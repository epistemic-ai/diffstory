"""Measure complete local v2 narration without provider or network access.

Run each phase in a fresh process, with the same explicitly pinned input:
  .venv/bin/python -B scripts/benchmark_narration.py --phase cold \
      --report /private/input.json --sha256 DIGEST --state /private/benchmark
  .venv/bin/python -B scripts/benchmark_narration.py --phase warm \
      --report /private/input.json --sha256 DIGEST --state /private/benchmark

``cold`` requires a nonexistent state directory. ``resume`` keeps all existing
lineage budgets after a failed/interrupted cold phase. No phase resets state.
Metrics contain only aggregates, fixed labels and digests. Raw local artifacts
are private and MUST NOT be submitted to a model. Generic fixture prose tests
accounting and validation, not semantic explanation quality or LLM performance.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import io
import json
import os
import resource
import shutil
import sys
import time
from collections import Counter
from contextlib import ExitStack
from contextlib import contextmanager
from contextlib import redirect_stderr
from contextlib import redirect_stdout
from functools import wraps
from pathlib import Path
from threading import Lock
from threading import local
from typing import TYPE_CHECKING
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from diffstory import analysis as analysis_module  # noqa: E402
from diffstory import cli as cli_module  # noqa: E402
from diffstory import narrative_hierarchy  # noqa: E402
from diffstory import narrative_manifest  # noqa: E402
from diffstory import narrative_plan  # noqa: E402
from diffstory import narrative_runtime  # noqa: E402
from diffstory import narrative_v2  # noqa: E402
from diffstory import render as render_module  # noqa: E402
from diffstory.analysis import apply_annotations  # noqa: E402
from diffstory.cli import main as cli_main  # noqa: E402
from diffstory.narrative import CodexCLIProvider  # noqa: E402
from diffstory.narrative import Narrator  # noqa: E402
from diffstory.narrative_manifest import persisted_manifest  # noqa: E402
from diffstory.narrative_plan import canonical_bytes  # noqa: E402
from diffstory.narrative_runtime import ArtifactCallError  # noqa: E402
from diffstory.narrative_runtime import ArtifactRuntimeError  # noqa: E402
from diffstory.narrative_runtime import ProviderReply  # noqa: E402
from diffstory.narrative_runtime import RuntimeStageError  # noqa: E402
from diffstory.narrative_storage import encode_invocation  # noqa: E402
from diffstory.narrative_storage import storage_digest  # noqa: E402
from diffstory.narrative_v2 import StageInputs  # noqa: E402
from diffstory.narrative_v2 import Transport  # noqa: E402
from diffstory.narrative_v2 import V2Options  # noqa: E402
from diffstory.usage import estimate_input  # noqa: E402

if TYPE_CHECKING:
    from collections.abc import Callable
    from collections.abc import Iterator
    from collections.abc import Sequence

    from diffstory.narrative_runtime import ProgressEvent

VERSION = "diffstory.local-benchmark.v1"
MODEL = "gpt-6.1-sol"
STAGES = ("detail", "planning", "section", "document")
DEFAULT_RESERVE = 2_000_000_000
SHA256_LENGTH = 64
PROGRESS_INTERVAL = 15
PHASE_LABELS = (
    "input.initial_read",
    "input.initial_json_parse",
    "input.canonical_json_parse",
    "canonicalization",
    "ledger.build",
    "ledger.index_build",
    "manifest.persisted_boundary",
    "manifest.structural_parse",
    "manifest.inline_validation",
    "result.stage_validation",
    "cache.lookup",
    "cache.read_parse",
    "cache.expand",
    "cache.record_parse",
    "proof.production",
    "proof.generated_manifest_audit",
    "proof.producer_request_audit",
    "proof.request_plan_audit",
    "proof.support.prose",
    "proof.support.consumed",
    "proof.support.forest",
    "source.resolve",
    "source.resolve_lookup",
    "apply.public",
    "persistence.serialize",
    "cli.reload.read_decode_validate",
    "cli.reload.json_parse",
    "render.safety_validation",
    "render.json_serialize",
    "render.html_assembly",
)
MEASUREMENT_QUALIFICATIONS = {
    "cli_reload": (
        "read_json includes file checks, bounded read, UTF-8 decode, object check, "
        "and JSON parse; the nested parse has its own timer."
    ),
    "cli_parser_input": (
        "The CLI JSON parser receives decoded text, so its byte-equivalent input "
        "is not re-encoded; the parser records characters."
    ),
    "manifest_results": (
        "Nested Pydantic result validators run inside GenerationManifest parsing; "
        "M1 measures that full parser and separate stage-result checks."
    ),
    "renderer": (
        "Renderer safety time includes generic checks and its nested full proof; "
        "same-thread child time is subtracted from exclusive time."
    ),
    "thread_cpu": (
        "Nested phase CPU uses per-thread CPU time; process_cpu_seconds is a "
        "separate run-wide total."
    ),
    "exclusive_time": (
        "Exclusive time subtracts nested instrumented phases on the same thread. "
        "Do not sum inclusive phase times."
    ),
    "peak_memory": (
        "peak_rss_bytes is a process-wide high-water mark, not a per-phase peak."
    ),
    "browser": (
        "Browser parsing and rendering are deferred to M6; Python HTML assembly "
        "does not measure browser work."
    ),
}


class BenchmarkError(Exception):
    """Carry only a fixed benchmark reason, never input or exception text."""


class PhaseMeasurements:
    """Collect fixed-label nested timings and source-free numeric counters."""

    def __init__(
        self,
        *,
        wall_clock: Callable[[], float] | None = None,
        thread_cpu_clock: Callable[[], float] | None = None,
    ) -> None:
        """Use monotonic wall and thread CPU clocks unless a test injects clocks."""
        self._wall_clock = wall_clock or time.perf_counter
        self._thread_cpu_clock = thread_cpu_clock or time.thread_time
        self._local = local()
        self._lock = Lock()
        self._values = {
            name: {
                "calls": 0,
                "failures": 0,
                "records": 0,
                "bytes": 0,
                "characters": 0,
                "wall_inclusive_seconds": 0.0,
                "wall_exclusive_seconds": 0.0,
                "thread_cpu_inclusive_seconds": 0.0,
                "thread_cpu_exclusive_seconds": 0.0,
            }
            for name in PHASE_LABELS
        }

    def _add(self, name: str, **counts: float) -> None:
        """Add numeric observations under a fixed phase label."""
        with self._lock:
            values = self._values[name]
            for key, value in counts.items():
                values[key] += value

    def observe(
        self,
        name: str,
        *,
        records: int = 0,
        byte_count: int = 0,
        characters: int = 0,
    ) -> None:
        """Record numeric counts without retaining or serializing source data."""
        self._add(
            name,
            records=records,
            bytes=byte_count,
            characters=characters,
        )

    @contextmanager
    def measure(self, name: str) -> Iterator[None]:
        """Measure one nested operation and preserve its result or exception."""
        if name not in self._values:
            msg = "unknown_measurement_phase"
            raise ValueError(msg)
        stack = getattr(self._local, "stack", None)
        if stack is None:
            stack = []
            self._local.stack = stack
        with self._lock:
            self._values[name]["calls"] += 1
        frame = {
            "wall_child": 0.0,
            "cpu_child": 0.0,
        }
        wall_start = self._wall_clock()
        cpu_start = self._thread_cpu_clock()
        stack.append(frame)
        failed = False
        try:
            yield
        except BaseException:
            failed = True
            raise
        finally:
            cpu_elapsed = max(0.0, self._thread_cpu_clock() - cpu_start)
            wall_elapsed = max(0.0, self._wall_clock() - wall_start)
            stack.pop()
            if stack:
                stack[-1]["wall_child"] += wall_elapsed
                stack[-1]["cpu_child"] += cpu_elapsed
            self._add(
                name,
                failures=int(failed),
                wall_inclusive_seconds=wall_elapsed,
                wall_exclusive_seconds=max(
                    0.0, wall_elapsed - frame["wall_child"]
                ),
                thread_cpu_inclusive_seconds=cpu_elapsed,
                thread_cpu_exclusive_seconds=max(
                    0.0, cpu_elapsed - frame["cpu_child"]
                ),
            )

    def wrap(  # noqa: PLR0913 # Timed hook counters stay explicit.
        self,
        name: str,
        function: Callable[..., object],
        *,
        records: Callable[..., int] | None = None,
        input_bytes: Callable[..., int] | None = None,
        input_characters: Callable[..., int] | None = None,
        output_records: Callable[[object], int] | None = None,
        output_bytes: Callable[[object], int] | None = None,
        output_characters: Callable[[object], int] | None = None,
    ) -> Callable[..., object]:
        """Wrap a hook while forwarding arguments, result, and exception intact."""

        @wraps(function)
        def wrapped(*args: object, **kwargs: object) -> object:
            with self.measure(name):
                if records is not None:
                    self.observe(name, records=records(*args, **kwargs))
                if input_bytes is not None:
                    self.observe(name, byte_count=input_bytes(*args, **kwargs))
                if input_characters is not None:
                    self.observe(
                        name,
                        characters=input_characters(*args, **kwargs),
                    )
                result = function(*args, **kwargs)
                if output_records is not None:
                    self.observe(name, records=output_records(result))
                if output_bytes is not None:
                    self.observe(name, byte_count=output_bytes(result))
                if output_characters is not None:
                    self.observe(name, characters=output_characters(result))
                return result

        return wrapped

    def snapshot(self) -> dict:
        """Return fixed-label counters and timings with no caller-owned objects."""
        with self._lock:
            return {name: dict(self._values[name]) for name in PHASE_LABELS}


class _ObservedJson:
    """Delegate JSON operations while replacing only the selected local hook."""

    def __init__(
        self, module: object, **overrides: Callable[..., object]
    ) -> None:
        """Keep the original JSON module and a small set of measured callables."""
        self._module = module
        self._overrides = overrides

    def __getattr__(self, name: str) -> object:
        """Forward every unmeasured JSON attribute unchanged."""
        if name in self._overrides:
            return self._overrides[name]
        return getattr(self._module, name)


def _bytes_length(*args: object, **_kwargs: object) -> int:
    """Return the length of one byte-buffer argument without copying it."""
    value = args[0] if args else None
    return (
        len(value) if isinstance(value, (bytes, bytearray, memoryview)) else 0
    )


def _manifest_record_count(value: object) -> int:
    """Count typed proof records using existing collections only."""
    names = ("details", "planning", "sections", "producers", "receipts")
    return 1 + sum(len(getattr(value, name, ())) for name in names)


def _ledger_record_count(value: object) -> int:
    """Count ledger records without building a serialized representation."""
    names = ("buffers", "spans", "units", "groups", "changes")
    return sum(len(getattr(value, name, ())) for name in names)


def _instrument_method(
    stack: ExitStack,
    owner: object,
    attribute: str,
    measurements: PhaseMeasurements,
    phase: str,
    **options: object,
) -> None:
    """Patch one module or class hook for this benchmark run only."""
    original = getattr(owner, attribute)
    stack.enter_context(
        patch.object(
            owner,
            attribute,
            measurements.wrap(phase, original, **options),
        )
    )


def _instrument_classmethod(
    stack: ExitStack,
    owner: type,
    attribute: str,
    measurements: PhaseMeasurements,
    phase: str,
    **options: object,
) -> None:
    """Patch a bound class validator while restoring its original descriptor."""
    original = getattr(owner, attribute)
    wrapped = measurements.wrap(phase, original, **options)

    def dispatch(_class: type, *args: object, **kwargs: object) -> object:
        return wrapped(*args, **kwargs)

    stack.enter_context(patch.object(owner, attribute, classmethod(dispatch)))


def install_measurement_hooks(
    stack: ExitStack, measurements: PhaseMeasurements
) -> None:
    """Install benchmark-local observers at the mapped narration boundaries."""
    for owner in (narrative_plan, narrative_v2, narrative_manifest):
        if hasattr(owner, "canonical_bytes"):
            _instrument_method(
                stack,
                owner,
                "canonical_bytes",
                measurements,
                "canonicalization",
                output_bytes=lambda value: len(value),
            )
    _instrument_method(
        stack,
        sys.modules[__name__],
        "canonical_bytes",
        measurements,
        "canonicalization",
        output_bytes=lambda value: len(value),
    )
    for owner in (narrative_plan, narrative_v2):
        if hasattr(owner, "build_evidence_ledger"):
            _instrument_method(
                stack,
                owner,
                "build_evidence_ledger",
                measurements,
                "ledger.build",
                output_records=_ledger_record_count,
            )
        if hasattr(owner, "build_ledger_index"):
            _instrument_method(
                stack,
                owner,
                "build_ledger_index",
                measurements,
                "ledger.index_build",
                records=lambda ledger, **_kwargs: len(ledger.units)
                + len(ledger.spans),
            )
    _instrument_method(
        stack,
        narrative_v2,
        "strict_json",
        measurements,
        "input.canonical_json_parse",
        records=lambda *_args, **_kwargs: 1,
        input_bytes=_bytes_length,
    )
    _instrument_method(
        stack,
        narrative_manifest,
        "persisted_manifest",
        measurements,
        "manifest.persisted_boundary",
        records=lambda _value, **_kwargs: 1,
    )
    _instrument_method(
        stack,
        sys.modules[__name__],
        "persisted_manifest",
        measurements,
        "manifest.persisted_boundary",
        records=lambda _value, **_kwargs: 1,
    )
    _instrument_classmethod(
        stack,
        narrative_manifest.GenerationManifest,
        "model_validate_json",
        measurements,
        "manifest.structural_parse",
        records=lambda *_args, **_kwargs: 1,
        input_bytes=_bytes_length,
    )
    _instrument_classmethod(
        stack,
        narrative_manifest.GenerationManifest,
        "model_validate",
        measurements,
        "manifest.inline_validation",
        records=lambda value, **_kwargs: _manifest_record_count(value),
    )
    _instrument_method(
        stack,
        narrative_v2.StageInputs,
        "validate",
        measurements,
        "result.stage_validation",
        records=lambda *_args, **_kwargs: 1,
    )

    _instrument_method(
        stack,
        narrative_runtime.ArtifactStore,
        "lookup",
        measurements,
        "cache.lookup",
        records=lambda *_args, **_kwargs: 1,
    )
    _instrument_method(
        stack,
        narrative_runtime.ArtifactStore,
        "read",
        measurements,
        "cache.read_parse",
        records=lambda *_args, **_kwargs: 1,
        input_bytes=lambda _self, serialized, **_kwargs: len(serialized),
    )
    _instrument_method(
        stack,
        narrative_runtime.ArtifactStore,
        "_expand",
        measurements,
        "cache.expand",
        records=lambda *_args, **_kwargs: 1,
    )
    for model in (
        narrative_runtime._StoredItem,  # noqa: SLF001 # Instrument the actual v3 leaf parser.
        narrative_runtime._StoredRequest,  # noqa: SLF001 # Instrument shared request records.
        narrative_runtime.StoredInvocation,
        narrative_runtime.StoredArtifact,
    ):
        _instrument_classmethod(
            stack,
            model,
            "model_validate_json",
            measurements,
            "cache.record_parse",
            records=lambda *_args, **_kwargs: 1,
            input_bytes=_bytes_length,
        )

    _instrument_method(
        stack,
        narrative_v2,
        "validate_production",
        measurements,
        "proof.production",
        records=lambda _report, manifest, **_kwargs: _manifest_record_count(
            manifest
        ),
    )
    _instrument_method(
        stack,
        narrative_v2.BatchedNarrator,
        "validate_manifest",
        measurements,
        "proof.generated_manifest_audit",
        records=lambda _self, manifest, **_kwargs: _manifest_record_count(
            manifest
        ),
    )
    _instrument_method(
        stack,
        narrative_v2.CodecReplay,
        "required",
        measurements,
        "proof.producer_request_audit",
        records=lambda _self, _invocation, requests, **_kwargs: len(requests),
    )
    _instrument_method(
        stack,
        narrative_v2,
        "validate_production_requests",
        measurements,
        "proof.request_plan_audit",
        records=lambda plan, **_kwargs: len(plan.requests),
    )
    _instrument_method(
        stack,
        narrative_v2,
        "validate_prose_support",
        measurements,
        "proof.support.prose",
        records=lambda *_args, **_kwargs: 1,
    )
    _instrument_method(
        stack,
        narrative_v2,
        "validate_support_forest",
        measurements,
        "proof.support.forest",
        records=lambda prose, *_args, **_kwargs: len(prose),
    )
    _instrument_method(
        stack,
        narrative_hierarchy,
        "validate_consumed_support",
        measurements,
        "proof.support.consumed",
        records=lambda *_args, **_kwargs: 1,
    )
    _instrument_method(
        stack,
        narrative_hierarchy,
        "validate_support_forest",
        measurements,
        "proof.support.forest",
        records=lambda prose, *_args, **_kwargs: len(prose),
    )
    _instrument_method(
        stack,
        narrative_plan,
        "resolve_span_ref",
        measurements,
        "source.resolve",
        records=lambda *_args, **_kwargs: 1,
    )
    _instrument_method(
        stack,
        narrative_plan,
        "_resolve_span_ref",
        measurements,
        "source.resolve_lookup",
        records=lambda *_args, **_kwargs: 1,
    )

    _instrument_method(
        stack,
        analysis_module,
        "apply_annotations",
        measurements,
        "apply.public",
        records=lambda *_args, **_kwargs: 1,
    )
    _instrument_method(
        stack,
        sys.modules[__name__],
        "apply_annotations",
        measurements,
        "apply.public",
        records=lambda *_args, **_kwargs: 1,
    )
    _instrument_method(
        stack,
        StorageGuard,
        "json",
        measurements,
        "persistence.serialize",
        records=lambda *_args, **_kwargs: 1,
        output_bytes=lambda value: value,
    )

    cli_json = cli_module.json
    stack.enter_context(
        patch.object(
            cli_module,
            "json",
            _ObservedJson(
                cli_json,
                loads=measurements.wrap(
                    "cli.reload.json_parse",
                    cli_json.loads,
                    records=lambda *_args, **_kwargs: 1,
                    input_characters=lambda value, **_kwargs: len(value)
                    if isinstance(value, str)
                    else 0,
                ),
            ),
        )
    )
    _instrument_method(
        stack,
        cli_module,
        "read_json",
        measurements,
        "cli.reload.read_decode_validate",
        records=lambda *_args, **_kwargs: 1,
    )
    _instrument_method(
        stack,
        render_module,
        "validate_report",
        measurements,
        "render.safety_validation",
        records=lambda *_args, **_kwargs: 1,
    )
    render_json = render_module.json
    stack.enter_context(
        patch.object(
            render_module,
            "json",
            _ObservedJson(
                render_json,
                dumps=measurements.wrap(
                    "render.json_serialize",
                    render_json.dumps,
                    records=lambda *_args, **_kwargs: 1,
                    output_characters=lambda value: len(value),
                ),
            ),
        )
    )
    _instrument_method(
        stack,
        render_module,
        "render",
        measurements,
        "render.html_assembly",
        records=lambda *_args, **_kwargs: 1,
        output_characters=lambda value: len(value),
    )
    _instrument_method(
        stack,
        cli_module,
        "render",
        measurements,
        "render.html_assembly",
        records=lambda *_args, **_kwargs: 1,
        output_characters=lambda value: len(value),
    )


def binding_summary(binding: dict) -> dict:
    """Export only fixed-key aggregates and a digest of the private binding."""
    return {
        "version": binding["version"],
        "binding_sha256": digest_json(binding),
        "candidate_file_count": len(binding["candidate_hashes"]),
        "source_bytes": binding["source_bytes"],
    }


def require(condition: bool, reason: str) -> None:  # noqa: FBT001
    """Stop at a concrete failed benchmark invariant."""
    if not condition:
        raise BenchmarkError(reason)


def file_digest(path: Path) -> str:
    """Hash a local file in bounded chunks without printing its contents."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def digest_json(value: object) -> str:
    """Bind a source-free summary or a local payload without exposing it."""
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def candidate_hashes() -> dict[str, str]:
    """Bind every runtime module and reader asset plus this benchmark."""
    paths = [*sorted((ROOT / "diffstory").glob("*.py"))]
    paths += sorted((ROOT / "diffstory" / "assets").glob("*"))
    paths.append(Path(__file__).resolve())
    return {
        str(p.relative_to(ROOT)): file_digest(p) for p in paths if p.is_file()
    }


def write_metrics(path: Path, value: dict) -> None:
    """Atomically replace a small, source-free checkpoint."""
    temporary = path.with_suffix(".pending")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    temporary.chmod(0o600)
    temporary.replace(path)


def statement(text: str, support: dict | None = None) -> dict:
    """Use declared support handles without inventing source interpretation."""
    support = support or {}
    claims = [x["handle"] for x in support.get("claims", [])[:1]]
    citations = [x["handle"] for x in support.get("citations", [])[:1]]
    basis = [x["handle"] for x in support.get("basis", [])[:1]]
    return {
        "kind": "implementation_claim"
        if claims or citations or basis
        else "editorial",
        "text": text,
        "claims": claims,
        "citations": citations,
        "basis": basis,
        "effect": None,
        "uncertainty": [],
    }


def fixture_response(data: dict) -> dict:
    """Retain every requested unit, obligation and required citation/limitation."""
    result = {"version": "diffstory.wire.v2", "items": []}
    if data["stage"] == "detail":
        result["acknowledgements"] = []
        for item in data["items"]:
            if not item["obligations"]:
                result["acknowledgements"].append(item["target"])
                continue
            result["items"].append(
                {
                    "target": item["target"],
                    "explanations": [
                        {
                            "obligations": [obligation["handle"]],
                            "before": "The base evidence is retained for this obligation.",
                            "after": "The head evidence is retained for this obligation.",
                            "effect": "This local fixture checks retention, not semantic behavior.",
                            "uncertainty": "No semantic quality or executed test outcome is inferred.",
                            "citations": list(
                                obligation["required_citations"]
                            ),
                            "limitations": [
                                x["handle"] for x in obligation["limitations"]
                            ],
                        }
                        for obligation in item["obligations"]
                    ],
                    "observations": [],
                    "questions": [],
                }
            )
        return result
    for item in data["items"]:
        prose = statement("Retained local fixture evidence.", item["support"])
        value = {"target": item["target"], "terminology": []}
        if data["stage"] == "planning":
            value["abstraction"] = [prose]
        elif data["stage"] == "section":
            value.update(
                title=statement("Local fixture section"),
                orientation=[statement("Inspect the retained evidence.")],
                synthesis=[prose],
                transition=[]
                if item["final"]
                else [statement("Continue to the next section.")],
            )
        else:
            value.update(
                preamble=[statement("Local structural benchmark.")],
                lead=[prose],
                closing=[
                    statement("Semantic quality requires separate review.")
                ],
            )
        result["items"].append(value)
    return result


class LocalCodexFixture(CodexCLIProvider):
    """Use the actual Codex codec/capacities with an unmistakable local identity.

    The closed Codex codec requires name='codex'. Destination, model, adapter
    type and inert executable label distinguish every artifact from real Codex
    output. The inherited prepare method and prefix are unchanged. No live
    completion or credential method is called, including legacy completion.
    """

    destination = "fixture://diffstory-local-benchmark/codex-codec/v1"

    def __init__(self) -> None:
        """Select real capacity metadata and replace all transport identity fields."""
        super().__init__(
            model=MODEL, executable="/nonexistent/diffstory-local-fixture"
        )
        self.model = f"local-fixture::{MODEL}"
        self._requested_model = self.model
        self.totals = {stage: Counter() for stage in STAGES}
        self.request_digests = {stage: hashlib.sha256() for stage in STAGES}
        self.response_digests = {stage: hashlib.sha256() for stage in STAGES}

    def require_credentials(self) -> None:
        """Local fixtures have no credentials or executable to resolve."""

    def complete(self, _body: bytes, _timeout: float | None) -> tuple:
        """Refuse the legacy transport entry point."""
        msg = "legacy_transport_forbidden"
        raise BenchmarkError(msg)

    def complete_reply(
        self, body: bytes, timeout: float, *, response_byte_limit: int
    ) -> ProviderReply:
        """Return one bounded fixture reply and aggregate bytes, never payloads."""
        started = time.monotonic()
        data = json.loads(body)["data"]
        stage = data["stage"]
        require(stage in STAGES, "unknown_stage")
        self.totals[stage]["attempts"] += 1
        raw = canonical_bytes(fixture_response(data))
        framed = self.request_prefix + body
        totals = self.totals[stage]
        totals["serialized_request_bytes"] += len(framed)
        totals["serialized_response_bytes"] += len(raw)
        totals["conservative_input_token_bound"] += estimate_input(
            framed, overhead_bytes=self.input_overhead_bytes
        )
        totals["conservative_response_token_bound"] += len(raw)
        totals["max_request_bytes"] = max(
            totals["max_request_bytes"], len(framed)
        )
        totals["max_response_bytes"] = max(
            totals["max_response_bytes"], len(raw)
        )
        self.request_digests[stage].update(hashlib.sha256(framed).digest())
        self.response_digests[stage].update(hashlib.sha256(raw).digest())
        if len(raw) > response_byte_limit:
            msg = "response_limit"
            raise ArtifactCallError(msg, response_bytes=len(raw))
        if time.monotonic() - started > timeout:
            msg = "timeout"
            raise ArtifactCallError(msg, response_bytes=len(raw))
        return ProviderReply(raw, None, len(raw))

    def metrics(self) -> dict:
        """Label unavailable token usage separately from conservative estimates."""
        return {
            stage: dict(self.totals[stage])
            | {
                "request_sequence_digest": self.request_digests[
                    stage
                ].hexdigest(),
                "response_sequence_digest": self.response_digests[
                    stage
                ].hexdigest(),
            }
            for stage in STAGES
        }


def forbid_external(*_args: object, **_kwargs: object) -> None:
    """Stop before any network, live provider, authentication or subprocess call."""
    msg = "external_io_forbidden"
    raise BenchmarkError(msg)


def offline_guard(stack: ExitStack) -> None:
    """Disable live transport entry points for generation and public reload."""
    for target in (
        "socket.socket",
        "socket.create_connection",
        "urllib.request.urlopen",
        "subprocess.Popen",
        "subprocess.run",
        "os.system",
        "diffstory.narrative.CodexCLIProvider.complete",
        "diffstory.narrative.CodexCLIProvider.complete_reply",
        "diffstory.narrative.CodexCLIProvider.require_credentials",
        "diffstory.narrative.OpenAIResponsesProvider.complete",
        "diffstory.narrative.OpenAIResponsesProvider.complete_reply",
        "diffstory.narrative.OpenAIResponsesProvider.require_credentials",
    ):
        stack.enter_context(patch(target, side_effect=forbid_external))


class StorageGuard:
    """Check free bytes before exact runtime records and bounded report writes."""

    def __init__(self, state: Path, reserve: int) -> None:
        """Retain an explicit free-space floor without changing cache layout."""
        self.state = state
        self.reserve = reserve
        self.minimum_free = shutil.disk_usage(state).free
        self.record_bytes = 0
        self.record_writes = 0
        self.shared_bytes = 0
        self.shared_writes = 0
        self.atomic = narrative_runtime._atomic_record  # noqa: SLF001 # Observe the existing writer, not a replacement cache format.
        self.immutable = narrative_runtime._immutable_cache_record  # noqa: SLF001 # Shared records and leaf temporaries use a distinct writer.

    def check(self, needed: int) -> None:
        """Refuse a write before it consumes the reserved filesystem space."""
        free = shutil.disk_usage(self.state).free
        self.minimum_free = min(self.minimum_free, free)
        require(free >= self.reserve + needed, "insufficient_storage")

    def record(self, path: Path, record: object) -> None:
        """Size the exact descriptor then call the unchanged atomic product writer."""
        size = len(record.model_dump_json().encode("utf-8"))
        self.check(size * 2)
        self.atomic(path, record)
        self.record_bytes += size
        self.record_writes += 1

    def shared_record(self, path: Path, record: object) -> None:
        """Reserve exact shared/leaf temporary bytes even for an existing address."""
        size = len(record.model_dump_json().encode("utf-8"))
        self.check(size * 2)
        self.immutable(path, record)
        self.shared_bytes += size
        self.shared_writes += 1

    def json(self, path: Path, value: dict) -> int:
        """Measure and stream complete JSON without a second serialized copy."""
        encoder = json.JSONEncoder(
            ensure_ascii=False, indent=2, allow_nan=False
        )
        size = sum(
            len(part.encode("utf-8")) for part in encoder.iterencode(value)
        )
        self.check(size * 2)
        with path.open("w", encoding="utf-8") as stream:
            for part in encoder.iterencode(value):
                stream.write(part)
        path.chmod(0o600)
        require(path.stat().st_size == size, "persisted_size_mismatch")
        return size


def storage_metrics(state: Path) -> dict:
    """Count private file bytes without exposing filenames or raw artifacts."""
    totals = {
        name: {"files": 0, "bytes": 0}
        for name in ("cache", "recovery", "run", "json", "html", "other")
    }
    for path in state.rglob("*"):
        if not path.is_file():
            continue
        relative = path.relative_to(state)
        kind = (
            "run"
            if "producer-runs-v2" in relative.parts
            or "runs-v2" in relative.parts
            else "cache"
            if relative.parts[0] == "cache"
            else "recovery"
            if relative.parts[0] == "recovery"
            else "html"
            if path.suffix == ".html"
            else "json"
            if relative.parts[0] == "output"
            else "other"
        )
        totals[kind]["files"] += 1
        totals[kind]["bytes"] += path.stat().st_size
    return totals


def cache_storage_metrics(state: Path) -> dict:
    """Count every optional cache file, including shared records and leftovers."""
    totals = {
        name: {"files": 0, "bytes": 0}
        for name in ("invocations", "requests", "items", "temporary", "other")
    }
    cache = state / "cache"
    for path in cache.rglob("*"):
        if not path.is_file():
            continue
        parts = path.relative_to(cache).parts
        kind = (
            "temporary"
            if path.name.startswith(".v3-")
            else parts[1]
            if len(parts) >= 3  # noqa: PLR2004 # artifacts-v3/<kind>/<record>.
            and parts[0] == "artifacts-v3"
            and parts[1] in {"invocations", "requests", "items"}
            else "other"
        )
        totals[kind]["files"] += 1
        totals[kind]["bytes"] += path.stat().st_size
    return totals


def ledger_metrics(ledger: object) -> dict:
    """Derive exact current source and obligation counts, never historical counts."""
    return {
        "buffers": len(ledger.buffers),
        "spans": len(ledger.spans),
        "units": len(ledger.units),
        "groups": len(ledger.groups),
        "changes": len(ledger.changes),
        "source_bytes": sum(b.byte_length for b in ledger.buffers),
        "context_units": sum(not u.obligations for u in ledger.units),
        "ungrouped_units": sum(u.group_key is None for u in ledger.units),
        "obligations": dict(
            Counter(o.kind for u in ledger.units for o in u.obligations)
        ),
        "limitations": sum(
            len(o.limitations) for u in ledger.units for o in u.obligations
        ),
        "ledger_digest": ledger.id,
    }


def detail_preflight(inputs: object) -> dict:
    """Measure compact initial metadata; results and later repair shapes are unknown."""
    stage = StageInputs(inputs, "detail", inputs.detail_items())
    batches = stage.pack(tuple(stage.items))
    repeated = compact = requests = leaves = 0
    for batch in batches:
        invocation = stage.invocation(batch)
        invocation_size = len(invocation.model_dump_json().encode("utf-8"))
        stored = encode_invocation(invocation)
        request = narrative_runtime._StoredRequest(  # noqa: SLF001 # Measure the actual shared storage contract.
            metadata=stage.prepare(
                batch.item_keys, repair=False, partition=batch.partition
            ).metadata
        )
        compact += len(stored.model_dump_json().encode("utf-8"))
        requests += len(request.model_dump_json().encode("utf-8"))
        repeated += invocation_size * len(batch.item_keys)
        invocation_digest = storage_digest(stored)
        request_digest = storage_digest(request)
        for key, semantic_key in invocation.artifact_keys().items():
            header = {
                "version": narrative_runtime._StoredItem.model_fields[  # noqa: SLF001 # Measure the product leaf envelope.
                    "version"
                ].default,
                "original_run_digest": inputs.run.digest,
                "invocation_digest": invocation_digest,
                "request_digest": request_digest,
                "item_key": key,
                "semantic_key": semantic_key,
            }
            # Exact leaf envelope size, excluding only the unknown result value.
            leaves += len(canonical_bytes(header)) + len(',"result":')
    return {
        "initial_detail_batches": len(batches),
        "dynamic_later_batches": "unknown_until_validated_results",
        "compact_invocation_bytes": compact,
        "successful_request_metadata_bytes_if_initial_batches_succeed": requests,
        "leaf_metadata_bytes_without_results": leaves,
        "initial_metadata_bytes_if_batches_succeed": compact
        + requests
        + leaves,
        "hypothetical_legacy_repeated_invocation_bytes": repeated,
        "excludes": "results, repairs, recovery, later stages, reports and HTML",
        "conservative_input_bound_sum": sum(b.input_bound for b in batches),
        "reserved_output_bound_sum": sum(
            b.output_reservation for b in batches
        ),
        "maximum_output_guard": max(b.output_guard for b in batches),
        "maximum_input_bound": max(b.input_bound for b in batches),
        "maximum_output_reservation": max(
            b.output_reservation for b in batches
        ),
    }


def coverage_metrics(candidate: dict, ledger: object) -> dict:
    """Check exact detail retention, every obligation, every group and receipts."""
    manifest = persisted_manifest(candidate["generation"]).model_dump(
        mode="json"
    )
    details = manifest["details"]
    displayed = [
        d for section in candidate["steps"] for d in section["details"]
    ] + candidate["context_details"]
    originals = {d["unit_key"]: digest_json(d) for d in details}
    retained = {d["unit_key"]: digest_json(d) for d in displayed}
    require(
        len(details) == len(displayed) == len(ledger.units), "detail_count"
    )
    require(originals == retained, "detail_retention")
    require(set(originals) == {u.id for u in ledger.units}, "unit_coverage")
    obligations = [o.id for u in ledger.units for o in u.obligations]
    explained = [
        o
        for d in details
        for e in d["explanations"]
        for o in e["obligation_ids"]
    ]
    require(Counter(obligations) == Counter(explained), "obligation_coverage")
    require(len(manifest["sections"]) == len(ledger.groups), "group_coverage")
    require(not manifest["sections"][-1]["transition"], "final_transition")
    return {
        "complete": True,
        "original_details": len(details),
        "retained_details": len(displayed),
        "explained_obligations": len(explained),
        "exact_detail_digest": digest_json(originals),
        "sections": len(manifest["sections"]),
        "final_transition_empty": True,
        "citations": len(candidate["citations"]),
        "support_provenance": len(candidate["support_provenance"]),
        "receipts": len(manifest["receipts"]),
        "receipt_dispositions": dict(
            Counter(r["disposition"] for r in manifest["receipts"])
        ),
        "plan_digest": digest_json(manifest["plan"]),
        "producer_count": len(manifest["producers"]),
        "producer_history_digest": digest_json(manifest["producers"]),
        "historical_production_requests": sum(
            len(p["requests"]) for p in manifest["producers"]
        ),
        "producer_stages": dict(
            Counter(
                p["invocation"]["batch"]["stage"]
                for p in manifest["producers"]
            )
        ),
        "producer_usage": "unavailable; fixture reports no LLM token usage",
    }


def initialize(args: argparse.Namespace) -> dict:
    """Bind state to exact source, code, options and explicit fixture identity."""
    require(
        len(args.sha256) == SHA256_LENGTH
        and all(c in "0123456789abcdef" for c in args.sha256),
        "invalid_pin",
    )
    require(file_digest(args.report) == args.sha256, "pin_mismatch")
    require(args.reserve_bytes >= 0, "invalid_reserve")
    configuration = Transport(LocalCodexFixture(), V2Options()).descriptor
    binding = {
        "version": VERSION,
        "local_fixture_only": True,
        "source_sha256": args.sha256,
        "source_bytes": args.report.stat().st_size,
        "configuration_digest": configuration.configuration_digest,
        "configuration": configuration.model_dump(mode="json"),
        "candidate_hashes": candidate_hashes(),
        "reserve_bytes": args.reserve_bytes,
        "concurrency": 1,
        "architecture": "v2",
    }
    marker = args.state / "LOCAL_FIXTURE_ONLY.json"
    if args.phase == "cold":
        require(not args.state.exists(), "state_exists")
        args.state.mkdir(mode=0o700, parents=True)
        write_metrics(marker, binding)
    else:
        require(marker.is_file(), "unmarked_state")
        require(
            json.loads(marker.read_text()) == binding, "state_binding_mismatch"
        )
        require(
            not (args.state / f"{args.phase}-metrics.json").exists(),
            "phase_already_recorded",
        )
        if args.phase == "warm":
            baseline = args.state / "resume-metrics.json"
            if not baseline.exists():
                baseline = args.state / "cold-metrics.json"
            require(
                json.loads(baseline.read_text())["status"] == "PASS",
                "cold_not_complete",
            )
    return binding


def execute(  # noqa: PLR0915 # Keep the sequential public boundary audit explicit.
    args: argparse.Namespace,
    metrics: dict,
    binding: dict,
    measurements: PhaseMeasurements,
) -> None:
    """Run actual generation, offline apply, persistence and supported CLI reload."""
    provider = LocalCodexFixture()
    guard = StorageGuard(args.state, args.reserve_bytes)
    metrics["storage_preflight"] = {
        "available_bytes": guard.minimum_free,
        "reserve_bytes": guard.reserve,
    }
    last_progress = 0.0

    def checkpoint(event: ProgressEvent) -> None:
        nonlocal last_progress
        if (
            time.monotonic() - last_progress < PROGRESS_INTERVAL
            and event.event not in {"stage_started", "stage_completed"}
        ):
            return
        last_progress = time.monotonic()
        metrics["progress"] = {
            "event": event.event,
            "completed": event.completed,
            "planned": event.planned,
            "attempts": event.provider_calls,
        }
        metrics["fixture"] = provider.metrics()
        metrics["phase_measurements"] = measurements.snapshot()
        write_metrics(args.state / "progress.json", metrics)

    narrator = Narrator(
        provider,
        architecture="v2",
        recovery_dir=args.state / "recovery",
        checkpoint_dir=args.state / "cache",
        concurrency=1,
        on_progress=checkpoint,
    )
    try:
        with ExitStack() as stack:
            install_measurement_hooks(stack, measurements)
            offline_guard(stack)
            stack.enter_context(
                patch(
                    "diffstory.narrative_runtime._atomic_record", guard.record
                )
            )
            stack.enter_context(
                patch(
                    "diffstory.narrative_runtime._immutable_cache_record",
                    guard.shared_record,
                )
            )
            metrics["stage"] = "load"
            with measurements.measure("input.initial_read"):
                report_bytes = args.report.read_bytes()
                measurements.observe(
                    "input.initial_read",
                    records=1,
                    byte_count=len(report_bytes),
                )
            with measurements.measure("input.initial_json_parse"):
                measurements.observe(
                    "input.initial_json_parse",
                    records=1,
                    byte_count=len(report_bytes),
                )
                report = json.loads(report_bytes)
            del report_bytes
            metrics["stage"] = "preflight"
            metrics["preview"] = narrator.preview(report)
            inputs = narrator._v2.inputs  # noqa: SLF001 # Inspect actual frozen public-preview inputs for measurement only.
            metrics["ledger"] = ledger_metrics(inputs.ledger)
            metrics["detail_preflight"] = detail_preflight(inputs)
            metrics["phase_measurements"] = measurements.snapshot()
            write_metrics(args.state / "progress.json", metrics)
            if args.phase == "cold":
                guard.check(
                    metrics["detail_preflight"][
                        "initial_metadata_bytes_if_batches_succeed"
                    ]
                )
            metrics["stage"] = "generate"
            candidate = narrator.generate(report)
            metrics["coverage"] = coverage_metrics(candidate, inputs.ledger)
            metrics["current_usage"] = candidate["generation"]["current_usage"]
            metrics["counters"] = narrator.v2_counters
            metrics["diagnostic_reasons"] = dict(
                Counter(d.reason for d in narrator.v2_diagnostics)
            )
            requests = [
                request
                for producer in candidate["generation"]["producers"]
                for request in producer["requests"]
            ]
            metrics["production_bounds"] = {
                "conservative_input_bound_sum": sum(
                    r["batch"]["input_bound"] for r in requests
                ),
                "reserved_output_bound_sum": sum(
                    r["batch"]["output_reservation"] for r in requests
                ),
                "maximum_output_guard": max(
                    r["batch"]["output_guard"] for r in requests
                ),
                "maximum_estimated_response_bytes": max(
                    r["batch"]["estimated_response_bytes"] for r in requests
                ),
                "maximum_timeout_seconds": max(
                    r["timeout_seconds"] for r in requests
                ),
                "usage_is_observed_llm_tokens": False,
            }
            del requests
            metrics["stage"] = "apply"
            annotated = apply_annotations(report, candidate)
            output = args.state / "output"
            output.mkdir(mode=0o700, exist_ok=True)
            annotation_path = output / f"{args.phase}.annotations.json"
            report_path = output / f"{args.phase}.report.json"
            metrics["stage"] = "persist"
            metrics["annotation_bytes"] = guard.json(
                annotation_path, candidate
            )
            metrics["report_bytes"] = guard.json(report_path, annotated)
            metrics["annotation_sha256"] = file_digest(annotation_path)
            metrics["report_sha256"] = file_digest(report_path)
            # Release production objects before the public reader reconstructs all
            # evidence and receipts. No trusted-cache flag is introduced.
            del candidate, annotated, inputs, report
            narrator._v2 = None  # noqa: SLF001 # Release the finished engine, never change runtime validation.
            gc.collect()
            metrics["stage"] = "public_cli_reload_render"
            # Reserve full report + a conservative JSON-escaping HTML allowance.
            # This is a storage guard, not a source or report coverage reduction.
            guard.check(metrics["report_bytes"] * 8 + 2_000_000)
            html_path = output / f"{args.phase}-reader.html"
            with (
                redirect_stdout(io.StringIO()),
                redirect_stderr(io.StringIO()),
            ):
                code = cli_main(
                    [
                        "render",
                        str(report_path),
                        "--out",
                        str(html_path),
                        "--max-report-bytes",
                        str(metrics["report_bytes"]),
                    ]
                )
            metrics["public_cli"] = {
                "exit_code": code,
                "max_report_bytes": metrics["report_bytes"],
                "explicit_measured_limit": True,
            }
            require(code == 0, "public_cli_reload_failed")
            require(
                file_digest(html_path.with_suffix(".report.json"))
                == metrics["report_sha256"],
                "cli_report_retention",
            )
            metrics["html_bytes"] = html_path.stat().st_size
            measurements.observe(
                "render.html_assembly", byte_count=metrics["html_bytes"]
            )
            metrics["html_sha256"] = file_digest(html_path)
            metrics["public_cli"]["report_bytes_preserved"] = True
            metrics["public_cli"]["strong_offline_validation"] = True
            if args.phase == "warm":
                baseline = args.state / "resume-metrics.json"
                if not baseline.exists():
                    baseline = args.state / "cold-metrics.json"
                cold = json.loads(baseline.read_text())
                require(
                    sum(t["attempts"] for t in provider.totals.values()) == 0,
                    "warm_attempts",
                )
                for key in ("ledger", "detail_preflight"):
                    require(metrics[key] == cold[key], "warm_plan_mismatch")
                for key in (
                    "plan_digest",
                    "exact_detail_digest",
                    "receipts",
                    "citations",
                    "support_provenance",
                    "producer_history_digest",
                ):
                    require(
                        metrics["coverage"][key] == cold["coverage"][key],
                        "warm_retention_mismatch",
                    )
                require(
                    metrics["pid"] != cold["pid"], "fresh_process_required"
                )
                metrics["fresh_process_warm"] = True
                require(
                    metrics["current_usage"]
                    == {
                        "input_tokens": 0,
                        "cached_input_tokens": 0,
                        "output_tokens": 0,
                        "calls": 0,
                        "unreported_calls": 0,
                    },
                    "warm_current_usage",
                )
            require(
                file_digest(args.report) == args.sha256,
                "source_changed_during_run",
            )
            require(
                candidate_hashes() == binding["candidate_hashes"],
                "candidate_changed_during_run",
            )
            metrics["status"] = "PASS"
            metrics["stage"] = "complete"
    finally:
        metrics["fixture"] = provider.metrics()
        metrics["counters"] = (
            narrator.v2_counters
            if narrator._v2 is not None  # noqa: SLF001 # A finished engine was explicitly released.
            else metrics.get("counters", {})
        )
        metrics["storage"] = storage_metrics(args.state)
        metrics["cache_storage"] = cache_storage_metrics(args.state)
        metrics["storage_guard"] = {
            "minimum_free_bytes": guard.minimum_free,
            "atomic_record_writes": guard.record_writes,
            "atomic_record_bytes_written": guard.record_bytes,
            "immutable_record_writes": guard.shared_writes,
            "immutable_record_bytes_written": guard.shared_bytes,
        }


class SafeParser(argparse.ArgumentParser):
    """Keep invalid command arguments and private paths out of diagnostics."""

    def error(self, _message: str) -> None:
        """Fail without echoing arbitrary user input."""
        msg = "invalid_arguments"
        raise BenchmarkError(msg)


def main(argv: Sequence[str] | None = None) -> int:
    """Run one fresh-process phase and emit only a fixed status and metrics."""
    parser = SafeParser(description=__doc__)
    parser.add_argument(
        "--phase", choices=("cold", "warm", "resume"), required=True
    )
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--sha256", required=True)
    parser.add_argument("--state", type=Path, required=True)
    parser.add_argument("--reserve-bytes", type=int, default=DEFAULT_RESERVE)
    started = time.monotonic()
    process_cpu_started = time.process_time()
    measurements = PhaseMeasurements()
    metrics = {
        "version": VERSION,
        "status": "BLOCKED",
        "stage": "initialize",
        "pid": os.getpid(),
        "live_provider_calls": 0,
        "network_calls": 0,
        "authentication_calls": 0,
        "semantic_quality": "not_measured; structural fixture only",
        "observed_llm_tokens": None,
    }
    ready = False
    binding = None
    try:
        args = parser.parse_args(argv)
        binding = initialize(args)
        metrics["binding"] = binding_summary(binding)
        ready = True
        metrics["phase"] = args.phase
        execute(args, metrics, binding, measurements)
    except BenchmarkError as error:
        metrics["reason"] = str(error)
    except (ArtifactRuntimeError, RuntimeStageError) as error:
        metrics["reason"] = (
            error.reason
            if isinstance(error, ArtifactRuntimeError)
            else error.diagnostic.reason
            if error.diagnostic
            else "runtime_failure"
        )
    except KeyboardInterrupt:
        metrics["reason"] = "interrupted"
    except MemoryError:
        metrics["reason"] = "memory_exhausted"
    except OSError:
        metrics["reason"] = "local_io_failure"
    except Exception as error:  # noqa: BLE001 # Never expose source-bearing validation errors or traceback.
        metrics["reason"] = "validation_or_internal_failure"
        metrics["exception_type_digest"] = hashlib.sha256(
            type(error).__name__.encode()
        ).hexdigest()
    metrics["elapsed_seconds"] = time.monotonic() - started
    metrics["process_cpu_seconds"] = time.process_time() - process_cpu_started
    metrics["phase_measurements"] = measurements.snapshot()
    metrics["measurement_qualifications"] = dict(MEASUREMENT_QUALIFICATIONS)
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    metrics["peak_rss_bytes"] = rss if sys.platform == "darwin" else rss * 1024
    metrics["peak_rss_scope"] = "process_run_high_water_mark"
    if ready:
        try:
            write_metrics(args.state / f"{args.phase}-metrics.json", metrics)
        except OSError:
            metrics["status"] = "BLOCKED"
            metrics["reason"] = "metrics_write_failed"
    sys.stdout.write(
        json.dumps(
            {
                key: metrics[key]
                for key in (
                    "status",
                    "stage",
                    "elapsed_seconds",
                    "process_cpu_seconds",
                    "peak_rss_bytes",
                    "reason",
                )
                if key in metrics
            }
        )
        + "\n"
    )
    return 0 if metrics["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
