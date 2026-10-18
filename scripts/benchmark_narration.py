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
import math
import os
import resource
import shlex
import shutil
import statistics
import subprocess
import sys
import tempfile
import threading
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
MODULE_WALL_STARTED = time.monotonic()
sys.path.insert(0, str(ROOT))

from diffstory import analysis as analysis_module  # noqa: E402
from diffstory import cli as cli_module  # noqa: E402
from diffstory import models as models_module  # noqa: E402
from diffstory import narrative_contracts as contracts_module  # noqa: E402
from diffstory import narrative_hierarchy  # noqa: E402
from diffstory import narrative_hierarchy as hierarchy_module  # noqa: E402
from diffstory import narrative_manifest  # noqa: E402
from diffstory import narrative_plan  # noqa: E402
from diffstory import narrative_protocol as protocol_module  # noqa: E402
from diffstory import narrative_runtime  # noqa: E402
from diffstory import narrative_session as session_module  # noqa: E402
from diffstory import narrative_storage as storage_module  # noqa: E402
from diffstory import narrative_v2  # noqa: E402
from diffstory import narrative_wire as wire_module  # noqa: E402
from diffstory import render as render_module  # noqa: E402
from diffstory import report_storage  # noqa: E402
from diffstory.analysis import apply_annotations  # noqa: E402
from diffstory.cli import main as cli_main  # noqa: E402
from diffstory.narrative import CodexCLIProvider  # noqa: E402
from diffstory.narrative import Narrator  # noqa: E402
from diffstory.narrative_manifest import expanded_generation  # noqa: E402
from diffstory.narrative_manifest import persisted_manifest  # noqa: E402
from diffstory.narrative_plan import canonical_bytes  # noqa: E402
from diffstory.narrative_runtime import ArtifactCallError  # noqa: E402
from diffstory.narrative_runtime import ArtifactRuntimeError  # noqa: E402
from diffstory.narrative_runtime import ProviderReply  # noqa: E402
from diffstory.narrative_runtime import RuntimeStageError  # noqa: E402
from diffstory.narrative_session import ReportSession  # noqa: E402
from diffstory.narrative_storage import encode_invocation  # noqa: E402
from diffstory.narrative_storage import storage_digest  # noqa: E402
from diffstory.narrative_v2 import StageInputs  # noqa: E402
from diffstory.narrative_v2 import Transport  # noqa: E402
from diffstory.narrative_v2 import V2Options  # noqa: E402
from diffstory.render import iter_render_validated  # noqa: E402
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
    "session.capture.snapshot",
    "session.capture.initialize",
    "session.capture.facade",
    "session.capture.freshness",
    "session.generate_validated",
    "session.apply",
    "session.finish_production",
    "session.shared_import",
    "session.indexed_import_proof",
    "session.stage.complete",
    "session.stage.proof_check",
    "session.records.assemble",
    "proof.indexed_production",
    "proof.stage.audit",
    "proof.producer.selected_lookup",
    "proof.producer.all_peer_iteration",
    "storage.unique_row_validation",
    "storage.graph.derive",
    "manifest.indexed_reader_assemble",
    "manifest.indexed_candidate_assemble",
    "manifest.indexed_applied_fields",
    "ledger.initial_bind",
    "ledger.retained_reuse",
    "source.metadata_fingerprint",
    "output.annotations.serialize",
    "output.report.serialize",
    "output.render.validated",
    "compat.generate",
    "compat.apply",
    "compat.public_validate",
    "compat.serialize",
    "compat.render",
    "shared.storage_parse",
    "shared.storage_expand",
    "benchmark.identity_extract",
    "benchmark.compatibility_expand",
    "cli.reload.report_comparison",
    "digest.content",
    "digest.canonical",
    "digest.semantic",
    "digest.canonical_stream",
    "digest.shape_stream",
)
MEASUREMENT_QUALIFICATIONS = {
    "cli_reload": (
        "The current report loader measures bounded byte reads and strict object "
        "decoding at this label; nested json.loads has its own timer."
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
    "pydantic_callbacks": (
        "Filtered Python call tracing counts discovered Pydantic field and model "
        "validator functions in separate profiled subprocesses. Pydantic-core "
        "and other C-internal callbacks are not observable."
    ),
    "synthetic": (
        "Synthetic timings use golden_report(count) and FixtureProvider only. "
        "Each sample is a fresh process; cold and warm artifact-cache runs are "
        "separate, and reported RSS is each process high-water mark."
    ),
    "compact_equivalence": (
        "The render CLI persists compact JSON. Retention compares compact JSON "
        "insertion order and array order, not the input file's pretty whitespace."
    ),
    "trace_observer": (
        "Full Python callback tracing is a separate, timing-ineligible pass. "
        "Its elapsed time, CPU, and RSS are not mixed with repeated timed runs."
    ),
    "browser": (
        "Browser parsing and rendering are deferred to M6; Python HTML assembly "
        "does not measure browser work."
    ),
}
SYNTHETIC_SCENARIOS = (
    "handle-paired",
    "compat-public",
)
SYNTHETIC_COORDINATE_FIELDS = (
    "count",
    "repeat",
    "scenario",
    "cache_state",
    "profile_kind",
)
SYNTHETIC_DEFAULT_COUNTS = (2, 4, 8)
SYNTHETIC_MIN_COUNTS = 3
SYNTHETIC_MIN_REPEATS = 3
SYNTHETIC_MAX_REPEATS = 5
SYNTHETIC_MAX_COUNT = 32
SYNTHETIC_MAX_OUTPUT_BYTES = 1 << 31
ALWAYS_SUMMARIZE_PHASES = frozenset(
    {
        "ledger.initial_bind",
        "ledger.retained_reuse",
        "proof.producer.selected_lookup",
        "proof.producer.all_peer_iteration",
        "source.resolve",
        "source.resolve_lookup",
        "source.metadata_fingerprint",
    }
)


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
        self._callback_counts: Counter[str] = Counter()
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

    def observe_callback(self, label: str) -> None:
        """Count one discovered Python Pydantic callback under a fixed label.

        Preconditions:
            ``label`` is derived from a loaded Diffstory model validator.
        Postconditions:
            The callback count increases without retaining call arguments.
        Essential behavior:
            Stores only class and validator identifiers; it never captures
            report data, request bodies, or callback locals.
        """
        with self._lock:
            self._callback_counts[label] += 1

    def callback_snapshot(self) -> dict[str, int]:
        """Return sorted Python validator callback counts for profiled runs.

        Preconditions:
            Callback profiling may have run in this process.
        Postconditions:
            Returns a detached name-to-count map with no call values.
        Essential behavior:
            Keeps callback totals separate from ordinary phase measurements.
        """
        with self._lock:
            return dict(sorted(self._callback_counts.items()))

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


def _pydantic_validator_codes() -> dict[object, str]:
    """Find loaded Diffstory field and model validator Python code objects.

    Preconditions:
        Product model modules have loaded their Pydantic decorators.
    Postconditions:
        Returns only code-object-to-fixed-label mappings; no model values or
        report data are retained.
    Essential behavior:
        Uses decorator metadata so profiling counts nested Python callbacks,
        not only public ``model_validate`` entry points.
    """
    modules = (
        models_module,
        contracts_module,
        narrative_hierarchy,
        narrative_manifest,
        narrative_plan,
        narrative_runtime,
        session_module,
        narrative_v2,
        protocol_module,
        report_storage,
        storage_module,
        wire_module,
    )
    callbacks: dict[object, str] = {}
    seen_classes: set[type] = set()
    for module in modules:
        for model in vars(module).values():
            if (
                not isinstance(model, type)
                or model in seen_classes
                or not model.__module__.startswith("diffstory.")
            ):
                continue
            seen_classes.add(model)
            decorators = getattr(model, "__pydantic_decorators__", None)
            if decorators is None:
                continue
            for kind in ("field_validators", "model_validators"):
                for name, decorator in getattr(decorators, kind, {}).items():
                    callback = getattr(decorator, "func", None)
                    callback = getattr(callback, "__func__", callback)
                    while callback is not None and hasattr(
                        callback, "__wrapped__"
                    ):
                        callback = callback.__wrapped__
                    code = getattr(callback, "__code__", None)
                    if code is not None:
                        prefix = (
                            "field" if kind == "field_validators" else "model"
                        )
                        callbacks[code] = f"{prefix}.{model.__name__}.{name}"
    return callbacks


@contextmanager
def profile_python_pydantic_validators(
    measurements: PhaseMeasurements,
) -> Iterator[None]:
    """Count Python Pydantic callbacks without timing their profiled run.

    Preconditions:
        The process is an isolated synthetic sample and no other task-owned
        profile hook must remain installed.
    Postconditions:
        Restores current-thread and future-thread profile hooks exactly.
    Essential behavior:
        Counts only discovered validator code objects and discards frames,
        arguments, and return values immediately.
    """
    callback_codes = _pydantic_validator_codes()

    def observe(frame: object, event: str, _arg: object) -> None:
        if event != "call":
            return
        label = callback_codes.get(frame.f_code)  # type: ignore[attr-defined]
        if label is not None:
            measurements.observe_callback(label)

    previous_thread_profile = threading.getprofile()
    previous_profile = sys.getprofile()
    sys.setprofile(observe)
    threading.setprofile(observe)
    try:
        yield
    finally:
        sys.setprofile(previous_profile)
        threading.setprofile(previous_thread_profile)


def _audit_stage_observer(
    stack: ExitStack, measurements: PhaseMeasurements
) -> None:
    """Count selected producers and complete original peers at one audit.

    Preconditions:
        ``_audit_stage_records`` is the shared expanded/indexed audit loop.
    Postconditions:
        Its arguments, return value, and exception behavior remain unchanged.
    Essential behavior:
        Counts receipt-selected producer lookups separately from every peer
        visited in each original invocation.
    """
    original = narrative_v2._audit_stage_records  # noqa: SLF001 # Observe the shared production audit helper.

    @wraps(original)
    def observed(*args: object, **kwargs: object) -> object:
        producers = args[2]
        receipts = args[3]
        all_peer_count = sum(
            len(producer.invocation.batch.item_keys) for producer in producers
        )
        measurements.observe(
            "proof.producer.selected_lookup",
            records=len(receipts),
        )
        measurements.observe(
            "proof.producer.all_peer_iteration",
            records=all_peer_count,
        )
        with measurements.measure("proof.stage.audit"):
            return original(*args, **kwargs)

    stack.enter_context(
        patch.object(narrative_v2, "_audit_stage_records", observed)
    )


def _ledger_binding_observer(
    stack: ExitStack, measurements: PhaseMeasurements
) -> None:
    """Separate initial ledger/index attachment from retained pair reuse.

    Preconditions:
        ``_CapturedInput.bind_ledger_index`` receives the exact pair produced
        or retained by one current production scope.
    Postconditions:
        The product binding check and returned objects are unchanged.
    Essential behavior:
        Counts initial attachment and later identity-checked reuse separately.
    """
    owner = session_module._CapturedInput  # noqa: SLF001 # Inspect only benchmark-local binding events.
    original = owner.bind_ledger_index

    @wraps(original)
    def observed(
        self: object, ledger: object, index: object, parameters: object
    ) -> object:
        retained = getattr(self, "_ledger", None) is not None
        label = "ledger.retained_reuse" if retained else "ledger.initial_bind"
        with measurements.measure(label):
            measurements.observe(
                label,
                records=len(getattr(index, "units", ()))
                + len(getattr(index, "spans", ())),
            )
            return original(self, ledger, index, parameters)

    stack.enter_context(patch.object(owner, "bind_ledger_index", observed))


def _instrument_digest_hooks(
    stack: ExitStack, measurements: PhaseMeasurements, *, trace_streams: bool
) -> None:
    """Measure canonical/digest calls and optionally count exact streamed bytes.

    Preconditions:
        All active modules use the canonical helpers already imported from
        ``narrative_plan``.
    Postconditions:
        Digest values remain identical and wrappers retain no input objects.
    Essential behavior:
        Counts byte-buffer digests in timed runs. Optional per-chunk byte
        tracing is enabled only in the separate profiled sample.
    """
    owners = (
        narrative_hierarchy,
        narrative_manifest,
        narrative_plan,
        narrative_runtime,
        session_module,
        narrative_v2,
        report_storage,
        storage_module,
        wire_module,
    )
    for owner in owners:
        if hasattr(owner, "content_digest"):
            _instrument_method(
                stack,
                owner,
                "content_digest",
                measurements,
                "digest.content",
                records=lambda *_args, **_kwargs: 1,
                input_bytes=_bytes_length,
            )
        if hasattr(owner, "canonical_digest"):
            _instrument_method(
                stack,
                owner,
                "canonical_digest",
                measurements,
                "digest.canonical",
                records=lambda *_args, **_kwargs: 1,
            )
        if hasattr(owner, "semantic_digest"):
            _instrument_method(
                stack,
                owner,
                "semantic_digest",
                measurements,
                "digest.semantic",
                records=lambda *_args, **_kwargs: 1,
            )
    if not trace_streams:
        return

    def count_chunks(
        name: str, original: Callable[..., Iterator[bytes]]
    ) -> Callable[..., Iterator[bytes]]:
        @wraps(original)
        def observed(*args: object, **kwargs: object) -> Iterator[bytes]:
            with measurements.measure(name):
                for chunk in original(*args, **kwargs):
                    measurements.observe(
                        name, records=1, byte_count=len(chunk)
                    )
                    yield chunk

        return observed

    canonical_chunks = narrative_plan._canonical_chunks  # noqa: SLF001 # Count exact canonical bytes only in profiled samples.
    shape_chunks = session_module._shape_chunks  # noqa: SLF001 # Count exact capture-shape bytes only in profiled samples.
    stack.enter_context(
        patch.object(
            narrative_plan,
            "_canonical_chunks",
            count_chunks("digest.canonical_stream", canonical_chunks),
        )
    )
    stack.enter_context(
        patch.object(
            session_module,
            "_shape_chunks",
            count_chunks("digest.shape_stream", shape_chunks),
        )
    )


def install_measurement_hooks(  # noqa: PLR0915 # Keep the mapped instrumentation surface in one reversible installer.
    stack: ExitStack,
    measurements: PhaseMeasurements,
    *,
    trace_digest_streams: bool = False,
) -> None:
    """Install benchmark-local observers at the mapped narration boundaries."""
    for owner in (
        narrative_plan,
        narrative_v2,
        narrative_manifest,
        narrative_hierarchy,
        narrative_runtime,
        session_module,
        protocol_module,
        report_storage,
        storage_module,
        wire_module,
    ):
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
    _instrument_digest_hooks(
        stack, measurements, trace_streams=trace_digest_streams
    )
    _instrument_method(
        stack,
        session_module,
        "_snapshot_report",
        measurements,
        "session.capture.snapshot",
        records=lambda *_args, **_kwargs: 1,
    )
    _instrument_method(
        stack,
        session_module._CapturedInput,  # noqa: SLF001 # Count exact capture initialization and freshness gates.
        "_initialize_capture",
        measurements,
        "session.capture.initialize",
        records=lambda *_args, **_kwargs: 1,
    )
    _instrument_method(
        stack,
        session_module._CapturedInput,  # noqa: SLF001 # Count cheap liveness/digest checks at real boundaries.
        "assert_capture_current",
        measurements,
        "session.capture.freshness",
        records=lambda *_args, **_kwargs: 1,
    )
    _ledger_binding_observer(stack, measurements)
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
        report_storage,
        "_parse_report_storage_for_import",
        measurements,
        "shared.storage_parse",
        records=lambda *_args, **_kwargs: 1,
        input_bytes=_bytes_length,
    )
    _instrument_method(
        stack,
        report_storage,
        "_logical_record_for_validation",
        measurements,
        "storage.unique_row_validation",
        records=lambda *_args, **_kwargs: 1,
        output_bytes=lambda result: len(result[1]),
    )
    for attribute in ("_derive_report_graph", "_derive_annotation_graph"):
        _instrument_method(
            stack,
            report_storage,
            attribute,
            measurements,
            "storage.graph.derive",
            records=lambda *_args, **_kwargs: 1,
        )
    _instrument_method(
        stack,
        narrative_manifest,
        "assemble_indexed_reader",
        measurements,
        "manifest.indexed_reader_assemble",
        records=lambda *_args, **_kwargs: 1,
    )
    _instrument_method(
        stack,
        narrative_manifest,
        "assemble_indexed_candidate",
        measurements,
        "manifest.indexed_candidate_assemble",
        records=lambda *_args, **_kwargs: 1,
    )
    _instrument_method(
        stack,
        narrative_manifest,
        "validate_indexed_applied_fields",
        measurements,
        "manifest.indexed_applied_fields",
        records=lambda *_args, **_kwargs: 1,
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
        narrative_v2,
        "validate_indexed_production",
        measurements,
        "proof.indexed_production",
        records=lambda _report, records, **_kwargs: (
            len(getattr(records, "details", ()))
            + len(getattr(records, "planning", ()))
            + len(getattr(records, "sections", ()))
            + 1
        ),
    )
    _audit_stage_observer(stack, measurements)
    _instrument_method(
        stack,
        narrative_v2,
        "_audit_indexed_import_stage",
        measurements,
        "proof.stage.audit",
        records=lambda _stage, references, **_kwargs: len(references),
    )
    _instrument_method(
        stack,
        session_module.ReportSession,
        "finish_production",
        measurements,
        "session.finish_production",
        records=lambda _self, _records, stage_proofs, **_kwargs: len(
            stage_proofs
        ),
    )
    _instrument_method(
        stack,
        session_module.ReportSession,
        "_complete_stage",
        measurements,
        "session.stage.complete",
        records=lambda *_args, **_kwargs: 1,
    )
    _instrument_method(
        stack,
        session_module.ReportSession,
        "_check_stage_proof",
        measurements,
        "session.stage.proof_check",
        records=lambda *_args, **_kwargs: 1,
    )
    _instrument_method(
        stack,
        session_module.ReportSession,
        "_recordset_for_production",
        measurements,
        "session.records.assemble",
        records=lambda _self, proofs, **_kwargs: len(proofs),
    )
    _instrument_method(
        stack,
        session_module.ReportSession,
        "apply",
        measurements,
        "session.apply",
        records=lambda *_args, **_kwargs: 1,
    )
    _instrument_method(
        stack,
        session_module.ReportSession,
        "_validate_indexed_import",
        measurements,
        "session.indexed_import_proof",
        records=lambda *_args, **_kwargs: 1,
    )
    _instrument_classmethod(
        stack,
        session_module.ReportSession,
        "import_indexed",
        measurements,
        "session.shared_import",
        records=lambda *_args, **_kwargs: 1,
        input_bytes=_bytes_length,
    )
    _instrument_method(
        stack,
        Narrator,
        "generate_validated",
        measurements,
        "session.generate_validated",
        records=lambda *_args, **_kwargs: 1,
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
        hierarchy_module._IndexedSourceState,  # noqa: SLF001 # Instrument the current metadata fingerprint, not only selector counts.
        "current_metadata_fingerprint",
        measurements,
        "source.metadata_fingerprint",
        records=lambda *_args, **_kwargs: 1,
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
        cli_module,
        "_read_bounded_bytes",
        measurements,
        "cli.reload.read_decode_validate",
        records=lambda *_args, **_kwargs: 1,
        output_bytes=lambda value: len(value),
    )
    _instrument_method(
        stack,
        cli_module,
        "_decode_json_object",
        measurements,
        "cli.reload.read_decode_validate",
        records=lambda *_args, **_kwargs: 1,
        input_bytes=_bytes_length,
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
    _instrument_method(
        stack,
        Narrator,
        "capture_session",
        measurements,
        "session.capture.facade",
        records=lambda *_args, **_kwargs: 1,
    )
    _instrument_method(
        stack,
        Narrator,
        "generate",
        measurements,
        "compat.generate",
        records=lambda *_args, **_kwargs: 1,
    )
    _instrument_method(
        stack,
        analysis_module,
        "apply_annotations",
        measurements,
        "compat.apply",
        records=lambda *_args, **_kwargs: 1,
    )
    _instrument_method(
        stack,
        render_module,
        "validate_report",
        measurements,
        "compat.public_validate",
        records=lambda *_args, **_kwargs: 1,
    )
    _instrument_method(
        stack,
        render_module,
        "render",
        measurements,
        "compat.render",
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


def compact_json_digest(value: object) -> str:
    """Hash compact JSON with the CLI serializer's insertion and array order.

    Preconditions:
        ``value`` is JSON-serializable without NaN or infinity.
    Postconditions:
        Returns the SHA-256 of the exact compact UTF-8 JSON body the CLI writes.
    Essential behavior:
        Streams encoder pieces into a digest without retaining a second JSON
        string or sorting object keys.
    """
    return compact_json_summary(value)["sha256"]


def compact_json_summary(value: object) -> dict[str, int | str]:
    """Stream the CLI's compact JSON identity and byte length.

    Preconditions:
        ``value`` is JSON-serializable without NaN or infinity.
    Postconditions:
        Returns only an exact UTF-8 byte count and a SHA-256 digest.
    Essential behavior:
        Preserves dictionary insertion and array order without retaining JSON.
    """
    encoder = json.JSONEncoder(
        ensure_ascii=False, allow_nan=False, separators=(",", ":")
    )
    digest = hashlib.sha256()
    byte_count = 0
    for part in encoder.iterencode(value):
        chunk = part.encode("utf-8")
        digest.update(chunk)
        byte_count += len(chunk)
    return {"bytes": byte_count, "sha256": digest.hexdigest()}


def rendered_report_digest(html: str | bytes) -> str:
    """Hash the exact logical report embedded by either renderer path.

    Preconditions:
        ``html`` is one self-contained Diffstory reader document.
    Postconditions:
        Returns a canonical digest of its embedded JSON report object.
    Essential behavior:
        Parses only the marked reader payload and exports no payload text.
    """
    value = html.encode("utf-8") if isinstance(html, str) else html
    prefix = b'<script type="application/json" id="report-data">'
    suffix = b"</script>"
    start = value.find(prefix)
    if start < 0:
        msg = "rendered_report_payload_missing"
        raise BenchmarkError(msg)
    start += len(prefix)
    end = value.find(suffix, start)
    if end < 0:
        msg = "rendered_report_payload_missing"
        raise BenchmarkError(msg)
    payload = value[start:end]
    try:
        report = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError):
        msg = "rendered_report_payload_invalid"
        raise BenchmarkError(msg) from None
    require(type(report) is dict, "rendered_report_payload_invalid")
    return digest_json(report)


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
            metrics["report_sha256"] = compact_json_digest(annotated)
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
            reloaded_report_path = html_path.with_suffix(".report.json")
            with measurements.measure("cli.reload.report_comparison"):
                reloaded_report_bytes = reloaded_report_path.stat().st_size
                reloaded_report_sha256 = file_digest(reloaded_report_path)
                measurements.observe(
                    "cli.reload.report_comparison",
                    records=1,
                    byte_count=reloaded_report_bytes,
                )
            require(
                reloaded_report_sha256 == metrics["report_sha256"],
                "cli_report_retention",
            )
            metrics["public_cli"]["compact_report_bytes"] = (
                reloaded_report_bytes
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


def parse_synthetic_counts(value: str) -> tuple[int, ...]:
    """Parse a finite set of at least three distinct synthetic row counts.

    Preconditions:
        ``value`` is a comma-separated command-line value.
    Postconditions:
        Returns counts in caller order, each between one and the fixed maximum.
    Essential behavior:
        Rejects malformed, repeated, or undersized matrices before any run.
    """
    try:
        parts = value.split(",")
        counts = tuple(int(part) for part in parts)
    except (AttributeError, ValueError):
        msg = "invalid_synthetic_counts"
        raise BenchmarkError(msg) from None
    require(
        len(counts) >= SYNTHETIC_MIN_COUNTS
        and len(set(counts)) == len(counts)
        and all(1 <= count <= SYNTHETIC_MAX_COUNT for count in counts),
        "invalid_synthetic_counts",
    )
    return counts


def _synthetic_fixture(count: int) -> tuple[dict, object]:
    """Build one authored compiler report and its offline fixture provider.

    Preconditions:
        ``count`` passed the finite synthetic matrix bounds.
    Postconditions:
        Returns ``golden_report(count)`` and the existing deterministic
        ``FixtureProvider``; neither reads arbitrary input files.
    Essential behavior:
        Adds the repository's test directory only for this short-lived local
        benchmark process.
    """
    test_directory = str(ROOT / "tests")
    if test_directory not in sys.path:
        sys.path.insert(0, test_directory)
    from test_narrative_v2 import FixtureProvider as SyntheticProvider
    from test_narrative_v2 import golden_report

    return golden_report(count), SyntheticProvider()


def _source_hash_labels() -> dict[str, str]:
    """Return fixed-label exact hashes for the benchmark and runtime surface.

    Preconditions:
        The assigned benchmark files and all runtime modules are readable.
    Postconditions:
        Returns two exact file hashes and one digest of the sorted runtime set.
    Essential behavior:
        Exports no repository paths or source bytes.
    """
    runtime = candidate_hashes()
    script_key = "scripts/benchmark_narration.py"
    return {
        "benchmark_script_sha256": runtime[script_key],
        "benchmark_tests_sha256": file_digest(
            ROOT / "tests" / "test_benchmark_narration.py"
        ),
        "runtime_source_set_sha256": digest_json(
            {key: value for key, value in runtime.items() if key != script_key}
        ),
    }


def _stream_summary(chunks: Iterator[bytes], *, retain: bool = False) -> dict:
    """Hash a UTF-8 stream and optionally retain bytes needed for import.

    Preconditions:
        ``chunks`` yields bytes from an accepted bounded output iterator.
    Postconditions:
        Returns exact byte/chunk counts and a digest; optional raw bytes are
        included only for the independent shared-import caller.
    Essential behavior:
        Never logs chunk values or converts streamed source/prose to text.
    """
    digest = hashlib.sha256()
    output = bytearray() if retain else None
    byte_count = 0
    chunk_count = 0
    for chunk in chunks:
        digest.update(chunk)
        byte_count += len(chunk)
        chunk_count += 1
        if output is not None:
            output.extend(chunk)
    result = {
        "bytes": byte_count,
        "sha256": digest.hexdigest(),
        "chunks": chunk_count,
    }
    if output is not None:
        result["raw"] = bytes(output)
    return result


def _measure_stream(
    measurements: PhaseMeasurements,
    phase: str,
    chunks: Iterator[bytes],
    *,
    retain: bool = False,
) -> dict:
    """Measure one fully consumed stream without storing its ordinary output.

    Preconditions:
        ``phase`` is a fixed output label and ``chunks`` is a valid iterator.
    Postconditions:
        Returns the source-free stream summary from :func:`_stream_summary`.
    Essential behavior:
        Counts one output record and its complete emitted bytes after timing.
    """
    with measurements.measure(phase):
        result = _stream_summary(chunks, retain=retain)
    measurements.observe(phase, records=1, byte_count=result["bytes"])
    if not retain:
        result.pop("raw", None)
    return result


def _request_identity_rows(manifest: dict) -> list[dict[str, str]]:
    """Select ordered request identities without returning request bodies.

    Preconditions:
        ``manifest`` is the expanded generation-v2 representation.
    Postconditions:
        Returns stage, invocation-batch, request-batch, and request-digest
        strings in producer/request order.
    Essential behavior:
        Excludes prompts, serialized inputs, result text, and provider bodies.
    """
    rows = []
    for producer in manifest.get("producers", ()):
        invocation = producer["invocation"]["batch"]
        for request in producer.get("requests", ()):
            batch = request["batch"]
            rows.append(
                {
                    "stage": batch["stage"],
                    "invocation_batch_id": invocation["id"],
                    "request_batch_id": batch["id"],
                    "request_digest": batch["request_digest"],
                }
            )
    return rows


def _request_identity_summary(rows: list[dict[str, str]]) -> dict:
    """Return request identity count, stage totals, and a one-way digest.

    Preconditions:
        ``rows`` contains only fixed request-identity metadata.
    Postconditions:
        No request IDs or digests are returned individually.
    Essential behavior:
        Binds exact ordered identities while keeping all request data private.
    """
    return {
        "request_count": len(rows),
        "request_identity_sha256": digest_json(rows),
        "request_stage_counts": dict(
            sorted(Counter(row["stage"] for row in rows).items())
        ),
    }


def _handle_request_identity_summary(
    session: ReportSession,
    generation: object,
    measurements: PhaseMeasurements,
) -> dict:
    """Read identity metadata from this session's exact checked generation.

    Preconditions:
        ``generation`` is live in ``session`` and its producer/request rows
        belong to the registered graph.
    Postconditions:
        Returns the same digest shape as the expanded compatibility adapter.
    Essential behavior:
        Uses checked handle access and resolves only producer/request metadata.
    """
    with measurements.measure("benchmark.identity_extract"):
        proof = session._checked_generation_data(generation)  # noqa: SLF001 # Use the exact live capability for source-free identity extraction.
        rows = []
        for producer_ref in proof.records.producers:
            producer = proof.graph.record(
                producer_ref.table, producer_ref.storage_id
            )
            invocation = proof.graph.record(
                "invocations", producer["invocation"]["$ref"]
            )
            invocation_batch = invocation["batch"]
            for request_ref in producer["requests"]:
                request = proof.graph.record("requests", request_ref["$ref"])
                batch = request["batch"]
                rows.append(
                    {
                        "stage": batch["stage"],
                        "invocation_batch_id": invocation_batch["id"],
                        "request_batch_id": batch["id"],
                        "request_digest": batch["request_digest"],
                    }
                )
    return _request_identity_summary(rows)


def _proof_call_total(snapshot: dict, labels: tuple[str, ...]) -> int:
    """Sum only complete production proof and independent-ingress calls.

    Preconditions:
        ``snapshot`` is a detached ``PhaseMeasurements`` value.
    Postconditions:
        Returns a nonnegative count for the named complete-proof boundaries.
    Essential behavior:
        Excludes cheap liveness, digest, structural, and stage-local checks.
    """
    return sum(snapshot[name]["calls"] for name in labels)


def _phase_delta(before: dict, after: dict) -> dict:
    """Return numeric per-phase deltas between two detached snapshots.

    Preconditions:
        Both snapshots came from the same ``PhaseMeasurements`` instance.
    Postconditions:
        Returns fixed-label numeric deltas without retaining either snapshot.
    Essential behavior:
        Keeps same-thread inclusive and exclusive phase measurements separate.
    """
    return {
        name: {
            key: after[name][key] - before[name][key] for key in before[name]
        }
        for name in PHASE_LABELS
    }


def _source_resolution_summary(inputs: object | None) -> dict[str, int]:
    """Read source-resolution counts, including current physical bindings.

    Preconditions:
        ``inputs`` is the current production-input object or ``None``.
    Postconditions:
        Returns counts only; reference keys and physical source metadata stay
        inside the current citation scope.
    Essential behavior:
        Uses the scope's explicit memoized physical-binding count.
    """
    scope = getattr(inputs, "citation_scope", None)
    if scope is None:
        return {
            "resolver_calls": 0,
            "distinct_current_physical_bindings": 0,
        }
    return {
        "resolver_calls": scope.resolution_count,
        "distinct_current_physical_bindings": scope.memoized_reference_count,
    }


class BenchmarkProvider:
    """Wrap the deterministic test fixture without retaining request bodies.

    Attributes:
        call_count: Number of local fixture replies returned in this run.
        _provider: Existing authored ``FixtureProvider`` implementation.

    Use:
        Construct once per synthetic process. It delegates the exact fixture
        codec and response rules, then clears its private request log.
    """

    def __init__(self, provider: object) -> None:
        """Retain one deterministic fixture provider and an empty call count.

        Preconditions:
            ``provider`` implements the existing local fixture protocol.
        Postconditions:
            No provider request has been sent or retained.
        Essential behavior:
            Keeps the test fixture response logic unchanged.
        """
        self._provider = provider
        self.call_count = 0

    def __getattr__(self, name: str) -> object:
        """Delegate static provider metadata and prepare behavior unchanged."""
        return getattr(self._provider, name)

    def complete_reply(
        self, body: bytes, timeout: float, *, response_byte_limit: int
    ) -> ProviderReply:
        """Return one existing fixture response, then release its raw request.

        Preconditions:
            ``body`` and ``response_byte_limit`` satisfy the local fixture
            provider's normal request boundary.
        Postconditions:
            Returns its unchanged reply and retains no request dictionary.
        Essential behavior:
            Counts provider attempts without recording prompt or source data.
        """
        self.call_count += 1
        try:
            return self._provider.complete_reply(
                body,
                timeout,
                response_byte_limit=response_byte_limit,
            )
        finally:
            self._provider.calls.clear()


def _measure_handle_outputs(  # noqa: PLR0913 # These boundaries share one live session and proof.
    session: ReportSession,
    generation: object,
    report_handle: object,
    storage: str,
    measurements: PhaseMeasurements,
    *,
    retain_report_bytes: bool = False,
) -> dict:
    """Stream annotation, report, and self-contained HTML output once.

    Preconditions:
        Both handles are live in ``session`` and ``storage`` is an accepted
        output mode.
    Postconditions:
        Returns only exact byte counts, chunk counts, and output digests.
    Essential behavior:
        Consumes the production iterators directly and retains no ordinary
        output string or byte buffer.
    """
    annotations = _measure_stream(
        measurements,
        "output.annotations.serialize",
        session.iter_annotations_json(
            generation,
            report_storage=storage,
            max_output_bytes=SYNTHETIC_MAX_OUTPUT_BYTES,
        ),
    )
    report = _measure_stream(
        measurements,
        "output.report.serialize",
        session.iter_report_json(
            report_handle,
            report_storage=storage,
            max_output_bytes=SYNTHETIC_MAX_OUTPUT_BYTES,
        ),
        retain=retain_report_bytes,
    )
    html = _measure_stream(
        measurements,
        "output.render.validated",
        iter_render_validated(
            session,
            report_handle,
            report_storage=storage,
            max_output_bytes=SYNTHETIC_MAX_OUTPUT_BYTES,
        ),
    )
    result = {
        "annotations": annotations,
        "report": report,
        "html": html,
        "report_storage": storage,
        "compact_generation_baseline": "diffstory.generation-storage.v1",
    }
    if storage == "expanded":
        result["annotations"]["logical_sha256"] = annotations["sha256"]
        result["report"]["logical_sha256"] = report["sha256"]
    if retain_report_bytes:
        result["_raw_report"] = report.pop("raw")
    return result


def _run_shared_import_scenario(
    shared_bytes: bytes,
    source_summary: dict,
    measurements: PhaseMeasurements,
) -> tuple[dict, dict]:
    """Reimport emitted shared report bytes in a new independent scope.

    Preconditions:
        ``shared_bytes`` came from the current generation's trusted report
        iterator and is within the local output cap.
    Postconditions:
        Returns shared output metrics and deterministic proof-work gates for
        the newly imported generation.
    Essential behavior:
        The exact emitted bytes, not an existing capability or parsed graph,
        enter ``ReportSession.import_indexed``.
    """
    before_import = measurements.snapshot()
    imported, imported_generation = ReportSession.import_indexed(
        shared_bytes,
        max_input_bytes=len(shared_bytes),
    )
    del shared_bytes
    after_import = measurements.snapshot()
    import_proof_delta = {
        label: after_import[label]["calls"] - before_import[label]["calls"]
        for label in (
            "proof.indexed_production",
            "session.indexed_import_proof",
            "ledger.build",
            "ledger.index_build",
        )
    }
    with imported:
        request_identity = _handle_request_identity_summary(
            imported, imported_generation, measurements
        )
        before_apply = measurements.snapshot()
        imported_report = imported.apply(imported_generation)
        after_apply = measurements.snapshot()
        outputs = _measure_handle_outputs(
            imported,
            imported_generation,
            imported_report,
            "shared",
            measurements,
        )
        after_output = measurements.snapshot()
        with measurements.measure("benchmark.compatibility_expand"):
            logical_report = imported.report_data(imported_report)
            logical_report_sha256 = digest_json(logical_report)
            logical_report_bytes = len(canonical_bytes(logical_report))
        measurements.observe(
            "benchmark.compatibility_expand",
            records=1,
            byte_count=logical_report_bytes,
        )
        after_expansion = measurements.snapshot()
        require(
            outputs["report"]["sha256"] == source_summary["sha256"],
            "shared_import_report_bytes_changed",
        )
        del logical_report
        proof_labels = (
            "session.finish_production",
            "proof.production",
            "proof.indexed_production",
            "session.indexed_import_proof",
        )
        ledger_labels = ("ledger.build", "ledger.index_build")
        gates = {
            "independent_import_complete_proofs": import_proof_delta[
                "proof.indexed_production"
            ],
            "independent_import_session_proofs": import_proof_delta[
                "session.indexed_import_proof"
            ],
            "apply_added_complete_proofs": _proof_call_total(
                after_apply, proof_labels
            )
            - _proof_call_total(before_apply, proof_labels),
            "output_added_complete_proofs": _proof_call_total(
                after_output, proof_labels
            )
            - _proof_call_total(after_apply, proof_labels),
            "apply_added_ledger_index_builds": sum(
                after_apply[label]["calls"] - before_apply[label]["calls"]
                for label in ledger_labels
            ),
            "output_added_ledger_index_builds": sum(
                after_output[label]["calls"] - after_apply[label]["calls"]
                for label in ledger_labels
            ),
        }
        outputs["request_identity"] = request_identity
        outputs["source_shared_report"] = source_summary
        outputs["logical_report"] = {
            "bytes": logical_report_bytes,
            "sha256": logical_report_sha256,
        }
        outputs["source_resolution"] = {
            "resolver_calls": 0,
            "distinct_current_physical_bindings": 0,
        }
        outputs["path_phase_measurements"] = {
            "independent_import_proof": _phase_delta(
                before_import, after_import
            ),
            "independent_import_apply_and_output": _phase_delta(
                after_import, after_output
            ),
            "independent_import_expansion": _phase_delta(
                after_output, after_expansion
            ),
        }
    return outputs, gates


def _run_expanded_public_scenario(
    session: ReportSession,
    report_handle: object,
    expanded_output: dict,
    measurements: PhaseMeasurements,
) -> tuple[dict, dict]:
    """Expand the same logical report and cross public validation/render APIs.

    Preconditions:
        The report handle was applied in the current session and remains live.
    Postconditions:
        Returns exact compact report/HTML size and digest data plus proof counts.
    Essential behavior:
        Measures public expanded boundaries separately from the trusted handle
        flow that supplied their equivalent logical report.
    """
    before_expansion = measurements.snapshot()
    with measurements.measure("benchmark.compatibility_expand"):
        expanded = session.report_data(report_handle)
        report_summary = compact_json_summary(expanded)
        logical_report_digest = digest_json(expanded)
    measurements.observe(
        "benchmark.compatibility_expand",
        records=1,
        byte_count=report_summary["bytes"],
    )
    after_expansion = measurements.snapshot()
    require(
        logical_report_digest == expanded_output["report"]["sha256"],
        "expanded_public_report_parity",
    )
    with measurements.measure("compat.public_validate"):
        render_module.validate_report(expanded)
    after_validation = measurements.snapshot()
    before_render = measurements.snapshot()
    with measurements.measure("compat.render"):
        html = render_module.render(expanded)
        encoded_html = html.encode("utf-8")
        html_summary = {
            "bytes": len(encoded_html),
            "sha256": hashlib.sha256(encoded_html).hexdigest(),
            "characters": len(html),
        }
    embedded_report_digest = rendered_report_digest(html)
    html_summary["embedded_report_sha256"] = embedded_report_digest
    require(
        embedded_report_digest == logical_report_digest,
        "expanded_public_render_payload_parity",
    )
    measurements.observe(
        "compat.render",
        records=1,
        byte_count=html_summary["bytes"],
        characters=html_summary["characters"],
    )
    del html, encoded_html
    after_render = measurements.snapshot()
    proof_labels = (
        "session.finish_production",
        "proof.production",
        "proof.indexed_production",
        "session.indexed_import_proof",
    )
    gates = {
        "public_expanded_complete_proofs": _proof_call_total(
            after_render, proof_labels
        )
        - _proof_call_total(before_expansion, proof_labels),
        "render_added_complete_proofs": _proof_call_total(
            after_render, proof_labels
        )
        - _proof_call_total(before_render, proof_labels),
    }
    return (
        {
            "annotations": None,
            "report": {
                **report_summary,
                "logical_sha256": logical_report_digest,
            },
            "html": html_summary,
            "report_storage": "expanded-public",
            "compact_generation_baseline": "diffstory.generation-storage.v1",
            "path_phase_measurements": {
                "expanded_public_conversion": _phase_delta(
                    before_expansion, after_expansion
                ),
                "expanded_public_validation": _phase_delta(
                    after_expansion, after_validation
                ),
                "expanded_public_render": _phase_delta(
                    after_validation, after_render
                ),
            },
        },
        {
            **gates,
            "logical_report_matches_trusted_expanded_stream": True,
            "public_html_payload_matches_logical_report": True,
        },
    )


def _run_compatibility_scenario(
    narrator: Narrator,
    report: dict,
    measurements: PhaseMeasurements,
) -> tuple[dict, dict]:
    """Run the current public ``generate/apply_annotations/render`` pipeline.

    Preconditions:
        ``narrator`` is configured with the local deterministic v2 provider.
    Postconditions:
        Returns source-free compact output sizes and exact request identity.
    Essential behavior:
        Uses each ordinary expanded dictionary boundary, including its current
        independent proof checks.
    """
    before_generate = measurements.snapshot()
    candidate = narrator.generate(report)
    after_generate = measurements.snapshot()
    generation = expanded_generation(candidate["generation"])
    request_identity = _request_identity_summary(
        _request_identity_rows(generation)
    )
    before_generation_serialization = measurements.snapshot()
    with measurements.measure("compat.serialize"):
        annotations = compact_json_summary(candidate)
        annotations["logical_sha256"] = digest_json(candidate)
    after_generation_serialization = measurements.snapshot()
    before_apply = measurements.snapshot()
    applied = apply_annotations(report, candidate)
    after_apply = measurements.snapshot()
    before_apply_serialization = measurements.snapshot()
    with measurements.measure("compat.serialize"):
        report_summary = compact_json_summary(applied)
        report_summary["logical_sha256"] = digest_json(applied)
    after_apply_serialization = measurements.snapshot()
    before_render = measurements.snapshot()
    html = render_module.render(applied)
    encoded_html = html.encode("utf-8")
    html_summary = {
        "bytes": len(encoded_html),
        "sha256": hashlib.sha256(encoded_html).hexdigest(),
        "characters": len(html),
        "embedded_report_sha256": rendered_report_digest(html),
    }
    after_render = measurements.snapshot()
    del candidate, applied, html, encoded_html
    return (
        {
            "annotations": annotations,
            "report": report_summary,
            "html": html_summary,
            "report_storage": "expanded-public",
            "compact_generation_baseline": "diffstory.generation-storage.v1",
            "request_identity": request_identity,
            "path_phase_measurements": {
                "compatibility_generate": _phase_delta(
                    before_generate, after_generate
                ),
                "compatibility_generation_serialization": _phase_delta(
                    before_generation_serialization,
                    after_generation_serialization,
                ),
                "compatibility_apply": _phase_delta(before_apply, after_apply),
                "compatibility_apply_serialization": _phase_delta(
                    before_apply_serialization, after_apply_serialization
                ),
                "compatibility_render": _phase_delta(
                    before_render, after_render
                ),
            },
        },
        {},
    )


def run_synthetic_sample(args: argparse.Namespace) -> dict:  # noqa: PLR0915 # Keep one complete synthetic path with its deterministic gates.
    """Run one bounded fresh-process synthetic evidence sample.

    Preconditions:
        Scenario, count, cache state, and cache directory came from the closed
        local benchmark CLI; no report input path is supplied.
    Postconditions:
        Returns source-free timings, phase counters, outputs, and proof gates.
    Essential behavior:
        Runs exactly one selected flow with the deterministic local provider.
    """
    started = MODULE_WALL_STARTED
    measurements = PhaseMeasurements()
    path_phase_measurements: dict[str, dict] = {}
    metrics = {
        "status": "BLOCKED",
        "scenario": args.synthetic_sample,
        "count": args.count,
        "cache_state": args.cache_state,
        "profiled_callbacks": bool(args.profile_callbacks),
        "live_provider_calls": 0,
        "network_calls": 0,
        "authentication_calls": 0,
        "semantic_quality": "not_measured; deterministic structural fixture",
        "observed_llm_tokens": None,
        "process_resource_scope": "whole_selected_scenario_child_process",
    }
    try:
        require(
            args.synthetic_sample in SYNTHETIC_SCENARIOS, "invalid_scenario"
        )
        require(
            1 <= args.count <= SYNTHETIC_MAX_COUNT, "invalid_synthetic_count"
        )
        require(args.cache_state in {"cold", "warm"}, "invalid_cache_state")
        report, fixture = _synthetic_fixture(args.count)
        provider = BenchmarkProvider(fixture)
        descriptor = Transport(provider, V2Options()).descriptor
        metrics.update(
            {
                "fixture_sha256": digest_json(report),
                "fixture_bytes": len(canonical_bytes(report)),
                "revision_binding_sha256": digest_json(
                    {
                        "base_sha": report["meta"]["base_sha"],
                        "head_sha": report["meta"]["head_sha"],
                    }
                ),
                "request_configuration_sha256": digest_json(
                    descriptor.model_dump(mode="json")
                ),
            }
        )
        narrator = Narrator(
            provider,
            architecture="v2",
            recovery_dir=args.cache_dir / "recovery",
            checkpoint_dir=args.cache_dir / "cache",
            concurrency=1,
        )
        with ExitStack() as stack:
            offline_guard(stack)
            install_measurement_hooks(
                stack,
                measurements,
                trace_digest_streams=args.profile_callbacks,
            )
            if args.profile_callbacks:
                stack.enter_context(
                    profile_python_pydantic_validators(measurements)
                )
            if args.synthetic_sample == "compat-public":
                outputs, gates = _run_compatibility_scenario(
                    narrator, report, measurements
                )
                path_phase_measurements = outputs.pop(
                    "path_phase_measurements"
                )
                inputs = getattr(narrator._v2, "inputs", None)  # noqa: SLF001 # Read only current source-scope counts.
                outputs["source_resolution"] = _source_resolution_summary(
                    inputs
                )
                request_identity = outputs["request_identity"]
            else:
                before_capture = measurements.snapshot()
                session = narrator.capture_session(report)
                after_capture = measurements.snapshot()
                with session:
                    before_generation = measurements.snapshot()
                    generation = narrator.generate_validated(
                        report, session=session
                    )
                    after_generation = measurements.snapshot()
                    generated_snapshot = measurements.snapshot()
                    request_identity = _handle_request_identity_summary(
                        session, generation, measurements
                    )
                    before_apply = measurements.snapshot()
                    report_handle = session.apply(generation)
                    after_apply = measurements.snapshot()
                    inputs = getattr(narrator._v2, "inputs", None)  # noqa: SLF001 # Read only current source-scope counts.
                    generation_source_resolution = _source_resolution_summary(
                        inputs
                    )
                    before_expanded = measurements.snapshot()
                    expanded_outputs = _measure_handle_outputs(
                        session,
                        generation,
                        report_handle,
                        "expanded",
                        measurements,
                    )
                    after_expanded = measurements.snapshot()
                    before_shared = measurements.snapshot()
                    shared_outputs = _measure_handle_outputs(
                        session,
                        generation,
                        report_handle,
                        "shared",
                        measurements,
                        retain_report_bytes=True,
                    )
                    after_shared = measurements.snapshot()
                    shared_bytes = shared_outputs.pop("_raw_report")
                    after_trusted_outputs = measurements.snapshot()
                    source_summary = shared_outputs["report"]
                    imported_outputs, imported_gates = (
                        _run_shared_import_scenario(
                            shared_bytes,
                            source_summary,
                            measurements,
                        )
                    )
                    del shared_bytes
                    expanded_public, expanded_public_gates = (
                        _run_expanded_public_scenario(
                            session,
                            report_handle,
                            expanded_outputs,
                            measurements,
                        )
                    )
                    require(
                        imported_outputs["logical_report"]["sha256"]
                        == expanded_public["report"]["logical_sha256"],
                        "independent_import_logical_report_parity",
                    )
                    require(
                        imported_outputs["request_identity"]
                        == request_identity,
                        "independent_import_request_identity_mismatch",
                    )
                    proof_labels = (
                        "session.finish_production",
                        "proof.production",
                        "proof.indexed_production",
                        "session.indexed_import_proof",
                    )
                    ledger_labels = ("ledger.build", "ledger.index_build")
                    trusted_gates = {
                        "fresh_capture_initializations": generated_snapshot[
                            "session.capture.initialize"
                        ]["calls"],
                        "generation_complete_proofs": _proof_call_total(
                            generated_snapshot, proof_labels
                        ),
                        "apply_added_complete_proofs": _proof_call_total(
                            after_apply, proof_labels
                        )
                        - _proof_call_total(before_apply, proof_labels),
                        "trusted_output_added_complete_proofs": _proof_call_total(
                            after_trusted_outputs, proof_labels
                        )
                        - _proof_call_total(after_apply, proof_labels),
                        "apply_added_ledger_index_builds": sum(
                            after_apply[label]["calls"]
                            - before_apply[label]["calls"]
                            for label in ledger_labels
                        ),
                        "trusted_output_added_ledger_index_builds": sum(
                            after_trusted_outputs[label]["calls"]
                            - after_apply[label]["calls"]
                            for label in ledger_labels
                        ),
                    }
                    gates = {
                        **trusted_gates,
                        **{
                            f"shared_import_{name}": value
                            for name, value in imported_gates.items()
                        },
                        **{
                            f"expanded_public_{name}": value
                            for name, value in expanded_public_gates.items()
                        },
                        "shared_import_logical_report_matches_expanded_public": True,
                        "shared_import_request_identity_matches_generation": True,
                        "expanded_shared_outputs_used_same_generation_handle": True,
                    }
                    outputs = {
                        "trusted_expanded": expanded_outputs,
                        "trusted_shared": shared_outputs,
                        "independent_shared_import": imported_outputs,
                        "expanded_public": expanded_public,
                        "source_resolution": generation_source_resolution,
                    }
                    path_phase_measurements = {
                        "capture_session": _phase_delta(
                            before_capture, after_capture
                        ),
                        "generate_validated": _phase_delta(
                            before_generation, after_generation
                        ),
                        "trusted_apply": _phase_delta(
                            before_apply, after_apply
                        ),
                        "trusted_expanded_output": _phase_delta(
                            before_expanded, after_expanded
                        ),
                        "trusted_shared_output": _phase_delta(
                            before_shared, after_shared
                        ),
                        **imported_outputs.pop("path_phase_measurements"),
                        **expanded_public.pop("path_phase_measurements"),
                    }
            metrics["request_identity"] = request_identity
            metrics["provider_calls"] = provider.call_count
            metrics["outputs"] = outputs
            metrics["deterministic_gates"] = gates
            require(
                provider.call_count > 0
                if args.cache_state == "cold"
                else provider.call_count == 0,
                "cache_state_provider_call_mismatch",
            )
            metrics["provider_call_gate"] = True
            if args.synthetic_sample == "handle-paired":
                for key in (
                    "apply_added_complete_proofs",
                    "trusted_output_added_complete_proofs",
                    "apply_added_ledger_index_builds",
                    "trusted_output_added_ledger_index_builds",
                ):
                    require(gates[key] == 0, "trusted_output_rebuilt_proof")
                require(
                    gates["fresh_capture_initializations"] == 1,
                    "fresh_capture_count_mismatch",
                )
                require(
                    gates["generation_complete_proofs"] == 1,
                    "generation_complete_proof_count_mismatch",
                )
                require(
                    gates["shared_import_independent_import_complete_proofs"]
                    == 1
                    and gates[
                        "shared_import_independent_import_session_proofs"
                    ]
                    == 1,
                    "shared_import_proof_count_mismatch",
                )
                require(
                    gates["shared_import_apply_added_complete_proofs"] == 0
                    and gates["shared_import_output_added_complete_proofs"]
                    == 0
                    and gates["shared_import_apply_added_ledger_index_builds"]
                    == 0
                    and gates["shared_import_output_added_ledger_index_builds"]
                    == 0,
                    "shared_handle_rebuilt_proof",
                )
            metrics["status"] = "PASS"
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
    except OSError:
        metrics["reason"] = "local_io_failure"
    except Exception as error:  # noqa: BLE001 # Never expose fixture or report data.
        metrics["reason"] = "validation_or_internal_failure"
        metrics["exception_class"] = type(error).__name__
        metrics["exception_type_digest"] = hashlib.sha256(
            type(error).__name__.encode()
        ).hexdigest()
    metrics["elapsed_seconds"] = time.monotonic() - started
    usage = resource.getrusage(resource.RUSAGE_SELF)
    metrics["process_cpu_seconds"] = usage.ru_utime + usage.ru_stime
    metrics["peak_rss_bytes"] = (
        usage.ru_maxrss if sys.platform == "darwin" else usage.ru_maxrss * 1024
    )
    metrics["peak_rss_scope"] = "fresh_process_high_water_mark"
    metrics["phase_measurements"] = measurements.snapshot()
    metrics["path_phase_measurements"] = path_phase_measurements
    metrics["pydantic_python_callbacks"] = measurements.callback_snapshot()
    return metrics


def _pointer_byte_stats(value: object) -> tuple[int, int, Counter[str]]:
    """Count closed storage pointer objects without retaining their targets.

    Preconditions:
        ``value`` is one parsed storage envelope or a table component.
    Postconditions:
        Returns pointer count, serialized pointer-object bytes, and a detached
        reference-count map keyed only by storage identifier.
    Essential behavior:
        Reads structural ``$ref`` objects only; it emits no referenced values.
    """
    count = 0
    byte_count = 0
    references: Counter[str] = Counter()
    pending = [value]
    while pending:
        current = pending.pop()
        if isinstance(current, dict):
            identifier = current.get("$ref")
            if type(identifier) is str:
                count += 1
                byte_count += len(canonical_bytes(current))
                references[identifier] += 1
            pending.extend(current.values())
        elif isinstance(current, (list, tuple)):
            pending.extend(current)
    return count, byte_count, references


def _analyze_shared_fixture(fixture_path: Path) -> dict:
    """Measure byte components and output parity for a supplied shared fixture.

    Preconditions:
        ``fixture_path`` names a caller-selected report-storage.v1 report.
    Postconditions:
        Returns source-free component bytes, table totals, proof timings, and
        exact expanded/shared output hashes.
    Essential behavior:
        Imports the fixture exactly once into a fresh ``ReportSession`` and
        never emits source, prose, request bodies, or the fixture path.
    """
    require(fixture_path.is_file(), "shared_fixture_unavailable")
    raw = fixture_path.read_bytes()
    envelope = json.loads(raw)
    require(
        envelope.get("storage_version") == "diffstory.report-storage.v1"
        and envelope.get("artifact_kind") == "report",
        "m6_fixture_kind_mismatch",
    )
    root_bytes = len(canonical_bytes(envelope["root"]))
    tables_value = envelope["tables"]
    tables_bytes = len(canonical_bytes(tables_value))
    table_bytes = {
        table: len(canonical_bytes(rows))
        for table, rows in sorted(tables_value.items())
    }
    table_row_component_bytes = sum(table_bytes.values())
    record_payload_bytes = sum(
        len(canonical_bytes(entry["record"]))
        for rows in tables_value.values()
        for entry in rows
    )
    pointer_count, pointer_object_bytes, reference_counts = (
        _pointer_byte_stats(envelope)
    )
    direct_duplicate_record_payload_bytes = 0
    for rows in tables_value.values():
        for entry in rows:
            reference_count = reference_counts[entry["id"]]
            if reference_count > 1:
                direct_duplicate_record_payload_bytes += (
                    reference_count - 1
                ) * len(canonical_bytes(entry["record"]))
    shared_input_bytes = len(raw)
    input_digest = hashlib.sha256(raw).hexdigest()
    del envelope

    measurements = PhaseMeasurements()
    with ExitStack() as stack:
        install_measurement_hooks(stack, measurements)
        imported, imported_generation = ReportSession.import_indexed(
            raw,
            max_input_bytes=len(raw),
        )
        with imported:
            report_handle = imported.apply(imported_generation)
            expanded = _measure_stream(
                measurements,
                "output.report.serialize",
                imported.iter_report_json(
                    report_handle,
                    report_storage="expanded",
                    max_output_bytes=SYNTHETIC_MAX_OUTPUT_BYTES,
                ),
            )
            shared = _measure_stream(
                measurements,
                "output.report.serialize",
                imported.iter_report_json(
                    report_handle,
                    report_storage="shared",
                    max_output_bytes=SYNTHETIC_MAX_OUTPUT_BYTES,
                ),
            )
            expanded_html = _measure_stream(
                measurements,
                "output.render.validated",
                iter_render_validated(
                    imported,
                    report_handle,
                    report_storage="expanded",
                    max_output_bytes=SYNTHETIC_MAX_OUTPUT_BYTES,
                ),
                retain=True,
            )
            expanded_html_payload_digest = rendered_report_digest(
                expanded_html.pop("raw")
            )
            logical_report = imported.report_data(report_handle)
            logical_report_digest = digest_json(logical_report)
            with measurements.measure("compat.public_validate"):
                render_module.validate_report(logical_report)
            with measurements.measure("compat.render"):
                public_html_text = render_module.render(logical_report)
            public_html_bytes = public_html_text.encode("utf-8")
            public_html_payload_digest = rendered_report_digest(
                public_html_text
            )
            public_html = {
                "bytes": len(public_html_bytes),
                "sha256": hashlib.sha256(public_html_bytes).hexdigest(),
                "embedded_report_sha256": public_html_payload_digest,
            }
            del logical_report, public_html_text, public_html_bytes
    require(
        expanded_html_payload_digest == logical_report_digest
        and public_html_payload_digest == logical_report_digest,
        "m6_public_render_payload_parity",
    )
    return {
        "input_sha256": input_digest,
        "shared_input_bytes": shared_input_bytes,
        "shared_output_bytes": shared["bytes"],
        "expanded_compact_output_bytes": expanded["bytes"],
        "shared_minus_expanded_bytes": shared["bytes"] - expanded["bytes"],
        "expanded_generation_storage_baseline": "diffstory.generation-storage.v1",
        "root_component_bytes": root_bytes,
        "tables_component_bytes": tables_bytes,
        "table_component_bytes": table_bytes,
        "table_entry_metadata_bytes": table_row_component_bytes
        - record_payload_bytes,
        "table_map_framing_bytes": tables_bytes - table_row_component_bytes,
        "unique_table_record_payload_bytes": record_payload_bytes,
        "shared_non_record_overhead_bytes": shared_input_bytes
        - record_payload_bytes,
        "expanded_compact_bytes_beyond_unique_table_payload": expanded["bytes"]
        - record_payload_bytes,
        "pointer_object_count": pointer_count,
        "pointer_object_bytes_subset": pointer_object_bytes,
        "direct_duplicate_record_payload_bytes_lower_bound": direct_duplicate_record_payload_bytes,
        "envelope_framing_bytes": shared_input_bytes
        - root_bytes
        - tables_bytes,
        "table_record_counts": {
            table: len(rows) for table, rows in sorted(tables_value.items())
        },
        "expanded_html_bytes": expanded_html["bytes"],
        "public_expanded_html_bytes": public_html["bytes"],
        "expanded_html_report_payload_sha256": expanded_html_payload_digest,
        "public_html_report_payload_sha256": public_html_payload_digest,
        "expanded_public_render_payload_parity": True,
        "metric_qualification": (
            "Pointer bytes are a subset of root/table bytes. Direct duplicate "
            "payload is a lower bound because nested reference closures are "
            "excluded; table and record components are actual canonical bytes. "
            "The public renderer validates and expands this same imported report."
        ),
        "phase_measurements": measurements.snapshot(),
    }


def _numeric_range(values: list[int | float]) -> dict[str, int | float]:
    """Summarize a finite numeric sample with median and observed endpoints.

    Preconditions:
        ``values`` contains one or more finite numeric observations.
    Postconditions:
        Returns a median, minimum, and maximum without dropping any sample.
    Essential behavior:
        Keeps three-run spread visible beside each reported median.
    """
    return {
        "median": statistics.median(values),
        "minimum": min(values),
        "maximum": max(values),
    }


def _output_byte_paths(
    value: object, prefix: tuple[str, ...] = ()
) -> dict[str, int]:
    """Collect output byte counts under fixed benchmark result labels.

    Preconditions:
        ``value`` is a source-free output metrics object.
    Postconditions:
        Returns only paths made from fixed scenario and output labels.
    Essential behavior:
        Ignores hashes, identities, prose, and phase timing values.
    """
    if not isinstance(value, dict):
        return {}
    if type(value.get("bytes")) is int and type(value.get("sha256")) is str:
        return {".".join(prefix): value["bytes"]}
    results: dict[str, int] = {}
    for name, item in value.items():
        if type(name) is str:
            results.update(_output_byte_paths(item, (*prefix, name)))
    return results


def _summarize_phase_maps(samples: list[dict], field: str) -> dict:
    """Aggregate phase counters and nested wall/thread CPU measurements.

    Preconditions:
        Every sample has a detached map under ``field`` with fixed phase keys.
    Postconditions:
        Returns median and min/max for each measured numeric phase value.
    Essential behavior:
        Omits uncalled phases and never sums inclusive nested timings.
    """
    phase_names: set[str] = set()
    for sample in samples:
        phase_names.update(sample.get(field, {}))
    summary = {}
    fields = (
        "calls",
        "failures",
        "records",
        "bytes",
        "characters",
        "wall_inclusive_seconds",
        "wall_exclusive_seconds",
        "thread_cpu_inclusive_seconds",
        "thread_cpu_exclusive_seconds",
    )
    for name in sorted(phase_names):
        phase_samples = [sample[field][name] for sample in samples]
        if (
            not any(
                item["calls"] or item["records"] or item["bytes"]
                for item in phase_samples
            )
            and name not in ALWAYS_SUMMARIZE_PHASES
        ):
            continue
        summary[name] = {
            key: _numeric_range([item[key] for item in phase_samples])
            for key in fields
        }
    return summary


def _summarize_path_phase_maps(samples: list[dict]) -> dict:
    """Aggregate boundary-local phase deltas without mixing distinct paths.

    Preconditions:
        Each sample has one map from path label to fixed phase deltas.
    Postconditions:
        Returns separate phase ranges for capture, proof, output, and public
        compatibility boundaries.
    Essential behavior:
        Never folds independent path timings or repeats into one total.
    """
    path_names: set[str] = set()
    for sample in samples:
        path_names.update(sample.get("path_phase_measurements", {}))
    result = {}
    for path in sorted(path_names):
        path_samples = [
            {"phases": sample.get("path_phase_measurements", {}).get(path, {})}
            for sample in samples
        ]
        result[path] = _summarize_phase_maps(path_samples, "phases")
    return result


def _matched_boundary_totals(samples: list[dict]) -> dict:
    """Summarize sequential end-to-end caller boundaries for matched paths.

    Preconditions:
        ``samples`` is one repeated scenario/cache group with path-local phase
        deltas for capture, generation, apply, output, or compatibility calls.
    Postconditions:
        Returns median and range for each complete non-overlapping boundary path.
    Essential behavior:
        Sums only the selected sequential boundary phases per sample, never a
        parent phase together with its nested children.
    """
    selections = {
        "trusted_expanded": (
            ("capture_session", "session.capture.facade"),
            ("generate_validated", "session.generate_validated"),
            ("trusted_apply", "session.apply"),
            ("trusted_expanded_output", "output.annotations.serialize"),
            ("trusted_expanded_output", "output.report.serialize"),
            ("trusted_expanded_output", "output.render.validated"),
        ),
        "trusted_shared": (
            ("capture_session", "session.capture.facade"),
            ("generate_validated", "session.generate_validated"),
            ("trusted_apply", "session.apply"),
            ("trusted_shared_output", "output.annotations.serialize"),
            ("trusted_shared_output", "output.report.serialize"),
            ("trusted_shared_output", "output.render.validated"),
        ),
        "public_compatibility": (
            ("compatibility_generate", "compat.generate"),
            ("compatibility_generation_serialization", "compat.serialize"),
            ("compatibility_apply", "compat.apply"),
            ("compatibility_apply_serialization", "compat.serialize"),
            ("compatibility_render", "compat.render"),
        ),
    }
    scenario = samples[0]["scenario"] if samples else None
    if scenario == "handle-paired":
        selections.pop("public_compatibility")
    elif scenario == "compat-public":
        selections = {
            "public_compatibility": selections["public_compatibility"]
        }
    totals = {}
    for name, phases in selections.items():
        wall_values = []
        cpu_values = []
        for sample in samples:
            paths = sample["metrics"].get("path_phase_measurements", {})
            wall_values.append(
                sum(
                    paths.get(path, {})
                    .get(phase, {})
                    .get("wall_inclusive_seconds", 0.0)
                    for path, phase in phases
                )
            )
            cpu_values.append(
                sum(
                    paths.get(path, {})
                    .get(phase, {})
                    .get("thread_cpu_inclusive_seconds", 0.0)
                    for path, phase in phases
                )
            )
        totals[name] = {
            "wall_seconds": _numeric_range(wall_values),
            "thread_cpu_seconds": _numeric_range(cpu_values),
            "boundary_phases": [phase for _, phase in phases],
        }
    return totals


def _sample_coordinates(sample: dict) -> dict:
    """Return the fixed source-free coordinates for one planned sample.

    Preconditions:
        ``sample`` is a controller row with count, repeat, scenario, cache,
        and profile labels.
    Postconditions:
        Returns only those five coordinate values.
    Essential behavior:
        Keeps failure reports useful without paths, payloads, or child stderr.
    """
    return {name: sample.get(name) for name in SYNTHETIC_COORDINATE_FIELDS}


def _phase_observation_complete(value: object) -> bool:
    """Check one fixed-label phase observation used by aggregate summaries.

    Preconditions:
        ``value`` may be any decoded child object.
    Postconditions:
        Returns true only when every exported count and time is finite.
    Essential behavior:
        Prevents missing phase fields from becoming implicit zero measurements.
    """
    fields = (
        "calls",
        "failures",
        "records",
        "bytes",
        "characters",
        "wall_inclusive_seconds",
        "wall_exclusive_seconds",
        "thread_cpu_inclusive_seconds",
        "thread_cpu_exclusive_seconds",
    )
    return isinstance(value, dict) and all(
        type(value.get(name)) in {int, float}
        and math.isfinite(value[name])
        and value[name] >= 0
        for name in fields
    )


def _path_phase_maps_complete(metrics: dict, required: dict) -> bool:
    """Check that each selected caller boundary contains every measured phase.

    Preconditions:
        ``metrics`` passed the common sample shape check; ``required`` maps
        fixed boundary labels to fixed phase labels.
    Postconditions:
        Returns false if any boundary or selected phase is absent or partial.
    Essential behavior:
        Keeps boundary medians tied to complete per-path phase observations.
    """
    paths = metrics.get("path_phase_measurements")
    return isinstance(paths, dict) and all(
        isinstance(paths.get(path), dict)
        and all(
            _phase_observation_complete(paths[path].get(phase))
            for phase in phases
        )
        for path, phases in required.items()
    )


def _is_sha256_digest(value: object) -> bool:
    """Check one lowercase hexadecimal SHA-256 value.

    Preconditions:
        ``value`` may be any decoded child field.
    Postconditions:
        Returns true only for a 64-character lowercase hexadecimal digest.
    Essential behavior:
        Keeps source-free identities complete without retaining source bytes.
    """
    return (
        type(value) is str
        and len(value) == SHA256_LENGTH
        and all(character in "0123456789abcdef" for character in value)
    )


def _byte_summary_complete(value: object) -> bool:
    """Check one serialized output summary used for byte aggregation.

    Preconditions:
        ``value`` may be any decoded child output field.
    Postconditions:
        Returns true only when nonnegative bytes have an exact SHA-256 digest.
    Essential behavior:
        Prevents missing output bytes from becoming a reduced successful run.
    """
    return (
        isinstance(value, dict)
        and type(value.get("bytes")) is int
        and value["bytes"] >= 0
        and _is_sha256_digest(value.get("sha256"))
    )


def _request_identity_complete(metrics: dict) -> bool:
    """Check the ordered request identity summary for a completed child.

    Preconditions:
        ``metrics`` is a source-free child metric map.
    Postconditions:
        Returns true only when count, digest, and stage counts are present.
    Essential behavior:
        Keeps pair comparisons bound to the complete ordered request set.
    """
    identity = metrics.get("request_identity")
    return (
        isinstance(identity, dict)
        and type(identity.get("request_count")) is int
        and identity["request_count"] >= 0
        and _is_sha256_digest(identity.get("request_identity_sha256"))
        and isinstance(identity.get("request_stage_counts"), dict)
        and all(
            type(name) is str and type(count) is int and count >= 0
            for name, count in identity["request_stage_counts"].items()
        )
    )


def _source_resolution_complete(value: object) -> bool:
    """Check source resolver counts, including distinct physical bindings.

    Preconditions:
        ``value`` may be any decoded source-free summary.
    Postconditions:
        Returns true only when both nonnegative counts are present.
    Essential behavior:
        Prevents a reduced source-resolution summary from entering evidence.
    """
    return isinstance(value, dict) and all(
        type(value.get(name)) is int and value[name] >= 0
        for name in ("resolver_calls", "distinct_current_physical_bindings")
    )


def _common_sample_metrics_complete(sample: dict, metrics: object) -> bool:
    """Check shared numeric, identity, and phase requirements for a sample.

    Preconditions:
        ``sample`` and ``metrics`` are decoded source-free child data.
    Postconditions:
        Returns true only when all common timing and identity fields are valid.
    Essential behavior:
        Rejects partial child rows before scenario-specific inspection.
    """
    if not isinstance(metrics, dict) or (
        sample.get("attempted") is not True
        or sample.get("exit_code") != 0
        or metrics.get("status") != "PASS"
        or any(name not in metrics for name in SYNTHETIC_IDENTITY_FIELDS)
        or not all(
            _is_sha256_digest(metrics.get(name))
            for name in (
                "fixture_sha256",
                "revision_binding_sha256",
                "request_configuration_sha256",
            )
        )
        or not _request_identity_complete(metrics)
        or metrics.get("provider_call_gate") is not True
    ):
        return False
    numeric_fields = (
        "elapsed_seconds",
        "process_cpu_seconds",
        "peak_rss_bytes",
        "provider_calls",
    )
    phases = metrics.get("phase_measurements")
    return (
        all(
            type(metrics.get(name)) in {int, float}
            and math.isfinite(metrics[name])
            and metrics[name] >= 0
            for name in numeric_fields
        )
        and isinstance(phases, dict)
        and set(phases) == set(PHASE_LABELS)
        and all(
            _phase_observation_complete(value) for value in phases.values()
        )
        and isinstance(metrics.get("path_phase_measurements"), dict)
        and isinstance(metrics.get("outputs"), dict)
        and isinstance(metrics.get("deterministic_gates"), dict)
        and metrics.get("peak_rss_scope") == "fresh_process_high_water_mark"
        and metrics.get("process_resource_scope")
        == "whole_selected_scenario_child_process"
    )


def _callback_metrics_complete(metrics: dict) -> bool:
    """Check actual per-code callback counts from a profiled child.

    Preconditions:
        ``metrics`` passed the common source-free sample checks.
    Postconditions:
        Returns true only for a warm callback profile with positive counts.
    Essential behavior:
        Does not treat callback child dispatches as validator invocations.
    """
    callbacks = metrics.get("pydantic_python_callbacks")
    return (
        isinstance(callbacks, dict)
        and bool(callbacks)
        and metrics.get("cache_state") == "warm"
        and metrics.get("provider_calls") == 0
        and all(
            type(name) is str and type(count) is int and count >= 0
            for name, count in callbacks.items()
        )
        and sum(callbacks.values()) > 0
    )


def _compatibility_metrics_complete(metrics: dict) -> bool:
    """Check the expanded public compatibility output summary shape.

    Preconditions:
        ``metrics`` passed the common source-free sample checks.
    Postconditions:
        Returns true only when exact annotation, report, and HTML hashes exist.
    Essential behavior:
        Keeps compatibility parity checks safe for failed child rows.
    """
    outputs = metrics["outputs"]
    if not all(
        isinstance(outputs.get(name), dict)
        for name in ("annotations", "report", "html", "source_resolution")
    ):
        return False
    return (
        all(
            _byte_summary_complete(outputs[name])
            for name in ("annotations", "report", "html")
        )
        and _is_sha256_digest(outputs["annotations"].get("logical_sha256"))
        and _is_sha256_digest(outputs["report"].get("logical_sha256"))
        and _is_sha256_digest(outputs["html"].get("embedded_report_sha256"))
        and _source_resolution_complete(outputs["source_resolution"])
        and metrics["deterministic_gates"] == {}
        and _path_phase_maps_complete(
            metrics,
            {
                "compatibility_generate": ("compat.generate",),
                "compatibility_generation_serialization": (
                    "compat.serialize",
                ),
                "compatibility_apply": ("compat.apply",),
                "compatibility_apply_serialization": ("compat.serialize",),
                "compatibility_render": ("compat.render",),
            },
        )
    )


def _trusted_output_summaries_complete(outputs: dict) -> bool:
    """Check every trusted expanded and shared output byte summary.

    Preconditions:
        Trusted output maps exist in ``outputs``.
    Postconditions:
        Returns true only for all three summaries in both formats.
    Essential behavior:
        Keeps missing output bytes or digests out of successful aggregates.
    """
    expanded = outputs["trusted_expanded"]
    shared = outputs["trusted_shared"]
    if not all(
        isinstance(expanded.get(name), dict)
        and isinstance(shared.get(name), dict)
        for name in ("annotations", "report", "html")
    ):
        return False
    return all(
        _byte_summary_complete(output[name])
        for output in (expanded, shared)
        for name in ("annotations", "report", "html")
    ) and all(
        _is_sha256_digest(expanded[name].get("logical_sha256"))
        for name in ("annotations", "report")
    )


def _imported_output_summaries_complete(outputs: dict) -> bool:
    """Check independent shared import output and logical report summaries.

    Preconditions:
        The independent import output map exists.
    Postconditions:
        Returns true only when every stream and logical hash is present.
    Essential behavior:
        Keeps fresh-import evidence distinct from trusted output evidence.
    """
    imported = outputs["independent_shared_import"]
    return all(
        _byte_summary_complete(imported.get(name))
        for name in ("annotations", "report", "html", "logical_report")
    )


def _public_output_summaries_complete(outputs: dict) -> bool:
    """Check expanded public report and rendered HTML summaries.

    Preconditions:
        The expanded public output map exists.
    Postconditions:
        Returns true only when report and HTML bytes and digests are present.
    Essential behavior:
        Requires the HTML payload digest used by the public parity oracle.
    """
    public = outputs["expanded_public"]
    return (
        all(
            _byte_summary_complete(public.get(name))
            for name in ("report", "html")
        )
        and _is_sha256_digest(public["report"].get("logical_sha256"))
        and _is_sha256_digest(public["html"].get("embedded_report_sha256"))
    )


def _handle_gate_values_complete(gates: dict) -> bool:
    """Check exact trusted proof counts and output parity gates.

    Preconditions:
        ``gates`` is the handle-paired deterministic gate map.
    Postconditions:
        Returns true only when every required count and boolean gate passes.
    Essential behavior:
        Rejects a missing or reduced gate map without weakening acceptance.
    """
    exact_gates = {
        "fresh_capture_initializations": 1,
        "generation_complete_proofs": 1,
        "apply_added_complete_proofs": 0,
        "trusted_output_added_complete_proofs": 0,
        "apply_added_ledger_index_builds": 0,
        "trusted_output_added_ledger_index_builds": 0,
        "shared_import_independent_import_complete_proofs": 1,
        "shared_import_independent_import_session_proofs": 1,
        "shared_import_apply_added_complete_proofs": 0,
        "shared_import_output_added_complete_proofs": 0,
        "shared_import_apply_added_ledger_index_builds": 0,
        "shared_import_output_added_ledger_index_builds": 0,
        "expanded_public_logical_report_matches_trusted_expanded_stream": True,
        "expanded_public_public_html_payload_matches_logical_report": True,
        "shared_import_logical_report_matches_expanded_public": True,
        "shared_import_request_identity_matches_generation": True,
        "expanded_shared_outputs_used_same_generation_handle": True,
    }
    count_gates = (
        "expanded_public_public_expanded_complete_proofs",
        "expanded_public_render_added_complete_proofs",
    )
    return all(
        gates.get(name) == expected for name, expected in exact_gates.items()
    ) and all(
        type(gates.get(name)) is int and gates[name] > 0
        for name in count_gates
    )


def _handle_metrics_complete(metrics: dict) -> bool:
    """Check handle-paired outputs and every required deterministic gate.

    Preconditions:
        ``metrics`` passed the common source-free sample checks.
    Postconditions:
        Returns true only when trusted reuse, import, and parity gates pass.
    Essential behavior:
        Prevents a reduced metric or gate map from entering timing summaries.
    """
    outputs = metrics["outputs"]
    if not all(
        isinstance(outputs.get(name), dict)
        for name in (
            "trusted_expanded",
            "trusted_shared",
            "independent_shared_import",
            "expanded_public",
            "source_resolution",
        )
    ):
        return False
    return (
        _trusted_output_summaries_complete(outputs)
        and _imported_output_summaries_complete(outputs)
        and _public_output_summaries_complete(outputs)
        and _source_resolution_complete(outputs["source_resolution"])
        and _path_phase_maps_complete(
            metrics,
            {
                "capture_session": ("session.capture.facade",),
                "generate_validated": ("session.generate_validated",),
                "trusted_apply": ("session.apply",),
                "trusted_expanded_output": (
                    "output.annotations.serialize",
                    "output.report.serialize",
                    "output.render.validated",
                ),
                "trusted_shared_output": (
                    "output.annotations.serialize",
                    "output.report.serialize",
                    "output.render.validated",
                ),
                "independent_import_proof": (
                    "session.shared_import",
                    "proof.indexed_production",
                ),
                "independent_import_apply_and_output": (
                    "session.apply",
                    "output.report.serialize",
                ),
                "independent_import_expansion": (
                    "benchmark.compatibility_expand",
                ),
                "expanded_public_conversion": (
                    "benchmark.compatibility_expand",
                ),
                "expanded_public_validation": ("compat.public_validate",),
                "expanded_public_render": ("compat.render",),
            },
        )
        and _handle_gate_values_complete(metrics["deterministic_gates"])
    )


def _sample_metrics_complete(sample: dict) -> bool:
    """Check that a child row contains the measurements its scenario needs.

    Preconditions:
        ``sample`` is one source-free timing or callback controller row.
    Postconditions:
        Returns false for failed, partial, reduced, or malformed child metrics.
    Essential behavior:
        Requires all phase labels, identity fields, outputs, and applicable
        deterministic gates before a row can enter an aggregate.
    """
    metrics = sample.get("metrics")
    if not _common_sample_metrics_complete(sample, metrics):
        return False
    if sample.get("profile_kind") == "callbacks":
        return _callback_metrics_complete(metrics)
    if sample.get("profile_kind") != "timing":
        return False
    if sample.get("scenario") == "compat-public":
        return _compatibility_metrics_complete(metrics)
    return sample.get(
        "scenario"
    ) == "handle-paired" and _handle_metrics_complete(metrics)


def _timing_sample_eligible(sample: dict, cold: dict | None = None) -> bool:
    """Require a complete cold/warm observation before timing aggregation.

    Preconditions:
        ``sample`` is a timing row; ``cold`` is its paired cold row if warm.
    Postconditions:
        A warm row is eligible only when the cold baseline and identity pass.
    Essential behavior:
        Prevents a failed cold cache write from becoming valid warm evidence.
    """
    if not _sample_metrics_complete(sample):
        return False
    metrics = sample["metrics"]
    if sample.get("cache_state") == "cold":
        return metrics["provider_calls"] > 0
    if sample.get("cache_state") != "warm" or cold is None:
        return False
    return (
        _sample_metrics_complete(cold)
        and all(
            metrics.get(name) == cold["metrics"].get(name)
            for name in SYNTHETIC_IDENTITY_FIELDS
        )
        and cold["metrics"]["provider_calls"] > 0
        and metrics["provider_calls"] == 0
    )


def _mark_timing_sample_eligibility(samples: list[dict]) -> None:
    """Mark complete timing rows whose cache pair supports comparison.

    Preconditions:
        ``samples`` contains the planned cold and warm timing coordinates.
    Postconditions:
        Each row records a boolean eligibility and fixed rejection reason.
    Essential behavior:
        A warm row cannot enter summaries after an incomplete cold baseline.
    """
    index = _synthetic_sample_index(samples)
    for sample in samples:
        cold = index.get(
            (
                sample.get("count"),
                sample.get("repeat"),
                sample.get("scenario"),
                "cold",
                "timing",
            )
        )
        eligible = _timing_sample_eligible(sample, cold)
        sample["aggregate_eligible"] = eligible
        if eligible:
            sample["aggregate_reason"] = None
        elif not _sample_metrics_complete(sample):
            sample["aggregate_reason"] = "sample_metrics_incomplete"
        elif sample.get(
            "cache_state"
        ) == "warm" and not _sample_metrics_complete(cold or {}):
            sample["aggregate_reason"] = "cold_sample_incomplete"
        elif sample.get("cache_state") == "cold":
            sample["aggregate_reason"] = "cold_provider_call_gate_failed"
        elif sample.get("cache_state") == "warm":
            sample["aggregate_reason"] = "warm_cache_pair_failed"
        else:
            sample["aggregate_reason"] = "invalid_cache_state"


def _group_synthetic_samples(samples: list[dict], expected_count: int) -> dict:
    """Build source-free summaries for one repeated scenario/cache group.

    Preconditions:
        ``samples`` contains every planned row for one scenario/cache group.
    Postconditions:
        Returns ranges only for complete eligible rows and identifies every
        omitted coordinate.
    Essential behavior:
        Never creates zero-success medians for an empty or incomplete group.
    """
    valid_samples = [
        sample
        for sample in samples
        if sample.get("aggregate_eligible") is True
        and _sample_metrics_complete(sample)
    ]
    failed_coordinates = [
        {
            **_sample_coordinates(sample),
            "reason": sample.get("aggregate_reason")
            or "sample_metrics_incomplete",
        }
        for sample in samples
        if sample not in valid_samples
    ]
    complete = (
        len(samples) == expected_count and len(valid_samples) == expected_count
    )
    result = {
        "status": "PASS" if complete else "INCOMPLETE",
        "expected_sample_count": expected_count,
        "attempted_sample_count": sum(
            sample.get("attempted") is True for sample in samples
        ),
        "valid_sample_count": len(valid_samples),
        "failed_sample_coordinates": failed_coordinates,
        "child_exit_codes": [sample.get("exit_code") for sample in samples],
    }
    if not valid_samples:
        return result
    outputs: dict[str, list[int]] = {}
    for sample in valid_samples:
        for name, byte_count in _output_byte_paths(
            sample["metrics"].get("outputs", {})
        ).items():
            outputs.setdefault(name, []).append(byte_count)
    gate_maps = [
        sample["metrics"]["deterministic_gates"] for sample in valid_samples
    ]
    result.update(
        {
            "process_wall_seconds": _numeric_range(
                [
                    sample["metrics"]["elapsed_seconds"]
                    for sample in valid_samples
                ]
            ),
            "process_cpu_seconds": _numeric_range(
                [
                    sample["metrics"]["process_cpu_seconds"]
                    for sample in valid_samples
                ]
            ),
            "process_peak_rss_bytes": _numeric_range(
                [
                    sample["metrics"]["peak_rss_bytes"]
                    for sample in valid_samples
                ]
            ),
            "provider_calls": _numeric_range(
                [
                    sample["metrics"]["provider_calls"]
                    for sample in valid_samples
                ]
            ),
            "output_bytes": {
                name: _numeric_range(values)
                for name, values in sorted(outputs.items())
            },
            "deterministic_gates_identical": all(
                gates == gate_maps[0] for gates in gate_maps[1:]
            ),
            "deterministic_gates": gate_maps[0],
            "phase_measurements": _summarize_phase_maps(
                [sample["metrics"] for sample in valid_samples],
                "phase_measurements",
            ),
            "path_phase_measurements": _summarize_path_phase_maps(
                [sample["metrics"] for sample in valid_samples]
            ),
            "matched_boundary_paths": _matched_boundary_totals(valid_samples),
        }
    )
    return result


def _run_synthetic_child(
    scenario: str,
    count: int,
    cache_state: str,
    cache_dir: Path,
    *,
    profile_callbacks: bool,
) -> tuple[dict, int]:
    """Run one source-free scenario in a fresh Python process.

    Preconditions:
        The scenario and count are inside the bounded local matrix, and the
        private scratch cache directory exists.
    Postconditions:
        Returns one parsed source-free child metric map and its exact exit code.
    Essential behavior:
        Sends no report path or provider body to the child process.
    """
    command = [
        sys.executable,
        "-B",
        str(Path(__file__).resolve()),
        "--synthetic-sample",
        scenario,
        "--count",
        str(count),
        "--cache-state",
        cache_state,
        "--cache-dir",
        str(cache_dir),
    ]
    if profile_callbacks:
        command.append("--profile-callbacks")
    try:
        completed = subprocess.run(  # noqa: S603 # All executable arguments come from this closed enum and bounded counts.
            command,
            cwd=ROOT,
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=1800,
        )
    except subprocess.TimeoutExpired:
        return {"status": "BLOCKED", "reason": "child_timeout"}, 124
    except OSError:
        return {"status": "BLOCKED", "reason": "child_process_failed"}, 125
    try:
        metrics = json.loads(completed.stdout)
    except (json.JSONDecodeError, TypeError):
        return {
            "status": "BLOCKED",
            "reason": "child_metrics_invalid",
        }, completed.returncode
    if type(metrics) is not dict:
        return {
            "status": "BLOCKED",
            "reason": "child_metrics_invalid",
        }, completed.returncode
    return metrics, completed.returncode


SYNTHETIC_IDENTITY_FIELDS = (
    "fixture_sha256",
    "revision_binding_sha256",
    "request_configuration_sha256",
    "request_identity",
)


def _synthetic_sample_index(samples: list[dict]) -> dict[tuple, dict]:
    """Index timing and callback samples by their fixed matrix coordinates.

    Preconditions:
        Rows contain count, repeat, scenario, state, and profile-kind labels.
    Postconditions:
        Returns rows keyed only by those five fixed coordinates.
    Essential behavior:
        Makes paired evidence lookup explicit and does not inspect child text.
    """
    return {
        (
            row["count"],
            row["repeat"],
            row["scenario"],
            row["cache_state"],
            row["profile_kind"],
        ): row
        for row in samples
    }


def _validate_cache_pair(
    index: dict[tuple, dict],
    count: int,
    repeat: int,
    scenario: str,
    failures: list[str],
) -> None:
    """Check one cold, warm, and profiled sample identity and cache gate.

    Preconditions:
        ``index`` contains the complete sample matrix or a missing row.
    Postconditions:
        Appends only fixed failure codes for missing or mismatched evidence.
    Essential behavior:
        Requires cold provider activity, zero warm activity, and nonempty
        separately profiled Python Pydantic callback counts.
    """
    cold = index.get((count, repeat, scenario, "cold", "timing"))
    warm = index.get((count, repeat, scenario, "warm", "timing"))
    profiled = index.get((count, 1, scenario, "warm", "callbacks"))
    if cold is None or warm is None or profiled is None:
        failures.append("paired_sample_missing")
        return
    if not all(
        _sample_metrics_complete(sample) for sample in (cold, warm, profiled)
    ):
        failures.append("paired_sample_incomplete")
        return
    cold_metrics = cold["metrics"]
    warm_metrics = warm["metrics"]
    profile_metrics = profiled["metrics"]
    if not all(
        cold_metrics.get(field) == warm_metrics.get(field)
        for field in SYNTHETIC_IDENTITY_FIELDS
    ):
        failures.append("cold_warm_identity_mismatch")
    if not all(
        warm_metrics.get(field) == profile_metrics.get(field)
        for field in SYNTHETIC_IDENTITY_FIELDS
    ):
        failures.append("callback_identity_mismatch")
    if cold_metrics.get("provider_calls", 0) <= 0:
        failures.append("cold_provider_call_gate_failed")
    if warm_metrics.get("provider_calls") != 0:
        failures.append("warm_provider_call_gate_failed")
    callback_counts = profile_metrics.get("pydantic_python_callbacks", {})
    if not callback_counts or sum(callback_counts.values()) <= 0:
        failures.append("pydantic_callback_gate_failed")


def _validate_compatibility_pair(
    index: dict[tuple, dict], count: int, repeat: int, failures: list[str]
) -> None:
    """Check one matched trusted/public expanded workload and exact semantics.

    Preconditions:
        The cold timing rows for both current paths were attempted.
    Postconditions:
        Appends fixed failures for identity, compact-size, or logical-content
        differences without exposing any report fields.
    Essential behavior:
        Compares canonical field/presence/array semantics and compact byte totals.
    """
    trusted = index.get((count, repeat, "handle-paired", "cold", "timing"))
    compatibility = index.get(
        (count, repeat, "compat-public", "cold", "timing")
    )
    if trusted is None or compatibility is None:
        failures.append("compatibility_pair_missing")
        return
    if not all(
        _sample_metrics_complete(sample) for sample in (trusted, compatibility)
    ):
        failures.append("compatibility_pair_incomplete")
        return
    trusted_metrics = trusted["metrics"]
    compatibility_metrics = compatibility["metrics"]
    if not all(
        trusted_metrics.get(field) == compatibility_metrics.get(field)
        for field in SYNTHETIC_IDENTITY_FIELDS
    ):
        failures.append("compatibility_request_identity_mismatch")
        return
    trusted_outputs = trusted_metrics["outputs"]["trusted_expanded"]
    compatibility_outputs = compatibility_metrics["outputs"]
    if not all(
        isinstance(value, dict)
        for value in (
            trusted_outputs.get("annotations"),
            trusted_outputs.get("report"),
            trusted_outputs.get("html"),
            compatibility_outputs.get("annotations"),
            compatibility_outputs.get("report"),
            compatibility_outputs.get("html"),
        )
    ):
        failures.append("compatibility_outputs_incomplete")
        return
    semantic_pairs = (
        (
            trusted_outputs["annotations"]["logical_sha256"],
            compatibility_outputs["annotations"]["logical_sha256"],
        ),
        (
            trusted_outputs["report"]["logical_sha256"],
            compatibility_outputs["report"]["logical_sha256"],
        ),
        (
            trusted_outputs["report"]["logical_sha256"],
            compatibility_outputs["html"]["embedded_report_sha256"],
        ),
    )
    byte_pairs = (
        (
            trusted_outputs["annotations"]["bytes"],
            compatibility_outputs["annotations"]["bytes"],
        ),
        (
            trusted_outputs["report"]["bytes"],
            compatibility_outputs["report"]["bytes"],
        ),
        (
            trusted_outputs["html"]["bytes"],
            compatibility_outputs["html"]["bytes"],
        ),
    )
    if any(left != right for left, right in semantic_pairs + byte_pairs):
        failures.append("compatibility_semantic_output_mismatch")


def _validate_synthetic_matrix(
    timing_samples: list[dict],
    callback_samples: list[dict],
    counts: tuple[int, ...],
    repeats: int,
) -> list[str]:
    """Check repeated cache, request, callback, and compatibility pair gates.

    Preconditions:
        The finite timing and separate callback sample rows have been collected.
    Postconditions:
        Returns fixed failure codes only; no child stderr or input data escapes.
    Essential behavior:
        Requires identical fixture, revisions, request configuration, request
        identities, and exact compatibility semantic digests across paired paths.
    """
    expected_timing_keys = {
        (count, repeat, scenario, cache_state, "timing")
        for count in counts
        for repeat in range(1, repeats + 1)
        for scenario in SYNTHETIC_SCENARIOS
        for cache_state in ("cold", "warm")
    }
    expected_callback_keys = {
        (count, 1, scenario, "warm", "callbacks")
        for count in counts
        for scenario in SYNTHETIC_SCENARIOS
    }
    actual_timing_keys = {
        tuple(row.get(name) for name in SYNTHETIC_COORDINATE_FIELDS)
        for row in timing_samples
    }
    actual_callback_keys = {
        tuple(row.get(name) for name in SYNTHETIC_COORDINATE_FIELDS)
        for row in callback_samples
    }
    failures = [
        *(
            ["timing_sample_coordinates_mismatch"]
            if len(timing_samples) != len(expected_timing_keys)
            or actual_timing_keys != expected_timing_keys
            else []
        ),
        *(
            ["callback_sample_coordinates_mismatch"]
            if len(callback_samples) != len(expected_callback_keys)
            or actual_callback_keys != expected_callback_keys
            else []
        ),
    ]
    all_rows = [*timing_samples, *callback_samples]
    for row in all_rows:
        metrics = row.get("metrics")
        if (
            row.get("attempted") is not True
            or row.get("exit_code") != 0
            or not isinstance(metrics, dict)
            or metrics.get("status") != "PASS"
        ):
            failures.append("child_sample_failed")
        if not _sample_metrics_complete(row):
            failures.append("child_sample_incomplete")
    index = _synthetic_sample_index(all_rows)
    for count in counts:
        for repeat in range(1, repeats + 1):
            for scenario in SYNTHETIC_SCENARIOS:
                _validate_cache_pair(index, count, repeat, scenario, failures)
            _validate_compatibility_pair(index, count, repeat, failures)
    gate_groups: dict[tuple, list[dict]] = {}
    for row in all_rows:
        key = (row.get("count"), row.get("scenario"), row.get("profile_kind"))
        if _sample_metrics_complete(row):
            gate_groups.setdefault(key, []).append(
                row["metrics"]["deterministic_gates"]
            )
    if any(
        any(gates != gate_maps[0] for gates in gate_maps[1:])
        for gate_maps in gate_groups.values()
    ):
        failures.append("deterministic_gate_mismatch")
    return sorted(set(failures))


def _build_synthetic_aggregates(
    timing_samples: list[dict], counts: tuple[int, ...], repeats: int
) -> dict:
    """Aggregate only finite repeated timing samples by count and cache state.

    Preconditions:
        ``timing_samples`` contains per-process metrics with no private paths.
    Postconditions:
        Returns process and nested phase medians with minimum/maximum spread.
    Essential behavior:
        Keeps repeated, profiled callback runs out of ordinary timing medians.
    """
    _mark_timing_sample_eligibility(timing_samples)
    groups = {}
    for count in counts:
        for scenario in SYNTHETIC_SCENARIOS:
            for cache_state in ("cold", "warm"):
                rows = [
                    row
                    for row in timing_samples
                    if row["count"] == count
                    and row["scenario"] == scenario
                    and row["cache_state"] == cache_state
                ]
                groups[f"{count}/{scenario}/{cache_state}"] = (
                    _group_synthetic_samples(rows, repeats)
                )
    return {
        "repeats_per_timing_group": repeats,
        "groups": groups,
    }


def _markdown_process_results(value: dict) -> list[str]:
    """Render process ranges without fabricating missing observations.

    Preconditions:
        value contains source-free aggregates for planned count groups.
    Postconditions:
        Returns one process summary row per requested group.
    Essential behavior:
        Empty groups display an em dash instead of a zero-valued median.
    """
    lines = [
        "## Process results",
        "",
        "| Count | Scenario | Cache | Status and rows | Wall median [min, max] | CPU median [min, max] | Peak RSS median [min, max] | Provider calls |",
        "|---:|---|---|---|---:|---:|---:|---:|",
    ]
    for key, group in sorted(value["aggregates"]["groups"].items()):
        count, scenario, cache_state = key.split("/", 2)
        wall = group.get("process_wall_seconds")
        cpu = group.get("process_cpu_seconds")
        rss = group.get("process_peak_rss_bytes")
        calls = group.get("provider_calls")
        if wall is None or cpu is None or rss is None or calls is None:
            lines.append(
                f"| {count} | {scenario} | {cache_state} | "
                f"{group['status']} ({group['valid_sample_count']}/"
                f"{group['expected_sample_count']}) | — | — | — | — |"
            )
            continue
        lines.append(
            f"| {count} | {scenario} | {cache_state} | "
            f"{group['status']} ({group['valid_sample_count']}/"
            f"{group['expected_sample_count']}) | "
            f"{wall['median']:.4f} [{wall['minimum']:.4f}, {wall['maximum']:.4f}] | "
            f"{cpu['median']:.4f} [{cpu['minimum']:.4f}, {cpu['maximum']:.4f}] | "
            f"{rss['median']} [{rss['minimum']}, {rss['maximum']}] | "
            f"{calls['median']} |"
        )
    return lines


def _markdown_output_results(value: dict) -> list[str]:
    """Render exact expanded/shared report sizes where complete rows exist.

    Preconditions:
        value contains aggregates grouped by synthetic count and cache.
    Postconditions:
        Returns a byte comparison for every requested count.
    Essential behavior:
        Incomplete groups remain visible without substituting zero-byte values.
    """
    lines = [
        "",
        "## Same-proof output bytes",
        "",
        "Trusted expanded and shared output use one generation proof per sample. The expanded size is compact JSON with generation-storage.v1 semantics.",
        "",
        "| Count | Expanded report bytes median [min, max] | Shared report bytes median [min, max] | Expanded minus shared bytes |",
        "|---:|---:|---:|---:|",
    ]
    groups = value["aggregates"]["groups"]
    for count in value["counts"]:
        group = groups[f"{count}/handle-paired/cold"]
        outputs = group.get("output_bytes", {})
        expanded = outputs.get("trusted_expanded.report")
        shared = outputs.get("trusted_shared.report")
        if expanded is None or shared is None:
            lines.append(f"| {count} | — | — | — |")
            continue
        lines.append(
            f"| {count} | {expanded['median']} "
            f"[{expanded['minimum']}, {expanded['maximum']}] | "
            f"{shared['median']} [{shared['minimum']}, {shared['maximum']}] | "
            f"{expanded['median'] - shared['median']} |"
        )
    return lines


def _markdown_boundary_results(value: dict) -> list[str]:
    """Render matched caller-boundary wall and thread CPU ranges.

    Preconditions:
        value contains per-path aggregates for each scenario and cache.
    Postconditions:
        Returns only measured totals; missing comparisons have no numeric row.
    Essential behavior:
        States that public annotation and report serialization are included.
    """
    lines = [
        "",
        "## Matched boundary timings",
        "",
        "Totals use inclusive time for selected sequential phases and do not add nested child times. The public compatibility total includes standalone annotation and report JSON serialization, plus its independent generate, apply, validate, and render checks.",
        "",
        "| Count | Cache | Path | Wall median [min, max] | Thread CPU median [min, max] |",
        "|---:|---|---|---:|---:|",
    ]
    groups = value["aggregates"]["groups"]
    for count in value["counts"]:
        for cache_state in ("cold", "warm"):
            for scenario in SYNTHETIC_SCENARIOS:
                group = groups[f"{count}/{scenario}/{cache_state}"]
                for path, summary in group.get(
                    "matched_boundary_paths", {}
                ).items():
                    wall = summary["wall_seconds"]
                    cpu = summary["thread_cpu_seconds"]
                    lines.append(
                        f"| {count} | {cache_state} | {path} | "
                        f"{wall['median']:.4f} "
                        f"[{wall['minimum']:.4f}, {wall['maximum']:.4f}] | "
                        f"{cpu['median']:.4f} "
                        f"[{cpu['minimum']:.4f}, {cpu['maximum']:.4f}] |"
                    )
    return lines


def _markdown_shared_fixture_results(value: dict) -> list[str]:
    """Render optional caller-supplied shared fixture analysis.

    Preconditions:
        value contains None or source-free component and phase metrics.
    Postconditions:
        Returns a pending note or the complete component comparison.
    Essential behavior:
        Reports actual byte increases as well as savings without a fixture path.
    """
    exact = value.get("shared_fixture_analysis")
    if exact is None:
        return [
            "",
            "## Optional shared fixture",
            "",
            "No shared report fixture was supplied; component-level byte analysis was not run.",
        ]
    shared_bytes = exact["shared_output_bytes"]
    expanded_bytes = exact["expanded_compact_output_bytes"]
    saved = expanded_bytes - shared_bytes
    saved_percent = (100 * saved / expanded_bytes) if expanded_bytes else 0.0
    phases = exact["phase_measurements"]
    import_seconds = phases["session.shared_import"]["wall_inclusive_seconds"]
    indexed_seconds = phases["proof.indexed_production"][
        "wall_inclusive_seconds"
    ]
    parse_seconds = phases["shared.storage_parse"]["wall_inclusive_seconds"]
    validation_seconds = phases["compat.public_validate"][
        "wall_inclusive_seconds"
    ]
    render_seconds = phases["compat.render"]["wall_inclusive_seconds"]
    lines = [
        "",
        "## Caller-supplied shared fixture bytes",
        "",
        f"The shared input was {exact['shared_input_bytes']} bytes. Its shared output was {shared_bytes} bytes and its expanded compact report output was {expanded_bytes} bytes: {saved} bytes ({saved_percent:.2f}%) saved. A negative value means shared storage used more bytes.",
        f"The envelope had {exact['root_component_bytes']} canonical root bytes and {exact['tables_component_bytes']} canonical table bytes. Unique record payload was {exact['unique_table_record_payload_bytes']} bytes; pointer objects occupied {exact['pointer_object_bytes_subset']} bytes across {exact['pointer_object_count']} references. The direct duplicate payload estimate is a lower bound: {exact['direct_duplicate_record_payload_bytes_lower_bound']} bytes. Pointer bytes are a subset of root/table bytes.",
        f"The shared envelope carried {exact['shared_non_record_overhead_bytes']} non-record-payload bytes. The expanded compact generation-storage.v1 baseline was {exact['expanded_compact_bytes_beyond_unique_table_payload']} bytes larger than the unique shared table record payload. This explains the net size result; the duplicate-payload lower bound is not a net-saving claim because generation-storage.v1 already compacts generation data.",
        f"Expanded and public-render HTML payload parity passed: {exact['expanded_public_render_payload_parity']}.",
        "",
        "## Shared fixture phase comparison",
        "",
        f"Full shared import took {import_seconds:.2f} seconds; its indexed production proof took {indexed_seconds:.2f} seconds, while shared parsing took {parse_seconds:.2f} seconds. The indexed proof is nested inside import and is not added to it. Explicit public validation took {validation_seconds:.2f} seconds; public rendering took {render_seconds:.2f} seconds and performs its own validation.",
        "",
        "| Phase | Calls | Wall seconds | Thread CPU seconds |",
        "|---|---:|---:|---:|",
    ]
    for name in (
        "session.shared_import",
        "shared.storage_parse",
        "proof.indexed_production",
        "session.indexed_import_proof",
        "storage.unique_row_validation",
        "compat.public_validate",
        "compat.render",
    ):
        phase = exact["phase_measurements"][name]
        lines.append(
            f"| {name} | {phase['calls']} | "
            f"{phase['wall_inclusive_seconds']:.4f} | "
            f"{phase['thread_cpu_inclusive_seconds']:.4f} |"
        )
    return lines


def _markdown_callback_results(value: dict) -> list[str]:
    """Render actual per-code Python validator callback totals.

    Preconditions:
        value contains separate callback-traced child rows.
    Postconditions:
        Returns one row for each complete callback observation.
    Essential behavior:
        Does not count child dispatches as validator callbacks.
    """
    lines = [
        "",
        "## Python validator callbacks",
        "",
        "Callback counts come from separate warm-cache callback-traced subprocesses. Their elapsed time, CPU, and RSS are excluded from timing tables.",
        "",
        "| Count | Scenario | Field callbacks | Model callbacks | Total callbacks |",
        "|---:|---|---:|---:|---:|",
    ]
    rows = [
        row
        for row in value["callback_samples"]
        if _sample_metrics_complete(row)
        and row.get("exit_code") == 0
        and row.get("attempted") is True
    ]
    if not rows:
        lines.append("| — | — | — | — | No complete callback sample |")
        return lines
    for row in rows:
        callbacks = row["metrics"]["pydantic_python_callbacks"]
        field_count = sum(
            count
            for name, count in callbacks.items()
            if name.startswith("field.")
        )
        model_count = sum(
            count
            for name, count in callbacks.items()
            if name.startswith("model.")
        )
        lines.append(
            f"| {row['count']} | {row['scenario']} | {field_count} | "
            f"{model_count} | {field_count + model_count} |"
        )
    return lines


def _markdown_failure_results(value: dict) -> list[str]:
    """Render failed and skipped sample coordinates with fixed reasons.

    Preconditions:
        value contains matrix failure codes and source-free sample summaries.
    Postconditions:
        Returns failure codes and every failed, skipped, or ineligible coordinate.
    Essential behavior:
        Displays no child stderr, private paths, request bodies, or prose.
    """
    if not value.get("matrix_failures") and not value.get("failed_samples"):
        return []
    lines = ["", "## Failed gates", ""]
    lines.extend(f"- {item}" for item in value.get("matrix_failures", []))
    if value.get("failed_samples"):
        lines.extend(
            [
                "",
                "| Count | Repeat | Scenario | Cache | Profile | Attempted | Exit | Status | Reason |",
                "|---:|---:|---|---|---|---:|---:|---|---|",
            ]
        )
        lines.extend(
            (
                f"| {sample['count']} | {sample['repeat']} | "
                f"{sample['scenario']} | {sample['cache_state']} | "
                f"{sample['profile_kind']} | {sample['attempted']} | "
                f"{sample['exit_code']} | {sample['status']} | "
                f"{sample['reason']} |"
            )
            for sample in value["failed_samples"]
        )
    return lines


def _result_markdown(value: dict, command: str) -> str:
    """Render complete or partial source-free benchmark evidence.

    Preconditions:
        value contains fixed labels and possibly incomplete child rows.
    Postconditions:
        Returns a reviewable Markdown result without inventing absent metrics.
    Essential behavior:
        Keeps the exact caller command only in this private result artifact.
    """
    lines = [
        "# Synthetic Python benchmark evidence",
        "",
        f"Status: {value['status']}",
        "",
        "## Run",
        "",
        "Command:",
        "",
        f"    {command}",
        "",
        f"Parent exit code: {value['parent_exit_code']}.",
        f"Timing rows: {value['timing_sample_count']} "
        f"({value['timing_attempted_child_count']} child runs); callback rows: "
        f"{value['callback_sample_count']} "
        f"({value['callback_attempted_child_count']} child runs).",
        f"Counts: {', '.join(str(item) for item in value['counts'])}; "
        f"repeats per timing group: {value['repeats']}.",
        "Process RSS is the fresh child process high-water mark. It is not per-phase memory.",
        "",
        "## Source hashes",
        "",
    ]
    lines.extend(
        f"- {label}: {digest}"
        for label, digest in sorted(value["source_hashes"].items())
    )
    lines.extend(
        ["", f"Progress: {value['progress'].get('label', 'unknown')}."]
    )
    lines.extend(_markdown_process_results(value))
    lines.extend(_markdown_output_results(value))
    lines.extend(_markdown_boundary_results(value))
    lines.extend(_markdown_shared_fixture_results(value))
    lines.extend(_markdown_callback_results(value))
    lines.extend(
        [
            "",
            "## Timed phase evidence",
            "",
            "Detailed phase and boundary medians and ranges are in measurements.json. Inclusive nested times are not summed. Callback tracing is separate from timing summaries.",
            "",
            "## Qualifications",
            "",
        ]
    )
    lines.extend(f"- {item}" for item in value["qualifications"])
    lines.extend(_markdown_failure_results(value))
    return "\n".join(lines) + "\n"


def _run_synthetic_child_row(
    coordinates: dict,
    cache_dir: Path,
) -> dict:
    """Return one attempted child row with fixed coordinates and exact status.

    Preconditions:
        The caller selected a planned synthetic scenario and cache directory.
    Postconditions:
        Returns child metrics and exact exit status, or a fixed local failure.
    Essential behavior:
        Never retains stderr, exception text, paths, or request data.
    """
    try:
        metrics, exit_code = _run_synthetic_child(
            coordinates["scenario"],
            coordinates["count"],
            coordinates["cache_state"],
            cache_dir,
            profile_callbacks=coordinates["profile_kind"] == "callbacks",
        )
    except Exception:  # noqa: BLE001 # Keep parent evidence source-free on child-controller errors.
        metrics, exit_code = (
            {
                "status": "BLOCKED",
                "reason": "child_controller_failure",
            },
            125,
        )
    return {
        **coordinates,
        "attempted": True,
        "exit_code": exit_code,
        "metrics": metrics,
    }


def _skipped_synthetic_child_row(
    coordinates: dict,
    reason: str,
) -> dict:
    """Represent an unrun coordinate after its required baseline failed.

    Preconditions:
        The cold or warm child needed by this coordinate did not pass.
    Postconditions:
        Returns a source-free planned row with no fabricated exit status.
    Essential behavior:
        Preserves a full coordinate matrix without treating skipped work as data.
    """
    return {
        **coordinates,
        "attempted": False,
        "exit_code": None,
        "metrics": {"status": "BLOCKED", "reason": reason},
    }


def _sample_succeeded(row: dict, cold: dict | None = None) -> bool:
    """Check whether one attempted sample is complete and passed its child gates.

    Preconditions:
        ``row`` contains a planned sample coordinate.
    Postconditions:
        Returns true only for a complete metric map with exact zero exit.
    Essential behavior:
        Makes later warm/profile scheduling depend on a valid prerequisite.
    """
    return (
        row.get("attempted") is True
        and row.get("exit_code") == 0
        and _sample_metrics_complete(row)
        and (
            row["metrics"]["provider_calls"] == 0
            if row.get("profile_kind") == "callbacks"
            else _timing_sample_eligible(row, cold)
        )
    )


def _failed_sample_summaries(samples: list[dict]) -> list[dict]:
    """Return failed, skipped, or aggregation-ineligible sample coordinates.

    Preconditions:
        ``samples`` contains timing and callback rows from one controller run.
    Postconditions:
        Returns coordinates, attempt flag, exact exit, status, and fixed reason.
    Essential behavior:
        Excludes child metric bodies, exception text, paths, and stderr.
    """
    failures = []
    for sample in samples:
        metrics = sample.get("metrics")
        valid = _sample_metrics_complete(sample)
        if sample.get("profile_kind") == "timing":
            valid = valid and sample.get("aggregate_eligible") is True
        if valid and sample.get("exit_code") == 0:
            continue
        failures.append(
            {
                **_sample_coordinates(sample),
                "attempted": sample.get("attempted") is True,
                "exit_code": sample.get("exit_code"),
                "status": metrics.get("status")
                if isinstance(metrics, dict)
                else "INVALID",
                "reason": metrics.get("reason")
                if isinstance(metrics, dict)
                else "sample_metrics_invalid",
            }
        )
    return failures


def _synthetic_command(args: argparse.Namespace) -> str:
    """Build the exact caller command for the private Markdown result.

    Preconditions:
        ``args`` contains the selected counts, repeats, output directory, and
        optional shared fixture.
    Postconditions:
        Returns a shell-quoted command without storing it in measurements JSON.
    Essential behavior:
        Uses only caller-provided paths and a generic repository command.
    """
    command = [
        ".venv/bin/python",
        "-B",
        "scripts/benchmark_narration.py",
        "--synthetic",
        "--counts",
        args.counts,
        "--repeats",
        str(args.repeats),
        "--output-dir",
        str(args.output_dir),
    ]
    shared_fixture = getattr(args, "shared_fixture", None)
    if shared_fixture is not None:
        command.extend(("--shared-fixture", str(shared_fixture)))
    return shlex.join(command)


def _write_synthetic_result(
    output_directory: Path, result: dict, command: str
) -> None:
    """Persist a source-free measurement checkpoint and private Markdown view.

    Preconditions:
        ``result`` contains only benchmark rows, fixed labels, and aggregates.
    Postconditions:
        Both artifacts describe the same current partial or final matrix.
    Essential behavior:
        Replaces the JSON atomically and restricts artifact permissions.
    """
    write_metrics(output_directory / "measurements.json", result)
    report_path = output_directory / "result.md"
    report_path.write_text(_result_markdown(result, command), encoding="utf-8")
    report_path.chmod(0o600)


def _synthetic_result(
    args: argparse.Namespace,
    samples: dict,
    controller: dict,
) -> dict:
    """Build a source-free complete or partial controller result.

    Preconditions:
        Rows contain only fixed coordinates and source-free child metrics.
    Postconditions:
        Returns PASS only for the entire requested coordinate and gate matrix.
    Essential behavior:
        Retains partial rows and successful-row aggregates without filling
        missing evidence with zero-valued medians.
    """
    timing_samples = samples["timing"]
    callback_samples = samples["callbacks"]
    matrix_failures = controller["matrix_failures"]
    counts = parse_synthetic_counts(args.counts)
    aggregates = _build_synthetic_aggregates(
        timing_samples, counts, args.repeats
    )
    expected_callbacks = len(counts) * len(SYNTHETIC_SCENARIOS)
    matrix_complete = (
        not matrix_failures
        and all(
            group["status"] == "PASS"
            for group in aggregates["groups"].values()
        )
        and len(callback_samples) == expected_callbacks
        and all(
            sample.get("attempted") is True
            and sample.get("exit_code") == 0
            and _sample_metrics_complete(sample)
            for sample in callback_samples
        )
    )
    status = "PASS" if matrix_complete else "BLOCKED"
    all_samples = [*timing_samples, *callback_samples]
    return {
        "version": "diffstory.synthetic-benchmark.v1",
        "status": status,
        "parent_exit_code": 0 if matrix_complete else 2,
        "counts": counts,
        "repeats": args.repeats,
        "scenarios": SYNTHETIC_SCENARIOS,
        "timing_sample_count": len(timing_samples),
        "timing_attempted_child_count": sum(
            sample.get("attempted") is True for sample in timing_samples
        ),
        "callback_sample_count": len(callback_samples),
        "callback_attempted_child_count": sum(
            sample.get("attempted") is True for sample in callback_samples
        ),
        "timing_samples": timing_samples,
        "callback_samples": callback_samples,
        "aggregates": aggregates,
        "pairing_gates": {
            "all_request_fixture_revision_and_configuration_pairs_match": matrix_complete,
            "compatibility_semantic_outputs_match": matrix_complete,
            "cold_provider_calls_positive_and_warm_calls_zero": matrix_complete,
            "profiled_runs_have_pydantic_callbacks": matrix_complete,
            "all_expected_coordinates_complete": matrix_complete,
            "all_deterministic_gate_maps_identical": matrix_complete,
        },
        "matrix_failures": sorted(set(matrix_failures)),
        "failed_samples": _failed_sample_summaries(all_samples),
        "source_hashes": controller["source_hashes"],
        "shared_fixture_analysis": controller["shared_fixture_analysis"],
        "progress": controller["progress"],
        "qualifications": [
            "Process wall time starts immediately before repository module imports; process user plus system CPU and peak RSS cover the fresh process lifetime.",
            "Each timing and callback sample uses a fresh process. Warm work runs only after cold success; callback tracing runs only after its warm timing sample passes.",
            "Phase wall and thread CPU values are nested inclusive and exclusive measurements on the current Python thread. Do not sum inclusive values. Process CPU is reported separately.",
            "Process peak RSS is a process high-water mark, not a per-phase peak. Tracemalloc per-phase peaks were not collected; no sampled-current value is presented as a phase peak.",
            "A handle-paired process includes expanded and shared output, independent shared import, and the expanded public validation/render oracle. Its process RSS cannot establish per-format memory improvement.",
            "Python Pydantic field and model validator callbacks are counted from loaded decorator code objects. Pydantic-core and schema validators implemented in C are not observable through Python profiling.",
            "Callback tracing materially changes runtime cost. Traced elapsed time, CPU, and RSS are excluded from timing summaries; callback rows retain actual per-code counts.",
            "An optional caller-supplied shared fixture reports root, table, record, and pointer bytes. Pointer bytes are a subset; direct duplicate payload savings are a lower bound that excludes nested reference closures.",
            "The expanded baseline is compact generation-storage.v1 output. No pretty-JSON size advantage is counted.",
            "Trusted and public compatibility paths use the same synthetic report, revision binding, request configuration, and ordered request-identity digest. The public path retains independent generation, apply, validation, and render checks.",
            "Matched public boundary totals include standalone annotation and report JSON serialization. The compatibility path also retains its independent proof checks.",
            "A repeated synthetic benchmark does not establish a universal size or speed improvement. The full real matrix remains subject to its runtime ordering gate.",
        ],
    }


def _collect_synthetic_samples(
    args: argparse.Namespace,
    counts: tuple[int, ...],
    output_directory: Path,
    record_row: Callable[[dict], None],
) -> None:
    """Run planned children while preserving each completed coordinate.

    Preconditions:
        Counts and repeats passed validation; record_row writes a checkpoint.
    Postconditions:
        Records every planned timing and callback coordinate, including skips.
    Essential behavior:
        Runs warm work only after cold success and profiles only after warm success.
    """
    for count in counts:
        for repeat in range(1, args.repeats + 1):
            for scenario in SYNTHETIC_SCENARIOS:
                with tempfile.TemporaryDirectory(
                    prefix="synthetic-cache-", dir=output_directory
                ) as temporary:
                    cache_dir = Path(temporary) / "state"
                    cache_dir.mkdir(mode=0o700)
                    cold_coordinates = {
                        "count": count,
                        "repeat": repeat,
                        "scenario": scenario,
                        "cache_state": "cold",
                        "profile_kind": "timing",
                    }
                    cold = _run_synthetic_child_row(
                        cold_coordinates, cache_dir
                    )
                    record_row(cold)
                    warm_coordinates = {
                        **cold_coordinates,
                        "cache_state": "warm",
                    }
                    warm = (
                        _run_synthetic_child_row(warm_coordinates, cache_dir)
                        if _sample_succeeded(cold)
                        else _skipped_synthetic_child_row(
                            warm_coordinates, "cold_sample_failed"
                        )
                    )
                    record_row(warm)
                    if repeat == 1:
                        callback_coordinates = {
                            **warm_coordinates,
                            "profile_kind": "callbacks",
                        }
                        callback = (
                            _run_synthetic_child_row(
                                callback_coordinates, cache_dir
                            )
                            if _sample_succeeded(warm, cold)
                            else _skipped_synthetic_child_row(
                                callback_coordinates,
                                "warm_sample_failed"
                                if _sample_succeeded(cold)
                                else "cold_sample_failed",
                            )
                        )
                        record_row(callback)


def run_synthetic_matrix(args: argparse.Namespace) -> tuple[dict, int]:
    """Run the finite cold/warm and separate callback-traced evidence matrix.

    Preconditions:
        Counts and repeats passed the bounded CLI; output is caller-selected.
    Postconditions:
        Writes source-free measurements and Markdown under the selected output.
    Essential behavior:
        Retains each row as it finishes and never reuses a failed cold cache.
    """
    counts = parse_synthetic_counts(args.counts)
    require(
        SYNTHETIC_MIN_REPEATS <= args.repeats <= SYNTHETIC_MAX_REPEATS,
        "invalid_synthetic_repeats",
    )
    output_directory = args.output_dir.resolve()
    output_directory.mkdir(mode=0o700, parents=True, exist_ok=True)
    output_directory.chmod(0o700)
    timing_samples: list[dict] = []
    callback_samples: list[dict] = []
    source_hashes = _source_hash_labels()
    command = _synthetic_command(args)
    planned_timing_count = (
        len(counts) * args.repeats * len(SYNTHETIC_SCENARIOS) * 2
    )
    planned_callback_count = len(counts) * len(SYNTHETIC_SCENARIOS)
    last_progress = {"label": "matrix_started"}

    def persist_partial() -> None:
        """Write current rows as blocked until final matrix validation.

        Preconditions:
            Current row arrays and fixed source hashes are available.
        Postconditions:
            The private artifacts retain the latest rows and failed gates.
        Essential behavior:
            A mid-run stop cannot leave a partial matrix marked PASS.
        """
        failures = _validate_synthetic_matrix(
            timing_samples,
            callback_samples,
            counts,
            args.repeats,
        )
        failures.append("matrix_in_progress")
        if (
            len(timing_samples) != planned_timing_count
            or len(callback_samples) != planned_callback_count
        ):
            failures.append("matrix_incomplete")
        result = _synthetic_result(
            args,
            {"timing": timing_samples, "callbacks": callback_samples},
            {
                "source_hashes": source_hashes,
                "matrix_failures": failures,
                "progress": last_progress.copy(),
                "shared_fixture_analysis": None,
            },
        )
        _write_synthetic_result(output_directory, result, command)

    def record_row(row: dict) -> None:
        """Append one row, update fixed progress, and persist the checkpoint.

        Preconditions:
            row contains fixed coordinates and source-free child metrics.
        Postconditions:
            The row is retained in its timing or callback collection.
        Essential behavior:
            Stores only a fixed progress label and sample coordinates.
        """
        nonlocal last_progress
        target = (
            timing_samples
            if row["profile_kind"] == "timing"
            else callback_samples
        )
        target.append(row)
        last_progress = {
            "label": "sample_recorded",
            "coordinate": _sample_coordinates(row),
        }
        persist_partial()

    persist_partial()
    _collect_synthetic_samples(args, counts, output_directory, record_row)
    failures = _validate_synthetic_matrix(
        timing_samples,
        callback_samples,
        counts,
        args.repeats,
    )
    shared_fixture_analysis = None
    if not failures:
        shared_fixture = getattr(args, "shared_fixture", None)
        if shared_fixture is not None:
            try:
                shared_fixture_analysis = _analyze_shared_fixture(
                    shared_fixture
                )
            except BenchmarkError as error:
                failures.append(str(error))
            except OSError:
                failures.append("shared_fixture_io_failure")
    last_progress = {"label": "matrix_complete"}
    result = _synthetic_result(
        args,
        {"timing": timing_samples, "callbacks": callback_samples},
        {
            "source_hashes": source_hashes,
            "matrix_failures": failures,
            "progress": last_progress,
            "shared_fixture_analysis": shared_fixture_analysis,
        },
    )
    _write_synthetic_result(output_directory, result, command)
    return result, result["parent_exit_code"]


class SafeParser(argparse.ArgumentParser):
    """Keep invalid command arguments and private paths out of diagnostics."""

    def error(self, _message: str) -> None:
        """Fail without echoing arbitrary user input."""
        msg = "invalid_arguments"
        raise BenchmarkError(msg)


def _synthetic_main(argv: Sequence[str]) -> int:
    """Run either one internal sample or the exact authorized finite matrix.

    Preconditions:
        Arguments select only the closed synthetic benchmark interface.
    Postconditions:
        Emits source-free status and writes parent artifacts under the
        caller-selected output directory.
    Essential behavior:
        Keeps the existing pinned-input phase parser and behavior unchanged.
    """
    parser = SafeParser(description="Run bounded offline synthetic evidence.")
    parser.add_argument("--synthetic", action="store_true")
    parser.add_argument("--synthetic-sample", choices=SYNTHETIC_SCENARIOS)
    parser.add_argument("--count", type=int)
    parser.add_argument("--cache-state", choices=("cold", "warm"))
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--profile-callbacks", action="store_true")
    parser.add_argument(
        "--counts",
        default=",".join(str(count) for count in SYNTHETIC_DEFAULT_COUNTS),
    )
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="caller-selected private directory for source-free results",
    )
    parser.add_argument(
        "--shared-fixture",
        type=Path,
        help="optional caller-supplied shared report for component analysis",
    )
    try:
        args = parser.parse_args(argv)
        if args.synthetic_sample is not None:
            require(not args.synthetic, "invalid_synthetic_mode")
            require(args.count is not None, "invalid_synthetic_count")
            require(args.cache_state is not None, "invalid_cache_state")
            require(args.cache_dir is not None, "invalid_synthetic_cache")
            require(args.output_dir is None, "invalid_synthetic_output")
            require(args.shared_fixture is None, "invalid_sample_fixture")
            metrics = run_synthetic_sample(args)
            sys.stdout.write(json.dumps(metrics, allow_nan=False) + "\n")
            return 0 if metrics["status"] == "PASS" else 2
        require(args.synthetic, "invalid_synthetic_mode")
        require(args.count is None, "invalid_synthetic_count")
        require(args.cache_state is None, "invalid_cache_state")
        require(args.cache_dir is None, "invalid_synthetic_cache")
        require(not args.profile_callbacks, "invalid_synthetic_profile_mode")
        require(args.output_dir is not None, "missing_synthetic_output")
        result, exit_code = run_synthetic_matrix(args)
        sys.stdout.write(
            json.dumps(
                {
                    "status": result["status"],
                    "timing_sample_count": result["timing_sample_count"],
                    "callback_sample_count": result["callback_sample_count"],
                    "parent_exit_code": exit_code,
                    "matrix_failures": result["matrix_failures"],
                },
                allow_nan=False,
            )
            + "\n"
        )
    except BenchmarkError as error:
        sys.stdout.write(
            json.dumps(
                {"status": "BLOCKED", "reason": str(error)},
                allow_nan=False,
            )
            + "\n"
        )
        return 2
    except OSError:
        sys.stdout.write('{"status":"BLOCKED","reason":"local_io_failure"}\n')
        return 2
    except Exception:  # noqa: BLE001 # Keep unexpected controller failures source-free.
        sys.stdout.write(
            '{"status":"BLOCKED","reason":"controller_failure"}\n'
        )
        return 2
    else:
        return exit_code


def main(argv: Sequence[str] | None = None) -> int:
    """Run a pinned phase or the bounded synthetic evidence interface."""
    arguments = tuple(sys.argv[1:] if argv is None else argv)
    if "--synthetic" in arguments or "--synthetic-sample" in arguments:
        return _synthetic_main(arguments)
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
        args = parser.parse_args(arguments)
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
