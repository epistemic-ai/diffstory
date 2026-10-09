"""Concurrency and checkpoint tests for the narration runtime."""

from __future__ import annotations

import gc
import hashlib
import json
import os
import select
import signal
import subprocess
import sys
import tempfile
import threading
import time
import traceback
import unittest
import weakref
from dataclasses import FrozenInstanceError
from dataclasses import replace
from pathlib import Path
from types import MappingProxyType
from typing import ClassVar
from unittest.mock import Mock
from unittest.mock import call
from unittest.mock import patch

from pydantic import ValidationError
from test_analysis import snap
from test_narrative_contracts import prose_result
from test_narrative_plan import AFTER
from test_narrative_plan import BEFORE
from test_narrative_plan import authored_result

from diffstory import narrative_runtime as runtime_v2
from diffstory.analysis import compile_snapshot
from diffstory.narrative_contracts import ArtifactDependency
from diffstory.narrative_contracts import BatchSpec
from diffstory.narrative_contracts import DetailResult
from diffstory.narrative_contracts import NarrativeResult
from diffstory.narrative_contracts import result_key
from diffstory.narrative_contracts import result_kind
from diffstory.narrative_contracts import result_stage
from diffstory.narrative_contracts import structured_output_schema
from diffstory.narrative_plan import RequestPackingLimits
from diffstory.narrative_plan import build_evidence_ledger
from diffstory.narrative_plan import build_ledger_index
from diffstory.narrative_plan import prepare_request_items
from diffstory.narrative_plan import resolve_span_ref
from diffstory.narrative_plan import span_reference
from diffstory.narrative_plan import validate_detail_result
from diffstory.narrative_runtime import CallSpec
from diffstory.narrative_runtime import CheckpointError
from diffstory.narrative_runtime import CheckpointRecord
from diffstory.narrative_runtime import CheckpointStore
from diffstory.narrative_runtime import GenerationRuntime
from diffstory.narrative_runtime import ProgressEvent
from diffstory.narrative_runtime import RunIdentity
from diffstory.narrative_runtime import RuntimeCallError
from diffstory.narrative_runtime import RuntimeStageError
from diffstory.usage import ModelCapacity
from diffstory.usage import RunUsage


def identity(
    *,
    base_revision: str = "base-a",
    head_revision: str = "head-b",
    report_signature: str = "report-1",
    provider_destination: str = "https://provider.example/v1",
    model: str = "model-a",
) -> RunIdentity:
    """Build a stable identity for one deterministic report and provider."""
    return RunIdentity(
        base_revision=base_revision,
        head_revision=head_revision,
        report_signature=report_signature,
        provider_destination=provider_destination,
        model=model,
    )


def spec(
    *,
    task_id: str = "chunk-1",
    request_digest: str = "request-1",
    validation_version: str = "v1",
) -> CallSpec:
    """Build one reusable semantic-unit specification."""
    return CallSpec(
        task_id=task_id,
        schema_name="chunk_schema",
        request_digest=request_digest,
        validation_version=validation_version,
    )


class NarrativeRuntimeTests(unittest.TestCase):
    """Verify isolated checkpoints, safe accounting, and stage behavior."""

    def test_identity_digest_includes_every_run_dimension(self) -> None:
        """Invalidate checkpoint namespaces when any run identity changes."""
        original = identity()
        changes = {
            "base_revision": "base-c",
            "head_revision": "head-d",
            "report_signature": "report-2",
            "provider_destination": "https://other.example/v1",
            "model": "model-b",
            "runtime_contract_version": "2",
        }
        for field, value in changes.items():
            with self.subTest(field=field):
                changed = original.model_dump()
                changed[field] = value
                self.assertNotEqual(
                    original.digest,
                    RunIdentity(**changed).digest,
                )

    def test_request_digest_includes_schema_and_reservation(self) -> None:
        """Change the request digest when prepared-call contracts change."""
        common = {
            "task_id": "chunk-1",
            "schema_name": "chunk_schema",
            "request": b"prepared request",
            "validation_version": "v1",
        }
        original = CallSpec.from_request(
            **common,
            schema={"type": "object"},
            output_reservation=100,
        )
        changed_schema = CallSpec.from_request(
            **common,
            schema={"type": "array"},
            output_reservation=100,
        )
        changed_reservation = CallSpec.from_request(
            **common,
            schema={"type": "object"},
            output_reservation=101,
        )

        self.assertNotEqual(
            original.request_digest, changed_schema.request_digest
        )
        self.assertNotEqual(
            original.request_digest,
            changed_reservation.request_digest,
        )

    def test_unresolved_model_can_run_without_checkpoints_only(self) -> None:
        """Allow ordinary runs while blocking unstable persistent identity."""
        unresolved = identity(model="Codex CLI default")
        GenerationRuntime(unresolved)
        with (
            tempfile.TemporaryDirectory() as directory,
            self.assertRaisesRegex(ValueError, "explicit provider model"),
        ):
            GenerationRuntime(unresolved, checkpoint_dir=directory)

    def test_runtime_accepts_records_and_callable_sequences_only(self) -> None:
        """Reject mapping adapters and indexed stage jobs."""
        runtime_identity = identity()
        with self.assertRaises(TypeError):
            GenerationRuntime(runtime_identity.model_dump())

        runtime = GenerationRuntime(runtime_identity)
        with self.assertRaises(TypeError):
            runtime.run_stage("chunks", [(0, lambda: {"value": 1})])
        with self.assertRaises(TypeError):
            runtime.invoke(
                spec().model_dump(),
                lambda: {"value": 1},
                lambda value: value,
            )

    def test_checkpoint_cache_isolated_and_usage_separate(self) -> None:
        """Reuse validated output without reusing historical calls or usage."""
        with tempfile.TemporaryDirectory() as directory:
            first = GenerationRuntime(identity(), checkpoint_dir=directory)
            first_result = first.invoke(
                spec(),
                lambda: first.call(
                    "chunk_schema",
                    lambda: (
                        {"value": 3},
                        {
                            "input_tokens": 10,
                            "cached_input_tokens": 2,
                            "output_tokens": 4,
                        },
                    ),
                ),
                lambda value: value,
            )
            second = GenerationRuntime(identity(), checkpoint_dir=directory)
            second_result = second.invoke(
                spec(),
                lambda: self.fail("A valid checkpoint should skip production"),
                lambda value: value,
            )

            self.assertEqual(first_result, {"value": 3})
            self.assertEqual(second_result, first_result)
            self.assertEqual(first.usage.report()["calls"], 1)
            self.assertEqual(second.usage.report()["calls"], 0)
            self.assertEqual(second.reused_checkpoints, 1)

            checkpoint_files = list(Path(directory).glob("*/*.json"))
            self.assertEqual(len(checkpoint_files), 1)
            saved = json.loads(checkpoint_files[0].read_text(encoding="utf-8"))
            self.assertEqual(saved["result"], {"value": 3})
            self.assertNotIn("origin", saved)
            self.assertNotIn("receipts", saved)
            self.assertNotIn("usage", saved)

    def test_checkpoint_identity_and_atomic_replace_are_isolated(self) -> None:
        """Keep identities separate and retain old data after a failed replace."""
        with tempfile.TemporaryDirectory() as directory:
            store = CheckpointStore(directory)
            first_identity = identity()
            second_identity = identity(report_signature="report-2")
            first_spec = spec()
            first_record = CheckpointRecord(
                identity_digest=first_identity.digest,
                task_id=first_spec.task_id,
                schema_name=first_spec.schema_name,
                request_digest=first_spec.request_digest,
                validation_version=first_spec.validation_version,
                result={"version": 1},
            )
            store.commit(first_record)
            self.assertIsNone(store.load(second_identity, first_spec))

            replacement = CheckpointRecord(
                identity_digest=first_identity.digest,
                task_id=first_spec.task_id,
                schema_name=first_spec.schema_name,
                request_digest=first_spec.request_digest,
                validation_version=first_spec.validation_version,
                result={"version": 2},
            )
            with (
                patch(
                    "pathlib.Path.replace",
                    side_effect=OSError,
                ),
                self.assertRaises(CheckpointError),
            ):
                store.commit(replacement)

            loaded = store.load(first_identity, first_spec)
            self.assertIsInstance(loaded, CheckpointRecord)
            self.assertEqual(loaded.result, {"version": 1})
            self.assertEqual(
                list(Path(directory).glob("*/*.tmp")),
                [],
            )

    def test_persisted_checkpoint_requires_explicit_current_format_version(
        self,
    ) -> None:
        """Treat missing, obsolete, boolean, and string versions as cache misses."""
        invalid_versions: tuple[tuple[str, object], ...] = (
            ("missing", None),
            ("obsolete", 0),
            ("boolean", True),
            ("string", "1"),
        )
        for label, version in invalid_versions:
            with (
                self.subTest(version=label),
                tempfile.TemporaryDirectory() as directory,
            ):
                first = GenerationRuntime(identity(), checkpoint_dir=directory)
                first.invoke(
                    spec(),
                    lambda: {"value": 1},
                    lambda value: value,
                )
                checkpoint_path = next(Path(directory).glob("*/*.json"))
                saved = json.loads(checkpoint_path.read_text(encoding="utf-8"))
                if label == "missing":
                    saved.pop("format_version")
                else:
                    saved["format_version"] = version
                checkpoint_path.write_text(
                    json.dumps(saved),
                    encoding="utf-8",
                )

                produced = 0

                def produce() -> dict[str, int]:
                    nonlocal produced
                    produced += 1
                    return {"value": 2}

                resumed = GenerationRuntime(
                    identity(), checkpoint_dir=directory
                )
                result = resumed.invoke(spec(), produce, lambda value: value)

                self.assertEqual(result, {"value": 2})
                self.assertEqual(produced, 1)
                self.assertEqual(resumed.reused_checkpoints, 0)

    def test_coordinator_keyboard_interrupt_stops_later_sends_and_keeps_result(
        self,
    ) -> None:
        """Stop later calls while preserving a valid in-flight checkpoint."""
        with tempfile.TemporaryDirectory() as directory:
            provider_started = threading.Event()
            provider_release = threading.Event()
            callback_interrupted = threading.Event()
            later_send_started = threading.Event()
            timers: list[threading.Timer] = []
            outcomes: list[str] = []

            def on_progress(event: ProgressEvent) -> None:
                if (
                    event.event == "call_started"
                    and not callback_interrupted.is_set()
                ):
                    self.assertTrue(provider_started.wait(timeout=2))
                    callback_interrupted.set()
                    timer = threading.Timer(0.2, provider_release.set)
                    timer.daemon = True
                    timers.append(timer)
                    timer.start()
                    raise KeyboardInterrupt

            runtime = GenerationRuntime(
                identity(),
                checkpoint_dir=directory,
                on_progress=on_progress,
            )

            def first_send() -> tuple[dict[str, int], dict[str, int]]:
                provider_started.set()
                if not provider_release.wait(timeout=3):
                    raise TimeoutError
                return {"value": 1}, {
                    "input_tokens": 2,
                    "cached_input_tokens": 0,
                    "output_tokens": 1,
                }

            def second_send() -> tuple[dict[str, int], None]:
                later_send_started.set()
                return {"value": 2}, None

            def job() -> dict[str, object]:
                result = runtime.invoke(
                    spec(),
                    lambda: runtime.call("chunk_schema", first_send),
                    lambda value: value,
                )
                runtime.call("chunk_schema", second_send)
                return result

            def coordinate() -> None:
                try:
                    runtime.run_stage("chunks", [job])
                except KeyboardInterrupt:
                    outcomes.append("keyboard_interrupt")

            coordinator = threading.Thread(target=coordinate)
            coordinator.start()
            self.assertTrue(callback_interrupted.wait(timeout=2))
            coordinator.join(timeout=4)
            for timer in timers:
                timer.join(timeout=1)

            self.assertFalse(coordinator.is_alive())
            self.assertEqual(outcomes, ["keyboard_interrupt"])
            self.assertFalse(later_send_started.is_set())
            self.assertEqual(runtime.usage.report()["calls"], 1)
            self.assertEqual(
                runtime.run_stage("followup", [lambda: {"ready": True}]),
                [{"ready": True}],
            )

            resumed = GenerationRuntime(identity(), checkpoint_dir=directory)
            result = resumed.invoke(
                spec(),
                lambda: self.fail("A valid in-flight result should be cached"),
                lambda value: value,
            )
            self.assertEqual(result, {"value": 1})
            self.assertEqual(resumed.reused_checkpoints, 1)

    def test_two_sigints_drain_active_task_before_cli_summary(self) -> None:
        """Keep usage and checkpoints when repeated signals interrupt cleanup."""
        if os.name != "posix":
            self.skipTest("This regression requires POSIX process signals")

        child = """
import json
import sys
import time
from pathlib import Path

from diffstory.narrative_runtime import CallSpec
from diffstory.narrative_runtime import GenerationRuntime
from diffstory.narrative_runtime import RunIdentity

cache = Path(sys.argv[1])
runtime = GenerationRuntime(
    RunIdentity(
        base_revision="base",
        head_revision="head",
        report_signature="report",
        provider_destination="local",
        model="model",
    ),
    checkpoint_dir=cache,
)
call_spec = CallSpec(
    task_id="chunk-1",
    schema_name="chunk_schema",
    request_digest="request",
    validation_version="v1",
)
sends = {"active": 0, "later": 0}

def active_send():
    sends["active"] += 1
    print("provider_started", flush=True)
    time.sleep(0.5)
    return {"value": 1}, {"input_tokens": 7, "output_tokens": 3}

def later_send():
    sends["later"] += 1
    return {"value": 2}, None

def job():
    runtime.invoke(
        call_spec,
        lambda: runtime.call("chunk_schema", active_send),
        lambda result: result,
    )
    runtime.call("chunk_schema", later_send)

try:
    runtime.run_stage("chunks", [job])
except KeyboardInterrupt:
    summary = {
        "usage_calls": runtime.usage.report()["calls"],
        "checkpoint_files": len(list(cache.glob("*/*.json"))),
        "active_sends": sends["active"],
        "later_sends": sends["later"],
    }
    print(json.dumps(summary), flush=True)
    raise SystemExit(130)
"""
        repository = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as cache_dir:
            process = subprocess.Popen(
                [sys.executable, "-c", child, cache_dir],
                cwd=repository,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            try:
                ready, _, _ = select.select([process.stdout], [], [], 3)
                if not ready:
                    stdout, stderr = process.communicate(timeout=1)
                    self.fail(
                        "The provider did not start in the child process: "
                        f"{stdout}{stderr}"
                    )
                self.assertEqual(
                    process.stdout.readline().strip(), "provider_started"
                )
                process.send_signal(signal.SIGINT)
                time.sleep(0.05)
                process.send_signal(signal.SIGINT)
                stdout, stderr = process.communicate(timeout=4)
            except BaseException:
                if process.poll() is None:
                    process.kill()
                process.communicate()
                raise

            self.assertEqual(process.returncode, 130, stderr)
            summary = json.loads(stdout)
            self.assertEqual(
                summary,
                {
                    "usage_calls": 1,
                    "checkpoint_files": 1,
                    "active_sends": 1,
                    "later_sends": 0,
                },
            )
            checkpoint = next(Path(cache_dir).glob("*/*.json"))
            self.assertEqual(
                json.loads(checkpoint.read_text(encoding="utf-8"))["result"],
                {"value": 1},
            )

    def test_invalid_cached_result_is_revalidated_and_replaced(self) -> None:
        """Treat a cached result that fails current validation as a cache miss."""
        with tempfile.TemporaryDirectory() as directory:
            runtime = GenerationRuntime(identity(), checkpoint_dir=directory)
            runtime.invoke(
                spec(),
                lambda: {"value": 1},
                lambda value: value,
            )
            production_count = 0

            def produce() -> dict[str, int]:
                nonlocal production_count
                production_count += 1
                return {"value": 2}

            def validate(value: dict[str, int]) -> dict[str, int]:
                if value["value"] < 2:
                    raise ValueError
                return value

            result = runtime.invoke(spec(), produce, validate)

            self.assertEqual(result, {"value": 2})
            self.assertEqual(production_count, 1)
            self.assertEqual(runtime.reused_checkpoints, 0)

    def test_call_accounts_provider_failure_without_leaking_message(
        self,
    ) -> None:
        """Count a failed attempt and retain only sanitized usage metadata."""
        runtime = GenerationRuntime(identity())

        class ProviderFailureError(RuntimeError):
            usage: ClassVar[dict[str, int]] = {
                "input_tokens": 5,
                "cached_input_tokens": 1,
                "output_tokens": 0,
            }

        provider_failure = ProviderFailureError()
        provider_failure.args = ("private provider body",)
        with self.assertRaises(RuntimeCallError) as raised:
            runtime.call(
                "chunk_schema",
                lambda: (_ for _ in ()).throw(
                    provider_failure,
                ),
            )

        self.assertNotIn("private provider body", str(raised.exception))
        self.assertEqual(
            runtime.usage.report(),
            {
                "input_tokens": 5,
                "cached_input_tokens": 1,
                "output_tokens": 0,
                "calls": 1,
                "unreported_calls": 0,
            },
        )
        self.assertEqual(runtime.usage.report()["calls"], 1)

    def test_runtime_call_categories_survive_stage_wrapping(self) -> None:
        """Preserve allowlisted categories and use their fixed safe actions."""
        actions = {
            "authentication": "Check provider credentials, then rerun the stage.",
            "rate_limit": "Wait before you rerun the stage.",
            "transport": "Check provider connectivity, then rerun the stage.",
            "timeout": "Check provider responsiveness, then rerun the stage.",
            "invalid_response": "Check the provider response against its schema.",
            "provider_error": "Check the provider setup and request, then rerun.",
            "validation_error": "Correct the response or validator, then rerun.",
            "checkpoint_error": "Check checkpoint storage and permissions, then rerun.",
            "callback_error": "Correct the progress callback, then rerun.",
            "task_error": "Correct the failed task, then rerun.",
            "cancelled": "Rerun the stage after stopping its remaining work.",
        }
        schema_name = "chunk_schema"
        for category, action in actions.items():
            with self.subTest(category=category):
                runtime = GenerationRuntime(identity())

                def send(
                    selected_category: str = category,
                ) -> tuple[dict[str, object], dict[str, int] | None]:
                    raise RuntimeCallError(schema_name, selected_category)

                with self.assertRaises(RuntimeStageError) as raised:
                    runtime.run_stage(
                        "chunks",
                        [
                            lambda selected_runtime=runtime,
                            selected_send=send: (
                                selected_runtime.call(
                                    schema_name, selected_send
                                )
                            ),
                        ],
                    )

                self.assertEqual(raised.exception.category, category)
                self.assertEqual(raised.exception.action, action)

        unknown = RuntimeCallError("chunk_schema", "private exception text")
        self.assertEqual(unknown.category, "provider_error")
        self.assertNotIn("private exception text", str(unknown))

    def test_transport_failure_keeps_category_and_redacts_message(
        self,
    ) -> None:
        """Classify transport errors without exposing their provider text."""
        runtime = GenerationRuntime(identity())
        failure = ConnectionError()
        failure.args = ("private transport response",)

        def send() -> tuple[dict[str, object], dict[str, int] | None]:
            raise failure

        with self.assertRaises(RuntimeStageError) as raised:
            runtime.run_stage(
                "chunks",
                [lambda: runtime.call("chunk_schema", send)],
            )

        self.assertEqual(raised.exception.category, "transport")
        self.assertNotIn("private transport response", str(raised.exception))
        self.assertEqual(runtime.usage.report()["calls"], 1)

    def test_provider_reply_must_be_a_response_and_usage_pair(self) -> None:
        """Reject bare provider dictionaries while counting the attempt."""
        runtime = GenerationRuntime(identity())

        with self.assertRaises(RuntimeStageError) as raised:
            runtime.run_stage(
                "chunks",
                [
                    lambda: runtime.call(
                        "chunk_schema",
                        lambda: {"value": 1},
                    ),
                ],
            )

        self.assertEqual(raised.exception.category, "invalid_response")
        self.assertEqual(runtime.usage.report()["calls"], 1)

    def test_stage_is_bounded_stops_submissions_and_drains_active_call(
        self,
    ) -> None:
        """Drain a started provider call and block later calls after failure."""
        runtime = GenerationRuntime(identity(), concurrency=2)
        provider_started = threading.Event()
        provider_release = threading.Event()
        failed_job_started = threading.Event()
        later_job_started = threading.Event()
        provider_calls: list[str] = []
        outcome: list[RuntimeStageError] = []

        def slow_send() -> tuple[dict[str, bool], dict[str, int]]:
            provider_started.set()
            if not provider_release.wait(timeout=3):
                raise TimeoutError
            provider_calls.append("first")
            return {"ok": True}, {
                "input_tokens": 2,
                "cached_input_tokens": 0,
                "output_tokens": 1,
            }

        def active_job() -> None:
            runtime.call("chunk_schema", slow_send)
            runtime.call(
                "chunk_schema",
                lambda: provider_calls.append("must-not-start"),
            )

        def failing_job() -> None:
            self.assertTrue(provider_started.wait(timeout=2))
            failed_job_started.set()
            failure = ValueError()
            failure.args = ("private provider response",)
            raise failure

        def later_job() -> None:
            later_job_started.set()

        def coordinate() -> None:
            try:
                runtime.run_stage(
                    "chunks",
                    [active_job, failing_job, later_job],
                )
            except RuntimeStageError as error:
                outcome.append(error)

        coordinator = threading.Thread(target=coordinate)
        coordinator.start()
        self.assertTrue(failed_job_started.wait(timeout=2))
        deadline = time.monotonic() + 2
        while (
            not runtime._stop_requested()  # noqa: SLF001
            and time.monotonic() < deadline
        ):
            time.sleep(0.01)

        self.assertTrue(runtime._stop_requested())  # noqa: SLF001
        self.assertTrue(coordinator.is_alive())
        self.assertFalse(later_job_started.is_set())
        provider_release.set()
        coordinator.join(timeout=3)

        self.assertFalse(coordinator.is_alive())
        self.assertEqual(len(outcome), 1)
        self.assertIsInstance(outcome[0], RuntimeStageError)
        self.assertNotIn("private provider response", str(outcome[0]))
        self.assertEqual(provider_calls, ["first"])
        self.assertEqual(runtime.usage.report()["calls"], 1)

    def test_progress_callback_runs_only_on_coordinator(self) -> None:
        """Queue worker events and deliver them from the coordinating thread."""
        callback_threads: list[int] = []
        runtime = GenerationRuntime(
            identity(),
            concurrency=2,
            on_progress=lambda _event: callback_threads.append(
                threading.get_ident(),
            ),
        )
        coordinator_threads: list[int] = []
        results: list[list[object]] = []

        def coordinate() -> None:
            coordinator_threads.append(threading.get_ident())
            results.append(
                runtime.run_stage(
                    "summaries",
                    [
                        lambda: runtime.call(
                            "summary_schema",
                            lambda: ({"value": 1}, None),
                        ),
                        lambda: runtime.call(
                            "summary_schema",
                            lambda: ({"value": 2}, None),
                        ),
                    ],
                ),
            )

        coordinator = threading.Thread(target=coordinate)
        coordinator.start()
        coordinator.join(timeout=3)

        self.assertFalse(coordinator.is_alive())
        self.assertEqual(results, [[{"value": 1}, {"value": 2}]])
        self.assertTrue(callback_threads)
        self.assertEqual(set(callback_threads), {coordinator_threads[0]})

    def test_stage_results_follow_input_order_after_reversed_completion(
        self,
    ) -> None:
        """Return callable results in input order after reverse completion."""
        runtime = GenerationRuntime(identity(), concurrency=2)
        second_finished = threading.Event()

        def first() -> dict[str, int]:
            self.assertTrue(second_finished.wait(timeout=2))
            return {"position": 1}

        def second() -> dict[str, int]:
            second_finished.set()
            return {"position": 2}

        self.assertEqual(
            runtime.run_stage("steps", [first, second]),
            [{"position": 1}, {"position": 2}],
        )

    def test_run_usage_is_safe_for_parallel_call_accounting(self) -> None:
        """Retain every provider attempt under concurrent updates."""
        usage = RunUsage()
        runtime = GenerationRuntime(identity(), concurrency=4, usage=usage)
        runtime.run_stage(
            "parallel",
            [
                lambda: runtime.call(
                    "schema",
                    lambda: (
                        {"ok": True},
                        {"input_tokens": 1, "output_tokens": 2},
                    ),
                )
                for _ in range(20)
            ],
        )
        self.assertEqual(usage.report()["calls"], 20)
        self.assertEqual(usage.report()["input_tokens"], 20)
        self.assertEqual(usage.report()["output_tokens"], 40)


def digest(value: object) -> str:
    """Hash fixture values without exposing any source in metadata."""
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def prepared_v2(
    keys: tuple[str, ...], purpose: str = "initial"
) -> runtime_v2.PreparedArtifactRequest:
    """Serialize actual test input and declare all capacity bounds."""
    body = json.dumps(
        {"items": keys, "purpose": purpose}, sort_keys=True
    ).encode()
    request_digest = hashlib.sha256(body).hexdigest()
    batch = BatchSpec(
        id=digest(["batch", request_digest]),
        stage="detail",
        item_keys=keys,
        group_keys=(),
        request_digest=request_digest,
        input_bound=len(body) + 1024,
        output_reservation=200,
        output_guard=200,
        estimated_response_bytes=1600,
        estimated_seconds=3,
        partition=0,
    )
    return runtime_v2.PreparedArtifactRequest(
        body=body,
        metadata=runtime_v2.PreparedMetadata(
            batch=batch,
            context_tokens=100_000,
            max_output_tokens=200,
            response_byte_limit=10_000,
            timeout_seconds=30,
        ),
    )


def fixture_v2(
    count: int = 2,
    run_identity: RunIdentity | None = None,
) -> tuple[
    runtime_v2.ArtifactInvocation,
    runtime_v2.DependencyRegistry,
    dict[str, DetailResult],
]:
    """Build strict context-only evidence results and full dependency descriptors."""
    keys = tuple(digest(["unit", i]) for i in range(count))
    static = runtime_v2.static_artifact_dependencies(
        identity() if run_identity is None else run_identity,
        configuration_digest=digest("configuration"),
        prompt_digest=digest("prompt"),
        schema_digest=digest("schema"),
        validator_digest=digest("validator"),
    )
    lookups = tuple(
        runtime_v2.ArtifactLookup(
            item_key=key, static_dependencies=static, validator_version="v2"
        )
        for key in keys
    )
    peers = tuple(
        ArtifactDependency(
            kind=kind, key=key, digest=digest(["evidence", key])
        )
        for key in keys
        for kind in ("evidence", "peer")
    )
    context = tuple(
        ArtifactDependency(
            kind=kind, key=digest([kind]), digest=digest([kind, "empty"])
        )
        for kind in ("context", "order", "terminology")
    )
    invocation = runtime_v2.ArtifactInvocation(
        batch=prepared_v2(keys).metadata.batch,
        lookups=lookups,
        dependencies=static + peers + context,
    )
    registry = runtime_v2.DependencyRegistry(
        dependencies=invocation.dependencies
    )
    results = {
        key: DetailResult(
            unit_key=key, explanations=(), observations=(), questions=()
        )
        for key in keys
    }
    return invocation, registry, results


def validate_v2(
    lookup: runtime_v2.ArtifactLookup, result: DetailResult
) -> None:
    """Require the current item's typed context-only contract."""
    if result.unit_key != lookup.item_key or result.explanations:
        msg = "coverage"
        raise runtime_v2.ArtifactRuntimeError(msg)


def prepare_v2(
    _invocation: runtime_v2.ArtifactInvocation,
    node: runtime_v2.RecoveryNode,
    purpose: str,
) -> runtime_v2.PreparedArtifactRequest:
    """Reconstruct the exact prior body for retries and new bodies for repairs."""
    mode = purpose
    if purpose == "transient":
        mode = next(
            a.purpose
            for a in reversed(node.attempts)
            if a.purpose != "transient"
        )
    return prepared_v2(node.missing_keys, mode)


def wire_v2(items: list[object], **extras: object) -> runtime_v2.ProviderReply:
    """Make a well-formed outer envelope without a circular model-echoed ID."""
    return runtime_v2.ProviderReply(
        {"version": "diffstory.batch-result.v2", "items": items, **extras},
        response_bytes=100,
    )


def changed_record(
    value: runtime_v2.Contract, **changes: object
) -> runtime_v2.Contract:
    """Revalidate fixture changes through the public strict constructor."""
    fields = value.model_dump()
    if isinstance(value, runtime_v2.PreparedArtifactRequest):
        fields["body"] = value.body
    return type(value).model_validate({**fields, **changes})


class ArtifactRuntimeTests(unittest.TestCase):
    """Exercise complete typed artifacts and durable bounded recovery."""

    def callbacks(
        self, results: dict[str, DetailResult], sends: list[tuple[str, ...]]
    ) -> runtime_v2.BatchCallbacks:
        """Return a deterministic successful provider with exact send tracking."""

        def send(
            prepared: runtime_v2.PreparedArtifactRequest,
        ) -> runtime_v2.ProviderReply:
            keys = prepared.metadata.batch.item_keys
            sends.append(keys)
            return wire_v2(
                [results[key].model_dump(mode="json") for key in keys]
            )

        return runtime_v2.BatchCallbacks(
            prepare=prepare_v2, send=send, validate=validate_v2
        )

    def test_stage_prefetch_validates_only_each_original_batch_matches(
        self,
    ) -> None:
        """Whole-stage hits require one full validation per selected artifact."""
        for count in (4, 8, 16):
            with (
                self.subTest(count=count),
                tempfile.TemporaryDirectory() as directory,
            ):
                full, registry, results = fixture_v2(count)
                invocations = tuple(
                    runtime_v2.ArtifactInvocation(
                        batch=prepared_v2((lookup.item_key,)).metadata.batch,
                        lookups=(lookup,),
                        dependencies=tuple(
                            d
                            for d in full.dependencies
                            if d.kind not in {"evidence", "peer"}
                            or d.key == lookup.item_key
                        ),
                    )
                    for lookup in full.lookups
                )
                sends = []
                callbacks = self.callbacks(results, sends)
                cold = runtime_v2.ArtifactRuntime(
                    identity(), recovery_dir=directory, artifact_dir=directory
                )
                expected = tuple(
                    cold.execute_batch(invocation, registry, callbacks)
                    for invocation in invocations
                )
                self.assertEqual(len(sends), count)
                states = tuple(
                    cold.recovery.load(invocation.semantic_key)
                    for invocation in invocations
                )
                warm = runtime_v2.ArtifactRuntime(
                    identity(), recovery_dir=directory, artifact_dir=directory
                )
                hits = warm.prefetch(
                    full.lookups, registry, callbacks.validate
                )
                self.assertEqual(hits, tuple(o.artifacts[0] for o in expected))
                sends.clear()
                audit = Mock(return_value=True)
                with (
                    patch.object(
                        runtime_v2.StoredArtifact,
                        "model_validate",
                        wraps=runtime_v2.StoredArtifact.model_validate,
                    ) as full_validate,
                    patch.object(
                        runtime_v2.ArtifactLookup,
                        "model_validate",
                        wraps=runtime_v2.ArtifactLookup.model_validate,
                    ) as header_validate,
                ):
                    actual = tuple(
                        warm.execute_batch(
                            invocation,
                            registry,
                            callbacks,
                            available=hits,
                            audit_candidate=audit,
                        )
                        for invocation in invocations
                    )
                self.assertEqual(actual, expected)
                self.assertEqual(full_validate.call_count, count)
                self.assertEqual(header_validate.call_count, count)
                self.assertEqual(audit.call_count, count)
                self.assertEqual(sends, [])
                self.assertEqual(warm.usage.report()["calls"], 0)
                self.assertEqual(warm.counters["attempts"], 0)
                self.assertEqual(
                    tuple(warm.recovery.load(state.key) for state in states),
                    states,
                )

    def test_v2_requires_durable_state_and_keeps_unresolved_legacy_rule(
        self,
    ) -> None:
        """Disable artifacts on unresolved models but preserve durable budgets."""
        with self.assertRaises(runtime_v2.ArtifactRuntimeError):
            runtime_v2.ArtifactRuntime(identity())
        invocation, registry, results = fixture_v2(
            1, identity(model="default")
        )
        with tempfile.TemporaryDirectory() as directory:
            runtime = runtime_v2.ArtifactRuntime(
                identity(model="default"),
                recovery_dir=directory,
                artifact_dir=directory,
            )
            self.assertIsNone(runtime.artifacts)
            runtime.execute_batch(
                invocation, registry, self.callbacks(results, [])
            )
            self.assertEqual(
                runtime.pending_lineages(registry)[0].consumed_attempts, 1
            )
            self.assertFalse((Path(directory) / "artifacts-v2").exists())
        with self.assertRaisesRegex(ValueError, "explicit provider model"):
            GenerationRuntime(
                identity(model="default"), checkpoint_dir="unused"
            )
        self.assertEqual(runtime_v2.CHECKPOINT_FORMAT_VERSION, 1)

    def test_independent_validation_preserves_success_and_repairs_only_target(
        self,
    ) -> None:
        """One invalid item cannot reject or regenerate a valid peer."""
        invocation, registry, results = fixture_v2()
        first, second = invocation.batch.item_keys
        sends = []

        def send(
            prepared: runtime_v2.PreparedArtifactRequest,
        ) -> runtime_v2.ProviderReply:
            sends.append(prepared.metadata.batch.item_keys)
            if len(sends) == 1:
                return wire_v2(
                    [
                        results[first].model_dump(mode="json"),
                        {"unit_key": second, "explanations": "malformed"},
                    ]
                )
            return wire_v2([results[second].model_dump(mode="json")])

        callbacks = runtime_v2.BatchCallbacks(
            prepare=prepare_v2, send=send, validate=validate_v2
        )
        with tempfile.TemporaryDirectory() as directory:
            runtime = runtime_v2.ArtifactRuntime(
                identity(), recovery_dir=directory, artifact_dir=directory
            )
            outcome = runtime.execute_batch(invocation, registry, callbacks)
            self.assertEqual(
                tuple(r.unit_key for r in outcome.detail_batch().items),
                (first, second),
            )
            self.assertEqual(sends, [(first, second), (second,)])
            state = runtime.pending_lineages(registry)[0]
            self.assertEqual(state.nodes[0].repair_count, 1)
            self.assertEqual(runtime.counters["attempts"], 2)
            self.assertEqual(runtime.diagnostics[0].valid_items, 1)
            self.assertEqual(runtime.diagnostics[0].invalid_items, 1)
            self.assertEqual(
                len(list(Path(directory).glob("artifacts-v2/*/*.json"))), 2
            )
            self.assertEqual(
                outcome.artifacts[1].production_request.batch.item_keys,
                (second,),
            )
            self.assertEqual(
                outcome.artifacts[1].invocation.batch.item_keys,
                (first, second),
            )

    def test_malformed_envelope_never_salvages_items(self) -> None:
        """Bad version, binding or envelope shape yields no guessed success."""
        invocation, registry, results = fixture_v2()
        for extras in (
            {"version": "v1"},
            {"batch_key": digest("wrong")},
            {"extra": "private"},
            {"failures": [{}]},
        ):
            with (
                self.subTest(extras=extras),
                tempfile.TemporaryDirectory() as directory,
            ):
                sends = []

                def send(_prepared: object) -> runtime_v2.ProviderReply:
                    sends.append(1)  # noqa: B023 # Callback finishes within this subtest.
                    return wire_v2(
                        [r.model_dump(mode="json") for r in results.values()],
                        **extras,  # noqa: B023 # Callback finishes within this subtest.
                    )

                runtime = runtime_v2.ArtifactRuntime(
                    identity(), recovery_dir=directory, artifact_dir=directory
                )
                with self.assertRaises(runtime_v2.ArtifactRuntimeError):
                    runtime.execute_batch(
                        invocation,
                        registry,
                        runtime_v2.BatchCallbacks(
                            prepare_v2, send, validate_v2
                        ),
                    )
                self.assertEqual(len(sends), 2)
                self.assertEqual(
                    runtime.prefetch(
                        invocation.lookups, registry, validate_v2
                    ),
                    (),
                )
                self.assertTrue(
                    all(d.valid_items == 0 for d in runtime.diagnostics)
                )

    def test_duplicate_target_repairs_once_unknown_extra_retains_known_success(
        self,
    ) -> None:
        """Duplicates invalidate their target; extras fail without losing peers."""
        invocation, registry, results = fixture_v2()
        first, second = invocation.batch.item_keys
        for duplicate in (True, False):
            with (
                self.subTest(duplicate=duplicate),
                tempfile.TemporaryDirectory() as directory,
            ):
                sends = []

                def send(
                    prepared: runtime_v2.PreparedArtifactRequest,
                ) -> runtime_v2.ProviderReply:
                    sends.append(prepared.metadata.batch.item_keys)  # noqa: B023 # Callback finishes within this subtest.
                    items = [
                        results[key].model_dump(mode="json")
                        for key in prepared.metadata.batch.item_keys
                    ]
                    if len(sends) == 1:  # noqa: B023 # Callback finishes within this subtest.
                        items.append(
                            results[second].model_dump(mode="json")
                            if duplicate  # noqa: B023 # Callback finishes within this subtest.
                            else {"unit_key": digest("unknown")}
                        )
                    return wire_v2(items)

                runtime = runtime_v2.ArtifactRuntime(
                    identity(), recovery_dir=directory, artifact_dir=directory
                )
                callbacks = runtime_v2.BatchCallbacks(
                    prepare_v2, send, validate_v2
                )
                if duplicate:
                    runtime.execute_batch(invocation, registry, callbacks)
                    self.assertEqual(sends, [(first, second), (second,)])
                else:
                    with self.assertRaisesRegex(
                        runtime_v2.ArtifactRuntimeError, "unknown_item"
                    ):
                        runtime.execute_batch(invocation, registry, callbacks)
                    self.assertEqual(len(sends), 1)
                    self.assertEqual(runtime.diagnostics[-1].unknown_items, 1)
                    self.assertEqual(runtime.counters["repairs"], 0)
                    state = runtime.pending_lineages(registry)[0]
                    self.assertEqual(state.consumed_attempts, 1)
                    self.assertEqual(state.nodes[0].status, "failed")
                    self.assertEqual(state.nodes[0].reason, "unknown_item")
                    self.assertEqual(state.nodes[0].missing_keys, ())
                self.assertEqual(
                    len(
                        runtime.prefetch(
                            invocation.lookups, registry, validate_v2
                        )
                    ),
                    2,
                )

    def test_citation_callback_is_required_on_production_and_cache(
        self,
    ) -> None:
        """Current caller rejection prevents both initial acceptance and reuse."""
        invocation, registry, results = fixture_v2()
        with tempfile.TemporaryDirectory() as directory:
            runtime = runtime_v2.ArtifactRuntime(
                identity(), recovery_dir=directory, artifact_dir=directory
            )
            runtime.execute_batch(
                invocation, registry, self.callbacks(results, [])
            )
            calls = []

            def reject(
                lookup: runtime_v2.ArtifactLookup, _result: object
            ) -> None:
                calls.append(lookup.item_key)
                msg = "citation"
                raise runtime_v2.ArtifactRuntimeError(msg)

            self.assertEqual(
                runtime.prefetch(invocation.lookups, registry, reject), ()
            )
            self.assertEqual(set(calls), set(invocation.batch.item_keys))
        with tempfile.TemporaryDirectory() as directory:
            runtime = runtime_v2.ArtifactRuntime(
                identity(), recovery_dir=directory, artifact_dir=directory
            )
            callbacks = self.callbacks(results, [])
            with self.assertRaisesRegex(
                runtime_v2.ArtifactRuntimeError, "citation"
            ):
                runtime.execute_batch(
                    invocation,
                    registry,
                    runtime_v2.BatchCallbacks(
                        callbacks.prepare, callbacks.send, reject
                    ),
                )
            self.assertEqual(
                runtime.prefetch(invocation.lookups, registry, validate_v2), ()
            )

    def test_cross_revision_reuse_checks_all_consumed_dependencies_before_repacking(
        self,
    ) -> None:
        """Unrelated insertion retains hits; every consumed dimension invalidates."""
        invocation, registry, results = fixture_v2()
        with tempfile.TemporaryDirectory() as directory:
            first = runtime_v2.ArtifactRuntime(
                identity(), recovery_dir=directory, artifact_dir=directory
            )
            first.execute_batch(
                invocation, registry, self.callbacks(results, [])
            )
            second = runtime_v2.ArtifactRuntime(
                identity(
                    base_revision="base-new",
                    head_revision="head-new",
                    report_signature="new",
                ),
                recovery_dir=directory,
                artifact_dir=directory,
            )
            inserted = ArtifactDependency(
                kind="evidence", key=digest("unrelated"), digest=digest("new")
            )
            current = runtime_v2.DependencyRegistry(
                dependencies=(*registry.dependencies, inserted)
            )
            hits = second.prefetch(invocation.lookups, current, validate_v2)
            self.assertEqual(len(hits), 2)
            second.execute_batch(
                invocation,
                current,
                self.callbacks(results, []),
                available=hits,
            )
            self.assertEqual(second.usage.report()["calls"], 0)
            self.assertEqual(second.reused_checkpoints, 0)
            self.assertEqual(second.reused_artifacts, 2)
            self.assertEqual(
                len(list(Path(directory).glob("runs-v2/*.json"))), 2
            )
            for dependency in registry.dependencies:
                with self.subTest(kind=dependency.kind):
                    changed = changed_record(
                        dependency,
                        digest=digest(["changed", dependency.digest]),
                    )
                    stale = runtime_v2.DependencyRegistry(
                        dependencies=tuple(
                            changed if d == dependency else d
                            for d in registry.dependencies
                        )
                    )
                    self.assertEqual(
                        second.prefetch(
                            invocation.lookups, stale, validate_v2
                        ),
                        (),
                    )

    def test_descriptor_canonical_order_and_required_bindings(self) -> None:
        """Reject incomplete peers, duplicate names, and result/key mismatches."""
        invocation, registry, results = fixture_v2()
        reordered = changed_record(
            invocation, dependencies=tuple(reversed(invocation.dependencies))
        )
        self.assertEqual(invocation, reordered)
        for kind in (
            "peer",
            "evidence",
            "context",
            "order",
            "terminology",
            "provider",
            "schema",
        ):
            with (
                self.subTest(kind=kind),
                self.assertRaises(
                    (ValueError, runtime_v2.ArtifactRuntimeError)
                ),
            ):
                changed_record(
                    invocation,
                    dependencies=tuple(
                        d for d in invocation.dependencies if d.kind != kind
                    ),
                )
        with self.assertRaises(runtime_v2.ArtifactRuntimeError):
            changed_record(
                registry,
                dependencies=(
                    *registry.dependencies,
                    registry.dependencies[0],
                ),
            )
        with tempfile.TemporaryDirectory() as directory:
            runtime = runtime_v2.ArtifactRuntime(
                identity(), recovery_dir=directory, artifact_dir=directory
            )
            saved = runtime.execute_batch(
                invocation, registry, self.callbacks(results, [])
            ).artifacts[0]
            with self.assertRaises(runtime_v2.ArtifactRuntimeError):
                changed_record(
                    saved,
                    record=changed_record(
                        saved.record,
                        result=results[invocation.batch.item_keys[1]],
                    ),
                )
            with self.assertRaises(runtime_v2.ArtifactRuntimeError):
                changed_record(
                    saved,
                    record=changed_record(
                        saved.record, original_request_digest=digest("wrong")
                    ),
                )

    def test_corrupt_cache_is_miss_and_guarded_import_never_upgrades_v1(
        self,
    ) -> None:
        """Only a complete matching typed v2 record can enter explicit import."""
        invocation, registry, results = fixture_v2(1)
        with tempfile.TemporaryDirectory() as directory:
            runtime = runtime_v2.ArtifactRuntime(
                identity(), recovery_dir=directory, artifact_dir=directory
            )
            artifact = runtime.execute_batch(
                invocation, registry, self.callbacks(results, [])
            ).artifacts[0]
            path = next(Path(directory).glob("artifacts-v2/*/*.json"))
            original = path.read_bytes()
            for corrupted in (
                b"{",
                original.replace(b'"format_version":2', b'"format_version":1'),
                original.replace(
                    b'"explanations":[]', b'"explanations":["private"]'
                ),
            ):
                path.write_bytes(corrupted)
                self.assertEqual(
                    runtime.prefetch(
                        invocation.lookups, registry, validate_v2
                    ),
                    (),
                )
            legacy = CheckpointRecord(
                identity_digest=identity().digest,
                task_id="old",
                schema_name="old",
                request_digest="old",
                validation_version="v1",
                result={"prose": "old body"},
            )
            store = runtime.artifacts
            self.assertFalse(
                store.import_typed(
                    legacy.model_dump_json().encode(),
                    original_run=identity(),
                    invocation=invocation,
                    registry=registry,
                    validate=validate_v2,
                )
            )
            self.assertFalse(
                store.import_typed(
                    original,
                    original_run=identity(report_signature="wrong"),
                    invocation=invocation,
                    registry=registry,
                    validate=validate_v2,
                )
            )
            self.assertTrue(
                store.import_typed(
                    original,
                    original_run=identity(),
                    invocation=invocation,
                    registry=registry,
                    validate=validate_v2,
                )
            )
            self.assertEqual(
                runtime.prefetch(invocation.lookups, registry, validate_v2)[0],
                artifact,
            )

    def test_transient_retry_and_targeted_repair_each_have_one_durable_slot(
        self,
    ) -> None:
        """A restart cannot grant a second retry or second repair."""
        invocation, registry, results = fixture_v2(1)
        for reason, expected_calls in (
            ("timeout", 2),
            ("transport", 2),
            ("rate_limit", 2),
            ("malformed_json", 2),
            ("auth", 1),
            ("prohibited_tools", 1),
        ):
            with (
                self.subTest(reason=reason),
                tempfile.TemporaryDirectory() as directory,
            ):

                def fail(_prepared: object) -> runtime_v2.ProviderReply:
                    raise runtime_v2.ArtifactCallError(reason)  # noqa: B023 # Callback finishes within this subtest.

                callbacks = runtime_v2.BatchCallbacks(
                    prepare_v2, fail, validate_v2
                )
                first = runtime_v2.ArtifactRuntime(
                    identity(), recovery_dir=directory
                )
                with self.assertRaises(runtime_v2.ArtifactRuntimeError):
                    first.execute_batch(invocation, registry, callbacks)
                self.assertEqual(first.usage.report()["calls"], expected_calls)
                second = runtime_v2.ArtifactRuntime(
                    identity(report_signature="unrelated"),
                    recovery_dir=directory,
                )
                with self.assertRaises(runtime_v2.ArtifactRuntimeError):
                    second.execute_batch(
                        invocation, registry, self.callbacks(results, [])
                    )
                self.assertEqual(second.usage.report()["calls"], 0)
                self.assertEqual(
                    second.pending_lineages(registry)[0].consumed_attempts,
                    expected_calls,
                )

    def test_one_transient_then_one_repair_accounts_for_every_attempt(
        self,
    ) -> None:
        """Malformed success and unreported failure still count as current calls."""
        invocation, registry, results = fixture_v2(1)
        sends = []

        def send(
            prepared: runtime_v2.PreparedArtifactRequest,
        ) -> runtime_v2.ProviderReply:
            sends.append(prepared.body)
            if len(sends) == 1:
                msg = "timeout"
                raise runtime_v2.ArtifactCallError(
                    msg, usage={"input_tokens": 3, "output_tokens": 0}
                )
            if len(sends) == 2:
                return runtime_v2.ProviderReply(
                    b"{bad json", usage={"input_tokens": 5, "output_tokens": 1}
                )
            return wire_v2(
                [next(iter(results.values())).model_dump(mode="json")]
            )

        with tempfile.TemporaryDirectory() as directory:
            runtime = runtime_v2.ArtifactRuntime(
                identity(), recovery_dir=directory
            )
            runtime.execute_batch(
                invocation,
                registry,
                runtime_v2.BatchCallbacks(prepare_v2, send, validate_v2),
            )
            self.assertEqual(sends[0], sends[1])
            self.assertNotEqual(sends[1], sends[2])
            self.assertEqual(
                runtime.counters,
                {
                    "attempts": 3,
                    "retries": 1,
                    "repairs": 1,
                    "splits": 0,
                    "reused_artifacts": 0,
                },
            )
            self.assertEqual(runtime.usage.report()["input_tokens"], 8)
            self.assertEqual(runtime.usage.report()["unreported_calls"], 1)

    def test_policy_change_and_repacking_cannot_create_fresh_budget(
        self,
    ) -> None:
        """Original pending keys remain claimed across unrelated report edits."""
        invocation, registry, _results = fixture_v2()
        with tempfile.TemporaryDirectory() as directory:
            store = runtime_v2.RecoveryStore(directory)
            state = store.ensure(
                invocation, runtime_v2.RetryPolicy(total_attempts=2), registry
            )
            store.reserve(
                state.key,
                state.key,
                "initial",
                prepared_v2(invocation.batch.item_keys).metadata,
            )
            with self.assertRaisesRegex(
                runtime_v2.ArtifactRuntimeError, "policy_conflict"
            ):
                store.ensure(
                    invocation,
                    runtime_v2.RetryPolicy(total_attempts=3),
                    registry,
                )
            subset = changed_record(
                invocation,
                batch=prepared_v2(
                    invocation.batch.item_keys[:1]
                ).metadata.batch,
                lookups=invocation.lookups[:1],
                dependencies=tuple(
                    d
                    for d in invocation.dependencies
                    if d.kind not in {"peer", "evidence"}
                    or d.key == invocation.batch.item_keys[0]
                ),
            )
            with self.assertRaisesRegex(
                runtime_v2.ArtifactRuntimeError, "pending_lineage"
            ):
                store.ensure(subset, state.policy, registry)
            self.assertEqual(store.load(state.key).consumed_attempts, 1)
            self.assertEqual(
                store.discover(registry)[0].invocation, invocation
            )

    def test_interruption_after_reservation_preserves_original_budget(
        self,
    ) -> None:
        """An interrupted send is counted, then only its existing retry remains."""
        invocation, registry, results = fixture_v2(1)
        with tempfile.TemporaryDirectory() as directory:
            first = runtime_v2.ArtifactRuntime(
                identity(), recovery_dir=directory
            )

            def interrupt(_prepared: object) -> runtime_v2.ProviderReply:
                raise KeyboardInterrupt

            with self.assertRaises(KeyboardInterrupt):
                first.execute_batch(
                    invocation,
                    registry,
                    runtime_v2.BatchCallbacks(
                        prepare_v2, interrupt, validate_v2
                    ),
                )
            self.assertEqual(first.usage.report()["calls"], 1)
            state = first.pending_lineages(registry)[0]
            self.assertEqual(state.nodes[0].attempts[0].outcome, "cancelled")
            resumed = runtime_v2.ArtifactRuntime(
                identity(report_signature="new"), recovery_dir=directory
            )
            resumed.execute_batch(
                state.invocation, registry, self.callbacks(results, [])
            )
            self.assertEqual(resumed.counters["retries"], 1)
            self.assertEqual(
                resumed.pending_lineages(registry)[0].consumed_attempts, 2
            )

    def test_split_lineage_shares_cap_across_restart(self) -> None:
        """Binary child nodes consume one original total allowance."""
        invocation, registry, results = fixture_v2(4)
        with tempfile.TemporaryDirectory() as directory:
            calls = []

            def fail(_prepared: object) -> runtime_v2.ProviderReply:
                calls.append(1)
                msg = "output_limit"
                raise runtime_v2.ArtifactCallError(msg)

            def split(
                node: runtime_v2.RecoveryNode,
            ) -> tuple[tuple[str, ...], ...]:
                middle = len(node.missing_keys) // 2
                return node.missing_keys[:middle], node.missing_keys[middle:]

            callbacks = runtime_v2.BatchCallbacks(
                prepare_v2, fail, validate_v2, split=split
            )
            policy = runtime_v2.RetryPolicy(total_attempts=2)
            runtime = runtime_v2.ArtifactRuntime(
                identity(), recovery_dir=directory
            )
            with self.assertRaisesRegex(
                runtime_v2.ArtifactRuntimeError, "attempt_limit"
            ):
                runtime.execute_batch(
                    invocation, registry, callbacks, policy=policy
                )
            self.assertEqual(len(calls), 2)
            state = runtime.pending_lineages(registry)[0]
            self.assertEqual(state.consumed_attempts, 2)
            self.assertGreater(len(state.nodes), 1)
            resumed = runtime_v2.ArtifactRuntime(
                identity(), recovery_dir=directory
            )
            with self.assertRaisesRegex(
                runtime_v2.ArtifactRuntimeError, "attempt_limit"
            ):
                resumed.execute_batch(
                    invocation,
                    registry,
                    self.callbacks(results, []),
                    policy=policy,
                )
            self.assertEqual(resumed.usage.report()["calls"], 0)

    def test_split_partitions_reject_overlap_omission_widening_and_singleton(
        self,
    ) -> None:
        """Only strict declared item boundaries can create child requests."""
        invocation, registry, _results = fixture_v2()
        a, b = invocation.batch.item_keys
        for parts in (
            ((a,), (a, b)),
            ((a,), ()),
            ((a,),),
            ((a,), (digest("other"),)),
            ((b,), (a,)),
        ):
            with (
                self.subTest(parts=parts),
                tempfile.TemporaryDirectory() as directory,
            ):
                store = runtime_v2.RecoveryStore(directory)
                state = store.ensure(
                    invocation, runtime_v2.RetryPolicy(), registry
                )
                with self.assertRaises(runtime_v2.ArtifactRuntimeError):
                    store.split(state.key, state.key, parts)
                self.assertEqual(len(store.load(state.key).nodes), 1)
        one, one_registry, _ = fixture_v2(1)
        with tempfile.TemporaryDirectory() as directory:
            store = runtime_v2.RecoveryStore(directory)
            state = store.ensure(one, runtime_v2.RetryPolicy(), one_registry)
            self.assertEqual(state.depth_cap, 0)
            self.assertEqual(state.attempt_cap, 3)
            with self.assertRaisesRegex(
                runtime_v2.ArtifactRuntimeError, "split_limit"
            ):
                store.split(state.key, state.key, ((one.batch.item_keys[0],),))

    def test_partial_artifacts_resume_original_node_after_commit_interruption(
        self,
    ) -> None:
        """A crash between item commits and outcome cannot reset or repack work."""
        invocation, registry, results = fixture_v2()
        first, second = invocation.batch.item_keys
        with tempfile.TemporaryDirectory() as directory:
            runtime = runtime_v2.ArtifactRuntime(
                identity(), recovery_dir=directory, artifact_dir=directory
            )
            original_commit = runtime.artifacts.commit

            def commit(artifact: runtime_v2.StoredArtifact) -> None:
                if artifact.lookup.item_key == second:
                    raise KeyboardInterrupt
                original_commit(artifact)

            with (
                patch.object(runtime.artifacts, "commit", side_effect=commit),
                self.assertRaises(KeyboardInterrupt),
            ):
                runtime.execute_batch(
                    invocation, registry, self.callbacks(results, [])
                )
            resumed = runtime_v2.ArtifactRuntime(
                identity(report_signature="new"),
                recovery_dir=directory,
                artifact_dir=directory,
            )
            state = resumed.pending_lineages(registry)[0]
            self.assertEqual(state.consumed_attempts, 1)
            sends = []
            outcome = resumed.execute_batch(
                state.invocation, registry, self.callbacks(results, sends)
            )
            self.assertEqual(sends, [(second,)])
            self.assertEqual(len(outcome.artifacts), 2)
            self.assertEqual(outcome.artifacts[0].lookup.item_key, first)
            self.assertEqual(
                resumed.pending_lineages(registry)[0].consumed_attempts, 2
            )

    def test_no_cache_restart_cannot_invent_completed_payloads_or_budget(
        self,
    ) -> None:
        """Metadata persists without prose; completed payload loss fails safely."""
        invocation, registry, results = fixture_v2(1)
        with tempfile.TemporaryDirectory() as directory:
            first = runtime_v2.ArtifactRuntime(
                identity(), recovery_dir=directory
            )
            outcome = first.execute_batch(
                invocation, registry, self.callbacks(results, [])
            )
            resumed = runtime_v2.ArtifactRuntime(
                identity(), recovery_dir=directory
            )
            with self.assertRaisesRegex(
                runtime_v2.ArtifactRuntimeError, "results_unavailable"
            ):
                resumed.execute_batch(
                    invocation, registry, self.callbacks(results, [])
                )
            self.assertEqual(resumed.usage.report()["calls"], 0)
            self.assertEqual(
                resumed.execute_batch(
                    invocation,
                    registry,
                    self.callbacks(results, []),
                    available=outcome.artifacts,
                ),
                outcome,
            )
            saved = next(
                Path(directory).glob("recovery-v2/*.json")
            ).read_text()
            self.assertNotIn("explanations", saved)
            self.assertNotIn('"body"', saved)
            self.assertNotIn('"response"', saved)

    def test_corrupt_recovery_fails_closed_without_a_provider_send(
        self,
    ) -> None:
        """Malformed, wrong-version or checksum-corrupt counters never reset."""
        invocation, registry, results = fixture_v2(1)
        for damage in ("truncate", "version", "counter", "missing"):
            with (
                self.subTest(damage=damage),
                tempfile.TemporaryDirectory() as directory,
            ):
                runtime = runtime_v2.ArtifactRuntime(
                    identity(), recovery_dir=directory
                )
                runtime.recovery.ensure(
                    invocation, runtime_v2.RetryPolicy(), registry
                )
                path = next(Path(directory).glob("recovery-v2/*.json"))
                raw = path.read_text()
                if damage == "truncate":
                    raw = raw[:20]
                elif damage == "version":
                    raw = raw.replace(
                        "diffstory.recovery.v2", "diffstory.recovery.v1"
                    )
                elif damage == "counter":
                    raw = raw.replace(
                        '"consumed_attempts":0', '"consumed_attempts":1'
                    )
                else:
                    raw = "{}"
                path.write_text(raw)
                sends = []
                with self.assertRaisesRegex(
                    runtime_v2.ArtifactRuntimeError, "state_corrupt"
                ):
                    runtime.execute_batch(
                        invocation, registry, self.callbacks(results, sends)
                    )
                self.assertEqual(sends, [])

    def test_actual_request_rechecked_on_retry_repair_and_before_initial_send(
        self,
    ) -> None:
        """Changed bytes, dishonest bounds and incompatible retries never send."""
        invocation, registry, results = fixture_v2(1)
        for variant in (
            "digest",
            "input",
            "guard",
            "response",
            "timeout",
            "retry",
        ):
            with (
                self.subTest(variant=variant),
                tempfile.TemporaryDirectory() as directory,
            ):
                sends = []

                def prepare(
                    inv: runtime_v2.ArtifactInvocation,
                    node: runtime_v2.RecoveryNode,
                    purpose: str,
                ) -> runtime_v2.PreparedArtifactRequest:
                    value = prepare_v2(inv, node, purpose)
                    if variant == "retry" and purpose == "initial":  # noqa: B023 # Callback finishes within this subtest.
                        return value
                    if variant in {"digest", "retry"}:  # noqa: B023 # Callback finishes within this subtest.
                        return changed_record(
                            value, body=value.body + b"changed"
                        )
                    metadata = value.metadata
                    if variant == "input":  # noqa: B023 # Callback finishes within this subtest.
                        metadata = changed_record(
                            metadata,
                            batch=changed_record(
                                metadata.batch, input_bound=1
                            ),
                        )
                    elif variant == "guard":  # noqa: B023 # Callback finishes within this subtest.
                        metadata = changed_record(
                            metadata, max_output_tokens=201
                        )
                    elif variant == "response":  # noqa: B023 # Callback finishes within this subtest.
                        metadata = changed_record(
                            metadata, response_byte_limit=1
                        )
                    else:
                        metadata = changed_record(metadata, timeout_seconds=1)
                    return changed_record(value, metadata=metadata)

                def send(_prepared: object) -> runtime_v2.ProviderReply:
                    sends.append(1)  # noqa: B023 # Callback finishes within this subtest.
                    msg = "timeout"
                    raise runtime_v2.ArtifactCallError(msg)

                runtime = runtime_v2.ArtifactRuntime(
                    identity(), recovery_dir=directory
                )
                with self.assertRaises(runtime_v2.ArtifactRuntimeError):
                    runtime.execute_batch(
                        invocation,
                        registry,
                        runtime_v2.BatchCallbacks(prepare, send, validate_v2),
                    )
                self.assertEqual(len(sends), int(variant == "retry"))
                self.assertEqual(
                    runtime.pending_lineages(registry)[0].consumed_attempts,
                    len(sends),
                )

    def test_atomic_storage_failures_preserve_prior_state_and_prevent_send(
        self,
    ) -> None:
        """A failed manifest/reservation write cannot authorize provider work."""
        invocation, registry, results = fixture_v2(1)
        with tempfile.TemporaryDirectory() as directory:
            runtime = runtime_v2.ArtifactRuntime(
                identity(), recovery_dir=directory
            )
            runtime.recovery.ensure(
                invocation, runtime_v2.RetryPolicy(), registry
            )
            path = next(Path(directory).glob("recovery-v2/*.json"))
            previous = path.read_bytes()
            original_replace = Path.replace

            def fail_reservation(source: Path, target: Path) -> Path:
                if target == path:
                    msg = "private error"
                    raise OSError(msg)
                return original_replace(source, target)

            sends = []
            with (
                patch.object(Path, "replace", fail_reservation),
                self.assertRaisesRegex(
                    runtime_v2.ArtifactRuntimeError, "storage"
                ),
            ):
                runtime.execute_batch(
                    invocation, registry, self.callbacks(results, sends)
                )
            self.assertEqual(path.read_bytes(), previous)
            self.assertEqual(sends, [])
            self.assertEqual(list(Path(directory).rglob(".v2-*")), [])
            self.assertEqual(path.stat().st_mode & 0o777, 0o600)
            with (
                patch.object(Path, "replace", side_effect=OSError),
                self.assertRaises(runtime_v2.ArtifactRuntimeError),
            ):
                runtime.execute_batch(
                    invocation, registry, self.callbacks(results, sends)
                )
            self.assertEqual(sends, [])

    def test_separate_process_reservations_are_serialized(self) -> None:
        """Two processes cannot both spend the same node's initial slot."""
        invocation, registry, _results = fixture_v2(1)
        script = """
import sys
sys.path.insert(0, "tests")
from test_narrative_runtime import fixture_v2, prepared_v2
from diffstory.narrative_runtime import RecoveryStore, RetryPolicy, ArtifactRuntimeError
inv, registry, _ = fixture_v2(1)
store = RecoveryStore(sys.argv[1])
print("ready", flush=True)
sys.stdin.readline()
try:
    state = store.ensure(inv, RetryPolicy(total_attempts=1), registry)
    store.reserve(state.key, state.key, "initial", prepared_v2(inv.batch.item_keys).metadata)
    print("reserved", flush=True)
except ArtifactRuntimeError as error:
    print(error.reason, flush=True)
"""
        with tempfile.TemporaryDirectory() as directory:
            store = runtime_v2.RecoveryStore(directory)
            state = store.ensure(
                invocation, runtime_v2.RetryPolicy(total_attempts=1), registry
            )
            processes = [
                subprocess.Popen(
                    [sys.executable, "-c", script, directory],
                    stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                )
                for _ in range(2)
            ]
            try:
                for process in processes:
                    self.assertEqual(
                        process.stdout.readline().strip(), "ready"
                    )
                for process in processes:
                    process.stdin.write("go\n")
                    process.stdin.flush()
                output = [
                    process.communicate(timeout=10) for process in processes
                ]
                self.assertEqual(
                    sorted(stdout.strip() for stdout, _stderr in output),
                    ["attempt_limit", "reserved"],
                )
                self.assertTrue(
                    all(process.returncode == 0 for process in processes)
                )
                self.assertEqual(store.load(state.key).consumed_attempts, 1)
            finally:
                for process in processes:
                    if process.poll() is None:
                        process.kill()
                    process.communicate()

    def test_safe_diagnostics_and_semantic_stage_wrapping(self) -> None:
        """Malicious exception strings cannot replace safe semantic task data."""
        invocation, registry, _results = fixture_v2(1)
        private_text = "PRIVATE SOURCE RESPONSE CREDENTIAL SENTINEL"
        with tempfile.TemporaryDirectory() as directory:
            events = []
            runtime = runtime_v2.ArtifactRuntime(
                identity(), recovery_dir=directory, on_progress=events.append
            )

            def fail(_prepared: object) -> runtime_v2.ProviderReply:
                raise runtime_v2.ArtifactCallError(
                    private_text,
                    usage={"private": private_text},
                    response_bytes=-1,
                )

            callbacks = runtime_v2.BatchCallbacks(
                prepare_v2, fail, validate_v2
            )
            job = runtime_v2.SemanticJob(
                invocation.semantic_key,
                lambda: runtime.execute_batch(invocation, registry, callbacks),
            )
            with self.assertRaises(RuntimeStageError) as caught:
                runtime.run_stage("details", [job])
            error = caught.exception
            self.assertEqual(error.task_digest, invocation.semantic_key)
            self.assertEqual(error.stage_position, 0)
            self.assertEqual(error.diagnostic.reason, "generic")
            self.assertEqual(error.diagnostic.stage_position, 0)
            self.assertIsNone(error.diagnostic.response_bytes)
            self.assertFalse(error.diagnostic.usage_available)
            serialized = "".join(
                path.read_text() for path in Path(directory).rglob("*.json")
            )
            serialized += str(error) + "".join(
                event.model_dump_json() for event in events
            )
            self.assertNotIn(private_text, serialized)
            self.assertEqual(runtime.usage.report()["calls"], 2)
            self.assertEqual(runtime.usage.report()["unreported_calls"], 2)

    def test_current_ledger_callback_rebinds_shifted_content_stable_citations(
        self,
    ) -> None:
        """Real M1 validation accepts a shifted span and rejects wrong-side refs."""
        old = build_evidence_ledger(
            compile_snapshot(snap({"x.py": BEFORE}, {"x.py": AFTER}))
        )
        current = build_evidence_ledger(
            compile_snapshot(
                snap(
                    {"x.py": "# prefix\n" + BEFORE},
                    {"x.py": "# prefix\n" + AFTER},
                )
            )
        )
        unit = next(u for u in old.units if u.change_key)
        new_unit = next(u for u in current.units if u.id == unit.id)
        result = authored_result(unit)
        template, _, _ = fixture_v2(1)
        lookup = changed_record(template.lookups[0], item_key=unit.id)
        deps = tuple(
            d
            for d in template.dependencies
            if d.kind not in {"evidence", "peer"}
        ) + tuple(
            ArtifactDependency(kind=kind, key=unit.id, digest=digest(unit.id))
            for kind in ("evidence", "peer")
        )
        invocation = runtime_v2.ArtifactInvocation(
            batch=prepared_v2((unit.id,)).metadata.batch,
            lookups=(lookup,),
            dependencies=deps,
        )
        registry = runtime_v2.DependencyRegistry(dependencies=deps)
        old_index = build_ledger_index(old)
        new_index = build_ledger_index(current)

        def validate_old(_lookup: object, value: DetailResult) -> None:
            validate_detail_result(value, unit, old, index=old_index)

        def validate_new(_lookup: object, value: DetailResult) -> None:
            validate_detail_result(value, new_unit, current, index=new_index)

        def send(_prepared: object) -> runtime_v2.ProviderReply:
            return wire_v2([result.model_dump(mode="json")])

        with tempfile.TemporaryDirectory() as directory:
            first = runtime_v2.ArtifactRuntime(
                identity(), recovery_dir=directory, artifact_dir=directory
            )
            first.execute_batch(
                invocation,
                registry,
                runtime_v2.BatchCallbacks(prepare_v2, send, validate_old),
            )
            second = runtime_v2.ArtifactRuntime(
                identity(head_revision="shifted"),
                recovery_dir=directory,
                artifact_dir=directory,
            )
            hit = second.prefetch((lookup,), registry, validate_new)[0]
            self.assertEqual(hit.record.result, result)
            ref = span_reference(unit.spans[0], unit)
            self.assertEqual(
                resolve_span_ref(
                    ref, new_unit, current, index=new_index
                ).line_start,
                unit.spans[0].line_start + 1,
            )
            explanations = list(result.explanations)
            cited = next(
                i
                for i, explanation in enumerate(explanations)
                if explanation.citations
            )
            refs = list(explanations[cited].citations)
            refs[0] = changed_record(
                refs[0], side="head" if refs[0].side == "base" else "base"
            )
            explanations[cited] = changed_record(
                explanations[cited], citations=tuple(refs)
            )
            corrupted = changed_record(
                result, explanations=tuple(explanations)
            )
            bad_artifact = changed_record(
                hit, record=changed_record(hit.record, result=corrupted)
            )
            for path in Path(directory).glob("artifacts-v2/*/*.json"):
                path.unlink()
            second.artifacts.commit(bad_artifact)
            self.assertEqual(
                second.prefetch((lookup,), registry, validate_new), ()
            )

    def test_missing_item_and_declared_failure_keep_successful_peer(
        self,
    ) -> None:
        """Missing or explicitly failed keys receive one targeted repair."""
        invocation, registry, results = fixture_v2()
        first, second = invocation.batch.item_keys
        for declared in (False, True):
            with (
                self.subTest(declared=declared),
                tempfile.TemporaryDirectory() as directory,
            ):
                sends = []

                def send(
                    prepared: runtime_v2.PreparedArtifactRequest,
                ) -> runtime_v2.ProviderReply:
                    sends.append(prepared.metadata.batch.item_keys)  # noqa: B023 # Callback finishes within this subtest.
                    if len(sends) > 1:  # noqa: B023 # Callback finishes within this subtest.
                        return wire_v2(
                            [results[second].model_dump(mode="json")]
                        )
                    failures = (
                        [{"item_key": second, "reason": "invalid_citation"}]
                        if declared  # noqa: B023 # Callback finishes within this subtest.
                        else []
                    )
                    return wire_v2(
                        [results[first].model_dump(mode="json")],
                        failures=failures,
                    )

                runtime = runtime_v2.ArtifactRuntime(
                    identity(), recovery_dir=directory, artifact_dir=directory
                )
                runtime.execute_batch(
                    invocation,
                    registry,
                    runtime_v2.BatchCallbacks(prepare_v2, send, validate_v2),
                )
                self.assertEqual(sends, [(first, second), (second,)])
                self.assertEqual(
                    len(
                        runtime.prefetch(
                            invocation.lookups, registry, validate_v2
                        )
                    ),
                    2,
                )

    def test_unknown_extra_repairs_only_missing_target_with_original_evidence(
        self,
    ) -> None:
        """Keep valid peers and unknown diagnostics while repairing known gaps."""
        invocation, registry, results = fixture_v2()
        first, second = invocation.batch.item_keys

        class EvidenceItem(runtime_v2.Contract):
            """Bind fixture evidence to one known request target."""

            unit_key: str
            evidence: str

        evidence = {
            key: EvidenceItem(
                unit_key=key, evidence=f"fixture evidence {index}"
            )
            for index, key in enumerate(invocation.batch.item_keys)
        }
        capacity = ModelCapacity(1_000_000, 100_000)
        limits = RequestPackingLimits()

        def prepare(
            keys: tuple[str, ...], purpose: str
        ) -> runtime_v2.PreparedArtifactRequest:
            request = prepare_request_items(
                tuple(evidence[key] for key in keys),
                stage="detail",
                keys=lambda item: item.unit_key,
                groups=lambda _item: (),
                serializer=lambda items, output: json.dumps(
                    {
                        "items": [
                            item.model_dump(mode="json") for item in items
                        ],
                        "purpose": purpose,
                        "output": output,
                    }
                ).encode(),
                allowance=lambda _items: 300,
                capacity=capacity,
                limits=limits,
            )
            return runtime_v2.bind_prepared_request(request, capacity, limits)

        invocation = changed_record(
            invocation,
            batch=prepare(
                invocation.batch.item_keys, "initial"
            ).metadata.batch,
        )
        unknown = changed_record(results[second], unit_key=digest("unknown"))
        send = Mock(
            side_effect=[
                replace(
                    wire_v2(
                        [
                            results[first].model_dump(mode="json"),
                            unknown.model_dump(mode="json"),
                        ]
                    ),
                    usage={"input_tokens": 11, "output_tokens": 3},
                ),
                replace(
                    wire_v2([results[second].model_dump(mode="json")]),
                    usage={"input_tokens": 7, "output_tokens": 2},
                ),
            ]
        )
        callbacks = runtime_v2.BatchCallbacks(
            lambda _inv, node, purpose: prepare(node.missing_keys, purpose),
            send,
            validate_v2,
        )
        with tempfile.TemporaryDirectory() as directory:
            runtime = runtime_v2.ArtifactRuntime(
                identity(), recovery_dir=directory, artifact_dir=directory
            )
            outcome = runtime.execute_batch(invocation, registry, callbacks)
            self.assertEqual(
                outcome.detail_batch().items, tuple(results.values())
            )
            requests = [call.args[0] for call in send.call_args_list]
            self.assertEqual(
                [request.metadata.batch.item_keys for request in requests],
                [(first, second), (second,)],
            )
            self.assertEqual(
                json.loads(requests[1].body),
                {
                    "items": [evidence[second].model_dump(mode="json")],
                    "purpose": "repair",
                    "output": requests[1].metadata.batch.output_reservation,
                },
            )
            self.assertEqual(
                runtime.usage.report(),
                {
                    "calls": 2,
                    "input_tokens": 18,
                    "cached_input_tokens": 0,
                    "output_tokens": 5,
                    "unreported_calls": 0,
                },
            )
            diagnostic = runtime.diagnostics[0]
            self.assertEqual(diagnostic.reason, "unknown_item")
            self.assertEqual(diagnostic.attempt, 1)
            self.assertEqual(diagnostic.valid_items, 1)
            self.assertEqual(diagnostic.missing_items, 1)
            self.assertEqual(diagnostic.invalid_items, 0)
            self.assertEqual(diagnostic.unknown_items, 1)
            self.assertTrue(diagnostic.usage_available)
            self.assertEqual(runtime.diagnostics[1].reason, "generic")
            state = runtime.pending_lineages(registry)[0]
            self.assertEqual(state.invocation, invocation)
            self.assertEqual(state.consumed_attempts, 2)
            self.assertEqual(state.nodes[0].repair_count, 1)
            self.assertEqual(state.nodes[0].status, "succeeded")
            self.assertEqual(state.nodes[0].missing_keys, ())
            self.assertEqual(state.nodes[0].attempts[0].reason, "unknown_item")
            self.assertEqual(state.nodes[0].attempts[0].outcome, "failed")
            self.assertEqual(
                len(list(Path(directory).glob("artifacts-v2/*/*.json"))), 2
            )
            for artifact, request in zip(
                outcome.artifacts, requests, strict=True
            ):
                self.assertEqual(artifact.invocation, invocation)
                self.assertEqual(
                    artifact.record.dependencies, invocation.dependencies
                )
                self.assertEqual(artifact.production_request, request.metadata)
            resumed = runtime_v2.ArtifactRuntime(
                identity(), recovery_dir=directory, artifact_dir=directory
            )
            self.assertEqual(
                resumed.execute_batch(invocation, registry, callbacks), outcome
            )
            self.assertEqual(send.call_count, 2)
            self.assertEqual(resumed.usage.report()["calls"], 0)
            self.assertEqual(resumed.usage.report()["input_tokens"], 0)
            self.assertEqual(resumed.usage.report()["output_tokens"], 0)

    def test_unknown_extra_bad_repair_exhausts_original_lineage_on_restart(
        self,
    ) -> None:
        """Unknown or invalid repairs cannot reset budgets or erase valid peers."""
        invocation, registry, results = fixture_v2()
        first, second = invocation.batch.item_keys
        unknown = {"unit_key": digest("unknown")}
        invalid = {"unit_key": second, "explanations": "malformed"}
        for repair_items, reason in (
            ([unknown], "unknown_item"),
            ([invalid], "schema"),
        ):
            with (
                self.subTest(reason=reason),
                tempfile.TemporaryDirectory() as directory,
            ):
                send = Mock(
                    side_effect=[
                        wire_v2(
                            [
                                results[first].model_dump(mode="json"),
                                invalid,
                                unknown,
                            ]
                        ),
                        wire_v2(repair_items),
                    ]
                )
                callbacks = runtime_v2.BatchCallbacks(
                    prepare_v2, send, validate_v2
                )
                runtime = runtime_v2.ArtifactRuntime(
                    identity(), recovery_dir=directory, artifact_dir=directory
                )
                with self.assertRaisesRegex(
                    runtime_v2.ArtifactRuntimeError, reason
                ):
                    runtime.execute_batch(invocation, registry, callbacks)
                self.assertEqual(
                    [
                        call.args[0].metadata.batch.item_keys
                        for call in send.call_args_list
                    ],
                    [(first, second), (second,)],
                )
                self.assertEqual(runtime.usage.report()["calls"], 2)
                self.assertEqual(runtime.usage.report()["unreported_calls"], 2)
                self.assertEqual(runtime.diagnostics[0].reason, "unknown_item")
                self.assertEqual(runtime.diagnostics[0].invalid_items, 1)
                self.assertEqual(runtime.diagnostics[0].unknown_items, 1)
                state = runtime.pending_lineages(registry)[0]
                self.assertEqual(state.consumed_attempts, 2)
                self.assertEqual(state.nodes[0].repair_count, 1)
                self.assertEqual(state.nodes[0].status, "failed")
                self.assertEqual(state.nodes[0].missing_keys, (second,))
                saved = runtime.prefetch(
                    invocation.lookups, registry, validate_v2
                )
                self.assertEqual(
                    tuple(a.lookup.item_key for a in saved), (first,)
                )
                self.assertEqual(
                    len(list(Path(directory).glob("artifacts-v2/*/*.json"))), 1
                )
                resumed = runtime_v2.ArtifactRuntime(
                    identity(report_signature="changed"),
                    recovery_dir=directory,
                    artifact_dir=directory,
                )
                with self.assertRaisesRegex(
                    runtime_v2.ArtifactRuntimeError, reason
                ):
                    resumed.execute_batch(invocation, registry, callbacks)
                with self.assertRaisesRegex(
                    runtime_v2.ArtifactRuntimeError, "policy_conflict"
                ):
                    resumed.execute_batch(
                        invocation,
                        registry,
                        callbacks,
                        policy=runtime_v2.RetryPolicy(total_attempts=13),
                    )
                subset = changed_record(
                    invocation,
                    batch=prepared_v2((second,)).metadata.batch,
                    lookups=invocation.lookups[1:],
                    dependencies=tuple(
                        dependency
                        for dependency in invocation.dependencies
                        if dependency.kind not in {"peer", "evidence"}
                        or dependency.key == second
                    ),
                )
                with self.assertRaisesRegex(
                    runtime_v2.ArtifactRuntimeError, "pending_lineage"
                ):
                    resumed.execute_batch(subset, registry, callbacks)
                self.assertEqual(send.call_count, 2)
                self.assertEqual(resumed.usage.report()["calls"], 0)
                self.assertEqual(resumed.counters["repairs"], 0)
                self.assertEqual(resumed.pending_lineages(registry), (state,))
                self.assertEqual(
                    resumed.prefetch(
                        invocation.lookups, registry, validate_v2
                    ),
                    saved,
                )

    def test_unknown_extra_interruption_before_repair_keeps_reservations(
        self,
    ) -> None:
        """Interrupt before or after repair reservation without renewing its slot."""
        invocation, registry, results = fixture_v2()
        first, second = invocation.batch.item_keys
        for stop_event in ("call_completed", "call_started"):
            with (
                self.subTest(stop_event=stop_event),
                tempfile.TemporaryDirectory() as directory,
            ):

                def interrupt(
                    event: ProgressEvent, *, stop: str = stop_event
                ) -> None:
                    if event.event == stop and event.provider_calls == 1:
                        raise KeyboardInterrupt

                send = Mock(
                    side_effect=[
                        wire_v2(
                            [
                                results[first].model_dump(mode="json"),
                                {"unit_key": digest("unknown")},
                            ]
                        ),
                        wire_v2([{"unit_key": digest("unknown")}]),
                    ]
                )
                callbacks = runtime_v2.BatchCallbacks(
                    prepare_v2, send, validate_v2
                )
                runtime = runtime_v2.ArtifactRuntime(
                    identity(),
                    recovery_dir=directory,
                    artifact_dir=directory,
                    on_progress=interrupt,
                )
                with self.assertRaises(KeyboardInterrupt):
                    runtime.execute_batch(invocation, registry, callbacks)
                reserved_repair = int(stop_event == "call_started")
                state = runtime.pending_lineages(registry)[0]
                self.assertEqual(state.consumed_attempts, 1 + reserved_repair)
                self.assertEqual(state.nodes[0].repair_count, reserved_repair)
                self.assertEqual(state.nodes[0].missing_keys, (second,))
                self.assertEqual(
                    state.nodes[0].attempts[0].reason, "unknown_item"
                )
                self.assertEqual(runtime.diagnostics[0].unknown_items, 1)
                self.assertEqual(
                    runtime.usage.report()["calls"], 1 + reserved_repair
                )
                self.assertEqual(send.call_count, 1)
                if reserved_repair:
                    self.assertEqual(
                        state.nodes[0].attempts[-1].outcome, "cancelled"
                    )
                resumed = runtime_v2.ArtifactRuntime(
                    identity(report_signature="changed"),
                    recovery_dir=directory,
                    artifact_dir=directory,
                )
                with self.assertRaisesRegex(
                    runtime_v2.ArtifactRuntimeError, "unknown_item"
                ):
                    resumed.execute_batch(
                        state.invocation,
                        registry,
                        callbacks,
                        policy=state.policy,
                    )
                self.assertEqual(
                    [
                        call.args[0].metadata.batch.item_keys
                        for call in send.call_args_list
                    ],
                    [(first, second), (second,)],
                )
                self.assertEqual(resumed.usage.report()["calls"], 1)
                self.assertEqual(
                    resumed.counters["repairs"], 1 - reserved_repair
                )
                self.assertEqual(resumed.counters["retries"], reserved_repair)
                final_state = resumed.pending_lineages(registry)[0]
                self.assertEqual(final_state.invocation, invocation)
                self.assertEqual(
                    final_state.consumed_attempts, 2 + reserved_repair
                )
                self.assertEqual(final_state.nodes[0].repair_count, 1)
                self.assertEqual(final_state.nodes[0].status, "failed")
                self.assertEqual(
                    tuple(a.purpose for a in final_state.nodes[0].attempts),
                    ("initial", "repair", "transient")
                    if reserved_repair
                    else ("initial", "repair"),
                )
                again = runtime_v2.ArtifactRuntime(
                    identity(), recovery_dir=directory, artifact_dir=directory
                )
                with self.assertRaisesRegex(
                    runtime_v2.ArtifactRuntimeError, "unknown_item"
                ):
                    again.execute_batch(invocation, registry, callbacks)
                self.assertEqual(send.call_count, 2)
                self.assertEqual(again.usage.report()["calls"], 0)
                self.assertEqual(
                    again.pending_lineages(registry), (final_state,)
                )
                self.assertEqual(
                    tuple(
                        a.lookup.item_key
                        for a in again.prefetch(
                            invocation.lookups, registry, validate_v2
                        )
                    ),
                    (first,),
                )

    def test_failed_repair_never_repeats_or_discards_valid_result(
        self,
    ) -> None:
        """A valid peer remains usable after the other item exhausts its repair."""
        invocation, registry, results = fixture_v2()
        first, second = invocation.batch.item_keys
        sends = []

        def send(
            prepared: runtime_v2.PreparedArtifactRequest,
        ) -> runtime_v2.ProviderReply:
            sends.append(prepared.metadata.batch.item_keys)
            return wire_v2(
                [results[first].model_dump(mode="json")]
                if len(sends) == 1
                else []
            )

        with tempfile.TemporaryDirectory() as directory:
            runtime = runtime_v2.ArtifactRuntime(
                identity(), recovery_dir=directory, artifact_dir=directory
            )
            callbacks = runtime_v2.BatchCallbacks(
                prepare_v2, send, validate_v2
            )
            with self.assertRaises(runtime_v2.ArtifactRuntimeError):
                runtime.execute_batch(invocation, registry, callbacks)
            self.assertEqual(sends, [(first, second), (second,)])
            self.assertEqual(
                runtime.prefetch(invocation.lookups, registry, validate_v2)[
                    0
                ].lookup.item_key,
                first,
            )
            resumed = runtime_v2.ArtifactRuntime(
                identity(), recovery_dir=directory, artifact_dir=directory
            )
            with self.assertRaises(runtime_v2.ArtifactRuntimeError):
                resumed.execute_batch(invocation, registry, callbacks)
            self.assertEqual(resumed.usage.report()["calls"], 0)

    def test_pre_send_size_failure_splits_without_sending_oversized_body(
        self,
    ) -> None:
        """All child sends use actual bodies and one original descriptor/budget."""
        invocation, registry, results = fixture_v2()
        sends = []

        def prepare(
            inv: runtime_v2.ArtifactInvocation,
            node: runtime_v2.RecoveryNode,
            purpose: str,
        ) -> runtime_v2.PreparedArtifactRequest:
            value = prepare_v2(inv, node, purpose)
            if len(node.missing_keys) > 1:
                value = changed_record(
                    value,
                    metadata=changed_record(value.metadata, context_tokens=1),
                )
            return value

        callbacks = self.callbacks(results, sends)
        with tempfile.TemporaryDirectory() as directory:
            runtime = runtime_v2.ArtifactRuntime(
                identity(), recovery_dir=directory, artifact_dir=directory
            )
            outcome = runtime.execute_batch(
                invocation,
                registry,
                runtime_v2.BatchCallbacks(
                    prepare,
                    callbacks.send,
                    validate_v2,
                    split=lambda node: tuple(
                        (key,) for key in node.missing_keys
                    ),
                ),
            )
            self.assertEqual(
                sends, [(key,) for key in invocation.batch.item_keys]
            )
            self.assertEqual(runtime.counters["attempts"], 2)
            self.assertEqual(runtime.counters["splits"], 1)
            self.assertTrue(
                all(a.invocation == invocation for a in outcome.artifacts)
            )
            self.assertTrue(
                all(
                    a.production_request.batch.item_keys
                    != invocation.batch.item_keys
                    for a in outcome.artifacts
                )
            )
            self.assertEqual(
                runtime.pending_lineages(registry)[0].nodes[0].attempts, ()
            )

    def test_response_size_failure_uses_shared_split_policy(self) -> None:
        """Measured response length overrides an optimistic response estimate."""
        invocation, registry, results = fixture_v2()

        def send(
            prepared: runtime_v2.PreparedArtifactRequest,
        ) -> runtime_v2.ProviderReply:
            keys = prepared.metadata.batch.item_keys
            reply = wire_v2(
                [results[key].model_dump(mode="json") for key in keys]
            )
            return runtime_v2.ProviderReply(
                reply.response, response_bytes=10_001 if len(keys) > 1 else 100
            )

        with tempfile.TemporaryDirectory() as directory:
            runtime = runtime_v2.ArtifactRuntime(
                identity(), recovery_dir=directory
            )
            runtime.execute_batch(
                invocation,
                registry,
                runtime_v2.BatchCallbacks(
                    prepare_v2,
                    send,
                    validate_v2,
                    split=lambda node: tuple(
                        (key,) for key in node.missing_keys
                    ),
                ),
            )
            self.assertEqual(runtime.counters["attempts"], 3)
            self.assertEqual(runtime.diagnostics[0].reason, "response_limit")
            self.assertEqual(runtime.diagnostics[0].response_bytes, 10_001)

    def test_split_depth_bound_and_original_item_cap_are_enforced(
        self,
    ) -> None:
        """Neither oversized singleton nor depth-limited children can widen work."""
        invocation, registry, _results = fixture_v2(3)
        with tempfile.TemporaryDirectory() as directory:
            store = runtime_v2.RecoveryStore(directory)
            state = store.ensure(
                invocation, runtime_v2.RetryPolicy(max_split_depth=1), registry
            )
            keys = invocation.batch.item_keys
            state = store.split(state.key, state.key, (keys[:1], keys[1:]))
            with self.assertRaisesRegex(
                runtime_v2.ArtifactRuntimeError, "split_limit"
            ):
                store.split(
                    state.key, state.nodes[-1].key, ((keys[1],), (keys[2],))
                )
            self.assertEqual(state.depth_cap, 1)
        one, current, _ = fixture_v2(1)
        with tempfile.TemporaryDirectory() as directory:
            state = runtime_v2.RecoveryStore(directory).ensure(
                one, runtime_v2.RetryPolicy(total_attempts=100), current
            )
            self.assertEqual(state.attempt_cap, 3)

    def test_atomic_artifact_failure_leaves_previous_committed_data(
        self,
    ) -> None:
        """A failed replacement cannot damage the last complete typed artifact."""
        invocation, registry, results = fixture_v2(1)
        with tempfile.TemporaryDirectory() as directory:
            runtime = runtime_v2.ArtifactRuntime(
                identity(), recovery_dir=directory, artifact_dir=directory
            )
            artifact = runtime.execute_batch(
                invocation, registry, self.callbacks(results, [])
            ).artifacts[0]
            path = next(Path(directory).glob("artifacts-v2/*/*.json"))
            previous = path.read_bytes()
            with (
                patch.object(
                    Path,
                    "replace",
                    side_effect=OSError("private storage text"),
                ),
                self.assertRaisesRegex(
                    runtime_v2.ArtifactRuntimeError, "storage"
                ),
            ):
                runtime.artifacts.commit(artifact)
            self.assertEqual(path.read_bytes(), previous)
            self.assertEqual(
                runtime.prefetch(invocation.lookups, registry, validate_v2),
                (artifact,),
            )
            self.assertEqual(path.stat().st_mode & 0o777, 0o600)

    def test_os_lock_wait_is_bounded_without_removing_owner_lock(self) -> None:
        """A busy live owner cannot be mistaken for a stale lock."""
        with tempfile.TemporaryDirectory() as directory:
            store = runtime_v2.RecoveryStore(directory, lock_timeout=0.02)
            with store.lease():
                with (
                    self.assertRaisesRegex(
                        runtime_v2.ArtifactRuntimeError, "busy"
                    ),
                    store.lease(),
                ):
                    self.fail("A second independent flock must not acquire")
                self.assertTrue((store.directory / "store.lock").exists())
            with store.lease():
                self.assertTrue((store.directory / "store.lock").exists())

    def test_frozen_strict_descriptors_and_registry_index(self) -> None:
        """Strict public records reject mutation and mutable caller substitutes."""
        invocation, registry, _results = fixture_v2()
        with self.assertRaises(ValidationError):
            invocation.batch = prepared_v2(
                invocation.batch.item_keys
            ).metadata.batch
        with self.assertRaises(ValidationError):
            changed_record(invocation, lookups=list(invocation.lookups))
        with self.assertRaises(ValidationError):
            changed_record(runtime_v2.RetryPolicy(), total_attempts="12")
        revalidated = runtime_v2.DependencyRegistry.model_validate(registry)
        self.assertTrue(revalidated.contains(invocation.dependencies))
        with self.assertRaises(TypeError):
            registry._index.values[("context", digest("x"))] = digest("y")  # noqa: SLF001 # Verify the private acceleration structure is immutable.
        with self.assertRaises(FrozenInstanceError):
            registry._index.dependencies = ()  # noqa: SLF001 # The binding must be immutable as well as the map.

    def test_registry_copy_cannot_reuse_a_previous_field_index(self) -> None:
        """Reject stale copies until strict validation binds their current tuple."""
        old = ArtifactDependency(kind="context", key="a" * 64, digest="b" * 64)
        new = old.model_copy(update={"digest": "c" * 64})
        original = runtime_v2.DependencyRegistry(dependencies=(old,))
        copied = original.model_copy(update={"dependencies": (new,)})
        self.assertEqual(copied.dependencies, (new,))
        self.assertFalse(copied.contains((old,)))
        self.assertFalse(copied.contains((new,)))
        self.assertFalse(copied.contains(()))
        self.assertTrue(original.contains((old,)))

        current = runtime_v2.DependencyRegistry.model_validate(copied)
        self.assertFalse(current.contains((old,)))
        self.assertTrue(current.contains((new,)))
        self.assertIsNot(current, copied)
        self.assertEqual(
            runtime_v2.Contract.model_config["revalidate_instances"], "always"
        )
        with self.assertRaises((TypeError, ValueError)):
            replace(current._index, values=original._index.values)  # noqa: SLF001 # Copies cannot inject a stale map.
        copied._index = replace(original._index, registry=copied)  # noqa: SLF001 # Rebinding an index must validate and derive current values.
        self.assertFalse(copied.contains((old,)))
        self.assertTrue(copied.contains((new,)))

    def test_revalidated_registries_do_not_need_cyclic_collection(
        self,
    ) -> None:
        """Discard full registry copies while cyclic collection is disabled."""
        invocation, registry, _results = fixture_v2(8)
        enabled = gc.isenabled()
        gc.disable()
        try:
            copies = [
                runtime_v2.DependencyRegistry.model_validate(registry)
                for _ in range(8)
            ]
            observations = [weakref.ref(value) for value in copies]
            self.assertTrue(
                all(
                    value.contains(invocation.dependencies) for value in copies
                )
            )
            self.assertTrue(
                all(
                    value.dependencies is not registry.dependencies
                    for value in copies
                )
            )
            self.assertEqual(
                len({id(value.dependencies) for value in copies}), 8
            )
            copies.clear()
            self.assertEqual(sum(ref() is not None for ref in observations), 0)
            self.assertTrue(registry.contains(invocation.dependencies))
        finally:
            if enabled:
                gc.enable()

    def test_public_cache_rejects_unbound_and_invalid_registry_contexts(
        self,
    ) -> None:
        """Real cache reads reject stale peers, context, and private index swaps."""
        invocation, registry, results = fixture_v2(1)
        variants = {
            "unchanged copy": registry.model_copy(),
            "unvalidated construction": runtime_v2.DependencyRegistry.model_construct(
                dependencies=registry.dependencies
            ),
        }
        for kind in ("context", "peer"):
            changed = tuple(
                d.model_copy(update={"digest": digest([kind, "changed"])})
                if d.kind == kind
                else d
                for d in registry.dependencies
            )
            variants[f"copied changed {kind}"] = registry.model_copy(
                update={"dependencies": changed}
            )
            variants[f"validated changed {kind}"] = (
                runtime_v2.DependencyRegistry(dependencies=changed)
            )
            missing = tuple(d for d in registry.dependencies if d.kind != kind)
            variants[f"copied missing {kind}"] = registry.model_copy(
                update={"dependencies": missing}
            )
            variants[f"validated missing {kind}"] = (
                runtime_v2.DependencyRegistry(dependencies=missing)
            )

        corrupt_fields = (
            (*registry.dependencies, registry.dependencies[0]),
            tuple(
                d.model_copy(update={"digest": "invalid"})
                if d.kind == "context"
                else d
                for d in registry.dependencies
            ),
            list(registry.dependencies),
        )
        for position, fields in enumerate(corrupt_fields):
            copied = registry.model_copy(update={"dependencies": fields})
            variants[f"corrupt fields {position}"] = copied
            with self.assertRaises(
                (ValidationError, runtime_v2.ArtifactRuntimeError)
            ):
                runtime_v2.DependencyRegistry.model_validate(copied)
            with self.assertRaises(
                (ValidationError, runtime_v2.ArtifactRuntimeError)
            ):
                replace(registry._index, registry=copied)  # noqa: SLF001 # Index construction must not bypass strict record validation.

        # PrivateAttr permits assignment even on a frozen Pydantic record.
        for position, index in enumerate(
            (
                None,
                {},
                MappingProxyType(
                    {(d.kind, d.key): d.digest for d in registry.dependencies}
                ),
                registry._index,  # noqa: SLF001 # Borrowed valid maps must not authenticate another registry.
            )
        ):
            rebound = runtime_v2.DependencyRegistry(
                dependencies=registry.dependencies
            )
            rebound._index = index  # noqa: SLF001
            variants[f"replaced private index {position}"] = rebound
        rebound = runtime_v2.DependencyRegistry(
            dependencies=registry.dependencies
        )
        object.__setattr__(
            rebound,
            "dependencies",
            variants["validated changed context"].dependencies,
        )
        variants["rebound dependency field"] = rebound
        equal_tuple = runtime_v2.DependencyRegistry(
            dependencies=registry.dependencies
        )
        object.__setattr__(
            equal_tuple, "dependencies", (*equal_tuple.dependencies,)
        )
        variants["equal but replaced dependency tuple"] = equal_tuple

        with tempfile.TemporaryDirectory() as directory:
            writer = runtime_v2.ArtifactRuntime(
                identity(), recovery_dir=directory, artifact_dir=directory
            )
            saved = writer.execute_batch(
                invocation, registry, self.callbacks(results, [])
            ).artifacts[0]
            store = runtime_v2.ArtifactStore(directory)
            for name, current in variants.items():
                with self.subTest(name=name):
                    validate = Mock(return_value=None)
                    self.assertFalse(current.contains(invocation.dependencies))
                    self.assertIsNone(
                        store.lookup(invocation.lookups[0], current, validate)
                    )
                    reader = runtime_v2.ArtifactRuntime(
                        identity(),
                        recovery_dir=directory,
                        artifact_dir=directory,
                    )
                    self.assertEqual(
                        reader.prefetch(invocation.lookups, current, validate),
                        (),
                    )
                    self.assertEqual(reader.reused_artifacts, 0)
                    self.assertEqual(reader.usage.report()["calls"], 0)
                    self.assertEqual(reader.pending_lineages(current), ())
                    self.assertFalse(
                        store.import_typed(
                            saved.model_dump_json().encode(),
                            original_run=identity(),
                            invocation=invocation,
                            registry=current,
                            validate=validate,
                        )
                    )
                    validate.assert_not_called()

            extra = ArtifactDependency(
                kind="evidence", key=digest("unrelated"), digest=digest("new")
            )
            copied = registry.model_copy(
                update={"dependencies": (*registry.dependencies, extra)}
            )
            for current in (
                registry,
                runtime_v2.DependencyRegistry.model_validate(
                    registry.model_copy()
                ),
                runtime_v2.DependencyRegistry.model_validate(copied),
            ):
                self.assertEqual(
                    store.lookup(invocation.lookups[0], current, validate_v2),
                    saved,
                )
                reader = runtime_v2.ArtifactRuntime(
                    identity(), recovery_dir=directory, artifact_dir=directory
                )
                self.assertEqual(
                    reader.prefetch(invocation.lookups, current, validate_v2),
                    (saved,),
                )
                self.assertEqual(reader.reused_artifacts, 1)
                self.assertEqual(reader.usage.report()["calls"], 0)

    def test_registry_lookup_work_does_not_scale_with_unrelated_dependencies(
        self,
    ) -> None:
        """Count dependency reads, without a machine-dependent timing limit."""
        invocation, registry, results = fixture_v2(1)
        current = runtime_v2.DependencyRegistry(
            dependencies=(
                *registry.dependencies,
                *(
                    ArtifactDependency(
                        kind="context",
                        key=digest(["unrelated", i]),
                        digest=digest(["value", i]),
                    )
                    for i in range(41_340 - len(registry.dependencies))
                ),
            )
        )
        reads = 0
        get_attribute = ArtifactDependency.__getattribute__

        def counted(dependency: ArtifactDependency, name: str) -> object:
            nonlocal reads
            reads += name in {"kind", "key", "digest"}
            return get_attribute(dependency, name)

        with patch.object(ArtifactDependency, "__getattribute__", counted):
            for _ in range(1000):
                self.assertTrue(current.contains(invocation.dependencies))
        self.assertEqual(reads, 3 * len(invocation.dependencies) * 1000)

        with tempfile.TemporaryDirectory() as directory:
            writer = runtime_v2.ArtifactRuntime(
                identity(), recovery_dir=directory, artifact_dir=directory
            )
            writer.execute_batch(
                invocation, registry, self.callbacks(results, [])
            )
            store = runtime_v2.ArtifactStore(directory)
            costs = []
            for selected in (registry, current):
                reads = 0
                with patch.object(
                    ArtifactDependency, "__getattribute__", counted
                ):
                    self.assertIsNotNone(
                        store.lookup(
                            invocation.lookups[0], selected, validate_v2
                        )
                    )
                    reader = runtime_v2.ArtifactRuntime(
                        identity(),
                        recovery_dir=directory,
                        artifact_dir=directory,
                    )
                    self.assertEqual(
                        len(
                            reader.prefetch(
                                invocation.lookups, selected, validate_v2
                            )
                        ),
                        1,
                    )
                costs.append(reads)
            self.assertEqual(costs[0], costs[1])

    def test_runtime_identity_rejects_a_mismatched_provider_or_model(
        self,
    ) -> None:
        """Caller-supplied dependencies must match the runtime's actual adapter."""
        invocation, registry, results = fixture_v2(1)
        for changed_identity in (
            identity(model="other"),
            identity(provider_destination="other"),
        ):
            with tempfile.TemporaryDirectory() as directory:
                runtime = runtime_v2.ArtifactRuntime(
                    changed_identity,
                    recovery_dir=directory,
                    artifact_dir=directory,
                )
                with self.assertRaisesRegex(
                    runtime_v2.ArtifactRuntimeError, "dependencies_changed"
                ):
                    runtime.execute_batch(
                        invocation, registry, self.callbacks(results, [])
                    )
                with self.assertRaisesRegex(
                    runtime_v2.ArtifactRuntimeError, "dependencies_changed"
                ):
                    runtime.prefetch(invocation.lookups, registry, validate_v2)
                self.assertEqual(runtime.usage.report()["calls"], 0)

    def test_changed_consumed_context_is_new_work_but_new_packing_is_not(
        self,
    ) -> None:
        """An actual PR-context change can create a new semantic lineage."""
        invocation, registry, results = fixture_v2(1)
        with tempfile.TemporaryDirectory() as directory:
            runtime = runtime_v2.ArtifactRuntime(
                identity(), recovery_dir=directory, artifact_dir=directory
            )
            runtime.execute_batch(
                invocation, registry, self.callbacks(results, [])
            )
            deps = tuple(
                changed_record(d, digest=digest("new PR context"))
                if d.kind == "context"
                else d
                for d in invocation.dependencies
            )
            changed = changed_record(invocation, dependencies=deps)
            current = runtime_v2.DependencyRegistry(dependencies=deps)
            runtime.execute_batch(
                changed, current, self.callbacks(results, [])
            )
            self.assertNotEqual(changed.semantic_key, invocation.semantic_key)
            self.assertEqual(runtime.usage.report()["calls"], 2)
            self.assertEqual(
                len(list(Path(directory).glob("recovery-v2/*.json"))), 2
            )
            self.assertEqual(len(runtime.pending_lineages(current)), 1)

    def test_stage_stops_new_jobs_drains_active_v2_results_and_counts_usage(
        self,
    ) -> None:
        """An auth failure stops queued sends while another valid result commits."""
        template, registry, results = fixture_v2(3)
        invocations = []
        for lookup in template.lookups:
            deps = tuple(
                d
                for d in template.dependencies
                if d.kind not in {"evidence", "peer"}
                or d.key == lookup.item_key
            )
            invocations.append(
                runtime_v2.ArtifactInvocation(
                    batch=prepared_v2((lookup.item_key,)).metadata.batch,
                    lookups=(lookup,),
                    dependencies=deps,
                )
            )
        active = threading.Event()
        failed = threading.Event()
        release = threading.Event()
        started = []
        errors = []
        with tempfile.TemporaryDirectory() as directory:
            runtime = runtime_v2.ArtifactRuntime(
                identity(),
                recovery_dir=directory,
                artifact_dir=directory,
                concurrency=2,
            )

            def send(
                prepared: runtime_v2.PreparedArtifactRequest,
            ) -> runtime_v2.ProviderReply:
                key = prepared.metadata.batch.item_keys[0]
                started.append(key)
                if key == invocations[0].batch.item_keys[0]:
                    active.set()
                    self.assertTrue(release.wait(3))
                    return wire_v2([results[key].model_dump(mode="json")])
                self.assertTrue(active.wait(3))
                failed.set()
                msg = "auth"
                raise runtime_v2.ArtifactCallError(msg)

            callbacks = runtime_v2.BatchCallbacks(
                prepare_v2, send, validate_v2
            )
            jobs = [
                runtime_v2.SemanticJob(
                    inv.semantic_key,
                    lambda inv=inv: runtime.execute_batch(
                        inv, registry, callbacks
                    ),
                )
                for inv in invocations
            ]

            def coordinate() -> None:
                try:
                    runtime.run_stage("details", jobs)
                except RuntimeStageError as error:
                    errors.append(error)

            coordinator = threading.Thread(target=coordinate)
            coordinator.start()
            try:
                self.assertTrue(failed.wait(3))
                deadline = time.monotonic() + 3
                while (
                    not runtime._stop_requested()  # noqa: SLF001 # Synchronize on cancellation.
                    and time.monotonic() < deadline
                ):
                    time.sleep(0.01)
                self.assertTrue(runtime._stop_requested())  # noqa: SLF001
                self.assertTrue(coordinator.is_alive())
            finally:
                release.set()
                coordinator.join(4)
            self.assertFalse(coordinator.is_alive())
            self.assertEqual(len(errors), 1)
            self.assertEqual(
                errors[0].task_digest, invocations[1].semantic_key
            )
            self.assertCountEqual(
                started, [inv.batch.item_keys[0] for inv in invocations[:2]]
            )
            self.assertEqual(runtime.usage.report()["calls"], 2)
            self.assertEqual(
                len(
                    runtime.prefetch(
                        invocations[0].lookups, registry, validate_v2
                    )
                ),
                1,
            )

    def test_deleted_recovery_state_with_original_claim_fails_closed(
        self,
    ) -> None:
        """A deleted committed state cannot silently recreate zero attempts."""
        invocation, registry, results = fixture_v2(1)
        with tempfile.TemporaryDirectory() as directory:
            runtime = runtime_v2.ArtifactRuntime(
                identity(), recovery_dir=directory
            )
            state = runtime.recovery.ensure(
                invocation, runtime_v2.RetryPolicy(), registry
            )
            runtime.recovery.reserve(
                state.key,
                state.key,
                "initial",
                prepared_v2(invocation.batch.item_keys).metadata,
            )
            (runtime.recovery.directory / f"{state.key}.json").unlink()
            with self.assertRaisesRegex(
                runtime_v2.ArtifactRuntimeError, "state_corrupt"
            ) as caught:
                runtime.execute_batch(
                    invocation, registry, self.callbacks(results, [])
                )
            self.assertEqual(
                caught.exception.diagnostic.task_digest,
                invocation.semantic_key,
            )
            self.assertIsNone(caught.exception.diagnostic.attempt)
            self.assertEqual(runtime.usage.report()["calls"], 0)

    def test_interrupted_item_validation_keeps_already_committed_peer(
        self,
    ) -> None:
        """Checkpoint each validated item before the next validator can interrupt."""
        invocation, registry, results = fixture_v2()
        first, second = invocation.batch.item_keys

        def validate(
            lookup: runtime_v2.ArtifactLookup, result: DetailResult
        ) -> None:
            if lookup.item_key == second:
                raise KeyboardInterrupt
            validate_v2(lookup, result)

        with tempfile.TemporaryDirectory() as directory:
            runtime = runtime_v2.ArtifactRuntime(
                identity(), recovery_dir=directory, artifact_dir=directory
            )
            callbacks = self.callbacks(results, [])
            with self.assertRaises(KeyboardInterrupt):
                runtime.execute_batch(
                    invocation,
                    registry,
                    runtime_v2.BatchCallbacks(
                        prepare_v2, callbacks.send, validate
                    ),
                )
            self.assertEqual(
                runtime.prefetch(invocation.lookups, registry, validate_v2)[
                    0
                ].lookup.item_key,
                first,
            )
            resumed = runtime_v2.ArtifactRuntime(
                identity(), recovery_dir=directory, artifact_dir=directory
            )
            sends = []
            resumed.execute_batch(
                invocation, registry, self.callbacks(results, sends)
            )
            self.assertEqual(sends, [(second,)])
            self.assertEqual(
                resumed.pending_lineages(registry)[0].consumed_attempts, 2
            )

    def test_duplicate_json_envelope_fields_are_not_salvaged(self) -> None:
        """Reject raw JSON with ambiguous duplicate envelope property names."""
        invocation, registry, results = fixture_v2(1)
        response = (
            '{"version":"diffstory.batch-result.v2","items":[],"items":'
            + json.dumps([r.model_dump(mode="json") for r in results.values()])
            + "}"
        )
        with tempfile.TemporaryDirectory() as directory:
            runtime = runtime_v2.ArtifactRuntime(
                identity(), recovery_dir=directory, artifact_dir=directory
            )
            callbacks = runtime_v2.BatchCallbacks(
                prepare_v2,
                lambda _prepared: runtime_v2.ProviderReply(response.encode()),
                validate_v2,
            )
            with self.assertRaisesRegex(
                runtime_v2.ArtifactRuntimeError, "malformed_json"
            ):
                runtime.execute_batch(invocation, registry, callbacks)
            self.assertEqual(
                runtime.prefetch(invocation.lookups, registry, validate_v2), ()
            )
            self.assertEqual(
                runtime.diagnostics[0].response_bytes, len(response.encode())
            )

    def test_repair_rechecks_new_body_and_never_sends_mismatched_digest(
        self,
    ) -> None:
        """The repair callback cannot reuse an old digest for changed bytes."""
        invocation, registry, results = fixture_v2()
        first, _second = invocation.batch.item_keys
        sends = []

        def prepare(
            inv: runtime_v2.ArtifactInvocation,
            node: runtime_v2.RecoveryNode,
            purpose: str,
        ) -> runtime_v2.PreparedArtifactRequest:
            request = prepare_v2(inv, node, purpose)
            return (
                changed_record(
                    request, body=request.body + b"unaccounted repair"
                )
                if purpose == "repair"
                else request
            )

        def send(
            prepared: runtime_v2.PreparedArtifactRequest,
        ) -> runtime_v2.ProviderReply:
            sends.append(prepared.metadata.batch.item_keys)
            return wire_v2([results[first].model_dump(mode="json")])

        with tempfile.TemporaryDirectory() as directory:
            runtime = runtime_v2.ArtifactRuntime(
                identity(), recovery_dir=directory, artifact_dir=directory
            )
            with self.assertRaisesRegex(
                runtime_v2.ArtifactRuntimeError, "request_mismatch"
            ):
                runtime.execute_batch(
                    invocation,
                    registry,
                    runtime_v2.BatchCallbacks(prepare, send, validate_v2),
                )
            self.assertEqual(len(sends), 1)
            self.assertEqual(
                runtime.pending_lineages(registry)[0].nodes[0].repair_count, 0
            )
            self.assertEqual(
                len(
                    runtime.prefetch(invocation.lookups, registry, validate_v2)
                ),
                1,
            )

    def test_typed_story_plan_uses_the_same_artifact_store_without_dict_escape(
        self,
    ) -> None:
        """The accepted StoryPlan domain union works through generic callbacks."""
        template, _, _ = fixture_v2(1)
        key = template.batch.item_keys[0]
        plan = runtime_v2.StoryPlan(
            id=key,
            context_digest=digest("context"),
            groups=(),
            ordered_group_keys=(),
            cycles=(),
            chapters=(),
            cards=(),
            root_card_keys=(),
            requests=(),
        )
        lookup = changed_record(template.lookups[0], result_kind="story_plan")
        batch = changed_record(template.batch, stage="planning")
        invocation = changed_record(template, batch=batch, lookups=(lookup,))
        registry = runtime_v2.DependencyRegistry(
            dependencies=invocation.dependencies
        )

        def prepare(
            inv: runtime_v2.ArtifactInvocation,
            node: runtime_v2.RecoveryNode,
            purpose: str,
        ) -> runtime_v2.PreparedArtifactRequest:
            request = prepare_v2(inv, node, purpose)
            return changed_record(
                request,
                metadata=changed_record(
                    request.metadata,
                    batch=changed_record(
                        request.metadata.batch, stage="planning"
                    ),
                ),
            )

        def validate(_lookup: object, value: runtime_v2.StoryPlan) -> None:
            self.assertEqual(value, plan)

        with tempfile.TemporaryDirectory() as directory:
            runtime = runtime_v2.ArtifactRuntime(
                identity(), recovery_dir=directory, artifact_dir=directory
            )
            outcome = runtime.execute_batch(
                invocation,
                registry,
                runtime_v2.BatchCallbacks(
                    prepare,
                    lambda _prepared: wire_v2([plan.model_dump(mode="json")]),
                    validate,
                ),
            )
            self.assertEqual(outcome.artifacts[0].record.result, plan)
            with self.assertRaises(runtime_v2.ArtifactRuntimeError):
                outcome.detail_batch()
            self.assertEqual(
                len(runtime.prefetch((lookup,), registry, validate)), 1
            )

    def test_plain_callable_stage_preserves_available_semantic_diagnostic(
        self,
    ) -> None:
        """Legacy callable jobs also retain a v2 invocation's semantic identity."""
        invocation, registry, _results = fixture_v2(1)

        def send(_prepared: object) -> runtime_v2.ProviderReply:
            msg = "schema"
            raise RuntimeCallError(
                msg,
                "authentication",
                {"input_tokens": 4, "output_tokens": 0},
            )

        with tempfile.TemporaryDirectory() as directory:
            runtime = runtime_v2.ArtifactRuntime(
                identity(), recovery_dir=directory
            )
            callbacks = runtime_v2.BatchCallbacks(
                prepare_v2, send, validate_v2
            )
            with self.assertRaises(RuntimeStageError) as caught:
                runtime.run_stage(
                    "details",
                    [
                        lambda: runtime.execute_batch(
                            invocation, registry, callbacks
                        )
                    ],
                )
            self.assertEqual(
                caught.exception.task_digest, invocation.semantic_key
            )
            self.assertEqual(caught.exception.stage_position, 0)
            self.assertEqual(caught.exception.category, "authentication")
            self.assertEqual(runtime.usage.report()["calls"], 1)
            self.assertEqual(runtime.usage.report()["input_tokens"], 4)

    def test_failed_usage_stays_allowlisted_numeric_and_current(self) -> None:
        """Keep reported auth tokens without trusting arbitrary exception fields."""
        invocation, registry, _results = fixture_v2(1)
        arbitrary = RuntimeError("UNTRUSTED_FAILURE_SENTINEL")
        arbitrary.usage = {"input_tokens": 19, "output_tokens": 3}
        arbitrary.category = "authentication"
        cases = (
            (
                RuntimeCallError(
                    "provider",
                    category="authentication",
                    usage={"input_tokens": 19, "output_tokens": 3},
                ),
                "auth",
                1,
                19,
                3,
                0,
            ),
            *(
                (
                    RuntimeCallError("provider", "authentication", usage),
                    "auth",
                    1,
                    0,
                    0,
                    1,
                )
                for usage in (
                    {"input_tokens": 19},
                    {"input_tokens": 19, "output_tokens": True},
                    {"input_tokens": -1, "output_tokens": 3},
                    {"input_tokens": 19, "output_tokens": "3"},
                    {
                        "input_tokens": 19,
                        "output_tokens": 3,
                        "cached_input_tokens": 20,
                    },
                )
            ),
            (arbitrary, "generic", 2, 0, 0, 2),
        )
        for position, (
            error,
            reason,
            calls,
            input_tokens,
            output_tokens,
            unreported,
        ) in enumerate(cases):
            with (
                self.subTest(case=position),
                tempfile.TemporaryDirectory() as directory,
            ):
                send = Mock(side_effect=error)
                callbacks = runtime_v2.BatchCallbacks(
                    prepare_v2, send, validate_v2
                )
                runtime = runtime_v2.ArtifactRuntime(
                    identity(), recovery_dir=directory
                )
                with self.assertRaises(
                    runtime_v2.ArtifactRuntimeError
                ) as caught:
                    runtime.execute_batch(invocation, registry, callbacks)
                self.assertEqual(caught.exception.reason, reason)
                self.assertEqual(send.call_count, calls)
                self.assertEqual(
                    runtime.usage.report(),
                    {
                        "input_tokens": input_tokens,
                        "cached_input_tokens": 0,
                        "output_tokens": output_tokens,
                        "calls": calls,
                        "unreported_calls": unreported,
                    },
                )
                serialized = "".join(
                    path.read_text()
                    for path in Path(directory).rglob("*.json")
                )
                serialized += "".join(
                    traceback.format_exception(caught.exception)
                )
                self.assertNotIn("UNTRUSTED_FAILURE_SENTINEL", serialized)
                resumed = runtime_v2.ArtifactRuntime(
                    identity(), recovery_dir=directory
                )
                with self.assertRaises(runtime_v2.ArtifactRuntimeError):
                    resumed.execute_batch(invocation, registry, callbacks)
                self.assertEqual(send.call_count, calls)
                self.assertEqual(resumed.usage.report()["calls"], 0)
                self.assertEqual(resumed.usage.report()["input_tokens"], 0)
                self.assertEqual(resumed.usage.report()["output_tokens"], 0)

    def test_declared_size_failures_split_only_failed_items_and_keep_valid_peer(
        self,
    ) -> None:
        """Typed item failure reasons retain their capacity meaning for recovery."""
        invocation, registry, results = fixture_v2(3)
        first, second, third = invocation.batch.item_keys
        sends = []

        def send(
            prepared: runtime_v2.PreparedArtifactRequest,
        ) -> runtime_v2.ProviderReply:
            keys = prepared.metadata.batch.item_keys
            sends.append(keys)
            if len(sends) == 1:
                return wire_v2(
                    [results[first].model_dump(mode="json")],
                    failures=[
                        {"item_key": key, "reason": "output_limit"}
                        for key in (second, third)
                    ],
                )
            return wire_v2(
                [results[key].model_dump(mode="json") for key in keys]
            )

        with tempfile.TemporaryDirectory() as directory:
            runtime = runtime_v2.ArtifactRuntime(
                identity(), recovery_dir=directory, artifact_dir=directory
            )
            outcome = runtime.execute_batch(
                invocation,
                registry,
                runtime_v2.BatchCallbacks(
                    prepare_v2,
                    send,
                    validate_v2,
                    split=lambda node: tuple(
                        (key,) for key in node.missing_keys
                    ),
                ),
            )
            self.assertEqual(
                sends, [(first, second, third), (second,), (third,)]
            )
            self.assertEqual(len(outcome.artifacts), 3)
            self.assertEqual(runtime.diagnostics[0].reason, "output_limit")
            self.assertEqual(runtime.diagnostics[0].valid_items, 1)
            self.assertEqual(runtime.counters["repairs"], 0)
            self.assertEqual(runtime.counters["splits"], 1)

    def test_arbitrary_body_and_exception_text_never_enter_metadata_or_errors(
        self,
    ) -> None:
        """Source bytes and arbitrary provider errors remain outside safe output."""
        invocation, registry, _results = fixture_v2(1)
        private_text = (
            "UNTRUSTED_SOURCE_BODY_ERROR_SENTINEL\ncredential-like-value"
        )
        body = json.dumps({"source": private_text}).encode()
        base = prepared_v2(invocation.batch.item_keys)
        batch = changed_record(
            base.metadata.batch,
            id=digest(["body", private_text]),
            request_digest=hashlib.sha256(body).hexdigest(),
            input_bound=len(body) + 1024,
        )
        invocation = changed_record(invocation, batch=batch)
        request = changed_record(
            base,
            body=body,
            metadata=changed_record(base.metadata, batch=batch),
        )

        def send(_prepared: object) -> runtime_v2.ProviderReply:
            raise RuntimeError(private_text)

        with tempfile.TemporaryDirectory() as directory:
            events = []
            runtime = runtime_v2.ArtifactRuntime(
                identity(), recovery_dir=directory, on_progress=events.append
            )
            callbacks = runtime_v2.BatchCallbacks(
                lambda _inv, _node, _purpose: request, send, validate_v2
            )
            with self.assertRaises(runtime_v2.ArtifactRuntimeError) as caught:
                runtime.execute_batch(invocation, registry, callbacks)
            encoded = "".join(
                path.read_text() for path in Path(directory).rglob("*.json")
            )
            encoded += "".join(event.model_dump_json() for event in events)
            encoded += "".join(traceback.format_exception(caught.exception))
            self.assertNotIn("UNTRUSTED_SOURCE_BODY_ERROR_SENTINEL", encoded)
            self.assertNotIn("credential-like-value", encoded)
            self.assertEqual(runtime.usage.report()["calls"], 2)
            self.assertEqual(caught.exception.diagnostic.reason, "generic")


def prepared_family(
    keys: tuple[str, ...], stage: str, purpose: str = "initial"
) -> runtime_v2.PreparedArtifactRequest:
    """Retain actual fixture bytes while naming the correct prose stage."""
    request = prepared_v2(keys, purpose)
    return changed_record(
        request,
        metadata=changed_record(
            request.metadata,
            batch=changed_record(request.metadata.batch, stage=stage),
        ),
    )


def invocation_for_results(
    results: tuple[NarrativeResult, ...],
    run_identity: RunIdentity | None = None,
) -> runtime_v2.ArtifactInvocation:
    """Describe every original peer using the supplied family's actual schema."""
    run = identity() if run_identity is None else run_identity
    template, _, _ = fixture_v2(1, run)
    static = runtime_v2.static_artifact_dependencies(
        run,
        configuration_digest=digest("configuration"),
        prompt_digest=digest("prose-prompt.v2"),
        schema_digest=digest(structured_output_schema(type(results[0]))),
        validator_digest=digest("prose-validator.v2"),
    )
    keys = tuple(result_key(result) for result in results)
    lookups = tuple(
        runtime_v2.ArtifactLookup(
            item_key=result_key(result),
            result_kind=result_kind(result),
            static_dependencies=static,
            validator_version="prose.v2",
        )
        for result in results
    )
    peers = tuple(
        ArtifactDependency(
            kind=kind, key=key, digest=digest(["trusted-context", key])
        )
        for key in keys
        for kind in ("evidence", "peer")
    )
    context = tuple(
        dependency
        for dependency in template.dependencies
        if dependency.kind in {"context", "order", "terminology"}
    )
    return runtime_v2.ArtifactInvocation(
        batch=prepared_family(
            keys, result_stage(result_kind(results[0]))
        ).metadata.batch,
        lookups=lookups,
        dependencies=static + peers + context,
    )


class ProseArtifactTests(unittest.TestCase):
    """All new families retain M2's independent validation and mandatory reuse checks."""

    def test_each_family_repairs_one_item_preserves_peers_and_revalidates_cache(
        self,
    ) -> None:
        """A malformed peer cannot erase accepted supported prose or its producer."""
        for kind in ("planning_card", "section", "document"):
            results = tuple(
                prose_result(
                    kind,
                    digest(["subject", i]),
                    context=digest(["context", i]),
                )
                for i in range(2)
            )
            invocation = invocation_for_results(results)
            registry = runtime_v2.DependencyRegistry(
                dependencies=invocation.dependencies
            )
            keys = invocation.batch.item_keys
            sends = []

            def prepare(
                inv: runtime_v2.ArtifactInvocation,
                node: runtime_v2.RecoveryNode,
                purpose: str,
            ) -> runtime_v2.PreparedArtifactRequest:
                return prepared_family(
                    node.missing_keys, inv.batch.stage, purpose
                )

            def send(
                request: runtime_v2.PreparedArtifactRequest,
                outputs: tuple = results,
                targets: tuple = keys,
                sent: list = sends,
            ) -> runtime_v2.ProviderReply:
                sent.append(request.metadata.batch.item_keys)
                if len(sent) == 1:
                    bad = outputs[1].model_dump(mode="json") | {
                        "unsupported_extra": "rejected"
                    }
                    return wire_v2([outputs[0].model_dump(mode="json"), bad])
                self.assertEqual(
                    request.metadata.batch.item_keys, (targets[1],)
                )
                return wire_v2([outputs[1].model_dump(mode="json")])

            validate = Mock(return_value=None)
            callbacks = runtime_v2.BatchCallbacks(prepare, send, validate)
            with (
                self.subTest(kind=kind),
                tempfile.TemporaryDirectory() as directory,
            ):
                runtime = runtime_v2.ArtifactRuntime(
                    identity(), recovery_dir=directory, artifact_dir=directory
                )
                outcome = runtime.execute_batch(
                    invocation, registry, callbacks
                )
                self.assertEqual(
                    tuple(
                        artifact.record.result
                        for artifact in outcome.artifacts
                    ),
                    results,
                )
                self.assertEqual(sends, [keys, (keys[1],)])
                self.assertEqual(validate.call_count, 2)
                self.assertEqual(
                    outcome.artifacts[1].invocation.batch.item_keys, keys
                )
                self.assertEqual(
                    outcome.artifacts[1].production_request.batch.item_keys,
                    (keys[1],),
                )
                for artifact in outcome.artifacts:
                    self.assertEqual(
                        runtime_v2.StoredArtifact.model_validate_json(
                            artifact.model_dump_json()
                        ),
                        artifact,
                    )
                with self.assertRaises(runtime_v2.ArtifactRuntimeError):
                    outcome.detail_batch()
                recheck = Mock(return_value=None)
                resumed = runtime_v2.ArtifactRuntime(
                    identity(), recovery_dir=directory, artifact_dir=directory
                )
                self.assertEqual(
                    len(
                        resumed.prefetch(invocation.lookups, registry, recheck)
                    ),
                    2,
                )
                self.assertEqual(recheck.call_count, 2)
                self.assertEqual(resumed.usage.report()["calls"], 0)
                reject = Mock(
                    side_effect=ValueError("invalid current support")
                )
                self.assertEqual(
                    resumed.prefetch(invocation.lookups, registry, reject), ()
                )
                self.assertEqual(reject.call_count, 2)
                with self.assertRaises(TypeError):
                    resumed.prefetch(invocation.lookups, registry)

    def test_prose_schema_validator_and_peer_changes_invalidate_artifacts(
        self,
    ) -> None:
        """A kind label cannot upgrade stale schemas, validators, or original peers."""
        results = (
            prose_result("section", digest("first")),
            prose_result("section", digest("second")),
        )
        invocation = invocation_for_results(results)
        registry = runtime_v2.DependencyRegistry(
            dependencies=invocation.dependencies
        )

        def prepare(
            inv: runtime_v2.ArtifactInvocation,
            node: runtime_v2.RecoveryNode,
            purpose: str,
        ) -> runtime_v2.PreparedArtifactRequest:
            return prepared_family(node.missing_keys, inv.batch.stage, purpose)

        with tempfile.TemporaryDirectory() as directory:
            runtime = runtime_v2.ArtifactRuntime(
                identity(), recovery_dir=directory, artifact_dir=directory
            )
            runtime.execute_batch(
                invocation,
                registry,
                runtime_v2.BatchCallbacks(
                    prepare,
                    lambda _request: wire_v2(
                        [result.model_dump(mode="json") for result in results]
                    ),
                    Mock(return_value=None),
                ),
            )
            for dimension in ("schema", "validator", "context", "peer"):
                dependencies = tuple(
                    changed_record(
                        dependency, digest=digest(["changed", dimension])
                    )
                    if dependency.kind == dimension
                    else dependency
                    for dependency in invocation.dependencies
                )
                current = runtime_v2.DependencyRegistry(
                    dependencies=dependencies
                )
                validate = Mock(return_value=None)
                self.assertEqual(
                    runtime.prefetch(invocation.lookups, current, validate), ()
                )
                validate.assert_not_called()
            for dimension in ("schema", "validator"):
                lookups = tuple(
                    changed_record(
                        lookup,
                        static_dependencies=tuple(
                            changed_record(
                                dependency,
                                digest=digest(["new-contract", dimension]),
                            )
                            if dependency.kind == dimension
                            else dependency
                            for dependency in lookup.static_dependencies
                        ),
                    )
                    for lookup in invocation.lookups
                )
                self.assertEqual(
                    runtime.prefetch(
                        lookups, registry, Mock(return_value=None)
                    ),
                    (),
                )
            incompatible = changed_record(
                invocation.lookups[0], result_kind="detail"
            )
            self.assertEqual(
                runtime.prefetch(
                    (incompatible,), registry, Mock(return_value=None)
                ),
                (),
            )

    def test_wrong_prose_key_family_and_document_stage_are_rejected(
        self,
    ) -> None:
        """Wrong keys and stages cannot be hidden behind generic result parsing."""
        result = prose_result("document", digest("subject"))
        invocation = invocation_for_results((result,))
        with self.assertRaises(runtime_v2.ArtifactRuntimeError):
            changed_record(
                invocation,
                batch=changed_record(invocation.batch, stage="detail"),
            )
        self.assertEqual(
            invocation.artifact_keys()[result.item_key],
            invocation.artifact_key(invocation.lookups[0]),
        )
        with self.assertRaises(runtime_v2.ArtifactRuntimeError):
            runtime_v2.BatchCallbacks(
                lambda *_args: None, lambda _request: None, None
            )

    def test_public_preparation_adapter_checks_actual_repair_body(
        self,
    ) -> None:
        """The runtime accepts the public single-node preparation without private arithmetic."""
        items = (prose_result("section", digest("first")),)
        limits = RequestPackingLimits()
        capacity = ModelCapacity(1_000_000, 100_000)
        request = prepare_request_items(
            items,
            stage="section",
            keys=result_key,
            groups=lambda item: (item.binding.subject_key,),
            serializer=lambda values, output: json.dumps(
                {
                    "items": [
                        value.model_dump(mode="json") for value in values
                    ],
                    "output": output,
                    "repair": "missing_item",
                }
            ).encode(),
            allowance=lambda _values: 300,
            capacity=capacity,
            limits=limits,
        )
        prepared = runtime_v2.bind_prepared_request(request, capacity, limits)
        prepared.metadata.check(prepared.body)
        self.assertEqual(prepared.body, request.body)
        self.assertEqual(
            prepared.metadata.batch.output_guard, capacity.max_output_tokens
        )
        with self.assertRaisesRegex(
            runtime_v2.ArtifactRuntimeError, "request_mismatch"
        ):
            prepared.metadata.check(prepared.body + b"changed")


if __name__ == "__main__":
    unittest.main()


class AvailableArtifactTests(unittest.TestCase):
    """Supplied records cannot bypass validation or reopen completed work."""

    def setUp(self) -> None:
        """Complete a real lineage without a disk cache that could mask misses."""
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.invocation, self.registry, results = fixture_v2()
        self.sends = []
        self.callbacks = ArtifactRuntimeTests().callbacks(results, self.sends)
        self.runtime = runtime_v2.ArtifactRuntime(
            identity(), recovery_dir=self.temp.name
        )
        self.outcome = self.runtime.execute_batch(
            self.invocation, self.registry, self.callbacks
        )
        self.artifact, self.peer = self.outcome.artifacts
        self.state = self.runtime.recovery.load(self.invocation.semantic_key)
        self.sends.clear()

    def assert_history_unchanged(self) -> None:
        """Compare the complete durable state and ensure no additional send."""
        self.assertEqual(self.sends, [])
        self.assertEqual(
            self.runtime.recovery.load(self.state.key), self.state
        )
        self.assertIsNone(self.runtime.artifacts)

    def execute(self, available: tuple, **kwargs: object) -> object:
        """Reuse the completed original and check its entire durable history."""
        try:
            return self.runtime.execute_batch(
                self.invocation,
                self.registry,
                kwargs.pop("callbacks", self.callbacks),
                available=available,
                **kwargs,
            )
        finally:
            self.assert_history_unchanged()

    def test_same_item_other_static_contracts_skip_full_validation(
        self,
    ) -> None:
        """Another valid config or validator cannot supply the requested item."""
        alternatives = []
        for dimension in ("configuration", "validator_version"):
            static = tuple(
                changed_record(d, digest=digest("other configuration"))
                if d.kind == dimension
                else d
                for d in self.artifact.lookup.static_dependencies
            )
            lookups = tuple(
                changed_record(
                    lookup,
                    static_dependencies=static,
                    validator_version=(
                        "other" if dimension == "validator_version" else "v2"
                    ),
                )
                for lookup in self.invocation.lookups
            )
            invocation = changed_record(
                self.invocation,
                lookups=lookups,
                dependencies=static
                + tuple(
                    d
                    for d in self.invocation.dependencies
                    if d.kind not in {s.kind for s in static}
                ),
            )
            registry = runtime_v2.DependencyRegistry(
                dependencies=invocation.dependencies
            )
            issuer = runtime_v2.ArtifactRuntime(
                identity(), recovery_dir=Path(self.temp.name) / dimension
            )
            alternatives.append(
                issuer.execute_batch(
                    invocation,
                    registry,
                    ArtifactRuntimeTests().callbacks(
                        {
                            a.lookup.item_key: a.record.result
                            for a in self.outcome.artifacts
                        },
                        [],
                    ),
                ).artifacts[0]
            )
        audit = Mock(return_value=True)
        with patch.object(
            runtime_v2.StoredArtifact,
            "model_validate",
            wraps=runtime_v2.StoredArtifact.model_validate,
        ) as validate:
            self.assertEqual(
                self.execute(
                    (*alternatives, *self.outcome.artifacts),
                    audit_candidate=audit,
                ),
                self.outcome,
            )
        self.assertEqual(validate.call_count, 2)
        self.assertEqual(audit.call_count, 2)
        for alternative in alternatives:
            with (
                self.subTest(lookup=alternative.lookup.key),
                self.assertRaisesRegex(
                    runtime_v2.ArtifactRuntimeError, "results_unavailable"
                ),
            ):
                self.execute((alternative, self.peer))

    def test_reordered_static_header_is_canonicalized_before_selection(
        self,
    ) -> None:
        """An equivalent unnormalized header stays eligible for full checks."""
        lookup = self.artifact.lookup.model_copy(
            update={
                "static_dependencies": tuple(
                    reversed(self.artifact.lookup.static_dependencies)
                )
            }
        )
        self.assertNotEqual(lookup.key, self.artifact.lookup.key)
        artifact = self.artifact.model_copy(update={"lookup": lookup})
        self.assertEqual(self.execute((artifact, self.peer)), self.outcome)

    def test_other_producer_requires_all_current_dependencies(self) -> None:
        """A matching valid lookup cannot hide stale original peer content."""
        dependencies = tuple(
            changed_record(d, digest=digest("changed peer content"))
            if d.kind in {"evidence", "peer"}
            and d.key == self.peer.lookup.item_key
            else d
            for d in self.invocation.dependencies
        )
        invocation = changed_record(self.invocation, dependencies=dependencies)
        registry = runtime_v2.DependencyRegistry(dependencies=dependencies)
        issuer = runtime_v2.ArtifactRuntime(
            identity(), recovery_dir=Path(self.temp.name) / "changed-peer"
        )
        artifact = issuer.execute_batch(
            invocation,
            registry,
            ArtifactRuntimeTests().callbacks(
                {
                    a.lookup.item_key: a.record.result
                    for a in self.outcome.artifacts
                },
                [],
            ),
        ).artifacts[0]
        self.assertEqual(artifact.lookup, self.artifact.lookup)
        audit = Mock(return_value=True)
        with self.assertRaisesRegex(
            runtime_v2.ArtifactRuntimeError, "results_unavailable"
        ):
            self.execute((artifact, self.peer), audit_candidate=audit)
        self.assertEqual(audit.call_args_list, [call(self.peer)])

    def test_lookup_is_checked_again_after_full_validation(self) -> None:
        """Selection does not authenticate fields replaced before validation."""
        full, registry, results = fixture_v2(3)
        issuer = runtime_v2.ArtifactRuntime(
            identity(), recovery_dir=Path(self.temp.name) / "other-producer"
        )
        unrelated = issuer.execute_batch(
            full, registry, ArtifactRuntimeTests().callbacks(results, [])
        ).artifacts[-1]
        self.registry = registry
        artifact = self.artifact.model_copy()
        validate = runtime_v2.StoredArtifact.model_validate

        def replace_before_validation(candidate: object) -> object:
            if candidate is artifact:
                for name in runtime_v2.StoredArtifact.model_fields:
                    object.__setattr__(
                        candidate, name, getattr(unrelated, name)
                    )
            return validate(candidate)

        audit = Mock(return_value=True)
        with (
            patch.object(
                runtime_v2.StoredArtifact,
                "model_validate",
                side_effect=replace_before_validation,
            ) as validated,
            self.assertRaisesRegex(
                runtime_v2.ArtifactRuntimeError, "results_unavailable"
            ),
        ):
            self.execute((artifact, self.peer), audit_candidate=audit)
        self.assertEqual(validated.call_count, 2)
        self.assertEqual(audit.call_args_list, [call(self.peer)])

    def test_matching_corruption_and_duplicate_orders_fail_closed(
        self,
    ) -> None:
        """A valid earlier duplicate cannot hide damaged payload or provenance."""
        artifact = self.artifact
        corruptions = {
            "invalid result": artifact.model_copy(
                update={
                    "record": artifact.record.model_copy(
                        update={"result": "bad"}
                    )
                }
            ),
            "wrong result": artifact.model_copy(
                update={
                    "record": artifact.record.model_copy(
                        update={"result": self.peer.record.result}
                    )
                }
            ),
            "missing peer": artifact.model_copy(
                update={
                    "invocation": artifact.invocation.model_copy(
                        update={
                            "dependencies": tuple(
                                d
                                for d in artifact.invocation.dependencies
                                if not (
                                    d.kind == "peer"
                                    and d.key == self.peer.lookup.item_key
                                )
                            )
                        }
                    )
                }
            ),
            "inconsistent dependency": artifact.model_copy(
                update={
                    "record": artifact.record.model_copy(
                        update={
                            "dependencies": tuple(
                                d.model_copy(
                                    update={"digest": digest("changed")}
                                )
                                if d.kind == "context"
                                else d
                                for d in artifact.record.dependencies
                            )
                        }
                    )
                }
            ),
            "original request": artifact.model_copy(
                update={
                    "record": artifact.record.model_copy(
                        update={
                            "original_request_digest": digest("wrong request")
                        }
                    )
                }
            ),
            "producer request": artifact.model_copy(
                update={
                    "production_request": artifact.production_request.model_copy(
                        update={
                            "batch": artifact.production_request.batch.model_copy(
                                update={"item_keys": (digest("other item"),)}
                            )
                        }
                    )
                }
            ),
        }
        for name, invalid in corruptions.items():
            for available in (
                (invalid, self.peer),
                (invalid, *self.outcome.artifacts),
                (*self.outcome.artifacts, invalid),
            ):
                with (
                    self.subTest(
                        name=name, invalid_last=available[-1] is invalid
                    ),
                    self.assertRaises(
                        (ValidationError, runtime_v2.ArtifactRuntimeError)
                    ),
                ):
                    self.execute(available)

    def test_current_result_and_citation_rejection_cannot_regenerate(
        self,
    ) -> None:
        """Current validation still runs after selection and complete structure."""
        for rejected in (False, runtime_v2.ArtifactRuntimeError("citation")):
            validate = Mock()
            if isinstance(rejected, Exception):
                validate.side_effect = rejected
            else:
                validate.return_value = rejected
            callbacks = replace(self.callbacks, validate=validate)
            with self.assertRaisesRegex(
                runtime_v2.ArtifactRuntimeError, "results_unavailable"
            ):
                self.execute(self.outcome.artifacts, callbacks=callbacks)
            self.assertEqual(validate.call_count, 2)

    def test_every_matching_duplicate_reaches_candidate_audit(self) -> None:
        """Duplicate selection never short circuits later consumer checks."""
        audit = Mock(return_value=True)
        available = self.outcome.artifacts * 2
        self.assertEqual(
            self.execute(available, audit_candidate=audit), self.outcome
        )
        self.assertEqual(audit.call_count, 4)
        for rejected in (False, None, 1, "true", ValueError("audit failed")):
            for position in (0, 2):
                values = [True] * len(available)
                values[position] = rejected
                audit = Mock(side_effect=values)
                with self.subTest(rejected=repr(rejected), position=position):
                    if rejected is False:
                        self.assertEqual(
                            self.execute(available, audit_candidate=audit),
                            self.outcome,
                        )
                        self.assertEqual(audit.call_count, 4)
                    else:
                        error = (
                            ValueError
                            if isinstance(rejected, Exception)
                            else runtime_v2.ArtifactRuntimeError
                        )
                        reason = (
                            "audit failed"
                            if error is ValueError
                            else "request_mismatch"
                        )
                        with self.assertRaisesRegex(error, reason):
                            self.execute(available, audit_candidate=audit)
                        self.assertEqual(audit.call_count, position + 1)
        with self.assertRaisesRegex(
            runtime_v2.ArtifactRuntimeError, "results_unavailable"
        ):
            self.execute(
                self.outcome.artifacts, audit_candidate=lambda _a: False
            )

    def test_header_changes_after_selection_are_read_again(self) -> None:
        """Earlier success cannot authenticate a copied or mutated header."""
        for mutation in (
            "item",
            "config",
            "version",
            "dict",
            "list",
            "missing",
        ):
            with self.subTest(mutation=mutation):
                lookup = self.artifact.lookup.model_copy()
                artifact = self.artifact.model_copy(update={"lookup": lookup})
                self.assertEqual(
                    self.execute((artifact, self.peer)), self.outcome
                )
                error = runtime_v2.ArtifactRuntimeError
                reason = "results_unavailable"
                if mutation == "item":
                    object.__setattr__(lookup, "item_key", digest("unrelated"))
                elif mutation == "config":
                    object.__setattr__(
                        lookup,
                        "static_dependencies",
                        tuple(
                            changed_record(d, digest=digest("other"))
                            if d.kind == "configuration"
                            else d
                            for d in lookup.static_dependencies
                        ),
                    )
                elif mutation == "version":
                    object.__setattr__(
                        artifact,
                        "lookup",
                        changed_record(lookup, validator_version="other"),
                    )
                else:
                    error, reason = ValidationError, ""
                    if mutation == "dict":
                        object.__setattr__(
                            artifact, "lookup", {"item_key": lookup.item_key}
                        )
                    elif mutation == "list":
                        object.__setattr__(
                            lookup,
                            "static_dependencies",
                            list(lookup.static_dependencies),
                        )
                    else:
                        object.__delattr__(lookup, "item_key")
                with self.assertRaisesRegex(error, reason):
                    self.execute((artifact, self.peer))

    def test_ambiguous_records_retain_full_validation(self) -> None:
        """Dicts, lists and incomplete construction cannot bypass structure."""

        class ExtendedLookup(runtime_v2.ArtifactLookup):
            """An unexpected header type must retain full record validation."""

        artifact = self.artifact
        self.assertEqual(
            self.execute((artifact.model_dump(), self.peer)), self.outcome
        )
        self.assertEqual(
            self.execute(
                (
                    runtime_v2.StoredArtifact.model_construct(
                        **artifact.model_dump()
                    ),
                    self.peer,
                )
            ),
            self.outcome,
        )
        invalid = (
            (
                "incomplete construction",
                runtime_v2.StoredArtifact.model_construct(
                    lookup=artifact.lookup
                ),
                1,
            ),
            (
                "mutable item key",
                artifact.model_copy(
                    update={
                        "lookup": artifact.lookup.model_copy(
                            update={"item_key": [artifact.lookup.item_key]}
                        )
                    }
                ),
                0,
            ),
            (
                "malformed matching dict",
                {**artifact.model_dump(), "record": {}},
                1,
            ),
            ("list record", list(artifact), 1),
            (
                "malformed unrelated dict",
                {
                    **artifact.model_dump(),
                    "lookup": {
                        **artifact.lookup.model_dump(),
                        "item_key": digest("unrelated"),
                    },
                    "record": {},
                },
                1,
            ),
            (
                "unexpected header type",
                artifact.model_copy(
                    update={
                        "lookup": ExtendedLookup.model_validate(
                            {
                                **artifact.lookup.model_dump(),
                                "item_key": digest("unrelated"),
                            }
                        ),
                        "record": {},
                    }
                ),
                1,
            ),
        )
        for name, candidate, validations in invalid:
            with (
                self.subTest(name=name),
                patch.object(
                    runtime_v2.StoredArtifact,
                    "model_validate",
                    wraps=runtime_v2.StoredArtifact.model_validate,
                ) as validate,
                self.assertRaises(
                    (ValidationError, runtime_v2.ArtifactRuntimeError)
                ),
            ):
                self.execute((candidate, self.peer))
            # Malformed typed headers can reject before the full record call.
            self.assertEqual(validate.call_count, validations)


class CandidateAuditTests(unittest.TestCase):
    """The optional consumer audit controls selection without hiding failures."""

    def test_candidate_audit_requires_bool_and_propagates_outside_cache_miss(
        self,
    ) -> None:
        """Fetched, supplied and recovery-internal records share the same contract."""
        invocation, registry, results = fixture_v2(1)
        with tempfile.TemporaryDirectory() as directory:
            sends = []
            runtime = runtime_v2.ArtifactRuntime(
                identity(), recovery_dir=directory, artifact_dir=directory
            )
            callbacks = ArtifactRuntimeTests().callbacks(results, sends)
            outcome = runtime.execute_batch(invocation, registry, callbacks)
            artifact = outcome.artifacts[0]
            for value in (None, 1, "true", (), object()):

                def invalid(
                    _artifact: runtime_v2.StoredArtifact, value: object = value
                ) -> object:
                    return value

                with self.subTest(value_type=type(value).__name__):
                    with self.assertRaisesRegex(
                        runtime_v2.ArtifactRuntimeError, "request_mismatch"
                    ):
                        runtime.artifacts.lookup(
                            artifact.lookup,
                            registry,
                            validate_v2,
                            audit_candidate=invalid,
                        )
                    for supplied in ((), (artifact,)):
                        with self.assertRaisesRegex(
                            runtime_v2.ArtifactRuntimeError, "request_mismatch"
                        ):
                            runtime.execute_batch(
                                invocation,
                                registry,
                                callbacks,
                                available=supplied,
                                audit_candidate=invalid,
                            )

            def damaged(_artifact: runtime_v2.StoredArtifact) -> bool:
                msg = "state_corrupt"
                raise runtime_v2.ArtifactRuntimeError(msg)

            with self.assertRaisesRegex(
                runtime_v2.ArtifactRuntimeError, "state_corrupt"
            ):
                runtime.prefetch(
                    invocation.lookups,
                    registry,
                    validate_v2,
                    audit_candidate=damaged,
                )
            with self.assertRaisesRegex(
                runtime_v2.ArtifactRuntimeError, "results_unavailable"
            ):
                runtime.execute_batch(
                    invocation,
                    registry,
                    callbacks,
                    available=(artifact,),
                    audit_candidate=lambda _a: False,
                )
            self.assertEqual(sends, [invocation.batch.item_keys])
            self.assertEqual(
                runtime.execute_batch(
                    invocation,
                    registry,
                    callbacks,
                    available=(artifact,),
                    audit_candidate=lambda _a: True,
                ),
                outcome,
            )
            self.assertEqual(
                runtime.execute_batch(invocation, registry, callbacks), outcome
            )
