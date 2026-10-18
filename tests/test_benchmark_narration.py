"""Check local benchmark isolation, exact codec, full retention and resumability."""

from __future__ import annotations

import argparse
import copy
import io
import json
import subprocess
import sys
import tempfile
import unittest
import weakref
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
from contextlib import redirect_stdout
from pathlib import Path
from threading import Barrier
from threading import local
from types import SimpleNamespace
from unittest.mock import patch

from test_narrative_storage import artifacts_v2
from test_narrative_v2 import CASE_AFTER
from test_narrative_v2 import CASE_BEFORE
from test_narrative_v2 import CASE_EFFECT
from test_narrative_v2 import STRIP_AFTER
from test_narrative_v2 import STRIP_BEFORE
from test_narrative_v2 import STRIP_EFFECT
from test_narrative_v2 import FixtureProvider
from test_narrative_v2 import golden_report

from diffstory import analysis as analysis_module
from diffstory import narrative_runtime
from diffstory import render as render_module
from diffstory.analysis import apply_annotations
from diffstory.narrative import CodexCLIProvider
from diffstory.narrative import Narrator
from diffstory.narrative_manifest import expanded_generation
from diffstory.narrative_plan import canonical_bytes
from diffstory.narrative_runtime import ArtifactCallError
from diffstory.narrative_session import ReportSession
from diffstory.narrative_v2 import Transport
from diffstory.narrative_v2 import V2Options
from diffstory.render import iter_render_validated
from diffstory.render import render
from scripts import benchmark_narration as benchmark


class BenchmarkTests(unittest.TestCase):
    """Use authored public source only, with no live provider access."""

    def setUp(self) -> None:
        """Give every check a distinct marked benchmark directory."""
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.report = self.root / "input.json"
        self.report.write_text(json.dumps(golden_report()), encoding="utf-8")
        self.state = self.root / "benchmark"

    def args(self, phase: str = "cold") -> argparse.Namespace:
        """Pin the exact authored input bytes."""
        return argparse.Namespace(
            phase=phase,
            report=self.report,
            sha256=benchmark.file_digest(self.report),
            state=self.state,
            reserve_bytes=benchmark.DEFAULT_RESERVE,
        )

    def command(self, phase: str) -> list[str]:
        """Start each measured phase in an independent Python process."""
        return [
            sys.executable,
            "-B",
            str(benchmark.ROOT / "scripts/benchmark_narration.py"),
            "--phase",
            phase,
            "--report",
            str(self.report),
            "--sha256",
            benchmark.file_digest(self.report),
            "--state",
            str(self.state),
        ]

    def mocked_synthetic_metrics(
        self,
        scenario: str,
        count: int,
        cache_state: str,
        *,
        profile_callbacks: bool,
    ) -> dict:
        """Build one complete source-free child row for controller tests.

        Preconditions:
            Scenario and cache labels are values from the synthetic CLI enums.
        Postconditions:
            Returns a complete result shape with matching public output digests.
        Essential behavior:
            Exercises parent aggregation without starting provider or runtime work.
        """

        def phase_value() -> dict:
            """Return one complete deterministic phase fixture.

            Preconditions:
                None.
            Postconditions:
                Returns finite nonnegative counts and times for one phase.
            Essential behavior:
                Supplies complete values so parent tests do not mask omissions.
            """
            return {
                "calls": 1,
                "failures": 0,
                "records": 1,
                "bytes": 8,
                "characters": 8,
                "wall_inclusive_seconds": 0.1,
                "wall_exclusive_seconds": 0.08,
                "thread_cpu_inclusive_seconds": 0.05,
                "thread_cpu_exclusive_seconds": 0.04,
            }

        def phase_map(names: tuple[str, ...]) -> dict:
            """Build observations for selected fixed phase labels.

            Preconditions:
                Each name is a benchmark phase label.
            Postconditions:
                Returns one complete phase value per input name.
            Essential behavior:
                Keeps path-phase fixtures deterministic and source-free.
            """
            return {name: phase_value() for name in names}

        request_identity = {
            "request_count": count,
            "request_identity_sha256": f"{count:064x}",
            "request_stage_counts": {"detail": count},
        }
        annotations = {
            "bytes": 10,
            "sha256": "a" * 64,
            "logical_sha256": "b" * 64,
        }
        report = {
            "bytes": 20,
            "sha256": "c" * 64,
            "logical_sha256": "d" * 64,
        }
        html = {
            "bytes": 30,
            "sha256": "e" * 64,
            "embedded_report_sha256": "d" * 64,
        }
        if scenario == "handle-paired":
            outputs = {
                "trusted_expanded": {
                    "annotations": annotations,
                    "report": report,
                    "html": {"bytes": 30, "sha256": "e" * 64},
                },
                "trusted_shared": {
                    "annotations": {"bytes": 10, "sha256": "a" * 64},
                    "report": {"bytes": 18, "sha256": "f" * 64},
                    "html": {"bytes": 28, "sha256": "1" * 64},
                },
                "independent_shared_import": {
                    "annotations": {
                        "bytes": 10,
                        "sha256": "a" * 64,
                    },
                    "report": {
                        "bytes": 18,
                        "sha256": "f" * 64,
                    },
                    "html": {
                        "bytes": 28,
                        "sha256": "1" * 64,
                    },
                    "logical_report": {"bytes": 20, "sha256": "d" * 64},
                },
                "expanded_public": {
                    "report": report,
                    "html": html,
                },
                "source_resolution": {
                    "resolver_calls": 1,
                    "distinct_current_physical_bindings": 1,
                },
            }
            gates = {
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
                "expanded_public_public_expanded_complete_proofs": 2,
                "expanded_public_render_added_complete_proofs": 1,
                "expanded_public_logical_report_matches_trusted_expanded_stream": True,
                "expanded_public_public_html_payload_matches_logical_report": True,
                "shared_import_logical_report_matches_expanded_public": True,
                "shared_import_request_identity_matches_generation": True,
                "expanded_shared_outputs_used_same_generation_handle": True,
            }
            path_phases = {
                "capture_session": phase_map(("session.capture.facade",)),
                "generate_validated": phase_map(
                    ("session.generate_validated",)
                ),
                "trusted_apply": phase_map(("session.apply",)),
                "trusted_expanded_output": phase_map(
                    (
                        "output.annotations.serialize",
                        "output.report.serialize",
                        "output.render.validated",
                    )
                ),
                "trusted_shared_output": phase_map(
                    (
                        "output.annotations.serialize",
                        "output.report.serialize",
                        "output.render.validated",
                    )
                ),
                "independent_import_proof": phase_map(
                    ("session.shared_import", "proof.indexed_production")
                ),
                "independent_import_apply_and_output": phase_map(
                    ("session.apply", "output.report.serialize")
                ),
                "independent_import_expansion": phase_map(
                    ("benchmark.compatibility_expand",)
                ),
                "expanded_public_conversion": phase_map(
                    ("benchmark.compatibility_expand",)
                ),
                "expanded_public_validation": phase_map(
                    ("compat.public_validate",)
                ),
                "expanded_public_render": phase_map(("compat.render",)),
            }
        else:
            outputs = {
                "annotations": annotations,
                "report": report,
                "html": html,
                "source_resolution": {
                    "resolver_calls": 1,
                    "distinct_current_physical_bindings": 1,
                },
                "request_identity": request_identity,
            }
            gates = {}
            path_phases = {
                "compatibility_generate": phase_map(("compat.generate",)),
                "compatibility_generation_serialization": phase_map(
                    ("compat.serialize",)
                ),
                "compatibility_apply": phase_map(("compat.apply",)),
                "compatibility_apply_serialization": phase_map(
                    ("compat.serialize",)
                ),
                "compatibility_render": phase_map(("compat.render",)),
            }
        return {
            "status": "PASS",
            "scenario": scenario,
            "count": count,
            "cache_state": cache_state,
            "profiled_callbacks": profile_callbacks,
            "fixture_sha256": f"{count:064x}",
            "revision_binding_sha256": "b" * 64,
            "request_configuration_sha256": "c" * 64,
            "request_identity": request_identity,
            "provider_calls": 1 if cache_state == "cold" else 0,
            "elapsed_seconds": 1.0 + count / 10,
            "process_cpu_seconds": 0.5,
            "peak_rss_bytes": 1024 + count,
            "peak_rss_scope": "fresh_process_high_water_mark",
            "process_resource_scope": "whole_selected_scenario_child_process",
            "outputs": outputs,
            "deterministic_gates": gates,
            "provider_call_gate": True,
            "phase_measurements": {
                name: phase_value() for name in benchmark.PHASE_LABELS
            },
            "path_phase_measurements": path_phases,
            "pydantic_python_callbacks": (
                {"field.fixture": 1, "model.fixture": 1}
                if profile_callbacks
                else {}
            ),
        }

    def test_nested_measurements_use_deterministic_self_time(self) -> None:
        """Subtract nested wall and thread CPU time without changing totals."""

        class Clock:
            wall = 0.0
            cpu = 0.0

            def wall_time(self) -> float:
                return self.wall

            def thread_cpu_time(self) -> float:
                return self.cpu

            def advance(self, *, wall: float, cpu: float) -> None:
                self.wall += wall
                self.cpu += cpu

        clock = Clock()
        measurements = benchmark.PhaseMeasurements(
            wall_clock=clock.wall_time,
            thread_cpu_clock=clock.thread_cpu_time,
        )
        with measurements.measure("proof.production"):
            clock.advance(wall=5.0, cpu=3.0)
            with measurements.measure("proof.support.forest"):
                clock.advance(wall=2.0, cpu=1.0)
            clock.advance(wall=1.0, cpu=1.0)

        outer = measurements.snapshot()["proof.production"]
        child = measurements.snapshot()["proof.support.forest"]
        self.assertEqual(outer["wall_inclusive_seconds"], 8.0)
        self.assertEqual(outer["wall_exclusive_seconds"], 6.0)
        self.assertEqual(outer["thread_cpu_inclusive_seconds"], 5.0)
        self.assertEqual(outer["thread_cpu_exclusive_seconds"], 4.0)
        self.assertEqual(child["wall_inclusive_seconds"], 2.0)
        self.assertEqual(child["thread_cpu_inclusive_seconds"], 1.0)

    def test_overlapping_phase_cpu_uses_each_threads_clock(self) -> None:
        """Concurrent waits do not multiply one process CPU interval per phase."""

        class ThreadClock:
            def __init__(self) -> None:
                self.values = local()

            def wall_time(self) -> float:
                return getattr(self.values, "wall", 0.0)

            def thread_cpu_time(self) -> float:
                return getattr(self.values, "cpu", 0.0)

            def advance(self, amount: float) -> None:
                self.values.wall = self.wall_time() + amount
                self.values.cpu = self.thread_cpu_time() + amount

        clock = ThreadClock()
        barrier = Barrier(2)
        measurements = benchmark.PhaseMeasurements(
            wall_clock=clock.wall_time,
            thread_cpu_clock=clock.thread_cpu_time,
        )

        def run_phase() -> None:
            with measurements.measure("cache.lookup"):
                clock.advance(2.0)
                barrier.wait(timeout=5)
                clock.advance(3.0)

        with ThreadPoolExecutor(max_workers=2) as executor:
            futures = [executor.submit(run_phase) for _ in range(2)]
            for future in futures:
                future.result(timeout=5)

        phase = measurements.snapshot()["cache.lookup"]
        self.assertEqual(phase["calls"], 2)
        self.assertEqual(phase["thread_cpu_inclusive_seconds"], 10.0)
        self.assertEqual(phase["thread_cpu_exclusive_seconds"], 10.0)

    def test_wrappers_restore_hooks_and_forward_original_exceptions(
        self,
    ) -> None:
        """ExitStack restores patched hooks even when an observed call fails."""
        original = narrative_runtime.ArtifactStore._expand  # noqa: SLF001 # Verify restoration of this installed hook.
        failure = ValueError("SYNTHETIC_PRIVATE_EXCEPTION_SENTINEL")

        def fail(_self: object, _item: object) -> object:
            raise failure

        measurements = benchmark.PhaseMeasurements()
        with (
            patch.object(narrative_runtime.ArtifactStore, "_expand", fail),
            ExitStack() as stack,
            self.assertRaises(ValueError) as caught,
        ):
            benchmark.install_measurement_hooks(stack, measurements)
            narrative_runtime.ArtifactStore._expand(  # noqa: SLF001 # Exercise the installed hook.
                SimpleNamespace(), SimpleNamespace()
            )
        self.assertIs(caught.exception, failure)
        self.assertIs(narrative_runtime.ArtifactStore._expand, original)  # noqa: SLF001 # Check restoration after failure.
        phase = measurements.snapshot()["cache.expand"]
        self.assertEqual(phase["calls"], 1)
        self.assertEqual(phase["failures"], 1)
        self.assertNotIn(
            "SYNTHETIC_PRIVATE_EXCEPTION_SENTINEL",
            json.dumps(measurements.snapshot()),
        )

    def test_initial_read_buffer_is_released_before_preflight(self) -> None:
        """The parsed report proceeds without retaining its raw input bytes."""

        class TrackedPayload(bytearray):
            pass

        class SyntheticReport:
            payload_ref: weakref.ReferenceType[TrackedPayload] | None = None

            def read_bytes(self) -> TrackedPayload:
                payload = TrackedPayload(
                    json.dumps(golden_report()).encode("utf-8")
                )
                self.payload_ref = weakref.ref(payload)
                return payload

        class StopBeforePipelineError(Exception):
            pass

        report_path = SyntheticReport()
        self.state.mkdir()
        args = argparse.Namespace(
            phase="resume",
            report=report_path,
            sha256="synthetic-only",
            state=self.state,
            reserve_bytes=benchmark.DEFAULT_RESERVE,
        )

        def preview_and_stop(_narrator: Narrator, _report: dict) -> None:
            payload_ref = report_path.payload_ref
            self.assertIsNotNone(payload_ref)
            self.assertIsNone(payload_ref())
            raise StopBeforePipelineError

        with (
            patch.object(Narrator, "preview", preview_and_stop),
            self.assertRaises(StopBeforePipelineError),
        ):
            benchmark.execute(
                args,
                {},
                {},
                benchmark.PhaseMeasurements(),
            )

    def test_binding_summary_redacts_relative_candidate_paths(self) -> None:
        """Export a fixed-key digest summary while retaining the full private binding."""
        binding = {
            "version": benchmark.VERSION,
            "source_bytes": 17,
            "candidate_hashes": {
                "private/RELATIVE_SOURCE_PATH_SENTINEL.py": "a" * 64,
            },
        }
        summary = benchmark.binding_summary(binding)
        self.assertEqual(
            set(summary),
            {
                "version",
                "binding_sha256",
                "candidate_file_count",
                "source_bytes",
            },
        )
        self.assertEqual(summary["candidate_file_count"], 1)
        exported = json.dumps(summary)
        self.assertNotIn("candidate_hashes", exported)
        self.assertNotIn("RELATIVE_SOURCE_PATH_SENTINEL", exported)
        self.assertNotIn("private/", exported)

    def test_measurement_wrappers_preserve_synthetic_public_outputs(
        self,
    ) -> None:
        """Instrumentation leaves candidate, applied report and reader bytes exact."""

        def run(
            prefix: str, *, instrument: bool
        ) -> tuple[dict, dict, str, dict]:
            report = golden_report()
            narrator = Narrator(
                FixtureProvider(),
                architecture="v2",
                recovery_dir=self.root / f"{prefix}-proof",
                checkpoint_dir=self.root / f"{prefix}-cache",
                concurrency=1,
            )
            measurements = benchmark.PhaseMeasurements()
            with ExitStack() as stack:
                benchmark.offline_guard(stack)
                if instrument:
                    benchmark.install_measurement_hooks(stack, measurements)
                candidate = narrator.generate(report)
                applied = analysis_module.apply_annotations(report, candidate)
                html = render_module.render(applied)
            return candidate, applied, html, measurements.snapshot()

        plain = run("plain", instrument=False)
        measured = run("measured", instrument=True)
        self.assertEqual(
            canonical_bytes(plain[0]), canonical_bytes(measured[0])
        )
        self.assertEqual(
            canonical_bytes(plain[1]), canonical_bytes(measured[1])
        )
        self.assertEqual(plain[2], measured[2])
        self.assertGreater(measured[3]["apply.public"]["calls"], 0)
        self.assertGreater(measured[3]["render.html_assembly"]["calls"], 0)

    def test_codec_capacity_guard_and_distinct_identity(self) -> None:
        """Keep built-in bytes and maximum output reservation without real cache identity."""
        live = CodexCLIProvider(model=benchmark.MODEL)
        local = benchmark.LocalCodexFixture()
        with ExitStack() as stack:
            benchmark.offline_guard(stack)
            local.require_credentials()
            for output in (1000, 10000):
                self.assertEqual(
                    local.prepare("system", {"x": "é"}, "schema", {}, output),
                    live.prepare("system", {"x": "é"}, "schema", {}, output),
                )
            transport = Transport(local, V2Options())
            self.assertFalse(transport.enforceable)
            self.assertEqual(
                transport.capacity.output_guard(1000), live.max_output_tokens
            )
            self.assertEqual(local.context_tokens, live.context_tokens)
            self.assertEqual(local.request_prefix, live.request_prefix)
            self.assertNotEqual(local.destination, live.destination)
            self.assertNotEqual(local.model, live.model)
            self.assertIn("local", transport.descriptor.destination)
            with self.assertRaises(benchmark.BenchmarkError):
                local.complete(b"{}", 1)

    def assert_phase_measurements(self, metrics: dict) -> None:
        """Require source-free positive coverage and well-formed nested totals."""
        expected_calls = (
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
            "proof.production",
            "compat.generate",
            "session.finish_production",
            "proof.stage.audit",
            "proof.producer_request_audit",
            "proof.request_plan_audit",
            "proof.support.prose",
            "proof.support.forest",
            "source.resolve",
            "apply.public",
            "persistence.serialize",
            "cli.reload.read_decode_validate",
            "cli.reload.json_parse",
            "render.safety_validation",
            "render.json_serialize",
            "render.html_assembly",
        )
        phases = metrics["phase_measurements"]
        self.assertEqual(set(phases), set(benchmark.PHASE_LABELS))
        for name in expected_calls:
            self.assertGreater(phases[name]["calls"], 0, name)
        for phase in phases.values():
            self.assertLessEqual(
                phase["wall_exclusive_seconds"],
                phase["wall_inclusive_seconds"] + 1e-9,
            )
            self.assertLessEqual(
                phase["thread_cpu_exclusive_seconds"],
                phase["thread_cpu_inclusive_seconds"] + 1e-9,
            )
        self.assertGreater(phases["input.initial_read"]["bytes"], 0)
        self.assertGreater(phases["input.initial_json_parse"]["bytes"], 0)
        self.assertEqual(
            metrics["peak_rss_scope"], "process_run_high_water_mark"
        )
        self.assertGreaterEqual(metrics["process_cpu_seconds"], 0.0)
        self.assertIn(
            "deferred to M6",
            metrics["measurement_qualifications"]["browser"],
        )
        if metrics["phase"] == "warm":
            self.assertGreater(phases["cache.record_parse"]["calls"], 0)
            self.assertGreater(phases["cache.record_parse"]["bytes"], 0)

    def assert_source_free_metrics(self, metrics: dict) -> None:
        """Check public measurement and progress exports contain no paths."""
        self.assertTrue(metrics["coverage"]["complete"])
        self.assertTrue(metrics["public_cli"]["report_bytes_preserved"])
        self.assertEqual(
            metrics["production_bounds"]["maximum_output_guard"], 128000
        )
        self.assertEqual(
            metrics["coverage"]["original_details"],
            metrics["coverage"]["retained_details"],
        )
        text = json.dumps(metrics)
        self.assertNotIn("normal_0.py", text)
        self.assertNotIn("value.rstrip", text)
        self.assertNotIn(str(self.report), text)
        self.assertNotIn("candidate_hashes", text)
        self.assertNotIn("diffstory/narrative_storage.py", text)

    def test_fresh_process_cold_warm_full_public_roundtrip(self) -> None:
        """Revalidate cache, support and persisted report with zero warm attempts."""
        values = []
        for phase in ("cold", "warm"):
            completed = subprocess.run(
                self.command(phase),
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(
                completed.returncode, 0, completed.stdout + completed.stderr
            )
            self.assertEqual(completed.stderr, "")
            values.append(
                json.loads((self.state / f"{phase}-metrics.json").read_text())
            )
        cold, warm = values
        self.assertNotEqual(cold["pid"], warm["pid"])
        for metrics in values:
            self.assert_phase_measurements(metrics)
        self.assertTrue(warm["fresh_process_warm"])
        self.assertGreater(cold["counters"]["attempts"], 0)
        self.assertEqual(warm["counters"]["attempts"], 0)
        self.assertGreater(warm["counters"]["reused_artifacts"], 0)
        self.assertEqual(cold["ledger"], warm["ledger"])
        self.assertEqual(
            cold["coverage"]["exact_detail_digest"],
            warm["coverage"]["exact_detail_digest"],
        )
        self.assertEqual(
            cold["coverage"]["producer_history_digest"],
            warm["coverage"]["producer_history_digest"],
        )
        self.assertGreater(cold["storage_guard"]["immutable_record_writes"], 0)
        self.assertEqual(warm["storage_guard"]["immutable_record_writes"], 0)
        cache_files = [
            p for p in (self.state / "cache").rglob("*") if p.is_file()
        ]
        self.assertEqual(
            sum(v["files"] for v in warm["cache_storage"].values()),
            len(cache_files),
        )
        self.assertEqual(
            sum(v["bytes"] for v in warm["cache_storage"].values()),
            sum(p.stat().st_size for p in cache_files),
        )
        self.assertGreater(warm["cache_storage"]["invocations"]["files"], 0)
        self.assertGreater(warm["cache_storage"]["requests"]["files"], 0)
        self.assertEqual(
            warm["cache_storage"]["items"]["files"],
            cold["coverage"]["receipts"],
        )
        invocation_files = [
            p
            for p in cache_files
            if p.parent.name == "invocations"
            and json.loads(p.read_bytes())["batch"]["stage"] == "detail"
        ]
        request_files = [
            p
            for p in cache_files
            if p.parent.name == "requests"
            and json.loads(p.read_bytes())["metadata"]["batch"]["stage"]
            == "detail"
        ]
        invocation_keys = {p.stem for p in invocation_files}
        leaf_metadata = 0
        for path in cache_files:
            if path.parent.parent.name == "items":
                leaf = json.loads(path.read_bytes())
                if leaf["invocation_digest"] in invocation_keys:
                    leaf_metadata += path.stat().st_size - len(
                        canonical_bytes(leaf["result"])
                    )
        self.assertEqual(
            cold["detail_preflight"]["compact_invocation_bytes"],
            sum(p.stat().st_size for p in invocation_files),
        )
        self.assertEqual(
            cold["detail_preflight"][
                "successful_request_metadata_bytes_if_initial_batches_succeed"
            ],
            sum(p.stat().st_size for p in request_files),
        )
        self.assertEqual(
            cold["detail_preflight"]["leaf_metadata_bytes_without_results"],
            leaf_metadata,
        )
        self.assertEqual(
            cold["current_usage"]["calls"],
            cold["current_usage"]["unreported_calls"],
        )
        self.assertIsNone(cold["observed_llm_tokens"])
        for metrics in values:
            self.assert_source_free_metrics(metrics)
        progress = (self.state / "progress.json").read_text()
        self.assertNotIn("candidate_hashes", progress)
        self.assertNotIn("diffstory/narrative_storage.py", progress)
        self.assertNotIn(str(self.report), progress)

    def test_state_never_resets_and_binding_changes_fail(self) -> None:
        """A repeat cold start or changed source/configuration preserves every byte."""
        args = self.args()
        benchmark.initialize(args)
        marker = self.state / "LOCAL_FIXTURE_ONLY.json"
        original = marker.read_bytes()
        self.assertIn("candidate_hashes", json.loads(original))
        self.assertIn(
            "diffstory/narrative_storage.py",
            json.loads(original)["candidate_hashes"],
        )
        with self.assertRaisesRegex(benchmark.BenchmarkError, "state_exists"):
            benchmark.initialize(args)
        args.phase = "resume"
        self.report.write_text(self.report.read_text() + " ")
        args.sha256 = benchmark.file_digest(self.report)
        with self.assertRaisesRegex(
            benchmark.BenchmarkError, "state_binding_mismatch"
        ):
            benchmark.initialize(args)
        self.assertEqual(marker.read_bytes(), original)

    def test_storage_module_is_bound_and_old_marker_is_not_rewritten(
        self,
    ) -> None:
        """Adding shared storage cannot silently renew an earlier benchmark lineage."""
        args = self.args()
        binding = benchmark.initialize(args)
        self.assertIn(
            "diffstory/narrative_storage.py", binding["candidate_hashes"]
        )
        marker = self.state / "LOCAL_FIXTURE_ONLY.json"
        binding["candidate_hashes"].pop("diffstory/narrative_storage.py")
        marker.write_text(json.dumps(binding))
        original = marker.read_bytes()
        args.phase = "resume"
        with self.assertRaisesRegex(
            benchmark.BenchmarkError, "state_binding_mismatch"
        ):
            benchmark.initialize(args)
        self.assertEqual(marker.read_bytes(), original)

    def test_floor_intercepts_each_shared_record_and_leaf_temporary(
        self,
    ) -> None:
        """The actual store cannot publish any next dependency without its reserve."""
        for fail_at in (1, 2, 3):
            with self.subTest(fail_at=fail_at):
                self.assert_shared_floor(fail_at)

    def assert_shared_floor(self, fail_at: int) -> None:
        """Refuse one exact writer stage without retaining a partially written leaf."""
        artifact = artifacts_v2()[0][0]
        state = self.root / f"floor-{fail_at}"
        state.mkdir()
        guard = benchmark.StorageGuard(state, benchmark.DEFAULT_RESERVE)
        store = narrative_runtime.ArtifactStore(state / "cache")
        writes = []
        original = guard.immutable

        def checked(path: Path, record: object) -> None:
            writes.append((path, record))
            original(path, record)

        guard.immutable = checked
        checks = []
        original_check = guard.check

        def check(needed: int) -> None:
            checks.append(needed)
            free = guard.reserve + needed - (len(checks) == fail_at)
            usage = type("Usage", (), {"free": free})()
            with patch.object(
                benchmark.shutil, "disk_usage", return_value=usage
            ):
                original_check(needed)

        guard.check = check
        with (
            patch.object(
                narrative_runtime,
                "_immutable_cache_record",
                guard.shared_record,
            ),
            self.assertRaisesRegex(
                benchmark.BenchmarkError, "insufficient_storage"
            ),
        ):
            store.commit(artifact)
        self.assertEqual(len(writes), fail_at - 1)
        self.assertEqual(len(checks), fail_at)
        for (_, record), needed in zip(writes, checks, strict=False):
            self.assertEqual(
                needed, 2 * len(record.model_dump_json().encode())
            )
        self.assertEqual(list(store.directory.glob("*/*.json")), [])
        self.assertEqual(list(store.shared_directory.rglob(".v3-*")), [])

    def test_existing_shared_address_and_recovery_still_need_temporary_space(
        self,
    ) -> None:
        """Neither an existing address nor the separate recovery writer bypasses the floor."""
        self.state.mkdir()
        guard = benchmark.StorageGuard(self.state, benchmark.DEFAULT_RESERVE)
        artifact = artifacts_v2()[0][0]
        store = narrative_runtime.ArtifactStore(self.state / "cache")
        store.commit(artifact)
        path = next((store.shared_directory / "invocations").glob("*.json"))
        before = path.read_bytes()
        record = narrative_runtime.StoredInvocation.model_validate_json(before)
        usage = type("Usage", (), {"free": guard.reserve})()
        with patch.object(benchmark.shutil, "disk_usage", return_value=usage):
            with self.assertRaisesRegex(
                benchmark.BenchmarkError, "insufficient_storage"
            ):
                guard.shared_record(path, record)
            with self.assertRaisesRegex(
                benchmark.BenchmarkError, "insufficient_storage"
            ):
                guard.record(self.state / "recovery.json", record)
        self.assertEqual(path.read_bytes(), before)
        self.assertFalse((self.state / "recovery.json").exists())
        self.assertEqual(list(store.shared_directory.rglob(".v3-*")), [])

    def test_all_cache_files_and_both_manifest_shapes_are_measured(
        self,
    ) -> None:
        """Include orphan temporaries and unrelated cache records in file totals."""
        for relative in (
            "artifacts-v3/invocations/a.json",
            "artifacts-v3/requests/b.json",
            "artifacts-v3/items/c/d.json",
            "artifacts-v3/items/c/.v3-orphan",
            "artifacts-v2/legacy.json",
            "producer-runs-v2/run.json",
        ):
            path = self.state / "cache" / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"fixture")
        counts = benchmark.cache_storage_metrics(self.state)
        self.assertEqual(sum(v["files"] for v in counts.values()), 6)
        self.assertEqual(sum(v["bytes"] for v in counts.values()), 42)
        self.assertEqual(counts["temporary"], {"files": 1, "bytes": 7})
        candidate = Narrator(
            FixtureProvider(),
            architecture="v2",
            recovery_dir=self.root / "proof",
            concurrency=1,
        ).generate(golden_report())
        from diffstory.narrative_plan import build_evidence_ledger

        ledger = build_evidence_ledger(golden_report())
        legacy = copy.deepcopy(candidate)
        legacy["generation"] = expanded_generation(legacy["generation"])
        self.assertEqual(
            benchmark.coverage_metrics(candidate, ledger),
            benchmark.coverage_metrics(legacy, ledger),
        )

    def test_unmarked_state_and_bad_pin_are_refused(self) -> None:
        """Never use a real narration cache or an unpinned input as fixture state."""
        args = self.args("resume")
        self.state.mkdir()
        with self.assertRaisesRegex(
            benchmark.BenchmarkError, "unmarked_state"
        ):
            benchmark.initialize(args)
        args = self.args()
        args.sha256 = "0" * 64
        with self.assertRaisesRegex(benchmark.BenchmarkError, "pin_mismatch"):
            benchmark.initialize(args)

    def test_storage_floor_and_response_limit_fail_before_unsafe_writes(
        self,
    ) -> None:
        """Use explicit resource failures without weakening coverage or output caps."""
        self.state.mkdir()
        guard = benchmark.StorageGuard(self.state, benchmark.DEFAULT_RESERVE)
        usage = type("Usage", (), {"free": 1})()
        with (
            patch.object(benchmark.shutil, "disk_usage", return_value=usage),
            self.assertRaisesRegex(
                benchmark.BenchmarkError, "insufficient_storage"
            ),
        ):
            guard.json(self.state / "refused.json", {"retained": "content"})
        self.assertFalse((self.state / "refused.json").exists())
        provider = benchmark.LocalCodexFixture()
        body = provider.prepare(
            "", {"stage": "detail", "items": []}, "test", {}, 1000
        )
        with self.assertRaises(ArtifactCallError) as error:
            provider.complete_reply(body, 10, response_byte_limit=1)
        self.assertEqual(error.exception.reason, "response_limit")
        self.assertEqual(provider.totals["detail"]["attempts"], 1)
        self.assertFalse(hasattr(provider, "calls"))

    def test_errors_never_emit_private_exception_or_path(self) -> None:
        """Catch arbitrary validation text before traceback or exception formatting."""
        sentinel = "SYNTHETIC_PRIVATE_PAYLOAD_SENTINEL"
        with (
            patch.object(
                benchmark, "initialize", side_effect=ValueError(sentinel)
            ),
            redirect_stdout(io.StringIO()) as output,
        ):
            code = benchmark.main(self.command("cold")[3:])
        self.assertEqual(code, 2)
        self.assertNotIn(sentinel, output.getvalue())
        self.assertNotIn(str(self.report), output.getvalue())

    def test_failed_run_resume_keeps_spent_lineage(self) -> None:
        """An authentication-class fixture fault cannot reset its stopped lineage."""
        with (
            patch.object(
                benchmark,
                "fixture_response",
                side_effect=ArtifactCallError("authentication"),
            ),
            redirect_stdout(io.StringIO()),
        ):
            code = benchmark.main(self.command("cold")[3:])
        self.assertEqual(code, 2)
        files = list((self.state / "recovery" / "recovery-v2").glob("*.json"))
        self.assertTrue(files)
        before = {p.name: p.read_bytes() for p in files}
        with redirect_stdout(io.StringIO()):
            code = benchmark.main(self.command("resume")[3:])
        self.assertEqual(code, 2)
        self.assertEqual(before, {p.name: p.read_bytes() for p in files})
        resumed = json.loads((self.state / "resume-metrics.json").read_text())
        self.assertEqual(resumed["counters"]["attempts"], 0)

    def test_authored_golden_mechanisms_survive_public_reader(self) -> None:
        """Separate authored semantic examples from generic full-pin accounting."""
        report = golden_report()
        with ExitStack() as stack:
            benchmark.offline_guard(stack)
            narrator = Narrator(
                FixtureProvider(),
                architecture="v2",
                recovery_dir=self.root / "golden-state",
                checkpoint_dir=self.root / "golden-cache",
                concurrency=1,
            )
            candidate = narrator.generate(report)
            annotated = apply_annotations(report, candidate)
            reloaded = json.loads(json.dumps(annotated))
            html = render(reloaded)
        details = candidate["generation"]["details"]
        retained = [
            d
            for s in reloaded["narrative_v2"]["sections"]
            for d in s["details"]
        ] + reloaded["narrative_v2"]["context_details"]
        self.assertEqual(
            {d["unit_key"]: d for d in details},
            {d["unit_key"]: d for d in retained},
        )
        for text in (
            STRIP_BEFORE,
            STRIP_AFTER,
            STRIP_EFFECT,
            CASE_BEFORE,
            CASE_AFTER,
            CASE_EFFECT,
        ):
            self.assertIn(text, html)
        self.assertIn("No executed test results or caller expectations", html)
        self.assertEqual(reloaded["generation"]["provenance"], "unverified")
        tampered = copy.deepcopy(candidate)
        tampered["generation"]["details"].pop()
        with self.assertRaises(ValueError):
            apply_annotations(report, tampered)

    def test_synthetic_count_matrix_rejects_invalid_sizes_before_runs(
        self,
    ) -> None:
        """Concern: the local matrix must reject malformed or unbounded sizes.

        Input/expected matrix:
            ``"2,4,8"`` -> ``(2, 4, 8)``; empty, duplicate, undersized,
            nonnumeric, zero, and over-limit values -> ``BenchmarkError``.
        """
        matrix = (
            ("2,4,8", (2, 4, 8)),
            ("", None),
            ("2,2,8", None),
            ("2,4", None),
            ("2,x,8", None),
            ("0,4,8", None),
            ("2,4,33", None),
        )
        for value, expected in matrix:
            with self.subTest(value=value):
                if expected is None:
                    with self.assertRaises(benchmark.BenchmarkError):
                        benchmark.parse_synthetic_counts(value)
                else:
                    self.assertEqual(
                        benchmark.parse_synthetic_counts(value), expected
                    )

    def test_00_failed_controller_preserves_rows_without_incomplete_medians(
        self,
    ) -> None:
        """Concern: child validation failures must not erase rows or fake medians.

        Input/expected matrix:
            each cold child returns only BLOCKED and presentation_order_changed
            with exit 2 -> every planned coordinate remains, warm/profile work
            is skipped, failure reasons persist, and no group has a median.
        """
        output = self.root / "synthetic-failure-results"

        def fail_child(*_args: object, **_kwargs: object) -> tuple[dict, int]:
            """Model an order-gate child with no completed measurements.

            Preconditions:
                The controller requests one cold, warm, or callback child.
            Postconditions:
                Returns the fixed failure reason and exact nonzero status.
            Essential behavior:
                Omits outputs, counters, and deterministic gates intentionally.
            """
            return {
                "status": "BLOCKED",
                "reason": "presentation_order_changed",
            }, 2

        with (
            patch.object(
                benchmark, "_run_synthetic_child", side_effect=fail_child
            ),
            redirect_stdout(io.StringIO()),
        ):
            exit_code = benchmark.main(
                [
                    "--synthetic",
                    "--counts",
                    "1,2,3",
                    "--repeats",
                    "3",
                    "--output-dir",
                    str(output),
                ]
            )
        result = json.loads((output / "measurements.json").read_text())
        expected = (
            ("timing_sample_count", 36),
            ("callback_sample_count", 6),
            ("timing_attempted_child_count", 18),
            ("callback_attempted_child_count", 0),
            ("status", "BLOCKED"),
            ("parent_exit_code", 2),
        )
        self.assertEqual(exit_code, 2)
        for name, value in expected:
            with self.subTest(field=name):
                self.assertEqual(result[name], value)
        failed_cold = next(
            row
            for row in result["timing_samples"]
            if row["count"] == 1
            and row["repeat"] == 1
            and row["scenario"] == "handle-paired"
            and row["cache_state"] == "cold"
        )
        self.assertEqual(failed_cold["exit_code"], 2)
        self.assertEqual(
            failed_cold["metrics"]["reason"],
            "presentation_order_changed",
        )
        skipped_warm = next(
            row
            for row in result["timing_samples"]
            if row["count"] == 1
            and row["repeat"] == 1
            and row["scenario"] == "handle-paired"
            and row["cache_state"] == "warm"
        )
        self.assertFalse(skipped_warm["attempted"])
        self.assertIsNone(skipped_warm["exit_code"])
        self.assertEqual(
            skipped_warm["metrics"]["reason"], "cold_sample_failed"
        )
        self.assertIn(
            {
                "count": 1,
                "repeat": 1,
                "scenario": "handle-paired",
                "cache_state": "cold",
                "profile_kind": "timing",
                "attempted": True,
                "exit_code": 2,
                "status": "BLOCKED",
                "reason": "presentation_order_changed",
            },
            result["failed_samples"],
        )
        self.assertTrue(
            all(
                group["valid_sample_count"] == 0
                and "process_wall_seconds" not in group
                and "provider_calls" not in group
                for group in result["aggregates"]["groups"].values()
            )
        )
        report = (output / "result.md").read_text()
        self.assertIn("presentation_order_changed", report)
        self.assertNotIn("KeyError", report)

    def test_01_successful_controller_preserves_all_complete_gates(
        self,
    ) -> None:
        """Concern: complete mocked rows must retain every pairing and output gate.

        Input/expected matrix:
            three counts by three repeats, both scenarios and cache states, plus
            six callback rows -> PASS, complete medians, matching semantics,
            and every controller gate true in a caller-selected directory.
        """
        output = self.root / "caller-selected-synthetic-results"

        def successful_child(
            scenario: str,
            count: int,
            cache_state: str,
            _cache_dir: Path,
            *,
            profile_callbacks: bool,
        ) -> tuple[dict, int]:
            """Return complete matched rows without starting child processes.

            Preconditions:
                The controller provides one allowed scenario and cache state.
            Postconditions:
                Returns a complete mocked metric map and exact zero status.
            Essential behavior:
                Lets the public parent path run without provider or runtime work.
            """
            return (
                self.mocked_synthetic_metrics(
                    scenario,
                    count,
                    cache_state,
                    profile_callbacks=profile_callbacks,
                ),
                0,
            )

        with (
            patch.object(
                benchmark,
                "_run_synthetic_child",
                side_effect=successful_child,
            ),
            redirect_stdout(io.StringIO()),
        ):
            exit_code = benchmark.main(
                [
                    "--synthetic",
                    "--counts",
                    "1,2,3",
                    "--repeats",
                    "3",
                    "--output-dir",
                    str(output),
                ]
            )
        result = json.loads((output / "measurements.json").read_text())
        self.assertEqual(exit_code, 0)
        self.assertEqual(result["status"], "PASS")
        self.assertEqual(result["matrix_failures"], [])
        self.assertTrue(all(result["pairing_gates"].values()))
        self.assertTrue(
            all(
                group["status"] == "PASS"
                and group["valid_sample_count"] == 3
                and "process_wall_seconds" in group
                for group in result["aggregates"]["groups"].values()
            )
        )
        self.assertEqual(result["shared_fixture_analysis"], None)
        self.assertEqual(
            result["aggregates"]["groups"]["2/compat-public/cold"][
                "matched_boundary_paths"
            ]["public_compatibility"]["boundary_phases"],
            [
                "compat.generate",
                "compat.serialize",
                "compat.apply",
                "compat.serialize",
                "compat.render",
            ],
        )

    def test_handle_output_modes_preserve_exact_logical_report_and_annotations(
        self,
    ) -> None:
        """Concern: both formats must preserve one proof's full ordered JSON.

        Input/expected matrix:
            expanded report / expanded annotations -> the current JSON values;
            shared report / shared annotations after fresh import -> the same
            dictionaries, field sets, and array order.
        """
        with tempfile.TemporaryDirectory(
            prefix="synthetic-parity-", dir=self.root
        ) as temporary:
            state = Path(temporary)
            report = golden_report(2)
            provider = FixtureProvider()
            narrator = Narrator(
                provider,
                architecture="v2",
                recovery_dir=state / "recovery",
                checkpoint_dir=state / "cache",
                concurrency=1,
            )
            with ExitStack() as stack:
                benchmark.offline_guard(stack)
                session = narrator.capture_session(report)
                with session:
                    generation = narrator.generate_validated(
                        report, session=session
                    )
                    report_handle = session.apply(generation)
                    outputs = {}
                    for storage, _expected in (
                        ("expanded", "current logical values"),
                        ("shared", "same logical values after import"),
                    ):
                        annotations = b"".join(
                            session.iter_annotations_json(
                                generation,
                                report_storage=storage,
                                max_output_bytes=benchmark.SYNTHETIC_MAX_OUTPUT_BYTES,
                            )
                        )
                        report_bytes = b"".join(
                            session.iter_report_json(
                                report_handle,
                                report_storage=storage,
                                max_output_bytes=benchmark.SYNTHETIC_MAX_OUTPUT_BYTES,
                            )
                        )
                        html_bytes = b"".join(
                            iter_render_validated(
                                session,
                                report_handle,
                                report_storage=storage,
                                max_output_bytes=benchmark.SYNTHETIC_MAX_OUTPUT_BYTES,
                            )
                        )
                        outputs[storage] = {
                            "annotations": annotations,
                            "report": report_bytes,
                            "html": html_bytes,
                        }

                    expanded_annotations = json.loads(
                        outputs["expanded"]["annotations"]
                    )
                    expanded_report = json.loads(outputs["expanded"]["report"])
                    shared_annotation_session, shared_generation = (
                        ReportSession.import_indexed(
                            outputs["shared"]["annotations"],
                            compiler_report=report,
                            max_input_bytes=len(
                                outputs["shared"]["annotations"]
                            ),
                        )
                    )
                    with shared_annotation_session:
                        shared_annotations = json.loads(
                            b"".join(
                                shared_annotation_session.iter_annotations_json(
                                    shared_generation,
                                    report_storage="expanded",
                                    max_output_bytes=benchmark.SYNTHETIC_MAX_OUTPUT_BYTES,
                                )
                            )
                        )
                    shared_report_session, shared_report_generation = (
                        ReportSession.import_indexed(
                            outputs["shared"]["report"],
                            max_input_bytes=len(outputs["shared"]["report"]),
                        )
                    )
                    with shared_report_session:
                        shared_report_handle = shared_report_session.apply(
                            shared_report_generation
                        )
                        shared_report = shared_report_session.report_data(
                            shared_report_handle
                        )
                    self.assertEqual(shared_annotations, expanded_annotations)
                    self.assertEqual(shared_report, expanded_report)
                    self.assertEqual(
                        benchmark.rendered_report_digest(
                            outputs["expanded"]["html"]
                        ),
                        benchmark.digest_json(expanded_report),
                    )
                    self.assertEqual(
                        benchmark.rendered_report_digest(
                            outputs["shared"]["html"]
                        ),
                        benchmark.digest_json(
                            json.loads(outputs["shared"]["report"])
                        ),
                    )

    def test_handle_flow_reuses_production_proof_and_import_proves_once(
        self,
    ) -> None:
        """Concern: trusted reuse and independent import must have exact proof counts.

        Input/expected matrix:
            one cold ``handle-paired`` run -> capture=1, fresh proof=1,
            apply/output extra proof=0, trusted ledger/index rebuilds=0,
            independent indexed/session import proofs=1 each, and imported
            apply/output extra proof or ledger/index rebuilds=0.
        """
        with tempfile.TemporaryDirectory(
            prefix="synthetic-gates-", dir=self.root
        ) as temporary:
            cache_dir = Path(temporary) / "state"
            cache_dir.mkdir(mode=0o700)
            metrics = benchmark.run_synthetic_sample(
                argparse.Namespace(
                    synthetic_sample="handle-paired",
                    count=2,
                    cache_state="cold",
                    cache_dir=cache_dir,
                    profile_callbacks=False,
                )
            )
        self.assertEqual(metrics["status"], "PASS")
        gates = metrics["deterministic_gates"]
        expected = (
            ("fresh_capture_initializations", 1),
            ("generation_complete_proofs", 1),
            ("apply_added_complete_proofs", 0),
            ("trusted_output_added_complete_proofs", 0),
            ("apply_added_ledger_index_builds", 0),
            ("trusted_output_added_ledger_index_builds", 0),
            ("shared_import_independent_import_complete_proofs", 1),
            ("shared_import_independent_import_session_proofs", 1),
            ("shared_import_apply_added_complete_proofs", 0),
            ("shared_import_output_added_complete_proofs", 0),
            ("shared_import_apply_added_ledger_index_builds", 0),
            ("shared_import_output_added_ledger_index_builds", 0),
        )
        for name, count in expected:
            with self.subTest(gate=name):
                self.assertEqual(gates[name], count)

    def test_python_pydantic_profile_counts_nested_callbacks(self) -> None:
        """Concern: profiling must observe Python field and model validators.

        Input/expected matrix:
            one typed fixture request-configuration capture -> at least one
            field-validator callback and one model-validator callback.
        """
        report = golden_report(1)
        provider = FixtureProvider()
        descriptor = Transport(provider, V2Options()).descriptor
        measurements = benchmark.PhaseMeasurements()
        with benchmark.profile_python_pydantic_validators(measurements):
            session = ReportSession.capture(report, descriptor)
            session.close()
        callbacks = measurements.callback_snapshot()
        expected = (
            ("field validator", "field."),
            ("model validator", "model."),
        )
        for label, prefix in expected:
            with self.subTest(callback=label):
                self.assertGreater(
                    sum(
                        count
                        for name, count in callbacks.items()
                        if name.startswith(prefix)
                    ),
                    0,
                )


if __name__ == "__main__":
    unittest.main()
