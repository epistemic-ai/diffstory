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
from contextlib import ExitStack
from contextlib import redirect_stdout
from pathlib import Path
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

from diffstory import narrative_runtime
from diffstory.analysis import apply_annotations
from diffstory.narrative import CodexCLIProvider
from diffstory.narrative import Narrator
from diffstory.narrative_manifest import expanded_generation
from diffstory.narrative_plan import canonical_bytes
from diffstory.narrative_runtime import ArtifactCallError
from diffstory.narrative_v2 import Transport
from diffstory.narrative_v2 import V2Options
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

    def test_state_never_resets_and_binding_changes_fail(self) -> None:
        """A repeat cold start or changed source/configuration preserves every byte."""
        args = self.args()
        benchmark.initialize(args)
        marker = self.state / "LOCAL_FIXTURE_ONLY.json"
        original = marker.read_bytes()
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


if __name__ == "__main__":
    unittest.main()
