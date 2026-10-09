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

from test_narrative_v2 import CASE_AFTER
from test_narrative_v2 import CASE_BEFORE
from test_narrative_v2 import CASE_EFFECT
from test_narrative_v2 import STRIP_AFTER
from test_narrative_v2 import STRIP_BEFORE
from test_narrative_v2 import STRIP_EFFECT
from test_narrative_v2 import FixtureProvider
from test_narrative_v2 import golden_report

from diffstory.analysis import apply_annotations
from diffstory.narrative import CodexCLIProvider
from diffstory.narrative import Narrator
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
            reserve_bytes=0,
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
            "--reserve-bytes",
            "0",
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
        guard = benchmark.StorageGuard(self.state, 0)
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
