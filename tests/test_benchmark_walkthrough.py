"""Check the small benchmark's offline default and reproducible metric contract."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from test_analysis import snap

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "walkthrough_benchmark", ROOT / "scripts/benchmark_walkthrough.py"
)
BENCHMARK = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(BENCHMARK)


class WalkthroughBenchmarkTests(unittest.TestCase):
    """Separate structural local measurements from live quality claims."""

    def test_offline_default_reports_bound_metrics_without_network(
        self,
    ) -> None:
        """An ordinary baseline must not transfer source or invent token usage."""
        fixture = "demo"
        with tempfile.TemporaryDirectory() as directory:
            args = argparse.Namespace(
                fixture=fixture,
                snapshot=None,
                live=False,
                yes=False,
                provider="codex",
                model="gpt-6.1-sol",
                output_dir=Path(directory),
            )
            with patch(
                "socket.create_connection",
                side_effect=AssertionError("Unexpected network"),
            ):
                metrics = BENCHMARK.run(args)
            self.assertEqual(metrics["mode"], "offline")
            self.assertEqual(metrics["usage"]["calls"], 1)
            self.assertEqual(metrics["usage"]["unreported_calls"], 1)
            self.assertEqual(metrics["usage"]["output_tokens"], 0)
            expected = hashlib.sha256(
                (ROOT / "examples/demo.snapshot.json").read_bytes()
            ).hexdigest()
            self.assertEqual(metrics["fixture_sha256"], expected)

    def test_complex_offline_fixture_runs_source_free_editor_adapter(
        self,
    ) -> None:
        """The benchmark editor fixture must follow the current grouping schema."""
        before_source = (
            "def process(value):\n    # "
            + "x" * 12_000
            + "\n    return value\n"
        )
        after_source = before_source.replace(
            "return value", "return value + 1"
        )
        before = {f"m{index}.py": before_source for index in range(4)}
        after = dict.fromkeys(before, after_source)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            snapshot = root / "input.snapshot.json"
            snapshot.write_text(json.dumps(snap(before, after)))
            args = argparse.Namespace(
                fixture="complex",
                snapshot=snapshot,
                live=False,
                yes=False,
                provider="codex",
                model="gpt-6.1-sol",
                output_dir=root / "output",
            )
            metrics = BENCHMARK.run(args)
        self.assertEqual(metrics["mode"], "offline")
        self.assertEqual(metrics["writing_requests"], 4)
        self.assertEqual(metrics["editorial_requests"], 1)
        self.assertEqual(metrics["usage"]["calls"], 5)

    def test_live_transfer_requires_explicit_confirmation_before_input(
        self,
    ) -> None:
        """A live option alone must not read or transmit private fixture source."""
        yes, expected = (False, ValueError)
        args = argparse.Namespace(live=True, yes=yes)
        with (
            patch.object(BENCHMARK, "_read_fixture") as read_fixture,
            self.assertRaises(expected),
        ):
            BENCHMARK.run(args)
        read_fixture.assert_not_called()


if __name__ == "__main__":
    unittest.main()
