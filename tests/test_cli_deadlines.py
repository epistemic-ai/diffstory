"""Keep large PR deadlines usable without renewing earlier narration budgets."""

import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from test_analysis import snap
from test_walkthrough import FixtureProvider

from diffstory.analysis import compile_snapshot
from diffstory.cli import main
from diffstory.walkthrough import WalkthroughNarrator


def parallel_snapshot() -> dict:
    """Build four independent modules that require four synthetic writers."""
    before = (
        "def process(value):\n    # " + "x" * 12_000 + "\n    return value\n"
    )
    return snap(
        {f"m{i}.py": before for i in range(4)},
        {
            f"m{i}.py": before.replace("return value", "return value + 1")
            for i in range(4)
        },
    )


class CliDeadlineTests(unittest.TestCase):
    """Bind finite deadlines to the plan, consent preview, and durable run state."""

    def test_new_parallel_runs_budget_writer_waves_and_editing(self) -> None:
        """A cold parallel plan must include editorial time in its single deadline."""
        cases = [
            ([], 480),
            (["--concurrency", "2"], 720),
            (["--concurrency", "1"], 1200),
            (["--timeout-seconds", "600"], 600),
        ]
        for options, seconds in cases:
            with (
                self.subTest(options=options),
                tempfile.TemporaryDirectory() as directory,
            ):
                root = Path(directory)
                provider = FixtureProvider()
                output = io.StringIO()
                with (
                    patch(
                        "diffstory.cli.from_github",
                        return_value=parallel_snapshot(),
                    ),
                    patch(
                        "diffstory.cli._create_provider", return_value=provider
                    ),
                    contextlib.redirect_stdout(output),
                ):
                    result = main(
                        [
                            "owner/project#123",
                            "--yes",
                            "--out",
                            str(root / "story.html"),
                            *options,
                        ]
                    )
                self.assertEqual(result, 0)
                self.assertEqual(provider.calls, 5)
                self.assertIn(f"{seconds} seconds total", output.getvalue())
                self.assertLessEqual(provider.timeout, 240)
                record = json.loads(
                    next(
                        (root / "story.narration-state").glob("*/run.json")
                    ).read_text()
                )
                self.assertEqual(record["limits"]["seconds"], seconds)
                self.assertEqual(record["state"], "complete")

    def test_prior_parallel_budgets_stay_expired_until_explicit_fresh_state(
        self,
    ) -> None:
        """A new default must not bypass old deadlines in either state layout."""
        for legacy in (False, True):
            with (
                self.subTest(legacy=legacy),
                tempfile.TemporaryDirectory() as directory,
            ):
                root = Path(directory)
                captured = parallel_snapshot()
                provider = FixtureProvider()
                narrator = WalkthroughNarrator(
                    provider, cache_dir=root / "story.narration-cache"
                )
                narrator.prepare(compile_snapshot(captured))
                state_root = root / "story.narration-state"
                narrator.state_dir = (
                    state_root
                    if legacy
                    else state_root / narrator.prepared_run_identity()
                )
                narrator.generate_prepared()
                record_path = narrator.state_dir / "run.json"
                saved = json.loads(record_path.read_text())
                saved["deadline_at"] = 0
                record_path.write_text(json.dumps(saved))
                contents = record_path.read_bytes()
                output, errors = io.StringIO(), io.StringIO()
                arguments = [
                    "owner/project#123",
                    "--yes",
                    "--out",
                    str(root / "story.html"),
                ]
                with (
                    patch("diffstory.cli.from_github", return_value=captured),
                    patch(
                        "diffstory.cli._create_provider", return_value=provider
                    ),
                    contextlib.redirect_stdout(output),
                    contextlib.redirect_stderr(errors),
                ):
                    self.assertEqual(main(arguments), 2)
                    self.assertEqual(provider.calls, 5)
                    self.assertIn("240 seconds total", output.getvalue())
                    self.assertIn("--fresh-state", errors.getvalue())
                    self.assertEqual(
                        main(
                            [
                                *arguments,
                                "--fresh-state",
                                str(root / "deliberate-new-state"),
                            ]
                        ),
                        0,
                    )
                self.assertEqual(provider.calls, 5)
                self.assertEqual(record_path.read_bytes(), contents)
                fresh = json.loads(
                    (root / "deliberate-new-state/run.json").read_text()
                )
                self.assertEqual(fresh["limits"]["seconds"], 480)
                self.assertEqual(fresh["state"], "complete")

    def test_equal_timeout_override_keeps_the_same_exhausted_budget(
        self,
    ) -> None:
        """An integer override equal to the default must not create a new identity."""
        cases = [
            (snap({}, {"a.py": "x = 1\n"}), 240, 1),
            (parallel_snapshot(), 480, 5),
        ]
        for captured, seconds, calls in cases:
            with (
                self.subTest(seconds=seconds),
                tempfile.TemporaryDirectory() as directory,
            ):
                root = Path(directory)
                provider = FixtureProvider()
                errors = io.StringIO()
                arguments = [
                    "owner/project#123",
                    "--yes",
                    "--no-cache",
                    "--out",
                    str(root / "story.html"),
                ]
                with (
                    patch("diffstory.cli.from_github", return_value=captured),
                    patch(
                        "diffstory.cli._create_provider", return_value=provider
                    ),
                    contextlib.redirect_stdout(io.StringIO()),
                    contextlib.redirect_stderr(errors),
                ):
                    self.assertEqual(main(arguments), 0)
                    record_path = next(
                        (root / "story.narration-state").glob("*/run.json")
                    )
                    saved = json.loads(record_path.read_text())
                    saved["attempts"] = dict.fromkeys(saved["attempts"], 2)
                    saved["usage"]["calls"] = sum(saved["attempts"].values())
                    record_path.write_text(json.dumps(saved))
                    self.assertEqual(
                        main([*arguments, "--timeout-seconds", str(seconds)]),
                        2,
                    )
                self.assertRegex(
                    errors.getvalue(),
                    "budget exhausted|stopped after a failed request",
                )
                self.assertEqual(provider.calls, calls)
                self.assertEqual(
                    list((root / "story.narration-state").glob("*/run.json")),
                    [record_path],
                )

    def test_invalid_timeouts_stop_before_ingestion_or_provider_setup(
        self,
    ) -> None:
        """A malformed or excessive override must not reach any source transfer."""
        for value in ("0", "-1", "nan", "inf", "1.5", "2161"):
            with (
                self.subTest(value=value),
                patch("diffstory.cli.from_github") as ingest,
                patch("diffstory.cli._create_provider") as create,
                contextlib.redirect_stderr(io.StringIO()),
            ):
                try:
                    result = main(
                        ["owner/project#123", "--timeout-seconds", value]
                    )
                except SystemExit as error:
                    result = error.code
                self.assertEqual(result, 2)
                ingest.assert_not_called()
                create.assert_not_called()
