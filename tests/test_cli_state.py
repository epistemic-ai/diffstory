"""Prevent default output reuse from mixing unrelated narration budgets."""

import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from test_analysis import snap
from test_walkthrough import FixtureProvider

from diffstory.cli import main
from diffstory.walkthrough_runtime import MAX_RECORD_BYTES


def run_cli(
    output: Path, captured: dict, provider: FixtureProvider, *options: str
) -> tuple[int, str]:
    """Run approved narration against synthetic source without external calls.

    Args:
        output: Shared HTML destination under an isolated temporary directory.
        captured: Complete synthetic snapshot returned by GitHub ingestion.
        provider: Offline provider that measures new requests.
        *options: Additional CLI options, including explicit state overrides.

    Returns:
        The exit status and sanitized error output. Provider calls and state
        writes occur only through the real CLI and generation runtime.
    """
    errors = io.StringIO()
    with (
        patch("diffstory.cli.from_github", return_value=captured),
        patch("diffstory.cli._create_provider", return_value=provider),
        contextlib.redirect_stdout(io.StringIO()),
        contextlib.redirect_stderr(errors),
    ):
        result = main(
            ["owner/project#123", "--out", str(output), "--yes", *options]
        )
    return result, errors.getvalue()


class CliStateTests(unittest.TestCase):
    """Keep exact-run resumption and explicit budget resets intact."""

    def test_default_state_isolates_plans_and_resumes_matching_input(
        self,
    ) -> None:
        """Changed source, model, or limits must not collide at one output path."""
        first = snap({}, {"a.py": "x = 1\n"})
        recaptured = snap({}, {"a.py": "x = 1\n"}, captured_at="later")
        other = snap({}, {"a.py": "x = 2\n"}, head_sha="c" * 40)
        cases = [
            (first, "model-one", [], 1, 1),
            (recaptured, "model-one", [], 1, 1),
            (other, "model-one", [], 2, 2),
            (first, "model-two", [], 3, 3),
            (first, "model-one", ["--concurrency", "1"], 3, 4),
            (first, "model-one", [], 3, 4),
        ]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            provider = FixtureProvider()
            for (
                captured,
                model,
                options,
                expected_calls,
                expected_states,
            ) in cases:
                with self.subTest(
                    model=model, options=options, calls=expected_calls
                ):
                    provider.model = model
                    result, errors = run_cli(
                        root / "walkthrough.html", captured, provider, *options
                    )
                    self.assertEqual(result, 0, errors)
                    self.assertEqual(provider.calls, expected_calls)
                    states = list(
                        (root / "walkthrough.narration-state").glob(
                            "*/run.json"
                        )
                    )
                    self.assertEqual(len(states), expected_states)
                    for state in states:
                        record = json.loads(state.read_text())
                        self.assertEqual(state.parent.name, record["identity"])

    def test_default_state_retains_spent_and_legacy_budgets(self) -> None:
        """Namespacing must not give an identical plan fresh attempts or time."""
        cases = [
            (False, "attempts", "request budget exhausted"),
            (True, "attempts", "request budget exhausted"),
            (False, "deadline", "run deadline"),
            (True, "deadline", "run deadline"),
        ]
        for legacy, exhausted, diagnostic in cases:
            with (
                self.subTest(legacy=legacy, exhausted=exhausted),
                tempfile.TemporaryDirectory() as directory,
            ):
                root = Path(directory)
                output = root / "walkthrough.html"
                captured = snap({}, {"a.py": "x = 1\n"})
                provider = FixtureProvider()
                self.assertEqual(
                    run_cli(output, captured, provider, "--no-cache"), (0, "")
                )
                state_root = root / "walkthrough.narration-state"
                path = next(state_root.glob("*/run.json"))
                saved = json.loads(path.read_text())
                if exhausted == "attempts":
                    key = next(iter(saved["attempts"]))
                    saved["attempts"][key] = 2
                    saved["usage"]["calls"] = 2
                    saved["state"] = "failed"
                else:
                    saved["deadline_at"] = 0
                path.write_text(json.dumps(saved))
                if legacy:
                    path = path.replace(state_root / "run.json")
                result, errors = run_cli(
                    output, captured, provider, "--no-cache"
                )
                self.assertEqual(result, 2)
                self.assertIn(diagnostic, errors)
                self.assertEqual(provider.calls, 1)
                retained = json.loads(path.read_text())
                self.assertEqual(retained["attempts"], saved["attempts"])
                self.assertEqual(retained["deadline_at"], saved["deadline_at"])
                previous_bytes = path.read_bytes()
                other = snap({}, {"a.py": "x = 2\n"})
                self.assertEqual(
                    run_cli(output, other, provider, "--no-cache"), (0, "")
                )
                self.assertEqual(provider.calls, 2)
                self.assertEqual(path.read_bytes(), previous_bytes)

    def test_unidentifiable_legacy_state_fails_before_confirmation(
        self,
    ) -> None:
        """Unreadable legacy identity must never silently grant a new budget."""
        cases = [
            b"not json",
            b"[]",
            b"{}",
            b'{"identity": 1}',
            b'{"identity": "not-a-digest"}',
            b" " * (MAX_RECORD_BYTES + 1),
        ]
        for contents in cases:
            with (
                self.subTest(contents=contents[:30]),
                tempfile.TemporaryDirectory() as directory,
            ):
                root = Path(directory)
                state_root = root / "walkthrough.narration-state"
                state_root.mkdir()
                path = state_root / "run.json"
                path.write_bytes(contents)
                provider = FixtureProvider()
                with patch(
                    "diffstory.cli._confirm_source_transfer"
                ) as confirm:
                    result, errors = run_cli(
                        root / "walkthrough.html",
                        snap({}, {"a.py": "x = 1\n"}),
                        provider,
                    )
                self.assertEqual(result, 2)
                self.assertIn("state is corrupt", errors)
                self.assertEqual(provider.calls, 0)
                confirm.assert_not_called()
                self.assertEqual(path.read_bytes(), contents)
                self.assertEqual(list(state_root.iterdir()), [path])

    def test_explicit_state_paths_keep_exact_directory_semantics(self) -> None:
        """Explicit state and fresh-state choices must not gain an automatic fallback."""
        cases = [
            ("--state-dir", "Run inputs or limits changed"),
            ("--fresh-state", "requires a new directory"),
        ]
        for option, diagnostic in cases:
            with (
                self.subTest(option=option),
                tempfile.TemporaryDirectory() as directory,
            ):
                root = Path(directory)
                output = root / "walkthrough.html"
                state = root / "explicit-state"
                provider = FixtureProvider()
                result, errors = run_cli(
                    output,
                    snap({}, {"a.py": "x = 1\n"}),
                    provider,
                    option,
                    str(state),
                )
                self.assertEqual(result, 0, errors)
                path = state / "run.json"
                contents = path.read_bytes()
                result, errors = run_cli(
                    output,
                    snap({}, {"a.py": "x = 2\n"}),
                    provider,
                    option,
                    str(state),
                )
                self.assertEqual(result, 2)
                self.assertIn(diagnostic, errors)
                self.assertEqual(provider.calls, 1)
                self.assertEqual(path.read_bytes(), contents)
