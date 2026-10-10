"""CLI contracts for narration checkpoints, progress, and run outcomes."""

from __future__ import annotations

import io
import json
import tempfile
import unittest
from contextlib import redirect_stderr
from contextlib import redirect_stdout
from pathlib import Path
from typing import TYPE_CHECKING
from typing import ClassVar
from unittest.mock import patch

from test_narrative import FakeProvider

from diffstory.analysis import compile_snapshot
from diffstory.cli import _build_parser
from diffstory.cli import _generate_narration
from diffstory.cli import _progress_callback
from diffstory.cli import main
from diffstory.narrative import OPENAI_MODEL
from diffstory.narrative import OPENAI_URL
from diffstory.narrative import Narrator
from diffstory.narrative_runtime import ProgressEvent
from diffstory.narrative_runtime import RuntimeStageError

if TYPE_CHECKING:
    from argparse import Namespace
    from collections.abc import Callable

USAGE = {
    "input_tokens": 17,
    "cached_input_tokens": 2,
    "output_tokens": 9,
    "calls": 1,
    "unreported_calls": 0,
}


def sample_snapshot() -> dict:
    """Build a small source snapshot that can be compiled without I/O.

    Returns:
        A valid two-revision snapshot for one changed Python function.
    """
    return {
        "schema": "diffstory.snapshot.v1",
        "meta": {
            "base_sha": "a" * 40,
            "head_sha": "b" * 40,
            "scope": "changed files",
            "changed_files": 1,
            "title": "A sample change",
        },
        "fragments": [
            {
                "path": "module.py",
                "side": "base",
                "text": "def calculate(value):\n    return value\n",
                "start_line": 1,
                "scope": "full",
            },
            {
                "path": "module.py",
                "side": "head",
                "text": "def calculate(value):\n    return value + 1\n",
                "start_line": 1,
                "scope": "full",
            },
        ],
        "file_evidence": [
            {
                "base": {
                    "path": "module.py",
                    "state": "supplied",
                    "coverage": "full",
                },
                "head": {
                    "path": "module.py",
                    "state": "supplied",
                    "coverage": "full",
                },
            },
        ],
        "warnings": [],
    }


def narration_args(out: Path, *options: str) -> Namespace:
    """Parse one complete GitHub narration command for focused CLI checks.

    Args:
        out: HTML output path for the command.
        *options: Additional narration options.

    Returns:
        Parsed arguments for a GitHub narration command.
    """
    return _build_parser().parse_args(
        [
            "github",
            "owner/repo#1",
            "--narrate",
            "--architecture",
            "legacy",
            "--yes",
            "--out",
            str(out),
            *options,
        ]
    )


class StubProvider:
    """Satisfy CLI provider display and credential checks without a request."""

    name = "openai"
    model = OPENAI_MODEL
    destination = OPENAI_URL

    def require_credentials(self) -> None:
        """Pass the local credential preflight without external access."""


class FakeUsage:
    """Return fixed current-run usage for CLI summary checks."""

    def report(self) -> dict:
        """Return a copy of the fixed usage record."""
        return dict(USAGE)


class RecordingNarrator:
    """Capture CLI runtime options and return deterministic annotations."""

    constructed: ClassVar[list[RecordingNarrator]] = []
    generation_error: ClassVar[BaseException | None] = None

    def __init__(
        self,
        provider: StubProvider,
        *,
        concurrency: int,
        checkpoint_dir: Path | None,
        on_progress: Callable[[ProgressEvent], None],
    ) -> None:
        """Store the options supplied by the CLI.

        Args:
            provider: Stub provider selected by the CLI.
            concurrency: Requested worker limit.
            checkpoint_dir: Selected cache directory, if any.
            on_progress: Runtime progress callback.
        """
        self.provider = provider
        self.concurrency = concurrency
        self.checkpoint_dir = checkpoint_dir
        self.on_progress = on_progress
        self.usage = FakeUsage()
        self.reused_checkpoints = 3
        type(self).constructed.append(self)

    def preview(self, _report: dict) -> dict:
        """Return a minimal local preview for the CLI transfer display."""
        return {"chunks": 1, "calls": 1}

    def generate(self, _report: dict) -> dict:
        """Return fixed annotations or raise the configured run outcome."""
        if type(self).generation_error is not None:
            raise type(self).generation_error
        return {"generation": {"usage": dict(USAGE)}}


class CliNarrationTests(unittest.TestCase):
    """Check CLI choices that connect narration to the generation runtime."""

    def setUp(self) -> None:
        """Reset the shared fake narrator state before each CLI check."""
        RecordingNarrator.constructed.clear()
        RecordingNarrator.generation_error = None

    def test_cache_modes_and_runtime_options_reach_narrator(self) -> None:
        """Prevent accidental cache paths or concurrency from being ignored."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cases = [
                ("default", [], root / "default.narration-cache"),
                (
                    "override",
                    ["--cache-dir", str(root / "chosen-cache")],
                    root / "chosen-cache",
                ),
                ("disabled", ["--no-cache"], None),
                (
                    "codex explicit model",
                    ["--provider", "codex", "--model", "gpt-6.1-sol"],
                    root / "codex-explicit-model.narration-cache",
                ),
            ]
            report = compile_snapshot(sample_snapshot())
            for name, options, expected_cache in cases:
                with self.subTest(case=name):
                    out = root / f"{name.replace(' ', '-')}.html"
                    args = narration_args(
                        out,
                        "--concurrency",
                        "2",
                        *options,
                    )
                    with (
                        patch(
                            "diffstory.cli.OpenAIResponsesProvider",
                            return_value=StubProvider(),
                        ),
                        patch(
                            "diffstory.cli.CodexCLIProvider",
                            return_value=StubProvider(),
                        ),
                        patch(
                            "diffstory.cli.Narrator",
                            RecordingNarrator,
                        ),
                        patch(
                            "diffstory.cli.apply_annotations",
                            return_value=report,
                        ),
                        redirect_stdout(io.StringIO()) as output,
                    ):
                        _generate_narration(args, report)

                    narrator = RecordingNarrator.constructed[-1]
                    self.assertEqual(narrator.concurrency, 2)
                    self.assertEqual(narrator.checkpoint_dir, expected_cache)
                    self.assertTrue(callable(narrator.on_progress))
                    self.assertIn(
                        "Provider-reported tokens:", output.getvalue()
                    )
                    self.assertIn(
                        "Reused narration checkpoints: 3",
                        output.getvalue(),
                    )

    def test_codex_default_disables_cache_and_requires_model_for_override(
        self,
    ) -> None:
        """Keep unresolved Codex model runs out of persistent checkpoint reuse."""
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory) / "walkthrough.html"
            report = compile_snapshot(sample_snapshot())
            args = narration_args(out, "--provider", "codex")
            with (
                patch(
                    "diffstory.cli.CodexCLIProvider",
                    return_value=StubProvider(),
                ),
                patch("diffstory.cli.Narrator", RecordingNarrator),
                patch("diffstory.cli.apply_annotations", return_value=report),
                redirect_stdout(io.StringIO()) as output,
            ):
                _generate_narration(args, report)

            self.assertIsNone(RecordingNarrator.constructed[-1].checkpoint_dir)
            self.assertIn(
                "Codex uses its unresolved default model", output.getvalue()
            )

            explicit_cache = narration_args(
                out,
                "--provider",
                "codex",
                "--cache-dir",
                str(Path(directory) / "cache"),
            )
            with (
                patch("diffstory.cli.CodexCLIProvider") as provider_factory,
                patch("diffstory.cli.Narrator") as narrator_factory,
                self.assertRaisesRegex(ValueError, "requires --model"),
            ):
                _generate_narration(explicit_cache, report)
            provider_factory.assert_not_called()
            narrator_factory.assert_not_called()

    def test_progress_is_throttled_but_stage_boundaries_are_always_shown(
        self,
    ) -> None:
        """Prevent frequent runtime events from hiding useful stage changes."""
        event_names = [
            "stage_started",
            "heartbeat",
            "call_started",
            "heartbeat",
            "stage_completed",
            "stage_started",
        ]
        events = [
            ProgressEvent(
                event=name,
                stage="chunks",
                task_id="task-0",
                active_calls=1,
                completed=1,
                planned=3,
                reused_checkpoints=2,
                provider_calls=4,
                elapsed_seconds=1.2,
            )
            for name in event_names
        ]
        with (
            patch(
                "diffstory.cli.time.monotonic",
                side_effect=[0.0, 1.0, 4.9, 5.0, 5.1, 5.2],
            ),
            redirect_stdout(io.StringIO()) as output,
        ):
            report_progress = _progress_callback()
            for event in events:
                report_progress(event)

        lines = output.getvalue().splitlines()
        self.assertEqual(len(lines), 4)
        self.assertIn("chunks", lines[0])
        self.assertIn("tasks 1/3", lines[0])
        self.assertIn("active 1", lines[0])
        self.assertIn("calls 4", lines[0])
        self.assertIn("reused 2", lines[0])
        self.assertIn("elapsed 1s", lines[0])
        self.assertTrue(lines[2].startswith("Narration progress: chunks"))
        self.assertTrue(lines[3].startswith("Narration progress: chunks"))

    def test_real_narrator_failure_reports_usage_reuse_and_safe_error(
        self,
    ) -> None:
        """Keep provider details private while reporting failed-run accounting."""

        class FailingProvider(StubProvider):
            """Fail one real runtime call with private provider text."""

            context_tokens = 1_050_000
            max_output_tokens = 128_000
            input_overhead_bytes = 0

            def prepare(
                self,
                _system: str,
                _data: dict,
                _schema_name: str,
                _schema: dict,
                _max_output_tokens: int,
            ) -> bytes:
                """Return a bounded placeholder request body."""
                return b"{}"

            def complete(
                self, _body: bytes, _timeout: float
            ) -> tuple[dict, dict]:
                """Raise a transport error that the runtime must sanitize."""
                msg = "private provider response text"
                raise ConnectionError(msg)

        class ObservedNarrator(Narrator):
            """Retain the real narrator instance created by the CLI."""

            instance: ObservedNarrator | None = None

            def __init__(self, *args: object, **kwargs: object) -> None:
                """Record the instance after normal runtime initialization."""
                super().__init__(*args, **kwargs)
                type(self).instance = self

        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory) / "walkthrough.html"
            error_output = io.StringIO()
            with (
                patch(
                    "diffstory.cli.from_github", return_value=sample_snapshot()
                ),
                patch(
                    "diffstory.cli.OpenAIResponsesProvider",
                    return_value=FailingProvider(),
                ),
                patch("diffstory.cli.Narrator", ObservedNarrator),
                redirect_stdout(io.StringIO()) as output,
                redirect_stderr(error_output),
            ):
                result = main(
                    [
                        "github",
                        "owner/repo#1",
                        "--narrate",
                        "--architecture",
                        "legacy",
                        "--yes",
                        "--out",
                        str(out),
                        "--concurrency",
                        "2",
                    ]
                )

            self.assertEqual(result, 2)
            self.assertIsNotNone(ObservedNarrator.instance)
            narrator = ObservedNarrator.instance
            self.assertEqual(narrator.concurrency, 2)
            self.assertEqual(
                narrator.checkpoint_dir,
                out.with_suffix(".narration-cache"),
            )
            self.assertTrue(callable(narrator.on_progress))
            self.assertIn(
                "Provider usage unavailable for 1 of 1 calls",
                output.getvalue(),
            )
            self.assertIn("Reused narration checkpoints: 0", output.getvalue())
            self.assertIn("transport", error_output.getvalue())
            self.assertNotIn(
                "private provider response text", error_output.getvalue()
            )
            self.assertFalse(out.exists())

    def test_interrupt_returns_130_and_reports_current_run_summary(
        self,
    ) -> None:
        """Preserve usage and reuse counts when the user interrupts generation."""
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory) / "walkthrough.html"
            error_output = io.StringIO()
            RecordingNarrator.generation_error = KeyboardInterrupt()
            with (
                patch(
                    "diffstory.cli.from_github", return_value=sample_snapshot()
                ),
                patch(
                    "diffstory.cli.OpenAIResponsesProvider",
                    return_value=StubProvider(),
                ),
                patch("diffstory.cli.Narrator", RecordingNarrator),
                redirect_stdout(io.StringIO()) as output,
                redirect_stderr(error_output),
            ):
                result = main(
                    [
                        "github",
                        "owner/repo#1",
                        "--narrate",
                        "--architecture",
                        "legacy",
                        "--yes",
                        "--out",
                        str(out),
                    ]
                )

            self.assertEqual(result, 130)
            self.assertIn("Provider-reported tokens:", output.getvalue())
            self.assertIn("Reused narration checkpoints: 3", output.getvalue())
            self.assertIn("diffstory: interrupted", error_output.getvalue())
            self.assertFalse(out.exists())

    def test_controlled_runtime_error_returns_a_cli_error(self) -> None:
        """Show safe runtime failure details without exposing a traceback."""
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory) / "walkthrough.html"
            error_output = io.StringIO()
            RecordingNarrator.generation_error = RuntimeStageError(
                "chunks",
                "task-0",
                "transport",
            )
            with (
                patch(
                    "diffstory.cli.from_github", return_value=sample_snapshot()
                ),
                patch(
                    "diffstory.cli.OpenAIResponsesProvider",
                    return_value=StubProvider(),
                ),
                patch("diffstory.cli.Narrator", RecordingNarrator),
                redirect_stdout(io.StringIO()) as output,
                redirect_stderr(error_output),
            ):
                result = main(
                    [
                        "github",
                        "owner/repo#1",
                        "--narrate",
                        "--architecture",
                        "legacy",
                        "--yes",
                        "--out",
                        str(out),
                    ]
                )

            self.assertEqual(result, 2)
            self.assertIn("Stage 'chunks' failed", error_output.getvalue())
            self.assertIn("transport", error_output.getvalue())
            self.assertIn("Provider-reported tokens:", output.getvalue())
            self.assertIn("Reused narration checkpoints: 3", output.getvalue())

    def test_cli_interrupt_resumes_saved_chunk_and_writes_outputs(
        self,
    ) -> None:
        """Resume a saved source chunk after a CLI interrupt without recounting it."""
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory) / "walkthrough.html"
            cache_dir = out.with_suffix(".narration-cache")
            first_snapshot = sample_snapshot()
            first_snapshot["meta"]["captured_at"] = "fetch-one"
            second_snapshot = sample_snapshot()
            second_snapshot["meta"]["captured_at"] = "fetch-two"
            first_provider = FakeProvider()
            resumed_provider = FakeProvider()

            def interrupt_after_chunk_checkpoint(
                event: ProgressEvent,
            ) -> None:
                if (
                    event.event == "stage_completed"
                    and event.stage == "chunks"
                ):
                    raise KeyboardInterrupt

            command = [
                "github",
                "owner/repo#1",
                "--narrate",
                "--architecture",
                "legacy",
                "--yes",
                "--concurrency",
                "1",
                "--out",
                str(out),
            ]
            first_error = io.StringIO()
            with (
                patch(
                    "diffstory.cli.from_github",
                    side_effect=[first_snapshot, second_snapshot],
                ),
                patch(
                    "diffstory.cli.OpenAIResponsesProvider",
                    side_effect=[first_provider, resumed_provider],
                ),
                patch(
                    "diffstory.cli._progress_callback",
                    side_effect=[
                        interrupt_after_chunk_checkpoint,
                        _progress_callback(),
                    ],
                ),
                redirect_stdout(io.StringIO()) as first_output,
                redirect_stderr(first_error),
            ):
                first_result = main(command)
                checkpoint_files = list(cache_dir.glob("*/*.json"))
                first_outputs_written = any(
                    path.exists()
                    for path in (
                        out,
                        out.with_suffix(".annotations.json"),
                        out.with_suffix(".report.json"),
                    )
                )
                with redirect_stdout(io.StringIO()) as resumed_output:
                    resumed_result = main(command)

            self.assertEqual(first_result, 130)
            self.assertIn("diffstory: interrupted", first_error.getvalue())
            self.assertIn(
                "Reused narration checkpoints: 0",
                first_output.getvalue(),
            )
            self.assertFalse(first_outputs_written)
            self.assertEqual(
                [call["name"] for call in first_provider.calls],
                ["diffstory_chunk"],
            )
            self.assertEqual(len(checkpoint_files), 1)
            self.assertEqual(resumed_result, 0)
            self.assertNotIn(
                "diffstory_chunk",
                [call["name"] for call in resumed_provider.calls],
            )
            self.assertEqual(len(resumed_provider.calls), 3)
            self.assertIn(
                "Reused narration checkpoints: 1", resumed_output.getvalue()
            )

            annotations_path = out.with_suffix(".annotations.json")
            report_path = out.with_suffix(".report.json")
            self.assertTrue(out.exists())
            self.assertTrue(annotations_path.exists())
            self.assertTrue(report_path.exists())
            annotations = json.loads(
                annotations_path.read_text(encoding="utf-8")
            )
            report = json.loads(report_path.read_text(encoding="utf-8"))
            self.assertEqual(annotations["schema"], "diffstory.annotations.v1")
            self.assertEqual(annotations["generation"]["usage"]["calls"], 3)
            self.assertEqual(
                annotations["generation"]["usage"]["input_tokens"], 360
            )
            self.assertEqual(
                report["generation"]["schema"], "diffstory.generation.v1"
            )
            self.assertIn(
                "Model-generated narration", out.read_text(encoding="utf-8")
            )


if __name__ == "__main__":
    unittest.main()
