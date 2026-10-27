"""Check flat narration, saved-report reload, and annotation CLI boundaries."""

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
from diffstory.walkthrough import validate_walkthrough


class CliWalkthroughTests(unittest.TestCase):
    """Keep the source-to-reader command and saved artifacts interoperable."""

    def test_snapshot_generates_and_reloads_flat_report(self) -> None:
        """Narration must save a valid flat report and reload without provider work."""
        source = "def f():\n    return 1\n"
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            snapshot = root / "input.json"
            snapshot.write_text(json.dumps(snap({}, {"a.py": source})))
            provider = FixtureProvider()
            with (
                patch("diffstory.cli._create_provider", return_value=provider),
                contextlib.redirect_stdout(io.StringIO()),
            ):
                self.assertEqual(
                    main(
                        [
                            "snapshot",
                            str(snapshot),
                            "--narrate",
                            "--yes",
                            "--out",
                            str(root / "story.html"),
                        ]
                    ),
                    0,
                )
                self.assertEqual(
                    main(
                        [
                            "render",
                            str(root / "story.report.json"),
                            "--out",
                            str(root / "reload.html"),
                        ]
                    ),
                    0,
                )
            report = json.loads((root / "story.report.json").read_text())
            validate_walkthrough(report)
            self.assertEqual(provider.calls, 1)
            self.assertEqual(
                json.loads((root / "reload.report.json").read_text()),
                report,
            )

    def test_annotations_reapply_to_matching_source_without_generation(
        self,
    ) -> None:
        """The optional small annotations export must not contain repeated source."""
        expected = "diffstory.annotations.walkthrough.v1"
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            snapshot = root / "input.json"
            snapshot.write_text(json.dumps(snap({}, {"a.py": "x = 1\n"})))
            with (
                patch(
                    "diffstory.cli._create_provider",
                    return_value=FixtureProvider(),
                ),
                contextlib.redirect_stdout(io.StringIO()),
            ):
                with patch(
                    "diffstory.cli._create_provider",
                    return_value=FixtureProvider(),
                ):
                    self.assertEqual(
                        main(
                            [
                                "snapshot",
                                str(snapshot),
                                "--narrate",
                                "--yes",
                                "--out",
                                str(root / "story.html"),
                            ]
                        ),
                        0,
                    )
                self.assertEqual(
                    main(
                        [
                            "snapshot",
                            str(snapshot),
                            "--annotations",
                            str(root / "story.annotations.json"),
                            "--out",
                            str(root / "reapplied.html"),
                        ]
                    ),
                    0,
                )
            annotations = json.loads(
                (root / "story.annotations.json").read_text()
            )
            self.assertEqual(annotations["schema"], expected)
            self.assertNotIn("packet", annotations)
            validate_walkthrough(
                json.loads((root / "reapplied.report.json").read_text())
            )
            for version in (
                "diffstory.generation.walkthrough.v1",
                "diffstory.generation.walkthrough.v2",
            ):
                with self.subTest(version=version):
                    archived = json.loads(json.dumps(annotations))
                    archived["generation"]["version"] = version
                    for field in (
                        "quick_summary",
                        "impact",
                        "route",
                        "author_reported_checks",
                        "review_checks",
                    ):
                        archived["story"].pop(field)
                    if version.endswith("v1"):
                        archived["story"].pop("preamble")
                    archived_path = root / f"{version[-2:]}.annotations.json"
                    archived_path.write_text(json.dumps(archived))
                    archived_output = root / f"{version[-2:]}.html"
                    with contextlib.redirect_stderr(io.StringIO()):
                        self.assertEqual(
                            main(
                                [
                                    "snapshot",
                                    str(snapshot),
                                    "--annotations",
                                    str(archived_path),
                                    "--out",
                                    str(archived_output),
                                ]
                            ),
                            2,
                        )
                    self.assertFalse(archived_output.exists())
                    self.assertFalse(
                        (root / f"{version[-2:]}.report.json").exists()
                    )

    def test_retired_cli_options_are_rejected(self) -> None:
        """The parser has no generation architecture or storage selector."""
        with tempfile.TemporaryDirectory() as directory:
            snapshot = Path(directory) / "input.json"
            snapshot.write_text(json.dumps(snap({}, {"a.py": "x = 1\n"})))
            for option in ("--architecture", "--report-storage"):
                with (
                    self.subTest(option=option),
                    contextlib.redirect_stderr(io.StringIO()),
                    self.assertRaises(SystemExit),
                ):
                    main(["snapshot", str(snapshot), option, "legacy"])

    def test_render_cli_rejects_retired_storage_markers(self) -> None:
        """The loaded-report boundary rejects each old storage envelope form."""
        markers = [
            {"storage_version": "diffstory.generation-storage.v1"},
            {"artifact_kind": "stored-report"},
            {"root": {}, "tables": {}},
        ]
        compiler_report = compile_snapshot(snap({}, {"a.py": "x = 1\n"}))
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for index, marker in enumerate(markers):
                with self.subTest(marker=marker):
                    stored = json.loads(json.dumps(compiler_report))
                    stored.update(marker)
                    input_path = root / f"stored-{index}.json"
                    output_path = root / f"stored-{index}.html"
                    input_path.write_text(json.dumps(stored))
                    with (
                        contextlib.redirect_stdout(io.StringIO()),
                        contextlib.redirect_stderr(io.StringIO()),
                    ):
                        result = main(
                            [
                                "render",
                                str(input_path),
                                "--out",
                                str(output_path),
                            ]
                        )
                    self.assertEqual(result, 2)
                    self.assertFalse(output_path.exists())
                    self.assertFalse(
                        output_path.with_suffix(".report.json").exists()
                    )

    def test_missing_consent_stops_before_provider_request(self) -> None:
        """A local preview appears before consent and no provider call starts."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            snapshot = root / "input.json"
            snapshot.write_text(json.dumps(snap({}, {"a.py": "x = 1\n"})))
            provider = FixtureProvider()
            output = io.StringIO()
            with (
                patch("diffstory.cli._create_provider", return_value=provider),
                patch("diffstory.cli.sys.stdin", io.StringIO("")),
                contextlib.redirect_stdout(output),
                contextlib.redirect_stderr(io.StringIO()),
            ):
                result = main(
                    [
                        "snapshot",
                        str(snapshot),
                        "--narrate",
                        "--out",
                        str(root / "story.html"),
                    ]
                )
            self.assertEqual(result, 2)
            self.assertEqual(provider.calls, 0)
            self.assertIn("Destination: fixture://offline", output.getvalue())
            self.assertFalse((root / "story.html").exists())
            self.assertFalse((root / "story.report.json").exists())

    def test_default_model_is_resolved_before_consent(self) -> None:
        """An omitted model must select Terra and accept caching for either provider."""
        cases = [
            ("openai", "gpt-6.1-terra"),
            ("codex", "gpt-6.1-terra"),
        ]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            snapshot = root / "input.json"
            snapshot.write_text(json.dumps(snap({}, {"a.py": "x = 1\n"})))
            for provider, expected_model in cases:
                output = io.StringIO()
                errors = io.StringIO()
                with (
                    self.subTest(provider=provider),
                    patch(
                        "diffstory.cli.OpenAIResponsesProvider.require_credentials"
                    ),
                    patch(
                        "diffstory.cli.CodexCLIProvider.require_credentials"
                    ),
                    patch("diffstory.cli.sys.stdin", io.StringIO("")),
                    contextlib.redirect_stdout(output),
                    contextlib.redirect_stderr(errors),
                ):
                    result = main(
                        [
                            "snapshot",
                            str(snapshot),
                            "--narrate",
                            "--provider",
                            provider,
                            "--cache-dir",
                            str(root / provider / "cache"),
                            "--out",
                            str(root / f"{provider}.html"),
                        ]
                    )
                    self.assertEqual(result, 2)
                    self.assertIn(
                        f"model: {expected_model}", output.getvalue()
                    )
                    self.assertIn("Source was not sent", errors.getvalue())

    def test_capacity_preflight_stops_before_provider_request(self) -> None:
        """An oversized prepared request fails before calling the provider."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            snapshot = root / "input.json"
            snapshot.write_text(json.dumps(snap({}, {"a.py": "x = 1\n"})))
            provider = FixtureProvider()
            provider.context_tokens = 1
            with (
                patch("diffstory.cli._create_provider", return_value=provider),
                contextlib.redirect_stdout(io.StringIO()),
                contextlib.redirect_stderr(io.StringIO()),
            ):
                result = main(
                    [
                        "snapshot",
                        str(snapshot),
                        "--narrate",
                        "--yes",
                        "--out",
                        str(root / "story.html"),
                    ]
                )
            self.assertEqual(result, 2)
            self.assertEqual(provider.calls, 0)
            self.assertFalse((root / "story.html").exists())

    def test_evidence_exports_compiler_and_flat_source_packets(self) -> None:
        """Evidence export accepts both manual and current walkthrough reports."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            snapshot = root / "input.json"
            snapshot.write_text(json.dumps(snap({}, {"a.py": "x = 1\n"})))
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(
                    main(
                        [
                            "snapshot",
                            str(snapshot),
                            "--out",
                            str(root / "manual.html"),
                        ]
                    ),
                    0,
                )
                self.assertEqual(
                    main(
                        [
                            "evidence",
                            str(root / "manual.report.json"),
                            "--out",
                            str(root / "manual.evidence.json"),
                        ]
                    ),
                    0,
                )
                with patch(
                    "diffstory.cli._create_provider",
                    return_value=FixtureProvider(),
                ):
                    self.assertEqual(
                        main(
                            [
                                "snapshot",
                                str(snapshot),
                                "--narrate",
                                "--yes",
                                "--out",
                                str(root / "story.html"),
                            ]
                        ),
                        0,
                    )
                self.assertEqual(
                    main(
                        [
                            "evidence",
                            str(root / "story.report.json"),
                            "--out",
                            str(root / "story.evidence.json"),
                        ]
                    ),
                    0,
                )
            manual = json.loads((root / "manual.evidence.json").read_text())
            flat = json.loads((root / "story.report.json").read_text())
            exported = json.loads((root / "story.evidence.json").read_text())
            self.assertEqual(
                manual["schema"], "diffstory.narrative-request.v1"
            )
            self.assertEqual(exported, flat["packet"])

    def test_html_limit_keeps_existing_destinations_unchanged(self) -> None:
        """A failed HTML cap leaves prior outputs and report path untouched."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            snapshot = root / "input.json"
            output = root / "story.html"
            snapshot.write_text(json.dumps(snap({}, {"a.py": "x = 1\n"})))
            output.write_text("existing output")
            with contextlib.redirect_stderr(io.StringIO()):
                result = main(
                    [
                        "snapshot",
                        str(snapshot),
                        "--out",
                        str(output),
                        "--max-html-bytes",
                        "1",
                    ]
                )
            self.assertEqual(result, 2)
            self.assertEqual(output.read_text(), "existing output")
            self.assertFalse((root / "story.report.json").exists())

    def test_report_input_limit_preserves_existing_html(self) -> None:
        """A bounded report read fails before the existing HTML is replaced."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            snapshot = root / "input.json"
            snapshot.write_text(json.dumps(snap({}, {"a.py": "x = 1\n"})))
            report_path = root / "report.json"
            output = root / "report.html"
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(
                    main(
                        [
                            "snapshot",
                            str(snapshot),
                            "--out",
                            str(output),
                        ]
                    ),
                    0,
                )
            source_report = root / "report.report.json"
            source_report.replace(report_path)
            output.write_text("existing output")
            with contextlib.redirect_stderr(io.StringIO()):
                result = main(
                    [
                        "render",
                        str(report_path),
                        "--out",
                        str(output),
                        "--max-report-bytes",
                        "1",
                    ]
                )
            self.assertEqual(result, 2)
            self.assertEqual(output.read_text(), "existing output")

    def test_setup_failure_does_not_echo_provider_or_source_text(self) -> None:
        """Provider exceptions and source text stay out of CLI diagnostics."""
        sentinel = "private_source_and_api_key_4821"
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            snapshot = root / "input.json"
            snapshot.write_text(
                json.dumps(snap({}, {"a.py": f"value = {sentinel!r}\n"}))
            )
            stdout = io.StringIO()
            stderr = io.StringIO()
            with (
                patch(
                    "diffstory.cli._create_provider",
                    side_effect=RuntimeError(sentinel),
                ),
                contextlib.redirect_stdout(stdout),
                contextlib.redirect_stderr(stderr),
            ):
                result = main(
                    [
                        "snapshot",
                        str(snapshot),
                        "--narrate",
                        "--yes",
                        "--out",
                        str(root / "story.html"),
                    ]
                )
            self.assertEqual(result, 2)
            self.assertNotIn(sentinel, stdout.getvalue())
            self.assertNotIn(sentinel, stderr.getvalue())
            self.assertFalse((root / "story.html").exists())


if __name__ == "__main__":
    unittest.main()
