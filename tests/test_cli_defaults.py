"""Check the short GitHub command and its source-transfer boundary."""

import contextlib
import io
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from test_analysis import snap
from test_walkthrough import FixtureProvider

from diffstory.analysis import SCHEMA
from diffstory.cli import main
from diffstory.walkthrough import validate_walkthrough


class CliDefaultTests(unittest.TestCase):
    """Keep convenient PR defaults compatible with explicit local modes."""

    def test_pr_forms_default_to_codex_sol_but_require_consent(self) -> None:
        """Each PR spelling must prepare the same model without sending source."""
        cases = [
            ["owner/project#123"],
            ["https://github.com/owner/project/pull/123"],
            ["github", "owner/project#123"],
        ]
        for arguments in cases:
            with (
                self.subTest(arguments=arguments),
                tempfile.TemporaryDirectory() as directory,
            ):
                output = io.StringIO()
                errors = io.StringIO()
                destination = Path(directory) / "story.html"
                with (
                    patch(
                        "diffstory.cli.from_github",
                        return_value=snap({}, {"a.py": "x = 1\n"}),
                    ) as ingest,
                    patch(
                        "diffstory.cli.CodexCLIProvider.require_credentials"
                    ),
                    patch(
                        "diffstory.cli.CodexCLIProvider.complete_reply"
                    ) as send,
                    patch("diffstory.cli.sys.stdin", io.StringIO("")),
                    contextlib.redirect_stdout(output),
                    contextlib.redirect_stderr(errors),
                ):
                    result = main([*arguments, "--out", str(destination)])
                self.assertEqual(result, 2)
                self.assertIn("model: gpt-6.1-sol", output.getvalue())
                self.assertIn("Source was not sent", errors.getvalue())
                send.assert_not_called()
                self.assertFalse(destination.exists())
                self.assertFalse(
                    destination.with_suffix(".report.json").exists()
                )
                self.assertFalse(ingest.call_args.kwargs["include_tests"])

    def test_short_command_writes_default_outputs_and_reapplies_annotations(
        self,
    ) -> None:
        """The one-argument command must generate reusable files in the current directory."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            previous = Path.cwd()
            provider = FixtureProvider()
            captured = snap({}, {"a.py": "x = 1\n"})
            os.chdir(root)
            try:
                with (
                    patch("diffstory.cli.from_github", return_value=captured),
                    patch(
                        "diffstory.cli._create_provider", return_value=provider
                    ),
                    patch(
                        "diffstory.cli.sys.argv",
                        ["diffstory", "owner/project#123", "--yes"],
                    ),
                    contextlib.redirect_stdout(io.StringIO()),
                ):
                    self.assertEqual(main(), 0)
                self.assertTrue((root / "walkthrough.html").is_file())
                self.assertTrue(
                    (root / "walkthrough.annotations.json").is_file()
                )
                report = json.loads(
                    (root / "walkthrough.report.json").read_text()
                )
                validate_walkthrough(report)
                self.assertEqual(provider.calls, 1)
                with (
                    patch("diffstory.cli.from_github", return_value=captured),
                    patch("diffstory.cli._create_provider") as create_provider,
                    contextlib.redirect_stdout(io.StringIO()),
                ):
                    self.assertEqual(
                        main(
                            [
                                "owner/project#123",
                                "--annotations",
                                "walkthrough.annotations.json",
                                "--out",
                                "reapplied.html",
                            ]
                        ),
                        0,
                    )
                create_provider.assert_not_called()
                self.assertEqual(
                    json.loads((root / "reapplied.report.json").read_text()),
                    report,
                )
            finally:
                os.chdir(previous)

    def test_basic_report_modes_do_not_set_up_a_provider(self) -> None:
        """Explicit GitHub opt-out and existing local defaults must stay local."""
        cases = [
            ["owner/project#123", "--no-narrate"],
            ["github", "owner/project#123", "--no-narrate"],
            ["git", "--base", "main"],
            ["snapshot"],
        ]
        for arguments in cases:
            with (
                self.subTest(arguments=arguments),
                tempfile.TemporaryDirectory() as directory,
            ):
                root = Path(directory)
                captured = snap({}, {"a.py": "x = 1\n"})
                command = list(arguments)
                if arguments == ["snapshot"]:
                    input_path = root / "input.json"
                    input_path.write_text(json.dumps(captured))
                    command.append(str(input_path))
                with (
                    patch("diffstory.cli.from_github", return_value=captured),
                    patch("diffstory.cli.from_git", return_value=captured),
                    patch("diffstory.cli._create_provider") as create_provider,
                    contextlib.redirect_stdout(io.StringIO()),
                ):
                    self.assertEqual(
                        main([*command, "--out", str(root / "basic.html")]),
                        0,
                    )
                create_provider.assert_not_called()
                report = json.loads((root / "basic.report.json").read_text())
                self.assertEqual(report["schema"], SCHEMA)

    def test_conflicting_modes_and_invalid_commands_stop_before_ingestion(
        self,
    ) -> None:
        """Shorthand must preserve option conflicts and normal parser errors."""
        cases = [
            (["owner/project#123", "--narrate", "--no-narrate"], True),
            (
                ["owner/project#123", "--narrate", "--annotations", "a.json"],
                False,
            ),
            (["not-a-command"], True),
        ]
        for arguments, parser_error in cases:
            with (
                self.subTest(arguments=arguments),
                patch("diffstory.cli.from_github") as ingest,
                patch("diffstory.cli._create_provider") as create_provider,
                contextlib.redirect_stderr(io.StringIO()),
            ):
                if parser_error:
                    with self.assertRaises(SystemExit) as raised:
                        main(arguments)
                    self.assertEqual(raised.exception.code, 2)
                else:
                    self.assertEqual(main(arguments), 2)
                ingest.assert_not_called()
                create_provider.assert_not_called()
