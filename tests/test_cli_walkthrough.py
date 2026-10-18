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
                            "--architecture",
                            "walkthrough",
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
                self.assertEqual(
                    main(
                        [
                            "snapshot",
                            str(snapshot),
                            "--narrate",
                            "--architecture",
                            "walkthrough",
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
                    with contextlib.redirect_stdout(io.StringIO()):
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
                            0,
                        )
                    archived_report = json.loads(
                        (root / f"{version[-2:]}.report.json").read_text()
                    )
                    self.assertEqual(
                        archived_report["generation"]["version"], version
                    )
                    validate_walkthrough(archived_report)


if __name__ == "__main__":
    unittest.main()
