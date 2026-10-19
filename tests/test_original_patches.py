"""Keep original file patches distinct from generated function comparisons."""

import copy
import unittest
from unittest.mock import patch

import test_integration
from test_analysis import snap

from diffstory.analysis import compile_snapshot
from diffstory.compact_packet import build_packet
from diffstory.compact_packet import validate_packet
from diffstory.ingest import from_github


class OriginalPatchTests(unittest.TestCase):
    """Exercise original alignment, missing patches and the moved-function view."""

    def test_repeated_lines_keep_the_original_deleted_occurrence(self) -> None:
        """A second alignment must not invent an edit to another identical line."""
        source = snap({"README.md": "same\nsame\n"}, {"README.md": "same\n"})
        original = "@@ -1 +0,0 @@\n-same\n"
        source["patches"] = [
            {"path": "README.md", "origin": "github", "patch": original}
        ]
        packet = build_packet(compile_snapshot(source))
        rows = packet["diffs"][0]["hunks"][0]["rows"]
        self.assertEqual(rows, [{"tag": "delete", "old": 1, "new": None}])
        self.assertEqual(len(packet["changes"]), 1)
        self.assertEqual(len(packet["regions"]), 1)

    def test_original_grouping_survives_compile_and_packet_export(
        self,
    ) -> None:
        """A zero-context PR patch must not acquire Python's generated context."""
        source = snap(
            {"a.py": "def calculate():\n    return 1\n"},
            {"a.py": "def calculate():\n    return 2\n"},
        )
        original = "@@ -2 +2 @@\n-    return 1\n+    return 2\n"
        source["patches"] = [
            {"path": "a.py", "origin": "github", "patch": original}
        ]
        report = compile_snapshot(source)
        file_diff = report["raw_files"][0]
        self.assertEqual(file_diff["original_patch"], original)
        self.assertEqual(file_diff["hunks"][0]["old_start"], 2)
        self.assertEqual(
            [row["tag"] for row in file_diff["hunks"][0]["rows"]],
            ["delete", "add"],
        )
        packet = build_packet(report)
        self.assertEqual(packet["diffs"][0]["original_patch"], original)
        self.assertEqual(packet["diffs"][0]["hunks"][0]["old_start"], 2)
        function = next(
            change
            for change in packet["changes"]
            if change["kind"] == "modified"
        )
        self.assertEqual(function["comparison_hunks"][0]["old_start"], 1)
        altered = copy.deepcopy(packet)
        altered["diffs"][0]["hunks"] = function["comparison_hunks"]
        with self.assertRaisesRegex(ValueError, "alignment"):
            validate_packet(altered)
        altered = copy.deepcopy(packet)
        altered["diffs"][0]["patch_status"] = "generated"
        with self.assertRaises(ValueError):
            validate_packet(altered)

    def test_nonzero_partial_source_origin_keeps_captured_gap_validation(
        self,
    ) -> None:
        """A forged source origin must not shift an accepted row onto wrong text."""
        source = snap(
            {"a.py": "same\nsame\n"},
            {"a.py": "same\nchanged\n"},
        )
        source["patches"] = [
            {
                "path": "a.py",
                "origin": "github",
                "patch": "@@ -2 +2 @@\n-same\n+changed\n",
            }
        ]
        packet = build_packet(compile_snapshot(source))
        for stored_source in packet["sources"]:
            stored_source["start_line"] = 2
            stored_source["scope"] = "partial"
        for record in packet["file_evidence"]:
            for side in ("base", "head"):
                if record[side]["state"] == "supplied":
                    record[side]["coverage"] = "partial"
        packet["diffs"][0]["scope"] = "partial"

        with self.assertRaisesRegex(
            ValueError, "omitted changed source lines"
        ):
            validate_packet(packet)

    def test_github_patch_is_retained_or_explicitly_unavailable(self) -> None:
        """A missing API patch must never be replaced with generated file rows."""
        for supplied in ("@@ -0,0 +1 @@\n+x=1", None):
            with self.subTest(patch_supplied=supplied is not None):
                row = {"filename": "f0.py", "status": "added"}
                if supplied is not None:
                    row["patch"] = supplied
                fake, _ = test_integration.GitHubTransportTests().fake(
                    file_records=[row]
                )
                with patch("diffstory.ingest.GitHubClient.get", new=fake):
                    snapshot = from_github("org/repo#1")
                packet = build_packet(compile_snapshot(snapshot))
                diff = packet["diffs"][0]
                self.assertEqual(diff["original_patch"], supplied)
                self.assertEqual(
                    diff["patch_status"],
                    "captured" if supplied else "unavailable",
                )
                self.assertEqual(bool(diff["hunks"]), supplied is not None)
                self.assertTrue(packet["changes"])
                self.assertTrue(packet["regions"])

    def test_moved_function_keeps_both_original_files_and_extra_comparison(
        self,
    ) -> None:
        """The documented move-plus-strip example retains both kinds of view."""
        before = "def normalize_name(name):\n    return name.lower()\n"
        after = "def normalize_name(name):\n    return name.strip().lower()\n"
        source = snap(
            {
                "users.py": before + "\nKEEP = 1\n",
                "text.py": "FORMAT = 'text'\n",
            },
            {
                "users.py": "KEEP = 1\n",
                "text.py": "FORMAT = 'text'\n\n" + after,
            },
        )
        source["patches"] = [
            {
                "path": "users.py",
                "origin": "github",
                "patch": "@@ -1,3 +0,0 @@\n-def normalize_name(name):\n-    return name.lower()\n-\n",
            },
            {
                "path": "text.py",
                "origin": "github",
                "patch": "@@ -1,0 +2,3 @@\n+\n+def normalize_name(name):\n+    return name.strip().lower()\n",
            },
        ]
        packet = build_packet(compile_snapshot(source))
        self.assertEqual(
            {diff["path"] for diff in packet["diffs"]}, {"users.py", "text.py"}
        )
        function = next(
            change
            for change in packet["changes"]
            if change["before"]
            and change["after"]
            and change["before"]["name"] == "normalize_name"
        )
        self.assertEqual(
            [row["tag"] for row in function["comparison_hunks"][0]["rows"]],
            ["context", "delete", "add"],
        )

    def test_original_empty_ranges_and_newline_markers_survive(self) -> None:
        """Preserve Git's insert/delete coordinates and retained EOF markers."""
        cases = [
            ("a\nb\n", "a\nx\nb\n", "@@ -1,0 +2 @@\n+x\n"),
            ("a\nx\nb\n", "a\nb\n", "@@ -2 +1,0 @@\n-x\n"),
            (
                "x=1",
                "x=2",
                "@@ -1 +1 @@\n-x=1\n\\ No newline at end of file\n+x=2\n\\ No newline at end of file\n",
            ),
        ]
        for before, after, original in cases:
            with self.subTest(original=original):
                source = snap({"a.py": before}, {"a.py": after})
                source["patches"] = [
                    {"path": "a.py", "origin": "git", "patch": original}
                ]
                packet = build_packet(compile_snapshot(source))
                self.assertEqual(
                    packet["diffs"][0]["original_patch"], original
                )
                self.assertEqual(
                    packet["diffs"][0]["patch_status"], "captured"
                )

    def test_incomplete_or_wrong_source_patch_is_rejected(self) -> None:
        """Do not repair an incomplete or stale patch by quietly recomputing it."""
        for original in (
            "@@ -2 +2 @@\n-    return 1\n",
            "@@ -2 +2 @@\n-    return 99\n+    return 2\n",
            "",
        ):
            with self.subTest(original=original):
                source = snap(
                    {"a.py": "def f():\n    return 1\n"},
                    {"a.py": "def f():\n    return 2\n"},
                )
                source["patches"] = [
                    {"path": "a.py", "origin": "github", "patch": original}
                ]
                with self.assertRaises(ValueError):
                    compile_snapshot(source)

    def test_missing_patch_metadata_is_rejected(self) -> None:
        """Old snapshots and packets must fail instead of rebuilding a file diff."""
        source = snap({"a.py": "x=1\n"}, {"a.py": "x=2\n"})
        packet = build_packet(compile_snapshot(source))
        source.pop("patches")
        with self.assertRaisesRegex(ValueError, "patches"):
            compile_snapshot(source)
        source["patches"] = []
        with self.assertRaisesRegex(ValueError, "matching original patch"):
            compile_snapshot(source)
        for field in ("diff_origin", "patch_status", "original_patch"):
            with self.subTest(field=field):
                altered = copy.deepcopy(packet)
                altered["diffs"][0].pop(field)
                with self.assertRaisesRegex(ValueError, field):
                    validate_packet(altered)

    def test_repeated_original_patch_paths_are_rejected(self) -> None:
        """Reject ambiguous source-to-patch ownership before compilation."""
        source = snap({"a.py": "x = 1\n"}, {"a.py": "x = 2\n"})
        source["patches"].append(copy.deepcopy(source["patches"][0]))
        with self.assertRaisesRegex(ValueError, "paths must be unique"):
            compile_snapshot(source)

    def test_original_patch_text_must_encode_as_utf8(self) -> None:
        """Reject lone surrogates instead of failing later during byte accounting."""
        source = snap({"a.py": "x = 1\n"}, {"a.py": "x = 2\n"})
        source["patches"][0]["patch"] = "\ud800"
        with self.assertRaisesRegex(ValueError, "valid UTF-8"):
            compile_snapshot(source)

    def test_aggregate_original_patch_bytes_are_bounded(self) -> None:
        """Accept the exact aggregate patch ceiling and reject one byte below it."""
        source = snap(
            {"a.py": "a", "b.py": "b"},
            {"a.py": "c", "b.py": "d"},
        )
        patch_bytes = sum(
            len((record["patch"] or "").encode("utf-8"))
            for record in source["patches"]
        )
        source_bytes = sum(
            len(fragment["text"].encode("utf-8"))
            for fragment in source["fragments"]
        )
        cases = [
            ("exact ceiling", patch_bytes, False),
            ("one byte below", patch_bytes - 1, True),
        ]
        self.assertLess(source_bytes, patch_bytes - 1)
        for name, limit, rejected in cases:
            with (
                self.subTest(case=name),
                patch("diffstory.models.MAX_SNAPSHOT_SOURCE_BYTES", limit),
            ):
                if rejected:
                    with self.assertRaisesRegex(
                        ValueError, "aggregate byte limit"
                    ):
                        compile_snapshot(copy.deepcopy(source))
                else:
                    compile_snapshot(copy.deepcopy(source))


if __name__ == "__main__":
    unittest.main()
