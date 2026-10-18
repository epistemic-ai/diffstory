"""Regression checks for source storage and complete changed-region accounting."""

import copy
import unittest

from test_analysis import snap

from diffstory.analysis import compile_snapshot
from diffstory.compact_packet import build_packet
from diffstory.compact_packet import packet_identity
from diffstory.compact_packet import resolve_citation
from diffstory.compact_packet import source_index
from diffstory.compact_packet import validate_packet


class CompactPacketTests(unittest.TestCase):
    """Keep source exact, changes reachable, and context out of writing work."""

    def test_changed_lines_remain_in_complete_diff_without_duplicated_text(
        self,
    ) -> None:
        """Raw edits outside definitions must remain accountable and accessible."""
        cases = [
            ("value = 1\n", "value = 2\n", 2),
            ("def f():\n    return 1\n", "def f():\n    return 2\n", 2),
            ("# before\n\n", "# after\n\n\n", 3),
        ]
        for before, after, changed_rows in cases:
            with self.subTest(before=before):
                packet = build_packet(
                    compile_snapshot(snap({"a.py": before}, {"a.py": after}))
                )
                rows = [
                    row
                    for diff in packet["diffs"]
                    for h in diff["hunks"]
                    for row in h["rows"]
                ]
                self.assertEqual(
                    sum(r["tag"] != "context" for r in rows), changed_rows
                )
                self.assertTrue(packet["regions"])
                self.assertTrue(all("text" not in row for row in rows))
                self.assertEqual(
                    [s["text"] for s in packet["sources"]], [before, after]
                )
                self.assertTrue(
                    all(
                        "source" not in (c.get(side) or {})
                        for c in packet["changes"]
                        for side in ("before", "after")
                    )
                )

    def test_unchanged_context_does_not_create_accounting_regions(
        self,
    ) -> None:
        """Supplying unchanged definitions must not force extra narration work."""
        cases = [("def stable():\n    return 1\n", 0), ("# unchanged\n", 0)]
        for source, expected in cases:
            with self.subTest(source=source):
                packet = build_packet(
                    compile_snapshot(snap({"a.py": source}, {"a.py": source}))
                )
                self.assertEqual(len(packet["regions"]), expected)
                self.assertEqual(len(packet["changes"]), expected)

    def test_original_coordinates_and_physical_newlines_resolve(self) -> None:
        """Unicode separators and CRLF must not shift original source lines."""
        cases = [
            ("x = 'one\u2028two'\r\ny = 2\r\n", 2, "y = 2\r\n"),
            ("x = 1\ry = 2\r", 2, "y = 2\r"),
        ]
        for source, line, expected in cases:
            with self.subTest(source=source):
                packet = build_packet(
                    compile_snapshot(snap({}, {"a.py": source}))
                )
                ref = {
                    "source_id": packet["sources"][0]["id"],
                    "revision": "b" * 40,
                    "start": line,
                    "end": line,
                }
                self.assertEqual(
                    resolve_citation(ref, source_index(packet)), expected
                )

    def test_wrong_revision_or_range_cannot_resolve(self) -> None:
        """A citation must never resolve different-revision or absent lines."""
        packet = build_packet(compile_snapshot(snap({}, {"a.py": "x = 1\n"})))
        base = {
            "source_id": packet["sources"][0]["id"],
            "revision": "b" * 40,
            "start": 1,
            "end": 1,
        }
        cases = [
            {"revision": "a" * 40},
            {"start": 0},
            {"end": 2},
            {"start": True},
            {"source_id": "missing"},
        ]
        for mutation in cases:
            with (
                self.subTest(mutation=mutation),
                self.assertRaises(ValueError),
            ):
                resolve_citation({**base, **mutation}, source_index(packet))

    def test_missing_region_or_modified_source_is_rejected(self) -> None:
        """Saved artifacts cannot omit accounting regions or alter source bytes."""
        original = build_packet(
            compile_snapshot(snap({}, {"a.py": "x = 1\n"}))
        )
        for mutation in ("region", "source", "revision"):
            with self.subTest(mutation=mutation):
                packet = copy.deepcopy(original)
                if mutation == "region":
                    packet["regions"].clear()
                elif mutation == "source":
                    packet["sources"][0]["text"] += "x = 2\n"
                else:
                    packet["sources"][0]["revision"] = "a" * 40
                with self.assertRaises(ValueError):
                    validate_packet(packet)

    def test_evidence_identity_ignores_capture_time_only(self) -> None:
        """Cache identity must change with source or author context, not capture time."""
        packet = build_packet(compile_snapshot(snap({}, {"a.py": "x = 1\n"})))
        cases = [
            ("captured_at", "tomorrow", True),
            ("description", "different intent", False),
            ("head_sha", "c" * 40, False),
        ]
        for key, value, equal in cases:
            with self.subTest(key=key):
                changed = copy.deepcopy(packet)
                changed["meta"][key] = value
                self.assertEqual(
                    packet_identity(packet) == packet_identity(changed), equal
                )

    def test_line_ending_edits_create_accounting_work(self) -> None:
        """Edits hidden by display-line normalization must remain explained."""
        for before, after in [("x = 1\n", "x = 1\r\n"), ("x = 1\n", "x = 1")]:
            with self.subTest(after=repr(after)):
                packet = build_packet(
                    compile_snapshot(snap({"a.py": before}, {"a.py": after}))
                )
                self.assertTrue(packet["regions"])
                self.assertEqual(
                    {c for r in packet["regions"] for c in r["change_ids"]},
                    {c["id"] for c in packet["changes"]},
                )

    def test_omitted_diff_and_changed_gap_are_rejected(self) -> None:
        """Deleting both a hunk and its accounting row must not conceal edits."""
        original = build_packet(
            compile_snapshot(snap({"a.py": "x = 1\n"}, {"a.py": "x = 2\n"}))
        )
        for target in ["hunks", "diffs"]:
            with self.subTest(target=target):
                packet = copy.deepcopy(original)
                if target == "hunks":
                    packet["diffs"][0]["hunks"].clear()
                else:
                    packet["diffs"].clear()
                packet["regions"].clear()
                with self.assertRaises(ValueError):
                    validate_packet(packet)

    def test_static_test_definitions_keep_original_source_without_extra_jobs(
        self,
    ) -> None:
        """Retaining static test evidence must not copy code or invent writing jobs."""
        source = "def test_value():\n    assert 1 == 1\n"
        packet = build_packet(
            compile_snapshot(snap({}, {"tests/test_value.py": source}))
        )
        test = packet["analysis"]["tests"][0]
        self.assertNotIn("text", test)
        self.assertEqual(
            resolve_citation(test["source"], source_index(packet)),
            source,
        )
        self.assertEqual(len(packet["regions"]), 1)

    def test_diff_sources_cannot_be_detached_from_supplied_buffers(
        self,
    ) -> None:
        """Removing diff buffers must not conceal edits behind source-only accounting."""
        for source_id in [None, "f" * 16]:
            with self.subTest(source_id=source_id):
                packet = build_packet(
                    compile_snapshot(
                        snap({"a.py": "x = 1\n"}, {"a.py": "x = 2\n"})
                    )
                )
                diff = packet["diffs"][0]
                diff.update(
                    base_source=source_id, head_source=source_id, hunks=[]
                )
                packet["regions"] = [
                    {
                        "id": f"change-{change['id']}",
                        "diff_id": None,
                        "hunk": None,
                        "change_ids": [change["id"]],
                        "citations": [
                            {
                                key: ref[key]
                                for key in (
                                    "source_id",
                                    "revision",
                                    "start",
                                    "end",
                                )
                            }
                            for ref in (change["before"], change["after"])
                            if ref
                        ],
                    }
                    for change in packet["changes"]
                ]
                with self.assertRaises(ValueError):
                    validate_packet(packet)

    def test_source_availability_cannot_contradict_embedded_buffers(
        self,
    ) -> None:
        """A saved packet must not relabel supplied source as absent or unread."""
        original = build_packet(
            compile_snapshot(snap({}, {"a.py": "x = 1\n"}))
        )
        for replacement in [
            {"state": "unavailable", "reason": "size_limit", "path": "a.py"},
            {"state": "supplied", "coverage": "full", "path": "other.py"},
        ]:
            with self.subTest(replacement=replacement):
                packet = copy.deepcopy(original)
                packet["file_evidence"][0]["head"] = replacement
                with self.assertRaises(ValueError):
                    validate_packet(packet)


if __name__ == "__main__":
    unittest.main()
