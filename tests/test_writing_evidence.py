"""Check bounded writer projections against exact validated packet evidence."""

from __future__ import annotations

import copy
import hashlib
import unittest

from test_analysis import snap

from diffstory.analysis import compile_snapshot
from diffstory.compact_packet import build_packet
from diffstory.source_lines import split_source_lines
from diffstory.writing_evidence import WRITING_EVIDENCE_VERSION
from diffstory.writing_evidence import WritingEvidence


def _two_hunk_packet() -> dict:
    """Build one CRLF source pair with two far-apart changed definitions."""
    before = "".join(
        [
            "title = '雪'\r\n",
            "def first():\r\n",
            "    one = 1\r\n",
            "    note = 'café 🧬'\r\n",
            "    two = 2\r\n",
            "    three = 3\r\n",
            "    value = 'old'\r\n",
            "    four = 4\r\n",
            "    five = 5\r\n",
            "    six = 6\r\n",
            "    seven = 7\r\n",
            "    eight = 8\r\n",
            "    return value\r\n",
            "\r\n",
        ]
        + [f"# gap {index}\r\n" for index in range(12)]
        + [
            "\r\n",
            "def second():\r\n",
            "    other = 1\r\n",
            "    return other\r\n",
        ]
    )
    after = before.replace("value = 'old'", "value = 'new'").replace(
        "other = 1", "other = 2"
    )
    packet = build_packet(
        compile_snapshot(
            snap({"src/module.py": before}, {"src/module.py": after})
        )
    )
    packet["meta"]["captured_at"] = "2026-10-27T12:00:00Z"
    return packet


class WritingEvidenceTests(unittest.TestCase):
    """Keep writing inputs within their assigned, exact source evidence."""

    def test_selected_hunk_and_source_slice_preserve_unicode_and_crlf(
        self,
    ) -> None:
        """A selected full hunk maps to its exact original physical-line slice."""
        packet = _two_hunk_packet()
        selected_region = packet["regions"][0]
        diff = packet["diffs"][0]
        hunk_index = selected_region["hunk"]
        original_hunk = diff["hunks"][hunk_index]

        projection = WritingEvidence(packet).project([selected_region])

        self.assertEqual(projection["version"], WRITING_EVIDENCE_VERSION)
        self.assertNotIn("captured_at", projection["meta"])
        self.assertEqual(len(projection["diffs"]), 1)
        projected_diff = projection["diffs"][0]
        self.assertNotIn("original_patch", projected_diff)
        self.assertEqual(
            projected_diff["hunks"],
            [{**original_hunk, "original_index": hunk_index}],
        )

        for original_source in packet["sources"]:
            coordinates = [
                row[field]
                for row in original_hunk["rows"]
                for field in ("old", "new")
                if row[field] is not None
                and original_source["id"]
                == diff["base_source" if field == "old" else "head_source"]
            ]
            expected_start = min(coordinates)
            expected_end = max(coordinates)
            physical = split_source_lines(
                original_source["text"], keepends=True
            )
            expected_text = "".join(
                physical[
                    expected_start
                    - original_source["start_line"] : expected_end
                    - original_source["start_line"]
                    + 1
                ]
            )
            excerpt = next(
                source
                for source in projection["sources"]
                if source["id"] == original_source["id"]
            )
            self.assertEqual(excerpt["path"], original_source["path"])
            self.assertEqual(excerpt["revision"], original_source["revision"])
            self.assertEqual(excerpt["start_line"], expected_start)
            self.assertEqual(excerpt["scope"], "selected excerpts")
            self.assertEqual(excerpt["text"], expected_text)
            self.assertIn("\r\n", excerpt["text"])
            self.assertIn("café 🧬", excerpt["text"])
            self.assertEqual(
                excerpt["digest"],
                hashlib.sha256(expected_text.encode("utf-8")).hexdigest(),
            )

    def test_unassigned_hunk_patch_and_comparison_are_excluded(self) -> None:
        """A writer receives only assigned hunks, without duplicate patch text."""
        packet = _two_hunk_packet()
        selected_region = packet["regions"][0]
        projection = WritingEvidence(packet).project([selected_region])

        self.assertEqual(projection["regions"], [selected_region])
        self.assertEqual(
            [change["id"] for change in projection["changes"]],
            selected_region["change_ids"],
        )
        self.assertEqual(len(projection["diffs"][0]["hunks"]), 1)
        self.assertEqual(
            projection["diffs"][0]["hunks"][0]["original_index"], 0
        )
        self.assertNotIn("original_patch", projection["diffs"][0])
        self.assertTrue(
            all(
                "comparison_hunks" not in change
                for change in projection["changes"]
            )
        )
        self.assertTrue(
            all(
                "def second()" not in source["text"]
                for source in projection["sources"]
            )
        )

        reversed_regions = list(reversed(packet["regions"]))
        all_changes = WritingEvidence(packet).project(reversed_regions)[
            "changes"
        ]
        self.assertEqual(
            [change["id"] for change in all_changes],
            [change["id"] for change in packet["changes"]],
        )

    def test_source_only_region_projects_its_cited_ranges(self) -> None:
        """A region without captured diff hunks still supplies its citations."""
        text = "def added():\n    return 'π'\n"
        snapshot = snap({}, {"src/new.py": text})
        snapshot["patches"][0]["patch"] = None
        packet = build_packet(compile_snapshot(snapshot))
        region = next(
            region for region in packet["regions"] if region["diff_id"] is None
        )

        projection = WritingEvidence(packet).project([region])

        self.assertEqual(projection["diffs"], [])
        self.assertEqual(len(projection["sources"]), 1)
        self.assertEqual(projection["sources"][0]["side"], "head")
        self.assertEqual(projection["sources"][0]["start_line"], 1)
        self.assertEqual(projection["sources"][0]["text"], text)
        self.assertEqual(
            projection["changes"][0]["definition_coverage"], {"after": "full"}
        )

    def test_definition_coverage_tracks_partial_and_full_source_slices(
        self,
    ) -> None:
        """Definition metadata reports partial hunk context and full expansion."""
        packet = _two_hunk_packet()
        region = packet["regions"][0]
        original_change = packet["changes"][0]

        partial = WritingEvidence(packet).project([region])
        partial_change = partial["changes"][0]
        self.assertEqual(
            partial_change["definition_coverage"],
            {"before": "partial", "after": "partial"},
        )
        self.assertEqual(partial_change["before"], original_change["before"])
        self.assertEqual(partial_change["after"], original_change["after"])

        full = WritingEvidence(packet).project(
            [region], include_definitions=True
        )
        self.assertEqual(
            full["changes"][0]["definition_coverage"],
            {"before": "full", "after": "full"},
        )
        for side in ("before", "after"):
            reference = original_change[side]
            source = next(
                source
                for source in full["sources"]
                if source["id"] == reference["source_id"]
            )
            self.assertEqual(source["start_line"], reference["start"])
            self.assertEqual(
                source["text"],
                "".join(
                    split_source_lines(
                        next(
                            item
                            for item in packet["sources"]
                            if item["id"] == reference["source_id"]
                        )["text"],
                        keepends=True,
                    )[reference["start"] - 1 : reference["end"]]
                ),
            )

    def test_unselected_moved_definition_is_marked_not_supplied(self) -> None:
        """A selected deletion keeps its opposite definition metadata explicit."""
        before = {
            "users.py": "def normalize_name(name):\n    return name.lower()\n\nKEEP = 1\n",
            "text.py": "FORMAT = 'text'\n",
        }
        after = {
            "users.py": "KEEP = 1\n",
            "text.py": (
                "FORMAT = 'text'\n\ndef normalize_name(name):\n"
                "    return name.strip().lower()\n"
            ),
        }
        packet = build_packet(compile_snapshot(snap(before, after)))
        users_diff = next(
            diff for diff in packet["diffs"] if diff["path"] == "users.py"
        )
        region = next(
            region
            for region in packet["regions"]
            if region["diff_id"] == users_diff["id"]
        )

        projection = WritingEvidence(packet).project([region])
        moved = next(
            change
            for change in projection["changes"]
            if change["kind"] == "moved_modified"
        )

        self.assertEqual(
            moved["definition_coverage"],
            {"before": "full", "after": "not_supplied"},
        )
        self.assertEqual(
            moved["after"]["source_id"],
            packet["changes"][0]["after"]["source_id"],
        )

    def test_packet_and_region_inputs_remain_unchanged(self) -> None:
        """Projection changes cannot mutate the caller's packet or selection."""
        packet = _two_hunk_packet()
        original_packet = copy.deepcopy(packet)
        regions = [copy.deepcopy(packet["regions"][0])]
        original_regions = copy.deepcopy(regions)

        projection = WritingEvidence(packet).project(regions)
        projection["sources"][0]["text"] = "changed by caller"
        projection["regions"].clear()

        self.assertEqual(packet, original_packet)
        self.assertEqual(regions, original_regions)


if __name__ == "__main__":
    unittest.main()
