"""Contracts for the continuous reader's source-bound paragraph annotations."""

import copy
import unittest

from diffstory.analysis import apply_annotations
from diffstory.analysis import compile_snapshot
from diffstory.analysis import evidence_packet
from diffstory.models import MAX_PREAMBLE_CHARS
from diffstory.render import render


def report() -> dict:
    """
    Compile a report containing two source definitions at known lines.

    Returns:
        A ``diffstory.report.v1`` mapping for passage validation.

    """
    return compile_snapshot(
        {
            "schema": "diffstory.snapshot.v1",
            "meta": {
                "base_sha": "a" * 40,
                "head_sha": "b" * 40,
                "scope": "full changed files",
                "changed_files": 2,
            },
            "fragments": [
                {
                    "path": "query.py",
                    "side": "head",
                    "start_line": 20,
                    "text": "def construct_query(x):\n    value = x + 1\n    return value\n",
                    "scope": "complete",
                },
                {
                    "path": "parse.py",
                    "side": "head",
                    "start_line": 1,
                    "text": "def parse_record(x):\n    return str(x)\n",
                    "scope": "complete",
                },
            ],
        }
    )


def annotations(r: dict) -> dict:
    """
    Build revision-bound annotations for the first report group.

    Args:
        r: Compiled report with at least one group and change.

    Returns:
        Annotation mapping with one source-bound definition passage.

    """
    g = r["groups"][0]
    return {
        "schema": "diffstory.annotations.v1",
        "base_sha": r["meta"]["base_sha"],
        "head_sha": r["meta"]["head_sha"],
        "document": {
            "lead": "Read the code.",
            "closing": "Trace the callers.",
        },
        "steps": [
            {
                "group_id": g["id"],
                "evidence_change_ids": g["change_ids"],
                "transition": "Now follow the next boundary.",
                "passages": [
                    {
                        "text": "Read `this` exact definition.",
                        "change_ids": [g["change_ids"][0]],
                        "view": "definition",
                    }
                ],
            }
        ],
    }


class LiterateReaderTests(unittest.TestCase):
    """Check source-bound narration, evidence, and safe HTML rendering."""

    def test_passages_roundtrip(self) -> None:
        """Apply document and passage annotations without changing deterministic report facts."""
        r = report()
        a = annotations(r)
        out = apply_annotations(r, a)
        self.assertEqual(out["document"], a["document"])
        self.assertEqual(
            out["groups"][0]["narrative"]["passages"],
            a["steps"][0]["passages"],
        )
        self.assertNotIn("preamble", out["document"])
        self.assertEqual(out["changes"], r["changes"])
        self.assertEqual(out["stats"], r["stats"])
        self.assertNotIn("document", r)

    def test_valid_focus_uses_original_lines(self) -> None:
        """Accept a focus range within the cited definition using its original line numbers."""
        r = report()
        a = annotations(r)
        c = next(
            c
            for c in r["changes"]
            if c["id"] == a["steps"][0]["passages"][0]["change_ids"][0]
        )
        s = c["after"] or c["before"]
        a["steps"][0]["passages"][0]["focus"] = {
            "start": s["start"] + 1,
            "end": s["end"],
        }
        out = apply_annotations(r, a)
        self.assertIn("reader", render(out).lower())

    def test_focus_out_of_source_rejected(self) -> None:
        """Reject a focus range outside the supplied source definition."""
        r = report()
        a = annotations(r)
        a["steps"][0]["passages"][0]["focus"] = {"start": 999, "end": 1000}
        with self.assertRaisesRegex(ValueError, "outside its source"):
            apply_annotations(r, a)

    def test_focus_boolean_is_not_integer(self) -> None:
        """Reject boolean line values even though booleans are integer subclasses."""
        r = report()
        a = annotations(r)
        a["steps"][0]["passages"][0]["focus"] = {"start": True, "end": 3}
        with self.assertRaisesRegex(ValueError, "outside its source"):
            apply_annotations(r, a)

    def test_focus_not_allowed_for_diff(self) -> None:
        """Reject focused ranges when the passage displays a diff."""
        r = report()
        a = annotations(r)
        a["steps"][0]["passages"][0].update(
            view="diff", focus={"start": 1, "end": 2}
        )
        with self.assertRaisesRegex(ValueError, "one definition"):
            apply_annotations(r, a)

    def test_passage_cannot_cite_other_group(self) -> None:
        """Reject passage evidence owned by a different narrative group."""
        r = report()
        a = annotations(r)
        a["steps"][0]["passages"][0]["change_ids"] = r["groups"][1][
            "change_ids"
        ]
        with self.assertRaisesRegex(ValueError, "outside its group"):
            apply_annotations(r, a)

    def test_duplicate_ids_in_passage_rejected(self) -> None:
        """Reject duplicate change IDs within one passage."""
        r = report()
        a = annotations(r)
        a["steps"][0]["passages"][0]["change_ids"] *= 2
        with self.assertRaisesRegex(ValueError, "duplicated"):
            apply_annotations(r, a)

    def test_same_definition_can_support_two_passages(self) -> None:
        """Allow separate passages to cite the same definition."""
        r = report()
        a = annotations(r)
        a["steps"][0]["passages"].append(
            copy.deepcopy(a["steps"][0]["passages"][0])
        )
        out = apply_annotations(r, a)
        self.assertEqual(len(out["groups"][0]["narrative"]["passages"]), 2)

    def test_passage_view_rejected(self) -> None:
        """Reject unsupported passage view values."""
        r = report()
        a = annotations(r)
        a["steps"][0]["passages"][0]["view"] = "<script>"
        with self.assertRaisesRegex(ValueError, "view"):
            apply_annotations(r, a)

    def test_empty_passage_text_rejected(self) -> None:
        """Reject empty or whitespace-only passage prose."""
        r = report()
        a = annotations(r)
        a["steps"][0]["passages"][0]["text"] = "  "
        with self.assertRaisesRegex(ValueError, "text"):
            apply_annotations(r, a)

    def test_bad_document_rejected(self) -> None:
        """Reject document fields outside the supported narration contract."""
        r = report()
        a = annotations(r)
        a["document"]["test_runs"] = 100
        with self.assertRaisesRegex(ValueError, "document"):
            apply_annotations(r, a)

    def test_reader_is_one_document_not_view_tabs(self) -> None:
        """Render a continuous reader with lazy source loading and no tabs or pager."""
        h = render(apply_annotations(report(), annotations(report())))
        self.assertIn('id="story"', h)
        self.assertNotIn('role="tablist"', h)
        self.assertNotIn('class="sidebar"', h)
        self.assertNotIn("Next →", h)
        self.assertIn("IntersectionObserver", h)
        self.assertIn("Load full diff", h)

    def test_malicious_prose_embedded_as_data(self) -> None:
        """Escape malicious narration as embedded data under a no-network policy."""
        r = report()
        a = annotations(r)
        attack = "</script><script>window.COMPROMISED=true</script>"
        a["document"]["lead"] = attack
        a["document"]["preamble"] = attack
        a["steps"][0]["passages"][0]["text"] = attack
        h = render(apply_annotations(r, a))
        self.assertNotIn(attack, h)
        self.assertIn("\\u003c/script\\u003e", h)
        self.assertIn("connect-src 'none'", h)

    def test_optional_preamble_is_preserved_and_must_contain_text(
        self,
    ) -> None:
        """Keep an optional authored preamble and reject a blank supplied value."""
        r = report()
        a = annotations(r)
        a["document"]["preamble"] = (
            "This change joins the report to its source.\n"
            "Map: report → sections → code."
        )
        out = apply_annotations(r, a)
        self.assertEqual(
            out["document"]["preamble"], a["document"]["preamble"]
        )
        self.assertIn("\\nMap: report", render(out))

        a["document"]["preamble"] = "  "
        with self.assertRaisesRegex(ValueError, "preamble"):
            apply_annotations(r, a)

    def test_authored_preamble_limit_applies_to_annotations_and_saved_reports(
        self,
    ) -> None:
        """Reject an authored preamble above its limit at both input boundaries."""
        source_report = report()
        for length, valid in (
            (MAX_PREAMBLE_CHARS, True),
            (MAX_PREAMBLE_CHARS + 1, False),
        ):
            with self.subTest(length=length):
                preamble = "x" * length
                candidate = annotations(source_report)
                candidate["document"]["preamble"] = preamble
                saved = apply_annotations(
                    source_report, annotations(source_report)
                )
                saved["document"]["preamble"] = preamble
                if valid:
                    apply_annotations(source_report, candidate)
                    render(saved)
                else:
                    with self.assertRaises(ValueError):
                        apply_annotations(source_report, candidate)
                    with self.assertRaises(ValueError):
                        render(saved)

    def test_renderer_validates_loaded_report_passages(self) -> None:
        """Validate persisted passage evidence again when loading a report for rendering."""
        r = apply_annotations(report(), annotations(report()))
        r["groups"][0]["narrative"]["passages"][0]["change_ids"] = ["f" * 16]
        with self.assertRaisesRegex(ValueError, "outside its group"):
            render(r)

    def test_evidence_packet_explains_passages(self) -> None:
        """Describe passage citations, original lines, and constraints in evidence."""
        i = evidence_packet(report())["instructions"]
        self.assertIn("passages", i)
        self.assertIn("original line numbers", i)
        self.assertIn("Do not rewrite source", i)

    def test_report_string_lists_render(self) -> None:
        """Notes and warnings are string lists, while old reports may omit notes."""
        compiled = compile_snapshot(
            {
                "schema": "diffstory.snapshot.v1",
                "meta": {"changed_files": 2},
                "fragments": [
                    {"path": "README.md", "side": "head", "text": "A note.\n"},
                    {
                        "path": "broken.py",
                        "side": "head",
                        "text": "def broken(:\n",
                    },
                ],
                "warnings": ["Incoming legacy warning."],
            },
        )
        old_report = copy.deepcopy(compiled)
        old_report.pop("notes")
        malformed_warnings = copy.deepcopy(compiled)
        malformed_warnings["warnings"] = [{"message": "not a string"}]
        malformed_notes = copy.deepcopy(compiled)
        malformed_notes["notes"] = [None]
        rows = [
            ("new report", compiled, True),
            ("old report", old_report, True),
            ("object warning", malformed_warnings, False),
            ("non-string note", malformed_notes, False),
        ]
        for name, candidate, valid in rows:
            with self.subTest(case=name):
                if valid:
                    render(candidate)
                else:
                    with self.assertRaisesRegex(ValueError, "list of strings"):
                        render(candidate)

    def test_annotations_preserve_kind_and_basis(self) -> None:
        """Authored prose must not change the compiler's file status or basis."""
        source_report = report()
        before = copy.deepcopy(source_report["changes"])
        annotated = apply_annotations(
            source_report, annotations(source_report)
        )
        self.assertEqual(annotated["changes"], before)
        self.assertTrue(
            all(change["basis"] for change in annotated["changes"])
        )

    def test_stale_change_ids_are_rejected(self) -> None:
        """Annotations citing IDs from a corrected kind must fail validation."""
        source = "def f():\n    return 1\n"
        old_report = compile_snapshot(
            {
                "schema": "diffstory.snapshot.v1",
                "meta": {"base_sha": "a" * 40, "head_sha": "b" * 40},
                "fragments": [
                    {
                        "path": "x.py",
                        "side": "head",
                        "text": source,
                        "start_line": 1,
                        "scope": "full",
                    },
                ],
            },
        )
        new_report = compile_snapshot(
            {
                "schema": "diffstory.snapshot.v1",
                "meta": {"base_sha": "a" * 40, "head_sha": "b" * 40},
                "fragments": [
                    {
                        "path": "x.py",
                        "side": "head",
                        "text": source,
                        "start_line": 1,
                        "scope": "full",
                    },
                ],
                "file_evidence": [
                    {
                        "base": {"path": "x.py", "state": "absent"},
                        "head": {
                            "path": "x.py",
                            "state": "supplied",
                            "coverage": "full",
                        },
                    },
                ],
            },
        )
        stale_id = old_report["changes"][0]["id"]
        current = annotations(new_report)
        current["steps"][0]["evidence_change_ids"] = [stale_id]
        current["steps"][0]["passages"][0]["change_ids"] = [stale_id]
        self.assertNotEqual(stale_id, new_report["changes"][0]["id"])
        with self.assertRaisesRegex(ValueError, "outside its group"):
            apply_annotations(new_report, current)


if __name__ == "__main__":
    unittest.main()
