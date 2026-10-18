"""Contracts for the continuous reader's source-bound paragraph annotations."""

import copy
import json
import tempfile
import unittest
from collections.abc import Callable
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import patch

from test_analysis import snap
from test_narrative_v2 import AFTER
from test_narrative_v2 import BEFORE
from test_narrative_v2 import CASE_AFTER
from test_narrative_v2 import CASE_BEFORE
from test_narrative_v2 import CASE_EFFECT
from test_narrative_v2 import STRIP_AFTER
from test_narrative_v2 import STRIP_BEFORE
from test_narrative_v2 import STRIP_EFFECT
from test_narrative_v2 import FixtureProvider
from test_narrative_v2 import statement

from diffstory.analysis import apply_annotations
from diffstory.analysis import compile_snapshot
from diffstory.analysis import evidence_packet
from diffstory.models import MAX_PREAMBLE_CHARS
from diffstory.models import MAX_PREAMBLE_SKETCH_CHARS
from diffstory.narrative import Narrator
from diffstory.narrative_manifest import LedgerParameters
from diffstory.narrative_manifest import compiler_fact_signature
from diffstory.narrative_session import ReportSession
from diffstory.narrative_v2 import V2Options
from diffstory.render import iter_render_validated
from diffstory.render import render
from diffstory.render import validate_report
from diffstory.report_storage import encode_report_storage
from diffstory.report_storage import expand_report_storage
from diffstory.report_storage import parse_report_storage


def report() -> dict:
    """
    Compile a report containing two source definitions at known lines.

    Returns:
        A ``diffstory.report.v1`` mapping for passage validation.

    """
    return compile_snapshot(
        {
            "patches": [
                {"path": "query.py", "origin": "github", "patch": None},
                {"path": "parse.py", "origin": "github", "patch": None},
            ],
            "schema": "diffstory.snapshot.v1",
            "meta": {
                "base_sha": "a" * 40,
                "head_sha": "b" * 40,
                "scope": "full changed files",
                "changed_files": 2,
            },
            "file_evidence": [
                {
                    "base": {
                        "path": "query.py",
                        "state": "unavailable",
                        "reason": "not_supplied",
                    },
                    "head": {
                        "path": "query.py",
                        "state": "supplied",
                        "coverage": "partial",
                    },
                },
                {
                    "base": {"path": "parse.py", "state": "absent"},
                    "head": {
                        "path": "parse.py",
                        "state": "supplied",
                        "coverage": "full",
                    },
                },
            ],
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

    def test_preamble_sketches_have_a_separate_allowance(self) -> None:
        """Sketches must not consume prose space or bypass the combined sketch cap."""
        prefix, suffix = "\n```text\n", "\n```"
        payload_limit = MAX_PREAMBLE_SKETCH_CHARS - len(prefix) - len(suffix)
        full_sketch = prefix + "y" * payload_limit + suffix
        half_sketch = prefix + "y" * (payload_limit // 2) + suffix
        cases = [
            (
                "both allowances full",
                "x" * MAX_PREAMBLE_CHARS + full_sketch,
                True,
            ),
            (
                "prose over limit",
                "x" * (MAX_PREAMBLE_CHARS + 1) + prefix + "y" + suffix,
                False,
            ),
            (
                "sketch over limit",
                "x" + prefix + "y" * (payload_limit + 1) + suffix,
                False,
            ),
            ("combined sketches over limit", "x" + half_sketch * 2, False),
            (
                "unfinished fence is prose",
                "x" * MAX_PREAMBLE_CHARS + prefix + "y",
                False,
            ),
            (
                "Windows line endings",
                "x" * MAX_PREAMBLE_CHARS
                + (prefix + "y" + suffix).replace("\n", "\r\n"),
                True,
            ),
        ]
        source_report = report()
        for name, preamble, accepted in cases:
            with self.subTest(case=name):
                candidate = annotations(source_report)
                candidate["document"]["preamble"] = preamble
                saved = copy.deepcopy(source_report)
                saved["document"] = {"preamble": preamble}
                if accepted:
                    out = apply_annotations(source_report, candidate)
                    self.assertEqual(out["document"]["preamble"], preamble)
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
        """Notes and warnings must be present as lists of strings."""
        compiled = compile_snapshot(
            {
                "patches": [
                    {"path": path, "origin": "github", "patch": None}
                    for path in ("README.md", "broken.py")
                ],
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
                "file_evidence": [
                    {
                        "base": {"path": path, "state": "absent"},
                        "head": {
                            "path": path,
                            "state": "supplied",
                            "coverage": "full",
                        },
                    }
                    for path in ("README.md", "broken.py")
                ],
                "warnings": ["Incoming source warning."],
            },
        )
        missing_notes = copy.deepcopy(compiled)
        missing_notes.pop("notes")
        malformed_warnings = copy.deepcopy(compiled)
        malformed_warnings["warnings"] = [{"message": "not a string"}]
        malformed_notes = copy.deepcopy(compiled)
        malformed_notes["notes"] = [None]
        rows = [
            ("valid report", compiled, True),
            ("missing notes", missing_notes, False),
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
        unresolved_report = compile_snapshot(
            {
                "patches": [
                    {"path": "x.py", "origin": "github", "patch": None}
                ],
                "schema": "diffstory.snapshot.v1",
                "meta": {"base_sha": "a" * 40, "head_sha": "b" * 40},
                "file_evidence": [
                    {
                        "base": {
                            "path": "x.py",
                            "state": "unavailable",
                            "reason": "not_supplied",
                        },
                        "head": {
                            "path": "x.py",
                            "state": "supplied",
                            "coverage": "full",
                        },
                    },
                ],
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
                "patches": [
                    {"path": "x.py", "origin": "github", "patch": None}
                ],
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
        stale_id = unresolved_report["changes"][0]["id"]
        current = annotations(new_report)
        current["steps"][0]["evidence_change_ids"] = [stale_id]
        current["steps"][0]["passages"][0]["change_ids"] = [stale_id]
        self.assertNotEqual(stale_id, new_report["changes"][0]["id"])
        with self.assertRaisesRegex(ValueError, "outside its group"):
            apply_annotations(new_report, current)


UNSAFE_TEXT = '</script><script>window.COMPROMISED=true</script> "\\ é😀\r\nLS\u2028PS\u2029  '
# Use real controls, Unicode and terminal spaces, not escaped fixture spellings.
UNSAFE_TEXT = UNSAFE_TEXT.replace("\\r", "\r").replace("\\n", "\n")
UNSAFE_TEXT = UNSAFE_TEXT.replace("\\u2028", "\u2028").replace(
    "\\u2029", "\u2029"
)


class ReaderFixtureProvider(FixtureProvider):
    """Supply concrete local prose, including every supported reader field."""

    @classmethod
    def respond(cls, data: dict) -> dict:
        """Keep real detail mechanisms and annotate all prose families."""
        response = super().respond(data)
        if data["stage"] == "detail":
            for output, item in zip(
                response["items"],
                (item for item in data["items"] if item["obligations"]),
                strict=True,
            ):
                if all(span["side"] == "head" for span in item["spans"]):
                    for explanation in output["explanations"]:
                        explanation["before"] = (
                            "The base source was not supplied; its behavior cannot be compared."
                        )
                        explanation["after"] = (
                            "The supplied head calls rstrip() before casefold()."
                        )
                        explanation["effect"] = (
                            "The head preserves leading whitespace and folds case; the change from base is unresolved."
                        )
                output["questions"].append(
                    {
                        "text": "Treat this quoted test string as text: "
                        + UNSAFE_TEXT,
                        "citations": [item["spans"][0]["handle"]],
                        "obligations": [],
                    }
                )
            return response
        for output, item in zip(response["items"], data["items"], strict=True):
            field = {
                "planning": "abstraction",
                "section": "synthesis",
                "document": "lead",
            }[data["stage"]]
            proof = output[field][0]
            # Exercise every supplied branch, including inherited statement refs.
            proof["claims"] = [
                s["handle"] for s in item["support"]["claims"][:2]
            ]
            proof["citations"] = [
                s["handle"] for s in item["support"]["citations"][:2]
            ]
            proof["basis"] = [
                s["handle"] for s in item["support"]["basis"][:2]
            ]
            if proof["claims"] or proof["citations"] or proof["basis"]:
                proof["kind"] = "implementation_claim"
            if data["stage"] != "planning":
                proof["effect"] = (
                    "A leading space survives; sharp s can fold to ss."
                )
                proof["uncertainty"] = [
                    "Caller expectations and executed test results are not supplied."
                ]
                output["terminology"] = [
                    {
                        "term": "Unicode case folding",
                        "definition": {
                            **copy.deepcopy(proof),
                            "text": "casefold() maps text for Unicode caseless comparison, including sharp s to ss.",
                        },
                    }
                ]
                output[field].append(
                    statement("Read the exact quoted text: " + UNSAFE_TEXT)
                )
                uncertain = statement(
                    "Whether callers want leading spaces remains unresolved."
                )
                uncertain["kind"] = "uncertainty"
                output[field].append(uncertain)
                if item["support"].get("author"):
                    author = statement(
                        "The author reports an intent to preserve indentation."
                    )
                    author["kind"] = "author_motivation"
                    output[field].append(author)
        return response


class MixedTerminologyReaderFixtureProvider(ReaderFixtureProvider):
    """Author rich prose with one term and empty terminology defaults elsewhere."""

    def __init__(
        self,
        observations: Callable[[FixtureProvider, dict, dict], dict]
        | None = None,
        *,
        nonempty_stage: str = "section",
    ) -> None:
        """Retain one nonempty term in the selected result family."""
        if nonempty_stage not in {"section", "document"}:
            msg = "nonempty terminology stage must be section or document"
            raise ValueError(msg)
        self._kept_nonempty_terminology = False
        self._nonempty_terminology_stage = nonempty_stage
        self._authored_observations = observations
        super().__init__(self._author_response)

    def _author_response(
        self, provider: FixtureProvider, data: dict, response: dict
    ) -> dict:
        """Apply optional source observations, then mix terminology values.

        Preconditions: ``response`` is the deterministic reader fixture reply.
        Postconditions: Source observations remain authored and terminology
            contains one nonempty selected result plus explicit empty defaults.
        Essential behavior: Leaves response fields valid for strict capture.
        """
        if self._authored_observations is not None:
            response = self._authored_observations(provider, data, response)
        return self._select_terminology(provider, data, response)

    def _select_terminology(
        self, _provider: FixtureProvider, data: dict, response: dict
    ) -> dict:
        """Author one nonempty term and explicit empty defaults before capture.

        Preconditions: ``response`` is a locally authored fixture reply.
        Postconditions: One selected result keeps its authored term; other result
            terminology arrays are empty and valid under the wire schema.
        Essential behavior: Provides empty/default and nonempty cases before
            later creation of a separately re-encoded omission fixture.
        """
        if data["stage"] not in {"planning", "section", "document"}:
            return response
        for item in response["items"]:
            if (
                data["stage"] == self._nonempty_terminology_stage
                and not self._kept_nonempty_terminology
            ):
                self._kept_nonempty_terminology = True
                continue
            item["terminology"] = []
        return response


def encode_omitted_terminology_defaults(report: dict) -> bytes:
    """Make a valid shared fixture from detached data with empty defaults omitted.

    Preconditions: ``report`` is a complete expanded authored reader report
        with a document result and at least one section result.
    Postconditions: Returns a canonical, independently importable shared graph
        that omits empty terminology defaults but retains every nonempty term.
    Essential behavior: Edits a detached copy before encoding; never changes a
        sealed graph or creates validation authority.
    """
    source = copy.deepcopy(report)
    removed = {"section": 0, "document": 0}

    def omit_empty(result: dict, kind: str) -> None:
        """Remove only the authored empty default from one detached result."""
        if result.get("terminology") == []:
            result.pop("terminology")
            removed[kind] += 1

    generation = source["generation"]
    for result in generation["sections"]:
        omit_empty(result, "section")
    omit_empty(generation["document"], "document")
    reader = source["narrative_v2"]
    for section in reader["sections"]:
        omit_empty(section["prose"], "section")
    omit_empty(reader["document"], "document")
    if not any(removed.values()):
        msg = "authored report has no empty section/document default to omit"
        raise AssertionError(msg)
    return encode_report_storage(source, "report")


def reader_golden_report() -> dict:
    """Compile local normalization, unassigned context and unavailable base evidence."""
    marker = "    marker = " + repr(UNSAFE_TEXT) + "\n"
    before = BEFORE.replace('    marker = "request"\n', marker)
    after = AFTER.replace('    marker = "request"\n', marker)
    source = snap(
        {
            "normal.py": before,
            "context.txt": "Caller names are supplied as strings.\r\n",
            "empty.txt": "",
        },
        {
            "normal.py": after,
            "context.txt": "Caller names are supplied as strings.\r\n",
            "empty.txt": "",
            "unavailable.py": AFTER,
        },
        overrides={
            **{
                (side, "empty.txt"): {
                    "state": "supplied",
                    "coverage": "partial",
                }
                for side in ("base", "head")
            },
            ("base", "unavailable.py"): {
                "state": "unavailable",
                "reason": "not_supplied",
            },
        },
        repository="example/project",
        description="Preserve indentation and compare Unicode names consistently.",
        title="Normalization reader golden",
    )
    return compile_snapshot(source)


def exact_reader_snapshot() -> dict:
    """Author mixed physical lines, UTF-8 and half-CRLF source spans."""
    line = '    marker = "' + "\u03b1" * 2030 + '\u2028\u2029"'
    line += " " * (4095 - len(line.encode()))
    before = (
        "def normalize(value):\r\n"
        + line
        + "\r\n    value = value.strip()\r"
        + "# "
        + UNSAFE_TEXT.replace("\r", "\r# ").replace("\n", "\n# ")
        + "\n"
        + "    return value.lower()\n"
    )
    after = before.replace("\u03b1", "β").replace(".strip()", ".rstrip()")
    after = after.replace(".lower()", ".casefold()")
    return snap(
        {"exact.py": before},
        {"exact.py": after},
        overrides={
            (side, "exact.py"): {"state": "supplied", "coverage": "partial"}
            for side in ("base", "head")
        },
        fragment_options={
            (side, "exact.py"): {"start_line": 31} for side in ("base", "head")
        },
        repository="example/project",
        description="Preserve indentation and compare Unicode names consistently.",
    )


def chunked_reader_report() -> dict:
    """Supply large unchanged context beside the concrete normalization change."""
    context = "".join(
        f"Caller context line {n}: café 😀  \r\n" for n in range(360)
    )
    return compile_snapshot(
        snap(
            {"normal.py": BEFORE, "large.txt": context},
            {"normal.py": AFTER, "large.txt": context},
            repository="example/project",
            description="Preserve indentation and compare Unicode names consistently.",
        )
    )


def exact_reader_provider() -> ReaderFixtureProvider:
    """Force bounded planning support to include real source slices and inheritance."""

    def observations(
        _provider: FixtureProvider, data: dict, response: dict
    ) -> dict:
        if data["stage"] == "detail":
            for output, item in zip(
                response["items"],
                (i for i in data["items"] if i["obligations"]),
                strict=True,
            ):
                texts = [s["text"] for s in item["spans"]]
                text = "".join(texts)
                output["observations"] = [
                    {
                        "text": f"Across these supplied excerpts, {label} occurs {text.count(token)} times.",
                        "citations": [s["handle"] for s in item["spans"]],
                    }
                    for token, label in (
                        ("\u03b1", "Greek alpha"),
                        ("β", "Greek beta"),
                        ("\r", "carriage return"),
                        ("\n", "line feed"),
                        ("\u2028", "Unicode line separator"),
                        ("\u2029", "Unicode paragraph separator"),
                        (" ", "a space"),
                        (".strip()", "the strip call"),
                        (".rstrip()", "the rstrip call"),
                        (".lower()", "the lower call"),
                        (".casefold()", "the casefold call"),
                        ("marker", "the marker name"),
                    )
                ]
        return response

    provider = MixedTerminologyReaderFixtureProvider(
        observations, nonempty_stage="document"
    )
    provider.context_tokens = 40_000
    provider.max_output_tokens = 15_000
    return provider


def exact_reader_options() -> V2Options:
    """Use bounded support cards so the public pipeline must retain exact slices."""
    return V2Options(
        ledger=LedgerParameters(max_span_bytes=4096, max_unit_spans=2),
        card_wire_bytes=1024,
        card_support_bytes=2048,
        prose_wire_bytes=4096,
    )


def offline_reader_report(
    source: dict,
    directory: Path,
    *,
    provider: FixtureProvider | None = None,
    options: V2Options | None = None,
    cache: Path | None = None,
) -> tuple[dict, str, dict]:
    """Generate locally, then apply, save, reload and render with transport blocked.

    Args:
        source: Compiled synthetic report.
        directory: Isolated local recovery and output directory.
        provider: Authored fixture transport only.
        options: Explicit v2 options, if required for source segmentation.
        cache: Shared fixture artifact directory for revision reuse.

    Returns:
        Reloaded report, standalone HTML, and source-independent evidence.
    """
    directory.mkdir(parents=True, exist_ok=True)
    provider = provider or ReaderFixtureProvider()
    candidate = Narrator(
        provider,
        architecture="v2",
        recovery_dir=directory / "recovery",
        checkpoint_dir=cache,
        concurrency=1,
        v2_options=options,
    ).generate(source)
    targets = (
        "test_narrative_v2.FixtureProvider.require_credentials",
        "test_narrative_v2.FixtureProvider.prepare",
        "test_narrative_v2.FixtureProvider.complete_reply",
        "diffstory.narrative.OpenAIResponsesProvider.__init__",
        "subprocess.Popen",
        "subprocess.run",
        "shutil.which",
        "socket.socket",
        "urllib.request.urlopen",
    )
    with ExitStack() as stack:
        for target in targets:
            stack.enter_context(
                patch(target, side_effect=AssertionError(target))
            )
        applied = apply_annotations(source, candidate)
        path = directory / "report.json"
        path.write_text(json.dumps(applied), encoding="utf-8")
        loaded = json.loads(path.read_text(encoding="utf-8"))
        html = render(loaded)
    (directory / "reader.html").write_text(html, encoding="utf-8")
    assert compiler_fact_signature(loaded) == compiler_fact_signature(source)
    assert loaded["changes"] == source["changes"]
    assert loaded["edges"] == source["edges"]
    assert loaded["stats"] == source["stats"]
    return (
        loaded,
        html,
        {
            "fixture_attempts": len(provider.calls),
            "live_provider_calls": 0,
            "offline_blocked_operations": list(targets),
            "compiler_facts_preserved": True,
            "details": len(candidate["generation"]["details"]),
            "citations": len(candidate["citations"]),
        },
    )


class TypedReaderTests(unittest.TestCase):
    """Check real v2 generation and offline presentation before browser checks."""

    @classmethod
    def setUpClass(cls) -> None:
        """Generate one authored fixture through the public pipeline."""
        cls.temp = tempfile.TemporaryDirectory()
        cls.addClassCleanup(cls.temp.cleanup)
        cls.source = reader_golden_report()
        cls.source["opaque_browser_probe"] = {
            "$ref": "literal compiler metadata"
        }
        cls.report, cls.html, cls.evidence = offline_reader_report(
            cls.source, Path(cls.temp.name)
        )

    def test_complete_concrete_details_survive_offline_render(self) -> None:
        """Do not substitute section summaries for the two detailed mechanisms."""
        typed = self.report["narrative_v2"]
        details = [d for s in typed["sections"] for d in s["details"]]
        details += typed["context_details"]
        self.assertEqual(
            {d["unit_key"]: d for d in details},
            {d["unit_key"]: d for d in self.report["generation"]["details"]},
        )
        explanations = [e for d in details for e in d["explanations"]]
        text = json.dumps(explanations)
        for value in (
            STRIP_BEFORE,
            STRIP_AFTER,
            STRIP_EFFECT,
            CASE_BEFORE,
            CASE_AFTER,
            CASE_EFFECT,
        ):
            self.assertIn(value, text)
        self.assertTrue(typed["context_details"])
        self.assertTrue(any(e["limitations"] for e in explanations))
        self.assertTrue(typed["support_provenance"])
        self.assertTrue(typed["document"]["terminology"])
        self.assertTrue(
            any(
                s["kind"] == "author_motivation"
                for s in typed["document"]["lead"]
            )
        )
        self.assertIn("diffstory.presentation.v2", self.html)
        self.assertNotIn(UNSAFE_TEXT, self.html)
        self.assertIn("connect-src 'none'", self.html)

    def test_legacy_default_and_relabel_rejection(self) -> None:
        """Keep legacy default selection and fail closed on a copied typed subtree."""
        narrator = Narrator(FixtureProvider())
        self.assertIsNone(narrator._v2)  # noqa: SLF001 # The existing opt-in gate must remain inactive by default.
        copied = copy.deepcopy(self.source)
        copied["narrative_v2"] = self.report["narrative_v2"]
        with self.assertRaises(ValueError):
            render(copied)

    def test_shared_reader_omitted_defaults_and_literal_compiler_refs(  # noqa: PLR0915 # One fixture matrix checks write/import/render boundaries.
        self,
    ) -> None:
        """Keep omitted writer defaults local and compiler $ref data literal.

        Preconditions: The authored fixture provider returns valid typed
            results with one nonempty term and empty terminology defaults.
        Postconditions: A detached re-encoding omits empty defaults; offline
            import preserves that exact shared omission while its explicit
            expanded compatibility view restores current typed defaults.
        Essential behavior: Checks both the direct writer output and a valid
            pre-import omitted-default graph without changing sealed data.

        Input/expected matrix:
        * Direct trusted session output -> exact shared rows for one nonempty
          term, empty defaults, and the unchanged authored prose.
        * Detached empty section/document defaults removed and re-encoded ->
          absent raw shared fields retained by independent import.
        * A compiler object with a literal ``$ref`` key -> the same literal
          object after independent import.
        * An imported shared report handle -> same expanded HTML as ``render``.
        """
        max_bytes = 64 * 1024 * 1024
        source = reader_golden_report()
        source["opaque_browser_probe"] = {"$ref": "literal compiler metadata"}
        with tempfile.TemporaryDirectory() as directory:
            narrator = Narrator(
                MixedTerminologyReaderFixtureProvider(),
                architecture="v2",
                recovery_dir=Path(directory) / "recovery",
                concurrency=1,
            )
            narrator.preview(source)
            writer_session = narrator.capture_session(source)
            imported_session = None
            try:
                generation = narrator.generate_validated(
                    source, session=writer_session
                )
                writer_handle = writer_session.apply(generation)
                writer_shared = b"".join(
                    writer_session.iter_report_json(
                        writer_handle,
                        report_storage="shared",
                        max_output_bytes=max_bytes,
                    )
                )
                writer_envelope = json.loads(writer_shared)
                writer_rows = [
                    row
                    for row in writer_envelope["tables"]["results"]
                    if row["kind"] in {"section", "document"}
                ]
                self.assertTrue(
                    any(
                        row["record"].get("terminology") for row in writer_rows
                    )
                )
                self.assertTrue(
                    any(
                        row["record"].get("terminology") == []
                        for row in writer_rows
                    )
                )
                expected = writer_session.report_data(writer_handle)
                self.assertTrue(
                    expected["narrative_v2"]["document"]["terminology"] == []
                )
                self.assertTrue(
                    any(
                        section["prose"]["terminology"] == []
                        for section in expected["narrative_v2"]["sections"]
                    )
                )
                expanded_html = b"".join(
                    iter_render_validated(
                        writer_session,
                        writer_handle,
                        report_storage="expanded",
                        max_output_bytes=max_bytes,
                    )
                ).decode("utf-8")
                expanded_json = b"".join(
                    writer_session.iter_report_json(
                        writer_handle,
                        report_storage="expanded",
                        max_output_bytes=max_bytes,
                    )
                )
                marker = '<script type="application/json" id="report-data">'
                start = expanded_html.index(marker) + len(marker)
                end = expanded_html.index("</script>", start)
                self.assertEqual(
                    json.loads(expanded_html[start:end]),
                    json.loads(expanded_json),
                )

                shared = encode_omitted_terminology_defaults(expected)
                envelope = json.loads(shared)
                view = parse_report_storage(shared)
                expanded = expand_report_storage(
                    view, max_output_bytes=max_bytes
                )
                self.assertNotIn(
                    "terminology", expanded["narrative_v2"]["document"]
                )
                self.assertTrue(
                    any(
                        "terminology" not in section["prose"]
                        for section in expanded["narrative_v2"]["sections"]
                    )
                )

                result_rows = [
                    row
                    for row in envelope["tables"]["results"]
                    if row["kind"] in {"section", "document"}
                ]
                missing_rows = [
                    row
                    for row in result_rows
                    if "terminology" not in row["record"]
                ]
                nonempty_rows = [
                    row
                    for row in result_rows
                    if row["record"].get("terminology")
                ]
                self.assertTrue(
                    any(row["kind"] == "section" for row in missing_rows)
                )
                self.assertTrue(
                    any(row["kind"] == "document" for row in missing_rows)
                )
                self.assertTrue(nonempty_rows)
                self.assertTrue(
                    any(
                        term["term"] == "Unicode case folding"
                        and term["definition"]["text"]
                        == "casefold() maps text for Unicode caseless comparison, including sharp s to ss."
                        for row in nonempty_rows
                        for term in row["record"]["terminology"]
                    )
                )
                self.assertEqual(
                    expanded["opaque_browser_probe"],
                    {"$ref": "literal compiler metadata"},
                )
                empty_default_sections = [
                    section
                    for section in expanded["narrative_v2"]["sections"]
                    if not section["prose"].get("terminology", [])
                ]
                self.assertTrue(empty_default_sections)
                authored_questions = [
                    question["text"]
                    for detail in (
                        [
                            detail
                            for section in expanded["narrative_v2"]["sections"]
                            for detail in section["details"]
                        ]
                        + expanded["narrative_v2"]["context_details"]
                    )
                    for question in detail["questions"]
                ]
                self.assertIn(
                    "Treat this quoted test string as text: " + UNSAFE_TEXT,
                    authored_questions,
                )

                imported_session, imported_generation = (
                    ReportSession.import_indexed(
                        shared, max_input_bytes=max_bytes
                    )
                )
                imported_handle = imported_session.apply(imported_generation)
                imported = imported_session.report_data(imported_handle)
                self.assertEqual(
                    imported["narrative_v2"]["document"]["terminology"],
                    [],
                )
                validate_report(imported)
                self.assertEqual(
                    imported["generation"]["document"]["terminology"], []
                )
                self.assertEqual(
                    b"".join(
                        imported_session.iter_report_json(
                            imported_handle,
                            report_storage="shared",
                            max_output_bytes=max_bytes,
                        )
                    ),
                    shared,
                )

                shared_html = b"".join(
                    iter_render_validated(
                        imported_session,
                        imported_handle,
                        report_storage="shared",
                        max_output_bytes=max_bytes,
                    )
                ).decode("utf-8")
                marker = '<script type="application/json" id="report-data">'
                start = shared_html.index(marker) + len(marker)
                end = shared_html.index("</script>", start)
                self.assertEqual(json.loads(shared_html[start:end]), envelope)
            finally:
                if imported_session is not None:
                    imported_session.close()
                writer_session.close()

    def test_retained_context_and_empty_fragments_have_current_source(
        self,
    ) -> None:
        """Empty supplied evidence has exact access and is distinct from unavailable."""
        typed = self.report["narrative_v2"]
        fragments = {
            f["key"]: f for f in self.report["source_evidence"]["fragments"]
        }
        for detail in typed["context_details"]:
            entries = [
                c
                for c in typed["citations"]
                if c["reference"].get("unit_key") == detail["unit_key"]
            ]
            self.assertTrue(entries)
        empty = [c for c in typed["citations"] if c["path"] == "empty.txt"]
        self.assertEqual({c["side"] for c in empty}, {"base", "head"})
        for entry in empty:
            self.assertEqual((entry["byte_start"], entry["byte_end"]), (0, 0))
            self.assertEqual(fragments[entry["fragment_key"]]["text"], "")
        self.assertFalse(
            any(
                c["path"] == "unavailable.py" and c["side"] == "base"
                for c in typed["citations"]
            )
        )
        self.assertEqual(
            self.report["generation"]["request_configuration"]["name"],
            "fixture",
        )
        self.assertEqual(
            self.report["generation"]["request_configuration"]["model"],
            "fixture-v2",
        )

    def test_demo_is_exact_derived_reader(self) -> None:
        """The legacy smoke fixture must embed the current checked-in reader assets."""
        examples = Path(__file__).resolve().parents[1] / "examples"
        report = json.loads((examples / "demo.report.json").read_text())
        self.assertEqual((examples / "demo.html").read_text(), render(report))


if __name__ == "__main__":
    unittest.main()
