"""Adversarial evidence, request sizing, and iterative story-planning checks."""

from __future__ import annotations

import copy
import json
import pickle
import unittest
from collections import Counter
from types import MappingProxyType
from typing import TYPE_CHECKING
from unittest import mock

from pydantic import ValidationError
from test_analysis import snap
from test_narrative import two_group_report

from diffstory import narrative_plan as plan_module
from diffstory.analysis import compile_snapshot
from diffstory.narrative_contracts import ClaimRef
from diffstory.narrative_contracts import Contract
from diffstory.narrative_contracts import DetailResult
from diffstory.narrative_contracts import Digest
from diffstory.narrative_contracts import EvidenceLedger
from diffstory.narrative_contracts import EvidenceQuestion
from diffstory.narrative_contracts import EvidenceUnit
from diffstory.narrative_contracts import Explanation
from diffstory.narrative_contracts import FactualObservation
from diffstory.narrative_contracts import GroupSpec
from diffstory.narrative_contracts import PlanningGroup
from diffstory.narrative_contracts import SourceSpan
from diffstory.narrative_contracts import SpanRef
from diffstory.narrative_contracts import SupportedStatement
from diffstory.narrative_contracts import Terminology
from diffstory.narrative_plan import CitationResolutionScope
from diffstory.narrative_plan import PackingError
from diffstory.narrative_plan import RequestPackingLimits
from diffstory.narrative_plan import build_evidence_ledger
from diffstory.narrative_plan import build_story_plan
from diffstory.narrative_plan import canonical_bytes
from diffstory.narrative_plan import capture_indexed_detail_results
from diffstory.narrative_plan import check_evidence_batch
from diffstory.narrative_plan import detail_output_allowance
from diffstory.narrative_plan import ordered_components
from diffstory.narrative_plan import pack_evidence_units
from diffstory.narrative_plan import pack_request_items
from diffstory.narrative_plan import planning_groups
from diffstory.narrative_plan import planning_groups_from_details
from diffstory.narrative_plan import prepare_request_items
from diffstory.narrative_plan import reconstruct_buffer
from diffstory.narrative_plan import resolve_span_ref
from diffstory.narrative_plan import semantic_digest
from diffstory.narrative_plan import span_reference
from diffstory.narrative_plan import validate_detail_result
from diffstory.narrative_plan import validate_detail_results
from diffstory.narrative_plan import validate_evidence_ledger
from diffstory.narrative_plan import validate_indexed_detail_results
from diffstory.narrative_plan import validate_planning_claims
from diffstory.narrative_plan import validate_story_order
from diffstory.narrative_plan import validate_story_plan
from diffstory.narrative_plan import validate_story_requests
from diffstory.usage import ModelCapacity

if TYPE_CHECKING:
    from collections.abc import Callable
    from types import FrameType


def _span_validator_profiler(
    counts: Counter[str],
) -> Callable[[FrameType, str, object], None]:
    """Build a profiler that counts actual nested span-validator calls.

    Preconditions:
        ``counts`` is the mutable counter for one measured operation.
    Postconditions:
        Returns a call profiler that increments ``span_ref_ranges`` only for
        actual ``SpanRef.ranges`` Python executions.
    Essential behavior:
        Exposes nested structural work that top-level parser mocks omit.
    """

    def profile(frame: FrameType, event: str, _arg: object) -> None:
        if (
            event == "call"
            and frame.f_code.co_name == "ranges"
            and isinstance(frame.f_locals.get("self"), SpanRef)
        ):
            counts["span_ref_ranges"] += 1

    return profile


BEFORE = 'def normalize(value):\n    value = value.strip()\n    marker = "request"\n    return value.lower()\n'
AFTER = 'def normalize(value):\n    value = value.rstrip()\n    marker = "request"\n    return value.casefold()\n'


def authored_result(unit: object) -> DetailResult:
    """Explain the concrete normalization fixture with complete source support."""
    refs = tuple(span_reference(span, unit) for span in unit.spans)
    return DetailResult(
        unit_key=unit.id,
        explanations=tuple(
            Explanation(
                obligation_ids=(obligation.id,),
                before="normalize removed whitespace at both ends, then used lower() to lowercase the result.",
                after="normalize removes only trailing whitespace, then uses casefold() to fold letter case.",
                effect="Leading whitespace now remains in the returned value. Case folding may produce different text from lower() for Unicode input.",
                uncertainty="The supplied source shows no callers or executed tests, so caller assumptions and runtime test outcomes remain unverified.",
                citations=tuple(
                    ref
                    for ref in refs
                    if ref.span_id in obligation.required_span_ids
                ),
                limitations=obligation.limitations,
            )
            for obligation in unit.obligations
        ),
        observations=(
            FactualObservation(
                id=semantic_digest("normalization observation", unit.id),
                text="The head implementation calls rstrip() before casefold().",
                citations=refs,
            ),
        )
        if refs
        else (),
        questions=(
            EvidenceQuestion(
                text="Do callers expect leading whitespace to be preserved?",
                citations=refs[:1],
                obligation_ids=(),
            ),
        )
        if refs
        else (),
    )


def body(items: tuple, output: int) -> bytes:
    """Serialize a complete fixture envelope, including instructions and schema."""
    return canonical_bytes(
        {
            "instructions": "Use source-backed detail.",
            "schema": {"type": "object", "additionalProperties": False},
            "items": items,
            "max_output_tokens": output,
        }
    )


def change_result(
    result: DetailResult, index: int, **fields: object
) -> DetailResult:
    """Build a deliberately invalid explanation for public-boundary tests."""
    explanations = list(result.explanations)
    explanations[index] = explanations[index].model_copy(update=fields)
    return result.model_copy(update={"explanations": tuple(explanations)})


class LedgerTests(unittest.TestCase):
    """Check exact original bytes, source ownership, and semantic identity."""

    def test_exact_bytes_empty_crlf_multibyte_and_same_line_parts(
        self,
    ) -> None:
        """Keep blank changes, terminal newline changes, and each empty buffer."""
        report = compile_snapshot(
            snap(
                {"notes.txt": "café" * 20 + "\r\n\r\n", "empty.txt": ""},
                {"notes.txt": "café" * 20 + "!\r\n\n", "empty.txt": ""},
            )
        )
        ledger = build_evidence_ledger(
            report, max_span_bytes=9, max_unit_spans=2
        )
        fragments = {
            f["key"]: f for f in report["source_evidence"]["fragments"]
        }
        for binding in ledger.bindings:
            self.assertEqual(
                reconstruct_buffer(ledger, binding.buffer_key),
                fragments[binding.fragment_key]["text"].encode(),
            )
        pieces = [
            s
            for s in ledger.spans
            if s.path == "notes.txt" and s.side == "base" and s.line_start == 1
        ]
        self.assertGreater(len(pieces), 5)
        self.assertEqual(len({p.id for p in pieces}), len(pieces))
        self.assertTrue(all(p.byte_end - p.byte_start <= 9 for p in pieces))
        self.assertTrue(all(len(unit.spans) <= 2 for unit in ledger.units))
        self.assertEqual(sum(b.byte_length == 0 for b in ledger.buffers), 2)
        validate_evidence_ledger(ledger)

    def test_original_runs_are_not_normalized_hunks(self) -> None:
        """Detect a newline-only change even when normalized hunks are removed."""
        report = compile_snapshot(
            snap(
                {"x.py": "def f():\r\n    pass\r\n"},
                {"x.py": "def f():\n    pass"},
            )
        )
        for change in report["changes"]:
            change["hunks"] = []
        ledger = build_evidence_ledger(report)
        changed = {
            sid
            for unit in ledger.units
            for o in unit.obligations
            if o.kind == "changed_region"
            for sid in o.required_span_ids
        }
        self.assertTrue(changed)
        self.assertEqual(changed, {s.id for s in ledger.spans})

    def test_identical_move_has_structural_obligation(self) -> None:
        """An unchanged function at a new path still requires an explanation."""
        report = compile_snapshot(snap({"a.py": BEFORE}, {"b.py": BEFORE}))
        ledger = build_evidence_ledger(report)
        moved = [
            u
            for u in ledger.units
            if u.classification and u.classification.kind == "moved"
        ]
        self.assertTrue(moved)
        self.assertTrue(
            any(o.kind == "structure" for u in moved for o in u.obligations)
        )
        self.assertFalse(
            any(
                o.kind == "changed_region"
                for u in moved
                for o in u.obligations
            )
        )
        self.assertFalse(
            any(
                o.kind == "limitation"
                for u in ledger.units
                for o in u.obligations
            )
        )
        self.assertEqual(
            {b.change_id for b in ledger.changes},
            {c["id"] for c in report["changes"]},
        )

    def test_missing_projection_or_explicit_binding_rejected(self) -> None:
        """Old reports cannot claim original-byte coverage without recompilation."""
        report = compile_snapshot(snap({"x.py": BEFORE}, {"x.py": AFTER}))
        missing = copy.deepcopy(report)
        del missing["source_evidence"]
        with self.assertRaisesRegex(
            ValueError, "original_source_projection_required"
        ):
            build_evidence_ledger(missing)
        del report["changes"][0]["before"]["fragment_key"]
        with self.assertRaisesRegex(
            ValueError, "source_fragment_binding_required"
        ):
            build_evidence_ledger(report)

    def test_unavailable_absent_and_unpaired_evidence_are_distinct(
        self,
    ) -> None:
        """Missing text cannot be interpreted as a confirmed absent file."""
        cases = [
            ({"state": "absent"}, None),
            ({"state": "unavailable", "reason": "read_failed"}, "unavailable"),
        ]
        for state, limitation in cases:
            with self.subTest(state=state):
                report = compile_snapshot(
                    snap(
                        {},
                        {"x.py": AFTER},
                        overrides={("base", "x.py"): state},
                    )
                )
                ledger = build_evidence_ledger(report)
                self.assertEqual(
                    next(
                        a.state
                        for a in ledger.availability
                        if a.side == "base"
                    ),
                    state["state"],
                )
                codes = {
                    limit.code
                    for u in ledger.units
                    for o in u.obligations
                    for limit in o.limitations
                }
                self.assertEqual(codes, {limitation} if limitation else set())
                if limitation:
                    self.assertEqual(
                        ledger.units[0].classification.kind, "observed_head"
                    )
                    self.assertEqual(
                        ledger.units[0].classification.uncertainty,
                        (report["changes"][0]["basis"],),
                    )
        source = snap(
            {"x.py": "# old excerpt\n"},
            {"x.py": AFTER},
            overrides={
                (side, "x.py"): {"state": "supplied", "coverage": "partial"}
                for side in ("base", "head")
            },
            fragment_options={
                (side, "x.py"): {"region": side, "start_line": 51}
                for side in ("base", "head")
            },
        )
        ledger = build_evidence_ledger(compile_snapshot(source))
        self.assertIn(
            "unpaired_excerpt",
            {
                limit.code
                for u in ledger.units
                for o in u.obligations
                for limit in o.limitations
            },
        )
        self.assertTrue(
            all(s.byte_start >= 0 and s.line_start >= 51 for s in ledger.spans)
        )

    def test_overlapping_context_has_explicit_shared_ownership(self) -> None:
        """Broad context intervals can overlap definition evidence without loss."""
        before = 'print("old")\n\ndef f():\n    return 1\n\nprint("old")\n'
        after = 'print("new")\n\ndef f():\n    return 2\n\nprint("new")\n'
        ledger = build_evidence_ledger(
            compile_snapshot(snap({"x.py": before}, {"x.py": after}))
        )
        self.assertTrue(any(o.shared for o in ledger.ownership))
        self.assertTrue(
            all(o.shared == (len(o.unit_keys) > 1) for o in ledger.ownership)
        )
        self.assertTrue(all(o.unit_keys for o in ledger.ownership))

    def test_content_keys_survive_revision_and_unrelated_fragment_insertion(
        self,
    ) -> None:
        """Legacy IDs and source positions do not define reusable evidence keys."""
        source = snap({"x.py": BEFORE}, {"x.py": AFTER})
        first = build_evidence_ledger(compile_snapshot(source))
        source["meta"].update(base_sha="c" * 40, head_sha="d" * 40)
        second = build_evidence_ledger(compile_snapshot(source))
        self.assertEqual(first.id, second.id)
        source["fragments"].reverse()
        reordered = build_evidence_ledger(compile_snapshot(source))
        self.assertEqual(first.id, reordered.id)
        inserted = snap(
            {"unrelated.txt": "same\n", "x.py": "# prefix\n" + BEFORE},
            {"unrelated.txt": "same\n", "x.py": "# prefix\n" + AFTER},
        )
        current = build_evidence_ledger(compile_snapshot(inserted))
        old_unit = next(u for u in first.units if u.change_key)
        new_unit = next(
            u for u in current.units if u.change_key == old_unit.change_key
        )
        self.assertEqual(old_unit.id, new_unit.id)
        old_ref = span_reference(old_unit.spans[0], old_unit)
        self.assertEqual(
            resolve_span_ref(old_ref, new_unit, current).line_start,
            old_unit.spans[0].line_start + 1,
        )
        validate_detail_result(authored_result(old_unit), new_unit, current)
        changed = build_evidence_ledger(
            compile_snapshot(
                snap(
                    {"x.py": BEFORE.replace('"request"', '"response"')},
                    {"x.py": AFTER.replace('"request"', '"response"')},
                )
            )
        )
        self.assertNotIn(old_unit.id, {u.id for u in changed.units})

    def test_duplicate_contents_keep_distinct_occurrences(self) -> None:
        """Separate equal fragments and equal same-line pieces safely."""
        source = snap(
            {"x.txt": "same\n"},
            {"x.txt": "same!\n"},
            overrides={
                (side, "x.txt"): {"state": "supplied", "coverage": "partial"}
                for side in ("base", "head")
            },
            fragment_options={
                (side, "x.txt"): {"region": "one"} for side in ("base", "head")
            },
        )
        source["fragments"].extend(
            {**f, "start_line": 100, "region": "two"}
            for f in list(source["fragments"])
        )
        ledger = build_evidence_ledger(compile_snapshot(source))
        self.assertEqual(len(ledger.buffers), 4)
        self.assertEqual(len({b.key for b in ledger.buffers}), 4)
        self.assertEqual(len({s.id for s in ledger.spans}), len(ledger.spans))

    def test_ledger_rejects_gap_and_modified_text(self) -> None:
        """Persisted-boundary validation checks bytes and coordinates again."""
        ledger = build_evidence_ledger(
            compile_snapshot(snap({"x.py": BEFORE}, {"x.py": AFTER}))
        )
        for update in (
            {"byte_start": 1, "byte_end": ledger.spans[0].byte_end + 1},
            {"text": ledger.spans[0].text.replace("def", "xxx")},
        ):
            damaged = ledger.model_copy(
                update={
                    "spans": (
                        ledger.spans[0].model_copy(update=update),
                        *ledger.spans[1:],
                    )
                }
            )
            with self.assertRaises(ValueError):
                validate_evidence_ledger(damaged)

    def test_unchanged_context_has_dispatch_but_no_explanation_obligation(
        self,
    ) -> None:
        """Unchanged lines and empty files remain covered without invented changes."""
        ledger = build_evidence_ledger(
            compile_snapshot(
                snap(
                    {"x.py": BEFORE, "empty.txt": ""},
                    {"x.py": BEFORE, "empty.txt": ""},
                )
            )
        )
        self.assertTrue(ledger.units)
        self.assertFalse(any(u.obligations for u in ledger.units))
        self.assertTrue(all(o.unit_keys for o in ledger.ownership))

    def test_long_change_splits_keep_both_comparison_sides(self) -> None:
        """A changed long line can be dispatched in bounded comparison pieces."""
        ledger = build_evidence_ledger(
            compile_snapshot(snap({"x.txt": "é" * 100}, {"x.txt": "λ" * 100})),
            max_span_bytes=8,
            max_unit_spans=2,
        )
        for unit in ledger.units:
            for obligation in unit.obligations:
                if obligation.kind == "changed_region":
                    self.assertEqual(
                        {
                            s.side
                            for s in unit.spans
                            if s.id in obligation.required_span_ids
                        },
                        {"base", "head"},
                    )
        self.assertTrue(all(len(u.spans) <= 2 for u in ledger.units))

    def test_text_only_status_is_not_a_parse_failure(self) -> None:
        """Preserve the compiler distinction between opaque text and failed Python."""
        ledger = build_evidence_ledger(
            compile_snapshot(snap({"x.txt": "a\n"}, {"x.txt": "b\n"}))
        )
        codes = {
            limit.code
            for u in ledger.units
            for o in u.obligations
            for limit in o.limitations
        }
        self.assertIn("text_only", codes)
        self.assertNotIn("parse_failed", codes)


class DetailValidationTests(unittest.TestCase):
    """Use realistic detail and invalid citations to test structural guarantees."""

    def setUp(self) -> None:
        """Compile two separated changed regions inside one normalization function."""
        self.ledger = build_evidence_ledger(
            compile_snapshot(snap({"x.py": BEFORE}, {"x.py": AFTER}))
        )
        self.unit = next(u for u in self.ledger.units if u.change_key)
        self.result = authored_result(self.unit)

    def test_realistic_detail_and_json_roundtrip(self) -> None:
        """Keep before/after mechanisms and source-backed factual observations."""
        self.assertEqual(
            sum(o.kind == "changed_region" for o in self.unit.obligations), 2
        )
        validate_detail_result(self.result, self.unit, self.ledger)
        restored = DetailResult.model_validate_json(
            self.result.model_dump_json()
        )
        self.assertEqual(restored, self.result)
        validate_detail_result(restored, self.unit, self.ledger)

    def test_change_citation_does_not_cover_omitted_changed_region(
        self,
    ) -> None:
        """Citing structural anchors cannot replace an explanation of the last edit."""
        missing = self.result.model_copy(
            update={"explanations": self.result.explanations[:-1]}
        )
        with self.assertRaisesRegex(ValueError, "obligation_coverage"):
            validate_detail_result(missing, self.unit, self.ledger)
        no_region = change_result(
            self.result, 2, citations=self.result.explanations[0].citations
        )
        with self.assertRaisesRegex(ValueError, "missing_changed_region"):
            validate_detail_result(no_region, self.unit, self.ledger)

    def test_wrong_side_path_digest_ranges_and_unknown_refs(self) -> None:
        """Reject each tampered citation field independently."""
        first = self.result.explanations[0]
        ref = first.citations[0]
        updates = [
            {"side": "head" if ref.side == "base" else "base"},
            {"path": "wrong.py"},
            {"content_digest": "f" * 64},
            {"byte_end": ref.byte_end + 1},
            {"line_start": 2, "line_end": 2},
            {"span_id": "e" * 64},
            {"unit_key": "d" * 64},
            {"group_key": "c" * 64},
        ]
        for update in updates:
            with self.subTest(update=update):
                damaged = change_result(
                    self.result,
                    0,
                    citations=(
                        ref.model_copy(update=update),
                        *first.citations[1:],
                    ),
                )
                with self.assertRaises(ValueError):
                    validate_detail_result(damaged, self.unit, self.ledger)

    def test_duplicate_refs_and_obligations(self) -> None:
        """Reject repeated evidence within a claim and repeated obligation coverage."""
        explanation = self.result.explanations[0]
        for damaged in (
            change_result(
                self.result,
                0,
                citations=(*explanation.citations, explanation.citations[0]),
            ),
            self.result.model_copy(
                update={
                    "explanations": (*self.result.explanations, explanation)
                }
            ),
        ):
            with self.assertRaises(ValueError):
                validate_detail_result(damaged, self.unit, self.ledger)

    def test_cross_group_and_cross_unit_and_unlinked_observations(
        self,
    ) -> None:
        """A known span from another item is still not valid evidence here."""
        ledger = build_evidence_ledger(two_group_report(), max_unit_spans=2)
        unit = ledger.units[0]
        other = next(u for u in ledger.units if u.group_key != unit.group_key)
        result = authored_result(unit)
        bad = result.observations[0].model_copy(
            update={"citations": (span_reference(other.spans[0], other),)}
        )
        with self.assertRaises(ValueError):
            validate_detail_result(
                result.model_copy(update={"observations": (bad,)}),
                unit,
                ledger,
            )
        same_group = next(
            u
            for u in ledger.units
            if u.group_key == unit.group_key and u.id != unit.id
        )
        bad = result.observations[0].model_copy(
            update={
                "citations": (span_reference(same_group.spans[0], same_group),)
            }
        )
        with self.assertRaises(ValueError):
            validate_detail_result(
                result.model_copy(update={"observations": (bad,)}),
                unit,
                ledger,
            )
        bad = self.result.observations[0].model_copy(update={"citations": ()})
        with self.assertRaises(ValidationError):
            validate_detail_result(
                self.result.model_copy(update={"observations": (bad,)}),
                self.unit,
                self.ledger,
            )
        with self.assertRaises(ValidationError):
            self.result.explanations[0].model_validate(
                {
                    **self.result.explanations[0].model_dump(),
                    "obligation_ids": ("legacy-change-id",),
                }
            )

    def test_limitation_must_be_explicit(self) -> None:
        """An unavailable base requires the exact structured limitation."""
        ledger = build_evidence_ledger(
            compile_snapshot(
                snap(
                    {},
                    {"x.py": AFTER},
                    overrides={
                        ("base", "x.py"): {
                            "state": "unavailable",
                            "reason": "read_failed",
                        }
                    },
                )
            )
        )
        unit = next(
            u
            for u in ledger.units
            if any(o.limitations for o in u.obligations)
        )
        result = authored_result(unit)
        validate_detail_result(result, unit, ledger)
        index = next(
            i for i, exp in enumerate(result.explanations) if exp.limitations
        )
        with self.assertRaisesRegex(ValueError, "explicit_limitation"):
            validate_detail_result(
                change_result(result, index, limitations=()), unit, ledger
            )

    def test_full_detail_set_and_planning_claims_require_validated_observations(
        self,
    ) -> None:
        """Cannot omit whole units or substitute arbitrary IDs as planning facts."""
        results = tuple(authored_result(u) for u in self.ledger.units)
        groups = planning_groups_from_details(self.ledger, results)
        validate_detail_results(results, self.ledger)
        validate_planning_claims(groups, results, self.ledger)
        with self.assertRaisesRegex(ValueError, "detail_set_coverage"):
            validate_detail_results(results[:-1], self.ledger)
        claim = (
            groups[0].claims[0].model_copy(update={"observation_id": "f" * 64})
        )
        forged = groups[0].model_copy(update={"claims": (claim,)})
        with self.assertRaisesRegex(ValueError, "unsupported_planning_claim"):
            validate_planning_claims((forged,), results, self.ledger)


class PackingTests(unittest.TestCase):
    """Verify actual-body sizing, output guard, and practical limits."""

    def setUp(self) -> None:
        """Use two independent compiler groups and a roomy fixture capacity."""
        self.ledger = build_evidence_ledger(two_group_report())
        self.capacity = ModelCapacity(1_000_000, 100_000)
        self.limits = RequestPackingLimits(timeout_seconds=10_000)

    def test_all_groups_fit_in_one_request(self) -> None:
        """Presentation group count sets no evidence-call floor."""
        batches = pack_evidence_units(
            self.ledger.units, self.capacity, self.limits, body
        )
        self.assertEqual(len(batches), 1)
        self.assertEqual(len(batches[0].group_keys), 2)
        by_key = {u.id: u for u in self.ledger.units}
        check_evidence_batch(
            batches[0],
            tuple(by_key[k] for k in batches[0].item_keys),
            self.capacity,
            self.limits,
            body,
        )
        self.assertEqual(
            batches[0].output_guard, self.capacity.max_output_tokens
        )

    def test_output_reservation_enforcement_is_additive(self) -> None:
        """Default Codex guard stays maximal; an enforceable cap frees input."""
        capacity = ModelCapacity(20_000, 15_000)
        self.assertEqual(capacity.input_upper_bound, 5_000)
        self.assertEqual(capacity.input_bound_for(100), 5_000)
        self.assertEqual(
            capacity.input_bound_for(100, enforceable_output_cap=True), 19_900
        )
        for bad in (True, 0, 15_001, "100"):
            with self.assertRaises(ValueError):
                capacity.output_guard(bad)
        unit = (self.ledger.units[0],)
        with self.assertRaisesRegex(PackingError, "input_limit"):
            pack_evidence_units(unit, capacity, self.limits, body)
        self.assertEqual(
            len(
                pack_evidence_units(
                    unit,
                    capacity,
                    self.limits,
                    body,
                    enforceable_output_cap=True,
                )
            ),
            1,
        )

    def test_schema_repair_and_framing_are_counted(self) -> None:
        """Extra schema, repair, and framing bytes can make an atomic item unsendable."""
        limits = RequestPackingLimits(
            framing_bytes=13,
            schema_reserve_bytes=17,
            repair_reserve_bytes=19,
            safety_tokens=23,
            repair_output_tokens=29,
            timeout_seconds=10_000,
        )
        spec = pack_evidence_units(
            self.ledger.units,
            self.capacity,
            limits,
            body,
            enforceable_output_cap=True,
        )[0]
        ordered = tuple(sorted(self.ledger.units, key=lambda u: u.id))
        self.assertEqual(
            spec.input_bound,
            len(body(ordered, spec.output_reservation)) + 1024 + 13 + 17 + 19,
        )
        self.assertEqual(
            spec.output_reservation,
            detail_output_allowance(ordered, limits) + 29,
        )
        huge = limits.model_copy(update={"repair_reserve_bytes": 1_000_000})
        with self.assertRaisesRegex(PackingError, "input_limit"):
            pack_evidence_units(self.ledger.units, self.capacity, huge, body)

    def test_response_time_and_output_limits(self) -> None:
        """Each practical bound independently rejects unsendable atomic evidence."""
        for updates, reason in (
            ({"response_byte_limit": 1}, "response_limit"),
            ({"timeout_seconds": 1}, "timeout_limit"),
        ):
            with self.subTest(reason=reason):
                limits = self.limits.model_copy(update=updates)
                with self.assertRaisesRegex(PackingError, reason):
                    pack_evidence_units(
                        self.ledger.units, self.capacity, limits, body
                    )
        with self.assertRaisesRegex(PackingError, "output_limit"):
            pack_evidence_units(
                self.ledger.units,
                ModelCapacity(1_000_000, 2),
                self.limits,
                body,
            )

    def test_more_changed_source_requires_more_output(self) -> None:
        """Output allowance follows obligations, detail bytes, citations, and JSON."""
        small = build_evidence_ledger(
            compile_snapshot(snap({"x.py": BEFORE}, {"x.py": AFTER}))
        )
        large = build_evidence_ledger(
            compile_snapshot(
                snap(
                    {"x.py": BEFORE},
                    {
                        "x.py": AFTER.replace(
                            '"request"', '"' + "a" * 10_000 + '"'
                        )
                    },
                )
            ),
            max_span_bytes=200,
        )
        self.assertGreater(
            detail_output_allowance(large.units, self.limits),
            detail_output_allowance(small.units, self.limits),
        )
        batches = pack_evidence_units(
            large.units,
            ModelCapacity(60_000, 20_000),
            self.limits,
            body,
            enforceable_output_cap=True,
        )
        self.assertGreater(len(batches), 1)
        self.assertEqual(
            {key for b in batches for key in b.item_keys},
            {u.id for u in large.units},
        )

    def test_actual_serializer_change_and_deterministic_partitions(
        self,
    ) -> None:
        """Recheck prepared bodies and retain stable ordering across input shuffles."""
        first = pack_evidence_units(
            self.ledger.units, self.capacity, self.limits, body
        )
        self.assertEqual(
            first,
            pack_evidence_units(
                tuple(reversed(self.ledger.units)),
                self.capacity,
                self.limits,
                body,
            ),
        )
        ordered = tuple(sorted(self.ledger.units, key=lambda u: u.id))
        with self.assertRaisesRegex(ValueError, "prepared_request_changed"):
            check_evidence_batch(
                first[0],
                ordered,
                self.capacity,
                self.limits,
                lambda items, output: body(items, output) + b" ",
            )


def graph_groups(
    count: int, *, cycle: bool = False, claims: int = 0
) -> tuple[PlanningGroup, ...]:
    """Build a large compiler-like chain or SCC with complete fact records."""
    keys = [semantic_digest("group", i) for i in range(count)]
    result = []
    for index, key in enumerate(keys):
        unit = semantic_digest("unit", index)
        refs = tuple(
            ClaimRef(
                unit_key=unit,
                observation_id=semantic_digest("observation", index, number),
            )
            for number in range(claims)
        )
        prerequisites = (keys[index - 1],) if index or cycle else ()
        result.append(
            PlanningGroup(
                group=GroupSpec(
                    key=key,
                    path=f"module{index}.py",
                    theme="implementation",
                    unit_keys=(unit,),
                    prerequisites=prerequisites,
                ),
                claims=refs,
                terminology=(
                    Terminology(
                        term="normalization",
                        definition="Removal of trailing whitespace followed by Unicode case folding.",
                        claims=refs,
                    ),
                )
                if refs
                else (),
            )
        )
    return tuple(result)


class StoryTests(unittest.TestCase):
    """Check nonrecursive graph order, complete hierarchy, and trusted validation."""

    def test_huge_chain_and_cycle_are_iterative(self) -> None:
        """Graphs larger than Python's recursion limit retain every node and edge."""
        nodes = [f"g{i:05d}" for i in range(2500)]
        chain = {
            node: (nodes[i - 1],) if i else () for i, node in enumerate(nodes)
        }
        order, cycles = ordered_components(chain)
        self.assertEqual(order, tuple(nodes))
        self.assertEqual(cycles, ())
        chain[nodes[0]] = (nodes[-1],)
        order, cycles = ordered_components(chain)
        self.assertEqual(len(cycles), 1)
        self.assertEqual(set(cycles[0]), set(nodes))
        validate_story_order(order, chain)

    def test_large_scc_spans_bounded_cards_and_chapters(self) -> None:
        """Keep cycle adjacency while every hierarchy request fits the serializer."""
        groups = graph_groups(1200, cycle=True)
        capacity = ModelCapacity(14_000, 5000)
        limits = RequestPackingLimits(repair_reserve_bytes=0, safety_tokens=0)
        plan = build_story_plan(
            groups,
            capacity,
            limits,
            body,
            context_digest="a" * 64,
            max_chapter_groups=19,
            enforceable_output_cap=True,
        )
        self.assertGreater(len(plan.chapters), 50)
        self.assertGreater(max(c.level for c in plan.cards), 1)
        self.assertEqual(len(plan.cycles[0]), 1200)
        self.assertTrue(
            all(
                r.input_bound + r.output_guard <= capacity.context_tokens
                for r in plan.requests
            )
        )
        validate_story_plan(plan, groups)
        validate_story_requests(
            plan, capacity, limits, body, enforceable_output_cap=True
        )
        self.assertTrue(
            any(c.cross_chapter_prerequisites for c in plan.chapters)
        )

    def test_large_card_declared_splits_preserve_claims_and_terms(
        self,
    ) -> None:
        """Split only between complete atoms and retain all references."""
        groups = graph_groups(1, claims=150)
        # A whole term with 150 references is indivisible; give it room, while
        # all claim atoms plus the term still exceed one card's capacity.
        capacity = ModelCapacity(75_000, 15_000)
        limits = RequestPackingLimits(repair_reserve_bytes=0, safety_tokens=0)
        plan = build_story_plan(
            groups,
            capacity,
            limits,
            body,
            context_digest="b" * 64,
            enforceable_output_cap=True,
        )
        leaves = [c for c in plan.cards if c.level == 0]
        self.assertGreater(len(leaves), 1)
        atoms = [a for c in leaves for a in c.atoms]
        self.assertEqual(
            tuple(a.claim for a in atoms if a.kind == "claim"),
            groups[0].claims,
        )
        self.assertEqual(
            tuple(a.term for a in atoms if a.kind == "term"),
            groups[0].terminology,
        )
        validate_story_plan(plan, groups)
        with self.assertRaises(PackingError):
            build_story_plan(
                groups,
                ModelCapacity(4000, 1000),
                limits,
                body,
                context_digest="b" * 64,
                enforceable_output_cap=True,
            )

    def test_invalid_order_missing_member_edge_claim_and_cycle(self) -> None:
        """External compiler facts remain authoritative for all proposed plans."""
        groups = graph_groups(3, claims=2)
        plan = build_story_plan(
            groups,
            ModelCapacity(100_000, 20_000),
            RequestPackingLimits(),
            body,
            context_digest="c" * 64,
        )
        for update in (
            {"ordered_group_keys": tuple(reversed(plan.ordered_group_keys))},
            {"ordered_group_keys": plan.ordered_group_keys[:-1]},
            {"cards": plan.cards[:-1]},
            {"cycles": (plan.ordered_group_keys,)},
            {"chapters": ()},
        ):
            with (
                self.subTest(fields=tuple(update)),
                self.assertRaises(ValueError),
            ):
                validate_story_plan(plan.model_copy(update=update), groups)
        damaged_group = groups[1].model_copy(update={"claims": ()})
        with self.assertRaisesRegex(ValueError, "planning_facts_changed"):
            validate_story_plan(
                plan.model_copy(
                    update={"groups": (groups[0], damaged_group, groups[2])}
                ),
                groups,
            )
        cyclic = {"a": ("b",), "b": ("a",), "c": ()}
        with self.assertRaisesRegex(ValueError, "story_cycle_not_adjacent"):
            validate_story_order(("a", "c", "b"), cyclic)

    def test_small_multigroup_planning_batches_and_stable_cards(self) -> None:
        """Leaf requests pack groups together, and unchanged leaf keys survive insertion."""
        groups = planning_groups(build_evidence_ledger(two_group_report()))
        args = (
            ModelCapacity(1_000_000, 100_000),
            RequestPackingLimits(timeout_seconds=10_000),
            body,
        )
        plan = build_story_plan(groups, *args, context_digest="d" * 64)
        leaf_keys = {c.id for c in plan.cards if c.level == 0}
        leaf_requests = [
            r for r in plan.requests if set(r.item_keys) <= leaf_keys
        ]
        self.assertEqual(len(leaf_requests), 1)
        self.assertEqual(len(plan.requests), 1)
        self.assertEqual(len(leaf_requests[0].group_keys), 2)
        added = build_story_plan(
            (*groups, *graph_groups(1)), *args, context_digest="e" * 64
        )
        self.assertTrue(
            leaf_keys <= {c.id for c in added.cards if c.level == 0}
        )
        restored = type(plan).model_validate_json(plan.model_dump_json())
        validate_story_plan(restored, groups)
        self.assertEqual(
            json.loads(plan.model_dump_json())["version"],
            "diffstory.story-plan.v2",
        )


class PhysicalSourceLedgerTests(unittest.TestCase):
    """Bind obligations to physical lines while retaining each original byte."""

    def test_unicode_function_changed_region_and_exact_reconstruction(
        self,
    ) -> None:
        """The return edit belongs to f on line three, never a phantom line."""
        for separator in ("\u2028", "\u2029", "\x85", "\v", "\f"):
            for eol in ("\n", "\r\n", "\r"):
                before = eol.join(
                    (
                        "def f():",
                        f"    value = 'a{separator}b'",
                        "    return 1",
                        "",
                    )
                )
                after = before.replace("return 1", "return 2")
                ledger = build_evidence_ledger(
                    compile_snapshot(
                        snap({"example.py": before}, {"example.py": after})
                    )
                )
                with self.subTest(separator=separator, eol=eol):
                    self.assertEqual(max(s.line_end for s in ledger.spans), 3)
                    units = [
                        u
                        for u in ledger.units
                        if u.classification
                        and u.classification.kind == "modified"
                    ]
                    self.assertTrue(units)
                    changed = {
                        sid
                        for u in units
                        for o in u.obligations
                        if o.kind == "changed_region"
                        for sid in o.required_span_ids
                    }
                    spans = [s for s in ledger.spans if s.id in changed]
                    self.assertEqual({s.line_start for s in spans}, {3})
                    self.assertEqual(
                        {s.text for s in spans},
                        {"    return 1" + eol, "    return 2" + eol},
                    )
                    for buffer in ledger.buffers:
                        self.assertEqual(
                            reconstruct_buffer(ledger, buffer.key),
                            (
                                before if buffer.side == "base" else after
                            ).encode(),
                        )

    def test_terminal_and_newline_only_ownership_with_each_eol(self) -> None:
        """Empty anchors and terminal delimiters retain exact source ownership."""
        for eol in ("\n", "\r\n", "\r"):
            for before, after, maximum_line in (
                ("", eol, 1),
                (eol, eol * 2, 2),
                ("a", "a" + eol, 1),
                ("é\u2028" * 30, "é\u2028" * 30 + eol, 1),
            ):
                with self.subTest(eol=eol, before_length=len(before)):
                    ledger = build_evidence_ledger(
                        compile_snapshot(
                            snap({"x.txt": before}, {"x.txt": after})
                        ),
                        max_span_bytes=4,
                        max_unit_spans=2,
                    )
                    index = plan_module.build_ledger_index(ledger)
                    self.assertEqual(
                        max(span.line_end for span in ledger.spans),
                        maximum_line,
                    )
                    self.assertEqual(
                        {o.span_id for o in ledger.ownership},
                        {span.id for span in ledger.spans},
                    )
                    self.assertTrue(
                        any(
                            o.kind == "changed_region"
                            for unit in ledger.units
                            for o in unit.obligations
                        )
                    )
                    for buffer in ledger.buffers:
                        self.assertEqual(
                            reconstruct_buffer(
                                ledger, buffer.key, index=index
                            ),
                            (
                                before if buffer.side == "base" else after
                            ).encode(),
                        )


class LedgerIndexTests(unittest.TestCase):
    """Reuse validated maps without trusting copied IDs or mutable dictionaries."""

    def setUp(self) -> None:
        """Create a complete, small ledger and a valid result."""
        self.ledger = build_evidence_ledger(
            compile_snapshot(snap({"x.py": BEFORE}, {"x.py": AFTER}))
        )
        self.unit = next(u for u in self.ledger.units if u.change_key)
        self.result = authored_result(self.unit)

    def test_new_derivation_boundary_rejects_deep_mutation_and_index_copies(
        self,
    ) -> None:
        """The production boundary rechecks snapshots without weakening indexed citations."""
        index = plan_module.build_ledger_index(self.ledger)
        self.assertIs(index.validated_ledger(), self.ledger)
        with self.assertRaisesRegex(ValueError, "unregistered_ledger_index"):
            copy.copy(index).validated_ledger()
        span = next(iter(index.spans.values()))
        original = span.text
        object.__setattr__(span, "text", "tampered cached source")
        try:
            with self.assertRaisesRegex(ValueError, "ledger_index_mutated"):
                index.validated_ledger()
        finally:
            object.__setattr__(span, "text", original)
        self.assertIs(index.validated_ledger(), self.ledger)
        original_map = index.spans
        object.__setattr__(index, "spans", dict(original_map))
        try:
            with self.assertRaisesRegex(ValueError, "ledger_index_mutated"):
                index.validated_ledger()
        finally:
            object.__setattr__(index, "spans", original_map)
        validate_detail_result(
            self.result, self.unit, self.ledger, index=index
        )

    def test_index_binding_immutability_and_malicious_copies(self) -> None:
        """Reject wrong instances, changed bindings, altered units, and bad types."""
        index = plan_module.build_ledger_index(self.ledger)
        with self.assertRaises((AttributeError, TypeError)):
            index.spans = {}
        with self.assertRaises(TypeError):
            index.spans[self.unit.spans[0].id] = self.unit.spans[0]
        ref = span_reference(self.unit.spans[0], self.unit)
        for ledger in (
            self.ledger.model_copy(),
            self.ledger.model_copy(
                update={
                    "bindings": (
                        self.ledger.bindings[0].model_copy(
                            update={"start_line": 90}
                        ),
                        *self.ledger.bindings[1:],
                    )
                }
            ),
        ):
            with self.assertRaisesRegex(ValueError, "ledger_index_mismatch"):
                resolve_span_ref(ref, self.unit, ledger, index=index)
        for unit in (
            self.unit.model_copy(update={"obligations": ()}),
            self.unit.model_copy(
                update={
                    "spans": (
                        self.unit.spans[0].model_copy(
                            update={"line_start": True}
                        ),
                        *self.unit.spans[1:],
                    )
                }
            ),
        ):
            with self.assertRaises(ValueError):
                validate_detail_result(
                    self.result, unit, self.ledger, index=index
                )
        bad_ref = ref.model_copy(update={"byte_start": False})
        with self.assertRaises(ValidationError):
            resolve_span_ref(bad_ref, self.unit, self.ledger, index=index)
        damaged = self.ledger.model_copy(update={"compiler_test_runs": False})
        with self.assertRaises(ValidationError):
            plan_module.build_ledger_index(damaged)
        damaged = self.ledger.model_copy(
            update={
                "spans": (
                    self.ledger.spans[0].model_copy(
                        update={"line_start": True}
                    ),
                    *self.ledger.spans[1:],
                )
            }
        )
        with self.assertRaises(ValidationError):
            plan_module.build_ledger_index(damaged)

    def test_rebound_ledger_and_shifted_source_reject_stale_index(
        self,
    ) -> None:
        """Index binding uses instance and current fields, never only a digest."""
        index = plan_module.build_ledger_index(self.ledger)
        original_bindings = self.ledger.bindings
        rebound = (
            original_bindings[0].model_copy(update={"start_line": 90}),
            *original_bindings[1:],
        )
        # Even a caller that bypasses the frozen top-level setter cannot rebind
        # current source coordinates while retaining this index.
        object.__setattr__(self.ledger, "bindings", rebound)
        try:
            with self.assertRaisesRegex(ValueError, "ledger_index_mismatch"):
                validate_detail_result(
                    self.result, self.unit, self.ledger, index=index
                )
        finally:
            object.__setattr__(self.ledger, "bindings", original_bindings)
        shifted = build_evidence_ledger(
            compile_snapshot(
                snap(
                    {"x.py": BEFORE},
                    {"x.py": AFTER},
                    overrides={
                        (side, "x.py"): {
                            "state": "supplied",
                            "coverage": "partial",
                        }
                        for side in ("base", "head")
                    },
                    fragment_options={
                        (side, "x.py"): {"start_line": 90}
                        for side in ("base", "head")
                    },
                )
            )
        )
        ref = span_reference(self.unit.spans[0], self.unit)
        with self.assertRaisesRegex(ValueError, "ledger_index_mismatch"):
            resolve_span_ref(ref, self.unit, shifted, index=index)

    def test_factory_rejects_duplicate_missing_and_altered_ownership(
        self,
    ) -> None:
        """All former whole-ledger rejection checks also guard index construction."""
        ledger = self.ledger
        for update in (
            {"units": (*ledger.units, ledger.units[0])},
            {"spans": ledger.spans[:-1]},
            {"spans": (*ledger.spans, ledger.spans[0])},
            {"ownership": ledger.ownership[:-1]},
            {
                "groups": (
                    ledger.groups[0].model_copy(update={"unit_keys": ()}),
                    *ledger.groups[1:],
                )
            },
            {
                "units": (
                    ledger.units[0].model_copy(update={"obligations": ()}),
                    *ledger.units[1:],
                )
            },
            {
                "spans": (
                    ledger.spans[0].model_copy(
                        update={"content_digest": "f" * 64}
                    ),
                    *ledger.spans[1:],
                )
            },
        ):
            with (
                self.subTest(fields=tuple(update)),
                self.assertRaises(ValueError),
            ):
                plan_module.build_ledger_index(
                    ledger.model_copy(update=update)
                )

    def test_supplied_index_retains_all_detail_rejections(self) -> None:
        """Index reuse must reject corrupt citations and incomplete detail sets."""
        index = plan_module.build_ledger_index(self.ledger)
        first = self.result.explanations[0]
        ref = first.citations[0]
        for update in (
            {"side": "head" if ref.side == "base" else "base"},
            {"span_id": "e" * 64},
            {"unit_key": "d" * 64},
            {"group_key": "c" * 64},
            {"content_digest": "f" * 64},
            {"byte_end": ref.byte_end + 1},
            {"line_end": 5},
        ):
            with self.subTest(update=update), self.assertRaises(ValueError):
                validate_detail_result(
                    change_result(
                        self.result,
                        0,
                        citations=(
                            ref.model_copy(update=update),
                            *first.citations[1:],
                        ),
                    ),
                    self.unit,
                    self.ledger,
                    index=index,
                )
        with self.assertRaisesRegex(ValueError, "duplicate_citation"):
            validate_detail_result(
                change_result(
                    self.result, 0, citations=(*first.citations, ref)
                ),
                self.unit,
                self.ledger,
                index=index,
            )
        results = tuple(authored_result(u) for u in self.ledger.units)
        for bad in (results[:-1], (*results, results[0])):
            with self.assertRaisesRegex(ValueError, "detail_set_coverage"):
                validate_detail_results(bad, self.ledger, index=index)

    def test_index_rejects_known_cross_unit_and_group_citations(self) -> None:
        """Known global spans still require the cited unit's local ownership."""
        ledger = build_evidence_ledger(two_group_report(), max_unit_spans=2)
        index = plan_module.build_ledger_index(ledger)
        unit = ledger.units[0]
        other_group = next(
            u for u in ledger.units if u.group_key != unit.group_key
        )
        same_group = next(
            u
            for u in ledger.units
            if u.group_key == unit.group_key and u.id != unit.id
        )
        result = authored_result(unit)
        for other in (other_group, same_group):
            ref = span_reference(other.spans[0], other)
            observation = result.observations[0].model_copy(
                update={"citations": (ref,)}
            )
            with self.assertRaises(ValueError):
                validate_detail_result(
                    result.model_copy(update={"observations": (observation,)}),
                    unit,
                    ledger,
                    index=index,
                )
        ref = (
            result.observations[0]
            .citations[0]
            .model_copy(update={"byte_start": False})
        )
        observation = result.observations[0].model_copy(
            update={"citations": (ref,)}
        )
        with self.assertRaises(ValidationError):
            validate_detail_result(
                result.model_copy(update={"observations": (observation,)}),
                unit,
                ledger,
                index=index,
            )

    def test_large_set_builds_one_index_and_does_not_scan_per_citation(
        self,
    ) -> None:
        """Count global validation and equality work on thousands of source spans."""
        before = {
            f"module_{i}.py": "def f():\n"
            + "    value = 1\n" * 100
            + "    return 1\n"
            for i in range(64)
        }
        after = {
            path: text.replace("return 1", "return 2")
            for path, text in before.items()
        }
        ledger = build_evidence_ledger(compile_snapshot(snap(before, after)))
        self.assertGreater(len(ledger.spans), 10000)
        results = tuple(authored_result(u) for u in ledger.units)
        with mock.patch.object(
            EvidenceLedger,
            "model_validate",
            wraps=EvidenceLedger.model_validate,
        ) as deep:
            validate_detail_results(results, ledger)
            self.assertEqual(deep.call_count, 1)
        index = plan_module.build_ledger_index(ledger)
        unit = ledger.units[-1]
        ref = span_reference(unit.spans[-1], unit)
        equality_calls = 0
        unit_equality_calls = 0
        original_unit = EvidenceUnit.__eq__
        original = SourceSpan.__eq__

        def counted_unit(left: EvidenceUnit, right: object) -> bool:
            nonlocal unit_equality_calls
            unit_equality_calls += 1
            return original_unit(left, right)

        def counted(left: SourceSpan, right: object) -> bool:
            nonlocal equality_calls
            equality_calls += 1
            return original(left, right)

        with (
            mock.patch.object(
                EvidenceLedger,
                "model_validate",
                wraps=EvidenceLedger.model_validate,
            ) as deep,
            mock.patch.object(SourceSpan, "__eq__", counted),
            mock.patch.object(EvidenceUnit, "__eq__", counted_unit),
        ):
            for _ in range(100):
                resolve_span_ref(ref, unit, ledger, index=index)
            validate_detail_results(results, ledger, index=index)
            groups = planning_groups_from_details(ledger, results, index=index)
            validate_planning_claims(groups, results, ledger, index=index)
            for buffer in ledger.buffers:
                reconstruct_buffer(ledger, buffer.key, index=index)
            self.assertEqual(deep.call_count, 0)
        # Only comparisons inside the selected units are needed. A global scan
        # for each citation exceeds this bound by several orders of magnitude.
        self.assertLessEqual(unit_equality_calls, 100)
        self.assertLessEqual(equality_calls, 100 * len(unit.spans))


class IndexedDetailValidationTests(unittest.TestCase):
    """Keep indexed detail data owned and current-source checks scope-local."""

    @classmethod
    def setUpClass(cls) -> None:
        """Create one complete synthetic ledger and one checked index."""
        cls.ledger = build_evidence_ledger(
            compile_snapshot(snap({"x.py": BEFORE}, {"x.py": AFTER}))
        )
        cls.index = plan_module.build_ledger_index(cls.ledger)
        cls.results = tuple(authored_result(unit) for unit in cls.ledger.units)

    def capture(
        self,
        results: tuple[DetailResult | dict, ...] | None = None,
        *,
        scope: CitationResolutionScope | None = None,
        revision_binding: tuple[str, str] | None = None,
    ) -> object:
        """Capture the whole detail set with the same current index."""
        return capture_indexed_detail_results(
            self.results if results is None else results,
            self.ledger,
            index=self.index,
            citation_scope=scope,
            revision_binding=revision_binding,
        )

    def _assert_parity_failure(
        self,
        result: DetailResult,
        *,
        reason: str | None = None,
    ) -> None:
        unit = self.index.units[result.unit_key]
        strict_failure: Exception | None = None
        with self.assertRaises(Exception) as strict:
            validate_detail_result(result, unit, self.ledger, index=self.index)
        strict_failure = strict.exception
        candidates = tuple(
            result if original.unit_key == result.unit_key else original
            for original in self.results
        )
        with self.assertRaises(type(strict_failure)) as indexed:
            self.capture(candidates)
        if reason is not None:
            self.assertIn(reason, str(strict_failure))
            self.assertIn(reason, str(indexed.exception))
        else:
            self.assertEqual(str(indexed.exception), str(strict_failure))

    def test_capture_preserves_full_detail_and_expanded_acceptance_order(
        self,
    ) -> None:
        """Keep every detail field and allow the same reordered set as strict validation."""
        reversed_results = tuple(reversed(self.results))
        validate_detail_results(
            reversed_results, self.ledger, index=self.index
        )
        context = self.capture(reversed_results)
        self.assertEqual(
            context.unit_keys,
            tuple(result.unit_key for result in reversed_results),
        )
        for result in reversed_results:
            self.assertEqual(
                canonical_bytes(context.detail_record(result.unit_key)),
                canonical_bytes(result.model_dump(mode="python")),
            )
        self.assertTrue(context.contains_unit(reversed_results[0].unit_key))
        self.assertFalse(context.contains_unit("f" * 64))
        context.assert_current(self.ledger, self.index)
        self.assertFalse(hasattr(context, "validated_generation"))
        self.assertFalse(hasattr(context, "validated_report"))

    def test_missing_duplicate_and_reordered_coverage_match_strict_oracle(
        self,
    ) -> None:
        """Reject incomplete or repeated units and retain accepted input order."""
        for candidates in (
            self.results[:-1],
            (*self.results, self.results[0]),
        ):
            with self.subTest(count=len(candidates)):
                with self.assertRaisesRegex(
                    ValueError, "detail_set_coverage_mismatch"
                ):
                    validate_detail_results(
                        candidates, self.ledger, index=self.index
                    )
                with self.assertRaisesRegex(
                    ValueError, "detail_set_coverage_mismatch"
                ):
                    self.capture(candidates)
        context = self.capture(tuple(reversed(self.results)))
        self.assertEqual(
            context.unit_keys,
            tuple(item.unit_key for item in reversed(self.results)),
        )

    def test_citation_and_obligation_rejections_match_strict_oracle(
        self,
    ) -> None:
        """Keep source, citation-uniqueness, obligation, and limitation checks."""
        result = next(item for item in self.results if item.explanations)
        first = result.explanations[0]
        ref = first.citations[0]
        for update in (
            {"side": "head" if ref.side == "base" else "base"},
            {"path": "wrong.py"},
            {"content_digest": "f" * 64},
            {"byte_end": ref.byte_end + 1},
            {"line_end": ref.line_end + 1},
            {"group_key": "c" * 64},
        ):
            with self.subTest(update=update):
                damaged = change_result(
                    result,
                    0,
                    citations=(
                        ref.model_copy(update=update),
                        *first.citations[1:],
                    ),
                )
                self._assert_parity_failure(damaged)
        duplicate = change_result(
            result, 0, citations=(*first.citations, first.citations[0])
        )
        self._assert_parity_failure(duplicate, reason="duplicate_citation")
        missing_support = change_result(result, 0, citations=())
        self._assert_parity_failure(
            missing_support, reason="missing_changed_region_citation"
        )
        repeated_obligation = result.model_copy(
            update={"explanations": (*result.explanations, first)}
        )
        self._assert_parity_failure(repeated_obligation)

    def test_invalid_observations_questions_and_limits_match_strict_oracle(
        self,
    ) -> None:
        """Keep observation IDs, question links, and explicit limitations checked."""
        result = next(item for item in self.results if item.observations)
        observation = result.observations[0]
        duplicate_observation = result.model_copy(
            update={"observations": (*result.observations, observation)}
        )
        self._assert_parity_failure(
            duplicate_observation, reason="duplicate_observation"
        )
        question = result.questions[0]
        unlinked_question = result.model_copy(
            update={
                "questions": (
                    question.model_copy(
                        update={"citations": (), "obligation_ids": ()}
                    ),
                )
            }
        )
        self._assert_parity_failure(
            unlinked_question, reason="unlinked_question"
        )

        limited_ledger = build_evidence_ledger(
            compile_snapshot(
                snap(
                    {},
                    {"x.py": AFTER},
                    overrides={
                        ("base", "x.py"): {
                            "state": "unavailable",
                            "reason": "read_failed",
                        }
                    },
                )
            )
        )
        limited_index = plan_module.build_ledger_index(limited_ledger)
        limited_unit = next(
            unit
            for unit in limited_ledger.units
            if any(obligation.limitations for obligation in unit.obligations)
        )
        limited_result = authored_result(limited_unit)
        explanation_index = next(
            i
            for i, item in enumerate(limited_result.explanations)
            if item.limitations
        )
        omitted_limit = change_result(
            limited_result, explanation_index, limitations=()
        )
        with self.assertRaisesRegex(
            ValueError, "explicit_limitation_required"
        ):
            validate_detail_result(
                omitted_limit,
                limited_unit,
                limited_ledger,
                index=limited_index,
            )
        limited_results = tuple(
            omitted_limit
            if item.unit_key == limited_result.unit_key
            else authored_result(unit)
            for item, unit in zip(
                (authored_result(unit) for unit in limited_ledger.units),
                limited_ledger.units,
                strict=True,
            )
        )
        with self.assertRaisesRegex(
            ValueError, "explicit_limitation_required"
        ):
            capture_indexed_detail_results(
                limited_results, limited_ledger, index=limited_index
            )

    def test_cross_unit_and_cross_group_citations_are_rejected(self) -> None:
        """Global source presence cannot substitute for unit and group ownership."""
        ledger = build_evidence_ledger(two_group_report(), max_unit_spans=2)
        index = plan_module.build_ledger_index(ledger)
        unit = ledger.units[0]
        result = authored_result(unit)
        other_group = next(
            candidate
            for candidate in ledger.units
            if candidate.group_key != unit.group_key
        )
        same_group = next(
            candidate
            for candidate in ledger.units
            if candidate.group_key == unit.group_key
            and candidate.id != unit.id
        )
        for foreign in (other_group, same_group):
            bad_observation = result.observations[0].model_copy(
                update={
                    "citations": (span_reference(foreign.spans[0], foreign),)
                }
            )
            bad_result = result.model_copy(
                update={"observations": (bad_observation,)}
            )
            with self.assertRaises(ValueError) as strict:
                validate_detail_result(bad_result, unit, ledger, index=index)
            candidates = tuple(
                bad_result if item.id == unit.id else authored_result(item)
                for item in ledger.units
            )
            with self.assertRaises(type(strict.exception)) as indexed:
                capture_indexed_detail_results(candidates, ledger, index=index)
            self.assertEqual(str(indexed.exception), str(strict.exception))

    def test_owned_stage_capture_isolated_from_models_dicts_and_accessors(
        self,
    ) -> None:
        """Detach nested source values and return a fresh mapping on every read."""
        source_models = list(self.results)
        context = self.capture(tuple(source_models))
        source_result = next(
            item for item in source_models if item.explanations
        )
        source_ref = source_result.explanations[0].citations[0]
        original_start = source_ref.byte_start
        object.__setattr__(source_ref, "byte_start", True)
        try:
            current = context.detail_record(source_result.unit_key)
            self.assertIs(
                type(current["explanations"][0]["citations"][0]["byte_start"]),
                int,
            )
            current["explanations"][0]["citations"][0]["byte_start"] = True
            again = context.detail_record(source_result.unit_key)
            self.assertEqual(
                again["explanations"][0]["citations"][0]["byte_start"],
                original_start,
            )
        finally:
            object.__setattr__(source_ref, "byte_start", original_start)

        payloads = [item.model_dump(mode="python") for item in self.results]
        dict_context = self.capture(tuple(payloads))
        expected = dict_context.detail_record(self.results[0].unit_key)
        payloads[0]["unit_key"] = "f" * 64
        self.assertEqual(
            dict_context.detail_record(self.results[0].unit_key), expected
        )

    def test_capture_rejects_bypassed_numeric_booleans_without_serializing(
        self,
    ) -> None:
        """Strict typed capture rejects bool offsets instead of accepting dump coercion."""
        payloads = [item.model_dump(mode="python") for item in self.results]
        payload = next(item for item in payloads if item["explanations"])
        payload["explanations"][0]["citations"][0]["byte_start"] = True
        with self.assertRaises(ValidationError):
            self.capture(tuple(payloads))

    def test_scope_memo_counts_real_resolutions_and_enforces_lifecycle(
        self,
    ) -> None:
        """Reuse exact references locally and release all cached spans on close."""
        scope = CitationResolutionScope(
            self.ledger,
            self.index,
            revision_binding=("base-revision-1", "head-revision-1"),
        )

        def reference_key(ref: object) -> tuple[object, ...]:
            return (
                ref.span_id,
                ref.unit_key,
                ref.group_key,
                ref.side,
                ref.path,
                ref.byte_start,
                ref.byte_end,
                ref.line_start,
                ref.line_end,
                ref.content_digest,
            )

        distinct = {
            reference_key(ref)
            for result in self.results
            for refs in (
                *(item.citations for item in result.explanations),
                *(item.citations for item in result.observations),
                *(item.citations for item in result.questions),
            )
            for ref in refs
        }
        with (
            mock.patch.object(
                plan_module,
                "_validate_detail_result",
                wraps=plan_module._validate_detail_result,  # noqa: SLF001 # Count the real contextual validator.
            ) as details,
            mock.patch.object(
                plan_module,
                "_resolve_span_ref",
                wraps=plan_module._resolve_span_ref,  # noqa: SLF001 # Count actual source resolver calls.
            ) as resolutions,
            mock.patch.object(
                plan_module,
                "build_evidence_ledger",
                side_effect=AssertionError(
                    "indexed validation rebuilt the evidence ledger"
                ),
            ) as ledger_builds,
            mock.patch.object(
                plan_module,
                "build_ledger_index",
                side_effect=AssertionError(
                    "indexed capture rebuilt the ledger index"
                ),
            ) as builds,
        ):
            context = self.capture(
                scope=scope,
                revision_binding=("base-revision-1", "head-revision-1"),
            )
            self.assertEqual(details.call_count, len(self.results))
            self.assertEqual(resolutions.call_count, len(distinct))
            for _ in range(3):
                for unit_key in context.unit_keys:
                    self.assertTrue(context.contains_unit(unit_key))
                    context.detail_record(unit_key)
            self.assertEqual(details.call_count, len(self.results))
            self.assertEqual(resolutions.call_count, len(distinct))
            self.assertEqual(ledger_builds.call_count, 0)
            self.assertEqual(builds.call_count, 0)
        self.assertEqual(scope.resolution_count, len(distinct))
        self.assertEqual(scope.memoized_reference_count, len(distinct))
        with self.assertRaises(AttributeError):
            scope._index = self.index  # noqa: SLF001 # Rebinding must fail.
        with self.assertRaises(AttributeError):
            object.__setattr__(
                scope,
                "_revision_binding",
                ("base-rebound", "head-rebound"),
            )
        self.assertEqual(
            scope.revision_binding,
            ("base-revision-1", "head-revision-1"),
        )
        with self.assertRaises(TypeError):
            copy.copy(scope)
        with self.assertRaises(TypeError):
            copy.deepcopy(scope)
        with self.assertRaises(TypeError):
            pickle.dumps(scope)
        with self.assertRaises(TypeError):
            copy.copy(context)
        with self.assertRaises(TypeError):
            pickle.dumps(context)
        with self.assertRaises(AttributeError):
            context._index = self.index  # noqa: SLF001 # Rebinding must fail.
        scope.close()
        self.assertEqual(scope.memoized_reference_count, 0)
        with self.assertRaises(AttributeError):
            object.__setattr__(scope, "_closed", False)
        self.assertTrue(scope.closed)
        with self.assertRaisesRegex(ValueError, "unregistered_detail_context"):
            _ = context.unit_keys
        with self.assertRaisesRegex(ValueError, "citation_scope_closed"):
            scope.resolve(
                self.results[0].explanations[0].citations[0],
                self.index.units[self.results[0].unit_key],
                revision_binding=("base-revision-1", "head-revision-1"),
            )

    def test_scope_rejects_rebound_index_revision_and_mutated_sources(  # noqa: PLR0915 # Exercise current ledger and index mutations together.
        self,
    ) -> None:
        """Content-stable references cannot reuse another binding or revision."""
        revision = ("base-revision-1", "head-revision-1")
        scope = CitationResolutionScope(
            self.ledger, self.index, revision_binding=revision
        )
        result = next(item for item in self.results if item.explanations)
        unit = self.index.units[result.unit_key]
        reference = result.explanations[0].citations[0]
        indexed_span = self.index.spans[reference.span_id]
        original_line_start = indexed_span.line_start
        object.__setattr__(indexed_span, "line_start", True)
        try:
            with self.assertRaisesRegex(ValueError, "ledger_index_mutated"):
                CitationResolutionScope(
                    self.ledger, self.index, revision_binding=revision
                )
        finally:
            object.__setattr__(indexed_span, "line_start", original_line_start)

        resolved = scope.resolve(reference, unit, revision_binding=revision)
        original_span = self.index.spans[reference.span_id]
        self.assertEqual(resolved.line_start, original_span.line_start)
        object.__setattr__(resolved, "text", "changed detached result")
        self.assertNotEqual(
            scope.resolve(reference, unit, revision_binding=revision).text,
            "changed detached result",
        )
        with self.assertRaisesRegex(
            ValueError, "citation_scope_revision_binding_required"
        ):
            scope.resolve(reference, unit)
        with self.assertRaisesRegex(
            ValueError, "citation_scope_revision_mismatch"
        ):
            scope.resolve(
                reference,
                unit,
                revision_binding=("base-revision-2", "head-revision-1"),
            )
        with self.assertRaises(AttributeError):
            object.__setattr__(
                scope,
                "_revision_binding",
                ("base-revision-2", "head-revision-1"),
            )
        context = self.capture(scope=scope, revision_binding=revision)
        context.assert_current(
            self.ledger, self.index, revision_binding=revision
        )
        with self.assertRaisesRegex(
            ValueError, "citation_scope_revision_mismatch"
        ):
            context.assert_current(
                self.ledger,
                self.index,
                revision_binding=("base-revision-2", "head-revision-1"),
            )

        shifted_spans = tuple(
            span.model_copy(
                update={
                    "line_start": span.line_start + 89,
                    "line_end": span.line_end + 89,
                }
            )
            for span in self.ledger.spans
        )
        shifted_by_id = {span.id: span for span in shifted_spans}
        shifted_units = tuple(
            unit.model_copy(
                update={
                    "spans": tuple(
                        shifted_by_id[span.id] for span in unit.spans
                    )
                }
            )
            for unit in self.ledger.units
        )
        shifted_bindings = tuple(
            binding.model_copy(update={"start_line": binding.start_line + 89})
            for binding in self.ledger.bindings
        )
        shifted_ledger = self.ledger.model_copy(
            update={
                "spans": shifted_spans,
                "units": shifted_units,
                "bindings": shifted_bindings,
            }
        )
        shifted_index = plan_module.build_ledger_index(shifted_ledger)
        shifted_unit = shifted_index.units[result.unit_key]
        shifted_ref = span_reference(
            shifted_index.spans[reference.span_id], shifted_unit
        )
        self.assertEqual(reference, shifted_ref)
        with self.assertRaisesRegex(
            ValueError, "citation_scope_binding_mismatch"
        ):
            capture_indexed_detail_results(
                tuple(authored_result(item) for item in shifted_ledger.units),
                shifted_ledger,
                index=shifted_index,
                citation_scope=scope,
                revision_binding=revision,
            )
        fresh_scope = CitationResolutionScope(
            shifted_ledger, shifted_index, revision_binding=revision
        )
        shifted_span = fresh_scope.resolve(
            shifted_ref,
            shifted_unit,
            revision_binding=revision,
        )
        self.assertNotEqual(shifted_span.line_start, original_span.line_start)
        self.assertEqual(fresh_scope.resolution_count, 1)

        current_span = self.index.spans[reference.span_id]
        original_text = current_span.text
        object.__setattr__(current_span, "text", "tampered source")
        try:
            with self.assertRaisesRegex(ValueError, "ledger_index_mutated"):
                scope.resolve(reference, unit, revision_binding=revision)
        finally:
            object.__setattr__(current_span, "text", original_text)

        original_line_start = current_span.line_start
        object.__setattr__(current_span, "line_start", True)
        try:
            with self.assertRaisesRegex(ValueError, "ledger_index_mutated"):
                scope.resolve(reference, unit, revision_binding=revision)
        finally:
            object.__setattr__(current_span, "line_start", original_line_start)

        current_unit = self.index.units[unit.id]
        original_group_key = current_unit.group_key
        object.__setattr__(current_unit, "group_key", "f" * 64)
        try:
            with self.assertRaisesRegex(ValueError, "ledger_index_mutated"):
                scope.resolve(reference, unit, revision_binding=revision)
        finally:
            object.__setattr__(current_unit, "group_key", original_group_key)

        current_binding = self.index.bindings[current_span.buffer_key]
        original_start_line = current_binding.start_line
        object.__setattr__(current_binding, "start_line", True)
        try:
            with self.assertRaisesRegex(ValueError, "ledger_index_mutated"):
                scope.resolve(reference, unit, revision_binding=revision)
        finally:
            object.__setattr__(
                current_binding, "start_line", original_start_line
            )

        current_availability = self.ledger.availability[0]
        original_counterpart = current_availability.counterpart_path
        object.__setattr__(
            current_availability, "counterpart_path", "mutated.py"
        )
        try:
            with self.assertRaisesRegex(ValueError, "ledger_index_mutated"):
                scope.resolve(reference, unit, revision_binding=revision)
        finally:
            object.__setattr__(
                current_availability, "counterpart_path", original_counterpart
            )
        scope.close()
        fresh_scope.close()

        changed_revision = ("base-revision-2", "head-revision-1")
        changed_revision_scope = CitationResolutionScope(
            self.ledger, self.index, revision_binding=changed_revision
        )
        changed_revision_scope.resolve(
            reference, unit, revision_binding=changed_revision
        )
        self.assertEqual(changed_revision_scope.resolution_count, 1)
        changed_revision_scope.close()

    def test_raw_boolean_span_coordinates_cannot_enter_or_reuse_a_scope(
        self,
    ) -> None:
        """Reject bool coordinates before scope capture and after memo seeding."""
        revision = ("base-raw-1", "head-raw-1")
        result = next(item for item in self.results if item.explanations)
        unit = self.index.units[result.unit_key]
        reference = result.explanations[0].citations[0]
        span = self.index.spans[reference.span_id]
        original_line_start = span.line_start

        object.__setattr__(span, "line_start", True)
        try:
            with self.assertRaisesRegex(ValueError, "ledger_index_mutated"):
                CitationResolutionScope(
                    self.ledger, self.index, revision_binding=revision
                )
        finally:
            object.__setattr__(span, "line_start", original_line_start)

        scope = CitationResolutionScope(
            self.ledger, self.index, revision_binding=revision
        )
        resolved = scope.resolve(reference, unit, revision_binding=revision)
        self.assertIs(type(resolved.line_start), int)
        object.__setattr__(span, "line_start", True)
        try:
            with self.assertRaisesRegex(ValueError, "ledger_index_mutated"):
                scope.resolve(reference, unit, revision_binding=revision)
        finally:
            object.__setattr__(span, "line_start", original_line_start)
            scope.close()

    def test_scope_owner_pins_revision_and_cannot_be_reopened(self) -> None:
        """Context reuse stays on factory revisions and closed owner state."""
        revision = ("base-owner-1", "head-owner-1")
        changed = ("base-owner-2", "head-owner-1")
        scope = CitationResolutionScope(
            self.ledger, self.index, revision_binding=revision
        )
        result = next(item for item in self.results if item.explanations)
        unit = self.index.units[result.unit_key]
        reference = result.explanations[0].citations[0]
        scope.resolve(reference, unit, revision_binding=revision)

        with self.assertRaises(AttributeError):
            object.__setattr__(scope, "_revision_binding", changed)
        self.assertEqual(scope.revision_binding, revision)
        context = self.capture(scope=scope, revision_binding=revision)
        context.assert_current(
            self.ledger, self.index, revision_binding=revision
        )
        with self.assertRaisesRegex(
            ValueError, "citation_scope_revision_mismatch"
        ):
            context.assert_current(
                self.ledger, self.index, revision_binding=changed
            )
        with self.assertRaisesRegex(
            ValueError, "citation_scope_revision_mismatch"
        ):
            scope.resolve(reference, unit, revision_binding=changed)

        scope.close()
        with self.assertRaises(AttributeError):
            object.__setattr__(scope, "_closed", False)
        with self.assertRaisesRegex(ValueError, "citation_scope_closed"):
            scope.resolve(reference, unit, revision_binding=revision)

    def test_public_expanded_validator_does_not_consume_indexed_context(
        self,
    ) -> None:
        """Public strict validation remains an independent boundary and oracle."""
        result = next(item for item in self.results if item.explanations)
        unit = self.index.units[result.unit_key]
        context = self.capture()
        with self.assertRaises(ValidationError):
            validate_detail_result(
                context, unit, self.ledger, index=self.index
            )
        expected_calls = sum(
            len(refs)
            for refs in (
                *(item.citations for item in result.explanations),
                *(item.citations for item in result.observations),
                *(item.citations for item in result.questions),
            )
        )
        with mock.patch.object(
            plan_module,
            "_resolve_span_ref",
            wraps=plan_module._resolve_span_ref,  # noqa: SLF001 # Count strict expanded resolutions.
        ) as strict_resolutions:
            validate_detail_result(result, unit, self.ledger, index=self.index)
        self.assertEqual(strict_resolutions.call_count, expected_calls)


class IndexedGraphDetailTests(unittest.TestCase):
    """Exercise the indexed adapter against a real accepted report graph."""

    @classmethod
    def setUpClass(cls) -> None:
        """Use one locally generated report graph for the graph adapter tests."""
        from test_report_storage import ReportStorageTests

        ReportStorageTests.setUpClass()
        fixture = ReportStorageTests(
            "test_expanded_and_compact_generation_inputs_have_exact_oracles"
        )
        cls.graph = fixture._typed_report_graph()  # noqa: SLF001 # Reuse the accepted synthetic graph fixture.
        cls.ledger = build_evidence_ledger(ReportStorageTests.report)
        cls.index = plan_module.build_ledger_index(cls.ledger)
        cls.expanded_details = ReportStorageTests.expanded_report[
            "generation"
        ]["details"]

    def test_graph_detail_validation_avoids_expansion_and_reuses_unique_work(
        self,
    ) -> None:
        """Validate graph addresses directly and keep returned data detached."""
        graph = self.graph
        ledger = self.ledger
        index = self.index
        references = graph.generation_result_refs["details"]
        expected_refs = {
            canonical_bytes(reference)
            for result_ref in references
            for reference in (
                item
                for group in (graph.result_record(result_ref)["explanations"],)
                for explanation in group
                for item in explanation["citations"]
            )
        }
        expected_refs.update(
            canonical_bytes(reference)
            for result_ref in references
            for section in (
                graph.result_record(result_ref).get("observations", []),
                graph.result_record(result_ref).get("questions", []),
            )
            for item in section
            for reference in item["citations"]
        )
        import diffstory.report_storage as storage_module

        expected_details = {
            detail["unit_key"]: detail for detail in self.expanded_details
        }

        with (
            mock.patch.object(
                plan_module,
                "_validate_detail_result",
                wraps=plan_module._validate_detail_result,  # noqa: SLF001 # Count the real contextual validator.
            ) as details,
            mock.patch.object(
                plan_module,
                "_resolve_span_ref",
                wraps=plan_module._resolve_span_ref,  # noqa: SLF001 # Count actual source resolver calls.
            ) as resolutions,
            mock.patch.object(
                plan_module,
                "build_evidence_ledger",
                side_effect=AssertionError(
                    "indexed validation rebuilt the evidence ledger"
                ),
            ) as ledger_builds,
            mock.patch.object(
                plan_module,
                "build_ledger_index",
                side_effect=AssertionError(
                    "indexed validation rebuilt the index"
                ),
            ) as builds,
            mock.patch.object(
                DetailResult,
                "model_validate",
                wraps=DetailResult.model_validate,
            ) as structural,
            mock.patch.object(
                storage_module,
                "expand_report_storage",
                side_effect=AssertionError(
                    "indexed validation expanded the report"
                ),
            ) as expand,
            mock.patch.object(
                storage_module,
                "encode_report_storage",
                side_effect=AssertionError(
                    "indexed validation encoded the report"
                ),
            ) as encode,
            mock.patch.object(
                storage_module,
                "parse_report_storage",
                side_effect=AssertionError(
                    "indexed validation parsed the report"
                ),
            ) as parse,
            mock.patch.object(
                storage_module.ReportStorageView,
                "to_bytes",
                side_effect=AssertionError(
                    "indexed validation serialized the graph"
                ),
            ) as to_bytes,
        ):
            context = validate_indexed_detail_results(
                tuple(reversed(references)), graph, ledger, index=index
            )
            self.assertEqual(
                context.references,
                tuple(reversed(references)),
            )
            self.assertEqual(details.call_count, len(references))
            self.assertEqual(structural.call_count, len(references))
            self.assertEqual(resolutions.call_count, len(expected_refs))
            self.assertEqual(
                context.citation_scope.resolution_count, len(expected_refs)
            )
            self.assertEqual(ledger_builds.call_count, 0)
            self.assertEqual(builds.call_count, 0)
            for operation in (expand, encode, parse, to_bytes):
                self.assertEqual(operation.call_count, 0)

        for unit_key in context.unit_keys:
            self.assertEqual(
                canonical_bytes(context.detail_record(unit_key)),
                canonical_bytes(expected_details[unit_key]),
            )

        first_unit = context.unit_keys[0]
        first_reference = context.references[0]
        original_reference = next(reversed(references))
        object.__setattr__(first_reference, "storage_id", "f" * 64)
        self.assertEqual(context.references[0], original_reference)
        first = context.detail_record(first_unit)
        first["explanations"].clear()
        self.assertTrue(context.detail_record(first_unit)["explanations"])
        with self.assertRaisesRegex(
            ValueError, "detail_set_coverage_mismatch"
        ):
            validate_indexed_detail_results(
                references[:-1], graph, ledger, index=index
            )
        with self.assertRaisesRegex(
            ValueError, "detail_set_coverage_mismatch"
        ):
            validate_indexed_detail_results(
                (*references, references[0]), graph, ledger, index=index
            )
        context.citation_scope.close()

    def test_graph_capture_rejects_mutated_addresses(self) -> None:
        """A caller-mutated storage address cannot select another record."""
        graph = self.graph
        ledger = self.ledger
        index = self.index
        references = graph.generation_result_refs["details"]
        reference = references[0]
        old_id = reference.storage_id
        object.__setattr__(reference, "storage_id", "f" * 64)
        try:
            with self.assertRaisesRegex(
                ValueError, "detail_set_coverage_mismatch"
            ):
                validate_indexed_detail_results(
                    references, graph, ledger, index=index
                )
        finally:
            object.__setattr__(reference, "storage_id", old_id)

    def test_context_owns_graph_view_when_caller_rebinds_tables(self) -> None:
        """Rebinding the caller's graph tables cannot change captured details."""
        import diffstory.report_storage as storage_module

        graph = self.graph
        references = graph.generation_result_refs["details"]
        reference = next(
            item
            for item in references
            if graph.result_record(item)["explanations"]
        )
        unit_key = reference.item_key
        context = validate_indexed_detail_results(
            references, graph, self.ledger, index=self.index
        )
        expected = context.detail_record(unit_key)
        original_tables = graph._tables  # noqa: SLF001 # Preserve the fixture's immutable source view.
        try:
            changed_record = graph.record("results", reference.storage_id)
            changed_record["explanations"][0]["before"] = (
                "changed after context capture"
            )
            changed_tables = dict(original_tables)
            changed_results = dict(original_tables["results"])
            changed_entry = dict(changed_results[reference.storage_id])
            changed_entry["record"] = storage_module._freeze(  # noqa: SLF001 # Rebuild a frozen synthetic caller view.
                changed_record
            )
            changed_results[reference.storage_id] = MappingProxyType(
                changed_entry
            )
            changed_tables["results"] = MappingProxyType(changed_results)
            object.__setattr__(
                graph, "_tables", MappingProxyType(changed_tables)
            )

            self.assertNotEqual(
                graph.result_record(reference)["explanations"][0]["before"],
                expected["explanations"][0]["before"],
            )
            self.assertIsNot(context._graph, graph)  # noqa: SLF001 # The context owns its top-level graph view.
            self.assertEqual(context.detail_record(unit_key), expected)
            context.assert_current(self.ledger, self.index)
        finally:
            object.__setattr__(graph, "_tables", original_tables)
            context.citation_scope.close()

    def test_arena_detail_binding_reuses_structure_and_keeps_owned_rows(
        self,
    ) -> None:
        """Reject structural reparsing after early arena detail capture."""
        import copy
        import sys

        from diffstory.narrative_plan import (
            _capture_indexed_arena_detail_results,
        )
        from diffstory.report_storage import ResultRef
        from diffstory.report_storage import _RecordArena

        caller_details = copy.deepcopy(self.expanded_details)
        arena = _RecordArena()
        validator_calls = Counter()
        trace_span_validator = _span_validator_profiler(validator_calls)
        previous_profile = sys.getprofile()
        sys.setprofile(trace_span_validator)
        capture_validator_calls = 0
        with (
            mock.patch.object(
                DetailResult,
                "model_validate_json",
                wraps=DetailResult.model_validate_json,
            ) as detail_parses,
            mock.patch.object(
                DetailResult,
                "model_validate",
                side_effect=AssertionError("arena adapter reparsed a detail"),
            ),
            mock.patch.object(
                SpanRef,
                "model_validate",
                side_effect=AssertionError(
                    "arena adapter reparsed a span ref"
                ),
            ),
            mock.patch.object(
                plan_module,
                "_validate_detail_result",
                wraps=plan_module._validate_detail_result,  # noqa: SLF001 # Count full contextual checks.
            ) as contextual_checks,
            mock.patch.object(
                plan_module,
                "_resolve_span_ref",
                wraps=plan_module._resolve_span_ref,  # noqa: SLF001 # Count actual source reads.
            ) as source_resolutions,
        ):
            try:
                references = tuple(
                    arena.put_result(item) for item in caller_details
                )
                capture_validator_calls = validator_calls["span_ref_ranges"]
                context = _capture_indexed_arena_detail_results(
                    references, arena, self.ledger, index=self.index
                )
            finally:
                sys.setprofile(previous_profile)

        self.assertEqual(detail_parses.call_count, len(references))
        self.assertEqual(contextual_checks.call_count, len(references))
        self.assertGreater(capture_validator_calls, 0)
        self.assertEqual(
            validator_calls["span_ref_ranges"], capture_validator_calls
        )
        self.assertGreaterEqual(
            arena.typed_validation_counts.get("evidence_refs:span_ref", 0), 1
        )
        self.assertEqual(
            source_resolutions.call_count,
            context.citation_scope.resolution_count,
        )
        self.assertGreater(source_resolutions.call_count, 0)
        self.assertEqual(tuple(context.references), references)
        self.assertEqual(context._graph, None)  # noqa: SLF001 # Arena ownership has no complete graph input.
        self.assertEqual(dict(context._typed_by_unit), {})  # noqa: SLF001 # The context stores no detail model map.
        self.assertEqual(arena._builder.typed_results, {})  # noqa: SLF001 # Capture releases parsed detail models.
        self.assertTrue(
            all(
                row["kind"] == "detail"
                for row in context._arena_reader._tables["results"].values()  # noqa: SLF001 # Verify the early shallow reader contains details only.
            )
        )
        for original in self.expanded_details:
            self.assertEqual(
                canonical_bytes(context.detail_record(original["unit_key"])),
                canonical_bytes(original),
            )
        caller_details[0]["unit_key"] = "0" * 64
        selected_unit = self.expanded_details[0]["unit_key"]
        detached = context.detail_record(selected_unit)
        detached["explanations"].clear()
        self.assertTrue(context.detail_record(selected_unit)["explanations"])

        altered_references = (
            (
                "storage address changed",
                ResultRef(
                    references[0].table,
                    references[0].kind,
                    "f" * 64,
                    references[0].item_key,
                ),
            ),
            (
                "item binding changed",
                ResultRef(
                    references[0].table,
                    references[0].kind,
                    references[0].storage_id,
                    "f" * 64,
                ),
            ),
        )
        for concern, altered_reference in altered_references:
            with (
                self.subTest(concern=concern),
                self.assertRaisesRegex(
                    ValueError, "foreign_storage_result_reference"
                ),
            ):
                _capture_indexed_arena_detail_results(
                    (altered_reference, *references[1:]),
                    arena,
                    self.ledger,
                    index=self.index,
                )
        context.citation_scope.close()
        with self.assertRaisesRegex(ValueError, "unregistered_detail_context"):
            _ = context.unit_keys

    def test_arena_detail_context_rejects_private_backing_rebinds(
        self,
    ) -> None:
        """Reject selectors or reader indexes rebound after scope registration."""
        import copy

        from diffstory.narrative_plan import (
            _capture_indexed_arena_detail_results,
        )
        from diffstory.report_storage import _RecordArena

        arena = _RecordArena()
        references = tuple(
            arena.put_result(copy.deepcopy(item))
            for item in self.expanded_details
        )
        context = _capture_indexed_arena_detail_results(
            references, arena, self.ledger, index=self.index
        )
        reader = context._arena_reader  # noqa: SLF001 # Exercise registered private backing identities.
        cases = (
            ("unit-to-result selectors", context, "_references_by_unit"),
            ("arena table index", reader, "_tables"),
            ("arena item-key index", reader, "_result_ids"),
        )
        for concern, owner, attribute in cases:
            with self.subTest(concern=concern):
                original = getattr(owner, attribute)
                object.__setattr__(owner, attribute, MappingProxyType({}))
                with self.assertRaisesRegex(
                    ValueError, "unregistered_detail_context"
                ):
                    _ = context.unit_keys
                object.__setattr__(owner, attribute, original)
                self.assertEqual(
                    context.unit_keys,
                    tuple(item["unit_key"] for item in self.expanded_details),
                )
        context.citation_scope.close()

    def test_arena_rejects_raw_bool_reference_before_ownership(self) -> None:
        """Reject bool values where a source citation requires a natural number."""
        import copy

        from diffstory.report_storage import _RecordArena

        candidate = copy.deepcopy(
            next(
                item
                for item in self.expanded_details
                if any(
                    entry.get("citations", ())
                    for family in (
                        "explanations",
                        "observations",
                        "questions",
                    )
                    for entry in item[family]
                )
            )
        )
        reference = next(
            citation
            for family in ("explanations", "observations", "questions")
            for entry in candidate[family]
            for citation in entry.get("citations", ())
        )
        reference["byte_start"] = True
        arena = _RecordArena()
        with self.assertRaises(ValidationError):
            arena.put_result(candidate)
        self.assertEqual(sum(arena.retained_record_counts.values()), 0)


class ProseInputFixture(Contract):
    """Complete immutable request atoms containing already-produced child prose."""

    key: Digest
    group_key: Digest
    statements: tuple[SupportedStatement, ...]


class PublicProsePackingTests(unittest.TestCase):
    """Use actual child text to exercise request count, order, and hard bounds."""

    def setUp(self) -> None:
        """Create immutable inputs whose text is retained by the serializer."""
        self.items = tuple(
            ProseInputFixture(
                key=semantic_digest("section-context", i),
                group_key=semantic_digest("presentation", i),
                statements=(
                    SupportedStatement(
                        kind="editorial",
                        text=f"Read section {i} before its dependent callers.",
                    ),
                ),
            )
            for i in range(40)
        )
        self.capacity = ModelCapacity(1_000_000, 100_000)
        self.limits = RequestPackingLimits(
            repair_output_tokens=0, timeout_seconds=10_000
        )
        self.options = {
            "stage": "section",
            "keys": lambda item: item.key,
            "groups": lambda item: (item.group_key,),
            "serializer": body,
            "allowance": lambda items: 300 * len(items),
            "capacity": self.capacity,
            "limits": self.limits,
        }

    def test_two_and_many_sections_and_cards_use_one_actual_request(
        self,
    ) -> None:
        """Presentation count does not impose a hidden per-section request floor."""
        for count in (2, 40):
            for stage in ("section", "planning"):
                with self.subTest(count=count, stage=stage):
                    batches = pack_request_items(
                        self.items[:count], **(self.options | {"stage": stage})
                    )
                    self.assertEqual(len(batches), 1)
                    self.assertEqual(len(batches[0].group_keys), count)
                    self.assertEqual(
                        set(batches[0].item_keys),
                        {item.key for item in self.items[:count]},
                    )
                    if stage == "section":
                        self.assertEqual(
                            batches[0].item_keys,
                            tuple(item.key for item in self.items[:count]),
                        )

    def test_output_bytes_and_time_give_different_adjacent_partitions(
        self,
    ) -> None:
        """Practical byte and timeout limits bind even with a large context."""
        cases = (
            (ModelCapacity(100_000, 1000), self.limits, (3, 3, 1)),
            (
                self.capacity,
                RequestPackingLimits(
                    repair_output_tokens=0, response_byte_limit=4800
                ),
                (2, 2, 2, 1),
            ),
            (
                self.capacity,
                RequestPackingLimits(
                    repair_output_tokens=0,
                    output_tokens_per_second=100,
                    fixed_latency_seconds=0,
                    timeout_seconds=3,
                ),
                (1,) * 7,
            ),
        )
        for capacity, limits, expected in cases:
            with self.subTest(expected=expected):
                batches = pack_request_items(
                    self.items[:7],
                    **(
                        self.options | {"capacity": capacity, "limits": limits}
                    ),
                )
                self.assertEqual(
                    tuple(len(batch.item_keys) for batch in batches), expected
                )
                self.assertEqual(
                    tuple(key for batch in batches for key in batch.item_keys),
                    tuple(item.key for item in self.items[:7]),
                )

    def test_unordered_card_partitions_are_stable_without_text_slicing(
        self,
    ) -> None:
        """Stable digest partitions are independent of arrival order."""
        options = self.options | {
            "stage": "planning",
            "capacity": ModelCapacity(100_000, 1000),
        }
        first = pack_request_items(self.items, **options)
        reverse = pack_request_items(tuple(reversed(self.items)), **options)
        self.assertEqual(first, reverse)
        self.assertEqual(
            {key for batch in first for key in batch.item_keys},
            {item.key for item in self.items},
        )

    def test_actual_repair_preparation_retains_prefix_schema_and_targets(
        self,
    ) -> None:
        """The exact returned body, including repair fields, is the send authority."""
        limits = RequestPackingLimits(
            framing_bytes=13,
            schema_reserve_bytes=17,
            repair_reserve_bytes=19,
            repair_output_tokens=23,
            safety_tokens=29,
        )

        def repair(items: tuple, output: int) -> bytes:
            return b"PROVIDER PREFIX\n" + canonical_bytes(
                {
                    "reason": "missing_item",
                    "schema": {"items": "complete"},
                    "items": items,
                    "output": output,
                }
            )

        prepared = prepare_request_items(
            (self.items[3],),
            **(self.options | {"serializer": repair, "limits": limits}),
        )
        self.assertTrue(prepared.body.startswith(b"PROVIDER PREFIX\n"))
        payload = json.loads(prepared.body.split(b"\n", 1)[1])
        self.assertEqual(
            payload["items"], [self.items[3].model_dump(mode="json")]
        )
        self.assertEqual(payload["reason"], "missing_item")
        self.assertEqual(prepared.batch.output_reservation, 323)
        self.assertEqual(
            prepared.batch.input_bound, len(prepared.body) + 1024 + 49
        )
        self.assertEqual(
            prepared.batch.request_digest,
            plan_module.content_digest(prepared.body),
        )
        self.assertNotIn("body", prepared.model_dump())
        self.assertEqual(prepared.batch.item_keys, (self.items[3].key,))

    def test_actual_child_growth_and_atomic_overflow_fail_before_send(
        self,
    ) -> None:
        """Produced child prose can require another hierarchy level; no preview wins."""
        capacity = ModelCapacity(8000, 1000)
        options = self.options | {"stage": "planning", "capacity": capacity}
        small = self.items[:2]
        self.assertEqual(len(pack_request_items(small, **options)), 1)
        large = ProseInputFixture(
            key=small[0].key,
            group_key=small[0].group_key,
            statements=(
                SupportedStatement(
                    kind="editorial", text="complete child statement " * 1000
                ),
            ),
        )
        with self.assertRaisesRegex(PackingError, "input_limit"):
            pack_request_items((large, small[1]), **options)
        self.assertEqual(len(large.statements[0].text), 25_000)

    def test_public_preparer_keeps_codex_guard_and_rejects_invalid_atoms(
        self,
    ) -> None:
        """Only an enforceable direct API cap can reserve less than full output."""
        options = self.options | {"capacity": ModelCapacity(20_000, 18_000)}
        with self.assertRaisesRegex(PackingError, "input_limit"):
            prepare_request_items(self.items[:1], **options)
        prepared = prepare_request_items(
            self.items[:1], **(options | {"enforceable_output_cap": True})
        )
        self.assertEqual(prepared.batch.output_guard, 300)
        for items in (
            (),
            (self.items[0], self.items[0]),
            [self.items[0]],
            ({"key": self.items[0].key},),
        ):
            with (
                self.subTest(shape=type(items).__name__),
                self.assertRaises(ValueError),
            ):
                prepare_request_items(items, **self.options)
        for allowance in (0, -1, True, "300"):
            with (
                self.subTest(allowance=allowance),
                self.assertRaises(ValueError),
            ):
                prepare_request_items(
                    self.items[:1],
                    **(
                        self.options
                        | {"allowance": lambda _items, value=allowance: value}
                    ),
                )

    def test_actual_repacking_preserves_structural_plan_identity_and_closure(
        self,
    ) -> None:
        """Replacing request bytes cannot drop deterministic members, claims, or terms."""
        groups = graph_groups(3, claims=2)
        plan = build_story_plan(
            groups,
            self.capacity,
            self.limits,
            body,
            context_digest=semantic_digest("trusted-context"),
        )

        def produced(cards: tuple, output: int) -> bytes:
            return canonical_bytes(
                {
                    "cards": cards,
                    "output": output,
                    "validated_child_prose": "All complete produced child statements are included here.",
                }
            )

        requests = pack_request_items(
            plan.cards,
            stage="planning",
            keys=lambda card: card.id,
            groups=lambda card: tuple(
                dict.fromkeys(atom.group_key for atom in card.atoms)
            ),
            serializer=produced,
            allowance=lambda cards: 1000 * len(cards),
            capacity=self.capacity,
            limits=self.limits,
        )
        repacked = type(plan).model_validate(
            plan.model_dump() | {"requests": requests}
        )
        validate_story_plan(repacked, groups)
        self.assertEqual(repacked.id, plan.id)
        self.assertNotEqual(repacked.requests, plan.requests)
        self.assertEqual(repacked.groups, groups)
        self.assertEqual(repacked.cards, plan.cards)
        self.assertTrue(
            all(
                group.terminology and group.claims for group in repacked.groups
            )
        )

    def test_bad_public_key_is_rejected_before_sizing_or_serialization(
        self,
    ) -> None:
        """A caller's mistaken key callback must not echo source in a size error."""
        serialize = mock.Mock(return_value=b"unused")
        options = self.options | {
            "keys": lambda _item: "PRIVATE_KEY_SENTINEL",
            "serializer": serialize,
            "allowance": lambda _items: 1_000_000,
        }
        for operation in (prepare_request_items, pack_request_items):
            with self.assertRaisesRegex(ValueError, "^invalid_request_key$"):
                operation(self.items[:1], **options)
        serialize.assert_not_called()


if __name__ == "__main__":
    unittest.main()
