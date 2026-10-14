"""Current-data production derivation, bounded handoffs and adversarial authority."""

from __future__ import annotations

import copy
import json
import pickle
import unittest
from collections import defaultdict
from unittest import mock

from test_analysis import snap
from test_narrative_plan import AFTER
from test_narrative_plan import BEFORE
from test_narrative_plan import authored_result
from test_narrative_v2 import FixtureProvider
from test_narrative_v2 import golden_report

from diffstory.analysis import compile_snapshot
from diffstory.narrative_contracts import AuthorContext
from diffstory.narrative_contracts import BatchSpec
from diffstory.narrative_contracts import DetailResult
from diffstory.narrative_contracts import FactualObservation
from diffstory.narrative_contracts import PlanningBounds
from diffstory.narrative_contracts import PlanningCardResult
from diffstory.narrative_contracts import ProseBinding
from diffstory.narrative_contracts import SourceSliceRef
from diffstory.narrative_contracts import StatementPath
from diffstory.narrative_contracts import StoryPlan
from diffstory.narrative_contracts import SupportedStatement
from diffstory.narrative_contracts import SupportSet
from diffstory.narrative_hierarchy import IndexedPlanningSources
from diffstory.narrative_hierarchy import PlanningShape
from diffstory.narrative_hierarchy import PlanningSources
from diffstory.narrative_hierarchy import _fragment
from diffstory.narrative_hierarchy import _indexed_field
from diffstory.narrative_hierarchy import atom_key
from diffstory.narrative_hierarchy import build_production_plan
from diffstory.narrative_hierarchy import derive_planning_sources
from diffstory.narrative_hierarchy import derive_planning_sources_indexed
from diffstory.narrative_hierarchy import fragment_material
from diffstory.narrative_hierarchy import inherited_support
from diffstory.narrative_hierarchy import material_support
from diffstory.narrative_hierarchy import planning_shapes
from diffstory.narrative_hierarchy import planning_waves
from diffstory.narrative_hierarchy import production_plan_id
from diffstory.narrative_hierarchy import record_key
from diffstory.narrative_hierarchy import record_material
from diffstory.narrative_hierarchy import source_reference
from diffstory.narrative_hierarchy import statement_reference
from diffstory.narrative_hierarchy import validate_consumed_support
from diffstory.narrative_hierarchy import validate_production_plan
from diffstory.narrative_hierarchy import validate_production_requests
from diffstory.narrative_hierarchy import validate_source_slice
from diffstory.narrative_hierarchy import validate_statement_reference
from diffstory.narrative_hierarchy import validate_support_chains
from diffstory.narrative_hierarchy import validate_support_forest
from diffstory.narrative_plan import IndexedDetailContext
from diffstory.narrative_plan import PackingError
from diffstory.narrative_plan import build_evidence_ledger
from diffstory.narrative_plan import build_ledger_index
from diffstory.narrative_plan import canonical_bytes
from diffstory.narrative_plan import capture_indexed_detail_results
from diffstory.narrative_plan import content_digest
from diffstory.narrative_plan import planning_atoms
from diffstory.narrative_plan import resolve_span_ref
from diffstory.narrative_plan import semantic_digest
from diffstory.narrative_plan import validate_story_plan
from diffstory.narrative_v2 import ProductionInputs
from diffstory.narrative_v2 import StageInputs
from diffstory.narrative_v2 import Transport
from diffstory.narrative_v2 import V2Options
from diffstory.narrative_wire import wire_schema

BOUNDS = PlanningBounds(
    card_wire_bytes=1024,
    card_support_bytes=2048,
    prose_wire_bytes=4096,
    output_guard=15000,
)
CONTEXT = semantic_digest("current normalized fixture context")


class SizingSerializer:
    """Measure JSON escaping, real schemas, framing, repair and output guard."""

    def __init__(self, capacity: int = 40000) -> None:
        """Keep only sizes; this fixture has no provider/send capability."""
        self.capacity = capacity
        self.sizes: list[int] = []
        self.guards: set[int] = set()
        self.schemas = {
            stage: wire_schema(stage)
            for stage in ("planning", "section", "document")
        }

    def __call__(self, shape: PlanningShape) -> None:
        """Raise the planner's measured-capacity signal only on real size failure."""
        self.guards.add(shape.output_guard)
        body = b"fixture framing\n" + canonical_bytes(
            {
                "schema": self.schemas[shape.stage],
                "instructions": "Use only supplied supported statements. Partial fields are continuations. Preserve author provenance.",
                "data": shape.sizing_payload(),
                "repair": {
                    "item": shape.subject_key,
                    "reason": "invalid_citation",
                },
                "output_guard": shape.output_guard,
                "result_wire_budget": shape.budget.result_wire_bytes,
            }
        )
        self.sizes.append(len(body))
        if len(body) + shape.output_guard + 512 > self.capacity:
            msg = "input_limit"
            raise PackingError(msg, shape.subject_key)


def fixture(
    report: dict | None = None,
    *,
    author: AuthorContext | None = None,
    max_span_bytes: int = 4096,
    max_unit_spans: int = 16,
) -> tuple:
    """Derive source facts and details from the current trusted compiler fixture."""
    ledger = build_evidence_ledger(
        report or golden_report(1),
        max_span_bytes=max_span_bytes,
        max_unit_spans=max_unit_spans,
    )
    index = build_ledger_index(ledger)
    details = tuple(authored_result(u) for u in index.units.values())
    sources = derive_planning_sources(
        index, details, author_context=author or AuthorContext()
    )
    return ledger, index, details, sources


def build(
    sources: PlanningSources,
    *,
    bounds: PlanningBounds = BOUNDS,
    check: SizingSerializer | None = None,
) -> StoryPlan:
    """Use the same declared caps for normal and oversized fixture inputs."""
    return build_production_plan(
        sources,
        context_digest=CONTEXT,
        bounds=bounds,
        check_shape=check or SizingSerializer(),
    )


def changed_detail(
    details: tuple[DetailResult, ...], text: str
) -> tuple[DetailResult, ...]:
    """Change one accepted field without changing any source/obligation coverage."""
    first = details[0]
    explanation = type(first.explanations[0]).model_validate(
        first.explanations[0].model_dump() | {"before": text}
    )
    return (
        DetailResult.model_validate(
            first.model_dump()
            | {"explanations": (explanation, *first.explanations[1:])}
        ),
        *details[1:],
    )


def rehash(plan: StoryPlan, changes: dict) -> StoryPlan:
    """Rehash a forged persisted hierarchy without using it as expected facts."""
    production = plan.production.model_dump(mode="json") | changes
    replacements = {}

    def replace(value: object) -> object:
        if isinstance(value, str):
            return replacements.get(value, value)
        if isinstance(value, list):
            return [replace(item) for item in value]
        if isinstance(value, dict):
            result = {key: replace(item) for key, item in value.items()}
            if "relationship_key" in result:
                result["relationship_key"] = semantic_digest(
                    "production.relationship.v1",
                    {
                        k: v
                        for k, v in result.items()
                        if k != "relationship_key"
                    },
                )
            return result
        return value

    fragments = []
    for fragment in production["fragments"]:
        old = fragment["id"]
        new = semantic_digest(
            "production.fragment.v1",
            {k: v for k, v in fragment.items() if k != "id"},
        )
        replacements[old] = new
        fragments.append(fragment | {"id": new})
    production["fragments"] = fragments
    cards = []
    for raw in production["auxiliary_cards"]:
        card = replace(raw)
        new = semantic_digest(
            "production.aux-card.v1",
            {k: v for k, v in card.items() if k != "id"},
        )
        replacements[raw["id"]] = new
        cards.append(card | {"id": new})
    production["auxiliary_cards"] = cards
    production = replace(production)
    candidate = StoryPlan.model_validate_json(
        canonical_bytes(
            plan.model_dump(mode="json") | {"production": production}
        )
    )
    return StoryPlan.model_validate(
        candidate.model_dump() | {"id": production_plan_id(candidate)}
    )


def prose(subject: str, statement: SupportedStatement) -> PlanningCardResult:
    """Bind one structural fixture statement to its exact producer identity."""
    binding = ProseBinding(
        kind="planning_card",
        subject_key=subject,
        plan_key=CONTEXT,
        context_digest=semantic_digest("consumed fixture", subject),
    )
    return PlanningCardResult(
        item_key=binding.key, binding=binding, abstraction=(statement,)
    )


class HierarchyTests(unittest.TestCase):
    """Lossless derivation and bounded hierarchy under the material reproduction caps."""

    def assert_valid(
        self,
        plan: StoryPlan,
        sources: PlanningSources,
        bounds: PlanningBounds = BOUNDS,
    ) -> None:
        """Check rederivation and every final shape, not discarded greedy candidates."""
        check = SizingSerializer()
        validate_production_plan(
            plan,
            sources,
            context_digest=CONTEXT,
            bounds=bounds,
            check_shape=check,
        )
        for shape in planning_shapes(plan, sources):
            check(shape)
        self.assertEqual(check.guards, {15000})
        subjects = tuple(key for wave in planning_waves(plan) for key in wave)
        self.assertEqual(len(subjects), len(set(subjects)))
        self.assertEqual(
            set(subjects),
            {c.id for c in plan.cards}
            | {c.id for c in plan.production.auxiliary_cards},
        )

    def test_normal_whole_fit_primary_tree_and_sizing_only(self) -> None:
        """Small records remain complete inline inputs and generic validation stays strict."""
        _, _, details, sources = fixture()
        plan = build(sources)
        self.assert_valid(plan, sources)
        self.assertEqual(plan.production.fragments, ())
        self.assertEqual(plan.production.auxiliary_cards, ())
        self.assertEqual(sources.details, details)
        self.assertFalse(
            hasattr(planning_shapes(plan, sources)[0], "request_digest")
        )
        self.assertFalse(hasattr(planning_shapes(plan, sources)[0], "send"))
        with self.assertRaisesRegex(
            ValueError, "production_validation_required"
        ):
            validate_story_plan(plan, sources.groups)
        with self.assertRaisesRegex(
            ValueError, "planning_request_membership_mismatch"
        ):
            validate_production_requests(plan)
        expected_atoms = tuple(
            a for g in plan.groups for a in planning_atoms(g)
        )
        actual = []
        cards = {c.id: c for c in plan.cards}
        pending = list(reversed(plan.root_card_keys))
        while pending:
            card = cards[pending.pop()]
            actual.extend(card.atoms)
            pending.extend(reversed(card.child_keys))
        self.assertEqual(tuple(actual), expected_atoms)
        self.assertEqual(
            {p.atom_key for p in plan.production.atom_projections},
            {atom_key(a) for a in expected_atoms},
        )

    def test_exact_47267_byte_accepted_wire_detail_handoff(self) -> None:
        """The saved diagnostic's accepted detail reaches a frozen bounded plan."""
        provider = FixtureProvider()
        inputs = ProductionInputs(
            golden_report(1), Transport(provider, V2Options())
        )
        stage = StageInputs(inputs, "detail", inputs.detail_items())
        request = stage.prepare(tuple(stage.items), repair=False, partition=0)
        raw = FixtureProvider.respond(json.loads(request.body)["data"])
        long_text = "Authored long-detail handoff fixture. " * 1200
        raw["items"][0]["explanations"][0]["before"] = long_text
        self.assertEqual(len(canonical_bytes(raw)), 47267)
        envelope = inputs.wire.envelope(
            raw, request.metadata.batch, stage.contexts
        )
        self.assertFalse(envelope.failures)
        details = tuple(
            DetailResult.model_validate_json(canonical_bytes(value))
            for value in envelope.items
        )
        for detail in details:
            stage.validate(stage.lookups[detail.unit_key], detail)
        sources = derive_planning_sources(
            inputs.index, details, author_context=AuthorContext()
        )
        plan = build(sources)
        indexed_context = capture_indexed_detail_results(
            details, inputs.ledger, index=inputs.index
        )
        with (
            mock.patch(
                "diffstory.narrative_hierarchy.derive_planning_sources",
                side_effect=AssertionError(
                    "indexed path used legacy derivation"
                ),
            ),
            mock.patch(
                "diffstory.report_storage.expand_report_storage",
                side_effect=AssertionError("indexed path expanded the report"),
            ),
            mock.patch(
                "diffstory.narrative_plan.build_evidence_ledger",
                side_effect=AssertionError("indexed path rebuilt the ledger"),
            ),
            mock.patch(
                "diffstory.narrative_plan.build_ledger_index",
                side_effect=AssertionError("indexed path rebuilt the index"),
            ),
            mock.patch(
                "diffstory.narrative_plan._validate_detail_result",
                side_effect=AssertionError("indexed path revalidated details"),
            ),
            mock.patch(
                "diffstory.narrative_plan._resolve_span_ref",
                side_effect=AssertionError(
                    "indexed path resolved citations again"
                ),
            ),
            mock.patch.object(
                IndexedDetailContext,
                "detail_record",
                side_effect=AssertionError("normal builder expanded a detail"),
            ),
        ):
            indexed = derive_planning_sources_indexed(
                inputs.index, indexed_context, author_context=AuthorContext()
            )
            indexed_plan = build(indexed)
        self.assertEqual(plan, indexed_plan)
        self.assertEqual(indexed.record_keys, sources.record_keys)
        for expanded_fragment, indexed_fragment in zip(
            plan.production.fragments,
            indexed_plan.production.fragments,
            strict=True,
        ):
            self.assertEqual(
                fragment_material(indexed, indexed_fragment),
                fragment_material(sources, expanded_fragment),
            )
        long_key = next(
            key
            for key in indexed.record_keys
            if record_material(indexed, key).owner.kind == "detail"
            and record_material(indexed, key).owner.record_kind
            == "explanation"
            and any(
                field.path == "before"
                and field.data == long_text.encode("utf-8")
                for field in record_material(indexed, key).fields
            )
        )
        self.assertEqual(
            record_material(indexed, long_key).fields[1].data,
            long_text.encode("utf-8"),
        )
        memo = indexed._source_state.field_memo  # noqa: SLF001 # Inspect the documented bounded field memo.
        memo_weight = sum(weight for _fields, weight in memo.values())
        self.assertLessEqual(memo_weight, 32_768)
        self.assertTrue(
            all(weight <= 8_192 for _fields, weight in memo.values())
        )
        self.assertNotIn((long_key, "before"), memo)
        memo_units = [
            indexed._source_state.detail_units_by_key[key]  # noqa: SLF001 # Inspect per-unit detail memo coverage.
            for key, _path in memo
            if key in indexed._source_state.detail_units_by_key  # noqa: SLF001 # Separate detail fields from other owners.
        ]
        self.assertEqual(len(memo_units), len(set(memo_units)))
        self.assert_valid(plan, sources)
        self.assertTrue(plan.production.fragments)
        self.assertTrue(
            any(p.material.root for p in plan.production.atom_projections)
        )
        self.assertEqual(sources.details, details)
        self.assertEqual(provider.calls, [])

    def test_unicode_physical_endings_and_complete_field_reconstruction(
        self,
    ) -> None:
        """Huge UTF-8 fields preserve CRLF, CR, LF, LS, PS and terminal spaces."""
        _, index, details, _ = fixture()
        text = '  \u03b1😀\r\nline\rpart\nLS\u2028PS\u2029\\"  ' * 2400
        details = changed_detail(details, text)
        sources = derive_planning_sources(
            index, details, author_context=AuthorContext()
        )
        plan = build(sources)
        self.assert_valid(plan, sources)
        grouped = defaultdict(list)
        for fragment in plan.production.fragments:
            grouped[(record_key(fragment.owner), fragment.path)].append(
                fragment
            )
        for (key, path), fragments in grouped.items():
            expected = next(
                f.data
                for f in record_material(sources, key).fields
                if f.path == path
            )
            start = 0
            restored = bytearray()
            for fragment in fragments:
                self.assertEqual(fragment.byte_start, start)
                data = fragment_material(sources, fragment)
                data.decode("utf-8")
                restored.extend(data)
                start = fragment.byte_end
            self.assertEqual(bytes(restored), expected)
        before = [
            f
            for f in plan.production.fragments
            if f.path == "before" and f.total_bytes == len(text.encode())
        ]
        self.assertGreater(len(before), 10)
        self.assertEqual(
            b"".join(fragment_material(sources, f) for f in before),
            text.encode(),
        )
        self.assertEqual(sources.details, details)

    def test_every_record_field_order_and_empty_value_is_derived(self) -> None:
        """Independent current detail fixtures supply all expected fields and framing."""
        _, _, details, sources = fixture(
            author=AuthorContext(
                title="", repository="", author_reported_motivation=""
            )
        )
        by_unit = {d.unit_key: d for d in details}
        observed = defaultdict(list)
        author_fields = None
        for key in sources.record_keys:
            record = record_material(sources, key)
            fields = {
                f.path: f.data.decode()
                if f.encoding == "utf8"
                else json.loads(f.data)
                for f in record.fields
            }
            if record.owner.kind == "author":
                author_fields = fields
                self.assertIsNone(
                    material_support(
                        sources, record_keys=(key,)
                    ).author_context_digest
                )
            if record.owner.kind != "detail":
                continue
            owner = record.owner
            detail = by_unit[owner.unit_key]
            if owner.record_kind == "detail":
                for name in ("explanations", "observations", "questions"):
                    self.assertEqual(
                        fields[name],
                        {
                            "type": "sequence",
                            "length": len(getattr(detail, name)),
                        },
                    )
            else:
                record_name = {
                    "explanation": "explanations",
                    "observation": "observations",
                    "question": "questions",
                }[owner.record_kind]
                expected = getattr(detail, record_name)[owner.ordinal]
                self.assertEqual(fields, expected.model_dump(mode="json"))
                observed[(owner.unit_key, record_name)].append(owner.ordinal)
        for detail in details:
            for name in ("explanations", "observations", "questions"):
                self.assertEqual(
                    observed[(detail.unit_key, name)],
                    list(range(len(getattr(detail, name)))),
                )
        self.assertEqual(author_fields, AuthorContext().model_dump())
        empty = DetailResult(
            unit_key=details[0].unit_key,
            explanations=(),
            observations=(),
            questions=(),
        )
        # Empty observations/questions remain explicit on an otherwise valid detail.
        empty = DetailResult.model_validate(
            empty.model_dump() | {"explanations": details[0].explanations}
        )
        current = derive_planning_sources(
            build_ledger_index(build_evidence_ledger(golden_report(1))),
            (empty,),
            author_context=AuthorContext(),
        )
        header = next(
            record_material(current, k)
            for k in current.record_keys
            if record_material(current, k).owner.kind == "detail"
        )
        self.assertEqual(
            json.loads(
                next(f.data for f in header.fields if f.path == "observations")
            ),
            {"type": "sequence", "length": 0},
        )

    def test_large_observation_citation_source_author_and_compiler_heading(
        self,
    ) -> None:
        """Every variable input family can split, including support collections."""
        path = "long_" + "heading" * 5000 + ".py"
        before = (
            "def f():\n"
            + "    value = 'old'\n" * 40
            + "    return '"
            + "\u03b1" * 20000
            + "'\n"
        )
        after = before.replace("old", "new").replace("\u03b1", "β")
        report = compile_snapshot(snap({path: before}, {path: after}))
        _, index, details, _ = fixture(
            report, max_span_bytes=100000, max_unit_spans=1000
        )
        changed = []
        for detail in details:
            observations = (
                tuple(
                    FactualObservation(
                        id=semantic_digest(
                            "large observation", detail.unit_key, n
                        ),
                        text="Observed β\r\n\u2028" * 2500,
                        citations=detail.observations[0].citations,
                    )
                    for n in range(2)
                )
                if detail.observations
                else ()
            )
            changed.append(
                DetailResult.model_validate(
                    detail.model_dump() | {"observations": observations}
                )
            )
        author = AuthorContext(
            title="title" * 7000,
            repository="repo" * 6000,
            author_reported_motivation="Author intent\r\n" * 6000,
        )
        sources = derive_planning_sources(
            index, tuple(changed), author_context=author
        )
        plan = build(sources)
        self.assert_valid(plan, sources)
        h = plan.production
        self.assertIsNotNone(h.author_projection.root)
        self.assertEqual(
            {f.owner.kind for f in h.fragments},
            {"source", "detail", "author", "heading"},
        )
        self.assertTrue(
            any(
                f.path == "citations" and f.byte_end < f.total_bytes
                for f in h.fragments
            )
        )
        self.assertTrue(
            any(
                f.path == "text"
                and f.owner.kind == "source"
                and f.byte_end < f.total_bytes
                for f in h.fragments
            )
        )
        self.assertTrue(
            any(
                f.path == "path" and f.byte_end < f.total_bytes
                for f in h.fragments
            )
        )
        self.assertEqual(sources.details, tuple(changed))
        source_fragment = next(
            f
            for f in h.fragments
            if f.owner.kind == "source" and f.path == "text"
        )
        source_support = material_support(
            sources, fragments=(source_fragment,)
        )
        self.assertEqual(source_support.claims, ())
        self.assertEqual(source_support.citations, ())
        self.assertEqual(len(source_support.basis), 1)
        self.assertIsNone(source_support.author_context_digest)
        self.assertEqual(
            validate_source_slice(source_support.basis[0], sources),
            fragment_material(sources, source_fragment),
        )
        detail_fragment = next(
            f
            for f in h.fragments
            if f.owner.kind == "detail" and f.path == "text"
        )
        self.assertEqual(
            material_support(sources, fragments=(detail_fragment,)),
            SupportSet(),
        )
        waves = {
            key: pos
            for pos, wave in enumerate(planning_waves(plan))
            for key in wave
        }
        for card in h.auxiliary_cards:
            for ref in card.sources:
                self.assertLess(waves[ref.target_key], waves[card.id])

    def test_64_group_63_prerequisite_context_and_scc_facts(self) -> None:
        """Targeted context reductions retain every current compiler relationship."""
        for cycle in (False, True):
            report = golden_report(64)
            groups = report["groups"]
            if cycle:
                for pos, group in enumerate(groups):
                    group["prerequisites"] = [groups[pos - 1]["id"]]
            else:
                groups[-1]["prerequisites"] = [g["id"] for g in groups[:-1]]
            ledger, _, _, sources = fixture(report)
            plan = build(sources)
            self.assert_valid(plan, sources)
            roots = {
                r.group_key: r.card_key for r in plan.production.group_roots
            }
            expected_edges = {
                (g.key, p) for g in ledger.groups for p in g.prerequisites
            }
            actual_edges = {
                (view.group_key, ref.source_group_key)
                for view in plan.production.section_views
                for ref in view.sources
                if ref.relationship == "prerequisite"
            }
            self.assertEqual(actual_edges, expected_edges)
            self.assertEqual(len(plan.production.section_views), 64)
            for view in plan.production.section_views:
                self.assertEqual(view.final, view.position == 63)
                own = next(s for s in view.sources if s.relationship == "own")
                self.assertEqual(own.target_key, roots[view.group_key])
                for ref in view.sources:
                    if ref.kind == "group":
                        self.assertEqual(
                            ref.target_key, roots[ref.source_group_key]
                        )
            if cycle:
                self.assertEqual(len(plan.cycles), 1)
                self.assertEqual(
                    set(plan.production.cycle_roots[0].group_keys),
                    {g.key for g in ledger.groups},
                )
                self.assertTrue(
                    any(
                        c.purpose == "cycle"
                        for c in plan.production.auxiliary_cards
                    )
                )
            else:
                target = max(
                    plan.production.section_views,
                    key=lambda v: sum(
                        r.relationship == "prerequisite" for r in v.sources
                    ),
                )
                self.assertEqual(
                    sum(
                        s.relationship == "prerequisite"
                        for s in target.sources
                    ),
                    63,
                )
                self.assertIsNotNone(target.context_root)

    def test_dynamic_budgets_keep_configured_guard_and_fixed_protocol_failure(
        self,
    ) -> None:
        """Smaller abstraction budgets allow progress without changing provider limits."""
        _, index, details, _ = fixture()
        sources = derive_planning_sources(
            index,
            changed_detail(details, "long field " * 6000),
            author_context=AuthorContext(),
        )
        check = SizingSerializer(capacity=23000)
        plan = build(sources, check=check)
        self.assertLess(
            plan.production.card_budget.result_wire_bytes,
            BOUNDS.card_wire_bytes,
        )
        self.assertEqual(plan.production.bounds, BOUNDS)
        self.assertEqual(check.guards, {15000})
        for shape in planning_shapes(plan, sources):
            check(shape)
        with self.assertRaisesRegex(
            PackingError, "production_protocol_capacity"
        ):
            build(sources, check=SizingSerializer(capacity=15010))

    def test_self_consistent_fragment_rehash_and_coverage_forgery_rejected(
        self,
    ) -> None:
        """Missing, repeated, overlapping, wrong-owner and altered content fail closed."""
        _, index, details, _ = fixture()
        sources = derive_planning_sources(
            index,
            changed_detail(details, "one huge field " * 6000),
            author_context=AuthorContext(),
        )
        plan = build(sources)
        fields = plan.production.model_dump(mode="json")
        fragments = fields["fragments"]
        target = next(
            i
            for i, f in enumerate(fragments)
            if f["path"] == "before" and f["byte_start"] > 0
        )
        for changes in (
            {"content_digest": "f" * 64},
            {"byte_start": fragments[target]["byte_start"] - 1},
            {"owner": fragments[target]["owner"] | {"ordinal": 999}},
            {"path": "text"},
        ):
            forged = [*fragments]
            forged[target] = forged[target] | changes
            candidate = rehash(plan, {"fragments": forged})
            self.assertEqual(candidate.id, production_plan_id(candidate))
            with self.assertRaisesRegex(ValueError, "production_plan_changed"):
                validate_production_plan(
                    candidate,
                    sources,
                    context_digest=CONTEXT,
                    bounds=BOUNDS,
                    check_shape=SizingSerializer(),
                )
        for forged in (fragments[1:], [*fragments, fragments[0]]):
            candidate = rehash(plan, {"fragments": forged})
            with self.assertRaisesRegex(ValueError, "production_plan_changed"):
                validate_production_plan(
                    candidate,
                    sources,
                    context_digest=CONTEXT,
                    bounds=BOUNDS,
                    check_shape=SizingSerializer(),
                )

    def test_forged_context_edges_cycles_unknown_and_shared_child_rejected(
        self,
    ) -> None:
        """Rehashed relationships cannot replace compiler facts or tree ownership."""
        report = golden_report(8)
        report["groups"][-1]["prerequisites"] = [
            g["id"] for g in report["groups"][:-1]
        ]
        _, index, details, _ = fixture(report)
        sources = derive_planning_sources(
            index,
            changed_detail(details, "huge " * 10000),
            author_context=AuthorContext(),
        )
        plan = build(sources)
        views = plan.production.model_dump(mode="json")["section_views"]
        views[0]["sources"][0]["source_group_key"] = "f" * 64
        views[0]["sources"][0]["target_group_key"] = "f" * 64
        candidate = rehash(plan, {"section_views": views})
        with self.assertRaisesRegex(ValueError, "production_plan_changed"):
            validate_production_plan(
                candidate,
                sources,
                context_digest=CONTEXT,
                bounds=BOUNDS,
                check_shape=SizingSerializer(),
            )
        cards = list(plan.production.auxiliary_cards)
        parent = next(c for c in cards if c.child_keys)
        for bad in (
            parent.model_copy(update={"child_keys": (parent.id,)}),
            parent.model_copy(update={"child_keys": ("f" * 64,)}),
            parent.model_copy(update={"owner_key": "f" * 64}),
        ):
            h = plan.production.model_copy(
                update={
                    "auxiliary_cards": tuple(
                        bad if c.id == parent.id else c for c in cards
                    )
                }
            )
            with self.assertRaises(ValueError):
                planning_waves(plan.model_copy(update={"production": h}))
        # A second parent cannot own the same internal reduction children.
        duplicate = parent.model_copy(update={"id": "e" * 64})
        h = plan.production.model_copy(
            update={"auxiliary_cards": (*cards, duplicate)}
        )
        with self.assertRaisesRegex(ValueError, "shared_auxiliary_child"):
            planning_waves(plan.model_copy(update={"production": h}))

    def test_plan_identity_excludes_scheduling_and_requests_but_includes_budgets(
        self,
    ) -> None:
        """Result arrival and actual audit partitions cannot rewrite the frozen plan."""
        _, index, details, sources = fixture(golden_report(3))
        plan = build(sources)
        reordered = derive_planning_sources(
            index, tuple(reversed(details)), author_context=AuthorContext()
        )
        self.assertEqual(plan, build(reordered))
        keys = tuple(key for wave in planning_waves(plan) for key in wave)

        def spec(items: tuple[str, ...], partition: int) -> BatchSpec:
            return BatchSpec(
                id=semantic_digest("actual fixture audit", items),
                stage="planning",
                item_keys=items,
                group_keys=(),
                request_digest=semantic_digest("actual bytes fixture", items),
                input_bound=100,
                output_reservation=100,
                output_guard=15000,
                estimated_response_bytes=1000,
                estimated_seconds=1,
                partition=partition,
            )

        a = plan.model_copy(update={"requests": (spec(keys, 0),)})
        b = plan.model_copy(
            update={
                "requests": tuple(
                    spec((k,), pos) for pos, k in enumerate(reversed(keys))
                )
            }
        )
        validate_production_requests(a)
        validate_production_requests(b)
        self.assertEqual(production_plan_id(a), production_plan_id(b))
        self.assertEqual(a.id, plan.id)
        for requests in (a.requests * 2, b.requests[:-1]):
            with self.assertRaisesRegex(
                ValueError, "planning_request_membership_mismatch"
            ):
                validate_production_requests(
                    plan.model_copy(update={"requests": requests})
                )
        bounds = PlanningBounds.model_validate(
            BOUNDS.model_dump() | {"card_wire_bytes": 900}
        )
        self.assertNotEqual(plan.id, build(sources, bounds=bounds).id)

    def test_content_identity_survives_revision_and_30_line_rebinding(
        self,
    ) -> None:
        """Source owners omit run signatures, URLs and absolute fragment coordinates."""
        plans = []
        references = []
        for start in (1, 31):
            report = compile_snapshot(
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
                        (side, "x.py"): {"start_line": start}
                        for side in ("base", "head")
                    },
                )
            )
            report["meta"]["base_sha"] = str(start) * 40
            report["meta"]["head_sha"] = str(start + 1) * 40
            ledger, index, details, _ = fixture(report)
            details = changed_detail(
                details, "unchanged detailed mechanism " * 2000
            )
            sources = derive_planning_sources(
                index, details, author_context=AuthorContext()
            )
            plan = build(sources)
            self.assert_valid(plan, sources)
            plans.append(plan)
            owner = next(
                f.owner
                for f in plan.production.fragments
                if f.owner.kind == "source"
            )
            ref = source_reference(sources, owner)
            span = resolve_span_ref(
                ref, index.units[ref.unit_key], ledger, index=index
            )
            references.append((ref, span.line_start))
        self.assertEqual(plans[0], plans[1])
        self.assertEqual(references[0][0], references[1][0])
        self.assertEqual(references[1][1] - references[0][1], 30)

    def test_authority_rejects_copy_rebind_nested_mutation_and_invalid_details(
        self,
    ) -> None:
        """Original snapshots and independently registered indexes resist cached forgery."""
        ledger, index, details, sources = fixture()
        for copied in (
            copy.copy(sources),
            copy.deepcopy(sources),
            PlanningSources(),
        ):
            with self.assertRaisesRegex(
                ValueError, "unregistered_planning_sources"
            ):
                build(copied)
        with self.assertRaisesRegex(ValueError, "unregistered_ledger_index"):
            derive_planning_sources(
                copy.copy(index), details, author_context=AuthorContext()
            )
        with self.assertRaises(ValueError):
            derive_planning_sources(index, (), author_context=AuthorContext())
        explanation = details[0].explanations[0]
        original = explanation.before
        object.__setattr__(
            explanation, "before", "changed behind frozen boundary"
        )
        with self.assertRaisesRegex(ValueError, "planning_sources_changed"):
            build(sources)
        object.__setattr__(explanation, "before", original)
        for span in (ledger.spans[0], next(iter(index.spans.values()))):
            original = span.text
            object.__setattr__(span, "text", "tampered nested cache")
            with self.assertRaisesRegex(ValueError, "ledger_index_mutated"):
                build(sources)
            object.__setattr__(span, "text", original)
        self.assert_valid(build(sources), sources)

    def test_fragment_lookup_does_not_scan_or_reparse_whole_source(
        self,
    ) -> None:
        """A large source is indexed once, then exact slices are constant-time lookups."""
        _, index, details, _ = fixture()
        sources = derive_planning_sources(
            index,
            changed_detail(details, "huge Unicode 😀 " * 6000),
            author_context=AuthorContext(),
        )
        plan = build(sources)
        with (
            mock.patch(
                "diffstory.narrative_hierarchy.record_material",
                side_effect=AssertionError("record reparsed"),
            ),
            mock.patch.object(
                type(index),
                "validated_ledger",
                side_effect=AssertionError("ledger rescanned"),
            ),
        ):
            for fragment in plan.production.fragments:
                self.assertEqual(
                    len(fragment_material(sources, fragment)),
                    fragment.byte_end - fragment.byte_start,
                )


class IndexedPlanningSourceTests(unittest.TestCase):
    """Compare the indexed hierarchy adapter with its expanded oracle."""

    def test_indexed_sources_match_expanded_material_and_plan(self) -> None:
        """All indexed source bytes and plan outputs must match legacy values."""
        cases = (
            (
                "default-author",
                AuthorContext(),
                None,
            ),
            (
                "unicode-crlf-empty-author",
                AuthorContext(
                    title="", repository="", author_reported_motivation=""
                ),
                '  \u03b1😀\r\nline\rpart\nLS\u2028PS\u2029\\"  ',
            ),
        )
        for name, author, before in cases:
            with self.subTest(name=name):
                ledger, index, details, _unused = fixture(author=author)
                if before is not None:
                    details = changed_detail(details, before)
                expanded = derive_planning_sources(
                    index, details, author_context=author
                )
                context = capture_indexed_detail_results(
                    details, ledger, index=index
                )
                expected_plan = build(expanded)
                with (
                    mock.patch(
                        "diffstory.narrative_hierarchy.derive_planning_sources",
                        side_effect=AssertionError(
                            "indexed path used legacy derivation"
                        ),
                    ),
                    mock.patch(
                        "diffstory.report_storage.expand_report_storage",
                        side_effect=AssertionError(
                            "indexed path expanded the report"
                        ),
                    ),
                    mock.patch(
                        "diffstory.narrative_plan.build_evidence_ledger",
                        side_effect=AssertionError(
                            "indexed path rebuilt the ledger"
                        ),
                    ),
                    mock.patch(
                        "diffstory.narrative_plan.build_ledger_index",
                        side_effect=AssertionError(
                            "indexed path rebuilt the index"
                        ),
                    ),
                    mock.patch(
                        "diffstory.narrative_plan._validate_detail_result",
                        side_effect=AssertionError(
                            "indexed path revalidated details"
                        ),
                    ),
                    mock.patch(
                        "diffstory.narrative_plan._resolve_span_ref",
                        side_effect=AssertionError(
                            "indexed path resolved citations again"
                        ),
                    ),
                    mock.patch.object(
                        IndexedDetailContext,
                        "detail_record",
                        side_effect=AssertionError(
                            "normal planning rebuilt complete details"
                        ),
                    ),
                ):
                    indexed = derive_planning_sources_indexed(
                        index, context, author_context=author
                    )
                    self.assertEqual(indexed.groups, expanded.groups)
                    self.assertEqual(indexed.record_keys, expanded.record_keys)
                    actual_plan = build(indexed)
                    for key in expanded.record_keys:
                        self.assertEqual(
                            record_material(indexed, key),
                            record_material(expanded, key),
                        )
                    for fragment in expected_plan.production.fragments:
                        self.assertEqual(
                            fragment_material(indexed, fragment),
                            fragment_material(expanded, fragment),
                        )
                    source_key = next(
                        key
                        for key in indexed.record_keys
                        if record_material(indexed, key).owner.kind == "source"
                    )
                    source_record = record_material(indexed, source_key)
                    source_text = next(
                        field
                        for field in source_record.fields
                        if field.path == "text" and field.data
                    )
                    source_fragment = _fragment(
                        source_record.owner,
                        source_text,
                        0,
                        min(12, len(source_text.data)),
                    )
                    self.assertEqual(
                        fragment_material(indexed, source_fragment),
                        fragment_material(expanded, source_fragment),
                    )
                    expected_support = material_support(
                        expanded,
                        record_keys=expanded.record_keys,
                        fragments=(source_fragment,),
                    )
                    actual_support = material_support(
                        indexed,
                        record_keys=indexed.record_keys,
                        fragments=(source_fragment,),
                    )
                    self.assertEqual(actual_support, expected_support)
                    self.assertEqual(
                        validate_source_slice(
                            actual_support.basis[0], indexed
                        ),
                        validate_source_slice(
                            expected_support.basis[0], expanded
                        ),
                    )
                self.assertEqual(actual_plan, expected_plan)
                self.assertEqual(
                    production_plan_id(actual_plan),
                    production_plan_id(expected_plan),
                )
                expanded_details = indexed.details
                self.assertEqual(expanded_details, details)
                if expanded_details[0].explanations:
                    original_before = details[0].explanations[0].before
                    object.__setattr__(
                        expanded_details[0].explanations[0],
                        "before",
                        "detached accessor mutation",
                    )
                    self.assertEqual(indexed.details, details)
                    self.assertEqual(
                        details[0].explanations[0].before, original_before
                    )
                context.citation_scope.close()

    def test_indexed_record_material_returns_a_detached_owner(self) -> None:
        """A returned owner alias could corrupt captured record metadata."""
        cases = (("explanation", 0, False, int, 0),)
        for (
            record_kind,
            ordinal,
            replacement,
            expected_type,
            expected_value,
        ) in cases:
            with self.subTest(record_kind=record_kind, ordinal=ordinal):
                ledger, index, details, _unused = fixture()
                context = capture_indexed_detail_results(
                    details, ledger, index=index
                )
                indexed = derive_planning_sources_indexed(
                    index, context, author_context=AuthorContext()
                )
                try:
                    state = indexed._source_state  # noqa: SLF001 # Inspect the exact retained owner.
                    key = next(
                        key
                        for key in indexed.record_keys
                        if state.record_owners[key].kind == "detail"
                        and state.record_owners[key].record_kind == record_kind
                        and state.record_owners[key].ordinal == ordinal
                    )
                    original_fingerprint = state.current_metadata_fingerprint()
                    public_material = record_material(indexed, key)
                    public_owner = public_material.owner
                    self.assertIsNot(public_owner, state.record_owners[key])
                    object.__setattr__(public_owner, "ordinal", replacement)
                    self.assertIs(type(public_owner.ordinal), bool)
                    self.assertIs(
                        type(state.record_owners[key].ordinal), expected_type
                    )
                    self.assertEqual(
                        state.record_owners[key].ordinal, expected_value
                    )
                    self.assertEqual(
                        state.current_metadata_fingerprint(),
                        original_fingerprint,
                    )
                    indexed.check_current()
                    later_material = record_material(indexed, key)
                    self.assertEqual(
                        later_material.owner, state.record_owners[key]
                    )
                    self.assertIs(
                        type(later_material.owner.ordinal), expected_type
                    )
                    self.assertEqual(
                        later_material.owner.ordinal, expected_value
                    )
                    self.assertEqual(
                        later_material.fields, public_material.fields
                    )
                    self.assertEqual(
                        later_material.citations, public_material.citations
                    )
                    self.assertEqual(
                        later_material.claim, public_material.claim
                    )
                finally:
                    context.citation_scope.close()

    def test_indexed_metadata_witness_preserves_raw_scalar_types(self) -> None:
        """Canonical model serialization could hide raw owner or reference changes."""
        cases = (
            ("record_owner", "ordinal", 0, False),
            ("source_reference", "line_start", 1, True),
        )
        for metadata_kind, field_name, original_value, replacement in cases:
            with self.subTest(
                metadata_kind=metadata_kind, field_name=field_name
            ):
                ledger, index, details, _unused = fixture()
                context = capture_indexed_detail_results(
                    details, ledger, index=index
                )
                indexed = derive_planning_sources_indexed(
                    index, context, author_context=AuthorContext()
                )
                try:
                    state = indexed._source_state  # noqa: SLF001 # Mutate retained metadata to exercise its witness.
                    if metadata_kind == "record_owner":
                        target = next(
                            owner
                            for owner in state.record_owners.values()
                            if owner.kind == "detail"
                            and owner.record_kind == "explanation"
                            and getattr(owner, field_name) == original_value
                        )
                    else:
                        target = next(
                            reference
                            for reference in state.source_by_record.values()
                            if type(getattr(reference, field_name)) is int
                            and getattr(reference, field_name)
                            == original_value
                        )
                    original = getattr(target, field_name)
                    original_fingerprint = state.current_metadata_fingerprint()
                    object.__setattr__(target, field_name, replacement)
                    self.assertIs(type(getattr(target, field_name)), bool)
                    self.assertNotEqual(
                        state.current_metadata_fingerprint(),
                        original_fingerprint,
                    )
                    with self.assertRaisesRegex(
                        ValueError, "planning_sources_changed"
                    ):
                        indexed.check_current()
                    object.__setattr__(target, field_name, original)
                    self.assertEqual(
                        state.current_metadata_fingerprint(),
                        original_fingerprint,
                    )
                    indexed.check_current()
                finally:
                    context.citation_scope.close()

    def test_indexed_sources_own_normalized_author_input(self) -> None:
        """Later caller mutation must not change the adapter's author record."""
        author = AuthorContext(
            title="Original title",
            repository="repo",
            author_reported_motivation="Keep the change small.",
        )
        ledger, index, details, _unused = fixture(author=author)
        context = capture_indexed_detail_results(details, ledger, index=index)
        indexed = derive_planning_sources_indexed(
            index, context, author_context=author
        )
        original_digest = indexed.author_digest
        object.__setattr__(author, "title", "mutated after capture")
        indexed.check_current()
        self.assertEqual(indexed.author.title, "Original title")
        self.assertEqual(indexed.author_digest, original_digest)
        context.citation_scope.close()

    def test_indexed_sources_reject_changed_index_revision_and_source_lines(
        self,
    ) -> None:
        """Source authority must stay on its current ledger, index, and revisions."""
        ledger, index, details, _unused = fixture()
        revision = ("base-revision-a", "head-revision-a")
        context = capture_indexed_detail_results(
            details,
            ledger,
            index=index,
            revision_binding=revision,
        )
        indexed = derive_planning_sources_indexed(
            index,
            context,
            author_context=AuthorContext(),
            revision_binding=revision,
        )
        with self.assertRaisesRegex(
            ValueError, "citation_scope_revision_mismatch"
        ):
            derive_planning_sources_indexed(
                index,
                context,
                author_context=AuthorContext(),
                revision_binding=("base-revision-b", revision[1]),
            )
        with self.assertRaisesRegex(
            ValueError, "citation_scope_binding_mismatch"
        ):
            derive_planning_sources_indexed(
                build_ledger_index(ledger),
                context,
                author_context=AuthorContext(),
                revision_binding=revision,
            )
        span = ledger.spans[0]
        original_line = span.line_start
        object.__setattr__(span, "line_start", original_line + 1)
        try:
            with self.assertRaisesRegex(ValueError, "ledger_index_mutated"):
                indexed.check_current()
        finally:
            object.__setattr__(span, "line_start", original_line)
        indexed.check_current()
        context.citation_scope.close()

    def test_indexed_source_registration_rejects_copies_and_closes_memo(
        self,
    ) -> None:
        """Copies cannot borrow source scope and closing releases its field memo."""
        ledger, index, details, _unused = fixture()
        context = capture_indexed_detail_results(details, ledger, index=index)
        indexed = derive_planning_sources_indexed(
            index, context, author_context=AuthorContext()
        )
        self.assertFalse(hasattr(indexed, "validated_generation"))
        self.assertFalse(hasattr(indexed, "validated_report"))
        with mock.patch.object(
            IndexedDetailContext,
            "detail_record",
            side_effect=AssertionError("normal planning expanded details"),
        ):
            build(indexed)
        state = indexed._source_state  # noqa: SLF001 # Inspect scope-owned cache behavior.
        detail_key = next(
            key
            for key in indexed.record_keys
            if state.record_owners[key].kind == "detail"
            and state.record_owners[key].record_kind == "explanation"
        )
        _indexed_field(state, detail_key, "before", cache=True)
        self.assertTrue(state.field_memo)
        _indexed_field(state, detail_key, "after", cache=True)
        self.assertNotIn((detail_key, "before"), state.field_memo)
        self.assertIn((detail_key, "after"), state.field_memo)
        for forged in (IndexedPlanningSources(state),):
            with self.assertRaisesRegex(
                ValueError, "unregistered_planning_sources"
            ):
                forged.check_current()
        with self.assertRaises(TypeError):
            context.citation_scope._register_indexed_planning_sources(  # noqa: SLF001 # Public callers cannot issue a scope registration.
                context,
                IndexedPlanningSources(state),
                state.binding_witness(),
                state.field_memo,
            )
        with self.assertRaisesRegex(TypeError, "cannot_be_copied"):
            copy.copy(indexed)
        with self.assertRaisesRegex(TypeError, "cannot_be_copied"):
            copy.deepcopy(indexed)
        with self.assertRaisesRegex(TypeError, "cannot_be_serialized"):
            pickle.dumps(indexed)
        original_units = context._unit_keys  # noqa: SLF001 # Preserve the accepted selector order.
        object.__setattr__(context, "_unit_keys", ("f" * 64,))
        try:
            with self.assertRaisesRegex(
                ValueError, "unregistered_detail_context"
            ):
                indexed.check_current()
        finally:
            object.__setattr__(context, "_unit_keys", original_units)
        original_backing = context._graph  # noqa: SLF001 # Preserve the accepted backing selector.
        object.__setattr__(context, "_graph", object())
        try:
            with self.assertRaisesRegex(
                ValueError, "unregistered_detail_context"
            ):
                indexed.check_current()
        finally:
            object.__setattr__(context, "_graph", original_backing)
        context.citation_scope.close()
        self.assertFalse(state.field_memo)
        with self.assertRaisesRegex(ValueError, "unregistered_detail_context"):
            indexed.check_current()

    def test_indexed_field_memo_evicts_oldest_by_configured_weight(
        self,
    ) -> None:
        """Indexed source fields must obey byte-weighted LRU eviction."""
        ledger, index, details, _unused = fixture(max_span_bytes=128)
        context = capture_indexed_detail_results(details, ledger, index=index)
        indexed = derive_planning_sources_indexed(
            index, context, author_context=AuthorContext()
        )
        state = indexed._source_state  # noqa: SLF001 # Inspect the bounded memo.
        source_keys = tuple(
            key
            for key in indexed.record_keys
            if state.record_owners[key].kind == "source"
        )
        measured = []
        for key in source_keys:
            _indexed_field(state, key, "text", cache=True)
            measured.append((key, state.field_memo[(key, "text")][1]))
        state.field_memo.clear()
        largest = sorted(measured, key=lambda item: item[1], reverse=True)[:3]
        self.assertEqual(len(largest), 3)
        limit = largest[0][1] + largest[1][1]
        with mock.patch(
            "diffstory.narrative_hierarchy.INDEXED_FIELD_MEMO_MAX_WEIGHT",
            limit,
        ):
            for key, _weight in largest:
                _indexed_field(state, key, "text", cache=True)
        self.assertLessEqual(
            sum(weight for _field, weight in state.field_memo.values()), limit
        )
        self.assertNotIn((largest[0][0], "text"), state.field_memo)
        self.assertIn((largest[1][0], "text"), state.field_memo)
        self.assertIn((largest[2][0], "text"), state.field_memo)
        context.citation_scope.close()


class SupportTests(unittest.TestCase):
    """Immediate consumption is distinct from structural provenance termination."""

    def setUp(self) -> None:
        """Use real current source references and accepted detail observations."""
        _, self.index, _, self.sources = fixture(
            author=AuthorContext(
                author_reported_motivation="Preserve indentation."
            )
        )
        self.source_key = next(
            key
            for key in self.sources.record_keys
            if record_material(self.sources, key).owner.kind == "source"
        )
        self.source_record = record_material(self.sources, self.source_key)
        self.ref = source_reference(self.sources, self.source_record.owner)
        self.text = next(
            f.data for f in self.source_record.fields if f.path == "text"
        )

    def test_slice_never_authorizes_whole_span_or_unseen_remainder(
        self,
    ) -> None:
        """Offsets are relative to the atomic text, not current absolute source bytes."""
        ref = SourceSliceRef(
            source=self.ref,
            byte_start=0,
            byte_end=1,
            content_digest=content_digest(self.text[:1]),
        )
        self.assertEqual(
            validate_source_slice(ref, self.sources), self.text[:1]
        )
        result = prose(
            "a" * 64,
            SupportedStatement(
                kind="implementation_claim",
                text="Fixture bytes are supplied.",
                basis=(ref,),
            ),
        )
        allowed = SupportSet(basis=(ref,))
        validate_consumed_support(result, allowed, self.sources, {})
        validate_support_chains(result, self.sources, {})
        expanded = prose(
            "b" * 64,
            SupportedStatement(
                kind="implementation_claim",
                text="Unseen bytes cannot be cited directly.",
                citations=(self.ref,),
            ),
        )
        with self.assertRaisesRegex(ValueError, "unconsumed_support"):
            validate_consumed_support(expanded, allowed, self.sources, {})
        for update in (
            {"byte_end": len(self.text) + 1},
            {"content_digest": "f" * 64},
            {"source": self.ref.model_copy(update={"line_start": 99})},
        ):
            with self.assertRaises(ValueError):
                validate_source_slice(
                    ref.model_copy(update=update), self.sources
                )

    def test_whole_claim_requires_whole_observation_and_all_direct_support(
        self,
    ) -> None:
        """An observation record alone is context, not a consumed-source grant."""
        observed = next(
            record_material(self.sources, key)
            for key in self.sources.record_keys
            if record_material(self.sources, key).claim is not None
        )
        partial = material_support(
            self.sources, record_keys=(observed.key, self.source_key)
        )
        self.assertEqual(partial.claims, ())
        sources = tuple(
            key
            for key in self.sources.record_keys
            if record_material(self.sources, key).owner.kind == "source"
        )
        whole = material_support(
            self.sources, record_keys=(observed.key, *sources)
        )
        self.assertEqual(whole.claims, (observed.claim,))
        self.assertEqual(set(whole.citations), set(observed.citations))
        author = next(
            key
            for key in self.sources.record_keys
            if record_material(self.sources, key).owner.kind == "author"
        )
        author_support = material_support(self.sources, record_keys=(author,))
        self.assertEqual(author_support.claims, ())
        self.assertEqual(author_support.citations, ())
        self.assertEqual(author_support.basis, ())
        self.assertEqual(
            author_support.author_context_digest, self.sources.author_digest
        )

    def test_statement_exact_path_result_digest_item_and_consumption(
        self,
    ) -> None:
        """Only exact consumed implementation statements can support a parent."""
        child = prose(
            "a" * 64,
            SupportedStatement(
                kind="implementation_claim",
                text="The source calls a method.",
                citations=(self.ref,),
            ),
        )
        ref = statement_reference(
            child, StatementPath(field="abstraction", index=0)
        )
        parent = prose(
            "b" * 64,
            SupportedStatement(
                kind="implementation_claim",
                text="A child establishes the method call.",
                basis=(ref,),
            ),
        )
        allowed = inherited_support((child,))
        validate_consumed_support(
            parent, allowed, self.sources, {child.item_key: child}
        )
        validate_support_chains(parent, self.sources, {child.item_key: child})
        for update in (
            {"result_digest": "f" * 64},
            {"statement_digest": "f" * 64},
            {"producer_item_key": "f" * 64},
            {"statement_path": StatementPath(field="synthesis")},
            {"statement_path": StatementPath(field="abstraction", index=1)},
        ):
            with self.assertRaises(ValueError):
                validate_statement_reference(
                    ref.model_copy(update=update), {child.item_key: child}
                )
        with self.assertRaisesRegex(
            ValueError, "unconsumed_statement_producer"
        ):
            validate_consumed_support(parent, allowed, self.sources, {})
        altered = child.model_copy(
            update={
                "abstraction": (
                    SupportedStatement(
                        kind="implementation_claim",
                        text="Different prose.",
                        citations=(self.ref,),
                    ),
                )
            }
        )
        with self.assertRaisesRegex(ValueError, "statement_reference_changed"):
            validate_statement_reference(ref, {child.item_key: altered})

    def test_editorial_author_and_uncertainty_do_not_confer_implementation_support(
        self,
    ) -> None:
        """Provenance remains distinct through consumed child results."""
        for kind in ("editorial", "uncertainty", "author_motivation"):
            child = prose(
                "a" * 64,
                SupportedStatement(
                    kind=kind,
                    text="Context without implementation proof.",
                    author_context_digest=self.sources.author_digest
                    if kind == "author_motivation"
                    else None,
                ),
            )
            with self.assertRaisesRegex(
                ValueError, "unsupported_statement_kind"
            ):
                statement_reference(child, StatementPath(field="abstraction"))
            support = inherited_support(
                (child,), author_context_digest=self.sources.author_digest
            )
            self.assertEqual(support.basis, ())
            self.assertEqual(
                support.author_context_digest,
                self.sources.author_digest
                if kind == "author_motivation"
                else None,
            )
        author = prose(
            "b" * 64,
            SupportedStatement(
                kind="author_motivation",
                text="The author wants this change.",
                author_context_digest=self.sources.author_digest,
            ),
        )
        with self.assertRaisesRegex(ValueError, "unconsumed_author_context"):
            validate_consumed_support(author, SupportSet(), self.sources, {})

    def test_deep_support_chains_are_iterative_and_revalidate_endpoints(
        self,
    ) -> None:
        """A long inherited chain terminates at current source without recursion."""
        current = prose(
            semantic_digest("child", 0),
            SupportedStatement(
                kind="implementation_claim",
                text="The source supports this statement.",
                citations=(self.ref,),
            ),
        )
        producers = {current.item_key: current}
        for number in range(1100):
            ref = statement_reference(
                current, StatementPath(field="abstraction")
            )
            current = prose(
                semantic_digest("child", number + 1),
                SupportedStatement(
                    kind="implementation_claim",
                    text="The consumed child supports this statement.",
                    basis=(ref,),
                ),
            )
            producers[current.item_key] = current
        validate_support_chains(current, self.sources, producers)
        producers.pop(next(iter(producers)))
        with self.assertRaisesRegex(
            ValueError, "unconsumed_statement_producer"
        ):
            validate_support_chains(current, self.sources, producers)

    def test_forest_checks_current_authority_once_and_shares_deep_branches(
        self,
    ) -> None:
        """Many roots share one deep source check and one iterative traversal."""
        current = prose(
            semantic_digest("forest", 0),
            SupportedStatement(
                kind="implementation_claim",
                text="Source endpoint.",
                citations=(self.ref,),
            ),
        )
        producers = {current.item_key: current}
        roots = []
        for number in range(1100):
            current = prose(
                semantic_digest("forest", number + 1),
                SupportedStatement(
                    kind="implementation_claim",
                    text="Consumed child.",
                    basis=(
                        statement_reference(
                            current, StatementPath(field="abstraction")
                        ),
                    ),
                ),
            )
            producers[current.item_key] = current
            if number >= 1088:
                roots.append(current)
        with mock.patch.object(
            PlanningSources,
            "check_current",
            autospec=True,
            side_effect=PlanningSources.check_current,
        ) as checked:
            validate_support_forest(tuple(roots), self.sources, producers)
            self.assertEqual(checked.call_count, 1)
        for sources in (copy.copy(self.sources), PlanningSources()):
            with self.assertRaisesRegex(
                ValueError, "unregistered_planning_sources"
            ):
                validate_support_forest(tuple(roots), sources, producers)
        endpoint = next(iter(producers))
        missing = dict(producers)
        missing.pop(endpoint)
        with self.assertRaisesRegex(
            ValueError, "unconsumed_statement_producer"
        ):
            validate_support_forest(tuple(roots), self.sources, missing)
        changed = dict(producers)
        changed[endpoint] = prose(
            semantic_digest("forged endpoint"),
            SupportedStatement(
                kind="editorial", text="No implementation support."
            ),
        )
        with self.assertRaises(ValueError):
            validate_support_forest(tuple(roots), self.sources, changed)
        span = next(iter(self.index.spans.values()))
        original = span.text
        object.__setattr__(span, "text", "changed current authority")
        try:
            with self.assertRaisesRegex(ValueError, "ledger_index_mutated"):
                validate_support_forest(tuple(roots), self.sources, producers)
        finally:
            object.__setattr__(span, "text", original)


if __name__ == "__main__":
    unittest.main()
