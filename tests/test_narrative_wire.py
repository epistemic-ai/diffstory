"""Raw JSON edge cases before provider output becomes a dictionary."""

from __future__ import annotations

import unittest
from unittest.mock import patch

from test_narrative_v2 import FixtureProvider
from test_narrative_v2 import golden_report

from diffstory.narrative_plan import build_evidence_ledger
from diffstory.narrative_plan import build_ledger_index
from diffstory.narrative_runtime import ArtifactRuntimeError
from diffstory.narrative_v2 import ProductionInputs
from diffstory.narrative_v2 import StageInputs
from diffstory.narrative_v2 import Transport
from diffstory.narrative_v2 import V2Options
from diffstory.narrative_wire import Aliases
from diffstory.narrative_wire import WireAdapter
from diffstory.narrative_wire import strict_json


class StrictJSONTests(unittest.TestCase):
    """Reject ambiguous properties and invalid Unicode or numeric values."""

    def test_nested_duplicates_constants_and_invalid_unicode(self) -> None:
        """Reject invalid values at every depth without echoing a body."""
        for raw in (
            b'{"items":[],"items":["private"]}',
            b'{"outer":{"text":"private","text":"replacement"}}',
            b'{"n":NaN}',
            b'{"n":Infinity}',
            b'{"n":-Infinity}',
            b'{"n":1e999}',
            b'{"text":"\xff"}',
            b'{"text":"\\ud800"}',
        ):
            with (
                self.subTest(raw=raw),
                self.assertRaises(ArtifactRuntimeError) as caught,
            ):
                strict_json(raw)
            self.assertEqual(caught.exception.reason, "malformed_json")
            self.assertNotIn("private", str(caught.exception))

    def test_valid_line_separators_and_source_newlines_survive(self) -> None:
        """Retain Unicode separators and escaped physical newlines exactly."""
        self.assertEqual(
            strict_json('{"source":"a\u2028b\u2029c\\r\\n"}'),
            {"source": "a\u2028b\u2029c\r\n"},
        )


class CompactWireTests(unittest.TestCase):
    """Reject ambiguous framing and support without losing independent targets."""

    def setUp(self) -> None:
        """Prepare actual fixture evidence and its runtime batch identity."""
        self.inputs = ProductionInputs(
            golden_report(), Transport(FixtureProvider(), V2Options())
        )
        self.stage = StageInputs(
            self.inputs, "detail", self.inputs.detail_items()
        )
        self.prepared = self.stage.prepare(
            tuple(self.stage.items), repair=False, partition=0
        )
        self.batch = self.prepared.metadata.batch
        self.data = strict_json(self.prepared.body)["data"]

    def test_alias_collision_is_detected_without_exposing_identity(
        self,
    ) -> None:
        """A compact-name collision cannot silently redirect a citation."""
        aliases = Aliases()
        with patch(
            "diffstory.narrative_wire.semantic_digest", return_value="a" * 64
        ):
            aliases.add("u", "first")
            with self.assertRaisesRegex(ArtifactRuntimeError, "schema"):
                aliases.add("u", "second")

    def test_bad_envelope_has_no_partial_salvage(self) -> None:
        """Version, shape and duplicate-property failures reject the whole envelope."""
        valid = FixtureProvider.respond(self.data)
        for raw in (
            valid | {"version": "older"},
            valid | {"extra": "private"},
            valid | {"items": {}},
            valid | {"acknowledgements": [None]},
            b'{"version":"diffstory.wire.v2","items":[],"items":[],"acknowledgements":[]}',
        ):
            with (
                self.subTest(raw_type=type(raw).__name__),
                self.assertRaises(ArtifactRuntimeError),
            ):
                self.inputs.wire.envelope(raw, self.batch, {})

    def test_cross_unit_unknown_and_duplicate_support_fail_only_affected_item(
        self,
    ) -> None:
        """A known target with invalid support remains an independently invalid item."""
        for mode in ("unknown", "duplicate", "peer"):
            with self.subTest(mode=mode):
                response = FixtureProvider.respond(self.data)
                target = response["items"][0]
                citations = target["explanations"][0]["citations"]
                if mode == "unknown":
                    citations.append("unknown-citation")
                elif mode == "duplicate":
                    citations.append(citations[0])
                else:
                    citations.append(
                        response["items"][1]["explanations"][0]["citations"][0]
                    )
                envelope = self.inputs.wire.envelope(response, self.batch, {})
                self.assertEqual(
                    len(envelope.items) + len(envelope.failures), 2
                )
                if mode == "peer":
                    bad = envelope.items[0]
                    from diffstory.narrative_contracts import DetailResult
                    from diffstory.narrative_plan import canonical_bytes

                    detail = DetailResult.model_validate_json(
                        canonical_bytes(bad)
                    )
                    with self.assertRaisesRegex(
                        ArtifactRuntimeError, "citation"
                    ):
                        self.stage.validate(
                            self.stage.lookups[detail.unit_key], detail
                        )
                else:
                    self.assertEqual(len(envelope.failures), 1)
                    self.assertEqual(len(envelope.items), 1)

    def test_ack_origin_cannot_be_forged_and_changed_units_cannot_ack(
        self,
    ) -> None:
        """Only the explicit ack array can acknowledge a no-obligation unit."""
        target = self.data["items"][0]["target"]
        forged = {
            "version": "diffstory.wire.v2",
            "items": [{"target": target, "ack": True}],
            "acknowledgements": [],
        }
        envelope = self.inputs.wire.envelope(forged, self.batch, {})
        self.assertEqual(envelope.items, ())
        self.assertEqual(len(envelope.failures), 1)
        explicit = {
            "version": "diffstory.wire.v2",
            "items": [],
            "acknowledgements": [target],
        }
        envelope = self.inputs.wire.envelope(explicit, self.batch, {})
        self.assertEqual(envelope.items, ())
        self.assertEqual(len(envelope.failures), 1)

    def test_alias_input_is_indexed_and_does_not_send_source_coordinates(
        self,
    ) -> None:
        """Wire source text stays exact without revision, fragment or line fields."""
        ledger = build_evidence_ledger(golden_report())
        wire = WireAdapter(build_ledger_index(ledger))
        for unit in ledger.units:
            value = wire.unit_input(unit)
            self.assertEqual(
                [s["text"] for s in value["spans"]],
                [s.text for s in unit.spans],
            )
            for span in value["spans"]:
                self.assertEqual(set(span), {"handle", "side", "path", "text"})

    def test_indexed_prose_context_serializes_selected_section_bytes(
        self,
    ) -> None:
        """Wire serialization could retain or rebuild complete consumed prose.

        Input/expected case matrix:
            expanded section: title, all arrays, one term, and aliases.
            indexed section: selected arena paths and the same alias order.
            request body: exact canonical bytes match between both routes.

        Preconditions:
            Details and prose rows share one owned arena and current indexed
            planning sources. The fixture has no external provider dispatch.
        Postconditions:
            ``ProseContext`` retains the indexed selector, while wire output and
            request bytes match the strict expanded compatibility adapter.
        """
        from diffstory.narrative_contracts import AuthorContext
        from diffstory.narrative_contracts import DetailResult
        from diffstory.narrative_contracts import ProseBinding
        from diffstory.narrative_contracts import SectionResult
        from diffstory.narrative_contracts import SupportedStatement
        from diffstory.narrative_contracts import SupportedTerminology
        from diffstory.narrative_contracts import SupportSet
        from diffstory.narrative_hierarchy import (
            capture_indexed_consumed_prose,
        )
        from diffstory.narrative_hierarchy import (
            derive_planning_sources_indexed,
        )
        from diffstory.narrative_manifest import validate_prose_support
        from diffstory.narrative_plan import (
            _capture_indexed_arena_detail_results,
        )
        from diffstory.narrative_plan import canonical_bytes
        from diffstory.narrative_plan import semantic_digest
        from diffstory.narrative_wire import ProseContext
        from diffstory.narrative_wire import WireAdapter
        from diffstory.report_storage import _RecordArena

        raw = FixtureProvider.respond(self.data)
        envelope = self.inputs.wire.envelope(raw, self.batch, {})
        self.assertFalse(envelope.failures)
        details = tuple(
            DetailResult.model_validate_json(canonical_bytes(item))
            for item in envelope.items
        )
        arena = _RecordArena()
        detail_refs = tuple(arena.put_result(item) for item in details)
        detail_context = _capture_indexed_arena_detail_results(
            detail_refs,
            arena,
            self.inputs.ledger,
            index=self.inputs.index,
        )
        sources = derive_planning_sources_indexed(
            self.inputs.index,
            detail_context,
            author_context=AuthorContext(),
        )
        claim = next(iter(sources._source_state.claim_records))  # noqa: SLF001 # Use one current accepted detail observation.
        statement = SupportedStatement(
            kind="implementation_claim",
            text="The selected claim remains in the request.",
            claims=(claim,),
        )
        subject = semantic_digest("wire section subject")
        plan_key = semantic_digest("wire section plan")
        binding = ProseBinding(
            kind="section",
            subject_key=subject,
            plan_key=plan_key,
            context_digest=semantic_digest("wire section context"),
        )
        expanded = SectionResult(
            item_key=binding.key,
            binding=binding,
            title=statement,
            orientation=(statement,),
            synthesis=(statement,),
            transition=(),
            terminology=(
                SupportedTerminology(term="wire term", definition=statement),
            ),
        )
        reference = arena.put_result(expanded)
        selector = capture_indexed_consumed_prose(
            arena._snapshot_reader(),  # noqa: SLF001 # Select the exact owned section result.
            (reference,),
            sources,
        )
        context = ProseContext(
            binding=binding,
            claims=(claim,),
            citations=(),
            author_digest=sources.author_digest,
            support=SupportSet(claims=(claim,)),
            consumed=selector,
            sources=sources,
        )
        self.assertIs(context.consumed, selector)

        validate_prose_support(
            expanded, context, WireAdapter(self.inputs.index)
        )
        expanded_input = WireAdapter(self.inputs.index).prose_input(expanded)
        indexed_input = WireAdapter(self.inputs.index).prose_input(
            selector.select(expanded.item_key)
        )
        self.assertEqual(
            canonical_bytes(indexed_input), canonical_bytes(expanded_input)
        )
        self.assertEqual(
            canonical_bytes(
                {"version": "diffstory.wire.v2", "data": indexed_input}
            ),
            canonical_bytes(
                {"version": "diffstory.wire.v2", "data": expanded_input}
            ),
        )
        other_index = build_ledger_index(
            build_evidence_ledger(golden_report())
        )
        with self.assertRaisesRegex(
            ValueError, "citation_scope_binding_mismatch"
        ):
            WireAdapter(other_index).prose_input(
                selector.select(expanded.item_key)
            )
        detail_context.citation_scope.close()

    def test_typed_basis_handles_reject_unconsumed_and_stale_proof(  # noqa: PLR0915 # Exercise both actual consumed support families.
        self,
    ) -> None:
        """Whole spans, unseen slices and changed child endpoints grant no support."""
        from dataclasses import replace
        from types import MappingProxyType

        from diffstory.narrative_contracts import DetailResult
        from diffstory.narrative_contracts import PlanningCardResult
        from diffstory.narrative_contracts import SourceSliceRef
        from diffstory.narrative_contracts import SupportSet
        from diffstory.narrative_hierarchy import planning_waves
        from diffstory.narrative_manifest import validate_prose_support
        from diffstory.narrative_plan import canonical_bytes
        from diffstory.narrative_plan import content_digest

        inputs = ProductionInputs(
            golden_report(1),
            Transport(
                FixtureProvider(),
                V2Options(
                    card_wire_bytes=1024,
                    card_support_bytes=2048,
                    prose_wire_bytes=4096,
                ),
            ),
        )
        inputs.transport.provider.context_tokens = 40000
        # Capacity/configuration is captured; construct again after explicit limits.
        inputs.transport.provider.max_output_tokens = 15000
        inputs = ProductionInputs(
            golden_report(1),
            Transport(inputs.transport.provider, inputs.options),
        )
        detail = StageInputs(inputs, "detail", inputs.detail_items())
        request = detail.prepare(
            tuple(detail.items), repair=False, partition=0
        )
        raw = FixtureProvider.respond(strict_json(request.body)["data"])
        raw["items"][0]["explanations"][0]["before"] = (
            "field continuation " * 3000
        )
        envelope = inputs.wire.envelope(raw, request.metadata.batch, {})
        details = tuple(
            DetailResult.model_validate_json(canonical_bytes(v))
            for v in envelope.items
        )
        plan = inputs.build_plan(inputs.bind_details(details))
        produced, sampled = {}, {}
        for wave in planning_waves(plan):
            stage = StageInputs(
                inputs,
                "planning",
                tuple(
                    inputs.card_item(inputs.cards[k], produced) for k in wave
                ),
            )
            for batch in stage.pack(tuple(stage.items)):
                request = stage.prepare(
                    batch.item_keys, repair=False, partition=batch.partition
                )
                raw = FixtureProvider.respond(
                    strict_json(request.body)["data"]
                )
                envelope = inputs.wire.envelope(
                    raw, request.metadata.batch, stage.contexts
                )
                self.assertEqual(envelope.failures, ())
                for value in envelope.items:
                    result = PlanningCardResult.model_validate_json(
                        canonical_bytes(value)
                    )
                    stage.validate(stage.lookups[result.item_key], result)
                    produced[result.binding.subject_key] = result
                    context = stage.contexts[result.item_key]
                    for ref in context.support.basis:
                        sampled.setdefault(ref.kind, (result, context, ref))
            if set(sampled) == {"source_slice", "statement"}:
                break
        self.assertEqual(set(sampled), {"source_slice", "statement"})
        result, context, ref = sampled["source_slice"]
        alias = inputs.wire.basis_alias(ref)
        self.assertTrue(alias.startswith("x"))
        with self.assertRaisesRegex(ArtifactRuntimeError, "citation"):
            inputs.wire.aliases.resolve("b", alias)
        statement = result.abstraction[0]
        for bad in (
            statement.model_copy(
                update={"basis": (), "citations": (ref.source,)}
            ),
            statement.model_copy(
                update={
                    "basis": (
                        SourceSliceRef(
                            source=ref.source,
                            byte_start=0,
                            byte_end=1,
                            content_digest=content_digest(
                                inputs.index.spans[
                                    ref.source.span_id
                                ].text.encode()[:1]
                            ),
                        ),
                    )
                }
            ),
        ):
            altered = result.model_copy(update={"abstraction": (bad,)})
            with self.assertRaisesRegex(ArtifactRuntimeError, "citation"):
                validate_prose_support(altered, context, inputs.wire)
        author = statement.model_copy(
            update={
                "kind": "author_motivation",
                "basis": (),
                "author_context_digest": inputs.author_digest,
            }
        )
        with self.assertRaisesRegex(ArtifactRuntimeError, "citation"):
            validate_prose_support(
                result.model_copy(update={"abstraction": (author,)}),
                context,
                inputs.wire,
            )
        result, context, ref = sampled["statement"]
        self.assertTrue(inputs.wire.basis_alias(ref).startswith("b"))
        for update in (
            {"result_digest": "0" * 64},
            {"statement_digest": "0" * 64},
            {"producer_item_key": "0" * 64},
            {
                "statement_path": ref.statement_path.model_copy(
                    update={"index": 999}
                )
            },
        ):
            bad = ref.model_copy(update=update)
            inputs.wire.basis_alias(bad)
            altered = result.model_copy(
                update={
                    "abstraction": (
                        result.abstraction[0].model_copy(
                            update={
                                "claims": (),
                                "citations": (),
                                "basis": (bad,),
                            }
                        ),
                    )
                }
            )
            forged = replace(context, support=SupportSet(basis=(bad,)))
            with self.assertRaisesRegex(ArtifactRuntimeError, "citation"):
                validate_prose_support(altered, forged, inputs.wire)
        with self.assertRaisesRegex(ArtifactRuntimeError, "citation"):
            validate_prose_support(
                result,
                replace(context, consumed=MappingProxyType({})),
                inputs.wire,
            )


if __name__ == "__main__":
    unittest.main()
