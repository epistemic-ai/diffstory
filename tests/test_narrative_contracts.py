"""Strict immutable contracts and provider-schema conversion tests."""

import json
import unittest

from pydantic import ValidationError

from diffstory.narrative_contracts import ArtifactDependency
from diffstory.narrative_contracts import ArtifactRecord
from diffstory.narrative_contracts import BatchResult
from diffstory.narrative_contracts import ClaimRef
from diffstory.narrative_contracts import DetailResult
from diffstory.narrative_contracts import DocumentResult
from diffstory.narrative_contracts import EvidenceUnit
from diffstory.narrative_contracts import ItemFailure
from diffstory.narrative_contracts import NarrativeResult
from diffstory.narrative_contracts import PlanningBounds
from diffstory.narrative_contracts import PlanningCardResult
from diffstory.narrative_contracts import ProseBinding
from diffstory.narrative_contracts import ProseKind
from diffstory.narrative_contracts import RecordFragment
from diffstory.narrative_contracts import SectionResult
from diffstory.narrative_contracts import SourceRecordOwner
from diffstory.narrative_contracts import SourceSliceRef
from diffstory.narrative_contracts import SourceSpan
from diffstory.narrative_contracts import SpanRef
from diffstory.narrative_contracts import StatementPath
from diffstory.narrative_contracts import StatementRef
from diffstory.narrative_contracts import SupportedStatement
from diffstory.narrative_contracts import SupportedTerminology
from diffstory.narrative_contracts import result_key
from diffstory.narrative_contracts import result_kind
from diffstory.narrative_contracts import result_model
from diffstory.narrative_contracts import result_stage
from diffstory.narrative_contracts import structured_output_schema
from diffstory.narrative_plan import content_digest
from diffstory.narrative_plan import semantic_digest


def prose_result(
    kind: ProseKind,
    subject: str,
    *,
    plan_key: str = "a" * 64,
    context: str = "b" * 64,
) -> NarrativeResult:
    """Produce structural supported-prose fixtures, not a semantic quality oracle."""
    binding = ProseBinding(
        kind=kind,
        subject_key=plan_key if kind == "document" else subject,
        plan_key=plan_key,
        context_digest=context,
    )
    statement = SupportedStatement(
        kind="implementation_claim",
        text="The new rstrip call retains leading whitespace while removing trailing whitespace.",
        claims=(
            ClaimRef(
                unit_key=subject,
                observation_id=semantic_digest("rstrip", subject),
            ),
        ),
        effect="Indentation remains in normalized output.",
        uncertainty=("No runtime test was executed.",),
    )
    editorial = SupportedStatement(
        kind="editorial", text="Read normalization before its callers."
    )
    fields = {"item_key": binding.key, "binding": binding}
    if kind == "planning_card":
        return PlanningCardResult(**fields, abstraction=(statement,))
    if kind == "section":
        return SectionResult(
            **fields,
            title=editorial,
            orientation=(editorial,),
            synthesis=(statement,),
            transition=(),
        )
    return DocumentResult(
        **fields,
        preamble=(statement,),
        lead=(editorial,),
        closing=(statement,),
    )


class ProseContractTests(unittest.TestCase):
    """Exercise family, support, version, and content-identity boundaries."""

    def test_version_extras_keys_and_family_are_strict(self) -> None:
        """Wrong context keys or families cannot be relabelled as valid prose."""
        for kind in ("planning_card", "section", "document"):
            result = prose_result(kind, "c" * 64)
            model = result_model(kind)
            self.assertEqual(
                model.model_validate_json(result.model_dump_json()), result
            )
            self.assertEqual(result_key(result), result.binding.key)
            self.assertEqual(result_kind(result), kind)
            for update in (
                {"version": "v1"},
                {"arbitrary": "field"},
                {"item_key": "f" * 64},
            ):
                with (
                    self.subTest(kind=kind, update=update),
                    self.assertRaises(ValidationError),
                ):
                    model.model_validate(result.model_dump() | update)
            with self.assertRaises(ValidationError):
                result.binding.context_digest = "e" * 64
            with self.assertRaises(ValidationError):
                model.model_validate(
                    result.model_copy(update={"item_key": "e" * 64})
                )
        card = prose_result("planning_card", "c" * 64)
        section_binding = ProseBinding(
            **(card.binding.model_dump() | {"kind": "section"})
        )
        with self.assertRaises(ValidationError):
            PlanningCardResult.model_validate(
                card.model_dump()
                | {"binding": section_binding, "item_key": section_binding.key}
            )
        self.assertEqual(result_stage("document"), "document")
        with self.assertRaises(KeyError):
            result_model("unknown")

    def test_claim_support_and_motivation_remain_distinct(self) -> None:
        """Implementation needs ClaimRef or SpanRef; author statements need attribution."""
        claim = ClaimRef(unit_key="c" * 64, observation_id="d" * 64)
        span = SpanRef(
            span_id="1" * 64,
            unit_key="c" * 64,
            group_key="e" * 64,
            side="head",
            path="normalize.py",
            byte_start=0,
            byte_end=9,
            line_start=1,
            line_end=1,
            content_digest="2" * 64,
        )
        for support in ({"claims": (claim,)}, {"citations": (span,)}):
            statement = SupportedStatement(
                kind="implementation_claim",
                text="Casefold replaces lower.",
                **support,
            )
            self.assertEqual(statement.text, "Casefold replaces lower.")
        for fields in (
            {"kind": "implementation_claim"},
            {"kind": "editorial", "effect": "Changes output"},
            {"kind": "author_motivation"},
            {"kind": "editorial", "author_context_digest": "a" * 64},
            {"kind": "implementation_claim", "claims": (claim, claim)},
            {"kind": "implementation_claim", "citations": (span, span)},
        ):
            with (
                self.subTest(fields=tuple(fields)),
                self.assertRaises(ValidationError),
            ):
                SupportedStatement(text="A statement", **fields)
        author = SupportedStatement(
            kind="author_motivation",
            text="The author wants Unicode normalization.",
            author_context_digest="3" * 64,
        )
        self.assertEqual(author.claims, ())
        with self.assertRaises(ValidationError):
            SupportedTerminology(term="normalization", definition=author)

    def test_prose_retains_effect_uncertainty_terms_and_empty_transition(
        self,
    ) -> None:
        """The content identity binds inputs and keeps full typed output unchanged."""
        section = prose_result("section", "c" * 64)
        term = SupportedTerminology(
            term="trimming", definition=section.synthesis[0]
        )
        section = SectionResult.model_validate(
            section.model_dump() | {"terminology": (term,)}
        )
        self.assertEqual(section.transition, ())
        self.assertEqual(
            section.terminology[0].definition.effect,
            "Indentation remains in normalized output.",
        )
        self.assertEqual(
            section.synthesis[0].uncertainty,
            ("No runtime test was executed.",),
        )
        altered = SectionResult.model_validate(
            section.model_dump()
            | {
                "title": SupportedStatement(
                    kind="editorial", text="Normalization tour"
                )
            }
        )
        self.assertEqual(altered.item_key, section.item_key)
        changed_context = prose_result("section", "c" * 64, context="d" * 64)
        self.assertNotEqual(changed_context.item_key, section.item_key)
        with self.assertRaises(ValidationError):
            SectionResult.model_validate(
                section.model_dump() | {"transition": ""}
            )

    def test_new_families_have_strict_schemas_and_typed_artifacts(
        self,
    ) -> None:
        """No schema permits extra objects, and no prose becomes a detail result."""
        for kind in ("planning_card", "section", "document"):
            result = prose_result(kind, "c" * 64)
            record = ArtifactRecord(
                semantic_key="d" * 64,
                dependencies=(),
                original_request_digest="e" * 64,
                validator_version="prose.v2",
                result=result,
            )
            self.assertEqual(
                ArtifactRecord.model_validate_json(record.model_dump_json()),
                record,
            )
            with self.assertRaises(ValidationError):
                BatchResult(batch_key="f" * 64, items=(result,), failures=())
            pending = [structured_output_schema(type(result))]
            while pending:
                node = pending.pop()
                if isinstance(node, dict):
                    if node.get("type") == "object":
                        self.assertFalse(node["additionalProperties"])
                        self.assertEqual(
                            set(node["required"]), set(node["properties"])
                        )
                    pending.extend(node.values())
                elif isinstance(node, list):
                    pending.extend(node)


class ContractTests(unittest.TestCase):
    """Reject scalar coercion, mutable containers, and untyped payload fields."""

    def setUp(self) -> None:
        """Create one complete strict source span."""
        self.fields = {
            "id": "a" * 64,
            "buffer_key": "b" * 64,
            "side": "base",
            "path": "src/x.py",
            "byte_start": 0,
            "byte_end": 2,
            "line_start": 1,
            "line_end": 1,
            "content_digest": content_digest("é".encode()),
            "text": "é",
        }

    def test_digest_path_and_integral_half_open_ranges(self) -> None:
        """Reject malformed digests, unsafe paths, coercions, and reversed bounds."""
        SourceSpan(**self.fields)
        invalid = [
            {"id": "a" * 63},
            {"id": "a" * 63 + "\n"},
            {"id": "A" * 64},
            {"path": "../x"},
            {"path": "/x"},
            {"path": "C:\\x"},
            {"path": "a//b"},
            {"path": "a/./b"},
            {"path": "a\\b"},
            {"byte_start": True},
            {"byte_end": "2"},
            {"byte_end": 2.0},
            {"byte_start": -1},
            {"byte_end": 1},
            {"line_start": 0},
            {"line_start": 2},
            {"extra": "no"},
            {"version": "v1"},
        ]
        for update in invalid:
            with self.subTest(update=update):
                with self.assertRaises(ValidationError):
                    SourceSpan(**(self.fields | update))
                with self.assertRaises(ValidationError):
                    SourceSpan.model_validate_json(
                        json.dumps(self.fields | update)
                    )

    def test_nested_models_and_tuples_are_frozen(self) -> None:
        """Strict Python construction rejects lists; JSON round trips retain tuples."""
        span = SourceSpan(**self.fields)
        unit = EvidenceUnit(
            id="c" * 64,
            group_key=None,
            change_key=None,
            classification=None,
            spans=(span,),
            obligations=(),
        )
        with self.assertRaises(ValidationError):
            unit.spans[0].text = "x"
        with self.assertRaises(ValidationError):
            unit.spans = ()
        with self.assertRaises(TypeError):
            unit.spans[0] = span
        with self.assertRaises(ValidationError):
            EvidenceUnit(**(unit.model_dump() | {"spans": [span]}))
        self.assertEqual(
            EvidenceUnit.model_validate_json(unit.model_dump_json()), unit
        )

    def test_batch_keys_and_artifact_dependencies_are_typed(self) -> None:
        """No arbitrary dictionary can replace a validated artifact payload."""
        result = DetailResult(
            unit_key="a" * 64, explanations=(), observations=(), questions=()
        )
        failure = ItemFailure(item_key="b" * 64, reason="invalid_citation")
        batch = BatchResult(
            batch_key="c" * 64, items=(result,), failures=(failure,)
        )
        self.assertEqual(
            BatchResult.model_validate_json(batch.model_dump_json()), batch
        )
        with self.assertRaises(ValidationError):
            BatchResult(
                batch_key="c" * 64, items=(result, result), failures=()
            )
        with self.assertRaises(ValidationError):
            BatchResult(
                batch_key="c" * 64,
                items=(result,),
                failures=(
                    failure.model_copy(update={"item_key": result.unit_key}),
                ),
            )
        record = ArtifactRecord(
            semantic_key="d" * 64,
            dependencies=(
                ArtifactDependency(kind="peer", key="e" * 64, digest="f" * 64),
            ),
            original_request_digest="1" * 64,
            validator_version="validator.v2",
            result=result,
        )
        self.assertEqual(
            ArtifactRecord.model_validate_json(record.model_dump_json()),
            record,
        )
        with self.assertRaises(ValidationError):
            ArtifactRecord(
                **(record.model_dump() | {"result": {"arbitrary": "payload"}})
            )
        with self.assertRaises(ValidationError):
            record.dependencies[0].kind = "model"

    def test_provider_schema_requires_nullable_fields_and_forbids_extras(
        self,
    ) -> None:
        """Provider conversion retains strict object fields and homogeneous arrays."""
        schema = structured_output_schema(DetailResult)
        pending = [schema]
        while pending:
            node = pending.pop()
            if isinstance(node, dict):
                self.assertNotIn("default", node)
                self.assertNotIn("prefixItems", node)
                if node.get("type") == "object":
                    self.assertFalse(node["additionalProperties"])
                    self.assertEqual(
                        set(node["required"]), set(node["properties"])
                    )
                pending.extend(node.values())
            elif isinstance(node, list):
                pending.extend(node)

    def test_production_owner_path_range_and_budget_contracts(self) -> None:
        """Production extensions remain frozen, discriminated and scalar-strict."""
        owner = SourceRecordOwner(
            unit_key="a" * 64, span_id="b" * 64, content_digest="c" * 64
        )
        fragment = RecordFragment(
            id="d" * 64,
            owner=owner,
            path="text",
            encoding="utf8",
            byte_start=0,
            byte_end=2,
            total_bytes=2,
            field_digest=content_digest(b"ab"),
            content_digest=content_digest(b"ab"),
        )
        self.assertEqual(
            RecordFragment.model_validate_json(fragment.model_dump_json()),
            fragment,
        )
        for update in (
            {"path": "arbitrary"},
            {"encoding": "normalized"},
            {"byte_start": True},
            {"byte_end": 3},
            {"byte_end": 0},
            {"owner": owner.model_dump() | {"kind": "author"}},
            {"owner": owner.model_dump() | {"revision": "current"}},
        ):
            with self.assertRaises(ValidationError):
                RecordFragment.model_validate(fragment.model_dump() | update)
        with self.assertRaises(ValidationError):
            fragment.owner.span_id = "f" * 64
        with self.assertRaises(ValidationError):
            PlanningBounds(
                card_wire_bytes=100,
                card_support_bytes=2048,
                prose_wire_bytes=4096,
                output_guard=15000,
            )

    def test_exact_basis_is_typed_and_does_not_change_detail_schema(
        self,
    ) -> None:
        """Only known basis families and statement paths enter supported prose."""
        ref = StatementRef(
            producer_item_key="a" * 64,
            result_digest="b" * 64,
            statement_path=StatementPath(field="abstraction"),
            statement_digest="c" * 64,
        )
        statement = SupportedStatement(
            kind="implementation_claim",
            text="An exact consumed statement supports this.",
            basis=(ref,),
        )
        self.assertEqual(
            SupportedStatement.model_validate_json(
                statement.model_dump_json()
            ),
            statement,
        )
        for update in (
            {"basis": (ref, ref)},
            {"basis": ({"kind": "arbitrary"},)},
        ):
            with self.assertRaises(ValidationError):
                SupportedStatement.model_validate(
                    statement.model_dump() | update
                )
        for path in (
            {"field": "uncertainty"},
            {"field": "title", "index": 1},
            {"field": "abstraction", "index": -1},
        ):
            with self.assertRaises(ValidationError):
                StatementPath(**path)
        span = SpanRef(
            span_id="1" * 64,
            unit_key="a" * 64,
            group_key=None,
            side="head",
            path="x.py",
            byte_start=0,
            byte_end=2,
            line_start=1,
            line_end=1,
            content_digest=content_digest(b"ab"),
        )
        with self.assertRaises(ValidationError):
            SourceSliceRef(
                source=span,
                byte_start=1,
                byte_end=1,
                content_digest=content_digest(b""),
            )
        self.assertNotIn(
            "basis", json.dumps(structured_output_schema(DetailResult))
        )
        self.assertNotIn(
            "SourceSliceRef", DetailResult.model_json_schema().get("$defs", {})
        )


if __name__ == "__main__":
    unittest.main()
