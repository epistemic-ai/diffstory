"""Strict immutable contracts and provider-schema conversion tests."""

import json
import unittest

from pydantic import ValidationError

from diffstory.narrative_contracts import ArtifactDependency
from diffstory.narrative_contracts import ArtifactRecord
from diffstory.narrative_contracts import BatchResult
from diffstory.narrative_contracts import DetailResult
from diffstory.narrative_contracts import EvidenceUnit
from diffstory.narrative_contracts import ItemFailure
from diffstory.narrative_contracts import SourceSpan
from diffstory.narrative_contracts import structured_output_schema
from diffstory.narrative_plan import content_digest


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


if __name__ == "__main__":
    unittest.main()
