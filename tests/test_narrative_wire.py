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


if __name__ == "__main__":
    unittest.main()
