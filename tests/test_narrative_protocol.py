"""Exact codec, configuration and legacy identity checks without remote calls."""

from __future__ import annotations

import json
import unittest
from unittest.mock import patch

from pydantic import ValidationError
from test_narrative_v2 import FixtureProvider

from diffstory.narrative import CodexCLIProvider
from diffstory.narrative import OpenAIResponsesProvider
from diffstory.narrative_contracts import Contract
from diffstory.narrative_manifest import LedgerParameters
from diffstory.narrative_plan import RequestPackingLimits
from diffstory.narrative_plan import canonical_bytes
from diffstory.narrative_plan import semantic_digest
from diffstory.narrative_protocol import RequestConfiguration
from diffstory.narrative_protocol import serialize_request
from diffstory.narrative_runtime import ArtifactRuntimeError
from diffstory.narrative_v2 import COMMON_PROMPT
from diffstory.narrative_v2 import PROMPTS
from diffstory.narrative_v2 import OfflineTransport
from diffstory.narrative_v2 import Transport
from diffstory.narrative_v2 import V2Options
from diffstory.narrative_wire import wire_schema


class LayeredProvider(FixtureProvider):
    """Declare one extra JSON string layer without an arbitrary stored callable."""

    request_json_layers = 1
    request_prefix = b"\xff local framing\n"
    input_overhead_bytes = 111

    def prepare(self, *args: object) -> bytes:
        """Serialize the declared canonical wrapper around the neutral fixture."""
        return canonical_bytes({"payload": super().prepare(*args).decode()})


class ProtocolTests(unittest.TestCase):
    """Pure serialization must be byte-identical to each supported live adapter."""

    def test_wire_schemas_are_detached(self) -> None:
        """Mutating returned schemas cannot change later results."""
        stages = ("detail", "planning", "section", "document")
        for stage in stages:
            with self.subTest(stage=stage):
                expected = wire_schema(stage)
                returned = wire_schema(stage)
                returned["properties"]["version"]["caller_mutation"] = stage
                returned["properties"]["items"]["items"]["caller_mutation"] = (
                    stage
                )
                self.assertEqual(wire_schema(stage), expected)

    def test_wire_schemas_reuse_fixed_templates(self) -> None:
        """Repeated schema reads do not rebuild their fixed templates."""
        stages = ("detail", "planning", "section", "document")
        with patch(
            "diffstory.narrative_wire.structured_output_schema",
            side_effect=AssertionError("fixed schema rebuilt"),
        ) as build_schema:
            for stage in stages:
                with self.subTest(stage=stage):
                    for _ in range(3):
                        wire_schema(stage)
            build_schema.assert_not_called()

    def test_all_codecs_roundtrip_exact_initial_and_repair_bytes(self) -> None:
        """Keep Unicode, schemas, output fields, key order, prefix and repair data."""
        providers = (
            FixtureProvider(),
            LayeredProvider(),
            CodexCLIProvider(model="gpt-6.1-sol", executable="/opaque/codex"),
            OpenAIResponsesProvider(api_key="unused-fixture"),
        )
        options = V2Options(
            ledger=LedgerParameters(max_span_bytes=128, max_unit_spans=4),
            packing=RequestPackingLimits(
                framing_bytes=71, stable_partition_count=3, safety_tokens=777
            ),
            card_wire_bytes=2048,
            prose_wire_bytes=4096,
            card_support_bytes=2048,
            max_chapter_groups=3,
        )
        for provider in providers:
            live = Transport(provider, options)
            descriptor = RequestConfiguration.model_validate_json(
                live.descriptor.model_dump_json()
            )
            self.assertEqual(descriptor.options, options)
            self.assertEqual(
                descriptor.configuration_digest, live.config_digest
            )
            # Only nonzero fixture framing extends the exact legacy payload.
            old = descriptor.legacy_configuration()
            legacy = semantic_digest("diffstory.transport.v2", old)
            self.assertEqual(descriptor.identity_digest("legacy"), legacy)
            if descriptor.fixture_json_layers:
                self.assertNotEqual(live.config_digest, legacy)
            else:
                self.assertEqual(live.config_digest, legacy)
            with (
                patch.object(
                    type(provider),
                    "prepare",
                    side_effect=AssertionError("provider prepare"),
                ),
                patch(
                    "shutil.which",
                    side_effect=AssertionError("executable resolution"),
                ),
                patch(
                    "pathlib.Path.read_bytes",
                    side_effect=AssertionError("executable read"),
                ),
                patch(
                    "subprocess.Popen",
                    side_effect=AssertionError("subprocess"),
                ),
                patch("socket.socket", side_effect=AssertionError("network")),
            ):
                offline = OfflineTransport(descriptor)
            for stage in PROMPTS:
                for repair in (False, True):
                    data = {
                        "version": "diffstory.request.v2",
                        "stage": stage,
                        "repair": repair,
                        "repair_reason": "coverage" if repair else None,
                        "items": [{"text": 'é\r\n😀"\\\u2028\u2029'}],
                    }
                    self.assertEqual(
                        live.body(stage, data, 321),
                        offline.body(stage, data, 321),
                    )
                    self.assertEqual(
                        offline.body(stage, data, 321),
                        serialize_request(
                            descriptor,
                            PROMPTS[stage] + COMMON_PROMPT,
                            data,
                            f"diffstory_{stage}_v2",
                            wire_schema(stage),
                            321,
                        ),
                    )

    def test_descriptor_is_strict_frozen_complete_and_versioned(self) -> None:
        """Reject unknown fields, missing settings, wrong versions and unsafe defaults."""
        descriptor = Transport(FixtureProvider(), V2Options()).descriptor
        for updates in (
            {"protocol": "opaque.serializer"},
            {"version": "future.v9"},
            {"context_tokens": 0},
            {"max_output_tokens": True},
            {"prefix_hex": "0xdead"},
            {"fixture_json_layers": 3},
            {"arbitrary_options": {}},
            {"stages": []},
            {"destination": "https://unexpected.example"},
        ):
            with (
                self.subTest(updates=updates),
                self.assertRaises(ValidationError),
            ):
                RequestConfiguration.model_validate_json(
                    json.dumps(descriptor.model_dump(mode="json") | updates)
                )
        for field in RequestConfiguration.model_fields:
            data = descriptor.model_dump(mode="json")
            del data[field]
            with (
                self.subTest(missing=field),
                self.assertRaises(ValidationError),
            ):
                RequestConfiguration.model_validate_json(json.dumps(data))
        for path in (
            ("options", "card_wire_bytes"),
            ("options", "packing", "framing_bytes"),
            ("options", "ledger", "max_span_bytes"),
        ):
            data = descriptor.model_dump(mode="json")
            parent = data
            for key in path[:-1]:
                parent = parent[key]
            del parent[path[-1]]
            with (
                self.subTest(missing_path=path),
                self.assertRaises(ValidationError),
            ):
                RequestConfiguration.model_validate_json(json.dumps(data))
        for value, field in (
            (descriptor, "model"),
            (descriptor.options, "card_wire_bytes"),
            (descriptor.options.packing, "safety_tokens"),
        ):
            self.assertIsInstance(value, Contract)
            with self.assertRaises(ValidationError):
                setattr(value, field, 9)

    def test_opaque_or_mismatched_serializer_fails_before_send(self) -> None:
        """A declaration cannot disguise unrecognized actual provider bytes."""
        provider = FixtureProvider()
        provider.request_protocol = "unknown"
        with self.assertRaises(ValidationError):
            Transport(provider, V2Options())
        provider = FixtureProvider()
        provider.prepare = lambda *_args: b"{}"
        live = Transport(provider, V2Options())
        with self.assertRaisesRegex(ArtifactRuntimeError, "request_mismatch"):
            live.body("detail", {}, 321)
        self.assertEqual(provider.calls, [])

    def test_stale_stage_identity_and_nested_bypass_fail(self) -> None:
        """Saved digests and frozen-model bypasses are revalidated against current code."""
        descriptor = Transport(FixtureProvider(), V2Options()).descriptor
        data = descriptor.model_dump(mode="json")
        data["stages"][0]["schema_digest"] = "0" * 64
        with self.assertRaisesRegex(
            ArtifactRuntimeError, "dependencies_changed"
        ):
            OfflineTransport(
                RequestConfiguration.model_validate_json(json.dumps(data))
            )
        unknown = descriptor.model_copy(
            update={"protocol": "opaque", "version": "unknown"}
        )
        with self.assertRaises(ValidationError):
            serialize_request(unknown, "system", {}, "schema", {}, 321)
        object.__setattr__(descriptor.options.packing, "safety_tokens", -1)
        with self.assertRaises(ValidationError):
            OfflineTransport(descriptor)

    def test_pre_m4_zero_and_builtin_configuration_detail_batch_and_request_pins(
        self,
    ) -> None:
        """These constants came from the immutable f9a62a6 source and actual bytes."""
        from test_narrative_v2 import golden_report

        from diffstory.narrative_v2 import ProductionInputs
        from diffstory.narrative_v2 import StageInputs

        providers = (
            FixtureProvider(),
            CodexCLIProvider(model="gpt-6.1-sol", executable="/opaque/codex"),
            OpenAIResponsesProvider(api_key="unused-fixture"),
        )
        expected = (
            (
                "ff436a482181ea3789f1707f91158cadafaa6a4deef9dc66814d04ec7f09872f",
                "675f0416e6e51833681fbbd622220f78b8ed99e94319c8a45d7a85d3fd3ad01f",
                "1d8653877ffbe3ed61e755380e33bc25ef174875edd02fa411be18147031b998",
                "89bcbafbfb6d5861924afa004884692c1199c093283c4bb785816d7b5b32781e",
                "a974111b0b26f3aded81d293243942aa0bbbc5dc413b375fa5340e02b3d3e5a1",
            ),
            (
                "d4570ebec5905c105d62c8474d54cdb79fae645184ad7254a47ac7ddecfa2db7",
                "fba2761469dc51ca9dd894ebc3c568da2cf3d90fc37fb200b92f214185121cec",
                "c571da4f35d5799d524b81fe86dc8a65cfee41693207c5d433e4b15d0865601f",
                "8e2fe37884a0bdca1968939dcc926b6b98da8f930a388baafe9bf80384c0dfb6",
                "41a99b7f5b458023147b500dcd45897ddaa303ad372b9297ec63283f6be3150c",
            ),
            (
                "a36cd82eb501b46e29904dcd9812946a42ede12a594ac9a85b63e841459978e9",
                "3f90ccc6936d36fc40260a22eba9a456942a77c40acb2287e21effb496dc28ec",
                "87608ec40d643737264bdf4fef912f4e21c20eb504b7ca6a12102e17d16cb76a",
                "8eccee3263fc3aed49ddeaf9fc96b440988156db8e967d734699822589820423",
                "b6fc0975d476ad4e95454404d000db3201254cbacff4eec3a2c2cfcee0210752",
            ),
        )
        for provider, pinned in zip(providers, expected, strict=True):
            transport = Transport(provider, V2Options())
            inputs = ProductionInputs(golden_report(1), transport)
            stage = StageInputs(inputs, "detail", inputs.detail_items())
            batch = stage.pack(tuple(stage.items))[0]
            invocation = stage.invocation(batch)
            self.assertEqual(
                (
                    transport.config_digest,
                    invocation.lookups[0].key,
                    invocation.semantic_key,
                    batch.id,
                    batch.request_digest,
                ),
                pinned,
            )
            self.assertEqual(
                OfflineTransport(transport.descriptor).config_digest, pinned[0]
            )
