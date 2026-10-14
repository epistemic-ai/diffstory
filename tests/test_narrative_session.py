"""Offline tests for immutable report sessions and imported proof reuse."""

from __future__ import annotations

import copy
import hashlib
import json
import pickle
import tempfile
import unittest
from unittest.mock import patch

from test_narrative_runtime import changed_record
from test_narrative_v2 import FixtureProvider
from test_narrative_v2 import golden_report

from diffstory.analysis import apply_annotations
from diffstory.narrative import Narrator
from diffstory.narrative_manifest import GenerationManifest
from diffstory.narrative_manifest import compiler_fact_signature
from diffstory.narrative_manifest import result_digest
from diffstory.narrative_plan import canonical_bytes
from diffstory.narrative_plan import canonical_digest
from diffstory.narrative_protocol import RequestConfiguration
from diffstory.narrative_protocol import V2Options
from diffstory.narrative_runtime import ArtifactRuntimeError
from diffstory.narrative_session import ReportSession
from diffstory.narrative_session import ValidatedGeneration
from diffstory.narrative_session import ValidatedReport
from diffstory.render import render
from diffstory.render import validate_report


def replace_result(
    manifest: GenerationManifest, family: str, position: int, result: object
) -> GenerationManifest:
    """Change one result and keep its receipt digest structurally coherent."""
    values = list(getattr(manifest, family))
    values[position] = result
    receipts = tuple(
        changed_record(receipt, result_digest=result_digest(result))
        if receipt.item_key == result.item_key
        else receipt
        for receipt in manifest.receipts
    )
    return changed_record(
        manifest, **{family: tuple(values)}, receipts=receipts
    )


class ReportSessionTests(unittest.TestCase):
    """Use complete synthetic public manifests and no live provider calls."""

    @classmethod
    def setUpClass(cls) -> None:
        """Build one fixture generation with the repository's local provider."""
        cls.report = golden_report()
        cls.provider = FixtureProvider()
        with tempfile.TemporaryDirectory() as directory:
            cls.annotations = Narrator(
                cls.provider,
                architecture="v2",
                recovery_dir=directory,
                concurrency=1,
            ).generate(cls.report)
        cls.configuration = copy.deepcopy(
            cls.annotations["generation"]["request_configuration"]
        )
        cls.applied = apply_annotations(cls.report, cls.annotations)

    def assert_import_rejected(
        self,
        report: dict,
        configuration: dict,
        manifest: object,
        *,
        candidate: object | None = None,
    ) -> None:
        """Require a failed import to issue no generation capability."""
        with (
            ReportSession.capture(report, configuration) as session,
            self.assertRaises((ValueError, ArtifactRuntimeError)),
        ):
            session.validate_import(manifest, candidate=candidate)

    def test_import_apply_matches_public_application_and_render(self) -> None:
        """A completed import reuses its proof and preserves public output."""
        from diffstory.narrative_plan import build_evidence_ledger
        from diffstory.narrative_plan import build_ledger_index
        from diffstory.narrative_v2 import validate_production

        with (
            patch(
                "test_narrative_v2.FixtureProvider.prepare",
                side_effect=AssertionError("provider request during import"),
            ),
            patch(
                "test_narrative_v2.FixtureProvider.complete_reply",
                side_effect=AssertionError("provider request during import"),
            ),
            patch(
                "diffstory.narrative_v2.validate_production",
                wraps=validate_production,
            ) as production,
            patch(
                "diffstory.narrative_v2.build_evidence_ledger",
                wraps=build_evidence_ledger,
            ) as ledgers,
            patch(
                "diffstory.narrative_v2.build_ledger_index",
                wraps=build_ledger_index,
            ) as indexes,
            ReportSession.capture(self.report, self.configuration) as session,
        ):
            generation = session.validate_import(
                self.annotations["generation"],
                candidate=self.annotations,
            )
            self.assertIs(type(generation), ValidatedGeneration)
            self.assertEqual(production.call_count, 1)
            self.assertEqual(ledgers.call_count, 1)
            self.assertEqual(indexes.call_count, 1)

            report_handle = session.apply(generation)
            self.assertIs(type(report_handle), ValidatedReport)
            self.assertEqual(production.call_count, 1)
            self.assertEqual(ledgers.call_count, 1)
            self.assertEqual(indexes.call_count, 1)

            detached = session.report_data(report_handle)
            self.assertEqual(detached, self.applied)
            validate_report(copy.deepcopy(detached))
            self.assertEqual(production.call_count, 2)
            html = render(copy.deepcopy(detached))
            self.assertIn("diffstory.presentation.v2", html)
            self.assertEqual(production.call_count, 3)

            public_applied = apply_annotations(self.report, self.annotations)
            self.assertEqual(public_applied, self.applied)
            self.assertEqual(production.call_count, 4)

    def test_capture_and_validated_data_are_detached(self) -> None:
        """Caller mutation cannot change a captured scope or retained proof."""
        report = copy.deepcopy(self.report)
        configuration = copy.deepcopy(self.configuration)
        candidate = copy.deepcopy(self.annotations)
        manifest = GenerationManifest.model_validate_json(
            canonical_bytes(candidate["generation"])
        )
        expected_run = copy.deepcopy(candidate["generation"]["run"])

        with ReportSession.capture(report, configuration) as session:
            report["source_evidence"]["fragments"][0]["text"] = "changed"
            configuration["options"]["ledger"]["max_span_bytes"] += 1
            generation = session.validate_import(manifest, candidate=candidate)

            object.__setattr__(
                manifest.details[0].explanations[0],
                "before",
                "changed frozen data",
            )
            candidate["steps"][0]["details"][0]["explanations"][0][
                "before"
            ] = "changed candidate"
            candidate["generation"]["run"]["report_signature"] = "0" * 64

            report_handle = session.apply(generation)
            detached = session.report_data(report_handle)
            self.assertEqual(detached, self.applied)
            self.assertEqual(detached["generation"]["run"], expected_run)
            detached["groups"][0]["title"] = "mutated returned data"
            detached["source_evidence"]["fragments"][0]["text"] = "mutated"
            self.assertEqual(session.report_data(report_handle), self.applied)

            captured = session.captured_report()
            captured["meta"]["description"] = "mutated projection"
            descriptor = session.request_configuration_data()
            descriptor["stages"][0]["prompt_digest"] = "0" * 64
            self.assertEqual(session.captured_report(), self.report)
            self.assertEqual(
                session.request_configuration_data(), self.configuration
            )

    def test_canonical_bytes_keeps_the_standard_encoder_for_native_and_models(
        self,
    ) -> None:
        """Per-leaf Python serialization could slow ordinary identity values."""
        native = {
            "unicode": "café 🧭",
            "values": [True, 1, None, -0.0],
            "nested": {"z": 2, "a": 1},
        }
        defaulted_options = V2Options()
        typed = RequestConfiguration.model_validate_json(
            canonical_bytes(self.configuration)
        )
        cases = [
            (
                "native",
                native,
                json.dumps(
                    native,
                    sort_keys=True,
                    separators=(",", ":"),
                    ensure_ascii=False,
                    allow_nan=False,
                ).encode("utf-8"),
            ),
            (
                "defaulted-model",
                defaulted_options,
                json.dumps(
                    defaulted_options.model_dump(mode="json"),
                    sort_keys=True,
                    separators=(",", ":"),
                    ensure_ascii=False,
                    allow_nan=False,
                ).encode("utf-8"),
            ),
            (
                "typed-model",
                typed,
                json.dumps(
                    typed.model_dump(mode="json"),
                    sort_keys=True,
                    separators=(",", ":"),
                    ensure_ascii=False,
                    allow_nan=False,
                ).encode("utf-8"),
            ),
        ]
        with patch(
            "diffstory.narrative_plan._canonical_chunks",
            side_effect=AssertionError(
                "ordinary values entered Python chunk loop"
            ),
        ):
            for name, value, expected in cases:
                with self.subTest(name=name):
                    self.assertEqual(canonical_bytes(value), expected)
        for name, value, expected in cases:
            with self.subTest(name=f"digest-{name}"):
                self.assertEqual(
                    canonical_digest(value),
                    hashlib.sha256(expected).hexdigest(),
                )

    def test_owned_report_digest_matches_standard_json_for_raw_python_shapes(
        self,
    ) -> None:
        """Streaming capture could change Unicode or Python-shape identity."""
        cases = [
            (
                "unicode-scalars-and-arrays",
                {
                    "meta": {"title": "café 🧭", "revision": None},
                    "values": [True, 1, False, 0, None, "e\u0301"],
                    "tuple_value": (True, 1),
                    "list_value": [True, 1],
                    "empty_mapping": {},
                    "empty_array": [],
                },
            )
        ]
        for name, report in cases:
            with (
                self.subTest(name=name),
                ReportSession.capture(report, self.configuration) as session,
            ):
                expected = json.dumps(
                    report,
                    sort_keys=True,
                    separators=(",", ":"),
                    ensure_ascii=False,
                    allow_nan=False,
                ).encode("utf-8")
                capture = session._captured_input  # noqa: SLF001 # Verify the internal owned stream without using the public expansion as its oracle.
                self.assertEqual(
                    capture.report_digest,
                    hashlib.sha256(expected).hexdigest(),
                )
                self.assertEqual(
                    canonical_bytes(capture.report_view), expected
                )
                self.assertEqual(
                    canonical_digest(capture.report_view),
                    hashlib.sha256(expected).hexdigest(),
                )
                self.assertEqual(session.captured_report(), report)

    def test_changed_owned_group_title_cannot_reuse_completed_import(
        self,
    ) -> None:
        """A changed captured title could be hidden by the derived overlay."""
        cases = [
            (
                "group-title",
                "changed captured title",
                "captured_input_changed",
            )
        ]
        for name, changed_title, expected_error in cases:
            with (
                self.subTest(name=name),
                ReportSession.capture(
                    self.report, self.configuration
                ) as session,
            ):
                generation = session.validate_import(
                    self.annotations["generation"],
                    candidate=self.annotations,
                )
                capture = session._captured_input  # noqa: SLF001 # Reproduce the frozen-owned-view bypass.
                root = capture.report_view
                group = root["groups"][0]
                original_entries = group.entries
                old_signature = compiler_fact_signature(root)
                object.__setattr__(
                    group,
                    "entries",
                    tuple(
                        (
                            key,
                            changed_title if key == "title" else value,
                        )
                        for key, value in original_entries
                    ),
                )
                self.assertEqual(compiler_fact_signature(root), old_signature)
                with self.assertRaisesRegex(ValueError, expected_error):
                    session.apply(generation)
                self.assertFalse(
                    any(
                        kind == "report"
                        for _handle, kind, _data in session._registrations.values()  # noqa: SLF001 # Confirm failed apply issued no report handle.
                    )
                )
                object.__setattr__(group, "entries", original_entries)

    def test_changed_session_version_cannot_issue_import_capability(
        self,
    ) -> None:
        """A changed proof version could issue a capability under a stale scope."""
        cases = [
            (
                "validation-version",
                "diffstory.session.import-proof.changed",
                "captured_input_changed",
            )
        ]
        for name, changed_version, expected_error in cases:
            with (
                self.subTest(name=name),
                ReportSession.capture(
                    self.report, self.configuration
                ) as session,
            ):
                with (
                    patch(
                        "diffstory.narrative_session._VALIDATION_VERSION",
                        changed_version,
                    ),
                    self.assertRaisesRegex(ValueError, expected_error),
                ):
                    session.validate_import(
                        self.annotations["generation"],
                        candidate=self.annotations,
                    )
                self.assertEqual(session._registrations, {})  # noqa: SLF001 # Confirm stale versions issued no session capability.

    def test_capture_accepts_configuration_models_and_serialized_snapshots(
        self,
    ) -> None:
        """Typed descriptors and immutable JSON snapshots use the same proof."""
        descriptor = RequestConfiguration.model_validate_json(
            canonical_bytes(self.configuration)
        )
        with ReportSession.capture(self.report, descriptor) as session:
            generation = session.validate_import(
                self.annotations["generation"], candidate=self.annotations
            )
            report_handle = session.apply(generation)
            self.assertEqual(session.report_data(report_handle), self.applied)

        serialized_report = canonical_bytes(self.report)
        serialized_config = canonical_bytes(self.configuration)
        input_report = json.loads(serialized_report)
        expected = apply_annotations(input_report, self.annotations)
        with ReportSession.capture(
            serialized_report, serialized_config
        ) as session:
            generation = session.validate_import(
                self.annotations["generation"], candidate=self.annotations
            )
            report_handle = session.apply(generation)
            self.assertEqual(session.report_data(report_handle), expected)

    def test_configuration_manifest_and_current_report_bindings(self) -> None:
        """Wrong descriptors and compiler facts cannot enter the proof scope."""
        wrong_configuration = copy.deepcopy(self.configuration)
        wrong_configuration["stages"][0]["prompt_digest"] = "0" * 64
        self.assert_import_rejected(
            self.report,
            wrong_configuration,
            self.annotations["generation"],
            candidate=self.annotations,
        )

        wrong_manifest = copy.deepcopy(self.annotations["generation"])
        wrong_manifest["request_configuration"]["model"] = "other-model"
        self.assert_import_rejected(
            self.report,
            self.configuration,
            wrong_manifest,
            candidate={**self.annotations, "generation": wrong_manifest},
        )

        wrong_current = copy.deepcopy(self.report)
        wrong_current["meta"]["description"] = "different compiler context"
        self.assert_import_rejected(
            wrong_current,
            self.configuration,
            self.annotations["generation"],
            candidate=self.annotations,
        )

    def test_parsed_fake_copied_cross_session_and_closed_handles_fail(
        self,
    ) -> None:
        """Only the exact live registered object may authorize an operation."""
        with self.assertRaises(TypeError):
            ReportSession(b"{}", {}, b"{}")
        session = ReportSession.capture(self.report, self.configuration)
        parsed = session.parse_import(
            self.annotations["generation"], candidate=self.annotations
        )
        with self.assertRaises(ValueError):
            session.apply(parsed)  # type: ignore[arg-type]
        with self.assertRaises(TypeError):
            ValidatedGeneration()
        forged_generation = object.__new__(ValidatedGeneration)
        with self.assertRaises(ValueError):
            session.apply(forged_generation)
        with self.assertRaises(TypeError):
            copy.copy(session)
        with self.assertRaises(TypeError):
            copy.copy(parsed)
        with self.assertRaises(TypeError):
            pickle.dumps(parsed)
        with self.assertRaises(TypeError):
            pickle.dumps(session)

        serialized = json.loads(json.dumps({"capability": repr(parsed)}))
        with self.assertRaises((ValueError, ArtifactRuntimeError)):
            session.validate_import(serialized)

        generation = session.validate_import(parsed)
        with self.assertRaises(AttributeError):
            generation.validation_complete = True  # type: ignore[attr-defined]
        with self.assertRaises(TypeError):
            copy.copy(generation)
        with self.assertRaises(TypeError):
            pickle.dumps(generation)
        with self.assertRaises(TypeError):
            json.dumps(generation)
        with self.assertRaises(ValueError):
            session.report_data(parsed)  # type: ignore[arg-type]
        other = ReportSession.capture(self.report, self.configuration)
        with self.assertRaises(ValueError):
            other.apply(generation)
        other.close()

        report_handle = session.apply(generation)
        forged_report = object.__new__(ValidatedReport)
        with self.assertRaises(ValueError):
            session.report_data(forged_report)
        session.close()
        with self.assertRaises(ValueError):
            session.apply(generation)
        with self.assertRaises(ValueError):
            session.report_data(report_handle)

    def test_reimport_of_applied_report_keeps_original_run_identity(
        self,
    ) -> None:
        """The original pre-annotation run is distinct from current report facts."""
        original_run = self.annotations["generation"]["run"]
        with ReportSession.capture(
            self.applied, self.configuration
        ) as session:
            generation = session.validate_import(
                self.annotations["generation"], candidate=self.annotations
            )
            applied_handle = session.apply(generation)
            result = session.report_data(applied_handle)
            self.assertEqual(result["generation"]["run"], original_run)
            self.assertEqual(
                result["generation"]["compiler_fact_signature"],
                compiler_fact_signature(result),
            )

    def test_tampered_source_peer_request_and_configuration_proofs_fail(
        self,
    ) -> None:
        """Coherent rehashes still need current source and request authority."""
        manifest = GenerationManifest.model_validate_json(
            canonical_bytes(self.annotations["generation"])
        )

        section_position = 0
        section = manifest.sections[section_position]
        statement_position = next(
            index
            for index, statement in enumerate(section.synthesis)
            if statement.basis
        )
        statement = section.synthesis[statement_position]
        bad_reference = changed_record(
            statement.basis[0], producer_item_key="0" * 64
        )
        bad_statement = changed_record(
            statement,
            basis=(bad_reference, *statement.basis[1:]),
        )
        bad_section = changed_record(
            section,
            synthesis=(
                *section.synthesis[:statement_position],
                bad_statement,
                *section.synthesis[statement_position + 1 :],
            ),
        )
        bad_source = replace_result(
            manifest, "sections", section_position, bad_section
        )
        self.assert_import_rejected(
            self.report, self.configuration, bad_source
        )

        original_producer = manifest.producers[0]
        dependencies = list(original_producer.invocation.dependencies)
        matching_key = next(
            dependency.key
            for dependency in dependencies
            if dependency.kind == "peer"
        )
        dependencies = [
            changed_record(dependency, digest="0" * 64)
            if dependency.key == matching_key
            and dependency.kind in {"peer", "evidence"}
            else dependency
            for dependency in dependencies
        ]
        bad_invocation = changed_record(
            original_producer.invocation, dependencies=tuple(dependencies)
        )
        bad_producer = changed_record(
            original_producer, invocation=bad_invocation
        )
        artifact_keys = bad_invocation.artifact_keys()
        bad_peers = changed_record(
            manifest,
            producers=(bad_producer, *manifest.producers[1:]),
            receipts=tuple(
                changed_record(
                    receipt,
                    producer_key=bad_producer.key,
                    artifact_key=artifact_keys[receipt.item_key],
                )
                if receipt.producer_key == original_producer.key
                else receipt
                for receipt in manifest.receipts
            ),
        )
        self.assert_import_rejected(self.report, self.configuration, bad_peers)

        first_request = original_producer.requests[0]
        bad_request = changed_record(
            first_request,
            batch=changed_record(
                first_request.batch,
                input_bound=first_request.batch.input_bound + 1,
            ),
        )
        request_producer = changed_record(
            original_producer,
            requests=(bad_request, *original_producer.requests[1:]),
        )
        bad_requests = changed_record(
            manifest,
            producers=(request_producer, *manifest.producers[1:]),
            receipts=tuple(
                changed_record(receipt, producer_key=request_producer.key)
                if receipt.producer_key == original_producer.key
                else receipt
                for receipt in manifest.receipts
            ),
        )
        self.assert_import_rejected(
            self.report, self.configuration, bad_requests
        )

        bad_configuration = changed_record(
            manifest.request_configuration,
            context_tokens=manifest.request_configuration.context_tokens + 1,
        )
        bad_configuration_manifest = changed_record(
            manifest, request_configuration=bad_configuration
        )
        self.assert_import_rejected(
            self.report,
            bad_configuration.model_dump(mode="json"),
            bad_configuration_manifest,
        )

    def test_coherently_rehashed_renderer_fields_cannot_seal(self) -> None:
        """Renderer IDs, exact numeric types and tags are checked after proof."""
        for field in (
            {"id": "f" * 64},
            {"number": True},
            {"tag": "invalid"},
        ):
            report = copy.deepcopy(self.report)
            report["extra_compiler_record"] = field
            candidate = copy.deepcopy(self.annotations)
            candidate["generation"]["compiler_fact_signature"] = (
                compiler_fact_signature(report)
            )
            with (
                self.subTest(field=field),
                ReportSession.capture(report, self.configuration) as session,
            ):
                generation = session.validate_import(
                    candidate["generation"], candidate=candidate
                )
                with self.assertRaisesRegex(
                    ValueError, "Invalid report|Invalid diff"
                ):
                    session.apply(generation)

    def test_invalid_links_do_not_receive_a_report_handle(self) -> None:
        """Candidate order and compiler prerequisite links remain checked."""
        candidate = copy.deepcopy(self.annotations)
        candidate["steps"][0]["next_id"] = "f" * 16
        self.assert_import_rejected(
            self.report,
            self.configuration,
            candidate["generation"],
            candidate=candidate,
        )

        report = copy.deepcopy(self.report)
        report["groups"][0]["prerequisites"] = ["f" * 16]
        candidate = copy.deepcopy(self.annotations)
        candidate["generation"]["compiler_fact_signature"] = (
            compiler_fact_signature(report)
        )
        with (
            ReportSession.capture(report, self.configuration) as session,
            self.assertRaises((KeyError, ValueError, ArtifactRuntimeError)),
        ):
            session.validate_import(
                candidate["generation"], candidate=candidate
            )


if __name__ == "__main__":
    unittest.main()
