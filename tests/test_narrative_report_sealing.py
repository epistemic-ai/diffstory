"""Source-free tests for indexed report sealing and the v2 narrator facade."""

from __future__ import annotations

import copy
import tempfile
import unittest
from dataclasses import replace
from types import MappingProxyType
from unittest.mock import patch

from test_narrative_v2 import FixtureProvider
from test_narrative_v2 import golden_report

from diffstory import narrative_session as _narrative_session
from diffstory.analysis import apply_annotations
from diffstory.narrative import Narrator
from diffstory.narrative_manifest import persisted_manifest
from diffstory.narrative_manifest import stored_generation
from diffstory.narrative_plan import _thaw_owned
from diffstory.narrative_plan import canonical_bytes
from diffstory.narrative_plan import content_digest
from diffstory.narrative_runtime import ArtifactRuntimeError
from diffstory.narrative_session import ReportSession
from diffstory.narrative_session import ValidatedGeneration
from diffstory.narrative_session import ValidatedReport
from diffstory.narrative_session import _CapturedInput
from diffstory.narrative_session import _freeze_report
from diffstory.narrative_session import (
    _snapshot_report as snapshot_report_for_probe,
)
from diffstory.report_storage import encode_report_storage
from diffstory.report_storage import expand_report_storage
from diffstory.report_storage import parse_report_storage


class IndexedReportSealingTests(unittest.TestCase):
    """Keep one real source-free generation alive for sealing comparisons.

    Members:
        report: Compiler fixture used for preview and production.
        narrator: v2 facade whose provider only returns local fixture data.
        session: Live owner of the captured report and issued capabilities.
        generation: Complete indexed generation handle from the facade.
        report_handle: Sealed applied report handle.
        expected_report: Independent expanded ``apply_annotations`` oracle.
    """

    @classmethod
    def setUpClass(cls) -> None:
        """Generate once through the facade and retain exact indexed handles."""
        cls.recovery = tempfile.TemporaryDirectory()
        cls.report = golden_report()
        cls.narrator = Narrator(
            FixtureProvider(),
            architecture="v2",
            recovery_dir=cls.recovery.name,
            concurrency=1,
        )
        with patch.object(
            _narrative_session,
            "_snapshot_report",
            wraps=snapshot_report_for_probe,
        ) as snapshot_report:
            cls.narrator.preview(cls.report)
            cls.preview_capture = cls.narrator._v2.inputs._captured_input  # noqa: SLF001 # Retain the exact preview capture for ownership evidence.
            cls.session = cls.narrator.capture_session(cls.report)
            cls.generation = cls.narrator.generate_validated(
                cls.report, session=cls.session
            )
            cls.snapshot_report_calls = snapshot_report.call_count
        cls.capture_adopted = (
            cls.session._captured_input is cls.preview_capture  # noqa: SLF001 # The facade must adopt this exact preview snapshot.
        )
        cls.proof = cls.session._registered_payload(  # noqa: SLF001 # Inspect only the session's exact registered generation payload.
            cls.generation, ValidatedGeneration, "generation"
        )
        cls.report_handle = cls.session.apply(cls.generation)
        cls.application = cls.session._checked_report_data(  # noqa: SLF001 # Verify the documented trusted payload accessor shape.
            cls.report_handle
        )
        cls.annotation_oracle = expand_report_storage(
            cls.proof.graph,
            max_output_bytes=128 * 1024 * 1024,
        )
        cls.annotation_oracle["generation"] = stored_generation(
            persisted_manifest(cls.annotation_oracle["generation"])
        )
        cls.annotation_bytes = encode_report_storage(
            cls.annotation_oracle, "annotations"
        )
        cls.expected_report = apply_annotations(
            cls.report, cls.annotation_oracle
        )

    @classmethod
    def tearDownClass(cls) -> None:
        """Close all retained session authority and temporary recovery data."""
        cls.session.close()
        cls.recovery.cleanup()

    def test_facade_returns_sealed_handle_with_compatibility_oracle_parity(
        self,
    ) -> None:
        """Concern: fresh v2 facade output keeps one capture and exact defaults.

        Input/expected matrix:
        * A previewed source-free report and matching session -> facade returns
          a ``ValidatedGeneration`` and apply returns a ``ValidatedReport``.
        * The exact preview capture -> owned by the session, and direct
          ``_snapshot_report`` instrumentation counts only one capture across
          preview, session creation, and generation.
        * Explicit ``report_data`` -> byte-equivalent expanded oracle and the
          existing generation-storage.v1 external representation.
        * Annotation/report graphs -> all non-reader rows are shared and the
          annotation-only reader row is absent from the report closure.
        """
        self.assertTrue(self.capture_adopted)
        self.assertEqual(self.snapshot_report_calls, 1)
        self.assertTrue(self.narrator._generation_started)  # noqa: SLF001 # The facade mirrors its one-run state.
        self.assertIsInstance(self.generation, ValidatedGeneration)
        self.assertIsInstance(self.report_handle, ValidatedReport)
        self.assertIs(
            self.application.base_report,
            self.session._captured_input.report_view,  # noqa: SLF001 # The checked payload retains the exact captured immutable base.
        )
        self.assertEqual(
            canonical_bytes(self.session.report_data(self.report_handle)),
            canonical_bytes(self.expected_report),
        )
        self.assertEqual(
            self.session.report_data(self.report_handle)["generation"][
                "storage_version"
            ],
            "diffstory.generation-storage.v1",
        )
        source_graph = self.proof.graph
        report_graph = self.application.graph
        self.assertIs(
            report_graph._root["generation"],  # noqa: SLF001 # The report root keeps the exact completed generation selector graph.
            source_graph._root["generation"],  # noqa: SLF001
        )
        old_reader_id = source_graph._root["reader"]["$ref"]  # noqa: SLF001 # Identify only the source annotation reader selector.
        new_reader_id = report_graph._root["reader"]["$ref"]  # noqa: SLF001
        self.assertNotEqual(old_reader_id, new_reader_id)
        self.assertNotIn(old_reader_id, report_graph._tables["readers"])  # noqa: SLF001 # Report closure omits the replaced annotation reader.
        for table, source_rows in source_graph._tables.items():  # noqa: SLF001 # Compare shared row identity without resolving records.
            if table == "readers":
                continue
            destination_rows = report_graph._tables[table]  # noqa: SLF001
            for identifier in source_rows:
                self.assertIs(
                    source_rows[identifier],
                    destination_rows[identifier],
                )

        stored_graph = report_graph.to_bytes()
        parsed_graph = parse_report_storage(stored_graph)
        self.assertEqual(parsed_graph.artifact_kind, "report")
        self.assertEqual(len(parsed_graph.storage_counts), 13)
        self.assertEqual(
            set(parsed_graph._tables),  # noqa: SLF001 # Strict parsing must keep the exact closed table set.
            set(report_graph._tables),  # noqa: SLF001
        )
        for table, parsed_rows in parsed_graph._tables.items():  # noqa: SLF001 # Compare exact serialized table identities.
            graph_rows = report_graph._tables[table]  # noqa: SLF001
            self.assertEqual(set(parsed_rows), set(graph_rows))
            for identifier, entry in parsed_rows.items():
                self.assertEqual(entry["id"], identifier)

    def test_capture_session_starts_preview_when_needed(self) -> None:
        """Concern: session creation cannot snapshot a report twice.

        Input/expected matrix:
        * A fresh v2 narrator and source-free report -> the factory runs its
          normal preview, returns a session over that exact capture, and
          `_snapshot_report` runs once.
        """
        recovery = tempfile.TemporaryDirectory()
        narrator = Narrator(
            FixtureProvider(),
            architecture="v2",
            recovery_dir=recovery.name,
            concurrency=1,
        )
        report = golden_report()
        with patch.object(
            _narrative_session,
            "_snapshot_report",
            wraps=snapshot_report_for_probe,
        ) as snapshot_report:
            session = narrator.capture_session(report)
        try:
            self.assertEqual(snapshot_report.call_count, 1)
            self.assertIs(
                session._captured_input,  # noqa: SLF001 # The session directly owns preview's exact capture.
                narrator._v2.inputs._captured_input,  # noqa: SLF001
            )
        finally:
            session.close()
            recovery.cleanup()

    def test_capture_session_rechecks_report_provider_options_and_run(
        self,
    ) -> None:
        """Concern: session creation requires the exact approved v2 inputs.

        Input/expected matrix:
        * Changed caller report, facade provider, provider configuration,
          v2 options, or production run -> no session is attached.
        """
        cases = (
            ("report", "dependencies_changed"),
            ("facade_provider", "dependencies_changed"),
            ("provider_configuration", "dependencies_changed"),
            ("options", "dependencies_changed"),
            ("run", "dependencies_changed"),
        )
        for name, expected_reason in cases:
            with self.subTest(changed_input=name):
                recovery = tempfile.TemporaryDirectory()
                provider = FixtureProvider()
                narrator = Narrator(
                    provider,
                    architecture="v2",
                    recovery_dir=recovery.name,
                    concurrency=1,
                )
                report = golden_report()
                narrator.preview(report)
                if name == "report":
                    report["stats"]["factory_probe"] = 1
                elif name == "facade_provider":
                    narrator.provider = FixtureProvider()
                elif name == "provider_configuration":
                    object.__setattr__(
                        provider, "context_tokens", provider.context_tokens + 1
                    )
                elif name == "options":
                    options = narrator._v2.options  # noqa: SLF001 # Mutate only the source-free frozen options fixture.
                    object.__setattr__(
                        options, "card_wire_bytes", options.card_wire_bytes + 1
                    )
                else:
                    run = narrator._v2.inputs.run  # noqa: SLF001 # Mutate only the source-free preview's captured run.
                    object.__setattr__(run, "model", f"{run.model}-changed")
                try:
                    with self.assertRaisesRegex(
                        ArtifactRuntimeError, expected_reason
                    ):
                        narrator.capture_session(report)
                finally:
                    recovery.cleanup()

    def test_capture_session_rejects_reused_or_closed_capture(self) -> None:
        """Concern: one preview capture cannot acquire a second session owner.

        Input/expected matrix:
        * A live capture already owned by one session -> second factory call
          fails with the ownership error.
        * A closed preview capture -> factory fails its liveness check.
        """
        for state in ("owned", "closed"):
            with self.subTest(state=state):
                recovery = tempfile.TemporaryDirectory()
                narrator = Narrator(
                    FixtureProvider(),
                    architecture="v2",
                    recovery_dir=recovery.name,
                    concurrency=1,
                )
                report = golden_report()
                session = narrator.capture_session(report)
                try:
                    if state == "closed":
                        session.close()
                        expected_error = ArtifactRuntimeError
                        expected_message = "dependencies_changed"
                    else:
                        expected_error = ValueError
                        expected_message = "captured_input_session_mismatch"
                    with self.assertRaisesRegex(
                        expected_error, expected_message
                    ):
                        narrator.capture_session(report)
                finally:
                    session.close()
                    recovery.cleanup()

    def test_capture_session_rejects_started_and_legacy_narrators(
        self,
    ) -> None:
        """Concern: only an unstarted v2 narrator can create its session.

        Input/expected matrix:
        * Legacy narrator -> explicit v2-only factory error.
        * Started v2 narrator -> one-generation guard rejects a second scope.
        """
        legacy = Narrator(FixtureProvider())
        with self.assertRaisesRegex(ValueError, "session_capture_requires_v2"):
            legacy.capture_session(self.report)

        recovery = tempfile.TemporaryDirectory()
        narrator = Narrator(
            FixtureProvider(),
            architecture="v2",
            recovery_dir=recovery.name,
            concurrency=1,
        )
        narrator.preview(self.report)
        narrator._generation_started = True  # noqa: SLF001 # Exercise the facade's one-generation state gate.
        try:
            with self.assertRaisesRegex(ValueError, "only one generation job"):
                narrator.capture_session(self.report)
        finally:
            recovery.cleanup()

    def test_apply_checks_capture_once_at_entry(self) -> None:
        """Concern: apply must not repeat a full capture check while issuing.

        Input/expected matrix:
        * A current registered generation -> apply invokes the actual
          `_CapturedInput.assert_capture_current` method exactly once.
        """
        capture = self.session._captured_input  # noqa: SLF001 # Count the current full capture boundary.
        original = _CapturedInput.assert_capture_current
        calls = []

        def count_capture_check(
            instance: object, *args: object, **kwargs: object
        ) -> None:
            calls.append(instance)
            original(instance, *args, **kwargs)

        with patch.object(
            _CapturedInput, "assert_capture_current", count_capture_check
        ):
            handle = self.session.apply(self.generation)
        self.assertIsInstance(handle, ValidatedReport)
        self.assertEqual(calls, [capture])

    def test_apply_rejects_stale_captured_data_before_registration(
        self,
    ) -> None:
        """Concern: mutated captured state cannot issue a report capability.

        Input/expected matrix:
        * A generation whose session capture witness changed -> apply rejects
          at its one entry check and registers no report handle.
        """
        capture = self.session._captured_input  # noqa: SLF001 # Mutate only a synthetic in-memory witness.
        original_digest = capture._report_digest  # noqa: SLF001 # Restore the fixture after the adversarial check.
        report_count = sum(
            kind == "report"
            for _handle, kind, _payload in self.session._registrations.values()  # noqa: SLF001 # Count only report capabilities already registered.
        )
        object.__setattr__(capture, "_report_digest", "0" * 64)
        try:
            with self.assertRaisesRegex(ValueError, "captured_input_changed"):
                self.session.apply(self.generation)
            self.assertEqual(
                sum(
                    kind == "report"
                    for _handle, kind, _payload in self.session._registrations.values()  # noqa: SLF001 # Failure must not add a report capability.
                ),
                report_count,
            )
        finally:
            object.__setattr__(capture, "_report_digest", original_digest)

    def test_apply_uses_completed_proof_without_expansion_or_rebuild(
        self,
    ) -> None:
        """Concern: sealing does not reconstruct generation proof or graph data.

        Input/expected matrix:
        * A live completed fresh generation -> apply succeeds while the full
          production validator and graph expander are forbidden.
        * Its capture and ledger object -> retain exact identity through apply.
        """
        capture = self.session._captured_input  # noqa: SLF001 # Pin the current immutable session scope.
        ledger = self.narrator._v2.inputs.ledger  # noqa: SLF001 # Pin the exact production ledger.
        with (
            patch(
                "diffstory.narrative_v2.validate_production",
                side_effect=AssertionError("apply rebuilt production proof"),
            ),
            patch(
                "diffstory.report_storage.expand_report_storage",
                side_effect=AssertionError("apply expanded the graph"),
            ),
        ):
            report_handle = self.session.apply(self.generation)
        self.assertIsInstance(report_handle, ValidatedReport)
        self.assertIs(self.session._captured_input, capture)  # noqa: SLF001
        self.assertIs(self.narrator._v2.inputs.ledger, ledger)  # noqa: SLF001

    def test_existing_renderer_predicates_reject_invalid_effective_fields(  # noqa: C901 # The required adversarial cases are one renderer matrix.
        self,
    ) -> None:
        """Concern: owned renderer checks cannot miss malformed applied fields.

        Input/expected matrix:
        * Invalid schema, collections, IDs, counts, tags, group links,
          passages, document values, and provenance labels -> renderer error.
        * Each case starts from the valid compiler base and deterministic
          overlay, then freezes exact owned views before the shared predicates.
        """
        from diffstory.render import _validate_indexed_report_surface

        overlay_template = _thaw_owned(self.application.overlay)
        reader = self.application.reader
        unknown_id = "f" * 16
        while any(
            unknown_id == group["id"] for group in self.report["groups"]
        ) or any(
            unknown_id == change["id"] for change in self.report["changes"]
        ):
            unknown_id = content_digest(unknown_id)[:16]

        def bad_schema(report: dict, _overlay: dict) -> None:
            report["schema"] = "wrong.schema"

        def bad_collection(report: dict, _overlay: dict) -> None:
            report["warnings"] = [None]

        def bool_number(_report: dict, overlay: dict) -> None:
            overlay["groups"][0]["number"] = True

        def negative_stat(report: dict, _overlay: dict) -> None:
            report["stats"]["renderer_probe"] = -1

        def invalid_identifier(report: dict, overlay: dict) -> None:
            invalid = "bad-id"
            report["groups"][0]["id"] = invalid
            overlay["groups"][0]["group_id"] = invalid

        def invalid_tag(report: dict, _overlay: dict) -> None:
            report["raw_files"] = [{"tag": "rename"}]

        def duplicate_groups(_report: dict, overlay: dict) -> None:
            overlay["groups"][1]["group_id"] = overlay["groups"][0]["group_id"]

        def duplicate_changes(report: dict, _overlay: dict) -> None:
            report["changes"][1]["id"] = report["changes"][0]["id"]

        def unknown_change(report: dict, overlay: dict) -> None:
            report["groups"][0]["change_ids"] = [unknown_id]
            overlay["groups"][0]["narrative"]["evidence_change_ids"] = [
                unknown_id
            ]

        def unknown_prerequisite(report: dict, _overlay: dict) -> None:
            report["groups"][0]["prerequisites"] = [unknown_id]

        def unknown_next_group(_report: dict, overlay: dict) -> None:
            overlay["groups"][0]["next_id"] = unknown_id

        def invalid_passage(report: dict, overlay: dict) -> None:
            change_id = report["groups"][0]["change_ids"][0]
            overlay["groups"][0]["narrative"]["passages"] = [
                {
                    "text": "A selected source passage.",
                    "change_ids": [change_id],
                    "view": "unsupported",
                }
            ]

        def invalid_document(_report: dict, overlay: dict) -> None:
            overlay["document"]["lead"] = 7

        def invalid_provenance(_report: dict, overlay: dict) -> None:
            overlay["groups"][0]["narrative"]["provenance"] = "verified"

        cases = (
            ("schema", bad_schema, reader),
            ("collection", bad_collection, reader),
            ("bool_number", bool_number, reader),
            ("negative_stat", negative_stat, reader),
            ("identifier", invalid_identifier, reader),
            ("tag", invalid_tag, reader),
            ("duplicate_group", duplicate_groups, reader),
            ("duplicate_change", duplicate_changes, reader),
            ("change_link", unknown_change, reader),
            ("prerequisite", unknown_prerequisite, reader),
            ("next_group", unknown_next_group, reader),
            ("passage", invalid_passage, reader),
            ("document", invalid_document, reader),
            ("provenance", invalid_provenance, reader),
            (
                "reader_label",
                lambda _report, _overlay: None,
                replace(reader, provenance="author-verified"),
            ),
        )
        for name, mutate, selected_reader in cases:
            with self.subTest(renderer_case=name):
                report = copy.deepcopy(self.report)
                overlay = copy.deepcopy(overlay_template)
                mutate(report, overlay)
                with self.assertRaises(ValueError):
                    _validate_indexed_report_surface(
                        _freeze_report(report),
                        _freeze_report(overlay),
                        selected_reader,
                    )

    def test_invalid_renderer_base_fails_before_report_capability_issue(
        self,
    ) -> None:
        """Concern: apply registers no report handle after a renderer failure.

        Input/expected matrix:
        * Valid indexed annotations plus a compiler base with a boolean stat ->
          import proof completes, renderer sealing rejects it, and no report
          capability enters the session registry.
        """
        invalid_report = copy.deepcopy(self.report)
        invalid_report["stats"]["renderer_probe"] = True
        recovery = tempfile.TemporaryDirectory()
        narrator = Narrator(
            FixtureProvider(),
            architecture="v2",
            recovery_dir=recovery.name,
            concurrency=1,
        )
        narrator.preview(invalid_report)
        session = narrator.capture_session(invalid_report)
        try:
            generation = narrator.generate_validated(
                invalid_report, session=session
            )
            with self.assertRaisesRegex(
                ValueError, "Invalid statistic: renderer_probe"
            ):
                session.apply(generation)
            self.assertFalse(
                any(
                    kind == "report"
                    for _handle, kind, _payload in session._registrations.values()  # noqa: SLF001 # Confirm failure occurred before report registration.
                )
            )
        finally:
            session.close()
            recovery.cleanup()

    def test_checked_accessor_rejects_graph_root_and_selector_substitution(
        self,
    ) -> None:
        """Concern: frozen-field substitutions cannot retain report authority.

        Input/expected matrix:
        * A copied-value graph root installed with ``object.__setattr__`` ->
          rejected at the next checked report-data boundary.
        * An equal-value reader projection replacement on the payload ->
          rejected at the same boundary by identity pins.
        """
        for substitution in ("root", "reader"):
            with self.subTest(substitution=substitution):
                session, generation = ReportSession.import_indexed(
                    self.annotation_bytes,
                    compiler_report=self.report,
                    max_input_bytes=len(self.annotation_bytes),
                )
                try:
                    handle = session.apply(generation)
                    payload = session._checked_report_data(  # noqa: SLF001 # Use the documented checked payload accessor.
                        handle
                    )
                    if substitution == "root":
                        graph = payload.graph
                        object.__setattr__(
                            graph,
                            "_root",
                            MappingProxyType(dict(graph._root)),  # noqa: SLF001 # Simulate a frozen root-field substitution.
                        )
                    else:
                        object.__setattr__(
                            payload,
                            "reader",
                            replace(payload.reader),
                        )
                    with self.assertRaisesRegex(
                        ValueError, "stale_session_capability"
                    ):
                        session._checked_report_data(  # noqa: SLF001 # Check that the accessor rejects a substituted selector.
                            handle
                        )
                finally:
                    session.close()

    def test_legacy_facade_rejects_session_handle_generation(self) -> None:
        """Concern: the new handle path remains explicit v2-only behavior.

        Input/expected matrix:
        * Library-default legacy narrator -> clear ``validated_generation``
          error before preview or provider dispatch.
        """
        legacy = Narrator(FixtureProvider())
        with self.assertRaisesRegex(
            ValueError, "validated_generation_requires_v2"
        ):
            legacy.generate_validated(self.report, session=object())
        with self.assertRaisesRegex(ValueError, "session_capture_requires_v2"):
            legacy.capture_session(self.report)


if __name__ == "__main__":
    unittest.main()
