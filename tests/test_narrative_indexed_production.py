# ruff: noqa: SLF001  # Proof authority tests inspect private session registrations.

"""Focused tests for session-owned fresh indexed production completion."""

from __future__ import annotations

import copy
import tempfile
import unittest
from dataclasses import replace
from unittest.mock import patch

from test_narrative_v2 import FixtureProvider
from test_narrative_v2 import golden_report

from diffstory.narrative import Narrator
from diffstory.narrative_hierarchy import planning_waves
from diffstory.narrative_manifest import expanded_generation
from diffstory.narrative_manifest import persisted_manifest
from diffstory.narrative_manifest import validate_generation
from diffstory.narrative_plan import build_evidence_ledger
from diffstory.narrative_plan import build_ledger_index
from diffstory.narrative_protocol import V2Options
from diffstory.narrative_session import ReportSession
from diffstory.narrative_session import StageProof
from diffstory.narrative_session import ValidatedGeneration
from diffstory.narrative_session import _context_witness
from diffstory.narrative_session import _IndexedGenerationProof
from diffstory.narrative_session import _stage_proof_witness
from diffstory.narrative_v2 import BatchedNarrator
from diffstory.narrative_v2 import Transport
from diffstory.report_storage import GenerationRecordSet


class NarrativeIndexedProductionTests(unittest.TestCase):
    """Use locally authored reports and an in-process provider only."""

    def _coordinator(
        self, provider: FixtureProvider, directory: str
    ) -> tuple[BatchedNarrator, ReportSession, dict]:
        """Create one coordinator and exact report/configuration session."""
        report = golden_report(2)
        options = V2Options()
        descriptor = Transport(provider, options).descriptor
        session = ReportSession.capture(report, descriptor)
        coordinator = BatchedNarrator(
            provider,
            recovery_dir=directory,
            concurrency=1,
            options=options,
        )
        return coordinator, session, report

    def test_fresh_finish_reuses_completed_owned_rows(self) -> None:
        """Offline reconstruction or whole-output expansion could replace stage evidence."""
        cases = [
            ("two synthetic units", 2, 1, 1, 0, 0),
        ]
        for (
            name,
            unit_count,
            expected_ledger_builds,
            expected_index_builds,
            expected_offline,
            expected_expansions,
        ) in cases:
            with (
                self.subTest(name=name),
                tempfile.TemporaryDirectory() as directory,
            ):
                report = golden_report(unit_count)
                provider = FixtureProvider()
                options = V2Options()
                session = ReportSession.capture(
                    report, Transport(provider, options).descriptor
                )
                coordinator = BatchedNarrator(
                    provider,
                    recovery_dir=directory,
                    concurrency=1,
                    options=options,
                )
                with (
                    patch(
                        "diffstory.narrative_v2.build_evidence_ledger",
                        wraps=build_evidence_ledger,
                    ) as ledger_builds,
                    patch(
                        "diffstory.narrative_v2.build_ledger_index",
                        wraps=build_ledger_index,
                    ) as index_builds,
                    patch(
                        "diffstory.narrative_v2.validate_production",
                        side_effect=AssertionError(
                            "fresh finish used offline proof"
                        ),
                    ) as offline_proof,
                    patch(
                        "diffstory.report_storage.expand_report_storage",
                        side_effect=AssertionError(
                            "validated generation expanded"
                        ),
                    ) as expansions,
                ):
                    generation = coordinator.generate_validated(
                        report, session=session
                    )

                proof = session._registered_payload(
                    generation, ValidatedGeneration, "generation"
                )
                self.assertIs(type(generation), ValidatedGeneration)
                self.assertIs(type(proof), _IndexedGenerationProof)
                self.assertEqual(
                    ledger_builds.call_count, expected_ledger_builds
                )
                self.assertEqual(
                    index_builds.call_count, expected_index_builds
                )
                self.assertEqual(offline_proof.call_count, expected_offline)
                self.assertEqual(expansions.call_count, expected_expansions)
                self.assertEqual(
                    len(proof.stage_proofs),
                    1 + len(planning_waves(coordinator.inputs.plan)) + 2,
                )
                self.assertTrue(
                    all(
                        type(stage_proof) is StageProof
                        for stage_proof in proof.stage_proofs
                    )
                )
                self.assertEqual(
                    tuple(
                        reference.item_key
                        for reference in proof.records.details
                    ),
                    tuple(coordinator.inputs.index.units),
                )
                session.close()

    def test_finish_checks_each_registered_stage_once(self) -> None:
        """Record-set construction could repeat the complete proof traversal.

        Input/expected case matrix:
            one authored two-unit report: one ordered proof check per registered
                detail, planning, section, and document stage.

        Preconditions:
            The coordinator returns all actual successful stage proofs and the
            session's pending data-only record set is unchanged.
        Postconditions:
            Finish validates each exact registered stage once before it issues
            the generation capability.
        """
        cases = (("two synthetic units", 2),)
        finish = ReportSession.finish_production
        check_proof = (
            ReportSession._check_stage_proof_current
        )  # Count the complete ordered traversal itself.

        for name, unit_count in cases:
            with (
                self.subTest(case=name),
                tempfile.TemporaryDirectory() as directory,
            ):
                report = golden_report(unit_count)
                provider = FixtureProvider()
                options = V2Options()
                session = ReportSession.capture(
                    report, Transport(provider, options).descriptor
                )
                coordinator = BatchedNarrator(
                    provider,
                    recovery_dir=directory,
                    concurrency=1,
                    options=options,
                )
                actual_checks = []

                def measured_finish(
                    target: ReportSession,
                    records: GenerationRecordSet,
                    stage_proofs: tuple[StageProof, ...],
                    _actual_checks: list[int] = actual_checks,
                ) -> ValidatedGeneration:
                    """Count only checks performed by finish itself."""
                    count = 0

                    def count_proof_check(
                        owner: ReportSession, *args: object, **kwargs: object
                    ) -> object:
                        nonlocal count
                        count += 1
                        return check_proof(owner, *args, **kwargs)

                    with patch.object(
                        ReportSession,
                        "_check_stage_proof_current",
                        new=count_proof_check,
                    ):
                        result = finish(target, records, stage_proofs)
                    _actual_checks.append(count)
                    self.assertEqual(count, len(stage_proofs))
                    return result

                with patch.object(
                    ReportSession, "finish_production", new=measured_finish
                ):
                    generation = coordinator.generate_validated(
                        report, session=session
                    )

                proof = session._registered_payload(
                    generation, ValidatedGeneration, "generation"
                )
                self.assertEqual(actual_checks, [len(proof.stage_proofs)])
                session.close()

    def test_empty_stage_proofs_keep_completion_rejection_reasons(
        self,
    ) -> None:
        """Empty proof sets could reach record indexing or change rejection reasons."""
        cases = (
            (
                "fresh session without stage registrations",
                False,
                "arbitrary",
                "stage_proof_coverage",
            ),
            (
                "incomplete stage binding with partial data",
                True,
                "partial",
                "incomplete_stage_proof",
            ),
            (
                "incomplete stage binding with arbitrary data",
                True,
                "arbitrary",
                "incomplete_stage_proof",
            ),
        )
        for (
            name,
            register_incomplete_stage,
            data_kind,
            expected_error,
        ) in cases:
            with (
                self.subTest(name=name),
                tempfile.TemporaryDirectory() as directory,
            ):
                provider = FixtureProvider()
                coordinator, session, report = self._coordinator(
                    provider, directory
                )
                if register_incomplete_stage:
                    coordinator.preview(report, session=session)
                    annotations = coordinator.generate(report)
                    records = copy.deepcopy(annotations["generation"])
                    if data_kind == "partial":
                        records["details"] = records["details"][:-1]
                else:
                    records = {"generation": {"details": [], "producers": []}}
                if data_kind == "arbitrary":
                    records = {"generation": {"details": [], "producers": []}}

                registrations_before = dict(session._registrations)
                with self.assertRaisesRegex(ValueError, expected_error):
                    session.finish_production(records, ())

                self.assertEqual(session._registrations, registrations_before)
                self.assertFalse(
                    any(
                        kind in {"generation", "report"}
                        for _handle, kind, _payload in session._registrations.values()
                    )
                )
                session.close()

    def test_stage_proof_check_validates_source_once_across_views(
        self,
    ) -> None:
        """Compatibility aliases could repeat a full source metadata walk.

        Input/expected case matrix:
            section proof with multiple compatibility contexts: 1 source check
                and 1 metadata fingerprint for this operation.
            document proof with multiple compatibility contexts: the same
                bounded counts.

        Preconditions:
            The proofs are complete and all compatibility views are generated
            by the in-process fixture provider.
        Postconditions:
            Context and selector checks use the exact just-checked source owner
            without repeating source metadata validation.
        """
        from diffstory.narrative_hierarchy import IndexedPlanningSources
        from diffstory.narrative_hierarchy import _IndexedSourceState

        with tempfile.TemporaryDirectory() as directory:
            provider = FixtureProvider()
            coordinator, session, report = self._coordinator(
                provider, directory
            )
            generation = coordinator.generate_validated(
                report, session=session
            )
            proof = session._registered_payload(
                generation, ValidatedGeneration, "generation"
            )
            stage_proofs = tuple(
                stage_proof
                for stage_proof in proof.stage_proofs
                if session._registrations[id(stage_proof)][
                    2
                ].binding.stage_name
                in {"section", "document"}
            )
            cases = (
                ("section", stage_proofs[0], (1, 1)),
                ("document", stage_proofs[-1], (1, 1)),
            )
            check_current = IndexedPlanningSources.check_current
            fingerprint = _IndexedSourceState.current_metadata_fingerprint
            context_witness = _context_witness

            for name, stage_proof, expected in cases:
                evidence = session._registrations[id(stage_proof)][2]
                context_count = sum(
                    len(view.contexts)
                    for view in (
                        *evidence.audit_replay.views,
                        *evidence.audit_replay.legacy_alternatives,
                    )
                )
                self.assertGreater(context_count, 1)
                counts = {"source": 0, "fingerprint": 0}
                context_ids = []

                def count_source(
                    sources: IndexedPlanningSources,
                    _counts: dict[str, int] = counts,
                ) -> None:
                    _counts["source"] += 1
                    check_current(sources)

                def count_fingerprint(
                    state: _IndexedSourceState,
                    _counts: dict[str, int] = counts,
                ) -> str:
                    _counts["fingerprint"] += 1
                    return fingerprint(state)

                def count_context(
                    context: object | None,
                    checked_sources: object | None,
                    _context_ids: list[int] = context_ids,
                ) -> str | None:
                    if context is not None:
                        _context_ids.append(id(context))
                    return context_witness(context, checked_sources)

                with (
                    self.subTest(stage=name),
                    patch.object(
                        IndexedPlanningSources,
                        "check_current",
                        new=count_source,
                    ),
                    patch.object(
                        _IndexedSourceState,
                        "current_metadata_fingerprint",
                        new=count_fingerprint,
                    ),
                    patch(
                        "diffstory.narrative_session._context_witness",
                        new=count_context,
                    ),
                ):
                    checked = session._check_stage_proof(stage_proof)

                self.assertIs(checked, evidence)
                self.assertEqual(
                    (counts["source"], counts["fingerprint"]), expected
                )
                self.assertEqual(len(context_ids), len(set(context_ids)))
            session.close()

    def test_stage_context_and_selector_require_the_checked_source_owner(
        self,
    ) -> None:
        """Equal content could let a context or selector borrow another owner.

        Input/expected case matrix:
            context.sources replaced by another object: reject the context.
            selector._sources replaced by another object: reject the selector.

        Preconditions:
            Each selected stage proof has a live indexed context and selector.
        Postconditions:
            The source adapter and consumed selector remain pinned to the exact
            source owner checked by the enclosing witness traversal.
        """
        with tempfile.TemporaryDirectory() as directory:
            provider = FixtureProvider()
            coordinator, session, report = self._coordinator(
                provider, directory
            )
            generation = coordinator.generate_validated(
                report, session=session
            )
            proof = session._registered_payload(
                generation, ValidatedGeneration, "generation"
            )
            stage_proof = proof.stage_proofs[-1]
            evidence = session._registrations[id(stage_proof)][2]
            context = next(iter(evidence.stage.contexts.values()))
            selector = context.consumed
            cases = (
                (
                    "context source owner",
                    context,
                    "sources",
                    "stage_context_source_mismatch",
                ),
                (
                    "selector source owner",
                    selector,
                    "_sources",
                    "stage_selector_source_mismatch",
                ),
            )
            for name, target, attribute, reason in cases:
                original = getattr(target, attribute)
                with self.subTest(owner=name):
                    object.__setattr__(target, attribute, object())
                    try:
                        with self.assertRaisesRegex(ValueError, reason):
                            session._check_stage_binding(evidence.stage)
                    finally:
                        object.__setattr__(target, attribute, original)
                    self.assertIs(
                        session._check_stage_binding(evidence.stage),
                        evidence.binding,
                    )
            session.close()

    def test_pending_recordset_rejects_changed_headers_and_references(  # noqa: PLR0915 # One matrix binds all mutable fields to the session witness.
        self,
    ) -> None:
        """Changed data-only records could replace the production the session completed."""

        def changed_reference(records: GenerationRecordSet, name: str) -> None:
            """Replace one captured address with a valid-shaped false address."""
            attribute = f"_{name}"
            current = object.__getattribute__(records, attribute)
            if name in {"document", "run_ref", "plan_ref"}:
                reference = list(current)
                reference[2] = (
                    "0" * 64 if reference[2] != "0" * 64 else "f" * 64
                )
                object.__setattr__(records, attribute, tuple(reference))
                return
            references = list(current)
            reference = list(references[0])
            reference[2] = "0" * 64 if reference[2] != "0" * 64 else "f" * 64
            references[0] = tuple(reference)
            object.__setattr__(records, attribute, tuple(references))

        def reversed_references(
            records: GenerationRecordSet, name: str
        ) -> None:
            """Reverse one ordered group when it has multiple references."""
            attribute = f"_{name}"
            references = object.__getattribute__(records, attribute)
            reordered = tuple(reversed(references))
            self.assertNotEqual(reordered, references)
            object.__setattr__(records, attribute, reordered)

        def rehashed_header(records: GenerationRecordSet) -> None:
            """Install a distinct header with its own valid captured digest."""
            header = dict(records.header)
            unit_keys = list(header["unit_keys"])
            header["unit_keys"] = list(reversed(unit_keys))
            if header["unit_keys"] == unit_keys:
                if "provenance" in header:
                    header.pop("provenance")
                else:
                    header["provenance"] = "unverified"
            changed = GenerationRecordSet(
                header,
                run_ref=records.run_ref,
                details=records.details,
                plan_ref=records.plan_ref,
                planning=records.planning,
                sections=records.sections,
                document=records.document,
                producers=records.producers,
                receipts=records.receipts,
            )
            object.__setattr__(records, "_header", changed._header)
            object.__setattr__(
                records, "_header_witness", changed._header_witness
            )

        sequence_fields = (
            "details",
            "planning",
            "sections",
            "producers",
            "receipts",
        )
        with tempfile.TemporaryDirectory() as directory:
            provider = FixtureProvider()
            coordinator, session, report = self._coordinator(
                provider, directory
            )
            finish = ReportSession.finish_production

            def reject_mutations_then_finish(
                target: ReportSession,
                records: GenerationRecordSet,
                stage_proofs: tuple[StageProof, ...],
            ) -> ValidatedGeneration:
                """Check pending and final uses against the saved session witness."""
                mutation_cases = [
                    (
                        "coherently rehashed valid header",
                        rehashed_header,
                        ("_header", "_header_witness"),
                        "production_recordset_changed",
                    )
                ]
                mutation_cases.extend(
                    (
                        f"changed {field} reference",
                        lambda value, field=field: changed_reference(
                            value, field
                        ),
                        (f"_{field}",),
                        "production_recordset_changed",
                    )
                    for field in sequence_fields
                )
                mutation_cases.append(
                    (
                        "changed document reference",
                        lambda value: changed_reference(value, "document"),
                        ("_document",),
                        "production_recordset_changed",
                    )
                )
                mutation_cases.extend(
                    (
                        f"changed {field} reference",
                        lambda value, field=field: changed_reference(
                            value, field
                        ),
                        (f"_{field}",),
                        "production_recordset_changed",
                    )
                    for field in ("run_ref", "plan_ref")
                )
                mutation_cases.extend(
                    (
                        f"reordered {field} references",
                        lambda value, field=field: reversed_references(
                            value, field
                        ),
                        (f"_{field}",),
                        "production_recordset_changed",
                    )
                    for field in sequence_fields
                    if len(object.__getattribute__(records, f"_{field}")) > 1
                )
                for name, mutate, fields, expected_error in mutation_cases:
                    saved = {
                        field: object.__getattribute__(records, field)
                        for field in fields
                    }
                    with self.subTest(name=name):
                        try:
                            mutate(records)
                            with self.assertRaisesRegex(
                                ValueError, expected_error
                            ):
                                target._recordset_for_production(stage_proofs)
                            with self.assertRaisesRegex(
                                ValueError, expected_error
                            ):
                                finish(target, records, stage_proofs)
                        finally:
                            for field, value in saved.items():
                                object.__setattr__(records, field, value)
                return finish(target, records, stage_proofs)

            with patch.object(
                ReportSession,
                "finish_production",
                new=reject_mutations_then_finish,
            ):
                generation = coordinator.generate_validated(
                    report, session=session
                )

            self.assertIs(type(generation), ValidatedGeneration)
            self.assertIsNone(session._pending_generation_records)
            self.assertIsNone(session._pending_generation_recordset_witness)
            session.close()

    def test_compatibility_generation_keeps_legacy_storage_and_public_validation(
        self,
    ) -> None:
        """Changing traversal must not change the public generation-storage wrapper."""
        cases = [
            ("authored fixture", 2, "diffstory.generation-storage.v1"),
        ]
        for name, unit_count, expected_version in cases:
            with (
                self.subTest(name=name),
                tempfile.TemporaryDirectory() as directory,
            ):
                report = golden_report(unit_count)
                provider = FixtureProvider()
                candidate = Narrator(
                    provider,
                    architecture="v2",
                    recovery_dir=directory,
                    concurrency=1,
                ).generate(report)

                self.assertEqual(
                    candidate["generation"]["version"],
                    "diffstory.generation.v2",
                )
                self.assertEqual(
                    candidate["generation"]["storage_version"],
                    expected_version,
                )
                manifest = persisted_manifest(
                    expanded_generation(candidate["generation"])
                )
                validate_generation(
                    report,
                    manifest,
                    candidate=candidate,
                )

    def test_compatibility_generate_discards_reusable_generation_authority(
        self,
    ) -> None:
        """An expanded compatibility dictionary must not retain a live capability."""
        cases = [("data-only compatibility output", dict, 0)]
        for name, expected_type, expected_registrations in cases:
            with (
                self.subTest(name=name),
                tempfile.TemporaryDirectory() as directory,
            ):
                report = golden_report(2)
                provider = FixtureProvider()
                coordinator = BatchedNarrator(
                    provider,
                    recovery_dir=directory,
                    concurrency=1,
                    options=V2Options(),
                )
                candidate = coordinator.generate(report)
                session = coordinator.session

                self.assertIs(type(candidate), expected_type)
                self.assertIsNotNone(session)
                self.assertEqual(
                    len(session._registrations), expected_registrations
                )
                self.assertFalse(
                    any(
                        kind in {"stage", "generation"}
                        for _handle, kind, _payload in session._registrations.values()
                    )
                )
                session.close()

    def test_warm_and_mixed_runs_bind_actual_reuse_dispositions(self) -> None:
        """Historical results must keep original producers and current sends distinct."""
        cases = [
            ("warm", None, 0, "reused", "reused"),
            (
                "mixed author context",
                "Review Unicode identifiers.",
                1,
                "reused",
                "sent",
            ),
        ]
        for (
            name,
            description,
            minimum_sends,
            expected_detail,
            expected_prose,
        ) in cases:
            with (
                self.subTest(name=name),
                tempfile.TemporaryDirectory() as directory,
            ):
                report = golden_report(2)
                first_provider = FixtureProvider()
                options = V2Options()
                first_session = ReportSession.capture(
                    report,
                    Transport(first_provider, options).descriptor,
                )
                first_coordinator = BatchedNarrator(
                    first_provider,
                    recovery_dir=f"{directory}/state",
                    checkpoint_dir=f"{directory}/cache",
                    concurrency=1,
                    options=options,
                )
                first_coordinator.generate_validated(
                    report, session=first_session
                )
                first_session.close()

                current_report = golden_report(2)
                if description is not None:
                    current_report["meta"]["description"] = description
                provider = FixtureProvider()
                session = ReportSession.capture(
                    current_report,
                    Transport(provider, options).descriptor,
                )
                coordinator = BatchedNarrator(
                    provider,
                    recovery_dir=f"{directory}/state",
                    checkpoint_dir=f"{directory}/cache",
                    concurrency=1,
                    options=options,
                )
                generation = coordinator.generate_validated(
                    current_report, session=session
                )
                proof = session._registered_payload(
                    generation, ValidatedGeneration, "generation"
                )
                receipts = tuple(
                    session._arena.receipt_view(reference)
                    for reference in proof.records.receipts
                )
                detail_dispositions = {
                    receipt["disposition"]
                    for receipt in receipts
                    if receipt["result_kind"] == "detail"
                }
                prose_dispositions = {
                    receipt["disposition"]
                    for receipt in receipts
                    if receipt["result_kind"] != "detail"
                }

                self.assertEqual(detail_dispositions, {expected_detail})
                self.assertEqual(prose_dispositions, {expected_prose})
                self.assertGreaterEqual(len(provider.calls), minimum_sends)
                if name == "warm":
                    self.assertEqual(provider.calls, [])
                session.close()

    def test_stage_proofs_reject_foreign_and_incomplete_authority(
        self,
    ) -> None:
        """Copied, foreign, missing, or reordered proofs must not close production."""
        cases = [
            ("copied capability", "copy", TypeError, "cannot be copied"),
            (
                "missing final proof",
                "missing",
                ValueError,
                "stage_proof_coverage",
            ),
            (
                "duplicate proof",
                "duplicate",
                ValueError,
                "stage_proof_coverage",
            ),
            (
                "reordered proofs",
                "reordered",
                ValueError,
                "stage_proof_coverage",
            ),
            (
                "foreign session",
                "foreign",
                ValueError,
                "stage_proof_coverage",
            ),
        ]
        for name, mutation, error_type, expected_error in cases:
            with (
                self.subTest(name=name),
                tempfile.TemporaryDirectory() as directory,
            ):
                provider = FixtureProvider()
                coordinator, session, report = self._coordinator(
                    provider, directory
                )
                generation = coordinator.generate_validated(
                    report, session=session
                )
                proof = session._registered_payload(
                    generation, ValidatedGeneration, "generation"
                )
                stage_proofs = proof.stage_proofs
                if mutation == "copy":
                    with self.assertRaisesRegex(error_type, expected_error):
                        copy.copy(stage_proofs[0])
                    session.close()
                    continue
                if mutation == "missing":
                    supplied = stage_proofs[:-1]
                    target = session
                elif mutation == "duplicate":
                    supplied = (stage_proofs[0], *stage_proofs[:-1])
                    target = session
                elif mutation == "reordered":
                    supplied = (
                        stage_proofs[1],
                        stage_proofs[0],
                        *stage_proofs[2:],
                    )
                    target = session
                else:
                    supplied = stage_proofs
                    other = ReportSession.capture(
                        report, Transport(provider, V2Options()).descriptor
                    )
                    self.addCleanup(other.close)
                    target = other
                with self.assertRaisesRegex(error_type, expected_error):
                    target._ordered_stage_proofs(supplied)
                session.close()

    def test_rehashed_false_predecessors_do_not_close_production(self) -> None:
        """A coherently rehashed proof must still name the required prior stages."""
        cases = [
            ("document omits required stages", "stage_predecessor_mismatch")
        ]
        for name, expected_error in cases:
            with (
                self.subTest(name=name),
                tempfile.TemporaryDirectory() as directory,
            ):
                provider = FixtureProvider()
                coordinator, session, report = self._coordinator(
                    provider, directory
                )
                generation = coordinator.generate_validated(
                    report, session=session
                )
                proof = session._registered_payload(
                    generation, ValidatedGeneration, "generation"
                )
                document_proof = proof.stage_proofs[-1]
                registration = session._registrations[id(document_proof)]
                evidence = registration[2]
                altered = replace(
                    evidence, predecessors=(proof.stage_proofs[0],)
                )
                altered = replace(
                    altered,
                    witness=_stage_proof_witness(altered),
                )
                session._registrations[id(document_proof)] = (
                    document_proof,
                    "stage",
                    altered,
                )

                with self.assertRaisesRegex(ValueError, expected_error):
                    session._ordered_stage_proofs(proof.stage_proofs)
                session.close()

    def test_closing_session_expires_stage_proofs(self) -> None:
        """A closed session must not keep completed stage capabilities live."""
        cases = [("session close", "session_closed")]
        for name, expected_error in cases:
            with (
                self.subTest(name=name),
                tempfile.TemporaryDirectory() as directory,
            ):
                provider = FixtureProvider()
                coordinator, session, report = self._coordinator(
                    provider, directory
                )
                generation = coordinator.generate_validated(
                    report, session=session
                )
                proof = session._registered_payload(
                    generation, ValidatedGeneration, "generation"
                )
                stage_proof = proof.stage_proofs[0]
                session.close()

                with self.assertRaisesRegex(ValueError, expected_error):
                    session._check_stage_proof(stage_proof)

    def test_stage_proof_pins_both_codec_audit_memos(self) -> None:
        """Mutating a replay memo must not preserve prior audit authority."""
        cases = [
            ("runtime replay memo", "stage"),
            ("completion audit memo", "completion"),
        ]
        with tempfile.TemporaryDirectory() as directory:
            provider = FixtureProvider()
            coordinator, session, report = self._coordinator(
                provider, directory
            )
            generation = coordinator.generate_validated(
                report, session=session
            )
            proof = session._registered_payload(
                generation, ValidatedGeneration, "generation"
            )
            stage_proof = proof.stage_proofs[0]
            evidence = session._check_stage_proof(stage_proof)
            for name, owner in cases:
                with self.subTest(name=name):
                    replay = (
                        evidence.stage.replay
                        if owner == "stage"
                        else evidence.audit_replay
                    )
                    replay._requests.add("caller-mutated-memo")
                    try:
                        with self.assertRaisesRegex(
                            ValueError, "codec_replay_changed"
                        ):
                            session._check_stage_proof(stage_proof)
                    finally:
                        replay._requests.remove("caller-mutated-memo")
                    self.assertIs(
                        session._check_stage_proof(stage_proof), evidence
                    )
            session.close()


if __name__ == "__main__":
    unittest.main()
