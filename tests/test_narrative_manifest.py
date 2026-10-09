"""Adversarial structural manifest checks; trusted report validation is later."""

from __future__ import annotations

import copy
import json
import tempfile
import unittest
from unittest.mock import patch

from pydantic import ValidationError
from test_narrative import two_group_report
from test_narrative_contracts import prose_result
from test_narrative_plan import authored_result
from test_narrative_plan import body
from test_narrative_runtime import changed_record
from test_narrative_runtime import identity
from test_narrative_runtime import invocation_for_results
from test_narrative_runtime import prepared_family

from diffstory.narrative_contracts import ClaimRef
from diffstory.narrative_contracts import SupportedStatement
from diffstory.narrative_contracts import result_key
from diffstory.narrative_contracts import result_kind
from diffstory.narrative_manifest import CurrentUsage
from diffstory.narrative_manifest import GenerationManifest
from diffstory.narrative_manifest import LedgerParameters
from diffstory.narrative_manifest import ProducerDescriptor
from diffstory.narrative_manifest import ResultReceipt
from diffstory.narrative_manifest import compiler_fact_signature
from diffstory.narrative_manifest import result_digest
from diffstory.narrative_manifest import validate_generation
from diffstory.narrative_plan import RequestPackingLimits
from diffstory.narrative_plan import build_evidence_ledger
from diffstory.narrative_plan import build_story_plan
from diffstory.narrative_plan import canonical_bytes
from diffstory.narrative_plan import content_digest
from diffstory.narrative_plan import planning_groups_from_details
from diffstory.narrative_plan import semantic_digest
from diffstory.narrative_runtime import ArtifactInvocation
from diffstory.narrative_runtime import ArtifactRuntimeError
from diffstory.usage import ModelCapacity


def manifest_fixture() -> tuple[dict, GenerationManifest]:
    """Bind all real ledger units and all structural prose to shared producers."""
    report = two_group_report()
    ledger = build_evidence_ledger(report)
    details = tuple(authored_result(unit) for unit in ledger.units)
    plan = build_story_plan(
        planning_groups_from_details(ledger, details),
        ModelCapacity(1_000_000, 100_000),
        RequestPackingLimits(),
        body,
        context_digest=semantic_digest("author-context"),
    )
    planning = tuple(
        prose_result("planning_card", card.id, plan_key=plan.id)
        for card in plan.cards
    )
    sections = tuple(
        prose_result("section", key, plan_key=plan.id)
        for key in plan.ordered_group_keys
    )
    document = prose_result("document", plan.id, plan_key=plan.id)
    run = identity(report_signature=content_digest(canonical_bytes(report)))
    producers = []
    receipts = []
    for results in (details, planning, sections, (document,)):
        invocation = invocation_for_results(results, run)
        # A repaired last item still consumes the original peers, including
        # accepted items absent from its smaller production request.
        requests = [
            prepared_family(
                invocation.batch.item_keys, invocation.batch.stage
            ).metadata
        ]
        if len(results) > 1:
            requests.append(
                prepared_family(
                    (result_key(results[-1]),),
                    invocation.batch.stage,
                    "repair",
                ).metadata
            )
        producer = ProducerDescriptor(
            original_run=run, invocation=invocation, requests=tuple(requests)
        )
        producers.append(producer)
        artifacts = invocation.artifact_keys()
        for position, result in enumerate(results):
            request = (
                requests[-1] if position == len(results) - 1 else requests[0]
            )
            receipts.append(
                ResultReceipt(
                    item_key=result_key(result),
                    result_kind=result_kind(result),
                    result_digest=result_digest(result),
                    producer_key=producer.key,
                    request_key=request.batch.id,
                    artifact_key=artifacts[result_key(result)],
                    disposition="sent",
                )
            )
    from test_narrative_v2 import FixtureProvider

    from diffstory.narrative_v2 import Transport
    from diffstory.narrative_v2 import V2Options

    manifest = GenerationManifest(
        request_configuration=Transport(
            FixtureProvider(), V2Options()
        ).descriptor,
        run=run,
        compiler_fact_signature=compiler_fact_signature(report),
        current_usage=CurrentUsage(
            input_tokens=100,
            cached_input_tokens=0,
            output_tokens=200,
            calls=sum(len(producer.requests) for producer in producers),
            unreported_calls=0,
        ),
        ledger_digest=ledger.id,
        ledger_parameters=LedgerParameters(
            max_span_bytes=4096, max_unit_spans=16
        ),
        unit_keys=tuple(unit.id for unit in ledger.units),
        details=details,
        plan=plan,
        planning=planning,
        sections=sections,
        document=document,
        producers=tuple(producers),
        receipts=tuple(receipts),
    )
    return report, manifest


class ManifestTests(unittest.TestCase):
    """Reject incomplete audit relationships without claiming semantic authority."""

    def setUp(self) -> None:
        """Prepare a complete multi-group manifest with actual detail contracts."""
        self.report, self.manifest = manifest_fixture()

    def test_roundtrip_is_versioned_frozen_and_keeps_source_only_in_report(
        self,
    ) -> None:
        """Shared producers and typed details survive without copying source buffers."""
        manifest = self.manifest
        self.assertEqual(
            GenerationManifest.model_validate_json(manifest.model_dump_json()),
            manifest,
        )
        self.assertEqual(manifest.provenance, "unverified")
        self.assertEqual(len(manifest.producers), 4)
        self.assertGreater(len(manifest.receipts), len(manifest.producers))
        self.assertNotIn("source_evidence", manifest.model_dump())
        self.assertNotIn("spans", manifest.model_dump())
        self.assertNotIn("ledger", manifest.model_dump())
        self.assertEqual(
            manifest.run.report_signature,
            content_digest(canonical_bytes(self.report)),
        )
        with self.assertRaises(ValidationError):
            manifest.current_usage.calls = 1
        for update in (
            {"version": "diffstory.generation.v1"},
            {"guarantee": "trusted"},
            {"provenance": "verified"},
            {"details": list(manifest.details)},
        ):
            with (
                self.subTest(fields=tuple(update)),
                self.assertRaises(ValidationError),
            ):
                GenerationManifest.model_validate(
                    manifest.model_dump() | update
                )

    def test_missing_duplicate_and_inconsistent_coverage_is_rejected(
        self,
    ) -> None:
        """Every declared unit and typed prose result must have one valid receipt."""
        manifest = self.manifest
        first = manifest.receipts[0]
        updates = (
            {"receipts": manifest.receipts[1:]},
            {"receipts": (*manifest.receipts, first)},
            {
                "receipts": (
                    changed_record(first, result_digest="f" * 64),
                    *manifest.receipts[1:],
                )
            },
            {
                "receipts": (
                    changed_record(first, producer_key="f" * 64),
                    *manifest.receipts[1:],
                )
            },
            {
                "receipts": (
                    changed_record(first, request_key="f" * 64),
                    *manifest.receipts[1:],
                )
            },
            {
                "receipts": (
                    changed_record(first, artifact_key="f" * 64),
                    *manifest.receipts[1:],
                )
            },
            {
                "receipts": (
                    changed_record(first, result_kind="section"),
                    *manifest.receipts[1:],
                )
            },
            {"details": manifest.details[1:]},
            {"details": (manifest.details[0],) * len(manifest.details)},
            {"unit_keys": (*manifest.unit_keys, manifest.unit_keys[0])},
            {"unit_keys": manifest.unit_keys[1:]},
            {"planning": manifest.planning[1:]},
            {"sections": tuple(reversed(manifest.sections))},
            {"producers": manifest.producers[1:]},
            {"producers": (*manifest.producers, manifest.producers[0])},
        )
        for update in updates:
            with (
                self.subTest(fields=tuple(update)),
                self.assertRaises(ValueError),
            ):
                changed_record(manifest, **update)

    def test_repair_receipt_retains_complete_original_peer_dependencies(
        self,
    ) -> None:
        """Dropping an original successful peer from a repaired producer fails."""
        producer = self.manifest.producers[0]
        repaired = producer.requests[-1]
        self.assertEqual(len(repaired.batch.item_keys), 1)
        self.assertGreater(len(producer.invocation.batch.item_keys), 1)
        original_peer = producer.invocation.batch.item_keys[0]
        self.assertNotIn(original_peer, repaired.batch.item_keys)
        with self.assertRaises(ArtifactRuntimeError):
            changed_record(
                producer.invocation,
                dependencies=tuple(
                    dependency
                    for dependency in producer.invocation.dependencies
                    if not (
                        dependency.kind == "peer"
                        and dependency.key == original_peer
                    )
                ),
            )
        first = self.manifest.receipts[0]
        wrong_request = changed_record(first, request_key=repaired.batch.id)
        with self.assertRaisesRegex(ValueError, "request_item_mismatch"):
            changed_record(
                self.manifest,
                receipts=(wrong_request, *self.manifest.receipts[1:]),
            )

    def test_changed_prose_requires_receipt_rebinding_but_not_trust(
        self,
    ) -> None:
        """Even internally consistent data still needs trusted current-ledger checks."""
        document = changed_record(
            self.manifest.document,
            closing=(
                SupportedStatement(
                    kind="implementation_claim",
                    text="Unsupported semantics with an existing reference remain unverified.",
                    claims=self.manifest.document.closing[0].claims,
                ),
            ),
        )
        with self.assertRaisesRegex(ValueError, "receipt_result_mismatch"):
            changed_record(self.manifest, document=document)
        receipts = tuple(
            changed_record(receipt, result_digest=result_digest(document))
            if receipt.item_key == document.item_key
            else receipt
            for receipt in self.manifest.receipts
        )
        bound = changed_record(
            self.manifest, document=document, receipts=receipts
        )
        self.assertEqual(bound.provenance, "unverified")

    def test_historical_reuse_does_not_add_current_usage_and_sent_run_is_exact(
        self,
    ) -> None:
        """Exact original producer identities survive reuse across report revisions."""
        current = identity(
            base_revision="base-next",
            head_revision="head-next",
            report_signature="report-next",
        )
        with self.assertRaisesRegex(ValueError, "sent_run_mismatch"):
            changed_record(self.manifest, run=current)
        reused = changed_record(
            self.manifest,
            run=current,
            receipts=tuple(
                changed_record(receipt, disposition="reused")
                for receipt in self.manifest.receipts
            ),
            current_usage=CurrentUsage(
                input_tokens=0,
                cached_input_tokens=0,
                output_tokens=0,
                calls=0,
                unreported_calls=0,
            ),
        )
        self.assertEqual(reused.current_usage.calls, 0)
        self.assertNotEqual(reused.run, reused.producers[0].original_run)
        for change in (
            {"model": "different-model"},
            {"provider_destination": "https://other.example"},
        ):
            with (
                self.subTest(change=change),
                self.assertRaisesRegex(
                    ValueError, "producer_identity_mismatch"
                ),
            ):
                changed_record(reused, run=changed_record(current, **change))
        with self.assertRaisesRegex(ValueError, "sent_usage_mismatch"):
            changed_record(self.manifest, current_usage=reused.current_usage)

    def test_producer_metadata_and_parameters_are_strict(self) -> None:
        """Original static identity, requests, and ledger parameters cannot drift."""
        producer = self.manifest.producers[0]
        for update in (
            {"version": "v1"},
            {"requests": (*producer.requests, producer.requests[0])},
            {"original_run": identity(model="different-model")},
        ):
            with (
                self.subTest(fields=tuple(update)),
                self.assertRaises(ValueError),
            ):
                changed_record(producer, **update)
        for update in (
            {"max_span_bytes": 3},
            {"max_unit_spans": 1},
            {"max_unit_spans": True},
            {"max_span_bytes": "4096"},
        ):
            with (
                self.subTest(fields=tuple(update)),
                self.assertRaises(ValidationError),
            ):
                changed_record(self.manifest.ledger_parameters, **update)
        request = producer.requests[0]
        with self.assertRaisesRegex(ValueError, "production_output_guard"):
            changed_record(
                producer,
                requests=(changed_record(request, max_output_tokens=1000),),
            )
        same_producer = changed_record(
            producer, requests=tuple(reversed(producer.requests))
        )
        with self.assertRaisesRegex(ValueError, "duplicate_original_producer"):
            changed_record(
                self.manifest,
                producers=(*self.manifest.producers, same_producer),
            )

    def test_manifest_hashes_shared_invocations_once_per_producer(
        self,
    ) -> None:
        """Receipt checks do not rehash a complete original batch for each item."""
        original = ArtifactInvocation.artifact_keys
        calls = []

        def tracked(invocation: ArtifactInvocation) -> dict[str, str]:
            calls.append(invocation.batch.id)
            return original(invocation)

        with patch.object(ArtifactInvocation, "artifact_keys", tracked):
            GenerationManifest.model_validate(self.manifest)
        self.assertEqual(len(calls), len(self.manifest.producers))


class CompilerFactSignatureTests(unittest.TestCase):
    """Bind original compiler facts after generated presentation is applied."""

    def test_only_declared_presentation_changes_preserve_signature(
        self,
    ) -> None:
        """Reordering and annotating groups does not invalidate compiler identity."""
        report = two_group_report()
        original = copy.deepcopy(report)
        signature = compiler_fact_signature(report)
        report["document"] = {
            "lead": "Generated opening",
            "closing": "Generated close",
        }
        report["generation"] = {"version": "diffstory.generation.v2"}
        report["groups"].reverse()
        for position, group in enumerate(report["groups"]):
            group.update(
                title="Authored heading",
                narrative={"passages": []},
                number=position + 1,
                next_id=None,
            )
        self.assertEqual(compiler_fact_signature(report), signature)
        self.assertNotEqual(
            content_digest(canonical_bytes(report)),
            content_digest(canonical_bytes(original)),
        )

    def test_compiler_and_source_changes_alter_signature(self) -> None:
        """Unknown fields are retained so newly introduced compiler facts stay bound."""
        report = two_group_report()
        signature = compiler_fact_signature(report)
        mutations = (
            lambda value: value["stats"].update(test_runs=99),
            lambda value: value["meta"].update(
                head_revision="changed-revision"
            ),
            lambda value: value["source_evidence"].update(
                new_compiler_fact="changed-source"
            ),
            lambda value: value["source_evidence"]["fragments"][0].update(
                text="different original source"
            ),
            lambda value: value["groups"][0].update(
                prerequisites=["changed-prerequisite"]
            ),
            lambda value: value["changes"][0].update(kind="removed"),
            lambda value: value["groups"][0].update(
                new_compiler_fact="must remain bound"
            ),
        )
        for position, mutate in enumerate(mutations):
            altered = copy.deepcopy(report)
            mutate(altered)
            with self.subTest(position=position):
                self.assertNotEqual(
                    compiler_fact_signature(altered), signature
                )
        self.assertEqual(compiler_fact_signature(report), signature)


class ProductionManifestTests(unittest.TestCase):
    """Use independently compiled current facts, not manifest self-consistency."""

    def setUp(self) -> None:
        """Generate one complete real fixture candidate through the public gate."""
        from test_narrative_v2 import FixtureProvider
        from test_narrative_v2 import golden_report

        from diffstory.narrative import Narrator

        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.report = golden_report()
        self.provider = FixtureProvider()
        self.candidate = Narrator(
            self.provider,
            architecture="v2",
            recovery_dir=directory.name,
            concurrency=1,
        ).generate(self.report)
        self.manifest = GenerationManifest.model_validate_json(
            json.dumps(self.candidate["generation"])
        )

    def test_self_consistent_unknown_claim_fails_current_support_validation(
        self,
    ) -> None:
        """Rehashing a result receipt cannot create a current detail observation."""
        manifest = self.manifest
        original = manifest.sections[0]
        unsupported = changed_record(
            original.synthesis[0],
            claims=(
                ClaimRef(
                    unit_key=manifest.details[0].unit_key,
                    observation_id="0" * 64,
                ),
            ),
        )
        section = changed_record(original, synthesis=(unsupported,))
        receipts = tuple(
            changed_record(r, result_digest=result_digest(section))
            if r.item_key == section.item_key
            else r
            for r in manifest.receipts
        )
        altered = changed_record(
            manifest,
            sections=(section, *manifest.sections[1:]),
            receipts=receipts,
        )
        self.assertIsInstance(altered, GenerationManifest)
        with self.assertRaisesRegex(ArtifactRuntimeError, "citation"):
            validate_generation(self.report, altered, provider=self.provider)

    def test_self_consistent_request_metadata_fails_actual_byte_audit(
        self,
    ) -> None:
        """Structural descriptor hashes do not establish actual prepared bounds."""
        manifest = self.manifest
        old = manifest.producers[0]
        first = old.requests[0]
        batch = changed_record(
            first.batch, input_bound=first.batch.input_bound + 1
        )
        producer = changed_record(
            old,
            requests=(changed_record(first, batch=batch), *old.requests[1:]),
        )
        receipts = tuple(
            changed_record(r, producer_key=producer.key)
            if r.producer_key == old.key
            else r
            for r in manifest.receipts
        )
        altered = changed_record(
            manifest,
            producers=(producer, *manifest.producers[1:]),
            receipts=receipts,
        )
        with self.assertRaisesRegex(ArtifactRuntimeError, "request_mismatch"):
            validate_generation(self.report, altered, provider=self.provider)

    def test_trusted_compiler_facts_and_exact_questions_remain_bound(
        self,
    ) -> None:
        """Reject a source-fact edit or an assembly that loses unresolved questions."""
        report = copy.deepcopy(self.report)
        report["changes"][0]["basis"] += " changed"
        with self.assertRaisesRegex(
            ArtifactRuntimeError, "dependencies_changed"
        ):
            validate_generation(report, self.manifest, provider=self.provider)
        candidate = copy.deepcopy(self.candidate)
        detail = next(
            d
            for step in candidate["steps"]
            for d in step["details"]
            if d["questions"]
        )
        detail["questions"] = []
        with self.assertRaisesRegex(ArtifactRuntimeError, "coverage"):
            validate_generation(
                self.report,
                self.manifest,
                provider=self.provider,
                candidate=candidate,
            )

    def test_missing_union_audit_and_unconsumed_whole_source_fail_public_validation(
        self,
    ) -> None:
        """An empty final audit or available-but-unconsumed source cannot pass."""
        manifest = self.manifest
        empty = changed_record(
            manifest, plan=changed_record(manifest.plan, requests=())
        )
        with self.assertRaisesRegex(
            ValueError, "planning_request_membership_mismatch"
        ):
            validate_generation(self.report, empty, provider=self.provider)
        original = manifest.sections[0]
        direct = manifest.details[0].explanations[0].citations[0]
        statement = changed_record(
            original.synthesis[0], claims=(), basis=(), citations=(direct,)
        )
        section = changed_record(original, synthesis=(statement,))
        altered = changed_record(
            manifest,
            sections=(section, *manifest.sections[1:]),
            receipts=tuple(
                changed_record(r, result_digest=result_digest(section))
                if r.item_key == section.item_key
                else r
                for r in manifest.receipts
            ),
        )
        with self.assertRaisesRegex(ArtifactRuntimeError, "citation"):
            validate_generation(self.report, altered, provider=self.provider)

    def test_rehashed_stale_statement_endpoints_fail_public_validation(
        self,
    ) -> None:
        """Rehashed receipts cannot authorize a changed producer, path or digest."""
        from diffstory.narrative_contracts import StatementPath

        original = self.manifest.sections[0]
        ref = original.synthesis[0].basis[0]
        for update in (
            {"producer_item_key": "0" * 64},
            {"result_digest": "0" * 64},
            {"statement_digest": "0" * 64},
            {"statement_path": StatementPath(field="abstraction", index=999)},
        ):
            with self.subTest(fields=tuple(update)):
                statement = changed_record(
                    original.synthesis[0],
                    basis=(changed_record(ref, **update),),
                )
                section = changed_record(original, synthesis=(statement,))
                altered = changed_record(
                    self.manifest,
                    sections=(section, *self.manifest.sections[1:]),
                    receipts=tuple(
                        changed_record(r, result_digest=result_digest(section))
                        if r.item_key == section.item_key
                        else r
                        for r in self.manifest.receipts
                    ),
                )
                with self.assertRaisesRegex(ArtifactRuntimeError, "citation"):
                    validate_generation(
                        self.report, altered, provider=self.provider
                    )

    def test_rederived_context_and_final_transition_reject_self_declarations(
        self,
    ) -> None:
        """Persisted relationship facts and a valid receipt cannot replace current facts."""
        hierarchy = self.manifest.plan.production
        view = hierarchy.section_views[0]
        changed = changed_record(view, chapter_key="0" * 64)
        altered = changed_record(
            self.manifest,
            plan=changed_record(
                self.manifest.plan,
                production=changed_record(
                    hierarchy,
                    section_views=(changed, *hierarchy.section_views[1:]),
                ),
            ),
        )
        with self.assertRaisesRegex(
            ArtifactRuntimeError, "dependencies_changed"
        ):
            validate_generation(self.report, altered, provider=self.provider)
        final = self.manifest.sections[-1]
        changed = changed_record(
            final,
            transition=(
                SupportedStatement(
                    kind="editorial", text="Invented next section."
                ),
            ),
        )
        altered = changed_record(
            self.manifest,
            sections=(*self.manifest.sections[:-1], changed),
            receipts=tuple(
                changed_record(r, result_digest=result_digest(changed))
                if r.item_key == changed.item_key
                else r
                for r in self.manifest.receipts
            ),
        )
        with self.assertRaisesRegex(ArtifactRuntimeError, "coverage"):
            validate_generation(self.report, altered, provider=self.provider)
