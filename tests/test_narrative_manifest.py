"""Adversarial structural manifest checks; trusted report validation is later."""

from __future__ import annotations

import copy
import json
import sys
import tempfile
import unittest
from collections import Counter
from typing import TYPE_CHECKING
from unittest.mock import patch

from pydantic import ValidationError

if TYPE_CHECKING:
    from collections.abc import Callable
    from collections.abc import Mapping
    from types import FrameType

from test_narrative import two_group_report
from test_narrative_contracts import prose_result
from test_narrative_plan import authored_result
from test_narrative_plan import body
from test_narrative_runtime import changed_record
from test_narrative_runtime import identity
from test_narrative_runtime import invocation_for_results
from test_narrative_runtime import prepared_family

from diffstory.narrative_contracts import AuthorContext
from diffstory.narrative_contracts import ClaimRef
from diffstory.narrative_contracts import EvidenceUnit
from diffstory.narrative_contracts import ProseBinding
from diffstory.narrative_contracts import SourceSliceRef
from diffstory.narrative_contracts import SpanRef
from diffstory.narrative_contracts import StatementPath
from diffstory.narrative_contracts import SupportedStatement
from diffstory.narrative_contracts import SupportedTerminology
from diffstory.narrative_contracts import result_key
from diffstory.narrative_contracts import result_kind
from diffstory.narrative_hierarchy import derive_planning_sources_indexed
from diffstory.narrative_hierarchy import span_reference
from diffstory.narrative_hierarchy import statement_reference
from diffstory.narrative_manifest import CurrentUsage
from diffstory.narrative_manifest import GenerationManifest
from diffstory.narrative_manifest import LedgerParameters
from diffstory.narrative_manifest import ProducerDescriptor
from diffstory.narrative_manifest import ResultReceipt
from diffstory.narrative_manifest import _document_view
from diffstory.narrative_manifest import _group_view
from diffstory.narrative_manifest import assemble_candidate
from diffstory.narrative_manifest import assemble_indexed_candidate
from diffstory.narrative_manifest import assemble_indexed_reader
from diffstory.narrative_manifest import candidate_storage_semantics
from diffstory.narrative_manifest import compiler_fact_signature
from diffstory.narrative_manifest import expanded_generation
from diffstory.narrative_manifest import indexed_applied_fields
from diffstory.narrative_manifest import persisted_manifest
from diffstory.narrative_manifest import presentation_from_candidate
from diffstory.narrative_manifest import result_digest
from diffstory.narrative_manifest import stored_generation
from diffstory.narrative_manifest import validate_generation
from diffstory.narrative_plan import RequestPackingLimits
from diffstory.narrative_plan import SourceSpan
from diffstory.narrative_plan import _capture_indexed_arena_detail_results
from diffstory.narrative_plan import build_evidence_ledger
from diffstory.narrative_plan import build_ledger_index
from diffstory.narrative_plan import build_story_plan
from diffstory.narrative_plan import canonical_bytes
from diffstory.narrative_plan import content_digest
from diffstory.narrative_plan import planning_groups_from_details
from diffstory.narrative_plan import semantic_digest
from diffstory.narrative_runtime import ArtifactInvocation
from diffstory.narrative_runtime import ArtifactRuntimeError
from diffstory.report_storage import GenerationRecordSet
from diffstory.report_storage import RecordRef
from diffstory.report_storage import ResultRef
from diffstory.report_storage import _RecordArena
from diffstory.report_storage import expand_report_storage
from diffstory.usage import ModelCapacity


def _span_validator_profiler(
    counts: Counter[str],
) -> Callable[[FrameType, str, object], None]:
    """Count real nested span model validators during one scoped operation.

    Preconditions:
        ``counts`` is the mutable counter for one measured call.
    Postconditions:
        Returns a profiler that records each executed ``SpanRef.ranges`` call.
    Essential behavior:
        Measures nested Pydantic validator execution separately from storage
        table rows and mocked top-level model dispatch.
    """

    def profile(frame: FrameType, event: str, _arg: object) -> None:
        if (
            event == "call"
            and frame.f_code.co_name == "ranges"
            and isinstance(frame.f_locals.get("self"), SpanRef)
        ):
            counts["span_ref_ranges"] += 1

    return profile


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


def saved_generation_metrics(manifest: GenerationManifest) -> dict:
    """Return source-free exact byte counts and typed roundtrip evidence."""
    expanded = manifest.model_dump(mode="json")
    compact = stored_generation(manifest)
    restored = persisted_manifest(compact)
    before = len(canonical_bytes(expanded))
    after = len(canonical_bytes(compact))
    return {
        "expanded_manifest_bytes": before,
        "compact_manifest_bytes": after,
        "saved_bytes": before - after,
        "expanded_invocation_bytes": sum(
            len(canonical_bytes(p["invocation"]))
            for p in expanded["producers"]
        ),
        "compact_invocation_bytes": sum(
            len(canonical_bytes(p["invocation"])) for p in compact["producers"]
        ),
        "producer_count": len(manifest.producers),
        "lookup_count": sum(
            len(p.invocation.lookups) for p in manifest.producers
        ),
        "expanded_equality": restored == manifest,
        "producer_keys_equal": tuple(p.key for p in restored.producers)
        == tuple(p.key for p in manifest.producers),
        "results_and_receipts_equal": all(
            getattr(restored, field) == getattr(manifest, field)
            for field in (
                "details",
                "planning",
                "sections",
                "document",
                "receipts",
            )
        ),
        "legacy_read_equal": persisted_manifest(expanded) == manifest,
    }


def indexed_assembly_inputs(
    report: dict,
    manifest: GenerationManifest,
    *,
    author: AuthorContext | None = None,
) -> tuple[object, object, _RecordArena, GenerationRecordSet, object, object]:
    """Capture authored generation rows and bind indexed detail/source views.

    Preconditions:
        ``report`` and ``manifest`` form one deterministic local fixture.
    Postconditions:
        Every result and audit row is in ``arena`` before assembly begins;
        the returned detail and source selectors share that arena and ledger
        and carry the report's exact ``(base_sha, head_sha)`` binding.
    Essential behavior:
        Exercises strict initial ownership once, then leaves normal assembly to
        read only selected records and captured scalar facts.
    """
    from diffstory.narrative_plan import build_evidence_ledger

    ledger = build_evidence_ledger(
        report, **manifest.ledger_parameters.model_dump()
    )
    index = build_ledger_index(ledger)
    revision_binding = (
        report["meta"]["base_sha"],
        report["meta"]["head_sha"],
    )
    arena = _RecordArena()
    detail_refs = tuple(arena.put_result(item) for item in manifest.details)
    details = _capture_indexed_arena_detail_results(
        detail_refs,
        arena,
        ledger,
        index=index,
        revision_binding=revision_binding,
    )
    plan_ref = arena.put_record("story_plan", manifest.plan)
    planning_refs = tuple(arena.put_result(item) for item in manifest.planning)
    section_refs = tuple(arena.put_result(item) for item in manifest.sections)
    document_ref = arena.put_result(manifest.document)
    run_ref = arena.put_record("run", manifest.run)
    producer_refs = tuple(
        arena.put_record("producer", item) for item in manifest.producers
    )
    receipt_refs = tuple(
        arena.put_record("receipt", item) for item in manifest.receipts
    )
    generation_data = manifest.model_dump(mode="json")
    header_fields = (
        "version",
        "provenance",
        "compiler_fact_signature",
        "current_usage",
        "ledger_digest",
        "ledger_parameters",
        "request_configuration",
        "unit_keys",
    )
    generation = GenerationRecordSet(
        {key: generation_data[key] for key in header_fields},
        run_ref=run_ref,
        details=detail_refs,
        plan_ref=plan_ref,
        planning=planning_refs,
        sections=section_refs,
        document=document_ref,
        producers=producer_refs,
        receipts=receipt_refs,
    )
    sources = derive_planning_sources_indexed(
        index,
        details,
        author_context=author if author is not None else AuthorContext(),
        revision_binding=revision_binding,
    )
    return ledger, index, arena, generation, details, sources


def indexed_generation_with(
    generation: GenerationRecordSet, **changes: object
) -> GenerationRecordSet:
    """Rebuild one typed indexed set with selected relationship changes.

    Preconditions:
        ``generation`` is a captured set and ``changes`` names constructor
        fields whose replacement values have the declared reference types.
    Postconditions:
        Returns a fresh data-only set with unchanged fields preserved.
    Essential behavior:
        Builds realistic adversarial relationship inputs through the same
        strict constructor used by valid indexed assemblies.
    """
    values: dict[str, object] = {
        "header": generation.header,
        "run_ref": generation.run_ref,
        "details": generation.details,
        "plan_ref": generation.plan_ref,
        "planning": generation.planning,
        "sections": generation.sections,
        "document": generation.document,
        "producers": generation.producers,
        "receipts": generation.receipts,
    }
    values.update(changes)
    return GenerationRecordSet(**values)


def indexed_support_manifest(
    report: dict,
    manifest: GenerationManifest,
    *,
    slice_error: str | None = None,
    statement_digest_error: bool = False,
    inherited_endpoint_error: bool = False,
) -> tuple[GenerationManifest, AuthorContext]:
    """Add an authored slice and two immediate inherited statement links.

    Preconditions:
        ``manifest`` has the local two-group source fixture and stable result
        item keys needed for reference construction.
    Postconditions:
        Returns a strict manifest whose receipts match the changed result rows;
        optional digest, endpoint, and slice errors remain structurally valid
        but fail current assembly.
    Essential behavior:
        Covers source slices, exact paths/digests, authorship, uncertainty,
        terminology, and a two-edge inherited statement chain.
    """
    ledger = build_evidence_ledger(
        report, **manifest.ledger_parameters.model_dump()
    )
    unit = ledger.units[0]
    span = unit.spans[0]
    source = span_reference(span, unit)
    source_bytes = span.text.encode("utf-8")
    slice_end = len(source_bytes) + (1 if slice_error == "coordinate" else 0)
    slice_digest = (
        "f" * 64 if slice_error == "digest" else content_digest(source_bytes)
    )
    source_slice = SourceSliceRef(
        source=source,
        byte_start=0,
        byte_end=slice_end,
        content_digest=slice_digest,
    )
    plan_results = list(manifest.planning)
    first_plan = plan_results[0]
    terminology = SupportedTerminology(
        term="casefold",
        definition=SupportedStatement(
            kind="implementation_claim",
            text="The selected term keeps Unicode case folding explicit.",
            effect="The terminology retains the generated definition.",
            basis=(source_slice,),
        ),
    )
    plan_results[0] = changed_record(first_plan, terminology=(terminology,))
    plan_reference = statement_reference(
        plan_results[0], StatementPath(field="abstraction", index=0)
    )
    if statement_digest_error:
        plan_reference = changed_record(
            plan_reference, statement_digest="f" * 64
        )
    elif inherited_endpoint_error:
        plan_reference = changed_record(
            plan_reference, producer_item_key="0" * 64
        )
    section_results = list(manifest.sections)
    first_section = section_results[0]
    inherited_claim = SupportedStatement(
        kind="implementation_claim",
        text="The section uses the selected source and planning statement.",
        effect="The inherited source remains linked to its original producer.",
        uncertainty=("Caller behavior remains unverified.",),
        basis=(source_slice, plan_reference),
    )
    section_results[0] = changed_record(
        first_section, synthesis=(inherited_claim,)
    )
    section_reference = statement_reference(
        section_results[0], StatementPath(field="synthesis", index=0)
    )
    author = AuthorContext(
        title="Fixture author",
        repository="example/project",
        author_reported_motivation="Keep the behavior visible.",
    )
    author_digest = semantic_digest("diffstory.author-context.v2", author)
    document_claim = SupportedStatement(
        kind="implementation_claim",
        text="The document preserves the inherited section reference.",
        effect="The second provenance edge remains explicit.",
        basis=(section_reference,),
    )
    motivation = SupportedStatement(
        kind="author_motivation",
        text="The author wants the behavior to remain visible.",
        author_context_digest=author_digest,
    )
    uncertainty = SupportedStatement(
        kind="uncertainty",
        text="The fixture has no executed runtime tests.",
        uncertainty=("Runtime behavior was not measured.",),
    )
    document = changed_record(
        manifest.document,
        preamble=(document_claim,),
        lead=(motivation,),
        closing=(uncertainty,),
    )
    changed_results = {
        item.item_key: item
        for item in (*plan_results, *section_results, document)
    }
    receipts = tuple(
        changed_record(
            receipt,
            result_digest=result_digest(changed_results[receipt.item_key]),
        )
        if receipt.item_key in changed_results
        else receipt
        for receipt in manifest.receipts
    )
    updated = changed_record(
        manifest,
        planning=tuple(plan_results),
        sections=tuple(section_results),
        document=document,
        receipts=receipts,
    )
    return updated, author


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

    def test_compact_generation_is_lossless_and_measures_static_savings(
        self,
    ) -> None:
        """All typed proofs survive; only repeated static dependency bytes shrink."""
        manifest = self.manifest
        expanded = manifest.model_dump(mode="json")
        compact = stored_generation(manifest)
        self.assertEqual(persisted_manifest(compact), manifest)
        self.assertEqual(persisted_manifest(expanded), manifest)
        self.assertEqual(expanded_generation(compact), expanded)
        self.assertEqual(compact["version"], expanded["version"])
        self.assertEqual(
            compact["request_configuration"], expanded["request_configuration"]
        )
        self.assertLess(
            len(canonical_bytes(compact)), len(canonical_bytes(expanded))
        )
        for before, after in zip(
            manifest.producers,
            persisted_manifest(compact).producers,
            strict=True,
        ):
            self.assertEqual(before.key, after.key)
            self.assertEqual(
                before.invocation.artifact_keys(),
                after.invocation.artifact_keys(),
            )
            self.assertEqual(before.original_run, after.original_run)
            self.assertEqual(before.requests, after.requests)
        self.assertEqual(
            canonical_bytes(compact).count(b'"static_dependencies"'),
            len(manifest.producers),
        )
        metrics = saved_generation_metrics(manifest)
        self.assertGreater(metrics["saved_bytes"], 0)
        self.assertTrue(metrics["expanded_equality"])
        self.assertTrue(metrics["producer_keys_equal"])
        self.assertTrue(metrics["results_and_receipts_equal"])

    def test_compact_storage_dispatch_rejects_malformed_and_lossy_shapes(
        self,
    ) -> None:
        """Versions, kinds, peer closure, order and exact scalar types stay strict."""
        mutations = (
            lambda v: v.update(storage_version="unknown"),
            lambda v: v.update(storage_version=1),
            lambda v: v.pop("storage_version"),
            lambda v: v.update(extra="unknown"),
            lambda v: v.update(producers={}),
            lambda v: v["producers"][0].update(extra="unknown"),
            lambda v: v["producers"][0]["invocation"].pop("version"),
            lambda v: v["producers"][0]["invocation"].update(
                version="diffstory.request-storage.v1"
            ),
            lambda v: v["producers"][0]["invocation"].update(extra=[]),
            lambda v: v["producers"][0]["invocation"].pop(
                "static_dependencies"
            ),
            lambda v: v["producers"][0]["invocation"][
                "static_dependencies"
            ].pop(),
            lambda v: v["producers"][0]["invocation"][
                "static_dependencies"
            ].reverse(),
            lambda v: v["producers"][0]["invocation"][
                "dynamic_dependencies"
            ].reverse(),
            lambda v: v["producers"][0]["invocation"][
                "dynamic_dependencies"
            ].pop(),
            lambda v: v["producers"][0]["invocation"].update(
                dynamic_dependencies=[
                    d
                    for d in v["producers"][0]["invocation"][
                        "dynamic_dependencies"
                    ]
                    if not (
                        d["kind"] == "peer"
                        and d["key"]
                        == self.manifest.producers[
                            0
                        ].invocation.batch.item_keys[0]
                    )
                ]
            ),
            lambda v: v["producers"][0]["invocation"]["lookups"].reverse(),
            lambda v: v["producers"][0]["invocation"]["lookups"][0].update(
                result_kind="section"
            ),
            lambda v: v["producers"][0]["invocation"]["batch"].update(
                partition=True
            ),
            lambda v: v["producers"][0]["requests"][0].update(
                safety_tokens=True
            ),
        )
        for position, mutate in enumerate(mutations):
            value = stored_generation(self.manifest)
            mutate(value)
            with (
                self.subTest(position=position),
                self.assertRaises((ValueError, ArtifactRuntimeError)),
            ):
                persisted_manifest(value)

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


class IndexedAssemblyTests(unittest.TestCase):
    """Keep indexed reader and application fields equal to expanded oracles."""

    def setUp(self) -> None:
        """Prepare one multi-group report with direct and inherited support."""
        self.report, base_manifest = manifest_fixture()
        self.manifest, self.author = indexed_support_manifest(
            self.report, base_manifest
        )

    def test_indexed_candidate_matches_expanded_reader_and_application(
        self,
    ) -> None:
        """The indexed path retains exact output without full record expansion.

        Input/expected case matrix:
            annotation reader: expected is ``assemble_candidate`` over the
            captured current ledger and strict detail oracle.
            document overlay: expected is ``_document_view`` over the explicit
            ``V2Presentation`` oracle.
            group overlays: expected is ``_group_view`` for each presentation
            section and current compiler group, in section order.
        """
        from diffstory.narrative_manifest import PresentationSection
        from diffstory.narrative_manifest import V2Presentation
        from diffstory.report_storage import ReportRecordGraph
        from diffstory.report_storage import _ArenaRecordReader

        ledger, index, arena, generation, details, sources = (
            indexed_assembly_inputs(
                self.report, self.manifest, author=self.author
            )
        )
        expected_candidate = assemble_candidate(
            self.report, self.manifest, ledger=ledger, index=index
        )
        expected_presentation = presentation_from_candidate(expected_candidate)
        groups_by_id = {group["id"]: group for group in self.report["groups"]}
        expected_group_overlays = tuple(
            {
                "group_id": section.group_id,
                **_group_view(section, groups_by_id[section.group_id]),
            }
            for section in expected_presentation.sections
        )
        previous_resolution_count = details.citation_scope.resolution_count
        with (
            patch(
                "diffstory.narrative_manifest.assemble_candidate",
                side_effect=AssertionError("indexed path rebuilt candidate"),
            ),
            patch(
                "diffstory.narrative_manifest.presentation_from_candidate",
                side_effect=AssertionError("indexed path built presentation"),
            ),
            patch(
                "diffstory.narrative_manifest.stored_generation",
                side_effect=AssertionError("indexed path expanded generation"),
            ),
            patch(
                "diffstory.narrative_manifest.expanded_generation",
                side_effect=AssertionError("indexed path expanded generation"),
            ),
            patch(
                "diffstory.report_storage.expand_report_storage",
                side_effect=AssertionError("indexed path expanded report"),
            ),
            patch.object(
                GenerationManifest,
                "model_validate",
                side_effect=AssertionError("indexed path parsed manifest"),
            ),
            patch.object(
                GenerationManifest,
                "model_validate_json",
                side_effect=AssertionError("indexed path parsed manifest"),
            ),
            patch.object(
                GenerationManifest,
                "__init__",
                side_effect=AssertionError("indexed path built manifest"),
            ),
            patch.object(
                V2Presentation,
                "model_validate",
                side_effect=AssertionError("indexed path parsed presentation"),
            ),
            patch.object(
                V2Presentation,
                "model_validate_json",
                side_effect=AssertionError("indexed path parsed presentation"),
            ),
            patch.object(
                V2Presentation,
                "__init__",
                side_effect=AssertionError("indexed path built presentation"),
            ),
            patch.object(
                PresentationSection,
                "model_validate",
                side_effect=AssertionError("indexed path parsed sections"),
            ),
            patch.object(
                PresentationSection,
                "model_validate_json",
                side_effect=AssertionError("indexed path parsed sections"),
            ),
            patch.object(
                PresentationSection,
                "__init__",
                side_effect=AssertionError("indexed path built sections"),
            ),
            patch.object(
                SupportedStatement,
                "model_validate",
                side_effect=AssertionError("indexed path reparsed statements"),
            ),
            patch.object(
                _RecordArena,
                "result_view",
                side_effect=AssertionError("indexed path expanded a result"),
            ),
            patch.object(
                _ArenaRecordReader,
                "result_view",
                side_effect=AssertionError("indexed path expanded a result"),
            ),
            patch.object(
                _ArenaRecordReader,
                "result_record",
                side_effect=AssertionError("indexed path expanded a result"),
            ),
        ):
            graph = assemble_indexed_candidate(
                self.report,
                generation,
                arena=arena,
                details=details,
                sources=sources,
            )
            overlay = indexed_applied_fields(
                graph.reader_projection,
                arena=arena,
                details=details,
                sources=sources,
                groups=groups_by_id,
            )

        self.assertIsInstance(graph, ReportRecordGraph)
        self.assertEqual(
            details.citation_scope.resolution_count,
            previous_resolution_count,
        )
        expanded = expand_report_storage(
            graph, max_output_bytes=128 * 1024 * 1024
        )
        cases = (
            ("annotation reader", expected_candidate, expanded),
            (
                "document overlay",
                _document_view(expected_presentation),
                overlay["document"],
            ),
            (
                "ordered group overlays",
                expected_group_overlays,
                overlay["groups"],
            ),
        )
        for label, expected, actual in cases:
            expected_value = (
                candidate_storage_semantics(expected)
                if label == "annotation reader"
                else expected
            )
            with self.subTest(case=label):
                self.assertEqual(
                    canonical_bytes(actual), canonical_bytes(expected_value)
                )

        self.assertEqual(len(expanded["support_provenance"]), 2)
        self.assertGreaterEqual(len(expanded["citations"]), 2)
        self.assertEqual(
            arena.retained_record_counts["inherited_provenance"], 2
        )
        for entry in arena._builder.records["inherited_provenance"].values():  # noqa: SLF001 # Confirm provenance stores addresses, not copied content.
            self.assertEqual(
                set(entry["record"]),
                {"key", "reference", "result", "consumption", "receipt"},
            )

    def test_indexed_assembly_counts_current_lookups_and_nested_validation(  # noqa: PLR0915 # Keep independently measured counters explicit.
        self,
    ) -> None:
        """Indexed work uses unique current bindings and reports real counters.

        Input/expected case matrix:
            ordered generation results: one selected receipt row read and
            result digest read per result.
            inherited endpoints: one additional direct receipt-index lookup per
            retained provenance reference, with no receipt scan.
            repeated source references: one checked selection per distinct
            reference/unit pair and one resolver call per new current binding.
            nested source spans: actual ``SpanRef.ranges`` calls are counted
            separately from arena row validation counters.
            owned results and plans: indexed work adds no result or plan rows;
            inherited provenance adds two reference-only rows.
        """
        import diffstory.narrative_plan as plan_module
        from diffstory.narrative_plan import CitationResolutionScope

        _ledger, _index, arena, generation, details, sources = (
            indexed_assembly_inputs(
                self.report, self.manifest, author=self.author
            )
        )
        scope = details.citation_scope
        prior_resolution_count = scope.resolution_count
        prior_memo_count = scope.memoized_reference_count
        prior_validation_counts = arena.typed_validation_counts
        prior_retained_counts = arena.retained_record_counts
        result_refs = (
            *generation.details,
            *generation.planning,
            *generation.sections,
            generation.document,
        )
        receipt_lookups: list[str] = []
        receipt_reads: list[str] = []
        result_digest_reads: list[str] = []
        resolution_requests: list[tuple[str, str]] = []
        span_validator_calls: Counter[str] = Counter()

        original_receipt_lookup = _RecordArena.receipt_reference_for_item
        original_receipt_view = _RecordArena.receipt_view
        original_result_digests = _RecordArena.result_digests
        original_resolution = (
            CitationResolutionScope._resolve_validated  # noqa: SLF001 # Count actual selected resolver calls.
        )

        def tracked_receipt_lookup(
            owner: _RecordArena, item_key: str
        ) -> RecordRef:
            receipt_lookups.append(item_key)
            return original_receipt_lookup(owner, item_key)

        def tracked_receipt_view(
            owner: _RecordArena, reference: RecordRef
        ) -> Mapping[str, object]:
            receipt_reads.append(reference.storage_id)
            return original_receipt_view(owner, reference)

        def tracked_result_digests(
            owner: _RecordArena, reference: ResultRef
        ) -> tuple[str, str]:
            result_digest_reads.append(reference.item_key)
            return original_result_digests(owner, reference)

        def tracked_resolution(
            scope_value: CitationResolutionScope,
            reference: SpanRef,
            unit: EvidenceUnit,
        ) -> SourceSpan:
            resolution_requests.append((semantic_digest(reference), unit.id))
            return original_resolution(scope_value, reference, unit)

        prior_profiler = sys.getprofile()
        sys.setprofile(_span_validator_profiler(span_validator_calls))
        try:
            with (
                patch.object(
                    _RecordArena,
                    "receipt_reference_for_item",
                    new=tracked_receipt_lookup,
                ),
                patch.object(
                    _RecordArena, "receipt_view", new=tracked_receipt_view
                ),
                patch.object(
                    _RecordArena, "result_digests", new=tracked_result_digests
                ),
                patch.object(
                    CitationResolutionScope,
                    "_resolve_validated",
                    new=tracked_resolution,
                ),
                patch.object(
                    plan_module,
                    "_resolve_span_ref",
                    wraps=plan_module._resolve_span_ref,  # noqa: SLF001 # Count calls at the existing current resolver.
                ) as actual_source_resolutions,
            ):
                projection = assemble_indexed_reader(
                    self.report,
                    generation,
                    arena=arena,
                    details=details,
                    sources=sources,
                )
        finally:
            sys.setprofile(prior_profiler)

        expected_result_keys = [
            reference.item_key for reference in result_refs
        ]
        self.assertEqual(
            receipt_lookups[: len(expected_result_keys)], expected_result_keys
        )
        self.assertEqual(
            len(receipt_lookups),
            len(expected_result_keys) + len(projection.support_provenance),
        )
        self.assertEqual(len(receipt_reads), len(result_refs))
        self.assertEqual(result_digest_reads, expected_result_keys)
        self.assertEqual(
            len(resolution_requests), len(set(resolution_requests))
        )
        new_resolutions = scope.resolution_count - prior_resolution_count
        new_memo_entries = scope.memoized_reference_count - prior_memo_count
        self.assertEqual(actual_source_resolutions.call_count, new_resolutions)
        self.assertEqual(new_resolutions, new_memo_entries)
        self.assertGreater(span_validator_calls["span_ref_ranges"], 0)
        self.assertEqual(
            len(projection.support_provenance),
            arena.retained_record_counts["inherited_provenance"]
            - prior_retained_counts["inherited_provenance"],
        )
        self.assertEqual(
            arena.retained_record_counts["inherited_provenance"]
            - prior_retained_counts["inherited_provenance"],
            2,
        )
        current_validation_counts = arena.typed_validation_counts
        for counter_name in set(prior_validation_counts) | set(
            current_validation_counts
        ):
            if counter_name.startswith(
                ("results:", "plans:", "producers:", "receipts:")
            ):
                self.assertEqual(
                    current_validation_counts.get(counter_name, 0),
                    prior_validation_counts.get(counter_name, 0),
                )
        current_retained_counts = arena.retained_record_counts
        for table in ("results", "plans", "producers", "receipts"):
            self.assertEqual(
                current_retained_counts[table], prior_retained_counts[table]
            )

    def test_indexed_reader_entry_matches_wrapper_and_keeps_receipt_order(
        self,
    ) -> None:
        """The unsealed projection preserves caller receipt order and seals once.

        Input/expected case matrix:
            original receipt order: expanded output keeps that order.
            reversed receipt order: reader membership is unchanged, while the
            generation array retains the supplied reversed order.
        """
        from diffstory.report_storage import GenerationRecordSet
        from diffstory.report_storage import ReportRecordGraph

        expected_ledger = build_evidence_ledger(
            self.report,
            **self.manifest.ledger_parameters.model_dump(),
        )
        expected_index = build_ledger_index(expected_ledger)
        expected_candidate = assemble_candidate(
            self.report,
            self.manifest,
            ledger=expected_ledger,
            index=expected_index,
        )
        outputs = []
        for label, reverse_receipts in (
            ("original", False),
            ("reversed", True),
        ):
            _ledger, _index, arena, generation, details, sources = (
                indexed_assembly_inputs(
                    self.report, self.manifest, author=self.author
                )
            )
            if reverse_receipts:
                generation = GenerationRecordSet(
                    generation.header,
                    run_ref=generation.run_ref,
                    details=generation.details,
                    plan_ref=generation.plan_ref,
                    planning=generation.planning,
                    sections=generation.sections,
                    document=generation.document,
                    producers=generation.producers,
                    receipts=tuple(reversed(generation.receipts)),
                )
            projection = assemble_indexed_reader(
                self.report,
                generation,
                arena=arena,
                details=details,
                sources=sources,
            )
            graph = arena.seal(
                generation,
                projection,
                {
                    "shape": "annotations",
                    "metadata": {
                        "schema": "diffstory.annotations.v2",
                        "base_sha": self.report["meta"]["base_sha"],
                        "head_sha": self.report["meta"]["head_sha"],
                    },
                },
            )
            self.assertIsInstance(graph, ReportRecordGraph)
            expanded = expand_report_storage(
                graph, max_output_bytes=128 * 1024 * 1024
            )
            expected = candidate_storage_semantics(
                copy.deepcopy(expected_candidate)
            )
            if reverse_receipts:
                expected["generation"]["receipts"].reverse()
            outputs.append((label, expected, expanded))

        for label, expected, output in outputs:
            with self.subTest(case=label):
                self.assertEqual(
                    canonical_bytes(output),
                    canonical_bytes(expected),
                )

    def test_section_and_document_coverage_uses_binding_subjects(self) -> None:
        """Context-bound item keys do not stand in for covered plan subjects.

        Input/expected case matrix:
            each section result: item key differs from its subject key, while
            projected group key equals that subject key.
            document result: item key differs from the plan subject, while the
            selected binding exposes the exact plan subject and plan key.
        """
        _ledger, _index, arena, generation, details, sources = (
            indexed_assembly_inputs(
                self.report, self.manifest, author=self.author
            )
        )
        reader = arena._snapshot_reader()  # noqa: SLF001 # Select immutable binding facts without full result expansion.
        section_cases = tuple(
            (
                reference.item_key,
                reader.result_binding_facts(reference)[1],
            )
            for reference in generation.sections
        )
        document_kind, document_subject, document_plan = (
            reader.result_binding_facts(generation.document)
        )
        identity_cases = (
            *(
                ("section", item_key, subject)
                for item_key, subject in section_cases
            ),
            ("document", generation.document.item_key, document_subject),
        )
        for label, item_key, subject in identity_cases:
            with self.subTest(case=label, subject=subject):
                self.assertNotEqual(item_key, subject)
        self.assertEqual(document_kind, "document")
        self.assertEqual(document_subject, document_plan)

        projection = assemble_indexed_reader(
            self.report,
            generation,
            arena=arena,
            details=details,
            sources=sources,
        )
        self.assertEqual(
            tuple(section.group_key for section in projection.sections),
            tuple(subject for _item_key, subject in section_cases),
        )

    def test_indexed_assembly_rejects_wrong_prose_subjects_and_plans(
        self,
    ) -> None:
        """Context-bound prose cannot claim another plan subject or plan.

        Input/expected case matrix:
            section bound to an unknown subject: ``section_coverage``.
            document bound to another plan: ``document_plan_mismatch``.
            planning card bound to another plan: ``prose_plan_mismatch``.
        """
        cases = ("section_subject", "document_plan", "planning_plan")
        for case in cases:
            with self.subTest(case=case):
                _ledger, _index, arena, generation, details, sources = (
                    indexed_assembly_inputs(
                        self.report, self.manifest, author=self.author
                    )
                )
                reason = "section_coverage"
                if case == "section_subject":
                    original = self.manifest.sections[0]
                    subject_key = semantic_digest("foreign indexed group")
                    binding = ProseBinding(
                        kind="section",
                        subject_key=subject_key,
                        plan_key=original.binding.plan_key,
                        context_digest=original.binding.context_digest,
                    )
                    wrong_result = original.model_copy(
                        update={"binding": binding, "item_key": binding.key}
                    )
                    wrong_reference = arena.put_result(wrong_result)
                    generation = indexed_generation_with(
                        generation,
                        sections=(wrong_reference, *generation.sections[1:]),
                    )
                elif case == "document_plan":
                    original = self.manifest.document
                    plan_key = semantic_digest("foreign indexed plan")
                    binding = ProseBinding(
                        kind="document",
                        subject_key=plan_key,
                        plan_key=plan_key,
                        context_digest=original.binding.context_digest,
                    )
                    wrong_result = original.model_copy(
                        update={"binding": binding, "item_key": binding.key}
                    )
                    wrong_reference = arena.put_result(wrong_result)
                    generation = indexed_generation_with(
                        generation, document=wrong_reference
                    )
                    reason = "document_plan_mismatch"
                else:
                    original = self.manifest.planning[0]
                    binding = ProseBinding(
                        kind="planning_card",
                        subject_key=original.binding.subject_key,
                        plan_key=semantic_digest("foreign indexed plan"),
                        context_digest=original.binding.context_digest,
                    )
                    wrong_result = original.model_copy(
                        update={"binding": binding, "item_key": binding.key}
                    )
                    wrong_reference = arena.put_result(wrong_result)
                    generation = indexed_generation_with(
                        generation,
                        planning=(wrong_reference, *generation.planning[1:]),
                    )
                    reason = "prose_plan_mismatch"

                with self.assertRaisesRegex(ValueError, reason):
                    assemble_indexed_reader(
                        self.report,
                        generation,
                        arena=arena,
                        details=details,
                        sources=sources,
                    )

    def test_indexed_assembly_checks_group_membership_against_current_index(
        self,
    ) -> None:
        """A stored plan cannot reassign current ledger units between groups.

        Input/expected case matrix:
            the original plan keeps each current group assignment and is
            accepted; a re-bound plan swaps actual unit memberships and fails
            with ``plan_unit_coverage``.
        """
        _ledger, _index, arena, generation, details, sources = (
            indexed_assembly_inputs(
                self.report, self.manifest, author=self.author
            )
        )
        original_groups = self.manifest.plan.groups
        self.assertGreaterEqual(len(original_groups), 2)
        first, second = original_groups[:2]
        changed_first = first.model_copy(
            update={
                "group": first.group.model_copy(
                    update={"unit_keys": second.group.unit_keys}
                )
            }
        )
        changed_second = second.model_copy(
            update={
                "group": second.group.model_copy(
                    update={"unit_keys": first.group.unit_keys}
                )
            }
        )
        changed_plan = self.manifest.plan.model_copy(
            update={
                "groups": (changed_first, changed_second, *original_groups[2:])
            }
        )
        plan_ref = arena.put_record("story_plan", changed_plan)
        generation = indexed_generation_with(generation, plan_ref=plan_ref)

        with self.assertRaisesRegex(ValueError, "plan_unit_coverage"):
            assemble_indexed_reader(
                self.report,
                generation,
                arena=arena,
                details=details,
                sources=sources,
            )

    def test_indexed_assembly_checks_receipt_membership_and_binding(
        self,
    ) -> None:
        """Receipt selectors must be a complete set with current result bindings.

        Input/expected case matrix:
            duplicate receipt membership fails with ``receipt_coverage``.
            an owned receipt naming a foreign producer fails with
            ``receipt_binding_mismatch``.
        """
        cases = ("receipt_membership", "receipt_producer")
        for case in cases:
            with self.subTest(case=case):
                manifest = self.manifest
                if case == "receipt_producer":
                    first = manifest.receipts[0].model_copy(
                        update={
                            "producer_key": semantic_digest(
                                "foreign indexed producer"
                            )
                        }
                    )
                    manifest = manifest.model_copy(
                        update={"receipts": (first, *manifest.receipts[1:])}
                    )
                _ledger, _index, arena, generation, details, sources = (
                    indexed_assembly_inputs(
                        self.report, manifest, author=self.author
                    )
                )
                reason = "receipt_binding_mismatch"
                if case == "receipt_membership":
                    receipts = generation.receipts
                    generation = indexed_generation_with(
                        generation,
                        receipts=(receipts[0], *receipts[:-1]),
                    )
                    reason = "receipt_coverage"
                with self.assertRaisesRegex(ValueError, reason):
                    assemble_indexed_reader(
                        self.report,
                        generation,
                        arena=arena,
                        details=details,
                        sources=sources,
                    )

    def test_indexed_applied_fields_reject_wrong_prose_identity(self) -> None:
        """Application cannot read a selector with another prose identity.

        Input/expected case matrix:
            planning result selected as section prose: rejected by kind.
            section selector rebound to another subject: rejected by subject.
            valid section row bound to another plan: rejected by plan.
            Each invalid selector fails with ``reader_section_mismatch``.
        """
        from diffstory.report_storage import IndexedReaderProjection
        from diffstory.report_storage import ReaderSectionRef

        _ledger, _index, arena, generation, details, sources = (
            indexed_assembly_inputs(
                self.report, self.manifest, author=self.author
            )
        )
        projection = assemble_indexed_reader(
            self.report,
            generation,
            arena=arena,
            details=details,
            sources=sources,
        )
        original = projection.sections[0]
        original_result = self.manifest.sections[0]
        other_plan = semantic_digest("foreign applied-fields plan")
        wrong_plan_binding = ProseBinding(
            kind="section",
            subject_key=original.group_key,
            plan_key=other_plan,
            context_digest=original_result.binding.context_digest,
        )
        wrong_plan_result = original_result.model_copy(
            update={
                "binding": wrong_plan_binding,
                "item_key": wrong_plan_binding.key,
            }
        )
        wrong_plan_reference = arena.put_result(wrong_plan_result)

        invalid_sections = (
            (
                "kind",
                ReaderSectionRef(
                    group_id=original.group_id,
                    group_key=original.group_key,
                    number=original.number,
                    next_id=original.next_id,
                    prose=generation.planning[0],
                    details=original.details,
                ),
            ),
            (
                "subject",
                ReaderSectionRef(
                    group_id=original.group_id,
                    group_key=semantic_digest("foreign applied-fields group"),
                    number=original.number,
                    next_id=original.next_id,
                    prose=original.prose,
                    details=original.details,
                ),
            ),
            (
                "plan",
                ReaderSectionRef(
                    group_id=original.group_id,
                    group_key=original.group_key,
                    number=original.number,
                    next_id=original.next_id,
                    prose=wrong_plan_reference,
                    details=original.details,
                ),
            ),
        )
        groups = {group["id"]: group for group in self.report["groups"]}
        for label, section in invalid_sections:
            with self.subTest(case=label):
                invalid_projection = IndexedReaderProjection(
                    shape=projection.shape,
                    document=projection.document,
                    sections=(section, *projection.sections[1:]),
                    context_details=projection.context_details,
                    citations=projection.citations,
                    support_provenance=projection.support_provenance,
                    version=projection.version,
                    provenance=projection.provenance,
                )
                with self.assertRaisesRegex(
                    ValueError, "reader_section_mismatch"
                ):
                    indexed_applied_fields(
                        invalid_projection,
                        arena=arena,
                        details=details,
                        sources=sources,
                        groups=groups,
                    )

    def test_indexed_assembly_rejects_current_binding_and_reference_errors(
        self,
    ) -> None:
        """Selected rows fail when their order, source, or provenance is unsupported.

        Input/expected case matrix:
            changed report revision: ``source_revision_changed``.
            reordered section references: ``section_coverage``.
            unsupported inherited producer endpoint:
                ``unconsumed_statement_producer``.
            wrong source-slice digest: ``source_slice_digest``.
            out-of-span slice coordinates: ``invalid_source_slice``.
            rehashed inherited statement reference: ``inherited_statement_mismatch``.
        """
        cases = (
            "revision",
            "sections",
            "slice_digest",
            "slice_coordinate",
            "statement_digest",
            "inherited_endpoint",
        )
        for case in cases:
            with self.subTest(case=case):
                report = self.report
                manifest = self.manifest
                if case in {
                    "slice_digest",
                    "slice_coordinate",
                    "statement_digest",
                    "inherited_endpoint",
                }:
                    error = {
                        "slice_digest": "digest",
                        "slice_coordinate": "coordinate",
                    }.get(case)
                    manifest, author = indexed_support_manifest(
                        self.report,
                        manifest_fixture()[1],
                        slice_error=error,
                        statement_digest_error=case == "statement_digest",
                        inherited_endpoint_error=case == "inherited_endpoint",
                    )
                else:
                    author = self.author
                _ledger, _index, arena, generation, details, sources = (
                    indexed_assembly_inputs(report, manifest, author=author)
                )
                if case == "revision":
                    report = copy.deepcopy(report)
                    report["meta"]["base_sha"] = "0" * 40
                    reason = "source_revision_changed"
                elif case == "sections":
                    generation = GenerationRecordSet(
                        generation.header,
                        run_ref=generation.run_ref,
                        details=generation.details,
                        plan_ref=generation.plan_ref,
                        planning=generation.planning,
                        sections=tuple(reversed(generation.sections)),
                        document=generation.document,
                        producers=generation.producers,
                        receipts=generation.receipts,
                    )
                    reason = "section_coverage"
                elif case == "slice_digest":
                    reason = "source_slice_digest"
                elif case == "slice_coordinate":
                    reason = "invalid_source_slice"
                elif case == "statement_digest":
                    reason = "inherited_statement_mismatch"
                else:
                    reason = "unconsumed_statement_producer"
                with self.assertRaisesRegex(ValueError, reason):
                    assemble_indexed_reader(
                        report,
                        generation,
                        arena=arena,
                        details=details,
                        sources=sources,
                    )


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
