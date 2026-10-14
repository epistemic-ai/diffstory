"""Generation bindings, trusted production validation and exact v2 assembly.

The data contract alone is not a trust authority. validate_generation rebuilds
current source and checks support, hierarchy and actual request metadata. The
public application and rendering boundaries must call it before using v2 data.
Legacy generation.v1 retains its existing guarantees.
"""

from __future__ import annotations

import json
from bisect import bisect_right
from collections import defaultdict
from collections.abc import Mapping
from collections.abc import Sequence
from typing import TYPE_CHECKING
from typing import Annotated
from typing import Literal

from pydantic import Field
from pydantic import field_validator
from pydantic import model_validator

from .narrative_contracts import Contract
from .narrative_contracts import DetailResult
from .narrative_contracts import Digest
from .narrative_contracts import DocumentResult
from .narrative_contracts import NarrativeResult
from .narrative_contracts import Natural
from .narrative_contracts import PlanningCardResult
from .narrative_contracts import Positive
from .narrative_contracts import ResultKind
from .narrative_contracts import SectionResult
from .narrative_contracts import SourceSliceRef
from .narrative_contracts import SpanRef
from .narrative_contracts import StatementRef
from .narrative_contracts import StoryPlan
from .narrative_contracts import SupportedStatement
from .narrative_contracts import result_key
from .narrative_contracts import result_kind
from .narrative_plan import _OwnedList
from .narrative_plan import _OwnedMapping
from .narrative_plan import _OwnedTuple
from .narrative_plan import _thaw_owned
from .narrative_plan import canonical_bytes
from .narrative_plan import content_digest
from .narrative_plan import semantic_digest
from .narrative_protocol import (
    LedgerParameters,  # noqa: TC001 # Runtime Pydantic annotation and public re-export.
)
from .narrative_protocol import (
    RequestConfiguration,  # noqa: TC001 # Runtime Pydantic annotation.
)
from .narrative_runtime import ArtifactInvocation
from .narrative_runtime import PreparedMetadata
from .narrative_runtime import RunIdentity
from .narrative_storage import StoredInvocation
from .narrative_storage import decode_invocation
from .narrative_storage import encode_invocation

if TYPE_CHECKING:
    from collections.abc import Callable

    from .narrative_contracts import EvidenceLedger
    from .narrative_hierarchy import IndexedPlanningSources
    from .narrative_hierarchy import IndexedProseResult
    from .narrative_plan import IndexedDetailContext
    from .narrative_plan import LedgerIndex
    from .narrative_protocol import V2Options
    from .narrative_runtime import StoredArtifact
    from .narrative_wire import ProseContext
    from .narrative_wire import WireAdapter
    from .report_storage import GenerationRecordSet
    from .report_storage import IndexedReaderProjection
    from .report_storage import RecordRef
    from .report_storage import ReportRecordGraph
    from .report_storage import ResultRef
    from .report_storage import _RecordArena


def _require(condition: bool, reason: str) -> None:  # noqa: FBT001 # Assertion predicate.
    if not condition:
        raise ValueError(reason)


def compiler_fact_signature(report: dict | _OwnedMapping) -> str:
    """Hash compiler facts after removing only declared generated presentation.

    Exclude root document/generation/narrative_v2 and group presentation.
    Canonicalize group order by compiler ID. Retain every other field, including
    metadata, revisions, source, classification, prerequisites and test status.
    Never compare an annotated report's full hash to its pre-annotation hash.

    Preconditions:
        ``report`` is a strict public dictionary or the private read-only
        primitive view owned by a ReportSession capture.
    Postconditions:
        Returns the existing compiler-fact signature with unchanged field and
        scalar semantics.
    Essential behavior:
        Hashes selected mapping views by stream, without thawing an owned report.
    """
    _require(
        type(report) in {dict, _OwnedMapping}
        and type(report.get("groups")) in {list, _OwnedList},
        "invalid_report_shape",
    )
    groups = report["groups"]
    _require(
        all(
            type(group) in {dict, _OwnedMapping}
            and type(group.get("id")) is str
            for group in groups
        ),
        "invalid_group_shape",
    )
    _require(
        len({group["id"] for group in groups}) == len(groups),
        "duplicate_group",
    )
    excluded = {"document", "generation", "groups", "narrative_v2"}
    projection = _OwnedMapping(
        (
            *tuple(
                (key, value)
                for key, value in report.items()
                if key not in excluded
            ),
            (
                "groups",
                _OwnedList(
                    tuple(
                        _OwnedMapping(
                            tuple(
                                (key, value)
                                for key, value in group.items()
                                if key
                                not in {
                                    "narrative",
                                    "title",
                                    "number",
                                    "next_id",
                                }
                            )
                        )
                        for group in sorted(
                            groups, key=lambda group: group["id"]
                        )
                    )
                ),
            ),
        )
    )
    return semantic_digest("diffstory.compiler-facts.v2", projection)


class CurrentUsage(Contract):
    """Current provider usage only; historical artifact usage is never added."""

    input_tokens: Natural
    cached_input_tokens: Natural
    output_tokens: Natural
    calls: Natural
    unreported_calls: Natural

    @model_validator(mode="after")
    def counters(self) -> CurrentUsage:
        """Keep subsets within their totals without inventing reported usage."""
        _require(
            self.cached_input_tokens <= self.input_tokens,
            "invalid_cached_usage",
        )
        _require(
            self.unreported_calls <= self.calls, "invalid_unreported_usage"
        )
        _require(
            self.calls > 0 or self.input_tokens + self.output_tokens == 0,
            "usage_without_calls",
        )
        return self


class ProducerDescriptor(Contract):
    """Store a full original invocation once, including every original peer.

    requests contains only actual production requests referenced by current
    receipts. A repair request may contain fewer items; invocation retains all
    original peer/context dependencies. Bodies and source buffers are absent.
    """

    version: Literal["diffstory.producer.v2"] = "diffstory.producer.v2"
    original_run: RunIdentity
    invocation: ArtifactInvocation
    requests: tuple[PreparedMetadata, ...] = Field(min_length=1)

    @field_validator("original_run")
    @classmethod
    def revalidate_run(cls, value: RunIdentity) -> RunIdentity:
        """Revalidate even a copied legacy runtime identity at this boundary."""
        return RunIdentity.model_validate(value.model_dump())

    @model_validator(mode="after")
    def bindings(self) -> ProducerDescriptor:
        """Require unique actual requests bounded by the original invocation."""
        invocation = self.invocation
        identity = {
            "provider": content_digest(
                canonical_bytes(self.original_run.provider_destination)
            ),
            "model": content_digest(canonical_bytes(self.original_run.model)),
        }
        _require(
            all(
                dependency.digest == identity[dependency.kind]
                for dependency in invocation.dependencies
                if dependency.kind in identity
            ),
            "producer_static_identity_mismatch",
        )
        original_items = set(invocation.batch.item_keys)
        _require(
            len({request.batch.id for request in self.requests})
            == len(self.requests),
            "duplicate_production_request",
        )
        for request in self.requests:
            _require(
                request.batch.stage == invocation.batch.stage,
                "production_stage_mismatch",
            )
            _require(
                set(request.batch.item_keys) <= original_items,
                "production_items_mismatch",
            )
            _require(
                request.batch.output_reservation <= request.max_output_tokens,
                "production_output_limit",
            )
            guard = (
                request.batch.output_reservation
                if request.enforceable_output_cap
                else request.max_output_tokens
            )
            _require(
                request.batch.output_guard >= guard, "production_output_guard"
            )
            _require(
                request.batch.input_bound
                + request.batch.output_guard
                + request.safety_tokens
                <= request.context_tokens,
                "production_input_limit",
            )
            _require(
                request.batch.estimated_response_bytes
                <= request.response_byte_limit,
                "production_response_limit",
            )
            _require(
                request.batch.estimated_seconds <= request.timeout_seconds,
                "production_timeout_limit",
            )
        return self

    @property
    def key(self) -> str:
        """Identify the shared producer descriptor without storing its body."""
        return semantic_digest("diffstory.producer-key.v2", self)


class ResultReceipt(Contract):
    """Bind one current result to an original producer and actual request."""

    item_key: Digest
    result_kind: ResultKind
    result_digest: Digest
    producer_key: Digest
    request_key: Digest
    artifact_key: Digest
    disposition: Literal["sent", "reused"]


def result_digest(result: NarrativeResult) -> str:
    """Bind a receipt to all typed prose, support, effects and uncertainty."""
    return semantic_digest("diffstory.receipt-result.v2", result)


class GenerationManifest(Contract):
    """Strict audit and coverage data; semantic authority stays with the report.

    unit_keys is the declared ledger coverage, including context-only units.
    Production validation compares a newly derived ledger; internal agreement cannot prove
    that the caller declared every unit. Source remains in report.source_evidence.
    Every detail and prose output has exactly one receipt. The deterministic
    plan is not a provider output and therefore has no receipt.
    """

    version: Literal["diffstory.generation.v2"] = "diffstory.generation.v2"
    provenance: Literal["unverified"] = "unverified"
    run: RunIdentity
    compiler_fact_signature: Digest
    current_usage: CurrentUsage
    ledger_digest: Digest
    ledger_parameters: LedgerParameters
    request_configuration: RequestConfiguration
    unit_keys: tuple[Digest, ...]
    details: tuple[DetailResult, ...]
    plan: StoryPlan
    planning: tuple[PlanningCardResult, ...]
    sections: tuple[SectionResult, ...]
    document: DocumentResult
    producers: tuple[ProducerDescriptor, ...] = Field(min_length=1)
    receipts: tuple[ResultReceipt, ...] = Field(min_length=1)

    @classmethod
    def model_validate_json(
        cls, json_data: str | bytes | bytearray, **kwargs: object
    ) -> GenerationManifest:
        """Dispatch saved storage while preserving strict JSON model validation.

        Runtime Python models stay expanded. Dispatch before Pydantic consumes
        JSON so arrays retain their strict JSON tuple semantics.
        """
        value = json.loads(json_data)
        if type(value) is dict and "storage_version" in value:
            json_data = canonical_bytes(expanded_generation(value))
        return super().model_validate_json(json_data, **kwargs)

    @field_validator("run")
    @classmethod
    def revalidate_run(cls, value: RunIdentity) -> RunIdentity:
        """Preserve exact pre-annotation identity, including original signature."""
        return RunIdentity.model_validate(value.model_dump())

    @model_validator(mode="after")
    def bindings(self) -> GenerationManifest:
        """Reject incomplete or inconsistent self-declared structural bindings."""
        units = set(self.unit_keys)
        _require(len(units) == len(self.unit_keys), "duplicate_ledger_unit")
        _require(
            len(self.details) == len(self.unit_keys)
            and {detail.unit_key for detail in self.details} == units,
            "detail_coverage",
        )
        _require(
            len({group.group.key for group in self.plan.groups})
            == len(self.plan.groups),
            "duplicate_plan_group",
        )
        _require(
            len(self.plan.ordered_group_keys) == len(self.plan.groups)
            and set(self.plan.ordered_group_keys)
            == {group.group.key for group in self.plan.groups},
            "plan_group_order",
        )
        _require(
            all(
                set(group.group.unit_keys) <= units
                for group in self.plan.groups
            ),
            "plan_unit_coverage",
        )
        _require(
            tuple(section.binding.subject_key for section in self.sections)
            == self.plan.ordered_group_keys,
            "section_coverage",
        )
        _require(
            len({card.id for card in self.plan.cards}) == len(self.plan.cards),
            "duplicate_plan_card",
        )
        subjects = (
            *self.plan.cards,
            *(
                self.plan.production.auxiliary_cards
                if self.plan.production
                else ()
            ),
        )
        _require(
            len({c.id for c in subjects}) == len(subjects),
            "duplicate_plan_card",
        )
        _require(
            len(self.planning) == len(subjects)
            and {card.binding.subject_key for card in self.planning}
            == {card.id for card in subjects},
            "planning_coverage",
        )
        prose = (*self.planning, *self.sections, self.document)
        _require(
            all(result.binding.plan_key == self.plan.id for result in prose),
            "prose_plan_mismatch",
        )
        results = (*self.details, *prose)
        by_key = {result_key(result): result for result in results}
        _require(len(by_key) == len(results), "duplicate_result")
        _require(
            len(self.receipts) == len(results)
            and {receipt.item_key for receipt in self.receipts} == set(by_key),
            "receipt_coverage",
        )
        self._bind_receipts(by_key)
        return self

    def _bind_receipts(self, results: dict[str, NarrativeResult]) -> None:
        producers = {producer.key: producer for producer in self.producers}
        _require(len(producers) == len(self.producers), "duplicate_producer")
        _require(
            len(
                {
                    (
                        producer.original_run.digest,
                        producer.invocation.semantic_key,
                        producer.invocation.batch.id,
                    )
                    for producer in self.producers
                }
            )
            == len(self.producers),
            "duplicate_original_producer",
        )
        _require(
            {receipt.producer_key for receipt in self.receipts}
            == set(producers),
            "producer_coverage",
        )
        requests = {
            key: {request.batch.id: request for request in producer.requests}
            for key, producer in producers.items()
        }
        lookups = {
            key: {
                lookup.item_key: lookup
                for lookup in producer.invocation.lookups
            }
            for key, producer in producers.items()
        }
        artifact_keys = {
            key: producer.invocation.artifact_keys()
            for key, producer in producers.items()
        }
        request_items = {
            (key, request.batch.id): set(request.batch.item_keys)
            for key, producer in producers.items()
            for request in producer.requests
        }
        used_requests = set()
        sent_requests = set()
        for receipt in self.receipts:
            producer = producers[receipt.producer_key]
            result = results[receipt.item_key]
            lookup = lookups[receipt.producer_key].get(receipt.item_key)
            request = requests[receipt.producer_key].get(receipt.request_key)
            _require(
                lookup is not None and request is not None,
                "missing_producer_binding",
            )
            _require(
                receipt.item_key
                in request_items[receipt.producer_key, receipt.request_key],
                "request_item_mismatch",
            )
            _require(
                receipt.result_kind
                == lookup.result_kind
                == result_kind(result),
                "receipt_kind_mismatch",
            )
            _require(
                receipt.result_digest == result_digest(result),
                "receipt_result_mismatch",
            )
            _require(
                receipt.artifact_key
                == artifact_keys[receipt.producer_key][receipt.item_key],
                "receipt_artifact_mismatch",
            )
            _require(
                producer.original_run.provider_destination
                == self.run.provider_destination
                and producer.original_run.model == self.run.model,
                "producer_identity_mismatch",
            )
            used_requests.add((receipt.producer_key, receipt.request_key))
            if receipt.disposition == "sent":
                _require(
                    producer.original_run == self.run, "sent_run_mismatch"
                )
                sent_requests.add((receipt.producer_key, receipt.request_key))
        _require(
            used_requests
            == {
                (key, request)
                for key, values in requests.items()
                for request in values
            },
            "unused_production_request",
        )
        _require(
            len(sent_requests) <= self.current_usage.calls,
            "sent_usage_mismatch",
        )


def validate_prose_support(  # noqa: C901 # Keep the existing current prose predicates in one shared validation path.
    result: NarrativeResult | IndexedProseResult,
    context: ProseContext,
    wire: WireAdapter,
) -> None:
    """Check binding and exact allowed support from a trusted current context.

    The production input builder supplies this context after validating details
    against the current ledger. A result cannot grant itself extra support.
    These checks establish references and coverage, not semantic truth.
    """
    from .narrative_hierarchy import validate_consumed_support
    from .narrative_runtime import ArtifactRuntimeError

    def require(condition: bool, reason: str = "citation") -> None:  # noqa: FBT001
        if not condition:
            raise ArtifactRuntimeError(reason)

    from .narrative_hierarchy import IndexedProseResult

    indexed = type(result) is IndexedProseResult
    binding = result.binding
    require(binding == context.binding, "dependencies_changed")
    item_key = result.item_key
    require(item_key == context.binding.key, "dependencies_changed")
    if context.sources is not None:
        try:
            validate_consumed_support(
                result, context.support, context.sources, context.consumed
            )
        except ValueError:
            msg = "citation"
            raise ArtifactRuntimeError(msg) from None
    allowed_claims = frozenset(context.claims)
    allowed_citations = frozenset(context.citations)
    indexed_entries = ()
    if indexed:
        indexed_entries = result.statement_entries()
        values = [statement for _path, statement in indexed_entries]
    else:
        values = [term.definition for term in result.terminology]
        for name in (
            "abstraction",
            "title",
            "orientation",
            "synthesis",
            "transition",
            "preamble",
            "lead",
            "closing",
        ):
            if hasattr(result, name):
                field = getattr(result, name)
                values.extend((field,) if name == "title" else field)
    for statement in values:
        require(set(statement.claims) <= allowed_claims)
        require(set(statement.citations) <= allowed_citations)
        require(set(statement.basis) <= set(context.support.basis))
        require(
            statement.author_context_digest
            == (
                context.author_digest
                if statement.kind == "author_motivation"
                else None
            )
        )
    if context.final:
        if indexed and result.kind == "section":
            require(
                not any(
                    path.field == "transition" for path, _ in indexed_entries
                ),
                "coverage",
            )
        elif isinstance(result, SectionResult):
            require(result.transition == (), "coverage")
    require(
        len(canonical_bytes(wire.prose_input(result)))
        <= context.max_wire_bytes,
        "schema",
    )


def build_receipts(
    artifacts: tuple[StoredArtifact, ...],
    original_run: Callable[[str], RunIdentity],
    reused_artifact_keys: frozenset[str],
) -> tuple[tuple[ProducerDescriptor, ...], tuple[ResultReceipt, ...]]:
    """Use real successful artifacts and share each full original descriptor."""
    groups = defaultdict(list)
    for artifact in artifacts:
        groups[
            (
                artifact.original_run_digest,
                artifact.invocation.semantic_key,
                artifact.invocation.batch.id,
            )
        ].append(artifact)
    producers = []
    receipts = []
    for values in groups.values():
        first = values[0]
        requests = {
            a.production_request.batch.id: a.production_request for a in values
        }
        producer = ProducerDescriptor(
            original_run=original_run(first.original_run_digest),
            invocation=first.invocation,
            requests=tuple(requests.values()),
        )
        producers.append(producer)
        for artifact in values:
            result = artifact.record.result
            receipts.append(
                ResultReceipt(
                    item_key=result_key(result),
                    result_kind=result_kind(result),
                    result_digest=result_digest(result),
                    producer_key=producer.key,
                    request_key=artifact.production_request.batch.id,
                    artifact_key=artifact.record.semantic_key,
                    disposition="reused"
                    if artifact.record.semantic_key in reused_artifact_keys
                    else "sent",
                )
            )
    return tuple(producers), tuple(receipts)


def assemble_candidate(  # noqa: C901, PLR0912, PLR0915 # Assemble all typed families, slice coordinates and immediate provenance.
    report: dict | _OwnedMapping,
    manifest: GenerationManifest,
    *,
    ledger: EvidenceLedger | None = None,
    index: LedgerIndex | None = None,
) -> dict:
    """Assemble exact typed prose and all details using current compiler bindings.

    This helper does not apply annotations or render HTML. Public consumers must
    first call validate_generation to reconstruct the declared pure protocol.
    Optional ledger/index values reuse the production builder's current index.
    Without them, source is rebuilt from the report and declared parameters.
    """
    from .analysis import source_url
    from .narrative_hierarchy import statement_at
    from .narrative_hierarchy import statement_entries
    from .narrative_plan import build_evidence_ledger
    from .narrative_plan import build_ledger_index
    from .narrative_plan import resolve_span_ref
    from .narrative_plan import span_reference
    from .narrative_plan import validate_detail_results
    from .source_lines import split_source_lines

    _require(
        compiler_fact_signature(report) == manifest.compiler_fact_signature,
        "compiler_facts_changed",
    )
    if ledger is None:
        ledger = build_evidence_ledger(
            report, **manifest.ledger_parameters.model_dump()
        )
    if index is None:
        index = build_ledger_index(ledger)
    index.check_ledger(ledger)
    _require(ledger.id == manifest.ledger_digest, "ledger_changed")
    validate_detail_results(manifest.details, ledger, index=index)
    group_ids = {g.group_key: g.group_id for g in ledger.group_bindings}
    details = {d.unit_key: d for d in manifest.details}
    assigned = {u for g in index.groups.values() for u in g.unit_keys}
    references = {}
    slices = {}
    provenance = {}
    prose = (*manifest.planning, *manifest.sections, manifest.document)
    producers = {r.item_key: r for r in prose}
    receipts = {r.item_key: r for r in manifest.receipts}
    for detail in manifest.details:
        # Retained context and empty acknowledgements need exact source links
        # even when no model statement cites them. Keep their original results.
        if detail.unit_key not in assigned or not (
            detail.explanations or detail.observations or detail.questions
        ):
            unit = index.units[detail.unit_key]
            for span in unit.spans:
                ref = span_reference(span, unit)
                references[semantic_digest(ref)] = ref
        for value in (
            *detail.explanations,
            *detail.observations,
            *detail.questions,
        ):
            for ref in value.citations:
                references[semantic_digest(ref)] = ref
    for result in prose:
        for _, value in statement_entries(result):
            for ref in value.citations:
                references[semantic_digest(ref)] = ref
            for ref in value.basis:
                if isinstance(ref, SourceSliceRef):
                    slices[semantic_digest(ref)] = ref
                else:
                    statement = statement_at(
                        producers[ref.producer_item_key], ref.statement_path
                    )
                    provenance[semantic_digest(ref)] = {
                        "key": semantic_digest(ref),
                        "reference": ref.model_dump(mode="json"),
                        "statement": statement.model_dump(mode="json"),
                        "consumption": "inherited_statement",
                        "receipt": receipts[ref.producer_item_key].model_dump(
                            mode="json"
                        ),
                    }
    resolved = []
    for key, ref in sorted(references.items()):
        span = resolve_span_ref(
            ref, index.units[ref.unit_key], ledger, index=index
        )
        binding = index.bindings[span.buffer_key]
        resolved.append(
            {
                "key": key,
                "reference": ref.model_dump(mode="json"),
                "consumption": "direct_source",
                "fragment_key": binding.fragment_key,
                "side": span.side,
                "path": span.path,
                "revision": report["meta"][
                    "base_sha" if span.side == "base" else "head_sha"
                ],
                "byte_start": span.byte_start,
                "byte_end": span.byte_end,
                "line_start": span.line_start,
                "line_end": span.line_end,
                "url": source_url(
                    report["meta"],
                    span.path,
                    span.side,
                    span.line_start,
                    span.line_end,
                ),
            }
        )
    physical_starts = {}
    original_fragments = {
        f["key"]: f for f in report["source_evidence"]["fragments"]
    }
    for key, ref in sorted(slices.items()):
        span = resolve_span_ref(
            ref.source, index.units[ref.source.unit_key], ledger, index=index
        )
        binding = index.bindings[span.buffer_key]
        if binding.fragment_key not in physical_starts:
            starts, offset = [], 0
            for line in split_source_lines(
                original_fragments[binding.fragment_key]["text"], keepends=True
            ):
                starts.append(offset)
                offset += len(line.encode("utf-8"))
            physical_starts[binding.fragment_key] = starts
        starts = physical_starts[binding.fragment_key]
        byte_start = span.byte_start + ref.byte_start
        byte_end = span.byte_start + ref.byte_end
        origin = bisect_right(starts, span.byte_start)
        line_start = (
            span.line_start + bisect_right(starts, byte_start) - origin
        )
        line_end = (
            span.line_start + bisect_right(starts, byte_end - 1) - origin
        )
        resolved.append(
            {
                "key": key,
                "reference": ref.model_dump(mode="json"),
                "consumption": "source_slice",
                "fragment_key": binding.fragment_key,
                "side": span.side,
                "path": span.path,
                "revision": report["meta"][
                    "base_sha" if span.side == "base" else "head_sha"
                ],
                "byte_start": byte_start,
                "byte_end": byte_end,
                "line_start": line_start,
                "line_end": line_end,
                "url": source_url(
                    report["meta"], span.path, span.side, line_start, line_end
                ),
            }
        )
    steps = []
    for position, section in enumerate(manifest.sections):
        key = section.binding.subject_key
        next_key = (
            manifest.plan.ordered_group_keys[position + 1]
            if position + 1 < len(manifest.sections)
            else None
        )
        steps.append(
            {
                "group_id": group_ids[key],
                "group_key": key,
                "number": position + 1,
                "next_id": group_ids[next_key] if next_key else None,
                "prose": section.model_dump(mode="json"),
                "details": [
                    details[u].model_dump(mode="json")
                    for u in index.groups[key].unit_keys
                ],
            }
        )
    return {
        "schema": "diffstory.annotations.v2",
        "base_sha": report["meta"]["base_sha"],
        "head_sha": report["meta"]["head_sha"],
        "document": manifest.document.model_dump(mode="json"),
        "steps": steps,
        "context_details": [
            d.model_dump(mode="json")
            for d in manifest.details
            if d.unit_key not in assigned
        ],
        "citations": resolved,
        "support_provenance": [provenance[key] for key in sorted(provenance)],
        "generation": stored_generation(manifest),
    }


def _indexed_assembly_bindings(  # noqa: PLR0913 # Keep exact capture ownership inputs explicit at this private boundary.
    report: dict | _OwnedMapping,
    generation: GenerationRecordSet,
    *,
    arena: _RecordArena,
    details: IndexedDetailContext,
    sources: IndexedPlanningSources,
    captured_input: object | None = None,
) -> tuple[object, object]:
    """Check one live arena/source binding before selected indexed work.

    Preconditions:
        ``report`` is the compiler report captured by ``generation``; the
        generation rows and detail context were captured in ``arena``; and
        ``sources`` is the exact live adapter for that detail context.
    Postconditions:
        Returns the registered indexed source state and one shallow immutable
        reader snapshot over the arena's current primitive rows.
    Essential behavior:
        Checks compiler facts, ledger identity, revisions, selector ownership,
        and detail order once without building a ledger or parsing results.
    """
    from .narrative_hierarchy import IndexedPlanningSources
    from .narrative_hierarchy import _state
    from .narrative_plan import IndexedDetailContext
    from .report_storage import GenerationRecordSet
    from .report_storage import _RecordArena

    if type(report) is dict:
        pass
    elif type(report) is _OwnedMapping:
        owns_report = getattr(captured_input, "owns_report_view", None)
        assert_current = getattr(
            captured_input, "assert_capture_current", None
        )
        _require(
            callable(owns_report)
            and owns_report(report)
            and callable(assert_current),
            "invalid_report_shape",
        )
        assert_current()
    else:
        _require(False, "invalid_report_shape")  # noqa: FBT003 # Deliberate stable failure branch.
    _require(
        type(generation) is GenerationRecordSet,
        "invalid_generation_record_set",
    )
    _require(type(arena) is _RecordArena, "invalid_record_arena")
    _require(
        type(details) is IndexedDetailContext, "invalid_indexed_detail_context"
    )
    _require(
        type(sources) is IndexedPlanningSources, "invalid_planning_sources"
    )
    sources.check_current()
    state = _state(sources)
    _require(
        state.details_context is details,
        "planning_sources_detail_context_mismatch",
    )
    detail_reader = details._arena_reader  # noqa: SLF001 # Require the accepted arena-backed context.
    _require(
        detail_reader is not None
        and detail_reader._owner_token is arena._owner_token,  # noqa: SLF001 # Bind context and arena without rebuilding records.
        "detail_context_arena_mismatch",
    )
    ledger = state.index._ledger  # noqa: SLF001 # Source check above validated this exact owner.
    header = generation.header
    _require(ledger.id == header.get("ledger_digest"), "ledger_changed")
    meta = report.get("meta")
    _require(type(meta) in {dict, _OwnedMapping}, "invalid_report_metadata")
    revisions = (meta.get("base_sha"), meta.get("head_sha"))
    _require(
        revisions == state.revision_binding,
        "source_revision_changed",
    )
    _require(
        compiler_fact_signature(report)
        == header.get("compiler_fact_signature"),
        "compiler_facts_changed",
    )

    ledger_unit_keys = tuple(state.index.units)
    detail_unit_keys = details.unit_keys
    header_unit_keys = header.get("unit_keys")
    _require(
        type(header_unit_keys) in {tuple, list}
        and tuple(header_unit_keys) == ledger_unit_keys
        and detail_unit_keys == ledger_unit_keys,
        "detail_set_coverage_mismatch",
    )
    generation_details = generation.details
    context_details = details.references
    _require(
        len(generation_details) == len(context_details)
        and generation_details == context_details
        and tuple(ref.item_key for ref in generation_details)
        == ledger_unit_keys,
        "detail_reference_mismatch",
    )
    reader = arena._snapshot_reader()  # noqa: SLF001 # One shallow selected-row view for this operation.
    return state, reader


def assemble_indexed_reader(  # noqa: C901, PLR0912, PLR0913, PLR0915 # Keep one checked pass over selected assembly records.
    report: dict | _OwnedMapping,
    generation: GenerationRecordSet,
    *,
    arena: _RecordArena,
    details: IndexedDetailContext,
    sources: IndexedPlanningSources,
    captured_input: object | None = None,
) -> IndexedReaderProjection:
    """Assemble ordered annotation references from accepted arena records.

    Preconditions:
        ``generation`` and ``details`` bind the same current arena and ledger;
        ``sources`` is their live indexed source adapter; all result, producer,
        and receipt rows were captured before this call; and the indexed detail
        and planning-source contexts carry the report's exact revision tuple.
    Postconditions:
        Returns exact section/detail membership, current resolved source
        citations, and reference-based inherited provenance in oracle order.
        The projection owns only typed record addresses and selector scalars.
    Essential behavior:
        Selects details, prose, plan facts, and receipts without reconstructing
        a ``GenerationManifest`` or ``V2Presentation``. The caller still owns
        full production, request, hierarchy, and standalone proof validation.

    Raises:
        ValueError: If current facts, result order, source bindings, receipt
            coverage, or any selected citation/provenance address is invalid.
    """
    from .analysis import source_url
    from .narrative_contracts import SourceSliceRef
    from .narrative_contracts import SpanRef
    from .narrative_contracts import StatementRef
    from .narrative_hierarchy import span_reference
    from .narrative_plan import content_digest
    from .narrative_plan import semantic_digest
    from .report_storage import IndexedReaderProjection
    from .report_storage import ReaderSectionRef
    from .report_storage import _thaw
    from .source_lines import split_source_lines

    state, reader = _indexed_assembly_bindings(
        report,
        generation,
        arena=arena,
        details=details,
        sources=sources,
        captured_input=captured_input,
    )
    plan_facts = arena.plan_assembly_facts(generation.plan_ref)
    plan_id = plan_facts.get("id")
    group_keys = plan_facts.get("group_keys")
    ordered_group_keys = plan_facts.get("ordered_group_keys")
    groups_by_key = plan_facts.get("groups")
    card_ids = plan_facts.get("card_ids")
    auxiliary_card_ids = plan_facts.get("auxiliary_card_ids")
    _require(
        type(plan_id) is str
        and type(group_keys) is tuple
        and type(ordered_group_keys) is tuple
        and isinstance(groups_by_key, Mapping)
        and type(card_ids) is tuple
        and type(auxiliary_card_ids) is tuple,
        "invalid_story_plan_facts",
    )
    _require(
        len(ordered_group_keys) == len(group_keys)
        and len(set(group_keys)) == len(group_keys)
        and len(set(ordered_group_keys)) == len(ordered_group_keys)
        and set(ordered_group_keys) == set(group_keys),
        "plan_group_order",
    )

    group_ids = {
        binding.group_key: binding.group_id
        for binding in state.index._ledger.group_bindings  # noqa: SLF001 # Current compiler group bindings were checked with the source state.
    }
    _require(
        set(group_ids) == set(group_keys)
        and all(type(group_ids[key]) is str for key in ordered_group_keys),
        "group_binding_mismatch",
    )
    current_group_keys = set(state.index.groups)
    _require(
        set(group_keys) == current_group_keys
        and set(groups_by_key) == current_group_keys,
        "plan_group_coverage",
    )
    for group_key in ordered_group_keys:
        group_unit_keys = groups_by_key.get(group_key)
        current_group = state.index.groups[group_key]
        _require(
            type(group_unit_keys) is tuple
            and group_unit_keys == current_group.unit_keys,
            "plan_unit_coverage",
        )

    section_refs = generation.sections
    section_bindings = tuple(
        reader.result_binding_facts(reference) for reference in section_refs
    )
    _require(
        len(section_refs) == len(ordered_group_keys)
        and all(
            kind == "section" for kind, _subject, _plan in section_bindings
        )
        and tuple(subject for _kind, subject, _plan in section_bindings)
        == ordered_group_keys,
        "section_coverage",
    )
    _require(
        all(plan == plan_id for _kind, _subject, plan in section_bindings),
        "prose_plan_mismatch",
    )
    document_binding = reader.result_binding_facts(generation.document)
    _require(
        document_binding == ("document", plan_id, plan_id),
        "document_plan_mismatch",
    )
    planning_refs = generation.planning
    expected_planning_keys = (*card_ids, *auxiliary_card_ids)
    planning_bindings = tuple(
        reader.result_binding_facts(reference) for reference in planning_refs
    )
    planning_subjects = tuple(
        subject for _kind, subject, _plan in planning_bindings
    )
    _require(
        len(planning_bindings) == len(expected_planning_keys)
        and all(
            kind == "planning_card"
            for kind, _subject, _plan in planning_bindings
        )
        and len(set(planning_subjects)) == len(planning_subjects)
        and set(planning_subjects) == set(expected_planning_keys),
        "planning_coverage",
    )
    _require(
        all(plan == plan_id for _kind, _subject, plan in planning_bindings),
        "prose_plan_mismatch",
    )

    detail_refs = generation.details
    details_by_unit = {
        reference.item_key: reference for reference in detail_refs
    }
    assigned_units = {
        unit_key
        for group_unit_keys in groups_by_key.values()
        for unit_key in group_unit_keys
    }
    _require(
        assigned_units <= set(details_by_unit)
        and set(details_by_unit) == set(state.index.units),
        "detail_coverage",
    )
    reader_sections = []
    for position, group_key in enumerate(ordered_group_keys):
        group_unit_keys = groups_by_key.get(group_key)
        _require(
            type(group_unit_keys) is tuple
            and all(
                unit_key in details_by_unit for unit_key in group_unit_keys
            ),
            "plan_unit_coverage",
        )
        next_group_key = (
            ordered_group_keys[position + 1]
            if position + 1 < len(ordered_group_keys)
            else None
        )
        reader_sections.append(
            ReaderSectionRef(
                group_id=group_ids[group_key],
                group_key=group_key,
                number=position + 1,
                next_id=(
                    group_ids[next_group_key]
                    if next_group_key is not None
                    else None
                ),
                prose=section_refs[position],
                details=tuple(
                    details_by_unit[unit_key] for unit_key in group_unit_keys
                ),
            )
        )

    result_refs = (
        *detail_refs,
        *planning_refs,
        *section_refs,
        generation.document,
    )
    result_keys = tuple(reference.item_key for reference in result_refs)
    _require(
        len(set(result_keys)) == len(result_keys), "duplicate_result_identity"
    )
    producer_keys = set(arena.producer_keys(generation.producers))
    receipt_refs = generation.receipts
    _require(len(receipt_refs) == len(result_refs), "receipt_coverage")
    selected_receipts = tuple(
        arena.receipt_reference_for_item(result_ref.item_key)
        for result_ref in result_refs
    )
    _require(
        len({reference.storage_id for reference in receipt_refs})
        == len(receipt_refs)
        and {reference.storage_id for reference in receipt_refs}
        == {reference.storage_id for reference in selected_receipts},
        "receipt_coverage",
    )
    for result_ref, receipt_ref in zip(
        result_refs, selected_receipts, strict=True
    ):
        receipt = arena.receipt_view(receipt_ref)
        receipt_digest, _result_content_digest = arena.result_digests(
            result_ref
        )
        _require(
            receipt.get("item_key") == result_ref.item_key
            and receipt.get("result_kind") == result_ref.kind
            and receipt.get("result_digest") == receipt_digest
            and receipt.get("producer_key") in producer_keys,
            "receipt_binding_mismatch",
        )

    direct_refs: dict[str, SpanRef] = {}
    source_slices: dict[str, SourceSliceRef] = {}
    all_spans: dict[str, SpanRef] = {}
    inherited: dict[str, tuple[RecordRef, ResultRef, RecordRef]] = {}

    def add_direct_reference(value: object) -> None:
        """Strictly capture one selected citation and deduplicate by identity."""
        reference = SpanRef.model_validate(_thaw(value))
        key = semantic_digest(reference)
        direct_refs.setdefault(key, reference)
        all_spans.setdefault(key, reference)

    for detail_ref in detail_refs:
        unit_key = detail_ref.item_key
        _digest, _observation_ids, counts = details.planning_detail_facts(
            unit_key
        )
        for record_kind, count in zip(
            ("explanation", "observation", "question"), counts, strict=True
        ):
            for ordinal in range(count):
                record = details.indexed_detail_record_view(
                    unit_key, record_kind, ordinal
                )
                for reference in record["citations"]:
                    add_direct_reference(reference)
        if unit_key not in assigned_units or not any(counts):
            unit = state.index.units[unit_key]
            for span in unit.spans:
                reference = span_reference(span, unit)
                key = semantic_digest(reference)
                direct_refs.setdefault(key, reference)
                all_spans.setdefault(key, reference)

    prose_refs = (*planning_refs, *section_refs, generation.document)
    prose_by_item: dict[str, ResultRef] = {}
    for result_ref in prose_refs:
        _require(
            result_ref.item_key not in prose_by_item, "duplicate_prose_result"
        )
        prose_by_item[result_ref.item_key] = result_ref
    for result_ref in prose_refs:
        for path in reader.statement_paths(result_ref):
            statement = reader.statement_view(result_ref, path)
            for reference in statement.get("citations", ()):
                add_direct_reference(reference)
            for basis in statement.get("basis", ()):
                if not isinstance(basis, Mapping):
                    msg = "invalid_statement_support"
                    raise ValueError(msg)
                if basis.get("kind") == "source_slice":
                    source_slice = SourceSliceRef.model_validate(_thaw(basis))
                    key = semantic_digest(source_slice)
                    source_slices.setdefault(key, source_slice)
                    source_key = semantic_digest(source_slice.source)
                    all_spans.setdefault(source_key, source_slice.source)
                else:
                    statement_ref = StatementRef.model_validate(_thaw(basis))
                    key = semantic_digest(statement_ref)
                    producer_ref = prose_by_item.get(
                        statement_ref.producer_item_key
                    )
                    _require(
                        producer_ref is not None,
                        "unconsumed_statement_producer",
                    )
                    reference_ref = arena.put_record(
                        "statement_ref", statement_ref
                    )
                    receipt_ref = arena.receipt_reference_for_item(
                        statement_ref.producer_item_key
                    )
                    provenance_ref = arena.put_indexed_inherited_provenance(
                        reference_ref, producer_ref, receipt_ref
                    )
                    inherited.setdefault(
                        key,
                        (provenance_ref, producer_ref, receipt_ref),
                    )

    scope = details.citation_scope
    resolved_spans = {
        key: scope._resolve_validated(  # noqa: SLF001 # The enclosing source check already established exact ledger/index/revision liveness.
            reference, state.index.units[reference.unit_key]
        )
        for key, reference in all_spans.items()
    }
    meta = report["meta"]
    fragment_texts = {
        fragment["key"]: fragment["text"]
        for fragment in report["source_evidence"]["fragments"]
        if isinstance(fragment, Mapping)
        and type(fragment.get("key")) is str
        and type(fragment.get("text")) is str
    }
    physical_starts: dict[str, list[int]] = {}
    citation_refs = []
    for key, reference in sorted(direct_refs.items()):
        span = resolved_spans[key]
        binding = state.index.bindings[span.buffer_key]
        citation = {
            "key": key,
            "reference": reference.model_dump(mode="json"),
            "consumption": "direct_source",
            "fragment_key": binding.fragment_key,
            "side": span.side,
            "path": span.path,
            "revision": meta[
                "base_sha" if span.side == "base" else "head_sha"
            ],
            "byte_start": span.byte_start,
            "byte_end": span.byte_end,
            "line_start": span.line_start,
            "line_end": span.line_end,
            "url": source_url(
                meta, span.path, span.side, span.line_start, span.line_end
            ),
        }
        citation_refs.append(arena.put_record("resolved_citation", citation))

    for key, reference in sorted(source_slices.items()):
        source_key = semantic_digest(reference.source)
        span = resolved_spans[source_key]
        source_ref = reference.source
        _require(
            source_ref in state.source_records,
            "invalid_source_slice",
        )
        source_bytes = span.text.encode("utf-8")
        _require(
            reference.byte_end <= len(source_bytes), "invalid_source_slice"
        )
        sliced_bytes = source_bytes[reference.byte_start : reference.byte_end]
        _require(
            content_digest(sliced_bytes) == reference.content_digest,
            "source_slice_digest",
        )
        try:
            sliced_bytes.decode("utf-8")
        except UnicodeError as error:
            msg = "source_slice_utf8_boundary"
            raise ValueError(msg) from error
        binding = state.index.bindings[span.buffer_key]
        fragment_text = fragment_texts.get(binding.fragment_key)
        _require(
            type(fragment_text) is str, "citation_source_fragment_missing"
        )
        if binding.fragment_key not in physical_starts:
            starts: list[int] = []
            offset = 0
            for line in split_source_lines(fragment_text, keepends=True):
                starts.append(offset)
                offset += len(line.encode("utf-8"))
            physical_starts[binding.fragment_key] = starts
        starts = physical_starts[binding.fragment_key]
        byte_start = span.byte_start + reference.byte_start
        byte_end = span.byte_start + reference.byte_end
        origin = bisect_right(starts, span.byte_start)
        line_start = (
            span.line_start + bisect_right(starts, byte_start) - origin
        )
        line_end = (
            span.line_start + bisect_right(starts, byte_end - 1) - origin
        )
        citation = {
            "key": key,
            "reference": reference.model_dump(mode="json"),
            "consumption": "source_slice",
            "fragment_key": binding.fragment_key,
            "side": span.side,
            "path": span.path,
            "revision": meta[
                "base_sha" if span.side == "base" else "head_sha"
            ],
            "byte_start": byte_start,
            "byte_end": byte_end,
            "line_start": line_start,
            "line_end": line_end,
            "url": source_url(
                meta, span.path, span.side, line_start, line_end
            ),
        }
        citation_refs.append(arena.put_record("resolved_citation", citation))

    provenance_refs = tuple(inherited[key][0] for key in sorted(inherited))
    context_refs = tuple(
        reference
        for reference in detail_refs
        if reference.item_key not in assigned_units
    )
    return IndexedReaderProjection(
        shape="annotations",
        document=generation.document,
        sections=tuple(reader_sections),
        context_details=context_refs,
        citations=tuple(citation_refs),
        support_provenance=provenance_refs,
    )


def assemble_indexed_candidate(
    report: dict,
    generation: GenerationRecordSet,
    *,
    arena: _RecordArena,
    details: IndexedDetailContext,
    sources: IndexedPlanningSources,
) -> ReportRecordGraph:
    """Seal an indexed annotation reader over the already owned record arena.

    Preconditions:
        All generation and detail rows are owned by ``arena`` and ``sources``
        remains the exact live source adapter for ``details``.
    Postconditions:
        Returns a closed annotations graph with the exact current base/head
        metadata and the assembled indexed reader projection.
    Essential behavior:
        Shares :func:`assemble_indexed_reader` and the existing arena integrity
        seal; the projection API remains available when the caller must choose
        a different final artifact root.
    """
    from .report_storage import _RecordArena

    _require(type(arena) is _RecordArena, "invalid_record_arena")
    projection = assemble_indexed_reader(
        report,
        generation,
        arena=arena,
        details=details,
        sources=sources,
    )
    meta = report["meta"]
    root = {
        "shape": "annotations",
        "metadata": {
            "schema": "diffstory.annotations.v2",
            "base_sha": meta["base_sha"],
            "head_sha": meta["head_sha"],
        },
    }
    return arena.seal(generation, projection, root)


def validate_generation(
    report: dict | _OwnedMapping,
    manifest: GenerationManifest,
    *,
    provider: object | None = None,
    options: V2Options | None = None,
    candidate: dict | None = None,
    _captured_input: object | None = None,
) -> dict:
    """Rebuild trusted production support, hierarchy, requests and exact assembly.

    Preconditions:
        Public calls supply an expanded dictionary. The private captured-input
        path supplies only the exact immutable view registered by its session.
    Postconditions:
        Returns a data candidate after the complete independent proof succeeds.
    Essential behavior:
        Does not let a report dictionary or partial session binding skip proof.
    """
    from .narrative_v2 import validate_production

    if _captured_input is None:
        _require(type(report) is dict, "invalid_report_shape")
    else:
        owns_report = getattr(_captured_input, "owns_report_view", None)
        _require(
            callable(owns_report) and owns_report(report),
            "invalid_report_shape",
        )
    return validate_production(
        report,
        manifest,
        provider=provider,
        options=options,
        candidate=candidate,
        _captured_input=_captured_input,
    )


LegacyIdentifier = Annotated[str, Field(pattern=r"^[a-f0-9]{16}$")]


class ResolvedCitation(Contract):
    """Bind direct source or an exact slice to current original fragment bytes."""

    key: Digest
    reference: SpanRef | SourceSliceRef
    consumption: Literal["direct_source", "source_slice"]
    fragment_key: str
    side: Literal["base", "head"]
    path: str
    revision: str
    byte_start: Natural
    byte_end: Natural
    line_start: Positive
    line_end: Positive
    url: str | None


class InheritedProvenance(Contract):
    """Identify the consumed statement and its actual successful request receipt."""

    key: Digest
    reference: StatementRef
    statement: SupportedStatement
    consumption: Literal["inherited_statement"]
    receipt: ResultReceipt


class PresentationSection(Contract):
    """Retain complete section prose and all compiler-assigned detail results."""

    group_id: LegacyIdentifier
    group_key: Digest
    number: Positive
    next_id: LegacyIdentifier | None
    prose: SectionResult
    details: tuple[DetailResult, ...]


class V2Presentation(Contract):
    """Explicit reader subtree; structural parsing alone never authorizes it."""

    version: Literal["diffstory.presentation.v2"]
    provenance: Literal["model-generated · unverified"]
    document: DocumentResult
    sections: tuple[PresentationSection, ...]
    context_details: tuple[DetailResult, ...]
    citations: tuple[ResolvedCitation, ...]
    support_provenance: tuple[InheritedProvenance, ...]


GENERATION_STORAGE_VERSION = "diffstory.generation-storage.v1"


def stored_generation(manifest: GenerationManifest) -> dict:
    """Embed a complete proof with one static dependency set per producer.

    The semantic manifest and all its identities remain expanded v2. This
    representation needs no cache files to reconstruct any producer or request.
    """
    value = manifest.model_dump(mode="json")
    value["storage_version"] = GENERATION_STORAGE_VERSION
    for producer, original in zip(
        value["producers"], manifest.producers, strict=True
    ):
        producer["invocation"] = encode_invocation(
            original.invocation
        ).model_dump(mode="json")
    return value


def expanded_generation(value: object) -> dict:
    """Normalize only the declared storage representation, losslessly.

    Retain raw application fields for exact candidate comparison. Do not round
    trip the whole manifest through models: defaults, normalized dependencies,
    or numeric coercion must not hide candidate changes. Compact invocations
    must be the exact encoding of their expanded form, including list order.
    """
    _require(
        type(value) is dict
        and value.get("version") == "diffstory.generation.v2",
        "invalid_generation_schema",
    )
    if "storage_version" not in value:
        return value
    _require(
        value["storage_version"] == GENERATION_STORAGE_VERSION,
        "invalid_generation_storage_version",
    )
    _require(type(value.get("producers")) is list, "invalid_stored_producers")
    producers = []
    for producer in value["producers"]:
        _require(type(producer) is dict, "invalid_stored_producer")
        raw = producer.get("invocation")
        stored = StoredInvocation.model_validate_json(canonical_bytes(raw))
        expanded = decode_invocation(stored, ArtifactInvocation)
        _require(
            canonical_bytes(raw)
            == canonical_bytes(encode_invocation(expanded)),
            "noncanonical_stored_invocation",
        )
        producers.append(
            {**producer, "invocation": expanded.model_dump(mode="json")}
        )
    return {
        **{k: v for k, v in value.items() if k != "storage_version"},
        "producers": producers,
    }


def candidate_storage_semantics(candidate: dict) -> dict:
    """Keep every presentation field while expanding its declared proof storage."""
    return {
        **candidate,
        "generation": expanded_generation(candidate.get("generation")),
    }


def persisted_manifest(value: object) -> GenerationManifest:
    """Parse JSON-shaped data without coercing a legacy declaration into v2."""
    if (
        type(value) is not dict
        or value.get("version") != "diffstory.generation.v2"
    ):
        msg = "invalid_generation_schema"
        raise ValueError(msg)
    return GenerationManifest.model_validate_json(canonical_bytes(value))


def presentation_from_candidate(candidate: dict) -> V2Presentation:
    """Construct a typed reader view from an independently validated assembly."""
    return V2Presentation.model_validate_json(
        canonical_bytes(
            {
                "version": "diffstory.presentation.v2",
                "provenance": "model-generated · unverified",
                "document": candidate["document"],
                "sections": candidate["steps"],
                "context_details": candidate["context_details"],
                "citations": candidate["citations"],
                "support_provenance": candidate["support_provenance"],
            }
        )
    )


def _text(
    statements: Sequence[SupportedStatement | Mapping[str, object]],
) -> str:
    """Join selected statement text with the stable application separator.

    Preconditions:
        Each value is a strict statement model or one selected owned statement
        mapping with a string ``text`` field.
    Postconditions:
        Returns text in input order, separated by two newline characters.
    Essential behavior:
        Shares the exact prose joining rule between expanded and indexed views.
    """
    values = []
    for statement in statements:
        if isinstance(statement, Mapping):
            value = statement.get("text")
        else:
            value = statement.text
        _require(type(value) is str, "invalid_statement_text")
        values.append(value)
    return "\n\n".join(values)


def _document_fields(
    preamble: Sequence[SupportedStatement | Mapping[str, object]],
    lead: Sequence[SupportedStatement | Mapping[str, object]],
    closing: Sequence[SupportedStatement | Mapping[str, object]],
) -> dict[str, str]:
    """Project document fields using the shared application text rule.

    Preconditions:
        Each statement sequence contains models or selected mappings in
        document order.
    Postconditions:
        Returns the exact three applied document fields.
    Essential behavior:
        Preserves empty fields and the same paragraph separator for both paths.
    """
    return {
        "preamble": _text(preamble),
        "lead": _text(lead),
        "closing": _text(closing),
    }


def _group_fields(  # noqa: PLR0913 # Keep exact deterministic group inputs explicit.
    *,
    title: str,
    number: int,
    next_id: str | None,
    change_ids: object,
    orientation: Sequence[SupportedStatement | Mapping[str, object]],
    synthesis: Sequence[SupportedStatement | Mapping[str, object]],
    transition: Sequence[SupportedStatement | Mapping[str, object]],
    questions: Sequence[str],
) -> dict:
    """Project one group with shared order, label, and generated-field rules.

    Preconditions:
        ``title``, group numbers/IDs, compiler change IDs, and selected text
        values have passed their owning record checks.
    Postconditions:
        Returns only the deterministic fields applied to one compiler group.
    Essential behavior:
        Keeps generated provenance, empty fields, statement order, and question
        order identical in expanded and indexed applications.
    """
    return {
        "title": title,
        "number": number,
        "next_id": next_id,
        "narrative": {
            "provenance": "model-generated · unverified",
            "evidence_change_ids": change_ids,
            "intent": _text(orientation),
            "why_now": "",
            "takeaway": _text(synthesis),
            "transition": _text(transition),
            "invariants": [],
            "questions": list(questions),
        },
    }


def _document_view(presentation: V2Presentation) -> dict:
    """Return the deterministic document overlay from an expanded presentation.

    Preconditions:
        ``presentation`` is a complete v2 presentation already accepted by its
        caller's public proof boundary.
    Postconditions:
        Returns only the applied document fields in their original order.
    Essential behavior:
        Delegates paragraph joining to the same helper used by indexed apply.
    """
    document = presentation.document
    return _document_fields(
        document.preamble,
        document.lead,
        document.closing,
    )


def _group_view(section: PresentationSection, group: dict) -> dict:
    """Return the deterministic group overlay from expanded reader records.

    Preconditions:
        ``section`` is an accepted presentation section and ``group`` is its
        current compiler group mapping.
    Postconditions:
        Returns the exact title, numbering, narrative, and question fields.
    Essential behavior:
        Shares all applied prose and label rules with indexed application.
    """
    return _group_fields(
        title=section.prose.title.text,
        number=section.number,
        next_id=section.next_id,
        change_ids=group["change_ids"],
        orientation=section.prose.orientation,
        synthesis=section.prose.synthesis,
        transition=section.prose.transition,
        questions=tuple(
            question.text
            for detail in section.details
            for question in detail.questions
        ),
    )


def indexed_applied_fields(  # noqa: PLR0913, PLR0915 # Select document and group fields in one bounded adapter.
    projection: IndexedReaderProjection,
    *,
    arena: _RecordArena,
    details: IndexedDetailContext,
    sources: IndexedPlanningSources,
    groups: Mapping[str, Mapping[str, object]],
    captured_input: object | None = None,
) -> dict:
    """Return a small deterministic document/group overlay from owned records.

    Preconditions:
        ``projection`` names result rows in ``arena``; ``details`` and
        ``sources`` are the matching live arena-backed detail context; and
        ``groups`` contains current compiler group facts indexed by group ID.
    Postconditions:
        Returns only a document mapping and ordered group overlays. No report,
        presentation, generation, or complete result tree is retained.
    Essential behavior:
        Selects text and questions by path and shares the expanded adapter's
        separator, numbering, ordering, transition, and provenance rules.

    Raises:
        ValueError: If a selected result, group fact, or detail membership is
            missing or rebound.
    """
    from .narrative_hierarchy import IndexedPlanningSources
    from .narrative_hierarchy import _state
    from .narrative_plan import IndexedDetailContext
    from .report_storage import IndexedReaderProjection
    from .report_storage import _RecordArena

    _require(
        type(projection) is IndexedReaderProjection,
        "invalid_indexed_reader_projection",
    )
    _require(type(arena) is _RecordArena, "invalid_record_arena")
    _require(
        type(details) is IndexedDetailContext, "invalid_indexed_detail_context"
    )
    _require(
        type(sources) is IndexedPlanningSources, "invalid_planning_sources"
    )
    _require(isinstance(groups, Mapping), "invalid_compiler_group_facts")
    owned_group_values: set[int] = set()
    if any(
        type(group) is _OwnedMapping
        or type(group.get("change_ids")) in {_OwnedList, _OwnedTuple}
        for group in groups.values()
        if isinstance(group, Mapping)
    ):
        report_view = getattr(captured_input, "report_view", None)
        assert_current = getattr(
            captured_input, "assert_capture_current", None
        )
        owned_groups = (
            report_view.get("groups")
            if isinstance(report_view, Mapping)
            else None
        )
        _require(
            callable(assert_current)
            and type(owned_groups) in {_OwnedList, _OwnedTuple},
            "invalid_compiler_group_facts",
        )
        assert_current()
        owned_group_values = {id(group) for group in owned_groups}
        _require(
            all(
                type(group) is _OwnedMapping
                and id(group) in owned_group_values
                for group in groups.values()
            ),
            "invalid_compiler_group_facts",
        )
    sources.check_current()
    state = _state(sources)
    _require(
        state.details_context is details,
        "planning_sources_detail_context_mismatch",
    )
    detail_reader = details._arena_reader  # noqa: SLF001 # Bind selected detail membership to the owned arena.
    _require(
        detail_reader is not None
        and detail_reader._owner_token is arena._owner_token,  # noqa: SLF001 # Compare exact arena ownership.
        "detail_context_arena_mismatch",
    )
    details_by_unit = {
        reference.item_key: reference for reference in details.references
    }
    _require(
        len(details_by_unit) == len(details.references),
        "detail_coverage",
    )
    reader = arena._snapshot_reader()  # noqa: SLF001 # One shallow selected-row view for this operation.

    group_ids_by_key = {
        binding.group_key: binding.group_id
        for binding in state.index._ledger.group_bindings  # noqa: SLF001 # Current group bindings passed the enclosing source check.
    }
    _require(
        set(group_ids_by_key.values()) == set(groups),
        "compiler_group_coverage_mismatch",
    )
    section_group_ids = tuple(
        section.group_id for section in projection.sections
    )
    _require(
        len(set(section_group_ids)) == len(section_group_ids)
        and set(section_group_ids) == set(groups),
        "reader_section_mismatch",
    )

    document_kind, document_subject, plan_id = reader.result_binding_facts(
        projection.document
    )
    _require(
        document_kind == "document" and document_subject == plan_id,
        "reader_prose_binding_mismatch",
    )

    document_fields: dict[str, list[Mapping[str, object]]] = {}
    document_paths = reader.statement_paths(projection.document)
    for field in ("preamble", "lead", "closing"):
        document_fields[field] = [
            reader.statement_view(projection.document, path)
            for path in document_paths
            if path.field == field
        ]
    document = _document_fields(
        document_fields["preamble"],
        document_fields["lead"],
        document_fields["closing"],
    )

    group_overlays = []
    for position, section in enumerate(projection.sections):
        section_kind, section_subject, section_plan = (
            reader.result_binding_facts(section.prose)
        )
        _require(
            section.group_id == group_ids_by_key.get(section.group_key)
            and section_kind == "section"
            and section_subject == section.group_key
            and section_plan == plan_id
            and section.number == position + 1
            and section.next_id
            == (
                projection.sections[position + 1].group_id
                if position + 1 < len(projection.sections)
                else None
            ),
            "reader_section_mismatch",
        )
        group = groups.get(section.group_id)
        _require(isinstance(group, Mapping), "missing_compiler_group_facts")
        change_ids = group.get("change_ids")
        _require(
            type(change_ids) in {tuple, list, _OwnedList, _OwnedTuple}
            and all(type(value) is str for value in change_ids),
            "invalid_compiler_group_facts",
        )
        if type(change_ids) in {_OwnedList, _OwnedTuple}:
            _require(
                id(group) in owned_group_values,
                "invalid_compiler_group_facts",
            )
            change_ids = list(change_ids)
        section_paths = reader.statement_paths(section.prose)
        title_values = [
            reader.statement_view(section.prose, path)
            for path in section_paths
            if path.field == "title"
        ]
        _require(len(title_values) == 1, "invalid_section_title")
        statement_fields = {
            field: [
                reader.statement_view(section.prose, path)
                for path in section_paths
                if path.field == field
            ]
            for field in ("orientation", "synthesis", "transition")
        }
        expected_units = state.index.groups.get(section.group_key)
        _require(expected_units is not None, "reader_section_mismatch")
        expected_unit_keys = expected_units.unit_keys
        _require(
            tuple(reference.item_key for reference in section.details)
            == expected_unit_keys,
            "reader_detail_mismatch",
        )
        questions = []
        for detail_ref in section.details:
            _require(
                detail_ref.kind == "detail"
                and details_by_unit.get(detail_ref.item_key) == detail_ref,
                "reader_detail_mismatch",
            )
            _digest, _observations, counts = details.planning_detail_facts(
                detail_ref.item_key
            )
            for ordinal in range(counts[2]):
                question = details.indexed_detail_record_view(
                    detail_ref.item_key, "question", ordinal
                )
                value = question.get("text")
                _require(type(value) is str, "invalid_detail_record")
                questions.append(value)
        fields = _group_fields(
            title=title_values[0]["text"],
            number=section.number,
            next_id=section.next_id,
            change_ids=change_ids,
            orientation=statement_fields["orientation"],
            synthesis=statement_fields["synthesis"],
            transition=statement_fields["transition"],
            questions=questions,
        )
        group_overlays.append({"group_id": section.group_id, **fields})
    _require(
        len({item["group_id"] for item in group_overlays})
        == len(group_overlays),
        "duplicate_reader_section",
    )
    return {"document": document, "groups": tuple(group_overlays)}


def _apply_v2_candidate(
    report: dict | _OwnedMapping,
    manifest: GenerationManifest,
    candidate: dict,
) -> dict:
    """Apply already validated deterministic v2 data to an owned report copy.

    This helper performs data assembly only. Callers at public dictionary
    boundaries must validate the complete proof before using it.
    """
    from copy import deepcopy

    presentation = presentation_from_candidate(candidate)
    result = (
        _thaw_owned(report)
        if type(report) is _OwnedMapping
        else deepcopy(report)
    )
    groups = {g["id"]: g for g in result["groups"]}
    result["groups"] = [groups[s.group_id] for s in presentation.sections]
    for section, group in zip(
        presentation.sections, result["groups"], strict=True
    ):
        group.update(_group_view(section, group))
    result["document"] = _document_view(presentation)
    result["narrative_v2"] = presentation.model_dump(mode="json")
    result["generation"] = stored_generation(manifest)
    _require(
        compiler_fact_signature(result) == manifest.compiler_fact_signature,
        "compiler_facts_changed",
    )
    return result


def apply_v2_annotations(report: dict, annotations: dict) -> dict:
    """Validate all offline proofs and attach deterministic typed presentation."""
    manifest = persisted_manifest(annotations.get("generation"))
    candidate = validate_generation(report, manifest, candidate=annotations)
    return _apply_v2_candidate(report, manifest, candidate)


def validate_v2_report(report: dict) -> None:
    """Rebuild current authority and require exact equality of every applied field.

    No cross-call mutable context or trusted persisted catalog is accepted. Each
    standalone boundary builds one index and validates the shared support forest.
    """
    manifest = persisted_manifest(report.get("generation"))
    candidate = validate_generation(report, manifest)
    expected = presentation_from_candidate(candidate)
    # JSON comparison preserves scalar types (True must not equal integer 1).
    _require(
        canonical_bytes(report.get("narrative_v2"))
        == canonical_bytes(expected),
        "presentation_changed",
    )
    _require(
        canonical_bytes(report.get("document"))
        == canonical_bytes(_document_view(expected)),
        "document_changed",
    )
    _require(
        [g["id"] for g in report["groups"]]
        == [s.group_id for s in expected.sections],
        "presentation_order_changed",
    )
    for section, group in zip(
        expected.sections, report["groups"], strict=True
    ):
        actual = {
            name: group.get(name)
            for name in ("title", "number", "next_id", "narrative")
        }
        _require(
            canonical_bytes(actual)
            == canonical_bytes(_group_view(section, group)),
            "section_presentation_changed",
        )


def validate_persisted_generation(report: dict, generation: dict) -> None:
    """Validate JSON generation data and any presentation already applied to it."""
    if "generation" in report or "narrative_v2" in report:
        _require(
            canonical_bytes(expanded_generation(generation))
            == canonical_bytes(expanded_generation(report.get("generation"))),
            "applied_generation_mismatch",
        )
        validate_v2_report(report)
    else:
        validate_generation(report, persisted_manifest(generation))
