"""Generation bindings, trusted production validation and exact v2 assembly.

The data contract alone is not a trust authority. validate_generation rebuilds
current source and checks support, hierarchy and actual request metadata. The
public application and rendering boundaries must call it before using v2 data.
Legacy generation.v1 retains its existing guarantees.
"""

from __future__ import annotations

from bisect import bisect_right
from collections import defaultdict
from typing import TYPE_CHECKING
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
from .narrative_contracts import ResultKind
from .narrative_contracts import SectionResult
from .narrative_contracts import SourceSliceRef
from .narrative_contracts import StoryPlan
from .narrative_contracts import result_key
from .narrative_contracts import result_kind
from .narrative_plan import canonical_bytes
from .narrative_plan import content_digest
from .narrative_plan import semantic_digest
from .narrative_runtime import ArtifactInvocation
from .narrative_runtime import PreparedMetadata
from .narrative_runtime import RunIdentity

if TYPE_CHECKING:
    from collections.abc import Callable

    from .narrative_contracts import EvidenceLedger
    from .narrative_plan import LedgerIndex
    from .narrative_runtime import StoredArtifact
    from .narrative_wire import ProseContext
    from .narrative_wire import WireAdapter


def _require(condition: bool, reason: str) -> None:  # noqa: FBT001 # Assertion predicate.
    if not condition:
        raise ValueError(reason)


def compiler_fact_signature(report: dict) -> str:
    """Hash compiler facts after removing only declared generated presentation.

    Exclude root document/generation and group narrative/title/number/next_id.
    Canonicalize group order by compiler ID. Retain every other field, including
    metadata, revisions, source, classification, prerequisites and test status.
    Never compare an annotated report's full hash to its pre-annotation hash.
    """
    _require(
        type(report) is dict and type(report.get("groups")) is list,
        "invalid_report_shape",
    )
    groups = report["groups"]
    _require(
        all(
            type(group) is dict and type(group.get("id")) is str
            for group in groups
        ),
        "invalid_group_shape",
    )
    _require(
        len({group["id"] for group in groups}) == len(groups),
        "duplicate_group",
    )
    projection = {
        key: value
        for key, value in report.items()
        if key not in {"document", "generation", "groups"}
    }
    projection["groups"] = [
        {
            key: value
            for key, value in group.items()
            if key not in {"narrative", "title", "number", "next_id"}
        }
        for group in sorted(groups, key=lambda group: group["id"])
    ]
    return semantic_digest("diffstory.compiler-facts.v2", projection)


class LedgerParameters(Contract):
    """Declare the exact ledger partition parameters for trusted reconstruction."""

    max_span_bytes: int = Field(ge=4)
    max_unit_spans: int = Field(ge=2)


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
    M4 compares it to a newly derived ledger; internal agreement cannot prove
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
    unit_keys: tuple[Digest, ...]
    details: tuple[DetailResult, ...]
    plan: StoryPlan
    planning: tuple[PlanningCardResult, ...]
    sections: tuple[SectionResult, ...]
    document: DocumentResult
    producers: tuple[ProducerDescriptor, ...] = Field(min_length=1)
    receipts: tuple[ResultReceipt, ...] = Field(min_length=1)

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


def validate_prose_support(
    result: NarrativeResult, context: ProseContext, wire: WireAdapter
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

    require(result.binding == context.binding, "dependencies_changed")
    require(result.item_key == context.binding.key, "dependencies_changed")
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
    if isinstance(result, SectionResult) and context.final:
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
    report: dict,
    manifest: GenerationManifest,
    *,
    ledger: EvidenceLedger | None = None,
    index: LedgerIndex | None = None,
) -> dict:
    """Assemble exact typed prose and all details using current compiler bindings.

    This helper does not apply annotations or render HTML. Public consumers must
    first call validate_production with their trusted provider/options contract.
    Optional ledger/index values reuse the production builder's current index.
    Without them, source is rebuilt from the report and declared parameters.
    """
    from .analysis import source_url
    from .narrative_hierarchy import statement_at
    from .narrative_hierarchy import statement_entries
    from .narrative_plan import build_evidence_ledger
    from .narrative_plan import build_ledger_index
    from .narrative_plan import resolve_span_ref
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
    references = {}
    slices = {}
    provenance = {}
    prose = (*manifest.planning, *manifest.sections, manifest.document)
    producers = {r.item_key: r for r in prose}
    for detail in manifest.details:
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
                "fragment_key": binding.fragment_key,
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
    assigned = {u for g in index.groups.values() for u in g.unit_keys}
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
        "generation": manifest.model_dump(mode="json"),
    }


def validate_generation(
    report: dict,
    manifest: GenerationManifest,
    *,
    provider: object,
    options: object = None,
    candidate: dict | None = None,
) -> None:
    """Rebuild trusted production support, hierarchy, requests and exact assembly."""
    from .narrative_v2 import validate_production

    validate_production(
        report,
        manifest,
        provider=provider,
        options=options,
        candidate=candidate,
    )
