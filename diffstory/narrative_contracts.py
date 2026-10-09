"""Immutable narration v2 values, independent of compiler and provider modules.

Byte bounds are half-open. SourceSpan coordinates refer to a current fragment;
SpanRef coordinates refer to the cited span itself. Semantic handles exclude
revision IDs and current fragment coordinates. JSON arrays become tuples through
Pydantic's strict JSON parser; Python callers must supply tuples themselves.
"""

from __future__ import annotations

import hashlib
import json
import re
from typing import Annotated
from typing import Literal

from pydantic import AfterValidator
from pydantic import BaseModel
from pydantic import ConfigDict
from pydantic import Field
from pydantic import model_validator


def _text(value: str) -> str:
    if not value.strip():
        msg = "Text must contain a non-whitespace character"
        raise ValueError(msg)
    return value


def _path(value: str) -> str:
    if (
        not value
        or "\0" in value
        or "\\" in value
        or value.startswith("/")
        or re.match(r"^[A-Za-z]:", value)
        or any(part in {"", ".", ".."} for part in value.split("/"))
    ):
        msg = "Path must be repository relative"
        raise ValueError(msg)
    return value


Digest = Annotated[
    str, Field(min_length=64, max_length=64, pattern=r"^[0-9a-f]{64}$")
]
Text = Annotated[str, AfterValidator(_text)]
RepositoryPath = Annotated[str, AfterValidator(_path)]
Natural = Annotated[int, Field(ge=0)]
Positive = Annotated[int, Field(ge=1)]
Side = Literal["base", "head"]
BatchStage = Literal["detail", "planning", "section", "document"]
ProseKind = Literal["planning_card", "section", "document"]
ResultKind = Literal[
    "detail", "story_plan", "planning_card", "section", "document"
]


class Contract(BaseModel):
    """Reject coercion, unknown fields, and nested mutation."""

    model_config = ConfigDict(
        strict=True,
        frozen=True,
        extra="forbid",
        revalidate_instances="always",
        hide_input_in_errors=True,
    )


class SourceSpan(Contract):
    """One UTF-8-safe part of a line in a current source buffer."""

    version: Literal["diffstory.source-span.v2"] = "diffstory.source-span.v2"
    id: Digest
    buffer_key: Digest
    side: Side
    path: RepositoryPath
    byte_start: Natural
    byte_end: Natural
    line_start: Positive
    line_end: Positive
    content_digest: Digest
    text: str

    @model_validator(mode="after")
    def ranges(self) -> SourceSpan:
        """Require ordered bounds and an exact byte count."""
        if self.byte_end - self.byte_start != len(self.text.encode("utf-8")):
            msg = "Span byte range does not match its text"
            raise ValueError(msg)
        if self.line_end < self.line_start:
            msg = "Span line range is reversed"
            raise ValueError(msg)
        return self


class SourceBuffer(Contract):
    """Account for an entire supplied buffer, including an empty one."""

    key: Digest
    side: Side
    path: RepositoryPath
    content_digest: Digest
    byte_length: Natural
    span_ids: tuple[Digest, ...]
    coverage: Literal["full", "partial"]
    parse_status: Literal["ok", "text_only", "failed"]


class BufferBinding(Contract):
    """Resolve a semantic buffer to its current compiler fragment."""

    buffer_key: Digest
    fragment_key: Text
    start_line: Positive
    region: Text


class FileAvailability(Contract):
    """Carry compiler file facts without inferring absence from missing text."""

    side: Side
    path: RepositoryPath
    counterpart_path: RepositoryPath
    state: Literal["supplied", "absent", "unavailable"]
    coverage: Literal["full", "partial"] | None
    reason: (
        Literal[
            "not_supplied",
            "size_limit",
            "binary_or_non_utf8",
            "unsupported_object",
            "invalid_content_size",
            "unsupported_encoding",
            "read_failed",
        ]
        | None
    )
    full_source: bool
    full_and_parsed: bool

    @model_validator(mode="after")
    def state_fields(self) -> FileAvailability:
        """Keep absent, unavailable, and supplied states distinct."""
        if (self.state == "supplied") != (self.coverage is not None):
            msg = "Coverage must describe supplied source only"
            raise ValueError(msg)
        if (self.state == "unavailable") != (self.reason is not None):
            msg = "Reason must describe unavailable source only"
            raise ValueError(msg)
        if self.full_and_parsed and not self.full_source:
            msg = "Parsed complete source requires complete source"
            raise ValueError(msg)
        if self.full_source != (self.coverage == "full"):
            msg = "Complete source must have full coverage"
            raise ValueError(msg)
        return self


class EvidenceLimitation(Contract):
    """A structured bound on what supplied evidence can establish."""

    code: Literal[
        "unavailable",
        "partial",
        "unpaired_excerpt",
        "unmatched_definition",
        "parse_failed",
        "text_only",
        "unclassified_changed_source",
    ]
    side: Side
    path: RepositoryPath
    reason: Text


class Classification(Contract):
    """Preserve compiler wording and unresolved classifications verbatim."""

    kind: Literal[
        "moved",
        "renamed",
        "moved_renamed",
        "moved_modified",
        "modified",
        "source_only",
        "added",
        "removed",
        "observed_head",
        "observed_base",
        "wiring",
        "text",
        "test",
    ]
    label: Text
    basis: Text
    uncertainty: tuple[Text, ...]
    before_name: str | None
    after_name: str | None
    before_is_test: bool | None
    after_is_test: bool | None


class Obligation(Contract):
    """One bounded source change, structural change, or evidence limitation."""

    id: Digest
    kind: Literal["changed_region", "structure", "limitation"]
    change_key: Digest | None
    required_span_ids: tuple[Digest, ...]
    limitations: tuple[EvidenceLimitation, ...]

    @model_validator(mode="after")
    def support(self) -> Obligation:
        """Require source support or an explicit limitation."""
        if not self.required_span_ids and not self.limitations:
            msg = "Obligation has neither source nor limitation"
            raise ValueError(msg)
        if len(set(self.required_span_ids)) != len(self.required_span_ids):
            msg = "Duplicate obligation spans"
            raise ValueError(msg)
        if self.kind == "limitation" and not self.limitations:
            msg = "Limitation obligation needs a limitation"
            raise ValueError(msg)
        return self


class EvidenceUnit(Contract):
    """A bounded request item; presentation groups do not set request bounds."""

    version: Literal["diffstory.evidence-unit.v2"] = (
        "diffstory.evidence-unit.v2"
    )
    id: Digest
    group_key: Digest | None
    change_key: Digest | None
    classification: Classification | None
    availability: tuple[FileAvailability, ...] = ()
    spans: tuple[SourceSpan, ...]
    obligations: tuple[Obligation, ...]

    @model_validator(mode="after")
    def ownership(self) -> EvidenceUnit:
        """Require unique owned spans and obligations."""
        spans = {span.id for span in self.spans}
        if len(spans) != len(self.spans):
            msg = "Duplicate unit spans"
            raise ValueError(msg)
        if len({item.id for item in self.obligations}) != len(
            self.obligations
        ):
            msg = "Duplicate unit obligations"
            raise ValueError(msg)
        if any(
            not set(item.required_span_ids) <= spans
            for item in self.obligations
        ):
            msg = "Obligation cites a span outside its unit"
            raise ValueError(msg)
        if any(
            item.change_key != self.change_key for item in self.obligations
        ):
            msg = "Obligation belongs to another change"
            raise ValueError(msg)
        return self


class ChangeBinding(Contract):
    """Bind a content change handle to a current report ID."""

    change_key: Digest
    change_id: Text
    group_key: Digest


class GroupBinding(Contract):
    """Bind a stable presentation group to a current report ID."""

    group_key: Digest
    group_id: Text


class SpanOwnership(Contract):
    """List every dispatch owner; multiple owners explicitly share context."""

    span_id: Digest
    unit_keys: tuple[Digest, ...]
    shared: bool


class GroupSpec(Contract):
    """Compiler group membership and all prerequisite edges."""

    key: Digest
    path: RepositoryPath
    theme: Text
    unit_keys: tuple[Digest, ...]
    prerequisites: tuple[Digest, ...]


class EvidenceLedger(Contract):
    """Separate gap-free source accounting from explanation obligations."""

    version: Literal["diffstory.evidence-ledger.v2"] = (
        "diffstory.evidence-ledger.v2"
    )
    id: Digest
    buffers: tuple[SourceBuffer, ...]
    bindings: tuple[BufferBinding, ...]
    spans: tuple[SourceSpan, ...]
    units: tuple[EvidenceUnit, ...]
    ownership: tuple[SpanOwnership, ...]
    availability: tuple[FileAvailability, ...]
    changes: tuple[ChangeBinding, ...]
    groups: tuple[GroupSpec, ...]
    group_bindings: tuple[GroupBinding, ...]
    compiler_test_runs: Natural


class SpanRef(Contract):
    """A stable citation with span-relative byte and inclusive line bounds.

    Full atomic spans are cited. resolve_span_ref supplies the current fragment
    coordinates. This permits reuse when unrelated text shifts source lines.
    """

    span_id: Digest
    unit_key: Digest
    group_key: Digest | None
    side: Side
    path: RepositoryPath
    byte_start: Natural
    byte_end: Natural
    line_start: Positive
    line_end: Positive
    content_digest: Digest

    @model_validator(mode="after")
    def ranges(self) -> SpanRef:
        """Require ordered citation bounds."""
        if self.byte_end < self.byte_start or self.line_end < self.line_start:
            msg = "Citation range is reversed"
            raise ValueError(msg)
        return self


class Explanation(Contract):
    """Explain a bounded obligation with source support and stated limits."""

    obligation_ids: tuple[Digest, ...] = Field(min_length=1)
    before: Text
    after: Text
    effect: Text
    uncertainty: Text
    citations: tuple[SpanRef, ...]
    limitations: tuple[EvidenceLimitation, ...]


class FactualObservation(Contract):
    """A reusable factual statement whose citations remain explicit."""

    id: Digest
    text: Text
    citations: tuple[SpanRef, ...] = Field(min_length=1)


class EvidenceQuestion(Contract):
    """An unresolved question tied to evidence or a named limitation."""

    text: Text
    citations: tuple[SpanRef, ...]
    obligation_ids: tuple[Digest, ...]


class DetailResult(Contract):
    """Detailed evidence writing, separate from later planning abstractions."""

    version: Literal["diffstory.detail-result.v2"] = (
        "diffstory.detail-result.v2"
    )
    unit_key: Digest
    explanations: tuple[Explanation, ...]
    observations: tuple[FactualObservation, ...]
    questions: tuple[EvidenceQuestion, ...]


class ClaimRef(Contract):
    """Refer to a validated detail observation without replacing its text."""

    unit_key: Digest
    observation_id: Digest


class Terminology(Contract):
    """Retain a complete term definition and its supporting claims."""

    term: Text
    definition: Text
    claims: tuple[ClaimRef, ...]


class SupportedStatement(Contract):
    """Retain prose and its support; consumers resolve refs in the current ledger.

    Author motivation refers to supplied author context, not source proof.
    Editorial text and uncertainty cannot establish implementation behavior.
    Support shape alone does not prove that the text follows from the evidence.
    """

    kind: Literal[
        "implementation_claim", "author_motivation", "editorial", "uncertainty"
    ]
    text: Text
    claims: tuple[ClaimRef, ...] = ()
    citations: tuple[SpanRef, ...] = ()
    effect: Text | None = None
    uncertainty: tuple[Text, ...] = ()
    author_context_digest: Digest | None = None

    @model_validator(mode="after")
    def support(self) -> SupportedStatement:
        """Require evidence for implementation and explicit author attribution."""
        if (
            self.kind == "implementation_claim" or self.effect is not None
        ) and not (self.claims or self.citations):
            msg = "Implementation claims and effects require support"
            raise ValueError(msg)
        if (self.kind == "author_motivation") != (
            self.author_context_digest is not None
        ):
            msg = "Author motivation requires an author context digest only"
            raise ValueError(msg)
        if len(set(self.claims)) != len(self.claims) or len(
            set(self.citations)
        ) != len(self.citations):
            msg = "Duplicate statement support"
            raise ValueError(msg)
        return self


class SupportedTerminology(Contract):
    """Carry proposed terminology with a source-supported definition."""

    term: Text
    definition: SupportedStatement

    @model_validator(mode="after")
    def support(self) -> SupportedTerminology:
        """Do not treat editorial definitions as validated implementation terms."""
        if self.definition.kind != "implementation_claim":
            msg = "Proposed terminology requires an implementation claim"
            raise ValueError(msg)
        return self


class ProseBinding(Contract):
    """Content identity supplied by the consumer for a card, group, or document.

    subject_key is the trusted card/group key, or plan_key for a document.
    context_digest covers consumed content and boundary facts, including actual
    child prose. It is not a digest of the result. Original batch dependencies
    remain in ArtifactInvocation; this item key does not assert independence.
    """

    version: Literal["diffstory.prose-binding.v2"] = (
        "diffstory.prose-binding.v2"
    )
    kind: ProseKind
    subject_key: Digest
    plan_key: Digest
    context_digest: Digest

    @model_validator(mode="after")
    def document_subject(self) -> ProseBinding:
        """A document belongs to its complete deterministic plan."""
        if self.kind == "document" and self.subject_key != self.plan_key:
            msg = "Document subject must be its plan"
            raise ValueError(msg)
        return self

    @property
    def key(self) -> str:
        """Hash the versioned trusted context, independent of generated text."""
        return hashlib.sha256(
            json.dumps(
                self.model_dump(mode="json"),
                sort_keys=True,
                separators=(",", ":"),
            ).encode()
        ).hexdigest()


class _ProseResult(Contract):
    item_key: Digest
    binding: ProseBinding
    terminology: tuple[SupportedTerminology, ...] = ()

    @model_validator(mode="after")
    def context_key(self) -> _ProseResult:
        if self.item_key != self.binding.key or type(self) is not result_model(
            self.binding.kind
        ):
            msg = "Prose result does not match its context or family"
            raise ValueError(msg)
        if len({term.term for term in self.terminology}) != len(
            self.terminology
        ):
            msg = "Duplicate proposed terminology"
            raise ValueError(msg)
        return self


class PlanningCardResult(_ProseResult):
    """Supported planning abstraction; never a substitute for DetailResult."""

    version: Literal["diffstory.planning-card-result.v2"] = (
        "diffstory.planning-card-result.v2"
    )
    abstraction: tuple[SupportedStatement, ...] = Field(min_length=1)


class SectionResult(_ProseResult):
    """Prose for one group; an empty tuple explicitly represents no transition.

    The consumer checks that the actual final section has an empty transition.
    Its trusted binding includes neighbors, order, terms, and prerequisites.
    """

    version: Literal["diffstory.section-result.v2"] = (
        "diffstory.section-result.v2"
    )
    title: SupportedStatement
    orientation: tuple[SupportedStatement, ...] = Field(min_length=1)
    synthesis: tuple[SupportedStatement, ...] = Field(min_length=1)
    transition: tuple[SupportedStatement, ...]


class DocumentResult(_ProseResult):
    """Whole-document prose using the same deterministic plan and support."""

    version: Literal["diffstory.document-result.v2"] = (
        "diffstory.document-result.v2"
    )
    preamble: tuple[SupportedStatement, ...] = Field(min_length=1)
    lead: tuple[SupportedStatement, ...] = Field(min_length=1)
    closing: tuple[SupportedStatement, ...] = Field(min_length=1)


class PlanningGroup(Contract):
    """Complete compiler membership and reusable planning facts for a group."""

    group: GroupSpec
    claims: tuple[ClaimRef, ...]
    terminology: tuple[Terminology, ...]


class PlanningAtom(Contract):
    """An indivisible planning record; large groups split between records."""

    group_key: Digest
    kind: Literal["member", "unit", "prerequisite", "claim", "term"]
    key: Digest | None
    claim: ClaimRef | None
    term: Terminology | None
    path: RepositoryPath | None = None
    theme: Text | None = None

    @model_validator(mode="after")
    def payload(self) -> PlanningAtom:
        """Require exactly the payload named by the atom kind."""
        expected = (
            self.kind in {"unit", "prerequisite"},
            self.kind == "claim",
            self.kind == "term",
        )
        if expected != (
            self.key is not None,
            self.claim is not None,
            self.term is not None,
        ):
            msg = "Planning atom payload does not match its kind"
            raise ValueError(msg)
        if (self.kind == "member") != (
            self.path is not None and self.theme is not None
        ):
            msg = "Only group membership atoms carry a path and theme"
            raise ValueError(msg)
        if self.kind != "member" and (
            self.path is not None or self.theme is not None
        ):
            msg = "Non-membership atoms cannot carry a partial heading"
            raise ValueError(msg)
        return self


class StoryCard(Contract):
    """A leaf of complete atoms or a parent of complete child handles."""

    id: Digest
    level: Natural
    atoms: tuple[PlanningAtom, ...]
    child_keys: tuple[Digest, ...]

    @model_validator(mode="after")
    def shape(self) -> StoryCard:
        """Keep leaves and hierarchy references separate."""
        if self.level == 0:
            if not self.atoms or self.child_keys:
                msg = "Leaf card requires atoms only"
                raise ValueError(msg)
        elif self.atoms or not self.child_keys:
            msg = "Parent card requires child handles only"
            raise ValueError(msg)
        return self


class Prerequisite(Contract):
    """One complete compiler prerequisite edge, including cycle edges."""

    group_key: Digest
    prerequisite_key: Digest


class StoryChapter(Contract):
    """A contiguous slice of a valid story order."""

    id: Digest
    group_keys: tuple[Digest, ...] = Field(min_length=1)
    cross_chapter_prerequisites: tuple[Prerequisite, ...]


class BatchSpec(Contract):
    """An actual serialized request bound and its explicit output estimates."""

    version: Literal["diffstory.batch-spec.v2"] = "diffstory.batch-spec.v2"
    id: Digest
    stage: BatchStage
    item_keys: tuple[Digest, ...] = Field(min_length=1)
    group_keys: tuple[Digest, ...]
    request_digest: Digest
    input_bound: Natural
    output_reservation: Positive
    output_guard: Positive
    estimated_response_bytes: Positive
    estimated_seconds: Positive
    partition: Natural

    @model_validator(mode="after")
    def bounds_and_keys(self) -> BatchSpec:
        """Reject repeated items and an output guard below the reservation."""
        if len(set(self.item_keys)) != len(self.item_keys) or len(
            set(self.group_keys)
        ) != len(self.group_keys):
            msg = "Batch keys must be unique"
            raise ValueError(msg)
        if self.output_guard < self.output_reservation:
            msg = "Output guard cannot be smaller than the reservation"
            raise ValueError(msg)
        return self


class StoryPlan(Contract):
    """Validated order and a lossless, bounded planning hierarchy."""

    version: Literal["diffstory.story-plan.v2"] = "diffstory.story-plan.v2"
    id: Digest
    context_digest: Digest
    groups: tuple[PlanningGroup, ...]
    ordered_group_keys: tuple[Digest, ...]
    cycles: tuple[tuple[Digest, ...], ...]
    chapters: tuple[StoryChapter, ...]
    cards: tuple[StoryCard, ...]
    root_card_keys: tuple[Digest, ...]
    requests: tuple[BatchSpec, ...]


NarrativeResult = (
    DetailResult
    | StoryPlan
    | PlanningCardResult
    | SectionResult
    | DocumentResult
)


def result_model(kind: ResultKind) -> type[Contract]:
    """Select an explicit family; unknown kinds never fall through to a plan."""
    return {
        "detail": DetailResult,
        "story_plan": StoryPlan,
        "planning_card": PlanningCardResult,
        "section": SectionResult,
        "document": DocumentResult,
    }[kind]


def result_key_field(kind: ResultKind) -> str:
    """Return the wire key field for a known result family."""
    result_model(kind)
    return {"detail": "unit_key", "story_plan": "id"}.get(kind, "item_key")


def result_kind(result: NarrativeResult) -> ResultKind:
    """Return the exact typed family without duck typing untrusted payloads."""
    for kind in (
        "detail",
        "story_plan",
        "planning_card",
        "section",
        "document",
    ):
        if type(result) is result_model(kind):
            return kind
    msg = "Unsupported narrative result type"
    raise ValueError(msg)


def result_key(result: NarrativeResult) -> str:
    """Get a result key through the shared family selector."""
    return getattr(result, result_key_field(result_kind(result)))


def result_stage(kind: ResultKind) -> BatchStage:
    """Map every result family to its actual provider stage."""
    return {
        "detail": "detail",
        "story_plan": "planning",
        "planning_card": "planning",
        "section": "section",
        "document": "document",
    }[kind]


class ItemFailure(Contract):
    """Allowlisted item failure; never persist arbitrary exception text."""

    item_key: Digest
    reason: Literal[
        "missing_item",
        "invalid_contract",
        "invalid_citation",
        "incomplete_coverage",
        "input_limit",
        "output_limit",
        "response_limit",
        "timeout_limit",
        "invalid_envelope",
        "hierarchy_cannot_reduce",
    ]


class BatchResult(Contract):
    """A typed envelope; consumers validate each result independently."""

    version: Literal["diffstory.batch-result.v2"] = "diffstory.batch-result.v2"
    batch_key: Digest
    items: tuple[DetailResult, ...]
    failures: tuple[ItemFailure, ...]

    @model_validator(mode="after")
    def unique_keys(self) -> BatchResult:
        """Reject duplicate or overlapping success and failure keys."""
        keys = [item.unit_key for item in self.items]
        keys.extend(item.item_key for item in self.failures)
        if len(set(keys)) != len(keys):
            msg = "Duplicate batch item key"
            raise ValueError(msg)
        return self


class ArtifactDependency(Contract):
    """A named semantic dependency, including each co-batched peer."""

    kind: Literal[
        "evidence",
        "peer",
        "context",
        "prompt",
        "schema",
        "validator",
        "provider",
        "model",
        "configuration",
        "order",
        "terminology",
    ]
    key: Digest
    digest: Digest


class ArtifactRecord(Contract):
    """Typed validated payload with explicit artifact dependencies for M2."""

    version: Literal["diffstory.artifact.v2"] = "diffstory.artifact.v2"
    semantic_key: Digest
    dependencies: tuple[ArtifactDependency, ...]
    original_request_digest: Digest
    validator_version: Text
    result: NarrativeResult

    @model_validator(mode="after")
    def unique_dependencies(self) -> ArtifactRecord:
        """Require one digest per named semantic dependency."""
        keys = [
            (dependency.kind, dependency.key)
            for dependency in self.dependencies
        ]
        if len(set(keys)) != len(keys):
            msg = "Artifact dependencies must be unique"
            raise ValueError(msg)
        return self


def structured_output_schema(model: type[Contract]) -> dict:
    """Return a provider schema with required fields and no object extras.

    Null remains a valid value for optional data; fields cannot be omitted.
    Fixed-size Python tuples are avoided in these contracts. Local validators
    remain necessary even when a provider accepts this JSON schema.
    """
    schema = model.model_json_schema()
    pending = [schema]
    while pending:
        node = pending.pop()
        if isinstance(node, dict):
            node.pop("default", None)
            if node.get("type") == "object":
                node["additionalProperties"] = False
                node["required"] = list(node.get("properties", {}))
            pending.extend(node.values())
        elif isinstance(node, list):
            pending.extend(node)
    return schema
