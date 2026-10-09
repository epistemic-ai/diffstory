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


class SourceSliceRef(Contract):
    """Exact UTF-8 bytes relative to one resolved atomic source, never its remainder."""

    kind: Literal["source_slice"] = "source_slice"
    source: SpanRef
    byte_start: Natural
    byte_end: Natural
    content_digest: Digest

    @model_validator(mode="after")
    def bounds(self) -> SourceSliceRef:
        """Reject empty and reversed slices; current-source checks are separate."""
        if self.byte_end <= self.byte_start:
            msg = "Source slice must contain bytes"
            raise ValueError(msg)
        return self


class StatementPath(Contract):
    """Allowlisted statement location; terminology indexes its definition."""

    field: Literal[
        "abstraction",
        "title",
        "orientation",
        "synthesis",
        "transition",
        "preamble",
        "lead",
        "closing",
        "terminology",
    ]
    index: Natural = 0

    @model_validator(mode="after")
    def scalar_title(self) -> StatementPath:
        """A title has one scalar location."""
        if self.field == "title" and self.index != 0:
            msg = "Title index must be zero"
            raise ValueError(msg)
        return self


class StatementRef(Contract):
    """Bind an exact consumed, validated child statement and complete result."""

    kind: Literal["statement"] = "statement"
    producer_item_key: Digest
    result_digest: Digest
    statement_path: StatementPath
    statement_digest: Digest


SupportBasis = Annotated[
    SourceSliceRef | StatementRef, Field(discriminator="kind")
]


class SupportSet(Contract):
    """Only support actually supplied in a request, with separate author provenance."""

    claims: tuple[ClaimRef, ...] = ()
    citations: tuple[SpanRef, ...] = ()
    basis: tuple[SupportBasis, ...] = ()
    author_context_digest: Digest | None = None

    @model_validator(mode="after")
    def unique(self) -> SupportSet:
        """Reject duplicate support handles in each typed family."""
        if any(
            len(set(values)) != len(values)
            for values in (self.claims, self.citations, self.basis)
        ):
            msg = "Duplicate consumed support"
            raise ValueError(msg)
        return self


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
    basis: tuple[SupportBasis, ...] = ()
    effect: Text | None = None
    uncertainty: tuple[Text, ...] = ()
    author_context_digest: Digest | None = None

    @model_validator(mode="after")
    def support(self) -> SupportedStatement:
        """Require evidence for implementation and explicit author attribution."""
        if (
            self.kind == "implementation_claim" or self.effect is not None
        ) and not (self.claims or self.citations or self.basis):
            msg = "Implementation claims and effects require support"
            raise ValueError(msg)
        if (self.kind == "author_motivation") != (
            self.author_context_digest is not None
        ):
            msg = "Author motivation requires an author context digest only"
            raise ValueError(msg)
        if (
            len(set(self.claims)) != len(self.claims)
            or len(set(self.citations)) != len(self.citations)
            or len(set(self.basis)) != len(self.basis)
        ):
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


class AuthorContext(Contract):
    """Current normalized author fields; empty strings remain explicit."""

    title: str = ""
    repository: str = ""
    author_reported_motivation: str = ""


class DetailRecordOwner(Contract):
    """One ordered record in an unchanged validated detail result."""

    kind: Literal["detail"] = "detail"
    unit_key: Digest
    detail_digest: Digest
    record_kind: Literal["detail", "explanation", "observation", "question"]
    ordinal: Natural


class SourceRecordOwner(Contract):
    """Content-stable source identity; coordinates come from the current registry."""

    kind: Literal["source"] = "source"
    unit_key: Digest
    span_id: Digest
    content_digest: Digest


class AuthorRecordOwner(Contract):
    """Author material cannot confer implementation support."""

    kind: Literal["author"] = "author"
    context_digest: Digest


class HeadingRecordOwner(Contract):
    """Immutable compiler path/theme, excluding generated group titles."""

    kind: Literal["heading"] = "heading"
    group_key: Digest


RecordOwner = Annotated[
    DetailRecordOwner
    | SourceRecordOwner
    | AuthorRecordOwner
    | HeadingRecordOwner,
    Field(discriminator="kind"),
]
RecordPath = Literal[
    "version",
    "unit_key",
    "explanations",
    "observations",
    "questions",
    "obligation_ids",
    "before",
    "after",
    "effect",
    "uncertainty",
    "citations",
    "limitations",
    "id",
    "text",
    "reference",
    "title",
    "repository",
    "author_reported_motivation",
    "path",
    "theme",
]
RecordEncoding = Literal["utf8", "canonical_json"]


class RecordFragment(Contract):
    """A contiguous part of an exact field; material is resolved from current data."""

    id: Digest
    owner: RecordOwner
    path: RecordPath
    encoding: RecordEncoding
    byte_start: Natural
    byte_end: Natural
    total_bytes: Natural
    field_digest: Digest
    content_digest: Digest

    @model_validator(mode="after")
    def bounds(self) -> RecordFragment:
        """Only empty fields have empty fragments."""
        if not 0 <= self.byte_start <= self.byte_end <= self.total_bytes or (
            self.byte_start == self.byte_end and self.total_bytes != 0
        ):
            msg = "Invalid record fragment range"
            raise ValueError(msg)
        return self


class CardBudget(Contract):
    """Caps on a generated wire record and its immediate forwarded support."""

    result_wire_bytes: Positive
    support_bytes: Positive


class PlanningBounds(Contract):
    """Frozen planning settings; configured provider output guards never shrink."""

    card_wire_bytes: Positive
    card_support_bytes: Positive
    prose_wire_bytes: Positive
    output_guard: Positive
    max_fragment_bytes: Positive = 4096
    max_chapter_groups: Positive = 32
    min_card_wire_bytes: Positive = 128
    min_card_support_bytes: Positive = 128

    @model_validator(mode="after")
    def minimums(self) -> PlanningBounds:
        """Require declared progress minima at or below configured caps."""
        if (
            self.min_card_wire_bytes > self.card_wire_bytes
            or self.min_card_support_bytes > self.card_support_bytes
        ):
            msg = "Minimum planning budgets exceed configured caps"
            raise ValueError(msg)
        return self


AuxPurpose = Literal[
    "support", "detail", "metadata", "cycle", "section_context"
]


class ReductionRef(Contract):
    """Shared auxiliary input, separate from exclusive reduction-tree children."""

    kind: Literal["reduction"] = "reduction"
    target_key: Digest
    owner_key: Digest
    purpose: AuxPurpose


class ContextSource(Contract):
    """One exact compiler-derived relationship to a primary or cycle root."""

    kind: Literal["group", "global", "cycle"]
    target_key: Digest
    relationship: Literal[
        "own",
        "prerequisite",
        "previous",
        "next",
        "global",
        "cycle",
        "cycle_member",
    ]
    relationship_key: Digest
    target_group_key: Digest | None = None
    source_group_key: Digest | None = None
    cycle_key: Digest | None = None

    @model_validator(mode="after")
    def family(self) -> ContextSource:
        """Keep group, global and cycle relationships in their declared families."""
        if self.kind == "group":
            valid = self.source_group_key is not None and (
                (
                    self.relationship == "cycle_member"
                    and self.cycle_key is not None
                    and self.target_group_key is None
                )
                or (
                    self.relationship
                    in {"own", "prerequisite", "previous", "next"}
                    and self.cycle_key is None
                    and self.target_group_key is not None
                )
            )
            if self.relationship == "own":
                valid = (
                    valid and self.source_group_key == self.target_group_key
                )
        elif self.kind == "global":
            valid = (
                self.relationship == "global"
                and self.source_group_key is None
                and self.cycle_key is None
            )
        else:
            valid = (
                self.relationship == "cycle"
                and self.source_group_key is None
                and self.target_group_key is not None
                and self.cycle_key is not None
            )
        if not valid:
            msg = "Context relationship does not match its family"
            raise ValueError(msg)
        return self


AuxReference = Annotated[
    ReductionRef | ContextSource, Field(discriminator="kind")
]


class AuxCard(Contract):
    """An owner-scoped reduction node with explicit shared input families."""

    id: Digest
    purpose: AuxPurpose
    owner_key: Digest
    fragment_keys: tuple[Digest, ...] = ()
    child_keys: tuple[Digest, ...] = ()
    sources: tuple[AuxReference, ...] = ()
    budget: CardBudget

    @model_validator(mode="after")
    def inputs(self) -> AuxCard:
        """A reducer has only children; a leaf has fragments or typed sources."""
        if (self.child_keys and (self.fragment_keys or self.sources)) or not (
            self.child_keys or self.fragment_keys or self.sources
        ):
            msg = "Invalid auxiliary inputs"
            raise ValueError(msg)
        if any(
            len(set(values)) != len(values)
            for values in (self.fragment_keys, self.child_keys, self.sources)
        ):
            msg = "Duplicate auxiliary inputs"
            raise ValueError(msg)
        return self


class MaterialProjection(Contract):
    """Complete inline records or one declared reduction root."""

    record_keys: tuple[Digest, ...] = ()
    root: ReductionRef | None = None

    @model_validator(mode="after")
    def exclusive(self) -> MaterialProjection:
        """Reduced material must not repeat its unbounded inline record list."""
        if (self.root is not None and self.record_keys) or len(
            set(self.record_keys)
        ) != len(self.record_keys):
            msg = "Invalid material projection"
            raise ValueError(msg)
        return self


class AtomProjection(Contract):
    """Bind an unchanged primary atom to its complete or reduced current material."""

    atom_key: Digest
    material: MaterialProjection


class GroupRoot(Contract):
    """An exact primary subtree for one compiler group."""

    group_key: Digest
    card_key: Digest


class CycleRoot(Contract):
    """One shared context forest for an exact compiler SCC."""

    cycle_key: Digest
    group_keys: tuple[Digest, ...] = Field(min_length=1)
    root: ReductionRef


class SectionView(Contract):
    """Full relationship authority plus an optional bounded provider projection."""

    group_key: Digest
    position: Natural
    previous_group_key: Digest | None
    next_group_key: Digest | None
    chapter_key: Digest
    cycle_key: Digest | None
    final: bool
    sources: tuple[ContextSource, ...]
    context_root: ReductionRef | None = None


class ProductionHierarchy(Contract):
    """Frozen production structure; validation requires current source derivation."""

    version: Literal["diffstory.production-hierarchy.v1"] = (
        "diffstory.production-hierarchy.v1"
    )
    bounds: PlanningBounds
    card_budget: CardBudget
    fragments: tuple[RecordFragment, ...]
    auxiliary_cards: tuple[AuxCard, ...]
    atom_projections: tuple[AtomProjection, ...]
    author_projection: MaterialProjection
    group_roots: tuple[GroupRoot, ...]
    cycle_roots: tuple[CycleRoot, ...]
    section_views: tuple[SectionView, ...]


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
    production: ProductionHierarchy | None = None


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
