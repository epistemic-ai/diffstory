"""Compact narration transport with strict framing and local support expansion.

Aliases identify content, never current revisions or source coordinates. The
consumer supplies the trusted binding and allowed support for every prose item.
An invalid envelope yields no items. Valid envelopes retain independent errors.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from dataclasses import field
from types import MappingProxyType
from typing import TYPE_CHECKING
from typing import Literal

from pydantic import Field

from .narrative_contracts import ClaimRef
from .narrative_contracts import Contract
from .narrative_contracts import DetailResult
from .narrative_contracts import EvidenceQuestion
from .narrative_contracts import Explanation
from .narrative_contracts import FactualObservation
from .narrative_contracts import ItemFailure
from .narrative_contracts import SourceSliceRef
from .narrative_contracts import StatementRef
from .narrative_contracts import SupportedStatement
from .narrative_contracts import SupportedTerminology
from .narrative_contracts import SupportSet
from .narrative_contracts import Text
from .narrative_contracts import result_model
from .narrative_contracts import structured_output_schema
from .narrative_hierarchy import IndexedConsumedProse
from .narrative_hierarchy import IndexedPlanningSources
from .narrative_hierarchy import IndexedProseResult
from .narrative_hierarchy import PlanningSource
from .narrative_hierarchy import ProseResult
from .narrative_plan import canonical_bytes
from .narrative_plan import semantic_digest
from .narrative_plan import span_reference
from .narrative_runtime import ArtifactRuntimeError
from .narrative_runtime import WireEnvelope

if TYPE_CHECKING:
    from collections.abc import Mapping

    from .narrative_contracts import BatchSpec
    from .narrative_contracts import EvidenceUnit
    from .narrative_contracts import ProseBinding
    from .narrative_contracts import SpanRef
    from .narrative_plan import LedgerIndex

WIRE_VERSION = "diffstory.wire.v2"


def _unique_object(pairs: list[tuple[str, object]]) -> dict:
    if len({key for key, _ in pairs}) != len(pairs):
        msg = "malformed_json"
        raise ArtifactRuntimeError(msg)
    return dict(pairs)


def _invalid_constant(_value: str) -> None:
    msg = "malformed_json"
    raise ArtifactRuntimeError(msg)


def strict_json(raw: bytes | str) -> object:
    """Reject duplicate properties, invalid constants and invalid UTF-8 early."""
    try:
        text = raw.decode("utf-8") if type(raw) is bytes else raw
        # Check lone surrogates before a parser can retain invalid Unicode.
        text.encode("utf-8")
        result = json.loads(
            text,
            object_pairs_hook=_unique_object,
            parse_constant=_invalid_constant,
        )
        json.dumps(result, ensure_ascii=False, allow_nan=False).encode("utf-8")
    except (ValueError, UnicodeError, TypeError):
        msg = "malformed_json"
        raise ArtifactRuntimeError(msg) from None
    else:
        return result


class Aliases:
    """Map stable short names to canonical identities with collision checks."""

    def __init__(self) -> None:
        """Start one local namespace shared by all stages of a run."""
        self._forward: dict[tuple[str, str], str] = {}
        self._reverse: dict[str, tuple[str, str]] = {}

    def add(self, kind: str, key: str) -> str:
        """Return a deterministic alias and reject any conflicting identity."""
        identity = (kind, key)
        alias = kind + semantic_digest("diffstory.alias.v2", *identity)[:20]
        old = self._reverse.get(alias)
        if old is not None and old != identity:
            msg = "schema"
            raise ArtifactRuntimeError(msg)
        self._reverse[alias] = identity
        self._forward[identity] = alias
        return alias

    def resolve(self, kind: str, alias: object) -> str:
        """Resolve only a declared name of the expected kind."""
        if type(alias) is not str:
            msg = "citation"
            raise ArtifactRuntimeError(msg)
        identity = self._reverse.get(alias)
        if identity is None or identity[0] != kind:
            msg = "citation"
            raise ArtifactRuntimeError(msg)
        return identity[1]


class WireExplanation(Contract):
    """Describe each obligation and cite only supplied short handles."""

    obligations: tuple[str, ...] = Field(min_length=1)
    before: Text
    after: Text
    effect: Text
    uncertainty: Text
    citations: tuple[str, ...]
    limitations: tuple[str, ...]


class WireObservation(Contract):
    """Let the consumer derive an observation ID from supported content."""

    text: Text
    citations: tuple[str, ...] = Field(min_length=1)


class WireQuestion(Contract):
    """Retain a question with explicit source or obligation support."""

    text: Text
    citations: tuple[str, ...]
    obligations: tuple[str, ...]


class WireDetail(Contract):
    """One independently validated detail item."""

    target: str
    explanations: tuple[WireExplanation, ...]
    observations: tuple[WireObservation, ...]
    questions: tuple[WireQuestion, ...]


class WireStatement(Contract):
    """Author context is bound locally; models never construct context hashes."""

    kind: Literal[
        "implementation_claim", "author_motivation", "editorial", "uncertainty"
    ]
    text: Text
    claims: tuple[str, ...]
    citations: tuple[str, ...]
    basis: tuple[str, ...] = ()
    effect: Text | None
    uncertainty: tuple[Text, ...]


class WireTerm(Contract):
    """Retain a supported proposed term without changing the compiler plan."""

    term: Text
    definition: WireStatement


class WireCard(Contract):
    """A bounded supported abstraction of complete planning input."""

    target: str
    abstraction: tuple[WireStatement, ...] = Field(min_length=1)
    terminology: tuple[WireTerm, ...]


class WireSection(Contract):
    """One section in the supplied immutable order."""

    target: str
    title: WireStatement
    orientation: tuple[WireStatement, ...] = Field(min_length=1)
    synthesis: tuple[WireStatement, ...] = Field(min_length=1)
    transition: tuple[WireStatement, ...]
    terminology: tuple[WireTerm, ...]


class WireDocument(Contract):
    """Global prose grounded in the same plan."""

    target: str
    preamble: tuple[WireStatement, ...] = Field(min_length=1)
    lead: tuple[WireStatement, ...] = Field(min_length=1)
    closing: tuple[WireStatement, ...] = Field(min_length=1)
    terminology: tuple[WireTerm, ...]


class DetailEnvelope(Contract):
    """Explicit acknowledgements apply only to units with no obligations."""

    version: Literal["diffstory.wire.v2"]
    items: tuple[WireDetail, ...]
    acknowledgements: tuple[str, ...]


class CardEnvelope(Contract):
    """Wire schema for planning outputs."""

    version: Literal["diffstory.wire.v2"]
    items: tuple[WireCard, ...]


class SectionEnvelope(Contract):
    """Wire schema for adjacent sections."""

    version: Literal["diffstory.wire.v2"]
    items: tuple[WireSection, ...]


class DocumentEnvelope(Contract):
    """Wire schema for document prose."""

    version: Literal["diffstory.wire.v2"]
    items: tuple[WireDocument, ...]


_WIRE_SCHEMA_TEMPLATES = MappingProxyType(
    {
        stage: json.dumps(
            structured_output_schema(model),
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
        ).encode("utf-8")
        for stage, model in (
            ("detail", DetailEnvelope),
            ("planning", CardEnvelope),
            ("section", SectionEnvelope),
            ("document", DocumentEnvelope),
        )
    }
)


def wire_schema(stage: str) -> dict:
    """Return a detached strict schema for one fixed output envelope.

    Args:
        stage: One of ``detail``, ``planning``, ``section``, or ``document``.

    Returns:
        A fresh dictionary decoded from the immutable byte template built once
        when this module loads. Templates remain fixed for the module lifetime,
        and caller mutations cannot affect later results.

    Raises:
        KeyError: If ``stage`` is not supported.
    """
    return json.loads(_WIRE_SCHEMA_TEMPLATES[stage])


@dataclass(frozen=True)
class ProseContext:
    """Trusted allowed support from exactly the supplied item context.

    ``consumed`` may retain the compatibility mapping of expanded producer
    models or an ``IndexedConsumedProse`` selector. The latter owns only arena
    references and small digests, and remains usable only while its exact
    indexed source and citation scope stay live. Neither form is complete
    production proof by itself.

    Attributes:
        binding: Expected item kind, subject, plan, and consumed-context digest.
        claims: Whole observation claims granted to this request.
        citations: Whole source references granted to this request.
        author_digest: Current author-context digest, or ``None``.
        final: Whether the item is the final item in its ordered family.
        max_wire_bytes: Maximum encoded prose-result bytes; default 4,096.
        support: Exact immediate claims, citations, slices, statements, and
            author context supplied to the item.
        consumed: Expanded producer mapping or one immutable arena selector.
        sources: Current legacy or indexed planning source context, if present.
        max_support_bytes: Maximum serialized support bytes; default 16,384.
    """

    binding: ProseBinding
    claims: tuple[ClaimRef, ...]
    citations: tuple[SpanRef, ...]
    author_digest: str | None
    final: bool = False
    max_wire_bytes: int = 4096
    support: SupportSet = field(default_factory=SupportSet)
    consumed: (
        Mapping[str, ProseResult | IndexedProseResult] | IndexedConsumedProse
    ) = field(default_factory=lambda: MappingProxyType({}))
    sources: PlanningSource | None = None
    max_support_bytes: int = 16384


class WireAdapter:
    """Expand compact provider output against one trusted current index."""

    def __init__(self, index: LedgerIndex) -> None:
        """Register complete detail support before any provider request."""
        self.index = index
        self.aliases = Aliases()
        self.references: dict[str, SpanRef] = {}
        self.claims: dict[str, ClaimRef] = {}
        self.bases: dict[str, SourceSliceRef | StatementRef] = {}
        self.limitations: dict[str, object] = {}
        self.unit_references: dict[str, dict[str, str]] = {}
        for group in index.groups.values():
            self.aliases.add("g", group.key)
        for unit in index.units.values():
            self.aliases.add("u", unit.id)
            self.unit_references[unit.id] = {}
            for span in unit.spans:
                ref = span_reference(span, unit)
                alias = self.aliases.add("s", semantic_digest(ref))
                self.references[alias] = ref
                self.unit_references[unit.id][span.id] = alias
            for obligation in unit.obligations:
                self.aliases.add("o", obligation.id)
                for limit in obligation.limitations:
                    alias = self.aliases.add("l", semantic_digest(limit))
                    self.limitations[alias] = limit

    def reference_alias(self, ref: SpanRef) -> str:
        """Register an already validated atomic reference."""
        alias = self.aliases.add("s", semantic_digest(ref))
        if alias not in self.references or self.references[alias] != ref:
            msg = "citation"
            raise ArtifactRuntimeError(msg)
        return alias

    def claim_alias(self, claim: ClaimRef) -> str:
        """Register a validated observation handle for later prose."""
        alias = self.aliases.add("q", semantic_digest(claim))
        self.claims[alias] = claim
        return alias

    def basis_alias(self, ref: SourceSliceRef | StatementRef) -> str:
        """Keep source slices and exact child statements in distinct namespaces."""
        if type(ref) not in {SourceSliceRef, StatementRef}:
            msg = "citation"
            raise ArtifactRuntimeError(msg)
        ref = type(ref).model_validate(ref)
        alias = self.aliases.add(
            "x" if isinstance(ref, SourceSliceRef) else "b",
            semantic_digest(ref),
        )
        if alias in self.bases and self.bases[alias] != ref:
            msg = "citation"
            raise ArtifactRuntimeError(msg)
        self.bases[alias] = ref
        return alias

    def support_input(self, support: SupportSet) -> dict:
        """Describe immediate handles only; source and child text stay in material.

        Hash aliases retain collision-checked full identities locally. They do
        not authorize any support outside the item's consumed SupportSet.
        """
        basis = []
        for ref in support.basis:
            value = {"handle": self.basis_alias(ref), "kind": ref.kind}
            if isinstance(ref, SourceSliceRef):
                value.update(
                    source=self.reference_alias(ref.source),
                    byte_start=ref.byte_start,
                    byte_end=ref.byte_end,
                    digest=self.aliases.add("d", ref.content_digest),
                )
            else:
                value.update(
                    producer=self.aliases.add("i", ref.producer_item_key),
                    result=self.aliases.add("r", ref.result_digest),
                    path=ref.statement_path.model_dump(mode="json"),
                    digest=self.aliases.add("d", ref.statement_digest),
                )
            basis.append(value)
        return {
            "claims": [
                {
                    "handle": self.claim_alias(c),
                    "unit": self.aliases.add("u", c.unit_key),
                    "observation": c.observation_id,
                }
                for c in support.claims
            ],
            "citations": [
                {"handle": self.reference_alias(r), "span": r.span_id}
                for r in support.citations
            ],
            "basis": basis,
            "author": self.aliases.add("a", support.author_context_digest)
            if support.author_context_digest
            else None,
        }

    def statement_input(self, statement: SupportedStatement) -> dict:
        """Serialize validated supported prose with compact current handles."""
        return {
            "kind": statement.kind,
            "text": statement.text,
            "claims": [self.claim_alias(c) for c in statement.claims],
            "citations": [
                self.reference_alias(c) for c in statement.citations
            ],
            "basis": [self.basis_alias(ref) for ref in statement.basis],
            "effect": statement.effect,
            "uncertainty": list(statement.uncertainty),
        }

    def prose_input(self, result: Contract | IndexedProseResult) -> dict:
        """Serialize all prose with the expanded or selected indexed adapter.

        Preconditions:
            ``result`` is a strict prose contract or a live indexed result
            selector bound to the current source context.
        Postconditions:
            Output fields, statement order, terminology, aliases, and defaults
            match the expanded wire adapter exactly.
        Essential behavior:
            Indexed inputs resolve only selected statements and never rebuild
            a complete producer model.
        """
        if type(result) is IndexedProseResult:
            return self.prose_input_indexed(result)
        values = {
            "target": self.aliases.add("i", result.item_key),
            "terminology": [
                {
                    "term": t.term,
                    "definition": self.statement_input(t.definition),
                }
                for t in result.terminology
            ],
        }
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
                values[name] = (
                    self.statement_input(field)
                    if name == "title"
                    else [self.statement_input(s) for s in field]
                )
        return values

    def prose_input_indexed(self, result: IndexedProseResult) -> dict:
        """Serialize a producer from ordered arena statement selectors.

        Preconditions:
            ``result`` selects one planning, section, or document record from
            a live ``IndexedConsumedProse`` owned by the current arena.
        Postconditions:
            The returned dictionary matches ``prose_input`` field-for-field,
            including title placement, term order, defaults, and aliases.
        Essential behavior:
            Materializes a transient tuple of selected statements for this
            producer before serialization. It retains no producer model or
            complete-result presentation tree.
        """
        return self._prose_input_indexed_checked(result)

    def _prose_input_indexed_checked(
        self,
        result: IndexedProseResult,
        checked_sources: IndexedPlanningSources | None = None,
    ) -> dict:
        """Encode selected prose within one current-source operation.

        Preconditions:
            A supplied ``checked_sources`` was just validated by the enclosing
            synchronous call, with no intervening external callback.
        Postconditions:
            Exact owner, row, and wire-index bindings hold. Bytes and aliases
            match the public adapter, and returned values carry no authority.
        Essential behavior:
            The default path checks sources freshly. Both paths use the same
            ordered statement encoding and preserve all field defaults.
        """
        item_key, result_kind, entries, terms = (
            result.prose_projection(index=self.index)
            if checked_sources is None
            else result._prose_projection_checked(  # noqa: SLF001 # The enclosing operation checked this exact owner.
                checked_sources, index=self.index
            )
        )
        by_field: dict[str, list[tuple[int, SupportedStatement]]] = {}
        for path, statement in entries:
            by_field.setdefault(path.field, []).append((path.index, statement))
        values = {
            "target": self.aliases.add("i", item_key),
            "terminology": [
                {
                    "term": term,
                    "definition": self.statement_input(statement),
                }
                for term, (_ordinal, statement) in zip(
                    terms,
                    by_field.get("terminology", ()),
                    strict=True,
                )
            ],
        }
        fields = {
            "planning_card": ("abstraction",),
            "section": ("title", "orientation", "synthesis", "transition"),
            "document": ("preamble", "lead", "closing"),
        }.get(result_kind)
        if fields is None:
            msg = "invalid_prose_family"
            raise ValueError(msg)
        for name in fields:
            selected = sorted(by_field.get(name, ()), key=lambda item: item[0])
            if name == "title":
                if len(selected) != 1 or selected[0][0] != 0:
                    msg = "invalid_statement_path"
                    raise ValueError(msg)
                values[name] = self.statement_input(selected[0][1])
            else:
                values[name] = [
                    self.statement_input(statement)
                    for _ordinal, statement in selected
                ]
        return values

    def detail_input(self, result: DetailResult) -> dict:
        """Retain every detailed field and observation in planning inputs."""
        return {
            "unit": self.aliases.add("u", result.unit_key),
            "explanations": [
                {
                    "obligations": [
                        self.aliases.add("o", o) for o in e.obligation_ids
                    ],
                    "before": e.before,
                    "after": e.after,
                    "effect": e.effect,
                    "uncertainty": e.uncertainty,
                    "citations": [
                        self.reference_alias(r) for r in e.citations
                    ],
                    "limitations": [
                        limit.model_dump(mode="json")
                        for limit in e.limitations
                    ],
                }
                for e in result.explanations
            ],
            "observations": [
                {
                    "handle": self.claim_alias(
                        ClaimRef(unit_key=result.unit_key, observation_id=o.id)
                    ),
                    "text": o.text,
                    "citations": [
                        self.reference_alias(r) for r in o.citations
                    ],
                }
                for o in result.observations
            ],
            "questions": [
                {
                    "text": q.text,
                    "citations": [
                        self.reference_alias(r) for r in q.citations
                    ],
                    "obligations": [
                        self.aliases.add("o", o) for o in q.obligation_ids
                    ],
                }
                for q in result.questions
            ],
        }

    def unit_input(self, unit: EvidenceUnit) -> dict:
        """Send exact text and compiler facts without absolute coordinates."""
        refs = self.unit_references[unit.id]
        return {
            "target": self.aliases.add("u", unit.id),
            "group": self.aliases.add("g", unit.group_key)
            if unit.group_key
            else None,
            "classification": unit.classification.model_dump(mode="json")
            if unit.classification
            else None,
            "availability": [
                a.model_dump(mode="json") for a in unit.availability
            ],
            "spans": [
                {
                    "handle": refs[s.id],
                    "side": s.side,
                    "path": s.path,
                    "text": s.text,
                }
                for s in unit.spans
            ],
            "obligations": [
                {
                    "handle": self.aliases.add("o", o.id),
                    "kind": o.kind,
                    "required_citations": [
                        refs[key] for key in o.required_span_ids
                    ],
                    "limitations": [
                        {
                            "handle": self.aliases.add(
                                "l", semantic_digest(limit)
                            ),
                            "fact": limit.model_dump(mode="json"),
                        }
                        for limit in o.limitations
                    ],
                }
                for o in unit.obligations
            ],
        }

    def _refs(self, handles: tuple[str, ...]) -> tuple[SpanRef, ...]:
        if len(set(handles)) != len(handles) or any(
            h not in self.references for h in handles
        ):
            msg = "citation"
            raise ArtifactRuntimeError(msg)
        return tuple(self.references[h] for h in handles)

    def _detail(self, raw: dict, key: str) -> DetailResult:
        item = WireDetail.model_validate_json(canonical_bytes(raw))
        explanations = []
        for value in item.explanations:
            if len(set(value.limitations)) != len(value.limitations) or any(
                h not in self.limitations for h in value.limitations
            ):
                msg = "citation"
                raise ArtifactRuntimeError(msg)
            explanations.append(
                Explanation(
                    obligation_ids=tuple(
                        self.aliases.resolve("o", h) for h in value.obligations
                    ),
                    before=value.before,
                    after=value.after,
                    effect=value.effect,
                    uncertainty=value.uncertainty,
                    citations=self._refs(value.citations),
                    limitations=tuple(
                        self.limitations[h] for h in value.limitations
                    ),
                )
            )
        observations = []
        for value in item.observations:
            refs = self._refs(value.citations)
            observations.append(
                FactualObservation(
                    id=semantic_digest(
                        "diffstory.observation.v2", key, value.text, refs
                    ),
                    text=value.text,
                    citations=refs,
                )
            )
        return DetailResult(
            unit_key=key,
            explanations=tuple(explanations),
            observations=tuple(observations),
            questions=tuple(
                EvidenceQuestion(
                    text=q.text,
                    citations=self._refs(q.citations),
                    obligation_ids=tuple(
                        self.aliases.resolve("o", h) for h in q.obligations
                    ),
                )
                for q in item.questions
            ),
        )

    def _statement(
        self, item: WireStatement, context: ProseContext
    ) -> SupportedStatement:
        if any(h not in self.claims for h in item.claims) or any(
            h not in self.bases for h in item.basis
        ):
            msg = "citation"
            raise ArtifactRuntimeError(msg)
        return SupportedStatement(
            kind=item.kind,
            text=item.text,
            claims=tuple(self.claims[h] for h in item.claims),
            citations=self._refs(item.citations),
            basis=tuple(self.bases[h] for h in item.basis),
            effect=item.effect,
            uncertainty=item.uncertainty,
            author_context_digest=context.author_digest
            if item.kind == "author_motivation"
            else None,
        )

    def _prose(self, raw: dict, context: ProseContext) -> Contract:
        if len(canonical_bytes(raw)) > context.max_wire_bytes:
            msg = "schema"
            raise ArtifactRuntimeError(msg)
        kind = context.binding.kind
        model = {
            "planning_card": WireCard,
            "section": WireSection,
            "document": WireDocument,
        }[kind]
        item = model.model_validate_json(canonical_bytes(raw))
        values = {
            "item_key": context.binding.key,
            "binding": context.binding,
            "terminology": tuple(
                SupportedTerminology(
                    term=t.term,
                    definition=self._statement(t.definition, context),
                )
                for t in item.terminology
            ),
        }
        for name in model.model_fields:
            if name in {"target", "terminology"}:
                continue
            field = getattr(item, name)
            values[name] = (
                self._statement(field, context)
                if name == "title"
                else tuple(self._statement(s, context) for s in field)
            )
        return result_model(kind)(**values)

    def envelope(  # noqa: C901 # Strict framing precedes independent expansion.
        self,
        raw: object,
        batch: BatchSpec,
        contexts: Mapping[str, ProseContext],
    ) -> WireEnvelope:
        """Validate framing first, then expand each target independently."""
        if type(raw) in {bytes, str}:
            raw = strict_json(raw)
        fields = (
            {"version", "items", "acknowledgements"}
            if batch.stage == "detail"
            else {"version", "items"}
        )
        if (
            type(raw) is not dict
            or set(raw) != fields
            or raw.get("version") != WIRE_VERSION
            or type(raw.get("items")) is not list
        ):
            msg = "malformed_envelope"
            raise ArtifactRuntimeError(msg)
        acknowledgements = raw.get("acknowledgements", [])
        if type(acknowledgements) is not list or any(
            type(a) is not str for a in acknowledgements
        ):
            msg = "malformed_envelope"
            raise ArtifactRuntimeError(msg)
        items, failures = [], []
        targets = [(value, False) for value in raw["items"]]
        targets.extend(({"target": alias}, True) for alias in acknowledgements)
        for value, acknowledged in targets:
            try:
                key = self.aliases.resolve(
                    "u" if batch.stage == "detail" else "i",
                    value.get("target") if type(value) is dict else None,
                )
            except ArtifactRuntimeError:
                # Retain unknown targets for the runtime's safe unknown-item policy.
                items.append({})
                continue
            if key not in batch.item_keys:
                items.append({})
                continue
            try:
                if batch.stage != "detail":
                    result = self._prose(value, contexts[key])
                elif acknowledged:
                    if self.index.units[key].obligations:
                        msg = "coverage"
                        raise ArtifactRuntimeError(msg)  # noqa: TRY301 # Keep this acknowledgement failure item-local.
                    result = DetailResult(
                        unit_key=key,
                        explanations=(),
                        observations=(),
                        questions=(),
                    )
                else:
                    result = self._detail(value, key)
                items.append(result.model_dump(mode="json"))
            except (
                ValueError,
                TypeError,
                KeyError,
                ArtifactRuntimeError,
            ) as error:
                reason = (
                    "invalid_citation"
                    if isinstance(error, ArtifactRuntimeError)
                    and error.reason == "citation"
                    else "invalid_contract"
                )
                failures.append(ItemFailure(item_key=key, reason=reason))
        return WireEnvelope(batch.id, tuple(items), tuple(failures))
