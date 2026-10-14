"""Current-data production hierarchy and sizing-only deterministic interfaces.

No value in this module can send a request. A provider integration must size
PlanningShape with its actual serializer, validate actual consumed support and
output budgets, and audit actual requests after execution. Details stay intact.
"""

from __future__ import annotations

import json
from collections import Counter
from collections import OrderedDict
from dataclasses import dataclass
from types import MappingProxyType
from typing import TYPE_CHECKING
from weakref import WeakKeyDictionary

from diffstory.narrative_contracts import AtomProjection
from diffstory.narrative_contracts import AuthorContext
from diffstory.narrative_contracts import AuthorRecordOwner
from diffstory.narrative_contracts import AuxCard
from diffstory.narrative_contracts import AuxPurpose
from diffstory.narrative_contracts import AuxReference
from diffstory.narrative_contracts import CardBudget
from diffstory.narrative_contracts import ClaimRef
from diffstory.narrative_contracts import ContextSource
from diffstory.narrative_contracts import CycleRoot
from diffstory.narrative_contracts import DetailRecordOwner
from diffstory.narrative_contracts import DetailResult
from diffstory.narrative_contracts import Digest
from diffstory.narrative_contracts import DocumentResult
from diffstory.narrative_contracts import EvidenceQuestion
from diffstory.narrative_contracts import Explanation
from diffstory.narrative_contracts import FactualObservation
from diffstory.narrative_contracts import GroupRoot
from diffstory.narrative_contracts import HeadingRecordOwner
from diffstory.narrative_contracts import MaterialProjection
from diffstory.narrative_contracts import PlanningAtom
from diffstory.narrative_contracts import PlanningBounds
from diffstory.narrative_contracts import PlanningCardResult
from diffstory.narrative_contracts import PlanningGroup
from diffstory.narrative_contracts import ProductionHierarchy
from diffstory.narrative_contracts import RecordEncoding
from diffstory.narrative_contracts import RecordFragment
from diffstory.narrative_contracts import RecordOwner
from diffstory.narrative_contracts import RecordPath
from diffstory.narrative_contracts import ReductionRef
from diffstory.narrative_contracts import SectionResult
from diffstory.narrative_contracts import SectionView
from diffstory.narrative_contracts import SourceRecordOwner
from diffstory.narrative_contracts import SourceSliceRef
from diffstory.narrative_contracts import SpanRef
from diffstory.narrative_contracts import StatementPath
from diffstory.narrative_contracts import StatementRef
from diffstory.narrative_contracts import StoryCard
from diffstory.narrative_contracts import StoryPlan
from diffstory.narrative_contracts import SupportedStatement
from diffstory.narrative_contracts import SupportSet
from diffstory.narrative_plan import _INDEXED_PLANNING_SOURCE_TOKEN
from diffstory.narrative_plan import IndexedDetailContext
from diffstory.narrative_plan import LedgerIndex
from diffstory.narrative_plan import PackingError
from diffstory.narrative_plan import _raw_integrity_bytes
from diffstory.narrative_plan import canonical_bytes
from diffstory.narrative_plan import content_digest
from diffstory.narrative_plan import indexed_detail_record_view
from diffstory.narrative_plan import ordered_components
from diffstory.narrative_plan import planning_atoms
from diffstory.narrative_plan import planning_groups_from_details
from diffstory.narrative_plan import planning_groups_from_indexed_details
from diffstory.narrative_plan import semantic_digest
from diffstory.narrative_plan import span_reference
from diffstory.narrative_plan import story_card
from diffstory.narrative_plan import story_chapters
from diffstory.narrative_plan import validate_primary_structure

if TYPE_CHECKING:
    from collections.abc import Callable
    from collections.abc import Mapping

ProseResult = PlanningCardResult | SectionResult | DocumentResult
UTF8_CONTINUATION = 0x80
DIGEST_HEX_LENGTH = 64
PRODUCTION_PLANNER_VERSION = "diffstory.production-planner.v1"
PRODUCTION_SUPPORT_VERSION = "diffstory.production-support.v1"
REVISION_BINDING_WIDTH = 2


def _require(condition: bool, reason: str) -> None:  # noqa: FBT001
    if not condition:
        raise ValueError(reason)


def record_key(owner: RecordOwner) -> str:
    """Return a namespaced content-stable record identity."""
    return semantic_digest("production.record.v1", owner)


def atom_key(atom: PlanningAtom) -> str:
    """Identify the complete original atom, including its compiler membership."""
    return semantic_digest("production.atom.v1", atom)


@dataclass(frozen=True)
class FieldMaterial:
    """Exact immutable field bytes; collection framing uses canonical JSON."""

    path: RecordPath
    encoding: RecordEncoding
    data: bytes
    digest: str


@dataclass(frozen=True)
class RecordMaterial:
    """A decoded copy of a trusted record, not cached mutable authority."""

    key: str
    owner: RecordOwner
    fields: tuple[FieldMaterial, ...]
    citations: tuple[SpanRef, ...]
    claim: ClaimRef | None = None


@dataclass(frozen=True, eq=False)
class PlanningSources:
    """Opaque, factory-registered source authority; copies have no authority.

    The registry stores bytes and immutable keys only. Accessors return fresh
    models. Current original inputs are checked once per build/validation, not
    once per fragment. Input snapshots never include report/run identities.
    """

    def check_current(self) -> None:
        """Reject changed original inputs and copied or rebound source registries."""
        state = _state(self)
        state.index.validated_ledger()
        _require(
            tuple(canonical_bytes(d) for d in state.original_details)
            == state.detail_snapshots
            and canonical_bytes(state.original_author) == state.author_bytes,
            "planning_sources_changed",
        )

    @property
    def groups(self) -> tuple[PlanningGroup, ...]:
        """Return independent copies of complete trusted compiler/claim facts."""
        return tuple(
            PlanningGroup.model_validate_json(b) for b in _state(self).groups
        )

    @property
    def details(self) -> tuple[DetailResult, ...]:
        """Return full original detail snapshots without shortening any field."""
        return tuple(
            DetailResult.model_validate_json(b)
            for b in _state(self).detail_snapshots
        )

    @property
    def author(self) -> AuthorContext:
        """Return a fresh normalized author value."""
        return AuthorContext.model_validate_json(_state(self).author_bytes)

    @property
    def author_digest(self) -> str:
        """Retain the existing normalized author-context digest convention."""
        return _state(self).author_digest

    @property
    def record_keys(self) -> tuple[str, ...]:
        """List all records in deterministic source/detail/metadata order."""
        return tuple(_state(self).record_metadata)


@dataclass(frozen=True)
class _SourceState:
    index: LedgerIndex
    original_details: tuple[DetailResult, ...]
    detail_snapshots: tuple[bytes, ...]
    original_author: AuthorContext
    author_bytes: bytes
    author_digest: str
    groups: tuple[bytes, ...]
    record_metadata: Mapping[str, bytes]
    record_fields: Mapping[str, tuple[tuple[str, str, bytes, str], ...]]
    source_by_record: Mapping[str, bytes]
    unit_records: Mapping[str, tuple[str, ...]]
    claim_records: Mapping[ClaimRef, str]
    heading_records: Mapping[str, str]
    source_records: Mapping[SpanRef, str]
    source_text: Mapping[SpanRef, bytes]
    author_record: str


INDEXED_FIELD_MEMO_MAX_WEIGHT = 32_768
INDEXED_FIELD_MEMO_MAX_ENTRY_WEIGHT = 8_192


@dataclass(frozen=True)
class _IndexedSourceState:
    """Selectors and small source metadata over one owned detail context.

    Attributes:
        index: Exact current ledger index accepted with ``details_context``.
        details_context: Scope-owned selectors for already validated details.
        revision_binding: The exact base/head revision pair accepted at capture.
        author_bytes: Canonical detached normalized author context.
        author_digest: Existing semantic author-context identity.
        group_bytes: Ordered canonical planning group facts.
        record_keys: Record identities in legacy source/detail/heading/author order.
        record_owners: Immutable owner metadata for each record key.
        detail_selectors: Detail record keys mapped to unit/kind/ordinal selectors.
        source_records: Exact source references mapped to their record keys.
        source_by_record: Source record keys mapped to exact atomic references.
        unit_records: Detail record keys in header/explanation/observation/question order.
        claim_records: Validated claims mapped to observation record keys.
        heading_records: Group keys mapped to heading record keys.
        author_record: The author metadata record key.
        field_memo: Scope-cleared LRU of bounded selected field bytes. Each
            entry weighs encoded data bytes, UTF-8 key/path/encoding bytes,
            ASCII digest bytes, and 64 fixed accounting bytes. Total weight is
            capped at 32,768 bytes and one entry at 8,192 bytes. Only one detail
            field per unit can remain cached.
        metadata_fingerprint: Digest of static selector and metadata state.
    """

    index: LedgerIndex
    details_context: IndexedDetailContext
    revision_binding: tuple[str, str]
    author_bytes: bytes
    author_digest: str
    group_bytes: tuple[bytes, ...]
    record_keys: tuple[str, ...]
    record_owners: Mapping[str, RecordOwner]
    detail_selectors: Mapping[str, tuple[str, str, int]]
    detail_units_by_key: Mapping[str, str]
    source_records: Mapping[SpanRef, str]
    source_by_record: Mapping[str, SpanRef]
    unit_records: Mapping[str, tuple[str, ...]]
    claim_records: Mapping[ClaimRef, str]
    heading_records: Mapping[str, str]
    author_record: str
    field_memo: OrderedDict
    metadata_fingerprint: str

    def binding_witness(self) -> tuple[object, ...]:
        """Return identity-only registration facts for the owning scope.

        Postconditions:
            Contains the identities of all immutable source selectors and the
            accepted context, revision, and normalized author bindings.
        Essential behavior:
            Excludes field-byte memo contents, which may change during use.
        """
        return (
            id(self),
            id(self.index),
            id(self.details_context),
            id(self.field_memo),
            id(self.record_keys),
            id(self.record_owners),
            id(self.detail_selectors),
            id(self.detail_units_by_key),
            id(self.source_records),
            id(self.source_by_record),
            id(self.unit_records),
            id(self.claim_records),
            id(self.heading_records),
            id(self.group_bytes),
            self.revision_binding,
            self.author_bytes,
            self.author_digest,
            self.metadata_fingerprint,
        )

    def current_metadata_fingerprint(self) -> str:
        """Hash static selectors and metadata without reading detail prose.

        Postconditions:
            The result identifies the current adapter metadata and selector
            values, excluding mutable memo contents.
        Essential behavior:
            Does not materialize complete detail records.
        """
        return semantic_digest(
            "production.indexed-source-metadata.v1",
            self.author_bytes.decode("utf-8"),
            self.author_digest,
            tuple(value.decode("utf-8") for value in self.group_bytes),
            self.record_keys,
            content_digest(_raw_integrity_bytes(self.record_owners)),
            tuple(sorted(self.detail_selectors.items())),
            tuple(sorted(self.detail_units_by_key.items())),
            content_digest(_raw_integrity_bytes(self.source_by_record)),
            content_digest(_raw_integrity_bytes(self.source_records)),
            tuple(sorted(self.unit_records.items())),
            content_digest(_raw_integrity_bytes(self.claim_records)),
            tuple(sorted(self.heading_records.items())),
            self.author_record,
            self.revision_binding,
        )


@dataclass(frozen=True, eq=False)
class IndexedPlanningSources:
    """Hierarchy source adapter over one live ``IndexedDetailContext``.

    The value owns normalized author input and small semantic indexes. It keeps
    no duplicate detail model, canonical detail snapshot, or all-field byte
    arrays. Its LRU counts encoded bytes, key/path/encoding/digest bytes, and a
    64-byte fixed allowance per entry. It evicts least-recently-used entries
    above 32,768 bytes, skips entries above 8,192 bytes, keeps at most one
    detail field per unit, and clears when its citation scope closes. A
    scope-local identity witness prevents copies or rebindings from borrowing
    its authority. The selected detail expansion property is an explicit
    compatibility accessor; the normal hierarchy builder does not call it.
    """

    _source_state: _IndexedSourceState

    def check_current(self) -> None:
        """Require the original ledger, selectors, author, and scope to remain live.

        Preconditions:
            This is the exact adapter registered by its detail context.
        Postconditions:
            Raises when ledger content, source revisions, detail witnesses,
            adapter metadata, scope identity, or registration changed.
        Essential behavior:
            Reuses the accepted index and context instead of rebuilding them.
        """
        state = _state(self)
        state.details_context.assert_current(
            state.details_context._ledger,  # noqa: SLF001 # Verify the context's exact ledger object.
            state.index,
            revision_binding=state.revision_binding,
        )
        state.details_context.assert_detail_witnesses_current()
        _require(
            state.current_metadata_fingerprint() == state.metadata_fingerprint,
            "planning_sources_changed",
        )

    @property
    def groups(self) -> tuple[PlanningGroup, ...]:
        """Return detached ordered compiler groups and observation claims.

        Preconditions:
            This adapter remains registered in its live detail scope.
        Postconditions:
            Returned values preserve the expanded oracle's group and claim
            order and are detached models.
        """
        return tuple(
            PlanningGroup.model_validate_json(value)
            for value in _state(self).group_bytes
        )

    @property
    def details(self) -> tuple[DetailResult, ...]:
        """Explicitly expand all detail results for compatibility callers.

        Preconditions:
            The original detail context is live.
        Postconditions:
            Returns detached strict models in captured detail order.
        Essential behavior:
            This compatibility accessor is not used by the normal builder.
        """
        state = _state(self)
        self.check_current()
        return tuple(
            DetailResult.model_validate_json(
                canonical_bytes(state.details_context.detail_record(unit_key))
            )
            for unit_key in state.details_context.unit_keys
        )

    @property
    def author(self) -> AuthorContext:
        """Return a fresh normalized copy of the owned author context.

        Postconditions:
            The caller receives a detached model with the stored author digest.
        """
        return AuthorContext.model_validate_json(_state(self).author_bytes)

    @property
    def author_digest(self) -> str:
        """Return the existing normalized author-context semantic digest.

        Postconditions:
            The digest uses the legacy normalized author semantic identity.
        """
        return _state(self).author_digest

    @property
    def record_keys(self) -> tuple[str, ...]:
        """List every record key in deterministic expanded-compatible order.

        Postconditions:
            Source, detail, heading, and author keys use their existing
            content-stable semantic identities.
        """
        return _state(self).record_keys

    def __copy__(self) -> IndexedPlanningSources:
        """Keep indexed source ownership with the original scope registration."""
        msg = "indexed_planning_sources_cannot_be_copied"
        raise TypeError(msg)

    def __deepcopy__(self, _memo: dict[int, object]) -> IndexedPlanningSources:
        """Keep indexed source ownership with the original scope registration."""
        msg = "indexed_planning_sources_cannot_be_copied"
        raise TypeError(msg)

    def __reduce_ex__(self, _protocol: int) -> object:
        """Prevent serialized data from transferring indexed source authority."""
        msg = "indexed_planning_sources_cannot_be_serialized"
        raise TypeError(msg)


PlanningSource = PlanningSources | IndexedPlanningSources


_SOURCES: WeakKeyDictionary = WeakKeyDictionary()


def _state(
    sources: PlanningSources | IndexedPlanningSources,
) -> _SourceState | _IndexedSourceState:
    """Resolve one exact legacy or scope-registered indexed source owner."""
    if type(sources) is PlanningSources:
        _require(sources in _SOURCES, "unregistered_planning_sources")
        return _SOURCES[sources]
    _require(
        type(sources) is IndexedPlanningSources,
        "unregistered_planning_sources",
    )
    state = sources._source_state  # noqa: SLF001 # Resolve this opaque adapter's private index.
    _require(
        type(state) is _IndexedSourceState, "unregistered_planning_sources"
    )
    scope = state.details_context.citation_scope
    _require(
        scope._owns_indexed_planning_sources(  # noqa: SLF001 # Verify local adapter identity.
            state.details_context, sources, state.binding_witness()
        ),
        "unregistered_planning_sources",
    )
    return state


def _record_bytes(
    owner: RecordOwner,
    values: Mapping[str, object],
    citations: tuple[SpanRef, ...] = (),
    claim: ClaimRef | None = None,
) -> bytes:
    return canonical_bytes(
        {
            "owner": owner,
            "fields": tuple(
                {
                    "path": path,
                    "encoding": "utf8"
                    if isinstance(value, str)
                    else "canonical_json",
                    "text": value
                    if isinstance(value, str)
                    else canonical_bytes(value).decode("utf-8"),
                }
                for path, value in values.items()
            ),
            "citations": citations,
            "claim": claim,
        }
    )


def record_material(sources: PlanningSource, key: str) -> RecordMaterial:
    """Resolve one record's exact current fields without scanning source text.

    Preconditions:
        ``sources`` is a live registered source owner and ``key`` is present.
    Postconditions:
        Returns fields, citations, and claim data with expanded-path ordering,
        framing, and logical digests.
    Essential behavior:
        Indexed sources resolve only this record's selected fields.
    """
    if type(sources) is IndexedPlanningSources:
        return _indexed_record_material(sources, key)
    state = _state(sources)
    _require(key in state.record_metadata, "unknown_record")
    value = json.loads(state.record_metadata[key])
    owners = {
        "detail": DetailRecordOwner,
        "source": SourceRecordOwner,
        "author": AuthorRecordOwner,
        "heading": HeadingRecordOwner,
    }
    owner = owners[value["owner"]["kind"]].model_validate_json(
        canonical_bytes(value["owner"])
    )
    return RecordMaterial(
        key,
        owner,
        tuple(FieldMaterial(*f) for f in state.record_fields[key]),
        tuple(
            SpanRef.model_validate_json(canonical_bytes(ref))
            for ref in value["citations"]
        ),
        ClaimRef.model_validate_json(canonical_bytes(value["claim"]))
        if value["claim"] is not None
        else None,
    )


def derive_planning_sources(
    index: LedgerIndex,
    details: tuple[DetailResult, ...],
    *,
    author_context: AuthorContext,
) -> PlanningSources:
    """Revalidate current details and bind one immutable indexed source registry.

    The input must already contain normalized immutable author fields. Header
    records use ledger path/theme only. Original detail ordering is retained in
    snapshots; lookup and plan derivation do not depend on result arrival order.
    """
    _require(
        type(index) is LedgerIndex
        and type(details) is tuple
        and type(author_context) is AuthorContext,
        "invalid_planning_sources",
    )
    ledger = index.validated_ledger()
    author = AuthorContext.model_validate(author_context)
    groups = planning_groups_from_details(ledger, details, index=index)
    snapshots = tuple(canonical_bytes(d) for d in details)
    records: dict[str, bytes] = {}
    unit_records: dict[str, tuple[str, ...]] = {}
    claim_records: dict[ClaimRef, str] = {}
    source_records: dict[SpanRef, str] = {}
    source_text: dict[SpanRef, bytes] = {}
    for unit in index.units.values():
        for span in unit.spans:
            ref = span_reference(span, unit)
            owner = SourceRecordOwner(
                unit_key=unit.id,
                span_id=span.id,
                content_digest=span.content_digest,
            )
            key = record_key(owner)
            records[key] = _record_bytes(
                owner, {"reference": ref, "text": span.text}
            )
            source_records[ref] = key
            source_text[ref] = span.text.encode("utf-8")
    by_unit = {
        d.unit_key: DetailResult.model_validate_json(b)
        for d, b in zip(details, snapshots, strict=True)
    }
    for unit_key in sorted(by_unit):
        detail = by_unit[unit_key]
        digest = content_digest(canonical_bytes(detail))
        owner = DetailRecordOwner(
            unit_key=unit_key,
            detail_digest=digest,
            record_kind="detail",
            ordinal=0,
        )
        header = record_key(owner)
        # Counts are explicit sequence framing; record ordinals retain order.
        records[header] = _record_bytes(
            owner,
            {
                "version": detail.version,
                "unit_key": detail.unit_key,
                **{
                    name: {
                        "type": "sequence",
                        "length": len(getattr(detail, name)),
                    }
                    for name in ("explanations", "observations", "questions")
                },
            },
        )
        keys = [header]
        for kind, values in (
            ("explanation", detail.explanations),
            ("observation", detail.observations),
            ("question", detail.questions),
        ):
            for ordinal, record in enumerate(values):
                owner = DetailRecordOwner(
                    unit_key=unit_key,
                    detail_digest=digest,
                    record_kind=kind,
                    ordinal=ordinal,
                )
                key = record_key(owner)
                claim = (
                    ClaimRef(unit_key=unit_key, observation_id=record.id)
                    if kind == "observation"
                    else None
                )
                records[key] = _record_bytes(
                    owner,
                    {
                        name: getattr(record, name)
                        for name in type(record).model_fields
                    },
                    record.citations,
                    claim,
                )
                keys.append(key)
                if claim is not None:
                    claim_records[claim] = key
        unit_records[unit_key] = tuple(keys)
    headings = {}
    for group in groups:
        owner = HeadingRecordOwner(group_key=group.group.key)
        key = record_key(owner)
        records[key] = _record_bytes(
            owner, {"path": group.group.path, "theme": group.group.theme}
        )
        headings[group.group.key] = key
    author_digest = semantic_digest("diffstory.author-context.v2", author)
    owner = AuthorRecordOwner(context_digest=author_digest)
    author_key = record_key(owner)
    records[author_key] = _record_bytes(owner, author.model_dump())
    metadata = {}
    fields = {}
    for key, raw in records.items():
        value = json.loads(raw)
        fields[key] = tuple(
            (
                f["path"],
                f["encoding"],
                f["text"].encode("utf-8"),
                content_digest(f["text"].encode("utf-8")),
            )
            for f in value.pop("fields")
        )
        metadata[key] = canonical_bytes(value)
    result = PlanningSources()
    _SOURCES[result] = _SourceState(
        index,
        details,
        snapshots,
        author_context,
        canonical_bytes(author),
        author_digest,
        tuple(canonical_bytes(g) for g in groups),
        MappingProxyType(metadata),
        MappingProxyType(fields),
        MappingProxyType(
            {key: canonical_bytes(ref) for ref, key in source_records.items()}
        ),
        MappingProxyType(unit_records),
        MappingProxyType(claim_records),
        MappingProxyType(headings),
        MappingProxyType(source_records),
        MappingProxyType(source_text),
        author_key,
    )
    return result


def _indexed_source_record_maps(
    index: LedgerIndex,
) -> tuple[
    list[str], dict[str, RecordOwner], dict[SpanRef, str], dict[str, SpanRef]
]:
    """Build source record addresses from the accepted current ledger index.

    Preconditions:
        ``index`` is the exact index bound to the detail context.
    Postconditions:
        Returns source keys in legacy unit/span order and direct reference
        lookups without copying any source text.
    Essential behavior:
        Reuses ``span_reference`` and ``record_key`` for every identity.
    """
    keys: list[str] = []
    owners: dict[str, RecordOwner] = {}
    by_reference: dict[SpanRef, str] = {}
    by_key: dict[str, SpanRef] = {}
    for unit in index.units.values():
        for span in unit.spans:
            reference = span_reference(span, unit)
            owner = SourceRecordOwner(
                unit_key=unit.id,
                span_id=span.id,
                content_digest=span.content_digest,
            )
            key = record_key(owner)
            keys.append(key)
            owners[key] = owner
            by_reference[reference] = key
            by_key[key] = reference
    return keys, owners, by_reference, by_key


def _indexed_detail_record_maps(
    details_context: IndexedDetailContext,
) -> tuple[
    list[str],
    dict[str, RecordOwner],
    dict[str, tuple[str, str, int]],
    dict[str, str],
    dict[str, tuple[str, ...]],
    dict[ClaimRef, str],
]:
    """Build compact detail selectors from captured digest and claim facts.

    Preconditions:
        ``details_context`` is the exact live context accepted for the ledger.
    Postconditions:
        Returns legacy header/detail key order and selectors without resolving
        complete detail records or retaining field bytes.
    Essential behavior:
        Uses the context's existing semantic digests and ordered observations.
    """
    keys: list[str] = []
    owners: dict[str, RecordOwner] = {}
    selectors: dict[str, tuple[str, str, int]] = {}
    units_by_key: dict[str, str] = {}
    unit_records: dict[str, tuple[str, ...]] = {}
    claim_records: dict[ClaimRef, str] = {}
    for unit_key in sorted(details_context.unit_keys):
        detail_digest, observation_ids, counts = (
            details_context.planning_detail_facts(unit_key)
        )
        header_owner = DetailRecordOwner(
            unit_key=unit_key,
            detail_digest=detail_digest,
            record_kind="detail",
            ordinal=0,
        )
        header = record_key(header_owner)
        keys.append(header)
        owners[header] = header_owner
        units_by_key[header] = unit_key
        unit_keys = [header]
        for record_kind, count in zip(
            ("explanation", "observation", "question"), counts, strict=True
        ):
            for ordinal in range(count):
                owner = DetailRecordOwner(
                    unit_key=unit_key,
                    detail_digest=detail_digest,
                    record_kind=record_kind,
                    ordinal=ordinal,
                )
                key = record_key(owner)
                keys.append(key)
                owners[key] = owner
                selectors[key] = (unit_key, record_kind, ordinal)
                units_by_key[key] = unit_key
                unit_keys.append(key)
                if record_kind == "observation":
                    claim = ClaimRef(
                        unit_key=unit_key,
                        observation_id=observation_ids[ordinal],
                    )
                    claim_records[claim] = key
        unit_records[unit_key] = tuple(unit_keys)
    return keys, owners, selectors, units_by_key, unit_records, claim_records


def derive_planning_sources_indexed(
    index: LedgerIndex,
    details_context: IndexedDetailContext,
    *,
    author_context: AuthorContext,
    revision_binding: tuple[str, str] | None = None,
) -> IndexedPlanningSources:
    """Create hierarchy sources over one accepted indexed detail context.

    Preconditions:
        ``index`` and ``details_context`` are exact live owners for the same
        validated ledger; all detail units and current revisions match; and
        ``author_context`` is a normalized ``AuthorContext``.
    Postconditions:
        The result preserves legacy group, record-key, owner, field, claim,
        source-reference, author, and semantic detail identities. It retains no
        complete detail models, canonical detail snapshots, or extracted
        detail field byte arrays.
    Essential behavior:
        Reuses captured detail digests/observation IDs and the existing ledger
        index; it does not call the expanded derivation path.
    """
    _require(
        type(index) is LedgerIndex
        and type(details_context) is IndexedDetailContext
        and type(author_context) is AuthorContext,
        "invalid_indexed_planning_sources",
    )
    groups = planning_groups_from_indexed_details(
        index, details_context, revision_binding=revision_binding
    )
    if revision_binding is not None:
        _require(
            type(revision_binding) is tuple
            and len(revision_binding) == REVISION_BINDING_WIDTH
            and all(type(value) is str for value in revision_binding),
            "citation_scope_revision_mismatch",
        )
    normalized_author = AuthorContext.model_validate_json(
        canonical_bytes(AuthorContext.model_validate(author_context))
    )
    author_bytes = canonical_bytes(normalized_author)
    author_digest = semantic_digest(
        "diffstory.author-context.v2", normalized_author
    )
    claim_records: dict[ClaimRef, str] = {}
    heading_records: dict[str, str] = {}
    source_keys, source_owners, source_records, source_by_record = (
        _indexed_source_record_maps(index)
    )
    (
        detail_keys,
        detail_owners,
        selectors,
        detail_units_by_key,
        unit_records,
        claim_records,
    ) = _indexed_detail_record_maps(details_context)
    owners = {**source_owners, **detail_owners}
    record_keys = [*source_keys, *detail_keys]

    for group in groups:
        owner = HeadingRecordOwner(group_key=group.group.key)
        key = record_key(owner)
        record_keys.append(key)
        owners[key] = owner
        heading_records[group.group.key] = key
    author_owner = AuthorRecordOwner(context_digest=author_digest)
    author_record = record_key(author_owner)
    record_keys.append(author_record)
    owners[author_record] = author_owner

    scope = details_context.citation_scope
    state = _IndexedSourceState(
        index=index,
        details_context=details_context,
        revision_binding=scope.revision_binding,
        author_bytes=author_bytes,
        author_digest=author_digest,
        group_bytes=tuple(canonical_bytes(group) for group in groups),
        record_keys=tuple(record_keys),
        record_owners=MappingProxyType(owners),
        detail_selectors=MappingProxyType(selectors),
        detail_units_by_key=MappingProxyType(detail_units_by_key),
        source_records=MappingProxyType(source_records),
        source_by_record=MappingProxyType(source_by_record),
        unit_records=MappingProxyType(unit_records),
        claim_records=MappingProxyType(claim_records),
        heading_records=MappingProxyType(heading_records),
        author_record=author_record,
        field_memo=OrderedDict(),
        metadata_fingerprint="",
    )
    object.__setattr__(
        state, "metadata_fingerprint", state.current_metadata_fingerprint()
    )
    result = IndexedPlanningSources(state)
    scope._register_indexed_planning_sources(  # noqa: SLF001 # Register authority in the exact owning scope.
        details_context,
        result,
        state.binding_witness(),
        state.field_memo,
        token=_INDEXED_PLANNING_SOURCE_TOKEN,
    )
    return result


def _indexed_record_values(
    state: _IndexedSourceState, key: str
) -> tuple[tuple[str, object], ...]:
    """Read one record's ordered field values from its accepted owner."""
    owner = state.record_owners[key]
    if owner.kind == "detail":
        detail_owner = DetailRecordOwner.model_validate(owner)
        selector = state.detail_selectors.get(key)
        if selector is None:
            _, _, counts = state.details_context.planning_detail_facts(
                detail_owner.unit_key
            )
            values = {
                "version": DetailResult.model_fields["version"].default,
                "unit_key": detail_owner.unit_key,
                **{
                    name: {"type": "sequence", "length": count}
                    for name, count in zip(
                        ("explanations", "observations", "questions"),
                        counts,
                        strict=True,
                    )
                },
            }
            return tuple(values.items())
        unit_key, record_kind, ordinal = selector
        selected = indexed_detail_record_view(
            state.details_context, unit_key, record_kind, ordinal
        )
        model = {
            "explanation": Explanation,
            "observation": FactualObservation,
            "question": EvidenceQuestion,
        }[record_kind]
        return tuple((name, selected[name]) for name in model.model_fields)
    if owner.kind == "source":
        source_owner = SourceRecordOwner.model_validate(owner)
        span = state.index.spans.get(source_owner.span_id)
        unit = state.index.units.get(source_owner.unit_key)
        reference = state.source_by_record[key]
        _require(
            span is not None
            and unit is not None
            and source_owner.span_id
            in state.index.unit_span_ids[source_owner.unit_key],
            "unknown_source_owner",
        )
        return (("reference", reference), ("text", span.text))
    if owner.kind == "heading":
        heading_owner = HeadingRecordOwner.model_validate(owner)
        group = state.index.groups.get(heading_owner.group_key)
        _require(group is not None, "unknown_heading_owner")
        return (("path", group.path), ("theme", group.theme))
    author = AuthorContext.model_validate_json(state.author_bytes)
    return tuple(author.model_dump().items())


def _encoded_field(path: str, value: object) -> tuple[str, str, bytes, str]:
    """Encode one selected value with the legacy field encoding convention."""
    encoding = "utf8" if isinstance(value, str) else "canonical_json"
    data = (
        value.encode("utf-8")
        if isinstance(value, str)
        else canonical_bytes(value)
    )
    return path, encoding, data, content_digest(data)


def _indexed_field(
    state: _IndexedSourceState, key: str, path: str, *, cache: bool
) -> tuple[str, str, bytes, str]:
    """Resolve one selected field and maintain a bounded scope-owned byte memo.

    Preconditions:
        ``key`` and ``path`` select a field in the indexed source state.
    Postconditions:
        Returns the legacy encoded bytes and digest. Cached weight is encoded
        bytes plus UTF-8 key/path/encoding bytes, ASCII digest bytes, and a
        64-byte accounting allowance; total and entry limits are enforced.
    Essential behavior:
        The LRU skips oversized fields, evicts its oldest entries at the total
        limit, and retains at most one detail field for each unit.
    """
    cache_key = (key, path)
    cached = state.field_memo.get(cache_key)
    if cached is not None:
        state.field_memo.move_to_end(cache_key)
        return cached[0]
    values = dict(_indexed_record_values(state, key))
    _require(path in values, "unknown_record_path")
    encoded = _encoded_field(path, values[path])
    if not cache:
        return encoded
    owner = state.record_owners[key]
    weight = (
        len(encoded[2])
        + len(key.encode("utf-8"))
        + len(path.encode("utf-8"))
        + len(encoded[1].encode("utf-8"))
        + len(encoded[3].encode("ascii"))
        + 64
    )
    if (
        weight > INDEXED_FIELD_MEMO_MAX_ENTRY_WEIGHT
        or weight > INDEXED_FIELD_MEMO_MAX_WEIGHT
    ):
        return encoded
    if owner.kind == "detail":
        unit_key = state.detail_units_by_key[key]
        for prior_key in tuple(state.field_memo):
            if state.detail_units_by_key.get(prior_key[0]) == unit_key:
                del state.field_memo[prior_key]
    while state.field_memo and (
        sum(item[1] for item in state.field_memo.values()) + weight
        > INDEXED_FIELD_MEMO_MAX_WEIGHT
    ):
        state.field_memo.popitem(last=False)
    state.field_memo[cache_key] = (encoded, weight)
    return encoded


def _indexed_record_material(
    sources: IndexedPlanningSources, key: str
) -> RecordMaterial:
    """Resolve one record with detached metadata and exact selected fields.

    Preconditions:
        ``key`` selects a record in the live indexed source owner.
    Postconditions:
        Returned owner, citations, and claim models do not alias retained
        source metadata.
    Essential behavior:
        Only the selected record's fields are resolved from its owner.
    """
    state = _state(sources)
    _require(key in state.record_owners, "unknown_record")
    owner = state.record_owners[key].model_copy(deep=True)
    values = _indexed_record_values(state, key)
    fields = tuple(
        FieldMaterial(*_encoded_field(path, value)) for path, value in values
    )
    citations: tuple[SpanRef, ...] = ()
    claim = None
    selector = state.detail_selectors.get(key)
    if selector is not None:
        selected = indexed_detail_record_view(state.details_context, *selector)
        citations = tuple(
            SpanRef.model_validate(value) for value in selected["citations"]
        )
        if selector[1] == "observation":
            claim = ClaimRef(
                unit_key=selector[0], observation_id=selected["id"]
            )
    return RecordMaterial(key, owner, fields, citations, claim)


def _source_bytes(sources: PlanningSource, reference: SpanRef) -> bytes | None:
    """Resolve one source's current bytes from its owning ledger index."""
    state = _state(sources)
    if type(sources) is not IndexedPlanningSources:
        return state.source_text.get(reference)
    key = state.source_records.get(reference)
    _require(key is not None, "invalid_source_slice")
    span = state.index.spans.get(reference.span_id)
    unit = state.index.units.get(reference.unit_key)
    _require(span is not None and unit is not None, "invalid_source_slice")
    _require(
        span_reference(span, unit) == reference
        and reference in state.source_records,
        "invalid_source_slice",
    )
    return span.text.encode("utf-8")


def _has_source_reference(sources: PlanningSource, reference: SpanRef) -> bool:
    """Check source membership through the selected legacy or indexed owner."""
    state = _state(sources)
    return (
        reference in state.source_records
        if type(sources) is IndexedPlanningSources
        else reference in state.source_text
    )


def _source_reference_for_record(sources: PlanningSource, key: str) -> SpanRef:
    """Return the exact source reference attached to one record key."""
    state = _state(sources)
    value = state.source_by_record.get(key)
    _require(value is not None, "unknown_source_owner")
    if type(sources) is IndexedPlanningSources:
        return SpanRef.model_validate(value)
    return SpanRef.model_validate_json(value)


def _fragment(
    owner: RecordOwner, field: FieldMaterial, start: int, end: int
) -> RecordFragment:
    fields = {
        "owner": owner,
        "path": field.path,
        "encoding": field.encoding,
        "byte_start": start,
        "byte_end": end,
        "total_bytes": len(field.data),
        "field_digest": field.digest,
        "content_digest": content_digest(field.data[start:end]),
    }
    return RecordFragment(
        id=semantic_digest("production.fragment.v1", fields), **fields
    )


def fragment_material(
    sources: PlanningSource, fragment: RecordFragment
) -> bytes:
    """Validate and return the exact current field byte range.

    Preconditions:
        ``sources`` is current and the fragment names one owned field.
    Postconditions:
        The returned bytes match both digests and decode as complete UTF-8.
    Essential behavior:
        Indexed sources resolve only the selected field and bound any memoized
        bytes by the scope-local byte-weighted LRU.
    """
    fragment = RecordFragment.model_validate(fragment)
    state = _state(sources)
    key = record_key(fragment.owner)
    if type(sources) is IndexedPlanningSources:
        _require(key in state.record_owners, "unknown_record")
        fields = {
            fragment.path: FieldMaterial(
                *_indexed_field(state, key, fragment.path, cache=True)
            )
        }
    else:
        _require(key in state.record_fields, "unknown_record")
        fields = {f[0]: FieldMaterial(*f) for f in state.record_fields[key]}
    _require(fragment.path in fields, "unknown_record_path")
    field = fields[fragment.path]
    _require(
        fragment
        == _fragment(
            fragment.owner, field, fragment.byte_start, fragment.byte_end
        ),
        "record_fragment_changed",
    )
    _require(fragment.byte_end <= len(field.data), "record_fragment_range")
    data = field.data[fragment.byte_start : fragment.byte_end]
    try:
        data.decode("utf-8")
    except UnicodeError:
        msg = "fragment_utf8_boundary"
        raise ValueError(msg) from None
    return data


def validate_source_slice(
    ref: SourceSliceRef, sources: PlanningSource
) -> bytes:
    """Resolve exact source bytes using relative offsets and current bindings.

    Preconditions:
        ``sources`` is current and ``ref`` names an owned source record.
    Postconditions:
        Returned UTF-8 bytes match ``ref.content_digest`` and byte range.
    Essential behavior:
        Absolute source line numbers never enter byte offsets; indexed sources
        read current text from the accepted ledger index.
    """
    ref = SourceSliceRef.model_validate(ref)
    data = _source_bytes(sources, ref.source)
    _require(
        data is not None and ref.byte_end <= len(data), "invalid_source_slice"
    )
    value = data[ref.byte_start : ref.byte_end]
    _require(
        content_digest(value) == ref.content_digest, "source_slice_digest"
    )
    try:
        value.decode("utf-8")
    except UnicodeError:
        msg = "source_slice_utf8_boundary"
        raise ValueError(msg) from None
    return value


@dataclass(frozen=True)
class FutureResult:
    """A complete future result/support budget, not invented generated prose."""

    subject_key: str
    budget: CardBudget


@dataclass(frozen=True)
class PlanningShape:
    """Sizing only: real leaf bytes and conservative future serialized records.

    check_shape must include the actual provider prefix/schema/aliases/framing,
    repair envelope, response limits and configured output guard. Inherited
    support uses compact StatementRef handles; it never expands source again.
    Forwarded result and support representations must obey these byte budgets.
    sizing_payload doubles escaping with backslashes at the JSON layer. Any
    additional provider serialization layers must be applied by the callback.
    """

    stage: str
    subject_key: str
    purpose: str
    material: bytes
    future: tuple[FutureResult, ...]
    budget: CardBudget
    output_guard: int

    def sizing_payload(self) -> dict:
        """Return fresh sizing data; this method grants no runtime send authority."""
        return {
            "subject": self.subject_key,
            "plan": "0" * 64,
            "material": json.loads(self.material),
            "children": [
                {
                    "subject": child.subject_key,
                    "item_key": "0" * 64,
                    "result_digest": "0" * 64,
                    "context_digest": "0" * 64,
                    "bounded_result": "\\" * child.budget.result_wire_bytes,
                    "bounded_support": "\\" * child.budget.support_bytes,
                }
                for child in self.future
            ],
        }


def projection_material(
    sources: PlanningSource, projection: MaterialProjection
) -> tuple[RecordMaterial, ...]:
    """Resolve whole inline records; reduced projections consume their root."""
    projection = MaterialProjection.model_validate(projection)
    return tuple(
        record_material(sources, key) for key in projection.record_keys
    )


def _inline(
    sources: PlanningSource, keys: tuple[str, ...]
) -> tuple[dict, ...]:
    # Complete structured fields need one JSON frame. Their citation metadata
    # is already a field; repeating the indexed metadata would inflate a whole-
    # fit record. Partial canonical fields use fragment text instead.
    state = _state(sources)
    if type(sources) is IndexedPlanningSources:
        output = []
        for key in keys:
            record = _indexed_record_material(sources, key)
            output.append(
                {
                    "key": key,
                    "owner": record.owner,
                    "fields": {
                        field.path: field.data.decode("utf-8")
                        if field.encoding == "utf8"
                        else json.loads(field.data)
                        for field in record.fields
                    },
                }
            )
        return tuple(output)
    return tuple(
        {
            "key": key,
            "owner": json.loads(state.record_metadata[key])["owner"],
            "fields": {
                path: data.decode("utf-8")
                if encoding == "utf8"
                else json.loads(data)
                for path, encoding, data, _ in state.record_fields[key]
            },
        }
        for key in keys
    )


def _projection_payload(
    sources: PlanningSource, projection: MaterialProjection
) -> dict:
    return {
        "records": _inline(sources, projection.record_keys),
        "root": projection.root,
    }


def _atom_records(
    sources: PlanningSource, atom: PlanningAtom
) -> tuple[str, ...]:
    state = _state(sources)
    if atom.kind == "member":
        return (state.heading_records[atom.group_key],)
    if atom.kind == "unit":
        return state.unit_records[atom.key]
    if atom.kind == "claim":
        return (state.claim_records[atom.claim],)
    return ()


def _with_sources(
    sources: PlanningSource, keys: tuple[str, ...]
) -> tuple[str, ...]:
    state = _state(sources)
    return tuple(
        dict.fromkeys(
            (
                *keys,
                *(
                    state.source_records[ref]
                    for key in keys
                    for ref in record_material(sources, key).citations
                ),
            )
        )
    )


def _aux_card(  # noqa: PLR0913 # Explicit disjoint input families define canonical identity.
    purpose: AuxPurpose,
    owner: str,
    budget: CardBudget,
    *,
    fragments: tuple[str, ...] = (),
    children: tuple[str, ...] = (),
    sources: tuple[AuxReference, ...] = (),
) -> AuxCard:
    fields = {
        "purpose": purpose,
        "owner_key": owner,
        "fragment_keys": fragments,
        "child_keys": children,
        "sources": sources,
        "budget": budget,
    }
    return AuxCard(
        id=semantic_digest("production.aux-card.v1", fields), **fields
    )


def _ref(card: AuxCard) -> ReductionRef:
    return ReductionRef(
        target_key=card.id, owner_key=card.owner_key, purpose=card.purpose
    )


def _relation(  # noqa: PLR0913 # Explicit typed relationship endpoints.
    kind: str,
    root: str,
    relationship: str,
    *,
    target: str | None = None,
    source: str | None = None,
    cycle: str | None = None,
) -> ContextSource:
    fields = {
        "kind": kind,
        "target_key": root,
        "relationship": relationship,
        "target_group_key": target,
        "source_group_key": source,
        "cycle_key": cycle,
    }
    return ContextSource(
        relationship_key=semantic_digest("production.relationship.v1", fields),
        **fields,
    )


class _Builder:
    def __init__(
        self,
        sources: PlanningSource,
        bounds: PlanningBounds,
        budget: CardBudget,
        check: Callable[[PlanningShape], None],
    ) -> None:
        self.sources = sources
        self.bounds = bounds
        self.budget = budget
        self.check = check
        self.fragments: dict[str, RecordFragment] = {}
        self.fragment_data: dict[str, bytes] = {}
        self.auxiliary: dict[str, AuxCard] = {}
        self.record_roots: dict[str, ReductionRef] = {}
        self.cards: list[StoryCard] = []
        self.projections: dict[str, AtomProjection] = {}
        self.author = MaterialProjection()

    def shape(
        self,
        stage: str,
        key: str,
        purpose: str,
        payload: object,
        dependencies: tuple[str, ...] = (),
    ) -> PlanningShape:  # Explicit sizing fields, with no execution authority.
        return PlanningShape(
            stage,
            key,
            purpose,
            canonical_bytes(payload),
            tuple(
                FutureResult(k, self.budget)
                for k in dict.fromkeys(dependencies)
            ),
            self.budget
            if stage == "planning"
            else CardBudget(
                result_wire_bytes=self.bounds.prose_wire_bytes,
                support_bytes=self.budget.support_bytes,
            ),
            self.bounds.output_guard,
        )

    def aux_shape(
        self, card: AuxCard, extra: tuple[RecordFragment, bytes] | None = None
    ) -> PlanningShape:
        fragments = []
        for key in card.fragment_keys:
            fragment, data = (
                extra
                if extra is not None and extra[0].id == key
                else (self.fragments[key], self.fragment_data[key])
            )
            fragments.append(
                {
                    "fragment": fragment,
                    "text": data.decode("utf-8"),
                    "continues_before": fragment.byte_start > 0,
                    "continues_after": fragment.byte_end
                    < fragment.total_bytes,
                }
            )
        return self.shape(
            "planning",
            card.id,
            card.purpose,
            {
                "owner": card.owner_key,
                "purpose": card.purpose,
                "fragments": fragments,
                "sources": card.sources,
            },
            (*card.child_keys, *(s.target_key for s in card.sources)),
        )

    def add_aux(self, card: AuxCard) -> AuxCard:
        self.check(self.aux_shape(card))
        _require(card.id not in self.auxiliary, "duplicate_auxiliary_card")
        self.auxiliary[card.id] = card
        return card

    def reduce_aux(
        self, purpose: AuxPurpose, owner: str, leaves: tuple[AuxCard, ...]
    ) -> AuxCard:
        level = leaves
        while len(level) > 1:
            parents = []
            pending: tuple[str, ...] = ()
            for child in level:
                candidate = _aux_card(
                    purpose, owner, self.budget, children=(*pending, child.id)
                )
                try:
                    self.check(self.aux_shape(candidate))
                except PackingError:
                    if not pending:
                        raise
                    parents.append(
                        self.add_aux(
                            _aux_card(
                                purpose, owner, self.budget, children=pending
                            )
                        )
                    )
                    pending = (child.id,)
                    self.check(
                        self.aux_shape(
                            _aux_card(
                                purpose, owner, self.budget, children=pending
                            )
                        )
                    )
                else:
                    pending += (child.id,)
            if pending:
                parents.append(
                    self.add_aux(
                        _aux_card(
                            purpose, owner, self.budget, children=pending
                        )
                    )
                )
            if len(parents) >= len(level):
                msg = "production_protocol_capacity"
                raise PackingError(msg, owner)
            level = tuple(parents)
        return level[0]

    def reference_root(
        self, purpose: AuxPurpose, owner: str, refs: tuple[AuxReference, ...]
    ) -> ReductionRef:
        _require(bool(refs), "empty_reduction")
        leaves = []
        pending: tuple[AuxReference, ...] = ()
        for ref in refs:
            candidate = _aux_card(
                purpose, owner, self.budget, sources=(*pending, ref)
            )
            try:
                self.check(self.aux_shape(candidate))
            except PackingError:
                if not pending:
                    raise
                leaves.append(
                    self.add_aux(
                        _aux_card(purpose, owner, self.budget, sources=pending)
                    )
                )
                pending = (ref,)
                self.check(
                    self.aux_shape(
                        _aux_card(purpose, owner, self.budget, sources=pending)
                    )
                )
            else:
                pending += (ref,)
        leaves.append(
            self.add_aux(
                _aux_card(purpose, owner, self.budget, sources=pending)
            )
        )
        return _ref(self.reduce_aux(purpose, owner, tuple(leaves)))

    def record_root(self, key: str) -> ReductionRef:  # noqa: C901, PLR0912 # Split exact fields and construct their support before reduction.
        if key in self.record_roots:
            return self.record_roots[key]
        record = record_material(self.sources, key)
        purpose: AuxPurpose = (
            "support"
            if record.owner.kind == "source"
            else "detail"
            if record.owner.kind == "detail"
            else "metadata"
        )
        support: tuple[ReductionRef, ...] = ()
        if record.citations:
            refs = tuple(
                self.record_root(_state(self.sources).source_records[r])
                for r in record.citations
            )
            support = (
                self.reference_root(
                    "support",
                    semantic_digest("production.record-support.v1", key),
                    refs,
                ),
            )
        leaves = []
        for field in record.fields:
            start = 0
            while start < len(field.data) or (not field.data and start == 0):
                end = min(
                    len(field.data), start + self.bounds.max_fragment_bytes
                )
                # UTF-8 continuation bytes are never split, including LS/PS.
                while (
                    end < len(field.data)
                    and field.data[end] & 0xC0 == UTF8_CONTINUATION
                ):
                    end -= 1
                if end == start and field.data:
                    end = start + 1
                    while (
                        end < len(field.data)
                        and field.data[end] & 0xC0 == UTF8_CONTINUATION
                    ):
                        end += 1
                while True:
                    fragment = _fragment(record.owner, field, start, end)
                    card = _aux_card(
                        purpose,
                        key,
                        self.budget,
                        fragments=(fragment.id,),
                        sources=support,
                    )
                    try:
                        self.check(
                            self.aux_shape(
                                card, (fragment, field.data[start:end])
                            )
                        )
                    except PackingError:
                        shorter = start + (end - start) // 2
                        while (
                            shorter > start
                            and field.data[shorter] & 0xC0 == UTF8_CONTINUATION
                        ):
                            shorter -= 1
                        if shorter == start:
                            raise
                        end = shorter
                    else:
                        break
                self.fragments[fragment.id] = fragment
                self.fragment_data[fragment.id] = field.data[start:end]
                leaves.append(self.add_aux(card))
                start = end
                if not field.data:
                    break
        root = _ref(self.reduce_aux(purpose, key, tuple(leaves)))
        self.record_roots[key] = root
        return root

    def projection(
        self, keys: tuple[str, ...], owner: str, purpose: AuxPurpose
    ) -> MaterialProjection:
        roots = tuple(self.record_root(key) for key in keys)
        root = (
            roots[0]
            if len(roots) == 1
            else self.reference_root(purpose, owner, roots)
        )
        return MaterialProjection(root=root)

    def primary_shape(self, card: StoryCard) -> PlanningShape:
        atoms = []
        dependencies = list(card.child_keys)
        for atom in card.atoms:
            projection = self.projections[atom_key(atom)].material
            # Large path/theme and complete detail are in the projection, not a
            # repeated atom header. Compiler relationship keys remain exact.
            atoms.append(
                {
                    "atom_key": atom_key(atom),
                    "group_key": atom.group_key,
                    "kind": atom.kind,
                    "key": atom.key,
                    "claim": atom.claim,
                    "term": atom.term,
                    "material": _projection_payload(self.sources, projection),
                }
            )
            if projection.root:
                dependencies.append(projection.root.target_key)
        author = (
            _projection_payload(self.sources, self.author)
            if card.atoms
            else None
        )
        if card.atoms and self.author.root:
            dependencies.append(self.author.root.target_key)
        return self.shape(
            "planning",
            card.id,
            "primary",
            {"atoms": atoms, "author": author},
            tuple(dependencies),
        )

    def add_primary(self, card: StoryCard) -> StoryCard:
        self.check(self.primary_shape(card))
        self.cards.append(card)
        return card

    def reduce_primary(self, level: tuple[StoryCard, ...]) -> StoryCard:
        while len(level) > 1:
            parents = []
            pending: tuple[str, ...] = ()
            for child in level:
                card = story_card(
                    child.level + 1, children=(*pending, child.id)
                )
                try:
                    self.check(self.primary_shape(card))
                except PackingError:
                    if not pending:
                        raise
                    parents.append(
                        self.add_primary(
                            story_card(child.level + 1, children=pending)
                        )
                    )
                    pending = (child.id,)
                    self.check(
                        self.primary_shape(
                            story_card(child.level + 1, children=pending)
                        )
                    )
                else:
                    pending += (child.id,)
            parents.append(
                self.add_primary(
                    story_card(level[0].level + 1, children=pending)
                )
            )
            if len(parents) >= len(level):
                msg = "production_protocol_capacity"
                raise PackingError(msg, level[0].id)
            level = tuple(parents)
        return level[0]

    def primary_group(self, group: PlanningGroup) -> StoryCard:
        leaves = []
        pending: tuple[PlanningAtom, ...] = ()
        for atom in planning_atoms(group):
            key = atom_key(atom)
            records = _atom_records(self.sources, atom)
            self.projections[key] = AtomProjection(
                atom_key=key,
                material=MaterialProjection(
                    record_keys=_with_sources(self.sources, records)
                ),
            )
            try:
                self.check(self.primary_shape(story_card(0, (atom,))))
            except PackingError:
                if not records:
                    raise
                self.projections[key] = AtomProjection(
                    atom_key=key,
                    material=self.projection(
                        records,
                        key,
                        "detail"
                        if atom.kind in {"unit", "claim"}
                        else "metadata",
                    ),
                )
                self.check(self.primary_shape(story_card(0, (atom,))))
            try:
                self.check(self.primary_shape(story_card(0, (*pending, atom))))
            except PackingError:
                _require(bool(pending), "unchecked_primary_leaf")
                leaves.append(self.add_primary(story_card(0, pending)))
                pending = (atom,)
            else:
                pending += (atom,)
        leaves.append(self.add_primary(story_card(0, pending)))
        return self.reduce_primary(tuple(leaves))

    def terminal_shape(
        self, view: SectionView | None, roots: tuple[str, ...]
    ) -> PlanningShape:
        if view is None:
            payload = {
                "global_roots": roots,
                "author": _projection_payload(self.sources, self.author),
            }
            dependencies = roots
            key = semantic_digest("production.document-shape.v1")
        else:
            payload = {
                "group_key": view.group_key,
                "position": view.position,
                "previous": view.previous_group_key,
                "next": view.next_group_key,
                "chapter_key": view.chapter_key,
                "cycle_key": view.cycle_key,
                "final": view.final,
                "relationship_count": len(view.sources),
                "context_root": view.context_root,
                "sources": () if view.context_root else view.sources,
                "author": _projection_payload(self.sources, self.author),
            }
            dependencies = (
                (view.context_root.target_key,)
                if view.context_root
                else tuple(s.target_key for s in view.sources)
            )
            key = view.group_key
        if self.author.root:
            dependencies += (self.author.root.target_key,)
        return self.shape(
            "document" if view is None else "section",
            key,
            "terminal",
            payload,
            dependencies,
        )

    def build(self, context_digest: str) -> StoryPlan:  # noqa: C901, PLR0915 # Freeze in dependency order, including all terminal shapes.
        groups = self.sources.groups
        order, cycles = ordered_components(
            {g.group.key: g.group.prerequisites for g in groups}
        )
        lookup = {g.group.key: g for g in groups}
        author_key = _state(self.sources).author_record
        self.author = MaterialProjection(record_keys=(author_key,))
        # First test metadata with two future children. If variable author data
        # does not fit, remove it from repeated envelopes through a metadata root.
        probe = self.shape(
            "planning",
            semantic_digest("production.probe.v1"),
            "metadata",
            {"author": _projection_payload(self.sources, self.author)},
            ("1" * 64, "2" * 64),
        )
        try:
            self.check(probe)
        except PackingError:
            self.author = self.projection(
                (author_key,), author_key, "metadata"
            )
        roots = [self.primary_group(lookup[key]) for key in order]
        group_roots = tuple(
            GroupRoot(group_key=key, card_key=root.id)
            for key, root in zip(order, roots, strict=True)
        )
        if roots:
            depth = max(root.level for root in roots)
            for pos, group_root in enumerate(roots):
                root = group_root
                while root.level < depth:
                    root = self.add_primary(
                        story_card(root.level + 1, children=(root.id,))
                    )
                roots[pos] = root
            global_roots = (self.reduce_primary(tuple(roots)).id,)
        else:
            global_roots = ()
        root_map = {r.group_key: r.card_key for r in group_roots}
        cycle_roots = []
        for cycle in cycles:
            cycle_key = semantic_digest("production.cycle.v1", cycle)
            refs = tuple(
                _relation(
                    "group",
                    root_map[key],
                    "cycle_member",
                    source=key,
                    cycle=cycle_key,
                )
                for key in cycle
            )
            cycle_roots.append(
                CycleRoot(
                    cycle_key=cycle_key,
                    group_keys=cycle,
                    root=self.reference_root("cycle", cycle_key, refs),
                )
            )
        by_cycle = {
            key: cycle for cycle in cycle_roots for key in cycle.group_keys
        }
        chapters = story_chapters(
            order, groups, self.bounds.max_chapter_groups
        )
        chapter_map = {key: c.id for c in chapters for key in c.group_keys}
        views = []
        for pos, key in enumerate(order):
            previous = order[pos - 1] if pos else None
            following = order[pos + 1] if pos + 1 < len(order) else None
            cycle = by_cycle.get(key)
            refs = [
                _relation(
                    "group", root_map[key], "own", target=key, source=key
                )
            ]
            refs.extend(
                _relation(
                    "group", root_map[p], "prerequisite", target=key, source=p
                )
                for p in lookup[key].group.prerequisites
            )
            if cycle:
                refs.append(
                    _relation(
                        "cycle",
                        cycle.root.target_key,
                        "cycle",
                        target=key,
                        cycle=cycle.cycle_key,
                    )
                )
            for other, relation in (
                (previous, "previous"),
                (following, "next"),
            ):
                if other is not None:
                    refs.append(
                        _relation(
                            "group",
                            root_map[other],
                            relation,
                            target=key,
                            source=other,
                        )
                    )
            refs.extend(
                _relation("global", root, "global", target=key)
                for root in global_roots
            )
            fields = {
                "group_key": key,
                "position": pos,
                "previous_group_key": previous,
                "next_group_key": following,
                "chapter_key": chapter_map[key],
                "cycle_key": cycle.cycle_key if cycle else None,
                "final": following is None,
                "sources": tuple(refs),
            }
            view = SectionView(**fields)
            try:
                self.check(self.terminal_shape(view, global_roots))
            except PackingError:
                view = SectionView(
                    **fields,
                    context_root=self.reference_root(
                        "section_context",
                        semantic_digest("production.section-context.v1", key),
                        tuple(refs),
                    ),
                )
                self.check(self.terminal_shape(view, global_roots))
            views.append(view)
        self.check(self.terminal_shape(None, global_roots))
        hierarchy = ProductionHierarchy(
            bounds=self.bounds,
            card_budget=self.budget,
            fragments=tuple(self.fragments.values()),
            auxiliary_cards=tuple(self.auxiliary.values()),
            atom_projections=tuple(self.projections.values()),
            author_projection=self.author,
            group_roots=group_roots,
            cycle_roots=tuple(cycle_roots),
            section_views=tuple(views),
        )
        plan = StoryPlan(
            id="0" * 64,
            context_digest=context_digest,
            groups=tuple(lookup[k] for k in order),
            ordered_group_keys=order,
            cycles=cycles,
            chapters=chapters,
            cards=tuple(self.cards),
            root_card_keys=global_roots,
            requests=(),
            production=hierarchy,
        )
        plan = StoryPlan.model_validate(
            plan.model_dump() | {"id": production_plan_id(plan)}
        )
        validate_primary_structure(plan, groups)
        planning_waves(plan)
        return plan


def production_plan_id(plan: StoryPlan) -> str:
    """Hash the entire frozen structure and bounds, excluding requests and prose."""
    _require(plan.production is not None, "production_hierarchy_required")
    return semantic_digest(
        "production.story-plan.v1", plan.model_dump(exclude={"id", "requests"})
    )


def build_production_plan(
    sources: PlanningSource,
    *,
    context_digest: Digest,
    bounds: PlanningBounds,
    check_shape: Callable[[PlanningShape], None],
) -> StoryPlan:
    """Freeze all hierarchy/terminal shapes before any planning result exists.

    A failed data leaf splits at UTF-8 boundaries. A reducer/terminal capacity
    failure reduces only new abstraction budgets and retries structure creation.
    Configured output guards and original detailed writing never change.
    check_shape signals measured capacity failures with PackingError only.

    Preconditions:
        ``sources`` is current, bounds are valid, and ``check_shape`` raises
        ``PackingError`` only for measured capacity failures.
    Postconditions:
        Returns the deterministic production structure derived from source
        records, bounds, and the requested context digest.
    Essential behavior:
        Uses either legacy sources or one scope-owned indexed adapter through
        the same sizing and planning algorithms.
    """
    _require(
        type(context_digest) is str
        and len(context_digest) == DIGEST_HEX_LENGTH
        and all(c in "0123456789abcdef" for c in context_digest),
        "invalid_context_digest",
    )
    sources.check_current()
    bounds = PlanningBounds.model_validate(bounds)
    wire, support = bounds.card_wire_bytes, bounds.card_support_bytes
    while True:
        budget = CardBudget(result_wire_bytes=wire, support_bytes=support)
        try:
            return _Builder(sources, bounds, budget, check_shape).build(
                context_digest
            )
        except PackingError:
            if (
                wire == bounds.min_card_wire_bytes
                and support == bounds.min_card_support_bytes
            ):
                msg = "production_protocol_capacity"
                raise PackingError(msg, context_digest) from None
            wire = max(bounds.min_card_wire_bytes, wire // 2)
            support = max(bounds.min_card_support_bytes, support // 2)


def validate_production_requests(plan: StoryPlan) -> None:
    """Require exact once-only coverage of both subject families in actual audits.

    The integration must additionally prove actual provider bytes and the
    structural-subject/runtime-result key bijection; sizing shapes cannot do so.
    """
    _require(plan.production is not None, "production_hierarchy_required")
    subjects = tuple(c.id for c in plan.cards) + tuple(
        c.id for c in plan.production.auxiliary_cards
    )
    requested = tuple(key for spec in plan.requests for key in spec.item_keys)
    _require(
        all(spec.stage == "planning" for spec in plan.requests)
        and Counter(requested) == Counter(subjects),
        "planning_request_membership_mismatch",
    )
    _require(
        len({spec.id for spec in plan.requests}) == len(plan.requests),
        "duplicate_planning_request",
    )


def validate_production_plan(
    plan: StoryPlan,
    sources: PlanningSource,
    *,
    context_digest: Digest,
    bounds: PlanningBounds,
    check_shape: Callable[[PlanningShape], None],
) -> None:
    """Rederive full structure from current trusted data, never from the manifest.

    A frozen pre-execution plan has no requests. If requests are attached, their
    subject union must be complete. Final consumers must always also call
    validate_production_requests, including when the supplied audit is empty.

    Preconditions:
        ``sources`` is current and ``check_shape`` measures the same serializer
        used to construct the candidate plan.
    Postconditions:
        Raises if the plan differs from a fresh build over current sources.
    Essential behavior:
        The plan is rederived through the shared legacy/indexed source adapter.
    """
    plan = StoryPlan.model_validate(plan)
    expected = build_production_plan(
        sources,
        context_digest=context_digest,
        bounds=bounds,
        check_shape=check_shape,
    )
    _require(
        plan.model_dump(exclude={"requests"})
        == expected.model_dump(exclude={"requests"}),
        "production_plan_changed",
    )
    if plan.requests:
        validate_production_requests(plan)


def planning_waves(plan: StoryPlan) -> tuple[tuple[Digest, ...], ...]:  # noqa: C901, PLR0912, PLR0915 # Validate both typed families before iterative graph traversal.
    """Return deterministic dependency-ready primary/auxiliary subject waves."""
    primary = {c.id: c for c in plan.cards}
    dependencies = {c.id: set(c.child_keys) for c in plan.cards}
    _require(len(primary) == len(plan.cards), "duplicate_story_card")
    if plan.production is not None:
        hierarchy = plan.production
        auxiliary = {c.id: c for c in hierarchy.auxiliary_cards}
        _require(
            len(auxiliary) == len(hierarchy.auxiliary_cards)
            and not primary.keys() & auxiliary.keys(),
            "duplicate_auxiliary_card",
        )
        projections = {p.atom_key: p for p in hierarchy.atom_projections}
        _require(
            len(projections) == len(hierarchy.atom_projections),
            "duplicate_atom_projection",
        )
        _require(
            set(projections)
            == {atom_key(a) for c in plan.cards for a in c.atoms},
            "atom_projection_membership",
        )
        fragments = {f.id: f for f in hierarchy.fragments}
        _require(
            len(fragments) == len(hierarchy.fragments),
            "duplicate_record_fragment",
        )
        _require(
            Counter(f for c in auxiliary.values() for f in c.fragment_keys)
            == Counter(fragments.keys()),
            "fragment_consumption_membership",
        )
        child_owners: Counter = Counter()
        for card in auxiliary.values():
            for key in card.fragment_keys:
                owner = fragments[key].owner
                _require(
                    card.owner_key == record_key(owner)
                    and card.purpose
                    == (
                        "support"
                        if owner.kind == "source"
                        else "detail"
                        if owner.kind == "detail"
                        else "metadata"
                    ),
                    "fragment_owner_mismatch",
                )
            for key in card.child_keys:
                _require(
                    key in auxiliary
                    and auxiliary[key].owner_key == card.owner_key
                    and auxiliary[key].purpose == card.purpose,
                    "auxiliary_child_owner",
                )
                child_owners[key] += 1
            for ref in card.sources:
                if isinstance(ref, ReductionRef):
                    _require(
                        ref.target_key in auxiliary
                        and _ref(auxiliary[ref.target_key]) == ref,
                        "invalid_reduction_reference",
                    )
                    _require(
                        ref.purpose
                        in {
                            "support": {"support"},
                            "detail": {"support", "detail"},
                            "metadata": {"metadata"},
                            "cycle": set(),
                            "section_context": set(),
                        }[card.purpose],
                        "auxiliary_dependency_family",
                    )
                else:
                    _require(
                        ref.target_key
                        in (auxiliary if ref.kind == "cycle" else primary),
                        "invalid_context_reference",
                    )
                    _require(
                        ref.kind != "cycle"
                        or auxiliary[ref.target_key].purpose == "cycle",
                        "context_root_family",
                    )
                    _require(
                        card.purpose in {"cycle", "section_context"}
                        and (
                            card.purpose != "cycle"
                            or (
                                ref.kind == "group"
                                and ref.relationship == "cycle_member"
                            )
                        ),
                        "context_dependency_family",
                    )
            dependencies[card.id] = set(card.child_keys) | {
                ref.target_key for ref in card.sources
            }
        _require(
            all(n == 1 for n in child_owners.values()),
            "shared_auxiliary_child",
        )
        _require(
            all(
                ref.target_key not in child_owners
                for card in auxiliary.values()
                for ref in card.sources
                if isinstance(ref, ReductionRef)
            ),
            "shared_nonroot_reduction",
        )
        for card in primary.values():
            for atom in card.atoms:
                _require(
                    atom_key(atom) in projections, "missing_atom_projection"
                )
                root = projections[atom_key(atom)].material.root
                if root:
                    _require(
                        root.target_key in auxiliary
                        and _ref(auxiliary[root.target_key]) == root,
                        "invalid_projection_root",
                    )
                    _require(
                        root.target_key not in child_owners
                        and root.purpose
                        == ("metadata" if atom.kind == "member" else "detail"),
                        "projection_dependency_family",
                    )
                    dependencies[card.id].add(root.target_key)
            if card.atoms and hierarchy.author_projection.root:
                root = hierarchy.author_projection.root
                _require(
                    root.target_key in auxiliary
                    and _ref(auxiliary[root.target_key]) == root,
                    "invalid_author_root",
                )
                _require(
                    root.target_key not in child_owners
                    and root.purpose == "metadata",
                    "author_dependency_family",
                )
                dependencies[card.id].add(root.target_key)
    _require(
        all(
            children <= dependencies.keys()
            for children in dependencies.values()
        ),
        "unknown_planning_dependency",
    )
    # Kahn traversal is iterative even for deep source/record/primary forests.
    reverse: dict[str, list[str]] = {key: [] for key in dependencies}
    remaining = {key: len(values) for key, values in dependencies.items()}
    for key, values in dependencies.items():
        for child in values:
            reverse[child].append(key)
    ready = tuple(sorted(key for key, count in remaining.items() if not count))
    waves = []
    while ready:
        waves.append(ready)
        following = []
        for key in ready:
            for parent in reverse[key]:
                remaining[parent] -= 1
                if remaining[parent] == 0:
                    following.append(parent)
        ready = tuple(sorted(following))
    _require(
        sum(map(len, waves)) == len(dependencies), "planning_dependency_cycle"
    )
    return tuple(waves)


def planning_shapes(
    plan: StoryPlan, sources: PlanningSource
) -> tuple[PlanningShape, ...]:
    """Reconstruct sizing descriptions after explicit current production validation.

    These are useful for serializer calibration/tests, never actual request
    audits. Call validate_production_plan before trusting a persisted plan.

    Preconditions:
        ``plan`` contains a production hierarchy and ``sources`` is current.
    Postconditions:
        Returns sizing shapes in deterministic dependency-wave order.
    Essential behavior:
        Resolves fragment bytes from either source adapter without expanding all
        indexed detail results.
    """
    _require(plan.production is not None, "production_hierarchy_required")
    h = plan.production
    builder = _Builder(sources, h.bounds, h.card_budget, lambda _shape: None)
    builder.fragments = {f.id: f for f in h.fragments}
    builder.fragment_data = {
        f.id: fragment_material(sources, f) for f in h.fragments
    }
    builder.projections = {p.atom_key: p for p in h.atom_projections}
    builder.author = h.author_projection
    shapes = {c.id: builder.aux_shape(c) for c in h.auxiliary_cards}
    shapes.update({c.id: builder.primary_shape(c) for c in plan.cards})
    return (
        tuple(shapes[key] for wave in planning_waves(plan) for key in wave)
        + tuple(
            builder.terminal_shape(v, plan.root_card_keys)
            for v in h.section_views
        )
        + (builder.terminal_shape(None, plan.root_card_keys),)
    )


def statement_at(
    result: ProseResult, path: StatementPath
) -> SupportedStatement:
    """Resolve only known statement fields in the exact result family."""
    path = StatementPath.model_validate(path)
    allowed = {
        PlanningCardResult: ("abstraction",),
        SectionResult: ("title", "orientation", "synthesis", "transition"),
        DocumentResult: ("preamble", "lead", "closing"),
    }
    _require(
        type(result) in allowed
        and (
            path.field == "terminology" or path.field in allowed[type(result)]
        ),
        "invalid_statement_path",
    )
    if path.field == "title":
        return result.title
    values = (
        result.terminology
        if path.field == "terminology"
        else getattr(result, path.field)
    )
    _require(path.index < len(values), "invalid_statement_path")
    value = values[path.index]
    return value.definition if path.field == "terminology" else value


def statement_reference(
    result: ProseResult, path: StatementPath
) -> StatementRef:
    """Bind an exact complete implementation statement; other kinds cannot prove it."""
    _require(
        type(result) in {PlanningCardResult, SectionResult, DocumentResult},
        "invalid_statement_producer",
    )
    result = type(result).model_validate(result)
    statement = statement_at(result, path)
    _require(
        statement.kind == "implementation_claim", "unsupported_statement_kind"
    )
    return StatementRef(
        producer_item_key=result.item_key,
        result_digest=content_digest(canonical_bytes(result)),
        statement_path=path,
        statement_digest=content_digest(canonical_bytes(statement)),
    )


def validate_statement_reference(
    ref: StatementRef, consumed: Mapping[str, ProseResult]
) -> SupportedStatement:
    """Require an exact consumed producer/result/path/digest, without guessing keys."""
    ref = StatementRef.model_validate(ref)
    _require(
        ref.producer_item_key in consumed, "unconsumed_statement_producer"
    )
    result = consumed[ref.producer_item_key]
    _require(
        type(result) in {PlanningCardResult, SectionResult, DocumentResult},
        "invalid_statement_producer",
    )
    result = type(result).model_validate(result)
    _require(
        result.item_key == ref.producer_item_key
        and statement_reference(result, ref.statement_path) == ref,
        "statement_reference_changed",
    )
    return statement_at(result, ref.statement_path)


def material_support(
    sources: PlanningSource,
    *,
    record_keys: tuple[str, ...] = (),
    fragments: tuple[RecordFragment, ...] = (),
) -> SupportSet:
    """Grant whole references only for complete supplied records/source texts.

    Fragment coverage alone does not grant a whole ClaimRef or SpanRef. A source
    text fragment grants just its exact slice. Author fragments retain author
    provenance, without adding implementation support.

    Preconditions:
        Every supplied key and fragment belongs to the current source owner.
    Postconditions:
        Whole claims/references are granted only by complete record material;
        text fragments grant exact source slices.
    Essential behavior:
        Uses the same support rules for legacy and indexed source state.
    """
    state = _state(sources)
    records = tuple(record_material(sources, key) for key in record_keys)
    whole = tuple(
        _source_reference_for_record(sources, key)
        for key in dict.fromkeys(record_keys)
        if key in state.source_by_record
    )
    claims = tuple(
        r.claim
        for r in records
        if r.claim is not None and set(r.citations) <= set(whole)
    )
    basis = []
    author = any(
        r.owner.kind == "author" and any(f.data for f in r.fields)
        for r in records
    )
    for fragment in fragments:
        data = fragment_material(sources, fragment)
        author = author or (fragment.owner.kind == "author" and bool(data))
        if (
            fragment.owner.kind == "source"
            and fragment.path == "text"
            and data
        ):
            owner = fragment.owner
            # Indexed lookup, never a whole-source scan per fragment.
            ref = source_reference(sources, owner)
            basis.append(
                SourceSliceRef(
                    source=ref,
                    byte_start=fragment.byte_start,
                    byte_end=fragment.byte_end,
                    content_digest=fragment.content_digest,
                )
            )
    return SupportSet(
        claims=tuple(dict.fromkeys(claims)),
        citations=whole,
        basis=tuple(dict.fromkeys(basis)),
        author_context_digest=state.author_digest if author else None,
    )


def source_reference(
    sources: PlanningSource, owner: SourceRecordOwner
) -> SpanRef:
    """Resolve a compact source owner to its exact current atomic reference.

    Preconditions:
        ``owner`` identifies a source record in the current source adapter.
    Postconditions:
        Returns the legacy stable span reference for that source record.
    """
    state = _state(sources)
    key = record_key(owner)
    if type(sources) is IndexedPlanningSources:
        reference = state.source_by_record.get(key)
        _require(reference is not None, "unknown_source_owner")
        return SpanRef.model_validate(reference)
    _require(key in state.source_by_record, "unknown_source_owner")
    return SpanRef.model_validate_json(state.source_by_record[key])


def statement_entries(
    result: ProseResult,
) -> tuple[tuple[StatementPath, SupportedStatement], ...]:
    """Enumerate exact statement locations for consumed-result support selection."""
    fields = {
        PlanningCardResult: ("abstraction",),
        SectionResult: ("title", "orientation", "synthesis", "transition"),
        DocumentResult: ("preamble", "lead", "closing"),
    }
    _require(type(result) in fields, "invalid_prose_family")
    values = []
    for name in (*fields[type(result)], "terminology"):
        items = (result.title,) if name == "title" else getattr(result, name)
        for index, item in enumerate(items):
            values.append(
                (
                    StatementPath(field=name, index=index),
                    item.definition if name == "terminology" else item,
                )
            )
    return tuple(values)


def inherited_support(
    consumed: tuple[ProseResult, ...],
    *,
    author_context_digest: str | None = None,
) -> SupportSet:
    """Forward exact implementation statements, never their recursively expanded text.

    Producers must already pass their consumed-support and receipt validation.
    Author attribution is forwarded only when its actual child statement exists.
    """
    refs = tuple(
        statement_reference(result, path)
        for result in consumed
        for path, statement in statement_entries(result)
        if statement.kind == "implementation_claim"
    )
    author = any(
        s.kind == "author_motivation"
        and s.author_context_digest == author_context_digest
        for r in consumed
        for _, s in statement_entries(r)
    )
    return SupportSet(
        basis=tuple(dict.fromkeys(refs)),
        author_context_digest=author_context_digest if author else None,
    )


def validate_consumed_support(
    result: ProseResult,
    allowed: SupportSet,
    sources: PlanningSource,
    consumed: Mapping[str, ProseResult],
) -> None:
    """Check actual immediate support grants and exact inherited statement endpoints.

    Integration supplies the validated, actually consumed producer set and owns
    receipt/binding/budget checks. Use validate_support_chains on final output to
    prove iterative termination at current source endpoints as well.

    Preconditions:
        ``allowed`` is the actual immediate grant set and ``consumed`` contains
        only validated producers consumed by this result.
    Postconditions:
        Every statement support reference is an exact member of those grants
        and resolves to the current source owner or consumed producer.
    Essential behavior:
        Accepts the shared legacy or indexed planning source adapter.
    """
    result = type(result).model_validate(result)
    allowed = SupportSet.model_validate(allowed)
    state = _state(sources)
    for _, statement in statement_entries(result):
        _require(
            set(statement.claims) <= set(allowed.claims)
            and set(statement.citations) <= set(allowed.citations)
            and set(statement.basis) <= set(allowed.basis),
            "unconsumed_support",
        )
        if statement.kind == "author_motivation":
            _require(
                statement.author_context_digest
                == allowed.author_context_digest
                == state.author_digest,
                "unconsumed_author_context",
            )
        for ref in statement.basis:
            if isinstance(ref, SourceSliceRef):
                validate_source_slice(ref, sources)
            else:
                validate_statement_reference(ref, consumed)
        _require(
            all(c in state.claim_records for c in statement.claims)
            and all(
                _has_source_reference(sources, r) for r in statement.citations
            ),
            "invalid_current_support",
        )


def validate_support_chains(
    result: ProseResult,
    sources: PlanningSource,
    producers: Mapping[str, ProseResult],
) -> None:
    """Validate one root through the shared current-source forest boundary."""
    validate_support_forest((result,), sources, producers)


def validate_support_forest(
    results: tuple[ProseResult, ...],
    sources: PlanningSource,
    producers: Mapping[str, ProseResult],
) -> None:
    """Iteratively validate complete inherited chains to valid current endpoints.

    This does not establish consumption or receipts; validate_consumed_support
    must have checked every producer against its independently derived inputs.
    Nonimplementation statements cannot establish implementation support.

    Preconditions:
        ``sources`` is current and producer keys match their result item keys.
    Postconditions:
        Every inherited implementation endpoint terminates at a valid current
        claim, citation, source slice, or consumed statement.
    Essential behavior:
        Uses current source membership shared by both source adapters.
    """
    sources.check_current()
    # The visited set belongs to this immutable endpoint map and this call only.
    producers = MappingProxyType(dict(producers))
    _require(type(results) is tuple, "invalid_prose_roots")
    for result in results:
        _require(
            type(result)
            in {PlanningCardResult, SectionResult, DocumentResult},
            "invalid_prose_family",
        )
        type(result).model_validate(result)
    state = _state(sources)
    visited: set[StatementRef] = set()
    active: set[StatementRef] = set()
    pending = [
        (s, None, False)
        for result in results
        for _, s in statement_entries(result)
    ]
    while pending:
        statement, identity, leaving = pending.pop()
        if leaving:
            active.remove(identity)
            visited.add(identity)
            continue
        if identity is not None:
            _require(identity not in active, "support_reference_cycle")
            if identity in visited:
                continue
            active.add(identity)
            pending.append((statement, identity, True))
        _require(
            all(c in state.claim_records for c in statement.claims)
            and all(
                _has_source_reference(sources, r) for r in statement.citations
            ),
            "invalid_current_support",
        )
        if statement.kind == "implementation_claim":
            _require(
                bool(
                    statement.claims or statement.citations or statement.basis
                ),
                "unsupported_implementation",
            )
        for ref in statement.basis:
            if isinstance(ref, SourceSliceRef):
                validate_source_slice(ref, sources)
            else:
                # Shared completed branches already passed exact endpoint checks
                # in this boundary. Never reuse this set across calls.
                if ref in visited:
                    continue
                child = validate_statement_reference(ref, producers)
                pending.append((child, ref, False))
