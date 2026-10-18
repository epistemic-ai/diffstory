"""Lossless, indexed storage for complete v2 reports and annotations.

The storage graph is a representation only. Its addresses and structurally
valid records do not validate a report or authorize use of its claims. Callers
must still run the existing proof boundary against the current compiler report.

``encode_report_storage`` accepts an expanded report/annotation mapping, or
the current ``generation-storage.v1`` form embedded in it. The latter is
normalized with :func:`expanded_generation`, which is also the compatibility
oracle for explicit expansion. ``parse_report_storage`` checks a canonical
closed envelope and exposes detached indexed access. Full reconstruction is
performed only by ``expand_report_storage`` and is limited before repeated
trees are allocated.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import re
from collections.abc import Callable
from collections.abc import Iterator
from collections.abc import Mapping
from collections.abc import Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import ClassVar
from typing import NoReturn

from pydantic import BaseModel

from .narrative_contracts import DetailResult
from .narrative_contracts import DocumentResult
from .narrative_contracts import PlanningCardResult
from .narrative_contracts import SectionResult
from .narrative_contracts import SourceSliceRef
from .narrative_contracts import SpanRef
from .narrative_contracts import StatementPath
from .narrative_contracts import StatementRef
from .narrative_contracts import StoryPlan
from .narrative_contracts import SupportedStatement
from .narrative_contracts import result_key
from .narrative_contracts import result_key_field
from .narrative_manifest import GENERATION_STORAGE_VERSION
from .narrative_manifest import CurrentUsage
from .narrative_manifest import GenerationManifest
from .narrative_manifest import InheritedProvenance
from .narrative_manifest import PresentationSection
from .narrative_manifest import ProducerDescriptor
from .narrative_manifest import ResolvedCitation
from .narrative_manifest import ResultReceipt
from .narrative_manifest import V2Presentation
from .narrative_manifest import expanded_generation
from .narrative_manifest import result_digest
from .narrative_plan import _OwnedList
from .narrative_plan import _OwnedMapping
from .narrative_plan import _OwnedTuple
from .narrative_plan import canonical_bytes
from .narrative_plan import content_digest
from .narrative_plan import semantic_digest
from .narrative_protocol import LedgerParameters
from .narrative_protocol import RequestConfiguration
from .narrative_runtime import ArtifactInvocation
from .narrative_runtime import PreparedMetadata
from .narrative_runtime import RunIdentity
from .narrative_storage import encode_invocation

STORAGE_VERSION = "diffstory.report-storage.v1"
_ID_PATTERN = re.compile(r"^[a-f0-9]{64}$")
_LEGACY_ID_PATTERN = re.compile(r"^[a-f0-9]{16}$")
_TABLES = (
    "results",
    "plans",
    "evidence_refs",
    "source_targets",
    "resolved_citations",
    "urls",
    "runs",
    "invocations",
    "requests",
    "producers",
    "receipts",
    "readers",
    "inherited_provenance",
)
_TABLE_RECORD_KINDS = {
    "results": {"detail", "planning_card", "section", "document"},
    "plans": {"story_plan"},
    "evidence_refs": {"span_ref", "source_slice_ref", "statement_ref"},
    "source_targets": {"source_target"},
    "resolved_citations": {"resolved_citation"},
    "urls": {"url"},
    "runs": {"run"},
    "invocations": {"invocation"},
    "requests": {"request"},
    "producers": {"producer"},
    "receipts": {"receipt"},
    "readers": {"reader"},
    "inherited_provenance": {"inherited_provenance"},
}
_RESULT_MODELS = {
    "detail": DetailResult,
    "planning_card": PlanningCardResult,
    "section": SectionResult,
    "document": DocumentResult,
}
_MAX_INPUT_BYTES = 256 * 1024 * 1024
_MAX_NESTING = 128
_MAX_RECORDS = 1_000_000
_JSON_BACKSLASH = ord("\\")
_JSON_QUOTE = ord('"')
_JSON_OPEN_DELIMITERS = frozenset({ord("["), ord("{")})
_JSON_CLOSE_DELIMITERS = frozenset({ord("]"), ord("}")})
_UNVISITED = 0
_ACTIVE = 1
_DONE = 2


def _fail(reason: str) -> NoReturn:
    raise ValueError(reason)


def _strict_json_value(value: object, *, depth: int = 0) -> None:  # noqa: C901 # Reject unsupported JSON primitives and nesting.
    """Reject Python values that cannot represent exact JSON primitives."""
    if depth > _MAX_NESTING:
        _fail("storage_json_too_deep")
    if isinstance(value, BaseModel):
        _strict_json_value(value.model_dump(mode="json"), depth=depth + 1)
        return
    if value is None or type(value) in {str, bool, int}:
        return
    if type(value) is float:
        if not math.isfinite(value):
            _fail("storage_non_finite_number")
        return
    if type(value) in {list, tuple}:
        for item in value:
            _strict_json_value(item, depth=depth + 1)
        return
    if type(value) is dict:
        for key, item in value.items():
            if type(key) is not str:
                _fail("storage_non_string_key")
            _strict_json_value(item, depth=depth + 1)
        return
    _fail("storage_non_json_value")


def _freeze(value: object) -> object:
    if isinstance(value, BaseModel):
        return _freeze(value.model_dump(mode="json"))
    if type(value) is dict:
        return MappingProxyType(
            {key: _freeze(item) for key, item in value.items()}
        )
    if type(value) in {list, tuple}:
        return tuple(_freeze(item) for item in value)
    return value


def _capture_import_base_report(value: object) -> _OwnedMapping:
    """Freeze one imported compiler base into the session's report views.

    Preconditions:
        ``value`` is the strict JSON object from a fresh canonical report
        storage parse, with only JSON values and string keys.
    Postconditions:
        Returns one recursively immutable ``_OwnedMapping`` report tree.
    Essential behavior:
        Makes the parser's base the exact object that ``_CapturedInput`` owns,
        without a later thaw and recapture of this complete tree.
    """
    captured = _capture_import_base_value(value)
    if type(captured) is not _OwnedMapping:
        _fail("invalid_report_base")
    return captured


def _capture_import_base_value(value: object) -> object:
    """Freeze one child of a fresh compiler-base JSON object.

    Preconditions:
        ``value`` is a strict JSON primitive, object, or array.
    Postconditions:
        Returns immutable report traversal views while preserving list shape.
    Essential behavior:
        Allocates each nested compiler-base container exactly once.
    """
    if type(value) is dict:
        entries: list[tuple[str, object]] = []
        for key, item in value.items():
            if type(key) is not str:
                _fail("invalid_report_base")
            entries.append((key, _capture_import_base_value(item)))
        return _OwnedMapping(tuple(entries))
    if type(value) is list:
        return _OwnedList(
            tuple(_capture_import_base_value(item) for item in value)
        )
    if type(value) is tuple:
        return _OwnedTuple(
            tuple(_capture_import_base_value(item) for item in value)
        )
    if value is None or type(value) in {bool, int, float, str}:
        return value
    _fail("invalid_report_base")
    return None


def _freeze_graph_root(value: dict) -> Mapping[str, object]:
    """Freeze a small graph root while sharing a captured compiler base.

    Preconditions:
        ``value`` passed arena root checks, and any ``_OwnedMapping`` base is
        the exact immutable compiler report owned by the current session.
    Postconditions:
        Root selectors are frozen, while the already immutable base object is
        retained by identity instead of copied into a second report tree.
    Essential behavior:
        Keeps the private application handoff to one captured base plus a
        small overlay.
    """
    if (
        value.get("shape") == "report"
        and type(value.get("base_report")) is _OwnedMapping
    ):
        return MappingProxyType(
            {
                "shape": "report",
                "base_report": value["base_report"],
                "generation": _freeze(value["generation"]),
                "reader": _freeze(value["reader"]),
            }
        )
    frozen = _freeze(value)
    if not isinstance(frozen, Mapping):
        _fail("invalid_storage_root")
    return frozen


def _thaw(value: object) -> object:
    if isinstance(value, Mapping):
        return {key: _thaw(item) for key, item in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return [_thaw(item) for item in value]
    return value


def _model_snapshot(value: BaseModel) -> dict:
    """Return a detached JSON snapshot that keeps explicitly absent defaults absent."""
    snapshot = value.model_dump(mode="json", exclude_unset=True)
    if type(snapshot) is not dict:
        _fail("invalid_typed_storage_record")
    _strict_json_value(snapshot)
    return snapshot


def _raw_json_snapshot(value: object) -> object:
    """Copy typed values without allowing serializers to coerce their scalars."""
    if isinstance(value, BaseModel):
        return _raw_model_snapshot(value)
    if type(value) in {list, tuple}:
        return [_raw_json_snapshot(item) for item in value]
    if isinstance(value, Mapping):
        result = {key: _raw_json_snapshot(item) for key, item in value.items()}
        _strict_json_value(result)
        return result
    if value is None or type(value) in {str, bool, int, float}:
        _strict_json_value(value)
        return value
    _fail("invalid_typed_storage_record")


def _raw_model_snapshot(value: BaseModel) -> dict:
    """Capture every declared field, including defaults, from raw attributes."""
    if not isinstance(value, BaseModel):
        _fail("invalid_typed_storage_record")
    fields = type(value).model_fields
    if set(vars(value)) - set(fields):
        _fail("invalid_typed_storage_record")
    snapshot = {
        name: _raw_json_snapshot(getattr(value, name)) for name in fields
    }
    _strict_json_value(snapshot)
    return snapshot


def _json_snapshot(value: object) -> object:
    """Copy one typed field to JSON primitives without making a report tree."""
    return _raw_json_snapshot(value)


def _ref(storage_id: str, kind: str) -> dict[str, str]:
    return {"$ref": storage_id, "kind": kind}


def _model(model: type[BaseModel], value: object) -> BaseModel:
    """Validate JSON-shaped input without Python-mode tuple coercion rules."""
    return model.model_validate_json(canonical_bytes(value))


def _is_ref(value: object) -> bool:
    return type(value) is dict and "$ref" in value


def _check_reference_shape(value: object) -> tuple[str, str]:
    if (
        not isinstance(value, Mapping)
        or set(value) != {"$ref", "kind"}
        or type(value.get("$ref")) is not str
        or not _ID_PATTERN.fullmatch(value["$ref"])
        or type(value.get("kind")) is not str
    ):
        _fail("invalid_storage_reference")
    return value["$ref"], value["kind"]


def storage_address(kind: str, record: dict) -> str:
    """Return a domain-separated address for one complete stored record."""
    return hashlib.sha256(
        canonical_bytes(
            {
                "storage_version": STORAGE_VERSION,
                "kind": kind,
                "record": record,
            }
        )
    ).hexdigest()


@dataclass(frozen=True, slots=True)
class RecordRef:
    """An integrity address into one table; it carries no proof authority."""

    table: str
    kind: str
    storage_id: str

    def __post_init__(self) -> None:
        """Reject an address whose table, kind, or digest is out of contract."""
        if (
            self.table not in _TABLES
            or self.kind not in _TABLE_RECORD_KINDS[self.table]
            or type(self.storage_id) is not str
            or not _ID_PATTERN.fullmatch(self.storage_id)
        ):
            _fail("invalid_storage_record_reference")


@dataclass(frozen=True, slots=True)
class ResultRef(RecordRef):
    """Typed address of one complete shared generation result."""

    item_key: str

    def __post_init__(self) -> None:
        """Require the address and semantic item key to identify a result."""
        super(ResultRef, self).__post_init__()
        if (
            self.table != "results"
            or type(self.item_key) is not str
            or not _ID_PATTERN.fullmatch(self.item_key)
        ):
            _fail("invalid_storage_result_reference")


def _record_ref_value(
    value: object, *, table: str, kind: str, result: bool = False
) -> tuple[str, str, str, str | None]:
    """Copy a typed address into primitive fields for owned bindings.

    Preconditions:
        ``value`` has the exact reference type and table/kind requested.
    Postconditions:
        The return value contains only strict address scalars.
    Essential behavior:
        Result item bindings are copied along with the digest.
    """
    expected = ResultRef if result else RecordRef
    if type(value) is not expected:
        _fail("invalid_generation_record_reference")
    if value.table != table or value.kind != kind:
        _fail("invalid_generation_record_reference")
    item_key = value.item_key if result else None
    return (value.table, value.kind, value.storage_id, item_key)


def _record_ref_from_value(
    value: tuple[str, str, str, str | None],
) -> RecordRef:
    """Return a fresh data-only address from captured primitive fields.

    Preconditions:
        ``value`` contains a captured table, kind, digest, and optional item key.
    Postconditions:
        The result constructor rechecks every scalar and returns a fresh value.
    Essential behavior:
        No caller-owned reference object is returned or retained.
    """
    table, kind, storage_id, item_key = value
    if item_key is None:
        return RecordRef(table, kind, storage_id)
    return ResultRef(table, kind, storage_id, item_key)


def _reference_values(
    values: Sequence[object],
    *,
    table: str,
    kind: str,
    result: bool = False,
) -> tuple[tuple[str, str, str, str | None], ...]:
    """Snapshot an ordered sequence of exact typed references.

    Preconditions:
        ``values`` is a list or tuple of references with the declared binding.
    Postconditions:
        The returned primitive tuples preserve input order and list length.
    Essential behavior:
        Each item passes through exact reference type and family checks.
    """
    if type(values) not in {tuple, list}:
        _fail("invalid_generation_record_references")
    return tuple(
        _record_ref_value(item, table=table, kind=kind, result=result)
        for item in values
    )


@dataclass(frozen=True, slots=True, init=False)
class GenerationRecordSet:
    """Data-only manifest header bound to ordered arena record references.

    The value stores ``version``, ``provenance``, ``compiler_fact_signature``,
    ``current_usage``, ``ledger_digest``, ``ledger_parameters``,
    ``request_configuration``, and ``unit_keys`` in ``header``. The default
    fields ``version`` and ``provenance`` may be absent to preserve caller field
    presence. ``run_ref`` names the run; the result groups bind ``details``,
    ``plan``, ``planning``, ``sections``, ``document``, ``producers``, and
    ``receipts`` in the same order as ``GenerationManifest``.

    This type owns a primitive header snapshot and copies reference fields. Its
    addresses are data only and do not prove generation completeness.

    Attributes:
        header: Immutable exact manifest-header fields and ordered unit keys.
        run_ref: Run record address, returned as a fresh reference value.
        details: Ordered detail result addresses.
        plan_ref: Story plan record address.
        planning: Ordered planning-card result addresses.
        sections: Ordered section result addresses.
        document: Document result address.
        producers: Ordered producer record addresses.
        receipts: Ordered receipt record addresses.
    """

    _header: Mapping[str, object]
    _header_witness: str
    _run_ref: tuple[str, str, str, str | None]
    _details: tuple[tuple[str, str, str, str | None], ...]
    _plan_ref: tuple[str, str, str, str | None]
    _planning: tuple[tuple[str, str, str, str | None], ...]
    _sections: tuple[tuple[str, str, str, str | None], ...]
    _document: tuple[str, str, str, str | None]
    _producers: tuple[tuple[str, str, str, str | None], ...]
    _receipts: tuple[tuple[str, str, str, str | None], ...]

    def __init__(  # noqa: PLR0913 # Keep the generation's typed ordered bindings explicit.
        self,
        header: Mapping[str, object],
        *,
        run_ref: RecordRef,
        details: Sequence[ResultRef],
        plan_ref: RecordRef,
        planning: Sequence[ResultRef],
        sections: Sequence[ResultRef],
        document: ResultRef,
        producers: Sequence[RecordRef],
        receipts: Sequence[RecordRef],
    ) -> None:
        """Capture strict header primitives and ordered references.

        Preconditions:
            ``header`` is a mapping with only the manifest-header fields listed
            in the class contract. Result and audit references have their
            declared table and kind. Each sequence is a list or tuple.
        Postconditions:
            Nested header values and every reference field are detached from
            caller-owned objects. Input ordering and field presence are kept.
        Essential behavior:
            Validates scalar and nested header contracts without constructing a
            ``GenerationManifest``; closure and arena ownership remain for seal.
        """
        snapshot = _raw_json_snapshot(header)
        if type(snapshot) is not dict:
            _fail("invalid_generation_header")
        allowed = {
            "version",
            "provenance",
            "compiler_fact_signature",
            "current_usage",
            "ledger_digest",
            "ledger_parameters",
            "request_configuration",
            "unit_keys",
        }
        required = allowed - {"version", "provenance"}
        if set(snapshot) - allowed or not required <= set(snapshot):
            _fail("invalid_generation_header")
        if (
            "version" in snapshot
            and snapshot["version"] != "diffstory.generation.v2"
        ):
            _fail("invalid_generation_header")
        if "provenance" in snapshot and snapshot["provenance"] != "unverified":
            _fail("invalid_generation_header")
        for name in ("compiler_fact_signature", "ledger_digest"):
            if type(snapshot[name]) is not str or not _ID_PATTERN.fullmatch(
                snapshot[name]
            ):
                _fail("invalid_generation_header")
        unit_keys = snapshot["unit_keys"]
        if type(unit_keys) is not list or any(
            type(item) is not str or not _ID_PATTERN.fullmatch(item)
            for item in unit_keys
        ):
            _fail("invalid_generation_header")
        _model(CurrentUsage, snapshot["current_usage"])
        _model(LedgerParameters, snapshot["ledger_parameters"])
        _model(RequestConfiguration, snapshot["request_configuration"])

        captured_header = _freeze(snapshot)
        object.__setattr__(self, "_header", captured_header)
        object.__setattr__(
            self, "_header_witness", content_digest(canonical_bytes(snapshot))
        )
        object.__setattr__(
            self,
            "_run_ref",
            _record_ref_value(run_ref, table="runs", kind="run"),
        )
        object.__setattr__(
            self,
            "_details",
            _reference_values(
                details, table="results", kind="detail", result=True
            ),
        )
        object.__setattr__(
            self,
            "_plan_ref",
            _record_ref_value(plan_ref, table="plans", kind="story_plan"),
        )
        object.__setattr__(
            self,
            "_planning",
            _reference_values(
                planning,
                table="results",
                kind="planning_card",
                result=True,
            ),
        )
        object.__setattr__(
            self,
            "_sections",
            _reference_values(
                sections, table="results", kind="section", result=True
            ),
        )
        object.__setattr__(
            self,
            "_document",
            _record_ref_value(
                document,
                table="results",
                kind="document",
                result=True,
            ),
        )
        object.__setattr__(
            self,
            "_producers",
            _reference_values(producers, table="producers", kind="producer"),
        )
        object.__setattr__(
            self,
            "_receipts",
            _reference_values(receipts, table="receipts", kind="receipt"),
        )

    @property
    def header(self) -> Mapping[str, object]:
        """Return immutable primitive header fields, preserving presence.

        Preconditions:
            The record set was constructed with a strict manifest header.
        Postconditions:
            The returned mapping and nested values remain immutable.
        Essential behavior:
            Exposes only the captured data-only header, without a manifest.
        """
        return self._header

    @property
    def run_ref(self) -> RecordRef:
        """Return the captured run address as a fresh value.

        Preconditions:
            The record set has an immutable run binding.
        Postconditions:
            The returned value is independent of caller references.
        Essential behavior:
            Reconstructs only the small typed address.
        """
        return _record_ref_from_value(self._run_ref)

    @property
    def details(self) -> tuple[ResultRef, ...]:
        """Return detail result references in captured order.

        Preconditions:
            The record set has immutable detail address fields.
        Postconditions:
            Fresh typed references preserve the original order and bindings.
        Essential behavior:
            Materializes addresses only, not result records.
        """
        return tuple(_record_ref_from_value(item) for item in self._details)

    @property
    def plan_ref(self) -> RecordRef:
        """Return the captured plan address as a fresh value.

        Preconditions:
            The record set has an immutable plan binding.
        Postconditions:
            The returned reference is independent of caller values.
        Essential behavior:
            Reconstructs only the small typed address.
        """
        return _record_ref_from_value(self._plan_ref)

    @property
    def planning(self) -> tuple[ResultRef, ...]:
        """Return planning result references in captured order.

        Preconditions:
            The record set has immutable planning address fields.
        Postconditions:
            Fresh typed references preserve the original order and bindings.
        Essential behavior:
            Materializes addresses only, not result records.
        """
        return tuple(_record_ref_from_value(item) for item in self._planning)

    @property
    def sections(self) -> tuple[ResultRef, ...]:
        """Return section result references in captured order.

        Preconditions:
            The record set has immutable section address fields.
        Postconditions:
            Fresh typed references preserve the original order and bindings.
        Essential behavior:
            Materializes addresses only, not section records.
        """
        return tuple(_record_ref_from_value(item) for item in self._sections)

    @property
    def document(self) -> ResultRef:
        """Return the captured document address as a fresh value.

        Preconditions:
            The record set has an immutable document binding.
        Postconditions:
            The returned reference is independent of caller values.
        Essential behavior:
            Materializes only the small typed address.
        """
        return _record_ref_from_value(self._document)

    @property
    def producers(self) -> tuple[RecordRef, ...]:
        """Return producer references in captured order.

        Preconditions:
            The record set has immutable producer address fields.
        Postconditions:
            Fresh typed references preserve the original order and bindings.
        Essential behavior:
            Materializes addresses only, not producer audit records.
        """
        return tuple(_record_ref_from_value(item) for item in self._producers)

    @property
    def receipts(self) -> tuple[RecordRef, ...]:
        """Return receipt references in captured order.

        Preconditions:
            The record set has immutable receipt address fields.
        Postconditions:
            Fresh typed references preserve the original order and bindings.
        Essential behavior:
            Materializes addresses only, not receipt audit records.
        """
        return tuple(_record_ref_from_value(item) for item in self._receipts)


@dataclass(frozen=True, slots=True)
class ReaderSectionRef:
    """Select one ordered reader section and its exact result memberships.

    Attributes:
        group_id: Stable section identifier used by the reader's ``next_id``.
        group_key: Plan group key bound to the section prose result.
        number: One-based visible section order.
        next_id: Next section identifier, or ``None`` for the final section.
        prose: Owned section result reference.
        details: Ordered detail result references assigned to this section.
    """

    group_id: str
    group_key: str
    number: int
    next_id: str | None
    prose: ResultRef
    details: tuple[ResultRef, ...]


@dataclass(frozen=True, slots=True)
class IndexedReaderProjection:
    """Ordered reader selectors over results and audit records owned by an arena.

    ``shape`` is ``presentation`` for reports or ``annotations`` for annotation
    artifacts. The document, section prose, section details, context details,
    citations, and support provenance fields are exact typed storage references
    in their source order. Presentation ``version`` and ``provenance`` default
    at seal when absent; annotations do not use those fields. Values are
    selectors only and do not establish reader or generation proof.

    Attributes:
        shape: Reader family bound to the artifact root.
        document: Owned document result address.
        sections: Ordered prose section selectors and their detail membership.
        context_details: Ordered details not assigned to a narrative section.
        citations: Ordered resolved citation record addresses.
        support_provenance: Ordered inherited statement record addresses.
        version: Optional presentation schema version.
        provenance: Optional presentation provenance label.
    """

    shape: str
    document: ResultRef
    sections: tuple[ReaderSectionRef, ...]
    context_details: tuple[ResultRef, ...]
    citations: tuple[RecordRef, ...]
    support_provenance: tuple[RecordRef, ...]
    version: str | None = None
    provenance: str | None = None


@dataclass(frozen=True, slots=True)
class ReportReaderProjection:
    """Borrowed typed reader inputs consumed while constructing a graph.

    The input models are not retained. ``ReportRecordGraph.from_typed`` copies
    each complete record into immutable JSON primitives before it returns.
    """

    shape: str
    document: DocumentResult
    sections: tuple[PresentationSection, ...]
    context_details: tuple[DetailResult, ...]
    citations: tuple[ResolvedCitation, ...]
    support_provenance: tuple[InheritedProvenance, ...]
    version: str | None = None
    provenance: str | None = None

    @classmethod
    def from_presentation(
        cls, presentation: V2Presentation
    ) -> ReportReaderProjection:
        """Create typed input from a complete v2 presentation model."""
        if not isinstance(presentation, V2Presentation):
            _fail("reader_projection_requires_typed_presentation")
        return cls(
            shape="presentation",
            document=presentation.document,
            sections=presentation.sections,
            context_details=presentation.context_details,
            citations=presentation.citations,
            support_provenance=presentation.support_provenance,
            version=presentation.version,
            provenance=presentation.provenance,
        )

    @classmethod
    def from_annotations(
        cls,
        *,
        document: DocumentResult,
        steps: tuple[PresentationSection, ...],
        context_details: tuple[DetailResult, ...],
        citations: tuple[ResolvedCitation, ...],
        support_provenance: tuple[InheritedProvenance, ...],
    ) -> ReportReaderProjection:
        """Create typed reader input for an annotations.v2 artifact."""
        return cls(
            shape="annotations",
            document=document,
            sections=steps,
            context_details=context_details,
            citations=citations,
            support_provenance=support_provenance,
        )


class _Builder:
    """Build typed records bottom-up and share exact canonical content."""

    def __init__(
        self,
        *,
        defer_validation: bool = False,
        retain_integrity_copies: bool = True,
        freeze_records: bool = False,
    ) -> None:
        """Initialize an append-only bottom-up table builder.

        Preconditions:
            Flags describe whether callers defer model validation, retain
            collision copies, or freeze primitive rows after insertion.
        Postconditions:
            Empty table indexes and audit maps are ready for new records.
        Essential behavior:
            The default mode keeps the parser's independent integrity copies;
            the private arena mode can disable those duplicate buffers.
        """
        self.defer_validation = defer_validation
        self.retain_integrity_copies = retain_integrity_copies
        self.freeze_records = freeze_records
        self.records: dict[str, dict[str, dict[str, object]]] = {
            name: {} for name in _TABLES
        }
        self.address_contents: dict[str, dict[str, bytes]] = (
            {name: {} for name in _TABLES} if retain_integrity_copies else {}
        )
        self.logical_records: dict[str, dict] = {}
        self.record_count = 0
        self.result_ids: dict[tuple[str, str], str] = {}
        self.typed_results: dict[str, BaseModel] = {}
        self.receipt_ids: dict[str, str] = {}

    def _capture_record(
        self, model: type[BaseModel], value: dict | BaseModel
    ) -> tuple[dict, BaseModel | None]:
        if isinstance(value, model):
            record = _raw_model_snapshot(value)
        elif type(value) is dict:
            record = copy.deepcopy(value)
            _strict_json_value(record)
        else:
            _fail("invalid_typed_storage_record")
        typed = None if self.defer_validation else _model(model, record)
        return record, typed

    @staticmethod
    def _array(record: dict, field: str) -> list | tuple:
        value = record.get(field, [])
        if type(value) not in {list, tuple}:
            _fail("invalid_typed_storage_record")
        return value

    @staticmethod
    def _model_array(value: object) -> tuple | list:
        if type(value) not in {tuple, list}:
            _fail("invalid_typed_storage_record")
        return value

    def put(
        self,
        table: str,
        kind: str,
        record: dict,
        *,
        address_record: dict | None = None,
        stream_address: bool = False,
    ) -> str:
        """Intern one typed table row by complete domain-separated content.

        Preconditions:
            ``record`` and optional ``address_record`` contain strict JSON
            primitives; ``table`` and ``kind`` are an allowed table contract.
        Postconditions:
            Exact duplicates return the existing address. New rows are stored
            once, optionally as recursively immutable primitives.
        Essential behavior:
            Existing parser builders retain canonical collision bytes. Arena
            builders compare against the owned logical row without retaining a
            second complete canonical record buffer.
        """
        if table not in self.records or kind not in _TABLE_RECORD_KINDS[table]:
            _fail("invalid_storage_record_kind")
        _strict_json_value(record)
        if stream_address:
            if table != "readers":
                _fail("invalid_streamed_storage_record")
            identifier = _streamed_storage_address(
                kind,
                _reader_logical_chunks(record, self.records),
            )
            encoded_complete = None
        else:
            complete = record if address_record is None else address_record
            _strict_json_value(complete)
            identifier = storage_address(kind, complete)
            encoded_complete = canonical_bytes(complete)
        prior = self.records[table].get(identifier)
        if prior is not None:
            if self.retain_integrity_copies:
                same_logical_content = (
                    _reader_records_stream_equal(
                        prior["record"], record, self.records
                    )
                    if stream_address
                    else self.address_contents[table][identifier]
                    == encoded_complete
                )
            elif stream_address:
                same_logical_content = _reader_records_stream_equal(
                    prior["record"], record, self.records
                )
            else:
                prior_logical = _arena_logical_record(
                    table, identifier, self.records
                )
                same_logical_content = canonical_bytes(
                    prior_logical
                ) == canonical_bytes(complete)
            if (
                prior["kind"] != kind
                or not same_logical_content
                or canonical_bytes(_thaw(prior["record"]))
                != canonical_bytes(record)
            ):
                _fail("storage_address_collision")
            return identifier
        if self.record_count >= _MAX_RECORDS:
            _fail("too_many_storage_records")
        stored_record = _freeze(record) if self.freeze_records else record
        entry = {
            "id": identifier,
            "kind": kind,
            "record": stored_record,
        }
        self.records[table][identifier] = (
            MappingProxyType(entry) if self.freeze_records else entry
        )
        if not stream_address and self.retain_integrity_copies:
            self.address_contents[table][identifier] = encoded_complete
            self.logical_records[f"{table}:{identifier}"] = complete
        self.record_count += 1
        return identifier

    def add_span(self, value: dict | SpanRef) -> dict[str, str]:
        value, _ = self._capture_record(SpanRef, value)
        return _ref(
            self.put("evidence_refs", "span_ref", copy.deepcopy(value)),
            "span_ref",
        )

    def add_evidence(
        self, value: dict | SpanRef | SourceSliceRef | StatementRef, kind: str
    ) -> dict[str, str]:
        if kind == "span_ref":
            return self.add_span(value)
        if kind == "source_slice_ref":
            value, _ = self._capture_record(SourceSliceRef, value)
            source = value.get("source")
            if type(source) is not dict:
                _fail("invalid_typed_storage_record")
            record = copy.deepcopy(value)
            record["source"] = self.add_span(source)
            return _ref(
                self.put("evidence_refs", kind, record, address_record=value),
                kind,
            )
        if kind == "statement_ref":
            value, _ = self._capture_record(StatementRef, value)
            return _ref(
                self.put("evidence_refs", kind, copy.deepcopy(value)), kind
            )
        _fail("invalid_evidence_reference_kind")

    def add_statement(
        self, value: dict | SupportedStatement
    ) -> dict[str, str]:
        value, _ = self._capture_record(SupportedStatement, value)
        record = copy.deepcopy(value)
        if "citations" in record:
            record["citations"] = [
                self.add_evidence(citation, "span_ref")
                for citation in self._array(record, "citations")
            ]
        if "basis" in record:
            record["basis"] = [
                self.add_evidence(
                    basis,
                    "source_slice_ref"
                    if type(basis) is dict
                    and basis.get("kind") == "source_slice"
                    else "statement_ref",
                )
                for basis in self._array(record, "basis")
            ]
        return record  # The containing result receives the table address.

    def add_result(  # noqa: C901, PLR0912 # Build one typed result table record bottom-up.
        self, kind: str, value: dict | BaseModel
    ) -> dict[str, str]:
        """Store one complete result and intern all typed nested references.

        Preconditions:
            ``kind`` selects a supported result model and ``value`` is an
            exact model or strict JSON-shaped record.
        Postconditions:
            The result row and its evidence edges have stable content addresses;
            duplicate semantic identities must have identical full content.
        Essential behavior:
            Preserve the caller's logical field presence for its address while
            storing child references in shared tables.
        """
        model = _RESULT_MODELS.get(kind)
        if model is None:
            _fail("invalid_result_kind")
        value, typed = self._capture_record(model, value)
        logical_id = storage_address(kind, value)
        key_field = result_key_field(kind)
        if key_field not in value or type(value[key_field]) is not str:
            _fail("invalid_typed_storage_record")
        pair = (kind, value[key_field])
        prior = self.result_ids.get(pair)
        if prior is not None:
            if prior != logical_id or not self._same_logical_content(
                "results", prior, value
            ):
                _fail("duplicate_result_identity")
            return _ref(prior, kind)
        record = copy.deepcopy(value)
        if kind == "detail":
            for family in ("explanations", "observations", "questions"):
                for item in self._array(record, family):
                    if type(item) is not dict:
                        _fail("invalid_typed_storage_record")
                    item["citations"] = [
                        self.add_evidence(citation, "span_ref")
                        for citation in self._array(item, "citations")
                    ]
        else:
            statement_fields = {
                "planning_card": ("abstraction",),
                "section": ("orientation", "synthesis", "transition"),
                "document": ("preamble", "lead", "closing"),
            }[kind]
            if kind == "section":
                record["title"] = self.add_statement(record["title"])
            for field in statement_fields:
                record[field] = [
                    self.add_statement(item)
                    for item in self._array(record, field)
                ]
            for terminology in self._array(record, "terminology"):
                if type(terminology) is not dict:
                    _fail("invalid_typed_storage_record")
                terminology["definition"] = self.add_statement(
                    terminology["definition"]
                )
        identifier = self.put("results", kind, record, address_record=value)
        self.result_ids[pair] = identifier
        if typed is not None:
            self.typed_results[identifier] = typed
        return _ref(identifier, kind)

    def _same_logical_content(
        self, table: str, identifier: str, incoming: dict
    ) -> bool:
        """Compare a reused address with its complete caller content.

        Preconditions:
            ``identifier`` names a row in ``table`` and ``incoming`` is its
            strict expanded logical record.
        Postconditions:
            Returns true only when complete logical values match.
        Essential behavior:
            Arena mode streams from owned rows and retains no second record
            buffer; parser mode uses its independent canonical snapshot.
        """
        if self.retain_integrity_copies:
            return self.address_contents[table][identifier] == canonical_bytes(
                incoming
            )
        return canonical_bytes(
            _arena_logical_record(table, identifier, self.records)
        ) == canonical_bytes(incoming)

    def add_citation(self, value: dict | ResolvedCitation) -> dict[str, str]:
        value, citation = self._capture_record(ResolvedCitation, value)
        record = copy.deepcopy(value)
        reference = record["reference"]
        if type(reference) is not dict:
            _fail("invalid_typed_storage_record")
        ref_kind = (
            "source_slice_ref"
            if reference.get("kind") == "source_slice"
            else "span_ref"
        )
        record["reference"] = self.add_evidence(reference, ref_kind)
        target = {
            name: record.pop(name)
            for name in ("fragment_key", "side", "path", "revision")
        }
        record["source_target"] = _ref(
            self.put("source_targets", "source_target", target),
            "source_target",
        )
        url = record.pop("url")
        record["url"] = _ref(self.put("urls", "url", {"value": url}), "url")
        if isinstance(citation, ResolvedCitation):
            _validate_citation_binding(citation, value)
        return _ref(
            self.put(
                "resolved_citations",
                "resolved_citation",
                record,
                address_record=value,
            ),
            "resolved_citation",
        )

    def add_run(self, value: dict | RunIdentity) -> dict[str, str]:
        value, _ = self._capture_record(RunIdentity, value)
        return _ref(self.put("runs", "run", copy.deepcopy(value)), "run")

    def add_receipt(self, value: dict | ResultReceipt) -> dict[str, str]:
        value, _ = self._capture_record(ResultReceipt, value)
        identifier = self.put("receipts", "receipt", copy.deepcopy(value))
        item_key = value.get("item_key")
        if type(item_key) is not str:
            _fail("invalid_typed_storage_record")
        prior = self.receipt_ids.get(item_key)
        if prior is not None and prior != identifier:
            _fail("duplicate_receipt_identity")
        self.receipt_ids[item_key] = identifier
        return _ref(identifier, "receipt")

    def add_producer(self, value: dict | ProducerDescriptor) -> dict[str, str]:
        value, _ = self._capture_record(ProducerDescriptor, value)
        record = copy.deepcopy(value)
        original_run = value.get("original_run")
        invocation = value.get("invocation")
        requests = value.get("requests")
        if (
            type(original_run) is not dict
            or type(invocation) is not dict
            or type(requests) not in {list, tuple}
        ):
            _fail("invalid_typed_storage_record")
        record["original_run"] = self.add_run(original_run)
        record["invocation"] = _ref(
            self.put("invocations", "invocation", copy.deepcopy(invocation)),
            "invocation",
        )
        record["requests"] = [
            _ref(
                self.put(
                    "requests",
                    "request",
                    copy.deepcopy(request),
                ),
                "request",
            )
            for request in requests
        ]
        return _ref(
            self.put("producers", "producer", record, address_record=value),
            "producer",
        )

    def add_generation(self, value: dict) -> dict:  # noqa: C901 # Convert typed manifest fields to references.
        """Create a generation skeleton while preserving typed defaults."""
        if isinstance(value, GenerationManifest):
            manifest = value
            generation: dict = {}
            for name in GenerationManifest.model_fields:
                item = getattr(manifest, name)
                if name == "run":
                    generation[name] = self.add_run(item)
                elif name == "details":
                    generation[name] = [
                        self.add_result("detail", result)
                        for result in self._model_array(item)
                    ]
                elif name == "plan":
                    generation[name] = _ref(
                        self.put(
                            "plans",
                            "story_plan",
                            _raw_model_snapshot(item),
                        ),
                        "story_plan",
                    )
                elif name in {"planning", "sections"}:
                    kind = "planning_card" if name == "planning" else "section"
                    generation[name] = [
                        self.add_result(kind, result)
                        for result in self._model_array(item)
                    ]
                elif name == "document":
                    generation[name] = self.add_result("document", item)
                elif name == "producers":
                    generation[name] = [
                        self.add_producer(producer)
                        for producer in self._model_array(item)
                    ]
                elif name == "receipts":
                    generation[name] = [
                        self.add_receipt(receipt)
                        for receipt in self._model_array(item)
                    ]
                else:
                    generation[name] = _json_snapshot(item)
            return generation
        generation = copy.deepcopy(value)
        generation["run"] = self.add_run(generation["run"])
        generation["details"] = [
            self.add_result("detail", item) for item in generation["details"]
        ]
        generation["plan"] = _ref(
            self.put("plans", "story_plan", generation["plan"]), "story_plan"
        )
        for kind, field in (
            ("planning_card", "planning"),
            ("section", "sections"),
        ):
            generation[field] = [
                self.add_result(kind, item) for item in generation[field]
            ]
        generation["document"] = self.add_result(
            "document", generation["document"]
        )
        generation["producers"] = [
            self.add_producer(item) for item in generation["producers"]
        ]
        generation["receipts"] = [
            self.add_receipt(item) for item in generation["receipts"]
        ]
        return generation

    def find_result(self, kind: str, key: str) -> dict[str, str]:
        identifier = self.result_ids.get((kind, key))
        if identifier is None:
            _fail("missing_provenance_result")
        return _ref(identifier, kind)

    def add_inherited(  # noqa: C901 # Check inherited statement and receipt bindings.
        self, value: dict | InheritedProvenance
    ) -> dict[str, str]:
        value, inherited = self._capture_record(InheritedProvenance, value)
        reference = value.get("reference")
        receipt = value.get("receipt")
        if type(reference) is not dict or type(receipt) is not dict:
            _fail("invalid_typed_storage_record")
        producer_item_key = reference.get("producer_item_key")
        receipt_item_key = receipt.get("item_key")
        if (
            type(producer_item_key) is not str
            or type(receipt_item_key) is not str
        ):
            _fail("invalid_typed_storage_record")
        if isinstance(inherited, InheritedProvenance) and (
            inherited.reference.producer_item_key != inherited.receipt.item_key
        ):
            _fail("reader_provenance_mismatch")
        result_id = self.result_ids.get(("planning_card", producer_item_key))
        result_kind_value = "planning_card"
        if result_id is None:
            for candidate_kind in ("section", "document"):
                result_id = self.result_ids.get(
                    (candidate_kind, producer_item_key)
                )
                if result_id is not None:
                    result_kind_value = candidate_kind
                    break
        if result_id is None:
            _fail("missing_provenance_result")
        result_pointer = _ref(result_id, result_kind_value)
        receipt_id = self.receipt_ids.get(receipt_item_key)
        if receipt_id is None:
            _fail("missing_provenance_receipt")
        if not self.defer_validation:
            typed_result = self.typed_results.get(result_id)
            if typed_result is None:
                _fail("missing_provenance_result")
            statement = _statement_from_path(
                {"version": typed_result.version},
                reference,
                typed_result=typed_result,
                typed_reference=(
                    inherited.reference
                    if isinstance(inherited, InheritedProvenance)
                    else None
                ),
            )
            if canonical_bytes(statement) != canonical_bytes(
                value["statement"]
            ):
                _fail("inherited_statement_mismatch")
            stored_receipt = self.records["receipts"][receipt_id]["record"]
            if canonical_bytes(stored_receipt) != canonical_bytes(
                value["receipt"]
            ):
                _fail("inherited_receipt_mismatch")
        evidence_value = (
            inherited.reference
            if isinstance(inherited, InheritedProvenance)
            else reference
        )
        ref_pointer = self.add_evidence(evidence_value, "statement_ref")
        record = {
            "key": value.get("key"),
            "reference": ref_pointer,
            "result": result_pointer,
            "consumption": "inherited_statement",
            "receipt": _ref(receipt_id, "receipt"),
        }
        return _ref(
            self.put(
                "inherited_provenance",
                "inherited_provenance",
                record,
                address_record=value,
            ),
            "inherited_provenance",
        )

    def add_reader_projection(
        self, projection: ReportReaderProjection
    ) -> dict[str, str]:
        """Build a reader table record directly from typed ordered inputs."""
        if projection.shape not in {"presentation", "annotations"}:
            _fail("invalid_reader_shape")
        arrays = (
            projection.sections,
            projection.context_details,
            projection.citations,
            projection.support_provenance,
        )
        if any(type(value) not in {tuple, list} for value in arrays):
            _fail("reader_projection_requires_typed_records")
        if (
            not isinstance(projection.document, DocumentResult)
            or any(
                not isinstance(section, PresentationSection)
                for section in projection.sections
            )
            or any(
                not isinstance(detail, DetailResult)
                for detail in projection.context_details
            )
            or any(
                not isinstance(citation, ResolvedCitation)
                for citation in projection.citations
            )
            or any(
                not isinstance(item, InheritedProvenance)
                for item in projection.support_provenance
            )
        ):
            _fail("reader_projection_requires_typed_records")
        section_field = (
            "sections" if projection.shape == "presentation" else "steps"
        )
        if projection.shape == "presentation":
            record = {
                "shape": "presentation",
                "version": projection.version,
                "provenance": projection.provenance,
                "document": self.add_result("document", projection.document),
                "sections": [],
                "context_details": [
                    self.add_result("detail", detail)
                    for detail in self._model_array(projection.context_details)
                ],
                "citations": [
                    self.add_citation(citation)
                    for citation in self._model_array(projection.citations)
                ],
                "support_provenance": [
                    self.add_inherited(item)
                    for item in self._model_array(
                        projection.support_provenance
                    )
                ],
            }
        else:
            record = {
                "shape": "annotations",
                "document": self.add_result("document", projection.document),
                "steps": [],
                "context_details": [
                    self.add_result("detail", detail)
                    for detail in self._model_array(projection.context_details)
                ],
                "citations": [
                    self.add_citation(citation)
                    for citation in self._model_array(projection.citations)
                ],
                "support_provenance": [
                    self.add_inherited(item)
                    for item in self._model_array(
                        projection.support_provenance
                    )
                ],
            }
        for section in self._model_array(projection.sections):
            record[section_field].append(
                {
                    "group_id": section.group_id,
                    "group_key": section.group_key,
                    "number": section.number,
                    "next_id": section.next_id,
                    "prose": self.add_result("section", section.prose),
                    "details": [
                        self.add_result("detail", detail)
                        for detail in self._model_array(section.details)
                    ],
                }
            )
        return _ref(
            self.put("readers", "reader", record, stream_address=True),
            "reader",
        )

    def add_reader(self, value: dict, *, shape: str) -> dict[str, str]:
        if shape == "presentation":
            _model(V2Presentation, value)
            record = {
                "shape": "presentation",
                "version": value["version"],
                "provenance": value["provenance"],
                "document": self.add_result("document", value["document"]),
                "sections": [],
                "context_details": [
                    self.add_result("detail", detail)
                    for detail in value["context_details"]
                ],
                "citations": [
                    self.add_citation(citation)
                    for citation in value["citations"]
                ],
                "support_provenance": [
                    self.add_inherited(item)
                    for item in value["support_provenance"]
                ],
            }
            for section in value["sections"]:
                typed = _model(PresentationSection, section)
                record["sections"].append(
                    {
                        "group_id": typed.group_id,
                        "group_key": typed.group_key,
                        "number": typed.number,
                        "next_id": typed.next_id,
                        "prose": self.add_result("section", section["prose"]),
                        "details": [
                            self.add_result("detail", detail)
                            for detail in section["details"]
                        ],
                    }
                )
        elif shape == "annotations":
            record = {
                "shape": "annotations",
                "document": self.add_result("document", value["document"]),
                "steps": [],
                "context_details": [
                    self.add_result("detail", detail)
                    for detail in value["context_details"]
                ],
                "citations": [
                    self.add_citation(citation)
                    for citation in value["citations"]
                ],
                "support_provenance": [
                    self.add_inherited(item)
                    for item in value["support_provenance"]
                ],
            }
            for step in value["steps"]:
                typed = _model(PresentationSection, step)
                record["steps"].append(
                    {
                        "group_id": typed.group_id,
                        "group_key": typed.group_key,
                        "number": typed.number,
                        "next_id": typed.next_id,
                        "prose": self.add_result("section", step["prose"]),
                        "details": [
                            self.add_result("detail", detail)
                            for detail in step["details"]
                        ],
                    }
                )
        else:
            _fail("invalid_reader_shape")
        return _ref(
            self.put("readers", "reader", record, address_record=value),
            "reader",
        )

    def envelope(self, artifact_kind: str, root: dict) -> dict:
        return {
            "storage_version": STORAGE_VERSION,
            "artifact_kind": artifact_kind,
            "root": root,
            "tables": {
                name: [
                    self.records[name][key]
                    for key in sorted(self.records[name])
                ]
                for name in _TABLES
            },
        }


def _validate_citation_binding(citation: ResolvedCitation, raw: dict) -> None:
    expected_consumption = (
        "source_slice"
        if isinstance(citation.reference, SourceSliceRef)
        else "direct_source"
    )
    if citation.consumption != expected_consumption:
        _fail("citation_consumption_mismatch")
    expected_key = semantic_digest(citation.reference)
    if citation.key != expected_key:
        _fail("citation_key_mismatch")
    if type(raw.get("url")) not in {str, type(None)}:
        _fail("invalid_citation_url")


def _expand_generation_input(value: object) -> dict:
    if type(value) is not dict:
        _fail("invalid_generation_schema")
    expanded = expanded_generation(value)
    GenerationManifest.model_validate_json(canonical_bytes(expanded))
    return expanded


def _json_mapping(value: object, error: str) -> dict:
    if type(value) is not dict:
        _fail(error)
    return value


def encode_report_storage(value: object, artifact_kind: str) -> bytes:
    """Encode an expanded v2 report or annotation into a canonical shared graph.

    ``generation-storage.v1`` is accepted in the ``generation`` field and is
    expanded using the existing compatibility helper before graph creation.
    Compiler report/source payloads remain pass-through JSON data.
    """
    if type(artifact_kind) is not str or artifact_kind not in {
        "report",
        "annotations",
    }:
        _fail("invalid_artifact_kind")
    if isinstance(value, BaseModel):
        value = value.model_dump(mode="json")
    _strict_json_value(value)
    source = _json_mapping(value, "invalid_artifact_root")
    source_bytes = canonical_bytes(source)
    if len(source_bytes) > _MAX_INPUT_BYTES:
        _fail("storage_input_too_large")
    source = json.loads(source_bytes)
    generation = _expand_generation_input(source.get("generation"))
    builder = _Builder()
    generation_skeleton = builder.add_generation(generation)

    if artifact_kind == "report":
        if "generation" not in source or "narrative_v2" not in source:
            _fail("invalid_report_root")
        presentation = _json_mapping(source["narrative_v2"], "invalid_reader")
        V2Presentation.model_validate_json(canonical_bytes(presentation))
        reader = builder.add_reader(presentation, shape="presentation")
        base_report = {
            key: item
            for key, item in source.items()
            if key not in {"generation", "narrative_v2"}
        }
        root = {
            "shape": "report",
            "base_report": base_report,
            "generation": generation_skeleton,
            "reader": reader,
        }
    else:
        required = {
            "schema",
            "base_sha",
            "head_sha",
            "generation",
            "document",
            "steps",
            "context_details",
            "citations",
            "support_provenance",
        }
        if (
            set(source) != required
            or source.get("schema") != "diffstory.annotations.v2"
        ):
            _fail("invalid_annotations_root")
        reader_source = {
            key: source[key]
            for key in required
            - {"schema", "base_sha", "head_sha", "generation"}
        }
        reader = builder.add_reader(reader_source, shape="annotations")
        root = {
            "shape": "annotations",
            "metadata": {
                "schema": source["schema"],
                "base_sha": source["base_sha"],
                "head_sha": source["head_sha"],
            },
            "generation": generation_skeleton,
            "reader": reader,
        }
    envelope = builder.envelope(artifact_kind, root)
    encoded = canonical_bytes(envelope)
    # Run the same structural boundary used for imported files. This catches
    # builder/reader divergence before returning a new persisted artifact.
    parse_report_storage(encoded)
    return encoded


def _strict_json_load(
    raw_bytes: bytes | str, *, max_input_bytes: int = _MAX_INPUT_BYTES
) -> tuple[dict, bytes]:
    if type(raw_bytes) is bytes:
        raw = raw_bytes
    elif type(raw_bytes) is str:
        raw = raw_bytes.encode("utf-8")
    else:
        _fail("storage_input_must_be_bytes_or_text")
    if (
        type(max_input_bytes) is not int
        or max_input_bytes < 0
        or len(raw) > max_input_bytes
    ):
        _fail("storage_input_too_large")
    _check_json_depth(raw)

    def pairs(items: list[tuple[str, object]]) -> dict:
        result = {}
        for key, value in items:
            if key in result:
                _fail("duplicate_json_key")
            result[key] = value
        return result

    def constant(_: str) -> object:
        _fail("storage_non_finite_number")

    try:
        value = json.loads(
            raw, object_pairs_hook=pairs, parse_constant=constant
        )
    except (UnicodeDecodeError, json.JSONDecodeError, RecursionError) as exc:
        msg = "invalid_storage_json"
        raise ValueError(msg) from exc
    _strict_json_value(value)
    if type(value) is not dict:
        _fail("invalid_storage_envelope")
    return value, raw


def _check_json_depth(raw: bytes) -> None:
    depth = 0
    in_string = False
    escaped = False
    for byte in raw:
        if in_string:
            if escaped:
                escaped = False
            elif byte == _JSON_BACKSLASH:
                escaped = True
            elif byte == _JSON_QUOTE:
                in_string = False
        elif byte == _JSON_QUOTE:
            in_string = True
        elif byte in _JSON_OPEN_DELIMITERS:
            depth += 1
            if depth > _MAX_NESTING:
                _fail("storage_json_too_deep")
        elif byte in _JSON_CLOSE_DELIMITERS:
            depth -= 1


def _expect_keys(value: object, keys: set[str], reason: str) -> dict:
    if type(value) is not dict or set(value) != keys:
        _fail(reason)
    return value


def _record_ref_locations(  # noqa: C901, PLR0912 # Enumerate the closed schema's legal pointer positions.
    table: str, kind: str, record: dict
) -> list[tuple[tuple[object, ...], str, str]]:
    """List only legal references for one typed record family."""
    refs: list[tuple[tuple[object, ...], str, str]] = []

    def add(
        path: tuple[object, ...], target_table: str, target_kind: str
    ) -> None:
        refs.append((path, target_table, target_kind))

    def statement(value: dict, prefix: tuple[object, ...]) -> None:
        for i, _ in enumerate(value.get("citations", [])):
            add((*prefix, "citations", i), "evidence_refs", "span_ref")
        for i, item in enumerate(value.get("basis", [])):
            basis_kind = (
                "source_slice_ref"
                if item.get("kind") in {"source_slice", "source_slice_ref"}
                else "statement_ref"
            )
            add((*prefix, "basis", i), "evidence_refs", basis_kind)

    if table == "results":
        if kind == "detail":
            for family in ("explanations", "observations", "questions"):
                for i, item in enumerate(record.get(family, [])):
                    for j, _ in enumerate(item.get("citations", [])):
                        add(
                            (family, i, "citations", j),
                            "evidence_refs",
                            "span_ref",
                        )
        elif kind in {"planning_card", "section", "document"}:
            statement_fields = {
                "planning_card": ("abstraction",),
                "section": ("orientation", "synthesis", "transition"),
                "document": ("preamble", "lead", "closing"),
            }[kind]
            if kind == "section" and "title" in record:
                statement(record["title"], ("title",))
            for field in statement_fields:
                for i, item in enumerate(record.get(field, [])):
                    statement(item, (field, i))
            for i, term in enumerate(record.get("terminology", [])):
                statement(term["definition"], ("terminology", i, "definition"))
    elif table == "evidence_refs" and kind == "source_slice_ref":
        add(("source",), "evidence_refs", "span_ref")
    elif table == "resolved_citations":
        add(("reference",), "evidence_refs", "")
        add(("source_target",), "source_targets", "source_target")
        add(("url",), "urls", "url")
    elif table == "producers":
        add(("original_run",), "runs", "run")
        add(("invocation",), "invocations", "invocation")
        for i, _ in enumerate(record.get("requests", [])):
            add(("requests", i), "requests", "request")
    elif table == "readers":
        add(("document",), "results", "document")
        for section_field in ("sections", "steps"):
            for i, section in enumerate(record.get(section_field, [])):
                add((section_field, i, "prose"), "results", "section")
                for j, _ in enumerate(section.get("details", [])):
                    add((section_field, i, "details", j), "results", "detail")
        for i, _ in enumerate(record.get("context_details", [])):
            add(("context_details", i), "results", "detail")
        for i, _ in enumerate(record.get("citations", [])):
            add(("citations", i), "resolved_citations", "resolved_citation")
        for i, _ in enumerate(record.get("support_provenance", [])):
            add(
                ("support_provenance", i),
                "inherited_provenance",
                "inherited_provenance",
            )
    elif table == "inherited_provenance":
        add(("reference",), "evidence_refs", "statement_ref")
        add(("result",), "results", "")
        add(("receipt",), "receipts", "receipt")
    return refs


def _root_ref_locations(
    root: dict,
) -> list[tuple[tuple[object, ...], str, str]]:
    refs = [(("reader",), "readers", "reader")]
    refs.extend(_generation_ref_locations(root["generation"]))
    return refs


def _generation_ref_locations(
    generation: dict,
) -> list[tuple[tuple[object, ...], str, str]]:
    refs: list[tuple[tuple[object, ...], str, str]] = []
    fixed = {
        "run": ("runs", "run"),
        "plan": ("plans", "story_plan"),
        "document": ("results", "document"),
    }
    for field, (table, kind) in fixed.items():
        if field in generation:
            refs.append((("generation", field), table, kind))
    for field, kind in (
        ("details", "detail"),
        ("planning", "planning_card"),
        ("sections", "section"),
    ):
        for i, _ in enumerate(generation.get(field, [])):
            refs.append((("generation", field, i), "results", kind))
    for field, table, kind in (
        ("producers", "producers", "producer"),
        ("receipts", "receipts", "receipt"),
    ):
        for i, _ in enumerate(generation.get(field, [])):
            refs.append((("generation", field, i), table, kind))
    return refs


def _at(value: object, path: tuple[object, ...]) -> object:
    for part in path:
        if type(part) is int:
            if type(value) not in {list, tuple} or part >= len(value):
                _fail("invalid_storage_reference_position")
            value = value[part]
        else:
            if not isinstance(value, Mapping) or part not in value:
                _fail("invalid_storage_reference_position")
            value = value[part]
    return value


def _read_root_generation_shape(root: dict) -> None:  # noqa: C901, PLR0912 # Validate each closed artifact-root branch.
    shape = root.get("shape")
    if shape == "report":
        _expect_keys(
            root,
            {"shape", "base_report", "generation", "reader"},
            "unknown_report_root_field",
        )
        if type(root["base_report"]) is not dict:
            _fail("invalid_report_base")
    elif shape == "annotations":
        _expect_keys(
            root,
            {"shape", "metadata", "generation", "reader"},
            "unknown_annotations_root_field",
        )
        metadata = _expect_keys(
            root["metadata"],
            {"schema", "base_sha", "head_sha"},
            "invalid_annotations_metadata",
        )
        if metadata["schema"] != "diffstory.annotations.v2" or any(
            type(metadata[k]) is not str for k in ("base_sha", "head_sha")
        ):
            _fail("invalid_annotations_metadata")
    else:
        _fail("invalid_storage_root_shape")
    generation = _json_mapping(
        root["generation"], "invalid_generation_skeleton"
    )
    allowed_generation_fields = set(GenerationManifest.model_fields)
    if not set(generation) <= allowed_generation_fields:
        _fail("unknown_generation_skeleton_field")
    for name, field in GenerationManifest.model_fields.items():
        if field.is_required() and name not in generation:
            _fail("incomplete_generation_skeleton")
    if (
        "version" in generation
        and generation["version"] != "diffstory.generation.v2"
    ):
        _fail("invalid_generation_schema")
    if "provenance" in generation and generation["provenance"] != "unverified":
        _fail("invalid_generation_provenance")
    for field_name in ("compiler_fact_signature", "ledger_digest"):
        if field_name in generation and (
            type(generation[field_name]) is not str
            or not _ID_PATTERN.fullmatch(generation[field_name])
        ):
            _fail("invalid_generation_digest")
    for field_name, model in (
        ("current_usage", CurrentUsage),
        ("ledger_parameters", LedgerParameters),
        ("request_configuration", RequestConfiguration),
    ):
        if field_name in generation:
            _model(model, generation[field_name])
    if "unit_keys" in generation and (
        type(generation["unit_keys"]) is not list
        or any(
            type(item) is not str or not _ID_PATTERN.fullmatch(item)
            for item in generation["unit_keys"]
        )
    ):
        _fail("invalid_generation_units")
    for field_name in (
        "details",
        "planning",
        "sections",
        "producers",
        "receipts",
    ):
        if (
            field_name in generation
            and type(generation[field_name]) is not list
        ):
            _fail("invalid_generation_array")


def _validate_envelope_shape(  # noqa: C901 # Validate envelope and all closed table entry shells.
    envelope: dict,
) -> tuple[str, dict, dict[str, dict[str, dict]]]:
    _expect_keys(
        envelope,
        {"storage_version", "artifact_kind", "root", "tables"},
        "unknown_storage_envelope_field",
    )
    if envelope["storage_version"] != STORAGE_VERSION:
        _fail("unknown_storage_version")
    artifact_kind = envelope["artifact_kind"]
    if artifact_kind not in {"report", "annotations"}:
        _fail("invalid_artifact_kind")
    root = _json_mapping(envelope["root"], "invalid_storage_root")
    _read_root_generation_shape(root)
    if (artifact_kind == "report") != (root.get("shape") == "report"):
        _fail("artifact_root_kind_mismatch")
    tables = _expect_keys(
        envelope["tables"], set(_TABLES), "unknown_storage_table"
    )
    indexes: dict[str, dict[str, dict]] = {}
    total = 0
    all_ids: set[str] = set()
    for table in _TABLES:
        entries = tables[table]
        if type(entries) is not list:
            _fail("invalid_storage_table")
        index: dict[str, dict] = {}
        previous = ""
        for entry in entries:
            _expect_keys(
                entry, {"id", "kind", "record"}, "unknown_storage_table_field"
            )
            identifier = entry["id"]
            kind = entry["kind"]
            if type(identifier) is not str or not _ID_PATTERN.fullmatch(
                identifier
            ):
                _fail("invalid_storage_address")
            if identifier in all_ids:
                _fail("duplicate_storage_address")
            if identifier <= previous:
                _fail("noncanonical_storage_table_order")
            previous = identifier
            if kind not in _TABLE_RECORD_KINDS[table]:
                _fail("invalid_storage_record_kind")
            if type(entry["record"]) is not dict:
                _fail("invalid_storage_record")
            index[identifier] = entry
            all_ids.add(identifier)
            total += 1
        indexes[table] = index
    if total > _MAX_RECORDS:
        _fail("too_many_storage_records")
    return artifact_kind, root, indexes


@dataclass(frozen=True, slots=True)
class ReportStorageView:
    """Immutable structural data view with detached indexed record access.

    It verifies storage structure and integrity only. It is not a reusable
    proof, validation result, or report-use capability.
    """

    _raw: bytes | None
    _artifact_kind: str
    _root: Mapping[str, object]
    _tables: Mapping[str, Mapping[str, Mapping[str, object]]]
    _result_by_key: Mapping[tuple[str, str], ResultRef]
    _result_by_id: Mapping[str, ResultRef]
    _typed_validation_counts: Mapping[str, int]

    @property
    def artifact_kind(self) -> str:
        """Return the parsed artifact family."""
        return self._artifact_kind

    @property
    def byte_length(self) -> int:
        """Return input bytes, or explicit stored-envelope size for a live graph."""
        if self._raw is not None:
            return len(self._raw)
        return len(self.to_bytes())

    def to_bytes(self) -> bytes:
        """Return canonical storage bytes, serializing live graphs on request."""
        if self._raw is not None:
            return self._raw
        envelope = {
            "storage_version": STORAGE_VERSION,
            "artifact_kind": self._artifact_kind,
            "root": _thaw(self._root),
            "tables": {
                table: [
                    _thaw(self._tables[table][identifier])
                    for identifier in sorted(self._tables[table])
                ]
                for table in _TABLES
            },
        }
        return canonical_bytes(envelope)

    @property
    def storage_counts(self) -> dict[str, int]:
        """Return detached record counts for each typed table."""
        return {name: len(self._tables[name]) for name in _TABLES}

    @property
    def typed_validation_counts(self) -> dict[str, int]:
        """Return unique table/kind counts from typed input or one parse pass."""
        return dict(self._typed_validation_counts)

    def lookup_result(self, kind: str, item_key: str) -> ResultRef:
        """Look up one result by its semantic kind and item key."""
        reference = self._result_by_key.get((kind, item_key))
        if reference is None:
            _fail("missing_storage_result")
        return reference

    def result_reference(self, storage_id: str) -> ResultRef:
        """Return a typed address for one indexed result."""
        reference = self._result_by_id.get(storage_id)
        if reference is None:
            _fail("missing_storage_result")
        return reference

    @property
    def generation_result_refs(self) -> Mapping[str, tuple[ResultRef, ...]]:
        """Return ordered generation result membership as typed addresses."""
        generation = self._root["generation"]
        output: dict[str, tuple[ResultRef, ...]] = {}
        for name in ("details", "planning", "sections"):
            output[name] = tuple(
                self._result_ref(item) for item in generation.get(name, [])
            )
        output["document"] = (self._result_ref(generation["document"]),)
        return MappingProxyType(output)

    @property
    def generation_record_refs(self) -> Mapping[str, object]:
        """Return run, plan, producer, receipt, and result addresses in order."""
        generation = self._root["generation"]
        output: dict[str, object] = {
            "results": self.generation_result_refs,
            "run": self._record_ref("runs", generation["run"]),
            "plan": self._record_ref("plans", generation["plan"]),
            "producers": tuple(
                self._record_ref("producers", item)
                for item in generation["producers"]
            ),
            "receipts": tuple(
                self._record_ref("receipts", item)
                for item in generation["receipts"]
            ),
        }
        return MappingProxyType(output)

    @property
    def reader_projection(self) -> IndexedReaderProjection:
        """Return ordered reader membership without materializing result data."""
        reader = self._reader_record()
        section_field = (
            "sections" if reader["shape"] == "presentation" else "steps"
        )
        sections = tuple(
            ReaderSectionRef(
                group_id=section["group_id"],
                group_key=section["group_key"],
                number=section["number"],
                next_id=section["next_id"],
                prose=self._result_ref(section["prose"]),
                details=tuple(
                    self._result_ref(item) for item in section["details"]
                ),
            )
            for section in reader[section_field]
        )
        return IndexedReaderProjection(
            shape=reader["shape"],
            document=self._result_ref(reader["document"]),
            sections=sections,
            context_details=tuple(
                self._result_ref(item) for item in reader["context_details"]
            ),
            citations=tuple(
                self._record_ref("resolved_citations", item)
                for item in reader["citations"]
            ),
            support_provenance=tuple(
                self._record_ref("inherited_provenance", item)
                for item in reader["support_provenance"]
            ),
            version=reader.get("version"),
            provenance=reader.get("provenance"),
        )

    def reader_section(self, group_id: str) -> ReaderSectionRef:
        """Return one ordered section's typed references without resolving peers."""
        for section in self.reader_projection.sections:
            if section.group_id == group_id:
                return section
        _fail("missing_reader_section")
        return None

    def record(self, table: str, storage_id: str) -> dict:
        """Return a detached normalized table record by table and address."""
        if table not in self._tables:
            _fail("unknown_storage_table")
        entry = self._tables[table].get(storage_id)
        if entry is None:
            _fail("missing_storage_record")
        return _thaw(entry["record"])

    def result(self, storage_id: str, *, kind: str | None = None) -> dict:
        """Resolve one result record without expanding the enclosing artifact."""
        entry = self._tables["results"].get(storage_id)
        if entry is None or (kind is not None and entry["kind"] != kind):
            _fail("missing_storage_result")
        return self._materialize("results", storage_id, {})

    def result_record(self, reference: ResultRef) -> dict:
        """Return one detached logical result for a typed result address."""
        if not isinstance(reference, ResultRef):
            _fail("invalid_storage_result_reference")
        return self.result(reference.storage_id, kind=reference.kind)

    def iter_sections(self) -> Iterator[dict]:
        """Yield ordered reader sections with detached prose and detail records."""
        reader_id = self._reader_id()
        reader = self._tables["readers"][reader_id]["record"]
        field = "sections" if reader["shape"] == "presentation" else "steps"
        for section in reader[field]:
            yield self._materialize_reader_section(section)

    def section_details(self, group_id: str) -> list[dict]:
        """Return one reader section's details in stored order."""
        reader = self._reader_record()
        field = "sections" if reader["shape"] == "presentation" else "steps"
        for section in reader[field]:
            if section["group_id"] == group_id:
                memo: dict[str, dict] = {}
                return [
                    self._materialize("results", item["$ref"], memo)
                    for item in section["details"]
                ]
        _fail("missing_reader_section")
        return None

    def citations(self) -> tuple[dict, ...]:
        """Return ordered resolved citation records with exact URLs."""
        reader = self._reader_record()
        return tuple(
            self._materialize("resolved_citations", item["$ref"], {})
            for item in reader["citations"]
        )

    def provenance(self) -> tuple[dict, ...]:
        """Return ordered inherited statements reconstructed from result paths."""
        reader = self._reader_record()
        return tuple(
            self._materialize("inherited_provenance", item["$ref"], {})
            for item in reader["support_provenance"]
        )

    def generation_audit(self) -> dict:
        """Return proof metadata and audit records without expanding result arrays.

        Result positions are returned as ``{kind, storage_id}`` index entries.
        The complete plan, producers, receipts, and original request metadata
        are resolved because they form the generation audit closure.
        """
        generation = self._root["generation"]
        output = {
            key: _thaw(value)
            for key, value in generation.items()
            if key
            not in {
                "run",
                "details",
                "plan",
                "planning",
                "sections",
                "document",
                "producers",
                "receipts",
            }
        }
        output["run"] = self._materialize(
            "runs", generation["run"]["$ref"], {}
        )
        for name in ("details", "planning", "sections"):
            output[name] = tuple(
                {"kind": item["kind"], "storage_id": item["$ref"]}
                for item in generation.get(name, [])
            )
        document = generation["document"]
        output["document"] = {
            "kind": document["kind"],
            "storage_id": document["$ref"],
        }
        output["plan"] = self._materialize(
            "plans", generation["plan"]["$ref"], {}
        )
        output["producers"] = tuple(
            self._materialize("producers", item["$ref"], {})
            for item in generation["producers"]
        )
        output["receipts"] = tuple(
            self._materialize("receipts", item["$ref"], {})
            for item in generation["receipts"]
        )
        return output

    def _reader_id(self) -> str:
        return self._root["reader"]["$ref"]

    def _record_ref(
        self, table: str, pointer: Mapping[str, object]
    ) -> RecordRef:
        identifier, kind = _check_reference_shape(pointer)
        entry = self._tables[table].get(identifier)
        if entry is None or entry["kind"] != kind:
            _fail("wrong_kind_storage_reference")
        return RecordRef(table, kind, identifier)

    def _result_ref(self, pointer: Mapping[str, object]) -> ResultRef:
        identifier, kind = _check_reference_shape(pointer)
        reference = self._result_by_id.get(identifier)
        if reference is None or reference.kind != kind:
            _fail("wrong_kind_storage_reference")
        return reference

    def _reader_record(self) -> Mapping[str, object]:
        return self._tables["readers"][self._reader_id()]["record"]

    def _materialize_reader_section(
        self, section: Mapping[str, object]
    ) -> dict:
        output = {
            key: _thaw(value)
            for key, value in section.items()
            if key not in {"prose", "details"}
        }
        output["prose"] = self._materialize(
            "results", section["prose"]["$ref"], {}
        )
        output["details"] = [
            self._materialize("results", item["$ref"], {})
            for item in section["details"]
        ]
        return output

    def _materialize(
        self, table: str, identifier: str, memo: dict[str, dict]
    ) -> dict:
        cache_key = f"{table}:{identifier}"
        if cache_key in memo:
            return copy.deepcopy(memo[cache_key])
        entry = self._tables[table].get(identifier)
        if entry is None:
            _fail("missing_storage_record")
        kind = entry["kind"]
        record = _thaw(entry["record"])
        locations = _record_ref_locations(table, kind, record)
        if table == "readers":
            section_field = (
                "sections" if record["shape"] == "presentation" else "steps"
            )
            section_locations = [
                item
                for item in locations
                if item[0][0] in {"sections", "steps"}
            ]
            other_locations = [
                item for item in locations if item not in section_locations
            ]
            result = _replace_paths(
                record,
                other_locations,
                lambda target_table, target_id: self._materialize(
                    target_table, target_id, memo
                ),
            )
            result[section_field] = [
                self._materialize_reader_section(item)
                for item in record[section_field]
            ]
        else:
            result = _replace_paths(
                record,
                locations,
                lambda target_table, target_id: self._materialize(
                    target_table, target_id, memo
                ),
            )
        if table == "resolved_citations":
            target = result.pop("source_target")
            result.update(target)
            result["url"] = result.pop("url")
            result["url"] = self._url_value(entry["record"]["url"]["$ref"])
        elif table == "inherited_provenance":
            source_result = result.pop("result")
            statement = _statement_from_path(
                source_result, result["reference"]
            )
            result["statement"] = statement
        memo[cache_key] = copy.deepcopy(result)
        return result

    def _validated_record(self, table: str, storage_id: str) -> dict:
        """Return one detached logical record already checked at the storage boundary.

        This narrow adapter supports indexed detail, support, and manifest
        validators. The returned mapping is data only; it carries no proof.
        """
        if table not in {
            "results",
            "plans",
            "evidence_refs",
            "resolved_citations",
            "runs",
            "invocations",
            "requests",
            "producers",
            "receipts",
            "inherited_provenance",
        }:
            _fail("unsupported_validated_record_access")
        return self._materialize(table, storage_id, {})

    def _url_value(self, identifier: str) -> str | None:
        record = self._tables["urls"][identifier]["record"]
        return record["value"]


def _validate_builder_reader_relationships(
    generation: dict,
    reader_pointer: dict,
    tables: dict[str, dict[str, dict]],
) -> None:
    """Check reader membership against builder indexes without resolving records."""
    reader_entry = tables["readers"].get(reader_pointer["$ref"])
    if reader_entry is None:
        _fail("missing_storage_record")
    reader = reader_entry["record"]
    generation_result_ids: set[str] = set()
    for field in ("details", "planning", "sections"):
        generation_result_ids.update(
            item["$ref"] for item in generation[field]
        )
    generation_result_ids.add(generation["document"]["$ref"])
    if reader["document"]["$ref"] != generation["document"]["$ref"]:
        _fail("reader_document_mismatch")
    generation_section_ids = {item["$ref"] for item in generation["sections"]}
    section_field = (
        "sections" if reader["shape"] == "presentation" else "steps"
    )
    for section in reader[section_field]:
        if section["prose"]["$ref"] not in generation_section_ids:
            _fail("reader_section_mismatch")
        if any(
            item["$ref"] not in generation_result_ids
            for item in section["details"]
        ):
            _fail("reader_detail_mismatch")
    if any(
        item["$ref"] not in generation_result_ids
        for item in reader["context_details"]
    ):
        _fail("reader_context_mismatch")

    generation_receipt_ids = {item["$ref"] for item in generation["receipts"]}
    for pointer in reader["support_provenance"]:
        provenance = tables["inherited_provenance"][pointer["$ref"]]["record"]
        reference = tables["evidence_refs"][provenance["reference"]["$ref"]][
            "record"
        ]
        receipt_pointer = provenance["receipt"]
        receipt = tables["receipts"][receipt_pointer["$ref"]]["record"]
        if (
            reference["producer_item_key"] != receipt["item_key"]
            or receipt_pointer["$ref"] not in generation_receipt_ids
        ):
            _fail("reader_provenance_receipt_mismatch")


@dataclass(frozen=True, slots=True)
class ReportRecordGraph(ReportStorageView):
    """Run-owned immutable graph built from typed records without report JSON.

    The graph shares the structural indexed accessors used by parsed views. It
    is data only: record addresses, structural checks, and attached reader
    references do not establish report or generation authority.
    """

    @classmethod
    def from_typed(
        cls,
        generation: GenerationManifest,
        reader: ReportReaderProjection,
        *,
        artifact_kind: str,
        base_report: dict | None = None,
        metadata: dict | None = None,
    ) -> ReportRecordGraph:
        """Build a shared graph from typed unique results and ordered reader input.

        ``base_report`` is an opaque pass-through compiler report mapping for a
        report graph. ``metadata`` contains the annotations.v2 root identity.
        The generation and reader models are snapshotted record by record; no
        expanded artifact is encoded, parsed, or retained.
        """
        if (
            type(artifact_kind) is not str
            or artifact_kind not in {"report", "annotations"}
            or not isinstance(generation, GenerationManifest)
            or not isinstance(reader, ReportReaderProjection)
        ):
            _fail("invalid_typed_graph_input")
        if (artifact_kind == "report") != (reader.shape == "presentation"):
            _fail("artifact_reader_kind_mismatch")
        builder = _Builder(defer_validation=True)
        generation_skeleton = builder.add_generation(generation)
        reader_pointer = builder.add_reader_projection(reader)
        if artifact_kind == "report":
            if (
                metadata is not None
                or type(base_report) is not dict
                or {"generation", "narrative_v2"} & set(base_report)
            ):
                _fail("invalid_report_graph_root")
            _strict_json_value(base_report)
            root = {
                "shape": "report",
                "base_report": copy.deepcopy(base_report),
                "generation": generation_skeleton,
                "reader": reader_pointer,
            }
        else:
            if base_report is not None or type(metadata) is not dict:
                _fail("invalid_annotations_graph_root")
            if set(metadata) != {"schema", "base_sha", "head_sha"}:
                _fail("invalid_annotations_metadata")
            if (
                metadata.get("schema") != "diffstory.annotations.v2"
                or type(metadata.get("base_sha")) is not str
                or type(metadata.get("head_sha")) is not str
            ):
                _fail("invalid_annotations_metadata")
            _strict_json_value(metadata)
            root = {
                "shape": "annotations",
                "metadata": copy.deepcopy(metadata),
                "generation": generation_skeleton,
                "reader": reader_pointer,
            }
        _read_root_generation_shape(root)
        _check_graph_and_addresses(root, builder.records)
        memo, _typed_records, validation_counts = _validate_table_records(
            root, builder.records
        )
        _verify_storage_addresses(builder.records)

        by_key: dict[tuple[str, str], ResultRef] = {}
        by_id: dict[str, ResultRef] = {}
        for (kind, item_key), identifier in builder.result_ids.items():
            reference = ResultRef("results", kind, identifier, item_key)
            by_key[(kind, item_key)] = reference
            by_id[identifier] = reference
        frozen_tables = {
            table: MappingProxyType(
                {
                    identifier: _freeze(entry)
                    for identifier, entry in records.items()
                }
            )
            for table, records in builder.records.items()
        }
        return cls(
            _raw=None,
            _artifact_kind=artifact_kind,
            _root=_freeze(root),
            _tables=MappingProxyType(frozen_tables),
            _result_by_key=MappingProxyType(by_key),
            _result_by_id=MappingProxyType(by_id),
            _typed_validation_counts=MappingProxyType(validation_counts),
        )


@dataclass(frozen=True, slots=True)
class _StorageImportFacts:
    """Small checked selectors transferred from one independent parse pass.

    Attributes:
        result_ids: Unique ``(kind, item_key)`` to storage-address bindings.
        receipt_ids: Unique result item keys to receipt storage addresses.
        result_digests: Typed receipt digests keyed by result address.
        result_content_digests: Typed logical content digests by result address.
        result_content_digest_matches_record: Whether raw logical bytes already
            match typed default-expanded bytes for safe digest reuse.
        plan_facts: Compact validated plan selectors by plan address.
        producer_facts: Compact audit selectors by producer key.
        producer_keys_by_record_id: Direct address-to-producer-key index.
        validation_counts: Unique logical row counts by table and kind.
    """

    result_ids: Mapping[tuple[str, str], str]
    receipt_ids: Mapping[str, str]
    result_digests: Mapping[str, str]
    result_content_digests: Mapping[str, str]
    result_content_digest_matches_record: Mapping[str, bool]
    plan_facts: Mapping[str, Mapping[str, object]]
    producer_facts: Mapping[str, Mapping[str, object]]
    producer_keys_by_record_id: Mapping[str, str]
    validation_counts: Mapping[str, int]


@dataclass(frozen=True, slots=True)
class _ParsedStorageImport:
    """Freshly parsed bytes prepared for one session-owned arena transfer.

    This private value stores no raw input buffer, expanded presentation,
    parsed model map, or logical-record map. Its root and tables contain only
    immutable primitives produced by the strict storage parser.
    """

    artifact_kind: str
    root: Mapping[str, object]
    tables: Mapping[str, Mapping[str, Mapping[str, object]]]
    result_by_key: Mapping[tuple[str, str], ResultRef]
    result_by_id: Mapping[str, ResultRef]
    facts: _StorageImportFacts

    def graph(self, *, base_report: object | None = None) -> ReportRecordGraph:
        """Return the independently parsed graph as immutable data only.

        Preconditions:
            This value came directly from ``_parse_report_storage_for_import``.
        Postconditions:
            The graph shares the exact table rows from that parse and retains
            no source bytes or mutable validation models.
        Essential behavior:
            It creates no authority; a ReportSession must still prove use.
        """
        root = self.root
        if base_report is not None:
            if (
                self.artifact_kind != "report"
                or type(base_report) is not _OwnedMapping
            ):
                _fail("invalid_report_base")
            root = MappingProxyType(
                {
                    "shape": "report",
                    "base_report": base_report,
                    "generation": self.root["generation"],
                    "reader": self.root["reader"],
                }
            )
        return ReportRecordGraph(
            _raw=None,
            _artifact_kind=self.artifact_kind,
            _root=root,
            _tables=self.tables,
            _result_by_key=self.result_by_key,
            _result_by_id=self.result_by_id,
            _typed_validation_counts=self.facts.validation_counts,
        )

    def generation_record_set(self) -> GenerationRecordSet:
        """Capture ordered header and row addresses without a manifest model.

        Preconditions:
            The normalized root passed closed generation-shell validation.
        Postconditions:
            Returns a data-only record set that preserves header field
            presence and every stored result, producer, and receipt order.
        Essential behavior:
            Resolves only root pointers and does not expand table records.
        """
        generation = self.root["generation"]
        header_names = {
            "version",
            "provenance",
            "compiler_fact_signature",
            "current_usage",
            "ledger_digest",
            "ledger_parameters",
            "request_configuration",
            "unit_keys",
        }
        header = {
            key: _thaw(value)
            for key, value in generation.items()
            if key in header_names
        }

        def record_ref(pointer: Mapping[str, object], table: str) -> RecordRef:
            """Resolve one checked root pointer to a strict data address."""
            identifier, kind = _check_reference_shape(pointer)
            entry = self.tables[table].get(identifier)
            if entry is None or entry["kind"] != kind:
                _fail("wrong_kind_storage_reference")
            return RecordRef(table, kind, identifier)

        def result_ref(pointer: Mapping[str, object]) -> ResultRef:
            """Resolve one checked result pointer and item binding."""
            identifier, kind = _check_reference_shape(pointer)
            result = self.result_by_id.get(identifier)
            if result is None or result.kind != kind:
                _fail("wrong_kind_storage_reference")
            return result

        return GenerationRecordSet(
            header,
            run_ref=record_ref(generation["run"], "runs"),
            details=tuple(result_ref(item) for item in generation["details"]),
            plan_ref=record_ref(generation["plan"], "plans"),
            planning=tuple(
                result_ref(item) for item in generation["planning"]
            ),
            sections=tuple(
                result_ref(item) for item in generation["sections"]
            ),
            document=result_ref(generation["document"]),
            producers=tuple(
                record_ref(item, "producers")
                for item in generation["producers"]
            ),
            receipts=tuple(
                record_ref(item, "receipts") for item in generation["receipts"]
            ),
        )


def _replace_paths(
    value: dict,
    locations: list[tuple[tuple[object, ...], str, str]],
    resolve: Callable[[str, str], dict],
) -> dict:
    result = _thaw(value)
    for path, target_table, _target_kind in locations:
        pointer = _at(result, path)
        identifier, _kind = _check_reference_shape(pointer)
        replacement = resolve(target_table, identifier)
        parent = result
        for item in path[:-1]:
            parent = parent[item]
        parent[path[-1]] = replacement
    return result


def _statement_value_from_normalized_result(  # noqa: C901 # Normalize each typed nested reference in place.
    indexes: Mapping[str, Mapping[str, Mapping[str, object]]],
    result_id: str,
    reference: Mapping[str, object],
) -> Mapping[str, object]:
    """Select one normalized statement without rebuilding its producer model.

    Preconditions:
        ``reference`` contains an allowlisted path into an owned result row.
    Postconditions:
        Returns the exact immutable normalized statement at that path.
    Essential behavior:
        Resolves terminology and statement arrays directly from primitive rows.
    """
    entry = indexes["results"].get(result_id)
    if entry is None:
        _fail("missing_provenance_result")
    result = entry["record"]
    path = reference.get("statement_path")
    if not isinstance(path, Mapping):
        _fail("invalid_statement_path")
    field_name = path.get("field")
    index = path.get("index", 0)
    if type(field_name) is not str or type(index) is not int or index < 0:
        _fail("invalid_statement_path")
    allowed = {
        "planning_card": {"abstraction", "terminology"},
        "section": {
            "title",
            "orientation",
            "synthesis",
            "transition",
            "terminology",
        },
        "document": {"preamble", "lead", "closing", "terminology"},
    }
    kind = entry["kind"]
    if kind not in allowed or field_name not in allowed[kind]:
        _fail("invalid_statement_path")
    if field_name == "title":
        if index != 0:
            _fail("invalid_statement_path")
        value = result.get("title")
    elif field_name == "terminology":
        values = result.get("terminology", ())
        if index >= len(values):
            _fail("invalid_statement_path")
        value = values[index]["definition"]
    else:
        values = result.get(field_name, ())
        if index >= len(values):
            _fail("invalid_statement_path")
        value = values[index]
    if not isinstance(value, Mapping):
        _fail("invalid_statement_path")
    return value


def _statement_reference_locations(
    statement: Mapping[str, object],
) -> dict[tuple[object, ...], tuple[str, str]]:
    """Return declared typed-reference positions inside one statement.

    Preconditions:
        ``statement`` is a normalized supported statement record.
    Postconditions:
        Every listed path names a legal evidence-reference position.
    Essential behavior:
        Distinguishes span citations, source slices, and statement bases.
    """
    locations: dict[tuple[object, ...], tuple[str, str]] = {}
    for field_name, kind in (
        ("citations", "span_ref"),
        ("basis", ""),
    ):
        values = statement.get(field_name, ())
        if type(values) not in {tuple, list}:
            continue
        for position, pointer in enumerate(values):
            pointer_kind_for_position = kind
            if field_name == "basis":
                _identifier, pointer_kind = _check_reference_shape(pointer)
                pointer_kind_for_position = pointer_kind
            locations[(field_name, position)] = (
                "evidence_refs",
                pointer_kind_for_position,
            )
    return locations


def _record_logical_chunks(
    table: str,
    identifier: str,
    indexes: Mapping[str, Mapping[str, Mapping[str, object]]],
) -> Iterator[bytes]:
    """Stream one owned record as canonical logical JSON through its references.

    Args:
        table: Storage table that owns the record.
        identifier: Address of the record in that table.
        indexes: Address indexes containing the row and legal reference targets.

    Yields:
        Canonical JSON byte chunks for the expanded logical record.

    Preconditions:
        ``indexes`` contains the row and each legal reference target.
    Postconditions: The chunks form its expanded logical record in canonical
        key order.
    Essential behavior:
        Resolves only the declared schema edges and never builds a full graph.
    """
    entry = indexes.get(table, {}).get(identifier)
    if entry is None:
        _fail("missing_storage_record")
    kind = entry["kind"]
    record = entry["record"]
    logical: dict = dict(record)
    references: dict[tuple[object, ...], tuple[str, str]] = {
        path: (target_table, target_kind)
        for path, target_table, target_kind in _record_ref_locations(
            table, kind, record
        )
    }
    if table == "readers":
        yield from _reader_logical_chunks(record, indexes)
        return
    if table == "resolved_citations":
        target_pointer = logical.pop("source_target")
        target_id, _target_kind = _check_reference_shape(target_pointer)
        target_entry = indexes["source_targets"].get(target_id)
        url_pointer = logical.pop("url")
        url_id, _url_kind = _check_reference_shape(url_pointer)
        url_entry = indexes["urls"].get(url_id)
        if target_entry is None or url_entry is None:
            _fail("missing_storage_record")
        logical.update(target_entry["record"])
        logical["url"] = url_entry["record"]["value"]
        references = {
            path: value
            for path, value in references.items()
            if path == ("reference",)
        }
    elif table == "inherited_provenance":
        result_pointer = logical.pop("result")
        result_id, _result_kind = _check_reference_shape(result_pointer)
        reference_id, _reference_kind = _check_reference_shape(
            record["reference"]
        )
        reference_entry = indexes["evidence_refs"].get(reference_id)
        if (
            reference_entry is None
            or reference_entry["kind"] != "statement_ref"
        ):
            _fail("wrong_kind_storage_reference")
        reference = reference_entry["record"]
        statement = _statement_value_from_normalized_result(
            indexes, result_id, reference
        )
        logical["statement"] = statement
        references = {
            path: value
            for path, value in references.items()
            if path in {("reference",), ("receipt",)}
        }
        references.update(
            {
                ("statement", *path): value
                for path, value in _statement_reference_locations(
                    statement
                ).items()
            }
        )
    yield from _iter_logical_record_json(
        logical,
        references=references,
        indexes=indexes,
        logical_lookup=lambda target_table, target_id: _record_logical_chunks(
            target_table, target_id, indexes
        ),
    )


def _reader_logical_chunks(
    record: Mapping[str, object],
    indexes: Mapping[str, Mapping[str, Mapping[str, object]]],
) -> Iterator[bytes]:
    """Stream the logical reader without its storage-only shape field.

    Args:
        record: Normalized reader record with storage references.
        indexes: Address indexes for the reader's reference targets.

    Yields:
        Canonical JSON byte chunks for the selected logical reader.

    Preconditions:
        ``record`` is a normalized reader selector record with owned targets.
    Postconditions: The chunks equal the canonical presentation or annotation
        reader.
    Essential behavior:
        Resolves ordered selectors one at a time through the chunk iterator.
    """
    logical_reader = {
        key: value for key, value in record.items() if key != "shape"
    }
    references = {
        path: (table, kind)
        for path, table, kind in _record_ref_locations(
            "readers", "reader", record
        )
    }
    yield from _iter_logical_record_json(
        logical_reader,
        references=references,
        indexes=indexes,
        logical_lookup=lambda table, identifier: _record_logical_chunks(
            table, identifier, indexes
        ),
    )


def _record_streamed_address(
    table: str,
    identifier: str,
    indexes: Mapping[str, Mapping[str, Mapping[str, object]]],
) -> str:
    """Recompute a complete-record address from immutable owned primitives.

    Args:
        table: Storage table that owns the record.
        identifier: Address of the record in that table.
        indexes: Address indexes containing the row and legal reference targets.

    Returns:
        str: Domain-separated SHA-256 address of the logical record.

    Preconditions:
        ``table`` and ``identifier`` select one row in ``indexes``.
    Postconditions: The address covers the complete logical record.
    Essential behavior:
        Hashes canonical chunks incrementally without retaining joined bytes.
    """
    entry = indexes[table][identifier]
    return _streamed_storage_address(
        str(entry["kind"]), _record_logical_chunks(table, identifier, indexes)
    )


def _streamed_storage_address(kind: str, chunks: Iterator[bytes]) -> str:
    """Hash one strict logical record stream under its declared storage kind.

    Args:
        kind: Table-record discriminator.
        chunks: Canonical logical JSON byte chunks for the record.

    Returns:
        str: Report-storage v1 address for the complete record content.

    Preconditions:
        ``kind`` is a table-record discriminator and ``chunks`` yield the
        record's canonical logical JSON.
    Postconditions: The returned address uses the storage-v1 domain.
    Essential behavior:
        Shares one domain-separated address calculation across parser and arena.
    """
    digest = hashlib.sha256()
    digest.update(b'{"kind":')
    digest.update(canonical_bytes(kind))
    digest.update(b',"record":')
    for chunk in chunks:
        digest.update(chunk)
    digest.update(b',"storage_version":')
    digest.update(canonical_bytes(STORAGE_VERSION))
    digest.update(b"}")
    return digest.hexdigest()


def _arena_logical_record(
    table: str,
    identifier: str,
    indexes: Mapping[str, Mapping[str, Mapping[str, object]]],
) -> dict:
    """Materialize one logical record for collision comparison or access.

    Args:
        table: Storage table that owns the record.
        identifier: Address of the record in that table.
        indexes: Address indexes containing the row and legal reference targets.

    Returns:
        dict: Detached logical record mapping.

    Preconditions:
        ``table`` and ``identifier`` select a valid owned row.
    Postconditions: The mapping is detached or a storage error is raised.
    Essential behavior:
        Joins only the explicitly selected record's canonical chunks.
    """
    try:
        value = json.loads(
            b"".join(_record_logical_chunks(table, identifier, indexes))
        )
    except (TypeError, json.JSONDecodeError) as error:
        msg = "invalid_logical_storage_record"
        raise ValueError(msg) from error
    if type(value) is not dict:
        _fail("invalid_logical_storage_record")
    return value


def _reader_records_stream_equal(
    left_record: Mapping[str, object],
    right_record: Mapping[str, object],
    indexes: Mapping[str, Mapping[str, Mapping[str, object]]],
) -> bool:
    """Compare reader logical streams in bounded chunks without joining them.

    Args:
        left_record: First normalized reader selector.
        right_record: Second normalized reader selector.
        indexes: Address indexes containing both readers' targets.

    Returns:
        bool: Whether the complete logical byte streams are identical.

    Preconditions:
        Both records are normalized readers whose references resolve in
        ``indexes``.
    Postconditions: True implies both complete logical streams are equal.
    Essential behavior:
        Advances one chunk from each stream without retaining either tree.
    """
    return _canonical_streams_equal(
        _reader_logical_chunks(left_record, indexes),
        _reader_logical_chunks(right_record, indexes),
    )


def _canonical_streams_equal(  # noqa: C901 # Advance bounded chunks without joining either stream.
    left: Iterator[bytes], right: Iterator[bytes]
) -> bool:
    """Compare canonical chunk streams without joining their contents.

    Args:
        left: First iterator of complete canonical JSON byte chunks.
        right: Second iterator of complete canonical JSON byte chunks.

    Returns:
        bool: Whether both byte streams are identical.

    Preconditions:
        Both iterators yield chunks for complete canonical JSON values.
    Postconditions: True implies both complete byte streams are identical.
    Essential behavior:
        Uses bounded cursors so a reader record is never expanded to one buffer.
    """
    left = iter(left)
    right = iter(right)
    left_chunk: bytes | None = None
    right_chunk: bytes | None = None
    left_offset = 0
    right_offset = 0
    left_done = False
    right_done = False
    while True:
        if left_chunk is None and not left_done:
            try:
                left_chunk = next(left)
                left_offset = 0
            except StopIteration:
                left_done = True
        if right_chunk is None and not right_done:
            try:
                right_chunk = next(right)
                right_offset = 0
            except StopIteration:
                right_done = True
        if left_done or right_done:
            return left_done and right_done
        if left_chunk is None or right_chunk is None:
            _fail("invalid_streamed_storage_record")
        left_remaining = len(left_chunk) - left_offset
        right_remaining = len(right_chunk) - right_offset
        if left_remaining == 0:
            left_chunk = None
            continue
        if right_remaining == 0:
            right_chunk = None
            continue
        length = min(left_remaining, right_remaining)
        if (
            left_chunk[left_offset : left_offset + length]
            != right_chunk[right_offset : right_offset + length]
        ):
            return False
        left_offset += length
        right_offset += length
        if left_offset == len(left_chunk):
            left_chunk = None
        if right_offset == len(right_chunk):
            right_chunk = None


def _capture_mapping(value: object) -> dict:
    """Copy one caller record before validation or serializer normalization.

    Preconditions:
        ``value`` is a typed model or an exact JSON dictionary.
    Postconditions:
        Returns a detached primitive dictionary with source scalar types kept.
    Essential behavior:
        Reads raw model attributes before any Pydantic JSON serializer runs.
    """
    if isinstance(value, BaseModel):
        return _raw_model_snapshot(value)
    if type(value) is dict:
        record = copy.deepcopy(value)
        _strict_json_value(record)
        return record
    _fail("invalid_typed_storage_record")


class _ArenaRecordReader:
    """Private immutable projection over records owned by one arena.

    Attributes:
        _tables: Shallow read-only table indexes over immutable arena records.
        _result_ids: Result address to immutable kind/item-key bindings.
        _owner_token: Arena identity token; it carries no proof authority.
    """

    __slots__ = (
        "_owner_token",
        "_result_content_digest_matches_record",
        "_result_content_digests",
        "_result_ids",
        "_tables",
    )

    def __init__(  # noqa: PLR0913 # The reader binds both exact arena digest witnesses.
        self,
        token: object,
        owner_token: object,
        tables: Mapping[str, Mapping[str, Mapping[str, object]]],
        result_ids: Mapping[str, tuple[str, str]],
        result_content_digests: Mapping[str, str],
        result_content_digest_matches_record: Mapping[str, bool],
    ) -> None:
        """Capture immutable record indexes for one private consumer.

        Preconditions:
            ``token`` is the module-private reader token and all record entries
            already have immutable primitive contents. Digest mappings are
            arena-owned facts for the same result addresses.
        Postconditions:
            The reader owns shallow index copies and no result models or trees.
        Essential behavior:
            Keeps an incremental detail context independent of later arena index
            additions or caller rebinding.
        """
        if token is not _ARENA_READER_TOKEN:
            _fail("invalid_arena_record_reader")
        self._owner_token = owner_token
        self._tables = MappingProxyType(
            {
                table: MappingProxyType(dict(records))
                for table, records in tables.items()
            }
        )
        self._result_ids = MappingProxyType(dict(result_ids))
        self._result_content_digests = result_content_digests
        self._result_content_digest_matches_record = (
            result_content_digest_matches_record
        )

    def __setattr__(self, name: str, value: object) -> None:
        """Keep an arena snapshot fixed after the private reader is created.

        Preconditions:
            Initialization assigns each slot once before it becomes visible.
        Postconditions:
            Rebinding an assigned slot raises ``AttributeError``.
        Essential behavior:
            Prevents callers from redirecting a context to another arena view.
        """
        if hasattr(self, name):
            msg = "arena_record_reader_is_immutable"
            raise AttributeError(msg)
        object.__setattr__(self, name, value)

    def _stored_result(self, reference: ResultRef) -> Mapping[str, object]:
        """Return one verified immutable normalized result record.

        Preconditions:
            ``reference`` is an exact result-reference object.
        Postconditions:
            The record family and item key agree with the captured address.
        Essential behavior:
            Rejects foreign result addresses even if their shapes look valid.
        """
        if type(reference) is not ResultRef:
            _fail("invalid_storage_result_reference")
        value = _record_ref_value(
            reference,
            table="results",
            kind=reference.kind,
            result=True,
        )
        table, kind, identifier, item_key = value
        expected = self._result_ids.get(identifier)
        if expected != (kind, item_key):
            _fail("foreign_storage_result_reference")
        entry = self._tables[table].get(identifier)
        if entry is None or entry["kind"] != kind:
            _fail("foreign_storage_result_reference")
        key_field = result_key_field(kind)
        if entry["record"].get(key_field) != item_key:
            _fail("foreign_storage_result_reference")
        return entry["record"]

    def result_binding_facts(
        self, reference: ResultRef
    ) -> tuple[str, str, str]:
        """Select the kind, subject, and plan from one owned prose row.

        Preconditions:
            ``reference`` names an owned planning-card, section, or document
            result captured with its strict prose binding.
        Postconditions:
            Returns ``(kind, subject_key, plan_key)`` from the immutable row.
        Essential behavior:
            Selects the context-bound identity fields without expanding a full
            result model or mistaking its item key for its subject key.
        """
        if type(reference) is not ResultRef or reference.kind not in {
            "planning_card",
            "section",
            "document",
        }:
            _fail("invalid_prose_binding")
        stored = self._stored_result(reference)
        binding = stored.get("binding")
        if not isinstance(binding, Mapping):
            _fail("invalid_prose_binding")
        kind = binding.get("kind")
        subject_key = binding.get("subject_key")
        plan_key = binding.get("plan_key")
        if (
            type(kind) is not str
            or type(subject_key) is not str
            or type(plan_key) is not str
        ):
            _fail("invalid_prose_binding")
        return kind, subject_key, plan_key

    def reusable_result_content_digest(
        self, reference: ResultRef
    ) -> str | None:
        """Return a captured digest only when its logical bytes are identical.

        Preconditions:
            ``reference`` is an exact result row in this reader snapshot.
        Postconditions:
            Returns the typed canonical content digest only when insertion
            proved the raw row already matched defaults and field presence;
            otherwise returns ``None`` for the existing streaming fallback.
        Essential behavior:
            Reuses a stable arena fact only under the same canonical result
            contract used by selected hierarchy results.
        """
        self._stored_result(reference)
        identifier = reference.storage_id
        if not self._result_content_digest_matches_record.get(
            identifier, False
        ):
            return None
        digest = self._result_content_digests.get(identifier)
        if type(digest) is not str:
            _fail("foreign_storage_result_reference")
        return digest

    def result_prose_binding(
        self, reference: ResultRef
    ) -> Mapping[str, object]:
        """Select the complete immutable binding for one owned prose result.

        Preconditions:
            ``reference`` names one arena-owned planning, section, or document
            result with a strictly captured ``ProseBinding``.
        Postconditions:
            Returns all declared binding fields without materializing the result.
        Essential behavior:
            Lets fresh stage validation compare actual owned binding bytes with
            the current request context while keeping storage addresses as data.
        """
        if type(reference) is not ResultRef or reference.kind not in {
            "planning_card",
            "section",
            "document",
        }:
            _fail("invalid_prose_binding")
        stored = self._stored_result(reference)
        binding = stored.get("binding")
        if not isinstance(binding, Mapping) or set(binding) != {
            "version",
            "kind",
            "subject_key",
            "plan_key",
            "context_digest",
        }:
            _fail("invalid_prose_binding")
        if any(type(binding.get(name)) is not str for name in binding):
            _fail("invalid_prose_binding")
        return binding

    def _result_view(self, reference: ResultRef) -> Mapping[str, object]:
        """Expose one immutable normalized result row to private adapters.

        Preconditions:
            ``reference`` names a result in this reader snapshot.
        Postconditions:
            Returns the exact frozen normalized row.
        Essential behavior:
            Performs the reader's exact family and item binding check.
        """
        return self._stored_result(reference)

    def _evidence_view(
        self, pointer: Mapping[str, object], *, kind: str
    ) -> Mapping[str, object]:
        """Return one immutable owned evidence record by an exact pointer.

        Preconditions:
            ``pointer`` is a strict reference to an evidence record of ``kind``.
        Postconditions:
            Returns the frozen evidence row with its stored kind checked.
        Essential behavior:
            Resolves the captured row without invoking a structural parser.
        """
        identifier, pointer_kind = _check_reference_shape(pointer)
        if pointer_kind != kind:
            _fail("wrong_kind_storage_reference")
        entry = self._tables["evidence_refs"].get(identifier)
        if entry is None or entry["kind"] != kind:
            _fail("foreign_storage_record_reference")
        return entry["record"]

    def result_view(self, reference: ResultRef) -> Mapping[str, object]:
        """Return one immutable, complete logical result projection.

        Preconditions:
            ``reference`` is an exact result address owned by this reader.
        Postconditions:
            The returned mapping and nested arrays cannot mutate arena records.
        Essential behavior:
            Resolves only this selected result and its declared nested refs.
        """
        self._stored_result(reference)
        return _freeze(
            _arena_logical_record(
                "results", reference.storage_id, self._tables
            )
        )

    def result_record(self, reference: ResultRef) -> dict:
        """Return one detached logical result for an explicit caller access.

        Preconditions:
            ``reference`` is an exact result address in this reader snapshot.
        Postconditions:
            The caller receives a mutable tree detached from owned primitives.
        Essential behavior:
            Materializes only the selected result and its nested references.
        """
        self._stored_result(reference)
        return _arena_logical_record(
            "results", reference.storage_id, self._tables
        )

    def statement_view(
        self, reference: ResultRef, path: StatementPath | Mapping[str, object]
    ) -> Mapping[str, object]:
        """Return one immutable supported statement by exact path.

        Preconditions:
            ``reference`` is an owned planning, section, or document result;
            ``path`` is a strict allowlisted ``StatementPath`` value.
        Postconditions:
            Only that statement and its typed evidence references are projected.
            All declared statement kinds are retained, including editorial,
            author-motivation, and uncertainty statements.
        Essential behavior:
            Does not reconstruct the full result model or retain the projection.
        """
        stored = self._stored_result(reference)
        kind = reference.kind
        if kind not in {"planning_card", "section", "document"}:
            _fail("invalid_statement_producer")
        raw_path = (
            _raw_model_snapshot(path)
            if isinstance(path, BaseModel)
            else _capture_mapping(path)
        )
        typed_path = _model(StatementPath, raw_path)
        path_value = _raw_model_snapshot(typed_path)
        pointer = {"statement_path": path_value}
        statement = _statement_value_from_normalized_result(
            self._tables, reference.storage_id, pointer
        )
        locations = _statement_reference_locations(statement)
        chunks = _iter_logical_record_json(
            statement,
            references=locations,
            indexes=self._tables,
            logical_lookup=lambda table, identifier: _record_logical_chunks(
                table, identifier, self._tables
            ),
        )
        try:
            logical = json.loads(b"".join(chunks))
        except json.JSONDecodeError as error:
            msg = "invalid_statement_path"
            raise ValueError(msg) from error
        if type(logical) is not dict or logical.get("kind") not in {
            "implementation_claim",
            "author_motivation",
            "editorial",
            "uncertainty",
        }:
            _fail("invalid_statement_path")
        if stored.get("item_key") != reference.item_key:
            _fail("foreign_storage_result_reference")
        return _freeze(logical)

    def statement_paths(
        self, reference: ResultRef
    ) -> tuple[StatementPath, ...]:
        """Enumerate exact prose statement locations without expanding a result.

        Preconditions:
            ``reference`` is an owned planning, section, or document result.
        Postconditions:
            Returns fresh paths in the expanded adapter's field order, followed
            by each ``terminology.definition`` location in input order.
        Essential behavior:
            Reads only the selected row's field lengths and does not project
            any statement or reconstruct its producer model.
        """
        stored = self._stored_result(reference)
        fields = {
            "planning_card": ("abstraction",),
            "section": ("title", "orientation", "synthesis", "transition"),
            "document": ("preamble", "lead", "closing"),
        }
        names = fields.get(reference.kind)
        if names is None:
            _fail("invalid_statement_producer")
        paths: list[StatementPath] = []
        for name in names:
            if name == "title":
                if "title" not in stored:
                    _fail("invalid_statement_path")
                paths.append(StatementPath(field="title"))
                continue
            values = stored.get(name, ())
            if type(values) not in {tuple, list}:
                _fail("invalid_statement_path")
            paths.extend(
                StatementPath(field=name, index=ordinal)
                for ordinal in range(len(values))
            )
        terminology = stored.get("terminology", ())
        if type(terminology) not in {tuple, list}:
            _fail("invalid_statement_path")
        paths.extend(
            StatementPath(field="terminology", index=ordinal)
            for ordinal in range(len(terminology))
        )
        return tuple(paths)

    def terminology_term(self, reference: ResultRef, index: int) -> str:
        """Return one term label while keeping its definition selected.

        Preconditions:
            ``reference`` is an owned prose result and ``index`` selects one
            of its declared terminology entries.
        Postconditions:
            Returns the exact immutable term string without materializing the
            result or its other terminology definitions.
        Essential behavior:
            Shares the existing result-address and path resolver checks.
        """
        stored = self._stored_result(reference)
        if reference.kind not in {"planning_card", "section", "document"}:
            _fail("invalid_statement_producer")
        if type(index) is not int or index < 0:
            _fail("invalid_statement_path")
        terms = stored.get("terminology", ())
        if type(terms) not in {tuple, list} or index >= len(terms):
            _fail("invalid_statement_path")
        term = terms[index]
        if not isinstance(term, Mapping) or type(term.get("term")) is not str:
            _fail("invalid_statement_path")
        return term["term"]

    def logical_chunks(self, reference: ResultRef) -> Iterator[bytes]:
        """Stream exact canonical logical bytes for one owned result.

        Preconditions:
            ``reference`` is an exact result address owned by this reader.
        Postconditions:
            The yielded byte chunks concatenate to the canonical expanded result.
        Essential behavior:
            Follows only the result schema's declared evidence-reference paths.
        """
        self._stored_result(reference)
        yield from _record_logical_chunks(
            "results", reference.storage_id, self._tables
        )


_ARENA_READER_TOKEN = object()


def _producer_relationship_facts(
    producer: ProducerDescriptor,
    record_id: str,
    run_id: str,
) -> Mapping[str, object]:
    """Project one strict producer into the fields used by shared checks.

    Args:
        producer: Strictly validated producer descriptor.
        record_id: Address of the owned producer record.
        run_id: Address of the producer's original run record.

    Returns:
        Mapping[str, object]: Immutable producer identity and request facts.

    Preconditions:
        ``producer`` passed ``ProducerDescriptor`` validation and ``record_id``
        and ``run_id`` select its owned producer and run records.
    Postconditions: The mapping contains identity, request membership, lookup
        kinds, and artifact keys without retaining the model.
    Essential behavior:
        Supplies the same narrow relationship facts to independent parsing and
        arena sealing.
    """
    invocation = producer.invocation
    return MappingProxyType(
        {
            "record_id": record_id,
            "key": producer.key,
            "run_id": run_id,
            "run_digest": content_digest(
                canonical_bytes(producer.original_run)
            ),
            "invocation_semantic_key": invocation.semantic_key,
            "invocation_id": invocation.batch.id,
            "lookup_by_item": MappingProxyType(
                {
                    item.item_key: item.result_kind
                    for item in invocation.lookups
                }
            ),
            "artifact_by_item": MappingProxyType(
                dict(invocation.artifact_keys())
            ),
            "request_items": MappingProxyType(
                {
                    request.batch.id: tuple(request.batch.item_keys)
                    for request in producer.requests
                }
            ),
        }
    )


def _plan_relationship_facts(plan: StoryPlan) -> Mapping[str, object]:
    """Project one strict plan into the selectors needed at graph sealing.

    Args:
        plan: A ``StoryPlan`` that passed strict model validation.

    Returns:
        Immutable identity, group/unit order, and subject ID facts.

    Preconditions:
        ``plan`` passed ``StoryPlan`` validation.
    Postconditions: The mapping has only scalar and tuple relationship facts.
    Essential behavior:
        Retains only relationship facts, not a duplicate plan model or tree.
    """
    return MappingProxyType(
        {
            "id": plan.id,
            "groups": MappingProxyType(
                {
                    item.group.key: tuple(item.group.unit_keys)
                    for item in plan.groups
                }
            ),
            "group_keys": tuple(item.group.key for item in plan.groups),
            "ordered_group_keys": tuple(plan.ordered_group_keys),
            "card_ids": tuple(item.id for item in plan.cards),
            "auxiliary_card_ids": (
                ()
                if plan.production is None
                else tuple(item.id for item in plan.production.auxiliary_cards)
            ),
        }
    )


class _RecordArena:
    """Incrementally own unique normalized results before generation exists.

    Attributes:
        _builder: Mutable append-only normalized table builder until seal.
        _sealed: True after a graph has been published; no later writes are legal.
        _validation_counts: Counts of unique owned rows that passed structural checks.
        _plan_facts: Small immutable group and subject witnesses by plan address.
        _result_digests: Small result-key to logical receipt-digest index.
        _producer_facts: Small immutable audit lookup witnesses by producer key.
    """

    __slots__ = (
        "_builder",
        "_owner_token",
        "_plan_facts",
        "_producer_facts",
        "_producer_keys_by_record_id",
        "_relationship_counts",
        "_result_content_digest_matches_record",
        "_result_content_digests",
        "_result_digests",
        "_sealed",
        "_sealed_tables",
        "_validation_counts",
    )

    _RECORD_TABLES: ClassVar[dict[str, tuple[str, str]]] = {
        "story_plan": ("plans", "story_plan"),
        "span_ref": ("evidence_refs", "span_ref"),
        "source_slice_ref": ("evidence_refs", "source_slice_ref"),
        "statement_ref": ("evidence_refs", "statement_ref"),
        "run": ("runs", "run"),
        "invocation": ("invocations", "invocation"),
        "request": ("requests", "request"),
        "producer": ("producers", "producer"),
        "receipt": ("receipts", "receipt"),
        "resolved_citation": ("resolved_citations", "resolved_citation"),
        "inherited_provenance": (
            "inherited_provenance",
            "inherited_provenance",
        ),
    }

    def __init__(self) -> None:
        """Start an empty append-only arena without a manifest or reader.

        Preconditions:
            None.
        Postconditions:
            No caller record, complete model, or canonical record buffer is held.
        Essential behavior:
            New records are strictly captured and structurally validated once;
            normalized evidence pointers share exact logical records.
        """
        self._builder = _Builder(
            defer_validation=True,
            retain_integrity_copies=False,
            freeze_records=True,
        )
        self._owner_token = object()
        self._plan_facts: dict[str, Mapping[str, object]] = {}
        self._producer_facts: dict[str, Mapping[str, object]] = {}
        self._producer_keys_by_record_id: dict[str, str] = {}
        self._result_content_digests: dict[str, str] = {}
        self._result_content_digest_matches_record: dict[str, bool] = {}
        self._result_digests: dict[str, str] = {}
        self._relationship_counts: dict[str, int] = {}
        self._sealed = False
        self._sealed_tables: (
            Mapping[str, Mapping[str, Mapping[str, object]]] | None
        ) = None
        self._validation_counts: dict[str, int] = {}

    @classmethod
    def _from_parsed_storage(
        cls, parsed: _ParsedStorageImport
    ) -> _RecordArena:
        """Transfer parser-owned immutable rows into one new session arena.

        Preconditions:
            ``parsed`` is the private result of a fresh bytes-only storage
            parse; its rows passed strict independent validation and all row
            contents are recursively immutable.
        Postconditions:
            Returns an open arena whose table entries are the exact same row
            objects. Only top-level indexes and small fact maps are copied.
        Essential behavior:
            It does not trust caller-supplied graph fields or validation counts.
        """
        if type(parsed) is not _ParsedStorageImport:
            _fail("invalid_parsed_storage_import")
        arena = cls()
        facts = parsed.facts
        builder = arena._builder
        builder.records = {
            table: dict(parsed.tables[table]) for table in _TABLES
        }
        builder.record_count = sum(
            len(records) for records in builder.records.values()
        )
        builder.result_ids = dict(facts.result_ids)
        builder.receipt_ids = dict(facts.receipt_ids)
        arena._result_digests = dict(facts.result_digests)
        arena._result_content_digests = dict(facts.result_content_digests)
        arena._result_content_digest_matches_record = dict(
            facts.result_content_digest_matches_record
        )
        arena._plan_facts = dict(facts.plan_facts)
        arena._producer_facts = dict(facts.producer_facts)
        arena._producer_keys_by_record_id = dict(
            facts.producer_keys_by_record_id
        )
        arena._validation_counts = dict(facts.validation_counts)
        return arena

    def _ensure_writable(self) -> _Builder:
        """Return the mutable builder only while this arena remains open.

        Preconditions:
            The caller is about to write or inspect an open capture.
        Postconditions:
            Returns the arena's builder or raises after successful publication.
        Essential behavior:
            No new record can change an already published graph.
        """
        if self._sealed:
            _fail("record_arena_sealed")
        return self._builder

    def _capture_checkpoint(self) -> dict[str, object]:
        """Snapshot indexes before one multi-record insertion attempt.

        Preconditions:
            The arena is open and no capture mutation is in progress.
        Postconditions:
            The checkpoint contains only table IDs and small index copies.
        Essential behavior:
            It enables a failed parent capture to discard only rows created by
            that attempt, leaving previously shared rows unchanged.
        """
        builder = self._ensure_writable()
        return {
            "table_ids": {
                table: set(rows) for table, rows in builder.records.items()
            },
            "record_count": builder.record_count,
            "result_ids": dict(builder.result_ids),
            "receipt_ids": dict(builder.receipt_ids),
            "typed_results": dict(builder.typed_results),
            "validation_counts": dict(self._validation_counts),
            "producer_facts": dict(self._producer_facts),
            "producer_keys_by_record_id": dict(
                self._producer_keys_by_record_id
            ),
            "plan_facts": dict(self._plan_facts),
            "result_digests": dict(self._result_digests),
            "result_content_digests": dict(self._result_content_digests),
            "result_content_digest_matches_record": dict(
                self._result_content_digest_matches_record
            ),
            "relationship_counts": dict(self._relationship_counts),
        }

    def _restore_checkpoint(self, checkpoint: Mapping[str, object]) -> None:
        """Remove rows and indexes added by one failed capture.

        Preconditions:
            ``checkpoint`` came from this arena before the failed insertion.
        Postconditions:
            Record tables, indexes, counters, and receipt witnesses match their
            prior contents exactly.
        Essential behavior:
            Shared rows from earlier successful captures are left in place.
        """
        builder = self._builder
        table_ids = checkpoint["table_ids"]
        if not isinstance(table_ids, Mapping):
            _fail("invalid_record_arena_checkpoint")
        for table, rows in builder.records.items():
            prior_ids = table_ids[table]
            for identifier in set(rows) - prior_ids:
                del rows[identifier]
        builder.record_count = checkpoint["record_count"]
        builder.result_ids = dict(checkpoint["result_ids"])
        builder.receipt_ids = dict(checkpoint["receipt_ids"])
        builder.typed_results = dict(checkpoint["typed_results"])
        self._validation_counts = dict(checkpoint["validation_counts"])
        self._producer_facts = dict(checkpoint["producer_facts"])
        self._producer_keys_by_record_id = dict(
            checkpoint["producer_keys_by_record_id"]
        )
        self._plan_facts = dict(checkpoint["plan_facts"])
        self._result_digests = dict(checkpoint["result_digests"])
        self._result_content_digests = dict(
            checkpoint["result_content_digests"]
        )
        self._result_content_digest_matches_record = dict(
            checkpoint["result_content_digest_matches_record"]
        )
        self._relationship_counts = dict(checkpoint["relationship_counts"])

    @property
    def typed_validation_counts(self) -> dict[str, int]:
        """Return unique table rows captured after strict structural checks.

        Returns:
            dict[str, int]: Detached counts by table and kind. These are row
            counts, not totals for nested Pydantic validator calls.

        Preconditions:
            The arena may be open or sealed.
        Postconditions: Mutating the returned dictionary cannot change arena
            counts.
        Essential behavior:
            Keeps nested checks performed by parent schemas distinct from row
            counts; tests measure those executions with validator tracing.
        """
        return dict(self._validation_counts)

    @property
    def retained_record_counts(self) -> dict[str, int]:
        """Return detached counts of primitive records currently owned.

        Preconditions:
            The arena may be open or sealed.
        Postconditions:
            Returns counts from the current immutable or mutable table indexes.
        Essential behavior:
            Counts rows only and does not materialize any record.
        """
        tables = (
            self._sealed_tables
            if self._sealed_tables is not None
            else self._builder.records
        )
        return {table: len(records) for table, records in tables.items()}

    def _add_validation(self, table: str, kind: str) -> None:
        """Count one new owned row after its strict structural check passed.

        Preconditions:
            The caller completed the row's declared strict schema, directly or
            as a nested value of its owning strict parent schema.
        Postconditions:
            The table/kind execution counter increases by one.
        Essential behavior:
            Does not claim to count nested validator executions, repeated uses,
            or failed rolled-back captures.
        """
        key = f"{table}:{kind}"
        self._validation_counts[key] = self._validation_counts.get(key, 0) + 1

    def _count_new_evidence_rows(self, previous_ids: set[str]) -> None:
        """Count newly owned evidence rows after their strict parent parse.

        Args:
            previous_ids: Evidence addresses owned before the parent parse.

        Preconditions:
            ``previous_ids`` is the evidence table's address set before a
            successful strict parse and builder insertion.
        Postconditions: Each new evidence row contributes one ownership count.
        Essential behavior:
            Counts unique stored rows separately from nested validator calls.
        """
        for identifier in (
            set(self._builder.records["evidence_refs"]) - previous_ids
        ):
            entry = self._builder.records["evidence_refs"][identifier]
            self._add_validation("evidence_refs", entry["kind"])

    def _snapshot_reader(self) -> _ArenaRecordReader:
        """Create a context-local shallow reader over immutable record rows.

        Preconditions:
            Every row visible in the current table indexes is already frozen.
        Postconditions:
            The new reader's table and result indexes do not follow later
            additions to this arena.
        Essential behavior:
            Retains only shallow address maps over the arena's one primitive
            copy of each record.
        """
        tables = (
            self._sealed_tables
            if self._sealed_tables is not None
            else self._builder.records
        )
        result_ids = {
            identifier: pair
            for pair, identifier in self._builder.result_ids.items()
        }
        if self._sealed_tables is not None:
            result_ids = {
                identifier: (reference.kind, reference.item_key)
                for identifier, reference in self._result_refs().items()
            }
        return _ArenaRecordReader(
            _ARENA_READER_TOKEN,
            self._owner_token,
            tables,
            result_ids,
            MappingProxyType(self._result_content_digests),
            MappingProxyType(self._result_content_digest_matches_record),
        )

    def _result_refs(self) -> dict[str, ResultRef]:
        """Build private result bindings from the owned result index.

        Preconditions:
            Each result index entry names a frozen owned result row.
        Postconditions:
            Returns fresh references bound to exact family/item pairs.
        Essential behavior:
            Copies address metadata only, not complete results.
        """
        refs: dict[str, ResultRef] = {}
        for (kind, item_key), identifier in self._builder.result_ids.items():
            refs[identifier] = ResultRef("results", kind, identifier, item_key)
        return refs

    def _snapshot(self, value: object) -> dict:
        """Capture raw model attributes or a caller mapping as JSON primitives.

        Preconditions:
            ``value`` is a supported model or strict JSON mapping.
        Postconditions:
            Returns a detached mapping that preserves raw scalar types.
        Essential behavior:
            Runs before any validation serializer can coerce source values.
        """
        return _capture_mapping(value)

    def _validate_evidence(
        self,
        kind: str,
        value: object,
    ) -> None:
        """Validate one new typed evidence reference from its actual content.

        Args:
            kind: Declared evidence-reference kind.
            value: Raw value or exact supported model to capture.

        Preconditions:
            ``value`` is one raw span, source-slice, or statement reference.
        Postconditions: New references pass their strict schema. Exact owned
            duplicates match complete logical content without another parse.
        Essential behavior:
            A source slice's real nested span is checked by its public schema.
        """
        builder = self._ensure_writable()
        if kind not in {"span_ref", "source_slice_ref", "statement_ref"}:
            _fail("invalid_evidence_reference_kind")
        raw = self._snapshot(value)
        identifier = storage_address(kind, raw)
        existing = builder.records["evidence_refs"].get(identifier)
        if existing is not None:
            if not builder._same_logical_content(  # noqa: SLF001 # Compare complete logical content on address reuse.
                "evidence_refs", identifier, raw
            ):
                _fail("storage_address_collision")
            return
        if kind == "span_ref":
            _model(SpanRef, raw)
        elif kind == "source_slice_ref":
            _model(SourceSliceRef, raw)
        else:
            _model(StatementRef, raw)

    def _validate_result(self, kind: str, raw: dict) -> tuple[str, str, bool]:
        """Strictly parse the complete supplied result and capture its digests.

        Args:
            kind: Registered result kind used to select the public schema.
            raw: Raw-type-preserving snapshot of the complete result.

        Returns:
            tuple[str, str, bool]: Semantic receipt digest, defaulted typed
            content digest, and whether the incoming JSON has identical
            canonical field presence and values.

        Preconditions:
            ``raw`` is a raw-type-preserving snapshot for one supported result.
        Postconditions: The public result schema checks actual nested evidence,
            and the digest witnesses include typed defaults.
        Essential behavior:
            Returns two scalar digests; it retains no typed result model or
            second complete result snapshot.
        """
        validated = _model(_RESULT_MODELS[kind], raw)
        self._add_validation("results", kind)
        logical = validated.model_dump(mode="json")
        logical_bytes = canonical_bytes(logical)
        return (
            semantic_digest("diffstory.receipt-result.v2", logical),
            content_digest(logical_bytes),
            canonical_bytes(raw) == logical_bytes,
        )

    @staticmethod
    def _kind_for_result(value: object, raw: Mapping[str, object]) -> str:
        """Resolve one supported result family from exact type or schema tag.

        Preconditions:
            ``raw`` is the detached snapshot of ``value``.
        Postconditions:
            Returns one declared result kind or rejects unknown inputs.
        Essential behavior:
            Model values require the exact registered model class.
        """
        for kind, model in _RESULT_MODELS.items():
            if type(value) is model:
                return kind
        if type(value) is dict:
            versions = {
                "diffstory.detail-result.v2": "detail",
                "diffstory.planning-card-result.v2": "planning_card",
                "diffstory.section-result.v2": "section",
                "diffstory.document-result.v2": "document",
            }
            kind = versions.get(raw.get("version"))
            if kind is not None:
                return kind
        _fail("invalid_result_kind")

    def _make_result_ref(
        self, kind: str, identifier: str, item_key: str
    ) -> ResultRef:
        """Return a result address bound to one semantic item key.

        Preconditions:
            ``kind``, ``identifier``, and ``item_key`` satisfy result bindings.
        Postconditions:
            Returns a fresh strict immutable result reference.
        Essential behavior:
            Carries the item binding alongside the content address.
        """
        return ResultRef("results", kind, identifier, item_key)

    def put_result(self, value: object) -> ResultRef:
        """Capture and intern one complete detail, planning, section, or document.

        Preconditions:
            ``value`` is an exact supported result model or a strict JSON mapping
            with its corresponding v2 result ``version`` field.
        Postconditions:
            One immutable primitive result row and its nested evidence references
            are owned. Repeated identical inputs return the same address.
        Essential behavior:
            Raw fields are copied before validation; the complete result is
            structurally parsed once and no typed result is retained.
        """
        builder = self._ensure_writable()
        raw = self._snapshot(value)
        kind = self._kind_for_result(value, raw)
        key_field = result_key_field(kind)
        item_key = raw.get(key_field)
        if type(item_key) is not str:
            _fail("invalid_typed_storage_record")
        identity = (kind, item_key)
        identifier = storage_address(kind, raw)
        prior = builder.result_ids.get(identity)
        if prior is not None:
            if prior != identifier or not builder._same_logical_content(  # noqa: SLF001 # Verify complete content on hash reuse.
                "results", prior, raw
            ):
                _fail("duplicate_result_identity")
            return self._make_result_ref(kind, prior, item_key)
        checkpoint = self._capture_checkpoint()
        try:
            (
                result_digest_value,
                result_content_digest,
                content_digest_matches_record,
            ) = self._validate_result(kind, raw)
            evidence_ids = set(builder.records["evidence_refs"])
            pointer = builder.add_result(kind, raw)
            self._count_new_evidence_rows(evidence_ids)
            result_id = pointer["$ref"]
            if result_id != identifier:
                _fail("storage_address_mismatch")
            reference = self._make_result_ref(kind, result_id, item_key)
            self._result_digests[result_id] = result_digest_value
            self._result_content_digests[result_id] = result_content_digest
            self._result_content_digest_matches_record[result_id] = (
                content_digest_matches_record
            )
        except Exception:
            self._restore_checkpoint(checkpoint)
            raise
        return reference

    def _producer_record(
        self,
        raw: dict,
    ) -> tuple[dict[str, str], Mapping[str, object]]:
        """Capture one producer and retain receipt lookup witnesses.

        Preconditions:
            ``raw`` is a detached producer descriptor with nested audit data.
        Postconditions:
            Returns its pointer and a small immutable producer-key index entry.
        Essential behavior:
            Releases the temporary model and retains no complete audit model.
        """
        model = _model(ProducerDescriptor, raw)
        builder = self._ensure_writable()
        pointer = builder.add_producer(raw)
        producer_id = pointer["$ref"]
        self._add_validation("producers", "producer")
        stored = builder.records["producers"][producer_id]["record"]
        run_id = stored["original_run"]["$ref"]
        facts = _producer_relationship_facts(model, producer_id, run_id)
        prior_facts = self._producer_facts.get(model.key)
        if prior_facts is not None and prior_facts["record_id"] != producer_id:
            _fail("duplicate_producer")
        prior_key = self._producer_keys_by_record_id.get(producer_id)
        if prior_key is not None and prior_key != model.key:
            _fail("duplicate_producer")
        self._producer_facts[model.key] = facts
        self._producer_keys_by_record_id[producer_id] = model.key
        return pointer, facts

    def _validate_citation(self, raw: dict) -> None:
        """Validate a citation with its actual source reference.

        Preconditions:
            ``raw`` is a detached resolved-citation mapping.
        Postconditions:
            The source reference, citation fields, consumption, and key agree.
        Essential behavior:
            One strict parent parse traverses the supplied nested reference.
        """
        reference = raw.get("reference")
        if type(reference) is not dict:
            _fail("invalid_typed_storage_record")
        _model(ResolvedCitation, raw)
        reference_kind = (
            "source_slice_ref"
            if reference.get("kind") == "source_slice"
            else "span_ref"
        )
        expected_consumption = (
            "source_slice"
            if reference_kind == "source_slice_ref"
            else "direct_source"
        )
        if raw.get("consumption") != expected_consumption:
            _fail("citation_consumption_mismatch")
        normalized_reference = (
            {
                "kind": "source_slice",
                "source": reference["source"],
                "byte_start": reference["byte_start"],
                "byte_end": reference["byte_end"],
                "content_digest": reference["content_digest"],
            }
            if reference_kind == "source_slice_ref"
            else reference
        )
        if raw.get("key") != semantic_digest(normalized_reference):
            _fail("citation_key_mismatch")

    def _validate_inherited(self, raw: dict) -> None:
        """Validate inherited provenance with its actual nested values.

        Preconditions:
            ``raw`` is a detached inherited-provenance mapping.
        Postconditions:
            The statement reference and statement evidence pass strict checks.
        Essential behavior:
            The public strict schema sees the supplied references, receipt, and
            statement exactly as captured.
        """
        reference = raw.get("reference")
        statement = raw.get("statement")
        if type(reference) is not dict or type(statement) is not dict:
            _fail("invalid_typed_storage_record")
        _model(InheritedProvenance, raw)

    def put_record(self, kind: str, value: object) -> RecordRef:
        """Capture one supported plan, evidence, citation, or audit record.

        Supported ``kind`` values are ``story_plan``, ``span_ref``,
        ``source_slice_ref``, ``statement_ref``, ``run``, ``invocation``,
        ``request``, ``producer``, ``receipt``, ``resolved_citation``, and
        ``inherited_provenance``. Each value is its corresponding strict model
        or a JSON-shaped mapping that meets that model's schema. Source targets,
        URLs, readers, and normalized storage rows are derived internally and
        are not accepted here.

        Preconditions:
            The arena is open, and nested producer/audit records referenced by
            inherited provenance have already been captured when required.
        Postconditions:
            One immutable primitive record is owned for new content; exact
            duplicates share its address and order-sensitive collections keep
            their caller order.
        Essential behavior:
            Captures raw model attributes before serialization, validates new
            records once, and retains no typed input models or canonical copies.
        """
        builder = self._ensure_writable()
        location = self._RECORD_TABLES.get(kind)
        if location is None:
            _fail("invalid_storage_record_kind")
        table, stored_kind = location
        raw = self._snapshot(value)
        identifier = storage_address(stored_kind, raw)
        if identifier in builder.records[table]:
            if not builder._same_logical_content(table, identifier, raw):  # noqa: SLF001 # Compare complete content on address reuse.
                _fail("storage_address_collision")
            return RecordRef(table, stored_kind, identifier)
        checkpoint = self._capture_checkpoint()
        try:
            return self._put_record_new(
                kind, raw, table, stored_kind, identifier
            )
        except Exception:
            self._restore_checkpoint(checkpoint)
            raise

    def _put_record_new(  # noqa: C901 # Capture one supported nested record family.
        self,
        kind: str,
        raw: dict,
        table: str,
        stored_kind: str,
        identifier: str,
    ) -> RecordRef:
        """Validate and intern one new non-result record inside a checkpoint.

        Preconditions:
            No row with ``identifier`` exists in the target table; ``raw`` is
            a detached strict JSON snapshot.
        Postconditions:
            The returned reference names one immutable row with matching kind
            and content address.
        Essential behavior:
            Dispatches to the exact structural validator for the requested
            record family and creates only normalized child references.
        """
        builder = self._ensure_writable()
        if kind == "story_plan":
            plan = _model(StoryPlan, raw)
            pointer = _ref(builder.put(table, stored_kind, raw), stored_kind)
            self._plan_facts[pointer["$ref"]] = _plan_relationship_facts(plan)
            self._add_validation(table, stored_kind)
        elif kind in {"span_ref", "source_slice_ref", "statement_ref"}:
            evidence_ids = set(builder.records["evidence_refs"])
            self._validate_evidence(kind, raw)
            pointer = builder.add_evidence(raw, kind)
            self._count_new_evidence_rows(evidence_ids)
        elif kind == "run":
            _model(RunIdentity, raw)
            pointer = builder.add_run(raw)
            self._add_validation(table, stored_kind)
        elif kind == "invocation":
            _model(ArtifactInvocation, raw)
            pointer = _ref(builder.put(table, stored_kind, raw), stored_kind)
            self._add_validation(table, stored_kind)
        elif kind == "request":
            _model(PreparedMetadata, raw)
            pointer = _ref(builder.put(table, stored_kind, raw), stored_kind)
            self._add_validation(table, stored_kind)
        elif kind == "producer":
            pointer, _facts = self._producer_record(raw)
        elif kind == "receipt":
            _model(ResultReceipt, raw)
            pointer = builder.add_receipt(raw)
            self._add_validation(table, stored_kind)
        elif kind == "resolved_citation":
            self._validate_citation(raw)
            evidence_ids = set(builder.records["evidence_refs"])
            prior_targets = set(builder.records["source_targets"])
            pointer = builder.add_citation(raw)
            self._count_new_evidence_rows(evidence_ids)
            self._add_validation(table, stored_kind)
            for target_id in (
                set(builder.records["source_targets"]) - prior_targets
            ):
                _validate_typed_table_record(
                    "source_targets",
                    "source_target",
                    _thaw(
                        builder.records["source_targets"][target_id]["record"]
                    ),
                )
                self._add_validation("source_targets", "source_target")
        else:
            self._validate_inherited(raw)
            evidence_ids = set(builder.records["evidence_refs"])
            pointer = builder.add_inherited(raw)
            self._count_new_evidence_rows(evidence_ids)
            self._add_validation(table, stored_kind)
        if pointer.get("$ref") != identifier:
            _fail("storage_address_mismatch")
        return RecordRef(table, stored_kind, identifier)

    def result_view(self, reference: ResultRef) -> Mapping[str, object]:
        """Return an immutable full logical projection of one owned result.

        Preconditions:
            ``reference`` has the exact result family, address, and item key
            already owned by this arena.
        Postconditions:
            The selected logical mapping and its nested values are immutable.
        Essential behavior:
            Materializes only the explicitly selected result and does not keep
            that projection after the caller releases it.
        """
        return self._snapshot_reader().result_view(reference)

    def result_digests(self, reference: ResultRef) -> tuple[str, str]:
        """Return the captured receipt and content digests for one result.

        Preconditions:
            ``reference`` is an exact result family, address, and item binding
            already owned by this arena.
        Postconditions:
            Returns the semantic receipt digest followed by the canonical
            logical-content digest captured during strict insertion.
        Essential behavior:
            Reads two scalar witnesses without selecting or parsing the result.
        """
        self._owned_result_reference(reference, kind=reference.kind)
        receipt_digest = self._result_digests.get(reference.storage_id)
        content_digest_value = self._result_content_digests.get(
            reference.storage_id
        )
        if receipt_digest is None or content_digest_value is None:
            _fail("foreign_storage_result_reference")
        return receipt_digest, content_digest_value

    def selected_result_model(self, reference: ResultRef) -> BaseModel:
        """Build one transient strict model for the accepted stage validator.

        Preconditions:
            ``reference`` is an exact result address captured in this arena.
        Postconditions:
            Returns one typed detail, planning, section, or document result.
        Essential behavior:
            Materializes only the selected result and retains no model alias.
        """
        pointer = self._owned_result_reference(reference, kind=reference.kind)
        logical = _arena_logical_record(
            "results", pointer["$ref"], self._builder.records
        )
        return _model(_RESULT_MODELS[reference.kind], logical)

    def selected_plan_model(self, reference: RecordRef) -> StoryPlan:
        """Build the exact plan model needed to rederive the current plan.

        Preconditions:
            ``reference`` selects this arena's one story-plan row.
        Postconditions:
            Returns a strict ``StoryPlan`` for the existing checked planner.
        Essential behavior:
            Does not construct a generation manifest or copy other audit rows.
        """
        pointer = self._owned_record_reference(
            reference, table="plans", kind="story_plan"
        )
        logical = _arena_logical_record(
            "plans", pointer["$ref"], self._builder.records
        )
        return _model(StoryPlan, logical)

    def selected_producer_model(
        self, reference: RecordRef
    ) -> ProducerDescriptor:
        """Build one producer's exact original request closure for audit.

        Preconditions:
            ``reference`` selects an owned producer named by the generation.
        Postconditions:
            Returns a strict producer with its original run, invocation peers,
            and successful-request metadata.
        Essential behavior:
            Materializes one selected producer instead of the full audit set.
        """
        pointer = self._owned_record_reference(
            reference, table="producers", kind="producer"
        )
        logical = _arena_logical_record(
            "producers", pointer["$ref"], self._builder.records
        )
        return _model(ProducerDescriptor, logical)

    def selected_receipt_model(self, reference: RecordRef) -> ResultReceipt:
        """Build one strict receipt for the current stage audit.

        Preconditions:
            ``reference`` selects an owned receipt named by the generation.
        Postconditions:
            Returns one strict receipt model with exact scalar values.
        Essential behavior:
            Keeps receipt validation transient and avoids a complete manifest.
        """
        pointer = self._owned_record_reference(
            reference, table="receipts", kind="receipt"
        )
        logical = _arena_logical_record(
            "receipts", pointer["$ref"], self._builder.records
        )
        return _model(ResultReceipt, logical)

    def plan_assembly_facts(
        self, reference: RecordRef
    ) -> Mapping[str, object]:
        """Return validated plan order and group membership selectors.

        Preconditions:
            ``reference`` is an owned ``story_plan`` record.
        Postconditions:
            Returns immutable plan ID, ordered group keys, ordered group unit
            membership, and planning-card subject IDs from strict capture.
        Essential behavior:
            Selects compact arena facts without reconstructing a ``StoryPlan``.
        """
        pointer = self._owned_record_reference(
            reference, table="plans", kind="story_plan"
        )
        facts = self._plan_facts.get(pointer["$ref"])
        if not isinstance(facts, Mapping):
            _fail("missing_validated_storage_model:plans:story_plan")
        return facts

    def producer_keys(
        self, references: Sequence[RecordRef]
    ) -> tuple[str, ...]:
        """Select producer identities for an ordered owned reference set.

        Preconditions:
            ``references`` is a list or tuple of owned producer references.
        Postconditions:
            Returns unique producer keys in the supplied order.
        Essential behavior:
            Uses each normalized producer row's semantic key to select its
            matching captured facts without scanning unrelated producers.
        """
        if type(references) not in {tuple, list}:
            _fail("invalid_producer_reference_set")
        keys = []
        for reference in references:
            pointer = self._owned_record_reference(
                reference, table="producers", kind="producer"
            )
            key = self._producer_keys_by_record_id.get(pointer["$ref"])
            facts = self._producer_facts.get(key)
            if (
                type(key) is not str
                or not isinstance(facts, Mapping)
                or facts.get("key") != key
                or facts.get("record_id") != pointer["$ref"]
            ):
                _fail("producer_coverage")
            keys.append(key)
        if len(set(keys)) != len(keys):
            _fail("duplicate_producer")
        return tuple(keys)

    def producer_audit_references(
        self, reference: RecordRef
    ) -> tuple[str, RecordRef, RecordRef, tuple[RecordRef, ...]]:
        """Select one producer's run, invocation, and ordered request rows.

        Preconditions:
            ``reference`` is an exact producer record owned by this arena.
        Postconditions:
            Returns its key and typed child references in original order.
        Essential behavior:
            Reads the normalized producer row without constructing a complete
            manifest or expanding unrelated producers.
        """
        pointer = self._owned_record_reference(
            reference, table="producers", kind="producer"
        )
        producer_key = self._producer_keys_by_record_id.get(pointer["$ref"])
        facts = self._producer_facts.get(producer_key)
        if (
            type(producer_key) is not str
            or not isinstance(facts, Mapping)
            or facts.get("key") != producer_key
            or facts.get("record_id") != pointer["$ref"]
        ):
            _fail("invalid_producer_record")
        record = self._builder.records["producers"][pointer["$ref"]]["record"]
        if not isinstance(record, Mapping) or record.get(
            "original_run", {}
        ).get("$ref") != facts.get("run_id"):
            _fail("invalid_producer_record")

        def selected(value: object, *, table: str, kind: str) -> RecordRef:
            identifier, actual_kind = _check_reference_shape(value)
            if actual_kind != kind:
                _fail("wrong_kind_storage_reference")
            pointer = self._owned_record_reference(
                RecordRef(table, kind, identifier), table=table, kind=kind
            )
            return RecordRef(table, kind, pointer["$ref"])

        raw_requests = record.get("requests")
        if type(raw_requests) not in {tuple, list}:
            _fail("invalid_producer_record")
        run_ref = selected(
            record.get("original_run"), table="runs", kind="run"
        )
        invocation_ref = selected(
            record.get("invocation"), table="invocations", kind="invocation"
        )
        request_refs = tuple(
            selected(item, table="requests", kind="request")
            for item in raw_requests
        )
        return producer_key, run_ref, invocation_ref, request_refs

    def selected_audit_record(
        self, reference: RecordRef, *, kind: str
    ) -> Mapping[str, object]:
        """Return one detached immutable run, invocation, or request record.

        Preconditions:
            ``reference`` names an owned row whose kind is one of the three
            supported producer-audit families.
        Postconditions:
            The selected logical mapping is immutable and detached from builder
            indexes; no neighboring row is expanded.
        Essential behavior:
            Supplies a narrow selector for exact completion evidence checks.
        """
        tables = {
            "run": ("runs", "run"),
            "invocation": ("invocations", "invocation"),
            "request": ("requests", "request"),
        }
        selected_table = tables.get(kind)
        if selected_table is None:
            _fail("invalid_storage_audit_kind")
        table, record_kind = selected_table
        pointer = self._owned_record_reference(
            reference, table=table, kind=record_kind
        )
        record = _arena_logical_record(
            table, pointer["$ref"], self._builder.records
        )
        return _freeze(record)

    def receipt_reference_for_item(self, item_key: str) -> RecordRef:
        """Select one captured receipt by its semantic result key.

        Preconditions:
            ``item_key`` is a string naming a receipt captured by this arena.
        Postconditions:
            Returns the exact owned receipt address or rejects missing content.
        Essential behavior:
            Uses the arena's existing item index for constant-time selection.
        """
        if type(item_key) is not str:
            _fail("missing_provenance_receipt")
        identifier = self._builder.receipt_ids.get(item_key)
        if identifier is None:
            _fail("missing_provenance_receipt")
        return RecordRef("receipts", "receipt", identifier)

    def receipt_view(self, reference: RecordRef) -> Mapping[str, object]:
        """Return one immutable selected receipt row without model parsing.

        Preconditions:
            ``reference`` is an exact receipt record owned by this arena.
        Postconditions:
            Returns the frozen primitive receipt fields for that address.
        Essential behavior:
            Resolves one row and does not scan the generation receipt set.
        """
        pointer = self._owned_record_reference(
            reference, table="receipts", kind="receipt"
        )
        return self._builder.records["receipts"][pointer["$ref"]]["record"]

    def put_indexed_inherited_provenance(  # noqa: C901 # Check one exact reference closure before insertion.
        self,
        reference: RecordRef,
        producer: ResultRef,
        receipt: RecordRef,
    ) -> RecordRef:
        """Insert provenance as references to an owned result and receipt.

        Preconditions:
            ``reference`` selects a strictly captured statement reference;
            ``producer`` selects its strictly captured planning, section, or
            document result; and ``receipt`` selects an owned receipt.
        Postconditions:
            One frozen provenance row points to the original result, exact path
            and digests, and matching receipt. No statement or receipt copy is
            retained beside its source rows.
        Essential behavior:
            Reuses selected-path resolution and streamed canonical addressing;
            it does not construct an ``InheritedProvenance`` model.
        """
        builder = self._ensure_writable()
        reference_pointer = self._owned_record_reference(
            reference, table="evidence_refs", kind="statement_ref"
        )
        if type(producer) is not ResultRef or producer.kind not in {
            "planning_card",
            "section",
            "document",
        }:
            _fail("invalid_statement_producer")
        result_pointer = self._owned_result_reference(
            producer, kind=producer.kind
        )
        receipt_pointer = self._owned_record_reference(
            receipt, table="receipts", kind="receipt"
        )
        raw_reference = _thaw(
            builder.records["evidence_refs"][reference_pointer["$ref"]][
                "record"
            ]
        )
        if (
            type(raw_reference) is not dict
            or raw_reference.get("producer_item_key") != producer.item_key
        ):
            _fail("reader_provenance_mismatch")
        result_content_digest = self._result_content_digests.get(
            producer.storage_id
        )
        if (
            result_content_digest is None
            or raw_reference.get("result_digest") != result_content_digest
        ):
            _fail("inherited_statement_mismatch")
        statement = _statement_value_from_normalized_result(
            builder.records, producer.storage_id, raw_reference
        )
        if statement.get("kind") != "implementation_claim":
            _fail("inherited_statement_mismatch")
        statement_locations = _statement_reference_locations(statement)
        statement_chunks = _iter_logical_record_json(
            statement,
            references=statement_locations,
            indexes=builder.records,
            logical_lookup=lambda table, identifier: _record_logical_chunks(
                table, identifier, builder.records
            ),
        )
        statement_digest = hashlib.sha256()
        for chunk in statement_chunks:
            statement_digest.update(chunk)
        if statement_digest.hexdigest() != raw_reference.get(
            "statement_digest"
        ):
            _fail("inherited_statement_mismatch")
        receipt_record = builder.records["receipts"][receipt_pointer["$ref"]][
            "record"
        ]
        if receipt_record.get("item_key") != producer.item_key:
            _fail("reader_provenance_mismatch")

        key = semantic_digest(raw_reference)
        logical_record = {
            "key": key,
            "reference": reference_pointer,
            "statement": statement,
            "consumption": "inherited_statement",
            "receipt": receipt_pointer,
        }
        logical_locations = {
            ("reference",): ("evidence_refs", "statement_ref"),
            ("receipt",): ("receipts", "receipt"),
        }
        logical_locations.update(
            {
                ("statement", *path): location
                for path, location in statement_locations.items()
            }
        )
        logical_chunks = _iter_logical_record_json(
            logical_record,
            references=logical_locations,
            indexes=builder.records,
            logical_lookup=lambda table, identifier: _record_logical_chunks(
                table, identifier, builder.records
            ),
        )
        identifier = _streamed_storage_address(
            "inherited_provenance", logical_chunks
        )
        record = {
            "key": key,
            "reference": reference_pointer,
            "result": result_pointer,
            "consumption": "inherited_statement",
            "receipt": receipt_pointer,
        }
        prior = builder.records["inherited_provenance"].get(identifier)
        if prior is not None:
            same_record = (
                prior["kind"] == "inherited_provenance"
                and canonical_bytes(_thaw(prior["record"]))
                == canonical_bytes(record)
                and _canonical_streams_equal(
                    _record_logical_chunks(
                        "inherited_provenance", identifier, builder.records
                    ),
                    _iter_logical_record_json(
                        logical_record,
                        references=logical_locations,
                        indexes=builder.records,
                        logical_lookup=lambda table,
                        target_id: _record_logical_chunks(
                            table, target_id, builder.records
                        ),
                    ),
                )
            )
            if not same_record:
                _fail("storage_address_collision")
            return RecordRef(
                "inherited_provenance", "inherited_provenance", identifier
            )
        if builder.record_count >= _MAX_RECORDS:
            _fail("too_many_storage_records")
        builder.records["inherited_provenance"][identifier] = MappingProxyType(
            {
                "id": identifier,
                "kind": "inherited_provenance",
                "record": _freeze(record),
            }
        )
        builder.record_count += 1
        self._add_validation("inherited_provenance", "inherited_provenance")
        return RecordRef(
            "inherited_provenance", "inherited_provenance", identifier
        )

    def statement_view(
        self, reference: ResultRef, path: StatementPath | Mapping[str, object]
    ) -> Mapping[str, object]:
        """Return one immutable statement projection by exact path.

        Preconditions:
            ``reference`` is an owned planning, section, or document result;
            ``path`` is a strict allowlisted statement location.
        Postconditions:
            The returned statement and nested evidence values are immutable.
        Essential behavior:
            Resolves only the selected statement and its evidence records.
        """
        return self._snapshot_reader().statement_view(reference, path)

    def statement_paths(
        self, reference: ResultRef
    ) -> tuple[StatementPath, ...]:
        """List exact statement locations without projecting the result.

        Preconditions:
            ``reference`` is an owned planning, section, or document result.
        Postconditions:
            Paths preserve prose-field and terminology input order.
        Essential behavior:
            Reads only stored array lengths from a shallow reader snapshot.
        """
        return self._snapshot_reader().statement_paths(reference)

    def terminology_term(self, reference: ResultRef, index: int) -> str:
        """Return one selected term label from an owned prose result.

        Preconditions:
            ``reference`` is an owned prose result and ``index`` selects an
            existing terminology entry.
        Postconditions:
            Returns the exact term string; no other result fields are exposed.
        Essential behavior:
            Uses the same checked arena result resolver as statement selection.
        """
        return self._snapshot_reader().terminology_term(reference, index)

    def logical_chunks(self, reference: ResultRef) -> Iterator[bytes]:
        """Stream canonical expanded bytes for one owned result address.

        Preconditions:
            ``reference`` is an exact result address already owned by this
            arena.
        Postconditions:
            Concatenated chunks equal the canonical logical result JSON.
        Essential behavior:
            Resolves references while streaming and keeps no expanded result.
        """
        yield from self._snapshot_reader().logical_chunks(reference)

    def _detail_reader(
        self, references: Sequence[ResultRef]
    ) -> _ArenaRecordReader:
        """Create the immutable reader used by private early detail binding.

        Preconditions:
            Every input is an exact result reference already owned by this
            arena and names a detail result.
        Postconditions:
            Returns a shallow immutable snapshot of the visible owned tables.
        Essential behavior:
            A later detail context cannot follow caller or arena index changes.
        """
        reader = self._snapshot_reader()
        for reference in references:
            reader._stored_result(reference)  # noqa: SLF001 # Verify this exact arena owned result binding.
            if reference.kind != "detail":
                _fail("invalid_indexed_detail_reference")
        return reader

    @property
    def relationship_counters(self) -> dict[str, int]:
        """Return receipt-index work performed by the latest successful seal.

        Preconditions:
            A seal has completed its relationship checks.
        Postconditions:
            Returns a detached count of index builds, records, and lookups.
        Essential behavior:
            Counters are incremented at the actual scan and lookup sites.
        """
        return dict(self._relationship_counts)

    def _owned_record_reference(
        self, reference: RecordRef, *, table: str, kind: str
    ) -> dict[str, str]:
        """Require one supplied reference to name an exact owned record.

        Preconditions:
            ``reference`` is a typed address expected for the supplied table
            and kind.
        Postconditions:
            The returned pointer matches the arena's stored kind and address.
        Essential behavior:
            A matching address from another arena is accepted only when its
            complete content is the same content already owned here.
        """
        if type(reference) is not RecordRef:
            _fail("foreign_storage_record_reference")
        value = _record_ref_value(reference, table=table, kind=kind)
        if table not in self._builder.records:
            _fail("foreign_storage_record_reference")
        entry = self._builder.records[table].get(value[2])
        if entry is None or entry["kind"] != kind:
            _fail("foreign_storage_record_reference")
        return _ref(value[2], kind)

    def _owned_result_reference(
        self, reference: ResultRef, *, kind: str
    ) -> dict[str, str]:
        """Require an exact result family, address, and semantic item binding.

        Preconditions:
            ``reference`` is an exact typed result reference.
        Postconditions:
            The returned pointer names a row whose family and result key match.
        Essential behavior:
            The arena's result index and stored raw key both confirm the item
            binding; the digest alone does not grant membership.
        """
        if type(reference) is not ResultRef:
            _fail("foreign_storage_result_reference")
        _table, actual_kind, identifier, item_key = _record_ref_value(
            reference,
            table="results",
            kind=kind,
            result=True,
        )
        entry = self._builder.records["results"].get(identifier)
        if entry is None or entry["kind"] != actual_kind:
            _fail("foreign_storage_result_reference")
        indexed = self._builder.result_ids.get((actual_kind, item_key))
        if indexed != identifier:
            _fail("foreign_storage_result_reference")
        result = entry["record"]
        if result.get(result_key_field(actual_kind)) != item_key:
            _fail("foreign_storage_result_reference")
        return _ref(identifier, actual_kind)

    def _generation_skeleton(self, references: GenerationRecordSet) -> dict:
        """Bind a data-only record set to exact owned arena records.

        Preconditions:
            ``references`` is an exact ``GenerationRecordSet`` value with an
            unchanged header witness.
        Postconditions:
            The returned tree contains only header primitives and normalized
            pointers that resolve to exact arena rows.
        Essential behavior:
            Does not reconstruct or validate a ``GenerationManifest`` model.
        """
        if type(references) is not GenerationRecordSet:
            _fail("invalid_generation_record_set")
        header = _thaw(references.header)
        if (
            type(header) is not dict
            or content_digest(canonical_bytes(header))
            != references._header_witness  # noqa: SLF001 # Verify the immutable constructor snapshot.
        ):
            _fail("invalid_generation_header")
        generation = dict(header)
        generation["run"] = self._owned_record_reference(
            references.run_ref, table="runs", kind="run"
        )
        generation["details"] = [
            self._owned_result_reference(item, kind="detail")
            for item in references.details
        ]
        generation["plan"] = self._owned_record_reference(
            references.plan_ref, table="plans", kind="story_plan"
        )
        generation["planning"] = [
            self._owned_result_reference(item, kind="planning_card")
            for item in references.planning
        ]
        generation["sections"] = [
            self._owned_result_reference(item, kind="section")
            for item in references.sections
        ]
        generation["document"] = self._owned_result_reference(
            references.document, kind="document"
        )
        generation["producers"] = [
            self._owned_record_reference(
                item, table="producers", kind="producer"
            )
            for item in references.producers
        ]
        generation["receipts"] = [
            self._owned_record_reference(
                item, table="receipts", kind="receipt"
            )
            for item in references.receipts
        ]
        return generation

    def _reader_record_from_projection(
        self, projection: IndexedReaderProjection, *, root_shape: str
    ) -> dict:
        """Copy ordered reader selectors into one normalized reader record.

        Preconditions:
            ``projection`` selects the reader family required by ``root_shape``
            and each address matches a row owned by this arena.
        Postconditions:
            The returned selector tree preserves caller order and owns no
            presentation or complete result models.
        Essential behavior:
            Exact content, family, and item bindings are checked while each
            pointer is copied.
        """
        if type(projection) is not IndexedReaderProjection:
            _fail("invalid_indexed_reader_projection")
        expected_shape = (
            "presentation" if root_shape == "report" else "annotations"
        )
        if projection.shape != expected_shape:
            _fail("artifact_reader_kind_mismatch")
        if any(
            type(value) not in {tuple, list}
            for value in (
                projection.sections,
                projection.context_details,
                projection.citations,
                projection.support_provenance,
            )
        ):
            _fail("invalid_indexed_reader_projection")
        document = self._owned_result_reference(
            projection.document, kind="document"
        )
        section_field = (
            "sections" if expected_shape == "presentation" else "steps"
        )
        if expected_shape == "presentation":
            version = getattr(projection, "version", None)
            provenance = getattr(projection, "provenance", None)
            if version is None:
                version = "diffstory.presentation.v2"
            if provenance is None:
                provenance = "model-generated · unverified"
            record: dict = {
                "shape": "presentation",
                "version": version,
                "provenance": provenance,
                "document": document,
                "sections": [],
                "context_details": [],
                "citations": [],
                "support_provenance": [],
            }
        else:
            record = {
                "shape": "annotations",
                "document": document,
                "steps": [],
                "context_details": [],
                "citations": [],
                "support_provenance": [],
            }
        for section in projection.sections:
            if type(section) is not ReaderSectionRef:
                _fail("invalid_indexed_reader_section")
            if type(section.details) not in {tuple, list}:
                _fail("invalid_indexed_reader_section")
            record[section_field].append(
                {
                    "group_id": section.group_id,
                    "group_key": section.group_key,
                    "number": section.number,
                    "next_id": section.next_id,
                    "prose": self._owned_result_reference(
                        section.prose, kind="section"
                    ),
                    "details": [
                        self._owned_result_reference(item, kind="detail")
                        for item in section.details
                    ],
                }
            )
        record["context_details"] = [
            self._owned_result_reference(item, kind="detail")
            for item in projection.context_details
        ]
        record["citations"] = [
            self._owned_record_reference(
                item,
                table="resolved_citations",
                kind="resolved_citation",
            )
            for item in projection.citations
        ]
        record["support_provenance"] = [
            self._owned_record_reference(
                item,
                table="inherited_provenance",
                kind="inherited_provenance",
            )
            for item in projection.support_provenance
        ]
        return record

    def _insert_reader(self, record: dict) -> str:
        """Address, validate, and retain one normalized reader skeleton.

        Preconditions:
            The selector tree uses the closed normalized reader schema and
            points only to already-owned result and audit records.
        Postconditions:
            One immutable reader row is inserted or an exact existing row is
            reused, with its address computed as a logical stream.
        Essential behavior:
            Validates the compact reader record once and retains no expanded
            presentation tree or canonical reader buffer.
        """
        builder = self._ensure_writable()
        _validate_reader_record(record)
        digest = hashlib.sha256()
        digest.update(b'{"kind":')
        digest.update(canonical_bytes("reader"))
        digest.update(b',"record":')
        for chunk in _reader_logical_chunks(record, builder.records):
            digest.update(chunk)
        digest.update(b',"storage_version":')
        digest.update(canonical_bytes(STORAGE_VERSION))
        digest.update(b"}")
        identifier = digest.hexdigest()
        if identifier in builder.records["readers"]:
            prior = builder.records["readers"][identifier]["record"]
            if not _reader_records_stream_equal(
                prior, record, builder.records
            ):
                _fail("storage_address_collision")
            return identifier
        if builder.record_count >= _MAX_RECORDS:
            _fail("too_many_storage_records")
        builder.records["readers"][identifier] = MappingProxyType(
            {
                "id": identifier,
                "kind": "reader",
                "record": _freeze(record),
            }
        )
        builder.record_count += 1
        self._add_validation("readers", "reader")
        return identifier

    @staticmethod
    def _check_arena_root(root: dict, artifact_kind: str) -> None:
        """Apply shared root validation and arena-specific root restrictions.

        Preconditions:
            ``root`` contains the captured artifact root and generation rows.
        Postconditions:
            Root and generation fields satisfy their closed scalar/array schema.
        Essential behavior:
            Reuses the independent parser's root and generation-shell checks.
        """
        if artifact_kind not in {"report", "annotations"}:
            _fail("invalid_artifact_kind")
        root_for_shape = root
        if (
            artifact_kind == "report"
            and type(root.get("base_report")) is _OwnedMapping
        ):
            root_for_shape = dict(root)
            root_for_shape["base_report"] = {}
        _read_root_generation_shape(root_for_shape)
        if (artifact_kind == "report") != (root.get("shape") == "report"):
            _fail("artifact_root_kind_mismatch")
        if artifact_kind == "report" and (
            type(root["base_report"]) not in {dict, _OwnedMapping}
            or {
                "generation",
                "narrative_v2",
            }
            & set(root["base_report"])
        ):
            _fail("invalid_report_base")
        if (
            artifact_kind == "report"
            and type(root["base_report"]) is _OwnedMapping
        ):
            _strict_json_value(root_for_shape)
        else:
            _strict_json_value(root)

    def compare_imported_reader(
        self,
        reader_id: str,
        projection: IndexedReaderProjection,
        *,
        root_shape: str,
    ) -> None:
        """Compare an incoming reader with a fresh indexed derivation.

        Preconditions:
            ``reader_id`` names the parser-validated reader row, and
            ``projection`` was derived from current source, result, and audit
            inputs in this exact arena.
        Postconditions:
            Returns only when complete logical reader bytes, scalar types,
            field presence, citation bindings, provenance, and order match.
        Essential behavior:
            Streams selected result records and never constructs a presentation
            or annotation model tree.
        """
        entry = self._builder.records["readers"].get(reader_id)
        if entry is None or entry["kind"] != "reader":
            _fail("missing_storage_record")
        expected = self._reader_record_from_projection(
            projection, root_shape=root_shape
        )
        if not _reader_records_stream_equal(
            entry["record"], expected, self._builder.records
        ):
            _fail("reader_mismatch")

    def drop_imported_reader(self, reader_id: str) -> None:
        """Remove the validated annotation root's obsolete reader row.

        Preconditions:
            ``reader_id`` is the exact reader row in this fresh imported arena;
            the row has already been compared with the independently derived
            annotation projection.
        Postconditions:
            The open arena no longer retains that root-only reader address.
        Essential behavior:
            Lets a derived report root keep exact table closure and shared rows.
        """
        builder = self._ensure_writable()
        entry = builder.records["readers"].get(reader_id)
        if entry is None or entry["kind"] != "reader":
            _fail("missing_storage_record")
        del builder.records["readers"][reader_id]
        builder.record_count -= 1
        key = "readers:reader"
        count = self._validation_counts.get(key, 0)
        if count <= 1:
            self._validation_counts.pop(key, None)
        else:
            self._validation_counts[key] = count - 1

    def derive_imported_report_graph(
        self,
        generation_refs: GenerationRecordSet,
        reader_projection: IndexedReaderProjection,
        *,
        base_report: object,
    ) -> ReportRecordGraph:
        """Derive a report root over the imported rows without a second proof.

        Preconditions:
            ``base_report`` is the exact immutable capture owned by the
            importing session; incoming annotation reader storage was compared
            before its obsolete row was removed; and generation references are
            the exact independently validated record set.
        Postconditions:
            Returns one report graph whose complete result and audit rows are
            physically shared with the import graph. Only the new reader row
            and small root are added.
        Essential behavior:
            Checks the new reader schema and root closure but does not rerun
            generation, producer, request, receipt, or support proof work.
        """
        builder = self._ensure_writable()
        if type(base_report) is not _OwnedMapping:
            _fail("invalid_report_base")
        reader_record = self._reader_record_from_projection(
            reader_projection, root_shape="report"
        )
        reader_id = self._insert_reader(reader_record)
        root = {
            "shape": "report",
            "base_report": base_report,
            "generation": self._generation_skeleton(generation_refs),
            "reader": _ref(reader_id, "reader"),
        }
        self._check_arena_root(root, "report")
        _check_graph_and_addresses(root, builder.records)

        frozen_tables = MappingProxyType(
            {
                table: MappingProxyType(dict(entries))
                for table, entries in builder.records.items()
            }
        )
        result_by_key: dict[tuple[str, str], ResultRef] = {}
        result_by_id: dict[str, ResultRef] = {}
        for (kind, item_key), identifier in builder.result_ids.items():
            reference = ResultRef("results", kind, identifier, item_key)
            result_by_key[(kind, item_key)] = reference
            result_by_id[identifier] = reference
        graph = ReportRecordGraph(
            _raw=None,
            _artifact_kind="report",
            _root=_freeze_graph_root(root),
            _tables=frozen_tables,
            _result_by_key=MappingProxyType(result_by_key),
            _result_by_id=MappingProxyType(result_by_id),
            _typed_validation_counts=MappingProxyType(
                dict(self._validation_counts)
            ),
        )
        self._sealed_tables = frozen_tables
        self._sealed = True
        return graph

    def seal(
        self,
        generation_refs: GenerationRecordSet,
        reader_projection: IndexedReaderProjection,
        root: Mapping[str, object],
    ) -> ReportRecordGraph:
        """Publish the same incrementally owned records as a closed graph.

        Preconditions:
            All generation and reader result/audit references have been captured
            in this arena. ``root`` is either a report base root or annotations
            metadata root. This arena has not already been sealed.
        Postconditions:
            A graph shares immutable primitive record rows with this arena; the
            graph cannot change and subsequent ingestion is rejected.
        Essential behavior:
            Rechecks closed root fields, addresses, closure, kinds, cycles,
            coverage, order, receipts, provenance, and contextual membership
            without parsing unique table records or reconstructing a manifest,
            reader model, or full presentation tree.
        """
        checkpoint = self._capture_checkpoint()
        try:
            return self._seal_owned_records(
                generation_refs, reader_projection, root
            )
        except Exception:
            if not self._sealed:
                self._restore_checkpoint(checkpoint)
            raise

    def _seal_owned_records(
        self,
        generation_refs: GenerationRecordSet,
        reader_projection: IndexedReaderProjection,
        root: Mapping[str, object],
    ) -> ReportRecordGraph:
        """Validate and publish one checkpointed graph candidate.

        Preconditions:
            The arena is open, and caller references are copied by this seal
            attempt before graph publication.
        Postconditions:
            Returns one immutable graph and closes the arena after all closure
            and relationship checks pass.
        Essential behavior:
            Failed candidates are rolled back by the public ``seal`` wrapper.
        """
        builder = self._ensure_writable()
        generation = self._generation_skeleton(generation_refs)
        captured_root = self._snapshot(root)
        if captured_root.get("shape") not in {"report", "annotations"}:
            _fail("invalid_storage_root_shape")
        reader_record = self._reader_record_from_projection(
            reader_projection, root_shape=captured_root["shape"]
        )
        reader_id = self._insert_reader(reader_record)
        captured_root["generation"] = generation
        captured_root["reader"] = _ref(reader_id, "reader")
        artifact_kind = captured_root["shape"]
        self._check_arena_root(captured_root, artifact_kind)
        _check_graph_and_addresses(captured_root, builder.records)
        _verify_storage_addresses(builder.records)
        producer_facts_by_id = {
            facts["record_id"]: facts
            for facts in self._producer_facts.values()
        }

        def inherited_statement_info(
            reference: Mapping[str, object],
            result_id: str,
        ) -> tuple[str, object]:
            """Return the normalized statement digest and kind.

            Preconditions:
                All pointers belong to the current immutable arena tables.
            Postconditions:
                Returns the exact canonical statement digest and its kind.
            Essential behavior:
                Reconstructs one selected statement through the shared
                canonical resolver without retaining a complete prose result.
            """
            return _normalized_statement_info(
                builder.records, result_id, reference
            )

        self._relationship_counts = {}
        _validate_shared_relationships(
            captured_root,
            builder.records,
            result_digests=self._result_digests,
            result_content_digests=self._result_content_digests,
            producer_facts_by_id=producer_facts_by_id,
            plan_facts_by_id=self._plan_facts,
            inherited_statement_info=inherited_statement_info,
            counters=self._relationship_counts,
        )

        frozen_tables = MappingProxyType(
            {
                table: MappingProxyType(dict(entries))
                for table, entries in builder.records.items()
            }
        )
        result_by_key: dict[tuple[str, str], ResultRef] = {}
        result_by_id: dict[str, ResultRef] = {}
        for (kind, item_key), identifier in builder.result_ids.items():
            reference = ResultRef("results", kind, identifier, item_key)
            result_by_key[(kind, item_key)] = reference
            result_by_id[identifier] = reference
        graph = ReportRecordGraph(
            _raw=None,
            _artifact_kind=artifact_kind,
            _root=_freeze(captured_root),
            _tables=frozen_tables,
            _result_by_key=MappingProxyType(result_by_key),
            _result_by_id=MappingProxyType(result_by_id),
            _typed_validation_counts=MappingProxyType(
                dict(self._validation_counts)
            ),
        )
        self._sealed_tables = frozen_tables
        self._sealed = True
        return graph


def _statement_from_path(
    result: dict,
    reference: dict,
    *,
    typed_result: BaseModel | None = None,
    typed_reference: StatementRef | None = None,
) -> dict:
    """Resolve an allowlisted statement path without copied prose storage."""
    ref = typed_reference or StatementRef.model_validate(reference)
    kind_by_version = {
        "diffstory.planning-card-result.v2": "planning_card",
        "diffstory.section-result.v2": "section",
        "diffstory.document-result.v2": "document",
    }
    result_kind_value = kind_by_version.get(result.get("version"))
    if result_kind_value is None:
        _fail("invalid_statement_path")
    model = _RESULT_MODELS[result_kind_value]
    if not isinstance(typed_result, model):
        typed_result = _model(model, result)
    from .narrative_hierarchy import statement_at

    statement = statement_at(typed_result, ref.statement_path).model_dump(
        mode="json", exclude_unset=True
    )
    if (
        typed_result.item_key != ref.producer_item_key
        or content_digest(canonical_bytes(typed_result)) != ref.result_digest
        or content_digest(canonical_bytes(statement)) != ref.statement_digest
        or statement.get("kind") != "implementation_claim"
    ):
        _fail("inherited_statement_mismatch")
    return statement


def _validate_logical_record(
    view_tables: dict[str, dict[str, dict]],
    table: str,
    identifier: str,
    memo: dict[str, dict],
    typed_records: dict[str, BaseModel] | None = None,
) -> dict:
    """Resolve one normalized row through the canonical streaming resolver.

    Preconditions:
        The row's declared reference positions have valid closed shapes and
        each target exists in ``view_tables``.
    Postconditions:
        Returns one strict logical record and stores its parsed model in the
        optional independent-parser index.
    Essential behavior:
        Uses the same path resolver as arena sealing for citations, inherited
        statements, readers, addresses, and canonical comparisons.
    """
    entry = view_tables[table][identifier]
    record = entry["record"]
    kind = entry["kind"]
    if table == "resolved_citations":
        _expect_keys(
            record,
            {
                "key",
                "reference",
                "byte_start",
                "byte_end",
                "line_start",
                "line_end",
                "consumption",
                "source_target",
                "url",
            },
            "invalid_normalized_resolved_citation",
        )
    elif table == "inherited_provenance":
        _expect_keys(
            record,
            {"key", "reference", "result", "consumption", "receipt"},
            "invalid_normalized_inherited_provenance",
        )
    elif table == "readers":
        _validate_reader_record(record)
    try:
        logical = json.loads(
            b"".join(_record_logical_chunks(table, identifier, view_tables))
        )
    except (TypeError, json.JSONDecodeError) as error:
        msg = "invalid_logical_storage_record"
        raise ValueError(msg) from error
    if type(logical) is not dict:
        _fail("invalid_logical_storage_record")
    typed = _validate_typed_table_record(table, kind, logical)
    if typed is not None and typed_records is not None:
        typed_records[f"{table}:{identifier}"] = typed
    memo[f"{table}:{identifier}"] = copy.deepcopy(logical)
    return logical


def _logical_record_for_validation(
    view_tables: dict[str, dict[str, dict]],
    table: str,
    identifier: str,
) -> tuple[dict | None, bytes, BaseModel | None]:
    """Resolve and strictly validate one logical row without retaining it.

    Preconditions:
        ``table`` and ``identifier`` select one row whose declared references
        passed envelope and graph-closure checks.
    Postconditions:
        Returns its transient logical mapping, exact logical bytes, and strict
        model where that row family has one. Reader rows return their skeleton
        only and are never expanded into a reader tree.
    Essential behavior:
        Shares the parser's closed row checks while keeping import lifetime to
        one logical row and one transient model at a time.
    """
    entry = view_tables[table][identifier]
    record = entry["record"]
    kind = entry["kind"]
    if table == "resolved_citations":
        _expect_keys(
            record,
            {
                "key",
                "reference",
                "byte_start",
                "byte_end",
                "line_start",
                "line_end",
                "consumption",
                "source_target",
                "url",
            },
            "invalid_normalized_resolved_citation",
        )
    elif table == "inherited_provenance":
        _expect_keys(
            record,
            {"key", "reference", "result", "consumption", "receipt"},
            "invalid_normalized_inherited_provenance",
        )
    elif table == "readers":
        _validate_reader_record(record)
        return None, b"", None
    if table == "urls":
        _expect_keys(record, {"value"}, "invalid_url_record")
        logical = {"value": record.get("value")}
        logical_bytes = canonical_bytes(logical)
    else:
        try:
            logical_bytes = b"".join(
                _record_logical_chunks(table, identifier, view_tables)
            )
            logical = json.loads(logical_bytes)
        except (TypeError, json.JSONDecodeError, RecursionError) as error:
            msg = "invalid_logical_storage_record"
            raise ValueError(msg) from error
    if type(logical) is not dict:
        _fail("invalid_logical_storage_record")
    typed = _validate_typed_table_record(table, kind, logical)
    return logical, logical_bytes, typed


def _logical_record(
    view_tables: dict[str, dict[str, dict]],
    table: str,
    identifier: str,
    memo: dict[str, dict],
    typed_records: dict[str, BaseModel] | None = None,
) -> dict:
    key = f"{table}:{identifier}"
    if key in memo:
        return copy.deepcopy(memo[key])
    if table == "urls":
        record = view_tables[table][identifier]["record"]
        _validate_typed_table_record(
            table, view_tables[table][identifier]["kind"], record
        )
        memo[key] = {"value": record["value"]}
        return copy.deepcopy(memo[key])
    return _validate_logical_record(
        view_tables, table, identifier, memo, typed_records
    )


def _logical_reader_section(
    view_tables: dict[str, dict[str, dict]],
    section: dict,
    memo: dict[str, dict],
) -> dict:
    output = {
        key: value
        for key, value in section.items()
        if key not in {"prose", "details"}
    }
    output["prose"] = _logical_record(
        view_tables, "results", section["prose"]["$ref"], memo
    )
    output["details"] = [
        _logical_record(view_tables, "results", item["$ref"], memo)
        for item in section["details"]
    ]
    return output


def _validate_typed_table_record(  # noqa: C901, PLR0911, PLR0912 # Dispatch each table/kind to its closed schema.
    table: str, kind: str, logical: dict
) -> BaseModel | None:
    """Structurally validate one complete logical record and return its model."""
    if table == "results":
        return _model(_RESULT_MODELS[kind], logical)
    if table == "plans":
        return _model(StoryPlan, logical)
    if table == "evidence_refs":
        return _model(
            {
                "span_ref": SpanRef,
                "source_slice_ref": SourceSliceRef,
                "statement_ref": StatementRef,
            }[kind],
            logical,
        )
    if table == "source_targets":
        _expect_keys(
            logical,
            {"fragment_key", "side", "path", "revision"},
            "invalid_source_target",
        )
        if any(type(logical[name]) is not str for name in logical):
            _fail("invalid_source_target")
        if logical["side"] not in {"base", "head"}:
            _fail("invalid_source_target")
    if table == "resolved_citations":
        citation = _model(ResolvedCitation, logical)
        _validate_citation_binding(citation, logical)
        return citation
    if table == "runs":
        return _model(RunIdentity, logical)
    if table == "invocations":
        return _model(ArtifactInvocation, logical)
    if table == "requests":
        return _model(PreparedMetadata, logical)
    if table == "producers":
        return _model(ProducerDescriptor, logical)
    if table == "receipts":
        return _model(ResultReceipt, logical)
    if table == "inherited_provenance":
        return _model(InheritedProvenance, logical)
    if table == "urls":
        _expect_keys(logical, {"value"}, "invalid_url_record")
        if type(logical["value"]) not in {str, type(None)}:
            _fail("invalid_url_record")
    return None


def _derive_report_graph(  # noqa: C901, PLR0915 # Derive one private report root over sealed rows.
    source_graph: ReportRecordGraph,
    records: GenerationRecordSet,
    reader_projection: IndexedReaderProjection,
    *,
    base_report: object,
) -> ReportRecordGraph:
    """Derive a report root over sealed annotation rows without reopening them.

    Preconditions:
        ``source_graph`` is the exact closed annotations graph for ``records``;
        ``reader_projection`` is the checked presentation selector; and
        ``base_report`` is the session-owned immutable compiler report.
    Postconditions:
        Returns an exact-closure report graph with the same immutable result,
        audit, plan, and evidence rows. Only the reader row and root change.
    Essential behavior:
        Reuses reader schema, record-reference, root, and closure predicates;
        it does not mutate or unseal the source graph or rerun production proof.
    """
    if (
        type(source_graph) is not ReportRecordGraph
        or type(records) is not GenerationRecordSet
        or type(reader_projection) is not IndexedReaderProjection
        or type(base_report) is not _OwnedMapping
        or source_graph._artifact_kind != "annotations"  # noqa: SLF001 # Inspect the exact imported graph family.
        or source_graph._raw is not None  # noqa: SLF001 # A session-owned graph must not retain input bytes.
        or source_graph._root.get("shape") != "annotations"  # noqa: SLF001 # Inspect its exact root shape.
    ):
        _fail("invalid_indexed_report_derivation")
    if reader_projection.shape != "presentation" or any(
        type(value) not in {tuple, list}
        for value in (
            reader_projection.sections,
            reader_projection.context_details,
            reader_projection.citations,
            reader_projection.support_provenance,
        )
    ):
        _fail("invalid_indexed_reader_projection")
    if {"generation", "narrative_v2"} & set(base_report):
        _fail("invalid_report_base")

    def result_pointer(reference: object, kind: str) -> dict[str, str]:
        """Resolve one selected result reference against sealed result rows."""
        if type(reference) is not ResultRef or reference.kind != kind:
            _fail("invalid_generation_record_reference")
        actual = source_graph._result_by_id.get(reference.storage_id)  # noqa: SLF001 # Resolve the typed selector in the sealed result index.
        entry = source_graph._tables["results"].get(reference.storage_id)  # noqa: SLF001 # Check the exact immutable result row.
        if (
            actual != reference
            or entry is None
            or entry.get("kind") != kind
            or entry["record"].get(result_key_field(kind))
            != reference.item_key
        ):
            _fail("foreign_storage_result_reference")
        return _ref(reference.storage_id, kind)

    def record_pointer(
        reference: object, table: str, kind: str
    ) -> dict[str, str]:
        """Resolve one selected audit reference against sealed table rows."""
        if (
            type(reference) is not RecordRef
            or reference.table != table
            or reference.kind != kind
        ):
            _fail("invalid_generation_record_reference")
        entry = source_graph._tables[table].get(reference.storage_id)  # noqa: SLF001 # Check the exact immutable audit row.
        if entry is None or entry.get("kind") != kind:
            _fail("foreign_storage_record_reference")
        return _ref(reference.storage_id, kind)

    record: dict[str, object] = {
        "shape": "presentation",
        "version": reader_projection.version or "diffstory.presentation.v2",
        "provenance": reader_projection.provenance
        or "model-generated · unverified",
        "document": result_pointer(reader_projection.document, "document"),
        "sections": [],
        "context_details": [],
        "citations": [],
        "support_provenance": [],
    }
    sections = []
    for section in reader_projection.sections:
        if type(section) is not ReaderSectionRef or type(
            section.details
        ) not in {
            tuple,
            list,
        }:
            _fail("invalid_indexed_reader_section")
        sections.append(
            {
                "group_id": section.group_id,
                "group_key": section.group_key,
                "number": section.number,
                "next_id": section.next_id,
                "prose": result_pointer(section.prose, "section"),
                "details": [
                    result_pointer(item, "detail") for item in section.details
                ],
            }
        )
    record["sections"] = sections
    record["context_details"] = [
        result_pointer(item, "detail")
        for item in reader_projection.context_details
    ]
    record["citations"] = [
        record_pointer(item, "resolved_citations", "resolved_citation")
        for item in reader_projection.citations
    ]
    record["support_provenance"] = [
        record_pointer(item, "inherited_provenance", "inherited_provenance")
        for item in reader_projection.support_provenance
    ]
    _validate_reader_record(record)

    source_root = source_graph._root  # noqa: SLF001 # Reuse the validated source generation selector.
    old_reader_id, old_reader_kind = _check_reference_shape(
        source_root["reader"]
    )
    old_reader = source_graph._tables["readers"].get(old_reader_id)  # noqa: SLF001 # Remove only the root-only annotation reader.
    if (
        old_reader_kind != "reader"
        or old_reader is None
        or old_reader.get("kind") != "reader"
        or old_reader["record"].get("shape") != "annotations"
    ):
        _fail("invalid_indexed_report_derivation")

    reader_id = _streamed_storage_address(
        "reader",
        _reader_logical_chunks(record, source_graph._tables),  # noqa: SLF001 # Resolve through the sealed shared rows.
    )
    reader_entry = _freeze(
        {"id": reader_id, "kind": "reader", "record": record}
    )
    reader_table = dict(source_graph._tables["readers"])  # noqa: SLF001 # Copy only the reader address table.
    reader_table.pop(old_reader_id)
    existing = reader_table.get(reader_id)
    if existing is not None and existing != reader_entry:
        _fail("storage_address_collision")
    reader_table[reader_id] = reader_entry
    tables = dict(source_graph._tables)  # noqa: SLF001 # Reuse every non-reader table view.
    tables["readers"] = MappingProxyType(reader_table)
    frozen_tables = MappingProxyType(tables)

    check_root = {
        "shape": "report",
        "base_report": base_report,
        "generation": _thaw(source_root["generation"]),
        "reader": _ref(reader_id, "reader"),
    }
    _RecordArena._check_arena_root(check_root, "report")  # noqa: SLF001 # Reuse the established private root predicate.
    _check_graph_and_addresses(check_root, frozen_tables)
    application_root = MappingProxyType(
        {
            "shape": "report",
            "base_report": base_report,
            "generation": source_root["generation"],
            "reader": MappingProxyType(_ref(reader_id, "reader")),
        }
    )
    return ReportRecordGraph(
        _raw=None,
        _artifact_kind="report",
        _root=application_root,
        _tables=frozen_tables,
        _result_by_key=source_graph._result_by_key,  # noqa: SLF001 # Share the immutable result selector index.
        _result_by_id=source_graph._result_by_id,  # noqa: SLF001 # Share the immutable reverse selector index.
        _typed_validation_counts=source_graph._typed_validation_counts,  # noqa: SLF001 # Reuse structural validation evidence.
    )


def _annotation_reader_record(
    source_graph: ReportRecordGraph,
    reader_projection: IndexedReaderProjection,
) -> dict[str, object]:
    """Build an annotation reader's pointers from a checked source projection.

    Preconditions:
        ``source_graph`` is a session-owned report graph and
        ``reader_projection`` is its checked annotation reader projection.
    Postconditions:
        Returns only the ordered reader skeleton with pointers to rows already
        owned by the graph.
    Essential behavior:
        Reuses accepted references and performs no graph-wide closure or
        address validation pass.
    """
    if (
        type(source_graph) is not ReportRecordGraph
        or source_graph._artifact_kind != "report"  # noqa: SLF001 # Require the exact parsed/applied report family.
        or source_graph._raw is not None  # noqa: SLF001 # Only session-owned immutable graph views can be derived.
        or source_graph._root.get("shape") != "report"  # noqa: SLF001 # Bind to the root selector checked by the session.
        or type(reader_projection) is not IndexedReaderProjection
        or reader_projection.shape != "annotations"
    ):
        _fail("invalid_indexed_annotation_derivation")

    def result_pointer(reference: object, kind: str) -> dict[str, str]:
        """Resolve one selected result reference against sealed result rows."""
        if type(reference) is not ResultRef or reference.kind != kind:
            _fail("invalid_generation_record_reference")
        actual = source_graph._result_by_id.get(reference.storage_id)  # noqa: SLF001 # Resolve the typed selector in the sealed result index.
        entry = source_graph._tables["results"].get(reference.storage_id)  # noqa: SLF001 # Check the exact immutable result row.
        if (
            actual != reference
            or entry is None
            or entry.get("kind") != kind
            or entry["record"].get(result_key_field(kind))
            != reference.item_key
        ):
            _fail("foreign_storage_result_reference")
        return _ref(reference.storage_id, kind)

    def record_pointer(
        reference: object, table: str, kind: str
    ) -> dict[str, str]:
        """Resolve one selected audit reference against sealed table rows."""
        if (
            type(reference) is not RecordRef
            or reference.table != table
            or reference.kind != kind
        ):
            _fail("invalid_generation_record_reference")
        entry = source_graph._tables[table].get(reference.storage_id)  # noqa: SLF001 # Check the exact immutable audit row.
        if entry is None or entry.get("kind") != kind:
            _fail("foreign_storage_record_reference")
        return _ref(reference.storage_id, kind)

    reader: dict[str, object] = {
        "shape": "annotations",
        "document": result_pointer(reader_projection.document, "document"),
        "steps": [],
        "context_details": [],
        "citations": [],
        "support_provenance": [],
    }
    steps = []
    for section in reader_projection.sections:
        if type(section) is not ReaderSectionRef:
            _fail("invalid_indexed_reader_section")
        steps.append(
            {
                "group_id": section.group_id,
                "group_key": section.group_key,
                "number": section.number,
                "next_id": section.next_id,
                "prose": result_pointer(section.prose, "section"),
                "details": [
                    result_pointer(item, "detail") for item in section.details
                ],
            }
        )
    reader["steps"] = steps
    reader["context_details"] = [
        result_pointer(item, "detail")
        for item in reader_projection.context_details
    ]
    reader["citations"] = [
        record_pointer(item, "resolved_citations", "resolved_citation")
        for item in reader_projection.citations
    ]
    reader["support_provenance"] = [
        record_pointer(item, "inherited_provenance", "inherited_provenance")
        for item in reader_projection.support_provenance
    ]
    return reader


def _derive_annotation_graph(
    source_graph: ReportRecordGraph,
    reader_projection: IndexedReaderProjection,
    *,
    metadata: Mapping[str, object],
) -> ReportRecordGraph:
    """Derive an annotations root over rows from a checked report graph.

    Preconditions:
        ``source_graph`` is a closed report graph, ``reader_projection`` is
        the accepted annotation reader, and ``metadata`` contains the current
        annotation schema and base/head revisions.
    Postconditions:
        Returns an annotation root that shares all accepted generation and
        audit rows and replaces the presentation reader with one annotation
        reader row.
    Essential behavior:
        Computes only the new reader address needed by shared storage; it does
        not revalidate the already sealed graph's closure or addresses.
    """
    if (
        type(source_graph) is not ReportRecordGraph
        or source_graph._artifact_kind != "report"  # noqa: SLF001 # Require the exact parsed/applied report family.
        or source_graph._raw is not None  # noqa: SLF001 # Only session-owned immutable graph views can be derived.
        or source_graph._root.get("shape") != "report"  # noqa: SLF001 # Bind to the root selector checked by the session.
        or type(reader_projection) is not IndexedReaderProjection
        or reader_projection.shape != "annotations"
        or set(metadata) != {"schema", "base_sha", "head_sha"}
        or metadata.get("schema") != "diffstory.annotations.v2"
        or any(
            type(metadata.get(key)) is not str
            for key in ("base_sha", "head_sha")
        )
    ):
        _fail("invalid_indexed_annotation_derivation")

    reader = _annotation_reader_record(source_graph, reader_projection)

    source_root = source_graph._root  # noqa: SLF001 # Reuse the validated report generation selector.
    old_reader_id, old_reader_kind = _check_reference_shape(
        source_root["reader"]
    )
    old_reader = source_graph._tables["readers"].get(old_reader_id)  # noqa: SLF001 # Replace only the root-selected presentation reader.
    if (
        old_reader_kind != "reader"
        or old_reader is None
        or old_reader.get("kind") != "reader"
        or old_reader["record"].get("shape") != "presentation"
    ):
        _fail("invalid_indexed_annotation_derivation")

    reader_id = _streamed_storage_address(
        "reader",
        _reader_logical_chunks(reader, source_graph._tables),  # noqa: SLF001 # Address the logical annotation reader over shared rows.
    )
    reader_entry = _freeze(
        {"id": reader_id, "kind": "reader", "record": reader}
    )
    reader_table = dict(source_graph._tables["readers"])  # noqa: SLF001 # Copy only the small root-reader index.
    reader_table.pop(old_reader_id)
    existing = reader_table.get(reader_id)
    if existing is not None and existing != reader_entry:
        _fail("storage_address_collision")
    reader_table[reader_id] = reader_entry
    tables = dict(source_graph._tables)  # noqa: SLF001 # Share every non-reader table index.
    tables["readers"] = MappingProxyType(reader_table)
    frozen_tables = MappingProxyType(tables)

    frozen_metadata = _freeze(dict(metadata))
    annotation_root = MappingProxyType(
        {
            "shape": "annotations",
            "metadata": frozen_metadata,
            "generation": source_root["generation"],
            "reader": MappingProxyType(_ref(reader_id, "reader")),
        }
    )
    return ReportRecordGraph(
        _raw=None,
        _artifact_kind="annotations",
        _root=annotation_root,
        _tables=frozen_tables,
        _result_by_key=source_graph._result_by_key,  # noqa: SLF001 # Share the immutable result selector index.
        _result_by_id=source_graph._result_by_id,  # noqa: SLF001 # Share the immutable reverse selector index.
        _typed_validation_counts=source_graph._typed_validation_counts,  # noqa: SLF001 # Reuse structural validation evidence.
    )


def _validate_annotation_reader(reader: dict) -> None:
    if set(reader) != {
        "document",
        "steps",
        "context_details",
        "citations",
        "support_provenance",
    }:
        _fail("invalid_annotation_reader")
    _model(DocumentResult, reader["document"])
    if type(reader["steps"]) is not list:
        _fail("invalid_annotation_reader")
    for step in reader["steps"]:
        _model(PresentationSection, step)
    for item in reader["context_details"]:
        _model(DetailResult, item)
    for item in reader["citations"]:
        _model(ResolvedCitation, item)
    for item in reader["support_provenance"]:
        _model(InheritedProvenance, item)


def _validate_reader_record(reader: dict) -> None:  # noqa: C901 # Validate typed reader fields and ordered sections.
    """Validate only the typed indexed skeleton, without expanding its results."""
    shape = reader.get("shape")
    if shape == "presentation":
        expected = {
            "shape",
            "version",
            "provenance",
            "document",
            "sections",
            "context_details",
            "citations",
            "support_provenance",
        }
        section_field = "sections"
        if (
            reader.get("version") != "diffstory.presentation.v2"
            or reader.get("provenance") != "model-generated · unverified"
        ):
            _fail("invalid_reader_metadata")
    elif shape == "annotations":
        expected = {
            "shape",
            "document",
            "steps",
            "context_details",
            "citations",
            "support_provenance",
        }
        section_field = "steps"
    else:
        _fail("invalid_reader_shape")
    _expect_keys(reader, expected, "invalid_reader_fields")
    for field_name in (
        "sections",
        "steps",
        "context_details",
        "citations",
        "support_provenance",
    ):
        if field_name in reader and type(reader[field_name]) not in {
            list,
            tuple,
        }:
            _fail("invalid_reader_array")
    if type(reader["document"]) is not dict:
        _fail("invalid_reader_document_reference")
    _check_reference_shape(reader["document"])
    for section in reader[section_field]:
        _expect_keys(
            section,
            {"group_id", "group_key", "number", "next_id", "prose", "details"},
            "invalid_reader_section_fields",
        )
        if (
            type(section["group_id"]) is not str
            or not _LEGACY_ID_PATTERN.fullmatch(section["group_id"])
            or type(section["group_key"]) is not str
            or not _ID_PATTERN.fullmatch(section["group_key"])
            or type(section["number"]) is not int
            or section["number"] < 1
            or (
                section["next_id"] is not None
                and (
                    type(section["next_id"]) is not str
                    or not _LEGACY_ID_PATTERN.fullmatch(section["next_id"])
                )
            )
            or type(section["details"]) not in {list, tuple}
        ):
            _fail("invalid_reader_section")
        _check_reference_shape(section["prose"])
        for detail in section["details"]:
            _check_reference_shape(detail)
    for field_name in ("context_details", "citations", "support_provenance"):
        for reference in reader[field_name]:
            _check_reference_shape(reference)


def _relationship_record_at(
    indexes: Mapping[str, Mapping[str, Mapping[str, object]]],
    table: str,
    pointer: Mapping[str, object],
    expected_kind: str,
) -> tuple[str, Mapping[str, object]]:
    """Resolve one checked relationship pointer to its normalized row.

    Preconditions:
        ``pointer`` is a strict table address and ``indexes`` contains the
        tables being checked.
    Postconditions:
        Returns the exact row only when its table and kind agree.
    Essential behavior:
        Gives parser and arena checks the same wrong-kind rejection path.
    """
    identifier, kind = _check_reference_shape(pointer)
    entry = indexes.get(table, {}).get(identifier)
    if (
        entry is None
        or (expected_kind and kind != expected_kind)
        or entry["kind"] != kind
    ):
        _fail("wrong_kind_storage_reference")
    return identifier, entry["record"]


def _validate_shared_relationships(  # noqa: C901, PLR0912, PLR0913, PLR0915 # One projection-based check serves parsing and sealing.
    root: Mapping[str, object],
    indexes: Mapping[str, Mapping[str, Mapping[str, object]]],
    *,
    result_digests: Mapping[str, str],
    result_content_digests: Mapping[str, str],
    producer_facts_by_id: Mapping[str, Mapping[str, object]],
    plan_facts_by_id: Mapping[str, Mapping[str, object]],
    inherited_statement_info: Callable[
        [Mapping[str, object], str], tuple[str, object]
    ],
    counters: dict[str, int] | None = None,
) -> None:
    """Validate generation and reader bindings over small shared projections.

    Args:
        root: Normalized graph root with generation and reader pointers.
        indexes: Normalized record tables indexed by storage address.
        result_digests: Receipt digests keyed by result address.
        result_content_digests: Content digests keyed by result address.
        producer_facts_by_id: Strict producer facts keyed by producer address.
        plan_facts_by_id: Strict plan facts keyed by plan address.
        inherited_statement_info: Resolver for selected inherited statements.
        counters: Optional instrumentation for indexed receipt work.

    Preconditions:
        Independent parsing has strictly parsed every row, or arena capture
        established strict structural witnesses for every owned row. Both
        inputs use the normalized table-reference representation.
    Postconditions:
        Coverage, order, producer, request, receipt, run, reader, and inherited
        statement relationships agree or raise their established error.
    Essential behavior:
        Applies one set of relationship rules without building complete models
        or reader/generation trees at seal time.

    Raises:
        ValueError: If any generation, receipt, producer, reader, or inherited
            statement relationship is invalid.
    """
    generation = root["generation"]
    if not isinstance(generation, Mapping):
        _fail("invalid_generation_skeleton")
    details = generation["details"]
    planning = generation["planning"]
    sections = generation["sections"]
    document_pointer = generation["document"]
    plan_pointer = generation["plan"]
    run = None
    if generation["run"] is not None:
        _, run = _relationship_record_at(
            indexes, "runs", generation["run"], "run"
        )
    current_run_digest = (
        None
        if run is None
        else content_digest(
            canonical_bytes(RunIdentity.model_construct(**_thaw(run)))
        )
    )
    plan_id, _plan = _relationship_record_at(
        indexes, "plans", plan_pointer, "story_plan"
    )
    plan = plan_facts_by_id.get(plan_id)
    if plan is None:
        _fail("missing_validated_storage_model:plans:story_plan")

    result_groups = (
        (details, "detail"),
        (planning, "planning_card"),
        (sections, "section"),
        ((document_pointer,), "document"),
    )
    result_by_key: dict[str, tuple[str, str, Mapping[str, object]]] = {}
    for pointers, expected_kind in result_groups:
        for pointer in pointers:
            identifier, record = _relationship_record_at(
                indexes, "results", pointer, expected_kind
            )
            item_key = record.get(result_key_field(expected_kind))
            if type(item_key) is not str:
                _fail("invalid_typed_storage_record")
            if item_key in result_by_key:
                _fail("duplicate_result_identity")
            result_by_key[item_key] = (expected_kind, identifier, record)

    unit_keys = generation["unit_keys"]
    units = set(unit_keys)
    if len(units) != len(unit_keys):
        _fail("duplicate_ledger_unit")
    detail_records = [
        _relationship_record_at(indexes, "results", pointer, "detail")[1]
        for pointer in details
    ]
    detail_by_unit = {
        record["unit_key"]: pointer
        for record, pointer in zip(detail_records, details, strict=True)
    }
    if len(detail_records) != len(unit_keys) or set(detail_by_unit) != units:
        _fail("detail_coverage")

    groups_by_key = plan["groups"]
    group_keys = plan["group_keys"]
    ordered_group_keys = plan["ordered_group_keys"]
    if len(set(group_keys)) != len(group_keys):
        _fail("duplicate_plan_group")
    if len(ordered_group_keys) != len(group_keys) or set(
        ordered_group_keys
    ) != set(group_keys):
        _fail("plan_group_order")
    if any(
        not set(group_unit_keys) <= units
        for group_unit_keys in groups_by_key.values()
    ):
        _fail("plan_unit_coverage")
    section_records = [
        _relationship_record_at(indexes, "results", pointer, "section")[1]
        for pointer in sections
    ]
    section_subjects = tuple(
        record["binding"]["subject_key"] for record in section_records
    )
    if section_subjects != tuple(ordered_group_keys):
        _fail("section_coverage")

    card_ids = plan["card_ids"]
    subjects = [*card_ids, *plan["auxiliary_card_ids"]]
    if len(set(card_ids)) != len(card_ids) or len(set(subjects)) != len(
        subjects
    ):
        _fail("duplicate_plan_card")
    planning_records = [
        _relationship_record_at(indexes, "results", pointer, "planning_card")[
            1
        ]
        for pointer in planning
    ]
    planning_subjects = [
        item["binding"]["subject_key"] for item in planning_records
    ]
    if len(planning_records) != len(subjects) or set(planning_subjects) != set(
        subjects
    ):
        _fail("planning_coverage")
    document = _relationship_record_at(
        indexes, "results", document_pointer, "document"
    )[1]
    prose_records = [*planning_records, *section_records, document]
    plan_id = plan["id"]
    if any(item["binding"]["plan_key"] != plan_id for item in prose_records):
        _fail("prose_plan_mismatch")

    producer_ids = [
        _relationship_record_at(indexes, "producers", pointer, "producer")[0]
        for pointer in generation["producers"]
    ]
    selected_producer_facts = [
        producer_facts_by_id.get(identifier) for identifier in producer_ids
    ]
    if any(facts is None for facts in selected_producer_facts):
        _fail("producer_coverage")
    producer_keys = [facts["key"] for facts in selected_producer_facts]
    if any(type(key) is not str for key in producer_keys):
        _fail("producer_coverage")
    if len(set(producer_ids)) != len(producer_ids) or len(
        set(producer_keys)
    ) != len(producer_keys):
        _fail("duplicate_producer")
    producer_facts_by_key = dict(
        zip(producer_keys, selected_producer_facts, strict=True)
    )
    producer_identities = [
        (
            facts["run_digest"],
            facts["invocation_semantic_key"],
            facts["invocation_id"],
        )
        for facts in producer_facts_by_key.values()
    ]
    if len(set(producer_identities)) != len(producer_identities):
        _fail("duplicate_original_producer")

    receipt_by_item: dict[str, tuple[str, Mapping[str, object]]] = {}
    receipt_scan_count = 0
    for pointer in generation["receipts"]:
        receipt_id, receipt = _relationship_record_at(
            indexes, "receipts", pointer, "receipt"
        )
        receipt_scan_count += 1
        item_key = receipt.get("item_key")
        if type(item_key) is not str or item_key in receipt_by_item:
            _fail("receipt_coverage")
        receipt_by_item[item_key] = (receipt_id, receipt)
    if counters is not None:
        counters["receipt_index_builds"] = (
            counters.get("receipt_index_builds", 0) + 1
        )
        counters["receipt_index_records_scanned"] = (
            counters.get("receipt_index_records_scanned", 0)
            + receipt_scan_count
        )
        counters.setdefault("receipt_lookups", 0)
    if set(receipt_by_item) != set(result_by_key):
        _fail("receipt_coverage")
    if {
        receipt["producer_key"]
        for _identifier, receipt in receipt_by_item.values()
    } != set(producer_facts_by_key):
        _fail("producer_coverage")

    used_requests: set[tuple[str, str]] = set()
    sent_requests: set[tuple[str, str]] = set()
    for receipt in (row for _identifier, row in receipt_by_item.values()):
        item = result_by_key.get(receipt["item_key"])
        producer_facts = producer_facts_by_key.get(receipt["producer_key"])
        if item is None or producer_facts is None:
            _fail("receipt_binding_mismatch")
        result_kind_value, result_id, _result = item
        if receipt["result_kind"] != result_kind_value:
            _fail("receipt_binding_mismatch")
        if receipt["result_digest"] != result_digests.get(result_id):
            _fail("receipt_result_mismatch")
        request_key = receipt["request_key"]
        request_items = producer_facts["request_items"].get(request_key)
        lookup_kind = producer_facts["lookup_by_item"].get(receipt["item_key"])
        if (
            request_items is None
            or lookup_kind is None
            or lookup_kind != receipt["result_kind"]
            or receipt["item_key"] not in request_items
        ):
            _fail("receipt_binding_mismatch")
        if receipt["artifact_key"] != producer_facts["artifact_by_item"].get(
            receipt["item_key"]
        ):
            _fail("receipt_artifact_mismatch")
        identity = (receipt["producer_key"], request_key)
        used_requests.add(identity)
        if receipt["disposition"] == "sent":
            sent_requests.add(identity)
        if run is not None:
            producer_run = indexes["runs"][producer_facts["run_id"]]["record"]
            if (
                producer_run["provider_destination"]
                != run["provider_destination"]
                or producer_run["model"] != run["model"]
            ):
                _fail("producer_identity_mismatch")
            if (
                receipt["disposition"] == "sent"
                and producer_facts["run_digest"] != current_run_digest
            ):
                _fail("sent_run_mismatch")
    expected_requests = {
        (producer_key, request_key)
        for producer_key, facts in producer_facts_by_key.items()
        for request_key in facts["request_items"]
    }
    if used_requests != expected_requests:
        _fail("unused_production_request")
    current_usage = generation.get("current_usage")
    if (
        not isinstance(current_usage, Mapping)
        or type(current_usage.get("calls")) is not int
        or len(sent_requests) > current_usage["calls"]
    ):
        _fail("sent_usage_mismatch")

    _reader_id, reader = _relationship_record_at(
        indexes, "readers", root["reader"], "reader"
    )
    if reader["document"]["$ref"] != document_pointer["$ref"]:
        _fail("reader_document_mismatch")
    section_field = (
        "sections" if reader["shape"] == "presentation" else "steps"
    )
    reader_sections = reader[section_field]
    if len(reader_sections) != len(sections) or len(reader_sections) != len(
        ordered_group_keys
    ):
        _fail("reader_section_mismatch")
    assigned_units = {
        unit_key
        for group_unit_keys in groups_by_key.values()
        for unit_key in group_unit_keys
    }
    for position, (section, prose_pointer, group_key) in enumerate(
        zip(reader_sections, sections, ordered_group_keys, strict=True)
    ):
        prose = indexes["results"][prose_pointer["$ref"]]["record"]
        expected_next = (
            reader_sections[position + 1]["group_id"]
            if position + 1 < len(reader_sections)
            else None
        )
        expected_details = [
            detail_by_unit[unit_key]["$ref"]
            for unit_key in groups_by_key[group_key]
        ]
        if (
            section["prose"]["$ref"] != prose_pointer["$ref"]
            or section["group_key"] != group_key
            or section["group_key"] != prose["binding"]["subject_key"]
            or section["number"] != position + 1
            or section["next_id"] != expected_next
        ):
            _fail("reader_section_mismatch")
        if [item["$ref"] for item in section["details"]] != expected_details:
            _fail("reader_detail_mismatch")
    expected_context = [
        pointer["$ref"]
        for pointer, record in zip(details, detail_records, strict=True)
        if record["unit_key"] not in assigned_units
    ]
    if [
        item["$ref"] for item in reader["context_details"]
    ] != expected_context:
        _fail("reader_context_mismatch")

    for pointer in reader["support_provenance"]:
        provenance_id, provenance = _relationship_record_at(
            indexes,
            "inherited_provenance",
            pointer,
            "inherited_provenance",
        )
        _reference_id, reference = _relationship_record_at(
            indexes,
            "evidence_refs",
            provenance["reference"],
            "statement_ref",
        )
        result_id, result = _relationship_record_at(
            indexes, "results", provenance["result"], ""
        )
        receipt_id, receipt = _relationship_record_at(
            indexes, "receipts", provenance["receipt"], "receipt"
        )
        if counters is not None:
            counters["receipt_lookups"] += 1
        matching = receipt_by_item.get(reference["producer_item_key"])
        if (
            reference["producer_item_key"] != receipt["item_key"]
            or reference["producer_item_key"]
            != result.get(
                result_key_field(indexes["results"][result_id]["kind"])
            )
            or reference["result_digest"]
            != result_content_digests.get(result_id)
            or provenance["result"]["kind"]
            != indexes["results"][result_id]["kind"]
            or matching is None
            or matching[0] != receipt_id
            or canonical_bytes(_thaw(matching[1]))
            != canonical_bytes(_thaw(receipt))
        ):
            _fail("reader_provenance_receipt_mismatch")
        statement_digest, statement_kind = inherited_statement_info(
            reference, result_id
        )
        if (
            statement_digest != reference["statement_digest"]
            or statement_kind != "implementation_claim"
        ):
            _fail("inherited_statement_mismatch")


def _parser_producer_facts(
    root: Mapping[str, object],
    indexes: Mapping[str, Mapping[str, Mapping[str, object]]],
    typed_records: Mapping[str, BaseModel],
) -> dict[str, Mapping[str, object]]:
    """Build small producer witnesses from independently parsed models.

    Args:
        root: Normalized graph root with producer references.
        indexes: Normalized record tables indexed by storage address.
        typed_records: Models produced by independent strict row parsing.

    Returns:
        dict[str, Mapping[str, object]]: Facts keyed by producer address.

    Preconditions:
        Every generation producer row passed strict table parsing.
    Postconditions: Each referenced producer has one immutable witness.
    Essential behavior:
        Adapts validated models to the same facts used by arena sealing.
    """
    generation = root["generation"]
    facts_by_id = {}
    for pointer in generation["producers"]:
        producer_id, _record = _relationship_record_at(
            indexes, "producers", pointer, "producer"
        )
        model = typed_records.get(f"producers:{producer_id}")
        if not isinstance(model, ProducerDescriptor):
            _fail("missing_validated_storage_model:producers:producer")
        producer_record = indexes["producers"][producer_id]["record"]
        run_id, _run = _relationship_record_at(
            indexes, "runs", producer_record["original_run"], "run"
        )
        facts_by_id[producer_id] = _producer_relationship_facts(
            model, producer_id, run_id
        )
    return facts_by_id


def _validate_parser_relationships(
    root: Mapping[str, object],
    indexes: Mapping[str, Mapping[str, Mapping[str, object]]],
    typed_records: Mapping[str, BaseModel],
    *,
    counters: dict[str, int] | None = None,
) -> None:
    """Adapt strict parser records to the shared relationship validator.

    Args:
        root: Normalized graph root with generation and reader pointers.
        indexes: Normalized record tables indexed by storage address.
        typed_records: Models produced by independent strict row parsing.
        counters: Optional instrumentation for indexed receipt work.

    Preconditions:
        All rows have passed independent strict parsing and logical capture.
    Postconditions:
        The shared generation and reader rules pass or preserve their errors.
    Essential behavior:
        Supplies only digest and audit witnesses, not another record tree.

    Raises:
        ValueError: If a required witness is missing or shared relationships
            fail.
    """
    result_digests = {}
    result_content_digests = {}
    for identifier in indexes["results"]:
        model = typed_records.get(f"results:{identifier}")
        if model is None:
            _fail("missing_validated_storage_model:results")
        result_digests[identifier] = result_digest(model)
        result_content_digests[identifier] = content_digest(
            canonical_bytes(model)
        )
    producer_facts_by_id = _parser_producer_facts(root, indexes, typed_records)
    plan_id, _ = _relationship_record_at(
        indexes, "plans", root["generation"]["plan"], "story_plan"
    )
    plan_model = typed_records.get(f"plans:{plan_id}")
    if not isinstance(plan_model, StoryPlan):
        _fail("missing_validated_storage_model:plans:story_plan")
    plan_facts_by_id = {plan_id: _plan_relationship_facts(plan_model)}

    def inherited_statement_info(
        reference: Mapping[str, object], result_id: str
    ) -> tuple[str, object]:
        """Return the selected statement digest and kind from parsed rows.

        Preconditions:
            The inherited-provenance row passed strict logical validation.
        Postconditions:
            Returns a digest and kind for the selected statement.
        Essential behavior:
            Uses the same normalized selector and streaming resolver as sealing.
        """
        return _normalized_statement_info(indexes, result_id, reference)

    _validate_shared_relationships(
        root,
        indexes,
        result_digests=result_digests,
        result_content_digests=result_content_digests,
        producer_facts_by_id=producer_facts_by_id,
        plan_facts_by_id=plan_facts_by_id,
        inherited_statement_info=inherited_statement_info,
        counters=counters,
    )


def _validate_table_records(
    root: dict,
    indexes: dict[str, dict[str, dict]],
    *,
    counters: dict[str, int] | None = None,
) -> tuple[dict[str, dict], dict[str, BaseModel], dict[str, int]]:
    memo: dict[str, dict] = {}
    typed_records: dict[str, BaseModel] = {}
    validation_counts: dict[str, int] = {}
    tables = indexes
    for table in _TABLES:
        for identifier, entry in tables[table].items():
            _logical_record(tables, table, identifier, memo, typed_records)
            count_key = f"{table}:{entry['kind']}"
            validation_counts[count_key] = (
                validation_counts.get(count_key, 0) + 1
            )
    _validate_parser_relationships(
        root, tables, typed_records, counters=counters
    )
    return memo, typed_records, validation_counts


def _validate_import_table_records(  # noqa: C901, PLR0912, PLR0915
    root: Mapping[str, object],
    indexes: dict[str, dict[str, dict]],
) -> _StorageImportFacts:
    """Validate imported rows one at a time and keep only compact witnesses.

    Preconditions:
        ``root`` and ``indexes`` passed the closed envelope, reference-closure,
        cycle, and canonical-address checks for one storage byte input.
    Postconditions:
        Every unique logical row passed its existing strict schema and shared
        relationship checks. Only immutable digest and relationship facts
        remain; parsed models and expanded logical maps are released per row.
    Essential behavior:
        Reader rows use their skeleton validator and are never expanded into a
        presentation or complete annotation tree.
    """
    validation_counts: dict[str, int] = {}
    result_ids: dict[tuple[str, str], str] = {}
    receipt_ids: dict[str, str] = {}
    result_digests: dict[str, str] = {}
    result_content_digests: dict[str, str] = {}
    result_content_digest_matches_record: dict[str, bool] = {}
    producer_facts_by_id: dict[str, Mapping[str, object]] = {}
    producer_facts_by_key: dict[str, Mapping[str, object]] = {}
    producer_keys_by_record_id: dict[str, str] = {}
    plan_facts: dict[str, Mapping[str, object]] = {}

    for table in _TABLES:
        for identifier, entry in indexes[table].items():
            logical, logical_bytes, model = _logical_record_for_validation(
                indexes, table, identifier
            )
            count_key = f"{table}:{entry['kind']}"
            validation_counts[count_key] = (
                validation_counts.get(count_key, 0) + 1
            )
            if table == "results":
                if model is None:
                    _fail("missing_validated_storage_model:results")
                item_key = result_key(model)
                pair = (entry["kind"], item_key)
                if pair in result_ids:
                    _fail("duplicate_result_identity")
                result_ids[pair] = identifier
                result_digests[identifier] = result_digest(model)
                typed_bytes = canonical_bytes(model)
                result_content_digests[identifier] = content_digest(
                    typed_bytes
                )
                result_content_digest_matches_record[identifier] = (
                    logical_bytes == typed_bytes
                )
            elif table == "plans":
                if not isinstance(model, StoryPlan):
                    _fail("missing_validated_storage_model:plans:story_plan")
                plan_facts[identifier] = _plan_relationship_facts(model)
            elif table == "producers":
                if not isinstance(model, ProducerDescriptor):
                    _fail("missing_validated_storage_model:producers:producer")
                if logical is None:
                    _fail("invalid_logical_storage_record")
                run_pointer = entry["record"].get("original_run")
                run_id, _run_kind = _check_reference_shape(run_pointer)
                facts = _producer_relationship_facts(model, identifier, run_id)
                producer_key = facts.get("key")
                if (
                    type(producer_key) is not str
                    or producer_key in producer_facts_by_key
                ):
                    _fail("duplicate_producer")
                producer_facts_by_id[identifier] = facts
                producer_facts_by_key[producer_key] = facts
                producer_keys_by_record_id[identifier] = producer_key
            elif table == "receipts":
                if not isinstance(model, ResultReceipt):
                    _fail("missing_validated_storage_model:receipts:receipt")
                if model.item_key in receipt_ids:
                    _fail("receipt_coverage")
                receipt_ids[model.item_key] = identifier
            del logical, logical_bytes, model

    plan_id, _plan = _relationship_record_at(
        indexes, "plans", root["generation"]["plan"], "story_plan"
    )
    if plan_id not in plan_facts:
        _fail("missing_validated_storage_model:plans:story_plan")

    def inherited_statement_info(
        reference: Mapping[str, object], result_id: str
    ) -> tuple[str, object]:
        """Return the selected statement facts through the shared resolver."""
        return _normalized_statement_info(indexes, result_id, reference)

    _validate_shared_relationships(
        root,
        indexes,
        result_digests=result_digests,
        result_content_digests=result_content_digests,
        producer_facts_by_id=producer_facts_by_id,
        plan_facts_by_id={plan_id: plan_facts[plan_id]},
        inherited_statement_info=inherited_statement_info,
    )
    return _StorageImportFacts(
        result_ids=MappingProxyType(result_ids),
        receipt_ids=MappingProxyType(receipt_ids),
        result_digests=MappingProxyType(result_digests),
        result_content_digests=MappingProxyType(result_content_digests),
        result_content_digest_matches_record=MappingProxyType(
            result_content_digest_matches_record
        ),
        plan_facts=MappingProxyType(
            {
                key: MappingProxyType(dict(value))
                for key, value in plan_facts.items()
            }
        ),
        producer_facts=MappingProxyType(
            {
                key: MappingProxyType(dict(value))
                for key, value in producer_facts_by_key.items()
            }
        ),
        producer_keys_by_record_id=MappingProxyType(
            producer_keys_by_record_id
        ),
        validation_counts=MappingProxyType(validation_counts),
    )


def _check_graph_and_addresses(  # noqa: C901, PLR0912 # Check every typed edge, cycle, and closure.
    root: dict, indexes: dict[str, dict[str, dict]]
) -> None:
    adjacency: dict[str, set[str]] = {
        identifier: set() for table in indexes.values() for identifier in table
    }
    owners: list[tuple[str | None, tuple[object, ...], str, str, object]] = []
    for path, table, kind in _root_ref_locations(root):
        owners.append((None, path, table, kind, _at(root, path)))
    for table, index in indexes.items():
        for identifier, entry in index.items():
            for path, target_table, target_kind in _record_ref_locations(
                table, entry["kind"], entry["record"]
            ):
                owners.append(
                    (
                        identifier,
                        path,
                        target_table,
                        target_kind,
                        _at(entry["record"], path),
                    )
                )
    pending: list[str] = []
    roots: set[str] = set()
    for owner, _path, table, expected_kind, value in owners:
        target_id, pointer_kind = _check_reference_shape(value)
        if target_id not in adjacency:
            pending.append("dangling_storage_reference")
            continue
        if owner is None:
            roots.add(target_id)
        else:
            adjacency[owner].add(target_id)
        actual = indexes[table].get(target_id)
        if (
            actual is None
            or actual["kind"] != pointer_kind
            or (expected_kind and pointer_kind != expected_kind)
        ):
            actual_kind = None if actual is None else actual["kind"]
            pending.append(
                f"wrong_kind_storage_reference:{table}:{expected_kind}:{pointer_kind}:{actual_kind}"
            )
    color: dict[str, int] = {}
    for identifier, initial_targets in adjacency.items():
        if color.get(identifier, _UNVISITED) == _DONE:
            continue
        color[identifier] = _ACTIVE
        stack = [(identifier, iter(initial_targets))]
        while stack:
            node, children = stack[-1]
            for target in children:
                state = color.get(target, _UNVISITED)
                if state == _ACTIVE:
                    _fail("cyclic_storage_reference")
                if state == _DONE:
                    continue
                color[target] = _ACTIVE
                stack.append((target, iter(adjacency[target])))
                break
            else:
                color[node] = _DONE
                stack.pop()
    if pending:
        _fail(pending[0])
    reachable: set[str] = set()
    pending_nodes = list(roots)
    while pending_nodes:
        node = pending_nodes.pop()
        if node in reachable:
            continue
        reachable.add(node)
        pending_nodes.extend(adjacency[node])
    if reachable != set(adjacency):
        _fail("unused_storage_record")


def _iter_logical_record_json(  # noqa: C901, PLR0912, PLR0913 # Stream canonical reader references.
    value: object,
    *,
    references: Mapping[tuple[object, ...], tuple[str, str]] | None = None,
    indexes: dict[str, dict[str, dict]] | None = None,
    memo: dict[str, dict] | None = None,
    logical_lookup: Callable[[str, str], Iterator[bytes]] | None = None,
    path: tuple[object, ...] = (),
) -> Iterator[bytes]:
    """Stream canonical JSON, resolving only listed schema reference paths.

    Preconditions:
        Each listed path contains a correctly shaped pointer in ``indexes``.
        Either ``memo`` or ``logical_lookup`` can resolve each target.
    Postconditions:
        Chunks form canonical JSON for the logical value, with array order kept.
    Essential behavior:
        Resolves only declared schema pointers and streams each referenced value
        without joining a full reader tree when ``logical_lookup`` is supplied.
    """
    target = None if references is None else references.get(path)
    if target is not None:
        if indexes is None or (memo is None and logical_lookup is None):
            _fail("invalid_storage_reference_position")
        table, expected_kind = target
        identifier, pointer_kind = _check_reference_shape(value)
        if expected_kind and pointer_kind != expected_kind:
            _fail("wrong_kind_storage_reference")
        entry = indexes[table].get(identifier)
        if entry is None or entry["kind"] != pointer_kind:
            _fail("wrong_kind_storage_reference")
        if logical_lookup is not None:
            yield from logical_lookup(table, identifier)
        else:
            if memo is None:
                _fail("missing_storage_record")
            logical = memo.get(f"{table}:{identifier}")
            if logical is None:
                _fail("missing_storage_record")
            yield from _iter_logical_record_json(logical)
        return
    if isinstance(value, Mapping):
        keys = tuple(value)
        if any(type(key) is not str for key in keys):
            _fail("storage_non_string_key")
        yield b"{"
        first = True
        for key in sorted(keys):
            if not first:
                yield bytes((44,))
            first = False
            yield from _iter_canonical_string_json(key)
            yield b":"
            yield from _iter_logical_record_json(
                value[key],
                references=references,
                indexes=indexes,
                memo=memo,
                logical_lookup=logical_lookup,
                path=(*path, key),
            )
        yield b"}"
        return
    if type(value) in {list, tuple, _OwnedList, _OwnedTuple}:
        yield b"["
        for position, item in enumerate(value):
            if position:
                yield bytes((44,))
            yield from _iter_logical_record_json(
                item,
                references=references,
                indexes=indexes,
                memo=memo,
                logical_lookup=logical_lookup,
                path=(*path, position),
            )
        yield b"]"
        return
    if type(value) is str:
        yield from _iter_canonical_string_json(value)
        return
    yield canonical_bytes(value)


def _iter_canonical_string_json(value: str) -> Iterator[bytes]:
    """Stream one canonical JSON string in bounded, UTF-8-safe pieces.

    Preconditions:
        ``value`` is a Python string accepted by the standard JSON encoder.
    Postconditions:
        Joined chunks equal ``canonical_bytes(value)`` without retaining one
        escaped copy of a long compiler source or prose string.
    Essential behavior:
        Uses the same escaping rules as the existing canonical serializer and
        splits only between Python string positions.
    """
    yield b'"'
    for offset in range(0, len(value), 1024):
        encoded = json.dumps(
            value[offset : offset + 1024],
            ensure_ascii=False,
            separators=(",", ":"),
            allow_nan=False,
        )
        yield encoded[1:-1].encode("utf-8")
    yield b'"'


def _normalized_statement_info(
    indexes: Mapping[str, Mapping[str, Mapping[str, object]]],
    result_id: str,
    reference: Mapping[str, object],
) -> tuple[str, object]:
    """Hash one selected normalized statement through the canonical resolver.

    Args:
        indexes: Normalized record tables indexed by storage address.
        result_id: Address of the owning result record.
        reference: Strict inherited-provenance selector.

    Returns:
        tuple[str, object]: Statement digest and its declared kind.

    Preconditions:
        ``result_id`` and ``reference`` select an owned result statement.
    Postconditions: The digest covers the selected canonical statement bytes.
    Essential behavior:
        Keeps only the selected statement mapping and streams its references.
    """
    statement = _statement_value_from_normalized_result(
        indexes, result_id, reference
    )
    chunks = _iter_logical_record_json(
        statement,
        references=_statement_reference_locations(statement),
        indexes=indexes,
        logical_lookup=lambda table, identifier: _record_logical_chunks(
            table, identifier, indexes
        ),
    )
    digest = hashlib.sha256()
    for chunk in chunks:
        digest.update(chunk)
    return digest.hexdigest(), statement.get("kind")


def _verify_storage_addresses(
    indexes: Mapping[str, Mapping[str, Mapping[str, object]]],
) -> None:
    """Recompute every record address from its canonical logical stream.

    Args:
        indexes: Normalized record tables indexed by storage address.

    Preconditions:
        Every referenced address resolves within ``indexes``.
    Postconditions: Every address matches its complete logical record.
    Essential behavior:
        Uses the shared streaming resolver without materializing the graph.

    Raises:
        ValueError: If a record has a missing reference or an address mismatch.
    """
    for table, index in indexes.items():
        for identifier in index:
            if (
                _record_streamed_address(table, identifier, indexes)
                != identifier
            ):
                _fail("storage_address_mismatch")


def parse_report_storage(raw_bytes: bytes | str) -> ReportStorageView:
    """Parse, integrity-check, and index a canonical stored report envelope."""
    envelope, raw = _strict_json_load(raw_bytes)
    artifact_kind, root, indexes = _validate_envelope_shape(envelope)
    _check_graph_and_addresses(root, indexes)
    memo, typed_records, validation_counts = _validate_table_records(
        root, indexes
    )
    _verify_storage_addresses(indexes)
    if raw != canonical_bytes(envelope):
        _fail("noncanonical_storage_json")
    frozen_tables = {
        table: MappingProxyType(
            {identifier: _freeze(entry) for identifier, entry in index.items()}
        )
        for table, index in indexes.items()
    }
    result_by_key: dict[tuple[str, str], ResultRef] = {}
    result_by_id: dict[str, ResultRef] = {}
    for identifier, entry in indexes["results"].items():
        typed = typed_records.get(f"results:{identifier}")
        if typed is None:
            _fail("missing_validated_storage_model:results")
        item_key = result_key(typed)
        reference = ResultRef("results", entry["kind"], identifier, item_key)
        result_by_key[(entry["kind"], item_key)] = reference
        result_by_id[identifier] = reference
    return ReportStorageView(
        _raw=raw,
        _artifact_kind=artifact_kind,
        _root=_freeze(root),
        _tables=MappingProxyType(frozen_tables),
        _result_by_key=MappingProxyType(result_by_key),
        _result_by_id=MappingProxyType(result_by_id),
        _typed_validation_counts=MappingProxyType(validation_counts),
    )


def _parse_report_storage_for_import(
    raw_bytes: bytes, *, max_input_bytes: int
) -> _ParsedStorageImport:
    """Strictly parse bytes for a new session without retaining parse copies.

    Preconditions:
        ``raw_bytes`` is exact bytes, and ``max_input_bytes`` is a nonnegative
        exact integer supplied by the caller.
    Postconditions:
        Returns immutable shared rows and compact checked facts. The result
        retains no raw bytes, full logical-record map, or parsed model map.
    Essential behavior:
        Reuses the existing closed envelope, pointer, address, row, and shared
        relationship validators while checking the explicit caller byte cap.
    """
    if type(raw_bytes) is not bytes:
        _fail("storage_input_must_be_bytes")
    envelope, raw = _strict_json_load(
        raw_bytes, max_input_bytes=max_input_bytes
    )
    artifact_kind, root, indexes = _validate_envelope_shape(envelope)
    _check_graph_and_addresses(root, indexes)
    facts = _validate_import_table_records(root, indexes)
    _verify_storage_addresses(indexes)
    if raw != canonical_bytes(envelope):
        _fail("noncanonical_storage_json")
    if artifact_kind == "report":
        root["base_report"] = _capture_import_base_report(root["base_report"])

    frozen_tables = MappingProxyType(
        {
            table: MappingProxyType(
                {
                    identifier: _freeze(entry)
                    for identifier, entry in index.items()
                }
            )
            for table, index in indexes.items()
        }
    )
    result_by_key: dict[tuple[str, str], ResultRef] = {}
    result_by_id: dict[str, ResultRef] = {}
    for (kind, item_key), identifier in facts.result_ids.items():
        reference = ResultRef("results", kind, identifier, item_key)
        result_by_key[(kind, item_key)] = reference
        result_by_id[identifier] = reference
    return _ParsedStorageImport(
        artifact_kind=artifact_kind,
        root=_freeze(root),
        tables=frozen_tables,
        result_by_key=MappingProxyType(result_by_key),
        result_by_id=MappingProxyType(result_by_id),
        facts=facts,
    )


def _logical_template(
    root: Mapping[str, object],
    tables: Mapping[str, Mapping[str, Mapping[str, object]]],
) -> tuple[dict, list[tuple[tuple[object, ...], str, str]]]:
    generation = _thaw(root["generation"])
    reader_id = root["reader"]["$ref"]
    reader = _thaw(tables["readers"][reader_id]["record"])
    shape = root["shape"]
    if shape == "report":
        template = dict(root["base_report"])
        template["generation"] = generation
        template["narrative_v2"] = {
            key: value for key, value in reader.items() if key != "shape"
        }
        prefix = "narrative_v2"
        ref_locations = _generation_ref_locations(generation)
        ref_locations.extend(
            _prefix_locations(
                _record_ref_locations("readers", "reader", reader), prefix
            )
        )
        return template, ref_locations
    metadata = _thaw(root["metadata"])
    template = dict(metadata)
    template["generation"] = generation
    reader.pop("shape")
    template.update(reader)
    ref_locations = _generation_ref_locations(generation)
    ref_locations.extend(
        _record_ref_locations(
            "readers", "reader", {"shape": "annotations", **reader}
        )
    )
    return template, ref_locations


def _prefix_locations(
    locations: list[tuple[tuple[object, ...], str, str]], prefix: str
) -> list[tuple[tuple[object, ...], str, str]]:
    return [((prefix, *path), table, kind) for path, table, kind in locations]


def _encoded_size(
    value: object,
    path: tuple[object, ...],
    refs: dict[tuple[object, ...], tuple[str, str]],
    size_record: Callable[[str, str], int],
) -> int:
    target = refs.get(path)
    if target is not None:
        table, identifier = target
        return size_record(table, identifier)
    if isinstance(value, Mapping):
        entries = list(value.items())
        total = 2 + max(0, len(entries) - 1)
        for key, child in entries:
            total += len(canonical_bytes(key)) + 1
            total += _encoded_size(child, (*path, key), refs, size_record)
        return total
    if type(value) in {list, tuple}:
        return (
            2
            + max(0, len(value) - 1)
            + sum(
                _encoded_size(child, (*path, index), refs, size_record)
                for index, child in enumerate(value)
            )
        )
    return len(canonical_bytes(value))


def _measurement_record(
    table: str,
    identifier: str,
    view: ReportStorageView,
    logical_memo: dict[str, dict],
) -> tuple[dict, list[tuple[tuple[object, ...], str, str]]]:
    entry = view._tables[table][identifier]  # noqa: SLF001 # Read the immutable index during size preflight.
    kind = entry["kind"]
    record = _thaw(entry["record"])
    locations = _record_ref_locations(table, kind, record)
    if table == "resolved_citations":
        target_ref = record.pop("source_target")
        target_id, _ = _check_reference_shape(target_ref)
        target = _thaw(view._tables["source_targets"][target_id]["record"])  # noqa: SLF001
        record.update(target)
        url_ref = record["url"]
        url_id, _ = _check_reference_shape(url_ref)
        record["url"] = view._tables["urls"][url_id]["record"]["value"]  # noqa: SLF001
        locations = [
            (path, target_table, target_kind)
            for path, target_table, target_kind in locations
            if path == ("reference",)
        ]
    elif table == "inherited_provenance":
        result_ref = record.pop("result")
        result_id, _ = _check_reference_shape(result_ref)
        reference_ref = record["reference"]
        reference_id, _ = _check_reference_shape(reference_ref)
        source_result = view._materialize("results", result_id, logical_memo)  # noqa: SLF001
        reference = view._materialize(  # noqa: SLF001
            "evidence_refs", reference_id, logical_memo
        )
        record["statement"] = _statement_from_path(source_result, reference)
        locations = [
            (path, target_table, target_kind)
            for path, target_table, target_kind in locations
            if path in {("reference",), ("receipt",)}
        ]
    elif table == "readers":
        record.pop("shape")
    return record, locations


def _record_size(  # noqa: PLR0913 # Carries the memoized sizing context without global state.
    table: str,
    identifier: str,
    view: ReportStorageView,
    memo: dict[str, int],
    active: set[str],
    logical_memo: dict[str, dict],
) -> int:
    key = f"{table}:{identifier}"
    if key in memo:
        return memo[key]
    if key in active:
        _fail("cyclic_storage_reference")
    active.add(key)
    record, locations = _measurement_record(
        table, identifier, view, logical_memo
    )
    refs = {}
    for path, target_table, _kind in locations:
        pointer = _at(record, path)
        target_id, _pointer_kind = _check_reference_shape(pointer)
        refs[path] = (target_table, target_id)
    size = _encoded_size(
        record,
        (),
        refs,
        lambda t, i: _record_size(t, i, view, memo, active, logical_memo),
    )
    active.remove(key)
    memo[key] = size
    return size


def _materialize_template(
    value: dict,
    locations: list[tuple[tuple[object, ...], str, str]],
    view: ReportStorageView,
    memo: dict[str, dict],
) -> dict:
    return _replace_paths(
        value,
        locations,
        lambda table, identifier: view._materialize(  # noqa: SLF001 # Resolve from the view's owned indexes.
            table, identifier, memo
        ),
    )


def expand_report_storage(
    view: ReportStorageView, *, max_output_bytes: int
) -> dict:
    """Expand a parsed view after measuring its projected canonical byte size.

    Repeated references are counted at every occurrence while unique records
    remain indexed. The caller's cap is checked before copied result trees are
    allocated. Cycles are rejected by parsing and checked again during sizing.
    """
    if (
        not isinstance(view, ReportStorageView)
        or type(max_output_bytes) is not int
        or max_output_bytes < 0
    ):
        _fail("invalid_expansion_limit")
    root, tables = view._root, view._tables  # noqa: SLF001 # Read immutable indexes for explicit expansion.
    template, locations = _logical_template(root, tables)
    refs = {
        path: (table, _check_reference_shape(_at(template, path))[0])
        for path, table, _kind in locations
    }
    size_memo: dict[str, int] = {}
    active: set[str] = set()
    logical_memo: dict[str, dict] = {}
    projected = _encoded_size(
        template,
        (),
        refs,
        lambda table, identifier: _record_size(
            table, identifier, view, size_memo, active, logical_memo
        ),
    )
    if projected > max_output_bytes:
        _fail("expanded_report_too_large")
    expanded = _materialize_template(template, locations, view, {})
    # This assertion also detects mistakes in the virtual projection formula.
    if len(canonical_bytes(expanded)) != projected:
        _fail("expansion_size_mismatch")
    return expanded


_JSON_OUTPUT_CHUNK_BYTES = 16 * 1024
_OVERLAY_GROUP_FIELDS = ("title", "number", "next_id", "narrative")


def _iter_bounded_json_chunks(
    chunks: Iterator[bytes], *, chunk_size: int = _JSON_OUTPUT_CHUNK_BYTES
) -> Iterator[bytes]:
    """Coalesce small JSON pieces into fixed-size output chunks.

    Preconditions:
        ``chunks`` yields UTF-8 JSON pieces and ``chunk_size`` is a positive
        exact integer.
    Postconditions:
        Every yielded byte string is nonempty and no larger than
        ``chunk_size``; concatenation preserves every input byte.
    Essential behavior:
        Keeps only one bounded staging buffer while avoiding one-byte writes
        for JSON punctuation and short scalar values.
    """
    if type(chunk_size) is not int or chunk_size <= 0:
        _fail("invalid_json_chunk_size")
    buffer = bytearray()
    for chunk in chunks:
        if type(chunk) is not bytes:
            _fail("invalid_json_chunk")
        offset = 0
        while offset < len(chunk):
            available = chunk_size - len(buffer)
            take = min(available, len(chunk) - offset)
            buffer.extend(chunk[offset : offset + take])
            offset += take
            if len(buffer) == chunk_size:
                yield bytes(buffer)
                buffer.clear()
    if buffer:
        yield bytes(buffer)


def _output_group_overlays(
    overlay: Mapping[str, object],
) -> dict[str, Mapping[str, object]]:
    """Index validated output group overlays without copying their contents.

    Preconditions:
        ``overlay`` is the immutable, renderer-checked application overlay.
    Postconditions:
        Returns one unique group-ID index whose values are the owned overlay
        rows; missing or duplicate rows are rejected.
    Essential behavior:
        Allocates only references for the bounded group projection.
    """
    values = overlay.get("groups")
    if type(values) not in {_OwnedList, _OwnedTuple, list, tuple}:
        _fail("invalid_indexed_applied_fields")
    expected = {"group_id", *_OVERLAY_GROUP_FIELDS}
    by_id: dict[str, Mapping[str, object]] = {}
    for item in values:
        if not isinstance(item, Mapping) or set(item) != expected:
            _fail("invalid_indexed_applied_fields")
        group_id = item.get("group_id")
        if type(group_id) is not str or group_id in by_id:
            _fail("invalid_indexed_applied_fields")
        by_id[group_id] = item
    return by_id


def _iter_effective_group_json(
    group: object, group_overlay: Mapping[str, object]
) -> Iterator[bytes]:
    """Stream one shallow group projection with proven applied fields.

    Preconditions:
        ``group`` and its overlay row are checked mappings for the same group.
    Postconditions:
        Emits their canonical merged JSON fields without retaining the group.
    Essential behavior:
        Creates one shallow group mapping at a time.
    """
    if not isinstance(group, Mapping):
        _fail("invalid_indexed_applied_fields")
    effective = dict(group.items())
    for field in _OVERLAY_GROUP_FIELDS:
        effective[field] = group_overlay[field]
    yield from _iter_logical_record_json(effective)


def _iter_effective_groups_json(
    base_groups: object, overlay: Mapping[str, object]
) -> Iterator[bytes]:
    """Stream groups in validated presentation order with generated fields.

    Preconditions:
        The report group sequence and exact overlay membership passed sealing.
    Postconditions:
        Emits groups in overlay order and rejects duplicate, missing, or
        unused compiler and overlay selectors.
    Essential behavior:
        Merges and releases one group projection at a time.
    """
    if type(base_groups) not in {_OwnedList, _OwnedTuple, list, tuple}:
        _fail("invalid_indexed_applied_fields")
    by_id = _output_group_overlays(overlay)
    overlay_groups = overlay["groups"]
    base_by_id: dict[str, Mapping[str, object]] = {}
    for group in base_groups:
        if not isinstance(group, Mapping):
            _fail("invalid_indexed_applied_fields")
        group_id = group.get("id")
        if type(group_id) is not str or group_id in base_by_id:
            _fail("invalid_indexed_applied_fields")
        base_by_id[group_id] = group
    if set(base_by_id) != set(by_id):
        _fail("invalid_indexed_applied_fields")

    yield b"["
    for position, group_overlay in enumerate(overlay_groups):
        group_id = group_overlay["group_id"]
        group = base_by_id.get(group_id)
        if group is None:
            _fail("invalid_indexed_applied_fields")
        if position:
            yield b","
        yield from _iter_effective_group_json(group, group_overlay)
    yield b"]"


def _iter_effective_base_report_json(
    base_report: object, overlay: Mapping[str, object]
) -> Iterator[bytes]:
    """Stream a report base with its applied document and group overlays.

    Preconditions:
        ``base_report`` and ``overlay`` are the exact checked session values.
    Postconditions:
        Emits the effective applied base in canonical key order.
    Essential behavior:
        Retains no complete compiler-base copy or effective group list.
    """
    if not isinstance(base_report, Mapping) or not isinstance(
        overlay, Mapping
    ):
        _fail("invalid_indexed_applied_fields")
    if {"generation", "narrative_v2"} & set(base_report):
        _fail("invalid_report_base")
    if set(overlay) != {"document", "groups"} or "groups" not in base_report:
        _fail("invalid_indexed_applied_fields")
    fields = set(base_report)
    fields.add("document")
    yield b"{"
    for position, field in enumerate(sorted(fields)):
        if position:
            yield b","
        yield from _iter_canonical_string_json(field)
        yield b":"
        if field == "document":
            yield from _iter_logical_record_json(overlay["document"])
        elif field == "groups":
            yield from _iter_effective_groups_json(
                base_report["groups"], overlay
            )
        else:
            yield from _iter_logical_record_json(base_report[field])
    yield b"}"


def _iter_shared_storage_json(  # noqa: C901 # Keep canonical root and 13-table emission in one closed-format writer.
    view: ReportStorageView, *, overlay: Mapping[str, object] | None = None
) -> Iterator[bytes]:
    """Stream the canonical closed report-storage envelope from owned indexes.

    Preconditions:
        ``view`` is a closed graph or parsed view. A report overlay, when
        supplied, is the session-checked effective document/group projection.
    Postconditions:
        Chunks equal canonical storage JSON without a mutable envelope or a
        complete serialized byte buffer.
    Essential behavior:
        Writes all 13 closed tables in canonical table and storage-ID order.
    """
    if not isinstance(view, ReportStorageView):
        _fail("invalid_storage_view")
    root = view._root  # noqa: SLF001 # Stream only the immutable root owned by this graph.
    tables = view._tables  # noqa: SLF001 # Stream only the immutable indexes owned by this graph.
    if (
        view._artifact_kind not in {"report", "annotations"}  # noqa: SLF001
        or set(tables) != set(_TABLES)
        or (overlay is not None and view._artifact_kind != "report")  # noqa: SLF001
    ):
        _fail("invalid_storage_view")
    expected_root = (
        {"shape", "base_report", "generation", "reader"}
        if view._artifact_kind == "report"  # noqa: SLF001
        else {"shape", "metadata", "generation", "reader"}
    )
    if set(root) != expected_root:
        _fail("invalid_storage_root")

    yield b'{"artifact_kind":'
    yield from _iter_canonical_string_json(view._artifact_kind)  # noqa: SLF001
    yield b',"root":{'
    for position, field in enumerate(sorted(root)):
        if position:
            yield b","
        yield from _iter_canonical_string_json(field)
        yield b":"
        if field == "base_report" and overlay is not None:
            yield from _iter_effective_base_report_json(root[field], overlay)
        else:
            yield from _iter_logical_record_json(root[field])
    yield b'},"storage_version":'
    yield from _iter_canonical_string_json(STORAGE_VERSION)
    yield b',"tables":{'
    for table_position, table in enumerate(sorted(_TABLES)):
        if table_position:
            yield b","
        yield from _iter_canonical_string_json(table)
        yield b":["
        index = tables[table]
        for row_position, identifier in enumerate(sorted(index)):
            if row_position:
                yield b","
            yield from _iter_logical_record_json(index[identifier])
        yield b"]"
    yield b"}}"


def _materialized_model(  # noqa: PLR0913 # Keep the selected table, kind, and codec contract explicit.
    view: ReportStorageView,
    pointer: object,
    *,
    table: str,
    kind: str,
    model_type: type[BaseModel],
    compact_invocation: bool = False,
) -> Iterator[bytes]:
    """Normalize and stream one selected typed record at a time.

    Preconditions:
        ``pointer`` selects a record of ``kind`` in ``table`` and the graph
        has already passed its independent structural and proof boundary.
    Postconditions:
        Emits the strict model's normal JSON representation; producer
        invocations use the existing generation-storage.v1 codec when asked.
    Essential behavior:
        Retains at most one selected logical record/model during compatibility
        default normalization.
    """
    identifier, pointer_kind = _check_reference_shape(pointer)
    entry = view._tables.get(table, {}).get(identifier)  # noqa: SLF001 # Resolve one selected address in the checked graph.
    if entry is None or pointer_kind != kind or entry.get("kind") != kind:
        _fail("wrong_kind_storage_reference")
    logical = view._materialize(table, identifier, {})  # noqa: SLF001 # Expand only this selected record for typed default normalization.
    model = model_type.model_validate(_tuple_model_input(logical))
    value = model.model_dump(mode="json")
    if compact_invocation:
        invocation = getattr(model, "invocation", None)
        if invocation is None or "invocation" not in value:
            _fail("invalid_generation_producer")
        value["invocation"] = encode_invocation(invocation).model_dump(
            mode="json"
        )
    yield from _iter_logical_record_json(value)


def _tuple_model_input(value: object) -> object:
    """Adapt JSON arrays to Python tuples for strict selected-model parsing.

    Preconditions:
        ``value`` is one already checked JSON header or selected table record.
    Postconditions:
        Mapping and scalar values are preserved while every array uses the
        Python shape accepted by strict ``model_validate`` tuple fields.
    Essential behavior:
        Avoids JSON re-encoding and never coerces scalar types.
    """
    if isinstance(value, Mapping):
        return {key: _tuple_model_input(item) for key, item in value.items()}
    if type(value) in {list, tuple, _OwnedList, _OwnedTuple}:
        return tuple(_tuple_model_input(item) for item in value)
    return value


def _iter_generation_record_array(  # noqa: PLR0913 # Keep the fixed reference adapter inputs explicit.
    view: ReportStorageView,
    generation: Mapping[str, object],
    *,
    field: str,
    table: str,
    kind: str,
    model_type: type[BaseModel],
    compact_invocation: bool = False,
) -> Iterator[bytes]:
    """Stream one ordered generation array through selected typed records.

    Preconditions:
        ``generation[field]`` is an ordered array of the exact record kind.
    Postconditions:
        Emits each selected normalized row in source order.
    Essential behavior:
        Retains only the current selected record and model.
    """
    values = generation.get(field)
    if type(values) not in {_OwnedList, _OwnedTuple, list, tuple}:
        _fail("invalid_generation_skeleton")
    yield b"["
    for position, pointer in enumerate(values):
        if position:
            yield b","
        yield from _materialized_model(
            view,
            pointer,
            table=table,
            kind=kind,
            model_type=model_type,
            compact_invocation=compact_invocation,
        )
    yield b"]"


def _iter_stored_generation_json(  # noqa: C901, PLR0912 # Dispatch the closed manifest fields to selected stream adapters.
    view: ReportStorageView,
) -> Iterator[bytes]:
    """Stream generation-storage.v1 with legacy model-default semantics.

    Preconditions:
        ``view`` owns a generation skeleton whose complete proof was checked
        by the current ReportSession.
    Postconditions:
        Emits the canonical JSON value produced by
        ``stored_generation(persisted_manifest(expanded_generation))``.
    Essential behavior:
        Normalizes one header or selected typed row at a time and never builds
        a whole ``GenerationManifest`` or manifest dump.
    """
    generation = view._root["generation"]  # noqa: SLF001 # Read the captured immutable generation selectors.
    if not isinstance(generation, Mapping):
        _fail("invalid_generation_skeleton")
    fields = set(GenerationManifest.model_fields)
    if set(generation) - fields:
        _fail("invalid_generation_skeleton")
    header_models: dict[str, type[BaseModel]] = {
        "current_usage": CurrentUsage,
        "ledger_parameters": LedgerParameters,
        "request_configuration": RequestConfiguration,
    }
    fixed_records: dict[str, tuple[str, str, type[BaseModel]]] = {
        "run": ("runs", "run", RunIdentity),
        "plan": ("plans", "story_plan", StoryPlan),
        "document": ("results", "document", DocumentResult),
    }
    array_records: dict[str, tuple[str, str, type[BaseModel], bool]] = {
        "details": ("results", "detail", DetailResult, False),
        "planning": ("results", "planning_card", PlanningCardResult, False),
        "sections": ("results", "section", SectionResult, False),
        "producers": ("producers", "producer", ProducerDescriptor, True),
        "receipts": ("receipts", "receipt", ResultReceipt, False),
    }
    output_fields = sorted((*fields, "storage_version"))
    yield b"{"
    for position, field in enumerate(output_fields):
        if position:
            yield b","
        yield from _iter_canonical_string_json(field)
        yield b":"
        if field == "storage_version":
            yield from _iter_canonical_string_json(GENERATION_STORAGE_VERSION)
        elif field in fixed_records:
            table, kind, model_type = fixed_records[field]
            yield from _materialized_model(
                view,
                generation.get(field),
                table=table,
                kind=kind,
                model_type=model_type,
            )
        elif field in array_records:
            table, kind, model_type, compact_invocation = array_records[field]
            yield from _iter_generation_record_array(
                view,
                generation,
                field=field,
                table=table,
                kind=kind,
                model_type=model_type,
                compact_invocation=compact_invocation,
            )
        elif field in header_models:
            value = generation.get(field)
            if value is None:
                _fail("invalid_generation_skeleton")
            model = header_models[field].model_validate(
                _tuple_model_input(_thaw(value))
            )
            yield from _iter_logical_record_json(model.model_dump(mode="json"))
        elif field == "version":
            yield from _iter_canonical_string_json(
                generation.get(field, "diffstory.generation.v2")
            )
        elif field == "provenance":
            yield from _iter_canonical_string_json(
                generation.get(field, "unverified")
            )
        elif field in {
            "compiler_fact_signature",
            "ledger_digest",
            "unit_keys",
        }:
            if field not in generation:
                _fail("invalid_generation_skeleton")
            yield from _iter_logical_record_json(generation[field])
        else:
            _fail("unknown_generation_skeleton_field")
    yield b"}"


def _reader_entry(
    view: ReportStorageView, root: Mapping[str, object]
) -> Mapping[str, object]:
    """Resolve the exact root-selected reader row without expanding it.

    Preconditions:
        ``root`` is a checked report or annotation root from ``view``.
    Postconditions:
        Returns only the typed reader record selected by the root.
    Essential behavior:
        Rejects missing and wrong-kind pointers without expanding the reader.
    """
    identifier, kind = _check_reference_shape(root["reader"])
    entry = view._tables["readers"].get(identifier)  # noqa: SLF001 # Follow only the closed reader root pointer.
    if kind != "reader" or entry is None or entry.get("kind") != "reader":
        _fail("wrong_kind_storage_reference")
    return entry["record"]


def _iter_reader_field_json(
    reader: Mapping[str, object],
    field: str,
    indexes: Mapping[str, object],
    references: Mapping[tuple[object, ...], tuple[str, str]],
) -> Iterator[bytes]:
    """Stream one logical reader field while resolving only schema pointers.

    Preconditions:
        ``reader`` is a checked annotation/presentation reader and ``field``
        is one of its declared member paths.
    Postconditions:
        Emits the logical field with only enumerated references resolved.
    Essential behavior:
        Leaves compiler maps and literal ``$ref``-shaped values untouched.
    """
    yield from _iter_logical_record_json(
        reader[field],
        references=references,
        indexes=indexes,  # type: ignore[arg-type]
        logical_lookup=lambda table, identifier: _record_logical_chunks(
            table,
            identifier,
            indexes,  # type: ignore[arg-type]
        ),
        path=(field,),
    )


def _iter_reader_model_array_json(
    view: ReportStorageView,
    values: object,
    *,
    table: str,
    kind: str,
    model_type: type[BaseModel],
) -> Iterator[bytes]:
    """Stream selected reader records through their strict compatibility model.

    Preconditions:
        ``values`` is an ordered array of typed references in ``view``.
    Postconditions:
        Emits each selected row with the same model defaults as public typed
        reader assembly, in the original reference order.
    Essential behavior:
        Materializes and validates only the current selected row.
    """
    if type(values) not in {_OwnedList, _OwnedTuple, list, tuple}:
        _fail("invalid_reader_array")
    yield b"["
    for position, pointer in enumerate(values):
        if position:
            yield b","
        yield from _materialized_model(
            view,
            pointer,
            table=table,
            kind=kind,
            model_type=model_type,
        )
    yield b"]"


def _iter_expanded_reader_section_json(
    view: ReportStorageView, section: Mapping[str, object]
) -> Iterator[bytes]:
    """Stream one reader section while normalizing its selected result rows.

    Preconditions:
        ``section`` is the checked ordered reader header with prose and detail
        references owned by ``view``.
    Postconditions:
        Emits that section with its original header fields and typed result
        defaults, without retaining a complete reader tree.
    Essential behavior:
        Uses the real result references and keeps only one selected row live.
    """
    expected = {
        "group_id",
        "group_key",
        "number",
        "next_id",
        "prose",
        "details",
    }
    if set(section) != expected:
        _fail("invalid_reader_section")
    yield b"{"
    for position, field in enumerate(sorted(expected)):
        if position:
            yield b","
        yield from _iter_canonical_string_json(field)
        yield b":"
        if field == "prose":
            yield from _materialized_model(
                view,
                section[field],
                table="results",
                kind="section",
                model_type=SectionResult,
            )
        elif field == "details":
            yield from _iter_reader_model_array_json(
                view,
                section[field],
                table="results",
                kind="detail",
                model_type=DetailResult,
            )
        else:
            yield from _iter_logical_record_json(section[field])
    yield b"}"


def _iter_expanded_reader_sections_json(
    view: ReportStorageView, values: object
) -> Iterator[bytes]:
    """Stream the reader's ordered section headers and selected result rows.

    Preconditions:
        ``values`` is the checked ``sections`` or ``steps`` array.
    Postconditions:
        Emits the same ordered section values with typed reader defaults.
    Essential behavior:
        Retains at most the current section and one typed result row.
    """
    if type(values) not in {_OwnedList, _OwnedTuple, list, tuple}:
        _fail("invalid_reader_array")
    yield b"["
    for position, section in enumerate(values):
        if position:
            yield b","
        if not isinstance(section, Mapping):
            _fail("invalid_reader_section")
        yield from _iter_expanded_reader_section_json(view, section)
    yield b"]"


def _iter_expanded_reader_field_json(
    view: ReportStorageView, reader: Mapping[str, object], field: str
) -> Iterator[bytes]:
    """Stream one selected reader field through its strict compatibility model.

    Preconditions:
        ``reader`` is a checked reader selected by ``view`` and ``field`` is
        one of its public expanded members.
    Postconditions:
        Emits the selected logical value with all typed defaults applied.
    Essential behavior:
        Normalizes one selected row at a time and preserves all references.
    """
    shape = reader.get("shape")
    if shape == "presentation":
        section_field = "sections"
        accepted_fields = {
            "version",
            "provenance",
            "document",
            "sections",
            "context_details",
            "citations",
            "support_provenance",
        }
    elif shape == "annotations":
        section_field = "steps"
        accepted_fields = {
            "document",
            "steps",
            "context_details",
            "citations",
            "support_provenance",
        }
    else:
        _fail("invalid_reader_shape")
    if field not in accepted_fields:
        _fail("invalid_reader_fields")
    if field == "document":
        yield from _materialized_model(
            view,
            reader[field],
            table="results",
            kind="document",
            model_type=DocumentResult,
        )
    elif field == section_field:
        yield from _iter_expanded_reader_sections_json(view, reader[field])
    elif field == "context_details":
        yield from _iter_reader_model_array_json(
            view,
            reader[field],
            table="results",
            kind="detail",
            model_type=DetailResult,
        )
    elif field == "citations":
        yield from _iter_reader_model_array_json(
            view,
            reader[field],
            table="resolved_citations",
            kind="resolved_citation",
            model_type=ResolvedCitation,
        )
    elif field == "support_provenance":
        yield from _iter_reader_model_array_json(
            view,
            reader[field],
            table="inherited_provenance",
            kind="inherited_provenance",
            model_type=InheritedProvenance,
        )
    else:
        yield from _iter_logical_record_json(reader[field])


def _iter_expanded_reader_json(
    view: ReportStorageView, reader: Mapping[str, object]
) -> Iterator[bytes]:
    """Stream an expanded presentation or annotation reader with defaults.

    Preconditions:
        ``reader`` is the checked reader selected by the report or annotation
        root of ``view``.
    Postconditions:
        Emits public typed assembly values while preserving order, content,
        source links, provenance, and all selected references.
    Essential behavior:
        Applies strict model defaults per selected row, without building a
        complete ``V2Presentation`` or annotation model.
    """
    shape = reader.get("shape")
    if shape == "presentation":
        expected = {
            "shape",
            "version",
            "provenance",
            "document",
            "sections",
            "context_details",
            "citations",
            "support_provenance",
        }
    elif shape == "annotations":
        expected = {
            "shape",
            "document",
            "steps",
            "context_details",
            "citations",
            "support_provenance",
        }
    else:
        _fail("invalid_reader_shape")
    if set(reader) != expected:
        _fail("invalid_reader_fields")
    fields = sorted(expected - {"shape"})
    yield b"{"
    for position, field in enumerate(fields):
        if position:
            yield b","
        yield from _iter_canonical_string_json(field)
        yield b":"
        yield from _iter_expanded_reader_field_json(view, reader, field)
    yield b"}"


def _iter_expanded_report_json(
    view: ReportStorageView, overlay: Mapping[str, object]
) -> Iterator[bytes]:
    """Stream an applied expanded report through its indexed reader and rows.

    Preconditions:
        ``view`` and ``overlay`` belong to a sealed applied report handle.
    Postconditions:
        Emits the canonical logical report with generation-storage.v1 defaults.
    Essential behavior:
        Resolves indexed result rows directly and keeps no presentation tree.
    """
    root = view._root  # noqa: SLF001 # Read only the session-checked report graph.
    if (
        view._artifact_kind != "report"  # noqa: SLF001
        or root.get("shape") != "report"
        or set(root) != {"shape", "base_report", "generation", "reader"}
        or not isinstance(root.get("base_report"), Mapping)
    ):
        _fail("invalid_storage_root")
    base = root["base_report"]
    if {"generation", "narrative_v2"} & set(base):
        _fail("invalid_report_base")
    reader = _reader_entry(view, root)
    if reader.get("shape") != "presentation":
        _fail("invalid_reader_shape")
    fields = set(base) | {"document", "groups", "generation", "narrative_v2"}
    yield b"{"
    for position, field in enumerate(sorted(fields)):
        if position:
            yield b","
        yield from _iter_canonical_string_json(field)
        yield b":"
        if field == "document":
            yield from _iter_logical_record_json(overlay["document"])
        elif field == "groups":
            yield from _iter_effective_groups_json(base.get("groups"), overlay)
        elif field == "generation":
            yield from _iter_stored_generation_json(view)
        elif field == "narrative_v2":
            yield from _iter_expanded_reader_json(view, reader)
        else:
            yield from _iter_logical_record_json(base[field])
    yield b"}"


def _iter_expanded_annotations_json(  # noqa: C901 # Select one accepted reader root and stream its fields.
    view: ReportStorageView,
    *,
    reader_record: Mapping[str, object] | None = None,
    metadata: Mapping[str, object] | None = None,
) -> Iterator[bytes]:
    """Stream an annotation candidate over its exact reader and proof rows.

    Preconditions:
        ``view`` owns a checked annotations root, or is a checked report graph
        paired with the annotation reader projection derived by its session.
    Postconditions:
        Emits the existing annotation logical fields and compatibility proof.
    Essential behavior:
        Resolves ordered reader pointers and selected audit rows as consumed;
        it does not need a storage-addressed reader when output is expanded.
    """
    root = view._root  # noqa: SLF001 # Read only the session-checked indexed graph.
    if view._artifact_kind == "annotations":  # noqa: SLF001 # Use the sealed annotation root when present.
        if (
            root.get("shape") != "annotations"
            or set(root) != {"shape", "metadata", "generation", "reader"}
            or not isinstance(root.get("metadata"), Mapping)
            or reader_record is not None
            or metadata is not None
        ):
            _fail("invalid_storage_root")
        reader = _reader_entry(view, root)
        metadata = root["metadata"]
    elif view._artifact_kind == "report":  # noqa: SLF001 # Consume a projection proven by the same report session.
        if (
            root.get("shape") != "report"
            or set(root) != {"shape", "base_report", "generation", "reader"}
            or reader_record is None
            or metadata is None
            or reader_record.get("shape") != "annotations"
        ):
            _fail("invalid_storage_root")
        reader = reader_record
    else:
        _fail("invalid_storage_root")
    if reader.get("shape") != "annotations":
        _fail("invalid_reader_shape")
    if not isinstance(metadata, Mapping):
        _fail("invalid_storage_root")
    reader_fields = set(reader) - {"shape"}
    fields = set(metadata) | reader_fields | {"generation"}
    yield b"{"
    for position, field in enumerate(sorted(fields)):
        if position:
            yield b","
        yield from _iter_canonical_string_json(field)
        yield b":"
        if field == "generation":
            yield from _iter_stored_generation_json(view)
        elif field in metadata:
            yield from _iter_logical_record_json(metadata[field])
        else:
            yield from _iter_expanded_reader_field_json(view, reader, field)
    yield b"}"


def _iter_report_output_json(
    view: ReportStorageView,
    overlay: Mapping[str, object],
    *,
    report_storage: str,
) -> Iterator[bytes]:
    """Select the expanded compatibility or normalized shared report stream.

    Preconditions:
        ``view`` and ``overlay`` passed the current session's report check.
    Postconditions:
        Emits exactly one selected report representation.
    Essential behavior:
        Does not change the compatibility default or construct an envelope.
    """
    if report_storage == "shared":
        yield from _iter_shared_storage_json(view, overlay=overlay)
    elif report_storage == "expanded":
        yield from _iter_expanded_report_json(view, overlay)
    else:
        _fail("unknown_report_storage")


def _iter_annotation_output_json(
    view: ReportStorageView,
    *,
    report_storage: str,
    reader_record: Mapping[str, object] | None = None,
    metadata: Mapping[str, object] | None = None,
) -> Iterator[bytes]:
    """Select the expanded compatibility or normalized shared annotation stream.

    Preconditions:
        ``view`` and any reader projection belong to one registered generation
        handle; shared output uses an annotation-root graph.
    Postconditions:
        Emits exactly one selected annotation representation.
    Essential behavior:
        Leaves all proof checks to the session and preserves expanded default.
    """
    if report_storage == "shared":
        if reader_record is not None or metadata is not None:
            _fail("invalid_indexed_annotation_derivation")
        yield from _iter_shared_storage_json(view)
    elif report_storage == "expanded":
        yield from _iter_expanded_annotations_json(
            view, reader_record=reader_record, metadata=metadata
        )
    else:
        _fail("unknown_report_storage")
