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
from dataclasses import dataclass
from types import MappingProxyType
from typing import NoReturn

from pydantic import BaseModel

from .narrative_contracts import DetailResult
from .narrative_contracts import DocumentResult
from .narrative_contracts import PlanningCardResult
from .narrative_contracts import SectionResult
from .narrative_contracts import SourceSliceRef
from .narrative_contracts import SpanRef
from .narrative_contracts import StatementRef
from .narrative_contracts import StoryPlan
from .narrative_contracts import SupportedStatement
from .narrative_contracts import result_key
from .narrative_contracts import result_key_field
from .narrative_contracts import result_kind
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
from .narrative_plan import canonical_bytes
from .narrative_plan import content_digest
from .narrative_plan import semantic_digest
from .narrative_protocol import LedgerParameters
from .narrative_protocol import RequestConfiguration
from .narrative_runtime import ArtifactInvocation
from .narrative_runtime import PreparedMetadata
from .narrative_runtime import RunIdentity

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


def _thaw(value: object) -> object:
    if isinstance(value, Mapping):
        return {key: _thaw(item) for key, item in value.items()}
    if type(value) in {list, tuple}:
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


@dataclass(frozen=True, slots=True)
class ReaderSectionRef:
    """Ordered reader membership with result addresses, without result copies."""

    group_id: str
    group_key: str
    number: int
    next_id: str | None
    prose: ResultRef
    details: tuple[ResultRef, ...]


@dataclass(frozen=True, slots=True)
class IndexedReaderProjection:
    """Read-only ordered reader references backed by a report record graph."""

    shape: str
    document: ResultRef
    sections: tuple[ReaderSectionRef, ...]
    context_details: tuple[ResultRef, ...]
    citations: tuple[RecordRef, ...]
    support_provenance: tuple[RecordRef, ...]


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

    def __init__(self, *, defer_validation: bool = False) -> None:
        self.defer_validation = defer_validation
        self.records: dict[str, dict[str, dict[str, object]]] = {
            name: {} for name in _TABLES
        }
        self.address_contents: dict[str, dict[str, bytes]] = {
            name: {} for name in _TABLES
        }
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
        if table not in self.records or kind not in _TABLE_RECORD_KINDS[table]:
            _fail("invalid_storage_record_kind")
        _strict_json_value(record)
        if stream_address:
            if table != "readers":
                _fail("invalid_streamed_storage_record")
            identifier = _logical_storage_address(
                table,
                "",
                {"kind": kind, "record": record},
                self.records,
                self.logical_records,
            )
            encoded_complete = None
        else:
            complete = record if address_record is None else address_record
            _strict_json_value(complete)
            identifier = storage_address(kind, complete)
            encoded_complete = canonical_bytes(complete)
        encoded_stored = canonical_bytes(record)
        prior = self.records[table].get(identifier)
        if prior is not None:
            same_logical_content = (
                _logical_reader_streams_equal(
                    prior["record"],
                    record,
                    self.records,
                    self.logical_records,
                )
                if stream_address
                else self.address_contents[table][identifier]
                == encoded_complete
            )
            if (
                prior["kind"] != kind
                or not same_logical_content
                or canonical_bytes(prior["record"]) != encoded_stored
            ):
                _fail("storage_address_collision")
            return identifier
        if self.record_count >= _MAX_RECORDS:
            _fail("too_many_storage_records")
        self.records[table][identifier] = {
            "id": identifier,
            "kind": kind,
            "record": record,
        }
        if not stream_address:
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
            if prior != logical_id:
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


def _strict_json_load(raw_bytes: bytes | str) -> tuple[dict, bytes]:
    if type(raw_bytes) is bytes:
        raw = raw_bytes
    elif type(raw_bytes) is str:
        raw = raw_bytes.encode("utf-8")
    else:
        _fail("storage_input_must_be_bytes_or_text")
    if len(raw) > _MAX_INPUT_BYTES:
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
        _verify_storage_addresses(builder.records, memo)

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
    entry = view_tables[table][identifier]
    record = entry["record"]
    kind = entry["kind"]

    def resolve(target_table: str, target_id: str) -> dict:
        return _logical_record(
            view_tables, target_table, target_id, memo, typed_records
        )

    locations = _record_ref_locations(table, kind, record)
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
    if table == "readers":
        _validate_reader_record(record)
        logical = copy.deepcopy(record)
    else:
        logical = _replace_paths(record, locations, resolve)
    if table == "resolved_citations":
        target = logical.pop("source_target")
        logical.update(target)
        logical["url"] = view_tables["urls"][record["url"]["$ref"]]["record"][
            "value"
        ]
    elif table == "inherited_provenance":
        source_result = logical.pop("result")
        result_pointer = record["result"]
        reference_pointer = record["reference"]
        typed_result = (
            None
            if typed_records is None
            else typed_records.get(f"results:{result_pointer['$ref']}")
        )
        typed_reference = (
            None
            if typed_records is None
            else typed_records.get(
                f"evidence_refs:{reference_pointer['$ref']}"
            )
        )
        logical["statement"] = _statement_from_path(
            source_result,
            logical["reference"],
            typed_result=typed_result,
            typed_reference=(
                typed_reference
                if isinstance(typed_reference, StatementRef)
                else None
            ),
        )
    typed = _validate_typed_table_record(table, kind, logical)
    if typed is not None and typed_records is not None:
        typed_records[f"{table}:{identifier}"] = typed
    memo[f"{table}:{identifier}"] = copy.deepcopy(logical)
    return logical


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


def _validate_generation_coverage(  # noqa: C901, PLR0913 # Check complete manifest coverage.
    generation: dict,
    details: list[BaseModel],
    planning: list[BaseModel],
    sections: list[BaseModel],
    document: BaseModel,
    plan: StoryPlan,
) -> dict[str, BaseModel]:
    unit_keys = generation["unit_keys"]
    units = set(unit_keys)
    if len(units) != len(unit_keys):
        _fail("duplicate_ledger_unit")
    if (
        len(details) != len(unit_keys)
        or {item.unit_key for item in details} != units
    ):
        _fail("detail_coverage")
    group_keys = [group.group.key for group in plan.groups]
    if len(set(group_keys)) != len(group_keys):
        _fail("duplicate_plan_group")
    if len(plan.ordered_group_keys) != len(plan.groups) or set(
        plan.ordered_group_keys
    ) != set(group_keys):
        _fail("plan_group_order")
    if any(not set(group.group.unit_keys) <= units for group in plan.groups):
        _fail("plan_unit_coverage")
    if tuple(section.binding.subject_key for section in sections) != tuple(
        plan.ordered_group_keys
    ):
        _fail("section_coverage")
    subjects = [*plan.cards]
    if len({card.id for card in subjects}) != len(subjects):
        _fail("duplicate_plan_card")
    if plan.production is not None:
        subjects.extend(plan.production.auxiliary_cards)
    subject_ids = [card.id for card in subjects]
    if len(set(subject_ids)) != len(subject_ids):
        _fail("duplicate_plan_card")
    if len(planning) != len(subjects) or {
        card.binding.subject_key for card in planning
    } != set(subject_ids):
        _fail("planning_coverage")
    prose = [*planning, *sections, document]
    if any(result.binding.plan_key != plan.id for result in prose):
        _fail("prose_plan_mismatch")
    ordered_results = [*details, *planning, *sections, document]
    logical_keys = [result_key(record) for record in ordered_results]
    if len(set(logical_keys)) != len(logical_keys):
        _fail("duplicate_result_identity")
    return dict(zip(logical_keys, ordered_results, strict=True))


def _validate_generation_producer_coverage(
    receipts: list[ResultReceipt], producers: list[ProducerDescriptor]
) -> dict[str, ProducerDescriptor]:
    producer_by_key = {item.key: item for item in producers}
    if len(producer_by_key) != len(producers):
        _fail("duplicate_producer")
    producer_identities = [
        (
            item.original_run.digest,
            item.invocation.semantic_key,
            item.invocation.batch.id,
        )
        for item in producers
    ]
    if len(set(producer_identities)) != len(producer_identities):
        _fail("duplicate_original_producer")
    if {receipt.producer_key for receipt in receipts} != set(producer_by_key):
        _fail("producer_coverage")
    return producer_by_key


def _validate_generation_receipts(  # noqa: C901, PLR0912, PLR0913 # Check producer and receipt bindings.
    generation: dict,
    receipts: list[ResultReceipt],
    result_by_key: dict[str, BaseModel],
    producers: list[ProducerDescriptor],
    producer_by_key: dict[str, ProducerDescriptor],
    run: RunIdentity | None,
) -> None:
    requests = {
        key: {request.batch.id: request for request in producer.requests}
        for key, producer in producer_by_key.items()
    }
    lookups = {
        key: {
            lookup.item_key: lookup for lookup in producer.invocation.lookups
        }
        for key, producer in producer_by_key.items()
    }
    artifact_keys = {
        key: producer.invocation.artifact_keys()
        for key, producer in producer_by_key.items()
    }
    request_items = {
        (key, request.batch.id): set(request.batch.item_keys)
        for key, producer in producer_by_key.items()
        for request in producer.requests
    }
    receipt_by_item = {item.item_key: item for item in receipts}
    if len(receipt_by_item) != len(receipts) or set(receipt_by_item) != set(
        result_by_key
    ):
        _fail("receipt_coverage")
    used_requests: set[tuple[str, str]] = set()
    sent_requests: set[tuple[str, str]] = set()
    for receipt in receipts:
        item = result_by_key.get(receipt.item_key)
        producer = producer_by_key.get(receipt.producer_key)
        if (
            item is None
            or producer is None
            or receipt.result_kind != result_kind(item)
        ):
            _fail("receipt_binding_mismatch")
        if receipt.result_digest != result_digest(item):
            _fail("receipt_result_mismatch")
        request_identity = (receipt.producer_key, receipt.request_key)
        request = requests[receipt.producer_key].get(receipt.request_key)
        lookup = lookups[receipt.producer_key].get(receipt.item_key)
        if (
            request is None
            or lookup is None
            or lookup.result_kind != receipt.result_kind
            or receipt.item_key
            not in request_items.get(request_identity, set())
        ):
            _fail("receipt_binding_mismatch")
        if receipt.artifact_key != artifact_keys[receipt.producer_key].get(
            receipt.item_key
        ):
            _fail("receipt_artifact_mismatch")
        used_requests.add(request_identity)
        if receipt.disposition == "sent":
            sent_requests.add(request_identity)
    if used_requests != {
        (key, request_id)
        for key, request_map in requests.items()
        for request_id in request_map
    }:
        _fail("unused_production_request")
    if run is not None:
        for producer in producers:
            if (
                producer.original_run.provider_destination
                != run.provider_destination
                or producer.original_run.model != run.model
            ):
                _fail("producer_identity_mismatch")
        for receipt in receipts:
            if (
                receipt.disposition == "sent"
                and producer_by_key[receipt.producer_key].original_run != run
            ):
                _fail("sent_run_mismatch")
    current_usage = generation.get("current_usage")
    if (
        type(current_usage) is not dict
        or type(current_usage.get("calls")) is not int
        or len(sent_requests) > current_usage["calls"]
    ):
        _fail("sent_usage_mismatch")


def _validate_generation_relationships(
    root: dict,
    typed_records: dict[str, BaseModel],
) -> None:
    generation = root["generation"]

    def get(table: str, pointer: dict) -> BaseModel:
        identifier, kind = _check_reference_shape(pointer)
        record = typed_records.get(f"{table}:{identifier}")
        if record is None:
            _fail(f"missing_validated_storage_model:{table}:{kind}")
        return record

    details = [get("results", item) for item in generation["details"]]
    planning = [get("results", item) for item in generation["planning"]]
    sections = [get("results", item) for item in generation["sections"]]
    document = get("results", generation["document"])
    plan = get("plans", generation["plan"])
    producers = [get("producers", item) for item in generation["producers"]]
    receipts = [get("receipts", item) for item in generation["receipts"]]
    if not isinstance(plan, StoryPlan):
        _fail("missing_validated_storage_model:plans:story_plan")
    if any(not isinstance(item, ProducerDescriptor) for item in producers):
        _fail("missing_validated_storage_model:producers:producer")
    if any(not isinstance(item, ResultReceipt) for item in receipts):
        _fail("missing_validated_storage_model:receipts:receipt")
    result_by_key = _validate_generation_coverage(
        generation, details, planning, sections, document, plan
    )
    producer_by_key = _validate_generation_producer_coverage(
        receipts, producers
    )
    run = None
    if generation["run"] is not None:
        run = get("runs", generation["run"])
        if not isinstance(run, RunIdentity):
            _fail("missing_validated_storage_model:runs:run")
    _validate_generation_receipts(
        generation, receipts, result_by_key, producers, producer_by_key, run
    )


def _validate_reader_relationships(  # noqa: C901, PLR0912 # Check ordered reader ownership and provenance.
    root: dict, tables: dict[str, dict[str, dict]], memo: dict[str, dict]
) -> None:
    reader = tables["readers"][root["reader"]["$ref"]]["record"]
    generation = root["generation"]
    generation_results: set[str] = set()
    for field in ("details", "planning", "sections"):
        generation_results.update(item["$ref"] for item in generation[field])
    generation_results.add(generation["document"]["$ref"])
    if reader["document"]["$ref"] != generation["document"]["$ref"]:
        _fail("reader_document_mismatch")
    section_field = (
        "sections" if reader["shape"] == "presentation" else "steps"
    )
    reader_sections = reader[section_field]
    generation_sections = generation["sections"]
    if len(reader_sections) != len(generation_sections):
        _fail("reader_section_mismatch")
    plan = _logical_record(tables, "plans", generation["plan"]["$ref"], memo)
    groups = {item["group"]["key"]: item["group"] for item in plan["groups"]}
    ordered_group_keys = plan["ordered_group_keys"]
    if len(reader_sections) != len(ordered_group_keys):
        _fail("reader_section_mismatch")
    details_by_unit = {
        _logical_record(tables, "results", item["$ref"], memo)[
            "unit_key"
        ]: item
        for item in generation["details"]
    }
    assigned_units = {
        unit_key
        for group in groups.values()
        for unit_key in group["unit_keys"]
    }
    for position, (section, generation_pointer, group_key) in enumerate(
        zip(
            reader_sections,
            generation_sections,
            ordered_group_keys,
            strict=True,
        )
    ):
        prose = _logical_record(
            tables, "results", generation_pointer["$ref"], memo
        )
        expected_next = (
            reader_sections[position + 1]["group_id"]
            if position + 1 < len(reader_sections)
            else None
        )
        expected_details = [
            details_by_unit[unit_key]["$ref"]
            for unit_key in groups[group_key]["unit_keys"]
        ]
        if (
            section["prose"]["$ref"] != generation_pointer["$ref"]
            or section["group_key"] != group_key
            or section["group_key"] != prose["binding"]["subject_key"]
            or section["number"] != position + 1
            or section["next_id"] != expected_next
        ):
            _fail("reader_section_mismatch")
        if [item["$ref"] for item in section["details"]] != expected_details:
            _fail("reader_detail_mismatch")
    expected_context = [
        item["$ref"]
        for item in generation["details"]
        if _logical_record(tables, "results", item["$ref"], memo)["unit_key"]
        not in assigned_units
    ]
    if [
        item["$ref"] for item in reader["context_details"]
    ] != expected_context:
        _fail("reader_context_mismatch")
    for section in reader_sections:
        if section["prose"]["$ref"] not in {
            item["$ref"] for item in generation_sections
        }:
            _fail("reader_section_mismatch")
        if any(
            item["$ref"] not in generation_results
            for item in section["details"]
        ):
            _fail("reader_detail_mismatch")
    if any(
        item["$ref"] not in generation_results
        for item in reader["context_details"]
    ):
        _fail("reader_context_mismatch")
    for item in reader["support_provenance"]:
        prov = _logical_record(
            tables, "inherited_provenance", item["$ref"], memo
        )
        if (
            prov["reference"]["producer_item_key"]
            != prov["receipt"]["item_key"]
        ):
            _fail("reader_provenance_mismatch")
        matching = next(
            (
                receipt
                for pointer in generation["receipts"]
                if (
                    receipt := _logical_record(
                        tables, "receipts", pointer["$ref"], memo
                    )
                )["item_key"]
                == prov["reference"]["producer_item_key"]
            ),
            None,
        )
        if matching is None or canonical_bytes(matching) != canonical_bytes(
            prov["receipt"]
        ):
            _fail("reader_provenance_receipt_mismatch")


def _validate_table_records(
    root: dict, indexes: dict[str, dict[str, dict]]
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
    _validate_generation_relationships(root, typed_records)
    _validate_reader_relationships(root, tables, memo)
    return memo, typed_records, validation_counts


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


def _iter_logical_record_json(  # noqa: C901 # Stream typed reader references in canonical order.
    value: object,
    *,
    references: Mapping[tuple[object, ...], tuple[str, str]] | None = None,
    indexes: dict[str, dict[str, dict]] | None = None,
    memo: dict[str, dict] | None = None,
    path: tuple[object, ...] = (),
) -> Iterator[bytes]:
    """Stream canonical JSON, resolving only listed schema reference paths."""
    target = None if references is None else references.get(path)
    if target is not None:
        if indexes is None or memo is None:
            _fail("invalid_storage_reference_position")
        table, expected_kind = target
        identifier, pointer_kind = _check_reference_shape(value)
        if expected_kind and pointer_kind != expected_kind:
            _fail("wrong_kind_storage_reference")
        entry = indexes[table].get(identifier)
        if entry is None or entry["kind"] != pointer_kind:
            _fail("wrong_kind_storage_reference")
        logical = memo.get(f"{table}:{identifier}")
        if logical is None:
            _fail("missing_storage_record")
        yield from _iter_logical_record_json(logical)
        return
    if isinstance(value, Mapping):
        yield b"{"
        first = True
        for key in sorted(value):
            if not first:
                yield bytes((44,))
            first = False
            yield canonical_bytes(key)
            yield b":"
            yield from _iter_logical_record_json(
                value[key],
                references=references,
                indexes=indexes,
                memo=memo,
                path=(*path, key),
            )
        yield b"}"
        return
    if type(value) in {list, tuple}:
        yield b"["
        for position, item in enumerate(value):
            if position:
                yield bytes((44,))
            yield from _iter_logical_record_json(
                item,
                references=references,
                indexes=indexes,
                memo=memo,
                path=(*path, position),
            )
        yield b"]"
        return
    yield canonical_bytes(value)


def _logical_storage_address(
    table: str,
    identifier: str,
    entry: dict,
    indexes: dict[str, dict[str, dict]],
    memo: dict[str, dict],
) -> str:
    kind = entry["kind"]
    if table != "readers":
        logical_record = memo.get(f"{table}:{identifier}")
        if logical_record is None:
            _fail("missing_storage_record")
        return storage_address(kind, logical_record)

    record = entry["record"]
    digest = hashlib.sha256()
    digest.update(b'{"kind":')
    digest.update(canonical_bytes(kind))
    digest.update(b',"record":')
    for chunk in _logical_reader_chunks(record, indexes, memo):
        digest.update(chunk)
    digest.update(b',"storage_version":')
    digest.update(canonical_bytes(STORAGE_VERSION))
    digest.update(b"}")
    return digest.hexdigest()


def _logical_reader_chunks(
    record: dict,
    indexes: dict[str, dict[str, dict]],
    memo: dict[str, dict],
) -> Iterator[bytes]:
    """Stream a reader's complete logical content through declared references."""
    logical_reader = {
        key: value for key, value in record.items() if key != "shape"
    }
    references = {
        path: (target_table, target_kind)
        for path, target_table, target_kind in _record_ref_locations(
            "readers", "reader", record
        )
    }
    yield from _iter_logical_record_json(
        logical_reader,
        references=references,
        indexes=indexes,
        memo=memo,
    )


def _logical_reader_streams_equal(  # noqa: C901 # Compare arbitrarily chunked logical reader streams.
    left_record: dict,
    right_record: dict,
    indexes: dict[str, dict[str, dict]],
    memo: dict[str, dict],
) -> bool:
    """Compare logical reader bytes in bounded chunks without joining them."""
    left = iter(_logical_reader_chunks(left_record, indexes, memo))
    right = iter(_logical_reader_chunks(right_record, indexes, memo))
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


def _verify_storage_addresses(
    indexes: dict[str, dict[str, dict]], memo: dict[str, dict]
) -> None:
    for table, index in indexes.items():
        for identifier, entry in index.items():
            if (
                _logical_storage_address(
                    table, identifier, entry, indexes, memo
                )
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
    _verify_storage_addresses(indexes, memo)
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
