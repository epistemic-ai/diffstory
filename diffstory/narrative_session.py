"""Run-scoped immutable validation and application capabilities for v2 reports.

The session owns canonical JSON snapshots. Imported generations receive a
capability only after the complete production proof succeeds. Capabilities are
opaque, registered by object identity, and expire when the session closes.
"""

from __future__ import annotations

from collections.abc import Iterator
from collections.abc import Mapping
from dataclasses import dataclass
from hashlib import sha256
from typing import Self
from weakref import ref as weakref_ref

from pydantic import BaseModel

from .narrative_manifest import _apply_v2_candidate
from .narrative_manifest import compiler_fact_signature
from .narrative_manifest import persisted_manifest
from .narrative_manifest import validate_generation
from .narrative_plan import _OwnedList
from .narrative_plan import _OwnedMapping
from .narrative_plan import _OwnedTuple
from .narrative_plan import _raw_integrity_bytes
from .narrative_plan import _thaw_owned
from .narrative_plan import canonical_bytes
from .narrative_plan import canonical_digest
from .narrative_plan import content_digest
from .narrative_plan import semantic_digest
from .narrative_protocol import RequestConfiguration
from .narrative_wire import strict_json
from .render import _validate_indexed_report_surface
from .render import _validate_report_collections
from .render import _validate_report_schema
from .render import _validate_v2_report_surface

_ISSUER_TOKEN = object()
_VALIDATION_VERSION = "diffstory.session.import-proof.v1"
_ASSEMBLY_VERSION = "diffstory.session.apply-v1"
_RENDER_SURFACE_VERSION = "diffstory.session.render-surface.v1"
_GRAPH_BINDING_SIZE = 8
_StageWitness = tuple[str, tuple[str, ...], str]
_StageWitnessCache = dict[
    int, tuple[object, object, object | None, _StageWitness]
]
_ContextWitnessCache = dict[int, tuple[object, object | None, str | None]]


def _require_session(condition: bool, reason: str) -> None:  # noqa: FBT001 # Keep internal boundary failures explicit.
    """Raise the stable session reason when an owned invariant fails."""
    if not condition:
        raise ValueError(reason)


def _production_binding_witness(production_inputs: object) -> str:
    """Hash the exact small mutable bindings held by production inputs.

    Preconditions:
        ``production_inputs`` is the coordinator-owned ProductionInputs value.
    Postconditions:
        Returns a type-preserving digest for its options and transport binding.
    Essential behavior:
        Leaves ledger and source validation to their existing scoped witnesses.
    """
    transport = getattr(production_inputs, "transport", None)
    if transport is None:
        msg = "captured_production_binding_changed"
        raise ValueError(msg)
    common = (
        id(production_inputs),
        id(transport),
        id(getattr(production_inputs, "options", None)),
        id(getattr(transport, "options", None)),
        id(getattr(transport, "descriptor", None)),
        id(getattr(transport, "limits", None)),
        content_digest(_raw_integrity_bytes(production_inputs.options)),
        content_digest(_raw_integrity_bytes(transport.options)),
        content_digest(_raw_integrity_bytes(transport.descriptor)),
        content_digest(_raw_integrity_bytes(transport.limits)),
        type(transport).__module__,
        type(transport).__qualname__,
        type(transport.capacity).__module__,
        type(transport.capacity).__qualname__,
        transport.capacity.context_tokens,
        transport.capacity.max_output_tokens,
        content_digest(transport.prefix),
        transport.config_digest,
        transport.enforceable,
    )
    if not hasattr(transport, "provider"):
        return semantic_digest(
            "diffstory.captured-input.offline-production-bindings.v1",
            common,
        )
    provider = transport.provider
    if (
        getattr(provider, "prepare", None) != transport.prepare_method
        or getattr(provider, "complete_reply", None) != transport.send_method
    ):
        msg = "captured_production_binding_changed"
        raise ValueError(msg)
    provider_fields = (
        "name",
        "model",
        "destination",
        "context_tokens",
        "max_output_tokens",
        "input_overhead_bytes",
        "request_prefix",
        "request_protocol",
        "request_json_layers",
        "enforceable_output_cap",
        "_requested_model",
        "_executable",
    )
    return semantic_digest(
        "diffstory.captured-input.production-bindings.v1",
        common,
        id(provider),
        content_digest(
            _raw_integrity_bytes(
                tuple(
                    (name, getattr(provider, name, None))
                    for name in provider_fields
                )
            )
        ),
        (type(provider).__module__, type(provider).__qualname__),
        content_digest(transport.config),
        transport.is_codex,
        transport.is_api,
        id(transport.prepare_method),
        id(transport.send_method),
    )


class _OpaqueCapability:
    """Identity-only token; all authority remains in a session registry."""

    __slots__ = ()

    def __new__(cls, issuer: object = None) -> Self:
        if issuer is not _ISSUER_TOKEN:
            msg = "Session capabilities are issued by a live ReportSession"
            raise TypeError(msg)
        return super().__new__(cls)

    def __copy__(self) -> _OpaqueCapability:
        msg = "Session capabilities cannot be copied"
        raise TypeError(msg)

    def __deepcopy__(self, _memo: dict) -> _OpaqueCapability:
        msg = "Session capabilities cannot be copied"
        raise TypeError(msg)

    def __reduce_ex__(self, _protocol: int) -> object:
        msg = "Session capabilities cannot be serialized"
        raise TypeError(msg)

    def __repr__(self) -> str:
        return f"<{type(self).__name__} opaque>"


class ParsedRecords(_OpaqueCapability):
    """Structurally parsed import data; it carries no proof authority."""

    __slots__ = ()


class ValidatedGeneration(_OpaqueCapability):
    """A complete fresh or imported generation proof for one live session."""

    __slots__ = ()


class StageProof(_OpaqueCapability):
    """A successfully validated production stage owned by one session."""

    __slots__ = ()


class ValidatedReport(_OpaqueCapability):
    """An applied report that passed this session's renderer safety checks."""

    __slots__ = ()


@dataclass(frozen=True, slots=True)
class _ParsedImport:
    scope_digest: str
    manifest_bytes: bytes
    candidate_bytes: bytes | None


@dataclass(frozen=True, slots=True)
class _GenerationProof:
    scope_digest: str
    report_digest: str
    configuration_bytes: bytes
    manifest_bytes: bytes
    candidate_bytes: bytes
    original_run_bytes: bytes
    compiler_fact_signature: str
    supplied_candidate_digest: str | None
    validation_version: str


@dataclass(frozen=True, slots=True)
class _ReportProof:
    scope_digest: str
    report_digest: str
    report_value: object
    configuration_bytes: bytes
    original_run_bytes: bytes
    compiler_fact_signature: str
    manifest_digest: str
    candidate_digest: str
    validation_version: str
    assembly_version: str
    render_surface_version: str


@dataclass(frozen=True, slots=True)
class _IndexedGenerationProof:
    """Sealed indexed generation evidence retained for the later apply unit.

    Attributes:
        scope_digest: Owning session identity.
        report_digest: Full captured pre-annotation report identity.
        configuration_bytes: Exact normalized request descriptor.
        compiler_fact_signature: Captured compiler-fact identity.
        records: Exact ordered generation record references.
        graph: Closed annotations graph sharing the owned result/audit rows.
        reader: Ordered reader references retained by that graph.
        overlay: Detached deterministic document/group fields for later apply.
        run_ref: Captured pre-annotation run row.
        application_payload: Data-only graph/base/overlay handoff.
        graph_binding: Identity pins for the source graph's immutable root and
            table/index views; rows remain shared and recursively immutable.
        stage_proofs: Exact capabilities that completed this production.
        versions_digest: Captured validation and compatibility versions.
        witness: Type-preserving digest of all retained payload bindings.
    """

    scope_digest: str
    report_digest: str
    configuration_bytes: bytes
    compiler_fact_signature: str
    records: object
    graph: object
    reader: object
    overlay: object
    run_ref: object
    application_payload: _IndexedApplicationPayload
    graph_binding: tuple[object, ...]
    stage_proofs: tuple[StageProof, ...]
    versions_digest: str
    witness: str


@dataclass(frozen=True, slots=True)
class _IndexedApplicationPayload:
    """Data-only M6 handoff over the captured base and shared graph rows.

    Attributes:
        graph: Report root with shared result/audit rows and a presentation
            reader projection. It is not a ``ValidatedReport`` capability.
        records: Exact registered generation record set for that root.
        reader: Ordered selected result/citation/provenance references.
        base_report: The exact immutable report owned by the live session.
        overlay: Small deterministic document and group fields derived from the
            independently validated annotation reader.
        original_run_bytes: Canonical bytes for the embedded pre-annotation
            run identity, separate from the current captured report digest.
        versions_digest: Captured validation and compatibility version binding.
        graph_binding: Identity pins for the derived report root, table views,
            and indexes. The graph's recursively immutable rows are shared.

    M6 may consume this value only through the exact registered generation
    proof. Before ``apply`` registers ``ValidatedReport``, it must adapt the
    existing renderer schema, collection, identifier, numeric, link, tag, and
    provenance predicates to ``base_report``, ``overlay``, and selected reader
    rows. This value carries data only and cannot skip those checks.
    """

    graph: object
    records: object
    reader: object
    base_report: object
    overlay: object
    original_run_bytes: bytes
    versions_digest: str
    graph_binding: tuple[object, ...]


class _CapturedInput:
    """One detached report/configuration snapshot shared by a live session.

    The report is one immutable primitive tree. Digests replace a retained
    canonical byte buffer. Ledger/index/source bindings are attached lazily by
    ProductionInputs and remain local to this capture.

    Attributes:
        _report: One owned report root; empty only after close.
        _report_digest: Lowercase 64-character SHA256 digest or empty after close.
        _report_shape_digest: Lowercase 64-character raw-shape digest or empty after close.
        _configuration_bytes: Compact normalized JSON descriptor or empty after close.
        _run_integrity_digest: None before run binding, else a lowercase 64-character digest.
        _author_integrity_digest: None before author binding, else a lowercase 64-character digest.
        _ledger_integrity_digest: None before ledger binding, else a lowercase 64-character digest.
        _ledger_parameters_integrity_digest: None before ledger binding, else a lowercase 64-character digest.
        _production_binding_digest: None before production binding, else a lowercase 64-character digest.
        _source_binding: None before prose derivation, else the exact current source adapter.
        _coordinator: None or a weak reference to the one local coordinator.
    """

    __slots__ = (
        "_author",
        "_author_digest",
        "_author_integrity_digest",
        "_closed",
        "_compiler_fact_signature",
        "_configuration_bytes",
        "_configuration_digest",
        "_coordinator",
        "_index",
        "_ledger",
        "_ledger_integrity_digest",
        "_ledger_parameters",
        "_ledger_parameters_integrity_digest",
        "_model_integrity_digest",
        "_owner_session",
        "_production_binding_digest",
        "_production_inputs",
        "_report",
        "_report_digest",
        "_report_shape_digest",
        "_report_type",
        "_revision_binding",
        "_run",
        "_run_integrity_digest",
        "_scope_digest",
        "_source_binding",
        "_versions_digest",
    )

    def __init__(self, report: object, request_configuration: object) -> None:
        """Capture one immutable report and normalized typed configuration.

        Preconditions:
            Inputs are JSON report data and a valid request-configuration
            descriptor or its serialized JSON form.
        Postconditions:
            Owns one read-only primitive report, small canonical configuration
            bytes, and digests for content, shape, scope, and stage versions.
        Essential behavior:
            Does not retain canonical report bytes or a second parsed tree.
        """
        self._initialize_capture(
            _snapshot_report(report),
            request_configuration,
            report_type=(
                type(report) if isinstance(report, BaseModel) else None
            ),
        )

    @classmethod
    def _from_parsed_report(
        cls, parsed: object, request_configuration: object
    ) -> Self:
        """Transfer the parser-owned immutable base into this capture.

        Preconditions:
            ``parsed`` is the exact private parser result for report bytes,
            and its base was frozen into this module's owned report types.
        Postconditions:
            The capture retains that exact base object and shares all normal
            configuration, digest, and version initialization checks.
        Essential behavior:
            Does not thaw or recursively copy the compiler base report.
        """
        from .report_storage import _ParsedStorageImport

        if (
            type(parsed) is not _ParsedStorageImport
            or parsed.artifact_kind != "report"
        ):
            msg = "invalid_parsed_report_capture"
            raise ValueError(msg)
        report_view = parsed.root.get("base_report")
        if type(report_view) is not _OwnedMapping:
            msg = "invalid_parsed_report_capture"
            raise ValueError(msg)
        instance = object.__new__(cls)
        instance._initialize_capture(  # noqa: SLF001 # Share the standard capture initializer without entering public capture.
            _owned_report_snapshot(report_view, None),
            request_configuration,
            report_type=None,
        )
        return instance

    def _initialize_capture(
        self,
        report_snapshot: tuple[_OwnedMapping, str, str, str | None],
        request_configuration: object,
        *,
        report_type: type[BaseModel] | None,
    ) -> None:
        """Initialize the shared capture state from one immutable report.

        Preconditions:
            ``report_snapshot`` contains one owned report and its exact raw
            digest, shape digest, and optional original-model witness.
        Postconditions:
            Installs the report plus normalized configuration and all current
            version, source-revision, compiler-fact, and scope witnesses.
        Essential behavior:
            Gives public detached capture and private parser transfer the same
            configuration and registration checks.
        """
        report_view, report_digest, shape_digest, model_digest = (
            report_snapshot
        )
        configuration_input = _snapshot_object(
            request_configuration, "request_configuration"
        )
        configuration = RequestConfiguration.model_validate_json(
            configuration_input
        )
        configuration_value = configuration.model_dump(mode="json")
        configuration_bytes = canonical_bytes(configuration_value)
        self._report = report_view
        self._report_digest = report_digest
        self._report_shape_digest = shape_digest
        self._model_integrity_digest = model_digest
        self._report_type = report_type
        self._configuration_bytes = configuration_bytes
        self._configuration_digest = content_digest(configuration_bytes)
        try:
            self._compiler_fact_signature = compiler_fact_signature(
                report_view
            )
        except (KeyError, TypeError, ValueError):
            # Capture remains a data boundary. Invalid compiler shape is
            # rejected by the later complete production proof.
            self._compiler_fact_signature = None
        self._revision_binding = self._read_revisions(report_view)
        self._versions_digest = semantic_digest(
            "diffstory.captured-input.versions.v1",
            _VALIDATION_VERSION,
            _ASSEMBLY_VERSION,
            _RENDER_SURFACE_VERSION,
            configuration.stages,
        )
        self._scope_digest = semantic_digest(
            "diffstory.report-session.scope.v2",
            report_digest,
            shape_digest,
            self._configuration_digest,
            self._versions_digest,
            self._revision_binding,
            self._compiler_fact_signature,
        )
        self._owner_session: ReportSession | None = None
        self._ledger = None
        self._index = None
        self._ledger_integrity_digest = None
        self._ledger_parameters_integrity_digest = None
        self._ledger_parameters = None
        self._run = None
        self._run_integrity_digest = None
        self._author = None
        self._author_digest = None
        self._author_integrity_digest = None
        self._production_inputs = None
        self._production_binding_digest = None
        self._source_binding = None
        self._coordinator = None
        self._closed = False

    @staticmethod
    def _read_revisions(report: _OwnedMapping) -> tuple[object, object]:
        """Read base/head revision values without expanding the report."""
        meta = report.get("meta")
        if type(meta) not in {dict, _OwnedMapping}:
            return None, None
        return meta.get("base_sha"), meta.get("head_sha")

    @property
    def report_view(self) -> _OwnedMapping:
        """Return the private immutable traversal view for owned internals."""
        self._require_open()
        return self._report

    @property
    def report_digest(self) -> str:
        """Return canonical bytes' digest without retaining those bytes."""
        self._require_open()
        return self._report_digest

    @property
    def shape_digest(self) -> str:
        """Return the type-preserving Python container/scalar witness."""
        self._require_open()
        return self._report_shape_digest

    @property
    def compiler_signature(self) -> str | None:
        """Return captured compiler-fact identity when the report shape allows it."""
        self._require_open()
        return self._compiler_fact_signature

    @property
    def configuration_bytes(self) -> bytes:
        """Return the small normalized request-configuration descriptor."""
        self._require_open()
        return self._configuration_bytes

    @property
    def scope_identity(self) -> str:
        """Return the private registration digest for this exact capture."""
        self._require_open()
        return self._scope_digest

    @property
    def configuration_identity(self) -> str:
        """Return the digest of the normalized request descriptor."""
        self._require_open()
        return self._configuration_digest

    @property
    def version_identity(self) -> str:
        """Return the current digest of session and stage contract versions.

        Preconditions:
            The capture is open and version constants still match capture time.
        Postconditions:
            Returns a lowercase 64-character SHA256 identity.
        Essential behavior:
            Rejects a code-version change before this value can bind a stage.
        """
        self._require_open()
        if self._current_versions_digest() != self._versions_digest:
            msg = "captured_version_changed"
            raise ValueError(msg)
        return self._versions_digest

    @property
    def owner_session(self) -> ReportSession | None:
        """Return the owning session only to local coordinator internals."""
        self._require_open()
        return self._owner_session

    @property
    def ledger(self) -> object | None:
        """Return the lazily shared ledger to local production internals."""
        self._require_open()
        return self._ledger

    @property
    def index(self) -> object | None:
        """Return the lazily shared index to local production internals."""
        self._require_open()
        return self._index

    @property
    def run(self) -> object | None:
        """Return the shared full run identity after production binding."""
        self._require_open()
        return self._run

    @property
    def author(self) -> object | None:
        """Return the exact normalized author model shared by production."""
        self._require_open()
        return self._author

    @property
    def author_digest(self) -> str | None:
        """Return the normalized author-context identity shared by production."""
        self._require_open()
        return self._author_digest

    def _current_versions_digest(self) -> str:
        """Recompute version identity from current code and captured stages.

        Preconditions:
            The stored request descriptor is valid canonical JSON.
        Postconditions:
            Returns a lowercase 64-character SHA256 identity.
        Essential behavior:
            Includes validation, assembly, rendering, and captured stage versions.
        """
        configuration = RequestConfiguration.model_validate_json(
            self._configuration_bytes
        )
        return semantic_digest(
            "diffstory.captured-input.versions.v1",
            _VALIDATION_VERSION,
            _ASSEMBLY_VERSION,
            _RENDER_SURFACE_VERSION,
            configuration.stages,
        )

    def assert_capture_current(
        self, production_inputs: object | None = None
    ) -> None:
        """Reject changed owned data and derived bindings at a trust boundary.

        Preconditions:
            The capture is open; ``production_inputs`` is its coordinator
            owner when supplied.
        Postconditions:
            Returns only when captured report, shape, configuration, versions,
            scope, and bound production data still match their initial values.
        Essential behavior:
            Reuses the one owned report and the ledger/source scopes already
            attached to this capture.
        """
        self._require_open()
        if (
            canonical_digest(self._report) != self._report_digest
            or _shape_digest(self._report) != self._report_shape_digest
            or content_digest(self._configuration_bytes)
            != self._configuration_digest
            or self._read_revisions(self._report) != self._revision_binding
            or self._current_versions_digest() != self._versions_digest
            or self._current_scope_digest() != self._scope_digest
        ):
            msg = "captured_input_changed"
            raise ValueError(msg)
        owner = production_inputs
        if owner is None and self._production_inputs is not None:
            owner = self._production_inputs()
        if owner is not None:
            self.assert_production_bindings(owner)

    def _current_scope_digest(self) -> str:
        """Recompute the input scope identity from its pinned components.

        Preconditions:
            The report and configuration witnesses were captured successfully.
        Postconditions:
            Returns a lowercase 64-character SHA256 identity.
        Essential behavior:
            Binds full report content, raw shape, revisions, facts, config, and versions.
        """
        return semantic_digest(
            "diffstory.report-session.scope.v2",
            self._report_digest,
            self._report_shape_digest,
            self._configuration_digest,
            self._versions_digest,
            self._revision_binding,
            self._compiler_fact_signature,
        )

    def bind_session(self, session: ReportSession) -> None:
        """Attach this exact capture to one live session identity."""
        self._require_open()
        if (
            self._owner_session is not None
            and self._owner_session is not session
        ):
            msg = "captured_input_session_mismatch"
            raise ValueError(msg)
        self._owner_session = session

    def owns_report_view(self, value: object) -> bool:
        """Check the exact readonly root accepted by the private proof path."""
        self._require_open()
        return value is self._report

    def check_report(self, value: object) -> None:
        """Reject caller changes while keeping only digests in session state.

        Preconditions:
            ``value`` is the mutable caller report retained for approval checks.
        Postconditions:
            Returns only when its canonical content and Python shape are current.
        Essential behavior:
            Uses raw model integrity for Pydantic models and streamed canonical
            hashing for mappings; it never rebuilds or retains report bytes.
        """
        self._require_open()
        if value is self._report:
            if (
                canonical_digest(self._report) != self._report_digest
                or _shape_digest(self._report) != self._report_shape_digest
            ):
                msg = "captured_input_changed"
                raise ValueError(msg)
            return
        digest, shape_digest, model_digest = _report_identity(value)
        if (
            digest != self._report_digest
            or shape_digest != self._report_shape_digest
        ):
            msg = "dependencies_changed"
            raise ValueError(msg)
        if self._report_type is not None and (
            type(value) is not self._report_type
            or model_digest != self._model_integrity_digest
        ):
            msg = "dependencies_changed"
            raise ValueError(msg)

    def expand_report(self) -> dict:
        """Return a detached report for explicit compatibility consumers."""
        self._require_open()
        value = _thaw_owned(self._report)
        if type(value) is not dict:
            msg = "invalid_owned_report"
            raise ValueError(msg)
        return value

    def canonical_report_bytes(self) -> bytes:
        """Create canonical input bytes only for an explicit compatibility call."""
        self._require_open()
        return canonical_bytes(self._report)

    def bind_run(self, run: object) -> object:
        """Register one exact type-preserving run identity.

        Preconditions:
            ``run`` is a supported immutable run model derived from this capture.
        Postconditions:
            Returns the first bound model only when its raw fields still match.
        Essential behavior:
            Pins every field, including the full pre-annotation report signature.
        """
        self._require_open()
        integrity_digest = content_digest(_raw_integrity_bytes(run))
        if self._run is None:
            self._run = run
            self._run_integrity_digest = integrity_digest
        elif (
            self._run_integrity_digest != integrity_digest
            or content_digest(_raw_integrity_bytes(self._run))
            != self._run_integrity_digest
        ):
            msg = "captured_run_mismatch"
            raise ValueError(msg)
        return self._run

    def bind_author(self, author: object, digest: str) -> tuple[object, str]:
        """Share one normalized author model and its semantic digest.

        Preconditions:
            ``author`` is the normalized author model derived from the report.
            ``digest`` is its existing semantic identity.
        Postconditions:
            Returns the first model and digest only while raw fields still match.
        Essential behavior:
            Keeps semantic identity unchanged and adds a raw integrity witness.
        """
        self._require_open()
        integrity_digest = content_digest(_raw_integrity_bytes(author))
        if self._author is None:
            self._author = author
            self._author_digest = digest
            self._author_integrity_digest = integrity_digest
        elif (
            self._author_integrity_digest != integrity_digest
            or content_digest(_raw_integrity_bytes(self._author))
            != self._author_integrity_digest
            or self._author_digest != digest
        ):
            msg = "captured_author_binding_mismatch"
            raise ValueError(msg)
        return self._author, self._author_digest

    def bind_ledger_index(
        self, ledger: object, index: object, ledger_parameters: object
    ) -> tuple[object, object]:
        """Share one ledger/index pair and pin its parameter model.

        Preconditions:
            The index validates the supplied ledger and parameters match capture.
        Postconditions:
            Returns the first ledger/index pair only while raw bindings remain.
        Essential behavior:
            Reuses the pair without building a second ledger or index.
        """
        self._require_open()
        ledger_digest = content_digest(_raw_integrity_bytes(ledger))
        parameters_digest = content_digest(
            _raw_integrity_bytes(ledger_parameters)
        )
        if self._ledger is None:
            self._ledger = ledger
            self._index = index
            self._ledger_parameters = ledger_parameters
            self._ledger_integrity_digest = ledger_digest
            self._ledger_parameters_integrity_digest = parameters_digest
        elif (
            self._ledger is not ledger
            or self._index is not index
            or self._ledger_integrity_digest != ledger_digest
            or content_digest(_raw_integrity_bytes(self._ledger))
            != self._ledger_integrity_digest
            or self._ledger_parameters_integrity_digest != parameters_digest
            or content_digest(_raw_integrity_bytes(self._ledger_parameters))
            != self._ledger_parameters_integrity_digest
        ):
            msg = "captured_ledger_binding_mismatch"
            raise ValueError(msg)
        return self._ledger, self._index

    def bind_production_inputs(self, production_inputs: object) -> None:
        """Pin the exact production input and its small transport configuration.

        Preconditions:
            ``production_inputs`` has bound this capture's run, author, ledger,
            index, options, and transport.
        Postconditions:
            Stores a weak owner reference and one lowercase 64-character witness.
        Essential behavior:
            Keeps close deterministic without retaining another report tree.
        """
        self._require_open()
        current = (
            self._production_inputs()
            if self._production_inputs is not None
            else None
        )
        if current is not None and current is not production_inputs:
            msg = "captured_production_binding_changed"
            raise ValueError(msg)
        self._production_inputs = weakref_ref(production_inputs)
        witness = _production_binding_witness(production_inputs)
        if self._production_binding_digest is None:
            self._production_binding_digest = witness
        elif self._production_binding_digest != witness:
            msg = "captured_production_binding_changed"
            raise ValueError(msg)

    def bind_coordinator(self, coordinator: object) -> None:
        """Retain one weak local link for coordinator cleanup on close.

        Preconditions:
            ``coordinator`` is the live BatchedNarrator using this capture.
        Postconditions:
            Stores a weak reference or confirms the same coordinator is bound.
        Essential behavior:
            Allows session close to expire preview aliases without a strong cycle.
        """
        self._require_open()
        current = (
            self._coordinator() if self._coordinator is not None else None
        )
        if current is not None and current is not coordinator:
            msg = "captured_coordinator_binding_changed"
            raise ValueError(msg)
        self._coordinator = weakref_ref(coordinator)

    def assert_production_bindings(self, production_inputs: object) -> None:
        """Reject raw changes to the shared run, ledger, author, and source.

        Preconditions:
            ``production_inputs`` is the exact live coordinator bound here.
        Postconditions:
            Returns only when every captured model and current source scope is
            unchanged, including nested frozen-model values.
        Essential behavior:
            Uses the existing ledger index and source guards without rebuilding
            either structure.
        """
        self._require_open()
        owner = (
            self._production_inputs()
            if self._production_inputs is not None
            else None
        )
        if owner is not production_inputs:
            msg = "captured_production_binding_changed"
            raise ValueError(msg)
        ledger = getattr(production_inputs, "ledger", None)
        index = getattr(production_inputs, "index", None)
        run = getattr(production_inputs, "run", None)
        sources = getattr(production_inputs, "sources", None)
        source_binding = getattr(production_inputs, "_source_binding", None)
        author = getattr(production_inputs, "author", None)
        author_digest = getattr(production_inputs, "author_digest", None)
        if (
            ledger is not self._ledger
            or index is not self._index
            or run is not self._run
            or sources is not source_binding
            or author is not self._author
            or author_digest != self._author_digest
            or content_digest(_raw_integrity_bytes(run))
            != self._run_integrity_digest
            or content_digest(_raw_integrity_bytes(self._run))
            != self._run_integrity_digest
            or content_digest(_raw_integrity_bytes(author))
            != self._author_integrity_digest
            or content_digest(_raw_integrity_bytes(self._author))
            != self._author_integrity_digest
            or content_digest(_raw_integrity_bytes(self._ledger_parameters))
            != self._ledger_parameters_integrity_digest
            or content_digest(_raw_integrity_bytes(ledger))
            != self._ledger_integrity_digest
            or _production_binding_witness(production_inputs)
            != self._production_binding_digest
        ):
            msg = "captured_production_binding_changed"
            raise ValueError(msg)
        validate_index = getattr(index, "validated_ledger", None)
        if not callable(validate_index) or validate_index() is not ledger:
            msg = "captured_ledger_binding_changed"
            raise ValueError(msg)
        if sources is not None:
            if self._source_binding is None:
                source_check = getattr(sources, "check_current", None)
                if (
                    type(sources).__module__ != "diffstory.narrative_hierarchy"
                    or type(sources).__name__
                    not in {"PlanningSources", "IndexedPlanningSources"}
                    or not callable(source_check)
                ):
                    msg = "captured_source_binding_changed"
                    raise ValueError(msg)
                self._source_binding = sources
            elif sources is not self._source_binding:
                msg = "captured_source_binding_changed"
                raise ValueError(msg)
            sources.check_current()

    def close(self) -> None:
        """Release source caches and every retained input authority reference.

        Preconditions:
            The capture may be open or already closed.
        Postconditions:
            Report, model, source, coordinator, and registry aliases are empty.
        Essential behavior:
            Is idempotent and clears state held through a retained coordinator.
        """
        if self._closed:
            return
        owner = self._production_inputs() if self._production_inputs else None
        if owner is not None:
            owner._release_capture_data()  # noqa: SLF001 # Release the current input alias when its owning session closes.
        _close_source_caches(self._source_binding)
        coordinator = self._coordinator() if self._coordinator else None
        if coordinator is not None:
            coordinator._release_capture_bindings()  # noqa: SLF001 # Expire the matching preview and coordinator-owned aliases.
        self._report = _OwnedMapping(())
        self._report_digest = ""
        self._report_shape_digest = ""
        self._model_integrity_digest = None
        self._report_type = None
        self._revision_binding = (None, None)
        self._compiler_fact_signature = None
        self._configuration_bytes = b""
        self._configuration_digest = ""
        self._owner_session = None
        self._ledger = None
        self._index = None
        self._ledger_integrity_digest = None
        self._ledger_parameters_integrity_digest = None
        self._ledger_parameters = None
        self._run = None
        self._run_integrity_digest = None
        self._author = None
        self._author_digest = None
        self._author_integrity_digest = None
        self._source_binding = None
        self._production_inputs = None
        self._production_binding_digest = None
        self._coordinator = None
        self._scope_digest = ""
        self._versions_digest = ""
        self._closed = True

    def _require_open(self) -> None:
        if self._closed:
            msg = "session_closed"
            raise ValueError(msg)


@dataclass(frozen=True, slots=True)
class _StageBinding:
    """Incomplete immutable snapshot of one actual production stage input.

    Attributes:
        stage: Exact mutable StageInputs object registered by this session.
        scope_digest: Session scope that owns this data binding.
        stage_name: Stage label from the coordinator's production plan.
        wave_index: Zero-based registration order within ``stage_name``.
        item_keys: Exact ordered request item keys.
        witness: Compact digest of payload, support, registry, config, and bounds.
        evidence_kind: Fixed label; bindings contain no execution evidence.

    This type is private data authority only. It cannot authorize generation,
    validation, application, or producer completion.
    """

    stage: object
    scope_digest: str
    stage_name: str
    wave_index: int
    sequence_index: int
    item_keys: tuple[str, ...]
    witness: str
    evidence_kind: str = "incomplete-stage-binding.v1"


@dataclass(frozen=True, slots=True)
class _StageProofEvidence:
    """Immutable owned references and audit membership for one completed stage.

    Attributes:
        scope_digest: Owning session identity.
        stage: Exact coordinator StageInputs object.
        binding: Exact pre-execution binding from the session registry.
        predecessors: Earlier StageProof tokens in their declared order.
        result_refs: Ordered actual accepted result rows for stage coverage.
        producer_refs: Ordered producer rows built from actual stored artifacts.
        invocation_refs: Original invocation rows covered by the audit.
        request_refs: Every successful request row in producer request order.
        run_refs: Original run rows selected by each producer.
        receipt_refs: Ordered receipts built from actual stored artifacts.
        subject_audits: Planning subject audits completed before the next wave.
        producer_views: Exact accepted compatibility view and binding per producer.
        audit_replay: The codec replay instance that executed required audits.
        replay_memo_witness: Digests of the stage and executing audit replay memos.
        witness: Type-preserving digest pinning every field above.
    """

    scope_digest: str
    stage: object
    binding: _StageBinding
    predecessors: tuple[StageProof, ...]
    result_refs: tuple[object, ...]
    producer_refs: tuple[object, ...]
    invocation_refs: tuple[object, ...]
    request_refs: tuple[object, ...]
    run_refs: tuple[object, ...]
    receipt_refs: tuple[object, ...]
    subject_audits: tuple[object, ...]
    producer_views: tuple[tuple[str, object, str], ...]
    audit_replay: object
    replay_memo_witness: str
    witness: str


def _close_source_caches(sources: object | None) -> None:
    """Close current citation scopes and selected-field memos when present."""
    if sources is None:
        return
    state = getattr(sources, "_source_state", None)
    if state is None:
        return
    field_memo = getattr(state, "field_memo", None)
    if hasattr(field_memo, "clear"):
        field_memo.clear()
    context = getattr(state, "details_context", None)
    scope = getattr(context, "_citation_scope", None)
    if scope is not None and not getattr(scope, "closed", True):
        scope.close()


class ReportSession:
    """Own one report/configuration scope and its completed offline proofs.

    Callers must retain exclusive ownership of mutable inputs while
    ``capture`` snapshots them. Serialized JSON inputs are accepted as an
    immutable alternative. Returned values are detached dictionaries.
    """

    __slots__ = (
        "_arena",
        "_captured_input",
        "_closed",
        "_pending_generation_records",
        "_pending_generation_recordset_witness",
        "_registrations",
        "_scope_digest",
        "_stage_bindings",
    )

    def __init__(
        self,
        captured_input: _CapturedInput,
        *,
        issuer: object = None,
    ) -> None:
        """Store one capture and local proof/stage registries.

        Preconditions:
            ``captured_input`` is a newly created private immutable snapshot.
        Postconditions:
            This session exclusively owns its capture and empty local registries.
        Essential behavior:
            Does not retain a canonical report buffer or parsed report tree copy.
        """
        if issuer is not _ISSUER_TOKEN:
            msg = "Use ReportSession.capture to create a validation scope"
            raise TypeError(msg)
        self._captured_input = captured_input
        captured_input.bind_session(self)
        self._scope_digest = semantic_digest(
            "diffstory.report-session.registration.v2",
            captured_input.scope_identity,
        )
        self._registrations: dict[
            int, tuple[_OpaqueCapability, str, object]
        ] = {}
        self._stage_bindings: dict[int, tuple[object, _StageBinding]] = {}
        self._arena = None
        self._pending_generation_records = None
        self._pending_generation_recordset_witness = None
        self._closed = False

    @classmethod
    def capture(
        cls, report: object, request_configuration: object
    ) -> ReportSession:
        """Snapshot report data and the complete typed request descriptor.

        Preconditions:
            ``report`` is a JSON object, and configuration is a valid request
            descriptor or serialized object.
        Postconditions:
            Returns a session with one detached report snapshot and no ledger
            until production preparation needs it.
        Essential behavior:
            Keeps only streamed report digests and small immutable metadata.
        """
        return cls(
            _CapturedInput(report, request_configuration),
            issuer=_ISSUER_TOKEN,
        )

    @classmethod
    def _from_preview_capture(cls, captured_input: object) -> ReportSession:
        """Attach one checked preview capture directly to a new session.

        Preconditions:
            ``captured_input`` is the current, unattached private capture from
            an approved v2 narrator preview.
        Postconditions:
            Returns one session that owns that exact detached capture.
        Essential behavior:
            Does not snapshot the caller report or allocate a second report
            tree. The capture's owner binding prevents another session from
            adopting it.
        """
        if (
            type(captured_input) is not _CapturedInput
            or captured_input.owner_session is not None
        ):
            msg = "captured_input_session_mismatch"
            raise ValueError(msg)
        return cls(captured_input, issuer=_ISSUER_TOKEN)

    @classmethod
    def import_indexed(
        cls,
        storage_bytes: bytes,
        *,
        compiler_report: object | None = None,
        max_input_bytes: int,
    ) -> tuple[ReportSession, ValidatedGeneration]:
        """Create a new session and independently prove shared storage bytes.

        Preconditions:
            ``storage_bytes`` is exact bytes; ``max_input_bytes`` is a
            nonnegative exact integer; and annotation storage has the matching
            compiler report. Report storage supplies its own complete base.
        Postconditions:
            Returns a new live session and one registered generation token
            after independent structure, production, reader, and application
            checks. No prior session or public graph object supplies authority.
        Essential behavior:
            Uses the parser-to-arena transfer path and retains one data-only M6
            application payload without issuing ``ValidatedReport``.
        """
        from .report_storage import _parse_report_storage_for_import
        from .report_storage import _RecordArena

        if type(max_input_bytes) is not int or max_input_bytes < 0:
            msg = "invalid_input_limit"
            raise ValueError(msg)
        parsed = _parse_report_storage_for_import(
            storage_bytes, max_input_bytes=max_input_bytes
        )
        if parsed.artifact_kind == "report" and compiler_report is not None:
            msg = "compiler_report_not_used_for_report"
            raise ValueError(msg)
        if parsed.artifact_kind == "annotations" and compiler_report is None:
            msg = "compiler_report_required"
            raise ValueError(msg)
        descriptor = RequestConfiguration.model_validate_json(
            canonical_bytes(parsed.root["generation"]["request_configuration"])
        )
        if parsed.artifact_kind == "report":
            captured_input = _CapturedInput._from_parsed_report(  # noqa: SLF001 # Transfer only the exact fresh parser result.
                parsed, descriptor
            )
        else:
            captured_input = _CapturedInput(compiler_report, descriptor)
        session = cls(captured_input, issuer=_ISSUER_TOKEN)
        try:
            session._arena = _RecordArena._from_parsed_storage(  # noqa: SLF001 # Transfer fresh parser-owned rows into the new session arena.
                parsed
            )
            graph = parsed.graph(
                base_report=session._captured_input.report_view
                if parsed.artifact_kind == "report"
                else None
            )
            records = parsed.generation_record_set()
            generation = session._validate_indexed_import(
                parsed, graph, records
            )
        except Exception:
            session.close()
            raise
        return session, generation

    def __enter__(self) -> Self:
        """Return this live session for a context-managed scope."""
        self._require_open()
        return self

    def __exit__(self, _exc_type: object, _exc: object, _tb: object) -> None:
        """Expire handles and retained proofs when the scope ends."""
        self.close()

    def __copy__(self) -> ReportSession:
        """Reject copies that could suggest shared authority."""
        msg = "ReportSession cannot be copied"
        raise TypeError(msg)

    def __deepcopy__(self, _memo: dict) -> ReportSession:
        """Reject copies that could suggest shared authority."""
        msg = "ReportSession cannot be copied"
        raise TypeError(msg)

    def __reduce_ex__(self, _protocol: int) -> object:
        """Reject serialization of a live authority registry."""
        msg = "ReportSession cannot be serialized"
        raise TypeError(msg)

    def close(self) -> None:
        """Expire handles and release stage, cache, and captured authority data."""
        if self._closed:
            return
        for stage, _binding in tuple(self._stage_bindings.values()):
            _close_stage_caches(stage)
        self._stage_bindings.clear()
        self._registrations.clear()
        self._arena = None
        self._pending_generation_records = None
        self._pending_generation_recordset_witness = None
        if self._captured_input is not None:
            self._captured_input.close()
        self._captured_input = None
        self._scope_digest = ""
        self._closed = True

    def _adopt_preview_capture(self, captured_input: object) -> None:
        """Adopt one matching unattached preview capture without recapturing.

        Preconditions:
            This session has no registered proof or stage, and the supplied
            private capture has the same report/configuration as this session.
        Postconditions:
            The exact preview capture is owned by this session.
        Essential behavior:
            Retains the preview's existing detached report, ledger, and index.
        """
        self._require_open()
        if type(captured_input) is not _CapturedInput:
            msg = "invalid_captured_input"
            raise ValueError(msg)
        current = self._captured_input
        if current is captured_input:
            return
        if (
            self._registrations
            or self._stage_bindings
            or current._production_inputs is not None  # noqa: SLF001 # Check whether this capture is already bound to production inputs.
            or captured_input.owner_session is not None
            or current.report_digest != captured_input.report_digest
            or current.shape_digest != captured_input.shape_digest
            or current.configuration_bytes
            != captured_input.configuration_bytes
        ):
            msg = "captured_input_session_mismatch"
            raise ValueError(msg)
        current.close()
        self._captured_input = captured_input
        captured_input.bind_session(self)
        self._scope_digest = semantic_digest(
            "diffstory.report-session.registration.v2",
            captured_input.scope_identity,
        )

    def captured_report(self) -> dict:
        """Return a detached explicit compatibility copy of the report."""
        self._require_open()
        return self._captured_input.expand_report()

    def request_configuration_data(self) -> dict:
        """Return a detached JSON descriptor, never the owned model."""
        self._require_open()
        return _decoded_object(
            self._captured_input.configuration_bytes, "request_configuration"
        )

    def _coordinator_capture(
        self, report: object, request_configuration: object
    ) -> _CapturedInput:
        """Bind one live coordinator to this capture after exact preflight checks.

        Preconditions:
            This is the active session; report and request descriptor are the
            same values that were supplied at capture.
        Postconditions:
            Returns the exact private capture for ProductionInputs.
        Essential behavior:
            Exposes no ledger, index, proof object, or reusable global cache.
        """
        self._require_open()
        try:
            self._captured_input.check_report(report)
            self._captured_input.assert_capture_current()
        except ValueError:
            msg = "dependencies_changed"
            raise ValueError(msg) from None
        configuration = RequestConfiguration.model_validate(
            request_configuration
        )
        if (
            canonical_bytes(configuration.model_dump(mode="json"))
            != self._captured_input.configuration_bytes
        ):
            msg = "dependencies_changed"
            raise ValueError(msg)
        return self._captured_input

    def parse_import(
        self, manifest: object, *, candidate: object | None = None
    ) -> ParsedRecords:
        """Parse detached import records without authorizing their use.

        Preconditions:
            The session is open and its captured inputs remain current.
        Postconditions:
            Returns a session-registered data handle, not a proof capability.
        Essential behavior:
            Checks capture integrity before parsing or registering the handle.
        """
        self._require_open()
        self._captured_input.assert_capture_current()
        parsed = self._parse_import(manifest, candidate)
        return self._register(ParsedRecords, "parsed", parsed)

    def validate_import(
        self,
        manifest: object,
        *,
        candidate: object | None = None,
    ) -> ValidatedGeneration:
        """Run one complete production proof and retain its expected output.

        Preconditions:
            The session and captured inputs are current; import data is complete.
        Postconditions:
            Returns a generation capability only after full production validation.
        Essential behavior:
            Reuses registered parse data when supplied and rechecks before issue.
        """
        self._require_open()
        self._captured_input.assert_capture_current()
        if type(manifest) is ParsedRecords:
            if candidate is not None:
                msg = "candidate_must_be_bound_during_parse"
                raise ValueError(msg)
            parsed = self._registered_payload(
                manifest, ParsedRecords, "parsed"
            )
            if type(parsed) is not _ParsedImport:
                msg = "invalid_session_capability"
                raise ValueError(msg)
        else:
            parsed = self._parse_import(manifest, candidate)
        return self._validate_parsed(parsed)

    def _validate_indexed_import(
        self, parsed: object, graph: object, records: object
    ) -> ValidatedGeneration:
        """Prove parsed rows and retain their data-only application handoff.

        Preconditions:
            ``parsed`` and ``graph`` came from this method's bytes-only parser;
            ``records`` names the same ordered graph closure; and this new
            session owns the compiler report and its configuration.
        Postconditions:
            Registers one complete generation proof only after current source,
            request, producer, receipt, support, reader, and applied-field
            comparisons pass.
        Essential behavior:
            Does not expand a generation, presentation, annotation, or report;
            application preparation does not rerun production proof.
        """
        from .narrative_manifest import validate_indexed_applied_fields
        from .narrative_runtime import RunIdentity
        from .narrative_v2 import validate_indexed_production
        from .report_storage import IndexedReaderProjection
        from .report_storage import ReportRecordGraph
        from .report_storage import _ParsedStorageImport

        self._require_open()
        _require_session(
            type(parsed) is _ParsedStorageImport
            and type(graph) is ReportRecordGraph,
            "invalid_parsed_storage_import",
        )
        arena = self._arena
        _require_session(arena is not None, "invalid_record_arena")
        report = self._captured_input.report_view
        reader, overlay, original_run = validate_indexed_production(
            report,
            records,
            arena=arena,
            captured_input=self._captured_input,
        )
        _require_session(
            type(reader) is IndexedReaderProjection
            and type(overlay) is dict
            and type(original_run) is RunIdentity,
            "invalid_indexed_import_result",
        )
        reader_id = parsed.root["reader"]["$ref"]
        if parsed.artifact_kind == "annotations":
            metadata = parsed.root["metadata"]
            meta = report.get("meta")
            _require_session(
                type(meta) in {dict, _OwnedMapping}
                and metadata.get("base_sha") == meta.get("base_sha")
                and metadata.get("head_sha") == meta.get("head_sha"),
                "dependencies_changed",
            )
            arena.compare_imported_reader(
                reader_id, reader, root_shape="annotations"
            )
            application_reader = IndexedReaderProjection(
                shape="presentation",
                document=reader.document,
                sections=reader.sections,
                context_details=reader.context_details,
                citations=reader.citations,
                support_provenance=reader.support_provenance,
                version="diffstory.presentation.v2",
                provenance="model-generated · unverified",
            )
            arena.drop_imported_reader(reader_id)
            application_graph = arena.derive_imported_report_graph(
                records,
                application_reader,
                base_report=report,
            )
        else:
            application_reader = IndexedReaderProjection(
                shape="presentation",
                document=reader.document,
                sections=reader.sections,
                context_details=reader.context_details,
                citations=reader.citations,
                support_provenance=reader.support_provenance,
                version="diffstory.presentation.v2",
                provenance="model-generated · unverified",
            )
            arena.compare_imported_reader(
                reader_id, application_reader, root_shape="report"
            )
            validate_indexed_applied_fields(report, overlay)
            application_graph = graph

        overlay_value = _freeze_report(overlay)
        application_payload = _IndexedApplicationPayload(
            graph=application_graph,
            records=records,
            reader=application_reader,
            base_report=report,
            overlay=overlay_value,
            original_run_bytes=canonical_bytes(
                original_run.model_dump(mode="json")
            ),
            versions_digest=self._captured_input._versions_digest,  # noqa: SLF001 # Bind current validation and compatibility rules.
            graph_binding=_capture_graph_binding(application_graph),
        )
        proof = _IndexedGenerationProof(
            scope_digest=self._scope_digest,
            report_digest=self._captured_input.report_digest,
            configuration_bytes=self._captured_input.configuration_bytes,
            compiler_fact_signature=self._captured_input.compiler_signature,
            records=records,
            graph=graph,
            reader=reader,
            overlay=overlay_value,
            run_ref=records.run_ref,
            application_payload=application_payload,
            graph_binding=_capture_graph_binding(graph),
            stage_proofs=(),
            versions_digest=self._captured_input._versions_digest,  # noqa: SLF001 # Bind captured validation and compatibility rules.
            witness="",
        )
        proof = _IndexedGenerationProof(
            **{
                field: getattr(proof, field)
                for field in proof.__dataclass_fields__
                if field != "witness"
            },
            witness=_indexed_generation_proof_witness(proof),
        )
        return self._register(ValidatedGeneration, "generation", proof)

    def apply(self, generation: ValidatedGeneration) -> ValidatedReport:
        """Apply the retained expected candidate without rebuilding proof.

        Preconditions:
            ``generation`` is the exact registered capability for this session.
        Postconditions:
            Returns a report capability only while the original capture is current.
        Essential behavior:
            Checks full captured content once before applying the proven
            candidate, then uses private current-scope registry helpers.
        """
        self._require_open()
        self._captured_input.assert_capture_current()
        proof = self._registered_payload_current(
            generation, ValidatedGeneration, "generation"
        )
        if type(proof) is _IndexedGenerationProof:
            application = self._check_indexed_generation_payload(proof)
            for stage_proof in proof.stage_proofs:
                evidence = self._registered_payload_current(
                    stage_proof, StageProof, "stage"
                )
                _require_session(
                    type(evidence) is _StageProofEvidence
                    and evidence.witness == _stage_proof_witness(evidence),
                    "stale_stage_proof",
                )
            _validate_indexed_report_surface(
                application.base_report,
                application.overlay,
                application.reader,
            )
            return self._register_current(ValidatedReport, "report", proof)
        if type(proof) is not _GenerationProof:
            msg = "invalid_session_capability"
            raise ValueError(msg)
        if (
            proof.scope_digest != self._scope_digest
            or proof.report_digest != self._captured_input.report_digest
            or proof.configuration_bytes
            != self._captured_input.configuration_bytes
            or proof.validation_version != _VALIDATION_VERSION
        ):
            msg = "stale_session_capability"
            raise ValueError(msg)

        report = self._captured_input.report_view
        manifest = persisted_manifest(strict_json(proof.manifest_bytes))
        expected_candidate = _decoded_object(
            proof.candidate_bytes, "candidate"
        )
        applied = _apply_v2_candidate(report, manifest, expected_candidate)

        # The imported generation proof and deterministic application above
        # establish the v2 fields. Keep the renderer's independent surface
        # checks, while avoiding a second production reconstruction.
        _validate_report_schema(applied)
        _validate_report_collections(applied)
        _validate_v2_report_surface(applied)

        report_proof = _ReportProof(
            scope_digest=self._scope_digest,
            report_digest=canonical_digest(applied),
            report_value=_freeze_report(applied),
            configuration_bytes=proof.configuration_bytes,
            original_run_bytes=proof.original_run_bytes,
            compiler_fact_signature=proof.compiler_fact_signature,
            manifest_digest=content_digest(proof.manifest_bytes),
            candidate_digest=content_digest(proof.candidate_bytes),
            validation_version=proof.validation_version,
            assembly_version=_ASSEMBLY_VERSION,
            render_surface_version=_RENDER_SURFACE_VERSION,
        )
        return self._register_current(ValidatedReport, "report", report_proof)

    def _checked_report_data(
        self, report: ValidatedReport
    ) -> _IndexedApplicationPayload:
        """Return the checked indexed payload for trusted output adapters.

        Preconditions:
            ``report`` is the exact live indexed report handle issued by this
            session, and captured report/source/configuration bindings remain
            current.
        Postconditions:
            Returns the registered ``_IndexedApplicationPayload`` with its
            graph, record set, presentation reader, captured base, overlay,
            original-run bytes, and version digest unchanged.
        Essential behavior:
            Rechecks capability identity, proof witness, graph root/table
            pins, selectors, and current capture before trusted use. This is a
            data accessor, not an authority source for caller dictionaries.
        """
        proof = self._registered_payload(report, ValidatedReport, "report")
        if type(proof) is not _IndexedGenerationProof:
            msg = "indexed_report_handle_required"
            raise ValueError(msg)
        return self._check_indexed_generation_payload(proof)

    def iter_report_json(
        self,
        report: ValidatedReport,
        *,
        report_storage: str = "expanded",
        max_output_bytes: int,
    ) -> Iterator[bytes]:
        """Return a bounded JSON stream for one registered applied report.

        Preconditions:
            ``report_storage`` is ``expanded`` or ``shared`` and
            ``max_output_bytes`` is an exact nonnegative integer. Iteration
            requires the exact live indexed ``ValidatedReport`` issued here.
        Postconditions:
            The complete stream is emitted only if it fits the byte cap and
            the same capture and graph remain current at the end boundary.
        Essential behavior:
            Streams immutable indexed data directly, applies the private
            overlay to expanded/shared report roots, and checks only registry
            identity between output chunks.
        """
        if type(report_storage) is not str or report_storage not in {
            "expanded",
            "shared",
        }:
            msg = "unknown_report_storage"
            raise ValueError(msg)
        if type(max_output_bytes) is not int or max_output_bytes < 0:
            msg = "invalid_output_limit"
            raise ValueError(msg)
        return self._iter_report_json(report, report_storage, max_output_bytes)

    def _iter_report_json(
        self,
        report: ValidatedReport,
        report_storage: str,
        max_output_bytes: int,
    ) -> Iterator[bytes]:
        """Consume the exact checked application graph as bounded JSON chunks.

        Preconditions:
            The public method accepted the format and exact nonnegative cap.
        Postconditions:
            Returns only complete chunks within the cap and performs a final
            full capture/graph check before iteration ends.
        Essential behavior:
            Checks identity/liveness between chunks without repeating the full
            input, ledger, source, or proof traversal.
        """
        from .report_storage import _iter_bounded_json_chunks
        from .report_storage import _iter_report_output_json

        application = self._checked_report_data(report)
        chunks = _iter_bounded_json_chunks(
            _iter_report_output_json(
                application.graph,
                application.overlay,
                report_storage=report_storage,
            )
        )
        total_bytes = 0
        for chunk in chunks:
            self._registered_payload_current(report, ValidatedReport, "report")
            if total_bytes + len(chunk) > max_output_bytes:
                msg = "report_output_too_large"
                raise ValueError(msg)
            total_bytes += len(chunk)
            yield chunk
        if self._checked_report_data(report) is not application:
            msg = "stale_session_capability"
            raise ValueError(msg)

    def iter_annotations_json(
        self,
        generation: ValidatedGeneration,
        *,
        report_storage: str = "expanded",
        max_output_bytes: int,
    ) -> Iterator[bytes]:
        """Return a bounded annotation JSON stream for a registered generation.

        Preconditions:
            ``report_storage`` is ``expanded`` or ``shared`` and
            ``max_output_bytes`` is an exact nonnegative integer. Iteration
            requires this session's exact live indexed generation handle.
        Postconditions:
            Emits the compatibility annotation candidate or its closed shared
            envelope only while the complete generation proof remains current.
        Essential behavior:
            Reuses the accepted proof graph and derives an annotation root over
            its shared rows when the proof was imported from report storage.
        """
        if type(report_storage) is not str or report_storage not in {
            "expanded",
            "shared",
        }:
            msg = "unknown_report_storage"
            raise ValueError(msg)
        if type(max_output_bytes) is not int or max_output_bytes < 0:
            msg = "invalid_output_limit"
            raise ValueError(msg)
        return self._iter_annotations_json(
            generation, report_storage, max_output_bytes
        )

    def _iter_annotations_json(  # noqa: C901, PLR0912 # Keep root selection and output boundaries together.
        self,
        generation: ValidatedGeneration,
        report_storage: str,
        max_output_bytes: int,
    ) -> Iterator[bytes]:
        """Consume the exact checked generation graph as bounded JSON chunks.

        Preconditions:
            ``generation`` is this session's complete registered indexed
            generation proof and the output options passed their public checks.
        Postconditions:
            Returns only complete chunks within the cap and rechecks the proof
            and capture before iteration ends.
        Essential behavior:
            Derives a root-only annotation view over existing rows when the
            source proof was a report, then streams without full expansion.
        """
        from .report_storage import IndexedReaderProjection
        from .report_storage import ReportRecordGraph
        from .report_storage import _annotation_reader_record
        from .report_storage import _derive_annotation_graph
        from .report_storage import _iter_annotation_output_json
        from .report_storage import _iter_bounded_json_chunks

        proof = self._checked_generation_data(generation)
        graph = proof.graph
        annotation_reader: Mapping[str, object] | None = None
        annotation_metadata: Mapping[str, object] | None = None
        if type(graph) is not ReportRecordGraph:
            msg = "indexed_generation_handle_required"
            raise ValueError(msg)
        if graph.artifact_kind == "annotations":
            annotation_graph = graph
        elif graph.artifact_kind == "report":
            reader = proof.reader
            if type(reader) is not IndexedReaderProjection:
                msg = "invalid_session_capability"
                raise ValueError(msg)
            if reader.shape == "presentation":
                reader = IndexedReaderProjection(
                    shape="annotations",
                    document=reader.document,
                    sections=reader.sections,
                    context_details=reader.context_details,
                    citations=reader.citations,
                    support_provenance=reader.support_provenance,
                )
            if reader.shape != "annotations":
                msg = "invalid_session_capability"
                raise ValueError(msg)
            meta = self._captured_input.report_view.get("meta")
            if not isinstance(meta, Mapping):
                msg = "invalid_report_metadata"
                raise ValueError(msg)
            annotation_metadata = {
                "schema": "diffstory.annotations.v2",
                "base_sha": meta["base_sha"],
                "head_sha": meta["head_sha"],
            }
            if report_storage == "shared":
                annotation_graph = _derive_annotation_graph(
                    graph, reader, metadata=annotation_metadata
                )
            else:
                annotation_reader = _annotation_reader_record(graph, reader)
                annotation_graph = graph
        else:
            msg = "invalid_session_capability"
            raise ValueError(msg)

        chunks = _iter_bounded_json_chunks(
            _iter_annotation_output_json(
                annotation_graph,
                report_storage=report_storage,
                reader_record=annotation_reader,
                metadata=(
                    annotation_metadata
                    if report_storage == "expanded"
                    else None
                ),
            )
        )
        total_bytes = 0
        for chunk in chunks:
            self._registered_payload_current(
                generation, ValidatedGeneration, "generation"
            )
            if total_bytes + len(chunk) > max_output_bytes:
                msg = "annotations_output_too_large"
                raise ValueError(msg)
            total_bytes += len(chunk)
            yield chunk
        if self._checked_generation_data(generation) is not proof:
            msg = "stale_session_capability"
            raise ValueError(msg)

    def _checked_generation_data(
        self, generation: ValidatedGeneration
    ) -> _IndexedGenerationProof:
        """Return an exact current indexed generation proof for output adapters.

        Preconditions:
            ``generation`` is the exact live registered generation capability.
        Postconditions:
            Returns its original indexed proof only after capture, graph,
            coverage, application pins, and every registered stage proof pass.
        Essential behavior:
            Reuses completed proof data without rebuilding a manifest, ledger,
            request audit, source index, or candidate.
        """
        payload = self._registered_payload(
            generation, ValidatedGeneration, "generation"
        )
        if type(payload) is not _IndexedGenerationProof:
            msg = "indexed_generation_handle_required"
            raise ValueError(msg)
        self._check_indexed_generation_payload(payload)
        for stage_proof in payload.stage_proofs:
            evidence = self._registered_payload_current(
                stage_proof, StageProof, "stage"
            )
            _require_session(
                type(evidence) is _StageProofEvidence
                and evidence.witness == _stage_proof_witness(evidence),
                "stale_stage_proof",
            )
        return payload

    def _check_indexed_generation_payload(
        self, proof: _IndexedGenerationProof
    ) -> _IndexedApplicationPayload:
        """Recheck a registered indexed proof at its apply or output boundary.

        Preconditions:
            ``proof`` is the exact private generation payload already resolved
            from this session's identity registry.
        Postconditions:
            Returns only its current application payload with exact graph,
            generation, reader, original-run, capture, configuration, and
            version bindings.
        Essential behavior:
            Reuses immutable completed proof evidence without constructing a
            new manifest, presentation, ledger, or generation graph.
        """
        capture = self._captured_input
        if (
            type(proof) is not _IndexedGenerationProof
            or type(proof.application_payload)
            is not _IndexedApplicationPayload
        ):
            msg = "invalid_session_capability"
            raise ValueError(msg)
        application = proof.application_payload
        try:
            current_witness = _indexed_generation_proof_witness(proof)
        except (AttributeError, TypeError, ValueError):
            msg = "stale_session_capability"
            raise ValueError(msg) from None
        if (
            proof.scope_digest != self._scope_digest
            or proof.report_digest != capture.report_digest
            or proof.configuration_bytes != capture.configuration_bytes
            or proof.compiler_fact_signature != capture.compiler_signature
            or proof.versions_digest != capture._versions_digest  # noqa: SLF001 # Compare the captured validation and compatibility versions.
            or proof.witness != current_witness
            or application.records is not proof.records
            or application.base_report is not capture.report_view
            or application.versions_digest != capture._versions_digest  # noqa: SLF001 # Compare the private application handoff version.
            or proof.run_ref != proof.records.run_ref
            or canonical_bytes(
                proof.records.header.get("request_configuration")
            )
            != proof.configuration_bytes
            or proof.records.header.get("compiler_fact_signature")
            != proof.compiler_fact_signature
            or not _graph_binding_is_current(proof.graph, proof.graph_binding)
            or not _graph_binding_is_current(
                application.graph, application.graph_binding
            )
        ):
            msg = "stale_session_capability"
            raise ValueError(msg)

        from .report_storage import GenerationRecordSet
        from .report_storage import IndexedReaderProjection
        from .report_storage import ReportRecordGraph
        from .report_storage import _check_reference_shape

        if (
            type(proof.graph) is not ReportRecordGraph
            or type(application.graph) is not ReportRecordGraph
            or type(proof.records) is not GenerationRecordSet
            or type(application.reader) is not IndexedReaderProjection
            or type(proof.reader) is not IndexedReaderProjection
        ):
            msg = "invalid_session_capability"
            raise ValueError(msg)
        report_root = application.graph._root  # noqa: SLF001 # Inspect only the application root selectors.
        if (
            report_root.get("shape") != "report"
            or set(report_root)
            != {"shape", "base_report", "generation", "reader"}
            or report_root.get("base_report") is not application.base_report
            or application.graph._artifact_kind != "report"  # noqa: SLF001 # Keep report-root kind exact.
            or not _indexed_graph_matches_records(proof.graph, proof.records)
            or not _indexed_graph_matches_records(
                application.graph, proof.records
            )
            or _indexed_reader_facts(application.graph.reader_projection)
            != _indexed_reader_facts(application.reader)
            or _indexed_reader_facts(proof.graph.reader_projection)
            != _indexed_reader_facts(
                proof.reader
                if proof.graph.artifact_kind == "annotations"
                else application.reader
            )
        ):
            msg = "indexed_graph_binding_changed"
            raise ValueError(msg)
        reader_id, reader_kind = _check_reference_shape(report_root["reader"])
        reader_entry = application.graph._tables["readers"].get(  # noqa: SLF001 # Resolve only the root-selected reader row.
            reader_id
        )
        run_reference = proof.records.run_ref
        run_entry = application.graph._tables["runs"].get(  # noqa: SLF001 # Resolve only the captured original-run row.
            run_reference.storage_id
        )
        if (
            reader_kind != "reader"
            or reader_entry is None
            or reader_entry.get("kind") != "reader"
            or reader_entry["record"].get("shape") != "presentation"
            or run_entry is None
            or run_entry.get("kind") != "run"
            or type(application.original_run_bytes) is not bytes
            or canonical_bytes(_thaw_owned(run_entry["record"]))
            != application.original_run_bytes
        ):
            msg = "indexed_graph_binding_changed"
            raise ValueError(msg)
        return application

    def report_data(self, report: ValidatedReport) -> dict:  # noqa: C901, PLR0915 # Keep the explicit compatibility expansion and selector checks at this boundary.
        """Return detached compatibility report data for a live report handle.

        Preconditions:
            ``report`` is the exact registered report capability for this open
            session.
        Postconditions:
            Returns detached report data; caller mutation cannot change any
            captured base, graph row, or report capability.
        Essential behavior:
            Expands indexed graphs only for this explicit compatibility API and
            restores the existing generation-storage.v1 representation.
        """
        proof = self._registered_payload(report, ValidatedReport, "report")
        if type(proof) is _IndexedGenerationProof:
            application = self._check_indexed_generation_payload(proof)
            from .narrative_manifest import V2Presentation
            from .narrative_manifest import stored_generation
            from .report_storage import _tuple_model_input
            from .report_storage import expand_report_storage

            expanded = expand_report_storage(
                application.graph,
                max_output_bytes=(1 << 63) - 1,
            )
            expanded["generation"] = stored_generation(
                persisted_manifest(expanded["generation"])
            )
            overlay = _thaw_owned(application.overlay)
            expanded["document"] = overlay["document"]
            groups = expanded.get("groups")
            overlay_groups = overlay.get("groups")
            if type(groups) not in {list, tuple} or type(
                overlay_groups
            ) not in {list, tuple}:
                msg = "invalid_indexed_applied_fields"
                raise ValueError(msg)
            groups_by_id: dict[str, dict] = {}
            for group in groups:
                if type(group) is not dict or type(group.get("id")) is not str:
                    msg = "invalid_indexed_applied_fields"
                    raise ValueError(msg)
                group_id = group["id"]
                if group_id in groups_by_id:
                    msg = "invalid_indexed_applied_fields"
                    raise ValueError(msg)
                groups_by_id[group_id] = group

            ordered_groups = []
            seen: set[str] = set()
            for item in overlay_groups:
                if (
                    type(item) is not dict
                    or set(item)
                    != {"group_id", "title", "number", "next_id", "narrative"}
                    or type(item.get("group_id")) is not str
                ):
                    msg = "invalid_indexed_applied_fields"
                    raise ValueError(msg)
                group_id = item["group_id"]
                group = groups_by_id.get(group_id)
                if group is None or group_id in seen:
                    msg = "invalid_indexed_applied_fields"
                    raise ValueError(msg)
                for field in ("title", "number", "next_id", "narrative"):
                    group[field] = item[field]
                seen.add(group_id)
                ordered_groups.append(group)
            if seen != set(groups_by_id):
                msg = "invalid_indexed_applied_fields"
                raise ValueError(msg)
            expanded["groups"] = ordered_groups
            presentation = V2Presentation.model_validate(
                _tuple_model_input(expanded["narrative_v2"])
            )
            expanded["narrative_v2"] = presentation.model_dump(mode="json")
            return expanded
        if type(proof) is not _ReportProof:
            msg = "invalid_session_capability"
            raise ValueError(msg)
        if (
            proof.scope_digest != self._scope_digest
            or proof.configuration_bytes
            != self._captured_input.configuration_bytes
            or proof.report_digest != canonical_digest(proof.report_value)
            or proof.validation_version != _VALIDATION_VERSION
            or proof.assembly_version != _ASSEMBLY_VERSION
            or proof.render_surface_version != _RENDER_SURFACE_VERSION
        ):
            msg = "stale_session_capability"
            raise ValueError(msg)
        return _thaw(proof.report_value)

    def _parse_import(
        self, manifest: object, candidate: object | None
    ) -> _ParsedImport:
        manifest_bytes = _snapshot_object(manifest, "manifest")
        parsed_manifest = persisted_manifest(strict_json(manifest_bytes))
        # Re-encode typed input once. This retains storage representation when
        # supplied as a mapping while avoiding aliases to caller-owned models.
        if isinstance(manifest, BaseModel):
            manifest_bytes = canonical_bytes(
                parsed_manifest.model_dump(mode="json")
            )
        candidate_bytes = (
            None
            if candidate is None
            else _snapshot_object(candidate, "candidate")
        )
        if candidate_bytes is not None:
            _decoded_object(candidate_bytes, "candidate")
        return _ParsedImport(
            scope_digest=self._scope_digest,
            manifest_bytes=manifest_bytes,
            candidate_bytes=candidate_bytes,
        )

    def _validate_parsed(self, parsed: _ParsedImport) -> ValidatedGeneration:
        self._captured_input.assert_capture_current()
        if parsed.scope_digest != self._scope_digest:
            msg = "stale_session_capability"
            raise ValueError(msg)
        manifest = persisted_manifest(strict_json(parsed.manifest_bytes))
        configuration = RequestConfiguration.model_validate_json(
            self._captured_input.configuration_bytes
        )
        if (
            canonical_bytes(
                manifest.request_configuration.model_dump(mode="json")
            )
            != self._captured_input.configuration_bytes
            or manifest.ledger_parameters != configuration.options.ledger
            or manifest.run.provider_destination != configuration.destination
            or manifest.run.model != configuration.model
        ):
            msg = "dependencies_changed"
            raise ValueError(msg)

        report = self._captured_input.report_view
        candidate = (
            None
            if parsed.candidate_bytes is None
            else _decoded_object(parsed.candidate_bytes, "candidate")
        )
        expected = validate_generation(
            report,
            manifest,
            candidate=candidate,
            _captured_input=self._captured_input,
        )
        proof = _GenerationProof(
            scope_digest=self._scope_digest,
            report_digest=self._captured_input.report_digest,
            configuration_bytes=self._captured_input.configuration_bytes,
            manifest_bytes=parsed.manifest_bytes,
            candidate_bytes=canonical_bytes(expected),
            original_run_bytes=canonical_bytes(
                manifest.run.model_dump(mode="json")
            ),
            compiler_fact_signature=manifest.compiler_fact_signature,
            supplied_candidate_digest=(
                None
                if parsed.candidate_bytes is None
                else content_digest(parsed.candidate_bytes)
            ),
            validation_version=_VALIDATION_VERSION,
        )
        return self._register(ValidatedGeneration, "generation", proof)

    def _register(self, cls: type, kind: str, payload: object) -> object:
        """Issue one local handle after rechecking its captured scope.

        Preconditions:
            The session is open and ``cls`` is a supported opaque capability.
        Postconditions:
            Returns a handle registered to this exact session and payload.
        Essential behavior:
            Never registers a value against stale captured data.
        """
        self._require_open()
        self._captured_input.assert_capture_current()
        handle = cls(_ISSUER_TOKEN)
        self._registrations[id(handle)] = (handle, kind, payload)
        return handle

    def _register_current(
        self, cls: type, kind: str, payload: object
    ) -> object:
        """Issue a local handle after the enclosing operation checked capture.

        Preconditions:
            The synchronous enclosing operation just checked captured input,
            and ``payload`` belongs to this live session scope.
        Postconditions:
            Returns a handle registered to this exact session and payload.
        Essential behavior:
            Preserves exact identity, kind, liveness, and scope checks without
            starting another full capture traversal.
        """
        self._require_open()
        if getattr(payload, "scope_digest", None) != self._scope_digest:
            msg = "stale_session_capability"
            raise ValueError(msg)
        handle = cls(_ISSUER_TOKEN)
        self._registrations[id(handle)] = (handle, kind, payload)
        return handle

    def _bind_stage(self, stage: object) -> _StageBinding:
        """Register exact immutable stage input facts without completion proof.

        Preconditions:
            ``stage`` is the live coordinator's StageInputs object and its
            ProductionInputs uses this session's exact captured input.
        Postconditions:
            Returns a frozen binding tagged incomplete and owned by this session.
        Essential behavior:
            Binds actual stage/wave data only; it never accepts result outcome
            dictionaries or issues a generation/report capability.
        """
        self._require_open()
        existing = self._stage_bindings.get(id(stage))
        if existing is not None:
            if existing[0] is not stage:
                msg = "invalid_stage_binding"
                raise ValueError(msg)
            self._check_stage_binding(stage)
            return existing[1]
        stage_name, item_keys, witness = _stage_witness(
            stage, self._captured_input
        )
        wave_index = sum(
            binding.stage_name == stage_name
            for _registered, binding in self._stage_bindings.values()
        )
        binding = _StageBinding(
            stage=stage,
            scope_digest=self._scope_digest,
            stage_name=stage_name,
            wave_index=wave_index,
            sequence_index=len(self._stage_bindings),
            item_keys=item_keys,
            witness=witness,
        )
        self._stage_bindings[id(stage)] = (stage, binding)
        return binding

    def _check_stage_binding(self, stage: object) -> _StageBinding:
        """Require the exact live stage and all captured input facts to match."""
        return self._check_stage_binding_operation(stage)[0]

    def _check_stage_binding_operation(
        self, stage: object
    ) -> tuple[
        _StageBinding,
        object,
        object | None,
        _StageWitnessCache,
        _ContextWitnessCache,
    ]:
        """Check one stage and retain witnesses only for this caller operation.

        Preconditions:
            ``stage`` is the exact registered stage for this open session.
        Postconditions:
            Returns the live binding, its exact checked input/source owners,
            and local identity-pinned witness maps for nested stage aliases.
        Essential behavior:
            Performs one full current-input check before private helpers reuse
            the checked owner within this synchronous traversal.
        """
        self._require_open()
        inputs = _stage_production_inputs(stage, self._captured_input)
        inputs.assert_capture_current()
        checked_sources = inputs.sources
        stage_witnesses: _StageWitnessCache = {}
        context_witnesses: _ContextWitnessCache = {}
        binding = self._check_stage_binding_current(
            stage,
            inputs,
            checked_sources,
            stage_witnesses,
            context_witnesses,
        )
        return (
            binding,
            inputs,
            checked_sources,
            stage_witnesses,
            context_witnesses,
        )

    def _check_stage_binding_current(
        self,
        stage: object,
        checked_inputs: object,
        checked_sources: object | None,
        stage_witnesses: _StageWitnessCache,
        context_witnesses: _ContextWitnessCache,
    ) -> _StageBinding:
        """Check one exact registered stage after its input owner was checked."""
        registration = self._stage_bindings.get(id(stage))
        if registration is None or registration[0] is not stage:
            msg = "invalid_stage_binding"
            raise ValueError(msg)
        binding = registration[1]
        if binding.scope_digest != self._scope_digest:
            msg = "stale_stage_binding"
            raise ValueError(msg)
        stage_name, item_keys, witness = _cached_stage_witness(
            stage,
            self._captured_input,
            checked_inputs,
            checked_sources,
            stage_witnesses,
            context_witnesses,
        )
        if (
            stage_name != binding.stage_name
            or item_keys != binding.item_keys
            or witness != binding.witness
        ):
            msg = "stage_binding_changed"
            raise ValueError(msg)
        return binding

    def _complete_stage(  # noqa: C901, PLR0915 # Keep stage completion and its exact evidence issuance in one validator.
        self,
        stage: object,
        artifacts: tuple[object, ...],
        *,
        predecessors: tuple[StageProof, ...],
    ) -> StageProof:
        """Validate actual successful stage records and issue one local proof.

        Preconditions:
            ``stage`` is registered by this session's coordinator; artifacts
            are the exact ordered successful runtime records; and predecessors
            are the completed StageProof tokens required by the fixed traversal.
        Postconditions:
            One opaque proof binds current result, producer, invocation,
            request, run, receipt, subject-audit, replay, and stage membership.
        Essential behavior:
            Strictly captures actual rows, executes the shared producer/result
            audit, and retains only owned references plus compact audits.
        """
        from .narrative_hierarchy import capture_indexed_consumed_prose
        from .narrative_manifest import build_receipts
        from .narrative_manifest import validate_prose_support
        from .narrative_runtime import StoredArtifact
        from .narrative_v2 import CodecReplay
        from .narrative_v2 import _audit_stage_records
        from .narrative_v2 import result_key

        self._require_open()
        (
            binding,
            inputs,
            checked_sources,
            stage_witnesses,
            context_witnesses,
        ) = self._check_stage_binding_operation(stage)
        _require_session(
            getattr(inputs, "_captured_input", None) is self._captured_input,
            "captured_input_mismatch",
        )
        coordinator_ref = self._captured_input._coordinator  # noqa: SLF001 # Completion uses the capture's exact bound coordinator.
        coordinator = (
            coordinator_ref() if coordinator_ref is not None else None
        )
        _require_session(
            coordinator is not None, "captured_coordinator_binding_changed"
        )
        coordinator.check_approval(coordinator.approved_report)
        _require_session(type(artifacts) is tuple, "invalid_stage_artifacts")
        _require_session(
            all(type(item) is StoredArtifact for item in artifacts),
            "invalid_stage_artifacts",
        )
        item_keys = tuple(item.lookup.item_key for item in artifacts)
        _require_session(item_keys == tuple(stage.items), "coverage")
        prior = self._completed_stage_entries()
        _require_session(
            len(prior) == binding.sequence_index,
            "incomplete_stage_predecessor",
        )
        expected_predecessors = self._expected_stage_predecessors(
            binding, inputs, prior
        )
        _require_session(
            type(predecessors) is tuple
            and len(predecessors) == len(expected_predecessors)
            and all(
                actual is expected
                for actual, expected in zip(
                    predecessors, expected_predecessors, strict=True
                )
            ),
            "stage_predecessor_mismatch",
        )
        for proof in predecessors:
            self._check_stage_proof(
                proof,
                checked_inputs=inputs,
                checked_sources=checked_sources,
                stage_witnesses=stage_witnesses,
                context_witnesses=context_witnesses,
            )

        from .report_storage import _RecordArena

        if self._arena is None:
            self._arena = _RecordArena()
        arena = self._arena
        result_refs = tuple(
            arena.put_result(item.record.result) for item in artifacts
        )
        _require_session(
            tuple(reference.item_key for reference in result_refs)
            == item_keys,
            "coverage",
        )
        for item, reference in zip(artifacts, result_refs, strict=True):
            _require_session(
                result_key(item.record.result) == reference.item_key,
                "coverage",
            )

        if binding.stage_name == "detail":
            _require_session(
                inputs._indexed_details_context is None,  # noqa: SLF001 # The one detail context is bound only at successful completion.
                "detail_context_already_bound",
            )
            inputs.bind_indexed_details(arena, result_refs)
            # Detail completion creates the indexed source owner. Check that
            # newly derived owner before any compatibility view uses it.
            inputs.assert_capture_current()
            checked_sources = inputs.sources
            stage_witnesses = {}
            context_witnesses = {}
            _require_session(
                self._check_stage_binding_current(
                    stage,
                    inputs,
                    checked_sources,
                    stage_witnesses,
                    context_witnesses,
                )
                is binding,
                "stage_binding_changed",
            )
            output_selector = None
        else:
            _require_session(
                inputs._indexed_details_context is not None  # noqa: SLF001 # All prose stages depend on the accepted detail rows.
                and inputs.sources is not None,
                "detail_context_required",
            )
            output_selector = capture_indexed_consumed_prose(
                arena._snapshot_reader(),  # noqa: SLF001 # Select actual owned stage outputs once.
                result_refs,
                inputs.sources,
            )

        producers, receipts = build_receipts(
            artifacts,
            coordinator.original_run,
            frozenset(coordinator.reused),
        )
        audit_replay = CodecReplay(stage)
        refs_by_key = {
            reference.item_key: reference for reference in result_refs
        }

        def validate_owned_result(item_key: str) -> None:
            """Run current checks against the exact arena-owned result row."""
            reference = refs_by_key.get(item_key)
            _require_session(reference is not None, "coverage")
            if binding.stage_name == "detail":
                _require_session(reference.kind == "detail", "coverage")
                return
            result = output_selector.select(item_key)
            context = stage.contexts.get(item_key)
            _require_session(context is not None, "dependencies_changed")
            validate_prose_support(result, context, inputs.wire)
            if binding.stage_name == "planning":
                _require_session(
                    len(canonical_bytes(inputs.forwarded_support(result)))
                    <= context.max_support_bytes,
                    "schema",
                )

        subject_audits, selected_views = _audit_stage_records(
            stage,
            item_keys,
            producers,
            receipts,
            audit_replay,
            validate_result=validate_owned_result,
        )
        receipt_by_item = {receipt.item_key: receipt for receipt in receipts}
        producer_views = []
        for producer_key, view in selected_views:
            view_name, view_item_keys, view_witness = _cached_stage_witness(
                view,
                self._captured_input,
                inputs,
                checked_sources,
                stage_witnesses,
                context_witnesses,
            )
            _require_session(
                view_name == binding.stage_name
                and view_item_keys == item_keys,
                "dependencies_changed",
            )
            for item in artifacts:
                item_key = result_key(item.record.result)
                if receipt_by_item[item_key].producer_key == producer_key:
                    _require_session(
                        item.lookup == view.lookups[item_key],
                        "dependencies_changed",
                    )
            producer_views.append((producer_key, view, view_witness))

        producer_refs = tuple(
            arena.put_record("producer", producer) for producer in producers
        )
        receipt_refs = tuple(
            arena.put_record("receipt", receipt) for receipt in receipts
        )
        invocation_refs = []
        request_refs = []
        run_refs = []
        for producer, producer_ref in zip(
            producers, producer_refs, strict=True
        ):
            producer_key, run_ref, invocation_ref, ordered_request_refs = (
                arena.producer_audit_references(producer_ref)
            )
            _require_session(producer_key == producer.key, "producer_coverage")
            _require_session(
                arena.selected_audit_record(run_ref, kind="run")
                == producer.original_run.model_dump(mode="json"),
                "producer_run_mismatch",
            )
            _require_session(
                arena.selected_audit_record(invocation_ref, kind="invocation")
                == producer.invocation.model_dump(mode="python"),
                "request_mismatch",
            )
            _require_session(
                len(ordered_request_refs) == len(producer.requests),
                "request_mismatch",
            )
            for request_ref, request in zip(
                ordered_request_refs, producer.requests, strict=True
            ):
                _require_session(
                    arena.selected_audit_record(request_ref, kind="request")
                    == request.model_dump(mode="python"),
                    "request_mismatch",
                )
            run_refs.append(run_ref)
            invocation_refs.append(invocation_ref)
            request_refs.extend(ordered_request_refs)

        replay_memo_witness = semantic_digest(
            "diffstory.session.stage-replay-memos.v1",
            _replay_memo_witness(stage),
            _codec_replay_memo_witness(audit_replay),
        )
        evidence = _StageProofEvidence(
            scope_digest=self._scope_digest,
            stage=stage,
            binding=binding,
            predecessors=predecessors,
            result_refs=result_refs,
            producer_refs=producer_refs,
            invocation_refs=tuple(invocation_refs),
            request_refs=tuple(request_refs),
            run_refs=tuple(run_refs),
            receipt_refs=receipt_refs,
            subject_audits=subject_audits,
            producer_views=tuple(producer_views),
            audit_replay=audit_replay,
            replay_memo_witness=replay_memo_witness,
            witness="",
        )
        evidence = _StageProofEvidence(
            **{
                field: getattr(evidence, field)
                for field in evidence.__dataclass_fields__
                if field != "witness"
            },
            witness=_stage_proof_witness(evidence),
        )
        return self._register(StageProof, "stage", evidence)

    def _completed_stage_entries(
        self,
    ) -> tuple[tuple[StageProof, _StageProofEvidence], ...]:
        """Return registered stage proofs in their exact coordinator order."""
        entries = []
        for handle, kind, payload in self._registrations.values():
            if kind == "stage":
                _require_session(
                    type(handle) is StageProof
                    and type(payload) is _StageProofEvidence,
                    "invalid_session_capability",
                )
                entries.append((handle, payload))
        entries.sort(key=lambda item: item[1].binding.sequence_index)
        return tuple(entries)

    def _expected_stage_predecessors(
        self,
        binding: _StageBinding,
        inputs: object,
        prior: tuple[tuple[StageProof, _StageProofEvidence], ...],
    ) -> tuple[StageProof, ...]:
        """Derive required predecessors from the fixed production traversal."""
        from .narrative_hierarchy import planning_waves

        by_name: dict[str, list[tuple[StageProof, _StageProofEvidence]]] = {
            "detail": [],
            "planning": [],
            "section": [],
            "document": [],
        }
        for entry in prior:
            by_name[entry[1].binding.stage_name].append(entry)
        _require_session(
            all(
                len(by_name[name]) <= 1
                for name in ("detail", "section", "document")
            ),
            "stage_order_mismatch",
        )
        planning = by_name["planning"]
        expected_waves = (
            len(planning_waves(inputs.plan)) if inputs.plan is not None else 0
        )
        if binding.stage_name == "detail":
            _require_session(
                not prior and binding.wave_index == 0, "stage_order_mismatch"
            )
            return ()
        _require_session(
            len(by_name["detail"]) == 1, "stage_predecessor_mismatch"
        )
        detail_token = by_name["detail"][0][0]
        if binding.stage_name == "planning":
            _require_session(
                not by_name["section"]
                and not by_name["document"]
                and binding.wave_index == len(planning)
                and binding.wave_index < expected_waves,
                "stage_order_mismatch",
            )
            return (detail_token, *(token for token, _ in planning))
        _require_session(
            len(planning) == expected_waves,
            "incomplete_planning_waves",
        )
        planning_tokens = tuple(token for token, _ in planning)
        if binding.stage_name == "section":
            _require_session(
                not by_name["section"]
                and not by_name["document"]
                and binding.wave_index == 0,
                "stage_order_mismatch",
            )
            return (detail_token, *planning_tokens)
        _require_session(
            binding.stage_name == "document"
            and len(by_name["section"]) == 1
            and not by_name["document"]
            and binding.wave_index == 0,
            "stage_predecessor_mismatch",
        )
        return (
            detail_token,
            *planning_tokens,
            by_name["section"][0][0],
        )

    def _check_stage_proof(
        self,
        proof: StageProof,
        *,
        checked_inputs: object | None = None,
        checked_sources: object | None = None,
        stage_witnesses: _StageWitnessCache | None = None,
        context_witnesses: _ContextWitnessCache | None = None,
    ) -> _StageProofEvidence:
        """Recheck exact proof identity, binding, replay memo, and row ownership.

        Preconditions:
            Direct calls omit operation owners and receive a full current-input
            check. An enclosing ordered traversal may pass its exact checked
            input/source owners and stack-local witness maps.
        Postconditions:
            Returns only the exact registered proof while all row and replay
            bindings remain current.
        Essential behavior:
            Reuses source checks only within one synchronous operation and
            keeps standalone calls fully guarded.
        """
        self._require_open()
        if checked_inputs is None:
            if (
                checked_sources is not None
                or stage_witnesses is not None
                or context_witnesses is not None
            ):
                msg = "captured_input_mismatch"
                raise ValueError(msg)
            evidence = self._registered_payload_current(
                proof, StageProof, "stage"
            )
            _require_session(
                type(evidence) is _StageProofEvidence,
                "invalid_session_capability",
            )
            inputs = _stage_production_inputs(
                evidence.stage, self._captured_input
            )
            inputs.assert_capture_current()
            checked_sources = inputs.sources
            stage_witnesses = {}
            context_witnesses = {}
        else:
            _require_session(
                stage_witnesses is not None and context_witnesses is not None,
                "invalid_stage_binding",
            )
            inputs = checked_inputs
        return self._check_stage_proof_current(
            proof,
            inputs,
            checked_sources,
            stage_witnesses,
            context_witnesses,
        )

    def _check_stage_proof_current(
        self,
        proof: StageProof,
        checked_inputs: object,
        checked_sources: object | None,
        stage_witnesses: _StageWitnessCache,
        context_witnesses: _ContextWitnessCache,
    ) -> _StageProofEvidence:
        """Check a proof using the exact source owner checked by its operation."""
        from .report_storage import _RecordArena

        evidence = self._registered_payload_current(proof, StageProof, "stage")
        _require_session(
            type(evidence) is _StageProofEvidence, "invalid_session_capability"
        )
        _require_session(
            evidence.witness == _stage_proof_witness(evidence)
            and evidence.scope_digest == self._scope_digest,
            "stale_stage_proof",
        )
        _require_session(
            self._check_stage_binding_current(
                evidence.stage,
                checked_inputs,
                checked_sources,
                stage_witnesses,
                context_witnesses,
            )
            is evidence.binding,
            "stale_stage_proof",
        )
        _require_session(
            evidence.audit_replay.stage is evidence.stage
            and semantic_digest(
                "diffstory.session.stage-replay-memos.v1",
                _replay_memo_witness(evidence.stage),
                _codec_replay_memo_witness(evidence.audit_replay),
            )
            == evidence.replay_memo_witness,
            "codec_replay_changed",
        )
        _require_session(
            type(self._arena) is _RecordArena, "missing_record_arena"
        )
        arena = self._arena
        _require_session(
            all(
                arena._owned_result_reference(reference, kind=reference.kind)  # noqa: SLF001 # Recheck exact arena row membership.
                for reference in evidence.result_refs
            ),
            "foreign_storage_result_reference",
        )
        _require_session(
            len(evidence.producer_refs)
            == len(evidence.invocation_refs)
            == len(evidence.run_refs)
            == len({id(item) for item in evidence.producer_refs}),
            "producer_coverage",
        )
        expected_invocations = []
        expected_requests = []
        expected_runs = []
        for reference in evidence.producer_refs:
            _producer_key, run_ref, invocation_ref, request_refs = (
                arena.producer_audit_references(reference)
            )
            expected_runs.append(run_ref)
            expected_invocations.append(invocation_ref)
            expected_requests.extend(request_refs)
        _require_session(
            tuple(expected_runs) == evidence.run_refs
            and tuple(expected_invocations) == evidence.invocation_refs
            and tuple(expected_requests) == evidence.request_refs,
            "stage_audit_membership_changed",
        )
        selected_view_map = {}
        for producer_key, view, witness in evidence.producer_views:
            checked_view = _cached_stage_witness(
                view,
                self._captured_input,
                checked_inputs,
                checked_sources,
                stage_witnesses,
                context_witnesses,
            )
            prior = selected_view_map.get(producer_key)
            _require_session(
                prior is None or (prior[0] is view and prior[1] == witness),
                "stage_compatibility_view_changed",
            )
            selected_view_map[producer_key] = (view, witness, checked_view)
        audit_views = (
            *evidence.audit_replay.views,
            *evidence.audit_replay.legacy_alternatives,
        )
        _require_session(
            len(selected_view_map) == len(evidence.producer_refs)
            and all(
                any(view is candidate for candidate in audit_views)
                and checked_view[2] == witness
                for view, witness, checked_view in selected_view_map.values()
            ),
            "stage_compatibility_view_changed",
        )
        for reference in evidence.producer_refs:
            producer_key, _run_ref, _invocation_ref, _request_refs = (
                arena.producer_audit_references(reference)
            )
            _require_session(
                producer_key in selected_view_map,
                "stage_compatibility_view_changed",
            )
        for reference in (
            *evidence.invocation_refs,
            *evidence.request_refs,
            *evidence.run_refs,
        ):
            _require_session(
                arena.selected_audit_record(
                    reference,
                    kind=reference.kind,
                )
                is not None,
                "stage_audit_membership_changed",
            )
        _require_session(
            all(
                arena._owned_record_reference(  # noqa: SLF001 # Recheck each exact receipt row without expansion.
                    reference, table="receipts", kind="receipt"
                )
                for reference in evidence.receipt_refs
            ),
            "stage_audit_membership_changed",
        )
        return evidence

    def _ordered_stage_proofs(
        self, stage_proofs: tuple[StageProof, ...]
    ) -> tuple[tuple[StageProof, _StageProofEvidence], ...]:
        """Require every registered completed stage in fixed traversal order."""
        from .narrative_hierarchy import planning_waves

        _require_session(type(stage_proofs) is tuple, "invalid_stage_proofs")
        registered = self._completed_stage_entries()
        _require_session(
            len(stage_proofs) == len(registered)
            and all(
                proof is entry[0]
                for proof, entry in zip(stage_proofs, registered, strict=True)
            ),
            "stage_proof_coverage",
        )
        bindings = sorted(
            (binding for _stage, binding in self._stage_bindings.values()),
            key=lambda item: item.sequence_index,
        )
        _require_session(
            len(bindings) == len(registered), "incomplete_stage_proof"
        )
        _require_session(bool(registered), "stage_proof_coverage")
        inputs = _stage_production_inputs(
            registered[0][1].stage, self._captured_input
        )
        _require_session(
            all(entry[1].stage.inputs is inputs for entry in registered),
            "captured_input_mismatch",
        )
        inputs.assert_capture_current()
        checked_sources = inputs.sources
        stage_witnesses: _StageWitnessCache = {}
        context_witnesses: _ContextWitnessCache = {}
        entries = []
        for position, (proof, registered_entry, binding) in enumerate(
            zip(stage_proofs, registered, bindings, strict=True)
        ):
            evidence = self._check_stage_proof(
                proof,
                checked_inputs=inputs,
                checked_sources=checked_sources,
                stage_witnesses=stage_witnesses,
                context_witnesses=context_witnesses,
            )
            _require_session(
                evidence is registered_entry[1]
                and evidence.binding is binding
                and binding.sequence_index == position,
                "stage_order_mismatch",
            )
            entries.append((proof, evidence))
        _require_session(bool(entries), "stage_proof_coverage")
        detail = entries[0][1]
        inputs = getattr(detail.stage, "inputs", None)
        _require_session(
            detail.binding.stage_name == "detail"
            and detail.binding.wave_index == 0
            and inputs is not None
            and inputs.plan is not None
            and inputs._indexed_details_context is not None,  # noqa: SLF001 # Fresh production must use the arena-backed detail context.
            "stage_order_mismatch",
        )
        expected_waves = len(planning_waves(inputs.plan))
        expected_names = (
            "detail",
            *("planning" for _ in range(expected_waves)),
            "section",
            "document",
        )
        _require_session(
            tuple(item[1].binding.stage_name for item in entries)
            == expected_names,
            "stage_proof_coverage",
        )
        for position, (proof, evidence) in enumerate(entries):
            prior = tuple(entries[:position])
            expected_predecessors = self._expected_stage_predecessors(
                evidence.binding, inputs, prior
            )
            _require_session(
                len(evidence.predecessors) == len(expected_predecessors)
                and all(
                    actual is expected
                    for actual, expected in zip(
                        evidence.predecessors,
                        expected_predecessors,
                        strict=True,
                    )
                ),
                "stage_predecessor_mismatch",
            )
            _require_session(
                proof is stage_proofs[position], "stage_order_mismatch"
            )
        return tuple(entries)

    def _recordset_for_production(  # noqa: PLR0915 # Assemble only the fixed ordered references from completed stages.
        self, stage_proofs: tuple[StageProof, ...]
    ) -> object:
        """Build the data-only generation record set from completed stage rows.

        Preconditions:
            ``stage_proofs`` contains every exact registered stage token in
            coordinator order; the owning rows are already session-owned.
        Postconditions:
            Returns one immutable record set assembled only from session-owned
            result and audit references.
        Essential behavior:
            Projects data and saved bindings only; ``finish_production`` does
            the complete ordered proof traversal before issuing authority.
        """
        from .narrative_contracts import StoryPlan
        from .narrative_hierarchy import planning_waves
        from .narrative_hierarchy import validate_production_requests
        from .narrative_manifest import CurrentUsage
        from .report_storage import GenerationRecordSet

        self._require_open()
        _require_session(type(stage_proofs) is tuple, "invalid_stage_proofs")
        entries = self._completed_stage_entries()
        _require_session(
            len(stage_proofs) == len(entries)
            and all(
                proof is entry[0]
                for proof, entry in zip(stage_proofs, entries, strict=True)
            ),
            "stage_proof_coverage",
        )
        _require_session(
            len(self._stage_bindings) == len(entries),
            "incomplete_stage_proof",
        )
        _require_session(bool(entries), "stage_proof_coverage")
        if self._pending_generation_records is not None:
            pending_witness = self._pending_generation_recordset_witness
            _require_session(
                pending_witness is not None
                and _generation_recordset_witness(
                    self._pending_generation_records
                )
                == pending_witness,
                "production_recordset_changed",
            )
            return self._pending_generation_records
        proofs = tuple(evidence for _proof, evidence in entries)
        inputs = proofs[0].stage.inputs
        coordinator_ref = self._captured_input._coordinator  # noqa: SLF001 # Use the exact bound usage/run owner.
        coordinator = (
            coordinator_ref() if coordinator_ref is not None else None
        )
        _require_session(
            coordinator is not None, "captured_coordinator_binding_changed"
        )
        plan = inputs.plan
        _require_session(
            type(plan) is StoryPlan and plan.requests == (),
            "production_plan_changed",
        )
        detail_evidence = proofs[0]
        detail_refs = detail_evidence.result_refs
        _require_session(
            tuple(detail_refs) == inputs._detail_refs  # noqa: SLF001 # Reuse the exact validated detail order.
            and tuple(ref.item_key for ref in detail_refs)
            == tuple(inputs.index.units),
            "detail_coverage",
        )
        planning_evidence = tuple(
            evidence
            for evidence in proofs
            if evidence.binding.stage_name == "planning"
        )
        planning_refs = tuple(
            reference
            for evidence in planning_evidence
            for reference in evidence.result_refs
        )
        section_evidence = next(
            evidence
            for evidence in proofs
            if evidence.binding.stage_name == "section"
        )
        document_evidence = next(
            evidence
            for evidence in proofs
            if evidence.binding.stage_name == "document"
        )
        section_refs = section_evidence.result_refs
        document_refs = document_evidence.result_refs
        _require_session(len(document_refs) == 1, "document_coverage")

        arena = self._arena
        reader = arena._snapshot_reader()  # noqa: SLF001 # Read selected binding/order facts without expanding models.
        expected_planning_subjects = tuple(
            key for wave in planning_waves(plan) for key in wave
        )
        planning_facts = tuple(
            reader.result_binding_facts(reference)
            for reference in planning_refs
        )
        _require_session(
            len(planning_refs) == len(expected_planning_subjects)
            and all(
                kind == "planning_card"
                for kind, _subject, _plan in planning_facts
            )
            and tuple(subject for _kind, subject, _plan in planning_facts)
            == expected_planning_subjects
            and all(
                plan_id == plan.id
                for _kind, _subject, plan_id in planning_facts
            ),
            "planning_coverage",
        )
        expected_section_subjects = tuple(
            view.group_key for view in plan.production.section_views
        )
        section_facts = tuple(
            reader.result_binding_facts(reference)
            for reference in section_refs
        )
        _require_session(
            len(section_refs) == len(expected_section_subjects)
            and all(
                kind == "section" for kind, _subject, _plan in section_facts
            )
            and tuple(subject for _kind, subject, _plan in section_facts)
            == expected_section_subjects
            and all(
                plan_id == plan.id
                for _kind, _subject, plan_id in section_facts
            ),
            "section_coverage",
        )
        document_facts = tuple(
            reader.result_binding_facts(reference)
            for reference in document_refs
        )
        _require_session(
            document_facts == (("document", plan.id, plan.id),),
            "document_coverage",
        )

        subject_audits = tuple(
            audit
            for evidence in planning_evidence
            for audit in evidence.subject_audits
        )
        audit_ids = tuple(audit.id for audit in subject_audits)
        _require_session(
            len(audit_ids) == len(set(audit_ids)),
            "duplicate_planning_request",
        )
        completed_plan = StoryPlan.model_validate(
            plan.model_dump(mode="python") | {"requests": subject_audits}
        )
        _require_session(
            completed_plan.model_dump(exclude={"requests"})
            == plan.model_dump(exclude={"requests"}),
            "production_plan_changed",
        )
        validate_production_requests(completed_plan)

        producer_refs = tuple(
            reference
            for evidence in proofs
            for reference in evidence.producer_refs
        )
        receipt_refs = tuple(
            reference
            for evidence in proofs
            for reference in evidence.receipt_refs
        )
        _require_session(
            len({(ref.kind, ref.storage_id) for ref in producer_refs})
            == len(producer_refs)
            and len({(ref.kind, ref.storage_id) for ref in receipt_refs})
            == len(receipt_refs),
            "producer_coverage",
        )
        current_run_json = inputs.run.model_dump(mode="json")
        receipt_views = tuple(arena.receipt_view(ref) for ref in receipt_refs)
        for producer_ref in producer_refs:
            producer_key, original_run_ref, _invocation_ref, _request_refs = (
                arena.producer_audit_references(producer_ref)
            )
            producer_receipts = tuple(
                receipt
                for receipt in receipt_views
                if receipt.get("producer_key") == producer_key
            )
            _require_session(bool(producer_receipts), "producer_coverage")
            dispositions = {
                receipt.get("disposition") for receipt in producer_receipts
            }
            _require_session(
                dispositions <= {"sent", "reused"}, "producer_coverage"
            )
            if "sent" in dispositions:
                _require_session(
                    canonical_bytes(
                        arena.selected_audit_record(
                            original_run_ref, kind="run"
                        )
                    )
                    == canonical_bytes(current_run_json),
                    "producer_run_mismatch",
                )

        run_ref = arena.put_record("run", inputs.run)
        plan_ref = arena.put_record("story_plan", completed_plan)
        compiler_signature = self._captured_input.compiler_signature
        _require_session(
            compiler_signature is not None, "compiler_facts_changed"
        )
        header = {
            "version": "diffstory.generation.v2",
            "provenance": "unverified",
            "compiler_fact_signature": compiler_signature,
            "current_usage": CurrentUsage(
                **coordinator.usage.report()
            ).model_dump(mode="json"),
            "ledger_digest": inputs.ledger.id,
            "ledger_parameters": inputs.options.ledger.model_dump(mode="json"),
            "request_configuration": inputs.transport.descriptor.model_dump(
                mode="json"
            ),
            "unit_keys": list(inputs.index.units),
        }
        records = GenerationRecordSet(
            header,
            run_ref=run_ref,
            details=detail_refs,
            plan_ref=plan_ref,
            planning=planning_refs,
            sections=section_refs,
            document=document_refs[0],
            producers=producer_refs,
            receipts=receipt_refs,
        )
        self._pending_generation_records = records
        self._pending_generation_recordset_witness = (
            _generation_recordset_witness(records)
        )
        return records

    def finish_production(
        self,
        records: object,
        stage_proofs: tuple[StageProof, ...],
    ) -> ValidatedGeneration:
        """Seal fresh indexed production after its exact stages are complete.

        Preconditions:
            ``records`` is the session's record set and ``stage_proofs`` are
            every exact completed StageProof in fixed order.
        Postconditions:
            Returns a session-registered generation token over the sealed
            indexed graph and checked deterministic application overlay.
        Essential behavior:
            Reuses completed stage evidence, runs the indexed support forest
            once, and performs no offline production or ledger reconstruction.
        """
        from .narrative_hierarchy import capture_indexed_consumed_prose
        from .narrative_hierarchy import validate_support_forest
        from .narrative_manifest import CurrentUsage
        from .narrative_manifest import assemble_indexed_reader
        from .narrative_manifest import indexed_applied_fields
        from .report_storage import GenerationRecordSet
        from .report_storage import IndexedReaderProjection
        from .report_storage import _derive_report_graph

        self._require_open()
        expected = self._recordset_for_production(stage_proofs)
        pending_witness = self._pending_generation_recordset_witness
        _require_session(
            type(records) is GenerationRecordSet
            and records is expected
            and pending_witness is not None
            and _generation_recordset_witness(records) == pending_witness,
            "production_recordset_changed",
        )
        entries = self._ordered_stage_proofs(stage_proofs)
        inputs = entries[0][1].stage.inputs
        coordinator_ref = self._captured_input._coordinator  # noqa: SLF001 # Usage and approval remain bound to this coordinator.
        coordinator = (
            coordinator_ref() if coordinator_ref is not None else None
        )
        _require_session(
            coordinator is not None, "captured_coordinator_binding_changed"
        )
        # The ordered traversal above checked capture, ledger, index, source,
        # run, configuration, and every registered stage exactly once.
        coordinator.check_approval(coordinator.approved_report)
        _require_session(
            content_digest(_raw_integrity_bytes(inputs.plan))
            == inputs._base_plan_integrity_digest,  # noqa: SLF001 # The request plan cannot drift after stage completion.
            "production_plan_changed",
        )
        scope_check = getattr(inputs.citation_scope, "_check_current", None)
        _require_session(callable(scope_check), "citation_scope_closed")
        scope_check(
            inputs.ledger,
            inputs.index,
            self._captured_input._revision_binding,  # noqa: SLF001 # Same explicit pair used before detail checks.
        )
        _require_session(
            self._arena._owned_record_reference(  # noqa: SLF001 # Exact recorded run reference remains session-owned.
                records.run_ref, table="runs", kind="run"
            )["$ref"]
            == records.run_ref.storage_id
            and records.header.get("current_usage")
            == CurrentUsage(**coordinator.usage.report()).model_dump(
                mode="json"
            ),
            "production_usage_changed",
        )

        prose_refs = (*records.planning, *records.sections, records.document)
        prose_selector = capture_indexed_consumed_prose(
            self._arena._snapshot_reader(),  # noqa: SLF001 # One final selector spans only owned prose references.
            tuple(prose_refs),
            inputs.sources,
        )
        prose_results = tuple(
            prose_selector.select(reference.item_key)
            for reference in prose_refs
        )
        validate_support_forest(prose_results, inputs.sources, prose_selector)

        report = self._captured_input.report_view
        groups_value = report.get("groups")
        _require_session(
            type(groups_value).__name__ in {"_OwnedList", "_OwnedTuple"},
            "invalid_compiler_group_facts",
        )
        groups = {}
        for group in groups_value:
            group_id = group.get("id")
            _require_session(
                type(group_id) is str and group_id not in groups,
                "invalid_compiler_group_facts",
            )
            groups[group_id] = group
        reader = assemble_indexed_reader(
            report,
            records,
            arena=self._arena,
            details=inputs._indexed_details_context,  # noqa: SLF001 # Exact arena-backed detail context.
            sources=inputs.sources,
            captured_input=self._captured_input,
        )
        overlay = indexed_applied_fields(
            reader,
            arena=self._arena,
            details=inputs._indexed_details_context,  # noqa: SLF001 # Keep the same current detail owner.
            sources=inputs.sources,
            groups=groups,
            captured_input=self._captured_input,
        )
        metadata = report["meta"]
        graph = self._arena.seal(
            records,
            reader,
            {
                "shape": "annotations",
                "metadata": {
                    "schema": "diffstory.annotations.v2",
                    "base_sha": metadata["base_sha"],
                    "head_sha": metadata["head_sha"],
                },
            },
        )
        application_reader = IndexedReaderProjection(
            shape="presentation",
            document=reader.document,
            sections=reader.sections,
            context_details=reader.context_details,
            citations=reader.citations,
            support_provenance=reader.support_provenance,
            version="diffstory.presentation.v2",
            provenance="model-generated · unverified",
        )
        application_graph = _derive_report_graph(
            graph,
            records,
            application_reader,
            base_report=report,
        )
        overlay_value = _freeze_report(overlay)
        original_run_bytes = canonical_bytes(
            inputs.run.model_dump(mode="json")
        )
        proof = _IndexedGenerationProof(
            scope_digest=self._scope_digest,
            report_digest=self._captured_input.report_digest,
            configuration_bytes=self._captured_input.configuration_bytes,
            compiler_fact_signature=self._captured_input.compiler_signature,
            records=records,
            graph=graph,
            reader=reader,
            overlay=overlay_value,
            run_ref=records.run_ref,
            application_payload=_IndexedApplicationPayload(
                graph=application_graph,
                records=records,
                reader=application_reader,
                base_report=report,
                overlay=overlay_value,
                original_run_bytes=original_run_bytes,
                versions_digest=self._captured_input._versions_digest,  # noqa: SLF001 # Bind the accepted validation and compatibility versions.
                graph_binding=_capture_graph_binding(application_graph),
            ),
            graph_binding=_capture_graph_binding(graph),
            stage_proofs=stage_proofs,
            versions_digest=self._captured_input._versions_digest,  # noqa: SLF001 # Bind the accepted validator/compatibility versions.
            witness="",
        )
        proof = _IndexedGenerationProof(
            **{
                field: getattr(proof, field)
                for field in proof.__dataclass_fields__
                if field != "witness"
            },
            witness=_indexed_generation_proof_witness(proof),
        )
        self._pending_generation_records = None
        self._pending_generation_recordset_witness = None
        return self._register(ValidatedGeneration, "generation", proof)

    def _discard_fresh_generation(
        self,
        generation: ValidatedGeneration,
        stage_proofs: tuple[StageProof, ...],
    ) -> None:
        """Expire temporary proofs after a data-only compatibility expansion.

        Preconditions:
            ``generation`` and ``stage_proofs`` are the exact completed fresh
            production capabilities registered in this open session.
        Postconditions:
            Those capabilities no longer authorize a follow-on session action.
        Essential behavior:
            Keeps compatibility ``generate`` output data-only while allowing
            ``generate_validated`` callers to retain their explicit token.
        """
        proof = self._registered_payload(
            generation, ValidatedGeneration, "generation"
        )
        _require_session(
            type(proof) is _IndexedGenerationProof
            and proof.stage_proofs == stage_proofs,
            "invalid_session_capability",
        )
        for stage_proof in stage_proofs:
            self._registered_payload(stage_proof, StageProof, "stage")
        for stage_proof in stage_proofs:
            self._registrations.pop(id(stage_proof))
        self._registrations.pop(id(generation))
        self._pending_generation_records = None
        self._pending_generation_recordset_witness = None

    def _registered_payload(
        self, handle: object, cls: type, kind: str
    ) -> object:
        """Return one exact current payload from the local capability registry.

        Preconditions:
            ``handle`` has the expected type and was issued by this session.
        Postconditions:
            Returns its payload only while capture and scope bindings are current.
        Essential behavior:
            Rejects copies, stale handles, wrong kinds, and changed inputs.
        """
        self._require_open()
        self._captured_input.assert_capture_current()
        return self._registered_payload_current(handle, cls, kind)

    def _registered_payload_current(
        self, handle: object, cls: type, kind: str
    ) -> object:
        """Resolve one exact registered payload after the operation checked input.

        Preconditions:
            The enclosing operation just checked this session's captured input.
        Postconditions:
            Returns only the payload registered to this exact handle identity.
        Essential behavior:
            Checks session, type, registry identity, kind, and scope without
            starting another full input/source traversal.
        """
        self._require_open()
        if type(handle) is not cls:
            msg = "invalid_session_capability"
            raise ValueError(msg)
        registration = self._registrations.get(id(handle))
        if registration is None or registration[0] is not handle:
            msg = "invalid_session_capability"
            raise ValueError(msg)
        if registration[1] != kind:
            msg = "invalid_session_capability"
            raise ValueError(msg)
        payload = registration[2]
        if getattr(payload, "scope_digest", None) != self._scope_digest:
            msg = "stale_session_capability"
            raise ValueError(msg)
        return payload

    def _require_open(self) -> None:
        if self._closed:
            msg = "session_closed"
            raise ValueError(msg)


def _stage_witness(
    stage: object,
    captured_input: _CapturedInput,
    *,
    checked_inputs: object | None = None,
    checked_sources: object | None = None,
    context_witnesses: _ContextWitnessCache | None = None,
) -> tuple[str, tuple[str, ...], str]:
    """Validate one stage boundary and hash its exact immutable binding.

    Preconditions:
        ``stage`` is a coordinator stage created from ``captured_input``.
    Postconditions:
        Returns stage name, ordered item keys, and an integrity witness.
    Essential behavior:
        Checks the exact current source owner once, then passes that owner to
        private context and selector checks for this synchronous traversal.
    """
    if checked_inputs is None:
        if checked_sources is not None or context_witnesses is not None:
            msg = "captured_input_mismatch"
            raise ValueError(msg)
        inputs = _stage_production_inputs(stage, captured_input)
        inputs.assert_capture_current()
        checked_sources = inputs.sources
        context_witnesses = {}
    elif context_witnesses is None:
        msg = "invalid_stage_binding"
        raise ValueError(msg)
    return _stage_witness_checked(
        stage,
        captured_input,
        checked_inputs if checked_inputs is not None else inputs,
        checked_sources,
        context_witnesses,
    )


def _stage_production_inputs(
    stage: object, captured_input: _CapturedInput
) -> object:
    """Return the exact production input owner attached to one stage.

    Preconditions:
        ``stage`` is expected to be a private ``StageInputs`` value.
    Postconditions:
        Returns its exact ``ProductionInputs`` object or raises a stable error.
    Essential behavior:
        Pins input identity without checking source liveness or content.
    """
    if (
        type(stage).__name__ != "StageInputs"
        or type(stage).__module__ != "diffstory.narrative_v2"
    ):
        msg = "invalid_stage_binding"
        raise ValueError(msg)
    inputs = getattr(stage, "inputs", None)
    if (
        type(inputs).__name__ != "ProductionInputs"
        or getattr(inputs, "_captured_input", None) is not captured_input
    ):
        msg = "captured_input_mismatch"
        raise ValueError(msg)
    return inputs


def _cached_stage_witness(  # noqa: PLR0913, PLR0917 # Keep exact owners and both stack-local witness maps explicit.
    stage: object,
    captured_input: _CapturedInput,
    checked_inputs: object,
    checked_sources: object | None,
    stage_witnesses: _StageWitnessCache,
    context_witnesses: _ContextWitnessCache,
) -> _StageWitness:
    """Reuse one exact stage witness within its enclosing synchronous call.

    Preconditions:
        ``checked_inputs`` and ``checked_sources`` were just checked by that
        operation; cache entries cannot escape the caller's stack frame.
    Postconditions:
        Returns the witness for this exact stage object and exact owners.
    Essential behavior:
        Reuses repeated aliases by object identity only; it keeps no session
        or process-wide trust memo.
    """
    prior = stage_witnesses.get(id(stage))
    if prior is not None:
        if (
            prior[0] is not stage
            or prior[1] is not checked_inputs
            or prior[2] is not checked_sources
        ):
            msg = "stage_binding_changed"
            raise ValueError(msg)
        return prior[3]
    witness = _stage_witness(
        stage,
        captured_input,
        checked_inputs=checked_inputs,
        checked_sources=checked_sources,
        context_witnesses=context_witnesses,
    )
    stage_witnesses[id(stage)] = (
        stage,
        checked_inputs,
        checked_sources,
        witness,
    )
    return witness


def _stage_witness_checked(
    stage: object,
    captured_input: _CapturedInput,
    checked_inputs: object,
    checked_sources: object | None,
    context_witnesses: _ContextWitnessCache,
) -> _StageWitness:
    """Hash one stage after the enclosing operation checked its source owner.

    Preconditions:
        ``checked_inputs`` is the exact current ProductionInputs for the
        capture, and ``checked_sources`` is its just-checked source owner.
    Postconditions:
        Returns the exact stage witness or rejects owner and raw-field drift.
    Essential behavior:
        Retains payload, context, compatibility-view, transport, and replay
        checks while selector helpers use the exact checked source object.
    """
    if (
        type(stage).__name__ != "StageInputs"
        or type(stage).__module__ != "diffstory.narrative_v2"
    ):
        msg = "invalid_stage_binding"
        raise ValueError(msg)
    inputs = getattr(stage, "inputs", None)
    if (
        inputs is not checked_inputs
        or getattr(inputs, "_captured_input", None) is not captured_input
        or getattr(inputs, "sources", None) is not checked_sources
    ):
        msg = "captured_input_mismatch"
        raise ValueError(msg)
    stage_name = getattr(stage, "stage", None)
    if stage_name not in {"detail", "planning", "section", "document"}:
        msg = "invalid_stage_binding"
        raise ValueError(msg)
    items = getattr(stage, "items", None)
    if not isinstance(items, Mapping):
        msg = "invalid_stage_binding"
        raise ValueError(msg)
    item_keys = tuple(items)
    item_facts = _items_witness(items, checked_sources, context_witnesses)
    transport_facts = _transport_witness(stage.transport)
    replay = getattr(stage, "replay", None)
    if replay is None:
        compatibility_views = None
    else:
        selected_views = (*replay.views, *replay.legacy_alternatives)
        compatibility_view_facts = []
        for view in selected_views:
            view_items = getattr(view, "items", None)
            view_contexts = getattr(view, "contexts", None)
            if not isinstance(view_items, Mapping) or not isinstance(
                view_contexts, Mapping
            ):
                msg = "invalid_stage_binding"
                raise ValueError(msg)
            compatibility_view_facts.append(
                (
                    id(view),
                    view.stage,
                    tuple(view_items),
                    _items_witness(
                        view_items, checked_sources, context_witnesses
                    ),
                    tuple(
                        (
                            key,
                            id(value),
                            _cached_context_witness(
                                value, checked_sources, context_witnesses
                            ),
                        )
                        for key, value in view_contexts.items()
                    ),
                    content_digest(_raw_integrity_bytes(view.handles)),
                    _transport_witness(view.transport),
                    content_digest(_raw_integrity_bytes(view.static)),
                    content_digest(_raw_integrity_bytes(view.lookups)),
                    content_digest(_raw_integrity_bytes(view.dynamic)),
                    content_digest(_raw_integrity_bytes(view.registry)),
                )
            )
        compatibility_views = tuple(compatibility_view_facts)
    witness = semantic_digest(
        "diffstory.session.stage-binding.v1",
        id(stage),
        id(inputs),
        id(captured_input),
        stage_name,
        item_keys,
        tuple(item_facts),
        tuple((key, id(value)) for key, value in stage.contexts.items()),
        content_digest(_raw_integrity_bytes(stage.handles)),
        content_digest(_raw_integrity_bytes(stage.static)),
        content_digest(_raw_integrity_bytes(stage.lookups)),
        content_digest(_raw_integrity_bytes(stage.dynamic)),
        content_digest(_raw_integrity_bytes(stage.registry)),
        transport_facts,
        None
        if replay is None
        else (
            id(replay),
            replay.layers,
            replay.fixture,
            id(replay.bound),
            id(replay.legacy),
            compatibility_views,
        ),
        captured_input.configuration_identity,
        captured_input.version_identity,
    )
    return stage_name, item_keys, witness


def _items_witness(
    items: Mapping[str, object],
    checked_sources: object | None,
    context_witnesses: _ContextWitnessCache,
) -> tuple[object, ...]:
    """Capture ordered payloads and contexts under one checked source owner."""
    facts = []
    for key, item in items.items():
        payload = getattr(item, "payload", None)
        if getattr(item, "key", None) != key or type(payload) is not bytes:
            msg = "invalid_stage_binding"
            raise ValueError(msg)
        facts.append(
            (
                key,
                id(item),
                item.subject,
                item.groups,
                id(payload),
                len(payload),
                content_digest(payload),
                item.output_tokens,
                _cached_context_witness(
                    item.context, checked_sources, context_witnesses
                ),
            )
        )
    return tuple(facts)


def _cached_context_witness(
    context: object | None,
    checked_sources: object | None,
    context_witnesses: _ContextWitnessCache,
) -> str | None:
    """Reuse one exact context witness within the current operation only."""
    if context is None:
        return None
    prior = context_witnesses.get(id(context))
    if prior is not None:
        if prior[0] is not context or prior[1] is not checked_sources:
            msg = "stage_context_changed"
            raise ValueError(msg)
        return prior[2]
    witness = _context_witness(context, checked_sources)
    context_witnesses[id(context)] = (context, checked_sources, witness)
    return witness


def _replay_memo_witness(stage: object) -> str:
    """Bind the exact post-execution codec memo without treating it as proof."""
    replay = getattr(stage, "replay", None)
    return _codec_replay_memo_witness(replay)


def _codec_replay_memo_witness(replay: object) -> str:
    """Bind one exact codec replay object's executed request memo."""
    memo = getattr(replay, "_requests", None)
    if (
        type(replay).__name__ != "CodecReplay"
        or type(replay).__module__ != "diffstory.narrative_v2"
        or type(memo) is not set
        or any(type(value) is not str for value in memo)
    ):
        msg = "codec_replay_changed"
        raise ValueError(msg)
    return semantic_digest(
        "diffstory.session.codec-replay-memo.v1",
        id(replay),
        content_digest(_raw_integrity_bytes(memo)),
    )


def _record_reference_facts(reference: object) -> tuple[object, ...]:
    """Return typed scalar facts for one immutable arena address."""
    from .report_storage import RecordRef
    from .report_storage import ResultRef

    if type(reference) is ResultRef:
        return (
            "result",
            reference.table,
            reference.kind,
            reference.storage_id,
            reference.item_key,
        )
    if type(reference) is RecordRef:
        return (
            "record",
            reference.table,
            reference.kind,
            reference.storage_id,
        )
    msg = "invalid_generation_record_reference"
    raise ValueError(msg)


def _stage_proof_witness(evidence: _StageProofEvidence) -> str:
    """Hash every immutable row and membership binding in one stage proof."""
    return semantic_digest(
        "diffstory.session.stage-proof.v1",
        evidence.scope_digest,
        id(evidence.stage),
        evidence.binding.stage_name,
        evidence.binding.wave_index,
        evidence.binding.sequence_index,
        evidence.binding.item_keys,
        evidence.binding.witness,
        tuple(id(item) for item in evidence.predecessors),
        tuple(_record_reference_facts(item) for item in evidence.result_refs),
        tuple(
            _record_reference_facts(item) for item in evidence.producer_refs
        ),
        tuple(
            _record_reference_facts(item) for item in evidence.invocation_refs
        ),
        tuple(_record_reference_facts(item) for item in evidence.request_refs),
        tuple(_record_reference_facts(item) for item in evidence.run_refs),
        tuple(_record_reference_facts(item) for item in evidence.receipt_refs),
        tuple(
            content_digest(_raw_integrity_bytes(item))
            for item in evidence.subject_audits
        ),
        tuple(
            (key, id(view), witness)
            for key, view, witness in evidence.producer_views
        ),
        evidence.replay_memo_witness,
    )


def _generation_recordset_witness(records: object) -> str:
    """Hash exact generation header and ordered arena references."""
    return semantic_digest(
        "diffstory.session.generation-record-set.v1",
        content_digest(_raw_integrity_bytes(records.header)),
        _record_reference_facts(records.run_ref),
        tuple(_record_reference_facts(item) for item in records.details),
        _record_reference_facts(records.plan_ref),
        tuple(_record_reference_facts(item) for item in records.planning),
        tuple(_record_reference_facts(item) for item in records.sections),
        _record_reference_facts(records.document),
        tuple(_record_reference_facts(item) for item in records.producers),
        tuple(_record_reference_facts(item) for item in records.receipts),
    )


def _capture_graph_binding(graph: object) -> tuple[object, ...]:
    """Pin graph roots, table maps, and indexes without hashing all rows.

    Preconditions:
        ``graph`` is the exact sealed ``ReportRecordGraph`` created by this
        session or its bytes-only import parser.
    Postconditions:
        Returns strong identity pins for the root, immutable table views, and
        indexes. Table rows are recursively immutable and remain shared.
    Essential behavior:
        Makes frozen-object field substitution visible at each consumer
        boundary without a full graph digest scan.
    """
    from .report_storage import ReportRecordGraph

    if type(graph) is not ReportRecordGraph:
        msg = "invalid_report_graph"
        raise ValueError(msg)
    table_views = tuple(
        (name, graph._tables[name])  # noqa: SLF001 # Retain only immutable table view identities.
        for name in graph._tables  # noqa: SLF001 # Preserve the exact current table set.
    )
    return (
        graph._root,  # noqa: SLF001 # The frozen root is identity-pinned.
        graph._tables,  # noqa: SLF001 # The frozen outer table map is identity-pinned.
        table_views,
        graph._result_by_key,  # noqa: SLF001 # The result selector index is identity-pinned.
        graph._result_by_id,  # noqa: SLF001 # The reverse result selector index is identity-pinned.
        graph._typed_validation_counts,  # noqa: SLF001 # Structural evidence remains from this graph.
        graph._artifact_kind,  # noqa: SLF001 # Root kind is a stable scalar binding.
        graph._raw,  # noqa: SLF001 # Session-created graphs do not retain input bytes.
    )


def _graph_binding_is_current(graph: object, binding: object) -> bool:
    """Check the exact immutable graph objects retained at issuance."""
    if type(binding) is not tuple or len(binding) != _GRAPH_BINDING_SIZE:
        return False
    try:
        root, tables, table_views, by_key, by_id, counts, kind, raw = binding
        return (
            graph._root is root  # noqa: SLF001 # Compare immutable root identity.
            and graph._tables is tables  # noqa: SLF001 # Compare immutable table-map identity.
            and type(table_views) is tuple
            and all(
                graph._tables.get(name) is view  # noqa: SLF001 # Compare each table view without scanning its records.
                for name, view in table_views
            )
            and len(table_views) == len(graph._tables)  # noqa: SLF001 # Detect substituted table membership.
            and graph._result_by_key is by_key  # noqa: SLF001 # Compare result index identity.
            and graph._result_by_id is by_id  # noqa: SLF001 # Compare reverse index identity.
            and graph._typed_validation_counts is counts  # noqa: SLF001 # Compare structural-count identity.
            and graph._artifact_kind == kind  # noqa: SLF001 # Compare report/annotation family.
            and graph._raw is raw  # noqa: SLF001 # Session graphs have no raw input alias.
        )
    except (AttributeError, TypeError, ValueError):
        return False


def _graph_binding_witness(graph: object, binding: object) -> str:
    """Witness expected and current graph identities at one trust boundary."""
    if type(binding) is not tuple or len(binding) != _GRAPH_BINDING_SIZE:
        return semantic_digest("diffstory.session.graph-binding.invalid.v1")
    try:
        expected_table_views = tuple(
            (name, id(view)) for name, view in binding[2]
        )
        current_table_views = tuple(
            (name, id(view))
            for name, view in graph._tables.items()  # noqa: SLF001 # Read table-map identities only.
        )
        current = (
            id(getattr(graph, "_root", None)),
            id(getattr(graph, "_tables", None)),
            current_table_views,
            id(getattr(graph, "_result_by_key", None)),
            id(getattr(graph, "_result_by_id", None)),
            id(getattr(graph, "_typed_validation_counts", None)),
            getattr(graph, "_artifact_kind", None),
            id(getattr(graph, "_raw", None)),
        )
        expected = (
            id(binding[0]),
            id(binding[1]),
            expected_table_views,
            id(binding[3]),
            id(binding[4]),
            id(binding[5]),
            binding[6],
            id(binding[7]),
        )
        return semantic_digest(
            "diffstory.session.graph-binding.v1",
            id(graph),
            id(binding),
            expected,
            current,
        )
    except (AttributeError, TypeError, ValueError):
        return semantic_digest("diffstory.session.graph-binding.invalid.v1")


def _indexed_generation_proof_witness(
    proof: _IndexedGenerationProof,
) -> str:
    """Pin the small owned generation payload without expanding its graph."""
    reader_facts = _indexed_reader_facts(proof.reader)
    application = proof.application_payload
    application_facts = (
        id(application),
        id(application.graph),
        id(application.records),
        _generation_recordset_witness(application.records),
        id(application.reader),
        _indexed_reader_facts(application.reader),
        id(application.base_report),
        id(application.overlay),
        content_digest(_raw_integrity_bytes(application.overlay)),
        id(application.original_run_bytes),
        content_digest(application.original_run_bytes),
        application.versions_digest,
        _graph_binding_witness(application.graph, application.graph_binding),
    )
    return semantic_digest(
        "diffstory.session.indexed-generation-proof.v1",
        proof.scope_digest,
        proof.report_digest,
        content_digest(proof.configuration_bytes),
        proof.compiler_fact_signature,
        id(proof.records),
        _generation_recordset_witness(proof.records),
        id(proof.graph),
        id(proof.graph_binding),
        _graph_binding_witness(proof.graph, proof.graph_binding),
        id(proof.reader),
        reader_facts,
        id(proof.overlay),
        content_digest(_raw_integrity_bytes(proof.overlay)),
        id(proof.run_ref),
        _record_reference_facts(proof.run_ref),
        application_facts,
        id(proof.stage_proofs),
        tuple(id(item) for item in proof.stage_proofs),
        proof.versions_digest,
    )


def _indexed_reader_facts(reader: object) -> tuple[object, ...]:
    """Return ordered primitive reference facts for one indexed reader.

    Preconditions:
        ``reader`` is the exact immutable projection retained by a session
        generation or M6 application payload.
    Postconditions:
        Returns only shape, scalar section selectors, and typed addresses.
    Essential behavior:
        Does not resolve or expand any referenced prose or audit record.
    """
    return (
        reader.shape,
        _record_reference_facts(reader.document),
        tuple(
            (
                section.group_id,
                section.group_key,
                section.number,
                section.next_id,
                _record_reference_facts(section.prose),
                tuple(
                    _record_reference_facts(item) for item in section.details
                ),
            )
            for section in reader.sections
        ),
        tuple(
            _record_reference_facts(item) for item in reader.context_details
        ),
        tuple(_record_reference_facts(item) for item in reader.citations),
        tuple(
            _record_reference_facts(item) for item in reader.support_provenance
        ),
        getattr(reader, "version", None),
        getattr(reader, "provenance", None),
    )


def _indexed_graph_matches_records(graph: object, records: object) -> bool:
    """Compare root generation selectors with the exact registered record set.

    Preconditions:
        ``graph`` and ``records`` are immutable session-owned graph and
        ``GenerationRecordSet`` values whose table rows passed structural
        validation at issue or import time.
    Postconditions:
        Returns true only when header fields and every ordered generation
        address select the same records and coverage.
    Essential behavior:
        Checks compact selectors without expanding result, audit, or prose rows.
    """
    from .report_storage import GenerationRecordSet
    from .report_storage import ReportRecordGraph

    if (
        type(graph) is not ReportRecordGraph
        or type(records) is not GenerationRecordSet
    ):
        return False
    generation = graph._root.get("generation")  # noqa: SLF001 # Inspect the exact root generation selector.
    if not isinstance(generation, Mapping):
        return False
    reference_fields = {
        "run",
        "details",
        "plan",
        "planning",
        "sections",
        "document",
        "producers",
        "receipts",
    }
    header = {
        key: value
        for key, value in generation.items()
        if key not in reference_fields
    }
    try:
        if canonical_bytes(header) != canonical_bytes(records.header):
            return False
        bindings = graph.generation_record_refs
        results = bindings["results"]
        return (
            bindings["run"] == records.run_ref
            and bindings["plan"] == records.plan_ref
            and bindings["producers"] == records.producers
            and bindings["receipts"] == records.receipts
            and results["details"] == records.details
            and results["planning"] == records.planning
            and results["sections"] == records.sections
            and results["document"] == (records.document,)
        )
    except (KeyError, TypeError, ValueError):
        return False


def _transport_witness(transport: object) -> tuple[object, ...]:
    """Capture immutable transport, codec, and output-bound facts compactly."""
    capacity = getattr(transport, "capacity", None)
    descriptor = transport.descriptor
    return (
        id(transport),
        content_digest(_raw_integrity_bytes(descriptor)),
        content_digest(_raw_integrity_bytes(transport.options)),
        getattr(capacity, "context_tokens", None),
        getattr(capacity, "max_output_tokens", None),
        content_digest(_raw_integrity_bytes(transport.limits)),
        getattr(transport, "config_digest", None),
        content_digest(getattr(transport, "prefix", b"")),
        getattr(transport, "enforceable", None),
        id(getattr(transport, "prepare_method", None)),
        id(getattr(transport, "send_method", None)),
        content_digest(getattr(transport, "config", b"")),
    )


def _context_witness(
    context: object | None, checked_sources: object | None
) -> str | None:
    """Hash exact context selectors under one already-checked source owner.

    Preconditions:
        The enclosing synchronous operation just checked ``checked_sources``
        through its exact ProductionInputs owner.
    Postconditions:
        Returns the raw context witness or rejects a mismatched owner/selector.
    Essential behavior:
        Keeps all support and selector binding checks while using ``_check_bound``
        instead of repeating the full indexed source traversal.
    """
    if context is None:
        return None
    sources = getattr(context, "sources", None)
    if sources is not checked_sources:
        msg = "stage_context_source_mismatch"
        raise ValueError(msg)
    consumed = getattr(context, "consumed", {})
    if isinstance(consumed, Mapping):
        consumed_facts = tuple(
            (
                key,
                id(value),
                content_digest(_raw_integrity_bytes(value)),
            )
            for key, value in consumed.items()
        )
    elif (
        type(consumed).__name__ == "IndexedConsumedProse"
        and type(consumed).__module__ == "diffstory.narrative_hierarchy"
        and callable(getattr(consumed, "_check_bound", None))
    ):
        if getattr(consumed, "_sources", None) is not checked_sources:
            msg = "stage_selector_source_mismatch"
            raise ValueError(msg)
        consumed._check_bound()  # noqa: SLF001 # The enclosing traversal checked this exact source owner.
        reference_values = getattr(consumed, "_reference_values", None)
        result_digests = getattr(consumed, "_result_digests", None)
        source_state = getattr(consumed, "_source_state", None)
        selector_binding = getattr(consumed, "_binding", None)
        if (
            type(reference_values) is not tuple
            or type(result_digests) is not tuple
            or len(reference_values) != len(result_digests)
            or source_state is None
            or type(selector_binding) is not tuple
        ):
            msg = "invalid_stage_binding"
            raise ValueError(msg)
        consumed_facts = (
            id(consumed),
            id(source_state),
            selector_binding,
            tuple(reference_values),
            tuple(result_digests),
        )
    else:
        msg = "invalid_stage_binding"
        raise ValueError(msg)
    return semantic_digest(
        "diffstory.session.stage-context.v1",
        id(context),
        content_digest(_raw_integrity_bytes(context.binding)),
        content_digest(_raw_integrity_bytes(context.claims)),
        content_digest(_raw_integrity_bytes(context.citations)),
        context.author_digest,
        context.final,
        context.max_wire_bytes,
        content_digest(_raw_integrity_bytes(context.support)),
        consumed_facts,
        id(sources),
        context.max_support_bytes,
    )


def _close_stage_caches(stage: object) -> None:
    """Release exact source scopes reachable from a registered stage."""
    contexts = getattr(stage, "contexts", {})
    if isinstance(contexts, Mapping):
        for context in tuple(contexts.values()):
            _close_source_caches(getattr(context, "sources", None))
    inputs = getattr(stage, "inputs", None)
    release = getattr(inputs, "_release_capture_data", None)
    if callable(release):
        release()


def _snapshot_object(value: object, name: str) -> bytes:
    """Create canonical owned bytes from JSON data or an immutable snapshot."""
    if isinstance(value, BaseModel):
        raw = canonical_bytes(value.model_dump(mode="json"))
    elif type(value) is dict:
        raw = canonical_bytes(value)
    elif type(value) is bytes:
        raw = value
    elif type(value) is str:
        raw = value.encode("utf-8")
    else:
        msg = f"{name}_must_be_json_object"
        raise TypeError(msg)
    parsed = strict_json(raw)
    if type(parsed) is not dict:
        msg = f"{name}_must_be_json_object"
        raise ValueError(msg)
    return canonical_bytes(parsed)


def _snapshot_report(
    value: object,
) -> tuple[_OwnedMapping, str, str, str | None]:
    """Capture one immutable report view and stream its integrity digests."""
    model_digest = None
    if isinstance(value, BaseModel):
        model_digest = content_digest(_raw_integrity_bytes(value))
        source = value.model_dump(mode="python")
    elif type(value) in {dict, bytes, str}:
        source = value
    else:
        msg = "report_must_be_json_object"
        raise TypeError(msg)

    if type(source) is bytes:
        parsed = strict_json(source)
        source = parsed
    elif type(source) is str:
        parsed = strict_json(source.encode("utf-8"))
        source = parsed

    if type(source) is not dict:
        msg = "report_must_be_json_object"
        raise ValueError(msg)
    report = _freeze_report(source)
    if type(report) is not _OwnedMapping:
        msg = "report_must_be_json_object"
        raise ValueError(msg)
    report_snapshot = _owned_report_snapshot(report, model_digest)
    if (
        isinstance(value, BaseModel)
        and canonical_digest(value) != report_snapshot[1]
    ):
        msg = "report_snapshot_mismatch"
        raise ValueError(msg)
    return report_snapshot


def _owned_report_snapshot(
    report_view: object, model_digest: str | None
) -> tuple[_OwnedMapping, str, str, str | None]:
    """Compute standard report witnesses for one already owned report.

    Preconditions:
        ``report_view`` is the exact immutable report representation produced
        by detached capture or the private fresh storage parser.
    Postconditions:
        Returns the report with its canonical digest, shape digest, and model
        integrity witness without allocating a report copy.
    Essential behavior:
        Keeps both capture routes on the same full-root digest semantics.
    """
    if type(report_view) is not _OwnedMapping:
        msg = "report_must_be_json_object"
        raise ValueError(msg)
    return (
        report_view,
        canonical_digest(report_view),
        _shape_digest(report_view),
        model_digest,
    )


def _report_identity(value: object) -> tuple[str, str, str | None]:
    """Stream current report/content shape witnesses without copying its tree."""
    model_digest = None
    if isinstance(value, BaseModel):
        source = value.model_dump(mode="python")
        report_digest = canonical_digest(value)
        model_digest = content_digest(_raw_integrity_bytes(value))
    elif type(value) is dict:
        source = value
        report_digest = canonical_digest(value)
    elif type(value) is bytes:
        source = strict_json(value)
        report_digest = canonical_digest(source)
    elif type(value) is str:
        source = strict_json(value.encode("utf-8"))
        report_digest = canonical_digest(source)
    else:
        msg = "report_must_be_json_object"
        raise TypeError(msg)
    if type(source) is not dict:
        msg = "report_must_be_json_object"
        raise ValueError(msg)
    return report_digest, _shape_digest(source), model_digest


def _freeze_report(value: object) -> object:
    """Recursively own primitive data without losing tuple/list distinctions."""
    if isinstance(value, BaseModel):
        return _freeze_report(value.model_dump(mode="python"))
    if type(value) is dict:
        items = []
        for key, item in value.items():
            if type(key) is not str:
                msg = "report_keys_must_be_strings"
                raise TypeError(msg)
            items.append((key, _freeze_report(item)))
        return _OwnedMapping(tuple(items))
    if type(value) is list:
        return _OwnedList(tuple(_freeze_report(item) for item in value))
    if type(value) is tuple:
        return _OwnedTuple(tuple(_freeze_report(item) for item in value))
    if value is None or type(value) in {bool, int, float, str}:
        return value
    msg = "report_contains_non_json_value"
    raise TypeError(msg)


def _shape_digest(value: object) -> str:
    """Hash the raw Python shape witness without building a second tree."""
    digest = sha256()
    for chunk in _shape_chunks(value):
        digest.update(chunk)
    return digest.hexdigest()


def _shape_chunks(value: object) -> Iterator[bytes]:
    """Yield canonical bytes for one type-preserving report-shape witness."""
    if type(value) in {dict, _OwnedMapping}:
        entries = value.items() if type(value) is dict else value.entries
        yield b'["mapping",['
        for position, (key, item) in enumerate(
            sorted(entries, key=lambda pair: pair[0])
        ):
            if position:
                yield b","
            yield b"["
            yield canonical_bytes(key)
            yield b","
            yield from _shape_chunks(item)
            yield b"]"
        yield b"]]"
        return
    if type(value) in {list, _OwnedList}:
        values = value if type(value) is list else value.items
        yield b'["list",['
        for position, item in enumerate(values):
            if position:
                yield b","
            yield from _shape_chunks(item)
        yield b"]]"
        return
    if type(value) in {tuple, _OwnedTuple}:
        values = value if type(value) is tuple else value.items
        yield b'["tuple",['
        for position, item in enumerate(values):
            if position:
                yield b","
            yield from _shape_chunks(item)
        yield b"]]"
        return
    yield b"["
    yield canonical_bytes(type(value).__name__)
    yield b","
    yield canonical_bytes(value)
    yield b"]"


def _thaw(value: object) -> dict:
    """Return a new mutable view of an immutable report graph."""
    if type(value) is _OwnedMapping:
        expanded = _thaw_owned(value)
        if type(expanded) is dict:
            return expanded
    msg = "invalid_owned_report"
    raise ValueError(msg)


def _decoded_object(raw: bytes, name: str) -> dict:
    value = strict_json(raw)
    if type(value) is not dict:
        msg = f"{name}_must_be_json_object"
        raise ValueError(msg)
    return value
