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
from .render import _validate_report_collections
from .render import _validate_report_schema
from .render import _validate_v2_report_surface

_ISSUER_TOKEN = object()
_VALIDATION_VERSION = "diffstory.session.import-proof.v1"
_ASSEMBLY_VERSION = "diffstory.session.apply-v1"
_RENDER_SURFACE_VERSION = "diffstory.session.render-surface.v1"


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
    """A complete imported-generation proof registered to one session."""

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
        report_view, report_digest, shape_digest, model_digest = (
            _snapshot_report(report)
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
        self._report_type = (
            type(report) if isinstance(report, BaseModel) else None
        )
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
    scope = getattr(context, "citation_scope", None)
    if scope is not None and not getattr(scope, "closed", True):
        scope.close()


class ReportSession:
    """Own one report/configuration scope and its completed offline proofs.

    Callers must retain exclusive ownership of mutable inputs while
    ``capture`` snapshots them. Serialized JSON inputs are accepted as an
    immutable alternative. Returned values are detached dictionaries.
    """

    __slots__ = (
        "_captured_input",
        "_closed",
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
        if self._captured_input is not None:
            self._captured_input.close()
        self._captured_input = None
        self._scope_digest = ""
        self._closed = True

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

    def apply(self, generation: ValidatedGeneration) -> ValidatedReport:
        """Apply the retained expected candidate without rebuilding proof.

        Preconditions:
            ``generation`` is the exact registered capability for this session.
        Postconditions:
            Returns a report capability only while the original capture is current.
        Essential behavior:
            Checks full captured content before applying the proven candidate.
        """
        self._require_open()
        self._captured_input.assert_capture_current()
        proof = self._registered_payload(
            generation, ValidatedGeneration, "generation"
        )
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
        return self._register(ValidatedReport, "report", report_proof)

    def report_data(self, report: ValidatedReport) -> dict:
        """Return a detached report only for a live registered report handle."""
        proof = self._registered_payload(report, ValidatedReport, "report")
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
        stage_name, item_keys, witness = _stage_witness(
            stage, self._captured_input
        )
        existing = self._stage_bindings.get(id(stage))
        if existing is not None:
            if existing[0] is not stage:
                msg = "invalid_stage_binding"
                raise ValueError(msg)
            self._check_stage_binding(stage)
            return existing[1]
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
        self._require_open()
        registration = self._stage_bindings.get(id(stage))
        if registration is None or registration[0] is not stage:
            msg = "invalid_stage_binding"
            raise ValueError(msg)
        binding = registration[1]
        if binding.scope_digest != self._scope_digest:
            msg = "stale_stage_binding"
            raise ValueError(msg)
        stage_name, item_keys, witness = _stage_witness(
            stage, self._captured_input
        )
        if (
            stage_name != binding.stage_name
            or item_keys != binding.item_keys
            or witness != binding.witness
        ):
            msg = "stage_binding_changed"
            raise ValueError(msg)
        return binding

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
    stage: object, captured_input: _CapturedInput
) -> tuple[str, tuple[str, ...], str]:
    """Hash only the selected immutable facts that define one stage binding."""
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
    inputs.assert_capture_current()
    stage_name = getattr(stage, "stage", None)
    if stage_name not in {"detail", "planning", "section", "document"}:
        msg = "invalid_stage_binding"
        raise ValueError(msg)
    items = getattr(stage, "items", None)
    if not isinstance(items, Mapping):
        msg = "invalid_stage_binding"
        raise ValueError(msg)
    item_keys = tuple(items)
    item_facts = _items_witness(items)
    transport_facts = _transport_witness(stage.transport)
    replay = getattr(stage, "replay", None)
    if replay is None:
        compatibility_views = None
    else:
        selected_views = (*replay.views, *replay.legacy_alternatives)
        compatibility_views = tuple(
            (
                id(view),
                view.stage,
                tuple(view.items),
                _items_witness(view.items),
                tuple(
                    (key, id(value), _context_witness(value))
                    for key, value in view.contexts.items()
                ),
                content_digest(_raw_integrity_bytes(view.handles)),
                _transport_witness(view.transport),
                content_digest(_raw_integrity_bytes(view.static)),
                content_digest(_raw_integrity_bytes(view.lookups)),
                content_digest(_raw_integrity_bytes(view.dynamic)),
                content_digest(_raw_integrity_bytes(view.registry)),
            )
            for view in selected_views
        )
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


def _items_witness(items: Mapping[str, object]) -> tuple[object, ...]:
    """Capture ordered request payload and support identities compactly."""
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
                _context_witness(item.context),
            )
        )
    return tuple(facts)


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


def _context_witness(context: object | None) -> str | None:
    """Hash exact consumed support selectors without retaining model copies."""
    if context is None:
        return None
    sources = getattr(context, "sources", None)
    check_current = getattr(sources, "check_current", None)
    if callable(check_current):
        check_current()
    consumed = getattr(context, "consumed", {})
    consumed_facts = tuple(
        (
            key,
            id(value),
            content_digest(_raw_integrity_bytes(value)),
        )
        for key, value in consumed.items()
    )
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
    report_digest = canonical_digest(report)
    shape_digest = _shape_digest(report)
    if (
        isinstance(value, BaseModel)
        and canonical_digest(value) != report_digest
    ):
        msg = "report_snapshot_mismatch"
        raise ValueError(msg)
    return report, report_digest, shape_digest, model_digest


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
