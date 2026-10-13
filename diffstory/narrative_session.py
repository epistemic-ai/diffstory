"""Run-scoped immutable validation and application capabilities for v2 reports.

The session owns canonical JSON snapshots. Imported generations receive a
capability only after the complete production proof succeeds. Capabilities are
opaque, registered by object identity, and expire when the session closes.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Self

from pydantic import BaseModel

from .narrative_manifest import _apply_v2_candidate
from .narrative_manifest import persisted_manifest
from .narrative_manifest import validate_generation
from .narrative_plan import canonical_bytes
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
class _FrozenMapping:
    items: tuple[tuple[str, object], ...]


@dataclass(frozen=True, slots=True)
class _FrozenList:
    items: tuple[object, ...]


@dataclass(frozen=True, slots=True)
class _FrozenTuple:
    items: tuple[object, ...]


@dataclass(frozen=True, slots=True)
class _GenerationProof:
    scope_digest: str
    report_bytes: bytes
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
    report_bytes: bytes
    report_value: object
    configuration_bytes: bytes
    original_run_bytes: bytes
    compiler_fact_signature: str
    manifest_digest: str
    candidate_digest: str
    validation_version: str
    assembly_version: str
    render_surface_version: str


class ReportSession:
    """Own one report/configuration scope and its completed offline proofs.

    Callers must retain exclusive ownership of mutable inputs while
    ``capture`` snapshots them. Serialized JSON inputs are accepted as an
    immutable alternative. Returned values are detached dictionaries.
    """

    __slots__ = (
        "_closed",
        "_configuration_bytes",
        "_registrations",
        "_report_bytes",
        "_report_snapshot",
        "_scope_digest",
    )

    def __init__(
        self,
        report_bytes: bytes,
        report_snapshot: object,
        configuration_bytes: bytes,
        *,
        issuer: object = None,
    ) -> None:
        """Store only detached immutable snapshots and a local registry."""
        if issuer is not _ISSUER_TOKEN:
            msg = "Use ReportSession.capture to create a validation scope"
            raise TypeError(msg)
        if canonical_bytes(_thaw(report_snapshot)) != report_bytes:
            msg = "report_snapshot_mismatch"
            raise ValueError(msg)
        self._report_bytes = report_bytes
        self._report_snapshot = report_snapshot
        self._configuration_bytes = configuration_bytes
        self._scope_digest = semantic_digest(
            "diffstory.report-session.scope.v1",
            content_digest(report_bytes),
            content_digest(canonical_bytes(_shape_value(report_snapshot))),
            content_digest(configuration_bytes),
        )
        self._registrations: dict[
            int, tuple[_OpaqueCapability, str, object]
        ] = {}
        self._closed = False

    @classmethod
    def capture(
        cls, report: object, request_configuration: object
    ) -> ReportSession:
        """Snapshot report data and the complete typed request descriptor."""
        report_bytes, report_snapshot = _snapshot_report(report)
        configuration_input = _snapshot_object(
            request_configuration, "request_configuration"
        )
        configuration = RequestConfiguration.model_validate_json(
            configuration_input
        )
        configuration_bytes = canonical_bytes(
            configuration.model_dump(mode="json")
        )
        return cls(
            report_bytes,
            report_snapshot,
            configuration_bytes,
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
        """Expire all handles and release retained proof data permanently."""
        if self._closed:
            return
        self._registrations.clear()
        self._report_bytes = b""
        self._report_snapshot = _FrozenMapping(())
        self._configuration_bytes = b""
        self._scope_digest = ""
        self._closed = True

    def captured_report(self) -> dict:
        """Return a detached copy of the captured report."""
        self._require_open()
        return _thaw(self._report_snapshot)

    def request_configuration_data(self) -> dict:
        """Return a detached JSON descriptor, never the owned model."""
        self._require_open()
        return _decoded_object(
            self._configuration_bytes, "request_configuration"
        )

    def parse_import(
        self, manifest: object, *, candidate: object | None = None
    ) -> ParsedRecords:
        """Parse detached import records without authorizing their use."""
        self._require_open()
        parsed = self._parse_import(manifest, candidate)
        return self._register(ParsedRecords, "parsed", parsed)

    def validate_import(
        self,
        manifest: object,
        *,
        candidate: object | None = None,
    ) -> ValidatedGeneration:
        """Run the complete production proof and retain its expected output."""
        self._require_open()
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
        """Apply the retained expected candidate without rebuilding proof."""
        proof = self._registered_payload(
            generation, ValidatedGeneration, "generation"
        )
        if type(proof) is not _GenerationProof:
            msg = "invalid_session_capability"
            raise ValueError(msg)
        if (
            proof.scope_digest != self._scope_digest
            or proof.report_bytes != self._report_bytes
            or proof.configuration_bytes != self._configuration_bytes
            or proof.validation_version != _VALIDATION_VERSION
        ):
            msg = "stale_session_capability"
            raise ValueError(msg)

        report = _thaw(self._report_snapshot)
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
            report_bytes=canonical_bytes(applied),
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
            or proof.configuration_bytes != self._configuration_bytes
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
        if parsed.scope_digest != self._scope_digest:
            msg = "stale_session_capability"
            raise ValueError(msg)
        manifest = persisted_manifest(strict_json(parsed.manifest_bytes))
        configuration = RequestConfiguration.model_validate_json(
            self._configuration_bytes
        )
        if (
            canonical_bytes(
                manifest.request_configuration.model_dump(mode="json")
            )
            != self._configuration_bytes
            or manifest.ledger_parameters != configuration.options.ledger
            or manifest.run.provider_destination != configuration.destination
            or manifest.run.model != configuration.model
        ):
            msg = "dependencies_changed"
            raise ValueError(msg)

        report = _decoded_object(self._report_bytes, "report")
        candidate = (
            None
            if parsed.candidate_bytes is None
            else _decoded_object(parsed.candidate_bytes, "candidate")
        )
        expected = validate_generation(report, manifest, candidate=candidate)
        proof = _GenerationProof(
            scope_digest=self._scope_digest,
            report_bytes=self._report_bytes,
            configuration_bytes=self._configuration_bytes,
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
        self._require_open()
        handle = cls(_ISSUER_TOKEN)
        self._registrations[id(handle)] = (handle, kind, payload)
        return handle

    def _registered_payload(
        self, handle: object, cls: type, kind: str
    ) -> object:
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


def _snapshot_report(value: object) -> tuple[bytes, object]:
    """Keep report container types while also validating canonical JSON bytes."""
    if isinstance(value, BaseModel):
        source = value.model_dump(mode="python")
    elif type(value) in {dict, bytes, str}:
        source = value
    else:
        msg = "report_must_be_json_object"
        raise TypeError(msg)

    if type(source) is bytes:
        parsed = strict_json(source)
        raw = canonical_bytes(parsed)
        source = parsed
    elif type(source) is str:
        parsed = strict_json(source.encode("utf-8"))
        raw = canonical_bytes(parsed)
        source = parsed
    else:
        raw = canonical_bytes(value)
        strict_json(raw)

    if type(source) is not dict:
        msg = "report_must_be_json_object"
        raise ValueError(msg)
    return raw, _freeze_report(source)


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
        return _FrozenMapping(tuple(items))
    if type(value) is list:
        return _FrozenList(tuple(_freeze_report(item) for item in value))
    if type(value) is tuple:
        return _FrozenTuple(tuple(_freeze_report(item) for item in value))
    if value is None or type(value) in {bool, int, float, str}:
        return value
    msg = "report_contains_non_json_value"
    raise TypeError(msg)


def _shape_value(value: object) -> object:
    """Encode container and scalar kinds into a deterministic scope identity."""
    if type(value) is _FrozenMapping:
        return [
            "mapping",
            [
                [key, _shape_value(item)]
                for key, item in sorted(value.items, key=lambda pair: pair[0])
            ],
        ]
    if type(value) is _FrozenList:
        return ["list", [_shape_value(item) for item in value.items]]
    if type(value) is _FrozenTuple:
        return ["tuple", [_shape_value(item) for item in value.items]]
    return [type(value).__name__, value]


def _thaw(value: object) -> dict:
    """Return a new mutable view of an immutable report graph."""
    if type(value) is _FrozenMapping:
        return {key: _thaw_value(item) for key, item in value.items}
    msg = "invalid_owned_report"
    raise ValueError(msg)


def _thaw_value(value: object) -> object:
    if type(value) is _FrozenMapping:
        return {key: _thaw_value(item) for key, item in value.items}
    if type(value) is _FrozenList:
        return [_thaw_value(item) for item in value.items]
    if type(value) is _FrozenTuple:
        return tuple(_thaw_value(item) for item in value.items)
    return value


def _decoded_object(raw: bytes, name: str) -> dict:
    value = strict_json(raw)
    if type(value) is not dict:
        msg = f"{name}_must_be_json_object"
        raise ValueError(msg)
    return value
