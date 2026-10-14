"""Single-file report renderer; no server, npm, CDN, or network dependency."""

from __future__ import annotations

import codecs
import json
import re
from pathlib import Path
from typing import TYPE_CHECKING

from .analysis import SCHEMA
from .analysis import validate_generated_report
from .analysis import validate_passages
from .models import validate_document
from .narrative_plan import _OwnedList
from .narrative_plan import _OwnedMapping
from .narrative_plan import _OwnedTuple

if TYPE_CHECKING:
    from collections.abc import Iterator
    from collections.abc import Mapping

    from .narrative_session import ReportSession
    from .narrative_session import ValidatedReport

REPORT_IDENTIFIER = re.compile(r"[a-f0-9]{16}")
NUMERIC_REPORT_FIELDS = frozenset(
    {
        "number",
        "start",
        "end",
        "old_start",
        "new_start",
        "old_count",
        "new_count",
        "old",
        "new",
        "changed_files",
        "additions",
        "deletions",
        "source_bytes",
    },
)


def _validate_nested_value(
    value: object, *, allow_owned: bool = False
) -> None:
    """Reject malformed report identifiers, numbers, and diff row tags.

    Args:
        value: Nested report value to validate recursively.
        allow_owned: Whether exact immutable session-owned containers are valid.

    Raises:
        ValueError: If a recognized identifier, number, or diff tag is invalid.
    """
    if isinstance(value, dict) or (
        allow_owned and type(value) is _OwnedMapping
    ):
        for key, item in value.items():
            is_identifier = (
                key == "id" or key.endswith("_id") or key in {"from", "to"}
            )
            if (
                item is not None
                and is_identifier
                and (
                    not isinstance(item, str)
                    or not REPORT_IDENTIFIER.fullmatch(item)
                )
            ):
                msg = f"Invalid report identifier: {key}"
                raise ValueError(msg)
            if (
                item is not None
                and key in NUMERIC_REPORT_FIELDS
                and (type(item) is not int or item < 0)
            ):
                msg = f"Invalid report number: {key}"
                raise ValueError(msg)
            if key == "tag" and item not in ("context", "add", "delete"):
                msg = "Invalid diff row tag"
                raise ValueError(msg)
            _validate_nested_value(item, allow_owned=allow_owned)
    elif isinstance(value, list) or (
        allow_owned and type(value) in {_OwnedList, _OwnedTuple, tuple}
    ):
        for item in value:
            _validate_nested_value(item, allow_owned=allow_owned)


def _validate_report_collections(
    report: dict, *, allow_owned: bool = False
) -> None:
    """Validate object collections and string-list report fields.

    Args:
        report: Parsed Diffstory report mapping.
        allow_owned: Whether exact immutable session-owned containers are valid.

    Raises:
        ValueError: If an object collection contains a non-object, or warnings
            and notes are not lists of strings.
    """

    def is_object(value: object) -> bool:
        return isinstance(value, dict) or (
            allow_owned and type(value) is _OwnedMapping
        )

    def is_array(value: object) -> bool:
        return isinstance(value, list) or (
            allow_owned and type(value) in {_OwnedList, _OwnedTuple, tuple}
        )

    for name in (
        "groups",
        "changes",
        "tests",
        "edges",
        "raw_files",
    ):
        records = report.get(name)
        if not is_array(records) or any(
            not is_object(record) for record in records
        ):
            msg = f"Report {name} must be a list of objects"
            raise ValueError(msg)
    warnings = report.get("warnings")
    if not is_array(warnings) or any(
        not isinstance(value, str) for value in warnings
    ):
        msg = "Report warnings must be a list of strings"
        raise ValueError(msg)
    notes = report.get("notes")
    if not is_array(notes) or any(
        not isinstance(value, str) for value in notes
    ):
        msg = "Report notes must be a list of strings"
        raise ValueError(msg)


def _validate_report_stats(
    stats: object, *, allow_owned: bool = False
) -> None:
    """Require nonnegative integer report statistics and classification counts.

    Args:
        stats: Report statistics field.
        allow_owned: Whether an immutable owned statistics mapping is valid.

    Raises:
        ValueError: If statistics are absent, malformed, or negative.
    """
    if not isinstance(stats, dict) and not (
        allow_owned and type(stats) is _OwnedMapping
    ):
        msg = "Report stats must be an object"
        raise ValueError(msg)
    for key, value in stats.items():
        if key == "by_kind":
            if (
                not isinstance(value, dict)
                and not (allow_owned and type(value) is _OwnedMapping)
            ) or any(
                type(count) is not int or count < 0 for count in value.values()
            ):
                msg = "Invalid classification counts"
                raise ValueError(msg)
        elif type(value) is not int or value < 0:
            msg = f"Invalid statistic: {key}"
            raise ValueError(msg)


def _validate_group_links(
    report: dict,
    groups: set[str],
    changes: set[str],
    changes_by_id: dict[str, dict],
    *,
    allow_owned: bool = False,
) -> None:
    """Validate group citations and prerequisite/next-group references.

    Args:
        report: Parsed report with validated collection records.
        groups: Valid group IDs.
        changes: Valid change IDs.
        changes_by_id: Change lookup used by passage validation.
        allow_owned: Whether owned narrative/passages need small strict views.

    Raises:
        ValueError: If a group references an unknown change or group.
    """
    for group in report["groups"]:
        narrative = group.get("narrative", {})
        if not isinstance(narrative, dict) and not (
            allow_owned and type(narrative) is _OwnedMapping
        ):
            msg = "Invalid group narrative"
            raise ValueError(msg)
        if "passages" in narrative:
            passages = narrative["passages"]
            group_value = group
            if allow_owned:
                passages = _passage_renderer_projection(passages)
                group_value = dict(group)
                change_ids = group_value.get("change_ids", ())
                if type(change_ids) in {_OwnedList, _OwnedTuple}:
                    group_value["change_ids"] = list(change_ids)
            validate_passages(passages, group_value, changes_by_id)
        if not set(group.get("change_ids", ())) <= changes:
            msg = "Unknown group change ID"
            raise ValueError(msg)
        if not set(group.get("prerequisites", ())) <= groups:
            msg = "Unknown prerequisite group ID"
            raise ValueError(msg)
        next_id = group.get("next_id")
        if next_id is not None and next_id not in groups:
            msg = "Unknown next group ID"
            raise ValueError(msg)


def _validate_report_schema(
    report: dict, *, allow_owned: bool = False
) -> None:
    """Require the renderer's report schema before inspecting its fields."""
    if (
        allow_owned
        and type(report) is not dict
        and not (allow_owned and type(report) is _OwnedMapping)
    ):
        msg = "Expected a diffstory report"
        raise ValueError(msg)
    if report.get("schema") != SCHEMA:
        msg = "Expected a diffstory report"
        raise ValueError(msg)


def _validate_v2_report_surface(
    report: dict, *, allow_owned: bool = False
) -> None:
    """Check v2 renderer fields after a separate complete proof check.

    This verifies renderer safety only. It does not validate narration proof or
    issue a capability. The public report boundary calls it after
    ``validate_generated_report``; a completed ReportSession import may call it
    after its own complete generation validation and deterministic application.
    The schema and collection checks must already have passed.
    """
    _validate_nested_value(
        {
            key: value
            for key, value in report.items()
            if key not in {"generation", "narrative_v2"}
        },
        allow_owned=allow_owned,
    )
    _validate_report_stats(report.get("stats"), allow_owned=allow_owned)
    groups = {g["id"] for g in report["groups"]}
    changes = {c["id"] for c in report["changes"]}
    if len(groups) != len(report["groups"]) or len(changes) != len(
        report["changes"],
    ):
        msg = "Duplicate report identifiers"
        raise ValueError(msg)
    changes_by_id = {c["id"]: c for c in report["changes"]}
    _validate_group_links(
        report,
        groups,
        changes,
        changes_by_id,
        allow_owned=allow_owned,
    )


def _passage_renderer_projection(value: object) -> list:
    """Build the small mutable view required by the existing passage checker.

    Preconditions:
        ``value`` is an owned immutable array of passage mappings, or a value
        whose invalid shape should be rejected by ``validate_passages``.
    Postconditions:
        Returns shallow passage dictionaries with only list/dict adapters for
        the existing strict passage predicate. Nested source rows stay shared.
    Essential behavior:
        Does not copy a compiler report or presentation result tree.
    """
    if type(value) not in {_OwnedList, _OwnedTuple, tuple}:
        return value  # type: ignore[return-value]
    passages = []
    for item in value:
        if type(item) is not _OwnedMapping:
            passages.append(item)
            continue
        projected = dict(item.items())
        change_ids = projected.get("change_ids")
        if type(change_ids) in {_OwnedList, _OwnedTuple}:
            projected["change_ids"] = list(change_ids)
        focus = projected.get("focus")
        if type(focus) is _OwnedMapping:
            projected["focus"] = dict(focus.items())
        passages.append(projected)
    return passages


def _indexed_group_overlays(
    overlay: _OwnedMapping,
) -> dict[str, Mapping[str, object]]:
    """Index exact generated group fields by their report group selector."""
    overlay_groups = overlay.get("groups")
    if type(overlay_groups) not in {_OwnedList, _OwnedTuple, tuple}:
        msg = "invalid_indexed_applied_fields"
        raise ValueError(msg)
    expected_fields = {"group_id", "title", "number", "next_id", "narrative"}
    by_group: dict[str, Mapping[str, object]] = {}
    for item in overlay_groups:
        if type(item) is not _OwnedMapping or set(item) != expected_fields:
            msg = "invalid_indexed_applied_fields"
            raise ValueError(msg)
        group_id = item.get("group_id")
        if type(group_id) is not str or group_id in by_group:
            msg = "Duplicate report identifiers"
            raise ValueError(msg)
        by_group[group_id] = item
    return by_group


def _indexed_effective_group(
    group: object, group_overlay: Mapping[str, object]
) -> dict:
    """Create one shallow effective group for shared renderer predicates.

    Preconditions:
        ``group`` is an owned compiler group and ``group_overlay`` is its exact
        deterministic applied-field row.
    Postconditions:
        Returns one shallow mapping with generated fields overriding compiler
        values; every nested base value remains shared.
    Essential behavior:
        Checks generated provenance and exact change coverage without building
        a complete report or presentation model.
    """
    if type(group) is not _OwnedMapping:
        msg = "invalid_indexed_applied_fields"
        raise ValueError(msg)
    group_value = dict(group.items())
    for field in ("title", "number", "next_id", "narrative"):
        group_value[field] = group_overlay[field]
    narrative = group_overlay["narrative"]
    if (
        type(narrative) is not _OwnedMapping
        or narrative.get("provenance") != "model-generated · unverified"
    ):
        msg = "Generated report lost its model-generated provenance label"
        raise ValueError(msg)
    evidence = narrative.get("evidence_change_ids")
    change_ids = group.get("change_ids")
    if (
        type(evidence) not in {_OwnedList, _OwnedTuple}
        or type(change_ids) not in {_OwnedList, _OwnedTuple}
        or any(type(item) is not str for item in evidence)
        or len(evidence) != len(set(evidence))
        or set(evidence) != set(change_ids)
    ):
        msg = "Generated step does not cover its expected changes"
        raise ValueError(msg)
    return group_value


def _indexed_effective_report(
    base_report: object, overlay: object, reader: object
) -> dict:
    """Project a small effective renderer view from owned base and overlay.

    Preconditions:
        ``base_report`` and ``overlay`` are immutable owned views retained by
        a complete generation proof. ``reader`` is its checked presentation
        projection. Public callers must use ``validate_report``.
    Postconditions:
        Returns a shallow report map and shallow group rows with applied fields
        taking precedence over compiler values.
    Essential behavior:
        Leaves result, source, and audit values shared with their owners.
    """
    from .report_storage import IndexedReaderProjection

    if (
        type(base_report) is not _OwnedMapping
        or type(overlay) is not _OwnedMapping
    ):
        msg = "invalid_indexed_report_surface"
        raise ValueError(msg)
    if type(reader) is not IndexedReaderProjection:
        msg = "invalid_indexed_reader_projection"
        raise ValueError(msg)
    if (
        reader.shape != "presentation"
        or reader.version != "diffstory.presentation.v2"
        or reader.provenance != "model-generated · unverified"
    ):
        msg = "Invalid generated reader metadata"
        raise ValueError(msg)
    if set(overlay) != {"document", "groups"}:
        msg = "invalid_indexed_applied_fields"
        raise ValueError(msg)

    effective = dict(base_report.items())
    effective["document"] = overlay["document"]
    group_values = base_report.get("groups")
    effective["groups"] = group_values
    _validate_report_schema(effective, allow_owned=True)
    _validate_report_collections(effective, allow_owned=True)

    by_group = _indexed_group_overlays(overlay)

    if type(group_values) not in {_OwnedList, _OwnedTuple}:
        msg = "invalid_indexed_applied_fields"
        raise ValueError(msg)
    effective_groups = []
    for group in group_values:
        group_id = group.get("id")
        group_overlay = by_group.get(group_id)
        if group_overlay is None:
            msg = "invalid_indexed_applied_fields"
            raise ValueError(msg)
        effective_groups.append(_indexed_effective_group(group, group_overlay))
    if set(by_group) != {group.get("id") for group in group_values}:
        msg = "invalid_indexed_applied_fields"
        raise ValueError(msg)
    document = overlay["document"]
    if type(document) is not _OwnedMapping:
        msg = "invalid_indexed_applied_fields"
        raise ValueError(msg)
    validate_document(dict(document.items()), generated=True)
    effective["groups"] = effective_groups
    return effective


def _validate_indexed_report_surface(
    base_report: object, overlay: object, reader: object
) -> None:  # Keep the existing renderer predicate sequence explicit.
    """Apply renderer predicates to one owned base plus its exact overlay.

    Preconditions:
        Inputs are the immutable base, deterministic overlay, and presentation
        reader retained by a complete indexed generation proof.
    Postconditions:
        The effective report fields pass the same schema, collection,
        identifier, numeric, tag, link, passage, document, and label checks as
        the expanded v2 renderer boundary.
    Essential behavior:
        Uses shallow projections and shared existing predicates. It does not
        issue authority or build an expanded report.
    """
    effective = _indexed_effective_report(base_report, overlay, reader)
    _validate_report_collections(effective, allow_owned=True)
    _validate_v2_report_surface(effective, allow_owned=True)


def validate_report(report: dict) -> None:
    """
    Validate report structure and values that affect safe rendering.

    Textual prose is rendered as escaped data by the browser UI. This check
    validates report identifiers, numeric fields, narrative evidence, and
    generated-report provenance before embedding the report.

    Args:
        report: Parsed Diffstory report mapping.

    Raises:
        ValueError: If the schema or any renderer-facing field is invalid.
        TypeError: If a value has an unsupported container or scalar type.

    """
    _validate_report_schema(report)
    _validate_report_collections(report)
    generation = report.get("generation")
    v2 = (
        isinstance(generation, dict)
        and generation.get("version") == "diffstory.generation.v2"
    )
    if v2:
        # Only these exact root paths are exempt after complete proof and
        # deterministic presentation validation. All compiler IDs stay 16-hex.
        validate_generated_report(report)
        _validate_v2_report_surface(report)
        return
    if "narrative_v2" in report:
        msg = "V2 presentation requires a validated v2 generation"
        raise ValueError(msg)
    _validate_nested_value(report)
    _validate_report_stats(report.get("stats"))
    groups = {g["id"] for g in report["groups"]}
    changes = {c["id"] for c in report["changes"]}
    if len(groups) != len(report["groups"]) or len(changes) != len(
        report["changes"],
    ):
        msg = "Duplicate report identifiers"
        raise ValueError(msg)
    changes_by_id = {c["id"]: c for c in report["changes"]}
    if "document" in report:
        validate_document(report["document"])
    _validate_group_links(report, groups, changes, changes_by_id)
    if "generation" in report:
        validate_generated_report(report)


def render(report: dict) -> str:
    """
    Render a validated report as a self-contained HTML document.

    Args:
        report: Report mapping to validate and embed.

    Returns:
        HTML containing the report data and packaged CSS, JavaScript, and mark.

    Raises:
        ValueError: If report validation fails.
        OSError: If a packaged renderer asset cannot be read.
        TypeError: If the report cannot be serialized as JSON.

    """
    validate_report(report)
    assets = Path(__file__).with_name("assets")
    payload = json.dumps(report, ensure_ascii=False, separators=(",", ":"))
    # application/json script elements can be closed by an unescaped </script>.
    payload = (
        payload.replace("&", "\\u0026")
        .replace("<", "\\u003c")
        .replace(">", "\\u003e")
        .replace("\u2028", "\\u2028")
        .replace("\u2029", "\\u2029")
    )
    return (
        assets.joinpath("app.html")
        .read_text(encoding="utf-8")
        .replace(
            "/*__CSS__*/",
            assets.joinpath("app.css").read_text(encoding="utf-8"),
        )
        .replace(
            "/*__JS__*/",
            assets.joinpath("app.js").read_text(encoding="utf-8"),
        )
        .replace(
            "__BRAND_MARK__",
            assets.joinpath("epistemic-mark.svg").read_text(encoding="utf-8"),
        )
        .replace("__REPORT_JSON__", payload)
    )


def iter_render_validated(
    session: ReportSession,
    report: ValidatedReport,
    *,
    report_storage: str = "expanded",
    max_output_bytes: int,
) -> Iterator[bytes]:
    """Return a bounded self-contained HTML stream for a sealed report.

    Args:
        session: The exact live session that issued ``report``.
        report: The exact registered indexed report capability.
        report_storage: ``expanded`` for the existing browser data shape or
            ``shared`` for a storage envelope.
        max_output_bytes: Exact nonnegative cap over all UTF-8 HTML bytes,
            including packaged assets and script-context escaping.

    Returns:
        Iterator[bytes]: Bounded UTF-8 chunks for the complete HTML document.

    Preconditions:
        The handle remains registered in ``session`` and its captured input,
        configuration, proof, and graph bindings remain current.
    Postconditions:
        The complete output is emitted only if every HTML byte fits the cap
        and the same report proof is current at the final boundary.
    Essential behavior:
        Replaces the packaged template assets, streams JSON through the
        application/json script-context escaper, and keeps the existing
        self-contained CSP. It does not build a complete JSON or HTML string.

    Raises:
        ValueError: If the format, cap, capability, or current bindings fail.
        OSError: If a packaged renderer asset cannot be read.
    """
    from .narrative_session import ReportSession as _ReportSession
    from .narrative_session import ValidatedReport as _ValidatedReport

    if (
        type(session) is not _ReportSession
        or type(report) is not _ValidatedReport
    ):
        msg = "indexed_report_handle_required"
        raise ValueError(msg)
    if type(report_storage) is not str or report_storage not in {
        "expanded",
        "shared",
    }:
        msg = "unknown_report_storage"
        raise ValueError(msg)
    if type(max_output_bytes) is not int or max_output_bytes < 0:
        msg = "invalid_output_limit"
        raise ValueError(msg)
    return _iter_render_validated(
        session, report, report_storage, max_output_bytes
    )


def _iter_render_validated(
    session: ReportSession,
    report: ValidatedReport,
    report_storage: str,
    max_output_bytes: int,
) -> Iterator[bytes]:
    """Consume one checked graph as capped, self-contained HTML bytes.

    Preconditions:
        The public adapter accepted the exact handle types, storage format,
        and nonnegative output cap.
    Postconditions:
        Emits complete UTF-8 template, JSON, and suffix chunks within the HTML
        cap, then performs one full proof and capture check before completion.
    Essential behavior:
        Holds only packaged template fragments and a bounded JSON/escape
        buffer; liveness is checked between chunks without a new capture walk.
    """
    from .narrative_session import ValidatedReport as _ValidatedReport
    from .report_storage import _iter_bounded_json_chunks
    from .report_storage import _iter_report_output_json

    application = session._checked_report_data(report)  # noqa: SLF001 # Use only the exact registered session payload.
    assets = Path(__file__).with_name("assets")
    template = (
        assets.joinpath("app.html")
        .read_text(encoding="utf-8")
        .replace(
            "/*__CSS__*/",
            assets.joinpath("app.css").read_text(encoding="utf-8"),
        )
        .replace(
            "/*__JS__*/",
            assets.joinpath("app.js").read_text(encoding="utf-8"),
        )
        .replace(
            "__BRAND_MARK__",
            assets.joinpath("epistemic-mark.svg").read_text(encoding="utf-8"),
        )
    )
    marker = "__REPORT_JSON__"
    if template.count(marker) != 1:
        msg = "invalid_renderer_template"
        raise ValueError(msg)
    prefix, suffix = template.split(marker, 1)
    prefix_bytes = prefix.encode("utf-8")
    suffix_bytes = suffix.encode("utf-8")
    json_chunks = _iter_bounded_json_chunks(
        _iter_report_output_json(
            application.graph,
            application.overlay,
            report_storage=report_storage,
        ),
        chunk_size=4096,
    )
    total_bytes = 0

    def checked_chunks(chunks: Iterator[bytes]) -> Iterator[bytes]:
        """Count complete HTML bytes and check handle liveness per chunk."""
        nonlocal total_bytes
        for chunk in chunks:
            session._registered_payload_current(  # noqa: SLF001 # Reject close/copy/foreign use between chunks without recapturing.
                report, _ValidatedReport, "report"
            )
            if total_bytes + len(chunk) > max_output_bytes:
                msg = "render_output_too_large"
                raise ValueError(msg)
            total_bytes += len(chunk)
            yield chunk

    for chunk in _iter_byte_slices(prefix_bytes):
        yield from checked_chunks(iter((chunk,)))
    escaped_json = _iter_application_json_escaped(json_chunks)
    yield from checked_chunks(escaped_json)
    for chunk in _iter_byte_slices(suffix_bytes):
        yield from checked_chunks(iter((chunk,)))
    if session._checked_report_data(report) is not application:  # noqa: SLF001 # Require a final complete capture and graph check.
        msg = "stale_session_capability"
        raise ValueError(msg)


def _iter_byte_slices(
    value: bytes, *, size: int = 16 * 1024
) -> Iterator[bytes]:
    """Split a small packaged HTML fragment into bounded output pieces.

    Preconditions:
        ``value`` is exact UTF-8 output bytes and ``size`` is positive.
    Postconditions:
        Joining the pieces restores ``value`` exactly.
    Essential behavior:
        Bounds each caller-visible asset or escaped-data chunk.
    """
    for offset in range(0, len(value), size):
        yield value[offset : offset + size]


def _iter_application_json_escaped(chunks: Iterator[bytes]) -> Iterator[bytes]:
    """Escape JSON text for a script element while preserving parsed values.

    Preconditions:
        ``chunks`` forms valid canonical UTF-8 JSON and is consumed in order.
    Postconditions:
        Joining the output produces JSON text that parses to the same value,
        with HTML-significant characters and JavaScript line separators escaped.
    Essential behavior:
        Carries UTF-8 decoder state across input chunks and keeps only two
        trailing Unicode characters between escape operations.
    """
    decoder = codecs.getincrementaldecoder("utf-8")()
    carry = ""
    carry_characters = 2

    def escaped_bytes(text: str) -> bytes:
        return (
            text.replace("&", "\\u0026")
            .replace("<", "\\u003c")
            .replace(">", "\\u003e")
            .replace("\u2028", "\\u2028")
            .replace("\u2029", "\\u2029")
        ).encode("utf-8")

    for chunk in chunks:
        text = carry + decoder.decode(chunk, final=False)
        if len(text) <= carry_characters:
            carry = text
            continue
        safe, carry = text[:-carry_characters], text[-carry_characters:]
        yield from _iter_byte_slices(escaped_bytes(safe))
    tail = carry + decoder.decode(b"", final=True)
    if tail:
        yield from _iter_byte_slices(escaped_bytes(tail))
