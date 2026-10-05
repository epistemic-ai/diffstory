"""Single-file report renderer; no server, npm, CDN, or network dependency."""

from __future__ import annotations

import json
import re
from pathlib import Path

from .analysis import SCHEMA
from .analysis import validate_generated_report
from .analysis import validate_passages
from .models import validate_document

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


def _validate_nested_value(value: object) -> None:
    """Reject malformed report identifiers, numbers, and diff row tags.

    Args:
        value: Nested report value to validate recursively.

    Raises:
        ValueError: If a recognized identifier, number, or diff tag is invalid.
    """
    if isinstance(value, dict):
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
            _validate_nested_value(item)
    elif isinstance(value, list):
        for item in value:
            _validate_nested_value(item)


def _validate_report_collections(report: dict) -> None:
    """Validate object collections and string-list report fields.

    Args:
        report: Parsed Diffstory report mapping.

    Raises:
        ValueError: If an object collection contains a non-object, or warnings
            and notes are not lists of strings.
    """
    for name in (
        "groups",
        "changes",
        "tests",
        "edges",
        "raw_files",
    ):
        records = report.get(name)
        if not isinstance(records, list) or any(
            not isinstance(record, dict) for record in records
        ):
            msg = f"Report {name} must be a list of objects"
            raise ValueError(msg)
    warnings = report.get("warnings")
    if not isinstance(warnings, list) or any(
        not isinstance(value, str) for value in warnings
    ):
        msg = "Report warnings must be a list of strings"
        raise ValueError(msg)
    notes = report.get("notes")
    if not isinstance(notes, list) or any(
        not isinstance(value, str) for value in notes
    ):
        msg = "Report notes must be a list of strings"
        raise ValueError(msg)


def _validate_report_stats(stats: object) -> None:
    """Require nonnegative integer report statistics and classification counts.

    Args:
        stats: Report statistics field.

    Raises:
        ValueError: If statistics are absent, malformed, or negative.
    """
    if not isinstance(stats, dict):
        msg = "Report stats must be an object"
        raise ValueError(msg)
    for key, value in stats.items():
        if key == "by_kind":
            if not isinstance(value, dict) or any(
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
) -> None:
    """Validate group citations and prerequisite/next-group references.

    Args:
        report: Parsed report with validated collection records.
        groups: Valid group IDs.
        changes: Valid change IDs.
        changes_by_id: Change lookup used by passage validation.

    Raises:
        ValueError: If a group references an unknown change or group.
    """
    for group in report["groups"]:
        narrative = group.get("narrative", {})
        if not isinstance(narrative, dict):
            msg = "Invalid group narrative"
            raise ValueError(msg)
        if "passages" in narrative:
            validate_passages(narrative["passages"], group, changes_by_id)
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
    if report.get("schema") != SCHEMA:
        msg = "Expected a diffstory report"
        raise ValueError(msg)
    _validate_report_collections(report)
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
