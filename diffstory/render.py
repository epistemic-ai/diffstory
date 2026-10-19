"""Single-file report renderer; no server, npm, CDN, or network dependency."""

from __future__ import annotations

import json
import re
from pathlib import Path

from .analysis import SCHEMA
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
    """Reject malformed report identifiers, numbers, and diff row tags."""
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
    """Validate object collections and string-list report fields."""
    for name in ("groups", "changes", "tests", "edges", "raw_files"):
        records = report.get(name)
        if not isinstance(records, list) or any(
            not isinstance(record, dict) for record in records
        ):
            msg = f"Report {name} must be a list of objects"
            raise ValueError(msg)
    for name in ("warnings", "notes"):
        values = report.get(name)
        if not isinstance(values, list) or any(
            not isinstance(value, str) for value in values
        ):
            msg = f"Report {name} must be a list of strings"
            raise ValueError(msg)


def _validate_report_stats(stats: object) -> None:
    """Require nonnegative integer report statistics and classification counts."""
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
    """Validate group citations and prerequisite/next-group references."""
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
    """Validate one current compiler or flat walkthrough report.

    Args:
        report: Untrusted decoded report mapping to validate before rendering,
            exporting evidence, or applying an annotation boundary.

    Returns:
        ``None`` when the report follows a supported current schema and all
        links checked by that schema resolve.

    Raises:
        ValueError: If the schema, report structure, IDs, links, annotations,
            or retired storage markers are unsupported or inconsistent.

    Preconditions:
        The report is parsed JSON-like data. This function does not accept a
        persisted storage envelope.
    Postconditions:
        Compiler v1 reports have valid narrative and source links; flat v2
        reports pass their complete walkthrough, packet, and coverage checks.
        Structural acceptance does not establish that prose is true.
    """
    if type(report) is not dict:
        msg = "Expected a diffstory report"
        raise ValueError(msg)
    if (
        "storage_version" in report
        or "artifact_kind" in report
        or ("root" in report and "tables" in report)
    ):
        msg = "Unsupported stored report format"
        raise ValueError(msg)
    if report.get("schema") == "diffstory.report.v2":
        from .walkthrough import validate_walkthrough

        validate_walkthrough(report)
        return
    if "generation" in report or "narrative_v2" in report:
        msg = "Unsupported generated report format"
        raise ValueError(msg)
    if report.get("schema") != SCHEMA:
        msg = "Expected a diffstory report"
        raise ValueError(msg)

    _validate_report_collections(report)
    _validate_nested_value(report)
    _validate_report_stats(report.get("stats"))
    groups = {group["id"] for group in report["groups"]}
    changes = {change["id"] for change in report["changes"]}
    if len(groups) != len(report["groups"]) or len(changes) != len(
        report["changes"]
    ):
        msg = "Duplicate report identifiers"
        raise ValueError(msg)
    changes_by_id = {change["id"]: change for change in report["changes"]}
    if "document" in report:
        validate_document(report["document"])
    _validate_group_links(report, groups, changes, changes_by_id)


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

    Preconditions:
        ``report`` is parsed report data and may be untrusted. The public
        renderer validates it on every call.
    Postconditions:
        The returned page embeds the validated report with JSON-context
        escaping and packaged local assets.

    """
    validate_report(report)
    return _render_prevalidated(report)


def _render_prevalidated(report: dict) -> str:
    """Render a report that a caller has just validated at its input boundary.

    Args:
        report: Report mapping that passed ``validate_report`` or was returned
            by the just-completed ``WalkthroughNarrator.generate_prepared``
            call and has not been changed since.

    Returns:
        Self-contained HTML containing the report and packaged reader assets.

    Raises:
        OSError: If a packaged renderer asset cannot be read.
        TypeError: If the report cannot be serialized as JSON.

    Preconditions:
        The exact mapping passed here was validated in the current synchronous
        call path. Do not call this with a loaded or changed report before its
        independent public validation boundary.
    Postconditions:
        Report content is escaped for the embedded JSON context. No validation
        result or trust marker is stored on the report.
    """
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
