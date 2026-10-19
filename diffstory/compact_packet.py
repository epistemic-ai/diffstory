"""Compact, revision-bound compiler evidence for the walkthrough writer.

Source buffers occur once. Changes and complete supplied diffs refer to those
buffers by ID and original line coordinates. A changed hunk is an accounting
region; unchanged context never creates an independent writing task.
"""

from __future__ import annotations

import copy
import hashlib
import json
from collections import defaultdict
from typing import TYPE_CHECKING

from .packet_schema import Packet
from .patches import parse_file_patch
from .source_lines import split_source_lines

if TYPE_CHECKING:
    from collections.abc import Mapping

PACKET_VERSION = "diffstory.packet.v1"


class SourceIndex(dict[str, dict]):
    """Index sources and split each physical buffer once per validation pass.

    Attributes:
        physical: Exact lines, including endings, keyed by source ID. Values
            are transient indexes and never become persisted source records.
        by_region: Source IDs keyed by region and revision side. More than one
            entry is ambiguous and rejected at source/diff boundaries.
    """

    def __init__(self, sources: list[dict]) -> None:
        """Index supplied records without copying their source buffers."""
        super().__init__((source["id"], source) for source in sources)
        self.physical = {
            source["id"]: split_source_lines(source["text"], keepends=True)
            for source in sources
        }
        self.by_region: dict[tuple[str, str], list[str]] = defaultdict(list)
        for source in sources:
            self.by_region[(source["region"], source["side"])].append(
                source["id"]
            )


def canonical_bytes(value: object) -> bytes:
    """Encode JSON deterministically, rejecting nonfinite values.

    Args:
        value: A JSON-compatible value with string mapping keys.

    Returns:
        Compact UTF-8 JSON bytes. The input is not changed.
    """
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def digest(value: object) -> str:
    """Return a stable SHA256 identity for a JSON-compatible value."""
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def _require(condition: bool, message: str) -> None:  # noqa: FBT001 # This internal assertion takes a computed invariant.
    """Reject a violated evidence invariant with an application-owned message."""
    if not condition:
        raise ValueError(message)


def source_index(packet: dict) -> SourceIndex:
    """Index already validated packet sources without copying their text.

    Args:
        packet: A packet produced by build_packet or validate_packet.

    Returns:
        A new mapping whose values are the original source records.
    """
    return SourceIndex(packet["sources"])


def resolve_citation(
    citation: dict,
    sources: Mapping[str, dict],
    *,
    lines: bool = False,
) -> str | list[str]:
    """Validate and resolve one explicit original-line citation.

    Args:
        citation: Source ID, exact revision, and inclusive start/end lines.
        sources: Source records indexed by their unique IDs.
        lines: Return individual physical lines instead of joined text.

    Returns:
        The exact cited source, preserving physical line endings for text.

    Raises:
        ValueError: If the source, revision, or range does not exist.
    """
    _require(type(citation) is dict, "Invalid source citation")
    source = sources.get(citation.get("source_id"))
    _require(source is not None, "Citation refers to an unknown source")
    _require(
        citation.get("revision") == source["revision"],
        "Citation has the wrong revision",
    )
    start, end = citation.get("start"), citation.get("end")
    _require(
        type(start) is int and type(end) is int,
        "Citation lines must be integers",
    )
    physical = (
        sources.physical[source["id"]]
        if isinstance(sources, SourceIndex)
        else split_source_lines(source["text"], keepends=True)
    )
    origin = source["start_line"]
    _require(
        origin <= start <= end < origin + len(physical),
        "Citation is outside supplied source",
    )
    selected = physical[start - origin : end - origin + 1]
    return selected if lines else "".join(selected)


def _reference(symbol: dict | None, sources: dict[str, dict]) -> dict | None:
    """Replace a compiler symbol buffer with an exact source reference."""
    if symbol is None:
        return None
    source_id = symbol.get("fragment_key")
    if source_id not in sources:
        matches = [
            s
            for s in sources.values()
            if s["side"] == symbol["side"]
            and s["path"] == symbol["path"]
            and s["start_line"] <= symbol["start"]
            and symbol["end"]
            < s["start_line"] + len(split_source_lines(s["text"]))
        ]
        _require(
            len(matches) == 1, "Compiler symbol has no unique source buffer"
        )
        source_id = matches[0]["id"]
    source = sources[source_id]
    ref = {
        "source_id": source_id,
        "revision": source["revision"],
        "start": symbol["start"],
        "end": symbol["end"],
        "name": symbol["name"],
        "url": symbol.get("url"),
    }
    resolve_citation(ref, sources)
    return ref


def _source_for_region(
    sources: SourceIndex, region: str, side: str
) -> str | None:
    """Select the compiler's sole source buffer for one diff region and side."""
    matches = sources.by_region.get((region, side), [])
    _require(len(matches) <= 1, "Diff region has ambiguous source buffers")
    return matches[0] if matches else None


def _row_citations(
    row: dict, diff: dict, sources: dict[str, dict]
) -> list[dict]:
    """Resolve each supplied diff row to its original source coordinates."""
    refs = []
    for side, field in (("base", "old"), ("head", "new")):
        number = row[field]
        if number is None:
            continue
        source_id = diff[f"{side}_source"]
        _require(source_id in sources, "Diff line has no supplied source")
        ref = {
            "source_id": source_id,
            "revision": sources[source_id]["revision"],
            "start": number,
            "end": number,
        }
        resolve_citation(ref, sources)
        refs.append(ref)
    return refs


def _region_changes(
    rows: list[dict], changes: list[dict], diff: dict
) -> list[str]:
    """Find compiler changes that overlap changed rows, excluding context."""
    result = []
    for change in changes:
        for field, side, number_field, tag in (
            ("before", "base", "old", "delete"),
            ("after", "head", "new", "add"),
        ):
            ref = change[field]
            if (
                ref
                and ref["source_id"] == diff[f"{side}_source"]
                and any(
                    row["tag"] == tag
                    and ref["start"] <= row[number_field] <= ref["end"]
                    for row in rows
                )
            ):
                result.append(change["id"])
                break
    return result


def _changed_citations(
    rows: list[dict], diff: dict, sources: dict[str, dict]
) -> list[dict]:
    """Bound changed source on each side without serializing candidate batches."""
    refs_by_source = defaultdict(list)
    for row in rows:
        if row["tag"] != "context":
            for ref in _row_citations(row, diff, sources):
                refs_by_source[ref["source_id"]].append(ref["start"])
    return [
        {
            "source_id": key,
            "revision": sources[key]["revision"],
            "start": min(numbers),
            "end": max(numbers),
        }
        for key, numbers in refs_by_source.items()
    ]


def _source_only_regions(
    changes: list[dict], regions: list[dict]
) -> list[dict]:
    """Account for exact-byte or structural changes absent from display hunks.

    Compiler display hunks ignore line endings. Exact source and compiler
    changes still expose such edits; they receive one collective accounting
    region rather than disappearing from writing work.
    """
    covered = {key for region in regions for key in region["change_ids"]}
    return [
        {
            "id": f"change-{change['id']}",
            "diff_id": None,
            "hunk": None,
            "change_ids": [change["id"]],
            "citations": [
                {
                    key: ref[key]
                    for key in ("source_id", "revision", "start", "end")
                }
                for ref in (change["before"], change["after"])
                if ref
            ],
        }
        for change in changes
        if change["id"] not in covered
    ]


def _project_hunks(hunks: list[dict]) -> list[dict]:
    """Retain row tags and coordinates while source stays in its own store."""
    return [
        {
            **hunk,
            "rows": [
                {key: row[key] for key in ("tag", "old", "new")}
                for row in hunk["rows"]
            ],
        }
        for hunk in hunks
    ]


def _validate_original_patch(diff: dict, sources: SourceIndex) -> None:
    """Require original-patch metadata and verify any captured alignment."""
    status = diff["patch_status"]
    patch = diff["original_patch"]
    if status in {"unavailable", "unbound"}:
        _require(
            not diff["hunks"]
            and (
                (status == "unavailable" and patch is None)
                or (status == "unbound" and isinstance(patch, str))
            ),
            "Unavailable original patch cannot contain replacement diff rows",
        )
        return
    _require(
        isinstance(patch, str), "Captured diff requires its original patch"
    )
    original = parse_file_patch(
        patch,
        (sources.get(diff["base_source"]) or {}).get("text", ""),
        (sources.get(diff["head_source"]) or {}).get("text", ""),
    )
    _require(
        _project_hunks(original) == diff["hunks"],
        "Original patch row alignment was changed",
    )


def _hunk_starts(diff: dict, hunk: dict) -> dict[str, int]:
    """Translate Git's empty-range header into the next source-line cursor."""
    starts = {"base": hunk["old_start"], "head": hunk["new_start"]}
    if diff.get("patch_status") == "captured":
        for side, count in (("base", "old_count"), ("head", "new_count")):
            if hunk[count] == 0:
                starts[side] += 1
    return starts


def _can_skip_captured_gap_walk(
    diff: dict,
    origins: dict[str, int],
    full_file_source_ids: set[str] | None,
) -> bool:
    """Allow parser coverage to replace the generic walk only for whole files.

    Args:
        diff: Persisted file diff with validated original-patch metadata.
        origins: Original first line for the supplied base and head buffers.
        full_file_source_ids: Source IDs backed by full file-evidence records.

    Returns:
        True only when supplied sources are complete files beginning at line
        one and the diff scope describes a full file.
    """
    if diff.get("patch_status") != "captured" or full_file_source_ids is None:
        return False
    source_ids = (diff["base_source"], diff["head_source"])
    return (
        all(
            source_id is None or source_id in full_file_source_ids
            for source_id in source_ids
        )
        and all(origins[side] == 1 for side in origins)
        and diff.get("scope") in {None, "full", "complete"}
    )


def _validate_diff_coverage(
    diff: dict,
    sources: SourceIndex,
    *,
    full_file_source_ids: set[str] | None = None,
) -> None:
    """Check diff source binding and any gaps not covered by patch parsing.

    Preconditions:
        Sources passed digest/revision validation and rows use known tags.
    Postconditions:
        Every source difference is present in the supplied display diff.
        Line endings remain exact in source even where display hunks omit them.

    Essential behavior:
        Captured patches skip the row/gap walk only when both supplied buffers
        are complete files starting at line one. Generated function comparisons
        still use this generic walk because they have no original file patch.
    """
    buffers, origins = {}, {}
    for side in ("base", "head"):
        source_id = diff[f"{side}_source"]
        _require(
            source_id == _source_for_region(sources, diff["region"], side),
            "Diff is detached from its supplied source buffers",
        )
        source = sources.get(source_id)
        if source:
            _require(
                source["side"] == side and source["region"] == diff["region"],
                "Diff source binding mismatch",
            )
        origins[side] = source["start_line"] if source else 1
        buffers[side] = split_source_lines(source["text"]) if source else []
    current = sources.get(diff["head_source"]) or sources.get(
        diff["base_source"]
    )
    previous = sources.get(diff["base_source"])
    _require(
        current is not None
        and diff["path"] == current["path"]
        and diff["old_path"] == (previous["path"] if previous else None),
        "Diff paths disagree with supplied source",
    )
    if _can_skip_captured_gap_walk(diff, origins, full_file_source_ids):
        return
    if diff.get("patch_status") in {"unavailable", "unbound"}:
        return
    cursor = dict(origins)
    for hunk in diff["hunks"]:
        starts = _hunk_starts(diff, hunk)
        _require(
            all(
                cursor[s] <= starts[s] <= origins[s] + len(buffers[s])
                for s in cursor
            ),
            "Invalid diff hunk ordering",
        )
        gaps = {
            s: buffers[s][cursor[s] - origins[s] : starts[s] - origins[s]]
            for s in cursor
        }
        _require(
            gaps["base"] == gaps["head"], "Diff omitted changed source lines"
        )
        cursor = dict(starts)
        for row in hunk["rows"]:
            for side, field in (("base", "old"), ("head", "new")):
                if row[field] is not None:
                    _require(
                        row[field] == cursor[side],
                        "Diff row coordinates are not contiguous",
                    )
                    cursor[side] += 1
            if row["tag"] == "context":
                _require(
                    buffers["base"][row["old"] - origins["base"]]
                    == buffers["head"][row["new"] - origins["head"]],
                    "Diff context differs between revisions",
                )
        _require(
            cursor["base"] - starts["base"] == hunk["old_count"]
            and cursor["head"] - starts["head"] == hunk["new_count"],
            "Diff hunk line counts disagree",
        )
    tails = {s: buffers[s][cursor[s] - origins[s] :] for s in cursor}
    _require(
        tails["base"] == tails["head"], "Diff omitted changed source lines"
    )


def build_packet(report: dict) -> dict:
    """Project compiler evidence without copying source into changes or diffs.

    Args:
        report: Decoded report data. It must be a current
            ``diffstory.report.v1`` compiler report with source evidence.

    Returns:
        A validated packet. Every supplied changed diff row remains present;
        only hunks with actual edits become accounting regions. Full source
        definitions remain available through change references.

    Raises:
        ValueError: If the value is not a current compiler report, it contains
            retired report markers, or revisions/source bindings are incomplete.

    Preconditions:
        ``report`` is untrusted decoded data. Current compiler reports may
        carry manual authored annotations, but must not contain retired
        generation or storage markers.
    Postconditions:
        The returned packet passes ``validate_packet``. The input report is not
        modified; retired root markers are rejected before projection.

    Essential behavior:
        Source buffers remain in one packet source list. Changes and diff rows
        refer to those buffers instead of copying source text.
    """
    _require(
        type(report) is dict and report.get("schema") == "diffstory.report.v1",
        "Compact packet requires a compiler report",
    )
    _require(
        not any(
            key in report
            for key in (
                "generation",
                "narrative_v2",
                "storage_version",
                "artifact_kind",
            )
        )
        and not ("root" in report and "tables" in report),
        "Unsupported compiler report format",
    )
    meta = copy.deepcopy(report["meta"])
    _require(
        all(
            isinstance(meta.get(f"{side}_sha"), str) and meta[f"{side}_sha"]
            for side in ("base", "head")
        ),
        "Narration requires pinned revisions",
    )
    evidence = report.get("source_evidence")
    _require(type(evidence) is dict, "Compiler source evidence is required")
    sources = [
        {
            "id": f["key"],
            "side": f["side"],
            "revision": meta[f"{f['side']}_sha"],
            "path": f["path"],
            "start_line": f["start_line"],
            "region": f["region"],
            "scope": f["scope"],
            "parse_status": f["parse_status"],
            "text": f["text"],
            "digest": f["content_digest"],
        }
        for f in evidence["fragments"]
    ]
    by_source = source_index({"sources": sources})
    changes = [
        {
            "id": c["id"],
            "kind": c["kind"],
            "label": c["label"],
            "basis": c["basis"],
            "before": _reference(c.get("before"), by_source),
            "after": _reference(c.get("after"), by_source),
        }
        for c in report["changes"]
    ]
    for original, change in zip(report["changes"], changes, strict=True):
        if (original.get("before") or {}).get("node_type") in {
            "FunctionDef",
            "AsyncFunctionDef",
        } and (original.get("after") or {}).get("node_type") in {
            "FunctionDef",
            "AsyncFunctionDef",
        }:
            # This extra view compares matched definitions, including moves.
            # It is separate from the original file patch below.
            change["comparison_hunks"] = _project_hunks(original["hunks"])
    diffs, regions = [], []
    for raw in report["raw_files"]:
        diff = {
            key: copy.deepcopy(raw[key])
            for key in (
                "id",
                "path",
                "old_path",
                "region",
                "scope",
                "before_url",
                "after_url",
                "diff_origin",
                "patch_status",
                "original_patch",
            )
        }
        diff.update(
            {
                f"{side}_source": _source_for_region(
                    by_source, raw["region"], side
                )
                for side in ("base", "head")
            }
        )
        diff["hunks"] = _project_hunks(raw["hunks"])
        for index, (hunk, original_hunk) in enumerate(
            zip(diff["hunks"], raw["hunks"], strict=True)
        ):
            rows = hunk["rows"]
            for row, original in zip(rows, original_hunk["rows"], strict=True):
                for ref in _row_citations(row, diff, by_source):
                    actual = resolve_citation(ref, by_source)
                    _require(
                        split_source_lines(actual)[0] == original["text"],
                        "Diff text disagrees with source",
                    )
            if any(row["tag"] != "context" for row in rows):
                regions.append(
                    {
                        "id": f"{raw['id']}-{index}",
                        "diff_id": raw["id"],
                        "hunk": index,
                        "citations": _changed_citations(rows, diff, by_source),
                        "change_ids": _region_changes(rows, changes, diff),
                    }
                )
        diffs.append(diff)
    regions.extend(_source_only_regions(changes, regions))
    packet = {
        "version": PACKET_VERSION,
        "meta": meta,
        "sources": sources,
        "file_evidence": [
            {
                side: {
                    key: record[side][key]
                    for key in ("state", "path", "coverage", "reason")
                    if key in record[side]
                }
                for side in ("base", "head")
            }
            for record in evidence["file_evidence"]
        ],
        "changes": changes,
        "diffs": diffs,
        "regions": regions,
        "groups": [
            {
                key: copy.deepcopy(g[key])
                for key in (
                    "id",
                    "title",
                    "path",
                    "change_ids",
                    "prerequisites",
                    "theme",
                    "test_links",
                )
            }
            for g in report["groups"]
        ],
        "dependencies": copy.deepcopy(report["edges"]),
        "analysis": {
            "tests": [
                {"id": test["id"], "source": _reference(test, by_source)}
                for test in report["tests"]
            ],
            "symbol_edges": copy.deepcopy(report["symbol_edges"]),
            "cycles": [list(cycle) for cycle in report["cycles"]],
            "unresolved": copy.deepcopy(report["unresolved"]),
            "method": copy.deepcopy(report["method"]),
        },
        "warnings": copy.deepcopy(report["warnings"]),
        "notes": copy.deepcopy(report["notes"]),
    }
    validate_packet(packet)
    return packet


def _validate_function_comparison(change: dict, sources: SourceIndex) -> None:
    """Keep a generated comparison tied to its two complete definitions."""
    comparison = change.get("comparison_hunks")
    if comparison is None:
        return
    _require(
        change["before"] and change["after"],
        "Function comparison needs both definitions",
    )
    # Reuse the linear row/gap check on just these two definitions. Checking
    # a saved comparison must not run a diff algorithm again.
    fragments = []
    for field in ("before", "after"):
        ref = change[field]
        fragments.append(
            {
                **sources[ref["source_id"]],
                "region": "function-comparison",
                "start_line": ref["start"],
                "text": resolve_citation(ref, sources),
            }
        )
    function_sources = source_index({"sources": fragments})
    diff = {
        "region": "function-comparison",
        "path": sources[change["after"]["source_id"]]["path"],
        "old_path": sources[change["before"]["source_id"]]["path"],
        "base_source": change["before"]["source_id"],
        "head_source": change["after"]["source_id"],
        "hunks": comparison,
    }
    for hunk in comparison:
        for row in hunk["rows"]:
            _require(
                (row["old"] is None) == (row["tag"] == "add")
                and (row["new"] is None) == (row["tag"] == "delete"),
                "Invalid function comparison row",
            )
            _row_citations(row, diff, function_sources)
    _validate_diff_coverage(diff, function_sources)


def validate_packet(packet: dict) -> None:  # noqa: C901, PLR0912 # Keep the direct source, diff, and accounting boundary together.
    """Validate packet schema and source-to-diff relationships.

    Args:
        packet: Untrusted decoded packet. Strict Pydantic models check field
            types, primitive ranges, identifiers, revisions, and closed record
            shapes before relational checks run.

    Returns:
        ``None`` when the packet is internally consistent.

    Raises:
        ValueError: If schema or primitive fields are invalid, or if source
            digests, revisions, references, availability, or accounting fail.

    Preconditions:
        The value is a decoded JSON-like mapping. It may be untrusted.
    Postconditions:
        All citations resolve to supplied source and exact revisions; source
        buffers match their digests; complete diffs, file availability, and
        changed-region accounting agree. The packet is not modified.

    Essential behavior:
        Recomputes changed-hunk membership from the complete supplied diffs;
        deleting a region while retaining its diff is rejected. Pydantic owns
        primitive checks; this function owns cross-record invariants.
    """
    Packet.model_validate(packet)
    sources = source_index(packet)
    _require(len(sources) == len(packet["sources"]), "Duplicate source IDs")
    _validate_file_evidence(packet, sources)
    full_file_paths = {
        (side, record[side]["path"])
        for record in packet["file_evidence"]
        for side in ("base", "head")
        if record[side]["state"] == "supplied"
        and record[side]["coverage"] == "full"
    }
    full_file_source_ids = {
        source["id"]
        for source in sources.values()
        if (source["side"], source["path"]) in full_file_paths
    }
    for source in sources.values():
        _require(
            source["revision"] == packet["meta"][f"{source['side']}_sha"],
            "Source revision mismatch",
        )
        _require(
            hashlib.sha256(source["text"].encode()).hexdigest()
            == source["digest"],
            "Source digest mismatch",
        )
    changes = {c["id"]: c for c in packet["changes"]}
    _require(len(changes) == len(packet["changes"]), "Duplicate change IDs")
    for change in changes.values():
        for field in ("before", "after"):
            if change[field]:
                resolve_citation(change[field], sources)
                _require(
                    sources[change[field]["source_id"]]["side"]
                    == ("base" if field == "before" else "head"),
                    "Change source side mismatch",
                )
        _validate_function_comparison(change, sources)
    for test in packet.get("analysis", {}).get("tests", []):
        resolve_citation(test["source"], sources)
    expected = {}
    diff_ids = set()
    for diff in packet["diffs"]:
        _require(diff["id"] not in diff_ids, "Duplicate diff IDs")
        diff_ids.add(diff["id"])
        for index, hunk in enumerate(diff["hunks"]):
            changed = False
            for row in hunk["rows"]:
                tag = row["tag"]
                _require(
                    (row["old"] is None) == (tag == "add")
                    and (row["new"] is None) == (tag == "delete"),
                    "Invalid diff row coordinates",
                )
                _row_citations(row, diff, sources)
                changed = changed or tag != "context"
            if changed:
                key = f"{diff['id']}-{index}"
                expected[key] = {
                    "id": key,
                    "diff_id": diff["id"],
                    "hunk": index,
                    "change_ids": _region_changes(
                        hunk["rows"], packet["changes"], diff
                    ),
                    "citations": _changed_citations(
                        hunk["rows"], diff, sources
                    ),
                }
        _validate_original_patch(diff, sources)
        _validate_diff_coverage(
            diff, sources, full_file_source_ids=full_file_source_ids
        )
    source_regions = defaultdict(dict)
    for source in sources.values():
        _require(
            source["side"] not in source_regions[source["region"]],
            "Ambiguous source region",
        )
        source_regions[source["region"]][source["side"]] = source["text"]
    expected_diff_regions = {
        key
        for key, sides in source_regions.items()
        if sides.get("base", "") != sides.get("head", "")
    }
    _require(
        {d["region"] for d in packet["diffs"]} == expected_diff_regions
        and len(packet["diffs"]) == len(expected_diff_regions),
        "Supplied diff inventory mismatch",
    )
    for region in _source_only_regions(
        packet["changes"], list(expected.values())
    ):
        expected[region["id"]] = region
    regions = {r["id"]: r for r in packet["regions"]}
    _require(
        len(regions) == len(packet["regions"])
        and set(regions) == set(expected),
        "Changed-region coverage mismatch",
    )
    for key, region in regions.items():
        _require(
            region == expected[key],
            "Changed-region binding mismatch",
        )
        _require(
            set(region["change_ids"]) <= set(changes),
            "Region refers to an unknown change",
        )
        for ref in region["citations"]:
            resolve_citation(ref, sources)


def _validate_file_evidence(packet: dict, sources: SourceIndex) -> None:
    """Bind availability and full-file claims to the buffers actually stored."""
    by_path = defaultdict(list)
    for source in sources.values():
        by_path[(source["side"], source["path"])].append(source)
    seen = set()
    supplied = set()
    for record in packet["file_evidence"]:
        for side in ("base", "head"):
            value = record[side]
            key = (side, value["path"])
            _require(key not in seen, "Duplicate file evidence path")
            seen.add(key)
            if value["state"] == "supplied":
                supplied.add(key)
                _require(bool(by_path[key]), "Supplied file has no source")
                if value["coverage"] == "full":
                    _require(
                        len(by_path[key]) == 1
                        and by_path[key][0]["start_line"] == 1
                        and by_path[key][0]["scope"]
                        in {None, "full", "complete"},
                        "Full file evidence must start at original line one",
                    )
    _require(
        supplied == set(by_path), "Source availability disagrees with buffers"
    )
    _require(
        packet["meta"]["source_bytes"]
        == sum(len(source["text"].encode()) for source in sources.values()),
        "Supplied source byte count disagrees with buffers",
    )


def packet_identity(packet: dict) -> str:
    """Bind caches to exact evidence while ignoring only capture timestamps."""
    meta = {k: v for k, v in packet["meta"].items() if k != "captured_at"}
    return digest({**packet, "meta": meta})


def item_sizes(packet: dict) -> dict[str, int]:
    """Measure each completed source/region once for incremental batch packing.

    Args:
        packet: A validated compact packet.

    Returns:
        Encoded byte lengths keyed by unique source and accounting-region IDs.
        Callers add these lengths without serializing growing candidate batches.
    """
    return {
        item["id"]: len(canonical_bytes(item))
        for field in ("sources", "regions")
        for item in packet[field]
    }
