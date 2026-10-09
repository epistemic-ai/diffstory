"""Deterministic evidence accounting, validation, packing, and story planning.

This module accepts trusted compiler reports. It does not import the compiler,
providers, or runtime. Public validators also serve later persisted boundaries.
Request serializers must include the real instructions, JSON schema, context,
and adapter envelope. Packing is an estimate; adapters must recheck before send.
"""

from __future__ import annotations

import hashlib
import heapq
import json
import math
import re
from bisect import bisect_right
from collections import Counter
from collections import defaultdict
from dataclasses import dataclass
from dataclasses import field
from difflib import SequenceMatcher
from types import MappingProxyType
from typing import TYPE_CHECKING
from typing import TypeVar
from weakref import WeakKeyDictionary

from pydantic import BaseModel
from pydantic import Field

from diffstory.narrative_contracts import BatchSpec
from diffstory.narrative_contracts import BatchStage
from diffstory.narrative_contracts import BufferBinding
from diffstory.narrative_contracts import ChangeBinding
from diffstory.narrative_contracts import ClaimRef
from diffstory.narrative_contracts import Classification
from diffstory.narrative_contracts import Contract
from diffstory.narrative_contracts import DetailResult
from diffstory.narrative_contracts import EvidenceLedger
from diffstory.narrative_contracts import EvidenceLimitation
from diffstory.narrative_contracts import EvidenceUnit
from diffstory.narrative_contracts import FileAvailability
from diffstory.narrative_contracts import GroupBinding
from diffstory.narrative_contracts import GroupSpec
from diffstory.narrative_contracts import Natural
from diffstory.narrative_contracts import Obligation
from diffstory.narrative_contracts import PlanningAtom
from diffstory.narrative_contracts import PlanningGroup
from diffstory.narrative_contracts import Positive
from diffstory.narrative_contracts import Prerequisite
from diffstory.narrative_contracts import SourceBuffer
from diffstory.narrative_contracts import SourceSpan
from diffstory.narrative_contracts import SpanOwnership
from diffstory.narrative_contracts import SpanRef
from diffstory.narrative_contracts import StoryCard
from diffstory.narrative_contracts import StoryChapter
from diffstory.narrative_contracts import StoryPlan
from diffstory.source_lines import split_source_lines
from diffstory.usage import ModelCapacity
from diffstory.usage import estimate_input

if TYPE_CHECKING:
    from collections.abc import Callable
    from collections.abc import Mapping
    from collections.abc import Sequence


def _json_default(value: object) -> object:
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json")
    msg = "Unsupported canonical value"
    raise TypeError(msg)


def canonical_bytes(value: object) -> bytes:
    """Encode identity material without whitespace or dictionary order effects."""
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
        default=_json_default,
    ).encode("utf-8")


def content_digest(value: bytes) -> str:
    """Return the exact SHA256 of a UTF-8 buffer or serialized request."""
    return hashlib.sha256(value).hexdigest()


def semantic_digest(*parts: object) -> str:
    """Hash versioned canonical content; callers exclude current bindings."""
    return content_digest(canonical_bytes(parts))


def _require(condition: bool, reason: str) -> None:  # noqa: FBT001 # Assertion predicate, not an option.
    if not condition:
        raise ValueError(reason)


def _unique(values: Sequence[str], reason: str) -> None:
    _require(len(set(values)) == len(values), reason)


def _line_bounds(text: str) -> tuple[list[str], list[int]]:
    lines = split_source_lines(text, keepends=True)
    bounds = [0]
    for line in lines:
        bounds.append(bounds[-1] + len(line.encode("utf-8")))
    return lines, bounds


def _pieces(
    raw: bytes, start: int, end: int, maximum: int
) -> list[tuple[int, int]]:
    result = []
    while start < end:
        stop = min(start + maximum, end)
        while stop < end and raw[stop] & 0xC0 == 0x80:  # noqa: PLR2004 # UTF-8 continuation-byte bits.
            stop -= 1
        _require(stop > start, "span_limit_too_small")
        result.append((start, stop))
        start = stop
    return result


@dataclass
class _Fragment:
    record: dict
    availability: FileAvailability
    key: str
    raw: bytes
    lines: list[str]
    bounds: list[int]


@dataclass
class _Change:
    record: dict
    key: str
    group_key: str
    classification: Classification
    # Current fragment key and its fragment-relative byte interval.
    ranges: dict[str, tuple[str, int, int]]
    limitations: tuple[EvidenceLimitation, ...]


def _projection(
    report: dict,
) -> tuple[list[_Fragment], tuple[FileAvailability, ...]]:
    evidence = report.get("source_evidence")
    _require(
        isinstance(evidence, dict)
        and evidence.get("schema") == "diffstory.source-evidence.v1",
        "original_source_projection_required",
    )
    states = {}
    fragment_states = {}
    for pair in evidence["file_evidence"]:
        for side, opposite in (("base", "head"), ("head", "base")):
            source = pair[side]
            state = FileAvailability(
                side=side,
                path=source["path"],
                counterpart_path=pair[opposite]["path"],
                state=source["state"],
                coverage=source.get("coverage"),
                reason=source.get("reason"),
                full_source=source["full_source"],
                full_and_parsed=source["full_and_parsed"],
            )
            _require((side, state.path) not in states, "duplicate_file_state")
            states[side, state.path] = state
            for key in source["fragment_keys"]:
                _require(
                    key not in fragment_states, "duplicate_fragment_binding"
                )
                fragment_states[key] = state
    fragments = []
    occurrences = Counter()
    records = sorted(
        evidence["fragments"],
        key=lambda r: (r["side"], r["path"], r["start_line"], r["key"]),
    )
    _unique([record["key"] for record in records], "duplicate_fragment_key")
    _require(
        set(fragment_states) == {r["key"] for r in records},
        "fragment_state_mismatch",
    )
    for record in records:
        raw = record["text"].encode("utf-8")
        _require(
            content_digest(raw) == record["content_digest"],
            "fragment_digest_mismatch",
        )
        state = fragment_states[record["key"]]
        _require(
            (state.side, state.path) == (record["side"], record["path"]),
            "fragment_path_mismatch",
        )
        seed = semantic_digest(
            "buffer.v2", state, record["parse_status"], record["text"]
        )
        ordinal = occurrences[seed]
        occurrences[seed] += 1
        lines, bounds = _line_bounds(record["text"])
        fragments.append(
            _Fragment(
                record,
                state,
                semantic_digest(seed, ordinal),
                raw,
                lines,
                bounds,
            )
        )
    return fragments, tuple(states[key] for key in sorted(states))


def _comparison_limits(
    record: dict,
    ranges: dict,
    fragments: dict[str, _Fragment],
    states: dict[tuple[str, str], FileAvailability],
) -> tuple[EvidenceLimitation, ...]:
    result = []
    for side, name, other in (
        ("base", "before", "after"),
        ("head", "after", "before"),
    ):
        source = record.get(name)
        counterpart = record.get(other)
        path = (
            source["path"]
            if source
            else states[
                (counterpart["side"], counterpart["path"])
            ].counterpart_path
        )
        state = states[side, path]
        code = None
        reason = None
        if state.state == "unavailable":
            code, reason = "unavailable", state.reason
        elif state.coverage == "partial":
            code = "partial" if side in ranges else "unpaired_excerpt"
            reason = "Only the supplied excerpt is available for comparison."
        elif state.state == "supplied" and not state.full_and_parsed:
            failed = any(
                f.record["parse_status"] == "failed"
                for f in fragments.values()
                if f.availability == state
            )
            code = "parse_failed" if failed else "text_only"
            reason = "The supplied file is not fully parsed."
        elif not source and state.state == "supplied":
            code, reason = (
                "unmatched_definition",
                "The compiler did not bind a counterpart definition; the file is supplied.",
            )
        if code:
            result.append(
                EvidenceLimitation(
                    code=code, side=side, path=path, reason=reason
                )
            )
        if source:
            fragment = fragments[source["fragment_key"]]
            if (
                fragment.record["parse_status"] == "failed"
                and code != "parse_failed"
            ):
                result.append(
                    EvidenceLimitation(
                        code="parse_failed",
                        side=side,
                        path=path,
                        reason="The supplied fragment did not parse.",
                    )
                )
    return tuple(result)


def _changes(
    report: dict,
    fragments: list[_Fragment],
    availability: tuple[FileAvailability, ...],
) -> tuple[list[_Change], dict[str, str]]:
    by_fragment = {fragment.record["key"]: fragment for fragment in fragments}
    states = {(state.side, state.path): state for state in availability}
    group_keys = {
        g["id"]: semantic_digest("group.v2", g["path"], g["theme"])
        for g in report["groups"]
    }
    _unique(list(group_keys.values()), "ambiguous_group_identity")
    owners = {}
    for group in report["groups"]:
        for change_id in group["change_ids"]:
            _require(change_id not in owners, "change_has_multiple_groups")
            owners[change_id] = group_keys[group["id"]]
    _unique([r["id"] for r in report["changes"]], "duplicate_change_id")
    _require(
        set(owners) == {r["id"] for r in report["changes"]},
        "change_group_membership_mismatch",
    )
    result = []
    occurrences = Counter()
    # Relative occurrence ranks distinguish identical records without hashing coordinates.
    records = sorted(
        report["changes"],
        key=lambda c: tuple(
            (c.get(n) or {}).get("start", 0) for n in ("before", "after")
        ),
    )
    for record in records:
        ranges = {}
        identity = []
        for side, name in (("base", "before"), ("head", "after")):
            source = record.get(name)
            if not source:
                identity.append(None)
                continue
            _require(
                source.get("fragment_key") in by_fragment,
                "source_fragment_binding_required",
            )
            fragment = by_fragment[source["fragment_key"]]
            _require(
                (side, source["path"])
                == (fragment.record["side"], fragment.record["path"]),
                "source_binding_mismatch",
            )
            start = source["start"] - fragment.record["start_line"]
            stop = source["end"] - fragment.record["start_line"] + 1
            _require(
                0 <= start <= stop <= len(fragment.lines),
                "source_line_range_mismatch",
            )
            lo, hi = fragment.bounds[start], fragment.bounds[stop]
            ranges[side] = (source["fragment_key"], lo, hi)
            identity.append(
                (
                    side,
                    source["path"],
                    source["name"],
                    source.get("node_type"),
                    fragment.raw[lo:hi].decode("utf-8"),
                )
            )
        _require(bool(ranges), "change_has_no_source_binding")
        limits = _comparison_limits(record, ranges, by_fragment, states)
        classification = Classification(
            kind=record["kind"],
            label=record["label"],
            basis=record["basis"],
            uncertainty=(record["basis"],)
            if record["kind"]
            in {"observed_base", "observed_head", "moved_modified"}
            else (),
            before_name=(record.get("before") or {}).get("name"),
            after_name=(record.get("after") or {}).get("name"),
            before_is_test=(record.get("before") or {}).get("is_test"),
            after_is_test=(record.get("after") or {}).get("is_test"),
        )
        seed = semantic_digest("change.v2", identity, classification, limits)
        ordinal = occurrences[seed]
        occurrences[seed] += 1
        result.append(
            _Change(
                record,
                semantic_digest(seed, ordinal),
                owners[record["id"]],
                classification,
                ranges,
                limits,
            )
        )
    return result, group_keys


def _make_spans(
    fragments: list[_Fragment], changes: list[_Change], maximum: int
) -> tuple[list[SourceBuffer], list[SourceSpan]]:
    owners = defaultdict(list)
    for change in changes:
        for fragment_key, lo, hi in change.ranges.values():
            owners[fragment_key].append((lo, hi, change.key))
    buffers = []
    spans = []
    occurrences = Counter()
    for fragment in fragments:
        record = fragment.record
        fragment_spans = []
        intervals = [
            piece
            for lo, hi in zip(
                fragment.bounds, fragment.bounds[1:], strict=False
            )
            for piece in _pieces(fragment.raw, lo, hi, maximum)
        ]
        if not intervals:
            intervals = [(0, 0)]
        for lo, hi in intervals:
            text = fragment.raw[lo:hi].decode("utf-8")
            change_keys = sorted(
                key
                for start, end, key in owners[record["key"]]
                if start <= lo and hi <= end
            )
            seed = semantic_digest(
                "span.v2", record["side"], record["path"], text, change_keys
            )
            ordinal = occurrences[seed]
            occurrences[seed] += 1
            line = record["start_line"] + max(
                0, bisect_right(fragment.bounds, lo) - 1
            )
            span = SourceSpan(
                id=semantic_digest(seed, ordinal),
                buffer_key=fragment.key,
                side=record["side"],
                path=record["path"],
                byte_start=lo,
                byte_end=hi,
                line_start=line,
                line_end=line,
                content_digest=content_digest(fragment.raw[lo:hi]),
                text=text,
            )
            spans.append(span)
            fragment_spans.append(span.id)
        buffers.append(
            SourceBuffer(
                key=fragment.key,
                side=record["side"],
                path=record["path"],
                content_digest=record["content_digest"],
                byte_length=len(fragment.raw),
                span_ids=tuple(fragment_spans),
                coverage=fragment.availability.coverage,
                parse_status=record["parse_status"],
            )
        )
    return buffers, spans


def _local_runs(
    change: _Change, fragments: dict[str, _Fragment]
) -> list[list[tuple[str, int, int]]]:
    texts = []
    for side in ("base", "head"):
        if side in change.ranges:
            key, lo, hi = change.ranges[side]
            lines, bounds = _line_bounds(
                fragments[key].raw[lo:hi].decode("utf-8")
            )
            texts.append((key, lo, lines, bounds))
        else:
            texts.append((None, 0, [], [0]))
    result = []
    for tag, a, b, c, d in SequenceMatcher(
        None, texts[0][2], texts[1][2], autojunk=False
    ).get_opcodes():
        if tag == "equal":
            continue
        run = []
        for text, start, stop in ((texts[0], a, b), (texts[1], c, d)):
            key, offset, _, bounds = text
            if key is not None and start < stop:
                run.append(
                    (key, offset + bounds[start], offset + bounds[stop])
                )
        result.append(run)
    return result


def _obligation(
    kind: str,
    change_key: str | None,
    ids: tuple[str, ...],
    limits: tuple[EvidenceLimitation, ...] = (),
) -> Obligation:
    return Obligation(
        id=semantic_digest("obligation.v2", kind, change_key, ids, limits),
        kind=kind,
        change_key=change_key,
        required_span_ids=ids,
        limitations=limits,
    )


def _unit(  # noqa: PLR0913 # Keep explicit evidence, serializer, and capacity inputs.
    group_key: str | None,
    change_key: str | None,
    classification: Classification | None,
    spans: tuple[SourceSpan, ...],
    obligations: tuple[Obligation, ...],
    *,
    availability: tuple[FileAvailability, ...] = (),
) -> EvidenceUnit:
    return EvidenceUnit(
        id=_unit_digest(
            group_key,
            change_key,
            classification,
            spans,
            obligations,
            availability,
        ),
        group_key=group_key,
        change_key=change_key,
        classification=classification,
        spans=spans,
        obligations=obligations,
        availability=availability,
    )


def _unit_digest(  # noqa: PLR0913 # Same identity inputs as the unit constructor.
    group_key: str | None,
    change_key: str | None,
    classification: Classification | None,
    spans: tuple[SourceSpan, ...],
    obligations: tuple[Obligation, ...],
    availability: tuple[FileAvailability, ...],
) -> str:
    return semantic_digest(
        "unit.v2",
        group_key,
        change_key,
        classification,
        tuple(span.id for span in spans),
        obligations,
        availability,
    )


def _change_units(
    change: _Change,
    fragments: dict[str, _Fragment],
    spans_by_fragment: dict[str, list[SourceSpan]],
    maximum: int,
) -> list[EvidenceUnit]:
    owned = []
    for key, lo, hi in change.ranges.values():
        owned.extend(
            s
            for s in spans_by_fragment[key]
            if lo <= s.byte_start and s.byte_end <= hi
        )
    span_map = {s.id: s for s in owned}
    obligations = []
    # Structural support uses one anchor from each bound side, even for moves
    # with an empty local diff. Remaining source still gets dispatch ownership.
    anchors = tuple(
        next(s.id for s in owned if s.side == side)
        for side in ("base", "head")
        if side in change.ranges
    )
    obligations.append(_obligation("structure", change.key, anchors))
    for run in _local_runs(change, fragments):
        obligations.extend(
            _obligation("changed_region", change.key, ids)
            for ids in _bounded_run_spans(run, spans_by_fragment, maximum)
        )
    if change.limitations:
        obligations.append(
            _obligation("limitation", change.key, anchors, change.limitations)
        )
    units = []
    assigned = set()
    pending = []
    pending_ids = []
    for obligation in obligations:
        combined = list(
            dict.fromkeys((*pending_ids, *obligation.required_span_ids))
        )
        if pending and len(combined) > maximum:
            units.append(
                _unit(
                    change.group_key,
                    change.key,
                    change.classification,
                    tuple(span_map[key] for key in pending_ids),
                    tuple(pending),
                )
            )
            pending, pending_ids = [], []
            combined = list(obligation.required_span_ids)
        pending.append(obligation)
        pending_ids = combined
        assigned.update(obligation.required_span_ids)
    context = [span for span in owned if span.id not in assigned]
    if pending:
        room = maximum - len(pending_ids)
        pending_ids.extend(span.id for span in context[:room])
        context = context[room:]
        units.append(
            _unit(
                change.group_key,
                change.key,
                change.classification,
                tuple(span_map[key] for key in pending_ids),
                tuple(pending),
            )
        )
    units.extend(
        _unit(
            change.group_key,
            change.key,
            change.classification,
            tuple(context[start : start + maximum]),
            (),
        )
        for start in range(0, len(context), maximum)
    )
    return units


def _bounded_run_spans(
    run: list[tuple[str, int, int]],
    spans_by_fragment: dict[str, list[SourceSpan]],
    maximum: int,
) -> list[tuple[str, ...]]:
    """Split only at span boundaries and keep comparison-side anchors together."""
    sides = [
        tuple(
            s.id
            for s in spans_by_fragment[key]
            if lo <= s.byte_start and s.byte_end <= hi
        )
        for key, lo, hi in run
    ]
    if len(sides) == 1:
        return [
            sides[0][start : start + maximum]
            for start in range(0, len(sides[0]), maximum)
        ]
    result = []
    pending: tuple[str, ...] = ()
    for index in range(max(len(side) for side in sides)):
        # Unequal runs retain the last opposite-side anchor. This is shared
        # evidence, not an assertion that the two pieces have equal meaning.
        pair = tuple(side[min(index, len(side) - 1)] for side in sides)
        combined = tuple(dict.fromkeys((*pending, *pair)))
        if len(combined) > maximum:
            result.append(pending)
            pending = pair
        else:
            pending = combined
    if pending:
        result.append(pending)
    return result


def _global_changed_spans(
    fragments: list[_Fragment], spans: list[SourceSpan]
) -> set[str]:
    regions = defaultdict(dict)
    for fragment in fragments:
        regions[fragment.record["region"]][fragment.record["side"]] = fragment
    by_buffer = defaultdict(list)
    for span in spans:
        by_buffer[span.buffer_key].append(span)
    changed = set()
    for pair in regions.values():
        before, after = pair.get("base"), pair.get("head")
        for tag, a, b, c, d in SequenceMatcher(
            None,
            before.lines if before else [],
            after.lines if after else [],
            autojunk=False,
        ).get_opcodes():
            if tag == "equal":
                continue
            for fragment, lo, hi in ((before, a, b), (after, c, d)):
                if fragment and lo < hi:
                    changed.update(
                        s.id
                        for s in by_buffer[fragment.key]
                        if fragment.bounds[lo] <= s.byte_start
                        and s.byte_end <= fragment.bounds[hi]
                    )
    return changed


def build_evidence_ledger(
    report: dict, *, max_span_bytes: int = 4096, max_unit_spans: int = 16
) -> EvidenceLedger:
    """Account for every original buffer and derive bounded local obligations.

    Original UTF-8 bytes, line endings, and empty buffers are retained. Source
    intervals may be shared by changes. Duplicate content uses occurrence ranks
    within the same semantic identity, never legacy IDs or absolute positions.
    A changed byte without compiler ownership gets an explicit limitation.

    Raises:
        ValueError: If original evidence or explicit bindings are missing.
    """
    _require(
        type(max_span_bytes) is int and max_span_bytes >= 4,  # noqa: PLR2004 # Maximum UTF-8 character width.
        "invalid_span_limit",
    )
    _require(
        type(max_unit_spans) is int and max_unit_spans >= 2,  # noqa: PLR2004 # One anchor per comparison side.
        "invalid_unit_limit",
    )
    fragments, availability = _projection(report)
    changes, group_keys = _changes(report, fragments, availability)
    buffers, spans = _make_spans(fragments, changes, max_span_bytes)
    by_fragment = {f.record["key"]: f for f in fragments}
    buffer_to_fragment = {f.key: f.record["key"] for f in fragments}
    spans_by_fragment = defaultdict(list)
    for span in spans:
        spans_by_fragment[buffer_to_fragment[span.buffer_key]].append(span)
    units = []
    for change in changes:
        units.extend(
            _change_units(
                change, by_fragment, spans_by_fragment, max_unit_spans
            )
        )
    explained = {
        key
        for unit in units
        for obligation in unit.obligations
        for key in obligation.required_span_ids
    }
    # Each compiler change has a structural obligation. Its full source is
    # accounted for across its units even when an identical move has no local
    # byte edits. Changed local runs still have their own required obligations.
    explained.update(span.id for unit in units for span in unit.spans)
    residual = _global_changed_spans(fragments, spans) - explained
    assigned = {s.id for unit in units for s in unit.spans}
    # Residual byte differences may fall outside normalized compiler ranges.
    # They remain explicit limitations, not invented compiler classifications.
    for span in spans:
        if span.id in residual:
            limit = EvidenceLimitation(
                code="unclassified_changed_source",
                side=span.side,
                path=span.path,
                reason="Original changed bytes lack a local compiler change obligation.",
            )
            obligation = _obligation("limitation", None, (span.id,), (limit,))
            units.append(_unit(None, None, None, (span,), (obligation,)))
            assigned.add(span.id)
    for fragment_spans in spans_by_fragment.values():
        context = [s for s in fragment_spans if s.id not in assigned]
        units.extend(
            _unit(
                None,
                None,
                None,
                tuple(context[start : start + max_unit_spans]),
                (),
            )
            for start in range(0, len(context), max_unit_spans)
        )
    # Include both file states in each item, including unchanged-context items.
    # Missing source then cannot acquire an implied absence during batching.
    state_map = {(s.side, s.path): s for s in availability}
    rebound = []
    for unit in units:
        relevant = {(span.side, span.path) for span in unit.spans}
        relevant.update(
            (
                "head" if side == "base" else "base",
                state_map[side, path].counterpart_path,
            )
            for side, path in tuple(relevant)
        )
        rebound.append(
            _unit(
                unit.group_key,
                unit.change_key,
                unit.classification,
                unit.spans,
                unit.obligations,
                availability=tuple(state_map[key] for key in sorted(relevant)),
            )
        )
    units = rebound
    units.sort(key=lambda u: u.id)
    owners = defaultdict(list)
    for unit in units:
        for span in unit.spans:
            owners[span.id].append(unit.id)
    groups = tuple(
        sorted(
            (
                GroupSpec(
                    key=group_keys[g["id"]],
                    path=g["path"],
                    theme=g["theme"],
                    unit_keys=tuple(
                        u.id
                        for u in units
                        if u.group_key == group_keys[g["id"]]
                    ),
                    prerequisites=tuple(
                        sorted(group_keys[key] for key in g["prerequisites"])
                    ),
                )
                for g in report["groups"]
            ),
            key=lambda g: g.key,
        )
    )
    key = semantic_digest(
        "ledger.v2",
        sorted(b.key for b in buffers),
        sorted(s.id for s in spans),
        [u.id for u in units],
        groups,
        availability,
        report["stats"]["test_runs"],
    )
    ledger = EvidenceLedger(
        id=key,
        buffers=tuple(buffers),
        bindings=tuple(
            BufferBinding(
                buffer_key=f.key,
                fragment_key=f.record["key"],
                start_line=f.record["start_line"],
                region=f.record["region"],
            )
            for f in fragments
        ),
        spans=tuple(spans),
        units=tuple(units),
        ownership=tuple(
            SpanOwnership(
                span_id=s.id,
                unit_keys=tuple(owners[s.id]),
                shared=len(owners[s.id]) > 1,
            )
            for s in spans
        ),
        availability=availability,
        changes=tuple(
            ChangeBinding(
                change_key=c.key,
                change_id=c.record["id"],
                group_key=c.group_key,
            )
            for c in changes
        ),
        groups=groups,
        group_bindings=tuple(
            GroupBinding(group_key=group_keys[g["id"]], group_id=g["id"])
            for g in report["groups"]
        ),
        compiler_test_runs=report["stats"]["test_runs"],
    )
    validate_evidence_ledger(ledger)
    return ledger


_INDEX_AUTHORITY: WeakKeyDictionary = WeakKeyDictionary()
_INDEX_MAP_NAMES = (
    "buffers",
    "spans",
    "units",
    "bindings",
    "groups",
    "unit_span_ids",
)


@dataclass(frozen=True, init=False, eq=False)
class LedgerIndex:
    """Read-only lookup context for one fully validated frozen ledger instance.

    Construction revalidates every contract and all semantic/ownership checks.
    Maps hold that validated copy, not unchecked caller objects. Reuse requires
    the original ledger instance and unchanged top-level field bindings. Frozen
    contracts prohibit nested mutation; model_copy returns a distinct instance
    and cannot borrow this context, even when its semantic digest is unchanged.
    This is an explicit caller-owned context, not a global or digest-keyed cache.
    """

    _ledger: EvidenceLedger = field(repr=False)
    _state: tuple[object, ...] = field(repr=False)
    buffers: Mapping[str, SourceBuffer]
    spans: Mapping[str, SourceSpan]
    units: Mapping[str, EvidenceUnit]
    bindings: Mapping[str, BufferBinding]
    groups: Mapping[str, GroupSpec]
    unit_span_ids: Mapping[str, frozenset[str]]

    def __init__(self, ledger: EvidenceLedger) -> None:
        """Validate before exposing immutable maps, including on direct calls."""
        validated = EvidenceLedger.model_validate(ledger)
        buffers = {item.key: item for item in validated.buffers}
        spans = {item.id: item for item in validated.spans}
        units = {item.id: item for item in validated.units}
        bindings = {item.buffer_key: item for item in validated.bindings}
        _validate_evidence_ledger(validated, buffers, spans, bindings)
        object.__setattr__(self, "_ledger", ledger)
        object.__setattr__(self, "_state", _ledger_state(ledger))
        for name, values in (
            ("buffers", buffers),
            ("spans", spans),
            ("units", units),
            ("bindings", bindings),
            ("groups", {item.key: item for item in validated.groups}),
            (
                "unit_span_ids",
                {
                    unit.id: frozenset(s.id for s in unit.spans)
                    for unit in validated.units
                },
            ),
        ):
            object.__setattr__(self, name, MappingProxyType(values))
        _INDEX_AUTHORITY[self] = (
            ledger,
            content_digest(canonical_bytes(ledger)),
            content_digest(self._map_bytes()),
            tuple(getattr(self, name) for name in _INDEX_MAP_NAMES),
        )

    def _map_bytes(self) -> bytes:
        return canonical_bytes(
            (
                *tuple(
                    dict(getattr(self, name))
                    for name in (
                        "buffers",
                        "spans",
                        "units",
                        "bindings",
                        "groups",
                    )
                ),
                tuple(
                    sorted(
                        (key, tuple(sorted(value)))
                        for key, value in self.unit_span_ids.items()
                    )
                ),
            )
        )

    def validated_ledger(self) -> EvidenceLedger:
        """Check deep current authority once at a new derivation boundary.

        Normal per-citation lookup keeps its constant-time check. This stronger
        boundary also rejects copied indexes and nested bypass mutation of
        frozen models or cached maps. Its immutable witness is not on the index.
        """
        authority = _INDEX_AUTHORITY.get(self)
        _require(authority is not None, "unregistered_ledger_index")
        ledger, snapshot, maps, bindings = authority
        self.check_ledger(ledger)
        _require(
            all(
                old is getattr(self, name)
                for name, old in zip(_INDEX_MAP_NAMES, bindings, strict=True)
            )
            and content_digest(canonical_bytes(ledger)) == snapshot
            and content_digest(self._map_bytes()) == maps,
            "ledger_index_mutated",
        )
        return ledger

    def check_ledger(self, ledger: EvidenceLedger) -> None:
        """Reject another ledger or rebound current source fields in constant time."""
        _require(
            ledger is self._ledger
            and all(
                old is current
                for old, current in zip(
                    self._state, _ledger_state(ledger), strict=True
                )
            ),
            "ledger_index_mismatch",
        )


def _ledger_state(ledger: EvidenceLedger) -> tuple[object, ...]:
    return tuple(getattr(ledger, key) for key in EvidenceLedger.model_fields)


def build_ledger_index(ledger: EvidenceLedger) -> LedgerIndex:
    """Validate once and retain this context for all checks on the current ledger."""
    return LedgerIndex(ledger)


def _index_for(
    ledger: EvidenceLedger, index: LedgerIndex | None
) -> LedgerIndex:
    if index is None:
        return build_ledger_index(ledger)
    _require(type(index) is LedgerIndex, "invalid_ledger_index")
    index.check_ledger(ledger)
    return index


def _reconstruct_buffer(
    buffer: SourceBuffer, spans: Mapping[str, SourceSpan]
) -> bytes:
    _require(
        all(key in spans for key in buffer.span_ids), "unknown_buffer_span"
    )
    return b"".join(spans[key].text.encode("utf-8") for key in buffer.span_ids)


def reconstruct_buffer(
    ledger: EvidenceLedger,
    buffer_key: str,
    *,
    index: LedgerIndex | None = None,
) -> bytes:
    """Reconstruct original bytes; reuse an index when reading several buffers."""
    index = _index_for(ledger, index)
    _require(buffer_key in index.buffers, "unknown_buffer")
    return _reconstruct_buffer(index.buffers[buffer_key], index.spans)


def validate_evidence_ledger(
    ledger: EvidenceLedger, *, index: LedgerIndex | None = None
) -> None:
    """Check byte reconstruction, current coordinates, and complete ownership."""
    _index_for(ledger, index)


def _validate_evidence_ledger(
    ledger: EvidenceLedger,
    buffers: Mapping[str, SourceBuffer],
    spans: Mapping[str, SourceSpan],
    bindings: Mapping[str, BufferBinding],
) -> None:
    _unique([b.key for b in ledger.buffers], "duplicate_buffer")
    _unique([s.id for s in ledger.spans], "duplicate_span")
    _unique([u.id for u in ledger.units], "duplicate_unit")
    _unique([g.key for g in ledger.groups], "duplicate_group")
    _unique([c.change_key for c in ledger.changes], "duplicate_change_binding")
    _unique([c.change_id for c in ledger.changes], "duplicate_current_change")
    _unique(
        [b.fragment_key for b in ledger.bindings], "duplicate_current_fragment"
    )
    _require(
        len(bindings) == len(ledger.bindings) == len(ledger.buffers),
        "buffer_binding_mismatch",
    )
    visited = []
    for buffer in buffers.values():
        _require(buffer.key in bindings, "missing_buffer_binding")
        raw = _reconstruct_buffer(buffer, spans)
        _require(
            len(raw) == buffer.byte_length
            and content_digest(raw) == buffer.content_digest,
            "buffer_content_mismatch",
        )
        _, bounds = _line_bounds(raw.decode("utf-8"))
        offset = 0
        _require(bool(buffer.span_ids), "missing_empty_buffer_span")
        for key in buffer.span_ids:
            span = spans[key]
            line = bindings[buffer.key].start_line + max(
                0, bisect_right(bounds, offset) - 1
            )
            _require(
                (
                    span.buffer_key,
                    span.side,
                    span.path,
                    span.byte_start,
                    span.line_start,
                    span.line_end,
                )
                == (buffer.key, buffer.side, buffer.path, offset, line, line),
                "source_coordinates_mismatch",
            )
            _require(
                content_digest(span.text.encode("utf-8"))
                == span.content_digest,
                "span_content_mismatch",
            )
            offset = span.byte_end
            visited.append(key)
    _require(
        Counter(visited) == Counter(spans.keys()), "source_coverage_mismatch"
    )
    actual = defaultdict(list)
    obligation_ids = []
    for unit in ledger.units:
        _require(
            unit.id
            == _unit_digest(
                unit.group_key,
                unit.change_key,
                unit.classification,
                unit.spans,
                unit.obligations,
                unit.availability,
            ),
            "unit_digest_mismatch",
        )
        for span in unit.spans:
            _require(spans.get(span.id) == span, "unit_span_mismatch")
            actual[span.id].append(unit.id)
        obligation_ids.extend(o.id for o in unit.obligations)
        for obligation in unit.obligations:
            _require(
                obligation
                == _obligation(
                    obligation.kind,
                    obligation.change_key,
                    obligation.required_span_ids,
                    obligation.limitations,
                ),
                "obligation_digest_mismatch",
            )
    _unique(obligation_ids, "duplicate_ledger_obligation")
    _require(len(ledger.ownership) == len(spans), "ownership_count_mismatch")
    _unique([o.span_id for o in ledger.ownership], "duplicate_ownership")
    for owner in ledger.ownership:
        _require(
            bool(owner.unit_keys)
            and owner.unit_keys == tuple(actual[owner.span_id])
            and owner.shared == (len(owner.unit_keys) > 1),
            "span_ownership_mismatch",
        )
    change_keys = {binding.change_key for binding in ledger.changes}
    _require(
        change_keys == {u.change_key for u in ledger.units if u.change_key},
        "change_ownership_mismatch",
    )
    _require(
        change_keys
        == {
            u.change_key
            for u in ledger.units
            if any(o.kind == "structure" for o in u.obligations)
        },
        "missing_change_structure_obligation",
    )
    groups = {g.key: g for g in ledger.groups}
    change_groups = {c.change_key: c.group_key for c in ledger.changes}
    _require(
        all(
            u.group_key in groups
            and u.group_key == change_groups[u.change_key]
            for u in ledger.units
            if u.change_key
        ),
        "change_group_ownership_mismatch",
    )
    _require(
        len(ledger.group_bindings) == len(groups)
        and {b.group_key for b in ledger.group_bindings} == set(groups),
        "group_binding_mismatch",
    )
    group_units = defaultdict(list)
    for unit in ledger.units:
        group_units[unit.group_key].append(unit.id)
    for group in ledger.groups:
        _require(
            group.unit_keys == tuple(group_units[group.key]),
            "group_ownership_mismatch",
        )
        _require(
            set(group.prerequisites) <= groups.keys(),
            "unknown_group_prerequisite",
        )
    _require(
        ledger.id
        == semantic_digest(
            "ledger.v2",
            sorted(b.key for b in ledger.buffers),
            sorted(s.id for s in ledger.spans),
            [u.id for u in ledger.units],
            ledger.groups,
            ledger.availability,
            ledger.compiler_test_runs,
        ),
        "ledger_digest_mismatch",
    )


def span_reference(span: SourceSpan, unit: EvidenceUnit) -> SpanRef:
    """Construct a stable reference to a complete owned atomic span."""
    _require(span in unit.spans, "cross_unit_citation")
    return SpanRef(
        span_id=span.id,
        unit_key=unit.id,
        group_key=unit.group_key,
        side=span.side,
        path=span.path,
        byte_start=0,
        byte_end=len(span.text.encode("utf-8")),
        line_start=1,
        line_end=span.line_end - span.line_start + 1,
        content_digest=span.content_digest,
    )


def _indexed_unit(unit: EvidenceUnit, index: LedgerIndex) -> EvidenceUnit:
    # Revalidate before equality: bool/int equality must not hide model_copy
    # corruption, and altered units cannot reuse an otherwise valid context.
    unit = EvidenceUnit.model_validate(unit)
    expected = index.units.get(unit.id)
    _require(
        expected is not None and unit == expected, "unknown_or_altered_unit"
    )
    return expected


def _resolve_span_ref(
    ref: SpanRef, unit: EvidenceUnit, index: LedgerIndex
) -> SourceSpan:
    _require(
        ref.span_id in index.unit_span_ids[unit.id],
        "unknown_or_cross_unit_citation",
    )
    span = index.spans[ref.span_id]
    _require(
        (
            ref.unit_key,
            ref.group_key,
            ref.side,
            ref.path,
            ref.byte_start,
            ref.byte_end,
            ref.line_start,
            ref.line_end,
            ref.content_digest,
        )
        == (
            unit.id,
            unit.group_key,
            span.side,
            span.path,
            0,
            span.byte_end - span.byte_start,
            1,
            span.line_end - span.line_start + 1,
            span.content_digest,
        ),
        "citation_content_or_coordinates_mismatch",
    )
    return span


def resolve_span_ref(
    ref: SpanRef,
    unit: EvidenceUnit,
    ledger: EvidenceLedger,
    *,
    index: LedgerIndex | None = None,
) -> SourceSpan:
    """Resolve a strictly validated reference against current validated source."""
    index = _index_for(ledger, index)
    return _resolve_span_ref(
        SpanRef.model_validate(ref), _indexed_unit(unit, index), index
    )


def _validate_citations(
    refs: tuple[SpanRef, ...], unit: EvidenceUnit, index: LedgerIndex
) -> tuple[SourceSpan, ...]:
    _unique([ref.span_id for ref in refs], "duplicate_citation")
    return tuple(_resolve_span_ref(ref, unit, index) for ref in refs)


def validate_citations(
    refs: tuple[SpanRef, ...],
    unit: EvidenceUnit,
    ledger: EvidenceLedger,
    *,
    index: LedgerIndex | None = None,
) -> tuple[SourceSpan, ...]:
    """Reject unknown, duplicate, cross-unit, or altered references."""
    index = _index_for(ledger, index)
    refs = tuple(SpanRef.model_validate(ref) for ref in refs)
    return _validate_citations(refs, _indexed_unit(unit, index), index)


def validate_detail_result(
    result: DetailResult,
    unit: EvidenceUnit,
    ledger: EvidenceLedger,
    *,
    index: LedgerIndex | None = None,
) -> None:
    """Require exact coverage and source-linked observations with one context.

    These are structural guarantees. They cannot prove prose is semantically
    correct or that an asserted effect follows from the cited implementation.
    """
    index = _index_for(ledger, index)
    _validate_detail_result(
        DetailResult.model_validate(result), _indexed_unit(unit, index), index
    )


def _validate_detail_result(
    result: DetailResult, unit: EvidenceUnit, index: LedgerIndex
) -> None:
    _require(result.unit_key == unit.id, "detail_unit_mismatch")
    obligations = {item.id: item for item in unit.obligations}
    covered = []
    for explanation in result.explanations:
        _unique(
            list(explanation.obligation_ids), "duplicate_explained_obligation"
        )
        _require(
            set(explanation.obligation_ids) <= obligations.keys(),
            "unknown_explained_obligation",
        )
        cited = {
            span.id
            for span in _validate_citations(explanation.citations, unit, index)
        }
        required = {
            key
            for oid in explanation.obligation_ids
            for key in obligations[oid].required_span_ids
        }
        _require(required <= cited, "missing_changed_region_citation")
        limits = tuple(
            dict.fromkeys(
                limit
                for oid in explanation.obligation_ids
                for limit in obligations[oid].limitations
            )
        )
        _require(
            set(explanation.limitations) == set(limits)
            and len(explanation.limitations) == len(limits),
            "explicit_limitation_required",
        )
        covered.extend(explanation.obligation_ids)
    _require(
        Counter(covered) == Counter(obligations.keys()),
        "incomplete_or_duplicate_obligation_coverage",
    )
    _unique([o.id for o in result.observations], "duplicate_observation")
    for observation in result.observations:
        _validate_citations(observation.citations, unit, index)
    for question in result.questions:
        _validate_citations(question.citations, unit, index)
        _unique(list(question.obligation_ids), "duplicate_question_obligation")
        _require(
            set(question.obligation_ids) <= obligations.keys(),
            "unknown_question_obligation",
        )
        _require(
            bool(question.citations or question.obligation_ids),
            "unlinked_question",
        )


def _validate_detail_results(
    results: tuple[DetailResult, ...], index: LedgerIndex
) -> tuple[DetailResult, ...]:
    results = tuple(DetailResult.model_validate(result) for result in results)
    _require(
        Counter(r.unit_key for r in results) == Counter(index.units.keys()),
        "detail_set_coverage_mismatch",
    )
    for result in results:
        _validate_detail_result(result, index.units[result.unit_key], index)
    return results


def validate_detail_results(
    results: tuple[DetailResult, ...],
    ledger: EvidenceLedger,
    *,
    index: LedgerIndex | None = None,
) -> None:
    """Require each ledger unit once; build or reuse one index for the full set."""
    _validate_detail_results(results, _index_for(ledger, index))


def validate_planning_claims(
    groups: tuple[PlanningGroup, ...],
    results: tuple[DetailResult, ...],
    ledger: EvidenceLedger,
    *,
    index: LedgerIndex | None = None,
) -> None:
    """Resolve all planning claims to validated source-linked observations."""
    index = _index_for(ledger, index)
    results = _validate_detail_results(results, index)
    groups = tuple(PlanningGroup.model_validate(group) for group in groups)
    expected = index.groups
    _require(
        len(groups) == len(expected)
        and {g.group.key: g.group for g in groups} == expected,
        "planning_group_bindings_changed",
    )
    observations = {
        (r.unit_key, o.id) for r in results for o in r.observations
    }
    for group in groups:
        _require(
            len(set(group.claims)) == len(group.claims),
            "duplicate_planning_claim",
        )
        _require(
            all(
                claim.unit_key in group.group.unit_keys
                and (claim.unit_key, claim.observation_id) in observations
                for claim in group.claims
            ),
            "unsupported_planning_claim",
        )
        _require(
            all(
                claim in group.claims
                for term in group.terminology
                for claim in term.claims
            ),
            "unsupported_terminology",
        )


def planning_groups_from_details(
    ledger: EvidenceLedger,
    results: tuple[DetailResult, ...],
    *,
    index: LedgerIndex | None = None,
) -> tuple[PlanningGroup, ...]:
    """Expose complete validated observation handles without shortening detail."""
    index = _index_for(ledger, index)
    results = _validate_detail_results(results, index)
    by_unit = {r.unit_key: r for r in results}
    return tuple(
        PlanningGroup(
            group=group,
            claims=tuple(
                ClaimRef(unit_key=key, observation_id=observation.id)
                for key in group.unit_keys
                for observation in by_unit[key].observations
            ),
            terminology=(),
        )
        for group in index.groups.values()
    )


class RequestPackingLimits(Contract):
    """Explicit estimates for output bytes, throughput, framing, and repairs.

    The serializer owns all actual JSON/schema bytes. These extra reserves cover
    adapter framing, a future targeted repair, and provider uncertainty. Throughput
    and response bytes per token are declared estimates, not provider guarantees.
    """

    response_byte_limit: Positive = 2_000_000
    timeout_seconds: Positive = 900
    output_tokens_per_second: Positive = 80
    response_bytes_per_token: Positive = 8
    fixed_latency_seconds: Natural = 5
    framing_bytes: Natural = 0
    schema_reserve_bytes: Natural = 0
    repair_reserve_bytes: Natural = 2048
    repair_output_tokens: Natural = 256
    safety_tokens: Natural = 1024
    detail_tokens_per_obligation: Positive = 256
    detail_tokens_per_source_byte: Positive = 1
    citation_tokens: Positive = 192
    observation_tokens_per_unit: Positive = 128
    json_tokens_per_item: Positive = 128
    planning_tokens_per_record: Positive = 64
    stable_partition_count: Positive = 16


class PackingError(ValueError):
    """Expose an allowlisted reason and opaque key, never source or bodies."""

    def __init__(self, reason: str, item_key: str) -> None:
        """Store the safe failure code for later diagnostics."""
        self.reason = reason
        self.item_key = item_key
        super().__init__(f"{reason}: {item_key}")


def detail_output_allowance(
    units: tuple[EvidenceUnit, ...], limits: RequestPackingLimits
) -> int:
    """Reserve detail, changed-source complexity, citations, observations, JSON."""
    total = 64
    for unit in units:
        spans = {span.id: span for span in unit.spans}
        total += (
            limits.json_tokens_per_item + limits.observation_tokens_per_unit
        )
        for obligation in unit.obligations:
            size = sum(
                len(spans[key].text.encode("utf-8"))
                for key in obligation.required_span_ids
            )
            total += (
                limits.detail_tokens_per_obligation
                + size * limits.detail_tokens_per_source_byte
            )
            total += len(obligation.required_span_ids) * limits.citation_tokens
            total += len(canonical_bytes(obligation.limitations))
    return total


Item = TypeVar("Item", bound=Contract)
_REQUEST_KEY = re.compile(r"[0-9a-f]{64}")


def _request_keys(
    items: tuple[Item, ...], keys: Callable[[Item], str]
) -> tuple[str, ...]:
    item_keys = tuple(keys(item) for item in items)
    _require(
        all(
            type(key) is str and _REQUEST_KEY.fullmatch(key) is not None
            for key in item_keys
        ),
        "invalid_request_key",
    )
    _unique(item_keys, "duplicate_request_item")
    return item_keys


class PreparedRequest(Contract):
    """Keep the exact actual provider bytes with their checked request bounds.

    Bytes are ephemeral and excluded from dumps. A serializer must include its
    full provider framing and schema; a preview is not send authority.
    """

    body: bytes = Field(exclude=True, repr=False)
    batch: BatchSpec


def prepare_request_items(  # noqa: PLR0913 # Explicit serializer and capacity form the send boundary.
    items: tuple[Item, ...],
    *,
    stage: BatchStage,
    keys: Callable[[Item], str],
    groups: Callable[[Item], tuple[str, ...]],
    serializer: Callable[[tuple[Item, ...], int], bytes],
    allowance: Callable[[tuple[Item, ...]], int],
    capacity: ModelCapacity,
    limits: RequestPackingLimits,
    enforceable_output_cap: bool = False,
    partition: int = 0,
) -> PreparedRequest:
    """Prepare one actual initial/repair/split/resume node without repacking.

    Supply exactly the immutable current items and a serializer for this node's
    purpose, including repair fields when applicable. The explicit allowance
    excludes limits.repair_output_tokens, which is reserved here. Nothing is
    clipped or split. Recreate this value at each send boundary.
    """
    _require(type(items) is tuple and bool(items), "invalid_request_items")
    _require(
        stage in {"detail", "planning", "section", "document"}, "invalid_stage"
    )
    for item in items:
        _require(isinstance(item, Contract), "mutable_request_item")
        type(item).model_validate(item)
    limits = RequestPackingLimits.model_validate(limits)
    requested = allowance(items)
    _require(
        type(requested) is int and requested > 0, "invalid_output_allowance"
    )
    output = requested + limits.repair_output_tokens
    item_keys = _request_keys(items, keys)
    if output > capacity.max_output_tokens:
        msg = "output_limit"
        raise PackingError(msg, item_keys[0])
    guard = capacity.output_guard(
        output, enforceable_output_cap=enforceable_output_cap
    )
    body = serializer(items, output)
    _require(type(body) is bytes, "invalid_serialized_request")
    bound = estimate_input(
        body,
        overhead_bytes=limits.framing_bytes
        + limits.schema_reserve_bytes
        + limits.repair_reserve_bytes,
    )
    if bound + guard + limits.safety_tokens > capacity.context_tokens:
        msg = "input_limit"
        raise PackingError(msg, item_keys[0])
    response_bytes = output * limits.response_bytes_per_token
    seconds = limits.fixed_latency_seconds + math.ceil(
        output / limits.output_tokens_per_second
    )
    if response_bytes > limits.response_byte_limit:
        msg = "response_limit"
        raise PackingError(msg, item_keys[0])
    if seconds > limits.timeout_seconds:
        msg = "timeout_limit"
        raise PackingError(msg, item_keys[0])
    request_digest = content_digest(body)
    batch = BatchSpec(
        id=semantic_digest(
            "batch.v2", stage, item_keys, request_digest, output, guard, limits
        ),
        stage=stage,
        item_keys=item_keys,
        group_keys=tuple(
            dict.fromkeys(g for item in items for g in groups(item))
        ),
        request_digest=request_digest,
        input_bound=bound,
        output_reservation=output,
        output_guard=guard,
        estimated_response_bytes=response_bytes,
        estimated_seconds=seconds,
        partition=partition,
    )
    return PreparedRequest(body=body, batch=batch)


def _request_spec(items: tuple[Item, ...], **kwargs: object) -> BatchSpec:
    return prepare_request_items(items, **kwargs).batch


def pack_request_items(  # noqa: PLR0913 # Keep all actual provider bounds explicit.
    items: tuple[Item, ...],
    *,
    stage: BatchStage,
    keys: Callable[[Item], str],
    groups: Callable[[Item], tuple[str, ...]],
    serializer: Callable[[tuple[Item, ...], int], bytes],
    allowance: Callable[[tuple[Item, ...]], int],
    capacity: ModelCapacity,
    limits: RequestPackingLimits,
    enforceable_output_cap: bool = False,
) -> tuple[BatchSpec, ...]:
    """Pack actual immutable cards or adjacent sections with no request floor.

    Whole-fit input uses one request. Section/document order is retained;
    evidence/planning use fixed digest partitions when the whole set cannot
    fit. Serialize complete produced child prose in parent items; the consumer
    must build another hierarchy level if whole parent input cannot fit.
    An indivisible oversized item fails instead of slicing text or references.
    """
    _require(type(items) is tuple, "invalid_request_items")
    _require(
        stage in {"detail", "planning", "section", "document"}, "invalid_stage"
    )
    limits = RequestPackingLimits.model_validate(limits)
    if not items:
        return ()
    _request_keys(items, keys)
    ordered = (
        items
        if stage in {"section", "document"}
        else tuple(sorted(items, key=keys))
    )
    kwargs = {
        "stage": stage,
        "keys": keys,
        "groups": groups,
        "serializer": serializer,
        "allowance": allowance,
        "capacity": capacity,
        "limits": limits,
        "enforceable_output_cap": enforceable_output_cap,
    }
    try:
        return (_request_spec(ordered, partition=0, **kwargs),)
    except PackingError:
        pass
    if stage in {"section", "document"}:
        return _pack(ordered, **kwargs)
    buckets = defaultdict(list)
    for item in ordered:
        buckets[int(keys(item), 16) % limits.stable_partition_count].append(
            item
        )
    return tuple(
        batch
        for partition in sorted(buckets)
        for batch in _pack(
            tuple(buckets[partition]), partition=partition, **kwargs
        )
    )


def _pack(  # noqa: PLR0913 # Keep explicit evidence, serializer, and capacity inputs.
    items: tuple[Item, ...],
    *,
    stage: str,
    keys: Callable[[Item], str],
    groups: Callable[[Item], tuple[str, ...]],
    serializer: Callable[[tuple[Item, ...], int], bytes],
    allowance: Callable[[tuple[Item, ...]], int],
    capacity: ModelCapacity,
    limits: RequestPackingLimits,
    enforceable_output_cap: bool,
    partition: int = 0,
) -> tuple[BatchSpec, ...]:
    result = []
    pending: tuple[Item, ...] = ()
    current = None
    for item in items:
        try:
            candidate = _request_spec(
                (*pending, item),
                stage=stage,
                keys=keys,
                groups=groups,
                serializer=serializer,
                allowance=allowance,
                capacity=capacity,
                limits=limits,
                enforceable_output_cap=enforceable_output_cap,
                partition=partition,
            )
        except PackingError:  # noqa: PERF203 # A size boundary ends the current batch.
            if not pending:
                raise
            result.append(current)
            pending = (item,)
            current = _request_spec(
                pending,
                stage=stage,
                keys=keys,
                groups=groups,
                serializer=serializer,
                allowance=allowance,
                capacity=capacity,
                limits=limits,
                enforceable_output_cap=enforceable_output_cap,
                partition=partition,
            )
        else:
            pending += (item,)
            current = candidate
    if pending:
        result.append(current)
    return tuple(result)


def pack_evidence_units(
    units: tuple[EvidenceUnit, ...],
    capacity: ModelCapacity,
    limits: RequestPackingLimits,
    serializer: Callable[[tuple[EvidenceUnit, ...], int], bytes],
    *,
    enforceable_output_cap: bool = False,
) -> tuple[BatchSpec, ...]:
    """Pack across groups using actual request bytes and obligation complexity.

    Try all inputs first: if they fit, there is exactly one batch. Otherwise,
    fixed digest buckets and semantic sorting localize peer dependencies. An
    atomic unit which still cannot fit is rejected, never clipped. Rebuild the
    ledger with smaller span/unit bounds when declared source boundaries permit.
    """
    if not units:
        return ()
    _unique([u.id for u in units], "duplicate_packing_unit")
    ordered = tuple(sorted(units, key=lambda u: u.id))
    kwargs = {
        "stage": "detail",
        "keys": lambda u: u.id,
        "groups": lambda u: (u.group_key,) if u.group_key else (),
        "serializer": serializer,
        "allowance": lambda items: detail_output_allowance(items, limits),
        "capacity": capacity,
        "limits": limits,
        "enforceable_output_cap": enforceable_output_cap,
    }
    try:
        return (_request_spec(ordered, partition=0, **kwargs),)
    except PackingError:
        pass
    buckets = defaultdict(list)
    for unit in ordered:
        buckets[int(unit.id, 16) % limits.stable_partition_count].append(unit)
    return tuple(
        batch
        for partition in sorted(buckets)
        for batch in _pack(
            tuple(buckets[partition]), partition=partition, **kwargs
        )
    )


def check_evidence_batch(  # noqa: PLR0913 # Keep explicit evidence, serializer, and capacity inputs.
    spec: BatchSpec,
    units: tuple[EvidenceUnit, ...],
    capacity: ModelCapacity,
    limits: RequestPackingLimits,
    serializer: Callable[[tuple[EvidenceUnit, ...], int], bytes],
    *,
    enforceable_output_cap: bool = False,
) -> None:
    """Recheck the exact current serializer and bounds immediately before send."""
    actual = _request_spec(
        units,
        stage="detail",
        keys=lambda u: u.id,
        groups=lambda u: (u.group_key,) if u.group_key else (),
        serializer=serializer,
        allowance=lambda items: detail_output_allowance(items, limits),
        capacity=capacity,
        limits=limits,
        enforceable_output_cap=enforceable_output_cap,
        partition=spec.partition,
    )
    _require(actual == spec, "prepared_request_changed")


def _finish_order(edges: dict[str, tuple[str, ...]]) -> list[str]:
    nodes = sorted(edges)
    visited = set()
    finished = []
    for node in nodes:
        if node in visited:
            continue
        visited.add(node)
        stack = [(node, iter(edges[node]))]
        while stack:
            current, iterator = stack[-1]
            neighbor = next(iterator, None)
            if neighbor is None:
                finished.append(current)
                stack.pop()
            elif neighbor not in visited:
                visited.add(neighbor)
                stack.append((neighbor, iter(edges[neighbor])))
    return finished


def _strong_components(
    edges: dict[str, tuple[str, ...]],
) -> tuple[list[tuple[str, ...]], dict[str, int]]:
    reverse = {node: [] for node in edges}
    for node, neighbors in edges.items():
        for neighbor in neighbors:
            reverse[neighbor].append(node)
    component_of = {}
    components = []
    for node in reversed(_finish_order(edges)):
        if node in component_of:
            continue
        number = len(components)
        members = []
        pending = [node]
        component_of[node] = number
        while pending:
            current = pending.pop()
            members.append(current)
            for neighbor in reverse[current]:
                if neighbor not in component_of:
                    component_of[neighbor] = number
                    pending.append(neighbor)
        components.append(tuple(sorted(members)))
    return components, component_of


def ordered_components(
    prerequisites: Mapping[str, Sequence[str]],
) -> tuple[tuple[str, ...], tuple[tuple[str, ...], ...]]:
    """Order an arbitrary-size graph through iterative SCC condensation.

    All edges are retained, including self edges. Nodes inside an SCC use stable
    lexical order. The component DAG uses a heap for deterministic ready nodes.
    """
    nodes = sorted(prerequisites)
    edges = {node: tuple(sorted(set(prerequisites[node]))) for node in nodes}
    _require(
        all(set(neighbors) <= edges.keys() for neighbors in edges.values()),
        "unknown_prerequisite",
    )
    components, component_of = _strong_components(edges)
    dependencies = [set() for _ in components]
    dependents = [set() for _ in components]
    for node, neighbors in edges.items():
        target = component_of[node]
        for neighbor in neighbors:
            source = component_of[neighbor]
            if source != target:
                dependencies[target].add(source)
                dependents[source].add(target)
    ready = [
        (components[index], index)
        for index, required in enumerate(dependencies)
        if not required
    ]
    heapq.heapify(ready)
    order = []
    cycles = []
    while ready:
        members, index = heapq.heappop(ready)
        order.extend(members)
        if len(members) > 1 or members[0] in edges[members[0]]:
            cycles.append(members)
        for dependent in sorted(dependents[index]):
            dependencies[dependent].remove(index)
            if not dependencies[dependent]:
                heapq.heappush(ready, (components[dependent], dependent))
    _require(len(order) == len(nodes), "invalid_component_graph")
    return tuple(order), tuple(cycles)


def validate_story_order(
    order: tuple[str, ...], prerequisites: Mapping[str, Sequence[str]]
) -> None:
    """Require each group once, prerequisites first, and complete SCC adjacency."""
    _require(
        len(order) == len(prerequisites) and set(order) == set(prerequisites),
        "story_membership_mismatch",
    )
    _, cycles = ordered_components(prerequisites)
    cycle_of = {
        key: index for index, members in enumerate(cycles) for key in members
    }
    positions = {key: index for index, key in enumerate(order)}
    for group, required in prerequisites.items():
        for prerequisite in required:
            same_cycle = group in cycle_of and cycle_of[group] == cycle_of.get(
                prerequisite
            )
            _require(
                same_cycle or positions[prerequisite] < positions[group],
                "story_prerequisite_order",
            )
    for members in cycles:
        places = [positions[key] for key in members]
        _require(
            max(places) - min(places) + 1 == len(members),
            "story_cycle_not_adjacent",
        )


def planning_groups(ledger: EvidenceLedger) -> tuple[PlanningGroup, ...]:
    """Provide a deterministic baseline without inventing detailed claims."""
    return tuple(
        PlanningGroup(group=g, claims=(), terminology=())
        for g in ledger.groups
    )


def _atoms(group: PlanningGroup) -> tuple[PlanningAtom, ...]:
    key = group.group.key
    result = [
        PlanningAtom(
            group_key=key,
            kind="member",
            key=None,
            claim=None,
            term=None,
            path=group.group.path,
            theme=group.group.theme,
        )
    ]
    result.extend(
        PlanningAtom(
            group_key=key, kind="unit", key=unit, claim=None, term=None
        )
        for unit in group.group.unit_keys
    )
    result.extend(
        PlanningAtom(
            group_key=key,
            kind="prerequisite",
            key=other,
            claim=None,
            term=None,
        )
        for other in group.group.prerequisites
    )
    result.extend(
        PlanningAtom(
            group_key=key, kind="claim", key=None, claim=claim, term=None
        )
        for claim in group.claims
    )
    result.extend(
        PlanningAtom(
            group_key=key, kind="term", key=None, claim=None, term=term
        )
        for term in group.terminology
    )
    return tuple(result)


def _card(
    level: int,
    atoms: tuple[PlanningAtom, ...] = (),
    children: tuple[str, ...] = (),
) -> StoryCard:
    return StoryCard(
        id=semantic_digest("card.v2", level, atoms, children),
        level=level,
        atoms=atoms,
        child_keys=children,
    )


def _planning_allowance(
    cards: tuple[StoryCard, ...], limits: RequestPackingLimits
) -> int:
    return 64 + sum(
        limits.json_tokens_per_item
        + limits.planning_tokens_per_record
        * (len(card.atoms) + len(card.child_keys))
        for card in cards
    )


def _leaf_cards(
    atoms: tuple[PlanningAtom, ...],
    check: Callable[[tuple[StoryCard, ...]], BatchSpec],
) -> tuple[StoryCard, ...]:
    cards = []
    pending: tuple[PlanningAtom, ...] = ()
    for atom in atoms:
        candidate = _card(0, (*pending, atom))
        try:
            check((candidate,))
        except PackingError:
            if not pending:
                raise
            cards.append(_card(0, pending))
            pending = (atom,)
            check((_card(0, pending),))
        else:
            pending += (atom,)
    if pending:
        cards.append(_card(0, pending))
    return tuple(cards)


def _parent_cards(
    children: tuple[StoryCard, ...],
    check: Callable[[tuple[StoryCard, ...]], BatchSpec],
) -> tuple[StoryCard, ...]:
    parents = []
    pending: tuple[str, ...] = ()
    level = children[0].level + 1
    for child in children:
        candidate = _card(level, children=(*pending, child.id))
        try:
            check((candidate,))
        except PackingError:
            if not pending:
                raise
            parents.append(_card(level, children=pending))
            pending = (child.id,)
            check((_card(level, children=pending),))
        else:
            pending += (child.id,)
    if pending:
        parents.append(_card(level, children=pending))
    if len(parents) >= len(children) and len(children) > 1:
        msg = "hierarchy_cannot_reduce"
        raise PackingError(msg, children[0].id)
    return tuple(parents)


def _chapters(
    order: tuple[str, ...], groups: dict[str, PlanningGroup], maximum: int
) -> tuple[StoryChapter, ...]:
    result = []
    for start in range(0, len(order), maximum):
        members = order[start : start + maximum]
        cross = tuple(
            Prerequisite(group_key=key, prerequisite_key=other)
            for key in members
            for other in groups[key].group.prerequisites
            if other not in members
        )
        result.append(
            StoryChapter(
                id=semantic_digest("chapter.v2", members, cross),
                group_keys=members,
                cross_chapter_prerequisites=cross,
            )
        )
    return tuple(result)


def build_story_plan(  # noqa: PLR0913 # Keep explicit evidence, serializer, and capacity inputs.
    groups: tuple[PlanningGroup, ...],
    capacity: ModelCapacity,
    limits: RequestPackingLimits,
    serializer: Callable[[tuple[StoryCard, ...], int], bytes],
    *,
    context_digest: str,
    max_chapter_groups: int = 32,
    enforceable_output_cap: bool = False,
) -> StoryPlan:
    """Build lossless bounded cards over contiguous SCC/topological order.

    A group can span leaf cards only between complete atoms. Parents reference
    all child cards, never sliced summaries. Structural requests use the supplied
    serializer and output limits. After generation, consumers must repack actual
    child prose and add a bounded level when needed; this function does not run
    that production loop. Large cycles can span cards/chapters; their groups
    remain adjacent in the complete story order.
    """
    _require(
        type(max_chapter_groups) is int and max_chapter_groups > 0,
        "invalid_chapter_limit",
    )
    _unique([g.group.key for g in groups], "duplicate_planning_group")
    lookup = {g.group.key: g for g in groups}
    order, cycles = ordered_components(
        {g.group.key: g.group.prerequisites for g in groups}
    )
    kwargs = {
        "stage": "planning",
        "keys": lambda c: c.id,
        "groups": lambda c: tuple(dict.fromkeys(a.group_key for a in c.atoms)),
        "serializer": serializer,
        "allowance": lambda cards: _planning_allowance(cards, limits),
        "capacity": capacity,
        "limits": limits,
        "enforceable_output_cap": enforceable_output_cap,
    }

    def check(cards: tuple[StoryCard, ...]) -> BatchSpec:
        return _request_spec(cards, partition=0, **kwargs)

    # Group-local leaf boundaries preserve unaffected card identities when an
    # unrelated group is inserted. Requests still batch cards across groups.
    leaves = tuple(
        card
        for key in order
        for card in _leaf_cards(_atoms(lookup[key]), check)
    )
    cards = list(leaves)
    current_requests = _pack(leaves, **kwargs)
    requests = list(current_requests)
    level = leaves
    while len(current_requests) > 1:
        parents = _parent_cards(level, check)
        cards.extend(parents)
        current_requests = _pack(parents, **kwargs)
        requests.extend(current_requests)
        level = parents
    chapters = _chapters(order, lookup, max_chapter_groups)
    key = semantic_digest(
        "story-plan.v2",
        context_digest,
        order,
        cycles,
        chapters,
        tuple(c.id for c in cards),
    )
    plan = StoryPlan(
        id=key,
        context_digest=context_digest,
        groups=tuple(lookup[key] for key in order),
        ordered_group_keys=order,
        cycles=cycles,
        chapters=chapters,
        cards=tuple(cards),
        root_card_keys=tuple(c.id for c in level),
        requests=tuple(requests),
    )
    validate_story_plan(plan, groups)
    return plan


def validate_primary_structure(
    plan: StoryPlan, expected_groups: tuple[PlanningGroup, ...]
) -> None:
    """Check only primary membership, facts, chapters, levels and exact traversal.

    This partial check is not plan acceptance. Generic callers must use
    validate_story_plan; production callers must rederive current sources with
    validate_production_plan and audit actual requests before final acceptance.
    """
    plan = StoryPlan.model_validate(plan)
    expected = {g.group.key: g for g in expected_groups}
    _require(len(expected) == len(expected_groups), "duplicate_expected_group")
    _require(
        {g.group.key: g for g in plan.groups} == expected
        and len(plan.groups) == len(expected),
        "planning_facts_changed",
    )
    prerequisites = {key: g.group.prerequisites for key, g in expected.items()}
    validate_story_order(plan.ordered_group_keys, prerequisites)
    _, cycles = ordered_components(prerequisites)
    _require(
        {frozenset(c) for c in plan.cycles} == {frozenset(c) for c in cycles}
        and len(plan.cycles) == len(cycles),
        "planning_cycles_changed",
    )
    chapter_order = tuple(
        key for chapter in plan.chapters for key in chapter.group_keys
    )
    _require(
        chapter_order == plan.ordered_group_keys, "chapter_membership_changed"
    )
    for chapter in plan.chapters:
        cross = tuple(
            Prerequisite(group_key=key, prerequisite_key=other)
            for key in chapter.group_keys
            for other in prerequisites[key]
            if other not in chapter.group_keys
        )
        _require(
            chapter.cross_chapter_prerequisites == cross,
            "cross_chapter_prerequisites_changed",
        )
        _require(
            chapter.id
            == semantic_digest("chapter.v2", chapter.group_keys, cross),
            "chapter_digest_mismatch",
        )
    cards = {card.id: card for card in plan.cards}
    _require(len(cards) == len(plan.cards), "duplicate_story_card")
    visited = set()
    atoms = []
    pending = list(reversed(plan.root_card_keys))
    while pending:
        key = pending.pop()
        _require(key in cards and key not in visited, "invalid_card_tree")
        card = cards[key]
        _require(
            card.id == _card(card.level, card.atoms, card.child_keys).id,
            "card_digest_mismatch",
        )
        visited.add(key)
        atoms.extend(card.atoms)
        for child in card.child_keys:
            _require(
                child in cards and cards[child].level == card.level - 1,
                "invalid_card_level",
            )
        pending.extend(reversed(card.child_keys))
    _require(set(visited) == set(cards), "unreachable_story_card")
    expected_atoms = tuple(
        atom
        for key in plan.ordered_group_keys
        for atom in _atoms(expected[key])
    )
    _require(tuple(atoms) == expected_atoms, "incomplete_planning_atoms")
    for group in expected.values():
        unit_keys = set(group.group.unit_keys)
        _unique(list(group.group.unit_keys), "duplicate_group_unit")
        _unique(
            list(group.group.prerequisites), "duplicate_group_prerequisite"
        )
        _require(
            all(claim.unit_key in unit_keys for claim in group.claims),
            "cross_group_claim",
        )
        _require(
            all(
                claim in group.claims
                for term in group.terminology
                for claim in term.claims
            ),
            "unsupported_terminology",
        )


def validate_story_plan(
    plan: StoryPlan, expected_groups: tuple[PlanningGroup, ...]
) -> None:
    """Validate the complete generic plan; production requires current-data validation."""
    plan = StoryPlan.model_validate(plan)
    _require(plan.production is None, "production_validation_required")
    validate_primary_structure(plan, expected_groups)
    requested = [key for request in plan.requests for key in request.item_keys]
    _require(
        Counter(requested) == Counter(card.id for card in plan.cards),
        "planning_request_membership_mismatch",
    )
    _require(
        plan.id
        == semantic_digest(
            "story-plan.v2",
            plan.context_digest,
            plan.ordered_group_keys,
            plan.cycles,
            plan.chapters,
            tuple(c.id for c in plan.cards),
        ),
        "story_plan_digest_mismatch",
    )


def planning_atoms(group: PlanningGroup) -> tuple[PlanningAtom, ...]:
    """Expose the exact ordered primary atoms without changing their semantics."""
    return _atoms(group)


def story_card(
    level: int,
    atoms: tuple[PlanningAtom, ...] = (),
    children: tuple[str, ...] = (),
) -> StoryCard:
    """Construct a primary card with the original canonical identity."""
    return _card(level, atoms, children)


def story_chapters(
    order: tuple[str, ...], groups: tuple[PlanningGroup, ...], maximum: int
) -> tuple[StoryChapter, ...]:
    """Derive complete chapter membership and cross-chapter compiler edges."""
    _require(type(maximum) is int and maximum > 0, "invalid_chapter_limit")
    return _chapters(order, {g.group.key: g for g in groups}, maximum)


def validate_story_requests(
    plan: StoryPlan,
    capacity: ModelCapacity,
    limits: RequestPackingLimits,
    serializer: Callable[[tuple[StoryCard, ...], int], bytes],
    *,
    enforceable_output_cap: bool = False,
) -> None:
    """Recheck actual serialization and allowances at every hierarchy level."""
    _require(plan.production is None, "production_validation_required")
    cards = {card.id: card for card in plan.cards}
    for spec in plan.requests:
        actual = _request_spec(
            tuple(cards[key] for key in spec.item_keys),
            stage="planning",
            keys=lambda c: c.id,
            groups=lambda c: tuple(
                dict.fromkeys(a.group_key for a in c.atoms)
            ),
            serializer=serializer,
            allowance=lambda items: _planning_allowance(items, limits),
            capacity=capacity,
            limits=limits,
            enforceable_output_cap=enforceable_output_cap,
            partition=spec.partition,
        )
        _require(actual == spec, "prepared_planning_request_changed")
