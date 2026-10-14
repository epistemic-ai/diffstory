"""Deterministic evidence accounting, validation, packing, and story planning.

This module accepts trusted compiler reports. It does not import the compiler,
providers, or runtime. Public validators also serve later persisted boundaries.
Request serializers must include the real instructions, JSON schema, context,
and adapter envelope. Packing is an estimate; adapters must recheck before send.
"""

from __future__ import annotations

import copy
import hashlib
import heapq
import json
import math
import re
from bisect import bisect_right
from collections import Counter
from collections import defaultdict
from collections.abc import Iterator
from collections.abc import Mapping
from collections.abc import MutableMapping
from collections.abc import Sequence as AbstractSequence
from dataclasses import dataclass
from dataclasses import field
from difflib import SequenceMatcher
from types import MappingProxyType
from typing import TYPE_CHECKING
from typing import Protocol
from typing import TypeVar
from weakref import WeakKeyDictionary
from weakref import ref as weakref_ref

from pydantic import BaseModel
from pydantic import Field
from typing_extensions import Self

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
from diffstory.narrative_contracts import EvidenceQuestion
from diffstory.narrative_contracts import EvidenceUnit
from diffstory.narrative_contracts import Explanation
from diffstory.narrative_contracts import FactualObservation
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
    from collections.abc import Sequence

    from diffstory.report_storage import ReportStorageView
    from diffstory.report_storage import ResultRef


@dataclass(frozen=True, slots=True)
class _OwnedMapping(Mapping[str, object]):
    """Read-only mapping for one session-owned JSON object.

    Attributes:
        entries: Ordered string keys and detached immutable JSON values.

    The mapping preserves source iteration order while canonical serialization
    sorts keys. It is an internal traversal view, not a public report adapter.
    """

    entries: tuple[tuple[str, object], ...]

    def __getitem__(self, key: str) -> object:
        """Return the value for an existing key.

        Preconditions:
            ``key`` is a string.
        Postconditions:
            Returns the exact immutable child owned by this mapping.
        Essential behavior:
            Does not copy or thaw the report tree.
        """
        for name, value in self.entries:
            if name == key:
                return value
        raise KeyError(key)

    def __iter__(self) -> Iterator[str]:
        """Iterate original keys without creating a mutable mapping."""
        return (key for key, _value in self.entries)

    def __len__(self) -> int:
        """Return the number of owned keys."""
        return len(self.entries)


@dataclass(frozen=True, slots=True)
class _OwnedList(AbstractSequence[object]):
    """Read-only view that preserves an input JSON list's Python shape."""

    items: tuple[object, ...]

    def __getitem__(self, index: int | slice) -> object:
        """Read one owned element or slice.

        Preconditions:
            ``index`` is a valid sequence index or slice.
        Postconditions:
            Returns an immutable owned element or tuple of owned elements.
        Essential behavior:
            Preserves list identity in the session's raw-shape witness.
        """
        return self.items[index]

    def __len__(self) -> int:
        """Return the number of owned list elements."""
        return len(self.items)


@dataclass(frozen=True, slots=True)
class _OwnedTuple(AbstractSequence[object]):
    """Read-only view that preserves an input Python tuple's shape."""

    items: tuple[object, ...]

    def __getitem__(self, index: int | slice) -> object:
        """Read one owned element or slice without exposing the backing tuple."""
        return self.items[index]

    def __len__(self) -> int:
        """Return the number of owned tuple elements."""
        return len(self.items)


def _canonical_chunks(value: object) -> Iterator[bytes]:
    """Yield canonical JSON bytes for native or owned JSON primitives.

    Preconditions:
        Values contain JSON primitives, mappings with string keys, sequences,
        or Pydantic models with JSON serializers.
    Postconditions:
        Joined chunks equal the previous ``json.dumps`` canonical wire bytes.
    Essential behavior:
        Traverses owned report views directly so hashing need not build a full
        temporary report dictionary or retain its encoded byte buffer.
    """
    if isinstance(value, BaseModel):
        value = value.model_dump(mode="json")
    if isinstance(value, Mapping):
        keys = tuple(value)
        if not all(type(key) is str for key in keys):
            # Keep json's original error behavior for non-JSON object keys.
            fallback = json.dumps(
                dict(value),
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
                allow_nan=False,
                default=_json_default,
            )
            yield fallback.encode("utf-8")
            return
        yield b"{"
        for position, key in enumerate(sorted(keys)):
            if position:
                yield b","
            yield json.dumps(key, ensure_ascii=False).encode("utf-8")
            yield b":"
            yield from _canonical_chunks(value[key])
        yield b"}"
        return
    if type(value) in {list, tuple, _OwnedList, _OwnedTuple}:
        yield b"["
        for position, item in enumerate(value):
            if position:
                yield b","
            yield from _canonical_chunks(item)
        yield b"]"
        return
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
        default=_json_default,
    )
    yield encoded.encode("utf-8")


def _thaw_owned(value: object) -> object:
    """Build a detached mutable compatibility value from an owned view."""
    if type(value) is _OwnedMapping:
        return {key: _thaw_owned(item) for key, item in value.entries}
    if type(value) is _OwnedList:
        return [_thaw_owned(item) for item in value.items]
    if type(value) is _OwnedTuple:
        return tuple(_thaw_owned(item) for item in value.items)
    return value


def _json_default(value: object) -> object:
    """Adapt typed models and owned views for the standard JSON encoder.

    Preconditions:
        ``value`` is a Pydantic model, mapping, or owned sequence view.
    Postconditions:
        Returns a shallow JSON-compatible container or model value.
    Essential behavior:
        Keeps ordinary trees on the standard encoder's optimized path while
        avoiding a recursive thaw of the session-owned report tree.
    """
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json")
    if isinstance(value, Mapping):
        return dict(value.items())
    if type(value) in {_OwnedList, _OwnedTuple}:
        return list(value.items)
    msg = "Unsupported canonical value"
    raise TypeError(msg)


def canonical_bytes(value: object) -> bytes:
    """Encode identity material without whitespace or dictionary order effects.

    Preconditions:
        ``value`` is JSON data, a typed model, a mapping, or an owned view.
    Postconditions:
        Returns the exact compact UTF-8 bytes used by the canonical identity.
    Essential behavior:
        Uses the standard encoder for native and typed values; owned views use
        shallow per-container adapters and are not deep-thawed.
    """
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
        default=_json_default,
    ).encode("utf-8")


def canonical_digest(value: object) -> str:
    """Hash canonical JSON by chunks without retaining its complete bytes.

    Preconditions:
        ``value`` is accepted by :func:`canonical_bytes`.
    Postconditions:
        Returns the SHA256 digest of exactly the canonical wire bytes.
    Essential behavior:
        Keeps only the active JSON path and hash state for owned report views.
    """
    digest = hashlib.sha256()
    for chunk in _canonical_chunks(value):
        digest.update(chunk)
    return digest.hexdigest()


def _raw_integrity_value(value: object) -> object:
    """Tag raw Python values before hashing mutable model integrity state.

    Pydantic serializers can normalize ``True`` to ``1`` for integer fields.
    Integrity checks must compare the exact current model value, including its
    scalar type, so this adapter walks model fields without serializing models.
    """
    if isinstance(value, BaseModel):
        return (
            "model",
            type(value).__module__,
            type(value).__qualname__,
            tuple(
                (name, _raw_integrity_value(getattr(value, name)))
                for name in type(value).model_fields
            ),
        )
    if isinstance(value, Mapping):
        entries = [
            (_raw_integrity_value(key), _raw_integrity_value(item))
            for key, item in value.items()
        ]
        entries.sort(key=lambda item: canonical_bytes(item[0]))
        return "mapping", tuple(entries)
    if type(value) in {tuple, list, _OwnedList, _OwnedTuple}:
        kind = "list" if type(value) in {list, _OwnedList} else "tuple"
        return kind, tuple(_raw_integrity_value(item) for item in value)
    if type(value) in {set, frozenset}:
        entries = [_raw_integrity_value(item) for item in value]
        entries.sort(key=canonical_bytes)
        return type(value).__name__, tuple(entries)
    if value is None or type(value) in {str, bool, int, float}:
        return "scalar", type(value).__name__, value
    if type(value) is bytes:
        return "bytes", value.hex()
    msg = "Unsupported raw integrity value"
    raise TypeError(msg)


def _raw_integrity_bytes(value: object) -> bytes:
    """Encode a type-preserving integrity witness for current model values."""
    return canonical_bytes(_raw_integrity_value(value))


def content_digest(value: bytes) -> str:
    """Return the exact SHA256 of a UTF-8 buffer or serialized request."""
    return hashlib.sha256(value).hexdigest()


def semantic_digest(*parts: object) -> str:
    """Hash versioned canonical content; callers exclude current bindings."""
    return canonical_digest(parts)


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
    report: dict | _OwnedMapping,
) -> tuple[list[_Fragment], tuple[FileAvailability, ...]]:
    evidence = report.get("source_evidence")
    _require(
        type(evidence) in {dict, _OwnedMapping}
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
    report: dict | _OwnedMapping,
    *,
    max_span_bytes: int = 4096,
    max_unit_spans: int = 16,
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
            content_digest(_raw_integrity_bytes(ledger)),
            content_digest(self._map_bytes()),
            tuple(getattr(self, name) for name in _INDEX_MAP_NAMES),
        )

    def _map_bytes(self) -> bytes:
        return _raw_integrity_bytes(
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
            and content_digest(_raw_integrity_bytes(ledger)) == snapshot
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


def _revision_pair(value: object) -> tuple[str, str]:
    _require(
        type(value) is tuple
        and len(value) == _REVISION_PAIR_SIZE
        and all(type(item) is str and item for item in value),
        "invalid_citation_revision_binding",
    )
    return value


def _ledger_side_revision(
    ledger: EvidenceLedger, index: LedgerIndex, side: str
) -> str:
    """Derive a source revision key when Git revisions are not available.

    The ledger format does not carry base/head commit IDs. This fallback binds
    the citation scope to the current side's paths, bytes, fragments, and
    absolute line starts. Callers that have report revisions should pass those
    exact values to ``CitationResolutionScope`` instead.
    """
    availability = tuple(
        sorted(
            (
                item.path,
                item.counterpart_path,
                item.state,
                item.coverage,
                item.reason,
                item.full_source,
                item.full_and_parsed,
            )
            for item in ledger.availability
            if item.side == side
        )
    )
    buffers = tuple(
        sorted(
            (
                buffer.key,
                buffer.path,
                buffer.content_digest,
                buffer.byte_length,
                buffer.coverage,
                buffer.parse_status,
                index.bindings[buffer.key].fragment_key,
                index.bindings[buffer.key].start_line,
                index.bindings[buffer.key].region,
            )
            for buffer in index.buffers.values()
            if buffer.side == side
        )
    )
    return semantic_digest(
        "citation-source-revision.v1", side, ledger.id, availability, buffers
    )


def _span_ref_key(ref: SpanRef) -> tuple[object, ...]:
    return (
        ref.span_id,
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


def _citation_binding_key(
    ref: SpanRef,
    unit: EvidenceUnit,
    index: LedgerIndex,
    revisions: tuple[str, str],
    availability_by_source: Mapping[tuple[str, str], tuple[object, ...]],
) -> tuple[object, ...]:
    """Bind one exact reference to its current physical source coordinates."""
    span = index.spans.get(ref.span_id)
    if span is None:
        return (_span_ref_key(ref), revisions, "missing-span")
    buffer = index.buffers.get(span.buffer_key)
    binding = index.bindings.get(span.buffer_key)
    availability = availability_by_source.get((span.side, span.path))
    return (
        _span_ref_key(ref),
        unit.id,
        unit.group_key,
        revisions,
        (
            span.buffer_key,
            span.side,
            span.path,
            span.byte_start,
            span.byte_end,
            span.line_start,
            span.line_end,
            span.content_digest,
        ),
        None
        if buffer is None
        else (
            buffer.key,
            buffer.side,
            buffer.path,
            buffer.content_digest,
            buffer.byte_length,
            buffer.coverage,
            buffer.parse_status,
        ),
        None
        if binding is None
        else (
            binding.buffer_key,
            binding.fragment_key,
            binding.start_line,
            binding.region,
        ),
        availability,
    )


_REVISION_PAIR_SIZE = 2

_CITATION_SCOPE_TOKEN = object()
_INDEXED_PLANNING_SOURCE_TOKEN = object()


@dataclass(frozen=True, slots=True)
class _DetailContextBinding:
    """Scope-local identity witness for one owned data-only detail context.

    Attributes:
        graph: Exact shallow graph view, or ``None`` for stage/arena contexts.
        arena_reader: Exact private arena projection, or ``None`` otherwise.
        references: Primitive address fields in the original detail order.
        unit_keys: Detail unit keys in the original order.
        reference_map: Exact immutable unit-to-reference map.
        typed_mapping: Exact immutable typed-stage map, if present.
        backing_tables: Exact immutable graph or arena table index.
        backing_result_ids: Exact arena result identity index, if present.
        detail_digests: Exact immutable semantic identities captured at validation.
        detail_raw_digests: Type-preserving integrity hashes for typed details.
        observation_ids: Ordered claim identifiers captured from validated details.
        detail_counts: Nested record counts used to build compact selectors.
    """

    graph: object | None
    arena_reader: object | None
    references: tuple[tuple[str, str, str, str], ...]
    unit_keys: tuple[str, ...]
    reference_map: object
    typed_mapping: object
    backing_tables: object | None
    backing_result_ids: object | None
    detail_digests: object
    detail_raw_digests: object
    observation_ids: object
    detail_counts: object


@dataclass(frozen=True, slots=True)
class _IndexedPlanningSourceBinding:
    """Scope-local ownership for one hierarchy source adapter.

    Attributes:
        context_ref: Callable weak reference to the context that issued the adapter.
        witness: Immutable adapter binding tuple captured at registration.
        field_memo: Bounded encoded-field memo released when the scope closes.
    """

    context_ref: Callable[[], IndexedDetailContext | None]
    witness: object
    field_memo: MutableMapping[object, object]


class _CitationScopeOwner:
    """Factory-owned binding and memo state for one exact scope object.

    Attributes:
        detail_contexts: Weak, exact registrations for accepted detail contexts.
        indexed_planning_sources: Weak source adapters scoped to this owner.
        memo: Scope-local current-source citation resolutions.
        closed: Whether close released all context and source memo state.
    """

    __slots__ = (
        "availability_by_source",
        "closed",
        "detail_contexts",
        "explicit_revision_binding",
        "index",
        "indexed_planning_sources",
        "ledger",
        "memo",
        "resolution_count",
        "revision_binding",
        "scope_ref",
    )

    def __init__(  # noqa: PLR0913 # The owner captures the complete local binding.
        self,
        token: object,
        scope: CitationResolutionScope,
        ledger: EvidenceLedger,
        index: LedgerIndex,
        revision_binding: tuple[str, str],
        explicit_revision_binding: bool,  # noqa: FBT001 # Preserve explicit Git revision binding semantics.
        availability_by_source: Mapping[tuple[str, str], tuple[object, ...]],
    ) -> None:
        _require(
            token is _CITATION_SCOPE_TOKEN, "invalid_citation_scope_owner"
        )
        self.scope_ref = weakref_ref(scope)
        self.ledger: EvidenceLedger | None = ledger
        self.index: LedgerIndex | None = index
        self.revision_binding = revision_binding
        self.explicit_revision_binding = explicit_revision_binding
        self.availability_by_source = availability_by_source
        self.memo: dict[tuple[object, ...], SourceSpan] = {}
        self.closed = False
        self.resolution_count = 0
        self.detail_contexts: WeakKeyDictionary = WeakKeyDictionary()
        self.indexed_planning_sources: WeakKeyDictionary = WeakKeyDictionary()


class CitationResolutionScope:
    """Run-local citation memo bound to one live ledger and its physical source.

    The scope stores resolved spans only while it is open. It is not proof
    authority. Public resolution methods strictly capture caller models and
    return detached spans. Indexed validators use the private checked path
    after validating their owned detail records.
    """

    __slots__ = ("__weakref__", "_owner")

    def __init__(
        self,
        ledger: EvidenceLedger,
        index: LedgerIndex,
        *,
        revision_binding: tuple[str, str] | None = None,
    ) -> None:
        """Bind to the existing validated index; never build a replacement."""
        _require(type(index) is LedgerIndex, "invalid_ledger_index")
        _require(type(ledger) is EvidenceLedger, "invalid_evidence_ledger")
        _require(index.validated_ledger() is ledger, "ledger_index_mismatch")
        explicit = revision_binding is not None
        revisions = (
            _revision_pair(revision_binding)
            if explicit
            else (
                _ledger_side_revision(ledger, index, "base"),
                _ledger_side_revision(ledger, index, "head"),
            )
        )
        availability_by_source = MappingProxyType(
            {
                (item.side, item.path): (
                    item.side,
                    item.path,
                    item.counterpart_path,
                    item.state,
                    item.coverage,
                    item.reason,
                    item.full_source,
                    item.full_and_parsed,
                )
                for item in ledger.availability
            }
        )
        object.__setattr__(
            self,
            "_owner",
            _CitationScopeOwner(
                _CITATION_SCOPE_TOKEN,
                self,
                ledger,
                index,
                revisions,
                explicit,
                availability_by_source,
            ),
        )

    def _owned_state(self) -> _CitationScopeOwner:
        owner = self._owner
        _require(
            type(owner) is _CitationScopeOwner and owner.scope_ref() is self,
            "invalid_citation_scope_owner",
        )
        return owner

    def __setattr__(self, name: str, value: object) -> None:
        """Keep ledger, index, and revision bindings fixed after capture."""
        if hasattr(self, name):
            msg = "citation_scope_binding_is_immutable"
            raise AttributeError(msg)
        object.__setattr__(self, name, value)

    @property
    def revision_binding(self) -> tuple[str, str]:
        """Return the immutable base/head revision pair used by this scope."""
        return self._owned_state().revision_binding

    @property
    def memoized_reference_count(self) -> int:
        """Return the number of physical citation bindings cached locally."""
        return len(self._owned_state().memo)

    @property
    def resolution_count(self) -> int:
        """Count successful calls to the source span resolver in this scope."""
        return self._owned_state().resolution_count

    @property
    def closed(self) -> bool:
        """Report whether memoized resolution state has been released."""
        return self._owned_state().closed

    def _check_current(
        self,
        ledger: EvidenceLedger,
        index: LedgerIndex,
        revision_binding: tuple[str, str] | None,
    ) -> None:
        owner = self._owned_state()
        _require(not owner.closed, "citation_scope_closed")
        _require(
            ledger is owner.ledger and index is owner.index,
            "citation_scope_binding_mismatch",
        )
        _require(type(index) is LedgerIndex, "invalid_ledger_index")
        _require(index.validated_ledger() is ledger, "ledger_index_mismatch")
        if owner.explicit_revision_binding:
            _require(
                revision_binding is not None,
                "citation_scope_revision_binding_required",
            )
        if revision_binding is not None:
            _require(
                _revision_pair(revision_binding) == owner.revision_binding,
                "citation_scope_revision_mismatch",
            )

    def _resolve_validated(
        self, ref: SpanRef, unit: EvidenceUnit
    ) -> SourceSpan:
        owner = self._owned_state()
        _require(not owner.closed, "citation_scope_closed")
        index = owner.index
        _require(index is not None, "citation_scope_closed")
        _require(
            index.units.get(unit.id) is unit,
            "citation_scope_unit_mismatch",
        )
        key = _citation_binding_key(
            ref,
            unit,
            index,
            owner.revision_binding,
            owner.availability_by_source,
        )
        resolved = owner.memo.get(key)
        if resolved is None:
            resolved = _resolve_span_ref(ref, unit, index)
            owner.memo[key] = resolved
            owner.resolution_count += 1
        return resolved

    def _check_index(self, index: LedgerIndex) -> None:
        """Require indexed detail checks to use the scope's original index."""
        owner = self._owned_state()
        _require(not owner.closed, "citation_scope_closed")
        _require(owner.index is index, "citation_scope_binding_mismatch")

    def _register_detail_context(self, context: IndexedDetailContext) -> None:
        """Register one exact detail context within this source scope.

        Preconditions:
            The context points to this scope's original ledger and index.
        Postconditions:
            The local owner pins graph/reader identity, ordered references, and
            unit order for subsequent liveness checks.
        Essential behavior:
            Stores identity and primitive witnesses without a global registry.
        """
        owner = self._owned_state()
        _require(not owner.closed, "citation_scope_closed")
        _require(
            context._citation_scope is self  # noqa: SLF001 # Context ownership is local to this scope.
            and context._ledger is owner.ledger  # noqa: SLF001 # Compare exact bound objects.
            and context._index is owner.index,  # noqa: SLF001 # Compare exact bound objects.
            "detail_context_binding_mismatch",
        )
        backing_tables = None
        backing_result_ids = None
        if context._graph is not None:  # noqa: SLF001 # Pin the exact graph table view.
            backing_tables = getattr(context._graph, "_tables", None)  # noqa: SLF001 # Retain only the immutable index object.
        elif context._arena_reader is not None:  # noqa: SLF001 # Pin the exact arena table view.
            backing_tables = getattr(context._arena_reader, "_tables", None)  # noqa: SLF001 # Retain only the immutable index object.
            backing_result_ids = getattr(
                context._arena_reader,  # noqa: SLF001 # Pin item-key bindings.
                "_result_ids",
                None,
            )
        owner.detail_contexts[context] = _DetailContextBinding(
            graph=context._graph,  # noqa: SLF001 # Pin the original shallow view.
            arena_reader=context._arena_reader,  # noqa: SLF001 # Pin the original arena projection.
            references=tuple(
                (
                    item.table,
                    item.kind,
                    item.storage_id,
                    item.item_key,
                )
                for item in context._references  # noqa: SLF001 # Store scalar witnesses only.
            ),
            unit_keys=context._unit_keys,  # noqa: SLF001 # Preserve original order.
            reference_map=context._references_by_unit,  # noqa: SLF001 # Pin the immutable unit-to-address map.
            typed_mapping=context._typed_by_unit,  # noqa: SLF001 # Pin the immutable typed-stage map.
            backing_tables=backing_tables,
            backing_result_ids=backing_result_ids,
            detail_digests=context._detail_digests,  # noqa: SLF001 # Pin compact semantic identities.
            detail_raw_digests=context._detail_raw_digests,  # noqa: SLF001 # Pin type-preserving integrity hashes.
            observation_ids=context._observation_ids,  # noqa: SLF001 # Pin ordered claim facts.
            detail_counts=context._detail_counts,  # noqa: SLF001 # Pin nested record counts.
        )

    def _owns_detail_context(self, context: IndexedDetailContext) -> bool:
        """Check identity registration without a process-global authority map.

        Preconditions:
            ``context`` is an indexed detail context presented to its scope.
        Postconditions:
            Returns true only while every pinned identity and scalar witness
            still matches the registered binding.
        Essential behavior:
            Rejects closure, object rebinding, reference mutation, and order
            changes before the context is used.
        """
        owner = self._owned_state()
        binding = owner.detail_contexts.get(context)
        backing_tables = None
        backing_result_ids = None
        if context._graph is not None:  # noqa: SLF001 # Read the registered graph view.
            backing_tables = getattr(context._graph, "_tables", None)  # noqa: SLF001 # Compare its immutable index identity.
        elif context._arena_reader is not None:  # noqa: SLF001 # Read the registered arena view.
            backing_tables = getattr(context._arena_reader, "_tables", None)  # noqa: SLF001 # Compare its immutable index identity.
            backing_result_ids = getattr(
                context._arena_reader,  # noqa: SLF001 # Compare exact result bindings.
                "_result_ids",
                None,
            )
        return bool(
            not owner.closed
            and binding is not None
            and context._citation_scope is self  # noqa: SLF001 # Verify the original local scope.
            and context._ledger is owner.ledger  # noqa: SLF001 # Verify the original ledger.
            and context._index is owner.index  # noqa: SLF001 # Verify the original checked index.
            and context._graph is binding.graph  # noqa: SLF001 # Reject graph rebinding.
            and context._arena_reader is binding.arena_reader  # noqa: SLF001 # Reject arena reader rebinding.
            and context._unit_keys == binding.unit_keys  # noqa: SLF001 # Reject order changes.
            and context._references_by_unit is binding.reference_map  # noqa: SLF001 # Reject map rebinding.
            and context._typed_by_unit is binding.typed_mapping  # noqa: SLF001 # Reject typed map rebinding.
            and tuple(
                (
                    item.table,
                    item.kind,
                    item.storage_id,
                    item.item_key,
                )
                for item in context._references  # noqa: SLF001 # Compare type-preserving scalar addresses.
            )
            == binding.references
            and context._references_by_unit is binding.reference_map  # noqa: SLF001 # Reject unit-to-reference map rebinding.
            and context._typed_by_unit is binding.typed_mapping  # noqa: SLF001 # Reject typed-stage map rebinding.
            and backing_tables is binding.backing_tables
            and backing_result_ids is binding.backing_result_ids
            and context._detail_digests is binding.detail_digests  # noqa: SLF001 # Keep detail identity facts fixed.
            and context._detail_raw_digests is binding.detail_raw_digests  # noqa: SLF001 # Keep raw-type witnesses fixed.
            and context._observation_ids is binding.observation_ids  # noqa: SLF001 # Keep group facts fixed.
            and context._detail_counts is binding.detail_counts  # noqa: SLF001 # Keep selected record counts fixed.
        )

    def _register_indexed_planning_sources(
        self,
        context: IndexedDetailContext,
        sources: object,
        witness: object,
        field_memo: MutableMapping[object, object],
        *,
        token: object,
    ) -> None:
        """Register hierarchy material access inside the owning detail scope.

        Preconditions:
            ``context`` is live in this scope; ``sources`` is the newly created
            identity-bearing adapter; and ``field_memo`` is its bounded cache.
        Postconditions:
            The exact adapter, context identity, witness, and memo are registered
            only for this scope's lifetime.
        Essential behavior:
            Avoids any process-global registry for indexed source authority.
        """
        owner = self._owned_state()
        _require(
            token is _INDEXED_PLANNING_SOURCE_TOKEN
            and not owner.closed
            and self._owns_detail_context(context),
            "unregistered_detail_context",
        )
        owner.indexed_planning_sources[sources] = (
            _IndexedPlanningSourceBinding(
                context_ref=weakref_ref(context),
                witness=witness,
                field_memo=field_memo,
            )
        )

    def _owns_indexed_planning_sources(
        self,
        context: IndexedDetailContext,
        sources: object,
        witness: object,
    ) -> bool:
        """Check exact live registration for one indexed source adapter.

        Preconditions:
            ``sources`` is an adapter presented with its original detail context.
        Postconditions:
            Returns true only when the scope is open and the original witness
            and context identity still match.
        Essential behavior:
            Rejects copied, rebound, closed, and cross-context source adapters.
        """
        owner = self._owned_state()
        binding = owner.indexed_planning_sources.get(sources)
        return bool(
            not owner.closed
            and binding is not None
            and binding.context_ref() is context
            and binding.witness == witness
            and self._owns_detail_context(context)
        )

    def resolve_many(
        self,
        refs: Sequence[SpanRef],
        unit: EvidenceUnit,
        *,
        revision_binding: tuple[str, str] | None = None,
    ) -> tuple[SourceSpan, ...]:
        """Strictly capture references and return detached current source spans.

        Use this batch API for later indexed assembly. It checks current ledger
        integrity once for the batch, then resolves each distinct physical
        reference at most once in this scope.
        """
        owner = self._owned_state()
        ledger = owner.ledger
        index = owner.index
        _require(
            ledger is not None and index is not None, "citation_scope_closed"
        )
        self._check_current(ledger, index, revision_binding)
        checked_unit = _indexed_unit(unit, index)
        checked_refs = tuple(SpanRef.model_validate(ref) for ref in refs)
        return tuple(
            self._resolve_validated(ref, checked_unit).model_copy(deep=True)
            for ref in checked_refs
        )

    def resolve(
        self,
        ref: SpanRef,
        unit: EvidenceUnit,
        *,
        revision_binding: tuple[str, str] | None = None,
    ) -> SourceSpan:
        """Strictly resolve one caller reference and return a detached span."""
        return self.resolve_many(
            (ref,), unit, revision_binding=revision_binding
        )[0]

    def close(self) -> None:
        """Release spans, indexed field bytes, and bound ledger/index references.

        Postconditions:
            The scope is closed and no source-field memo remains registered.
        Essential behavior:
            Ends the lifetime of both citation and indexed-planning caches.
        """
        owner = self._owned_state()
        owner.memo.clear()
        for binding in tuple(owner.indexed_planning_sources.values()):
            binding.field_memo.clear()
        owner.indexed_planning_sources.clear()
        owner.detail_contexts.clear()
        owner.ledger = None
        owner.index = None
        owner.availability_by_source = MappingProxyType({})
        owner.closed = True

    def __enter__(self) -> Self:
        """Return this live scope for a managed block."""
        _require(not self._owned_state().closed, "citation_scope_closed")
        return self

    def __exit__(self, *_exc: object) -> None:
        """Close the scope and release its memo on block exit."""
        self.close()

    def __copy__(self) -> CitationResolutionScope:
        """Do not transfer memo state into a copied scope."""
        msg = "citation_scope_cannot_be_copied"
        raise TypeError(msg)

    def __deepcopy__(
        self, _memo: dict[int, object]
    ) -> CitationResolutionScope:
        """Do not transfer memo state into a copied scope."""
        msg = "citation_scope_cannot_be_copied"
        raise TypeError(msg)

    def __reduce_ex__(self, _protocol: int) -> object:
        """Prevent serialized scope state from becoming reusable."""
        msg = "citation_scope_cannot_be_serialized"
        raise TypeError(msg)


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
    refs: tuple[SpanRef, ...],
    unit: EvidenceUnit,
    index: LedgerIndex,
    *,
    citation_scope: CitationResolutionScope | None = None,
) -> tuple[SourceSpan, ...]:
    _unique([ref.span_id for ref in refs], "duplicate_citation")
    if citation_scope is not None:
        citation_scope._check_index(index)  # noqa: SLF001 # Keep indexed use on this exact bound index.
        return tuple(
            citation_scope._resolve_validated(ref, unit)  # noqa: SLF001 # Owned detail records use the strict internal path.
            for ref in refs
        )
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
    result: DetailResult,
    unit: EvidenceUnit,
    index: LedgerIndex,
    *,
    citation_scope: CitationResolutionScope | None = None,
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
            for span in _validate_citations(
                explanation.citations,
                unit,
                index,
                citation_scope=citation_scope,
            )
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
        _validate_citations(
            observation.citations,
            unit,
            index,
            citation_scope=citation_scope,
        )
    for question in result.questions:
        _validate_citations(
            question.citations,
            unit,
            index,
            citation_scope=citation_scope,
        )
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


_DETAIL_CONTEXT_TOKEN = object()


class _DetailRecordReader(Protocol):
    """Narrow immutable result reader for arena-backed detail contexts.

    Implementations expose owned primitive records and detached selected
    results. They do not reconstruct a manifest or retain typed results.
    """

    def _result_view(self, reference: ResultRef) -> Mapping[str, object]:
        """Return one arena-owned normalized result row.

        Preconditions:
            ``reference`` is an exact detail address owned by the reader.
        Postconditions:
            The row and nested values are immutable primitives.
        Essential behavior:
            Selects one captured row without model parsing or reconstruction.
        """

    def _evidence_view(
        self, pointer: Mapping[str, object], *, kind: str
    ) -> Mapping[str, object]:
        """Return one arena-owned evidence-reference row.

        Preconditions:
            ``pointer`` has the exact strict address shape and requested kind.
        Postconditions:
            The selected evidence mapping is immutable.
        Essential behavior:
            Resolves the captured nested reference without revalidating its
            structural model.
        """

    def result_record(self, reference: ResultRef) -> dict:
        """Return a detached logical result selected by its address.

        Preconditions:
            ``reference`` is an exact address owned by the reader.
        Postconditions:
            The returned mapping is detached and may be changed by its caller.
        Essential behavior:
            Materializes only the explicitly selected result for access.
        """


def _tuple_detail_record(value: object) -> object:
    """Adapt one owned JSON-shaped detail record to strict tuple contracts.

    Storage records use JSON arrays, while the typed contracts use tuples.
    This conversion changes only those enumerated array positions. It keeps
    every scalar untouched, so bool/int and other type substitutions remain
    visible to strict Pydantic validation.
    """
    if type(value) is not dict:
        return value
    record = dict(value)
    for field_name in ("explanations", "observations", "questions"):
        entries = record.get(field_name)
        if type(entries) not in {list, tuple}:
            continue
        captured_entries = []
        for entry in entries:
            if type(entry) is not dict:
                captured_entries.append(entry)
                continue
            captured = dict(entry)
            for array_name in (
                "obligation_ids",
                "citations",
                "limitations",
            ):
                nested = captured.get(array_name)
                if type(nested) is list:
                    captured[array_name] = tuple(nested)
            captured_entries.append(captured)
        record[field_name] = tuple(captured_entries)
    return record


def _detach_detail_value(value: object) -> object:
    """Copy one selected detail value without changing its scalar types.

    Preconditions:
        ``value`` is an owned JSON-shaped value or typed immutable contract.
    Postconditions:
        Mappings and sequences are detached; booleans, integers, and strings
        retain their exact Python types.
    Essential behavior:
        Supports selected indexed detail fields without adapting a full result.
    """
    if isinstance(value, BaseModel):
        return value.model_copy(deep=True)
    if isinstance(value, Mapping):
        return {key: _detach_detail_value(item) for key, item in value.items()}
    if type(value) is tuple:
        return tuple(_detach_detail_value(item) for item in value)
    if type(value) is list:
        return [_detach_detail_value(item) for item in value]
    if value is None or type(value) in {str, bool, int, float}:
        return value
    _require(type(value) is bytes, "invalid_typed_detail_record")
    return bytes(value)


@dataclass(frozen=True, slots=True)
class _DetailExplanationView:
    """Ephemeral explanation fields for ledger contextual checks.

    Attributes:
        obligation_ids: Non-empty, ordered 64-character lowercase hex
            obligation digests from one owned detail. Contextual validation
            rejects duplicates and unknown digests.
        citations: Ordered, strictly prevalidated ``SpanRef`` values for this
            check; current unit, group, and source bindings are checked later.
        limitations: Ordered, strictly prevalidated ``EvidenceLimitation``
            values for this check. Codes are ``unavailable``, ``partial``,
            ``unpaired_excerpt``, ``unmatched_definition``, ``parse_failed``,
            ``text_only``, or ``unclassified_changed_source``; sides are
            ``base`` or ``head``; paths are repository-relative; and reasons
            are exact non-whitespace text.

    The selected values are temporary and preserve the stored field values and
    array order. They are used only by the existing contextual validator.
    """

    obligation_ids: tuple[str, ...]
    citations: tuple[SpanRef, ...]
    limitations: tuple[EvidenceLimitation, ...]


@dataclass(frozen=True, slots=True)
class _DetailObservationView:
    """Ephemeral observation identity and citation fields for one check.

    Attributes:
        id: Observation digest owned by the complete result record.
        citations: Strictly prevalidated source-span models for this check.
    """

    id: str
    citations: tuple[SpanRef, ...]


@dataclass(frozen=True, slots=True)
class _DetailQuestionView:
    """Ephemeral question citation and obligation fields for one check.

    Attributes:
        citations: Strictly prevalidated source-span models for this check.
        obligation_ids: Ordered obligation digests from the question.
    """

    citations: tuple[SpanRef, ...]
    obligation_ids: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class _DetailResultView:
    """Duck-typed detail projection used only by existing contextual checks.

    Attributes:
        unit_key: Ledger unit digest bound to the selected result address.
        explanations: Ordered obligation explanations for the unit.
        observations: Ordered factual observations for the unit.
        questions: Ordered unresolved questions for the unit.
    """

    unit_key: str
    explanations: tuple[_DetailExplanationView, ...]
    observations: tuple[_DetailObservationView, ...]
    questions: tuple[_DetailQuestionView, ...]


def _detail_result_check_view(
    record: Mapping[str, object], reader: _DetailRecordReader
) -> _DetailResultView:
    """Build one ephemeral contextual projection without model revalidation.

    Preconditions:
        ``record`` is an immutable normalized detail row already structurally
        validated by its owning arena. ``reader`` resolves its typed span refs.
    Postconditions:
        The returned plain dataclasses contain only values needed by the full
        ledger-unit, obligation, citation, and context checks.
    Essential behavior:
        No complete ``DetailResult`` or second detail snapshot is retained.
        Limitation rows become temporary typed values only after their owning
        arena has strictly parsed the complete result.
    """
    _require(
        type(record.get("unit_key")) is str,
        "detail_result_reference_mismatch",
    )

    def array(value: object) -> tuple:
        """Read one already checked ordered result array.

        Preconditions:
            ``value`` is a JSON list or immutable tuple from an owned row.
        Postconditions:
            Returns the same items in an ephemeral tuple.
        Essential behavior:
            Preserves item order and scalar types for contextual validation.
        """
        _require(type(value) in {tuple, list}, "invalid_typed_storage_record")
        return tuple(value)

    def spans(value: object) -> tuple[SpanRef, ...]:
        """Project validated span rows into temporary lightweight models.

        Preconditions:
            Each item in ``value`` is a strict span address resolved by reader.
        Postconditions:
            Returns temporary ``SpanRef`` values for existing contextual checks.
        Essential behavior:
            Uses model construction only after the arena's structural parse.
        """
        output = []
        for pointer in array(value):
            raw = reader._evidence_view(pointer, kind="span_ref")  # noqa: SLF001 # Resolve the arena-owned strict reference.
            output.append(SpanRef.model_construct(**dict(raw)))
        return tuple(output)

    def limitations(value: object) -> tuple[EvidenceLimitation, ...]:
        """Project validated limitation rows into temporary typed values.

        Preconditions:
            Each item in ``value`` is an arena-owned mapping that already
            passed the strict ``EvidenceLimitation`` structural validation.
        Postconditions:
            Returns the exact code, side, path, reason, and order in temporary
            ``EvidenceLimitation`` values.
        Essential behavior:
            Uses model construction only after the arena's structural parse;
            it does not supply defaults or normalize stored fields.
        """
        output = []
        for raw in array(value):
            _require(isinstance(raw, Mapping), "invalid_typed_storage_record")
            output.append(EvidenceLimitation.model_construct(**dict(raw)))
        return tuple(output)

    explanations = []
    for item in array(record.get("explanations", ())):
        _require(isinstance(item, Mapping), "invalid_typed_storage_record")
        explanations.append(
            _DetailExplanationView(
                obligation_ids=array(item.get("obligation_ids", ())),
                citations=spans(item.get("citations", ())),
                limitations=limitations(item["limitations"]),
            )
        )
    observations = []
    for item in array(record.get("observations", ())):
        _require(
            isinstance(item, Mapping) and type(item.get("id")) is str,
            "invalid_typed_storage_record",
        )
        observations.append(
            _DetailObservationView(
                id=item["id"], citations=spans(item.get("citations", ()))
            )
        )
    questions = []
    for item in array(record.get("questions", ())):
        _require(isinstance(item, Mapping), "invalid_typed_storage_record")
        questions.append(
            _DetailQuestionView(
                citations=spans(item.get("citations", ())),
                obligation_ids=array(item.get("obligation_ids", ())),
            )
        )
    return _DetailResultView(
        unit_key=record["unit_key"],
        explanations=tuple(explanations),
        observations=tuple(observations),
        questions=tuple(questions),
    )


class IndexedDetailContext:
    """Owned, data-only detail records checked against one current ledger.

    Graph-backed contexts keep immutable graph addresses and a private shallow
    view over the graph's immutable primitive tables. Arena-backed contexts
    keep ordered result addresses and a private reader projection over owned
    primitive records. Stage-backed contexts keep detached typed details.
    Public accessors return fresh dictionaries, so returned mutations cannot
    change context records. No form is a generation or report capability.

    Attributes:
        unit_keys: Exact detail-unit membership in captured input order.
        references: Result addresses for graph/arena contexts; empty for stage
            model contexts. Access returns fresh typed reference objects.
        detail_record: Explicit detached access to one selected logical detail.
        planning_detail_facts: Compact semantic digest, observation IDs, and
            nested record counts captured by the strict validator.
        detail raw hashes: Type-preserving integrity fingerprints for mutable
            typed-stage models; these are not source or record identities.
        indexed_detail_record_view: Detached view of one selected nested row.
    """

    __slots__ = (
        "__weakref__",
        "_arena_reader",
        "_citation_scope",
        "_detail_counts",
        "_detail_digests",
        "_detail_raw_digests",
        "_graph",
        "_index",
        "_ledger",
        "_observation_ids",
        "_references",
        "_references_by_unit",
        "_typed_by_unit",
        "_unit_keys",
    )

    def __init__(  # noqa: PLR0913 # The private constructor receives the complete owned state.
        self,
        *,
        token: object,
        ledger: EvidenceLedger,
        index: LedgerIndex,
        citation_scope: CitationResolutionScope,
        unit_keys: tuple[str, ...],
        graph: ReportStorageView | None = None,
        arena_reader: _DetailRecordReader | None = None,
        references: tuple[ResultRef, ...] = (),
        typed_by_unit: Mapping[str, DetailResult] | None = None,
        detail_digests: Mapping[str, str],
        detail_raw_digests: Mapping[str, str],
        observation_ids: Mapping[str, tuple[str, ...]],
        detail_counts: Mapping[str, tuple[int, int, int]],
    ) -> None:
        """Store one graph-backed, arena-backed, or typed stage context.

        Preconditions:
            ``token`` is module-private and the selected backing matches the
            supplied references or typed-detail mapping.
        Postconditions:
            Ordered unit keys, references, and backing selectors are detached
            and immutable to ordinary callers. Compact semantic identities,
            raw-type integrity hashes, observation IDs, and nested record
            counts match validated details.
        Essential behavior:
            Arena-backed contexts keep a reader and addresses, not complete
            detail models or duplicate logical snapshots. Validation captures
            only small detail facts for later planning-source construction.
        """
        _require(token is _DETAIL_CONTEXT_TOKEN, "invalid_detail_context")
        self._ledger = ledger
        self._index = index
        self._citation_scope = citation_scope
        self._graph = graph
        self._arena_reader = arena_reader
        self._references = references
        self._references_by_unit = MappingProxyType(
            dict(zip(unit_keys, references, strict=True))
            if graph is not None or arena_reader is not None
            else {}
        )
        self._typed_by_unit = MappingProxyType(dict(typed_by_unit or {}))
        self._detail_digests = MappingProxyType(dict(detail_digests))
        self._detail_raw_digests = MappingProxyType(dict(detail_raw_digests))
        self._observation_ids = MappingProxyType(dict(observation_ids))
        self._detail_counts = MappingProxyType(dict(detail_counts))
        self._unit_keys = unit_keys

    def __setattr__(self, name: str, value: object) -> None:
        """Keep captured detail ownership and backing records fixed."""
        if hasattr(self, name):
            msg = "indexed_detail_context_is_immutable"
            raise AttributeError(msg)
        object.__setattr__(self, name, value)

    def _check_registered(self) -> None:
        _require(
            self._citation_scope._owns_detail_context(self),  # noqa: SLF001 # Verify identity in the local owner.
            "unregistered_detail_context",
        )

    @property
    def citation_scope(self) -> CitationResolutionScope:
        """Return the local memo used by indexed detail citation checks."""
        self._check_registered()
        return self._citation_scope

    @property
    def unit_keys(self) -> tuple[str, ...]:
        """Return detail unit keys in the captured input order."""
        self._check_registered()
        return self._unit_keys

    @property
    def references(self) -> tuple[ResultRef, ...]:
        """Return fresh typed graph addresses in the captured input order."""
        self._check_registered()
        return tuple(_copy_result_ref(ref) for ref in self._references)

    def contains_unit(self, unit_key: str) -> bool:
        """Check membership without revalidating or resolving the detail."""
        self._check_registered()
        return unit_key in self._unit_keys

    def planning_detail_facts(
        self, unit_key: str
    ) -> tuple[str, tuple[str, ...], tuple[int, int, int]]:
        """Return the small, already validated facts used by planning sources.

        Preconditions:
            The context is registered and ``unit_key`` is in its ordered set.
        Postconditions:
            Returns the exact semantic detail digest, observation IDs, and
            explanation/observation/question counts captured at validation.
        Essential behavior:
            Does not expand or revalidate a complete detail result.
        """
        self._check_registered()
        digest = self._detail_digests.get(unit_key)
        observations = self._observation_ids.get(unit_key)
        counts = self._detail_counts.get(unit_key)
        _require(
            digest is not None
            and observations is not None
            and counts is not None,
            "unknown_detail_unit",
        )
        return digest, observations, counts

    def assert_detail_witnesses_current(self) -> None:
        """Reject bypass mutation of typed details captured by this context.

        Preconditions:
            The context is registered and open.
        Postconditions:
            Typed details still match their original semantic digests; graph and
            arena contexts retain their registered immutable backing selectors.
        Essential behavior:
            Checks current detail data without nested model revalidation.
        """
        self._check_registered()
        if self._typed_by_unit:
            for unit_key, detail in self._typed_by_unit.items():
                _require(
                    content_digest(_raw_integrity_bytes(detail))
                    == self._detail_raw_digests.get(unit_key),
                    "detail_context_changed",
                )

    def indexed_detail_record_view(
        self, unit_key: str, record_kind: str, ordinal: int
    ) -> Mapping[str, object]:
        """Resolve one selected explanation, observation, or question record.

        Preconditions:
            The context is registered; ``record_kind`` is one of the three
            detail families; and ``ordinal`` selects an existing row.
        Postconditions:
            Returns a detached mapping with the exact logical fields and span
            citation records for only that selected nested record.
        Essential behavior:
            Reads one nested row from owned selectors without reconstructing or
            parsing the complete detail result.
        """
        self._check_registered()
        _require(
            type(unit_key) is str
            and record_kind in {"explanation", "observation", "question"}
            and type(ordinal) is int
            and ordinal >= 0,
            "invalid_detail_record_selector",
        )
        family, model = {
            "explanation": ("explanations", Explanation),
            "observation": ("observations", FactualObservation),
            "question": ("questions", EvidenceQuestion),
        }[record_kind]
        if self._typed_by_unit:
            detail = self._typed_by_unit.get(unit_key)
            _require(detail is not None, "unknown_detail_unit")
            records = getattr(detail, family)
            _require(ordinal < len(records), "unknown_detail_record")
            selected = records[ordinal]
            _require(type(selected) is model, "invalid_typed_detail_record")
            return {
                name: copy.deepcopy(getattr(selected, name))
                for name in model.model_fields
            }

        reference = self._references_by_unit.get(unit_key)
        _require(reference is not None, "unknown_detail_unit")
        if self._graph is not None:
            entry = self._graph._tables["results"].get(  # noqa: SLF001 # Read one immutable normalized result row.
                reference.storage_id
            )
            _require(
                entry is not None
                and entry["kind"] == "detail"
                and entry["record"].get("unit_key") == unit_key,
                "detail_result_reference_mismatch",
            )
            row = entry["record"]
        else:
            _require(self._arena_reader is not None, "invalid_detail_context")
            row = self._arena_reader._result_view(  # noqa: SLF001 # Select a checked arena row without materialization.
                _copy_result_ref(reference)
            )
            _require(
                row.get("unit_key") == unit_key,
                "detail_result_reference_mismatch",
            )
        records = row.get(family)
        _require(
            type(records) in {tuple, list} and ordinal < len(records),
            "unknown_detail_record",
        )
        selected = records[ordinal]
        _require(isinstance(selected, Mapping), "invalid_typed_detail_record")
        _require(
            set(selected) == set(model.model_fields),
            "invalid_typed_detail_record",
        )
        output = {
            name: _detach_detail_value(selected[name])
            for name in model.model_fields
        }
        citations = output["citations"]
        _require(
            type(citations) in {tuple, list}, "invalid_typed_detail_record"
        )
        output["citations"] = tuple(
            self._indexed_span_record(pointer) for pointer in citations
        )
        return output

    def _indexed_span_record(self, pointer: object) -> Mapping[str, object]:
        """Resolve one validated nested citation to its current logical record."""
        if self._arena_reader is not None:
            value = self._arena_reader._evidence_view(  # noqa: SLF001 # Use the reader's strict evidence selector.
                pointer, kind="span_ref"
            )
            return _detach_detail_value(value)
        _require(
            self._graph is not None and isinstance(pointer, Mapping),
            "invalid_typed_detail_record",
        )
        _require(
            set(pointer) == {"$ref", "kind"}
            and type(pointer.get("$ref")) is str
            and pointer.get("kind") == "span_ref",
            "invalid_typed_detail_record",
        )
        entry = self._graph._tables["evidence_refs"].get(  # noqa: SLF001 # Resolve one accepted graph reference.
            pointer["$ref"]
        )
        _require(
            entry is not None and entry["kind"] == "span_ref",
            "invalid_typed_detail_record",
        )
        return _detach_detail_value(entry["record"])

    def detail_record(self, unit_key: str) -> dict:
        """Return one detached full detail mapping for a validated unit.

        Preconditions:
            The context is registered and live, and ``unit_key`` was captured.
        Postconditions:
            The returned logical detail is mutable and cannot change context
            records or future access results.
        Essential behavior:
            Materializes only the selected result from its owned backing.
        """
        self._check_registered()
        if self._arena_reader is not None:
            reference = self._references_by_unit.get(unit_key)
            _require(reference is not None, "unknown_detail_unit")
            return self._arena_reader.result_record(
                _copy_result_ref(reference)
            )
        if self._graph is not None:
            reference = self._references_by_unit.get(unit_key)
            _require(reference is not None, "unknown_detail_unit")
            return self._graph.result_record(_copy_result_ref(reference))
        result = self._typed_by_unit.get(unit_key)
        _require(result is not None, "unknown_detail_unit")
        return result.model_dump(mode="python")

    def assert_current(
        self,
        ledger: EvidenceLedger,
        index: LedgerIndex,
        *,
        revision_binding: tuple[str, str] | None = None,
    ) -> None:
        """Recheck current ledger/index binding before a later proof boundary."""
        self._check_registered()
        self._citation_scope._check_current(  # noqa: SLF001 # Context delegates integrity to its bound scope.
            ledger, index, revision_binding
        )
        _require(
            ledger is self._ledger and index is self._index,
            "detail_context_binding_mismatch",
        )

    def __copy__(self) -> IndexedDetailContext:
        """Keep ownership with the original validated context."""
        msg = "indexed_detail_context_cannot_be_copied"
        raise TypeError(msg)

    def __deepcopy__(self, _memo: dict[int, object]) -> IndexedDetailContext:
        """Keep ownership with the original validated context."""
        msg = "indexed_detail_context_cannot_be_copied"
        raise TypeError(msg)

    def __reduce_ex__(self, _protocol: int) -> object:
        """Prevent serialized context data from being treated as verified."""
        msg = "indexed_detail_context_cannot_be_serialized"
        raise TypeError(msg)


def planning_groups_from_indexed_details(
    index: LedgerIndex,
    details_context: IndexedDetailContext,
    *,
    revision_binding: tuple[str, str] | None = None,
) -> tuple[PlanningGroup, ...]:
    """Build ordered claim groups from one already validated detail context.

    Preconditions:
        ``index`` and ``details_context`` are exact live owners for the same
        ledger, and the optional revision binding matches their local scope.
    Postconditions:
        Returns the same group and claim ordering as
        :func:`planning_groups_from_details` without parsing details again.
    Essential behavior:
        Reads only captured observation IDs and current group/unit membership.
    """
    _require(
        type(index) is LedgerIndex
        and type(details_context) is IndexedDetailContext,
        "invalid_indexed_detail_context",
    )
    details_context.assert_current(
        details_context._ledger,  # noqa: SLF001 # Reuse the context's exact owner ledger.
        index,
        revision_binding=revision_binding,
    )
    details_context.assert_detail_witnesses_current()
    _require(
        Counter(details_context.unit_keys) == Counter(index.units.keys()),
        "detail_set_coverage_mismatch",
    )
    observations = {
        unit_key: details_context.planning_detail_facts(unit_key)[1]
        for unit_key in details_context.unit_keys
    }
    return tuple(
        PlanningGroup(
            group=group,
            claims=tuple(
                ClaimRef(unit_key=unit_key, observation_id=observation_id)
                for unit_key in group.unit_keys
                for observation_id in observations[unit_key]
            ),
            terminology=(),
        )
        for group in index.groups.values()
    )


def indexed_detail_semantic_digest(
    details_context: IndexedDetailContext, unit_key: str
) -> str:
    """Return one captured DetailResult semantic digest without expanding it.

    Preconditions:
        ``details_context`` is live and ``unit_key`` belongs to its validated
        detail set.
    Postconditions:
        The digest equals ``content_digest(canonical_bytes(DetailResult))``,
        including the existing default ``version`` field.
    Essential behavior:
        Reads only the compact identity retained during strict detail capture.
    """
    _require(
        type(details_context) is IndexedDetailContext,
        "invalid_indexed_detail_context",
    )
    return details_context.planning_detail_facts(unit_key)[0]


def indexed_detail_record_view(
    details_context: IndexedDetailContext,
    unit_key: str,
    record_kind: str,
    ordinal: int,
) -> Mapping[str, object]:
    """Expose one checked nested detail record to the planning source adapter.

    Preconditions:
        ``details_context`` is the exact registered owner and the selector names
        one existing explanation, observation, or question.
    Postconditions:
        Returns a detached selected-record view without rebuilding its detail.
    Essential behavior:
        Keeps the indexed adapter limited to the detail fields it consumes.
    """
    _require(
        type(details_context) is IndexedDetailContext,
        "invalid_indexed_detail_context",
    )
    return details_context.indexed_detail_record_view(
        unit_key, record_kind, ordinal
    )


def _copy_result_ref(reference: ResultRef) -> ResultRef:
    """Copy one exact typed result reference without accepting subclasses.

    Preconditions:
        ``reference`` is an exact storage ``ResultRef``.
    Postconditions:
        The returned value is a fresh immutable reference with strict scalar
        types checked by its constructor.
    Essential behavior:
        Caller object identity is not retained as a context binding.
    """
    from diffstory.report_storage import ResultRef as StorageResultRef

    _require(
        type(reference) is StorageResultRef, "invalid_storage_result_reference"
    )
    return StorageResultRef(
        reference.table,
        reference.kind,
        reference.storage_id,
        reference.item_key,
    )


def _detail_scope(
    ledger: EvidenceLedger,
    index: LedgerIndex,
    citation_scope: CitationResolutionScope | None,
    revision_binding: tuple[str, str] | None,
) -> tuple[CitationResolutionScope, bool]:
    _require(type(ledger) is EvidenceLedger, "invalid_evidence_ledger")
    _require(type(index) is LedgerIndex, "invalid_ledger_index")
    if citation_scope is None:
        return (
            CitationResolutionScope(
                ledger, index, revision_binding=revision_binding
            ),
            True,
        )
    _require(
        type(citation_scope) is CitationResolutionScope,
        "invalid_citation_scope",
    )
    citation_scope._check_current(  # noqa: SLF001 # Bind the supplied local scope once per validation batch.
        ledger, index, revision_binding
    )
    return citation_scope, False


def _validate_indexed_detail_models(
    results: tuple[DetailResult, ...],
    ledger: EvidenceLedger,
    index: LedgerIndex,
    citation_scope: CitationResolutionScope,
) -> IndexedDetailContext:
    """Validate stage details and capture compact planning facts.

    Preconditions:
        Every result is a strictly parsed ``DetailResult`` and the scope is
        bound to this exact ledger and index.
    Postconditions:
        Returns a registered context with exact semantic digests, ordered
        observation IDs, and nested record counts for every unit.
    Essential behavior:
        Reuses the strict detail validator once and retains the typed stage
        owner without creating canonical snapshots or field-byte copies.
    """
    _require(
        Counter(result.unit_key for result in results)
        == Counter(index.units.keys()),
        "detail_set_coverage_mismatch",
    )
    for result in results:
        _validate_detail_result(
            result,
            index.units[result.unit_key],
            index,
            citation_scope=citation_scope,
        )
    detail_digests = {
        result.unit_key: content_digest(canonical_bytes(result))
        for result in results
    }
    detail_raw_digests = {
        result.unit_key: content_digest(_raw_integrity_bytes(result))
        for result in results
    }
    observation_ids = {
        result.unit_key: tuple(item.id for item in result.observations)
        for result in results
    }
    detail_counts = {
        result.unit_key: (
            len(result.explanations),
            len(result.observations),
            len(result.questions),
        )
        for result in results
    }
    context = IndexedDetailContext(
        token=_DETAIL_CONTEXT_TOKEN,
        ledger=ledger,
        index=index,
        citation_scope=citation_scope,
        unit_keys=tuple(result.unit_key for result in results),
        typed_by_unit={result.unit_key: result for result in results},
        detail_digests=detail_digests,
        detail_raw_digests=detail_raw_digests,
        observation_ids=observation_ids,
        detail_counts=detail_counts,
    )
    citation_scope._register_detail_context(  # noqa: SLF001 # Keep registration in the owning scope.
        context
    )
    return context


def capture_indexed_detail_results(
    results: Sequence[DetailResult | dict],
    ledger: EvidenceLedger,
    *,
    index: LedgerIndex,
    citation_scope: CitationResolutionScope | None = None,
    revision_binding: tuple[str, str] | None = None,
) -> IndexedDetailContext:
    """Strictly own fresh typed stage details before a manifest is complete.

    The capture validates every supplied detail model and nested citation,
    checks full ledger-unit coverage, and returns a data-only context. It does
    not issue a stage proof or generation/report capability.
    """
    scope, owned_scope = _detail_scope(
        ledger, index, citation_scope, revision_binding
    )
    try:
        captured = tuple(
            DetailResult.model_validate(_tuple_detail_record(item))
            for item in results
        )
        return _validate_indexed_detail_models(
            captured,
            ledger,
            index,
            scope,
        )
    except Exception:
        if owned_scope:
            scope.close()
        raise


def validate_indexed_detail_results(  # noqa: PLR0913 # Explicit current source/index bindings are required.
    result_refs: Sequence[ResultRef],
    graph: ReportStorageView,
    ledger: EvidenceLedger,
    *,
    index: LedgerIndex,
    citation_scope: CitationResolutionScope | None = None,
    revision_binding: tuple[str, str] | None = None,
) -> IndexedDetailContext:
    """Validate ordered graph detail references without expanding the report.

    ``result_refs`` must name unique detail results in ``graph`` and collectively
    cover every unit in ``ledger``. The graph's addresses are only structural
    indexes; each fetched record is strictly parsed and checked against the
    current ledger. The returned context retains addresses and compact detail
    digests, observation IDs, and record counts, not duplicate detail models or
    canonical detail snapshots.

    Preconditions:
        References select each current graph detail exactly once, and the
        supplied ledger/index/revision binding passes the scope guards.
    Postconditions:
        Every selected detail has passed strict nested and citation validation;
        the returned context is registered to the local citation scope.
    Essential behavior:
        Does not expand the complete report or retain complete detail models.
    """
    from diffstory.report_storage import ReportRecordGraph
    from diffstory.report_storage import ReportStorageView
    from diffstory.report_storage import ResultRef as StorageResultRef

    _require(
        type(graph) in {ReportRecordGraph, ReportStorageView},
        "invalid_indexed_detail_graph",
    )
    owned_graph = copy.copy(graph)
    scope, owned_scope = _detail_scope(
        ledger, index, citation_scope, revision_binding
    )
    try:
        _require(
            all(type(ref) is StorageResultRef for ref in result_refs),
            "invalid_indexed_detail_reference",
        )
        references = tuple(_copy_result_ref(ref) for ref in result_refs)
        _unique(
            [reference.storage_id for reference in references],
            "detail_set_coverage_mismatch",
        )
        _unique(
            [reference.item_key for reference in references],
            "detail_set_coverage_mismatch",
        )
        generation_refs = owned_graph.generation_result_refs["details"]
        generation_ids = {
            reference.storage_id for reference in generation_refs
        }
        results = []
        unit_keys = []
        detail_digests = {}
        observation_ids = {}
        detail_counts = {}
        for reference in references:
            _require(
                reference.table == "results" and reference.kind == "detail",
                "invalid_indexed_detail_reference",
            )
            try:
                canonical_ref = owned_graph.lookup_result(
                    "detail", reference.item_key
                )
            except ValueError as error:
                msg = "detail_set_coverage_mismatch"
                raise ValueError(msg) from error
            _require(
                reference == canonical_ref
                and reference.storage_id in generation_ids,
                "detail_set_coverage_mismatch",
            )
            try:
                raw_record = owned_graph._validated_record(  # noqa: SLF001 # Read the graph-owned structurally checked record.
                    "results", canonical_ref.storage_id
                )
            except ValueError as error:
                msg = "detail_set_coverage_mismatch"
                raise ValueError(msg) from error
            result = DetailResult.model_validate(
                _tuple_detail_record(raw_record)
            )
            _require(
                result.unit_key == canonical_ref.item_key,
                "detail_result_reference_mismatch",
            )
            detail_digests[result.unit_key] = content_digest(
                canonical_bytes(result)
            )
            observation_ids[result.unit_key] = tuple(
                item.id for item in result.observations
            )
            detail_counts[result.unit_key] = (
                len(result.explanations),
                len(result.observations),
                len(result.questions),
            )
            results.append(result)
            unit_keys.append(result.unit_key)
        _require(
            Counter(unit_keys) == Counter(index.units.keys()),
            "detail_set_coverage_mismatch",
        )
        for result in results:
            _validate_detail_result(
                result,
                index.units[result.unit_key],
                index,
                citation_scope=scope,
            )
        context = IndexedDetailContext(
            token=_DETAIL_CONTEXT_TOKEN,
            ledger=ledger,
            index=index,
            citation_scope=scope,
            unit_keys=tuple(unit_keys),
            graph=owned_graph,
            references=references,
            detail_digests=detail_digests,
            detail_raw_digests={},
            observation_ids=observation_ids,
            detail_counts=detail_counts,
        )
        scope._register_detail_context(  # noqa: SLF001 # Keep registration in the owning scope.
            context
        )
    except Exception:
        if owned_scope:
            scope.close()
        raise
    else:
        return context


def _capture_indexed_arena_detail_results(  # noqa: PLR0913 # Current ledger and source bindings remain explicit.
    result_refs: Sequence[ResultRef],
    arena: object,
    ledger: EvidenceLedger,
    *,
    index: LedgerIndex,
    citation_scope: CitationResolutionScope | None = None,
    revision_binding: tuple[str, str] | None = None,
) -> IndexedDetailContext:
    """Bind early arena details to the current ledger without reparsing models.

    Preconditions:
        ``arena`` is the exact private ``_RecordArena`` type; every reference
        names an arena-owned detail; and ``ledger``/``index`` pass the existing
        scope and current-source integrity checks.
    Postconditions:
        Full ledger, obligation, context, and citation validation has run for
        every ordered detail. Semantic detail digests, observation IDs, and
        nested record counts are captured; the context stores only copied
        addresses and a shallow immutable arena reader.
    Essential behavior:
        Reuses the arena's strict structural view for nested validation. It
        transiently materializes each selected logical result once to capture
        the exact normalized semantic digest, including ``DetailResult``'s
        default version, then discards that mapping. Later planning-source
        derivation uses the compact facts and selected-record views. Current
        source citations use the same scope-local memo as existing adapters.
    """
    from diffstory.report_storage import ResultRef as StorageResultRef
    from diffstory.report_storage import _RecordArena

    _require(type(arena) is _RecordArena, "invalid_indexed_detail_arena")
    scope, owned_scope = _detail_scope(
        ledger, index, citation_scope, revision_binding
    )
    try:
        _require(
            type(result_refs) in {tuple, list}
            and all(type(ref) is StorageResultRef for ref in result_refs),
            "invalid_indexed_detail_reference",
        )
        references = tuple(_copy_result_ref(ref) for ref in result_refs)
        _unique(
            [reference.storage_id for reference in references],
            "detail_set_coverage_mismatch",
        )
        _unique(
            [reference.item_key for reference in references],
            "detail_set_coverage_mismatch",
        )
        reader = arena._detail_reader(references)  # noqa: SLF001 # Private owned-record path.
        detail_views = []
        unit_keys = []
        detail_digests = {}
        observation_ids = {}
        detail_counts = {}
        for reference in references:
            raw = reader._result_view(reference)  # noqa: SLF001 # Read the strict normalized result.
            detail = _detail_result_check_view(raw, reader)
            _require(
                detail.unit_key == reference.item_key,
                "detail_result_reference_mismatch",
            )
            raw_detail = reader.result_record(_copy_result_ref(reference))
            _require(type(raw_detail) is dict, "invalid_typed_storage_record")
            raw_detail.setdefault(
                "version", DetailResult.model_fields["version"].default
            )
            detail_digests[detail.unit_key] = content_digest(
                canonical_bytes(raw_detail)
            )
            observation_ids[detail.unit_key] = tuple(
                item.id for item in detail.observations
            )
            detail_counts[detail.unit_key] = (
                len(detail.explanations),
                len(detail.observations),
                len(detail.questions),
            )
            detail_views.append(detail)
            unit_keys.append(detail.unit_key)
        _require(
            Counter(unit_keys) == Counter(index.units.keys()),
            "detail_set_coverage_mismatch",
        )
        for result in detail_views:
            _validate_detail_result(
                result,
                index.units[result.unit_key],
                index,
                citation_scope=scope,
            )
        context = IndexedDetailContext(
            token=_DETAIL_CONTEXT_TOKEN,
            ledger=ledger,
            index=index,
            citation_scope=scope,
            unit_keys=tuple(unit_keys),
            arena_reader=reader,
            references=references,
            detail_digests=detail_digests,
            detail_raw_digests={},
            observation_ids=observation_ids,
            detail_counts=detail_counts,
        )
        scope._register_detail_context(  # noqa: SLF001 # Keep registration in the owning scope.
            context
        )
    except Exception:
        if owned_scope:
            scope.close()
        raise
    return context


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
