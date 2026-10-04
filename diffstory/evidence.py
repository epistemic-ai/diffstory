"""File coverage and definition claims derived from validated source evidence."""

from __future__ import annotations

from collections import defaultdict
from typing import TYPE_CHECKING

from .models import AbsentFile
from .models import FileEvidence
from .models import ParseStatus
from .models import RevisionSide
from .models import SourceSide
from .models import StrictModel
from .models import SuppliedFile
from .models import UnavailableFile

if TYPE_CHECKING:
    from collections.abc import Mapping
    from collections.abc import Sequence

UNAVAILABLE_DESCRIPTIONS = {
    "not_supplied": "The {side} source was not supplied; file absence is not confirmed.",
    "size_limit": "The {side} source is unavailable because it exceeds the file size limit.",
    "binary_or_non_utf8": "The {side} source is unavailable because it is binary or is not UTF-8.",
    "unsupported_object": "The {side} source is unavailable because it is not a supported regular file.",
    "invalid_content_size": "The {side} source is unavailable because its declared size is invalid.",
    "unsupported_encoding": "The {side} source is unavailable because its encoding is unsupported.",
    "read_failed": "The {side} source could not be read; file absence is not confirmed.",
}


class FileCoverage(StrictModel):
    """Join a file's source state to the fragments actually inspected.

    Attributes:
        source: Validated availability and coverage for this revision.
        counterpart_path: Linked path at the other revision.
        fragment_ids: IDs of the source fragments assigned to this file.
        parse_statuses: Parse outcomes in the same order as the fragment IDs.
    """

    source: SourceSide
    counterpart_path: str
    fragment_ids: tuple[str, ...]
    parse_statuses: tuple[ParseStatus, ...]

    @property
    def full_and_parsed(self) -> bool:
        """Return whether every byte of this file has valid Python AST evidence."""
        return (
            isinstance(self.source, SuppliedFile)
            and self.source.coverage == "full"
            and self.parse_statuses == ("ok",)
        )

    @property
    def absent(self) -> bool:
        """Return whether the producer confirmed that this file does not exist."""
        return isinstance(self.source, AbsentFile)

    def blockers(self, side: RevisionSide) -> list[str]:
        """Explain why this file cannot establish a definition's absence.

        Args:
            side: Revision label used in the returned prose.

        Returns:
            Deterministic blocker sentences, or none for confirmed absence.
        """
        if isinstance(self.source, AbsentFile):
            return []
        if isinstance(self.source, UnavailableFile):
            return [
                UNAVAILABLE_DESCRIPTIONS[self.source.reason].format(side=side)
            ]
        conditions = (
            (
                self.source.coverage != "full",
                f"Only part of the {side} file is supplied.",
            ),
            (
                "failed" in self.parse_statuses,
                f"The {side} Python source could not be parsed.",
            ),
            (
                "text_only" in self.parse_statuses,
                f"The {side} source is text-only; Python parsing is not available.",
            ),
        )
        return [message for blocked, message in conditions if blocked]


class FilePair(StrictModel):
    """Hold inspected base and head coverage for one comparison.

    Attributes:
        base: Base source state, assigned fragments, and head path.
        head: Head source state, assigned fragments, and base path.
    """

    base: FileCoverage
    head: FileCoverage


def _validate_full_source(
    fragment_ids: Sequence[str], fragments: Mapping[str, dict]
) -> None:
    """Require one complete fragment before accepting a full-file claim.

    Args:
        fragment_ids: Nonempty IDs assigned to a supplied file side.
        fragments: Source records indexed by ID.

    Raises:
        ValueError: If full source is split, offset, or marked as an excerpt.
    """
    if len(fragment_ids) != 1:
        msg = "Full file evidence requires exactly one fragment"
        raise ValueError(msg)
    fragment = fragments[fragment_ids[0]]
    if fragment.get("start_line", 1) != 1:
        msg = "Full file evidence must start at line 1"
        raise ValueError(msg)
    if fragment.get("scope") not in (None, "full", "complete"):
        msg = "Full file evidence conflicts with excerpt scope"
        raise ValueError(msg)


def _coverage_side(
    source: SourceSide,
    counterpart_path: str,
    fragment_ids: Sequence[str],
    fragments: Mapping[str, dict],
    parse_statuses: Mapping[str, ParseStatus],
) -> FileCoverage:
    """Validate a side's source claim against its assigned fragments.

    Args:
        source: Validated file state from explicit or legacy evidence.
        counterpart_path: Linked path at the other revision.
        fragment_ids: IDs assigned to this side and path.
        fragments: Indexed source records.
        parse_statuses: Parse result for each indexed fragment.

    Returns:
        Typed coverage whose source availability agrees with its fragments.

    Raises:
        ValueError: If supplied source is missing or unread source has fragments.
    """
    if isinstance(source, SuppliedFile):
        if not fragment_ids:
            msg = "A supplied file evidence side must have a fragment"
            raise ValueError(msg)
        if source.coverage == "full":
            _validate_full_source(fragment_ids, fragments)
    elif fragment_ids:
        msg = f"{source.state.capitalize()} file evidence cannot contain fragments"
        raise ValueError(msg)
    return FileCoverage(
        source=source,
        counterpart_path=counterpart_path,
        fragment_ids=tuple(fragment_ids),
        parse_statuses=tuple(parse_statuses[item] for item in fragment_ids),
    )


def _legacy_path_pairs(
    fragments: Mapping[str, dict],
) -> list[tuple[str | None, str | None]]:
    """Pair old snapshots by equal paths or an unambiguous explicit region.

    Args:
        fragments: Validated source records from a snapshot without file evidence.

    Returns:
        Sorted base/head pairs, followed by remaining single-sided paths.
    """
    paths = {side: set() for side in ("base", "head")}
    regions = {side: defaultdict(set) for side in ("base", "head")}
    for fragment in fragments.values():
        side, path = fragment["side"], fragment["path"]
        paths[side].add(path)
        if "region" in fragment:
            regions[side][fragment["region"]].add(path)
    pairs = [(path, path) for path in sorted(paths["base"] & paths["head"])]
    used_base = {base for base, _ in pairs}
    used_head = {head for _, head in pairs}
    candidates, reverse = defaultdict(set), defaultdict(set)
    for region in regions["base"].keys() & regions["head"].keys():
        bases, heads = regions["base"][region], regions["head"][region]
        if len(bases) == len(heads) == 1:
            base, head = next(iter(bases)), next(iter(heads))
            if base != head:
                candidates[base].add(head)
                reverse[head].add(base)
    for base, heads in sorted(candidates.items()):
        if len(heads) != 1 or base in used_base:
            continue
        head = next(iter(heads))
        if head in used_head or len(reverse[head]) != 1:
            continue
        pairs.append((base, head))
        used_base.add(base)
        used_head.add(head)
    pairs.extend((path, None) for path in sorted(paths["base"] - used_base))
    pairs.extend((None, path) for path in sorted(paths["head"] - used_head))
    return pairs


def _legacy_side(
    path: str,
    ids: Sequence[str],
    fragments: Mapping[str, dict],
    *,
    excerpts: bool,
) -> SourceSide:
    """Infer coverage conservatively when a producer omitted file evidence.

    Args:
        path: Original path, or the paired path when source is missing.
        ids: Fragment IDs assigned to this side.
        fragments: Source records indexed by ID.
        excerpts: Whether snapshot metadata declares selected excerpts.

    Returns:
        Unavailable source when omitted; full coverage only for one explicit
        line-one full fragment outside a selected-excerpt snapshot.
    """
    if not ids:
        return UnavailableFile(path=path, reason="not_supplied")
    fragment = fragments[ids[0]]
    full = (
        not excerpts
        and len(ids) == 1
        and fragment.get("start_line", 1) == 1
        and fragment.get("scope") == "full"
    )
    return SuppliedFile(path=path, coverage="full" if full else "partial")


def _legacy_evidence(
    fragments: Mapping[str, dict],
    by_file: Mapping[tuple[str, str], list[str]],
    *,
    excerpts: bool,
) -> list[FileEvidence]:
    """Build typed comparison records for old snapshots.

    Args:
        fragments: Validated source records indexed by ID.
        by_file: Fragment IDs grouped by revision and path.
        excerpts: Whether the producer declared selected excerpts.

    Returns:
        Conservative source pairs without inferring file absence.
    """
    records = []
    for base_path, head_path in _legacy_path_pairs(fragments):
        base = base_path or head_path
        head = head_path or base_path
        records.append(
            FileEvidence(
                base=_legacy_side(
                    base,
                    by_file.get(("base", base), []),
                    fragments,
                    excerpts=excerpts,
                ),
                head=_legacy_side(
                    head,
                    by_file.get(("head", head), []),
                    fragments,
                    excerpts=excerpts,
                ),
            )
        )
    return records


def _assign_regions(
    pairs: Sequence[FilePair], fragments: Mapping[str, dict], *, explicit: bool
) -> None:
    """Assign raw diff regions without merging unrelated source records.

    Args:
        pairs: Validated base/head coverage pairs.
        fragments: Indexed source records to receive ``_raw_region`` values.
        explicit: Whether pair ownership comes from explicit file evidence.

    Raises:
        ValueError: If regions repeat on a side, cross explicit records, or
            fail to pair two full file versions.
    """
    occupied, owners = set(), {}
    for index, pair in enumerate(pairs):
        regions = {"base": [], "head": []}
        for side_name in ("base", "head"):
            side = getattr(pair, side_name)
            default = pair.head.source.path if explicit else side.source.path
            for fragment_id in side.fragment_ids:
                fragment = fragments[fragment_id]
                region = fragment.get("region", default)
                key = (region, side_name)
                if key in occupied:
                    msg = "Duplicate raw region for one revision side"
                    raise ValueError(msg)
                occupied.add(key)
                if explicit and owners.setdefault(region, index) != index:
                    msg = "A raw region cannot join different file evidence records"
                    raise ValueError(msg)
                fragment["_raw_region"] = region
                regions[side_name].append(region)
        full_pair = all(
            isinstance(side.source, SuppliedFile)
            and side.source.coverage == "full"
            for side in (pair.base, pair.head)
        )
        if explicit and full_pair and regions["base"] != regions["head"]:
            msg = "Full file sides in one evidence record must share a region"
            raise ValueError(msg)


def build_file_coverage(
    evidence: Sequence[FileEvidence] | None,
    meta: Mapping[str, object],
    fragments: Mapping[str, dict],
    parse_statuses: Mapping[str, ParseStatus],
) -> dict[tuple[str, str], FileCoverage]:
    """Index file coverage from explicit records or conservative legacy inference.

    Args:
        evidence: Validated explicit records, or ``None`` for an older snapshot.
        meta: Snapshot metadata used only to identify selected excerpts.
        fragments: Validated source records indexed by ID.
        parse_statuses: Parse outcome for every source record.

    Returns:
        Typed coverage indexed by revision side and path.

    Raises:
        ValueError: If records omit source or contradict coverage and regions.
    """
    by_file = defaultdict(list)
    for fragment_id, fragment in fragments.items():
        by_file[(fragment["side"], fragment["path"])].append(fragment_id)
    explicit = evidence is not None
    records = (
        evidence
        if explicit
        else _legacy_evidence(
            fragments,
            by_file,
            excerpts=meta.get("scope") == "selected excerpts",
        )
    )
    pairs = []
    coverage = {}
    for record in records:
        sides = {}
        for side_name, opposite in (("base", "head"), ("head", "base")):
            source = getattr(record, side_name)
            key = (side_name, source.path)
            side = _coverage_side(
                source,
                getattr(record, opposite).path,
                by_file.get(key, []),
                fragments,
                parse_statuses,
            )
            coverage[key] = side
            sides[side_name] = side
        pairs.append(FilePair(**sides))
    if set(by_file) - coverage.keys():
        msg = "File evidence must cover every supplied fragment"
        raise ValueError(msg)
    _assign_regions(pairs, fragments, explicit=explicit)
    return coverage


class DefinitionEvidence(StrictModel):
    """Evaluate one unmatched definition against its own file and counterpart.

    Attributes:
        side: Revision containing the unmatched definition.
        own: Coverage for the definition's file, or none when unlinked.
        counterpart: Coverage for the paired file at the other revision.
        declaration_remains: Whether a same-name, same-type declaration remains.
    """

    side: RevisionSide
    own: FileCoverage | None
    counterpart: FileCoverage | None
    declaration_remains: bool

    @property
    def confirmed(self) -> bool:
        """Return whether file evidence confirms this addition or removal."""
        if self.own is None or self.counterpart is None:
            return False
        return (
            self.own.full_and_parsed
            and (self.counterpart.absent or self.counterpart.full_and_parsed)
            and not self.declaration_remains
        )

    @property
    def kind(self) -> str:
        """Return the confirmed or unresolved classification for this revision."""
        if self.confirmed:
            return "removed" if self.side == "base" else "added"
        return f"observed_{self.side}"

    def _confirmed_basis(self, name: str, path: str) -> str:
        """Describe the file claim supported by complete counterpart evidence.

        Args:
            name: Definition name from parsed source.
            path: Path containing that definition.

        Returns:
            Confirmed basis text with the repository-wide limitation.
        """
        action = "removed from" if self.side == "base" else "added to"
        basis = f"Definition {name} was {action} the supplied file {path}. "
        if self.counterpart.absent:
            revision = "head" if self.side == "base" else "base"
            parsed = f"The {self.side} file is fully supplied and parsed. "
            absent = (
                f"The file is confirmed absent at the {revision} revision."
            )
            basis += (
                parsed + absent
                if self.side == "base"
                else absent + " " + parsed.rstrip()
            )
        else:
            basis += "Both file versions are fully supplied and parsed. No counterpart was matched."
        return (
            basis
            + " This result does not establish whether the definition or its behavior exists elsewhere in the repository."
        )

    def _unresolved_basis(self, name: str, path: str) -> str:
        """Explain the source limits that leave this definition unresolved.

        Args:
            name: Definition name from parsed source.
            path: Path containing that definition.

        Returns:
            Reasons in base-then-head order, followed by the declaration guard.
        """
        base, head = (
            (self.own, self.counterpart)
            if self.side == "base"
            else (self.counterpart, self.own)
        )
        blockers = []
        for side_name, coverage in (("base", base), ("head", head)):
            blockers.extend(
                coverage.blockers(side_name)
                if coverage
                else [
                    UNAVAILABLE_DESCRIPTIONS["not_supplied"].format(
                        side=side_name
                    )
                ]
            )
        if self.declaration_remains:
            blockers.append(
                "A definition with the same name and type remains in the other file version, but the matcher did not form one pair."
            )
        reason = " ".join(blocker.rstrip(".") for blocker in blockers)
        action = "Removal" if self.side == "base" else "Addition"
        return f"Definition {name} is present in the supplied {self.side} source for {path}. {action} from this file is unresolved: {reason}."

    def basis(self, name: str, path: str) -> str:
        """Return the classification's basis with both paths when they differ.

        Args:
            name: Definition name from parsed source.
            path: Definition path, used when coverage is unlinked.

        Returns:
            Plain-language file evidence without claiming repository behavior.
        """
        own_path = self.own.source.path if self.own else path
        other_path = (
            self.counterpart.source.path if self.counterpart else own_path
        )
        basis = (
            self._confirmed_basis(name, own_path)
            if self.confirmed
            else self._unresolved_basis(name, own_path)
        )
        if own_path != other_path:
            base, head = (
                (own_path, other_path)
                if self.side == "base"
                else (other_path, own_path)
            )
            basis += f" Base path: {base}. Head path: {head}."
        return basis
