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
    from collections.abc import Iterator
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
    def full_source(self) -> bool:
        """Return whether the supplied fragment covers the whole file."""
        return (
            isinstance(self.source, SuppliedFile)
            and self.source.coverage == "full"
        )

    @property
    def full_and_parsed(self) -> bool:
        """Return whether the whole file has valid Python AST evidence."""
        return self.full_source and self.parse_statuses == ("ok",)

    @property
    def absent(self) -> bool:
        """Return whether the producer confirmed that this file does not exist."""
        return isinstance(self.source, AbsentFile)

    def blockers(self, side: RevisionSide) -> Iterator[str]:
        """Explain why this file cannot establish a definition's absence.

        Args:
            side: Revision label used in the returned prose.

        Yields:
            Blocker sentences in a fixed order; none for confirmed absence.
        """
        if isinstance(self.source, AbsentFile):
            return
        if isinstance(self.source, UnavailableFile):
            yield UNAVAILABLE_DESCRIPTIONS[self.source.reason].format(
                side=side
            )
            return
        if self.source.coverage != "full":
            yield f"Only part of the {side} file is supplied."
        if "failed" in self.parse_statuses:
            yield f"The {side} Python source could not be parsed."
        if "text_only" in self.parse_statuses:
            yield f"The {side} source is text-only; Python parsing is not available."


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
        source: Validated file state from the snapshot evidence.
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


def _assign_regions(
    pairs: Sequence[FilePair], fragments: Mapping[str, dict]
) -> None:
    """Assign raw diff regions without merging unrelated source records.

    For each file pair:
        Use the head path as the default region.
        For each fragment, reject a repeated (region, side) or a region owned
        by another file pair. Store the accepted region on that fragment.
        Require matching region labels when both sides supply a full file.

    Args:
        pairs: Validated base/head coverage pairs.
        fragments: Indexed source records to receive ``_raw_region`` values.

    Raises:
        ValueError: If regions repeat on a side, cross file records, or
            fail to pair two full file versions.
    """
    occupied = set()
    owners = {}
    for index, pair in enumerate(pairs):
        for side_name in ("base", "head"):
            side = getattr(pair, side_name)
            default = pair.head.source.path
            for fragment_id in side.fragment_ids:
                fragment = fragments[fragment_id]
                region = fragment.get("region", default)
                key = (region, side_name)
                if key in occupied:
                    msg = "Duplicate raw region for one revision side"
                    raise ValueError(msg)
                occupied.add(key)
                if region in owners and owners[region] != index:
                    msg = "A raw region cannot join different file evidence records"
                    raise ValueError(msg)
                owners[region] = index
                fragment["_raw_region"] = region
        if pair.base.full_source and pair.head.full_source:
            base_region = fragments[pair.base.fragment_ids[0]]["_raw_region"]
            head_region = fragments[pair.head.fragment_ids[0]]["_raw_region"]
            if base_region != head_region:
                msg = "Full file sides in one evidence record must share a region"
                raise ValueError(msg)


def build_file_coverage(
    fragments: Mapping[str, dict],
    parse_statuses: Mapping[str, ParseStatus],
    *,
    evidence: Sequence[FileEvidence],
) -> dict[tuple[str, str], FileCoverage]:
    """Index file coverage from the required comparison records.

    Args:
        fragments: Validated source records indexed by ID.
        parse_statuses: Parse outcome for every source record.
        evidence: Validated records linking every supplied file to its counterpart.

    Returns:
        Typed coverage indexed by revision side and path.

    Raises:
        ValueError: If records omit source or contradict coverage and regions.
    """
    by_file = defaultdict(list)
    for fragment_id, fragment in fragments.items():
        by_file[(fragment["side"], fragment["path"])].append(fragment_id)
    pairs = []
    coverage = {}
    for record in evidence:
        sides = {}
        for side_name, opposite in (("base", "head"), ("head", "base")):
            source = getattr(record, side_name)
            key = (side_name, source.path)
            side = _coverage_side(
                source,
                getattr(record, opposite).path,
                by_file.get(key, ()),
                fragments,
                parse_statuses,
            )
            coverage[key] = side
            sides[side_name] = side
        pairs.append(FilePair(**sides))
    if set(by_file) - coverage.keys():
        msg = "File evidence must cover every supplied fragment"
        raise ValueError(msg)
    _assign_regions(pairs, fragments)
    return coverage


class DefinitionEvidence(StrictModel):
    """Evaluate one unmatched definition against its own file and counterpart.

    Attributes:
        side: Revision containing the unmatched definition.
        own: Coverage for the definition's file.
        counterpart: Coverage for the paired file at the other revision.
        declaration_remains: Whether a same-name, same-type declaration remains.
    """

    side: RevisionSide
    own: FileCoverage
    counterpart: FileCoverage
    declaration_remains: bool

    @property
    def confirmed(self) -> bool:
        """Return whether file evidence confirms this addition or removal."""
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
            blockers.extend(coverage.blockers(side_name))
        if self.declaration_remains:
            blockers.append(
                "A definition with the same name and type remains in the other file version, but the matcher did not form one pair."
            )
        reason = " ".join(blocker.rstrip(".") for blocker in blockers)
        action = "Removal" if self.side == "base" else "Addition"
        return f"Definition {name} is present in the supplied {self.side} source for {path}. {action} from this file is unresolved: {reason}."

    def basis(self, name: str) -> str:
        """Return the classification's basis with both paths when they differ.

        Args:
            name: Definition name from parsed source.

        Returns:
            Plain-language file evidence without claiming repository behavior.
        """
        own_path = self.own.source.path
        other_path = self.counterpart.source.path
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
