"""Build bounded writer evidence from a validated full packet."""

from __future__ import annotations

import copy
import hashlib

from .compact_packet import SourceIndex
from .compact_packet import source_index

WRITING_EVIDENCE_VERSION = "diffstory.writing-evidence.v2"
_PROJECTED_FIELDS = frozenset({"sources", "changes", "regions", "diffs"})


class WritingEvidence:
    """Index a validated packet and project exact evidence for selected regions.

    Each projection contains only assigned regions, their complete original
    diff hunks, and contiguous source slices that cover the selected evidence.

    Attributes:
        _packet: Private snapshot of the caller's validated complete packet.
        _sources: Source records and exact physical lines indexed by source ID.
        _changes: Complete change records indexed by their original IDs.
        _change_order: Change IDs in the original packet order.
        _regions: Complete changed regions indexed by their original IDs.
        _diffs: Complete original diffs indexed by their original IDs.
    """

    def __init__(self, packet: dict) -> None:
        """Index a private copy of an already validated complete packet.

        Args:
            packet: Complete packet returned by ``build_packet`` or accepted
                by ``validate_packet``, with all source and diff evidence.

        Preconditions:
            The caller has validated the complete packet.
        Postconditions:
            Later projection operations cannot mutate the caller's packet.
        """
        self._packet = copy.deepcopy(packet)
        self._sources: SourceIndex = source_index(self._packet)
        self._changes = {
            change["id"]: change for change in self._packet["changes"]
        }
        self._change_order = [
            change["id"] for change in self._packet["changes"]
        ]
        self._regions = {
            region["id"]: region for region in self._packet["regions"]
        }
        self._diffs = {diff["id"]: diff for diff in self._packet["diffs"]}

    def project(
        self,
        regions: list[dict],
        *,
        include_definitions: bool = False,
    ) -> dict:
        """Return writer-only evidence for the selected full-packet regions.

        Args:
            regions: Exact region records from the validated full packet.
            include_definitions: Whether to include complete source ranges for
                each selected change's before and after definitions.

        Returns:
            A versioned packet projection with selected regions, changes,
            complete selected hunks, and exact contiguous source excerpts.

        Raises:
            TypeError: If the region list or definition flag has the wrong type.
            ValueError: If a region is unknown, duplicated, or differs from its
                validated full-packet record.
        """
        selected_regions = self._select_regions(regions)
        if type(include_definitions) is not bool:
            msg = "Definition inclusion must be a boolean"
            raise TypeError(msg)

        change_ids = {
            change_id
            for region in selected_regions
            for change_id in region["change_ids"]
        }
        selected_changes = [
            self._changes[change_id]
            for change_id in self._change_order
            if change_id in change_ids
        ]

        ranges, selected_hunks = self._collect_ranges(selected_regions)
        if include_definitions:
            self._include_definitions(selected_changes, ranges)

        sources = self._project_sources(ranges)
        changes = [
            self._project_change(change, ranges) for change in selected_changes
        ]
        diffs = self._project_diffs(selected_hunks)
        projection = {
            key: copy.deepcopy(value)
            for key, value in self._packet.items()
            if key not in _PROJECTED_FIELDS
        }
        projection["version"] = WRITING_EVIDENCE_VERSION
        projection["meta"].pop("captured_at", None)
        projection.update(
            {
                "sources": sources,
                "changes": changes,
                "regions": copy.deepcopy(selected_regions),
                "diffs": diffs,
            }
        )
        return projection

    def _collect_ranges(
        self, regions: list[dict]
    ) -> tuple[dict[str, tuple[int, int]], dict[str, set[int]]]:
        """Collect selected hunk coordinates and source-only citations."""
        ranges: dict[str, tuple[int, int]] = {}
        selected_hunks: dict[str, set[int]] = {}
        for region in regions:
            self._collect_region_ranges(region, ranges, selected_hunks)
        return ranges, selected_hunks

    def _collect_region_ranges(
        self,
        region: dict,
        ranges: dict[str, tuple[int, int]],
        selected_hunks: dict[str, set[int]],
    ) -> None:
        """Add all coordinates belonging to one assigned region."""
        if region["diff_id"] is None:
            for citation in region["citations"]:
                self._include_range(
                    ranges,
                    citation["source_id"],
                    citation["start"],
                    citation["end"],
                )
            return

        diff_id = region["diff_id"]
        hunk_index = region["hunk"]
        diff = self._diffs[diff_id]
        selected_hunks.setdefault(diff_id, set()).add(hunk_index)
        hunk = diff["hunks"][hunk_index]
        for row in hunk["rows"]:
            for source_id, coordinate in (
                (diff["base_source"], row["old"]),
                (diff["head_source"], row["new"]),
            ):
                if source_id is not None and coordinate is not None:
                    self._include_range(
                        ranges, source_id, coordinate, coordinate
                    )

    def _include_definitions(
        self, changes: list[dict], ranges: dict[str, tuple[int, int]]
    ) -> None:
        """Expand selected source slices to complete associated definitions."""
        for change in changes:
            for side in ("before", "after"):
                reference = change[side]
                if reference is not None:
                    self._include_range(
                        ranges,
                        reference["source_id"],
                        reference["start"],
                        reference["end"],
                    )

    def _select_regions(self, regions: list[dict]) -> list[dict]:
        """Resolve selected records to their canonical packet copies."""
        if type(regions) is not list:
            msg = "Writing regions must be a list"
            raise TypeError(msg)
        selected_ids = set()
        for region in regions:
            if type(region) is not dict:
                msg = "Writing regions must contain objects"
                raise TypeError(msg)
            region_id = region.get("id")
            original = self._regions.get(region_id)
            if original is None or region != original:
                msg = "Writing region is not an exact full-packet record"
                raise ValueError(msg)
            if region_id in selected_ids:
                msg = "Writing region is assigned more than once"
                raise ValueError(msg)
            selected_ids.add(region_id)
        return [
            region
            for region in self._packet["regions"]
            if region["id"] in selected_ids
        ]

    @staticmethod
    def _include_range(
        ranges: dict[str, tuple[int, int]],
        source_id: str,
        start: int,
        end: int,
    ) -> None:
        """Merge one exact source interval into its contiguous projection."""
        current = ranges.get(source_id)
        ranges[source_id] = (
            min(start, current[0]) if current else start,
            max(end, current[1]) if current else end,
        )

    def _project_sources(
        self, ranges: dict[str, tuple[int, int]]
    ) -> list[dict]:
        """Copy exact physical source lines into one slice per source ID."""
        result = []
        for source in self._packet["sources"]:
            selected = ranges.get(source["id"])
            if selected is None:
                continue
            start, end = selected
            source_start = source["start_line"]
            physical = self._sources.physical[source["id"]]
            lines = physical[start - source_start : end - source_start + 1]
            text = "".join(lines)
            excerpt = copy.deepcopy(source)
            excerpt.update(
                {
                    "start_line": start,
                    "scope": "selected excerpts",
                    "text": text,
                    "digest": hashlib.sha256(text.encode("utf-8")).hexdigest(),
                }
            )
            result.append(excerpt)
        return result

    @staticmethod
    def _definition_coverage(
        reference: dict, ranges: dict[str, tuple[int, int]]
    ) -> str:
        """Classify how much of one definition falls inside its source slice."""
        selected = ranges.get(reference["source_id"])
        if selected is None or (
            selected[1] < reference["start"] or selected[0] > reference["end"]
        ):
            return "not_supplied"
        if (
            selected[0] <= reference["start"]
            and selected[1] >= reference["end"]
        ):
            return "full"
        return "partial"

    def _project_change(
        self, change: dict, ranges: dict[str, tuple[int, int]]
    ) -> dict:
        """Remove duplicate comparisons and describe supplied definitions."""
        result = copy.deepcopy(change)
        result.pop("comparison_hunks", None)
        result["definition_coverage"] = {
            side: self._definition_coverage(change[side], ranges)
            for side in ("before", "after")
            if change[side] is not None
        }
        return result

    def _project_diffs(
        self, selected_hunks: dict[str, set[int]]
    ) -> list[dict]:
        """Keep only selected diffs and their complete original hunks."""
        result = []
        for diff in self._packet["diffs"]:
            indexes = selected_hunks.get(diff["id"])
            if not indexes:
                continue
            projected = {
                key: copy.deepcopy(value)
                for key, value in diff.items()
                if key not in {"hunks", "original_patch"}
            }
            projected["hunks"] = []
            for index, hunk in enumerate(diff["hunks"]):
                if index in indexes:
                    projected["hunks"].append(
                        {**copy.deepcopy(hunk), "original_index": index}
                    )
            result.append(projected)
        return result
