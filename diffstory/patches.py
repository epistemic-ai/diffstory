"""Read an original unified patch without choosing a new line alignment.

Git and GitHub already decided which lines belong together. This module keeps
those decisions and checks the rows against the captured file versions. It
does not run a diff algorithm or repair an incomplete patch.
"""

from __future__ import annotations

import re

from .source_lines import split_source_lines

HUNK_HEADER = re.compile(r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@(?:.*)$")


def _require(condition: object, message: str) -> None:
    """Reject invalid patch data without including private source in errors."""
    if not condition:
        raise ValueError(message)


def _read_hunks(patch: str) -> list[dict]:
    """Parse the supplied grouping; never choose new matches between lines."""
    hunks: list[dict] = []
    current = None
    patch_lines = patch.split("\n")
    for index, raw in enumerate(patch_lines):
        line = raw.removesuffix("\r")
        header = HUNK_HEADER.fullmatch(line)
        if header:
            current = {
                "old_start": int(header[1]),
                "old_count": int(header[2]) if header[2] is not None else 1,
                "new_start": int(header[3]),
                "new_count": int(header[4]) if header[4] is not None else 1,
                "rows": [],
            }
            hunks.append(current)
            continue
        if current is None:
            # File headers and rename/mode metadata are retained verbatim in
            # the original patch. Paths here never control source lookup.
            _require(
                not line.startswith("@@"), "Invalid original patch hunk header"
            )
            continue
        if line == "\\ No newline at end of file":
            _require(
                bool(current["rows"]),
                "Original patch newline marker has no row",
            )
            continue
        if not line and index == len(patch_lines) - 1:
            continue
        _require(bool(line) and line[0] in " +-", "Invalid original patch row")
        current["rows"].append(
            {
                "tag": {" ": "context", "+": "add", "-": "delete"}[line[0]],
                "text": raw[1:],
            }
        )
    return hunks


def _row_matches_source(text: str, source_line: str) -> bool:
    """Allow CRLF display normalization while retaining every bare CR byte."""
    content = source_line.removesuffix("\n")
    return text == content or (
        source_line.endswith("\r\n") and text == content.removesuffix("\r")
    )


def _bind_hunk(
    hunk: dict, old_lines: list[str], new_lines: list[str]
) -> list[str]:
    """Bind exact row bytes to LF-delimited source, then normalize CRLF for display."""
    old_number = hunk["old_start"]
    new_number = hunk["new_start"]
    old_count = new_count = 0
    result = []
    for row in hunk["rows"]:
        tag = row["tag"]
        row["old"] = old_number if tag != "add" else None
        row["new"] = new_number if tag != "delete" else None
        if tag != "add":
            _require(
                1 <= old_number <= len(old_lines)
                and _row_matches_source(
                    row["text"], old_lines[old_number - 1]
                ),
                "Original patch row disagrees with the earlier source",
            )
            old_number += 1
            old_count += 1
        if tag != "delete":
            _require(
                1 <= new_number <= len(new_lines)
                and _row_matches_source(
                    row["text"], new_lines[new_number - 1]
                ),
                "Original patch row disagrees with the later source",
            )
            if tag == "context":
                _require(
                    old_lines[old_number - 2] == new_lines[new_number - 1],
                    "Original patch context line endings disagree with source",
                )
            source_line = new_lines[new_number - 1]
            result.append(source_line)
            new_number += 1
            new_count += 1
        else:
            source_line = old_lines[old_number - 2]
        row["text"] = split_source_lines(source_line)[0]
    _require(
        (old_count, new_count) == (hunk["old_count"], hunk["new_count"]),
        "Original patch hunk is incomplete",
    )
    return result


def parse_file_patch(patch: str, before: str, after: str) -> list[dict]:
    """Keep original hunks and reject rows that disagree with either file.

    The supplied source must contain complete files. Checking that the patch
    produces ``after`` also catches a patch that stops before its last change.
    No-newline markers stay in the retained patch text; they are not code rows.
    """
    old_lines = split_source_lines(before, keepends=True)
    new_lines = split_source_lines(after, keepends=True)
    hunks = _read_hunks(patch)

    result: list[str] = []
    old_cursor = 0
    for hunk in hunks:
        old_begin = hunk["old_start"] - (hunk["old_count"] > 0)
        new_begin = hunk["new_start"] - (hunk["new_count"] > 0)
        _require(
            old_cursor <= old_begin <= len(old_lines),
            "Original patch hunks overlap or exceed the earlier file",
        )
        result.extend(old_lines[old_cursor:old_begin])
        _require(
            len(result) == new_begin,
            "Original patch later line numbers disagree with source",
        )
        result.extend(_bind_hunk(hunk, old_lines, new_lines))
        old_cursor = old_begin + hunk["old_count"]
    result.extend(old_lines[old_cursor:])
    _require(
        result == new_lines and (bool(hunks) or before == after),
        "Original patch does not include all source changes",
    )
    return hunks
