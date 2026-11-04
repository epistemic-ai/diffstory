"""Select test or production file evidence without changing retained source."""

from __future__ import annotations

import copy
from pathlib import PurePosixPath

from .models import SnapshotInput


def is_test_path(path: str) -> bool:
    """Recognize common test directories, filenames, and test support files.

    Args:
        path: Repository-relative, slash-separated source path.

    Returns:
        Whether the path uses a test convention. Similar names such as
        ``testing.py`` and API definitions under ``specs`` remain included.
    """
    value = PurePosixPath(path)
    return (
        any(
            part in {"test", "tests", "__tests__"} for part in value.parts[:-1]
        )
        or value.name == "conftest.py"
        or value.name.startswith("test_")
        or value.stem.endswith(("_test", "_spec", ".test", ".spec"))
    )


def record_test_exclusions(snapshot: dict, excluded: int, total: int) -> dict:
    """Label a selected-file snapshot and explain its intentionally omitted tests.

    Args:
        snapshot: Newly constructed snapshot with only retained file pairs.
        excluded: Number of complete test file pairs omitted from this input.
        total: Number of changed file pairs before selection.

    Returns:
        The same snapshot, with selected scope, counts, and a warning when any
        tests were excluded. Retained file coverage and source remain intact.
    """
    if excluded:
        snapshot["meta"].update(
            scope="selected excerpts",
            changed_files=total - excluded,
            total_changed_files=total,
            excluded_test_files=excluded,
        )
        noun = "file" if excluded == 1 else "files"
        snapshot["warnings"].append(
            f"Excluded {excluded} changed test {noun} by default. "
            "Use --include-tests when capturing input to retain tests."
        )
    return snapshot


def without_test_files(snapshot: dict) -> dict:
    """Copy a bounded snapshot while removing complete test-only file pairs.

    Args:
        snapshot: Untrusted v1 input, validated for primitive fields and bounds
            before filtering. The compiler validates retained evidence normally.

    Returns:
        An independent snapshot with test-only comparisons, their source, and
        their original patches removed. Renames with a production path on
        either side retain both source sides. Unmapped input remains present
        so filtering cannot hide invalid production evidence.

    Raises:
        ValueError: If input fields or aggregate source/patch bounds are invalid.
    """
    result = copy.deepcopy(snapshot)
    source = SnapshotInput.model_validate(result)
    excluded = [
        record
        for record in source.file_evidence
        if is_test_path(record.base.path) and is_test_path(record.head.path)
    ]
    if not excluded:
        return result
    paths = {
        (side, getattr(record, side).path)
        for record in excluded
        for side in ("base", "head")
    }
    patch_paths = {record.head.path for record in excluded}
    result["file_evidence"] = [
        record
        for record in result["file_evidence"]
        if ("head", record["head"]["path"]) not in paths
    ]
    result["fragments"] = [
        fragment
        for fragment in result.get("fragments", [])
        if (fragment["side"], fragment["path"]) not in paths
    ]
    result["patches"] = [
        record
        for record in result["patches"]
        if record["path"] not in patch_paths
    ]
    result.setdefault("meta", {})["source_bytes"] = sum(
        len(fragment["text"].encode("utf-8"))
        for fragment in result["fragments"]
    )
    result.setdefault("warnings", [])
    return record_test_exclusions(
        result, len(excluded), len(source.file_evidence)
    )
