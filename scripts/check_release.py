#!/usr/bin/env python3
"""Check public release inputs and optionally seal/verify an exact-file manifest.

This is a deterministic guardrail, not a comprehensive secret detector or an
independent licensing audit. Review examples and screenshots before publication.
No file contents or candidate secret values are printed on errors.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = "PUBLICATION.json"
SCHEMA = "diffstory.publication.v1"
MAX_RELEASE_FILE_BYTES = 12_000_000
TOP_FILES = {
    "README.md",
    "LICENSE",
    "NOTICE",
    "CONTRIBUTING.md",
    "SECURITY.md",
    "CODE_OF_CONDUCT.md",
    "CHANGELOG.md",
    "pyproject.toml",
    ".python-version",
    "uv.lock",
    "MANIFEST.in",
    ".gitignore",
    ".gitattributes",
    ".editorconfig",
}
TREES = {"diffstory", "tests", "examples", "docs", "scripts", ".github"}
SKIP_DIRS = {
    ".git",
    ".venv",
    "venv",
    "__pycache__",
    ".pytest_cache",
    ".ruff_cache",
    "build",
    "dist",
}
SKIP_SUFFIXES = {".pyc", ".pyo"}
TEXT_SUFFIXES = {
    ".py",
    ".js",
    ".css",
    ".html",
    ".json",
    ".md",
    ".yml",
    ".yaml",
    ".svg",
    ".sh",
    ".toml",
    ".in",
}
PUBLIC_DATA = {
    "examples/demo.html",
    "examples/demo.report.json",
    "examples/demo.snapshot.json",
    "examples/demo.annotations.json",
}
# Deliberately assembled: the source for this guard must not match its own rules.
PRIVATE_MARKERS = tuple(
    value.lower()
    for value in (
        "epistemic-ai/" + "foundation",
        "pr" + "20781",
        "eai_" + "query",
        "CF" + "TR2",
        "Clin" + "Var",
        "ba7ce6c20e0e699b7034d" + "1105604758e74552da2",
        "8a56c02f3c49bf4e1dfb2a9" + "cec3d5207ca594181",
    )
)
SECRET_PATTERNS = (
    re.compile(r"\bgh[pousr]_[A-Za-z0-9]{30,}\b"),
    re.compile(r"\bgithub_pat_[A-Za-z0-9_]{30,}\b"),
    re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
    re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
    re.compile(r'https://[^\s"<>]+[?&]token=[A-Za-z0-9_-]{16,}'),
)


def _publication_file_is_approved(
    path: Path,
    directory: Path,
    root: Path,
) -> bool:
    """Apply file-specific publication policy and return whether to include it.

    Args:
        path: Candidate file to inspect.
        directory: Directory containing the candidate.
        root: Publication root used to classify top-level and relative paths.

    Returns:
        Whether the candidate belongs in the publication manifest.

    Raises:
        ValueError: If the candidate violates publication policy.
    """
    relative = path.relative_to(root).as_posix()
    if path.name == MANIFEST and directory == root:
        return False
    if path.suffix in SKIP_SUFFIXES or path.name == ".DS_Store":
        return False
    if path.name.startswith(".env") or path.suffix.lower() in {
        ".pem",
        ".key",
        ".p12",
        ".pfx",
    }:
        msg = f"Credential-like file refused: {relative}"
        raise ValueError(msg)
    if directory == root and path.name not in TOP_FILES:
        msg = f"Unrecognized release file: {relative}"
        raise ValueError(msg)
    is_public_reader_image = path.suffix == ".png" and relative.startswith(
        "docs/reader/"
    )
    if (
        directory != root
        and path.suffix not in TEXT_SUFFIXES
        and not is_public_reader_image
    ):
        msg = f"Unsupported public file type: {relative}"
        raise ValueError(msg)
    report_suffixes = (
        ".report.json",
        ".snapshot.json",
        ".annotations.json",
        ".review.json",
    )
    if any(relative.endswith(suffix) for suffix in report_suffixes) and (
        relative not in PUBLIC_DATA
    ):
        msg = f"Unreviewed report data refused: {relative}"
        raise ValueError(msg)
    public_html = PUBLIC_DATA | {"diffstory/assets/app.html"}
    if path.suffix == ".html" and relative not in public_html:
        msg = f"Unreviewed rendered HTML refused: {relative}"
        raise ValueError(msg)
    return True


def _walk_publication_tree(
    directory: Path,
    root: Path,
    files: list[Path],
) -> None:
    """Collect approved files below one directory.

    Args:
        directory: Directory currently being visited.
        root: Publication root for relative-path and top-level checks.
        files: Mutable accumulator for approved files.

    Raises:
        OSError: If a directory cannot be read.
        ValueError: If a symlink or unrecognized directory is encountered.
    """
    for path in sorted(directory.iterdir()):
        relative = path.relative_to(root).as_posix()
        if path.is_symlink():
            msg = f"Symlinks are not public release inputs: {relative}"
            raise ValueError(msg)
        if path.is_dir():
            if path.name in SKIP_DIRS or path.name.endswith(".egg-info"):
                continue
            if directory == root and path.name not in TREES:
                msg = f"Unrecognized release directory: {relative}"
                raise ValueError(msg)
            _walk_publication_tree(path, root, files)
            continue
        if _publication_file_is_approved(path, directory, root):
            files.append(path)


def publication_files(root: Path) -> list[Path]:
    """Return approved release files, rejecting unrecognized inputs and symlinks.

    Args:
        root: Candidate publication root.

    Returns:
        Sorted paths included in the exact publication manifest.

    Raises:
        OSError: If a directory cannot be traversed.
        ValueError: If unknown files, directories, symlinks, credential-like
            files, or unapproved report data are encountered.
    """
    files: list[Path] = []
    _walk_publication_tree(root, root, files)
    return sorted(files, key=lambda p: p.relative_to(root).as_posix())


def inspect_files(root: Path) -> list[Path]:
    """Validate the release candidate set, contents, and synthetic demo data.

    Args:
        root: Candidate publication root.

    Returns:
        The validated, sorted candidate paths.

    Raises:
        OSError: If a candidate file cannot be read.
        ValueError: If a file is too large, contains a private marker or
            possible secret, has an invalid image signature, or demo metadata
            is missing or not explicitly synthetic.
        UnicodeError: If a text candidate is not UTF-8.
    """
    files = publication_files(root)
    for path in files:
        if path.stat().st_size > MAX_RELEASE_FILE_BYTES:
            msg = f"Unexpectedly large release file: {path.relative_to(root)}"
            raise ValueError(msg)
        if path.suffix == ".png":
            if not path.read_bytes().startswith(b"\x89PNG\r\n\x1a\n"):
                msg = f"Invalid PNG signature: {path.relative_to(root)}"
                raise ValueError(msg)
            continue
        text = path.read_text(encoding="utf-8")
        if any(marker in text.lower() for marker in PRIVATE_MARKERS):
            msg = f"Private prototype marker found in {path.relative_to(root)}"
            raise ValueError(msg)
        if any(pattern.search(text) for pattern in SECRET_PATTERNS):
            msg = f"Possible secret found in {path.relative_to(root)}"
            raise ValueError(msg)
    for name in ("demo.snapshot.json", "demo.report.json"):
        path = root / "examples" / name
        if not path.is_file():
            msg = f"Missing public synthetic example: {path.relative_to(root)}"
            raise ValueError(msg)
        data = json.loads(path.read_text())
        meta = data.get("meta", {})
        if (
            meta.get("input") != "synthetic example"
            or meta.get("number")
            or meta.get("url")
        ):
            msg = f"Public demo must be explicitly synthetic: {path.relative_to(root)}"
            raise ValueError(msg)
    return files


def entries(root: Path, files: list[Path]) -> list[dict[str, str]]:
    """Calculate relative paths and SHA-256 digests for a file set.

    Args:
        root: Candidate publication root used to form relative paths.
        files: Validated files to fingerprint.

    Returns:
        Manifest entries in the caller-provided order.

    Raises:
        OSError: If a file cannot be read.
    """
    return [
        {
            "path": path.relative_to(root).as_posix(),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
        for path in files
    ]


def check(
    root: Path = ROOT, *, verify: bool = False, write: bool = False
) -> int:
    """Check public release inputs and optionally verify or write a manifest.

    Args:
        root: Candidate publication root.
        verify: Compare the exact current paths and hashes to its manifest.
        write: Write a new manifest for the validated candidate set.

    Returns:
        Number of validated public files.

    Raises:
        OSError: If a file or manifest cannot be accessed.
        ValueError: If a release input is unsafe or a verified manifest differs.
        json.JSONDecodeError: If an existing manifest is malformed.

    Side Effects:
        Writes ``PUBLICATION.json`` only when ``write`` is true.
    """
    root = root.resolve()
    files = inspect_files(root)
    current = entries(root, files)
    if write:
        meta = {
            "schema": SCHEMA,
            "repository": "epistemic-ai/diffstory",
            "visibility": "public",
            "files": current,
        }
        (root / MANIFEST).write_text(
            json.dumps(meta, indent=2) + "\n", encoding="utf-8"
        )
    if verify:
        recorded = json.loads((root / MANIFEST).read_text(encoding="utf-8"))
        if (
            recorded.get("schema") != SCHEMA
            or recorded.get("repository") != "epistemic-ai/diffstory"
            or recorded.get("visibility") != "public"
            or recorded.get("files") != current
        ):
            msg = "Publication manifest does not match the exact current file set and hashes"
            raise ValueError(msg)
    return len(files)


def main(argv: Sequence[str] | None = None) -> int:
    """Run the release checker CLI and report validation status.

    Args:
        argv: Optional argument sequence; defaults to process arguments.

    Returns:
        Zero when validation succeeds; one for handled input or I/O failures.

    Side Effects:
        May write a publication manifest when requested and prints status to
        standard output or a sanitized failure to standard error.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    action = parser.add_mutually_exclusive_group()
    action.add_argument(
        "--manifest",
        action="store_true",
        help="Verify sealed paths and hashes",
    )
    action.add_argument(
        "--write-manifest",
        action="store_true",
        help="Seal reviewed public inputs",
    )
    args = parser.parse_args(argv)
    try:
        count = check(
            args.root, verify=args.manifest, write=args.write_manifest
        )
    except (OSError, ValueError, UnicodeError) as exc:
        print(f"Release check failed: {exc}", file=sys.stderr)
        return 1
    print(
        f"Release check passed: {count} public files"
        + ("; exact manifest verified." if args.manifest else ".")
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
