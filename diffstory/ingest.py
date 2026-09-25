"""Read-only Git and GitHub ingestion. No repository code or hooks are run."""
from __future__ import annotations

import base64
import json
import os
import re
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import quote
from urllib.request import HTTPRedirectHandler, Request, build_opener

from . import __version__
from .analysis import MAX_SNAPSHOT_SOURCE_BYTES

MAX_FILE = 8_000_000
MAX_FILES = 500


def _validate_source_limit(max_source_bytes: int) -> None:
    if type(max_source_bytes) is not int or max_source_bytes <= 0:
        raise ValueError("--max-source-bytes must be a positive integer")
    if max_source_bytes > MAX_SNAPSHOT_SOURCE_BYTES:
        raise ValueError(f"--max-source-bytes cannot exceed the {MAX_SNAPSHOT_SOURCE_BYTES}-byte hard limit")


def _git(repo: Path, *args: str, limit: int = MAX_FILE) -> bytes:
    # No shell interpolation, no external diff drivers and no optional git locks.
    command = [
        "git",
        "--no-pager",
        "-c",
        "core.hooksPath=/dev/null",
        "-C",
        str(repo),
        *args,
    ]
    environment = {
        **os.environ,
        "GIT_OPTIONAL_LOCKS": "0",
        "GIT_TERMINAL_PROMPT": "0",
    }
    proc = subprocess.run(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=90,
        env=environment,
    )
    if proc.returncode:
        raise ValueError(f"git {args[0]} failed: {proc.stderr.decode(errors='replace')[:400]}")
    if len(proc.stdout) > limit:
        raise ValueError(f"Git output exceeds size limit ({limit} bytes)")
    return proc.stdout


def _resolve(repo: Path, ref: str) -> str:
    if not ref or ref.startswith("-"):
        raise ValueError("Invalid git revision")
    revision = ref + "^{commit}"
    sha = _git(
        repo,
        "rev-parse",
        "--verify",
        "--end-of-options",
        revision,
    ).decode().strip()
    if not re.fullmatch(r"[0-9a-f]{40,64}", sha):
        raise ValueError("Git did not resolve a commit ID")
    return sha


def from_git(
    repo: str,
    base: str,
    head: str,
    *,
    two_dot: bool = False,
    max_files: int = MAX_FILES,
    max_source_bytes: int = MAX_SNAPSHOT_SOURCE_BYTES,
) -> dict:
    _validate_source_limit(max_source_bytes)
    root = Path(repo).resolve()
    base_sha, head_sha = _resolve(root, base), _resolve(root, head)
    effective = (
        base_sha
        if two_dot
        else _git(root, "merge-base", base_sha, head_sha).decode().strip()
    )
    # --no-renames intentionally presents file renames as delete+add. The compiler
    # then detects moved declarations across paths using evidence, not heuristics in Git.
    names = _git(
        root,
        "diff",
        "--name-status",
        "-z",
        "--no-renames",
        "--no-ext-diff",
        "--no-textconv",
        effective,
        head_sha,
        "--",
    ).split(b"\0")
    items = [
        (names[index].decode(), names[index + 1].decode("utf-8"))
        for index in range(0, len(names) - 1, 2)
    ]
    if len(items) > max_files:
        raise ValueError(
            f"{len(items)} changed files exceeds --max-files {max_files}; "
            "no partial report was written"
        )
    fragments, warnings = [], []
    source_bytes = 0
    for status, path in items:
        for side, sha, present in (("base", effective, status != "A"), ("head", head_sha, status != "D")):
            if not present:
                continue
            # Inspect object type, mode and length before reading. Skip gitlinks and symlinks.
            entries = _git(root, "ls-tree", "-z", sha, "--", path).split(b"\0")
            entry = next(
                (
                    item
                    for item in entries
                    if item and item.split(b"\t", 1)[-1].decode() == path
                ),
                None,
            )
            if entry is None:
                raise ValueError(f"Missing {side} tree entry: {path}")
            prefix = entry.split(b"\t", 1)[0].decode().split()
            mode, kind, oid = prefix[:3]
            if mode not in {"100644", "100755"} or kind != "blob":
                warnings.append(f"Skipped {side} {path}: mode {mode}, object type {kind}")
                continue
            size = int(_git(root, "cat-file", "-s", oid).decode())
            source_bytes += size
            if source_bytes > max_source_bytes:
                raise ValueError(f"Snapshot source exceeds --max-source-bytes {max_source_bytes}; no report was written")
            if size > MAX_FILE:
                warnings.append(f"Skipped {side} {path}: {size} bytes exceeds {MAX_FILE}")
                continue
            data = _git(root, "cat-file", "blob", oid)
            try:
                if b"\0" in data:
                    raise UnicodeError("NUL byte")
                text = data.decode("utf-8")
            except UnicodeError:
                warnings.append(f"Skipped {side} {path}: binary or non-UTF-8 source")
                continue
            fragments.append(
                {
                    "path": path,
                    "side": side,
                    "text": text,
                    "start_line": 1,
                    "scope": "full",
                }
            )

    meta = {
        "title": f"{base} → {head}",
        "repository": "",
        "base_sha": effective,
        "requested_base_sha": base_sha,
        "head_sha": head_sha,
        "scope": "changed files",
        "input": "local git",
        "comparison": "two-dot" if two_dot else "merge-base to head",
        "changed_files": len(items),
        "source_bytes": source_bytes,
        "captured_at": datetime.now(timezone.utc).isoformat(),
        "description": (
            "Source was read from committed Git objects, not the working tree. "
            "Untracked and uncommitted edits are excluded."
        ),
    }
    return {
        "schema": "diffstory.snapshot.v1",
        "meta": meta,
        "fragments": fragments,
        "warnings": warnings,
    }


class RejectRedirects(HTTPRedirectHandler):
    """Never forward authorization or follow an API redirect to another resource."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise ValueError("GitHub API redirect rejected; use the repository's canonical name.")


class GitHubClient:
    """Small GitHub REST client with bounded requests and no persistent token storage."""

    def __init__(self, token: str | None = None):
        self.token = token

    def get(self, path: str, *, optional: bool = False):
        if not path.startswith("/repos/"):
            raise ValueError("Unsupported GitHub API path")

        headers = {
            "Accept": "application/vnd.github+json",
            "User-Agent": f"diffstory/{__version__}",
        }
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        request = Request("https://api.github.com" + path, headers=headers)

        for attempt in range(3):
            try:
                opener = build_opener(RejectRedirects())
                with opener.open(request, timeout=40) as response:
                    payload = response.read(20_000_001)
                if len(payload) > 20_000_000:
                    raise ValueError("GitHub response exceeds size limit")
                return json.loads(payload)
            except HTTPError as error:
                if error.code == 404 and optional:
                    return None
                if error.code in (502, 503, 504) and attempt < 2:
                    time.sleep(1 + attempt)
                    continue
                if error.code in (401, 403, 404):
                    explanation = "Check repository access and GITHUB_TOKEN."
                else:
                    explanation = "Request failed."
                raise ValueError(f"GitHub HTTP {error.code}. {explanation}") from None
            except URLError as error:
                raise ValueError(
                    f"GitHub connection failed: {error.reason}"
                ) from None
        raise ValueError("GitHub request exhausted retries")


def parse_pr(value: str) -> tuple[str, int]:
    pattern = (
        r"(?:https://github\.com/)?"
        r"([\w.-]+/[\w.-]+)"
        r"(?:/pull/|#)([0-9]+)"
        r"(?:/(?:files|changes))?/?"
    )
    match = re.fullmatch(pattern, value)
    if not match:
        raise ValueError(
            "Use owner/repo#123 or https://github.com/owner/repo/pull/123"
        )
    return match.group(1), int(match.group(2))


def _read_github_file(api: GitHubClient, task: tuple) -> tuple[dict | None, str | None, int]:
    owner, path, side, revision, region = task
    contents_path = f"/repos/{owner}/contents/{quote(path, safe='/')}?ref={revision}"
    content_info = api.get(contents_path)
    if not isinstance(content_info, dict) or content_info.get("type") != "file":
        return None, f"Skipped {side} {path}: not a regular file", 0

    declared_size = content_info.get("size", 0)
    if type(declared_size) is not int or declared_size < 0:
        return None, f"Skipped {side} {path}: invalid content size", 0
    if declared_size > MAX_FILE:
        return None, f"Skipped {side} {path}: exceeds {MAX_FILE} bytes", declared_size

    if content_info.get("encoding") == "base64":
        data = base64.b64decode(content_info["content"])
    else:
        blob_path = f"/repos/{owner}/git/blobs/{content_info['sha']}"
        blob = api.get(blob_path)
        if blob.get("encoding") != "base64":
            return (
                None,
                f"Skipped {side} {path}: unsupported blob encoding",
                declared_size,
            )
        data = base64.b64decode(blob["content"])

    if len(data) > MAX_FILE:
        return None, f"Skipped {side} {path}: exceeds {MAX_FILE} bytes", len(data)
    try:
        if b"\0" in data:
            raise UnicodeError("NUL byte")
        text = data.decode("utf-8")
    except UnicodeError:
        return None, f"Skipped {side} {path}: binary or non-UTF-8 source", len(data)

    fragment = {
        "path": path,
        "side": side,
        "text": text,
        "start_line": 1,
        "scope": "full",
        "region": region,
    }
    return fragment, None, len(data)


def from_github(
    value: str,
    *,
    token_env: str = "GITHUB_TOKEN",
    max_files: int = MAX_FILES,
    max_source_bytes: int = MAX_SNAPSHOT_SOURCE_BYTES,
) -> dict:
    _validate_source_limit(max_source_bytes)
    repo, number = parse_pr(value)
    api = GitHubClient(os.getenv(token_env))
    pr = api.get(f"/repos/{repo}/pulls/{number}")

    # GitHub PR changes are relative to merge-base, not necessarily the current base tip.
    base_tip = pr["base"]["sha"]
    head = pr["head"]["sha"]
    compare = api.get(f"/repos/{repo}/compare/{base_tip}...{head}?per_page=1")
    base = compare["merge_base_commit"]["sha"]

    files, page = [], 1
    while True:
        files_path = f"/repos/{repo}/pulls/{number}/files?per_page=100&page={page}"
        batch = api.get(files_path)
        if not isinstance(batch, list):
            raise ValueError("Unexpected GitHub file-list response")
        files.extend(batch)
        if len(files) > max_files:
            raise ValueError(
                f"Changed file count exceeds --max-files {max_files}; "
                "no partial report was written"
            )
        if len(batch) < 100:
            break
        page += 1

    if len(files) != pr.get("changed_files", len(files)):
        raise ValueError(
            "GitHub did not return the complete file list; refusing an apparently "
            "complete report"
        )

    warnings = []
    tasks = []
    head_repo = (pr.get("head", {}).get("repo") or {}).get("full_name") or repo
    for file_info in files:
        old_path = file_info.get("previous_filename", file_info["filename"])
        if file_info["status"] != "added":
            tasks.append((repo, old_path, "base", base, file_info["filename"]))
        if file_info["status"] != "removed":
            tasks.append(
                (
                    head_repo,
                    file_info["filename"],
                    "head",
                    head,
                    file_info["filename"],
                )
            )

    fragments = []
    source_bytes = 0
    for task in tasks:
        fragment, warning, size = _read_github_file(api, task)
        source_bytes += size
        if source_bytes > max_source_bytes:
            raise ValueError(
                f"Snapshot source exceeds --max-source-bytes {max_source_bytes}; "
                "no report was written"
            )
        if fragment:
            fragments.append(fragment)
        if warning:
            warnings.append(warning)

    # Fail if the PR changed during the read; don't silently mix multiple revisions.
    end = api.get(f"/repos/{repo}/pulls/{number}")
    if end["head"]["sha"] != head or end["base"]["sha"] != base_tip:
        raise ValueError(
            "PR revisions changed while fetching. Retry to capture a consistent snapshot."
        )

    meta = {
        "title": pr["title"],
        "repository": repo,
        "number": number,
        "url": pr["html_url"],
        "base_sha": base,
        "requested_base_sha": base_tip,
        "head_sha": head,
        "head_repository": head_repo,
        "scope": "changed files",
        "input": "GitHub REST",
        "comparison": "merge-base to head",
        "changed_files": len(files),
        "source_bytes": source_bytes,
        "additions": pr.get("additions"),
        "deletions": pr.get("deletions"),
        "description": pr.get("body") or "",
        "draft": pr.get("draft", False),
        "state": pr.get("state"),
        "captured_at": datetime.now(timezone.utc).isoformat(),
        "validation": (
            "PR description is author-reported evidence. "
            "No test run was executed or verified by this compiler."
        ),
    }
    return {
        "schema": "diffstory.snapshot.v1",
        "meta": meta,
        "fragments": fragments,
        "warnings": warnings,
    }
