"""Read-only Git and GitHub ingestion. No repository code or hooks are run."""

from __future__ import annotations

import base64
import json
import os
import re
import shutil
import subprocess
import time
from datetime import datetime
from datetime import timezone
from pathlib import Path
from typing import TYPE_CHECKING
from typing import NoReturn
from urllib.error import HTTPError
from urllib.error import URLError
from urllib.parse import quote
from urllib.request import HTTPRedirectHandler
from urllib.request import Request
from urllib.request import build_opener

from . import __version__
from .analysis import MAX_SNAPSHOT_SOURCE_BYTES

if TYPE_CHECKING:
    from collections.abc import Sequence

MAX_FILE = 8_000_000
MAX_FILES = 500
MAX_API_RESPONSE_BYTES = 20_000_000
GITHUB_FILES_PER_PAGE = 100
GITHUB_MAX_ATTEMPTS = 3
GITHUB_RETRYABLE_STATUS_CODES = frozenset({502, 503, 504})
HTTP_NOT_FOUND = 404
HTTP_AUTH_FAILURE_CODES = frozenset({401, 403, HTTP_NOT_FOUND})
GITHUB_AUTH_ENV = "GITHUB_TOKEN"


def _validate_source_limit(max_source_bytes: int) -> None:
    """
    Require a positive source-byte limit no higher than the hard cap.

    Args:
        max_source_bytes: Aggregate source limit requested for ingestion.

    Raises:
        ValueError: If the limit is non-positive, not an integer, or above the
            repository-wide hard cap.

    """
    if type(max_source_bytes) is not int or max_source_bytes <= 0:
        msg = "--max-source-bytes must be a positive integer"
        raise ValueError(msg)
    if max_source_bytes > MAX_SNAPSHOT_SOURCE_BYTES:
        msg = f"--max-source-bytes cannot exceed the {MAX_SNAPSHOT_SOURCE_BYTES}-byte hard limit"
        raise ValueError(msg)


def _git(
    repo: Path,
    *args: str,
    limit: int = MAX_FILE,
) -> bytes:
    """
    Run a bounded, non-interactive Git command without invoking hooks.

    Args:
        repo: Repository directory passed to ``git -C``.
        *args: Git subcommand and its arguments; passed without a shell.
        limit: Maximum stdout size accepted from the command.

    Returns:
        Command standard output as bytes.

    Raises:
        ValueError: If Git exits unsuccessfully or output exceeds ``limit``.
        subprocess.TimeoutExpired: If Git does not finish within 90 seconds.

    """
    # No shell interpolation, no external diff drivers and no optional git locks.
    git_executable = shutil.which("git") or "git"
    command = [
        git_executable,
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
    # Fixed argv, no shell, hooks, or external diff drivers.
    proc = subprocess.run(  # noqa: S603
        command,
        capture_output=True,
        timeout=90,
        env=environment,
        check=False,
    )
    if proc.returncode:
        msg = f"git {args[0]} failed: {proc.stderr.decode(errors='replace')[:400]}"
        raise ValueError(msg)
    if len(proc.stdout) > limit:
        msg = f"Git output exceeds size limit ({limit} bytes)"
        raise ValueError(msg)
    return proc.stdout


def _git_object_info(
    repo: Path,
    revision: str,
    path: str,
    side: str,
) -> tuple[tuple[str, int] | None, str | None, str | None]:
    """
    Inspect a tree entry and report its object ID and size when it is a file.

    Args:
        repo: Local Git repository containing the revision.
        revision: Resolved commit ID to inspect.
        path: Repository-relative path from the diff.
        side: Snapshot side used in any warning.

    Returns:
        A ``(object_id, size)`` pair, warning, and no reason for a regular file;
        or no object information, warning, and ``unsupported_object`` for an
        unsupported tree entry.

    Raises:
        ValueError: If the diff path is absent from the resolved tree.
    """
    entries = _git(repo, "ls-tree", "-z", revision, "--", path).split(b"\0")
    entry = next(
        (
            item
            for item in entries
            if item and item.split(b"\t", 1)[-1].decode() == path
        ),
        None,
    )
    if entry is None:
        msg = f"Missing {side} tree entry: {path}"
        raise ValueError(msg)

    mode, kind, object_id = entry.split(b"\t", 1)[0].decode().split()
    if mode not in {"100644", "100755"} or kind != "blob":
        warning = f"Skipped {side} {path}: mode {mode}, object type {kind}"
        return None, warning, "unsupported_object"

    size = int(_git(repo, "cat-file", "-s", object_id).decode())
    return (object_id, size), None, None


def _resolve(repo: Path, ref: str) -> str:
    """
    Resolve a revision expression to a verified commit object ID.

    Args:
        repo: Repository directory containing the revision.
        ref: User-supplied Git revision expression.

    Returns:
        The full hexadecimal commit ID.

    Raises:
        ValueError: If the revision is missing, option-like, unresolved, or
            does not resolve to a commit ID.

    """
    if not ref or ref.startswith("-"):
        msg = "Invalid git revision"
        raise ValueError(msg)
    revision = ref + "^{commit}"
    sha = (
        _git(
            repo,
            "rev-parse",
            "--verify",
            "--end-of-options",
            revision,
        )
        .decode()
        .strip()
    )
    if not re.fullmatch(r"[0-9a-f]{40,64}", sha):
        msg = "Git did not resolve a commit ID"
        raise ValueError(msg)
    return sha


def from_git(  # noqa: PLR0913  # Revision selectors and separate source caps are public options.
    repo: str,
    base: str,
    head: str,
    *,
    two_dot: bool = False,
    max_files: int = MAX_FILES,
    max_source_bytes: int = MAX_SNAPSHOT_SOURCE_BYTES,
) -> dict:
    """
    Build a source snapshot from committed objects at two Git revisions.

    The working tree is never read, so uncommitted and untracked changes are
    excluded. File-count and aggregate-byte limits are enforced before a
    snapshot is returned.

    Args:
        repo: Local Git repository path.
        base: Base revision expression.
        head: Head revision expression.
        two_dot: Compare the exact base commit to head instead of using their
            merge base.
        max_files: Maximum number of changed paths to accept.
        max_source_bytes: Maximum aggregate source bytes to inspect.

    Returns:
        A ``diffstory.snapshot.v1`` mapping with metadata, source fragments,
        per-side file evidence, and warnings for skipped files.

    Raises:
        ValueError: If revisions, limits, Git objects, or source contents are
            invalid, or if the snapshot exceeds a configured bound.
        subprocess.TimeoutExpired: If a Git command exceeds its time limit.

    """
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
        msg = (
            f"{len(items)} changed files exceeds --max-files {max_files}; "
            "no partial report was written"
        )
        raise ValueError(
            msg,
        )
    fragments, warnings = [], []
    file_evidence = []
    source_bytes = 0
    for status, path in items:
        base_state = (
            {"path": path, "state": "absent"}
            if status == "A"
            else {
                "path": path,
                "state": "unavailable",
                "reason": "not_supplied",
            }
        )
        head_state = (
            {"path": path, "state": "absent"}
            if status == "D"
            else {
                "path": path,
                "state": "unavailable",
                "reason": "not_supplied",
            }
        )
        evidence_record = {"base": base_state, "head": head_state}
        file_evidence.append(evidence_record)
        for side, sha, present in (
            ("base", effective, status != "A"),
            ("head", head_sha, status != "D"),
        ):
            if not present:
                continue
            object_info, warning, reason = _git_object_info(
                root, sha, path, side
            )
            if warning:
                warnings.append(warning)
                evidence_record[side] = {
                    "path": path,
                    "state": "unavailable",
                    "reason": reason,
                }
                continue
            oid, size = object_info
            source_bytes += size
            if source_bytes > max_source_bytes:
                msg = f"Snapshot source exceeds --max-source-bytes {max_source_bytes}; no report was written"
                raise ValueError(msg)
            if size > MAX_FILE:
                warnings.append(
                    f"Skipped {side} {path}: {size} bytes exceeds {MAX_FILE}",
                )
                evidence_record[side] = {
                    "path": path,
                    "state": "unavailable",
                    "reason": "size_limit",
                }
                continue
            data = _git(root, "cat-file", "blob", oid)
            if b"\0" in data:
                warnings.append(
                    f"Skipped {side} {path}: binary or non-UTF-8 source",
                )
                evidence_record[side] = {
                    "path": path,
                    "state": "unavailable",
                    "reason": "binary_or_non_utf8",
                }
                continue
            try:
                text = data.decode("utf-8")
            except UnicodeError:
                warnings.append(
                    f"Skipped {side} {path}: binary or non-UTF-8 source",
                )
                evidence_record[side] = {
                    "path": path,
                    "state": "unavailable",
                    "reason": "binary_or_non_utf8",
                }
                continue
            fragments.append(
                {
                    "path": path,
                    "side": side,
                    "text": text,
                    "start_line": 1,
                    "scope": "full",
                },
            )
            evidence_record[side] = {
                "path": path,
                "state": "supplied",
                "coverage": "full",
            }

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
        "file_evidence": file_evidence,
        "warnings": warnings,
    }


class RejectRedirects(HTTPRedirectHandler):
    """Never forward authorization or follow an API redirect to another resource."""

    def redirect_request(
        self,
        _request: Request,
        _response: object,
        _code: int,
        _message: str,
        _headers: object,
        _new_url: str,
    ) -> NoReturn:
        """
        Reject API redirects so authorization cannot reach another URL.

        Args:
            _request: Original request, unused because redirects are rejected.
            _response: Original response, unused because redirects are rejected.
            _code: HTTP redirect status, unused because redirects are rejected.
            _message: HTTP redirect message, unused because redirects are rejected.
            _headers: Response headers, unused because redirects are rejected.
            _new_url: Proposed URL, unused because redirects are rejected.

        Raises:
            ValueError: Always, because redirects are not followed.

        """
        msg = "GitHub API redirect rejected; use the repository's canonical name."
        raise ValueError(msg)


class GitHubClient:
    """Small GitHub REST client with bounded requests and no persistent token storage."""

    def __init__(self, token: str | None = None) -> None:
        """
        Create a client that keeps the optional token in memory only.

        Args:
            token: GitHub bearer token for requests, or ``None`` for anonymous
                access.

        """
        self.token = token

    def get(
        self,
        path: str,
        *,
        optional: bool = False,
    ) -> dict | list | None:
        """
        Fetch and decode a repository-scoped GitHub REST resource.

        Args:
            path: API path beginning with ``/repos/``.
            optional: Return ``None`` for a 404 response when the resource is
                explicitly optional.

        Returns:
            The decoded JSON payload, or ``None`` for an optional missing
            resource.

        Raises:
            ValueError: If the path is outside the supported endpoint scope,
                the response is invalid or too large, or the request fails.

        """
        if not path.startswith("/repos/"):
            msg = "Unsupported GitHub API path"
            raise ValueError(msg)

        headers = {
            "Accept": "application/vnd.github+json",
            "User-Agent": f"diffstory/{__version__}",
        }
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        request = Request("https://api.github.com" + path, headers=headers)

        for attempt in range(GITHUB_MAX_ATTEMPTS):
            try:
                opener = build_opener(RejectRedirects())
                with opener.open(request, timeout=40) as response:
                    payload = response.read(MAX_API_RESPONSE_BYTES + 1)
                if len(payload) > MAX_API_RESPONSE_BYTES:
                    msg = "GitHub response exceeds size limit"
                    raise ValueError(msg)
                return json.loads(payload)
            except HTTPError as error:  # noqa: PERF203
                if error.code == HTTP_NOT_FOUND and optional:
                    return None
                if (
                    error.code in GITHUB_RETRYABLE_STATUS_CODES
                    and attempt + 1 < GITHUB_MAX_ATTEMPTS
                ):
                    time.sleep(1 + attempt)
                    continue
                if error.code in HTTP_AUTH_FAILURE_CODES:
                    explanation = (
                        "Check repository access and GitHub CLI sign-in (`gh auth status`) "
                        "or GITHUB_TOKEN."
                    )
                else:
                    explanation = "Request failed."
                msg = f"GitHub HTTP {error.code}. {explanation}"
                raise ValueError(msg) from None
            except URLError as error:
                msg = f"GitHub connection failed: {error.reason}"
                raise ValueError(
                    msg,
                ) from None
        msg = "GitHub request exhausted retries"
        raise ValueError(msg)


def parse_pr(value: str) -> tuple[str, int]:
    """
    Parse a GitHub pull-request URL or ``owner/repo#number`` reference.

    Args:
        value: Pull-request reference to parse.

    Returns:
        The repository slug and positive numeric pull-request identifier.

    Raises:
        ValueError: If ``value`` does not match a supported reference format.

    """
    pattern = (
        r"(?:https://github\.com/)?"
        r"([\w.-]+/[\w.-]+)"
        r"(?:/pull/|#)([0-9]+)"
        r"(?:/(?:files|changes))?/?"
    )
    match = re.fullmatch(pattern, value)
    if not match:
        msg = "Use owner/repo#123 or https://github.com/owner/repo/pull/123"
        raise ValueError(
            msg,
        )
    return match.group(1), int(match.group(2))


def _github_token(token_env: str) -> str | None:
    """
    Resolve a GitHub token from the environment or signed-in CLI.

    Args:
        token_env: Environment variable checked before invoking ``gh``.

    Returns:
        Token text, or ``None`` when neither source provides a token.

    Side Effects:
        Reads the environment and may run ``gh auth token`` with a timeout.

    """
    token = os.getenv(token_env)
    if token:
        return token
    try:
        # Fixed gh argv; no shell or user-controlled arguments.
        gh_executable = shutil.which("gh") or "gh"
        proc = subprocess.run(  # noqa: S603
            [gh_executable, "auth", "token"],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            timeout=10,
            text=True,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    token = proc.stdout.strip() if proc.returncode == 0 else ""
    return token or None


def _github_file_bytes(
    api: GitHubClient,
    content_info: dict,
    owner: str,
    side: str,
    path: str,
) -> tuple[bytes | None, str | None, int, str | None]:
    """
    Fetch and decode the base64 payload for one GitHub file.

    Args:
        api: Client used for authenticated GitHub requests.
        content_info: GitHub contents response for a regular file.
        owner: Repository owner used to fetch a blob fallback.
        side: Snapshot side used in any warning.
        path: Repository-relative file path.

    Returns:
        Decoded bytes, warning, byte count, and one reason code. Codes are
        ``invalid_content_size``, ``size_limit``, or
        ``unsupported_encoding``; successful reads have no reason.

    Raises:
        ValueError: If a GitHub request fails or its response is invalid.

    Side Effects:
        Reads file contents from the GitHub API.

    """
    declared_size = content_info.get("size", 0)
    if type(declared_size) is not int or declared_size < 0:
        return (
            None,
            f"Skipped {side} {path}: invalid content size",
            0,
            "invalid_content_size",
        )
    if declared_size > MAX_FILE:
        return (
            None,
            f"Skipped {side} {path}: exceeds {MAX_FILE} bytes",
            declared_size,
            "size_limit",
        )

    if content_info.get("encoding") == "base64":
        data = base64.b64decode(content_info["content"])
    else:
        blob_path = f"/repos/{owner}/git/blobs/{content_info['sha']}"
        blob = api.get(blob_path)
        if not isinstance(blob, dict) or blob.get("encoding") != "base64":
            return (
                None,
                f"Skipped {side} {path}: unsupported blob encoding",
                declared_size,
                "unsupported_encoding",
            )
        data = base64.b64decode(blob["content"])

    if len(data) > MAX_FILE:
        return (
            None,
            f"Skipped {side} {path}: exceeds {MAX_FILE} bytes",
            len(data),
            "size_limit",
        )
    return data, None, len(data), None


def _decode_github_source(data: bytes) -> str | None:
    """
    Decode one source file as UTF-8, excluding binary content.

    Args:
        data: Decoded GitHub file bytes.

    Returns:
        UTF-8 source text, or ``None`` when the file contains binary or invalid
        UTF-8 data.
    """
    if b"\0" in data:
        return None
    try:
        return data.decode("utf-8")
    except UnicodeError:
        return None


def _read_github_file(
    api: GitHubClient,
    task: tuple[str, str, str, str, str, int],
) -> tuple[dict | None, str | None, int, str | None]:
    """
    Fetch and decode one GitHub source file for a snapshot.

    Args:
        api: Client used for authenticated GitHub requests.
        task: Repository, path, revision side, revision SHA, region path, and
            file-evidence record index.

    Returns:
        Fragment, warning, bytes read or declared, and one availability reason:
        ``unsupported_object``, a reason from :func:`_github_file_bytes`,
        ``binary_or_non_utf8``, or ``None`` after a successful read.

    Raises:
        ValueError: If a GitHub request fails or its response is invalid.

    Side Effects:
        Reads file contents from the GitHub API.

    """
    owner, path, side, revision, region, _record_index = task
    contents_path = (
        f"/repos/{owner}/contents/{quote(path, safe='/')}?ref={revision}"
    )
    content_info = api.get(contents_path)
    if (
        not isinstance(content_info, dict)
        or content_info.get("type") != "file"
    ):
        return (
            None,
            f"Skipped {side} {path}: not a regular file",
            0,
            "unsupported_object",
        )

    data, warning, size, reason = _github_file_bytes(
        api,
        content_info,
        owner,
        side,
        path,
    )
    if warning:
        return None, warning, size, reason
    text = _decode_github_source(data)
    if text is None:
        return (
            None,
            f"Skipped {side} {path}: binary or non-UTF-8 source",
            size,
            "binary_or_non_utf8",
        )

    fragment = {
        "path": path,
        "side": side,
        "text": text,
        "start_line": 1,
        "scope": "full",
        "region": region,
    }
    return fragment, None, size, None


def _list_github_files(
    api: GitHubClient,
    repo: str,
    number: int,
    max_files: int,
    expected_count: int | None,
) -> Sequence[dict]:
    """
    Read every changed-file page and verify GitHub's advertised total.

    Args:
        api: Client used for authenticated GitHub requests.
        repo: Repository containing the pull request.
        number: Pull request number.
        max_files: Maximum changed files accepted by the caller.
        expected_count: Changed-file total advertised by the PR, if present.

    Returns:
        Changed-file records in the order returned by GitHub.

    Raises:
        ValueError: If GitHub returns a malformed or incomplete file list, or
            the list exceeds ``max_files``.
    """
    files = []
    page = 1
    while True:
        files_path = (
            f"/repos/{repo}/pulls/{number}/files?"
            f"per_page={GITHUB_FILES_PER_PAGE}&page={page}"
        )
        batch = api.get(files_path)
        if not isinstance(batch, list):
            msg = "Unexpected GitHub file-list response"
            raise ValueError(msg)
        files.extend(batch)
        if len(files) > max_files:
            msg = (
                f"Changed file count exceeds --max-files {max_files}; "
                "no partial report was written"
            )
            raise ValueError(msg)
        if len(batch) < GITHUB_FILES_PER_PAGE:
            break
        page += 1

    if expected_count is not None and len(files) != expected_count:
        msg = (
            "GitHub did not return the complete file list; refusing an "
            "apparently complete report"
        )
        raise ValueError(msg)
    return files


def _github_file_tasks(
    files: Sequence[dict],
    repo: str,
    head_repo: str,
    base: str,
    head: str,
) -> tuple[tuple[tuple[str, str, str, str, str, int], ...], list[dict]]:
    """
    Plan base and head content requests for each changed path.

    Args:
        files: Verified pull-request changed-file records.
        repo: Base repository name.
        head_repo: Repository containing the pull-request head.
        base: Merge-base commit ID.
        head: Head commit ID.

    Returns:
        Content tasks and initial file evidence in changed-file order. Added
        and removed files receive one absent side. Renames link the required
        ``previous_filename`` to the new path.
    """
    tasks = []
    evidence = []
    for record_index, file_info in enumerate(files):
        path = file_info["filename"]
        status = file_info["status"]
        if status == "renamed":
            old_path = file_info.get("previous_filename")
            if not isinstance(old_path, str) or not old_path:
                msg = "GitHub renamed file record requires previous_filename"
                raise ValueError(msg)
        else:
            old_path = path
        base_side = (
            {"path": path, "state": "absent"}
            if status == "added"
            else {
                "path": old_path,
                "state": "unavailable",
                "reason": "not_supplied",
            }
        )
        head_side = (
            {"path": path, "state": "absent"}
            if status == "removed"
            else {
                "path": path,
                "state": "unavailable",
                "reason": "not_supplied",
            }
        )
        evidence.append({"base": base_side, "head": head_side})
        if status != "added":
            tasks.append((repo, old_path, "base", base, path, record_index))
        if status != "removed":
            tasks.append((head_repo, path, "head", head, path, record_index))
    return tuple(tasks), evidence


def _read_github_tasks(
    api: GitHubClient,
    tasks: Sequence[tuple[str, str, str, str, str, int]],
    max_source_bytes: int,
) -> tuple[list[dict], list[str], int, list[dict]]:
    """
    Fetch source fragments while enforcing the aggregate byte cap.

    Args:
        api: Client used for authenticated GitHub requests.
        tasks: Content-fetch plan from :func:`_github_file_tasks`.
        max_source_bytes: Maximum aggregate source bytes accepted.

    Returns:
        Fragments, warnings, byte count, and structured side-read results.

    Raises:
        ValueError: If the aggregate source exceeds ``max_source_bytes``.
    """
    fragments = []
    warnings = []
    results = []
    source_bytes = 0
    for task in tasks:
        fragment, warning, size, reason = _read_github_file(api, task)
        source_bytes += size
        if source_bytes > max_source_bytes:
            msg = (
                f"Snapshot source exceeds --max-source-bytes {max_source_bytes}; "
                "no report was written"
            )
            raise ValueError(msg)
        if fragment:
            fragments.append(fragment)
            results.append(
                {
                    "record_index": task[5],
                    "side": task[2],
                    "state": "supplied",
                },
            )
        if warning:
            warnings.append(warning)
            results.append(
                {
                    "record_index": task[5],
                    "side": task[2],
                    "state": "unavailable",
                    "reason": reason,
                },
            )
    return fragments, warnings, source_bytes, results


def from_github(
    value: str,
    *,
    token_env: str = GITHUB_AUTH_ENV,
    max_files: int = MAX_FILES,
    max_source_bytes: int = MAX_SNAPSHOT_SOURCE_BYTES,
) -> dict:
    """
    Fetch a complete, size-bounded pull request snapshot from GitHub.

    The pull request's merge base defines the base revision. Changed-file pages
    are verified before the snapshot is returned, and files are fetched from
    the correct repository for each side of a fork.

    Args:
        value: Pull-request URL or ``owner/repo#number`` reference.
        token_env: Environment variable used before the signed-in ``gh`` token.
        max_files: Maximum changed-file count to accept.
        max_source_bytes: Maximum aggregate source bytes to inspect.

    Returns:
        A ``diffstory.snapshot.v1`` mapping with source fragments, metadata,
        per-side file evidence, and warnings for unsupported files.

    Raises:
        ValueError: If the PR is inaccessible, incomplete, malformed, changes
            during retrieval, or exceeds a configured limit.

    Side Effects:
        Reads the PR and changed source from GitHub and may invoke ``gh`` to
        obtain the saved CLI token.

    """
    _validate_source_limit(max_source_bytes)
    repo, number = parse_pr(value)
    api = GitHubClient(_github_token(token_env))
    pr = api.get(f"/repos/{repo}/pulls/{number}")

    # GitHub PR changes are relative to merge-base, not necessarily the current base tip.
    base_tip = pr["base"]["sha"]
    head = pr["head"]["sha"]
    compare = api.get(f"/repos/{repo}/compare/{base_tip}...{head}?per_page=1")
    base = compare["merge_base_commit"]["sha"]

    files = _list_github_files(
        api,
        repo,
        number,
        max_files,
        pr.get("changed_files"),
    )
    head_repo = (pr.get("head", {}).get("repo") or {}).get("full_name") or repo
    tasks, file_evidence = _github_file_tasks(
        files, repo, head_repo, base, head
    )
    fragments, warnings, source_bytes, read_results = _read_github_tasks(
        api,
        tasks,
        max_source_bytes,
    )
    for result in read_results:
        side_name = result["side"]
        side = file_evidence[result["record_index"]][side_name]
        if result["state"] == "supplied":
            file_evidence[result["record_index"]][side_name] = {
                "path": side["path"],
                "state": "supplied",
                "coverage": "full",
            }
        else:
            file_evidence[result["record_index"]][side_name] = {
                "path": side["path"],
                "state": "unavailable",
                "reason": result["reason"],
            }

    # Fail if the PR changed during the read; don't silently mix multiple revisions.
    end = api.get(f"/repos/{repo}/pulls/{number}")
    if end["head"]["sha"] != head or end["base"]["sha"] != base_tip:
        msg = "PR revisions changed while fetching. Retry to capture a consistent snapshot."
        raise ValueError(
            msg,
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
        "file_evidence": file_evidence,
        "warnings": warnings,
    }
