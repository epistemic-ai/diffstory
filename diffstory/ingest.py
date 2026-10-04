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

from pydantic import Field

from . import __version__
from .models import MAX_SNAPSHOT_SOURCE_BYTES
from .models import AbsentFile
from .models import AvailabilityReason
from .models import FileEvidence
from .models import RevisionSide
from .models import SourceFragment
from .models import StrictModel
from .models import SuppliedFile
from .models import UnavailableFile

if TYPE_CHECKING:
    from collections.abc import Iterable
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


class _GitObject(StrictModel):
    """Describe a regular blob at a pinned Git revision.

    Attributes:
        object_id: Blob object ID accepted by ``git cat-file``.
        size: Nonnegative declared blob bytes.
    """

    object_id: str
    size: int = Field(ge=0)


class _SkippedFile(StrictModel):
    """Carry a nonfatal source-read failure without implying file absence.

    Attributes:
        path: File path retained in unavailable evidence.
        reason: Supported availability code.
        warning: Human-readable skip explanation.
        size: Nonnegative declared or actual bytes charged to the source cap.
    """

    path: str
    reason: AvailabilityReason
    warning: str
    size: int = Field(default=0, ge=0)

    @property
    def evidence(self) -> UnavailableFile:
        """Return unavailable evidence for this path and skip reason."""
        return UnavailableFile(path=self.path, reason=self.reason)


class _GitHubFileTask(StrictModel):
    """Identify one pinned source read and its comparison record.

    Attributes:
        repository: Repository containing the requested revision.
        path: Path to fetch from that revision.
        side: Base or head revision.
        revision: Pinned commit ID.
        region: New path joining before/after source for a copy or rename.
        record_index: Position of the comparison record to update.
    """

    repository: str
    path: str
    side: RevisionSide
    revision: str
    region: str
    record_index: int = Field(ge=0)


class _GitHubFilePlan(StrictModel):
    """Keep pinned content requests with their file evidence.

    Attributes:
        tasks: Source reads in changed-file and base-then-head order.
        file_evidence: Initial absent or unavailable states for each file.
    """

    tasks: tuple[_GitHubFileTask, ...]
    file_evidence: list[FileEvidence]


class _SourceBytes(StrictModel):
    """Hold successfully decoded source bytes before UTF-8 validation.

    Attributes:
        data: Decoded content, including a zero-byte file.
        size: Nonnegative actual bytes charged to the source limit.
    """

    data: bytes
    size: int = Field(ge=0)


class _SourceRead(StrictModel):
    """Keep a file's read result and source evidence together.

    Attributes:
        evidence: Supplied state for the requested file side.
        size: Nonnegative bytes charged to the source limit.
        fragment: Validated full source after a successful read.
    """

    evidence: SuppliedFile
    size: int = Field(ge=0)
    fragment: SourceFragment


class _SourceCollection(StrictModel):
    """Collect GitHub source reads without a second reconstruction pass.

    Attributes:
        fragments: Successfully read source in request order.
        warnings: Skip warnings in request order.
        source_bytes: Actual or declared bytes charged to the aggregate limit.
        file_evidence: Comparison records updated with each read's source state.
    """

    fragments: list[SourceFragment]
    warnings: list[str]
    source_bytes: int = Field(ge=0)
    file_evidence: list[FileEvidence]


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
) -> _GitObject | _SkippedFile:
    """
    Inspect a tree entry and report its object ID and size when it is a file.

    Args:
        repo: Local Git repository containing the revision.
        revision: Resolved commit ID to inspect.
        path: Repository-relative path from the diff.
        side: Snapshot side used in any warning.

    Returns:
        Blob ID and size, or an unsupported-object warning and reason.

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
        return _SkippedFile(
            path=path, warning=warning, reason="unsupported_object"
        )

    size = int(_git(repo, "cat-file", "-s", object_id).decode())
    return _GitObject(object_id=object_id, size=size)


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
        evidence_record = FileEvidence(
            base=AbsentFile(path=path)
            if status == "A"
            else UnavailableFile(path=path, reason="not_supplied"),
            head=AbsentFile(path=path)
            if status == "D"
            else UnavailableFile(path=path, reason="not_supplied"),
        )
        file_evidence.append(evidence_record)
        for side, sha, present in (
            ("base", effective, status != "A"),
            ("head", head_sha, status != "D"),
        ):
            if not present:
                continue
            object_info = _git_object_info(root, sha, path, side)
            if isinstance(object_info, _SkippedFile):
                warnings.append(object_info.warning)
                setattr(evidence_record, side, object_info.evidence)
                continue
            size = object_info.size
            source_bytes += size
            if source_bytes > max_source_bytes:
                msg = f"Snapshot source exceeds --max-source-bytes {max_source_bytes}; no report was written"
                raise ValueError(msg)
            if size > MAX_FILE:
                warnings.append(
                    f"Skipped {side} {path}: {size} bytes exceeds {MAX_FILE}",
                )
                setattr(
                    evidence_record,
                    side,
                    UnavailableFile(path=path, reason="size_limit"),
                )
                continue
            data = _git(root, "cat-file", "blob", object_info.object_id)
            text = _decode_source(data)
            if text is None:
                warnings.append(
                    f"Skipped {side} {path}: binary or non-UTF-8 source",
                )
                setattr(
                    evidence_record,
                    side,
                    UnavailableFile(path=path, reason="binary_or_non_utf8"),
                )
                continue
            fragments.append(
                SourceFragment(
                    path=path,
                    side=side,
                    text=text,
                    start_line=1,
                    scope="full",
                )
            )
            setattr(
                evidence_record, side, SuppliedFile(path=path, coverage="full")
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
        "fragments": [
            fragment.model_dump(exclude_unset=True) for fragment in fragments
        ],
        "file_evidence": [record.model_dump() for record in file_evidence],
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
) -> _SourceBytes | _SkippedFile:
    """
    Fetch and decode the base64 payload for one GitHub file.

    Args:
        api: Client used for authenticated GitHub requests.
        content_info: GitHub contents response for a regular file.
        owner: Repository owner used to fetch a blob fallback.
        side: Snapshot side used in any warning.
        path: Repository-relative file path.

    Returns:
        Decoded bytes or a typed skip result. Codes are
        ``invalid_content_size``, ``size_limit``, or
        ``unsupported_encoding``; successful reads have no reason.

    Raises:
        ValueError: If a GitHub request fails or its response is invalid.

    Side Effects:
        Reads file contents from the GitHub API.

    """
    declared_size = content_info.get("size", 0)
    if type(declared_size) is not int or declared_size < 0:
        return _SkippedFile(
            path=path,
            warning=f"Skipped {side} {path}: invalid content size",
            size=0,
            reason="invalid_content_size",
        )
    if declared_size > MAX_FILE:
        return _SkippedFile(
            path=path,
            warning=f"Skipped {side} {path}: exceeds {MAX_FILE} bytes",
            size=declared_size,
            reason="size_limit",
        )

    if content_info.get("encoding") == "base64":
        data = base64.b64decode(content_info["content"])
    else:
        blob_path = f"/repos/{owner}/git/blobs/{content_info['sha']}"
        blob = api.get(blob_path)
        if not isinstance(blob, dict) or blob.get("encoding") != "base64":
            return _SkippedFile(
                path=path,
                warning=f"Skipped {side} {path}: unsupported blob encoding",
                size=declared_size,
                reason="unsupported_encoding",
            )
        data = base64.b64decode(blob["content"])

    if len(data) > MAX_FILE:
        return _SkippedFile(
            path=path,
            warning=f"Skipped {side} {path}: exceeds {MAX_FILE} bytes",
            size=len(data),
            reason="size_limit",
        )
    return _SourceBytes(data=data, size=len(data))


def _decode_source(data: bytes) -> str | None:
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
    task: _GitHubFileTask,
) -> _SourceRead | _SkippedFile:
    """
    Fetch and decode one GitHub source file for a snapshot.

    Args:
        api: Client used for authenticated GitHub requests.
        task: Pinned source request and its evidence record index.

    Returns:
        Source fragment and supplied evidence, or a skipped-read warning with
        unavailable evidence. Both results include their byte count.

    Raises:
        ValueError: If a GitHub request fails or its response is invalid.

    Side Effects:
        Reads file contents from the GitHub API.

    """
    owner = task.repository
    path = task.path
    side = task.side
    contents_path = (
        f"/repos/{owner}/contents/{quote(path, safe='/')}?ref={task.revision}"
    )
    content_info = api.get(contents_path)
    if (
        not isinstance(content_info, dict)
        or content_info.get("type") != "file"
    ):
        return _SkippedFile(
            path=path,
            reason="unsupported_object",
            warning=f"Skipped {side} {path}: not a regular file",
            size=0,
        )

    payload = _github_file_bytes(
        api,
        content_info,
        owner,
        side,
        path,
    )
    if isinstance(payload, _SkippedFile):
        return payload
    text = _decode_source(payload.data)
    if text is None:
        return _SkippedFile(
            path=path,
            reason="binary_or_non_utf8",
            warning=f"Skipped {side} {path}: binary or non-UTF-8 source",
            size=payload.size,
        )

    fragment = {
        "path": path,
        "side": side,
        "text": text,
        "start_line": 1,
        "scope": "full",
        "region": task.region,
    }
    return _SourceRead(
        evidence=SuppliedFile(path=path, coverage="full"),
        fragment=fragment,
        size=payload.size,
    )


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
    files: Iterable[dict],
    repo: str,
    head_repo: str,
    base: str,
    head: str,
) -> _GitHubFilePlan:
    """Plan pinned content reads and initial evidence for each changed file.

    Args:
        files: Verified pull-request changed-file records.
        repo: Repository containing the base revision.
        head_repo: Repository containing the head revision.
        base: Effective merge-base commit ID.
        head: Head commit ID.

    Returns:
        Named tasks and comparison records in changed-file order.

    Raises:
        ValueError: If a copy or rename omits its original path.
    """
    tasks = []
    evidence = []
    # Each changed file has one comparison record and at most two source reads.
    # Copies and renames use their old base path and new head path.
    for index, file_info in enumerate(files):
        path, status = file_info["filename"], file_info["status"]
        old_path = path
        if status in {"renamed", "copied"}:
            old_path = file_info.get("previous_filename")
            if not isinstance(old_path, str) or not old_path:
                msg = f"GitHub {status} file record requires previous_filename"
                raise ValueError(msg)
        record = FileEvidence(
            base=AbsentFile(path=path)
            if status == "added"
            else UnavailableFile(path=old_path, reason="not_supplied"),
            head=AbsentFile(path=path)
            if status == "removed"
            else UnavailableFile(path=path, reason="not_supplied"),
        )
        evidence.append(record)
        for side, owner, revision in (
            ("base", repo, base),
            ("head", head_repo, head),
        ):
            source = getattr(record, side)
            if isinstance(source, AbsentFile):
                continue
            tasks.append(
                _GitHubFileTask(
                    repository=owner,
                    path=source.path,
                    side=side,
                    revision=revision,
                    region=path,
                    record_index=index,
                )
            )
    return _GitHubFilePlan(tasks=tuple(tasks), file_evidence=evidence)


def _read_github_tasks(
    api: GitHubClient,
    plan: _GitHubFilePlan,
    max_source_bytes: int,
) -> _SourceCollection:
    """Read a pinned plan and update each file's evidence as the read completes.

    Args:
        api: Client used for authenticated GitHub requests.
        plan: Content reads and their initial comparison records.
        max_source_bytes: Maximum aggregate source bytes accepted.

    Returns:
        Collected source and warnings with final comparison states.

    Raises:
        ValueError: If source exceeds the aggregate byte cap.
    """
    result = _SourceCollection(
        fragments=[],
        warnings=[],
        source_bytes=0,
        file_evidence=plan.file_evidence,
    )
    for task in plan.tasks:
        read = _read_github_file(api, task)
        result.source_bytes += read.size
        if result.source_bytes > max_source_bytes:
            msg = f"Snapshot source exceeds --max-source-bytes {max_source_bytes}; no report was written"
            raise ValueError(msg)
        record = result.file_evidence[task.record_index]
        setattr(record, task.side, read.evidence)
        if isinstance(read, _SourceRead):
            result.fragments.append(read.fragment)
        else:
            result.warnings.append(read.warning)
    return result


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
    plan = _github_file_tasks(files, repo, head_repo, base, head)
    source = _read_github_tasks(
        api,
        plan,
        max_source_bytes,
    )

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
        "source_bytes": source.source_bytes,
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
        "fragments": [
            fragment.model_dump(exclude_unset=True)
            for fragment in source.fragments
        ],
        "file_evidence": [
            record.model_dump() for record in source.file_evidence
        ],
        "warnings": source.warnings,
    }
