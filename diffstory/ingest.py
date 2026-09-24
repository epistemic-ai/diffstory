"""Read-only Git and GitHub ingestion. No repository code or hooks are run."""
from __future__ import annotations

import base64
import json
import os
import re
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlencode
from urllib.request import HTTPRedirectHandler, Request, build_opener
from . import __version__

MAX_FILE = 8_000_000
MAX_FILES = 500


def _git(repo: Path, *args: str, limit: int = MAX_FILE) -> bytes:
    # No shell interpolation, no external diff drivers and no optional git locks.
    proc = subprocess.run(["git", "--no-pager", "-c", "core.hooksPath=/dev/null", "-C", str(repo), *args],
                          stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=90,
                          env={**os.environ, "GIT_OPTIONAL_LOCKS": "0", "GIT_TERMINAL_PROMPT": "0"})
    if proc.returncode:
        raise ValueError(f"git {args[0]} failed: {proc.stderr.decode(errors='replace')[:400]}")
    if len(proc.stdout) > limit: raise ValueError(f"Git output exceeds size limit ({limit} bytes)")
    return proc.stdout


def _resolve(repo: Path, ref: str) -> str:
    if not ref or ref.startswith("-"): raise ValueError("Invalid git revision")
    sha = _git(repo, "rev-parse", "--verify", "--end-of-options", ref + "^{commit}").decode().strip()
    if not re.fullmatch(r"[0-9a-f]{40,64}", sha): raise ValueError("Git did not resolve a commit ID")
    return sha


def from_git(repo: str, base: str, head: str, *, two_dot: bool = False, max_files: int = MAX_FILES) -> dict:
    root = Path(repo).resolve()
    base_sha, head_sha = _resolve(root, base), _resolve(root, head)
    effective = base_sha if two_dot else _git(root, "merge-base", base_sha, head_sha).decode().strip()
    # --no-renames intentionally presents file renames as delete+add. The compiler
    # then detects moved declarations across paths using evidence, not heuristics in Git.
    names = _git(root, "diff", "--name-status", "-z", "--no-renames", "--no-ext-diff", "--no-textconv", effective, head_sha, "--").split(b"\0")
    items = [(names[i].decode(), names[i+1].decode("utf-8")) for i in range(0, len(names)-1, 2)]
    if len(items) > max_files: raise ValueError(f"{len(items)} changed files exceeds --max-files {max_files}; no partial report was written")
    fragments, warnings = [], []
    for status, path in items:
        for side, sha, present in (("base", effective, status != "A"), ("head", head_sha, status != "D")):
            if not present: continue
            # Inspect object type, mode and length before reading. Skip gitlinks and symlinks.
            entries = _git(root, "ls-tree", "-z", sha, "--", path).split(b"\0")
            entry = next((x for x in entries if x and x.split(b"\t", 1)[-1].decode() == path), None)
            if entry is None: raise ValueError(f"Missing {side} tree entry: {path}")
            prefix = entry.split(b"\t", 1)[0].decode().split()
            mode, kind, oid = prefix[:3]
            if mode not in {"100644", "100755"} or kind != "blob":
                warnings.append(f"Skipped {side} {path}: mode {mode}, object type {kind}"); continue
            size = int(_git(root, "cat-file", "-s", oid).decode())
            if size > MAX_FILE:
                warnings.append(f"Skipped {side} {path}: {size} bytes exceeds {MAX_FILE}"); continue
            data = _git(root, "cat-file", "blob", oid)
            try:
                if b"\0" in data: raise UnicodeError("NUL byte")
                text = data.decode("utf-8")
            except UnicodeError:
                warnings.append(f"Skipped {side} {path}: binary or non-UTF-8 source"); continue
            fragments.append({"path": path, "side": side, "text": text, "start_line": 1, "scope": "full"})
    return {"schema": "diffstory.snapshot.v1", "meta": {
        "title": f"{base} → {head}", "repository": "", "base_sha": effective, "requested_base_sha": base_sha,
        "head_sha": head_sha, "scope": "changed files", "input": "local git", "comparison": "two-dot" if two_dot else "merge-base to head",
        "changed_files": len(items), "captured_at": datetime.now(timezone.utc).isoformat(),
        "description": "Source was read from committed Git objects, not the working tree. Untracked and uncommitted edits are excluded."},
        "fragments": fragments, "warnings": warnings}


class RejectRedirects(HTTPRedirectHandler):
    """Never forward authorization or follow an API redirect to another resource."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise ValueError("GitHub API redirect rejected; use the repository's canonical name.")


class GitHubClient:
    """Small GitHub REST client with bounded requests and no persistent token storage."""
    def __init__(self, token: str | None = None): self.token = token

    def get(self, path: str, *, optional: bool = False):
        if not path.startswith("/repos/"): raise ValueError("Unsupported GitHub API path")
        req = Request("https://api.github.com" + path, headers={
            "Accept": "application/vnd.github+json", "User-Agent": f"diffstory/{__version__}",
            **({"Authorization": f"Bearer {self.token}"} if self.token else {})})
        for attempt in range(3):
            try:
                with build_opener(RejectRedirects()).open(req, timeout=40) as response:
                    payload = response.read(20_000_001)
                if len(payload) > 20_000_000: raise ValueError("GitHub response exceeds size limit")
                return json.loads(payload)
            except HTTPError as e:
                if e.code == 404 and optional: return None
                if e.code in (502, 503, 504) and attempt < 2: time.sleep(1 + attempt); continue
                explanation = "Check repository access and GITHUB_TOKEN." if e.code in (401, 403, 404) else "Request failed."
                raise ValueError(f"GitHub HTTP {e.code}. {explanation}") from None
            except URLError as e:
                raise ValueError(f"GitHub connection failed: {e.reason}") from None
        raise ValueError("GitHub request exhausted retries")


def parse_pr(value: str) -> tuple[str, int]:
    m = re.fullmatch(r"(?:https://github\.com/)?([\w.-]+/[\w.-]+)(?:/pull/|#)([0-9]+)(?:/(?:files|changes))?/?", value)
    if not m: raise ValueError("Use owner/repo#123 or https://github.com/owner/repo/pull/123")
    return m.group(1), int(m.group(2))


def from_github(value: str, *, token_env: str = "GITHUB_TOKEN", max_files: int = MAX_FILES) -> dict:
    repo, number = parse_pr(value); api = GitHubClient(os.getenv(token_env))
    pr = api.get(f"/repos/{repo}/pulls/{number}")
    # GitHub PR changes are relative to merge-base, not necessarily the current base tip.
    base_tip, head = pr["base"]["sha"], pr["head"]["sha"]
    compare = api.get(f"/repos/{repo}/compare/{base_tip}...{head}?per_page=1")
    base = compare["merge_base_commit"]["sha"]
    files, page = [], 1
    while True:
        batch = api.get(f"/repos/{repo}/pulls/{number}/files?per_page=100&page={page}")
        if not isinstance(batch, list): raise ValueError("Unexpected GitHub file-list response")
        files.extend(batch)
        if len(files) > max_files: raise ValueError(f"Changed file count exceeds --max-files {max_files}; no partial report was written")
        if len(batch) < 100: break
        page += 1
    if len(files) != pr.get("changed_files", len(files)):
        raise ValueError("GitHub did not return the complete file list; refusing an apparently complete report")
    warnings = []; tasks = []
    head_repo = (pr.get("head", {}).get("repo") or {}).get("full_name") or repo
    for f in files:
        old = f.get("previous_filename", f["filename"])
        if f["status"] != "added": tasks.append((repo, old, "base", base, f["filename"]))
        if f["status"] != "removed": tasks.append((head_repo, f["filename"], "head", head, f["filename"]))

    def read(task):
        owner, path, side, sha, region = task
        obj = api.get(f"/repos/{owner}/contents/{quote(path, safe='/')}?ref={sha}")
        if not isinstance(obj, dict) or obj.get("type") != "file": return None, f"Skipped {side} {path}: not a regular file"
        if obj.get("size", 0) > MAX_FILE: return None, f"Skipped {side} {path}: exceeds {MAX_FILE} bytes"
        if obj.get("encoding") == "base64": data = base64.b64decode(obj["content"])
        else:
            blob = api.get(f"/repos/{owner}/git/blobs/{obj['sha']}")
            if blob.get("encoding") != "base64": return None, f"Skipped {side} {path}: unsupported blob encoding"
            data = base64.b64decode(blob["content"])
        if len(data) > MAX_FILE: return None, f"Skipped {side} {path}: exceeds {MAX_FILE} bytes"
        try:
            if b"\0" in data: raise UnicodeError("NUL byte")
            text = data.decode("utf-8")
        except UnicodeError: return None, f"Skipped {side} {path}: binary or non-UTF-8 source"
        return {"path": path, "side": side, "text": text, "start_line": 1, "scope": "full", "region": region}, None
    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(read, tasks))
    fragments = []
    for f, warning in results:
        if f: fragments.append(f)
        if warning: warnings.append(warning)
    # Fail if the PR changed during the read; don't silently mix multiple revisions.
    end = api.get(f"/repos/{repo}/pulls/{number}")
    if end["head"]["sha"] != head or end["base"]["sha"] != base_tip:
        raise ValueError("PR revisions changed while fetching. Retry to capture a consistent snapshot.")
    meta = {"title": pr["title"], "repository": repo, "number": number, "url": pr["html_url"],
            "base_sha": base, "requested_base_sha": base_tip, "head_sha": head, "head_repository": head_repo, "scope": "changed files", "input": "GitHub REST",
            "comparison": "merge-base to head", "changed_files": len(files), "additions": pr.get("additions"), "deletions": pr.get("deletions"),
            "description": pr.get("body") or "", "draft": pr.get("draft", False), "state": pr.get("state"),
            "captured_at": datetime.now(timezone.utc).isoformat(), "validation": "PR description is author-reported evidence. No test run was executed or verified by this compiler."}
    return {"schema": "diffstory.snapshot.v1", "meta": meta, "fragments": fragments, "warnings": warnings}
