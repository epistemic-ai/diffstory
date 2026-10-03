"""Actual local-Git integration and mocked GitHub transport contracts."""

import base64
import copy
import io
import json
import shutil
import subprocess
import tempfile
import unittest
from collections.abc import Callable
from collections.abc import Sequence
from contextlib import redirect_stderr
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import Mock
from unittest.mock import patch

from diffstory.analysis import MAX_SNAPSHOT_SOURCE_BYTES
from diffstory.analysis import apply_annotations
from diffstory.analysis import compile_snapshot
from diffstory.cli import main
from diffstory.ingest import MAX_FILE
from diffstory.ingest import _github_token
from diffstory.ingest import from_git
from diffstory.ingest import from_github


class GitIntegrationTests(unittest.TestCase):
    """Exercise repository ingestion and CLI output against temporary Git."""

    def setUp(self) -> None:
        """
        Create a temporary Git repository with a committed file move.

        Side Effects:
            Initializes Git history and stores temporary repository paths on the test.
        """
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.git("init", "-q")
        self.git("config", "user.name", "Prototype Test")
        self.git("config", "user.email", "test@example.invalid")
        (self.root / "old.py").write_text(
            "def parse(x):\n    return x.strip()\n"
        )
        (self.root / "caller.py").write_text(
            "from old import parse\ndef run(x):\n    return parse(x)\n"
        )
        self.git("add", ".")
        self.git("commit", "-qm", "base")
        self.base = self.git("rev-parse", "HEAD").strip()
        (self.root / "new.py").write_text((self.root / "old.py").read_text())
        (self.root / "old.py").unlink()
        (self.root / "caller.py").write_text(
            "from new import parse\ndef run(x):\n    return parse(x)\n"
        )
        self.git("add", ".")
        self.git("commit", "-qm", "extract parser")
        self.head = self.git("rev-parse", "HEAD").strip()

    def tearDown(self) -> None:
        """Remove the temporary Git repository created by ``setUp``."""
        self.tmp.cleanup()

    def git(self, *args: str) -> str:
        """
        Run Git in the test repository and return decoded output.

        Args:
            *args: Git subcommand and arguments.

        Returns:
            Standard output as text.

        Raises:
            subprocess.CalledProcessError: If the Git command fails.

        """
        return subprocess.check_output(
            ["git", "-C", str(self.root), *args],
            stderr=subprocess.STDOUT,
            text=True,
        )

    def test_git_move_and_dirty_worktree_ignored(self) -> None:
        """Read committed revisions while excluding uncommitted worktree edits."""
        (self.root / "new.py").write_text("THIS IS AN UNCOMMITTED CHANGE\n")
        s = from_git(str(self.root), self.base, self.head)
        self.assertEqual(s["meta"]["base_sha"], self.base)
        self.assertEqual(s["meta"]["head_sha"], self.head)
        self.assertTrue(
            any(
                f["path"] == "new.py" and "return x.strip()" in f["text"]
                for f in s["fragments"]
            )
        )
        self.assertEqual(
            compile_snapshot(s)["stats"]["identical_ast_moves"], 1
        )
        evidence = {
            record["base"]["path"]: record for record in s["file_evidence"]
        }
        self.assertEqual(evidence["new.py"]["base"]["state"], "absent")
        self.assertEqual(
            evidence["new.py"]["head"],
            {"path": "new.py", "state": "supplied", "coverage": "full"},
        )
        self.assertEqual(evidence["old.py"]["head"]["state"], "absent")
        self.assertEqual(evidence["caller.py"]["base"]["coverage"], "full")

    def test_cli_roundtrip(self) -> None:
        """Write HTML and snapshot outputs, then export revision-bound evidence."""
        out = self.root / "reader.html"
        snapshot = self.root / "snapshot.json"
        with redirect_stdout(io.StringIO()):
            code = main(
                [
                    "git",
                    "--repo",
                    str(self.root),
                    "--base",
                    self.base,
                    "--head",
                    self.head,
                    "--out",
                    str(out),
                    "--save-snapshot",
                    str(snapshot),
                ]
            )
        self.assertEqual(code, 0)
        self.assertIn("report-data", out.read_text())
        report = out.with_suffix(".report.json")
        with redirect_stdout(io.StringIO()):
            code = main(
                [
                    "evidence",
                    str(report),
                    "--out",
                    str(self.root / "evidence.json"),
                ]
            )
        self.assertEqual(code, 0)
        e = json.loads((self.root / "evidence.json").read_text())
        self.assertEqual(e["base_sha"], self.base)
        self.assertEqual(e["head_sha"], self.head)

    def test_empty_comparison(self) -> None:
        """Compile equal Git revisions without reporting semantic changes."""
        s = from_git(str(self.root), self.head, self.head)
        r = compile_snapshot(s)
        self.assertEqual(r["stats"]["units"], 0)

    def test_git_empty_file_is_supplied(self) -> None:
        """A zero-byte Git blob is a full supplied file, not an unavailable read."""
        (self.root / "empty.txt").write_bytes(b"")
        self.git("add", ".")
        self.git("commit", "-qm", "empty source")
        snapshot = from_git(str(self.root), self.head, "HEAD")
        record = next(
            item
            for item in snapshot["file_evidence"]
            if item["head"]["path"] == "empty.txt"
        )
        self.assertEqual(record["base"]["state"], "absent")
        self.assertEqual(
            record["head"],
            {"path": "empty.txt", "state": "supplied", "coverage": "full"},
        )
        self.assertEqual(
            next(f for f in snapshot["fragments"] if f["path"] == "empty.txt")[
                "text"
            ],
            "",
        )

    def test_git_skip_reason_is_structured(self) -> None:
        """Git skip branches must set the reason at the branch that detects them."""
        rows = [
            ("binary", "binary.py", b"\0binary", "binary_or_non_utf8"),
            ("size limit", "large.py", b"x" * (MAX_FILE + 1), "size_limit"),
            ("unsupported object", "linked.py", None, "unsupported_object"),
        ]
        for name, path, content, expected_reason in rows:
            with self.subTest(case=name):
                target = self.root / path
                if content is None:
                    target.symlink_to("missing-target.py")
                else:
                    target.write_bytes(content)
                self.git("add", path)
                self.git("commit", "-qm", f"add {name}")
                snapshot = from_git(str(self.root), self.head, "HEAD")
                record = next(
                    item
                    for item in snapshot["file_evidence"]
                    if item["head"]["path"] == path
                )
                self.assertEqual(record["head"]["state"], "unavailable")
                self.assertEqual(record["head"]["reason"], expected_reason)

    def test_file_limit_refuses_partial(self) -> None:
        """Reject a changed-file count above the configured limit."""
        with self.assertRaisesRegex(ValueError, "exceeds"):
            from_git(str(self.root), self.base, self.head, max_files=1)

    def test_bad_revision_error_is_readable(self) -> None:
        """Return a readable CLI error for an unresolved Git revision."""
        with redirect_stderr(io.StringIO()) as err:
            code = main(
                [
                    "git",
                    "--repo",
                    str(self.root),
                    "--base",
                    "NOT_A_REF",
                    "--out",
                    str(self.root / "x.html"),
                ]
            )
        self.assertEqual(code, 2)
        self.assertIn("diffstory:", err.getvalue())

    def test_skipped_side_does_not_claim_removal(self) -> None:
        """Treat a skipped binary source side as unresolved evidence instead of a proven removal."""
        (self.root / "new.py").write_bytes(b"\x00binary")
        self.git("add", ".")
        self.git("commit", "-qm", "binary")
        s = from_git(str(self.root), self.head, "HEAD")
        r = compile_snapshot(s)
        record = next(
            item
            for item in s["file_evidence"]
            if item["head"]["path"] == "new.py"
        )
        self.assertEqual(record["head"]["state"], "unavailable")
        self.assertEqual(record["head"]["reason"], "binary_or_non_utf8")
        self.assertTrue(r["warnings"])
        self.assertTrue(
            any(c["kind"] == "observed_base" for c in r["changes"])
        )
        self.assertFalse(any(c["kind"] == "removed" for c in r["changes"]))

    def test_wrong_base_annotations_rejected(self) -> None:
        """Reject annotations tied to a different base revision."""
        r = compile_snapshot(from_git(str(self.root), self.base, self.head))
        with self.assertRaisesRegex(ValueError, "different base"):
            apply_annotations(
                r,
                {
                    "schema": "diffstory.annotations.v1",
                    "base_sha": "wrong",
                    "head_sha": self.head,
                    "steps": [],
                },
            )


class GitHubAuthTests(unittest.TestCase):
    """Verify environment and saved ``gh`` authentication selection."""

    @patch.dict("os.environ", {"GITHUB_TOKEN": "explicit"}, clear=False)
    @patch("diffstory.ingest.subprocess.run")
    def test_environment_token_takes_precedence(self, run: Mock) -> None:
        """Prefer the configured token without calling gh."""
        self.assertEqual(_github_token("GITHUB_TOKEN"), "explicit")
        run.assert_not_called()

    @patch.dict("os.environ", {}, clear=True)
    @patch("diffstory.ingest.subprocess.run")
    def test_uses_authenticated_gh_token(self, run: Mock) -> None:
        """Use the saved GitHub CLI token when the environment is empty."""
        run.return_value = subprocess.CompletedProcess(
            ["gh", "auth", "token"], 0, stdout="from-gh\n"
        )
        self.assertEqual(_github_token("GITHUB_TOKEN"), "from-gh")
        run.assert_called_once_with(
            [shutil.which("gh") or "gh", "auth", "token"],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            timeout=10,
            text=True,
            check=False,
        )

    @patch.dict("os.environ", {}, clear=True)
    @patch("diffstory.ingest.subprocess.run", side_effect=FileNotFoundError)
    def test_missing_gh_falls_back_to_unauthenticated(
        self,
        _run: Mock,
    ) -> None:
        """Continue unauthenticated when gh is not installed."""
        self.assertIsNone(_github_token("GITHUB_TOKEN"))

    @patch.dict("os.environ", {}, clear=True)
    @patch("diffstory.ingest.subprocess.run")
    def test_unauthenticated_gh_falls_back_to_unauthenticated(
        self,
        run: Mock,
    ) -> None:
        """Continue unauthenticated when gh has no saved sign-in."""
        run.return_value = subprocess.CompletedProcess(
            ["gh", "auth", "token"], 1, stdout=""
        )
        self.assertIsNone(_github_token("GITHUB_TOKEN"))


class GitHubTransportTests(unittest.TestCase):
    """These verify transport behavior against fake responses, not a live API."""

    def fake(  # noqa: PLR0913  # Each API fixture endpoint has an explicit input.
        self,
        n: int = 1,
        *,
        changed_mid_read: bool = False,
        returned: int | None = None,
        file_records: Sequence[dict] | None = None,
        content_response: dict | None = None,
        blob_response: dict | None = None,
    ) -> tuple[Callable[..., object], list[str]]:
        """
        Build a deterministic fake GitHub API response handler.

        Args:
            n: Number of changed files advertised by the pull request.
            changed_mid_read: Change the head revision on a repeated PR read when true.
            returned: Optional number of file records actually returned.
            file_records: Optional exact changed-file records to return.
            content_response: Optional contents endpoint response for each path.
            blob_response: Optional blob endpoint response.

        Returns:
            A fake ``get`` callable and the list of requested API paths.

        """
        pr = {
            "base": {"sha": "a" * 40},
            "head": {"sha": "b" * 40, "repo": {"full_name": "org/repo"}},
            "changed_files": n if file_records is None else len(file_records),
            "title": "Example",
            "html_url": "https://github.com/org/repo/pull/1",
            "body": "No tests run.",
        }
        count = 0
        paths = []

        def get(_client: object, path: str, **_kwargs: object) -> object:
            """
            Return one deterministic response for a requested fake API path.

            Args:
                _client: Ignored client receiver used by the patched method.
                path: GitHub REST path to serve.
                **_kwargs: Ignored optional request arguments.

            Returns:
                Fixture payload matching the endpoint path.

            Raises:
                AssertionError: If the test requests an unconfigured endpoint.

            """
            nonlocal count
            paths.append(path)
            if path.endswith("/pulls/1"):
                count += 1
                p = copy.deepcopy(pr)
                if changed_mid_read and count > 1:
                    p["head"]["sha"] = "c" * 40
                return p
            if "/compare/" in path:
                return {"merge_base_commit": {"sha": "d" * 40}}
            if "/files?" in path:
                page = int(path.rsplit("page=", 1)[-1])
                records = (
                    file_records
                    if file_records is not None
                    else [
                        {"filename": f"f{i}.py", "status": "added"}
                        for i in range(n)
                    ]
                )
                amount = len(records) if returned is None else returned
                return list(
                    records[(page - 1) * 100 : min(page * 100, amount)]
                )
            if "/contents/" in path:
                return (
                    copy.deepcopy(content_response)
                    if content_response is not None
                    else {
                        "type": "file",
                        "size": 4,
                        "encoding": "base64",
                        "content": base64.b64encode(b"x=1\n").decode(),
                    }
                )
            if "/git/blobs/" in path:
                return (
                    copy.deepcopy(blob_response)
                    if blob_response is not None
                    else {
                        "encoding": "base64",
                        "content": base64.b64encode(b"x=1\n").decode(),
                    }
                )
            raise AssertionError(path)

        return get, paths

    def test_uses_merge_base_not_tip(self) -> None:
        """Use the pull request merge base rather than the current base tip."""
        fake, paths = self.fake()
        with patch("diffstory.ingest.GitHubClient.get", new=fake):
            s = from_github("org/repo#1")
        self.assertEqual(s["meta"]["base_sha"], "d" * 40)
        self.assertEqual(s["meta"]["requested_base_sha"], "a" * 40)

    def test_ingest_maps_github_file_status_to_evidence(self) -> None:
        """Each file status must keep its base and head evidence paths."""
        rows = [
            (
                "added",
                {"filename": "new.py", "status": "added"},
                {"base": ("new.py", "absent"), "head": ("new.py", "supplied")},
            ),
            (
                "removed",
                {"filename": "old.py", "status": "removed"},
                {"base": ("old.py", "supplied"), "head": ("old.py", "absent")},
            ),
            (
                "renamed",
                {
                    "filename": "new.py",
                    "previous_filename": "old.py",
                    "status": "renamed",
                },
                {
                    "base": ("old.py", "supplied"),
                    "head": ("new.py", "supplied"),
                },
            ),
            (
                "copied",
                {
                    "filename": "copy.py",
                    "previous_filename": "original.py",
                    "status": "copied",
                },
                {
                    "base": ("original.py", "supplied"),
                    "head": ("copy.py", "supplied"),
                },
            ),
        ]
        for name, file_info, expected in rows:
            with self.subTest(status=name):
                fake, paths = self.fake(file_records=(file_info,))
                with patch("diffstory.ingest.GitHubClient.get", new=fake):
                    snapshot = from_github("org/repo#1")
                record = snapshot["file_evidence"][0]
                for side_name, (path, state) in expected.items():
                    self.assertEqual(record[side_name]["path"], path)
                    self.assertEqual(record[side_name]["state"], state)
                    if state == "supplied":
                        self.assertEqual(record[side_name]["coverage"], "full")
                if name in {"renamed", "copied"}:
                    self.assertTrue(
                        any(
                            "/contents/" + expected["base"][0] in p
                            for p in paths
                        )
                    )
                    self.assertTrue(
                        any(
                            "/contents/" + expected["head"][0] in p
                            for p in paths
                        )
                    )

    def test_github_rename_requires_previous_path(self) -> None:
        """A copy or rename without its old path must fail without guessing."""
        for status in ("renamed", "copied"):
            with self.subTest(status=status):
                fake, _ = self.fake(
                    file_records=({"filename": "new.py", "status": status},),
                )
                with (
                    patch("diffstory.ingest.GitHubClient.get", new=fake),
                    self.assertRaisesRegex(ValueError, "previous_filename"),
                ):
                    from_github("org/repo#1")

    def test_github_skip_reason_is_structured(self) -> None:
        """Each nonfatal GitHub read skip must keep its reason separate from warning text."""
        rows = [
            (
                "unsupported object",
                {"type": "symlink", "size": 4},
                None,
                "unsupported_object",
            ),
            (
                "invalid size",
                {"type": "file", "size": -1},
                None,
                "invalid_content_size",
            ),
            (
                "size limit",
                {"type": "file", "size": 8_000_001},
                None,
                "size_limit",
            ),
            (
                "binary source",
                {
                    "type": "file",
                    "size": 1,
                    "encoding": "base64",
                    "content": base64.b64encode(b"\0").decode(),
                },
                None,
                "binary_or_non_utf8",
            ),
            (
                "unsupported encoding",
                {
                    "type": "file",
                    "size": 4,
                    "encoding": "utf-8",
                    "sha": "blob",
                },
                {"encoding": "utf-8", "content": ""},
                "unsupported_encoding",
            ),
        ]
        for name, content, blob, expected_reason in rows:
            with self.subTest(reason=name):
                fake, _ = self.fake(
                    file_records=[{"filename": "new.py", "status": "added"}],
                    content_response=content,
                    blob_response=blob,
                )
                with patch("diffstory.ingest.GitHubClient.get", new=fake):
                    snapshot = from_github("org/repo#1")
                side = snapshot["file_evidence"][0]["head"]
                self.assertEqual(side["state"], "unavailable")
                self.assertEqual(side["reason"], expected_reason)

    def test_github_empty_source_is_full(self) -> None:
        """A successfully decoded zero-byte response is supplied full source."""
        content = {
            "type": "file",
            "size": 0,
            "encoding": "base64",
            "content": "",
        }
        fake, _ = self.fake(
            file_records=[{"filename": "empty.py", "status": "added"}],
            content_response=content,
        )
        with patch("diffstory.ingest.GitHubClient.get", new=fake):
            snapshot = from_github("org/repo#1")
        self.assertEqual(
            snapshot["file_evidence"][0]["head"],
            {"path": "empty.py", "state": "supplied", "coverage": "full"},
        )
        self.assertEqual(snapshot["fragments"][0]["text"], "")

    def test_fork_metadata_and_fetch_paths(self) -> None:
        """Read head source and build links using the contributor fork repository."""
        fake, paths = self.fake()

        def fork(client: object, path: str, **kwargs: object) -> object:
            """
            Wrap the fake endpoint and replace the pull request head owner.

            Args:
                client: Patched GitHub client receiver.
                path: REST path requested by ingestion.
                **kwargs: Optional request arguments forwarded to the fake.

            Returns:
                The fake response, with fork metadata changed for the PR endpoint.

            """
            data = fake(client, path, **kwargs)
            if path.endswith("/pulls/1"):
                data["head"]["repo"]["full_name"] = "contributor/repo"
            return data

        with patch("diffstory.ingest.GitHubClient.get", new=fork):
            snapshot = from_github("org/repo#1")
        self.assertEqual(
            snapshot["meta"]["head_repository"], "contributor/repo"
        )
        self.assertTrue(
            any(
                p.startswith("/repos/contributor/repo/contents/")
                for p in paths
            )
        )

    def test_paginates_all_changed_files(self) -> None:
        """Fetch every changed-file page before compiling the snapshot."""
        fake, paths = self.fake(101)
        with patch("diffstory.ingest.GitHubClient.get", new=fake):
            s = from_github("org/repo#1")
        self.assertEqual(len(s["fragments"]), 101)
        self.assertTrue(any("per_page=100&page=2" in p for p in paths))

    def test_incomplete_file_list_rejected(self) -> None:
        """Reject a GitHub response that omits advertised changed files."""
        fake, _ = self.fake(2, returned=1)
        with (
            patch("diffstory.ingest.GitHubClient.get", new=fake),
            self.assertRaisesRegex(ValueError, "complete file list"),
        ):
            from_github("org/repo#1")

    def test_aggregate_source_limit_stops_before_fetching_remaining_files(
        self,
    ) -> None:
        """Stop source retrieval as soon as the aggregate byte limit is exceeded."""
        fake, paths = self.fake(3)
        with (
            patch("diffstory.ingest.GitHubClient.get", new=fake),
            self.assertRaisesRegex(ValueError, "max-source-bytes"),
        ):
            from_github("org/repo#1", max_source_bytes=7)
        content_requests = [path for path in paths if "/contents/" in path]
        self.assertEqual(len(content_requests), 2)

    def test_source_limit_cannot_raise_hard_cap_before_api_access(
        self,
    ) -> None:
        """Reject a source limit above the hard cap before making any GitHub request."""
        with (
            patch(
                "diffstory.ingest.GitHubClient.get",
                side_effect=AssertionError("unexpected GitHub request"),
            ),
            self.assertRaisesRegex(ValueError, "hard limit"),
        ):
            from_github(
                "org/repo#1",
                max_source_bytes=MAX_SNAPSHOT_SOURCE_BYTES + 1,
            )

    def test_changing_revision_rejected(self) -> None:
        """Reject a pull request whose head revision changes during source retrieval."""
        fake, _ = self.fake(changed_mid_read=True)
        with (
            patch("diffstory.ingest.GitHubClient.get", new=fake),
            self.assertRaisesRegex(ValueError, "changed while fetching"),
        ):
            from_github("org/repo#1")


if __name__ == "__main__":
    unittest.main()
