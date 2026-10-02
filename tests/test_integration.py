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
from contextlib import redirect_stderr
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import Mock
from unittest.mock import patch

from diffstory.analysis import MAX_SNAPSHOT_SOURCE_BYTES
from diffstory.analysis import apply_annotations
from diffstory.analysis import compile_snapshot
from diffstory.cli import main
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

    def fake(
        self,
        n: int = 1,
        *,
        changed_mid_read: bool = False,
        returned: int | None = None,
    ) -> tuple[Callable[..., object], list[str]]:
        """
        Build a deterministic fake GitHub API response handler.

        Args:
            n: Number of changed files advertised by the pull request.
            changed_mid_read: Change the head revision on a repeated PR read when true.
            returned: Optional number of file records actually returned.

        Returns:
            A fake ``get`` callable and the list of requested API paths.

        """
        pr = {
            "base": {"sha": "a" * 40},
            "head": {"sha": "b" * 40, "repo": {"full_name": "org/repo"}},
            "changed_files": n,
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
                amount = n if returned is None else returned
                return [
                    {"filename": f"f{i}.py", "status": "added"}
                    for i in range((page - 1) * 100, min(page * 100, amount))
                ]
            if "/contents/" in path:
                return {
                    "type": "file",
                    "size": 4,
                    "encoding": "base64",
                    "content": base64.b64encode(b"x=1\n").decode(),
                }
            raise AssertionError(path)

        return get, paths

    def test_uses_merge_base_not_tip(self) -> None:
        """Use the pull request merge base rather than the current base tip."""
        fake, paths = self.fake()
        with patch("diffstory.ingest.GitHubClient.get", new=fake):
            s = from_github("org/repo#1")
        self.assertEqual(s["meta"]["base_sha"], "d" * 40)
        self.assertEqual(s["meta"]["requested_base_sha"], "a" * 40)

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
