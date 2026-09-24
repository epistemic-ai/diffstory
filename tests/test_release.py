"""Release controls and read-only transport contracts. No live GitHub writes."""
from __future__ import annotations

import contextlib
import importlib.util
import io
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch
from urllib.error import HTTPError
from urllib.request import Request

from diffstory import __version__
from diffstory.analysis import source_url
from diffstory.cli import main as cli_main
from diffstory.ingest import GitHubClient, RejectRedirects

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('release_checker', ROOT / 'scripts/check_release.py')
checker = importlib.util.module_from_spec(spec)
spec.loader.exec_module(checker)


class TransportHardeningTests(unittest.TestCase):
    def test_redirect_rejected_without_token_or_location_in_error(self):
        request = Request('https://api.github.com/repos/example/catalog',
                          headers={'Authorization': 'Bearer never-print-this-value'})
        for url in ('https://other.invalid/path', 'https://api.github.com/repos/new/name'):
            with self.assertRaises(ValueError) as error:
                RejectRedirects().redirect_request(request, None, 302, 'Found', {}, url)
            self.assertNotIn('never-print-this-value', str(error.exception))
            self.assertNotIn(url, str(error.exception))

    def test_auth_stays_in_request_not_error_message(self):
        opener = MagicMock()
        opener.open.side_effect = HTTPError('https://api.github.com/repos/a/b', 403, 'Forbidden', {}, None)
        with patch('diffstory.ingest.build_opener', return_value=opener), self.assertRaises(ValueError) as error:
            GitHubClient('never-print-this-value').get('/repos/a/b')
        request = opener.open.call_args.args[0]
        self.assertEqual(request.get_header('Authorization'), 'Bearer never-print-this-value')
        self.assertNotIn('never-print-this-value', str(error.exception))

    def test_client_installs_no_redirect_handler(self):
        opener = MagicMock()
        opener.open.return_value.__enter__.return_value.read.return_value = b'{"ok": true}'
        with patch('diffstory.ingest.build_opener', return_value=opener) as factory:
            self.assertEqual(GitHubClient().get('/repos/example/catalog'), {'ok': True})
        self.assertIsInstance(factory.call_args.args[0], RejectRedirects)
        self.assertEqual(opener.open.call_args.kwargs['timeout'], 40)

    def test_rejects_non_repository_endpoints(self):
        with self.assertRaises(ValueError):
            GitHubClient().get('/user')

    def test_fork_source_links_use_correct_owner_on_each_side(self):
        meta = {'repository': 'example/catalog', 'head_repository': 'contributor/catalog',
                'base_sha': 'a'*40, 'head_sha': 'b'*40}
        self.assertIn('/example/catalog/', source_url(meta, 'one.py', 'base', 1, 4))
        self.assertIn('/contributor/catalog/', source_url(meta, 'two.py', 'head', 1, 4))
        meta.pop('head_repository')
        self.assertIn('/example/catalog/', source_url(meta, 'two.py', 'head', 1, 4))

    def test_invalid_fork_owner_does_not_produce_link(self):
        meta = {'repository': 'example/catalog', 'head_repository': 'https://untrusted.invalid',
                'head_sha': 'b'*40}
        self.assertIsNone(source_url(meta, 'two.py', 'head', 1, 4))

    def test_version_flag(self):
        with contextlib.redirect_stdout(io.StringIO()) as output:
            with self.assertRaises(SystemExit) as status:
                cli_main(['--version'])
        self.assertEqual(status.exception.code, 0)
        self.assertIn(__version__, output.getvalue())


class PublicationTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name) / 'release'
        self.root.mkdir()
        (self.root / 'examples').mkdir()
        (self.root / 'README.md').write_text('Public test fixture.\n')
        for name in ('demo.snapshot.json', 'demo.report.json'):
            (self.root / 'examples' / name).write_text(json.dumps({'meta': {'input': 'synthetic example'}}))

    def tearDown(self):
        self.tmp.cleanup()

    def test_manifest_roundtrip(self):
        count = checker.check(self.root, write=True)
        self.assertEqual(checker.check(self.root, verify=True), count)

    def test_modified_file_invalidates_manifest(self):
        checker.check(self.root, write=True)
        (self.root / 'README.md').write_text('Changed after sealing.')
        with self.assertRaisesRegex(ValueError, 'manifest'):
            checker.check(self.root, verify=True)

    def test_extra_candidate_invalidates_manifest(self):
        checker.check(self.root, write=True)
        (self.root / 'LICENSE').write_text('Some license')
        with self.assertRaisesRegex(ValueError, 'manifest'):
            checker.check(self.root, verify=True)

    def test_private_marker_is_rejected(self):
        (self.root / 'README.md').write_text('epistemic-ai/' + 'foundation')
        with self.assertRaisesRegex(ValueError, 'Private prototype marker'):
            checker.check(self.root)

    def test_secret_is_rejected_without_printing_value(self):
        value = 'ghp_' + 'Z'*36
        (self.root / 'README.md').write_text(value)
        with self.assertRaises(ValueError) as error:
            checker.check(self.root)
        self.assertNotIn(value, str(error.exception))

    def test_symlink_is_rejected(self):
        (self.root / 'LICENSE').symlink_to(self.root / 'README.md')
        with self.assertRaisesRegex(ValueError, 'Symlinks'):
            checker.check(self.root)

    def test_unapproved_report_is_rejected(self):
        (self.root / 'examples' / 'customer.report.json').write_text('{}')
        with self.assertRaisesRegex(ValueError, 'Unreviewed report'):
            checker.check(self.root)

    def test_demo_cannot_silently_become_real_pr(self):
        (self.root / 'examples' / 'demo.report.json').write_text(json.dumps({'meta': {'input': 'GitHub REST'}}))
        with self.assertRaisesRegex(ValueError, 'explicitly synthetic'):
            checker.check(self.root)

    def test_fonts_are_not_published(self):
        (self.root / 'examples' / 'custom.woff2').write_bytes(b'no fonts')
        with self.assertRaisesRegex(ValueError, 'Unsupported public file type'):
            checker.check(self.root)

    def test_environment_file_is_rejected(self):
        (self.root / '.env').write_text('example=not-a-secret')
        with self.assertRaisesRegex(ValueError, 'Credential-like file'):
            checker.check(self.root)

    def test_generated_distributions_are_not_manifest_inputs(self):
        (self.root / 'dist').mkdir()
        (self.root / 'dist' / 'local.whl').write_bytes(b'not a real wheel')
        checker.check(self.root, write=True)
        manifest = json.loads((self.root / 'PUBLICATION.json').read_text())
        self.assertFalse(any(f['path'].startswith('dist/') for f in manifest['files']))

    @unittest.skipUnless(shutil.which('git') and shutil.which('bash'), 'Git and bash required')
    def test_publication_helper_with_fake_github_and_real_local_git(self):
        """Exercise create/push/verify without any GitHub request or organization write."""
        (self.root / 'scripts').mkdir()
        for name in ('check_release.py', 'publish.sh'):
            shutil.copyfile(ROOT / 'scripts' / name, self.root / 'scripts' / name)
        (self.root / 'tests').mkdir()
        (self.root / 'tests' / 'test_smoke.py').write_text('import unittest\nclass T(unittest.TestCase):\n    def test_ok(self): self.assertTrue(True)\n')
        checker.check(self.root, write=True)
        bindir = Path(self.tmp.name) / 'bin'; bindir.mkdir()
        remote = Path(self.tmp.name) / 'remote.git'
        log = Path(self.tmp.name) / 'calls.jsonl'
        fake = bindir / 'gh'
        fake.write_text('#!' + sys.executable + '\n' + '''import json, os, subprocess, sys
from pathlib import Path
args = sys.argv[1:]
with open(os.environ['FAKE_LOG'], 'a') as log: log.write(json.dumps(args) + '\\n')
remote = os.environ['FAKE_REMOTE']
if args[:2] == ['auth', 'status']: pass
elif args[:2] == ['repo', 'view']:
    field = args[args.index('--json') + 1]
    if field == 'nameWithOwner': raise SystemExit(1)
    if field == 'isPrivate': print('false')
    elif field == 'url': print('https://github.com/epistemic-ai/diffstory')
elif args[:2] == ['api', 'user']: print('Publisher Test\\t1001\\tpublisher-test')
elif args[:2] == ['repo', 'create']:
    assert args[2] == 'epistemic-ai/diffstory' and '--public' in args and '--push' in args
    root = next(x.split('=', 1)[1] for x in args if x.startswith('--source='))
    subprocess.run(['git', 'init', '--bare', remote], check=True, capture_output=True)
    subprocess.run(['git', '-C', root, 'remote', 'add', 'origin', remote], check=True)
    subprocess.run(['git', '-C', root, 'push', '-u', 'origin', 'main'], check=True, capture_output=True)
elif args[0] == 'api' and args[1].endswith('/git/ref/heads/main'):
    print(subprocess.check_output(['git', '--git-dir', remote, 'rev-parse', 'main'], text=True).strip())
else: raise RuntimeError('Unexpected fake gh call')
''')
        fake.chmod(0o755)
        env = {**os.environ, 'PATH': str(bindir) + os.pathsep + os.environ['PATH'],
               'FAKE_REMOTE': str(remote), 'FAKE_LOG': str(log), 'GIT_CONFIG_GLOBAL': os.devnull,
               'GIT_CONFIG_NOSYSTEM': '1', 'GIT_TERMINAL_PROMPT': '0'}
        for key in ('GIT_DIR', 'GIT_WORK_TREE', 'GIT_INDEX_FILE'):
            env.pop(key, None)
        result = subprocess.run(['bash', str(self.root / 'scripts/publish.sh')], env=env,
                                capture_output=True, text=True, timeout=30)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn('Published and verified', result.stdout)
        files = subprocess.check_output(['git', '-C', str(self.root), 'ls-files'], text=True).splitlines()
        manifest = json.loads((self.root / 'PUBLICATION.json').read_text())
        self.assertEqual(set(files), {f['path'] for f in manifest['files']} | {'PUBLICATION.json'})
        calls = [json.loads(line) for line in log.read_text().splitlines()]
        self.assertEqual(sum(a[:2] == ['repo', 'create'] for a in calls), 1)
        # A second invocation refuses the newly initialized checkout before any write.
        again = subprocess.run(['bash', str(self.root / 'scripts/publish.sh')], env=env,
                               capture_output=True, text=True, timeout=10)
        self.assertNotEqual(again.returncode, 0)
        self.assertIn('existing Git history', again.stderr)


if __name__ == '__main__':
    unittest.main()
