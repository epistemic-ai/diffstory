"""Actual local-Git integration and mocked GitHub transport contracts."""
import base64
import copy
import io
import json
import subprocess
import tempfile
import unittest
from contextlib import redirect_stdout, redirect_stderr
from pathlib import Path
from unittest.mock import patch

from diffstory.analysis import compile_snapshot, apply_annotations
from diffstory.cli import main
from diffstory.ingest import from_git, from_github


class GitIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory()
        self.root=Path(self.tmp.name)
        self.git('init','-q')
        self.git('config','user.name','Prototype Test')
        self.git('config','user.email','test@example.invalid')
        (self.root/'old.py').write_text('def parse(x):\n    return x.strip()\n')
        (self.root/'caller.py').write_text('from old import parse\ndef run(x):\n    return parse(x)\n')
        self.git('add','.');self.git('commit','-qm','base')
        self.base=self.git('rev-parse','HEAD').strip()
        (self.root/'new.py').write_text((self.root/'old.py').read_text())
        (self.root/'old.py').unlink()
        (self.root/'caller.py').write_text('from new import parse\ndef run(x):\n    return parse(x)\n')
        self.git('add','.');self.git('commit','-qm','extract parser')
        self.head=self.git('rev-parse','HEAD').strip()

    def tearDown(self): self.tmp.cleanup()
    def git(self,*args):
        return subprocess.check_output(['git','-C',str(self.root),*args],stderr=subprocess.STDOUT,text=True)

    def test_git_move_and_dirty_worktree_ignored(self):
        (self.root/'new.py').write_text('THIS IS AN UNCOMMITTED CHANGE\n')
        s=from_git(str(self.root),self.base,self.head)
        self.assertEqual(s['meta']['base_sha'],self.base)
        self.assertEqual(s['meta']['head_sha'],self.head)
        self.assertTrue(any(f['path']=='new.py' and 'return x.strip()' in f['text'] for f in s['fragments']))
        self.assertEqual(compile_snapshot(s)['stats']['identical_ast_moves'],1)

    def test_cli_roundtrip(self):
        out=self.root/'reader.html';snapshot=self.root/'snapshot.json'
        with redirect_stdout(io.StringIO()):
            code=main(['git','--repo',str(self.root),'--base',self.base,'--head',self.head,'--out',str(out),'--save-snapshot',str(snapshot)])
        self.assertEqual(code,0);self.assertIn('report-data',out.read_text())
        report=out.with_suffix('.report.json')
        with redirect_stdout(io.StringIO()):
            code=main(['evidence',str(report),'--out',str(self.root/'evidence.json')])
        self.assertEqual(code,0)
        e=json.loads((self.root/'evidence.json').read_text())
        self.assertEqual(e['base_sha'],self.base)
        self.assertEqual(e['head_sha'],self.head)

    def test_empty_comparison(self):
        s=from_git(str(self.root),self.head,self.head)
        r=compile_snapshot(s)
        self.assertEqual(r['stats']['units'],0)

    def test_file_limit_refuses_partial(self):
        with self.assertRaisesRegex(ValueError,'exceeds'):from_git(str(self.root),self.base,self.head,max_files=1)

    def test_bad_revision_error_is_readable(self):
        with redirect_stderr(io.StringIO()) as err:
            code=main(['git','--repo',str(self.root),'--base','NOT_A_REF','--out',str(self.root/'x.html')])
        self.assertEqual(code,2);self.assertIn('diffstory:',err.getvalue())

    def test_skipped_side_does_not_claim_removal(self):
        (self.root/'new.py').write_bytes(b'\x00binary')
        self.git('add','.');self.git('commit','-qm','binary')
        s=from_git(str(self.root),self.head,'HEAD')
        r=compile_snapshot(s)
        self.assertTrue(r['warnings'])
        self.assertTrue(any(c['kind']=='observed_base' for c in r['changes']))
        self.assertFalse(any(c['kind']=='removed' for c in r['changes']))

    def test_wrong_base_annotations_rejected(self):
        r=compile_snapshot(from_git(str(self.root),self.base,self.head))
        with self.assertRaisesRegex(ValueError,'different base'):
            apply_annotations(r,{'schema':'diffstory.annotations.v1','base_sha':'wrong','head_sha':self.head,'steps':[]})


class GitHubTransportTests(unittest.TestCase):
    """These verify transport behavior against fake responses, not a live API."""
    def fake(self,n=1,changed_mid_read=False,returned=None):
        pr={'base':{'sha':'a'*40},'head':{'sha':'b'*40,'repo':{'full_name':'org/repo'}},'changed_files':n,
            'title':'Example','html_url':'https://github.com/org/repo/pull/1','body':'No tests run.'}
        count=0;paths=[]
        def get(_client,path,**kwargs):
            nonlocal count
            paths.append(path)
            if path.endswith('/pulls/1'):
                count+=1;p=copy.deepcopy(pr)
                if changed_mid_read and count>1:p['head']['sha']='c'*40
                return p
            if '/compare/' in path:return {'merge_base_commit':{'sha':'d'*40}}
            if '/files?' in path:
                page=int(path.rsplit('page=',1)[-1]);amount=n if returned is None else returned
                return [{'filename':f'f{i}.py','status':'added'} for i in range((page-1)*100,min(page*100,amount))]
            if '/contents/' in path:
                return {'type':'file','size':4,'encoding':'base64','content':base64.b64encode(b'x=1\n').decode()}
            raise AssertionError(path)
        return get,paths

    def test_uses_merge_base_not_tip(self):
        fake,paths=self.fake()
        with patch('diffstory.ingest.GitHubClient.get',new=fake):s=from_github('org/repo#1')
        self.assertEqual(s['meta']['base_sha'],'d'*40)
        self.assertEqual(s['meta']['requested_base_sha'],'a'*40)

    def test_fork_metadata_and_fetch_paths(self):
        fake, paths = self.fake()
        def fork(client, path, **kwargs):
            data = fake(client, path, **kwargs)
            if path.endswith('/pulls/1'): data['head']['repo']['full_name'] = 'contributor/repo'
            return data
        with patch('diffstory.ingest.GitHubClient.get', new=fork):
            snapshot = from_github('org/repo#1')
        self.assertEqual(snapshot['meta']['head_repository'], 'contributor/repo')
        self.assertTrue(any(p.startswith('/repos/contributor/repo/contents/') for p in paths))

    def test_paginates_all_changed_files(self):
        fake,paths=self.fake(101)
        with patch('diffstory.ingest.GitHubClient.get',new=fake):s=from_github('org/repo#1')
        self.assertEqual(len(s['fragments']),101)
        self.assertTrue(any('per_page=100&page=2' in p for p in paths))

    def test_incomplete_file_list_rejected(self):
        fake,_=self.fake(2,returned=1)
        with patch('diffstory.ingest.GitHubClient.get',new=fake),self.assertRaisesRegex(ValueError,'complete file list'):
            from_github('org/repo#1')

    def test_changing_revision_rejected(self):
        fake,_=self.fake(changed_mid_read=True)
        with patch('diffstory.ingest.GitHubClient.get',new=fake),self.assertRaisesRegex(ValueError,'changed while fetching'):
            from_github('org/repo#1')

if __name__=='__main__':unittest.main()
