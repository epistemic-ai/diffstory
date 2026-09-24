"""Contracts for the continuous reader's source-bound paragraph annotations."""
import copy
import json
import unittest
from diffstory.analysis import compile_snapshot, apply_annotations, evidence_packet
from diffstory.render import render


def report():
    return compile_snapshot({
        'schema': 'diffstory.snapshot.v1',
        'meta': {'base_sha': 'a' * 40, 'head_sha': 'b' * 40, 'scope': 'full changed files', 'changed_files': 2},
        'fragments': [
            {'path': 'query.py', 'side': 'head', 'start_line': 20, 'text': 'def construct_query(x):\n    value = x + 1\n    return value\n', 'scope': 'complete'},
            {'path': 'parse.py', 'side': 'head', 'start_line': 1, 'text': 'def parse_record(x):\n    return str(x)\n', 'scope': 'complete'},
        ],
    })


def annotations(r):
    g = r['groups'][0]
    return {
        'schema': 'diffstory.annotations.v1', 'base_sha': r['meta']['base_sha'], 'head_sha': r['meta']['head_sha'],
        'document': {'lead': 'Read the code.', 'closing': 'Trace the callers.'},
        'steps': [{'group_id': g['id'], 'evidence_change_ids': g['change_ids'],
                   'transition': 'Now follow the next boundary.',
                   'passages': [{'text': 'Read `this` exact definition.', 'change_ids': [g['change_ids'][0]], 'view': 'definition'}]}]
    }


class LiterateReaderTests(unittest.TestCase):
    def test_passages_roundtrip(self):
        r = report(); a = annotations(r)
        out = apply_annotations(r, a)
        self.assertEqual(out['document'], a['document'])
        self.assertEqual(out['groups'][0]['narrative']['passages'], a['steps'][0]['passages'])
        self.assertEqual(out['changes'], r['changes'])
        self.assertEqual(out['stats'], r['stats'])
        self.assertNotIn('document', r)

    def test_valid_focus_uses_original_lines(self):
        r = report(); a = annotations(r)
        c = next(c for c in r['changes'] if c['id'] == a['steps'][0]['passages'][0]['change_ids'][0])
        s = c['after'] or c['before']
        a['steps'][0]['passages'][0]['focus'] = {'start': s['start'] + 1, 'end': s['end']}
        out = apply_annotations(r, a)
        self.assertIn('reader', render(out).lower())

    def test_focus_out_of_source_rejected(self):
        r = report(); a = annotations(r)
        a['steps'][0]['passages'][0]['focus'] = {'start': 999, 'end': 1000}
        with self.assertRaisesRegex(ValueError, 'outside its source'): apply_annotations(r, a)

    def test_focus_boolean_is_not_integer(self):
        r = report(); a = annotations(r)
        a['steps'][0]['passages'][0]['focus'] = {'start': True, 'end': 3}
        with self.assertRaisesRegex(ValueError, 'outside its source'): apply_annotations(r, a)

    def test_focus_not_allowed_for_diff(self):
        r = report(); a = annotations(r)
        a['steps'][0]['passages'][0].update(view='diff', focus={'start': 1, 'end': 2})
        with self.assertRaisesRegex(ValueError, 'one definition'): apply_annotations(r, a)

    def test_passage_cannot_cite_other_group(self):
        r = report(); a = annotations(r)
        a['steps'][0]['passages'][0]['change_ids'] = r['groups'][1]['change_ids']
        with self.assertRaisesRegex(ValueError, 'outside its group'): apply_annotations(r, a)

    def test_duplicate_ids_in_passage_rejected(self):
        r = report(); a = annotations(r)
        a['steps'][0]['passages'][0]['change_ids'] *= 2
        with self.assertRaisesRegex(ValueError, 'duplicated'): apply_annotations(r, a)

    def test_same_definition_can_support_two_passages(self):
        r = report(); a = annotations(r)
        a['steps'][0]['passages'].append(copy.deepcopy(a['steps'][0]['passages'][0]))
        out = apply_annotations(r, a)
        self.assertEqual(len(out['groups'][0]['narrative']['passages']), 2)

    def test_passage_view_rejected(self):
        r = report(); a = annotations(r)
        a['steps'][0]['passages'][0]['view'] = '<script>'
        with self.assertRaisesRegex(ValueError, 'view'): apply_annotations(r, a)

    def test_empty_passage_text_rejected(self):
        r = report(); a = annotations(r)
        a['steps'][0]['passages'][0]['text'] = '  '
        with self.assertRaisesRegex(ValueError, 'text'): apply_annotations(r, a)

    def test_bad_document_rejected(self):
        r = report(); a = annotations(r); a['document']['test_runs'] = 100
        with self.assertRaisesRegex(ValueError, 'document'): apply_annotations(r, a)

    def test_reader_is_one_document_not_view_tabs(self):
        h = render(apply_annotations(report(), annotations(report())))
        self.assertIn('id="story"', h)
        self.assertNotIn('role="tablist"', h)
        self.assertNotIn('class="sidebar"', h)
        self.assertNotIn('Next →', h)
        self.assertIn('IntersectionObserver', h)
        self.assertIn('Load full diff', h)

    def test_malicious_prose_embedded_as_data(self):
        r = report(); a = annotations(r)
        attack = '</script><script>window.COMPROMISED=true</script>'
        a['document']['lead'] = attack
        a['steps'][0]['passages'][0]['text'] = attack
        h = render(apply_annotations(r, a))
        self.assertNotIn(attack, h)
        self.assertIn('\\u003c/script\\u003e', h)
        self.assertIn("connect-src 'none'", h)

    def test_renderer_validates_loaded_report_passages(self):
        r = apply_annotations(report(), annotations(report()))
        r['groups'][0]['narrative']['passages'][0]['change_ids'] = ['f' * 16]
        with self.assertRaisesRegex(ValueError, 'outside its group'): render(r)

    def test_evidence_packet_explains_passages(self):
        i = evidence_packet(report())['instructions']
        self.assertIn('passages', i)
        self.assertIn('original line numbers', i)
        self.assertIn('Do not rewrite source', i)


if __name__ == '__main__': unittest.main()
