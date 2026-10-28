"""Evaluation provenance, overlap rejection, pending scores and safe HTML."""
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location('evaluation_suite', ROOT/'scripts/evaluation_suite.py')
evaluation = importlib.util.module_from_spec(spec)
spec.loader.exec_module(evaluation)


class EvaluationContracts(unittest.TestCase):
    def test_two_variants_have_independent_audio_and_links(self):
        from html.parser import HTMLParser
        class Links(HTMLParser):
            def __init__(self):
                super().__init__()
                self.audio, self.links = [], []
            def handle_starttag(self, tag, attrs):
                attrs = dict(attrs)
                if tag == 'audio': self.audio.append(attrs.get('src'))
                if tag == 'a': self.links.append(attrs.get('href'))
        cases, reports = self.fixtures()
        evaluation.write_review(self.root, cases, reports, {})
        parser = Links()
        parser.feed((self.root/'listening.html').read_text(encoding='utf-8'))
        self.assertEqual(parser.audio, ['reference.wav', 'probe/base.wav', 'probe/adapter.wav'])
        self.assertEqual(parser.links, ['probe/base.wav', 'probe/adapter.wav'])

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.train = self.root/'train.jsonl'
        self.train.write_text(json.dumps(dict(text='Training only.')), encoding='utf-8')
        self.prompts = self.root/'prompts.jsonl'
        self.row = dict(id='probe', text='New text.', language='English', category='statement', checks='Listen.')

    def read(self, rows=None):
        self.prompts.write_text('\n'.join(json.dumps(r) for r in (rows if rows is not None else [self.row])), encoding='utf-8')
        return evaluation.read_suite(self.prompts, self.train)

    def test_six_fixed_cases_are_unique_and_not_training_texts(self):
        rows = evaluation.read_suite(ROOT/'tests/fixtures/evaluation/prompts.jsonl', self.train)
        self.assertEqual(len(rows), 6)
        self.assertEqual({r['language'] for r in rows}, {'English', 'Chinese'})

    def test_overlap_and_duplicates_fail(self):
        for rows in ([{**self.row, 'text':' TRAINING   ONLY. '}], [self.row, self.row],
                     [self.row, {**self.row, 'id':'other', 'text':'NEW TEXT.'}]):
            with self.subTest(rows=rows), self.assertRaises(ValueError):
                self.read(rows)

    def test_empty_and_unsafe_fields_fail(self):
        for rows in ([], [{**self.row, 'id':'../escape'}], [{**self.row, 'checks':''}],
                     [{**self.row, 'language':'Unknown'}]):
            with self.subTest(rows=rows), self.assertRaises(ValueError):
                self.read(rows)

    def fixtures(self):
        self.row['text'] = '</script><img src=x onerror=alert(1)>'
        cases = self.read()
        (self.root/'probe').mkdir()
        outputs = {}
        for variant in ('base', 'adapter'):
            data = variant.encode()  # hash-only artifact double, no audio claim
            (self.root/'probe'/f'{variant}.wav').write_bytes(data)
            outputs[variant] = dict(sha256=hashlib.sha256(data).hexdigest(), seconds=1,
                                    ended_with_eos=variant=='base')
        return cases, [dict(case_id='probe', outputs=outputs)]

    def test_review_remains_unscored_and_html_escaped(self):
        cases, reports = self.fixtures()
        result = evaluation.write_review(self.root, cases, reports, {'source':'unit test'})
        self.assertEqual(result['eos_count'], 1)
        self.assertIsNone(result['quality_improvement'])
        review = json.loads((self.root/'review-template.json').read_text(encoding='utf-8'))
        self.assertEqual(review['status'], 'pending')
        self.assertTrue(all(r['speaker_similarity'] is None for r in review['ratings']))
        self.assertEqual(review['suite_sha256'], hashlib.sha256((self.root/'suite.json').read_bytes()).hexdigest())
        page = (self.root/'listening.html').read_text(encoding='utf-8')
        self.assertNotIn('<img src=x', page)
        self.assertIn('&lt;img', page)
        self.assertIn('可能截断', page)
        self.assertIn('probe/base.wav', page)

    def test_changed_audio_and_missing_result_rejected(self):
        cases, reports = self.fixtures()
        with self.assertRaises(ValueError):
            evaluation.write_review(self.root, cases, [], {})
        (self.root/'probe/base.wav').write_bytes(b'changed')
        with self.assertRaisesRegex(ValueError, 'hash mismatch'):
            evaluation.write_review(self.root, cases, reports, {})

    def test_experimental_variants_are_rejected(self):
        cases, reports = self.fixtures()
        reports[0]['outputs']['base_tail'] = reports[0]['outputs']['base']
        with self.assertRaises(ValueError):
            evaluation.write_review(self.root, cases, reports, {})


if __name__ == '__main__':
    unittest.main(verbosity=2)
