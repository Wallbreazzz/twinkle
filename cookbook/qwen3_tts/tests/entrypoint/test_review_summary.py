"""Review binding, completeness and score validation on explicit file doubles."""
import copy
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import unittest
import test_evaluation_contracts as fixtures

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location('summary', ROOT/'scripts/summarize_review.py')
summary = importlib.util.module_from_spec(spec)
spec.loader.exec_module(summary)


class ReviewSummary(unittest.TestCase):
    def setUp(self):
        fixture = fixtures.EvaluationContracts()
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        cases, reports = fixture.fixtures()
        self.root = fixture.root
        fixtures.evaluation.write_review(self.root, cases, reports, {})
        self.suite = self.root/'suite.json'
        self.path = self.root/'review-template.json'
        self.review = json.loads(self.path.read_text(encoding='utf-8'))

    def run_summary(self):
        self.path.write_text(json.dumps(self.review), encoding='utf-8')
        return summary.summarize(self.suite, self.path)

    def test_pending_and_partial_do_not_invent_scores(self):
        result = self.run_summary()
        self.assertEqual(result['status'], 'pending')
        self.assertEqual(result['score_samples']['base']['speaker_similarity'], [])
        self.review['reviewer'] = 'listener'
        self.review['ratings'][0]['content'] = 'uncertain'
        result = self.run_summary()
        self.assertEqual(result['status'], 'partial')
        self.assertIsNone(result['quality_improvement'])

    def test_complete_means_reviewed_not_passed(self):
        self.review['reviewer'] = 'listener'
        self.review['status'] = 'complete'
        for row in self.review['ratings']:
            row.update(content='fail', speaker_similarity=1, naturalness=2)
        result = self.run_summary()
        self.assertEqual(result['status'], 'complete')
        self.assertEqual(result['content_counts']['adapter']['fail'], 1)
        self.assertIsNone(result['quality_improvement'])

    def test_wrong_suite_and_modified_audio_fail(self):
        self.review['suite_sha256'] = 'wrong'
        with self.assertRaises(ValueError): self.run_summary()
        self.review['suite_sha256'] = summary.digest(self.suite)
        (self.root/'probe/base.wav').write_bytes(b'changed')
        with self.assertRaises(ValueError): self.run_summary()

    def test_missing_duplicate_and_mismatched_rows_fail(self):
        original = copy.deepcopy(self.review)
        for kind in ('missing', 'duplicate', 'path', 'hash'):
            self.review = copy.deepcopy(original)
            if kind == 'missing': self.review['ratings'].pop()
            elif kind == 'duplicate': self.review['ratings'].append(self.review['ratings'][0])
            elif kind == 'path': self.review['ratings'][0]['audio'] = 'other.wav'
            else: self.review['ratings'][0]['audio_sha256'] = 'wrong'
            with self.subTest(kind=kind), self.assertRaises(ValueError): self.run_summary()

    def test_invalid_scores_and_false_completion_fail(self):
        self.review['reviewer'] = 'listener'
        for value in (True, 0, 6, 1.5, '5'):
            self.review['ratings'][0]['naturalness'] = value
            with self.subTest(value=value), self.assertRaises(ValueError): self.run_summary()
        self.review['ratings'][0]['naturalness'] = None
        self.review['status'] = 'complete'
        with self.assertRaises(ValueError): self.run_summary()

    def test_cli_preserves_existing_summary(self):
        output = self.root/'summary.json'
        command = [sys.executable, str(ROOT/'scripts/summarize_review.py'),
                   '--suite', str(self.suite), '--review', str(self.path), '--output', str(output)]
        first = subprocess.run(command, capture_output=True, timeout=30)
        self.assertEqual(first.returncode, 0, first.stderr)
        original = output.read_bytes()
        second = subprocess.run(command, capture_output=True, timeout=30)
        self.assertNotEqual(second.returncode, 0)
        self.assertEqual(output.read_bytes(), original)


if __name__ == '__main__': unittest.main(verbosity=2)
