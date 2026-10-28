"""Default sampled CLI -> twelve WAVs -> hash-bound pending review summary."""
import json
from pathlib import Path
import subprocess
import sys
import unittest
import test_generation_npu as baseline

ROOT = Path(__file__).resolve().parents[2]


class SampledWorkflow(baseline.GenerationAcceptance):
    prompts_path = ROOT/'tests/fixtures/evaluation/prompts.jsonl'
    decoding = None  # Exercise the user-facing default, not an explicit override.
    seed = 43
    timeout = 3600
    artifact_prefix = 'sampled-workflow-'

    def verify_suite(self, artifact, summary):
        self.assertEqual(summary['provenance']['decoding'], 'sampled')
        self.assertEqual(summary['provenance']['tail_policy'], 'none')
        self.assertEqual(summary['eos_count'], 12)
        for case in summary['cases']:
            self.assertEqual(set(case['outputs']), {'base', 'adapter'})
            self.assertTrue(case['settings']['do_sample'])
            self.assertTrue(case['settings']['subtalker_dosample'])
            self.assertEqual(case['tail_policy'], 'none')
        output = artifact/'review-summary.json'
        subprocess.run([sys.executable, str(ROOT/'scripts/summarize_review.py'),
                        '--suite', str(artifact/'suite.json'),
                        '--review', str(artifact/'review-template.json'),
                        '--output', str(output)], check=True, timeout=30)
        result = json.loads(output.read_text(encoding='utf-8'))
        self.assertEqual(result['status'], 'pending')
        self.assertEqual(result['audio_count'], 12)
        self.assertIsNone(result['quality_improvement'])
        self.assertEqual(result['content_counts'], {'base': {'pending': 6}, 'adapter': {'pending': 6}})


if __name__ == '__main__': unittest.main(verbosity=2)
