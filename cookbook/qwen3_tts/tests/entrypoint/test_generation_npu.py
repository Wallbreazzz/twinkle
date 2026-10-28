"""Train a public-fixture adapter, reload it through Twinkle, retain paired WAVs."""
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import uuid
import numpy as np
import soundfile as sf
ROOT = Path(__file__).resolve().parents[2]

class GenerationAcceptance(unittest.TestCase):
    prompts_path = None
    decoding = 'greedy'
    seed = 42
    timeout = 1200
    artifact_prefix = None

    def verify_suite(self, artifact, summary):
        pass

    def test_new_text_base_adapter_audio(self):
        model = os.environ.get('MODEL_PATH', '/models/Qwen3-TTS-12Hz-1.7B-Base')
        outputs = ROOT / 'outputs'
        outputs.mkdir(exist_ok=True)
        artifact = ROOT / 'outputs' / ((self.artifact_prefix or ('evaluation-' if self.prompts_path else 'generation-')) + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ') + '-' + uuid.uuid4().hex[:8])
        text = 'Today we will test a new voice.'
        with tempfile.TemporaryDirectory(prefix='generation-', dir=outputs) as folder:
            root = Path(folder)
            reference = None
            train_id = None
            for split, fixture in [('train', ROOT / 'tests/fixtures/librispeech'), ('dev', ROOT / 'tests/fixtures/librispeech/second')]:
                meta = json.loads((fixture / 'sample.json').read_text())
                audio = fixture / meta['audio']
                self.assertEqual(hashlib.sha256(audio.read_bytes()).hexdigest(), meta['sha256'])
                self.assertNotEqual(text, meta['text'])
                if split == 'train':
                    reference, train_id = (audio, meta['id'])
                row = dict(id=meta['id'], text=meta['text'], audio=str(audio), speaker_id='1272', group_id=meta['id'])
                (root / f'{split}.jsonl').write_text(json.dumps(row) + '\n', encoding='utf-8')

            def run(script, *args):
                subprocess.run([sys.executable, '-u', str(ROOT / 'scripts' / script), *map(str, args)], check=True, timeout=self.timeout)
            derived = root / 'derived'
            run('prepare_split_manifest.py', '--train_jsonl', root / 'train.jsonl', '--dev_jsonl', root / 'dev.jsonl', '--reference_id', train_id, '--model_path', model, '--output_dir', derived)
            trained = root / 'trained'
            run('train_qwen3_tts_lora.py', '--model_path', model, '--train_jsonl', derived / 'train.jsonl', '--output_dir', trained, '--epochs', '1', '--grad_accum', '2')
            result = json.loads((trained / 'result.json').read_text())
            self.assertEqual(result['optimizer_steps'], 1)
            prompt_args = ['--prompts_jsonl', self.prompts_path, '--train_jsonl', derived / 'train.jsonl'] if self.prompts_path else ['--text', text, '--language', 'English']
            run('generate_comparison.py', '--model_path', model, '--checkpoint', result['checkpoint'], '--reference_audio', reference, *prompt_args, *(['--decoding', self.decoding] if self.decoding is not None else []), '--seed', self.seed, '--max_new_tokens', '256' if self.prompts_path else '128', '--output_dir', artifact)
            if self.prompts_path:
                summary = json.loads((artifact / 'suite.json').read_text())
                self.assertEqual((summary['case_count'], summary['audio_count']), (6, 12))
                self.assertEqual(summary['human_review_status'], 'pending')
                self.assertIsNone(summary['quality_improvement'])
                self.assertTrue(summary['provenance']['text_overlap_checked'])
                self.assertEqual(summary['provenance']['train_manifest_sha256'], hashlib.sha256((derived / 'train.jsonl').read_bytes()).hexdigest())
                for case in summary['cases']:
                    self.assertTrue(case['adapter_weights_verified'])
                    for item in case['outputs'].values():
                        audio = artifact / case['case_id'] / item['file']
                        wave, sr = sf.read(audio)
                        self.assertTrue(wave.ndim == 1 and wave.size > 0 and np.isfinite(wave).all())
                        self.assertEqual(sr, item['sample_rate'])
                        self.assertGreater(item['rms'], 0)
                        self.assertEqual(hashlib.sha256(audio.read_bytes()).hexdigest(), item['sha256'])
                pending = json.loads((artifact / 'review-template.json').read_text())
                self.assertEqual(pending['suite_sha256'], hashlib.sha256((artifact / 'suite.json').read_bytes()).hexdigest())
                self.assertEqual(len(pending['ratings']), summary['audio_count'])
                self.assertTrue(all((r['content'] == 'pending' and r['naturalness'] is None for r in pending['ratings'])))
                self.assertTrue((artifact / 'reference.wav').is_file())
                self.assertTrue((artifact / 'listening.html').is_file())
                (artifact / 'acceptance.json').write_text(json.dumps(dict(train_sample_id=train_id, optimizer_steps=1, quality_evaluated=False)), encoding='utf-8')
                self.verify_suite(artifact, summary)
                print(json.dumps(dict(stage='多文本评估产物通过 / Multi-text evaluation artifacts passed', artifact_dir=str(artifact), cases=6, audios=summary['audio_count'], normal_eos=summary['eos_count'], human_review='pending'), ensure_ascii=False), flush=True)
                return
            report = json.loads((artifact / 'comparison.json').read_text())
            self.assertTrue(report['adapter_weights_verified'])
            self.assertEqual(report['text'], text)
            self.assertEqual(set(report['outputs']), {'base', 'adapter'})
            for label, item in report['outputs'].items():
                audio = artifact / item['file']
                wave, sr = sf.read(audio)
                self.assertEqual(sr, item['sample_rate'])
                self.assertTrue(wave.size > 0 and np.isfinite(wave).all())
                self.assertGreater(item['rms'], 0)
                self.assertEqual(hashlib.sha256(audio.read_bytes()).hexdigest(), item['sha256'])
                self.assertEqual(item['potentially_truncated'], not item['ended_with_eos'])
            report['acceptance'] = dict(train_sample_id=train_id, optimizer_steps=1, held_out_quality_evaluated=False, source='public LibriSpeech regression fixture')
            (artifact / 'comparison.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
            print(json.dumps(dict(stage='音频生成机制通过 / Audio generation mechanism passed', artifact_dir=str(artifact), outputs=report['outputs'], visible_devices=report['visible_devices'], quality_evaluated=False), ensure_ascii=False), flush=True)
if __name__ == '__main__':
    unittest.main(verbosity=2)
