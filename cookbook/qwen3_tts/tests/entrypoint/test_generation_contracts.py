"""CPU tests for audio artifacts and generation termination observation."""
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location('comparison', ROOT/'scripts/generate_comparison.py')
generation = importlib.util.module_from_spec(spec)
spec.loader.exec_module(generation)


class GenerationContracts(unittest.TestCase):
    def test_default_sampling_and_explicit_diagnostic_modes(self):
        base = ['--model_path', 'model', '--checkpoint', 'adapter', '--reference_audio', 'ref.wav',
                '--output_dir', 'out', '--text', 'Hello.']
        args = generation.parse_args(base)
        self.assertEqual(args.decoding, 'sampled')
        self.assertEqual(generation.parse_args(base + ['--decoding', 'greedy']).decoding, 'greedy')
        with self.assertRaises(SystemExit):
            generation.parse_args(base + ['--decoding', 'greedy', '--tail_frames', '3'])

    def test_audio_values_preserved_without_clipping(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'test.wav'
            wave = np.array([0, .25, -1.2, 1.3], dtype=np.float32)
            stats = generation.write_wave(path, wave, 24000)
            actual, sr = sf.read(path, dtype='float32')
            np.testing.assert_array_equal(actual, wave)
            self.assertEqual(sr, 24000)
            self.assertEqual(stats['fraction_outside_unit'], .5)
            self.assertEqual(stats['sha256'], generation.file_hash(path))

    def test_invalid_empty_silent_and_nonfinite_audio_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'test.wav'
            for wave in ([], [0, 0], [np.nan], [np.inf], [[1, 2]]):
                with self.subTest(wave=wave), self.assertRaises(ValueError):
                    generation.write_wave(path, wave, 24000)
            with self.assertRaises(ValueError):
                generation.write_wave(path, [1], 0)

    def test_eos_observation_and_method_restoration(self):
        class Talker:
            def generate(self):
                return SimpleNamespace(sequences=np.array([[1, 2, 9]]))
        talker = Talker()
        original = talker.generate
        with generation.capture_stop(talker, 9) as state:
            talker.generate()
        self.assertEqual(state, {'generated_tokens': 3, 'ended_with_eos': True})
        self.assertEqual(talker.generate, original)
        self.assertNotIn('generate', vars(talker))

    def test_truncated_and_failed_generation_restore_override(self):
        talker = SimpleNamespace(generate=lambda: SimpleNamespace(sequences=np.array([[1, 2]])))
        original = talker.generate
        with generation.capture_stop(talker, 9) as state:
            talker.generate()
        self.assertFalse(state['ended_with_eos'])
        self.assertIs(talker.generate, original)
        with self.assertRaises(RuntimeError):
            with generation.capture_stop(talker, 9):
                raise RuntimeError('failure')
        self.assertIs(talker.generate, original)

    def test_cli_help_without_npu_imports(self):
        proc = subprocess.run([sys.executable, str(ROOT/'scripts/generate_comparison.py'), '--help'],
                              capture_output=True, text=True, timeout=30)
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertIn('--checkpoint', proc.stdout)


if __name__ == '__main__':
    unittest.main(verbosity=2)
