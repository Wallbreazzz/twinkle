"""Twinkle-loaded base/adapter paired generation, local files only, one NPU."""
import argparse
from contextlib import contextmanager, nullcontext
import hashlib
import json
import os
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def write_wave(path, wave, sample_rate):
    import soundfile as sf
    wave = np.asarray(wave, dtype=np.float32)
    if wave.ndim != 1 or not wave.size or not np.isfinite(wave).all():
        raise ValueError('Generated waveform must be finite nonempty mono audio')
    if not isinstance(sample_rate, (int, np.integer)) or sample_rate <= 0:
        raise ValueError('Invalid generated sample rate')
    peak = float(np.max(np.abs(wave)))
    rms = float(np.sqrt(np.mean(wave.astype(np.float64) ** 2)))
    if rms == 0:
        raise ValueError('Generated waveform is entirely silent')
    # Float WAV preserves raw levels; never silently clip or normalize for comparison.
    sf.write(path, wave, sample_rate, subtype='FLOAT')
    return dict(samples=len(wave), sample_rate=int(sample_rate), seconds=len(wave)/sample_rate,
                peak=peak, rms=rms, fraction_outside_unit=float(np.mean(np.abs(wave) > 1)),
                sha256=file_hash(path))


@contextmanager
def capture_stop(talker, eos_id):
    """Observe the actual generation sequence, not an estimate from audio duration."""
    original = talker.generate
    had_local = 'generate' in vars(talker)
    local = vars(talker).get('generate')
    result = {}

    def generate(*args, **kwargs):
        output = original(*args, **kwargs)
        sequence = output.sequences
        if sequence.ndim != 2 or sequence.shape[0] != 1 or sequence.shape[1] == 0:
            raise ValueError('Expected a single nonempty generated token sequence')
        result.update(generated_tokens=int(sequence.shape[1]),
                      ended_with_eos=int(sequence[0, -1].item()) == eos_id)
        return output

    talker.generate = generate
    try:
        yield result
    finally:
        if had_local:
            talker.generate = local
        else:
            delattr(talker, 'generate')


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('model_path', 'checkpoint', 'reference_audio', 'output_dir'):
        parser.add_argument('--' + name, required=True)
    choice = parser.add_mutually_exclusive_group(required=True)
    choice.add_argument('--text')
    choice.add_argument('--prompts_jsonl')
    parser.add_argument('--train_jsonl', help='Required for a suite: rejects text overlap with this training manifest')
    parser.add_argument('--language', default='Auto', help='Single-pair language; Auto matches the training prefix')
    parser.add_argument('--conditioning_language', choices=('Auto', 'prompt'), default='Auto',
                        help='Auto matches training; prompt explicitly uses each text language')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--max_new_tokens', type=int, default=128)
    parser.add_argument('--decoding', choices=('greedy','sampled'), default='sampled',
                        help='Default: sampled; greedy is retained for controlled diagnostics')
    args = parser.parse_args(argv)
    if (args.text is not None and not args.text.strip()) or args.max_new_tokens < 2:
        parser.error('Nonempty text and max_new_tokens >= 2 required')
    if args.prompts_jsonl and not args.train_jsonl:
        parser.error('--train_jsonl is required with --prompts_jsonl')
    return args


def effective_language(case, args):
    return 'Auto' if args.conditioning_language == 'Auto' else case['language']


def main():
    args = parse_args()
    if args.prompts_jsonl:
        from evaluation_suite import read_suite, write_review
        cases = read_suite(args.prompts_jsonl, args.train_jsonl)
    else:
        cases = [dict(id='single', text=args.text, language=args.language)]
    os.environ['HF_HUB_OFFLINE'] = '1'
    os.environ['TRANSFORMERS_OFFLINE'] = '1'
    import random
    import torch
    import torch_npu  # noqa: F401
    import twinkle
    from twinkle import DeviceMesh
    from transformers import AutoProcessor
    from safetensors.torch import load_file
    from peft.tuners.lora import LoraLayer
    from qwen_tts import Qwen3TTSModel as OfficialModel
    from tts.qwen3.runtime import load_twinkle_model, ADAPTER
    from tts.qwen3.tokenization import load_text_tokenizer, assert_tokenizers_match
    from train_qwen3_tts_lora import model_fingerprint
    from prepare_split_manifest import read_audio

    path = Path(args.model_path).resolve(strict=True)
    checkpoint = Path(args.checkpoint).resolve(strict=True)
    reference = Path(args.reference_audio).resolve(strict=True)
    for name in ('adapter_config.json', 'adapter_model.safetensors'):
        if not (checkpoint / name).is_file():
            raise FileNotFoundError(checkpoint / name)
    out = Path(args.output_dir).resolve()
    out.mkdir(parents=True, exist_ok=False)
    torch.npu.set_device(0)
    mesh = DeviceMesh.from_sizes(fsdp_size=1, dp_size=1)
    twinkle.initialize(mode='local', global_device_mesh=mesh)
    model = load_twinkle_model(path, mesh)
    model.add_adapter_to_model(ADAPTER, str(checkpoint), is_trainable=False)
    model._lazy_wrap_model()
    peft_model = model.strategy.unwrap_model(model.model)
    peft_model.set_adapter(ADAPTER)
    peft_model.requires_grad_(False)
    peft_model.eval()
    # Bypass generic PEFT generate dispatch while keeping injected LoRA layers active.
    core = peft_model.get_base_model()
    saved = load_file(str(checkpoint / 'adapter_model.safetensors'))
    loaded = {n.replace('.' + ADAPTER + '.', '.'): p.detach().cpu() for n, p in peft_model.named_parameters()
              if '.lora_' in n}
    if saved.keys() != loaded.keys():
        raise AssertionError('Loaded adapter parameter keys differ from checkpoint')
    for name in saved:
        torch.testing.assert_close(saved[name], loaded[name], rtol=0, atol=0, check_dtype=False)
    codec = core.speech_tokenizer
    codec.model.to(device='cpu', dtype=torch.float32).eval()
    codec.device = torch.device('cpu')
    processor = AutoProcessor.from_pretrained(str(path), local_files_only=True, fix_mistral_regex=True)
    assert_tokenizers_match(load_text_tokenizer(path), [processor.tokenizer], [case['text'] for case in cases])
    official = OfficialModel(core, processor, generate_defaults=core.generate_config)
    wave, _ = read_audio(reference)
    # Speaker-only conditioning matches training's reference-mel speaker embedding.
    prompt = official.create_voice_clone_prompt(ref_audio=(wave, 24000), x_vector_only_mode=True)
    settings = dict(do_sample=False, subtalker_dosample=False, max_new_tokens=args.max_new_tokens,
                    repetition_penalty=1.05, non_streaming_mode=True)
    if args.decoding=='sampled':
        settings=official._merge_generate_kwargs(do_sample=True,subtalker_dosample=True,
            max_new_tokens=args.max_new_tokens,non_streaming_mode=True)
    common = dict(format_version=1, seed=args.seed, decoding=args.decoding,
                  tail_policy='none',
                  mode='x_vector_only', conditioning_language=args.conditioning_language, settings=settings, reference_sha256=file_hash(reference),
                  model_content_sha256=model_fingerprint(path),
                  adapter_sha256=file_hash(checkpoint/'adapter_model.safetensors'),
                  adapter_config_sha256=file_hash(checkpoint/'adapter_config.json'),
                  adapter_weights_verified=True, visible_devices=os.environ.get('ASCEND_RT_VISIBLE_DEVICES'),
                  outputs={}, interpretation='mechanism comparison only; no quality improvement claim')
    lora_layers = [m for m in core.modules() if isinstance(m, LoraLayer)]
    if not lora_layers:
        raise AssertionError('No LoRA layers loaded')
    reports = []
    for case in cases:
        case_out = out/case['id'] if args.prompts_jsonl else out
        if args.prompts_jsonl:
            case_out.mkdir()
        report = dict(common, case_id=case['id'], text=case['text'], language=case['language'],
                      effective_language=effective_language(case, args), outputs={})
        for label in ('base', 'adapter'):
            random.seed(args.seed)
            np.random.seed(args.seed)
            torch.manual_seed(args.seed)
            torch.npu.manual_seed(args.seed)
            with (peft_model.disable_adapter() if label == 'base' else nullcontext()):
                if any(m.disable_adapters != (label == 'base') or ADAPTER not in m.active_adapters for m in lora_layers):
                    raise AssertionError('Unexpected adapter activation state')
                with torch.inference_mode(), capture_stop(core.talker, core.config.talker_config.codec_eos_token_id) as stop:
                    waves, sr = official.generate_voice_clone(text=case['text'], language=effective_language(case, args),
                        voice_clone_prompt=prompt, **settings)
            if len(waves) != 1 or not stop:
                raise AssertionError('Expected one observed generation and waveform')
            stats = write_wave(case_out/f'{label}.wav', waves[0], sr)
            report['outputs'][label] = dict(stats, **stop, file=f'{label}.wav',
                                           potentially_truncated=not stop['ended_with_eos'])
            (case_out/'comparison.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
            print(json.dumps(dict(stage='生成完成 / Generation completed', case_id=case['id'], variant=label,
                                 **report['outputs'][label]), ensure_ascii=False), flush=True)
        reports.append(report)
    if args.prompts_jsonl:
        reference_stats = write_wave(out/'reference.wav', wave, 24000)
        provenance = dict(model_content_sha256=common['model_content_sha256'],
            adapter_sha256=common['adapter_sha256'], reference_source_sha256=common['reference_sha256'],
            reference_audio=reference_stats, prompts_sha256=file_hash(args.prompts_jsonl),
            train_manifest_sha256=file_hash(args.train_jsonl), seed=args.seed, settings=settings,
            decoding=args.decoding, conditioning_language=args.conditioning_language,
            tail_policy='none',tail_frames=0,
            text_overlap_checked=True, pretraining_overlap_unknown=True)
        (out/'prompts.jsonl').write_bytes(Path(args.prompts_jsonl).read_bytes())
        summary = write_review(out, cases, reports, provenance)
        print(json.dumps(dict(stage='多文本生成完成 / Evaluation suite generated', cases=summary['case_count'],
            audios=summary['audio_count'], normal_eos=summary['eos_count'], human_review='pending'), ensure_ascii=False), flush=True)
    print('试听目录 / Listening directory: ' + str(out), flush=True)


if __name__ == '__main__':
    main()
