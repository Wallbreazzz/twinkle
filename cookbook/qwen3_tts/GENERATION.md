# Paired generation and human review

This follow-up builds on the training cookbook. It generates the same text with
the base model (adapter disabled) and the loaded adapter, using the same seed,
reference and generation settings. It verifies loaded adapter weights and
activation state. A difference in voice is not evidence of improved similarity.

Use the optional dependencies and local 1.7B Base model described in README.md.
Run from `cookbook/qwen3_tts`, selecting one Ascend device:

```bash
export ASCEND_RT_VISIBLE_DEVICES=0
export MODEL_PATH=/models/Qwen3-TTS-12Hz-1.7B-Base
python scripts/generate_comparison.py \
  --model_path "$MODEL_PATH" \
  --checkpoint /data/run/checkpoint-update-10 \
  --reference_audio /data/train-reference.wav \
  --train_jsonl /data/derived/train.jsonl \
  --prompts_jsonl tests/fixtures/evaluation/prompts.jsonl \
  --output_dir outputs/comparison-new --seed 43 --max_new_tokens 256
```

The output directory must be new. Use the intended training reference for
speaker-only conditioning. The default is sampled decoding in both prediction
branches, using the pinned official wrapper's resolved settings. `--decoding
greedy` is an explicit diagnostic alternative. EOS deferral and the historical
diagnostic scripts are not included. All models are loaded locally; there is no
download fallback. Codec execution uses CPU fp32 and the TTS model NPU bf16.

For a single pair, replace `--prompts_jsonl` and `--train_jsonl` with
`--text "A new sentence." --language English`. Single-pair mode does not check
training-text overlap or create the suite review page.

Suite JSONL rows require nonempty `id`, `text`, `language`, `category` and
`checks`. IDs are safe lowercase path components. Languages currently supported
by the suite are English and Chinese. Normalized duplicate text and overlap
with the supplied training manifest are rejected; pretraining overlap remains
unknown. The bundled six prompts are regression probes, not a quality benchmark.

## Artifacts and review

Each case contains `base.wav`, `adapter.wav` and `comparison.json`. The suite
adds `suite.json`, `reference.wav`, `listening.html`, `prompts.jsonl` and
`review-template.json`. Reports contain effective generation settings, seeds,
model/adapter/reference hashes, waveform statistics and observed EOS status.
Float WAV preserves generated values without normalizing or clipping them.
EOS and nonempty audio are structural checks, not content correctness.

Keep the whole directory together and open `listening.html`. The page labels
the two versions and provides direct audio links; it is not a blinded test.
Listen to the reference, check content and endings, and fill only ratings you
can judge. Unfilled scores stay null. Export the review JSON, then run:

```bash
python scripts/summarize_review.py \
  --suite outputs/comparison-new/suite.json \
  --review /data/reviews/human-review.json \
  --output outputs/comparison-new/human-review-summary.json
```

The summarizer requires only Python's standard library. It verifies the suite
and generated-audio hashes, row identities, coverage, score ranges and claimed
completion. Output must not already exist. It reports content counts and raw
human scores without calculating an automatic quality winner. `complete`
means all fields were filled, including when every content rating is `fail`.

## Tests and validation status

`python tests/run.py --isolated` includes the generation/review CPU contracts.
They cover default decoding, waveform preservation, generation observation,
safe review artifacts, overlap rejection, score validation and output protection.
For the targeted real-model workflow after installing dependencies:

```bash
MODEL_PATH="$MODEL_PATH" python tests/entrypoint/test_sampled_workflow_npu.py
```

It prepares two public fixtures, trains one update, reloads in a fresh process,
generates six base/adapter pairs with default sampling, and reads the pending
review template through the CLI. Generated artifacts remain under
`outputs/sampled-workflow-*`; the temporary training checkpoint is discarded.
`python tests/run.py --npu` runs this plus the full training regression.

The earlier development implementation passed 21 remote workflow tests, and
its twelve waveforms matched previously listened-to outputs sample for sample.
This reorganized follow-up removes experimental branches and imports the
packaged runtime, so it still needs its own targeted NPU run before upstream
submission. Neither result establishes convergence or target-speaker similarity.
