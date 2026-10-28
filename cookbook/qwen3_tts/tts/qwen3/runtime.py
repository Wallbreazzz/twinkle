'Local Qwen3-TTS loading and assertions used by the single-card integration tests.'
import hashlib
import torch

from .data import Qwen3TTSDataConfig

DEFAULT_MODEL_PATH = '/models/Qwen3-TTS-12Hz-1.7B-Base'
ADAPTER = 'smoke'


def data_config(config):
    tc = config.talker_config
    return Qwen3TTSDataConfig(
        **{name: getattr(config, name) for name in
           ('tts_pad_token_id', 'tts_bos_token_id', 'tts_eos_token_id')},
        **{name: getattr(tc, name) for name in
           ('codec_pad_id', 'codec_bos_id', 'codec_eos_token_id',
            'codec_nothink_id', 'codec_think_bos_id', 'codec_think_eos_id',
            'num_code_groups')})


def load_twinkle_model(path, device_mesh):
    # Twinkle constructs a Template (AutoConfig/AutoProcessor) as well as the
    # supplied model_cls. Passing config alone does not register these classes.
    from transformers import AutoConfig, AutoModel, AutoProcessor
    from qwen_tts.core.models import Qwen3TTSConfig, Qwen3TTSForConditionalGeneration, Qwen3TTSProcessor
    from .model import Qwen3TTSModel
    AutoConfig.register('qwen3_tts', Qwen3TTSConfig, exist_ok=True)
    AutoModel.register(Qwen3TTSConfig, Qwen3TTSForConditionalGeneration, exist_ok=True)
    AutoProcessor.register(Qwen3TTSConfig, Qwen3TTSProcessor, exist_ok=True)
    model = Qwen3TTSModel(
        model_id=str(path), model_cls=Qwen3TTSForConditionalGeneration,
        config=Qwen3TTSConfig.from_pretrained(path, local_files_only=True), device_mesh=device_mesh,
        mixed_precision='bf16', torch_dtype=torch.bfloat16,
        attn_implementation='eager', local_files_only=True)
    # A single short sample does not need activation checkpointing. In
    # particular, reentrant checkpointing can sever gradients from frozen inputs.
    model.model.gradient_checkpointing_disable()
    return model


def trainable_snapshot(model):
    params = {n: p for n, p in model.named_parameters() if p.requires_grad}
    if not params or any('.lora_' not in n for n in params):
        raise AssertionError('Only LoRA parameters may be trainable')
    for branch in ('.talker.model.', '.talker.code_predictor.'):
        if not any(branch in n for n in params):
            raise AssertionError('Missing trainable branch: ' + branch)
    return {n: p.detach().cpu().clone() for n, p in params.items()}


def frozen_digest(model):
    """Hash all frozen weights, one tensor at a time; do not retain a base copy."""
    digest = hashlib.sha256()
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            continue
        digest.update(name.encode())
        value = parameter.detach().cpu().contiguous()
        digest.update(str((tuple(value.shape), value.dtype)).encode())
        digest.update(value.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def assert_gradients(model):
    nonzero = []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            if p.grad is not None:
                raise AssertionError('Frozen parameter has gradient: ' + name)
            continue
        if p.grad is None or not torch.isfinite(p.grad).all():
            raise AssertionError('Missing/nonfinite LoRA gradient: ' + name)
        if torch.count_nonzero(p.grad).item():
            nonzero.append(name)
    # Standard LoRA starts B at zero: A can legitimately have zero gradient.
    for branch in ('.talker.model.', '.talker.code_predictor.'):
        if not any(branch in name for name in nonzero):
            raise AssertionError('No nonzero gradient in ' + branch)
    return nonzero


def assert_updated(before, after):
    if before.keys() != after.keys():
        raise AssertionError('Trainable parameter set changed')
    changed = [name for name in before if not torch.equal(before[name], after[name])]
    if any(not torch.isfinite(value).all() for value in after.values()):
        raise AssertionError('Nonfinite updated weights')
    for branch in ('.talker.model.', '.talker.code_predictor.'):
        if not any(branch in name for name in changed):
            raise AssertionError('No parameter update in ' + branch)
    return changed
