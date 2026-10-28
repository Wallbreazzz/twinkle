"""Qwen3-TTS 两段 loss：Talker 时间轴 CE + 0.3 倍残差码本同位 CE。
源码依据：Qwen3-TTS 022e286。其 residual logits 与 cb1..cb15 已对齐，再 shift 会丢 cb1 的直接监督，
并让最后 head 无直接 CE；cb15 标签会被错配给前一 head。此结论不代表官方权重的训练方式或音质收益。"""

import torch
import torch.nn.functional as F

from twinkle.data_format import LossOutput

from tts.common import MultiCodebookLoss


def _shift_ce_like_official(logits, labels, ignore_index=-100):
    """复刻 transformers ForCausalLMLoss 的 shift 语义（含 off-by-one，仅供 shift_labels=True 对照）。

    logits [N, C, V]，labels [N, C]。
    """
    from transformers.loss.loss_utils import ForCausalLMLoss

    return ForCausalLMLoss(
        logits=logits,
        labels=labels,
        vocab_size=logits.shape[-1],
        ignore_index=ignore_index,
    )


class Qwen3TTSLoss(MultiCodebookLoss):
    """Qwen3-TTS 2 段式 loss。

    Args:
        sub_loss_weight: code_predictor 段权重（官方 0.3）。
        shift_labels: code_predictor 段是否做标准 LM shift。
            False（默认）= 修正 off-by-one，同位 per-head CE（无 shift）；
            True = 复现官方 ForCausalLMLoss 含 shift（仅用于对照）。
        ignore_index: 忽略的 label 值（非音频帧 / pad）。
    """

    def __init__(
        self,
        sub_loss_weight: float = 0.3,
        shift_labels: bool = False,
        ignore_index: int = -100,
        ar_codebook: int = 0,
    ):
        super().__init__(ar_codebook=ar_codebook)
        self.sub_loss_weight = sub_loss_weight
        self.shift_labels = shift_labels
        self.ignore_index = ignore_index

    def __call__(self, inputs, outputs, **kwargs) -> LossOutput:
        # talker 段（码本0）：透传编排算好的 mean CE（标准时间轴 shift 正确）
        talker_loss = outputs['talker_loss']
        # code_predictor 段（码本1-15）
        sub_logits = outputs['sub_talker_logits']   # [N, 15, V]
        sub_labels = outputs['sub_talker_labels']   # [N, 15]

        if self.shift_labels:
            sub_loss = _shift_ce_like_official(sub_logits, sub_labels, self.ignore_index)
        else:
            V = sub_logits.shape[-1]
            # 同位 per-head CE（无 shift）：logits[:,j] 预测码本 j+1 ↔ labels[:,j]=码本 j+1
            sub_loss = F.cross_entropy(
                sub_logits.reshape(-1, V),
                sub_labels.reshape(-1),
                ignore_index=self.ignore_index,
                reduction='mean',
            )

        total = talker_loss + self.sub_loss_weight * sub_loss
        # 两段均 per-token mean（镜像官方 sft_12hz.py：outputs.loss 与 sub_loss 都是 mean）。
        # num_tokens=0 走 Twinkle PER-TOKEN-MEAN 归一（grad_accum 路径），
        # 与 accelerate.backward(loss)+gradient_accumulation_steps 一致。
        return LossOutput(loss=total, num_tokens=0)
