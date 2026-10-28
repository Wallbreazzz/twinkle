"""多码本 loss 契约基类。各模型负责区分时间轴 next-token 与码本轴同位监督。"""

from typing import Optional, Sequence

from twinkle.data_format.output import LossOutput
from twinkle.loss.base import Loss


class MultiCodebookLoss(Loss):
    """多码本 next-token CE 基类。

    Args:
        codebook_weights: 各码本 CE 权重，None=等权（Qwen3 的残差段权重 0.3 在子类里体现）。
        ar_codebook: AR 主码本 index（该码本由主模型 next-token 预测，走标准 shift+CE）。
    """

    require_logps = False
    require_logits = True

    def __init__(
        self,
        codebook_weights: Optional[Sequence[float]] = None,
        ar_codebook: int = 0,
    ):
        self.codebook_weights = codebook_weights
        self.ar_codebook = ar_codebook

    def __call__(self, inputs, outputs, **kwargs) -> LossOutput:
        raise NotImplementedError(
            '子类实现：按模型多码本组织算 CE。Qwen3 见 tts.qwen3.loss.Qwen3TTSLoss。'
        )
