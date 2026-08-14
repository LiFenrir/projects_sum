"""LoRA 注入,语义对齐 JAX 侧 openpi.models.lora。"""

import torch
from torch import nn
import torch.nn.functional as F  # noqa: N812

# attn 作用于 q/k/v/o_proj,ffn 作用于 gate/up/down_proj(与 JAX gemma.py 一致)
_ATTN_TARGETS = ("q_proj", "k_proj", "v_proj", "o_proj")
_FFN_TARGETS = ("gate_proj", "up_proj", "down_proj")


class LoRALinear(nn.Linear):
    """冻结 weight/bias 的 nn.Linear 上叠加低秩分支:y = base(x) + (alpha/rank) * B(A(x))。

    继承 nn.Linear 并复用原参数对象,保持 weight/bias 参数名不变,与 base ckpt 的 key 兼容。
    """

    def __init__(self, base: nn.Linear, rank: int, alpha: float):
        super().__init__(base.in_features, base.out_features, bias=base.bias is not None)
        self.weight = base.weight
        if base.bias is not None:
            self.bias = base.bias
        self.weight.requires_grad = False
        if self.bias is not None:
            self.bias.requires_grad = False
        # 与 JAX 一致:A/B 均以 normal(stddev=0.01) 初始化
        self.lora_a = nn.Parameter(torch.randn(rank, base.in_features) * 0.01)
        self.lora_b = nn.Parameter(torch.randn(base.out_features, rank) * 0.01)
        self.scaling = alpha / rank

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        result = F.linear(x, self.weight, self.bias)
        lora = F.linear(F.linear(x.to(self.lora_a.dtype), self.lora_a), self.lora_b)
        return result + lora.to(result.dtype) * self.scaling


def inject_lora(language_model: nn.Module, lora_configs: dict) -> None:
    """将 Gemma 文本模型各层的 attn/ffn 线性层原位替换为 LoRALinear。lora_configs 键为 "attn"/"ffn"。"""
    for layer in language_model.layers:
        if (cfg := lora_configs.get("attn")) is not None:
            for name in _ATTN_TARGETS:
                setattr(layer.self_attn, name, LoRALinear(getattr(layer.self_attn, name), cfg.rank, cfg.alpha))
        if (cfg := lora_configs.get("ffn")) is not None:
            for name in _FFN_TARGETS:
                setattr(layer.mlp, name, LoRALinear(getattr(layer.mlp, name), cfg.rank, cfg.alpha))


def freeze_non_lora(module: nn.Module) -> None:
    """冻结除 lora_a/lora_b 外的所有参数(对齐 JAX get_freeze_filter)。"""
    for name, p in module.named_parameters():
        p.requires_grad = "lora_" in name
