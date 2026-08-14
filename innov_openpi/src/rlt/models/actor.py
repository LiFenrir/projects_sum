"""Actor 网络（VLA 参考 + 学习残差）。

确定性前向；探索噪声由 RolloutWorker 在 rollout 时显式添加。
"""

import torch
from torch import Tensor, nn

from rlt.models.networks import MLP


class Actor(nn.Module):
    """VLA 参考 + 学习残差的 Actor。

    forward 始终返回确定性 μ；探索噪声由 RolloutWorker 显式添加，
    与 TD3 actor loss 计算（确定性动作）彻底解耦。

    Args:
        state_dim: RL state 维度 (z_rl + s^p)。
        action_chunk_dim: 展平动作块维度 (C * d)。
        hidden_dim: MLP 隐藏层宽度。
        num_hidden_layers: MLP 隐藏层数。
        sigma: 探索噪声标准差（仅 rollout 侧使用）。
        ref_dropout: training 时清零参考动作的概率。
    """

    def __init__(
        self,
        state_dim: int,
        action_chunk_dim: int,
        hidden_dim: int = 256,
        num_hidden_layers: int = 2,
        sigma: float = 0.1,
        ref_dropout: float = 0.5,
    ) -> None:
        super().__init__()
        self.action_chunk_dim = action_chunk_dim
        self.sigma = sigma
        self.ref_dropout = ref_dropout

        self.mlp = MLP(
            input_dim=state_dim + action_chunk_dim,
            output_dim=action_chunk_dim,
            hidden_dim=hidden_dim,
            num_hidden_layers=num_hidden_layers,
        )

        # Zero-init the last linear layer so the residual starts at zero,
        # meaning the actor initially reproduces the VLA reference exactly.
        last_linear = [m for m in self.mlp.net if isinstance(m, nn.Linear)][-1]
        nn.init.zeros_(last_linear.weight)
        nn.init.zeros_(last_linear.bias)

    def forward(self, x: Tensor, a_tilde: Tensor) -> Tensor:
        """确定性前向：VLA 参考 + 学习残差。

        Actor loss / TD target 用确定性动作；探索噪声由调用方在 rollout 时显式添加。

        Args:
            x: RL state [B, state_dim]。
            a_tilde: VLA 参考动作块 [B, action_chunk_dim]。

        Returns:
            mu: 动作块 [B, action_chunk_dim]，clamp 到 [-1, 1]。
            training 时应用 ref_dropout。
        """
        a_tilde_input = self._apply_ref_dropout(a_tilde)
        residual = self.mlp(torch.cat([x, a_tilde_input], dim=-1))
        mu = (a_tilde + residual).clamp(-1.0, 1.0)
        return mu

    def _apply_ref_dropout(self, a_tilde: Tensor) -> Tensor:
        """Zero out reference actions for a fraction of the batch during training."""
        if not self.training or self.ref_dropout == 0.0:
            return a_tilde

        B = a_tilde.shape[0]
        # Per-sample dropout mask: 1 = keep, 0 = drop
        keep_mask = torch.rand(B, 1, device=a_tilde.device) >= self.ref_dropout
        return a_tilde * keep_mask
