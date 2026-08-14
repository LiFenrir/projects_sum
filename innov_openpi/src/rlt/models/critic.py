"""TD3-style twin Q-critic with target networks.

Each Q-network maps (state, action_chunk) → scalar Q-value.
TwinQCritic maintains two online + two target copies with Polyak averaging.
"""

import copy

import torch
from torch import Tensor, nn

from rlt.models.networks import MLP


class QNetwork(nn.Module):
    """Single Q-network: (state, action_chunk) → scalar.

    Args:
        state_dim: Dimension of RL state (z_rl + s^p).
        action_chunk_dim: Dimension of flattened action chunk (C * d).
        hidden_dim: MLP hidden layer width.
        num_hidden_layers: Number of MLP hidden layers.
    """

    def __init__(
        self,
        state_dim: int,
        action_chunk_dim: int,
        hidden_dim: int = 256,
        num_hidden_layers: int = 2,
    ) -> None:
        super().__init__()
        self.mlp = MLP(
            input_dim=state_dim + action_chunk_dim,
            output_dim=1,
            hidden_dim=hidden_dim,
            num_hidden_layers=num_hidden_layers,
        )

    def forward(self, x: Tensor, a: Tensor) -> Tensor:
        """Compute Q-value.

        Args:
            x: RL state [B, state_dim].
            a: Flattened action chunk [B, action_chunk_dim].

        Returns:
            Q-value [B, 1].
        """
        return self.mlp(torch.cat([x, a], dim=-1))


class TwinQCritic(nn.Module):
    """Twin Q-networks with target copies for TD3.

    Args:
        state_dim: Dimension of RL state.
        action_chunk_dim: Dimension of flattened action chunk.
        hidden_dim: MLP hidden layer width.
        num_hidden_layers: Number of MLP hidden layers.
    """

    def __init__(
        self,
        state_dim: int,
        action_chunk_dim: int,
        hidden_dim: int = 256,
        num_hidden_layers: int = 2,
    ) -> None:
        super().__init__()
        self.q1 = QNetwork(state_dim, action_chunk_dim, hidden_dim, num_hidden_layers)
        self.q2 = QNetwork(state_dim, action_chunk_dim, hidden_dim, num_hidden_layers)

        # Frozen target copies
        self.q1_target = copy.deepcopy(self.q1)
        self.q2_target = copy.deepcopy(self.q2)
        for param in self.q1_target.parameters():
            param.requires_grad_(False)
        for param in self.q2_target.parameters():
            param.requires_grad_(False)

    def forward(self, x: Tensor, a: Tensor) -> tuple[Tensor, Tensor]:
        """Compute Q-values from both online networks.

        Returns:
            (q1, q2): Each [B, 1].
        """
        return self.q1(x, a), self.q2(x, a)

    def q_min(self, x: Tensor, a: Tensor) -> Tensor:
        """Min of online Q-values (used in actor loss).

        Returns:
            min(q1, q2) [B, 1].
        """
        q1, q2 = self.forward(x, a)
        return torch.min(q1, q2)

    @torch.no_grad()
    def target_q_min(self, x: Tensor, a: Tensor) -> Tensor:
        """Min of target Q-values (used in TD target computation).

        Returns:
            min(q1_target, q2_target) [B, 1].
        """
        q1_t = self.q1_target(x, a)
        q2_t = self.q2_target(x, a)
        return torch.min(q1_t, q2_t)

    @torch.no_grad()
    def update_targets(self, tau: float) -> None:
        """Polyak-average online params into target networks.

        θ_target ← (1 - τ) * θ_target + τ * θ_online
        """
        for online, target in [
            (self.q1, self.q1_target),
            (self.q2, self.q2_target),
        ]:
            for p_online, p_target in zip(
                online.parameters(),
                target.parameters(),
                strict=True,
            ):
                p_target.data.lerp_(p_online.data, tau)
