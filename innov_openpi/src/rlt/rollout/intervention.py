"""Human intervention interface for real-robot rollouts.

This module provides a stub ``InterventionManager`` that always reports
no intervention.  Subclass it for real hardware setups where an operator
can override the RL agent's actions (e.g. via keyboard or joystick).

Because real-robot intervention (e.g. VR teleoperation) must step the
robot at the single-action level to compute relative velocity commands,
``get_human_action`` returns an ``InterventionResult`` containing both the
action chunk *and* the resulting env outputs (next_obs, rewards, done,
info).  The rollout worker uses these directly and skips its own
``env.step()`` call.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import NDArray


@dataclass
class InterventionResult:
    """Result of a human-controlled chunk, mirroring env.step() outputs."""

    action_chunk: NDArray  # [C, action_dim]
    next_obs: dict[str, Any]
    rewards: NDArray  # [C]
    done: bool
    info: dict[str, Any] = field(default_factory=dict)


class InterventionManager:
    """Stub intervention manager (no-op).

    Subclass and override ``check_intervention`` and ``get_human_action``
    for real hardware setups.

    Args:
        enabled: Whether intervention checking is active.
    """

    def __init__(self, enabled: bool = False) -> None:
        self.enabled = enabled

    def check_intervention(self) -> bool:
        """Return True if the human operator is intervening."""
        return False

    def get_human_action(
        self, action_dim: int, chunk_length: int
    ) -> InterventionResult | None:
        """Collect a human-controlled action chunk.

        Implementations that step the robot internally should return an
        ``InterventionResult`` so the rollout worker can skip its own
        ``env.step()`` call.

        Args:
            action_dim: Dimension of a single-step action.
            chunk_length: Number of steps in the chunk.

        Returns:
            ``InterventionResult`` if the human provided actions,
            otherwise ``None``.
        """
        return None
