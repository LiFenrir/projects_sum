"""Chunk-level environment wrapper for online RL.

Wraps a gym-style environment so that each ``step`` executes an entire
action chunk of C single actions, returning per-step rewards and a single
next observation.
"""

from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np
from numpy.typing import NDArray


class SimEnv:
    """Chunk-level wrapper around a gymnasium environment.

    Args:
        env: A gymnasium environment instance.
        action_dim: Dimension of a single-step action.
        chunk_length: C, number of steps per action chunk.
    """

    def __init__(self, env: gym.Env, action_dim: int, chunk_length: int, obs_stride: int = 2) -> None:
        self.env = env
        self._action_dim = action_dim
        self._chunk_length = chunk_length
        self._obs_stride = max(1, obs_stride)

    @property
    def action_dim(self) -> int:
        return self._action_dim

    @property
    def chunk_length(self) -> int:
        return self._chunk_length

    def _make_obs_dict(self, obs: NDArray) -> dict[str, Any]:
        """Convert raw gym observation to an openpi-compatible dict.

        Subclass or override this for environments whose observations
        already contain images / language instructions.
        """
        return {"state": np.asarray(obs, dtype=np.float32)}

    def reset(self, **kwargs: Any) -> dict[str, Any]:
        """Reset the environment.

        Returns:
            Observation dict with at least a ``"state"`` key.
        """
        obs, _info = self.env.reset(**kwargs)
        return self._make_obs_dict(obs)

    def step(self, action_chunk: NDArray) -> tuple[dict[str, Any], NDArray, bool, dict[str, Any]]:
        """Execute an action chunk of C single-step actions.

        Args:
            action_chunk: Actions to execute, shape ``[C, action_dim]``.

        Returns:
            next_obs: Observation dict after the last executed step.
            rewards: Per-step rewards, shape ``[C]``.  If the episode
                terminates at step k < C, rewards[k+1:] are zero.
            done: True if the episode ended during this chunk.
            info: Info dict from the last executed step, plus stride
                subsampling history (``obs_history``/``actions_history``/
                ``rewards_history``/``done_history``).
        """
        C = self._chunk_length
        stride = self._obs_stride
        rewards = np.zeros(C, dtype=np.float32)
        done = False
        info: dict[str, Any] = {}
        obs = None

        obs_history: list[dict[str, Any]] = []
        actions_history: list[NDArray] = []
        rewards_history: list[float] = []
        done_history: list[float] = []

        for k in range(C):
            obs, reward, terminated, truncated, info = self.env.step(action_chunk[k])
            rewards[k] = float(reward)
            done = terminated or truncated
            actions_history.append(np.asarray(action_chunk[k], dtype=np.float32))
            rewards_history.append(float(reward))
            done_history.append(1.0 if done else 0.0)
            if done:
                break
            # Stride subsampling: exclude the final step (k+1 == C) — that
            # observation equals the next chunk's start, already held by
            # the caller.
            if (k + 1) % stride == 0 and k + 1 < C:
                obs_history.append(self._make_obs_dict(obs))

        info["steps_executed"] = k + 1
        info["obs_history"] = obs_history
        info["actions_history"] = np.stack(actions_history) if actions_history else None
        info["rewards_history"] = np.asarray(rewards_history, dtype=np.float32)
        info["done_history"] = np.asarray(done_history, dtype=np.float32)

        # obs is guaranteed non-None because C >= 1
        assert obs is not None
        return self._make_obs_dict(obs), rewards, done, info
