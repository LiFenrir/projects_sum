"""VLA wrapper for RL Token training — loads model via ``create_trained_policy`` (same as serve_policy)."""

from __future__ import annotations

import logging
import os
import pathlib
from typing import Any

import jax
import numpy as np
import torch
from openpi.models.model import Observation
from openpi.policies import policy_config as _policy_config
from openpi.training import checkpoints as _checkpoints
from openpi.training.config import get_config
from openpi.transforms import InjectDefaultPrompt, Normalize, Unnormalize, compose
import openpi.transforms as _transforms
from torch import Tensor

logger = logging.getLogger(__name__)


class VLAWrapper:
    """VLA model wrapper used by RL rollout workers and trainers.

    Model loading and transform chains match serve_policy exactly
    (``create_trained_policy``, bfloat16, eval mode).
    """

    def __init__(
        self,
        checkpoint_path: str,
        config_name: str,
        device: torch.device | str = "cuda",
        data_transforms: _transforms.Group | None = None,
        default_prompt: str | None = None,
    ) -> None:
        self.device = torch.device(device)
        self.train_config = get_config(config_name)

        # Determine checkpoint directory (handle both file and dir paths)
        if os.path.isdir(checkpoint_path):
            checkpoint_dir = pathlib.Path(checkpoint_path)
        else:
            checkpoint_dir = pathlib.Path(checkpoint_path).parent

        data_config = self.train_config.data.create(
            self.train_config.assets_dirs, self.train_config.model
        )

        if data_config.asset_id is None:
            raise ValueError("Asset id is required to load norm stats.")
        norm_stats = self._load_norm_stats(checkpoint_dir, data_config)

        # Reuse serve_policy's loading path (includes bfloat16 + eval())
        self._policy = _policy_config.create_trained_policy(
            self.train_config,
            checkpoint_path,
            default_prompt=default_prompt,
            norm_stats=norm_stats,
            pytorch_device=str(self.device),
        )

        # Extract internals from the Policy object
        self.pi0 = self._policy._model
        self._input_transform = self._policy._input_transform
        self._output_transform = self._policy._output_transform
        self._sample_actions = self._policy._sample_actions

        self.action_dim = self.train_config.model.action_dim
        self.action_horizon = self.train_config.model.action_horizon

        # Optional: custom data_transforms override (used by train_rl_token.py)
        if data_transforms is not None:
            use_q = data_config.use_quantile_norm
            self._input_transform = compose([
                *data_config.repack_transforms.inputs,
                InjectDefaultPrompt(default_prompt),
                *data_transforms.inputs,
                Normalize(norm_stats, use_quantiles=use_q),
                *data_config.model_transforms.inputs,
            ])
            self._output_transform = compose([
                *data_config.model_transforms.outputs,
                Unnormalize(norm_stats, use_quantiles=use_q),
                *data_transforms.outputs,
                *data_config.repack_transforms.outputs,
            ])

    @staticmethod
    def _load_norm_stats(checkpoint_dir: pathlib.Path, data_config) -> dict[str, _transforms.NormStats]:
        """Load norm stats from checkpoint/assets/, falling back to config norm_stats."""
        asset_id = data_config.asset_id
        try:
            norm_stats = _checkpoints.load_norm_stats(checkpoint_dir / "assets", asset_id)
            logger.info("Loaded norm stats from checkpoint: %s/assets/%s", checkpoint_dir, asset_id)
            return norm_stats
        except FileNotFoundError:
            pass

        if data_config.norm_stats is not None:
            logger.info("Checkpoint has no embedded assets; using norm stats from config assets dir")
            return data_config.norm_stats

        raise FileNotFoundError(
            f"No norm stats found in checkpoint ({checkpoint_dir}/assets/{asset_id}) "
            f"or config assets dir. Run compute_norm_stats.py first."
        )

    def preprocess_obs(self, obs: dict[str, Any]) -> Observation:
        """Apply input transforms to raw obs → batched Observation."""
        transformed = self._input_transform(dict(obs))

        batched = jax.tree.map(
            lambda x: torch.from_numpy(np.array(x)).to(self.device)[None, ...],
            transformed,
        )
        return Observation.from_dict(batched)

    def preprocess_obs_batch(self, obs_list: list[dict[str, Any]]) -> Observation:
        """批量预处理观测列表 → batched Observation [N, ...]。

        用于 stride 子采样:多个错位起点观测打包一次 VLA 前向。
        """
        transformed = [self._input_transform(dict(o)) for o in obs_list]
        batched = jax.tree.map(
            lambda *arrs: torch.from_numpy(np.stack([np.asarray(a) for a in arrs])).to(self.device),
            *transformed,
        )
        return Observation.from_dict(batched)

    def extract_embeddings(
        self,
        observation: Observation,
    ) -> tuple[Tensor, Tensor]:
        """Extract post-transformer prefix embeddings z and pad_mask."""
        return self.pi0.extract_prefix_embeddings(observation)

    def sample_reference_actions(
        self,
        observation: Observation,
    ) -> Tensor:
        """Run VLA inference → unnormalize → [B, H, action_dim] robot-space actions.

        Matches ``Policy.infer()`` exactly: model inference → remove batch dim
        → output_transform once.
        """
        raw = self.pi0.sample_actions(self.device, observation)

        # Match Policy.infer(): remove batch dim → output_transform once.
        # Policy.infer() does: jax.tree.map(lambda x: np.asarray(x[0, ...].detach().cpu()), outputs)
        outputs = {
            "state": observation.state[0].detach().cpu().numpy(),
            "actions": raw[0].detach().cpu().numpy(),
        }
        outputs = self._output_transform(outputs)

        # Re-batch to [1, H, action_dim] for callers expecting batched output.
        return torch.as_tensor(outputs["actions"][None, ...], device=self.device)

    def get_rl_chunk_reference(
        self,
        observation: Observation,
        chunk_length: int = 10,
    ) -> Tensor:
        """Slice first C steps from VLA inference as RL reference a_tilde [B, C, action_dim]."""
        full_actions = self.sample_reference_actions(observation)
        return full_actions[:, :chunk_length, :]

    def compute_vla_loss(
        self,
        observation: dict[str, Any] | Observation,
        actions: Tensor,
    ) -> Tensor:
        """Flow-matching denoising loss on demo data [B, H, action_dim] → scalar."""
        per_element_loss = self.pi0.forward(observation, actions)
        return per_element_loss.mean()

    def compute_vla_loss_with_embeddings(
        self,
        observation: Observation,
        actions: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Single forward: returns (z, pad_mask, vla_loss) for joint training."""
        per_element_loss, z, pad_mask = self.pi0.forward_with_prefix_embeddings(
            observation, actions
        )
        return z, pad_mask, per_element_loss.mean()

    def unfreeze(self) -> None:
        """Re-enable gradients on VLA parameters for joint fine-tuning."""
        self.pi0.train()
        for param in self.pi0.parameters():
            param.requires_grad_(True)

    def trainable_parameters(self):
        """Return VLA parameters that require gradients (for optimizer)."""
        return [p for p in self.pi0.parameters() if p.requires_grad]
