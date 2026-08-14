"""CFG-aware Pi0 action model for ReCap training.

Extends PI0Pytorch with classifier-free guidance: during training, advantage
labels route each sample through a conditional (guidance prompt) or
unconditional (original prompt) language path. At inference, the standard
CFG interpolation formula guides action generation toward the positive
direction.

Migrated from rlinf/models/embodiment/openpi_cfg/openpi_cfg_action_model.py.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor
import torch.nn.functional as F

from openpi.models_pytorch.pi0_pytorch import PI0Pytorch
from openpi.models_pytorch.pi0_pytorch import make_att_2d_masks
import openpi.models_pytorch.preprocessing_pytorch as _preprocessing

_VALID_GUIDANCE_TYPES = ("positive", "negative", "no_guide")


# ---------------------------------------------------------------------------
# Routing mask helper — module-level so it can be tested independently
# ---------------------------------------------------------------------------


def compute_cfg_routing_masks(
    advantage: Tensor,
    *,
    positive_only_conditional: bool,
    unconditional_prob: float,
    random_values: Tensor | None = None,
) -> dict[str, Tensor]:
    """Compute per-sample routing masks for CFG training.

    Args:
        advantage: Boolean tensor ``[B]`` where ``True`` marks positive samples.
        positive_only_conditional: When True, only positive samples can be
            routed to the conditional branch; negatives are always unconditional.
        unconditional_prob: Probability that a sample is routed to the
            unconditional branch (applies to eligible samples only).
        random_values: Optional ``[B]`` uniform noise in ``[0, 1)`` (for tests).

    Returns:
        Dict of boolean masks ``[B]``:
        - ``positive_mask`` / ``negative_mask``
        - ``conditional_mask``
        - ``positive_conditional_mask`` / ``positive_unconditional_mask``
        - ``negative_conditional_mask`` / ``negative_unconditional_mask``
    """
    advantage = advantage.to(dtype=torch.bool)
    batch_size = advantage.shape[0]
    device = advantage.device

    random_values = torch.rand(batch_size, device=device) if random_values is None else random_values.to(device=device)

    positive_mask = advantage
    negative_mask = ~positive_mask

    if positive_only_conditional:
        positive_conditional_mask = positive_mask & (random_values > unconditional_prob)
        negative_conditional_mask = torch.zeros_like(positive_mask)
    else:
        guidance_mask = random_values > unconditional_prob
        positive_conditional_mask = positive_mask & guidance_mask
        negative_conditional_mask = negative_mask & guidance_mask

    conditional_mask = positive_conditional_mask | negative_conditional_mask
    positive_unconditional_mask = positive_mask & ~positive_conditional_mask
    negative_unconditional_mask = negative_mask & ~negative_conditional_mask

    return {
        "positive_mask": positive_mask,
        "negative_mask": negative_mask,
        "conditional_mask": conditional_mask,
        "positive_conditional_mask": positive_conditional_mask,
        "positive_unconditional_mask": positive_unconditional_mask,
        "negative_conditional_mask": negative_conditional_mask,
        "negative_unconditional_mask": negative_unconditional_mask,
    }


# ---------------------------------------------------------------------------
# CFG-aware Pi0 model
# ---------------------------------------------------------------------------


class OpenPi0ForCFGActionPrediction(PI0Pytorch):
    """Pi0 model extended with classifier-free guidance for ReCap training.

    Overrides ``_preprocess_observation``, ``forward``, and ``sample_actions``
    to support three-way prompt routing (neutral / positive guidance /
    negative guidance) based on pre-computed advantage labels.
    """

    def __init__(
        self,
        config,
        *,
        cfgrl_guidance_scale: float = 1.0,
        unconditional_prob: float = 0.1,
        guidance_type: str = "positive",
        positive_only_conditional: bool = True,
    ):
        super().__init__(config)

        if guidance_type not in _VALID_GUIDANCE_TYPES:
            raise ValueError(
                f"guidance_type must be one of {_VALID_GUIDANCE_TYPES}, got '{guidance_type}'"
            )
        if not 0.0 <= unconditional_prob <= 1.0:
            raise ValueError(
                f"unconditional_prob must be in [0, 1], got {unconditional_prob}"
            )

        self._cfg_guidance_scale = cfgrl_guidance_scale
        self._cfg_unconditional_prob = unconditional_prob
        self._cfg_guidance_type = guidance_type
        self._cfg_positive_only_conditional = positive_only_conditional

    # ------------------------------------------------------------------
    # Observation preprocessing (extended for guidance tokens)
    # ------------------------------------------------------------------

    def _preprocess_observation(self, observation, *, train=True):
        """Preprocess observation, additionally returning guidance language tokens.

        Returns a 9-tuple:
            (images, img_masks,
             lang_tokens, lang_masks,
             positive_guidance_lang_tokens, positive_guidance_lang_masks,
             negative_guidance_lang_tokens, negative_guidance_lang_masks,
             state)
        """
        # Call standard preprocessing for images and base language tokens
        base = _preprocessing.preprocess_observation_pytorch(observation, train=train)

        images = list(base.images.values())
        img_masks = list(base.image_masks.values())
        lang_tokens = base.tokenized_prompt
        lang_masks = base.tokenized_prompt_mask
        state = base.state

        # Extract guidance tokens (may be None in pure SFT mode)
        positive_lang_tokens = getattr(observation, "tokenized_positive_guidance_prompt", None)
        positive_lang_masks = getattr(observation, "tokenized_positive_guidance_prompt_mask", None)
        negative_lang_tokens = getattr(observation, "tokenized_negative_guidance_prompt", None)
        negative_lang_masks = getattr(observation, "tokenized_negative_guidance_prompt_mask", None)

        return (
            images,
            img_masks,
            lang_tokens,
            lang_masks,
            positive_lang_tokens,
            positive_lang_masks,
            negative_lang_tokens,
            negative_lang_masks,
            state,
        )

    # ------------------------------------------------------------------
    # Flow loss helper (single time-step)
    # ------------------------------------------------------------------

    def _compute_flow_losses(
        self,
        images,
        img_masks,
        state,
        actions,
        lang_tokens,
        lang_masks,
        device,
        time=None,
        noise=None,
    ) -> tuple[Tensor, Tensor]:
        """Compute flow-matching loss and per-sample detached loss.

        Returns:
            flow_loss: Scalar MSE averaged over all elements.
            per_sample_loss: ``[B]`` detached per-sample MSE.
        """
        images = [img.to(device) for img in images]
        img_masks = [m.to(device) for m in img_masks]
        state = state.to(device)
        actions = actions.to(device, dtype=torch.float32)

        if time is None:
            time = self.sample_time(actions.shape[0], device)
        if noise is None:
            noise = self.sample_noise(actions.shape, device)

        time_expanded = time[:, None, None]
        x_t = time_expanded * noise + (1 - time_expanded) * actions
        u_t = noise - actions

        prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(
            images, img_masks, lang_tokens, lang_masks
        )
        suffix_embs, suffix_pad_masks, suffix_att_masks, adarms_cond = (
            self.embed_suffix(state, x_t, time)
        )

        if (
            self.paligemma_with_expert.paligemma.language_model.layers[
                0
            ].self_attn.q_proj.weight.dtype
            == torch.bfloat16
        ):
            suffix_embs = suffix_embs.to(dtype=torch.bfloat16)
            prefix_embs = prefix_embs.to(dtype=torch.bfloat16)

        pad_masks = torch.cat([prefix_pad_masks, suffix_pad_masks], dim=1)
        att_masks = torch.cat([prefix_att_masks, suffix_att_masks], dim=1)

        att_2d_masks = make_att_2d_masks(pad_masks, att_masks)
        position_ids = torch.cumsum(pad_masks, dim=1) - 1
        att_2d_masks_4d = self._prepare_attention_masks_4d(att_2d_masks)

        def forward_func(
            prefix_embs, suffix_embs, att_2d_masks_4d, position_ids, adarms_cond
        ):
            (_, suffix_out), _ = self.paligemma_with_expert.forward(
                attention_mask=att_2d_masks_4d,
                position_ids=position_ids,
                past_key_values=None,
                inputs_embeds=[prefix_embs, suffix_embs],
                use_cache=False,
                adarms_cond=[None, adarms_cond],
            )
            return suffix_out

        suffix_out = self._apply_checkpoint(
            forward_func,
            prefix_embs,
            suffix_embs,
            att_2d_masks_4d,
            position_ids,
            adarms_cond,
        )
        suffix_out = suffix_out[:, -self.config.action_horizon :]
        suffix_out = suffix_out.to(dtype=torch.float32)

        def action_out_proj_func(suffix_out):
            return self.action_out_proj(suffix_out)

        v_t = self._apply_checkpoint(action_out_proj_func, suffix_out)
        per_element_loss = F.mse_loss(u_t, v_t, reduction="none")
        flow_loss = per_element_loss.mean()
        per_sample_loss = per_element_loss.detach().mean(dim=(-1, -2))
        return flow_loss, per_sample_loss

    # ------------------------------------------------------------------
    # Training forward with CFG routing
    # ------------------------------------------------------------------

    @staticmethod
    def _masked_loss_sum(per_sample_loss: Tensor, mask: Tensor) -> float:
        if mask.numel() == 0 or not torch.any(mask):
            return 0.0
        return (per_sample_loss * mask.float()).sum().item()

    def forward(
        self,
        data: dict[str, Any],
        **kwargs,
    ) -> tuple[Tensor, dict[str, Any]]:
        """CFG training forward.

        Args:
            data: Dict with keys:
                - ``observation``: preprocessed observation (with guidance tokens)
                - ``actions``: ground-truth actions ``[B, H, action_dim]``
                - ``advantage``: boolean advantage labels ``[B]``

        Returns:
            (flow_loss, metrics_dict)
        """
        observation = data["observation"]
        actions = data["actions"]
        device = actions.device
        advantage = data["advantage"].to(device=device, dtype=torch.bool)

        (
            images,
            img_masks,
            lang_tokens,
            lang_masks,
            positive_guidance_lang_tokens,
            positive_guidance_lang_masks,
            negative_guidance_lang_tokens,
            negative_guidance_lang_masks,
            state,
        ) = self._preprocess_observation(observation, train=True)

        # --- Routing ---
        routing = compute_cfg_routing_masks(
            advantage,
            positive_only_conditional=self._cfg_positive_only_conditional,
            unconditional_prob=self._cfg_unconditional_prob,
        )
        positive_mask = routing["positive_mask"]
        conditional_mask = routing["conditional_mask"]
        positive_conditional_mask = routing["positive_conditional_mask"]
        positive_unconditional_mask = routing["positive_unconditional_mask"]
        negative_conditional_mask = routing["negative_conditional_mask"]
        negative_unconditional_mask = routing["negative_unconditional_mask"]

        # --- Select language tokens per sample ---
        if self._cfg_positive_only_conditional:
            final_lang_tokens = torch.where(
                positive_conditional_mask.unsqueeze(-1),
                positive_guidance_lang_tokens,
                lang_tokens,
            )
            final_lang_masks = torch.where(
                positive_conditional_mask.unsqueeze(-1),
                positive_guidance_lang_masks,
                lang_masks,
            )
        else:
            guidance_lang_tokens = torch.where(
                positive_mask.unsqueeze(-1),
                positive_guidance_lang_tokens,
                negative_guidance_lang_tokens,
            )
            guidance_lang_masks = torch.where(
                positive_mask.unsqueeze(-1),
                positive_guidance_lang_masks,
                negative_guidance_lang_masks,
            )
            final_lang_tokens = torch.where(
                conditional_mask.unsqueeze(-1),
                guidance_lang_tokens,
                lang_tokens,
            )
            final_lang_masks = torch.where(
                conditional_mask.unsqueeze(-1),
                guidance_lang_masks,
                lang_masks,
            )

        # --- Flow loss ---
        actions = actions.to(device, dtype=torch.float32)
        time = kwargs.get("time", self.sample_time(actions.shape[0], device))
        noise = kwargs.get("noise", self.sample_noise(actions.shape, device))

        flow_loss, per_sample_loss = self._compute_flow_losses(
            images=images,
            img_masks=img_masks,
            state=state,
            actions=actions,
            lang_tokens=final_lang_tokens,
            lang_masks=final_lang_masks,
            device=device,
            time=time,
            noise=noise,
        )

        # --- Metrics ---
        metrics = {
            "conditional_count": conditional_mask.sum().item(),
            "unconditional_count": (~conditional_mask).sum().item(),
            "conditional_loss_sum": self._masked_loss_sum(
                per_sample_loss, conditional_mask
            ),
            "unconditional_loss_sum": self._masked_loss_sum(
                per_sample_loss, ~conditional_mask
            ),
            "positive_label_count": positive_mask.sum().item(),
            "negative_label_count": (~positive_mask).sum().item(),
            "positive_conditional_count": positive_conditional_mask.sum().item(),
            "positive_unconditional_count": positive_unconditional_mask.sum().item(),
            "negative_conditional_count": negative_conditional_mask.sum().item(),
            "negative_unconditional_count": negative_unconditional_mask.sum().item(),
            "positive_conditional_loss_sum": self._masked_loss_sum(
                per_sample_loss, positive_conditional_mask
            ),
            "positive_unconditional_loss_sum": self._masked_loss_sum(
                per_sample_loss, positive_unconditional_mask
            ),
            "negative_conditional_loss_sum": self._masked_loss_sum(
                per_sample_loss, negative_conditional_mask
            ),
            "negative_unconditional_loss_sum": self._masked_loss_sum(
                per_sample_loss, negative_unconditional_mask
            ),
        }

        return flow_loss, metrics

    # ------------------------------------------------------------------
    # CFG inference (classifier-free guidance sampling)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def sample_actions(
        self,
        observation,
        noise=None,
        num_steps=None,
    ) -> dict:
        """Sample actions with classifier-free guidance.

        At each denoising step:

            v = (1 - w) * v_uncond + w * v_cond

        where *w* is ``cfgrl_guidance_scale``, *v_uncond* uses the neutral
        prompt, and *v_cond* uses the guidance prompt (positive or negative
        depending on ``guidance_type``).

        When ``guidance_type`` is ``"no_guide"``, falls back to standard
        unconditional sampling.
        """
        guidance_type = self._cfg_guidance_type
        scale = self._cfg_guidance_scale
        if self._cfg_positive_only_conditional and guidance_type == "negative":
            raise ValueError(
                "guidance_type='negative' is incompatible with "
                "positive_only_conditional training."
            )

        bsize = observation.state.shape[0]
        device = observation.state.device
        n_steps = num_steps if num_steps is not None else getattr(
            self.config, "num_steps", 10
        )

        if noise is None:
            actions_shape = (bsize, self.config.action_horizon, self.config.action_dim)
            noise = self.sample_noise(actions_shape, device)

        (
            images,
            img_masks,
            lang_tokens,
            lang_masks,
            positive_guidance_lang_tokens,
            positive_guidance_lang_masks,
            negative_guidance_lang_tokens,
            negative_guidance_lang_masks,
            state,
        ) = self._preprocess_observation(observation, train=False)

        # --- Unconditional prefix KV cache ---
        prefix_embs_uncond, prefix_pad_masks_uncond, prefix_att_masks_uncond = (
            self.embed_prefix(images, img_masks, lang_tokens, lang_masks)
        )
        prefix_att_2d_masks_uncond = make_att_2d_masks(
            prefix_pad_masks_uncond, prefix_att_masks_uncond
        )
        prefix_position_ids_uncond = torch.cumsum(prefix_pad_masks_uncond, dim=1) - 1
        prefix_att_2d_masks_4d_uncond = self._prepare_attention_masks_4d(
            prefix_att_2d_masks_uncond
        )

        self.paligemma_with_expert.paligemma.language_model.config._attn_implementation = (
            "eager"
        )

        _, past_key_values_uncond = self.paligemma_with_expert.forward(
            attention_mask=prefix_att_2d_masks_4d_uncond,
            position_ids=prefix_position_ids_uncond,
            past_key_values=None,
            inputs_embeds=[prefix_embs_uncond, None],
            use_cache=True,
        )

        # --- Conditional prefix KV cache ---
        if guidance_type != "no_guide":
            if guidance_type == "positive":
                guidance_lang_tokens = positive_guidance_lang_tokens
                guidance_lang_masks = positive_guidance_lang_masks
            else:
                guidance_lang_tokens = negative_guidance_lang_tokens
                guidance_lang_masks = negative_guidance_lang_masks

            prefix_embs_cond, prefix_pad_masks_cond, prefix_att_masks_cond = (
                self.embed_prefix(
                    images, img_masks, guidance_lang_tokens, guidance_lang_masks
                )
            )
            prefix_att_2d_masks_cond = make_att_2d_masks(
                prefix_pad_masks_cond, prefix_att_masks_cond
            )
            prefix_position_ids_cond = torch.cumsum(prefix_pad_masks_cond, dim=1) - 1
            prefix_att_2d_masks_4d_cond = self._prepare_attention_masks_4d(
                prefix_att_2d_masks_cond
            )

            _, past_key_values_cond = self.paligemma_with_expert.forward(
                attention_mask=prefix_att_2d_masks_4d_cond,
                position_ids=prefix_position_ids_cond,
                past_key_values=None,
                inputs_embeds=[prefix_embs_cond, None],
                use_cache=True,
            )
        else:
            prefix_pad_masks_cond = None
            past_key_values_cond = None

        # --- Iterative denoising ---
        dt = -1.0 / n_steps
        dt = torch.tensor(dt, dtype=torch.float32, device=device)

        x_t = noise
        time = torch.tensor(1.0, dtype=torch.float32, device=device)
        while time >= -dt / 2:
            expanded_time = time.expand(bsize)
            v_t_uncond = self.denoise_step(
                state,
                prefix_pad_masks_uncond,
                past_key_values_uncond,
                x_t,
                expanded_time,
            )

            if guidance_type == "no_guide":
                v_t = v_t_uncond
            else:
                v_t_cond = self.denoise_step(
                    state,
                    prefix_pad_masks_cond,
                    past_key_values_cond,
                    x_t,
                    expanded_time,
                )
                v_t = (1 - scale) * v_t_uncond + scale * v_t_cond

            x_t = x_t + dt * v_t
            time += dt

        return {"actions": x_t}

    # ------------------------------------------------------------------
    # Observation preprocessing for env rollout
    # ------------------------------------------------------------------

    def obs_processor(self, env_obs: dict) -> dict:
        """Convert raw env observation to model-ready dict with guidance prompts."""
        processed = {
            "observation/image": env_obs["main_images"],
            "prompt": env_obs["task_descriptions"],
        }

        # Positive / negative guidance prompts
        if isinstance(env_obs["task_descriptions"], list):
            processed["positive_guidance_prompt"] = [
                f"{desc}\nAdvantage: positive"
                for desc in env_obs["task_descriptions"]
            ]
            processed["negative_guidance_prompt"] = [
                f"{desc}\nAdvantage: negative"
                for desc in env_obs["task_descriptions"]
            ]
        else:
            desc = env_obs["task_descriptions"]
            processed["positive_guidance_prompt"] = f"{desc}\nAdvantage: positive"
            processed["negative_guidance_prompt"] = f"{desc}\nAdvantage: negative"

        # State
        state = env_obs.get("states")
        if state is not None:
            if torch.is_tensor(state):
                state = state.to(dtype=torch.float32)
            processed["observation/state"] = state

        # Wrist image
        wrist = env_obs.get("wrist_images")
        if wrist is not None:
            processed["observation/wrist_image"] = wrist

        return processed
