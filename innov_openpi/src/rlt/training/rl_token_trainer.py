"""Stage 1 trainer: RL token encoder-decoder on demonstration data.

Trains the RLTokenModel (encoder-decoder) to compress VLA embeddings
into a single RL token z_rl via reconstruction loss.

Supports two modes (selected automatically by ``vla_finetune_alpha``):
- **Frozen VLA** (alpha=0): Only trains the encoder-decoder on
  on-the-fly VLA embeddings.
- **Joint training** (alpha>0): Simultaneously trains the RL token
  encoder-decoder (L_ro) and fine-tunes the VLA (alpha * L_vla).
  The combined objective is: L = L_ro(phi) + alpha * L_vla(theta_vla).
  Gradients are independent — L_ro only updates phi (VLA embeddings
  are always detached in the encoder-decoder), and L_vla only updates
  theta_vla.

DDP (Distributed Data Parallel) is supported via the ``use_ddp`` flag.
When enabled, the RL token model is wrapped with
:class:`torch.nn.parallel.DistributedDataParallel`, checkpointing only
happens on rank 0, and the progress bar is suppressed on non-zero ranks.
"""

from __future__ import annotations

from collections.abc import Iterator
import datetime as dt
import logging
import math
from pathlib import Path
from typing import Any

import torch
from torch import Tensor
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm import tqdm

from openpi.models.model import Observation
from openpi.training.vla_wrapper import VLAWrapper
from rlt.models.rl_token import RLTokenModel
from rlt.training.config import RLTokenTrainConfig
from rlt.training.ddp_utils import is_main_process

logger = logging.getLogger(__name__)


class RLTokenTrainer:
    """Stage 1 trainer for the RL token encoder-decoder.

    Mode is selected automatically by ``config.training.vla_finetune_alpha``:
    - alpha == 0: frozen VLA, trains encoder-decoder only.
    - alpha > 0:  joint training, fine-tunes VLA alongside encoder-decoder.

    Usage::

        trainer = RLTokenTrainer(config, device="cuda")
        trainer.train(vla, dataloader, log_fn=logger.log)

    Args:
        config: Stage 1 training hyperparameters.
        device: Torch device for training.
        use_ddp: If True, wrap the RL token model with DDP.
    """

    def __init__(
        self,
        config: RLTokenTrainConfig,
        device: torch.device | str = "cuda",
        use_ddp: bool = False,
    ) -> None:
        self.config = config
        self.device = torch.device(device)
        self.use_ddp = use_ddp

        # Build RL token model
        self.model = RLTokenModel(
            embedding_dim=config.arch.embedding_dim,
            encoder_layers=config.arch.encoder_layers,
            encoder_heads=config.arch.encoder_heads,
            decoder_layers=config.arch.decoder_layers,
            decoder_heads=config.arch.decoder_heads,
        ).to(self.device)

        # Optimizer for the RL token model (created BEFORE DDP wrapping so
        # it holds references to the original parameter objects — DDP's
        # forward() redirects gradients back to them automatically).
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config.training.peak_lr,
            weight_decay=config.training.weight_decay,
        )
        # LR is set per-step via _compute_lr(), matching OpenPI's train_pytorch.py pattern.
        # No torch.optim.lr_scheduler is used.

        # Wrap with DDP after optimizer creation
        if use_ddp:
            self.model = DDP(
                self.model,
                device_ids=[self.device.index] if self.device.type == "cuda" else None,
                find_unused_parameters=False,
            )
            logger.info("RL token model wrapped with DDP (device=%s)", self.device)

        # VLA joint training state (created by _setup_joint_training)
        self._vla: VLAWrapper | None = None
        self.vla_optimizer: torch.optim.Optimizer | None = None

        self._global_step = 0

    @property
    def joint(self) -> bool:
        """Whether the trainer is in joint training mode."""
        return self.config.training.vla_finetune_alpha > 0

    def _unwrap_model(self) -> RLTokenModel:
        """Return the underlying model, unwrapping DDP if needed."""
        if isinstance(self.model, DDP):
            return self.model.module
        return self.model

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def step(
        self,
        vla: VLAWrapper,
        observations: Any,
        actions: Tensor,
    ) -> dict[str, float]:
        """Run one training step, auto-selecting frozen or joint mode.

        Args:
            vla: VLA wrapper.
            observations: Batched Observation (or dict) for the VLA.
            actions: Ground-truth demo actions [B, H, action_dim].

        Returns:
            Dict of logged metrics.
        """
        if self.joint:
            return self._step_joint(vla, observations, actions)
        return self._step_frozen(vla, observations)

    def train(
        self,
        vla: VLAWrapper,
        dataloader: Iterator[tuple[Any, Tensor]],
        log_fn: Any | None = None,
    ) -> None:
        """Run the full training loop.

        Automatically sets up joint training if alpha > 0.

        When DDP is active, the underlying DataLoader should use a
        :class:`~torch.utils.data.distributed.DistributedSampler` and
        ``set_epoch()`` will be called at the start of each epoch.

        Args:
            vla: VLA wrapper.
            dataloader: Infinite iterator yielding (observations, actions).
            log_fn: Optional callable ``log_fn(metrics_dict)`` for logging.
        """
        train_cfg = self.config.training
        alpha = train_cfg.vla_finetune_alpha
        main = is_main_process()

        if self.joint:
            self._setup_joint_training(vla)
            if main:
                logger.info(
                    "Starting Stage 1 joint training for %d steps (alpha=%.3f)",
                    train_cfg.num_train_steps,
                    alpha,
                )
        elif main:
            logger.info(
                "Starting Stage 1 frozen-VLA training for %d steps",
                train_cfg.num_train_steps,
            )

        if main:
            logger.info(
                "LR schedule: warmup=%d, peak_lr=%.2e, decay_steps=%d, decay_lr=%.2e",
                train_cfg.warmup_steps,
                train_cfg.peak_lr,
                train_cfg.decay_steps,
                train_cfg.decay_lr,
            )

        if self.config.checkpoint.resume_checkpoint:
            self.load(self.config.checkpoint.resume_checkpoint)
            if main:
                logger.info("Resumed from step %d", self._global_step)

        # Only show progress bar on rank 0 (single tqdm instance shared by loop + set_postfix)
        if main:
            pbar = tqdm(range(1, train_cfg.num_train_steps + 1), desc="Stage 1")
        else:
            pbar = None
        for step_idx in (pbar if pbar is not None else range(1, train_cfg.num_train_steps + 1)):
            # Set epoch for DistributedSampler shuffling
            if self.use_ddp and hasattr(dataloader, "sampler"):
                sampler = dataloader.sampler
                if hasattr(sampler, "set_epoch"):
                    sampler.set_epoch(step_idx)

            try:
                observations, actions = next(dataloader)
            except StopIteration:
                logger.warning("Dataloader exhausted at step %d", step_idx)
                break

            metrics = self.step(vla, observations, actions)

            if (step_idx == 1 or step_idx % self.config.checkpoint.print_every == 0) and main:
                ts = dt.datetime.now(tz=dt.UTC).astimezone().strftime("%Y-%m-%d %H:%M:%S,%f")[:-3]
                if self.joint:
                    msg = (
                        f"{ts} [rlt.training.rl_token_trainer] INFO "
                        f"step={metrics['step']} loss={metrics['loss']:.6f} "
                        f"l_ro={metrics['l_ro']:.6f} l_vla={metrics['l_vla']:.6f} "
                        f"grad_norm={metrics['grad_norm']:.6f} vla_grad={metrics['vla_grad_norm']:.6f} "
                        f"lr={metrics['lr']:.2e}"
                    )
                else:
                    msg = (
                        f"{ts} [rlt.training.rl_token_trainer] INFO "
                        f"step={metrics['step']} loss={metrics['loss']:.6f} "
                        f"grad_norm={metrics['grad_norm']:.6f} lr={metrics['lr']:.2e}"
                    )
                if pbar is not None:
                    tqdm.write(msg)
                else:
                    logger.info(msg)

            # Progress bar (rank 0 only)
            if pbar is not None:
                if self.joint:
                    pbar.set_postfix(
                        l_ro=f"{metrics['l_ro']:.4f}",
                        l_vla=f"{metrics['l_vla']:.4f}",
                        lr=f"{metrics['lr']:.2e}",
                    )
                else:
                    pbar.set_postfix(loss=f"{metrics['loss']:.4f}", lr=f"{metrics['lr']:.2e}")

            # wandb logging (every log_every steps, rank 0 only)
            if step_idx % self.config.checkpoint.log_every == 0 and log_fn is not None and main:
                log_fn(metrics, step=metrics.get("step"))

            if step_idx % self.config.checkpoint.save_every == 0:
                self.save()

        if self._global_step % self.config.checkpoint.save_every != 0:
            self.save()

        if main:
            logger.info("Stage 1 training complete (%d steps)", self._global_step)

    # ------------------------------------------------------------------
    # Checkpointing
    # ------------------------------------------------------------------

    def save(self, path: str | None = None) -> Path | None:
        """Save model and optimizer state.

        When DDP is active, only rank 0 writes the checkpoint.  Other
        ranks return ``None``.

        Args:
            path: Override save path. Defaults to config.checkpoint.save_dir.

        Returns:
            Path to the saved checkpoint, or ``None`` on non-zero ranks.
        """
        if not is_main_process():
            # Barrier to prevent other ranks from racing ahead
            if dist.is_initialized():
                dist.barrier()
            return None

        ckpt_cfg = self.config.checkpoint
        save_dir = Path(path or ckpt_cfg.save_dir) / ckpt_cfg.run_name
        save_dir.mkdir(parents=True, exist_ok=True)
        ckpt_path = save_dir / f"rl_token_step{self._global_step}.pt"
        state = {
            "model": self._unwrap_model().state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "step": self._global_step,
            "config": self.config,
        }
        if self._vla is not None:
            state["vla_model"] = self._vla.pi0.state_dict()
        if self.vla_optimizer is not None:
            state["vla_optimizer"] = self.vla_optimizer.state_dict()
        torch.save(state, ckpt_path)
        logger.info("Saved checkpoint to %s", ckpt_path)

        # Barrier so other ranks wait for the write to finish
        if dist.is_initialized():
            dist.barrier()
        return ckpt_path

    def load(self, ckpt_path: str) -> None:
        """Load model and optimizer state from checkpoint.

        State is loaded into the underlying (unwrapped) model so it
        works correctly with or without DDP.

        Args:
            ckpt_path: Path to a saved checkpoint file.
        """
        ckpt = torch.load(ckpt_path, map_location=self.device, weights_only=False)
        self._unwrap_model().load_state_dict(ckpt["model"])
        self.optimizer.load_state_dict(ckpt["optimizer"])
        self._global_step = ckpt["step"]
        if "vla_model" in ckpt and self._vla is not None:
            self._vla.pi0.load_state_dict(ckpt["vla_model"])
            logger.info("Restored fine-tuned VLA weights from checkpoint")
        if "vla_optimizer" in ckpt and self.vla_optimizer is not None:
            self.vla_optimizer.load_state_dict(ckpt["vla_optimizer"])
        logger.info("Loaded checkpoint from %s (step %d)", ckpt_path, self._global_step)

    # ------------------------------------------------------------------
    # Private: mode-specific steps
    # ------------------------------------------------------------------

    def _compute_lr(self, step: int, *, peak_lr: float | None = None, decay_lr: float | None = None) -> float:
        """Compute learning rate for a given step (warmup → cosine decay).

        Matches OpenPI's ``train_pytorch.py`` lr_schedule exactly:
        - Warmup: linear from ``peak_lr / (warmup_steps + 1)`` to ``peak_lr``.
        - Decay: cosine from ``peak_lr`` to ``decay_lr``.

        Args:
            step: Current global step (0-indexed).
            peak_lr: Override peak learning rate. Defaults to ``config.training.peak_lr``.
            decay_lr: Override end learning rate. Defaults to ``config.training.decay_lr``.
        """
        train_cfg = self.config.training
        warmup_steps = train_cfg.warmup_steps
        _peak_lr = peak_lr if peak_lr is not None else train_cfg.peak_lr
        _decay_lr = decay_lr if decay_lr is not None else train_cfg.decay_lr
        decay_steps = train_cfg.decay_steps

        if step < warmup_steps:
            # Match JAX behavior: start from peak_lr / (warmup_steps + 1)
            init_lr = _peak_lr / (warmup_steps + 1)
            return init_lr + (_peak_lr - init_lr) * step / warmup_steps

        # Cosine decay
        progress = min(1.0, (step - warmup_steps) / max(1, decay_steps - warmup_steps))
        cos = 0.5 * (1.0 + math.cos(math.pi * progress))
        return _decay_lr + (_peak_lr - _decay_lr) * cos

    def _setup_joint_training(self, vla: VLAWrapper) -> None:
        """Unfreeze VLA and create its optimizer (called once by train())."""
        train_cfg = self.config.training
        vla.unfreeze()
        if train_cfg.gradient_checkpointing:
            vla.pi0.gradient_checkpointing_enable()
            logger.info("Enabled gradient checkpointing on VLA")
        self._vla = vla
        vla_params = vla.trainable_parameters()
        logger.info("Unfroze VLA: %d trainable parameters", sum(p.numel() for p in vla_params))

        self.vla_optimizer = torch.optim.AdamW(
            vla_params,
            lr=train_cfg.vla_learning_rate,
            weight_decay=train_cfg.weight_decay,
        )
        # VLA LR is set per-step via _compute_lr(), same as the RL token optimizer.

    def _step_frozen(
        self,
        vla: VLAWrapper,
        observations: Any,
    ) -> dict[str, float]:
        """Frozen VLA step: extract embeddings (no grad) → L_ro only."""
        self.model.train()
        train_cfg = self.config.training

        # Set LR for this step (matching OpenPI's train_pytorch.py pattern)
        lr = self._compute_lr(self._global_step)
        for pg in self.optimizer.param_groups:
            pg["lr"] = lr

        observations = _obs_to_device(observations, self.device)
        with torch.no_grad():
            z, pad_mask = vla.extract_embeddings(observations)

        z = z.to(self.device)
        pad_mask = pad_mask.to(self.device)
        loss, _z_rl, _z_hat = self.model(z, pad_mask)

        self.optimizer.zero_grad()
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(
            self._unwrap_model().parameters(), max_norm=train_cfg.max_grad_norm
        )
        self.optimizer.step()

        self._global_step += 1
        return {"loss": loss.item(), "grad_norm": grad_norm.item(), "lr": lr, "step": self._global_step}

    def _step_joint(
        self,
        vla: VLAWrapper,
        observations: Any,
        actions: Tensor,
    ) -> dict[str, float]:
        """Joint step: single VLA forward → L_ro + alpha * L_vla."""
        train_cfg = self.config.training
        alpha = train_cfg.vla_finetune_alpha
        self.model.train()

        # Set LR for this step (matching OpenPI's train_pytorch.py pattern)
        lr = self._compute_lr(self._global_step)
        for pg in self.optimizer.param_groups:
            pg["lr"] = lr
        if self.vla_optimizer is not None:
            vla_lr = self._compute_lr(self._global_step, peak_lr=train_cfg.vla_learning_rate)
            for pg in self.vla_optimizer.param_groups:
                pg["lr"] = vla_lr

        observations = _obs_to_device(observations, self.device)
        actions = actions.to(self.device)

        # Single VLA forward: detached embeddings + flow-matching loss
        z, pad_mask, l_vla = vla.compute_vla_loss_with_embeddings(observations, actions)

        # L_ro: RL token reconstruction loss
        z = z.to(self.device)
        pad_mask = pad_mask.to(self.device)
        l_ro, _z_rl, _z_hat = self.model(z, pad_mask)

        # Combined backward (disjoint graphs: L_ro→φ, L_vla→θ_vla)
        total_loss = l_ro + alpha * l_vla

        self.optimizer.zero_grad()
        if self.vla_optimizer is not None:
            self.vla_optimizer.zero_grad()

        total_loss.backward()

        grad_norm = torch.nn.utils.clip_grad_norm_(
            self._unwrap_model().parameters(), max_norm=train_cfg.max_grad_norm
        )
        self.optimizer.step()
        if self.vla_optimizer is not None:
            vla_grad_norm = torch.nn.utils.clip_grad_norm_(
                self._vla.trainable_parameters(), max_norm=train_cfg.max_grad_norm
            )
            self.vla_optimizer.step()
        else:
            vla_grad_norm = torch.tensor(0.0)

        self._global_step += 1
        return {
            "loss": total_loss.item(),
            "l_ro": l_ro.item(),
            "l_vla": l_vla.item(),
            "grad_norm": grad_norm.item(),
            "vla_grad_norm": vla_grad_norm.item(),
            "lr": lr,
            "step": self._global_step,
        }


def _obs_to_device(obs: Any, device: torch.device) -> Any:
    """Recursively move an Observation (or dict) of tensors to a device."""
    if isinstance(obs, Observation):
        return Observation(
            images={k: v.to(device) for k, v in obs.images.items()},
            image_masks={k: v.to(device) for k, v in obs.image_masks.items()},
            state=obs.state.to(device),
            tokenized_prompt=obs.tokenized_prompt.to(device) if obs.tokenized_prompt is not None else None,
            tokenized_prompt_mask=obs.tokenized_prompt_mask.to(device) if obs.tokenized_prompt_mask is not None else None,
            token_ar_mask=obs.token_ar_mask.to(device) if obs.token_ar_mask is not None else None,
            token_loss_mask=obs.token_loss_mask.to(device) if obs.token_loss_mask is not None else None,
        )
    if isinstance(obs, dict):
        return {k: _obs_to_device(v, device) for k, v in obs.items()}
    if isinstance(obs, torch.Tensor):
        return obs.to(device)
    return obs
