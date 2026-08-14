"""CFG (Classifier-Free Guidance) SFT Trainer."""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from dataclasses import field
import logging
from pathlib import Path
from typing import Any

import numpy as np
import safetensors.torch
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from openpi.training import config as _openpi_config
from recap.models.cfg_action_model import OpenPi0ForCFGActionPrediction

from .ddp import is_main_process

logger = logging.getLogger(__name__)


@dataclass
class CfgTrainConfig:
    """Configuration for CFG SFT training."""

    # Model
    model_path: str = ""
    model_type: str = "pi05"
    precision: str = "bf16"

    # Data
    train_data_paths: list[dict] = field(default_factory=list)
    advantage_tag: str | None = None
    balance_dataset_weights: bool = True
    num_workers: int = 8
    seed: int = 42

    # Training
    micro_batch_size: int = 32
    global_batch_size: int = 256
    max_steps: int = 30000
    save_interval: int = 3000
    val_check_interval: int = 500

    # Optimizer
    lr: float = 8.0e-6
    adam_beta1: float = 0.9
    adam_beta2: float = 0.95
    adam_eps: float = 1.0e-8
    weight_decay: float = 1.0e-10
    clip_grad: float = 1.0
    lr_scheduler: str = "cosine"
    lr_warmup_steps: int = 3000

    # CFG-specific
    cfgrl_guidance_scale: float = 1.0
    unconditional_prob: float = 0.1
    guidance_type: str = "positive"
    positive_only_conditional: bool = True

    # Logging
    log_every: int = 10
    save_dir: str = "./checkpoints/cfg_sft"
    run_name: str = "cfg_sft"


class CfgTrainer:
    """CFG SFT trainer using DDP.

    Trains a Pi0.5 model with classifier-free guidance using
    pre-computed advantage labels.
    """

    def __init__(
        self,
        config: CfgTrainConfig,
        device: torch.device | str = "cuda",
        use_ddp: bool = False,
    ) -> None:
        self.config = config
        self.device = torch.device(device)
        self.use_ddp = use_ddp
        self._global_step = 0

        # Build Pi0 config then create CFG-aware model
        train_config = _openpi_config.get_config("pi05_libero")
        model_config = train_config.model
        self.model = OpenPi0ForCFGActionPrediction(
            config=model_config,
            cfgrl_guidance_scale=config.cfgrl_guidance_scale,
            unconditional_prob=config.unconditional_prob,
            guidance_type=config.guidance_type,
            positive_only_conditional=config.positive_only_conditional,
        )
        # Load pre-trained weights (CFG model has same parameter structure as base PI0Pytorch)
        safetensors.torch.load_model(self.model, config.model_path)
        self.model = self.model.to(self.device)

        # Optimizer
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config.lr,
            betas=(config.adam_beta1, config.adam_beta2),
            eps=config.adam_eps,
            weight_decay=config.weight_decay,
        )
        self.scheduler = self._build_scheduler(self.optimizer)
        self.scaler = torch.amp.GradScaler("cuda")

        # DDP wrap
        if use_ddp:
            self.model = DDP(
                self.model,
                device_ids=[self.device.index] if self.device.type == "cuda" else None,
                find_unused_parameters=True,
            )

    def _build_scheduler(self, optimizer):
        warmup = self.config.lr_warmup_steps
        total = self.config.max_steps

        def lr_lambda(step):
            if step < warmup:
                return step / max(1, warmup)
            if self.config.lr_scheduler == "cosine":
                progress = (step - warmup) / max(1, total - warmup)
                return max(0.0, 0.5 * (1.0 + np.cos(np.pi * progress)))
            return 1.0

        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    def _unwrap_model(self):
        if isinstance(self.model, DDP):
            return self.model.module
        return self.model

    def step(self, observation, actions, advantage) -> dict[str, float]:
        """Single training step with CFG loss."""
        self.model.train()

        # Move observation to device (handles nested dicts / dataclass attrs)
        observation = _obs_to_device(observation, self.device)
        actions = actions.to(self.device)
        advantage = advantage.to(self.device)

        with torch.amp.autocast("cuda"):
            loss, step_metrics = self._unwrap_model()(
                data={
                    "observation": observation,
                    "actions": actions,
                    "advantage": advantage,
                },
            )

        self.optimizer.zero_grad()
        self.scaler.scale(loss).backward()
        self.scaler.unscale_(self.optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(
            self._unwrap_model().parameters(), max_norm=self.config.clip_grad
        )
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.scheduler.step()

        self._global_step += 1
        return {
            "loss": loss.item(),
            "grad_norm": grad_norm.item() if isinstance(grad_norm, torch.Tensor) else grad_norm,
            "lr": self.optimizer.param_groups[0]["lr"],
            "step": self._global_step,
            **{f"cfg/{k}": v for k, v in step_metrics.items()},
        }

    def train(self, dataloader, log_fn=None) -> None:
        """Full training loop."""
        main = is_main_process()
        if main:
            logger.info("Starting CFG SFT training for %d steps", self.config.max_steps)

        for step_idx in range(1, self.config.max_steps + 1):
            if self.use_ddp and hasattr(dataloader, "sampler"):
                sampler = dataloader.sampler
                if hasattr(sampler, "set_epoch"):
                    sampler.set_epoch(step_idx)

            try:
                observation, actions, advantage = next(dataloader)
            except StopIteration:
                logger.warning("Dataloader exhausted at step %d", step_idx)
                break

            metrics = self.step(observation, actions, advantage)

            if step_idx % self.config.log_every == 0 and log_fn is not None and main:
                log_fn(metrics, step=step_idx)

            if step_idx % self.config.save_interval == 0:
                self.save()

        self.save()
        if main:
            logger.info("CFG SFT training complete (%d steps)", self._global_step)

    def save(self, path: str | None = None) -> Path | None:
        """Save checkpoint (rank 0 only)."""
        if not is_main_process():
            if dist.is_initialized():
                dist.barrier()
            return None

        save_dir = Path(path or self.config.save_dir) / self.config.run_name
        save_dir.mkdir(parents=True, exist_ok=True)
        ckpt_path = save_dir / f"cfg_step{self._global_step}.pt"
        state = {
            "model": self._unwrap_model().state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "scaler": self.scaler.state_dict(),
            "step": self._global_step,
            "config": self.config,
        }
        torch.save(state, ckpt_path)
        logger.info("Saved checkpoint to %s", ckpt_path)

        if dist.is_initialized():
            dist.barrier()
        return ckpt_path

    def load(self, ckpt_path: str) -> None:
        """Load checkpoint."""
        ckpt = torch.load(ckpt_path, map_location=self.device, weights_only=False)
        self._unwrap_model().load_state_dict(ckpt["model"])
        self.optimizer.load_state_dict(ckpt["optimizer"])
        if "scheduler" in ckpt:
            self.scheduler.load_state_dict(ckpt["scheduler"])
        if "scaler" in ckpt:
            self.scaler.load_state_dict(ckpt["scaler"])
        self._global_step = ckpt["step"]
        logger.info("Loaded checkpoint from %s (step %d)", ckpt_path, self._global_step)


def _obs_to_device(obs: Any, device: torch.device) -> Any:
    """Recursively move observation tensors to device.

    Handles dicts, dataclass instances, and raw tensors.
    """
    if torch.is_tensor(obs):
        return obs.to(device)
    if isinstance(obs, dict):
        return {k: _obs_to_device(v, device) for k, v in obs.items()}
    if dataclasses.is_dataclass(obs) and not isinstance(obs, type):
        kwargs = {}
        for f in dataclasses.fields(obs):
            kwargs[f.name] = _obs_to_device(getattr(obs, f.name), device)
        return type(obs)(**kwargs)
    return obs
