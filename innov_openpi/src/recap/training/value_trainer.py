"""Value Model SFT Trainer."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from .ddp import is_main_process

logger = logging.getLogger(__name__)


@dataclass
class ValueTrainConfig:
    """Configuration for Value Model SFT training."""

    # Model
    critic_expert_variant: str = "gemma_100m"
    num_bins: int = 201
    v_min: float = -1.0
    v_max: float = 0.0
    siglip_path: str = ""
    gemma3_path: str = ""
    tokenizer_path: str = ""
    precision: str = "bf16"
    freeze_vlm: bool = False
    action_dim: int = 7
    action_horizon: int = 10

    # Data
    camera_map: dict[str, str] | None = None
    train_data_paths: list[dict] = field(default_factory=list)
    eval_data_paths: list[dict] = field(default_factory=list)
    tag: str | None = None
    include_state: bool = True
    gamma: float = 1.0
    normalize_to_minus_one_zero: bool = True
    include_next_obs: bool = False
    robot_type: str = "libero"
    model_type: str = "pi05"
    balance_weights: bool = True
    train_num_workers: int = 8
    eval_num_workers: int = 4
    seed: int = 42

    # Training
    micro_batch_size: int = 64
    global_batch_size: int = 512
    max_steps: int = 8000
    save_interval: int = 3000
    val_check_interval: int = 500

    # Optimizer
    lr: float = 1.0e-4
    value_lr: float = 2.0e-4
    adam_beta1: float = 0.9
    adam_beta2: float = 0.95
    adam_eps: float = 1.0e-8
    weight_decay: float = 1.0e-10
    clip_grad: float = 1.0
    lr_scheduler: str = "constant"
    lr_warmup_steps: int = 1000

    # Logging
    log_every: int = 10
    save_dir: str = "./checkpoints/value_sft"
    run_name: str = "value_sft"


class ValueTrainer:
    """Value Model SFT trainer using DDP.

    Trains a ValueCriticModel to predict return values from observations.
    """

    def __init__(
        self,
        config: ValueTrainConfig,
        device: torch.device | str = "cuda",
        use_ddp: bool = False,
    ) -> None:
        self.config = config
        self.device = torch.device(device)
        self.use_ddp = use_ddp
        self._global_step = 0

        # Build ValueCriticModel
        from recap.models.value_critic import get_model

        model_cfg = {
            "critic_expert_variant": config.critic_expert_variant,
            "num_bins": config.num_bins,
            "v_min": config.v_min,
            "v_max": config.v_max,
            "siglip_path": config.siglip_path,
            "gemma3_path": config.gemma3_path,
            "action_dim": config.action_dim,
            "action_horizon": config.action_horizon,
            "freeze_vlm": config.freeze_vlm,
            "precision": config.precision,
            "max_token_len": 200,
        }
        self.model = get_model(model_cfg).to(self.device)

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

        def lr_lambda(step):
            if step < warmup:
                return step / max(1, warmup)
            return 1.0

        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    def _unwrap_model(self):
        if isinstance(self.model, DDP):
            return self.model.module
        return self.model

    def step(self, batch: dict) -> dict[str, float]:
        """Single training step for value prediction."""
        self.model.train()

        batch = {
            k: v.to(self.device) if isinstance(v, torch.Tensor) else v
            for k, v in batch.items()
        }

        with torch.amp.autocast("cuda"):
            result = self._unwrap_model()(observation=batch, target_values=batch.get("target_values"))
            loss = result.loss

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
        metrics = {
            "loss": loss.item(),
            "grad_norm": grad_norm.item() if isinstance(grad_norm, torch.Tensor) else grad_norm,
            "lr": self.optimizer.param_groups[0]["lr"],
            "step": self._global_step,
        }
        if hasattr(result, "cat_acc_best"):
            metrics["cat_acc_best"] = result.cat_acc_best.item()
        if hasattr(result, "mae"):
            metrics["mae"] = result.mae.item()
        return metrics

    def train(self, dataloader, eval_dataloaders=None, log_fn=None) -> None:
        """Full training loop."""
        main = is_main_process()
        if main:
            logger.info("Starting Value SFT training for %d steps", self.config.max_steps)

        for step_idx in range(1, self.config.max_steps + 1):
            if self.use_ddp and hasattr(dataloader, "sampler"):
                sampler = dataloader.sampler
                if hasattr(sampler, "set_epoch"):
                    sampler.set_epoch(step_idx)

            try:
                batch = next(dataloader)
            except StopIteration:
                logger.warning("Dataloader exhausted at step %d", step_idx)
                break

            metrics = self.step(batch)

            if step_idx % self.config.log_every == 0 and log_fn is not None and main:
                log_fn(metrics, step=step_idx)

            if step_idx % self.config.save_interval == 0:
                self.save()

        self.save()
        if main:
            logger.info("Value SFT training complete (%d steps)", self._global_step)

    def save(self, path: str | None = None) -> Path | None:
        """Save checkpoint (rank 0 only)."""
        if not is_main_process():
            if dist.is_initialized():
                dist.barrier()
            return None

        save_dir = Path(path or self.config.save_dir) / self.config.run_name
        save_dir.mkdir(parents=True, exist_ok=True)
        ckpt_path = save_dir / f"value_step{self._global_step}.pt"
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
