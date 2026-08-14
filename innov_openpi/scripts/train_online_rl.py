"""Stage 2: Online RL training with frozen VLA + RL token (Algorithm 1).

Runs the online RL loop: VLA → RL Token → Actor → Environment using TD3
with twin Q-critics and BC regularization.

DDP support is provided for the model placement and multi-GPU inference.
The episode collection loop runs on rank 0 only (it interacts with a
physical or simulated environment); the TD3 update step uses the local
models on each rank.

Usage:
    # Single GPU:
    python scripts/train_online_rl.py --config configs/rlt/stage2_online_rl.yaml

    # Multi-GPU (single node):
    torchrun --standalone --nnodes=1 --nproc_per_node=4 \\
        scripts/train_online_rl.py --config configs/rlt/stage2_online_rl.yaml

    # YAML + CLI overrides:
    python scripts/train_online_rl.py --config configs/rlt/stage2_online_rl.yaml \\
        --max-env-steps 200000

    # Pure CLI:
    python scripts/train_online_rl.py --vla-config-name pi05_droid_finetune \\
        --vla-checkpoint-dir /path/to/vla.safetensors \\
        --rl-token-checkpoint /path/to/rl_token.pt \\
        --env-factory rlt_openpi.envs.franka.env_factory.make_franka_env
"""

from __future__ import annotations

import logging
import sys

import torch
import tyro

from rlt.rollout.factory import make_env, make_intervention
from rlt.rollout.intervention import InterventionManager
from rlt.training.config import OnlineRLTrainConfig
from rlt.training.ddp_utils import cleanup_ddp, is_main_process, setup_ddp
from rlt.training.online_rl_trainer import OnlineRLTrainer
from rlt.utils.checkpoint import load_rl_token_model
from rlt.utils.config_loader import load_config_with_cli
from rlt.utils.logging import Logger
from openpi.training.vla_wrapper import VLAWrapper

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(message)s")
log = logging.getLogger(__name__)


def main(config: OnlineRLTrainConfig) -> None:
    """Run online RL training (Stage 2, Algorithm 1)."""
    # Set up DDP (no-op when WORLD_SIZE=1)
    use_ddp, local_rank, device = setup_ddp()
    main = is_main_process()

    if main:
        log.info("Stage 2 config: %s", config)
        log.info("DDP: %s, device: %s", "enabled" if use_ddp else "disabled", device)

    # Set up logger (rank 0 only)
    rl_logger = Logger.from_train_config(config) if main else None

    # Load frozen VLA
    if main:
        log.info("Loading VLA: config=%s, checkpoint=%s",
                 config.checkpoint.vla_config_name, config.checkpoint.vla_checkpoint_dir)
    vla = VLAWrapper(
        checkpoint_path=config.checkpoint.vla_checkpoint_dir,
        config_name=config.checkpoint.vla_config_name,
        device=device,
    )

    # Load frozen RL token model from Stage 1
    if main:
        log.info("Loading RL token model from %s", config.checkpoint.rl_token_checkpoint)
    rl_token_model = load_rl_token_model(config.checkpoint.rl_token_checkpoint, device=device)

    # Restore fine-tuned VLA weights from Stage 1 checkpoint (if available).
    # Load to CPU first to avoid OOM — the VLA + RL token already occupy most VRAM.
    stage1_ckpt = torch.load(config.checkpoint.rl_token_checkpoint, map_location="cpu", weights_only=False)
    if "vla_model" in stage1_ckpt:
        vla.pi0.load_state_dict(stage1_ckpt["vla_model"])
        if main:
            log.info("Restored fine-tuned VLA weights from Stage 1 checkpoint")
    else:
        if main:
            log.warning("No fine-tuned VLA weights found in Stage 1 checkpoint; using base VLA")
    del stage1_ckpt
    torch.cuda.empty_cache()

    # Create trainer (models are placed on the local device)
    trainer = OnlineRLTrainer(
        config=config,
        vla=vla,
        rl_token_model=rl_token_model,
        device=device,
    )

    # Resume from checkpoint if provided
    if config.checkpoint.resume_checkpoint:
        if main:
            log.info("Resuming from checkpoint: %s", config.checkpoint.resume_checkpoint)
        trainer.load(config.checkpoint.resume_checkpoint)

    # Create environment via pluggable factory (rank 0 only).
    if not config.env.env_factory:
        log.error("--env-factory is required. Provide a Python import path to an env factory function.")
        raise SystemExit(1)

    env = make_env(
        config.env.env_factory,
        action_dim=config.action_dim,
        chunk_length=config.chunk_length,
        task_prompt=config.env.task_prompt,
        max_episode_chunks=config.env.max_episode_chunks,
        **config.env.extra,
    )
    if main:
        log.info("Environment created: action_dim=%d, chunk_length=%d", env.action_dim, env.chunk_length)

    # Create intervention manager (VR teleoperation, etc.) if specified.
    intervention_mgr: InterventionManager | None = None
    if config.env.intervention_factory:
        intervention_mgr = make_intervention(config.env.intervention_factory, env=env)
        if main:
            log.info("Intervention manager created via %s", config.env.intervention_factory)

    trainer.train(env=env, intervention_mgr=intervention_mgr, log_fn=rl_logger.log if rl_logger else None)

    if rl_logger is not None:
        rl_logger.finish()

    cleanup_ddp()


if __name__ == "__main__":
    # Support --config <yaml_path> with optional CLI overrides
    argv = sys.argv[1:]
    yaml_path = None
    if "--config" in argv:
        idx = argv.index("--config")
        yaml_path = argv[idx + 1]
        del argv[idx : idx + 2]

    config = load_config_with_cli(OnlineRLTrainConfig, yaml_path=yaml_path, cli_args=argv)
    main(config)
