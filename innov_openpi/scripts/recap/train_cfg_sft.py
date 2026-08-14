#!/usr/bin/env python3
"""Entry point for CFG SFT training with ReCap advantages.

Trains a Pi0.5 model with classifier-free guidance using pre-computed
advantage labels from compute_advantages.py.

Data pipeline per dataset:
    1. LeRobotDataset (with action delta_timestamps)
    2. PromptFromLeRobotTask (resolves task prompt)
    3. RepackTransform → Data transforms → Normalize → Model transforms
       (TokenizePromptWithGuidance replaces standard TokenizePrompt)
    4. AdvantagePreservingDataset (injects advantage bool from parquet)
    5. CfgMixtureDataset (weighted multi-dataset sampling)
    6. PyTorch DataLoader + CFGDataLoaderImpl

Usage:
    # Single GPU
    python scripts/recap/train_cfg_sft.py --config configs/recap/recap_cfg_sft.yaml

    # Multi-GPU (DDP)
    torchrun --nproc_per_node=4 scripts/recap/train_cfg_sft.py --config configs/recap/recap_cfg_sft.yaml
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
import pandas as pd
import torch
import yaml

from openpi import transforms as _transforms
from openpi.models import tokenizer as _tokenizer
from openpi.training import config as _openpi_config
from openpi.training.config import LeRobotDataConfig
from openpi.training.data_loader import TransformedDataset
from recap.data.cfg_dataset import AdvantagePreservingDataset
from recap.data.cfg_dataset import CFGDataLoaderImpl
from recap.data.cfg_dataset import CfgMixtureDataset
from recap.data.cfg_dataset import TokenizePromptWithGuidance
from recap.data.utils import cast_image_features
from recap.data.utils import decode_image_struct_batch
from recap.training.cfg_trainer import CfgTrainConfig
from recap.training.cfg_trainer import CfgTrainer
from recap.training.ddp import setup_ddp

try:
    import wandb
except ImportError:
    wandb = None

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------


def _load_advantages_lookup(
    data_path: str,
    advantage_tag: str | None = None,
) -> dict[tuple[int, int], bool]:
    """Load advantage lookup from meta/advantages_{tag}.parquet.

    Returns dict mapping ``(episode_index, frame_index) → bool``.
    """
    if advantage_tag:
        meta_path = Path(data_path) / "meta" / f"advantages_{advantage_tag}.parquet"
    else:
        meta_path = Path(data_path) / "meta" / "advantages.parquet"

    if not meta_path.exists():
        raise FileNotFoundError(
            f"Advantage file not found: {meta_path}. "
            "Run compute_advantages.py first."
        )

    adv_df = pd.read_parquet(meta_path)
    ep_idx = adv_df["episode_index"].to_numpy().astype(int).tolist()
    fr_idx = adv_df["frame_index"].to_numpy().astype(int).tolist()
    adv_vals = adv_df["advantage"].to_numpy().astype(bool).tolist()
    return dict(zip(zip(ep_idx, fr_idx, strict=True), adv_vals, strict=True))


def build_cfg_dataloader(
    config: CfgTrainConfig,
    local_rank: int,
    world_size: int,
) -> CFGDataLoaderImpl:
    """Build the full CFG data loading pipeline.

    Uses ``LeRobotDataConfig`` from ``openpi.training.config`` for repack,
    data, and normalization transforms. Model transforms are built from
    ``ModelTransformFactory`` with ``TokenizePrompt`` replaced by
    ``TokenizePromptWithGuidance`` for CFG training.
    """
    data_cfg = config.train_data_paths
    if not data_cfg:
        raise ValueError("At least one dataset must be specified in train_data_paths.")

    # --- OpenPI model config ---
    openpi_train_config = _openpi_config.get_config("configs/bi_s1/pi05_finetune.yaml")
    model_config = openpi_train_config.model

    # --- Tokenizer ---
    tokenizer = _tokenizer.PaligemmaTokenizer(model_config.max_token_len)

    # --- Build dataset list ---
    datasets_with_weights: list[tuple[Any, float]] = []

    for ds_cfg in data_cfg:
        data_path = ds_cfg["dataset_path"]
        weight = float(ds_cfg.get("weight", 1.0))
        episodes = ds_cfg.get("episodes")

        camera_map = ds_cfg["camera_map"]
        action_dim = int(ds_cfg["action_dim"])
        robot_type = ds_cfg.get("robot_type", "bi_s1")

        if local_rank == 0:
            logger.info(
                "Loading dataset: %s (robot_type=%s, action_dim=%d)",
                data_path, robot_type, action_dim,
            )

        # 1. Load LeRobot dataset
        meta = LeRobotDatasetMetadata(data_path)
        delta_timestamps = {
            "action": [t / meta.fps for t in range(model_config.action_horizon)],
        }

        base_dataset = LeRobotDataset(
            data_path,
            episodes=episodes,
            delta_timestamps=delta_timestamps,  # type: ignore[arg-type]
        )
        base_dataset.hf_dataset = cast_image_features(base_dataset.hf_dataset)
        base_dataset.hf_dataset.set_transform(decode_image_struct_batch)

        if local_rank == 0:
            logger.info("  Dataset: %d samples, %.1f fps", len(base_dataset), meta.fps)

        # 2. Add task prompts (if dataset has task info)
        tasks = {}
        tasks_path = Path(data_path) / "meta" / "tasks.jsonl"
        if tasks_path.exists():
            with open(tasks_path) as f:
                for line in f:
                    entry = json.loads(line.strip())
                    tasks[entry.get("task_index", len(tasks))] = entry.get("task", "")

        prompt_transforms = []
        if tasks:
            prompt_transforms.append(_transforms.PromptFromLeRobotTask(tasks))

        # 3. Build transform chain via LeRobotDataConfig
        assets_dirs = Path(config.model_path).parent
        data_config = LeRobotDataConfig(
            repo_id=data_path,
            robot_type=robot_type,
            camera_map=camera_map,
            action_dim=action_dim,
        ).create(assets_dirs, model_config)

        # Build model transforms, replacing TokenizePrompt → TokenizePromptWithGuidance
        model_transforms = _build_cfg_model_transforms(
            data_config.model_transforms.inputs,
            tokenizer=tokenizer,
        )

        transform_list = [
            *data_config.repack_transforms.inputs,
            *data_config.data_transforms.inputs,
            _transforms.Normalize(data_config.norm_stats, use_quantiles=data_config.use_quantile_norm),
            *model_transforms,
        ]

        # Apply prompt transforms first
        if prompt_transforms:
            base_dataset = TransformedDataset(base_dataset, prompt_transforms)

        transformed_dataset = TransformedDataset(base_dataset, transform_list)

        # 4. Load advantage lookup
        advantage_tag = config.advantage_tag
        advantages_lookup = _load_advantages_lookup(data_path, advantage_tag)

        if local_rank == 0:
            adv_filename = (
                f"advantages_{advantage_tag}.parquet"
                if advantage_tag
                else "advantages.parquet"
            )
            logger.info(
                "  Loaded advantages from meta/%s (%d entries)",
                adv_filename,
                len(advantages_lookup),
            )

        # 5. Wrap with AdvantagePreservingDataset
        final_dataset = AdvantagePreservingDataset(
            base_dataset=base_dataset,
            transformed_dataset=transformed_dataset,
            advantages_lookup=advantages_lookup,
        )

        datasets_with_weights.append((final_dataset, weight))

        if local_rank == 0:
            logger.info(
                "  Final dataset: %d samples (weight=%.2f)",
                len(final_dataset),
                weight,
            )

    # --- Combined mixture dataset ---
    combined_dataset = CfgMixtureDataset(
        datasets=datasets_with_weights,
        mode="train",
        balance_dataset_weights=config.balance_dataset_weights,
        seed=config.seed,
    )

    if local_rank == 0:
        logger.info(
            "Mixture dataset: %d total samples across %d datasets",
            len(combined_dataset),
            len(datasets_with_weights),
        )

    # --- PyTorch DataLoader with DDP ---
    micro_batch_size = config.micro_batch_size
    sampler = None
    shuffle = True

    if world_size > 1:
        sampler = torch.utils.data.distributed.DistributedSampler(
            combined_dataset,
            num_replicas=world_size,
            rank=local_rank,
            shuffle=True,
            drop_last=True,
            seed=config.seed,
        )
        shuffle = False
        local_batch_size = micro_batch_size
    else:
        local_batch_size = micro_batch_size

    torch_loader = torch.utils.data.DataLoader(
        combined_dataset,
        batch_size=local_batch_size,
        shuffle=shuffle,
        sampler=sampler,
        drop_last=True,
        num_workers=config.num_workers,
        pin_memory=True,
        prefetch_factor=4 if config.num_workers > 0 else None,
        persistent_workers=config.num_workers > 0,
    )

    # --- Wrap with CFG iteration semantics ---
    dataloader = CFGDataLoaderImpl(None, torch_loader)
    dataloader.sampler = sampler  # store for set_epoch forwarding

    return dataloader


def _build_cfg_model_transforms(
    standard_inputs: list,
    tokenizer: Any,
) -> list:
    """Build CFG model transforms by replacing TokenizePrompt with guidance variant.

    Takes the standard model_transforms.inputs list from a DataConfig and
    swaps ``TokenizePrompt`` for ``TokenizePromptWithGuidance``.
    """
    result = []
    for t in standard_inputs:
        if type(t).__name__ == "TokenizePrompt":
            result.append(
                TokenizePromptWithGuidance(
                    tokenizer=tokenizer,
                    discrete_state_input=getattr(t, "discrete_state_input", False),
                )
            )
        else:
            result.append(t)
    return result


# ---------------------------------------------------------------------------
# Wandb logging helper
# ---------------------------------------------------------------------------


def _create_log_fn(project: str = "rlt-openpi"):
    """Create a wandb logging function if wandb is available."""
    if wandb is None:
        return None

    def log_fn(metrics: dict, step: int) -> None:
        wandb.log(metrics, step=step)

    return log_fn


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="CFG SFT Training")
    parser.add_argument(
        "--config", type=str, required=True, help="YAML config file path"
    )
    parser.add_argument("--resume", type=str, default=None, help="Resume from checkpoint")
    parser.add_argument(
        "--wandb_project", type=str, default=None, help="Wandb project name (optional)"
    )
    args = parser.parse_args()

    # Load config
    with open(args.config) as f:
        cfg_dict = yaml.safe_load(f)

    config = CfgTrainConfig(
        **{k: v for k, v in cfg_dict.items() if k in CfgTrainConfig.__dataclass_fields__}
    )

    # Setup DDP
    rank, world_size, local_rank = setup_ddp()
    use_ddp = world_size > 1
    device = f"cuda:{local_rank}"

    logging.basicConfig(
        level=logging.INFO if rank == 0 else logging.WARNING,
        format="[%(asctime)s][%(name)s][%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    if rank == 0:
        logger.info("CFG SFT training starting...")
        logger.info("Config: %s", config)
        logger.info("DDP: %s (world_size=%d)", use_ddp, world_size)

    # Build dataloader
    dataloader = build_cfg_dataloader(config, local_rank, world_size)

    # Wandb logging
    log_fn = None
    if args.wandb_project and rank == 0 and wandb is not None:
        wandb.init(project=args.wandb_project, config=cfg_dict, name=config.run_name)
        log_fn = _create_log_fn(args.wandb_project)

    # Build trainer
    trainer = CfgTrainer(config=config, device=device, use_ddp=use_ddp)

    if args.resume:
        trainer.load(args.resume)

    # Train
    try:
        trainer.train(dataloader, log_fn=log_fn)
    except KeyboardInterrupt:
        logger.info("Training interrupted by user.")
        trainer.save()

    if rank == 0 and log_fn is not None and wandb is not None:
        wandb.finish()


if __name__ == "__main__":
    main()
