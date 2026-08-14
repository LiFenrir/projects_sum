#!/usr/bin/env python3
"""Entry point for Value Model SFT training.

Usage:
    # Single GPU
    python scripts/recap/train_value_sft.py --config configs/recap/recap_value_sft.yaml

    # Multi-GPU
    torchrun --nproc_per_node=4 scripts/recap/train_value_sft.py --config configs/recap/recap_value_sft.yaml
"""

import argparse
import logging
import sys
from pathlib import Path

import torch
import yaml

from recap.training.ddp import cleanup_ddp, setup_ddp
from recap.training.value_trainer import ValueTrainConfig, ValueTrainer

logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Value Model SFT Training")
    parser.add_argument("--config", type=str, required=True,
                        help="YAML config file path")
    parser.add_argument("--resume", type=str, default=None,
                        help="Resume from checkpoint")
    args = parser.parse_args()

    # Load config
    with open(args.config, "r") as f:
        cfg_dict = yaml.safe_load(f)

    config = ValueTrainConfig(**{k: v for k, v in cfg_dict.items()
                                 if k in ValueTrainConfig.__dataclass_fields__})

    # Setup DDP
    rank, world_size, local_rank = setup_ddp()
    use_ddp = world_size > 1
    device = f"cuda:{local_rank}"

    # Build trainer
    trainer = ValueTrainer(config=config, device=device, use_ddp=use_ddp)

    if args.resume:
        trainer.load(args.resume)

    logger.info("Value SFT training ready. Replace with actual DataLoader.")
    logger.info(f"Config: {config}")

    # TODO: Build actual dataloader using rlt.recap.data modules
    # from recap.data.value_dataset import ValueDataset, ...

    cleanup_ddp()


if __name__ == "__main__":
    main()
