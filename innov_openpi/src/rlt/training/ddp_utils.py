"""DDP (Distributed Data Parallel) utilities for RLT training.

Provides lightweight setup/cleanup helpers that mirror the pattern used
by ``scripts/train_pytorch.py``, adapted for the RLT trainer lifecycle.

Usage::

    from rlt.training.ddp_utils import setup_ddp, cleanup_ddp, is_main_process

    use_ddp, local_rank, device = setup_ddp()
    # ... create trainer, run training ...
    cleanup_ddp()
"""

from __future__ import annotations

import logging
import os

import torch
import torch.distributed as dist

logger = logging.getLogger(__name__)


def setup_ddp() -> tuple[bool, int, torch.device]:
    """Initialize the distributed process group and return DDP metadata.

    Reads ``WORLD_SIZE``, ``RANK``, ``LOCAL_RANK`` from the environment
    (set by ``torchrun``).  When ``WORLD_SIZE=1``, DDP is disabled and
    the function returns ``(False, 0, device)``.

    Returns:
        Tuple of ``(use_ddp, local_rank, device)``.
    """
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    use_ddp = world_size > 1

    if use_ddp and not dist.is_initialized():
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        dist.init_process_group(backend=backend, init_method="env://")

        if os.environ.get("TORCH_DISTRIBUTED_DEBUG") is None:
            os.environ["TORCH_DISTRIBUTED_DEBUG"] = "INFO"

    local_rank = int(os.environ.get("LOCAL_RANK", os.environ.get("RANK", "0")))
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        torch.cuda.set_device(device)

    if is_main_process():
        logger.info(
            "DDP: world_size=%d, rank=%d, local_rank=%d, device=%s",
            world_size,
            dist.get_rank() if use_ddp else 0,
            local_rank,
            device,
        )

    return use_ddp, local_rank, device


def cleanup_ddp() -> None:
    """Destroy the process group if initialized."""
    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


def is_main_process() -> bool:
    """Return True on rank 0 (or when DDP is not active)."""
    if not dist.is_initialized():
        return True
    return dist.get_rank() == 0


def get_rank() -> int:
    """Return the current process rank, or 0 if DDP is not active."""
    if not dist.is_initialized():
        return 0
    return dist.get_rank()


def get_world_size() -> int:
    """Return the total number of processes, or 1 if DDP is not active."""
    if not dist.is_initialized():
        return 1
    return dist.get_world_size()
