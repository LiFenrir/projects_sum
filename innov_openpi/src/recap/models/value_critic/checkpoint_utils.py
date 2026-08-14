"""Checkpoint loading and input transform utilities for value models.

Migrated from rlinf/models/embodiment/value_model/checkpoint_utils.py.
"""

import glob
import json
import logging
import pathlib

import numpy as np
import openpi.transforms as _openpi_transforms
import safetensors.torch
import torch

logger = logging.getLogger(__name__)


def load_state_dict_from_checkpoint(checkpoint_path: pathlib.Path) -> dict:
    """Load state dict from checkpoint directory or file.

    Supports:
    - Directory with .safetensors files
    - Directory with .pt/.pth files
    - Single .safetensors file
    - Single .pt/.pth file

    Args:
        checkpoint_path: Path to checkpoint directory or file

    Returns:
        Combined state dict from all files
    """
    if checkpoint_path.is_file():
        if str(checkpoint_path).endswith(".safetensors"):
            return safetensors.torch.load_file(str(checkpoint_path), device="cpu")
        return torch.load(
            str(checkpoint_path), map_location="cpu", weights_only=False
        )

    safetensor_files = sorted(glob.glob(str(checkpoint_path / "*.safetensors")))
    if safetensor_files:
        state_dict = {}
        for f in safetensor_files:
            state_dict.update(safetensors.torch.load_file(f, device="cpu"))
        return state_dict

    pt_files = sorted(glob.glob(str(checkpoint_path / "*.pt"))) + sorted(
        glob.glob(str(checkpoint_path / "*.pth"))
    )
    if pt_files:
        state_dict = {}
        for f in pt_files:
            state_dict.update(torch.load(f, map_location="cpu", weights_only=False))
        return state_dict

    raise FileNotFoundError(f"No checkpoint files found in {checkpoint_path}")


def has_tokenizer_files(checkpoint_dir: pathlib.Path) -> bool:
    """Check if checkpoint directory has tokenizer files."""
    tokenizer_files = [
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
    ]
    return any((checkpoint_dir / f).exists() for f in tokenizer_files)


def load_norm_stats(assets_dir: pathlib.Path, asset_id: str) -> dict:
    """Load normalization statistics from assets/{asset_id}/norm_stats.json.

    Args:
        assets_dir: Path to the assets directory.
        asset_id: Asset identifier (e.g., "bi_s1", "libero"), specified in YAML config.

    Returns:
        Dictionary mapping stat names to openpi NormStats objects.
    """
    path = assets_dir / asset_id / "norm_stats.json"
    if not path.exists():
        logger.warning(f"Norm stats not found at {path}, proceeding without normalization")
        return None

    logger.info(f"Loading norm stats from {path}")
    with open(path) as f:
        data = json.load(f)

    if "norm_stats" in data:
        data = data["norm_stats"]

    result = {}
    for k, v in data.items():
        result[k] = _openpi_transforms.NormStats(
            mean=np.asarray(v["mean"], dtype=np.float32),
            std=np.asarray(v["std"], dtype=np.float32),
            q01=np.asarray(v["q01"], dtype=np.float32) if v.get("q01") is not None else None,
            q99=np.asarray(v["q99"], dtype=np.float32) if v.get("q99") is not None else None,
        )
    return result


