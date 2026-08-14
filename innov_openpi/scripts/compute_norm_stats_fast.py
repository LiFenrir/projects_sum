"""Compute normalization statistics for a config.

This script reads state and action data directly from parquet files,
bypassing video decoding entirely. This is orders of magnitude faster
since only the two required fields (observation.state, action) are loaded.

Usage:
    python scripts/compute_norm_stats_fast.py --config-yaml configs/bi_s1/pi05_finetune.yaml
    python scripts/compute_norm_stats_fast.py --config-yaml configs/bi_s1/pi05_finetune.yaml --max-frames 5000
"""

import glob
import os

import numpy as np
import pandas as pd
import tqdm
import tyro

import openpi.shared.normalize as normalize
import openpi.training.config as _config
import openpi.transforms as transforms


def _resolve_dataset_paths(repo_id):
    """Resolve repo_id to a flat list of dataset root directories."""
    if repo_id is None:
        raise ValueError("Data config must have a repo_id")

    if isinstance(repo_id, str) and os.path.exists(repo_id):
        contents = os.listdir(repo_id)
        if "data" not in contents and "videos" not in contents:
            repo_id = [
                os.path.join(repo_id, d)
                for d in contents
                if os.path.isdir(os.path.join(repo_id, d))
            ]

    if not isinstance(repo_id, list):
        repo_id = [repo_id]

    return repo_id


def _find_parquet_files(dataset_paths):
    """Find all parquet files under data/chunk-*/ in each dataset directory."""
    parquet_files = []
    for dataset_path in dataset_paths:
        pattern = os.path.join(dataset_path, "data", "chunk-*", "*.parquet")
        found = sorted(glob.glob(pattern))
        if not found:
            print(f"Warning: No parquet files found in {dataset_path}")
        parquet_files.extend(found)
    return parquet_files


def _process_parquet_file(parquet_path, action_dim):
    """Read state and action from a single parquet file and apply transforms.

    Returns (state_array, action_array) as float32 numpy arrays of shape
    [num_frames, action_dim] after padding and outlier clamping.
    """
    df = pd.read_parquet(parquet_path, columns=["observation.state", "action"])

    state = np.array(df["observation.state"].tolist(), dtype=np.float32)
    action_raw = np.array(df["action"].tolist(), dtype=np.float32)

    # Clamp outliers
    state = np.where(state > np.pi, 0, state)
    state = np.where(state < -np.pi, 0, state)
    action_raw = np.where(action_raw > np.pi, 0, action_raw)
    action_raw = np.where(action_raw < -np.pi, 0, action_raw)

    # Pad to target action_dim
    state = transforms.pad_to_dim(state, action_dim, axis=-1)
    action_raw = transforms.pad_to_dim(action_raw, action_dim, axis=-1)

    return state, action_raw


def compute_stats_from_parquet(repo_id, action_dim, max_frames=None):
    """Compute RunningStats for state and actions by reading parquet files directly."""
    dataset_paths = _resolve_dataset_paths(repo_id)
    parquet_files = _find_parquet_files(dataset_paths)

    if not parquet_files:
        raise FileNotFoundError(f"No parquet files found in dataset paths: {dataset_paths}")

    print(f"Found {len(parquet_files)} parquet files across {len(dataset_paths)} dataset(s)")
    print(f"Dataset paths: {dataset_paths}")

    stats = {
        "state": normalize.RunningStats(),
        "actions": normalize.RunningStats(),
    }

    total_frames = 0
    pbar = tqdm.tqdm(parquet_files, desc="Processing parquet files")
    for pf in pbar:
        state, actions = _process_parquet_file(pf, action_dim)
        n_frames = state.shape[0]

        stats["state"].update(state)
        stats["actions"].update(actions)

        total_frames += n_frames
        pbar.set_postfix({"frames": total_frames})

        if max_frames is not None and total_frames >= max_frames:
            break

    print(f"Processed {total_frames} frames total")
    return stats


def main(config_yaml: str = "", max_frames: int | None = None):
    """Compute normalization statistics for a dataset.

    Args:
        config_yaml: Path to a YAML config file.
        max_frames: Maximum number of frames to use for computing stats.
    """
    if not config_yaml:
        raise ValueError(
            "--config-yaml is required. Usage:\n"
            "  python scripts/compute_norm_stats_fast.py --config-yaml configs/bi_s1/pi05_finetune.yaml"
        )

    config = _config.load_config(config_yaml)
    data_config = config.data.create(config.assets_dirs, config.model)

    # 与训练时 _load_norm_stats 路径保持一致: assets/{asset_id}/
    asset_id = data_config.asset_id
    if asset_id is None:
        raise ValueError("data.assets.asset_id 未配置，请在 YAML 中设置 assets.asset_id")
    output_path = config.assets_dirs / asset_id

    print(f"Output path: {output_path}")

    # Compute stats directly from parquet files (no video decoding)
    stats = compute_stats_from_parquet(data_config.repo_id, config.model.action_dim, max_frames)

    norm_stats = {key: s.get_statistics() for key, s in stats.items()}

    print(f"Writing stats to: {output_path}")
    normalize.save(output_path, norm_stats)

    print("Done.")
    print(f"  state mean:  {norm_stats['state'].mean}")
    print(f"  state std:   {norm_stats['state'].std}")
    print(f"  action mean: {norm_stats['actions'].mean}")
    print(f"  action std:  {norm_stats['actions'].std}")


if __name__ == "__main__":
    tyro.cli(main)
