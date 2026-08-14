"""Compute normalization statistics for a config.

This script reads state and action data directly from parquet files,
bypassing video decoding entirely. This is orders of magnitude faster
since only the two required fields (observation.state, action) are loaded.

Usage:
    python scripts/compute_norm_stats_fast.py --config-name LeRobot_pi05_finetune
    python scripts/compute_norm_stats_fast.py --config-name LeRobot_pi05_finetune --max-frames 5000
"""
import glob
import os
import pathlib

import numpy as np
import pandas as pd
import tqdm
import tyro

import openpi_value.shared.normalize as normalize
import openpi_value.training.config as _config
import openpi_value.transforms as transforms


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


def compute_stats_from_parquet(repo_id, action_dim):
    """Compute RunningStats for state and actions by reading parquet files directly."""
    dataset_paths = _resolve_dataset_paths(repo_id)
    parquet_files = _find_parquet_files(dataset_paths)

    if not parquet_files:
        raise FileNotFoundError(f"No parquet files found in dataset paths: {dataset_paths}")

    print(f"Found {len(parquet_files)} parquet files across {len(dataset_paths)} dataset(s)")

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

    print(f"Processed {total_frames} frames total")
    return stats


def main(config_name: str):
    """Compute normalization statistics for a dataset.

    Args:
        config_name: Name of the registered training config.
    """
    config = _config.get_config(config_name)
    data_config: _config.DataConfig = config.data.create(config.assets_dirs, config.model)

    assets_dir = pathlib.Path(config.data.assets.assets_dir) if config.data.assets.assets_dir else config.assets_dirs
    if data_config.asset_id is not None:
        output_path = assets_dir / data_config.asset_id
    else:
        repo_id = data_config.repo_id
        if repo_id is None:
            raise ValueError("Data config must have a repo_id or assets.asset_id")
        if isinstance(repo_id, list):
            if len(repo_id) != 1:
                raise ValueError("Need to specify assets.asset_id when using multiple datasets")
            repo_id = repo_id[0]
        output_path = assets_dir / pathlib.Path(repo_id).name

    print(f"Output path: {output_path}")

    # Compute stats directly from parquet files (no video decoding)
    stats = compute_stats_from_parquet(data_config.repo_id, config.model.action_dim)

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
