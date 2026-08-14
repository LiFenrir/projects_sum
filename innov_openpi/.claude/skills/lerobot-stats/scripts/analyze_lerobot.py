#!/usr/bin/env python3
"""Analyze LeRobot dataset statistics for a given date (MMDD)."""

import argparse
import glob
import json
import os
import sys

import numpy as np
import pandas as pd


def analyze_dataset(ds_path: str) -> dict | None:
    """Analyze a single LeRobot dataset directory. Returns None if empty."""
    parquet_files = sorted(glob.glob(f"{ds_path}/data/**/*.parquet", recursive=True))
    if not parquet_files:
        return None

    all_eps = []
    for pf in parquet_files:
        df = pd.read_parquet(pf)
        for ep_idx in df["episode_index"].unique():
            ep_df = df[df["episode_index"] == ep_idx]
            frames = len(ep_df)
            has_fail = "is_failure_data" in df.columns
            has_infer = "is_infer_data" in df.columns
            has_task = "task_index" in df.columns

            ep_data = {
                "episode": int(ep_idx),
                "frames": frames,
                "duration_s": round(frames / 50.0, 1),
            }
            if has_fail:
                ep_data["is_failure"] = bool(ep_df["is_failure_data"].iloc[0])
            if has_infer:
                n_infer = int((ep_df["is_infer_data"] == 1).sum())
                n_teleop = int((ep_df["is_infer_data"] == 0).sum())
                ep_data["teleop_frames"] = n_teleop
                ep_data["infer_frames"] = n_infer
            if has_task:
                ep_data["task_index"] = int(ep_df["task_index"].iloc[0])
            all_eps.append(ep_data)

    return {
        "name": os.path.basename(ds_path),
        "path": ds_path,
        "total_episodes": len(all_eps),
        "total_frames": sum(e["frames"] for e in all_eps),
        "total_duration_s": round(sum(e["frames"] for e in all_eps) / 50.0, 1),
        "episodes": sorted(all_eps, key=lambda e: e["episode"]),
    }


def compute_summary(results: list[dict]) -> dict:
    """Aggregate summary across all datasets."""
    all_eps = []
    for ds in results:
        if ds is None:
            continue
        for ep in ds["episodes"]:
            ep_copy = dict(ep)
            ep_copy["dataset"] = ds["name"]
            all_eps.append(ep_copy)

    if not all_eps:
        return {"total_episodes": 0, "total_frames": 0, "total_duration_s": 0}

    frames_list = [e["frames"] for e in all_eps]

    summary = {
        "total_datasets": sum(1 for r in results if r is not None),
        "empty_datasets": sum(1 for r in results if r is None),
        "total_episodes": len(all_eps),
        "total_frames": sum(frames_list),
        "total_duration_s": round(sum(frames_list) / 50.0, 1),
    }

    # Frame distribution
    summary["frame_distribution"] = {
        "min": int(np.min(frames_list)),
        "max": int(np.max(frames_list)),
        "mean": round(np.mean(frames_list), 1),
        "median": round(np.median(frames_list), 1),
        "std": round(np.std(frames_list), 1),
    }

    # Success/failure breakdown
    if any("is_failure" in e for e in all_eps):
        fail_eps = [e for e in all_eps if e.get("is_failure")]
        succ_eps = [e for e in all_eps if not e.get("is_failure")]
        summary["success_failure"] = {
            "success": {
                "episodes": len(succ_eps),
                "frames": sum(e["frames"] for e in succ_eps),
            },
            "failure": {
                "episodes": len(fail_eps),
                "frames": sum(e["frames"] for e in fail_eps),
            },
        }

    # Teleop/infer/DAGGER breakdown
    if any("teleop_frames" in e for e in all_eps):
        pure_teleop = [e for e in all_eps if e.get("infer_frames", 0) == 0]
        pure_infer = [e for e in all_eps if e.get("teleop_frames", 0) == 0]
        dagger = [e for e in all_eps if e.get("teleop_frames", 0) > 0 and e.get("infer_frames", 0) > 0]
        summary["collection_type"] = {
            "pure_teleop": {
                "episodes": len(pure_teleop),
                "frames": sum(e["frames"] for e in pure_teleop),
            },
            "pure_infer": {
                "episodes": len(pure_infer),
                "frames": sum(e["frames"] for e in pure_infer),
            },
            "dagger": {
                "episodes": len(dagger),
                "frames": sum(e["frames"] for e in dagger),
            },
        }

    # Task grouping
    if any("task_index" in e for e in all_eps):
        task_groups = {}
        for e in all_eps:
            ti = e.get("task_index", "N/A")
            task_groups.setdefault(ti, {"episodes": 0, "frames": 0})
            task_groups[ti]["episodes"] += 1
            task_groups[ti]["frames"] += e["frames"]
        summary["task_groups"] = {str(k): v for k, v in sorted(task_groups.items())}

    return summary


def main():
    parser = argparse.ArgumentParser(description="Analyze LeRobot dataset statistics")
    parser.add_argument("--data-path", required=True, help="LeRobot dataset root directory")
    parser.add_argument("--date", required=True, help="Date in MMDD format, e.g. 0528")
    args = parser.parse_args()

    date = args.date
    if not (len(date) == 4 and date.isdigit()):
        print("Error: --date must be in MMDD format (4 digits)", file=sys.stderr)
        sys.exit(1)

    pattern = f"{args.data_path.rstrip('/')}/bi_s1_{date}_*"
    ds_dirs = sorted(glob.glob(pattern))

    if not ds_dirs:
        print(f"No datasets found matching: {pattern}", file=sys.stderr)
        sys.exit(1)

    results = []
    for ds_dir in ds_dirs:
        r = analyze_dataset(ds_dir)
        results.append(r)

    summary = compute_summary(results)

    output = {
        "date": date,
        "data_path": args.data_path,
        "datasets": [r for r in results if r is not None],
        "summary": summary,
    }

    print(json.dumps(output, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
