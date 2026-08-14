#!/usr/bin/env python
"""Open-loop evaluation of VLA policy on a LeRobot dataset.

Loads the first N episodes, runs inference at 50-frame windows, compares predicted
action chunks with ground-truth actions, computes MSE / smoothness / chunk-viability
metrics, plots comparison charts, and saves a JSON results file.

Usage:
    python scripts/eval_openloop.py \
        --config configs/bi_s1/pi05_finetune.yaml \
        --dir checkpoints/bi_s1_pi05_sft/bi_s1_0624/14000/ \
        --default-prompt "Grasp a single layer of the cloth with the gripper, then place the cloth onto the board"
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import time
import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore", category=UserWarning)

# Suppress noisy matplotlib font messages unless debugging
os.environ.setdefault("MPLBACKEND", "Agg")
import matplotlib.pyplot as plt  # noqa: E402
from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata  # noqa: E402

from openpi.policies import policy_config as _policy_config  # noqa: E402
from openpi.training import config as _config  # noqa: E402

# ── matplotlib style ──────────────────────────────────────────────────────────
plt.rcParams.update(
    {
        "figure.dpi": 120,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "font.size": 9,
        "axes.titlesize": 11,
    }
)

# ── helpers ───────────────────────────────────────────────────────────────────


def _to_numpy(tensor_or_array):
    """Convert torch.Tensor / numpy array to contiguous float64 numpy array."""
    if hasattr(tensor_or_array, "numpy"):
        return tensor_or_array.numpy().astype(np.float64)
    return np.asarray(tensor_or_array, dtype=np.float64)


def _build_obs(raw_frame: dict, default_prompt: str | None = None) -> dict:
    """Convert a LeRobot raw frame dict into the observation dict expected by policy.infer()."""
    images = {}
    for key, val in raw_frame.items():
        if key.startswith("observation.images."):
            cam_name = key.split("observation.images.")[1]
            images[cam_name] = _to_numpy(val).transpose(1, 2, 0)  # CHW -> HWC, float32 [0,1]

    state = _to_numpy(raw_frame["observation.state"])  # [14]
    prompt = raw_frame.get("task", default_prompt or "")

    return {"state": state, "images": images, "prompt": prompt}


def compute_mse_per_step(pred_chunks: list[np.ndarray], gt_chunks: list[np.ndarray]) -> np.ndarray:
    """Return per-horizon-step MSE averaged across all chunks. Shape: [action_horizon, action_dim]."""
    pred = np.stack(pred_chunks, axis=0)  # [N, H, D]
    gt = np.stack(gt_chunks, axis=0)
    return np.mean((pred - gt) ** 2, axis=0)  # [H, D]


def compute_intra_chunk_smoothness(chunks: list[np.ndarray]) -> np.ndarray:
    """Mean absolute step-to-step difference within each chunk → per-dimension.

    Returns: [action_dim] array (mean |a[t+1] - a[t]| across all chunks and steps).
    """
    diffs = []
    for c in chunks:  # c: [H, D]
        diffs.append(np.abs(np.diff(c, axis=0)))
    return np.mean(np.concatenate(diffs, axis=0), axis=0)  # [D]


def compute_inter_chunk_jumps(chunks: list[np.ndarray]) -> np.ndarray:
    """Mean absolute difference between end of chunk N and start of chunk N+1.

    Returns: [action_dim] array.
    """
    if len(chunks) < 2:
        return np.zeros(chunks[0].shape[-1])
    jumps = []
    for i in range(len(chunks) - 1):
        jumps.append(np.abs(chunks[i][-1] - chunks[i + 1][0]))
    return np.mean(jumps, axis=0)  # [D]


def infer_once(policy, obs: dict) -> tuple[np.ndarray, float]:
    """Run a single inference. Returns (actions [H,D], infer_time_ms)."""
    result = policy.infer(obs)
    return np.asarray(result["actions"], dtype=np.float64), result["policy_timing"]["infer_ms"]


# ── main ──────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description="Open-loop VLA policy evaluation")
    parser.add_argument("--config", required=True, help="Path to YAML training config")
    parser.add_argument("--dir", required=True, help="Checkpoint directory containing model.safetensors")
    parser.add_argument("--default-prompt", default=None, help="Default task prompt if dataset has no task field")
    parser.add_argument("--num-episodes", type=int, default=5, help="Number of episodes to evaluate (default: 5)")
    parser.add_argument("--window-size", type=int, default=50, help="Window size in frames (default: 50)")
    parser.add_argument("--output", default="eval_openloop_results", help="Output prefix for .png and .json files")
    parser.add_argument("--device", default=None, help="PyTorch device (e.g., cuda:0, cuda:1, cpu). Auto-detected if not set.")
    parser.add_argument("--action-chunk", type=int, default=None, help="Number of action steps to use (truncates model output). Default: full action_horizon.")
    parser.add_argument("--warmup", action="store_true", default=True, help="Run a warm-up inference before timing")
    parser.add_argument("--no-warmup", dest="warmup", action="store_false", help="Skip warm-up inference")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    logger = logging.getLogger("eval_openloop")

    # ── Load policy ───────────────────────────────────────────────────────
    logger.info("Loading config from %s", args.config)
    train_config = _config.load_config(args.config)
    action_horizon = train_config.model.action_horizon  # 16
    action_dim = 14  # bi_s1 – hard-coded by LeRobotOutputs truncation

    logger.info("Creating policy from checkpoint %s …", args.dir)
    policy = _policy_config.create_trained_policy(
        train_config, args.dir, default_prompt=args.default_prompt, pytorch_device=args.device,
        action_chunk=args.action_chunk,
    )

    # ── Load dataset ──────────────────────────────────────────────────────
    repo_id = train_config.data.repo_id
    logger.info("Loading dataset metadata from %s", repo_id)
    ds_meta = LeRobotDatasetMetadata(repo_id)
    fps = ds_meta.fps

    logger.info("Loading raw LeRobot dataset (delta_timestamps for %d future actions)", action_horizon)
    raw_ds = LeRobotDataset(
        repo_id,
        delta_timestamps={"action": [float(t) / fps for t in range(action_horizon)]},
    )

    num_episodes = min(args.num_episodes, len(ds_meta.episodes))
    logger.info("Dataset: %d total episodes, using first %d", len(ds_meta.episodes), num_episodes)

    # ── Warm-up inference (triggers torch.compile JIT) ────────────────────
    if args.warmup:
        logger.info(">>> Running warm-up inference (torch.compile JIT — may take minutes) …")
        warmup_frame = raw_ds[0]
        warmup_obs = _build_obs(warmup_frame, args.default_prompt)
        t0 = time.monotonic()
        infer_once(policy, warmup_obs)
        logger.info(">>> Warm-up complete in %.0f s", time.monotonic() - t0)

    # ── Collect predictions ───────────────────────────────────────────────
    pred_chunks: list[np.ndarray] = []  # each: [H, 14]
    gt_chunks: list[np.ndarray] = []  # each: [H, 14]
    infer_times: list[float] = []

    total_windows = 0
    for ep_idx in range(num_episodes):
        ep_from = int(raw_ds.episode_data_index["from"][ep_idx])
        ep_to = int(raw_ds.episode_data_index["to"][ep_idx])

        # Walk through the episode in non-overlapping action_horizon-step windows,
        # capped at window_size from the episode end.
        window_step = action_horizon  # 16-frame stride = no overlap
        for ws in range(ep_from, max(ep_from, ep_to - args.window_size), window_step):
            frame = raw_ds[ws]
            obs = _build_obs(frame, args.default_prompt)
            gt = _to_numpy(frame["action"])  # [H, 14]

            pred, t_ms = infer_once(policy, obs)
            pred_chunks.append(pred)
            gt_chunks.append(gt)
            infer_times.append(t_ms)
            total_windows += 1

        logger.info(
            "Episode %d: frames %d-%d → %d windows",
            ep_idx,
            ep_from,
            ep_to,
            (ep_to - ep_from) // window_step if ep_to > ep_from else 0,
        )

    infer_times = np.array(infer_times)
    logger.info("Collected %d inference windows across %d episodes", total_windows, num_episodes)

    # ── Compute metrics ───────────────────────────────────────────────────

    # 1. MSE
    mse_per_step = compute_mse_per_step(pred_chunks, gt_chunks)  # [H, D]
    mean_mse_over_horizon = mse_per_step.mean(axis=0)  # [D] – per-dim MSE averaged over horizon
    overall_mse = float(mse_per_step.mean())

    # 2. Smoothness
    intra_smooth = compute_intra_chunk_smoothness(pred_chunks)  # [D]
    inter_jump = compute_inter_chunk_jumps(pred_chunks)  # [D]
    smoothness_ratio = inter_jump / (intra_smooth + 1e-8)  # [D]
    mean_ratio = float(np.mean(smoothness_ratio))

    # 3. Inference timing
    timing = {
        "mean_ms": float(np.mean(infer_times)),
        "std_ms": float(np.std(infer_times)),
        "min_ms": float(np.min(infer_times)),
        "max_ms": float(np.max(infer_times)),
        "p50_ms": float(np.percentile(infer_times, 50)),
        "p95_ms": float(np.percentile(infer_times, 95)),
        "p99_ms": float(np.percentile(infer_times, 99)),
    }

    # 4. Smoothing parameter recommendations
    control_period_ms = 1000.0 / fps  # ~33 ms at 30 Hz
    rec_latency_k = max(1, int(np.ceil(timing["mean_ms"] / control_period_ms)))
    if mean_ratio < 2:
        rec_min_smooth = 4
    elif mean_ratio < 5:
        rec_min_smooth = 8
    else:
        rec_min_smooth = 12
    rec_inference_rate = max(1, int(1000.0 / timing["mean_ms"]))  # Hz, capped at reasonable max

    recommendations = {
        "latency_k": rec_latency_k,
        "min_smooth_steps": rec_min_smooth,
        "inference_rate_hz": rec_inference_rate,
        "smoothness_ratio": round(mean_ratio, 3),
        "intra_chunk_smoothness_mean": round(float(np.mean(intra_smooth)), 5),
        "inter_chunk_jump_mean": round(float(np.mean(inter_jump)), 5),
    }

    # 5. Action chunk viability
    cumulative_mse = np.array([float(mse_per_step[: h + 1].mean()) for h in range(action_horizon)])
    # Baseline: MSE of the first 4 steps (near-term predictions are most reliable)
    baseline_mse = float(mse_per_step[:4].mean()) if action_horizon >= 4 else float(mse_per_step.mean())
    # Find the last horizon step where cumulative MSE ≤ 2× baseline
    viable_mask = cumulative_mse <= baseline_mse * 2.0
    usable_chunk = int(np.argmin(viable_mask)) if not viable_mask[-1] else action_horizon
    if usable_chunk == 0:
        usable_chunk = action_horizon  # fallback if all below threshold

    chunk_viability = {
        "action_horizon": action_horizon,
        "baseline_mse_first_4": round(baseline_mse, 6),
        "recommended_chunk": usable_chunk,
        "cumulative_mse_by_horizon": [round(float(v), 6) for v in cumulative_mse],
    }

    # ── Print report ──────────────────────────────────────────────────────
    joint_names = [
        "L_j1", "L_j2", "L_j3", "L_j4", "L_j5", "L_j6", "L_grip",
        "R_j1", "R_j2", "R_j3", "R_j4", "R_j5", "R_j6", "R_grip",
    ]

    print("\n" + "=" * 72)
    print("  OPEN-LOOP EVALUATION REPORT")
    print("=" * 72)
    print(f"  Episodes:         {num_episodes}")
    print(f"  Windows:          {total_windows}")
    print(f"  Action horizon:   {action_horizon}")
    print(f"  Action dim:       {action_dim}")
    print(f"  Overall MSE:      {overall_mse:.6f}")
    print()
    print("  Per-joint MSE:")
    for i, name in enumerate(joint_names):
        print(f"    {name:>8s}: {mean_mse_over_horizon[i]:.6f}")
    print()
    print("  Inference timing (ms):")
    for k, v in timing.items():
        print(f"    {k:>8s}: {v:7.1f}")
    print()
    print("  Smoothing recommendations:")
    for k, v in recommendations.items():
        print(f"    {k:>30s}: {v}")
    print()
    print("  Chunk viability:")
    for k, v in chunk_viability.items():
        if isinstance(v, list):
            print(f"    {k:>30s}: {v[:6]}...")
        else:
            print(f"    {k:>30s}: {v}")
    print("=" * 72)

    # ── Plots ─────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(3, 2, figsize=(14, 15))
    fig.suptitle("Open-Loop VLA Evaluation", fontsize=14, fontweight="bold")

    # [1] MSE vs Horizon Position
    ax = axes[0, 0]
    h_mean = mse_per_step.mean(axis=1)  # [H] – mean over dims
    h_std = mse_per_step.std(axis=1)
    xs = np.arange(1, action_horizon + 1)
    ax.plot(xs, h_mean, "b-o", markersize=4, label="Mean MSE")
    ax.fill_between(xs, h_mean - h_std, h_mean + h_std, alpha=0.2)
    ax.axvline(usable_chunk, color="red", linestyle="--", alpha=0.6, label=f"Recommended chunk={usable_chunk}")
    ax.set_xlabel("Horizon step")
    ax.set_ylabel("MSE")
    ax.set_title("MSE vs Horizon Position")
    ax.legend(fontsize=8)

    # [2] Per-Joint MSE
    ax = axes[0, 1]
    colors = ["#1f77b4"] * 6 + ["#ff7f0e"] + ["#2ca02c"] * 6 + ["#d62728"]
    bars = ax.bar(range(action_dim), mean_mse_over_horizon, color=colors, edgecolor="white", linewidth=0.5)
    ax.set_xticks(range(action_dim))
    ax.set_xticklabels(joint_names, rotation=45, ha="right", fontsize=7)
    ax.set_ylabel("MSE")
    ax.set_title("Per-Joint MSE")
    # Legend for arm colors
    from matplotlib.patches import Patch
    ax.legend(
        handles=[Patch(facecolor="#1f77b4", label="Left arm"), Patch(facecolor="#2ca02c", label="Right arm")],
        fontsize=8,
    )

    # [3] Action trajectory comparison (pick 3 dims: left j1, left grip, right j1)
    ax = axes[1, 0]
    demo_dims = [0, 6, 7]  # L_j1, L_grip, R_j1
    demo_names = [joint_names[d] for d in demo_dims]
    if len(pred_chunks) > 0:
        # Show first 5 chunks concatenated for a longer trajectory view
        n_show = min(5, len(pred_chunks))
        pred_concat = np.concatenate(pred_chunks[:n_show], axis=0)  # [n_show*H, D]
        gt_concat = np.concatenate(gt_chunks[:n_show], axis=0)
        t = np.arange(len(pred_concat))
        for d, name in zip(demo_dims, demo_names):
            ax.plot(t, pred_concat[:, d], linewidth=0.8, alpha=0.7, label=f"{name} pred")
            ax.plot(t, gt_concat[:, d], linewidth=0.8, alpha=0.7, linestyle="--", label=f"{name} GT")
        ax.set_xlabel("Step (concatenated chunks)")
        ax.set_ylabel("Action value")
        ax.set_title("Action Trajectory: Pred vs GT (first 5 chunks)")
        ax.legend(fontsize=7, ncol=3, loc="upper right")

    # [4] Inference time distribution
    ax = axes[1, 1]
    ax.hist(infer_times, bins=30, color="steelblue", edgecolor="white", alpha=0.8)
    ax.axvline(timing["mean_ms"], color="red", linestyle="--", label=f"mean={timing['mean_ms']:.0f}ms")
    ax.axvline(timing["p95_ms"], color="orange", linestyle="--", label=f"p95={timing['p95_ms']:.0f}ms")
    ax.set_xlabel("Inference time (ms)")
    ax.set_ylabel("Frequency")
    ax.set_title("Inference Time Distribution")
    ax.legend(fontsize=8)

    # [5] Intra-chunk smoothness vs inter-chunk jump
    ax = axes[2, 0]
    x_bar = np.arange(action_dim)
    w = 0.35
    ax.bar(x_bar - w / 2, intra_smooth, w, label="Intra-chunk Δ", color="steelblue", edgecolor="white", linewidth=0.5)
    ax.bar(x_bar + w / 2, inter_jump, w, label="Inter-chunk jump", color="coral", edgecolor="white", linewidth=0.5)
    ax.set_xticks(x_bar)
    ax.set_xticklabels(joint_names, rotation=45, ha="right", fontsize=7)
    ax.set_ylabel("Mean |Δaction|")
    ax.set_title(f"Intra-chunk Smoothness vs Inter-chunk Jump  (ratio={mean_ratio:.2f})")
    ax.legend(fontsize=8)

    # [6] Chunk viability
    ax = axes[2, 1]
    ax.plot(xs, cumulative_mse, "b-o", markersize=4, label="Cumulative MSE")
    ax.axhline(baseline_mse * 2.0, color="gray", linestyle=":", alpha=0.6, label="2× baseline")
    ax.axvline(usable_chunk, color="red", linestyle="--", alpha=0.6, label=f"Recommended: {usable_chunk}")
    ax.set_xlabel("Chunk length (horizon steps)")
    ax.set_ylabel("Cumulative MSE")
    ax.set_title(f"Action Chunk Viability (baseline MSE={baseline_mse:.4f})")
    ax.legend(fontsize=8)

    plt.tight_layout()
    png_path = f"{args.output}.png"
    fig.savefig(png_path, dpi=150, bbox_inches="tight")
    logger.info("Saved plot to %s", png_path)

    # ── Save JSON ─────────────────────────────────────────────────────────
    results = {
        "num_episodes": num_episodes,
        "num_windows": total_windows,
        "action_horizon": action_horizon,
        "action_dim": action_dim,
        "overall_mse": round(overall_mse, 6),
        "per_joint_mse": [round(float(v), 6) for v in mean_mse_over_horizon],
        "mse_per_step": [[round(float(v), 6) for v in row] for row in mse_per_step],
        "timing": timing,
        "smoothing_recommendations": recommendations,
        "chunk_viability": chunk_viability,
        "per_joint_intra_smoothness": [round(float(v), 6) for v in intra_smooth],
        "per_joint_inter_jump": [round(float(v), 6) for v in inter_jump],
        "per_joint_smoothness_ratio": [round(float(v), 4) for v in smoothness_ratio],
    }
    json_path = f"{args.output}.json"
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info("Saved results to %s", json_path)

    print(f"\nResults saved: {png_path}, {json_path}")


if __name__ == "__main__":
    main()
