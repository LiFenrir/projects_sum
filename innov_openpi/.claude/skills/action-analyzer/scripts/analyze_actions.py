#!/usr/bin/env python
"""Analyze policy action JSONL logs and recommend temporal smoothing parameters.

Handles two log formats:

  Server (serve_policy.py --log_file):
    One entry per inference call. `actions` = full 50-frame chunk.
    Step = inference index. State is null.
    Comparison: chunk-to-chunk (how much the policy plan changes between calls).

  Client (inspect_policy_action.py --log_file):
    One entry per control step (30Hz). `action` = current frame,
    `actions_full` = full chunk, `chunk_infer`/`chunk_frame` = position in chunk.
    State is real robot observation.
    Comparison: step-to-step + cross-chunk jump at boundaries.

Usage:
    python analyze_actions.py --server-log actions.jsonl --fps 30
    python analyze_actions.py --client-log policy_actions.jsonl --fps 30
    python analyze_actions.py --server-log a.jsonl --client-log b.jsonl --fps 30
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np


def _load(path):
    with open(path) as f:
        return [json.loads(l) for l in f if l.strip()]


def _spotlight(value):
    return f"\033[1;33m{value}\033[0m" if sys.stdout.isatty() else str(value)


# ── server-side analysis (per inference call) ──────────────────────────

def analyze_server(entries, fps):
    """Each entry = one inference call with full action chunk."""
    N = len(entries)
    chunks = np.array([e["actions"] for e in entries])          # (N, H, D)
    infer_ms = [e.get("infer_ms", 0) for e in entries]
    infer_ms = [v for v in infer_ms if v and v > 0]

    H, D = chunks.shape[1], chunks.shape[2]
    total_frames = N * H

    # within-chunk frame deltas
    chunk_deltas = np.abs(np.diff(chunks, axis=1))             # (N, H-1, D)
    cd_mean = chunk_deltas.mean(axis=(0, 1))
    cd_overall = {
        "mean": float(chunk_deltas.mean()),
        "std": float(chunk_deltas.std()),
        "p50": float(np.median(chunk_deltas)),
        "p99": float(np.percentile(chunk_deltas, 99)),
    }

    # chunk-to-chunk: action[0] between consecutive inference calls
    first_frames = chunks[:, 0, :]                             # (N, D)
    inter_chunk_deltas = np.abs(np.diff(first_frames, axis=0))  # (N-1, D)
    ic_mean = inter_chunk_deltas.mean(axis=0)
    ic_overall = {
        "mean": float(inter_chunk_deltas.mean()),
        "std": float(inter_chunk_deltas.std()),
        "p50": float(np.median(inter_chunk_deltas)),
        "p99": float(np.percentile(inter_chunk_deltas, 99)),
    }

    # ratio inter-chunk / within-chunk
    ratios = np.where(cd_mean > 1e-10, ic_mean / cd_mean, 0)
    ratio_overall = ic_overall["mean"] / cd_overall["mean"] if cd_overall["mean"] > 1e-10 else 0

    # chunk endpoint divergence
    endpoint_div = np.abs(chunks[:, -1, :] - chunks[:, 0, :])  # (N, D)
    ep_mean = endpoint_div.mean(axis=0)
    ep_max = endpoint_div.max(axis=0)

    # timing
    timing = {
        "mean_ms": float(np.mean(infer_ms)) if infer_ms else 0,
        "std_ms": float(np.std(infer_ms)) if infer_ms else 0,
        "min_ms": float(np.min(infer_ms)) if infer_ms else 0,
        "max_ms": float(np.max(infer_ms)) if infer_ms else 0,
    }

    # per-dim
    per_dim = []
    for i in range(D):
        per_dim.append({
            "dim": i,
            "chunk_mean": float(cd_mean[i]),
            "step_mean": float(ic_mean[i]),
            "ratio": float(ratios[i]),
            "endpoint_mean": float(ep_mean[i]),
            "endpoint_max": float(ep_max[i]),
            "action_mean": float(first_frames[:, i].mean()),
            "action_std": float(first_frames[:, i].std()),
        })

    # recommendations
    avg_infer_ms = timing["mean_ms"]
    rec = _recommend(avg_infer_ms, ratio_overall, fps)

    flagged = [i for i in range(D) if ratios[i] > max(2 * ratio_overall, 3)]

    return {
        "label": "Server (per inference call)",
        "entries": N,
        "total_frames": total_frames,
        "time_span_s": float(entries[-1]["timestamp"] - entries[0]["timestamp"]),
        "format": "server",
        "action_dim": D,
        "action_horizon": H,
        "timing": timing,
        "chunk_delta": cd_overall,
        "step_delta": ic_overall,
        "step_label": "chunk-to-chunk Δ  ",
        "ratio": ratio_overall,
        "per_dim": per_dim,
        "flagged_dims": flagged,
        "recommendation": rec,
        "has_state": False,
    }


# ── client-side analysis (per control step) ────────────────────────────

def analyze_client(entries, fps):
    """Each entry = one control step. action = current frame, actions_full = full chunk."""
    N = len(entries)

    actions = np.array([e["action"] for e in entries])            # (N, D)
    chunks = np.array([e["actions_full"] for e in entries])       # (N, H, D)
    states = np.array([e["state"] for e in entries])              # (N, D)
    chunk_ids = np.array([e.get("chunk_infer", 0) for e in entries])
    chunk_frames = np.array([e.get("chunk_frame", 0) for e in entries])
    infer_ms = [e.get("infer_ms", 0) for e in entries]
    infer_ms = [v for v in infer_ms if v and v > 0]

    H, D = chunks.shape[1], chunks.shape[2]

    # within-chunk frame deltas (same as server)
    chunk_deltas = np.abs(np.diff(chunks, axis=1))
    cd_mean = chunk_deltas.mean(axis=(0, 1))
    cd_overall = {
        "mean": float(chunk_deltas.mean()),
        "std": float(chunk_deltas.std()),
        "p50": float(np.median(chunk_deltas)),
        "p99": float(np.percentile(chunk_deltas, 99)),
    }

    # step-to-step: consecutive control steps
    step_deltas = np.abs(np.diff(actions, axis=0))
    sd_mean = step_deltas.mean(axis=0)
    sd_overall = {
        "mean": float(step_deltas.mean()),
        "std": float(step_deltas.std()),
        "p50": float(np.median(step_deltas)),
        "p99": float(np.percentile(step_deltas, 99)),
    }

    # cross-chunk jump: action at chunk boundary (chunk_frame goes back to 0)
    boundary_mask = np.diff(chunk_ids) != 0                       # True where chunk_id changes
    boundary_indices = np.where(boundary_mask)[0]
    cross_chunk_jumps = []
    if len(boundary_indices) > 0:
        for bi in boundary_indices:
            if bi + 1 < N:
                cross_chunk_jumps.append(np.abs(actions[bi + 1] - actions[bi]))
        cross_chunk_jumps = np.array(cross_chunk_jumps)           # (M, D)
        cc_mean = cross_chunk_jumps.mean(axis=0)
        cc_overall = {
            "mean": float(cross_chunk_jumps.mean()),
            "std": float(cross_chunk_jumps.std()),
            "p50": float(np.median(cross_chunk_jumps)),
            "p99": float(np.percentile(cross_chunk_jumps, 99)),
            "count": len(cross_chunk_jumps),
        }
        # ratio cross-chunk / within-chunk (the key metric for smoothing)
        cc_ratios = np.where(cd_mean > 1e-10, cc_mean / cd_mean, 0)
        cc_ratio_overall = cc_overall["mean"] / cd_overall["mean"] if cd_overall["mean"] > 1e-10 else 0
    else:
        cc_overall = {"mean": 0, "std": 0, "p50": 0, "p99": 0, "count": 0}
        cc_ratios = np.zeros(D)
        cc_ratio_overall = 0

    # ratio step-to-step / within-chunk
    ratios = np.where(cd_mean > 1e-10, sd_mean / cd_mean, 0)
    ratio_overall = sd_overall["mean"] / cd_overall["mean"] if cd_overall["mean"] > 1e-10 else 0

    # chunk endpoint divergence
    endpoint_div = np.abs(chunks[:, -1, :] - chunks[:, 0, :])
    ep_mean = endpoint_div.mean(axis=0)
    ep_max = endpoint_div.max(axis=0)

    # state variance
    state_std = states.std(axis=0)
    state_constant = state_std < 1e-8

    # timing
    timing = {
        "mean_ms": float(np.mean(infer_ms)) if infer_ms else 0,
        "std_ms": float(np.std(infer_ms)) if infer_ms else 0,
        "min_ms": float(np.min(infer_ms)) if infer_ms else 0,
        "max_ms": float(np.max(infer_ms)) if infer_ms else 0,
    }

    # per-dim
    per_dim = []
    for i in range(D):
        per_dim.append({
            "dim": i,
            "chunk_mean": float(cd_mean[i]),
            "step_mean": float(sd_mean[i]),
            "ratio": float(ratios[i]),
            "endpoint_mean": float(ep_mean[i]),
            "endpoint_max": float(ep_max[i]),
            "action_mean": float(actions[:, i].mean()),
            "action_std": float(actions[:, i].std()),
            "state_std": float(state_std[i]),
            "state_constant": bool(state_constant[i]),
        })

    # recommendations — use cross-chunk ratio if available (more informative)
    effective_ratio = cc_ratio_overall if cc_overall["count"] > 0 else ratio_overall
    avg_infer_ms = timing["mean_ms"]
    rec = _recommend(avg_infer_ms, effective_ratio, fps)

    flagged = [i for i in range(D) if ratios[i] > max(2 * ratio_overall, 3)]

    return {
        "label": "Client (per control step)",
        "entries": N,
        "total_frames": N,
        "time_span_s": float(entries[-1]["timestamp"] - entries[0]["timestamp"]),
        "format": "client",
        "action_dim": D,
        "action_horizon": H,
        "timing": timing,
        "chunk_delta": cd_overall,
        "step_delta": sd_overall,
        "step_label": "step-to-step Δ      ",
        "ratio": ratio_overall,
        "cross_chunk": cc_overall,
        "cross_chunk_ratio": cc_ratio_overall,
        "cross_chunk_ratios": cc_ratios if len(cc_ratios) > 0 else None,
        "per_dim": per_dim,
        "flagged_dims": flagged,
        "recommendation": rec,
        "has_state": True,
        "boundary_count": len(boundary_indices),
    }


def _recommend(avg_infer_ms, ratio, fps):
    rec_inference_rate = round(min(1000 / avg_infer_ms, fps), 1) if avg_infer_ms > 0 else fps
    rec_latency_k = max(int(np.ceil(avg_infer_ms / 1000 * fps)), 4)

    if ratio < 3:
        rec_min_smooth = 4
    elif ratio < 6:
        rec_min_smooth = 8
    else:
        rec_min_smooth = 12

    return {
        "inference_rate": rec_inference_rate,
        "latency_k": rec_latency_k,
        "min_smooth_steps": rec_min_smooth,
    }


# ── report ──────────────────────────────────────────────────────────────

def print_report(results, fps):
    print()
    print("=" * 72)
    print("  ACTION ANALYSIS REPORT")
    print("=" * 72)

    for r in results:
        print()
        print(f"── {r['label']} ──")
        print(f"  Entries: {r['entries']}  "
              f"Frames: {r['total_frames']}  "
              f"Span: {r['time_span_s']:.0f}s  "
              f"Dim: {r['action_dim']}  "
              f"Horizon: {r['action_horizon']}")

        # Timing
        t = r["timing"]
        print(f"\n  [Inference timing]")
        print(f"    mean={t['mean_ms']:.0f}ms  std={t['std_ms']:.0f}ms  "
              f"min={t['min_ms']:.0f}ms  max={t['max_ms']:.0f}ms")

        # Delta comparison
        cd = r["chunk_delta"]
        sd = r["step_delta"]
        label = r.get("step_label", "step-to-step")
        print(f"\n  [Delta comparison]")
        print(f"    {'':>20} {'mean':>10} {'std':>10} {'p50':>10} {'p99':>10}")
        print(f"    {'within-chunk':>20} {cd['mean']:10.4f} {cd['std']:10.4f} "
              f"{cd['p50']:10.4f} {cd['p99']:10.4f}")
        print(f"    {label:>20} {sd['mean']:10.4f} {sd['std']:10.4f} "
              f"{sd['p50']:10.4f} {sd['p99']:10.4f}")
        ratio_str = f"{r['ratio']:.2f}"
        print(f"    {'ratio':>20} {_spotlight(ratio_str):>10}")

        # Cross-chunk (client only)
        if r["format"] == "client" and r.get("cross_chunk", {}).get("count", 0) > 0:
            cc = r["cross_chunk"]
            print(f"\n  [Cross-chunk jump]  ({cc['count']} boundaries)")
            print(f"    {'mean':>10} {'std':>10} {'p50':>10} {'p99':>10}  {'ratio_vs_chunk':>16}")
            ccr_str = f"{r['cross_chunk_ratio']:.2f}"
            print(f"    {cc['mean']:10.4f} {cc['std']:10.4f} {cc['p50']:10.4f} "
                  f"{cc['p99']:10.4f}  {_spotlight(ccr_str):>16}")
            if r.get("cross_chunk_ratios") is not None:
                ccr = r["cross_chunk_ratios"]
                top = np.argsort(-ccr)[:3]
                print(f"    Top dims with largest cross-chunk jump ratio: "
                      f"{', '.join(f'd{i}({ccr[i]:.1f}x)' for i in top)}")

        # Per-dim
        has_state = r.get("has_state", False)
        if has_state:
            header = (f"{'dim':>4} {'chunk_Δ':>10} {'step_Δ':>10} {'ratio':>7} "
                      f"{'endpt_Δ':>9} {'act_σ':>9} {'state_σ':>9}")
        else:
            header = (f"{'dim':>4} {'chunk_Δ':>10} {'step_Δ':>10} {'ratio':>7} "
                      f"{'endpt_Δ':>9} {'act_σ':>9}")
        print(f"\n  [Per-dimension]")
        print(f"    {header}")
        print(f"    {'-' * (len(header) - 4)}")
        for d in r["per_dim"]:
            flag = " !" if d["dim"] in r.get("flagged_dims", []) else ""
            const = " CONST" if d.get("state_constant") else ""
            if has_state:
                state_s = f"{d['state_std']:.4f}{const}"
                print(f"    {d['dim']:4d} {d['chunk_mean']:10.4f} {d['step_mean']:10.4f} "
                      f"{d['ratio']:7.2f} {d['endpoint_mean']:9.4f} {d['action_std']:9.4f} "
                      f"{state_s:>9}{flag}")
            else:
                print(f"    {d['dim']:4d} {d['chunk_mean']:10.4f} {d['step_mean']:10.4f} "
                      f"{d['ratio']:7.2f} {d['endpoint_mean']:9.4f} {d['action_std']:9.4f}{flag}")

        if not has_state:
            print(f"    (state unavailable — server-side logging)")

        if r.get("flagged_dims"):
            print(f"    ⚠ flagged: {r['flagged_dims']}")

        # Recommendations
        rec = r["recommendation"]
        print(f"\n  [Recommended parameters]")
        print(f"    --inference_rate   {_spotlight(rec['inference_rate'])}")
        print(f"    --latency_k        {_spotlight(rec['latency_k'])}")
        print(f"    --min_smooth_steps {_spotlight(rec['min_smooth_steps'])}")
        print(f"    Reasoning:")
        print(f"      inference_rate:    min(1000/{t['mean_ms']:.0f}ms, {fps}fps)")
        print(f"      latency_k:         ceil({t['mean_ms']:.0f}ms / 1000 * {fps}fps)")

        if r["format"] == "client" and r.get("cross_chunk", {}).get("count", 0) > 0:
            print(f"      min_smooth_steps:  cross-chunk ratio={r['cross_chunk_ratio']:.1f} "
                  f"→ tier={rec['min_smooth_steps']}")
        else:
            print(f"      min_smooth_steps:  ratio={r['ratio']:.1f} "
                  f"→ tier={rec['min_smooth_steps']}")

    # Cross-file
    if len(results) == 2:
        print(f"\n── Cross-file check ──")
        r0, r1 = results
        dims_ok = r0["action_dim"] == r1["action_dim"]
        time_ok = abs(r0["timing"]["mean_ms"] - r1["timing"]["mean_ms"]) < 50
        cd_close = abs(r0["chunk_delta"]["mean"] - r1["chunk_delta"]["mean"]) < 0.01
        print(f"  Dims match: {dims_ok}  |  "
              f"Infer times match (Δ={abs(r0['timing']['mean_ms'] - r1['timing']['mean_ms']):.0f}ms): {time_ok}  |  "
              f"Chunk deltas close: {cd_close}")
        if dims_ok:
            print(f"  Action dim-0 mean: server={r0['per_dim'][0]['action_mean']:.4f}  "
                  f"client={r1['per_dim'][0]['action_mean']:.4f}")

    print()
    print("=" * 72)


# ── main ────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Analyze policy action JSONL logs and recommend smoothing parameters"
    )
    parser.add_argument("--server-log", type=str, default=None)
    parser.add_argument("--client-log", type=str, default=None)
    parser.add_argument("--fps", type=float, default=30.0)
    args = parser.parse_args()

    if not args.server_log and not args.client_log:
        print("ERROR: at least one of --server-log or --client-log is required")
        sys.exit(1)

    results = []
    if args.server_log:
        p = Path(args.server_log)
        if not p.exists():
            print(f"ERROR: {p} not found")
            sys.exit(1)
        entries = _load(p)
        results.append(analyze_server(entries, args.fps))

    if args.client_log:
        p = Path(args.client_log)
        if not p.exists():
            print(f"ERROR: {p} not found")
            sys.exit(1)
        entries = _load(p)
        results.append(analyze_client(entries, args.fps))

    print_report(results, args.fps)


if __name__ == "__main__":
    main()
