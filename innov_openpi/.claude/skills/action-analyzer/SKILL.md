---
name: action-analyzer
description: >
  Analyze policy action JSONL log files to compute within-chunk smoothness, step-to-step
  jump amplitude, chunk endpoint divergence, per-dimension stats, and state variance.
  Recommend optimal temporal smoothing parameters (latency_k, min_smooth_steps, inference_rate)
  for StreamActionBuffer in record_s1_inference.py. Trigger when user provides action
  log files, asks about action analysis, smoothing parameters, latency_k, min_smooth_steps,
  inference_rate, or wants to tune temporal smoothing for policy deployment.
---

# Action Analyzer

Analyze policy action JSONL logs produced by `serve_policy.py --log_file` or
`inspect_policy_action.py --log_file`. Produces a structured report and recommends
temporal smoothing parameters.

## Quick start

```bash
python .claude/skills/action-analyzer/scripts/analyze_actions.py \
    --server-log policy_and_value/policy_offline_and_value/actions.jsonl \
    --client-log deploy/data_collection/policy_actions.jsonl \
    --fps 30
```

Either `--server-log` or `--client-log` can be omitted if only one file exists.

## Report sections

The script outputs these sections in order:

1. **Data overview** — entry counts, time spans, field presence
2. **Inference timing** — mean/std/min/max infer_ms
3. **Within-chunk smoothness** — per-dim |action[t] - action[t-1]| inside each 50-frame chunk
4. **Step-to-step jumps** — per-dim |action[0]_step_n - action[0]_step_n-1|
5. **Chunk endpoint divergence** — per-dim |action[-1] - action[0]| (how far the predicted trajectory goes)
6. **State variance** — per-dim state_std, flags constant dimensions
7. **ratio step/chunk** — key metric: how much larger step jumps are vs chunk internal variation
8. **Parameter recommendation** — suggested values for `latency_k`, `min_smooth_steps`, `inference_rate`

## Parameter recommendation logic

Given `fps` (control frequency, default 30) and `action_horizon` (chunk length, e.g. 50):

### inference_rate
```
inference_rate = min(1000 / infer_ms_avg, fps)
```
Cannot exceed what the model can sustain. Typical: ~400ms → 2.5 Hz.

### latency_k
```
latency_k = max(ceil(infer_ms_avg / 1000 * fps), 4)
```
How many control steps pass during one inference call. These leading actions in
each new chunk are stale and should be dropped.

### min_smooth_steps
Based on **step/chunk ratio** (how much larger step-to-step jumps are vs within-chunk frame deltas):

| ratio | min_smooth_steps | reasoning |
|-------|-----------------|-----------|
| < 3   | 4 | chunks are consistent across steps, light smoothing |
| 3–6   | 8 | moderate chunk-to-chunk variation |
| > 6   | 12 | large jumps between chunks, need heavy blending |

If some individual dimensions have outsized step/chunk ratio (> 2x the mean),
they are flagged in the report — consider checking whether those joints need
mechanical attention or whether the policy is uncertain about them.

## Interpreting common patterns

**State constant across all steps** → robot is stationary during logging.
Actions will appear consistent but the analysis is limited — the policy isn't
responding to state changes. Re-run with `inspect_policy_action.py` sending
actions to the robot.

**Large chunk endpoint divergence on specific dims** → policy predicts
substantial movement on those joints over the 50-frame horizon. These joints
benefit more from temporal smoothing.

**Gripper dims (typically 6, 13) near 1.0–1.3** → gripper open. Near 0.0 →
gripper closed. If gripper action oscillates, the policy may be uncertain about
grasp timing.
