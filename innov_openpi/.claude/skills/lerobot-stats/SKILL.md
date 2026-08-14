---
name: lerobot-stats
description: Analyze LeRobot dataset statistics for a given MMDD date. Use this skill whenever the user asks to count, analyze, or summarize LeRobot / s1_data / bi_s1 datasets, especially with a date like "0528". Covers total episodes/frames, success vs failure breakdown, teleop/infer/DAGGER type breakdown, task grouping, frame distribution, and duration estimation. Outputs JSON.
---

# LeRobot Dataset Analyzer

Analyze LeRobot-format datasets (bi_s1_MMDD_*) for a given date.

## When to trigger

- User mentions "统计", "分析", "有多少数据", "数据量" + "lerobot" / "s1_data" / "bi_s1"
- User asks about data counts, failure rates, collection types, task breakdown for robot datasets
- User provides a MMDD date and wants to understand what data was collected that day

## Workflow

1. Ask the user for two required inputs if not already provided:
   - `--data-path`: The LeRobot dataset root directory (e.g., `robodeploy/s1_data/lerobot/`)
   - `--date`: The date in MMDD format (e.g., `0528`)

2. Run the analysis script:

```bash
python scripts/analyze_lerobot.py --data-path <path> --date <MMDD>
```

3. Parse the JSON output and present it to the user in a readable format. Include:
   - Total episodes, frames, and estimated duration (at 50Hz)
   - Success vs failure breakdown
   - Pure teleop / pure infer / DAGGER breakdown
   - Per-task grouping
   - Frame distribution stats (min/max/mean/median/std)

## Script

The analysis script is `scripts/analyze_lerobot.py`. It reads parquet files directly using pandas, so it needs Python with pandas, numpy, and pyarrow available in the current environment.
