#!/usr/bin/env python3
"""
从所有视频中随机抽帧,用于 labelme 标注。

每集视频按时序分前/中/后三段,随机选择 1~3 段,在被选段内随机抽帧;
所有视频的抽样总数不超过 MAX_FRAMES。输出 PNG + manifest.json。
"""

import json
import random
import subprocess
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

# =========================
# 配置
# =========================

VIDEO_DIR = Path(
    "/home/kemove/INNOV/datasets/innov_arm/innov_0730_merged/videos/chunk-000/observation.images.front"
)
OUTPUT_DIR = Path("/home/kemove/INNOV/datasets/innov_arm/labelme_sampled")
MAX_FRAMES = 1000
RANDOM_SEED = 42
NUM_SEGMENTS = 3  # 前/中/后三段


def sample_indices(n_frames: int, quota: int, rng: random.Random) -> list:
    """把视频分三段,随机选 1~3 段,在被选段内随机抽 quota 个不重复帧号。"""
    bounds = [(i * n_frames // NUM_SEGMENTS, (i + 1) * n_frames // NUM_SEGMENTS)
              for i in range(NUM_SEGMENTS)]
    k = rng.randint(1, min(NUM_SEGMENTS, quota))
    seg_ids = rng.sample(range(NUM_SEGMENTS), k)

    # 把 quota 随机分配到被选段,每段至少 1 帧
    cuts = sorted(rng.sample(range(1, quota), k - 1))
    parts = [b - a for a, b in zip([0] + cuts, cuts + [quota])]

    indices = []
    for seg, cnt in zip(seg_ids, parts):
        lo, hi = bounds[seg]
        indices.extend(rng.sample(range(lo, hi), min(cnt, hi - lo)))
    return sorted(indices)


def get_frame_count(video: Path) -> int:
    """ffprobe 读视频总帧数(头信息缺失时用 时长×fps 估算)。"""
    out = subprocess.run(
        ["ffprobe", "-v", "quiet", "-select_streams", "v:0",
         "-show_entries", "stream=nb_frames,r_frame_rate,duration",
         "-of", "json", str(video)],
        capture_output=True, text=True, check=True).stdout
    st = json.loads(out)["streams"][0]
    if st.get("nb_frames", "N/A") not in ("N/A", None):
        return int(st["nb_frames"])
    num, den = st["r_frame_rate"].split("/")
    return int(float(st["duration"]) * int(num) / int(den))


def extract_frames(video: Path, indices: list, out_dir: Path, stem: str):
    """单次 ffmpeg select 抽出指定帧,重命名为 stem_f{idx:05d}.png。"""
    if not indices:
        return
    select = "+".join(f"eq(n\\,{i})" for i in indices)
    with tempfile.TemporaryDirectory() as tmp:
        subprocess.run(
            ["ffmpeg", "-v", "error", "-i", str(video),
             "-vf", f"select='{select}'", "-fps_mode", "passthrough",
             f"{tmp}/%04d.png"],
            check=True)
        # select 按帧序输出,与升序 indices 一一对应
        for out_idx, frame_idx in enumerate(indices, 1):
            src = Path(tmp) / f"{out_idx:04d}.png"
            if src.exists():
                src.rename(out_dir / f"{stem}_f{frame_idx:05d}.png")


def process_video(args):
    vp, quota, rng_seed = args
    rng = random.Random(rng_seed)
    n_frames = get_frame_count(vp)
    picked = sample_indices(n_frames, quota, rng)
    extract_frames(vp, picked, OUTPUT_DIR, vp.stem)
    return vp.name, {"n_frames": n_frames, "sampled": picked}


def main():
    rng = random.Random(RANDOM_SEED)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    videos = sorted(VIDEO_DIR.glob("*.mp4"))
    n = len(videos)
    # 每集基础配额,余数随机分给部分视频
    quotas = [MAX_FRAMES // n] * n
    for i in rng.sample(range(n), MAX_FRAMES % n):
        quotas[i] += 1
    print(f"{n} videos, total quota {sum(quotas)}", flush=True)

    # 每集独立随机种子,保证可复现;ffmpeg 软解 AV1 较慢,并行加速
    jobs = [(vp, q, RANDOM_SEED + i) for i, (vp, q) in enumerate(zip(videos, quotas))]
    manifest = {}
    with ThreadPoolExecutor(max_workers=8) as pool:
        for done, (name, info) in enumerate(pool.map(process_video, jobs), 1):
            manifest[name] = info
            if done % 20 == 0:
                print(f"{done}/{n}", flush=True)

    (OUTPUT_DIR / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False))
    total = sum(len(v["sampled"]) for v in manifest.values())
    print(f"Done: {total} frames -> {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
