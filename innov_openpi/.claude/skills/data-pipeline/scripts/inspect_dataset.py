#!/usr/bin/env python3
"""
LeRobot 数据集质量检查工具。

对指定数据集进行四个维度的检查：
  1. 视频有效性 — ffprobe 读取每个视频的实际帧数
  2. 帧对齐 — parquet 行数 vs 视频帧数是否一致，各视图间是否一致
  3. index 列连续性 — 全局 index 是否从 0 开始连续无断点
  4. 数据分布 — 成功/失败/推理类型统计

用法:
    python inspect_dataset.py <dataset_path>
    python inspect_dataset.py <dataset_path> --fix-broken  # 自动删除异常 episode
    python inspect_dataset.py <dataset_path> --json        # JSON 格式输出
"""

import argparse
import json
import subprocess
import sys
from collections import Counter
from pathlib import Path

import pyarrow.parquet as pq


def get_video_nb_frames(vpath: Path) -> int | None:
    """ffprobe 获取视频实际帧数。"""
    r = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=nb_frames", "-of", "csv=p=0", str(vpath)],
        capture_output=True, text=True, timeout=30,
    )
    if r.returncode == 0 and r.stdout.strip():
        try:
            return int(r.stdout.strip())
        except ValueError:
            pass
    # fallback: -count_frames
    r2 = subprocess.run(
        ["ffprobe", "-v", "error", "-count_frames", "-select_streams", "v:0",
         "-show_entries", "stream=nb_read_frames", "-of", "csv=p=0", str(vpath)],
        capture_output=True, text=True, timeout=30,
    )
    if r2.returncode == 0 and r2.stdout.strip():
        try:
            return int(r2.stdout.strip())
        except ValueError:
            pass
    return None


def check_videos(ds: Path, episodes: list[dict], video_keys: list[str]) -> dict:
    """检查视频有效性。返回 {ep_idx: {vk: n_frames or error}}"""
    result = {}
    total, ok, missing, corrupt = 0, 0, 0, 0
    for ep in episodes:
        ep_idx = ep["episode_index"]
        chunk = ep_idx // 1000
        result[ep_idx] = {}
        for vk in video_keys:
            total += 1
            vpath = ds / f"videos/chunk-{chunk:03d}/{vk}/episode_{ep_idx:06d}.mp4"
            if not vpath.exists():
                result[ep_idx][vk] = "missing"
                missing += 1
            else:
                nf = get_video_nb_frames(vpath)
                if nf is None:
                    result[ep_idx][vk] = "corrupt"
                    corrupt += 1
                else:
                    result[ep_idx][vk] = nf
                    ok += 1
    return {"detail": result, "total": total, "ok": ok, "missing": missing, "corrupt": corrupt}


def check_alignment(ds: Path, episodes: list[dict], video_keys: list[str],
                    video_info: dict) -> dict:
    """检查 parquet 行数与视频帧数是否对齐。"""
    aligned, misaligned_pq, misaligned_view, skipped = 0, 0, 0, 0
    deltas = []
    details = []

    for ep in episodes:
        ep_idx = ep["episode_index"]
        chunk = ep_idx // 1000
        pq_path = ds / f"data/chunk-{chunk:03d}/episode_{ep_idx:06d}.parquet"
        n_pq = pq.read_metadata(str(pq_path)).num_rows

        # 收集各视图的有效帧数
        vframes = {}
        vi = video_info["detail"].get(ep_idx, {})
        for vk in video_keys:
            vf = vi.get(vk)
            if isinstance(vf, int):
                vframes[vk] = vf

        if not vframes:
            skipped += 1
            details.append({"episode": ep_idx, "issue": "no_valid_video"})
            continue

        ref_vk = list(vframes.keys())[0]
        ref_n = vframes[ref_vk]
        ep_ok = True

        # 各视图间一致性
        for vk, nf in vframes.items():
            if nf != ref_n:
                details.append({"episode": ep_idx, "issue": "cross_view",
                                "detail": f"{ref_vk}={ref_n} vs {vk}={nf}"})
                misaligned_view += 1
                ep_ok = False

        # parquet vs 视频
        delta = n_pq - ref_n
        deltas.append(delta)
        if delta != 0:
            details.append({"episode": ep_idx, "issue": "pq_vs_video",
                            "parquet": n_pq, "video": ref_n, "delta": delta})
            misaligned_pq += 1
            ep_ok = False

        if ep_ok:
            aligned += 1

    return {
        "aligned": aligned, "misaligned_pq": misaligned_pq,
        "misaligned_view": misaligned_view, "skipped": skipped,
        "delta_distribution": dict(Counter(deltas)), "details": details,
    }


def check_index(ds: Path, episodes: list[dict]) -> dict:
    """检查 index 列在该数据集内是否连续。"""
    prev_max = -1
    gaps = []
    for ep in sorted(episodes, key=lambda e: e["episode_index"]):
        ep_idx = ep["episode_index"]
        chunk = ep_idx // 1000
        pq_path = ds / f"data/chunk-{chunk:03d}/episode_{ep_idx:06d}.parquet"
        if not pq_path.exists():
            gaps.append({"episode": ep_idx, "issue": "missing_parquet"})
            continue
        idx_arr = pq.read_table(str(pq_path), columns=["index"]).column("index").to_numpy()

        # 跨 episode 连续性
        if prev_max >= 0 and idx_arr[0] != prev_max + 1:
            gaps.append({"episode": ep_idx, "issue": "cross_episode_gap",
                         "expected": prev_max + 1, "actual": int(idx_arr[0])})

        # episode 内部连续性
        for i in range(1, len(idx_arr)):
            if idx_arr[i] != idx_arr[i - 1] + 1:
                gaps.append({"episode": ep_idx, "issue": "internal_gap",
                             "pos": i, "prev": int(idx_arr[i - 1]), "curr": int(idx_arr[i])})

        prev_max = int(idx_arr[-1])

    return {"contiguous": len(gaps) == 0, "last_index": prev_max, "gaps": gaps}


def check_distribution(ds: Path, episodes: list[dict]) -> dict:
    """统计成功/失败/推理类型分布。"""
    success_ep = fail_ep = 0
    success_fr = fail_fr = 0
    infer_ep = Counter()
    infer_fr = Counter()

    for ep in episodes:
        ep_idx = ep["episode_index"]
        chunk = ep_idx // 1000
        table = pq.read_table(str(ds / f"data/chunk-{chunk:03d}/episode_{ep_idx:06d}.parquet"))
        n = table.num_rows
        cols = set(table.column_names)

        fail_arr = table.column("is_failure_data").to_numpy() if "is_failure_data" in cols else None
        infer_arr = table.column("is_infer_data").to_numpy() if "is_infer_data" in cols else None

        is_fail = bool(fail_arr.mean() > 0.5) if fail_arr is not None else False
        if is_fail:
            fail_ep += 1; fail_fr += n
        else:
            success_ep += 1; success_fr += n

        if infer_arr is not None:
            ni = int((infer_arr == 1).sum())
            nt = n - ni
            key = "mixed(DAGGER)" if (ni > 0 and nt > 0) else ("pure_infer" if ni > 0 else "pure_teleop")
            infer_ep[key] += 1; infer_fr[key] += n

    return {
        "success": {"episodes": success_ep, "frames": success_fr},
        "failure": {"episodes": fail_ep, "frames": fail_fr},
        "infer_type_episodes": dict(infer_ep),
        "infer_type_frames": dict(infer_fr),
    }


def print_report(video, alignment, index_ok, distribution, info):
    """格式化打印检查报告。"""
    print("=" * 60)
    print(f"数据集: {info.get('robot_type', 'unknown')}")
    print(f"Episodes: {info['total_episodes']}, Frames: {info['total_frames']}")
    print("=" * 60)

    print("\n[1] 视频有效性")
    print(f"  总视频: {video['total']}, 有效: {video['ok']}, 缺失: {video['missing']}, 损坏: {video['corrupt']}")
    broken = []
    for ep_idx, views in video["detail"].items():
        for vk, v in views.items():
            if isinstance(v, str):
                broken.append((ep_idx, vk.split(".")[-1], v))
    if broken:
        print(f"  异常详情:")
        for ep, view, reason in broken:
            print(f"    ep={ep} {view}: {reason}")

    print("\n[2] 帧对齐")
    print(f"  对齐: {alignment['aligned']}, parquet偏移: {alignment['misaligned_pq']}, 视图不一致: {alignment['misaligned_view']}, 跳过: {alignment['skipped']}")
    if alignment["delta_distribution"]:
        print(f"  帧差分布: {alignment['delta_distribution']}")
    if alignment["details"]:
        print(f"  异常详情:")
        for d in alignment["details"]:
            print(f"    {d}")

    print("\n[3] index 列连续性")
    if index_ok["contiguous"]:
        print(f"  ✓ 完全连续, 最后 index: {index_ok['last_index']}")
    else:
        print(f"  ✗ 发现 {len(index_ok['gaps'])} 处断点:")
        for g in index_ok["gaps"]:
            print(f"    {g}")

    print("\n[4] 数据分布")
    s = distribution["success"]; f = distribution["failure"]
    total_ep = s["episodes"] + f["episodes"]
    print(f"  成功: {s['episodes']} episodes ({s['episodes']*100//total_ep}%), {s['frames']} frames")
    print(f"  失败: {f['episodes']} episodes ({f['episodes']*100//total_ep}%), {f['frames']} frames")
    print(f"  推理类型(ep): {distribution['infer_type_episodes']}")
    print(f"  推理类型(fr):  {distribution['infer_type_frames']}")

    # 汇总判断
    all_ok = (video["missing"] == 0 and video["corrupt"] == 0
              and alignment["misaligned_pq"] == 0 and alignment["misaligned_view"] == 0
              and alignment["skipped"] == 0 and index_ok["contiguous"])
    print("\n" + "=" * 60)
    if all_ok:
        print("✓ 数据集全部检查通过")
    else:
        print("⚠ 发现问题，建议修正后重新检查")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="LeRobot 数据集质量检查工具")
    parser.add_argument("dataset", type=str, help="数据集路径")
    parser.add_argument("--json", action="store_true", help="JSON 格式输出")
    parser.add_argument("--fix-broken", action="store_true",
                        help="自动删除有视频问题的 episode（需 delete_episodes.py 可用）")
    args = parser.parse_args()

    ds = Path(args.dataset).resolve()
    if not (ds / "meta/info.json").exists():
        print(f"错误: 不是有效的 LeRobot 数据集 ({ds})", file=sys.stderr)
        sys.exit(1)

    with open(ds / "meta/info.json") as f:
        info = json.load(f)
    with open(ds / "meta/episodes.jsonl") as f:
        episodes = [json.loads(l) for l in f if l.strip()]

    features = info.get("features", {})
    video_keys = [k for k, v in features.items() if v.get("dtype") == "video"]

    # 执行检查
    video = check_videos(ds, episodes, video_keys)
    alignment = check_alignment(ds, episodes, video_keys, video)
    index_ok = check_index(ds, episodes)
    distribution = check_distribution(ds, episodes)

    if args.json:
        print(json.dumps({
            "video": {k: v for k, v in video.items() if k != "detail"},
            "video_broken": [
                {"episode": ep, "view": vk.split(".")[-1], "error": v}
                for ep, views in video["detail"].items()
                for vk, v in views.items() if isinstance(v, str)
            ],
            "alignment": alignment,
            "index": index_ok,
            "distribution": distribution,
        }, indent=2, ensure_ascii=False))
    else:
        print_report(video, alignment, index_ok, distribution, info)

    # --fix-broken: 删除有视频问题的 episode
    if args.fix_broken:
        broken_eps = set()
        for ep_idx, views in video["detail"].items():
            for vk, v in views.items():
                if isinstance(v, str):
                    broken_eps.add(ep_idx)

        if not broken_eps:
            print("没有需要修复的 episode")
            return

        print(f"\n将删除 {len(broken_eps)} 个异常 episode: {sorted(broken_eps)}")

        # 调用 delete_episodes.py
        delete_script = Path(__file__).resolve().parents[3] / "src/robodeploy/scripts/delete_episodes.py"
        if not delete_script.exists():
            print(f"错误: 找不到 delete_episodes.py ({delete_script})", file=sys.stderr)
            sys.exit(1)
        cmd = [sys.executable, str(delete_script), str(ds)] + [str(e) for e in sorted(broken_eps)]
        subprocess.run(cmd)

        # 删除后重新检查 index
        print("\n重新检查 index 连续性...")
        with open(ds / "meta/info.json") as f:
            info2 = json.load(f)
        with open(ds / "meta/episodes.jsonl") as f:
            eps2 = [json.loads(l) for l in f if l.strip()]
        idx2 = check_index(ds, eps2)
        if idx2["contiguous"]:
            print("  ✓ index 连续")
        else:
            print(f"  ✗ 仍有断点: {idx2['gaps']}")


if __name__ == "__main__":
    main()
