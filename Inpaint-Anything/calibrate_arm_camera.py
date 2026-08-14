"""URDF 渲染叠加标定工具:手调相机内参与双臂基座外参。

用法:
  1. --init 生成默认标定 JSON
  2. 渲染指定帧叠加图,肉眼比对后编辑 JSON 重跑
  3. --sweep 对某参数扫描,一次出多张叠加图加快收敛

Example:
    python calibrate_arm_camera.py --dataset_root /data/innov_0730_merged \
        --episode 100 --frames 0 400 --calib calib_front.json
    python calibrate_arm_camera.py ... --sweep left.base_xyz.0=-0.5:0.1:5
"""
import argparse
import copy
import json
from pathlib import Path

import imageio.v3 as iio
import numpy as np

from episode_data import load_episode_joints
from render_arm_video import DEFAULT_URDF
from urdf_render import DualArmRenderer, default_calib


def set_nested(d, key, value):
    """按点分路径写嵌套字典,如 left.base_xyz.0。"""
    parts = key.split(".")
    for p in parts[:-1]:
        d = d[int(p)] if isinstance(d, list) else d[p]
    last = parts[-1]
    if isinstance(d, list):
        d[int(last)] = value
    else:
        d[last] = value


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset_root")
    p.add_argument("--episode", type=int, default=0)
    p.add_argument("--camera", default="observation.images.front")
    p.add_argument("--urdf", default=DEFAULT_URDF)
    p.add_argument("--calib", required=True)
    p.add_argument("--init", action="store_true", help="写入默认标定后退出")
    p.add_argument("--frames", type=int, nargs="+", default=[0])
    p.add_argument("--out_dir", default="/tmp/urdf_calib")
    p.add_argument("--sweep", type=str, default=None,
                   help="形如 left.base_xyz.0=-0.5:0.1:5 (初值:步长:个数)")
    p.add_argument("--alpha", type=float, default=0.5, help="叠加透明度")
    return p.parse_args()


def render_overlay(renderer, frame, joints, i, alpha):
    rgb, mask, _ = renderer.render(
        joints["left_q"][i], joints["left_grip"][i],
        joints["right_q"][i], joints["right_grip"][i])
    out = frame.copy()
    # 渲染区染红叠加,便于看对齐
    tint = rgb.copy()
    tint[..., 0] = 255
    out[mask] = ((1 - alpha) * frame[mask] + alpha * tint[mask]).astype(np.uint8)
    return out, rgb


def main():
    args = parse_args()
    calib_path = Path(args.calib)

    if args.init:
        calib = default_calib(848, 480)
        calib_path.write_text(json.dumps(calib, indent=2))
        print(f"wrote {calib_path}")
        return

    calib = json.loads(calib_path.read_text())
    joints = load_episode_joints(
        Path(args.dataset_root) / "data" / "chunk-000"
        / f"episode_{args.episode:06d}.parquet")
    video_path = (Path(args.dataset_root) / "videos" / "chunk-000"
                  / args.camera / f"episode_{args.episode:06d}.mp4")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    variants = [("", calib)]
    if args.sweep:
        key, spec = args.sweep.split("=")
        start, step, count = (float(x) for x in spec.split(":"))
        variants = []
        for i in range(int(count)):
            c = copy.deepcopy(calib)
            set_nested(c, key, start + i * step)
            variants.append((f"_{key.replace('.', '-')}_{i}", c))

    probe = iio.imread(video_path, index=0)
    h, w = probe.shape[:2]
    for tag, c in variants:
        renderer = DualArmRenderer(args.urdf, c, w, h)
        for fi in args.frames:
            frame = iio.imread(video_path, index=fi)
            out, rgb = render_overlay(renderer, frame, joints, fi, args.alpha)
            p = out_dir / f"ep{args.episode:06d}_f{fi:04d}{tag}.png"
            iio.imwrite(p, out)
            print(p)
        renderer.delete()


if __name__ == "__main__":
    main()
