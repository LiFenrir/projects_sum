"""按关节角渲染 URDF 双臂,输出独立渲染视频 + 每帧 mask。

Example:
    python render_arm_video.py \
        --dataset_root /data/innov_0730_merged --episode 100 \
        --calib calib_front.json --out_dir /data/urdf_render --overlay
"""
import argparse
import json
from pathlib import Path

import imageio.v2 as imageio
import imageio.v3 as iio
import numpy as np

from episode_data import load_episode_joints
from urdf_render import DualArmRenderer

DEFAULT_URDF = "/home/kemove/INNOV/infra/robot_SDK/robot-arm-4340/urdf/urdf/urdf.urdf"


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset_root", required=True)
    p.add_argument("--episode", type=int, required=True)
    p.add_argument("--camera", default="observation.images.front")
    p.add_argument("--urdf", default=DEFAULT_URDF)
    p.add_argument("--calib", required=True, help="标定 JSON(见 urdf_render.default_calib)")
    p.add_argument("--out_dir", required=True)
    p.add_argument("--overlay", action="store_true",
                   help="额外输出叠加到原视频的渲染检查视频")
    p.add_argument("--limit", type=int, default=0, help="只渲染前 N 帧,0 = 全部")
    p.add_argument("--fps", type=int, default=30)
    return p.parse_args()


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ep_tag = f"episode_{args.episode:06d}"

    joints = load_episode_joints(
        Path(args.dataset_root) / "data" / "chunk-000" / f"{ep_tag}.parquet")
    if joints["reconstructed"]:
        print(f"reconstructed from action: {joints['reconstructed']}")

    video_path = (Path(args.dataset_root) / "videos" / "chunk-000"
                  / args.camera / f"{ep_tag}.mp4")
    probe = iio.imread(video_path, index=0)
    h, w = probe.shape[:2]
    n_frames = len(joints["left_q"])
    if args.limit:
        n_frames = min(n_frames, args.limit)

    calib = json.loads(Path(args.calib).read_text())
    renderer = DualArmRenderer(args.urdf, calib, w, h)

    render_path = out_dir / f"{ep_tag}_render.mp4"
    overlay_path = out_dir / f"{ep_tag}_overlay.mp4"
    writer = imageio.get_writer(render_path, fps=args.fps, quality=8)
    overlay_writer = (imageio.get_writer(overlay_path, fps=args.fps, quality=8)
                      if args.overlay else None)
    masks = np.zeros((n_frames, h, w), dtype=bool)

    for i in range(n_frames):
        rgb, mask, _ = renderer.render(
            joints["left_q"][i], joints["left_grip"][i],
            joints["right_q"][i], joints["right_grip"][i])
        writer.append_data(rgb)
        masks[i] = mask
        if overlay_writer is not None:
            frame = iio.imread(video_path, index=i)
            frame = frame.copy()
            frame[mask] = (0.55 * frame[mask] + 0.45 * rgb[mask]).astype(np.uint8)
            overlay_writer.append_data(frame)
        if (i + 1) % 200 == 0:
            print(f"  {i + 1}/{n_frames}", flush=True)

    writer.close()
    if overlay_writer is not None:
        overlay_writer.close()
    np.savez_compressed(out_dir / f"{ep_tag}_masks.npz", masks=masks)
    renderer.delete()
    print(f"render: {render_path}")
    if args.overlay:
        print(f"overlay: {overlay_path}")
    print(f"masks: {out_dir / (ep_tag + '_masks.npz')}")


if __name__ == "__main__":
    main()
