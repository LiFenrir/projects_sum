"""世界锚定的相机标定:Nelder-Mead 最大化渲染/真实 mask IoU + 基座边线约束。

世界系:x 沿两基座连线(左→右),y 指向桌面深处,z 向上。
左基座内边(x=-0.275)在 f0 的投影 x≈242,右基座内边(x=+0.275)投影 x≈635,
基座 10cm 见方、正朝向,两臂 URDF base 相对基座台有共享 yaw 偏置 θ0。
待优化:相机位姿 6DoF + fx,fy,k1 + θ0,两臂共用,尺度由 55cm 间距锁死。

Example:
    python optimize_arm_calib.py --dataset_root /data/innov_0730_merged \
        --episode 100 --frames 0 --sam_masks /tmp/urdf_calib/sam_masks_f0.npy \
        --calib /tmp/urdf_calib/calib_front.json
"""
import argparse
import json
from pathlib import Path

import cv2
import imageio.v3 as iio
import numpy as np
from scipy.optimize import minimize

from episode_data import load_episode_joints
from render_arm_video import DEFAULT_URDF
from urdf_render import DualArmRenderer, rpy_to_matrix

# 基座内边世界坐标(z=0 为基座台顶面/URDF base 原点)
BASE_INNER = {"left": np.array([-0.275, 0.0, 0.0]),
              "right": np.array([0.275, 0.0, 0.0])}
# f0 实测内边投影像素 x
BASE_INNER_PX = {"left": 242.0, "right": 635.0}


def iou(a, b):
    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    return inter / max(union, 1)


class CalibOptimizer:
    """半分辨率渲染器 + 多帧目标 mask,世界锚定参数化优化。

    参数向量 x = [cam_xyz(3), cam_rpy(3), theta0, fx, fy, k1]:
    T_cam_base = T_cam_world @ T_world_base(theta0),两臂共享。
    """

    def __init__(self, urdf, calib, width, height, joints, frame_targets,
                 scale=0.5):
        self.scale = scale
        self.calib = json.loads(json.dumps(calib))
        c = json.loads(json.dumps(calib))
        for k in ("fx", "fy", "cx", "cy"):
            c[k] *= scale
        self.width = int(width * scale)
        self.height = int(height * scale)
        self.renderer = DualArmRenderer(urdf, c, self.width, self.height)
        self.joints = joints
        # {"left": [(frame_idx, 半分辨率 mask), ...], "right": [...]}
        s = int(1 / scale)
        self.frame_targets = {
            side: [(fi, t[::s, ::s]) for fi, t in fts]
            for side, fts in frame_targets.items()
        }

    def unpack(self, x):
        cam_pos, cam_rpy, theta0 = x[:3], x[3:6], x[6]
        fx, k1 = x[7], x[8]
        R = rpy_to_matrix(cam_rpy)[:3, :3]
        T_cw = np.eye(4)
        T_cw[:3, :3] = R
        T_cw[:3, 3] = -R @ cam_pos  # 相机世界坐标 → world→cam 平移
        return T_cw, theta0, fx, fx, k1  # 方形像素:fy = fx

    def apply(self, x):
        T_cw, theta0, fx, fy, k1 = self.unpack(x)
        for side, wx in (("left", -0.325), ("right", 0.325)):
            T_wb = rpy_to_matrix([0, 0, theta0])
            T_wb[:3, 3] = [wx, 0, 0]
            T_cb = T_cw @ T_wb
            self.renderer.arms[side].base_pose = T_cb
        cam = self.renderer.cam_node.camera
        cam.fx, cam.fy = fx, fy
        self.renderer.set_distortion(k1, 0.0)

    def score(self, x):
        self.apply(x)
        total = 0.0
        for side in ("left", "right"):
            for fi, target in self.frame_targets[side]:
                m = self.renderer.render_arm_mask(
                    side, self.joints[f"{side}_q"][fi],
                    self.joints[f"{side}_grip"][fi])
                total += iou(m, target)
        total -= self.edge_penalty(x)
        return total

    def edge_penalty(self, x):
        """基座内边投影与实测像素的偏差(归一化到 IoU 量纲)。"""
        T_cw, theta0, fx, fy, k1 = self.unpack(x)
        cam = self.renderer.cam_node.camera
        K = np.array([[fx, 0, cam.cx], [0, fy, cam.cy], [0, 0, 1]])
        rvec, _ = cv2.Rodrigues(T_cw[:3, :3])
        err = 0.0
        for side in ("left", "right"):
            px, _ = cv2.projectPoints(
                BASE_INNER[side].reshape(1, 3), rvec, T_cw[:3, 3], K,
                np.array([k1, 0.0, 0.0, 0.0]))
            err += abs(px[0, 0, 0] - BASE_INNER_PX[side] * self.scale)
        return err / 200.0  # 200px 误差 = 1 IoU

    def optimize(self, x0, sigmas, bounds, maxiter=400):
        def neg(x):
            x = np.clip(x, [b[0] for b in bounds], [b[1] for b in bounds])
            return -self.score(x)

        n = len(x0)
        simplex = np.tile(x0, (n + 1, 1))
        simplex[1:] += np.diag(sigmas)
        res = minimize(neg, np.asarray(x0, dtype=float), method="Nelder-Mead",
                       options={"maxiter": maxiter, "xatol": 1e-3,
                                "fatol": 1e-4, "initial_simplex": simplex})
        x = np.clip(res.x, [b[0] for b in bounds], [b[1] for b in bounds])
        self.apply(x)
        return -res.fun, x


def filter_arm_masks(masks, rois):
    """按基座 ROI 过滤 SAM 实例:返回 left/right 的合并目标 mask。

    rois: {"left": (x0,y0,x1,y1), "right": (...)},机械臂基座固定区域。
    """
    out = {}
    for side, (x0, y0, x1, y1) in rois.items():
        sel = np.zeros(masks.shape[1:], dtype=bool)
        for m in masks:
            if m[y0:y1, x0:x1].any():
                sel |= m
        out[side] = sel
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset_root", required=True)
    p.add_argument("--episode", type=int, required=True)
    p.add_argument("--frames", type=int, nargs="+", required=True)
    p.add_argument("--camera", default="observation.images.front")
    p.add_argument("--urdf", default=DEFAULT_URDF)
    p.add_argument("--sam_masks", nargs="+", required=True,
                   help="与 --frames 一一对应的 SAM 分割 npy(可含多实例)")
    p.add_argument("--calib", required=True)
    p.add_argument("--maxiter", type=int, default=400)
    args = p.parse_args()

    calib = json.loads(Path(args.calib).read_text())
    joints = load_episode_joints(
        Path(args.dataset_root) / "data" / "chunk-000"
        / f"episode_{args.episode:06d}.parquet")

    video_path = (Path(args.dataset_root) / "videos" / "chunk-000"
                  / args.camera / f"episode_{args.episode:06d}.mp4")
    h, w = iio.imread(video_path, index=0).shape[:2]
    # 机械臂基座 ROI:左下/右下角
    rois = {"left": (0, int(h * 0.55), int(w * 0.35), h),
            "right": (int(w * 0.65), int(h * 0.55), w, h)}

    frame_targets = {"left": [], "right": []}
    for fi, npy in zip(args.frames, args.sam_masks):
        sel = filter_arm_masks(np.load(npy), rois)
        for side in ("left", "right"):
            frame_targets[side].append((fi, sel[side]))

    opt = CalibOptimizer(args.urdf, calib, w, h, joints, frame_targets)

    # 初始猜测:相机在基座连线中点后上方,look-at 桌面中心
    eye = np.array([0.0, -0.30, 0.65])
    center = np.array([0.0, 0.25, 0.0])
    fwd = center - eye
    fwd /= np.linalg.norm(fwd)
    right = np.cross(fwd, [0, 0, 1])
    right /= np.linalg.norm(right)
    down = np.cross(right, fwd)
    R0 = np.stack([right, down, fwd], axis=0)
    ry0 = np.arcsin(np.clip(-R0[2, 0], -1, 1))
    rx0 = np.arctan2(R0[2, 1], R0[2, 2])
    rz0 = np.arctan2(R0[1, 0], R0[0, 0])

    x0 = [*eye, rx0, ry0, rz0, 0.0, 450.0, 0.0]
    sigmas = [0.05, 0.05, 0.05, 0.1, 0.1, 0.1, 0.3, 60, 0.15]
    bounds = [(-0.4, 0.4), (-0.6, 0.2), (0.3, 1.2),
              (-3.2, 3.2), (-3.2, 3.2), (-3.2, 3.2),
              (-3.2, 3.2), (200, 800), (-0.8, 0.2)]
    for rnd in range(4):
        score, x = opt.optimize(x0, sigmas, bounds, args.maxiter)
        print(f"round{rnd} score: {score:.4f}")
        print("  cam_xyz,rpy:", np.round(x[:6], 4),
              "theta0:", round(x[6], 4),
              "fx,k1:", np.round(x[7:], 2))
        x0 = x

    # 写回全分辨率 calib:参数 → 两臂 base 位姿(相机系)
    T_cw, theta0, fx, fy, k1 = opt.unpack(x)
    out = opt.calib
    out["fx"], out["fy"], out["k1"] = float(fx) / opt.scale, \
        float(fy) / opt.scale, float(k1)
    for side, wx in (("left", -0.325), ("right", 0.325)):
        T_wb = rpy_to_matrix([0, 0, theta0])
        T_wb[:3, 3] = [wx, 0, 0]
        T_cb = T_cw @ T_wb
        R = T_cb[:3, :3]
        ry = np.arcsin(np.clip(-R[2, 0], -1, 1))
        rx = np.arctan2(R[2, 1], R[2, 2])
        rz = np.arctan2(R[1, 0], R[0, 0])
        out[side]["base_xyz"] = [float(v) for v in T_cb[:3, 3]]
        out[side]["base_rpy"] = [float(rx), float(ry), float(rz)]
    Path(args.calib).write_text(json.dumps(out, indent=2))
    print(f"saved {args.calib}")


if __name__ == "__main__":
    main()
