"""URDF 双臂渲染器:关节角 → 相机视角 RGB/mask/depth。

坐标约定:世界系即相机系(OpenCV 约定,x 右、y 下、z 前),
双臂基座位姿 T_cam_base 为待标定量。pyrender 相机取 OpenGL 约定
(看 -Z、+Y 上),内部做翻转转换。
"""

import copy
import os

import numpy as np

os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

import cv2
import pyrender
import yourdfpy

# OpenCV 相机系 → OpenGL 相机系
_CV_TO_GL = np.diag([1.0, -1.0, -1.0, 1.0])

ARM_JOINTS = ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6"]
PAW_JOINTS = ["joint_L", "joint_R"]


def load_robot(urdf_path):
    """加载 URDF,package:// 路径按 URDF 所在包目录解析。"""
    pkg_dir = urdf_path.rsplit("/urdf/", 1)[0]

    def handler(fname):
        return fname.replace("package://urdf/", pkg_dir + "/")

    return yourdfpy.URDF.load(urdf_path, filename_handler=handler)


def rpy_to_matrix(rpy):
    """固定轴 XYZ 欧拉角(弧度)→ 4x4 旋转矩阵。"""
    rx, ry, rz = rpy
    cx, sx = np.cos(rx), np.sin(rx)
    cy, sy = np.cos(ry), np.sin(ry)
    cz, sz = np.cos(rz), np.sin(rz)
    Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    R = Rz @ Ry @ Rx
    T = np.eye(4)
    T[:3, :3] = R
    return T


class ArmInstance:
    """单臂:yourdfpy FK + pyrender 节点列表。"""

    def __init__(self, scene, urdf_path, base_pose):
        self.robot = load_robot(urdf_path)
        self.nodes = {}  # 几何名 → (node, mesh, trimesh scene)
        tm_scene = self.robot.scene
        for geom_name, geom in tm_scene.geometry.items():
            mesh = pyrender.Mesh.from_trimesh(geom, smooth=False)
            node = scene.add(mesh)
            self.nodes[geom_name] = (node, mesh, tm_scene)
        self.base_pose = base_pose  # T_cam_base,4x4

    def set_config(self, cfg):
        """cfg: 关节名→值。FK 后把各连杆世界位姿写入渲染节点。"""
        self.robot.update_cfg(cfg)
        for geom_name, (node, _, tm_scene) in self.nodes.items():
            T_link_geom, _ = tm_scene.graph[geom_name]
            node.matrix = self.base_pose @ T_link_geom

    def set_visible(self, flag):
        """显隐切换(pyrender 渲染器检查的是 mesh.is_visible)。"""
        for _, mesh, _ in self.nodes.values():
            mesh.is_visible = flag


class DualArmRenderer:
    """双臂离屏渲染器。

    calib 字典:
      fx, fy, cx, cy: 相机内参(像素)
      left/right:
        base_xyz, base_rpy: 基座在相机系下的位姿
        joint_sign: 6 维 ±1,关节方向修正
        joint_offset: 6 维弧度,零位偏移
        grip_scale, grip_offset: 夹爪线性映射 paw = (g - offset) * scale
    """

    def __init__(self, urdf_path, calib, width, height):
        self.calib = calib
        self.width = width
        self.height = height
        self.scene = pyrender.Scene(ambient_light=[0.4, 0.4, 0.4])
        self.arms = {}
        for side in ("left", "right"):
            c = calib[side]
            base = rpy_to_matrix(c["base_rpy"])
            base[:3, 3] = c["base_xyz"]
            self.arms[side] = ArmInstance(self.scene, urdf_path, base)

        cam = pyrender.IntrinsicsCamera(
            fx=calib["fx"], fy=calib["fy"], cx=calib["cx"], cy=calib["cy"],
            znear=0.05, zfar=5.0,
        )
        self.cam_node = self.scene.add(cam, pose=_CV_TO_GL)
        # 主光源挂相机上,保证机械臂被照亮
        light = pyrender.DirectionalLight(color=np.ones(3), intensity=3.0)
        self.scene.add(light, pose=_CV_TO_GL)
        self.renderer = pyrender.OffscreenRenderer(width, height)
        self._dist_map = None
        self.set_distortion(calib.get("k1", 0.0), calib.get("k2", 0.0))

    def set_distortion(self, k1, k2):
        """设置 Brown 径向畸变(k1,k2),渲染结果 warp 到真实畸变图像。

        畸变像素 xd 采样自针孔渲染图的去畸变坐标,即
        真实畸变照片 ≈ warp(针孔渲染)。
        """
        if abs(k1) < 1e-9 and abs(k2) < 1e-9:
            self._dist_map = None
            return
        cam = self.cam_node.camera
        K = np.array([[cam.fx, 0, cam.cx], [0, cam.fy, cam.cy], [0, 0, 1]])
        xs, ys = np.meshgrid(np.arange(self.width, dtype=np.float32),
                             np.arange(self.height, dtype=np.float32))
        pts = np.stack([xs, ys], axis=-1).reshape(-1, 1, 2)
        undist = cv2.undistortPoints(pts, K, np.array([k1, k2, 0.0, 0.0]),
                                     P=K).reshape(self.height, self.width, 2)
        self._dist_map = (undist[..., 0].astype(np.float32),
                          undist[..., 1].astype(np.float32))

    def _warp(self, img):
        """按畸变映射 warp 渲染结果;无畸变时原样返回。"""
        if self._dist_map is None:
            return img
        return cv2.remap(img, self._dist_map[0], self._dist_map[1],
                         interpolation=cv2.INTER_LINEAR,
                         borderMode=cv2.BORDER_CONSTANT, borderValue=0)

    def _arm_cfg(self, side, q6, grip):
        c = self.calib[side]
        q = np.asarray(q6, dtype=float) * np.asarray(c["joint_sign"]) \
            + np.asarray(c["joint_offset"])
        cfg = dict(zip(ARM_JOINTS, q))
        paw = (float(grip) - c["grip_offset"]) * c["grip_scale"]
        paw = float(np.clip(paw, 0.0, 0.05))
        cfg["joint_L"] = paw
        cfg["joint_R"] = paw
        return cfg

    def render(self, left_q6, left_grip, right_q6, right_grip):
        """渲染一帧。返回 rgb (H,W,3)uint8、mask (H,W)bool、depth (H,)float。"""
        self.arms["left"].set_config(self._arm_cfg("left", left_q6, left_grip))
        self.arms["right"].set_config(self._arm_cfg("right", right_q6, right_grip))
        color, depth = self.renderer.render(self.scene)
        color = self._warp(color)
        depth = self._warp(depth.astype(np.float32))
        mask = depth > 0
        return color, mask, depth

    def render_arm_mask(self, side, q6, grip):
        """只渲染单臂 mask(另一侧隐藏),供标定优化。"""
        other = "right" if side == "left" else "left"
        self.arms[other].set_visible(False)
        self.arms[side].set_config(self._arm_cfg(side, q6, grip))
        _, depth = self.renderer.render(self.scene)
        self.arms[other].set_visible(True)
        return self._warp(depth.astype(np.float32)) > 0

    def delete(self):
        self.renderer.delete()


def default_calib(width, height):
    """初始标定猜测值,供手调起点。"""
    arm = {
        "base_xyz": [0.0, 0.0, 0.5],
        "base_rpy": [0.0, 0.0, 0.0],
        "joint_sign": [1, 1, 1, 1, 1, 1],
        "joint_offset": [0.0] * 6,
        "grip_scale": 0.1,
        "grip_offset": 0.4,
    }
    return {
        "fx": float(width), "fy": float(width),
        "cx": width / 2.0, "cy": height / 2.0,
        "left": copy.deepcopy(arm),
        "right": copy.deepcopy(arm),
    }
