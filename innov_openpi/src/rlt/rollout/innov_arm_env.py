"""innov_arm 真机环境接线(直接驱动,不经 WebSocket 桥接)。

把 robodeploy 的 innov_arm 驱动接入 RobotEnv 的 step_fn/reset_fn/get_obs_fn,
供 Stage 2 在线 RL 直接控制真机。支持单臂(innov_arm_v1)与双臂(bi_innov_arm_v1)。

机器人参数通道(优先级:工厂 kwargs > 环境变量):
- robot_type / INNOV_ARM_ROBOT_TYPE       "innov_arm_v1" | "bi_innov_arm_v1"
- port / INNOV_ARM_PORT                    单臂串口
- left_port,right_port / INNOV_ARM_LEFT_PORT,INNOV_ARM_RIGHT_PORT  双臂串口
- cameras / INNOV_ARM_CAMERAS               JSON,例:
    {"front": {"type": "realsense", "width": 960, "height": 848, "fps": 30},
     "left_wrist": {"type": "opencv", "source": "/dev/video1"}}
- control_hz / INNOV_ARM_CONTROL_HZ        控制频率(默认 15)

观测组装为 LeRobot 直接格式:{"state": [d], "images": {cam: RGB HWC}, "prompt"},
与 LeRobotInputs 输入契约一致(相机输出即为 RGB HWC,resize 由模型 transform 链处理)。
"""

from __future__ import annotations

import json
import logging
import os
import time
from typing import Any

import numpy as np
from numpy.typing import NDArray

from rlt.rollout.intervention import InterventionManager, InterventionResult
from rlt.rollout.keyboard_ctrl import KeyboardCtrl
from rlt.rollout.robot_env import RobotEnv

logger = logging.getLogger(__name__)

try:
    from robodeploy.cameras.opencv.configuration_opencv import OpenCVCameraConfig
    from robodeploy.cameras.realsense.configuration_realsense import RealSenseCameraConfig
    from robodeploy.robots.lerobot_robot_my_arm.bi_innov_arm_v1 import BiInnovArmV1Robot
    from robodeploy.robots.lerobot_robot_my_arm.config_innov_arm import BiInnovArmV1Config
    from robodeploy.robots.lerobot_robot_my_arm.config_innov_arm import InnovArmV1Config
    from robodeploy.robots.lerobot_robot_my_arm.innov_arm_v1 import InnovArmV1Robot
    from robodeploy.utils.leader_follower_align import reset_to_zero

    _ROBODEPLOY_AVAILABLE = True
except ImportError:  # robodeploy 未安装(仿真/无真机环境)
    _ROBODEPLOY_AVAILABLE = False

_ARM_DIM = {"innov_arm_v1": 7, "bi_innov_arm_v1": 14}


def _env_or(kwargs: dict[str, Any], key: str, default: Any = None) -> Any:
    """kwargs 优先,回退环境变量(INNOV_ARM_<KEY>)。"""
    if key in kwargs and kwargs[key] is not None:
        return kwargs[key]
    return os.environ.get(f"INNOV_ARM_{key.upper()}", default)


def _parse_cameras(cameras: str | dict) -> dict[str, Any]:
    """环境变量 JSON 或 dict → {cam_name: CameraConfig}。"""
    spec = json.loads(cameras) if isinstance(cameras, str) else dict(cameras)
    result: dict[str, Any] = {}
    for name, cfg in spec.items():
        ctype = cfg.pop("type", "realsense")
        if ctype in ("realsense", "intelrealsense"):  # robodeploy 注册名兼容
            result[name] = RealSenseCameraConfig(
                serial_number_or_name=cfg.pop("serial_number_or_name", name),
                fps=cfg.pop("fps", 30),
                width=cfg.pop("width", 848),
                height=cfg.pop("height", 480),
                **cfg,
            )
        elif ctype == "opencv":
            result[name] = OpenCVCameraConfig(
                index_or_path=cfg.pop("source", 0),
                fps=cfg.pop("fps", 30),
                width=cfg.pop("width", 848),
                height=cfg.pop("height", 480),
                **cfg,
            )
        else:
            raise ValueError(f"Unknown camera type: {ctype}")
    return result


class BodyTeachingIntervention(InterventionManager):
    """本体复合夹爪示教:按 i 键进入重力补偿示教,拖拽机械臂读关节位置攒 chunk。

    进入:驱动 set_mode("collect")(重力补偿,人可自由拖拽);
    示教:循环 get_action() 读关节位置,攒满 chunk_length 帧;
    退出:再按 i(不足帧末尾补齐)或攒满自动退出,恢复 control 模式。
    SDK 已支持 collect/control 模式切换(RobotController.set_mode)。
    """

    def __init__(
        self,
        robot: Any,
        ctrl: KeyboardCtrl | None,
        action_dim: int,
        chunk_length: int,
        get_obs_fn: Any,
        teach_hz: int = 15,
    ) -> None:
        super().__init__(enabled=True)
        self.robot = robot
        self.ctrl = ctrl
        self.action_dim = action_dim
        self.chunk_length = chunk_length
        self.get_obs_fn = get_obs_fn
        self._teach_period = 1.0 / max(1, teach_hz)
        self._motor_keys = list(robot.action_features.keys())
        self._teaching = False

    def check_intervention(self) -> bool:
        """检测 i 键:切换示教开关并切换驱动模式。"""
        if self.ctrl is not None and self.ctrl.poll() == "i":
            self._teaching = not self._teaching
            if self._teaching:
                self.robot.set_mode("collect")  # 重力补偿,人可拖拽
                logger.info("Body teaching ON (gravity compensation)")
            else:
                self.robot.set_mode("control")
                logger.info("Body teaching OFF (position control)")
        return self._teaching

    def _read_action(self) -> NDArray:
        """读当前关节位置(本体示教动作)。"""
        raw = self.robot.get_action()
        return np.array([float(raw[k]) for k in self._motor_keys], dtype=np.float32)

    def get_human_action(
        self, action_dim: int, chunk_length: int
    ) -> InterventionResult:
        """示教一个 chunk:读关节位置攒满 C 帧。"""
        frames: list[NDArray] = []
        while len(frames) < chunk_length:
            t_start = time.time()
            if self.ctrl is not None and self.ctrl.poll() == "i":
                self._teaching = False
                self.robot.set_mode("control")
                logger.info("Body teaching ended early (%d frames)", len(frames))
                break
            frames.append(self._read_action())
            sleep_t = self._teach_period - (time.time() - t_start)
            if sleep_t > 0:
                time.sleep(sleep_t)

        if self._teaching:  # 攒满自动退出
            self._teaching = False
            self.robot.set_mode("control")
            logger.info("Body teaching chunk complete (%d frames)", chunk_length)

        # 不足 C 帧时末尾帧补齐
        while len(frames) < chunk_length:
            frames.append(frames[-1].copy())

        next_obs = self.get_obs_fn()
        return InterventionResult(
            action_chunk=np.stack(frames[:chunk_length]),
            next_obs=next_obs,
            rewards=np.zeros(chunk_length, dtype=np.float32),
            done=False,
            info={},
        )


class InnovArmRobot:
    """innov_arm 驱动的 RobotEnv 接线(step_fn/reset_fn/get_obs_fn 三件套)。"""

    def __init__(
        self,
        robot: Any,
        action_dim: int,
        task_prompt: str,
    ) -> None:
        self.robot = robot
        self.action_dim = action_dim
        self.task_prompt = task_prompt
        # 动作/状态键序与驱动 action_features 一致(与数据集 features 同序)
        self._motor_keys = list(robot.action_features.keys())
        self._cam_names = list(robot.cameras.keys())

    def get_obs(self) -> dict[str, Any]:
        """驱动观测 → LeRobot 直接格式 {"state", "images", "prompt"}。

        front 相机为 front + front_1 垂直拼接(与采集 _stack_front_cameras 一致,
        数据集 images 键为 front/left_wrist/right_wrist)。
        """
        raw = self.robot.get_observation()
        state = np.array([float(raw[k]) for k in self._motor_keys], dtype=np.float32)
        images = {name: raw[name] for name in self._cam_names}
        if "front" in images and "front_1" in images:
            images["front"] = np.concatenate(
                [np.asarray(images["front"]), np.asarray(images["front_1"])], axis=0
            )
            del images["front_1"]
        return {"state": state, "images": images, "prompt": self.task_prompt}

    def step(self, action: NDArray) -> None:
        """单步动作 [action_dim] → send_action dict。"""
        act_dict = {k: float(action[i]) for i, k in enumerate(self._motor_keys)}
        self.robot.send_action(act_dict)

    def reset(self) -> None:
        """余弦插值归零至 home 位。"""
        reset_to_zero(self.robot, teleop=None, action_features=self.robot.action_features)


def make_innov_arm_env(
    action_dim: int = 14,
    chunk_length: int = 10,
    task_prompt: str = "",
    max_episode_chunks: int = 150,
    **kwargs: Any,
) -> RobotEnv:
    """创建直接驱动 innov_arm 真机的 RobotEnv。

    Args:
        action_dim: 动作维数(单臂 7,双臂 14;默认按 robot_type 推断)。
        chunk_length: C,每个 chunk 的单步动作数。
        task_prompt: 任务指令。
        max_episode_chunks: 单 episode 最大 chunk 数(超时强制终止)。
        **kwargs: robot_type/port/left_port/right_port/cameras/control_hz 覆盖。

    Returns:
        就绪的 RobotEnv(已 connect 真机)。
    """
    if not _ROBODEPLOY_AVAILABLE:
        raise RuntimeError("robodeploy 未安装,无法使用 innov_arm 真机环境")

    robot_type = _env_or(kwargs, "robot_type", "bi_innov_arm_v1")
    control_hz = int(_env_or(kwargs, "control_hz", 15))
    if action_dim <= 0:
        action_dim = _ARM_DIM.get(robot_type, 14)
    if action_dim != _ARM_DIM.get(robot_type):
        logger.warning(
            "robot_type=%s 期望 action_dim=%d,实际 %d",
            robot_type, _ARM_DIM.get(robot_type), action_dim,
        )

    cameras = _parse_cameras(_env_or(kwargs, "cameras", "{}"))
    if not cameras:
        raise ValueError("至少需要一个相机:设置 INNOV_ARM_CAMERAS 或传入 cameras=kwargs")

    if robot_type == "innov_arm_v1":
        cfg = InnovArmV1Config(
            port=_env_or(kwargs, "port", ""),
            mode="control",
            cameras=cameras,
        )
        robot = InnovArmV1Robot(cfg)
    elif robot_type == "bi_innov_arm_v1":
        cfg = BiInnovArmV1Config(
            left_port=_env_or(kwargs, "left_port", ""),
            right_port=_env_or(kwargs, "right_port", ""),
            mode="control",
            cameras=cameras,
        )
        robot = BiInnovArmV1Robot(cfg)
    else:
        raise ValueError(f"Unknown robot_type: {robot_type}")

    robot.connect()
    logger.info(
        "innov_arm connected: type=%s ports=%s cameras=%s",
        robot_type, _env_or(kwargs, "port", _env_or(kwargs, "left_port", "")), list(cameras),
    )

    controller = InnovArmRobot(robot, action_dim, task_prompt)
    env = RobotEnv(
        step_fn=controller.step,
        reset_fn=controller.reset,
        get_obs_fn=controller.get_obs,
        action_dim=action_dim,
        chunk_length=chunk_length,
        control_hz=control_hz,
        max_episode_chunks=max_episode_chunks,
    )
    # 暴露驱动与观测读取,供 intervention 工厂(BodyTeachingIntervention)使用
    env.robot = robot
    env.obs_fn = controller.get_obs
    return env


def make_body_teaching_intervention(
    env: Any = None, teach_hz: int = 15, **kwargs: Any
) -> BodyTeachingIntervention:
    """创建本体示教干预(供 --intervention-factory 使用)。

    Args:
        env: make_innov_arm_env 创建的 RobotEnv(需 robot/ctrl/obs_fn 属性)。
        teach_hz: 示教采样频率(读关节位置)。

    Returns:
        BodyTeachingIntervention 实例。
    """
    if env is None or not hasattr(env, "robot"):
        raise ValueError("make_body_teaching_intervention 需要 make_innov_arm_env 创建的 env")
    return BodyTeachingIntervention(
        robot=env.robot,
        ctrl=env.ctrl,
        action_dim=env.action_dim,
        chunk_length=env.chunk_length,
        get_obs_fn=env.obs_fn,
        teach_hz=teach_hz,
    )
