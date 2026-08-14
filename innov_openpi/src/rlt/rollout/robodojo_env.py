"""RoboDojo 仿真环境(rlt 侧 client):chunk 级接口,对齐 SimEnv/RobotEnv。

通过 TCP(length-prefix + msgpack,见 ipc.py)驱动 RoboDojo 侧的
robodojo_sim_server.py 子进程;Isaac Sim 崩溃时自动 respawn 并重连。

观测映射(RoboDojo → openpi 格式):
    state  ← state_keys 指定的关节键按序拼接(默认双臂 14-D)
    images ← camera_map {openpi 键: robodojo 相机} 的 color 图
    prompt ← task_prompt 覆盖,否则用 task 的 instruction

动作映射:[action_dim] 按 action_keys(默认 left/right arm+ee)切分为
RoboDojo 单步动作 dict(绝对关节目标)。
"""

from __future__ import annotations

import contextlib
import json
import logging
import os
from pathlib import Path
import socket
import subprocess
import time
from typing import Any

import numpy as np
from numpy.typing import NDArray

from rlt.rollout import ipc

logger = logging.getLogger(__name__)

_DEFAULT_CAMERA_MAP = {
    "front": "cam_head",
    "left_wrist": "cam_left_wrist",
    "right_wrist": "cam_right_wrist",
}


def _env_or(kwargs: dict[str, Any], key: str, default: Any = None) -> Any:
    """kwargs 优先,回退环境变量(ROBODOJO_<KEY>)。"""
    if key in kwargs and kwargs[key] is not None:
        return kwargs[key]
    return os.environ.get(f"ROBODOJO_{key.upper()}", default)


class SimCrashedError(Exception):
    """sim server 进程死亡或连接断开。"""


class RoboDojoEnv:
    """RoboDojo 仿真环境(chunk 级)。

    Args:
        robodojo_root: RoboDojo 仓库根目录。
        python: RoboDojo/Isaac Sim 环境的 Python 解释器路径。
        task_name: RoboDojo 任务名(task/RoboDojo/tasks/{task_name}.py)。
        action_dim: 单步动作维数(双臂 dual_x5 为 14)。
        chunk_length: C,每个 chunk 的单步动作数。
        env_cfg_type: env_cfg/{env_cfg_type}.yml(默认 arx_x5)。
        seed: eval layout 种子目录。
        device_id: sim 使用的 GPU(写入 CUDA_VISIBLE_DEVICES)。
        port: TCP 端口;0 = 自动选空闲端口。
        camera_map: {openpi 图像键: robodojo 相机名}。
        max_episode_chunks: 单 episode 最大 chunk 数(超时强制终止)。
        obs_stride: 每 N 步采一次中间观测。
        startup_timeout: 等 server 就绪的超时(Isaac Sim 启动很慢)。
        request_timeout: 单次请求超时(reset 含场景加载,需给足)。
        launch: True = 拉起子进程;False = 仅连接已运行的 server。
        server_script: server 脚本路径(测试可替换为 fake server)。
    """

    def __init__(
        self,
        robodojo_root: str = "",
        python: str = "python3",
        task_name: str = "",
        action_dim: int = 14,
        chunk_length: int = 10,
        env_cfg_type: str = "arx_x5",
        seed: int = 0,
        device_id: str | None = None,
        port: int = 0,
        host: str = "127.0.0.1",
        camera_map: dict[str, str] | None = None,
        task_prompt: str = "",
        max_episode_chunks: int = 150,
        obs_stride: int = 2,
        startup_timeout: float = 900.0,
        request_timeout: float = 600.0,
        *,
        launch: bool = True,
        server_script: str | None = None,
    ) -> None:
        if not task_name:
            raise ValueError("task_name 不能为空")
        self._root = robodojo_root
        self._python = python
        self._task_name = task_name
        self._env_cfg_type = env_cfg_type
        self._seed = seed
        self._device_id = device_id
        self._host = host
        self._port = port or self._free_port()
        self._camera_map = dict(camera_map or _DEFAULT_CAMERA_MAP)
        self._task_prompt = task_prompt
        self._max_episode_chunks = max_episode_chunks
        self._obs_stride = max(1, obs_stride)
        self._startup_timeout = startup_timeout
        self._request_timeout = request_timeout
        self._launch = launch
        self._server_script = server_script or str(Path(__file__).resolve().parent / "robodojo_sim_server.py")

        self._action_dim = action_dim
        self._chunk_length = chunk_length
        self._chunk_count = 0
        self._proc: subprocess.Popen | None = None
        self._sock: socket.socket | None = None
        self._state_keys: list[str] = []
        self._action_keys: list[tuple[str, int]] = []  # (键名, 维数),按动作向量顺序

        self._start_server()

    # ------------------------------------------------------------------
    # 进程与连接管理
    # ------------------------------------------------------------------

    @staticmethod
    def _free_port() -> int:
        with socket.socket() as s:
            s.bind(("127.0.0.1", 0))
            return s.getsockname()[1]

    def _launch_proc(self) -> None:
        cmd = [
            self._python,
            self._server_script,
            "--root_dir",
            self._root,
            "--task_name",
            self._task_name,
            "--env_cfg_type",
            self._env_cfg_type,
            "--host",
            self._host,
            "--port",
            str(self._port),
            "--seed",
            str(self._seed),
            "--headless",
        ]
        env = os.environ.copy()
        env["PYTHONPATH"] = self._root + os.pathsep + env.get("PYTHONPATH", "")
        if self._device_id is not None:
            env["CUDA_VISIBLE_DEVICES"] = str(self._device_id)
        logger.info("Launching RoboDojo sim server: %s", " ".join(cmd))
        self._proc = subprocess.Popen(cmd, cwd=self._root or None, env=env)

    def _connect(self) -> None:
        deadline = time.time() + self._startup_timeout
        while True:
            if self._proc is not None and self._proc.poll() is not None:
                raise RuntimeError(f"sim server exited during startup, rc={self._proc.returncode}")
            try:
                sock = socket.create_connection((self._host, self._port), timeout=5.0)
                break
            except OSError:
                if time.time() > deadline:
                    raise TimeoutError(f"sim server not ready within {self._startup_timeout}s") from None
                time.sleep(2.0)
        sock.settimeout(self._request_timeout)
        sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        self._sock = sock

    def _start_server(self) -> None:
        """(重)启动 server 并完成 hello 握手。"""
        self._close_sock()
        if self._launch:
            self._kill_proc()
            self._launch_proc()
        self._connect()
        info = self._rpc({"op": "hello"})
        self._setup_mappings(info)
        self._chunk_count = 0
        logger.info(
            "RoboDojo env ready: task=%s step_lim=%s state_keys=%s cameras=%s",
            info["task_name"],
            info["step_lim"],
            self._state_keys,
            info["cameras"],
        )

    def _setup_mappings(self, hello: dict) -> None:
        """按 hello 返回的机器人维度信息构建 state/action 键序。"""
        arm_dims = hello["arm_dim"]
        ee_dims = hello["ee_dim"]
        prefixes = [""] if len(arm_dims) == 1 else ["left_", "right_"]
        self._state_keys = []
        self._action_keys = []
        for i, p in enumerate(prefixes):
            self._state_keys += [f"{p}arm_joint_state", f"{p}ee_joint_state"]
            self._action_keys += [(f"{p}arm_joint_state", arm_dims[i]), (f"{p}ee_joint_state", ee_dims[i])]
        expect = sum(arm_dims) + sum(ee_dims)
        if self._action_dim != expect:
            raise ValueError(f"action_dim={self._action_dim} 与机器人维度 {expect} 不一致(task={self._task_name})")

    def _close_sock(self) -> None:
        if self._sock is not None:
            with contextlib.suppress(OSError):
                self._sock.close()
            self._sock = None

    def _kill_proc(self) -> None:
        if self._proc is not None and self._proc.poll() is None:
            self._proc.kill()
            self._proc.wait(timeout=10)
        self._proc = None

    # ------------------------------------------------------------------
    # RPC
    # ------------------------------------------------------------------

    def _rpc(self, req: dict) -> dict:
        """同步请求-响应;连接异常统一转为 SimCrashedError。"""
        try:
            assert self._sock is not None
            ipc.send_msg(self._sock, req)
            resp = ipc.recv_msg(self._sock)
        except (TimeoutError, ConnectionError, OSError) as e:
            raise SimCrashedError(str(e)) from e
        if not resp.get("ok"):
            raise RuntimeError(f"sim server error:\n{resp.get('error')}")
        return resp["result"]

    # ------------------------------------------------------------------
    # obs/action 映射
    # ------------------------------------------------------------------

    def _map_obs(self, raw: dict) -> dict[str, Any]:
        state = np.concatenate([np.asarray(raw["state"][k], dtype=np.float32) for k in self._state_keys])
        images = {dst: raw["vision"][src] for dst, src in self._camera_map.items() if src in raw["vision"]}
        return {"state": state, "images": images, "prompt": self._task_prompt or raw.get("instruction", "")}

    def _map_action(self, action: NDArray) -> dict[str, NDArray]:
        """[action_dim] → RoboDojo 单步动作 dict。"""
        out: dict[str, NDArray] = {}
        offset = 0
        for key, dim in self._action_keys:
            out[key] = np.asarray(action[offset : offset + dim], dtype=np.float32)
            offset += dim
        return out

    # ------------------------------------------------------------------
    # chunk 级环境接口(对齐 SimEnv)
    # ------------------------------------------------------------------

    @property
    def action_dim(self) -> int:
        return self._action_dim

    @property
    def chunk_length(self) -> int:
        return self._chunk_length

    def reset(self, **kwargs: Any) -> dict[str, Any]:
        """重置仿真(换下一个 eval layout),返回 openpi 格式观测。"""
        try:
            result = self._rpc({"op": "reset", "seed": kwargs.get("seed")})
        except (SimCrashedError, RuntimeError):
            logger.warning("reset failed, restarting sim server", exc_info=True)
            self._start_server()
            result = self._rpc({"op": "reset", "seed": kwargs.get("seed")})
        self._chunk_count = 0
        return self._map_obs(result["obs"])

    def step(self, action_chunk: NDArray) -> tuple[dict[str, Any], NDArray, bool, dict[str, Any]]:
        """执行 chunk_length 个单步动作。

        Returns:
            next_obs, rewards[chunk_length](稀疏,成功步为 1), done, info
            (success/timeout/sim_crash + stride 子采样 history)。
        """
        n_steps = self._chunk_length
        stride = self._obs_stride
        rewards = np.zeros(n_steps, dtype=np.float32)
        done = False
        info: dict[str, Any] = {}
        obs = None

        obs_history: list[dict[str, Any]] = []
        actions_history: list[NDArray] = []
        rewards_history: list[float] = []
        done_history: list[float] = []

        try:
            for k in range(n_steps):
                want_obs = (k + 1) % stride == 0 and k + 1 < n_steps
                result = self._rpc({"op": "step", "action": self._map_action(action_chunk[k]), "want_obs": want_obs})
                rewards[k] = float(result["reward"])
                done = bool(result["done"])
                actions_history.append(np.asarray(action_chunk[k], dtype=np.float32))
                rewards_history.append(float(result["reward"]))
                done_history.append(1.0 if done else 0.0)
                if "obs" in result:
                    obs = self._map_obs(result["obs"])
                    if want_obs:
                        obs_history.append(obs)
                if done:
                    info["success"] = bool(result["success"])
                    if result.get("score") is not None:
                        info["score"] = result["score"]
                    break
        except SimCrashedError:
            # Isaac Sim 崩溃:重启 server,本 episode 记失败结束
            logger.warning("sim server crashed at chunk %d step, restarting", self._chunk_count, exc_info=True)
            self._start_server()
            obs = self.reset()
            done = True
            info["success"] = False
            info["sim_crash"] = True

        executed = len(actions_history)
        info["steps_executed"] = executed
        info["obs_history"] = obs_history
        info["actions_history"] = np.stack(actions_history) if actions_history else None
        info["rewards_history"] = np.asarray(rewards_history, dtype=np.float32)
        info["done_history"] = np.asarray(done_history, dtype=np.float32)
        self._chunk_count += 1

        if not done and self._chunk_count >= self._max_episode_chunks:
            done = True
            info["success"] = False
            info["timeout"] = True

        if obs is None:
            # 未请求过观测:补一帧当前观测作为 next_obs
            obs = self._map_obs(self._rpc({"op": "observe"})["obs"])

        return obs, rewards, done, info

    def close(self) -> None:
        """关闭 server 与子进程。"""
        if self._sock is not None:
            with contextlib.suppress(SimCrashedError, RuntimeError):
                self._rpc({"op": "close"})
        self._close_sock()
        self._kill_proc()

    def __del__(self) -> None:
        with contextlib.suppress(Exception):
            self.close()


def make_robodojo_env(
    action_dim: int = 14,
    chunk_length: int = 10,
    task_prompt: str = "",
    max_episode_chunks: int = 150,
    **kwargs: Any,
) -> RoboDojoEnv:
    """创建 RoboDojo 仿真环境(供 --env-factory 使用)。

    kwargs/环境变量(ROBODOJO_*)双通道:
        root/robodojo_root: RoboDojo 仓库根目录(必填)
        python: RoboDojo 环境解释器路径(默认 python3)
        task_name: 任务名(必填,如 stack_blocks)
        env_cfg_type/seed/device_id/port/host/obs_stride/startup_timeout
        camera_map: JSON 字符串或 dict
    """
    root = _env_or(kwargs, "root", None) or _env_or(kwargs, "robodojo_root", "")
    if not root:
        raise ValueError("需要 robodojo_root:设置 ROBODOJO_ROOT 或传入 root=kwargs")
    task_name = _env_or(kwargs, "task_name", "")
    if not task_name:
        raise ValueError("需要 task_name:设置 ROBODOJO_TASK_NAME 或传入 task_name=kwargs")

    camera_map = _env_or(kwargs, "camera_map", None)
    if isinstance(camera_map, str):
        camera_map = json.loads(camera_map)

    return RoboDojoEnv(
        robodojo_root=root,
        python=_env_or(kwargs, "python", "python3"),
        task_name=task_name,
        action_dim=action_dim,
        chunk_length=chunk_length,
        env_cfg_type=_env_or(kwargs, "env_cfg_type", "arx_x5"),
        seed=int(_env_or(kwargs, "seed", 0)),
        device_id=_env_or(kwargs, "device_id", None),
        port=int(_env_or(kwargs, "port", 0)),
        host=_env_or(kwargs, "host", "127.0.0.1"),
        camera_map=camera_map,
        task_prompt=task_prompt,
        max_episode_chunks=max_episode_chunks,
        obs_stride=int(_env_or(kwargs, "obs_stride", 2)),
        startup_timeout=float(_env_or(kwargs, "startup_timeout", 900.0)),
        launch=bool(int(_env_or(kwargs, "launch", 1))),
        server_script=_env_or(kwargs, "server_script", None),
    )
