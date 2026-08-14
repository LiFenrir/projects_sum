"""RoboDojoEnv 单元测试:fake TCP server(ipc 协议)验证映射/chunk/奖励/崩溃重启。"""

from __future__ import annotations

import socket
import sys
import textwrap
import threading
import time

import numpy as np
import pytest

from rlt.rollout import ipc
from rlt.rollout.robodojo_env import RoboDojoEnv

_STATE_KEYS = [
    "left_arm_joint_state",
    "left_ee_joint_state",
    "right_arm_joint_state",
    "right_ee_joint_state",
]
_CAMS = ["cam_head", "cam_left_wrist", "cam_right_wrist"]


def _fake_obs():
    return {
        "vision": {cam: np.full((8, 8, 3), i, dtype=np.uint8) for i, cam in enumerate(_CAMS)},
        "state": {k: np.zeros(d, dtype=np.float32) for k, d in zip(_STATE_KEYS, [6, 1, 6, 1], strict=True)},
        "instruction": "fake task instruction",
    }


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class FakeSimServer:
    """线程内 fake server:第 success_at 步给 reward=1/done/success。"""

    def __init__(self, port: int, success_at: int = 3, step_lim: int = 5):
        self.port = port
        self.success_at = success_at
        self.step_lim = step_lim
        self.count = 0
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._serve, daemon=True)
        self._thread.start()

    def _serve(self):
        listen = socket.socket()
        listen.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        listen.bind(("127.0.0.1", self.port))
        listen.listen(1)
        listen.settimeout(0.5)
        while not self._stop.is_set():
            try:
                conn, _ = listen.accept()
            except TimeoutError:
                continue
            conn.settimeout(5.0)
            try:
                while not self._stop.is_set():
                    req = ipc.recv_msg(conn)
                    ipc.send_msg(conn, {"ok": True, "result": self._handle(req)})
                    if req.get("op") == "close":
                        break
            except (TimeoutError, ConnectionError, OSError):
                pass
            finally:
                conn.close()
        listen.close()

    def _handle(self, req: dict) -> dict:
        op = req["op"]
        if op == "hello":
            return {
                "task_name": "fake_task",
                "arm_dim": [6, 6],
                "ee_dim": [1, 1],
                "step_lim": self.step_lim,
                "cameras": _CAMS,
                "state_keys": _STATE_KEYS,
                "instruction": "fake task instruction",
            }
        if op == "reset":
            self.count = 0
            return {"obs": _fake_obs(), "seed": 0}
        if op == "observe":
            return {"obs": _fake_obs()}
        if op == "step":
            assert set(req["action"]) == set(_STATE_KEYS)
            assert req["action"]["left_arm_joint_state"].shape == (6,)
            self.count += 1
            done = self.count >= self.success_at
            result = {
                "reward": 1.0 if done else 0.0,
                "done": done,
                "success": done,
                "score": None,
                "step_count": self.count,
            }
            if req.get("want_obs") or done:
                result["obs"] = _fake_obs()
            return result
        if op == "close":
            return {}
        raise ValueError(op)

    def close(self):
        self._stop.set()
        self._thread.join(timeout=2)


@pytest.fixture
def fake_server():
    server = FakeSimServer(port=_free_port())
    yield server
    server.close()


def _attach_env(server: FakeSimServer, **kwargs) -> RoboDojoEnv:
    return RoboDojoEnv(
        task_name="fake_task",
        action_dim=14,
        chunk_length=10,
        host="127.0.0.1",
        port=server.port,
        launch=False,
        startup_timeout=10.0,
        request_timeout=10.0,
        **kwargs,
    )


def test_obs_mapping(fake_server):
    env = _attach_env(fake_server)
    obs = env.reset()
    assert obs["state"].shape == (14,)
    assert set(obs["images"]) == {"front", "left_wrist", "right_wrist"}
    assert obs["images"]["front"].shape == (8, 8, 3)
    assert obs["prompt"] == "fake task instruction"
    env.close()


def test_task_prompt_override(fake_server):
    env = _attach_env(fake_server, task_prompt="custom prompt")
    assert env.reset()["prompt"] == "custom prompt"
    env.close()


def test_step_chunk_success_reward(fake_server):
    env = _attach_env(fake_server)
    env.reset()
    # success_at=3:第一个 chunk 第 3 步成功,done 中断
    obs, rewards, done, info = env.step(np.zeros((10, 14), dtype=np.float32))
    assert done
    assert info["success"]
    assert rewards[2] == 1.0
    assert rewards.sum() == 1.0
    assert info["steps_executed"] == 3
    assert obs["state"].shape == (14,)
    env.close()


def test_chunk_timeout(fake_server):
    fake_server.success_at = 10**9  # 永不成功
    env = _attach_env(fake_server, max_episode_chunks=2)
    env.reset()
    _, _, done1, _ = env.step(np.zeros((10, 14), dtype=np.float32))
    _, _, done2, info2 = env.step(np.zeros((10, 14), dtype=np.float32))
    assert not done1
    assert done2
    assert info2.get("timeout")
    assert not info2["success"]
    env.close()


def test_stride_obs_history(fake_server):
    fake_server.success_at = 10**9
    env = _attach_env(fake_server, obs_stride=2)
    env.reset()
    _, _, done, info = env.step(np.zeros((10, 14), dtype=np.float32))
    assert not done
    # stride=2、C=10:第 2/4/6/8 步采样(末步不采),共 4 帧
    assert len(info["obs_history"]) == 4
    assert info["actions_history"].shape == (10, 14)
    env.close()


_CRASH_SCRIPT = textwrap.dedent(
    """
    import socket, sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from rlt.rollout import ipc
    import numpy as np

    port = int(sys.argv[sys.argv.index("--port") + 1])
    listen = socket.socket()
    listen.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    listen.bind(("127.0.0.1", port))
    listen.listen(1)
    while True:
        conn, _ = listen.accept()
        while True:
            req = ipc.recv_msg(conn)
            if req["op"] == "hello":
                ipc.send_msg(conn, {"ok": True, "result": {
                    "task_name": "crash_task", "arm_dim": [6, 6], "ee_dim": [1, 1],
                    "step_lim": 100, "cameras": [], "state_keys": [
                        "left_arm_joint_state", "left_ee_joint_state",
                        "right_arm_joint_state", "right_ee_joint_state"],
                    "instruction": ""}})
            elif req["op"] == "reset":
                obs = {"vision": {}, "state": {
                    "left_arm_joint_state": np.zeros(6, dtype=np.float32),
                    "left_ee_joint_state": np.zeros(1, dtype=np.float32),
                    "right_arm_joint_state": np.zeros(6, dtype=np.float32),
                    "right_ee_joint_state": np.zeros(1, dtype=np.float32)},
                    "instruction": ""}
                ipc.send_msg(conn, {"ok": True, "result": {"obs": obs, "seed": 0}})
            elif req["op"] == "step":
                sys.exit(1)  # 模拟 Isaac Sim 崩溃
    """
)


def test_crash_restart(tmp_path):
    """server 在 step 时崩溃 → env 自动重启,本 episode 记失败结束。"""
    script = tmp_path / "crash_server.py"
    script.write_text(_CRASH_SCRIPT)
    port = _free_port()
    env = RoboDojoEnv(
        task_name="crash_task",
        action_dim=14,
        chunk_length=10,
        port=port,
        launch=True,
        python=sys.executable,
        server_script=str(script),
        startup_timeout=30.0,
        request_timeout=10.0,
    )
    try:
        env.reset()
        t0 = time.time()
        obs, _, done, info = env.step(np.zeros((10, 14), dtype=np.float32))
        assert done
        assert not info["success"]
        assert info.get("sim_crash")
        assert obs["state"].shape == (14,)  # 重启后 reset 的新观测
        assert time.time() - t0 < 30
    finally:
        env.close()
