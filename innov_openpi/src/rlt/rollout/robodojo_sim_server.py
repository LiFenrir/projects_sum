"""RoboDojo 仿真 server:在 RoboDojo/Isaac Sim 的 Python 环境下运行,通过 TCP 向 rlt 提供 reset/step。

复用 create_eval_env(EvalEnv) 的 reset/take_action/is_episode_end 逻辑,
仅 monkeypatch 掉 WsModelClient(改 stub)与视频落盘(_stream_vision 置空)。

启动(由 rlt 侧 RoboDojoEnv 自动拉起,也可手动):
    /path/to/robodojo_env/bin/python robodojo_sim_server.py \
        --root_dir /path/to/RoboDojo --task_name stack_blocks \
        --env_cfg_type arx_x5 --port 29501 --seed 0 --headless

协议(length-prefix TCP + msgpack,见 ipc.py):
    hello                -> {task_name, arm_dim, ee_dim, step_lim, cameras, state_keys}
    reset {seed?}        -> {obs, seed}   (seed 省略时按 eval layout 轮转)
    step {action, want_obs} -> {reward, done, success, score, step_count, obs?}
    close                -> ok 后退出
"""

import argparse
from datetime import UTC
from datetime import datetime
import os
from pathlib import Path
import socket
import sys
import traceback

import numpy as np

# 同目录导入 ipc(独立于 rlt 包,避免触发包的 __init__ 依赖链)
sys.path.insert(0, str(Path(__file__).resolve().parent))
import ipc

parser = argparse.ArgumentParser()
parser.add_argument("--root_dir", required=True, help="RoboDojo 仓库根目录")
parser.add_argument("--task_name", required=True)
parser.add_argument("--env_cfg_type", default="arx_x5")
parser.add_argument("--port", type=int, required=True)
parser.add_argument("--host", default="127.0.0.1")
parser.add_argument("--seed", type=int, default=0, help="eval layout 种子目录(Assets/Eval_Layout/.../<seed>/)")
parser.add_argument("--device_id", default="0")
parser.add_argument("--max_reset_retries", type=int, default=5, help="场景不稳定时换下一个 layout 重试次数")

from isaaclab.app import AppLauncher  # noqa: E402

AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# RoboDojo 内部以相对路径写 eval_result/,且 env/task/utils 包需从根目录导入
os.chdir(args_cli.root_dir)
sys.path.insert(0, args_cli.root_dir)
os.environ.setdefault("ROBODOJO_RUN_ID", datetime.now(tz=UTC).strftime("%Y-%m-%d_%H-%M-%S"))
if args_cli.device_id:
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", args_cli.device_id)

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# 以下导入依赖已启动的 SimulationApp,必须保持在 AppLauncher 之后(模块级,与 eval_client/main.py 一致)
import importlib  # noqa: E402

from env.global_configs import BENCHMARK  # noqa: E402
from env.global_configs import ENV_CONFIG_PATH  # noqa: E402
from omegaconf import OmegaConf  # noqa: E402
import src.eval_client.eval_env as eval_env_mod  # noqa: E402
from utils.cluttered_generator import UnStableError  # noqa: E402
from utils.load_file import load_yaml  # noqa: E402
from utils.pipeline_utils import process_config  # noqa: E402
from utils.pipeline_utils import process_randomization  # noqa: E402


class _StubModelClient:
    """替代 WsModelClient 的空客户端:EvalEnv 的 policy 调用全部变为 no-op。"""

    def __init__(self, **kwargs):
        pass

    def call(self, func_name=None, **kwargs):
        return None

    def close(self):
        pass


def _build_env():
    """复刻 eval_client/main.py 的 env_cfg 组装(num_envs=1,无 policy server)。"""
    task_registry = importlib.import_module(f"task.{BENCHMARK}.task_registry")

    eval_cfg = load_yaml(os.path.join(ENV_CONFIG_PATH, args_cli.env_cfg_type + ".yml"))
    eval_cfg["task_name"] = args_cli.task_name
    eval_cfg["num_envs"] = 1
    eval_cfg["device_id"] = args_cli.device_id
    eval_cfg["eval_batch"] = False
    eval_cfg["policy_name"] = "rlt_sim"
    eval_cfg["additional_info"] = ""
    eval_cfg["seed"] = args_cli.seed
    eval_cfg["physx_monitor_enabled"] = False

    deploy_cfg = {"policy_name": "rlt_sim", "port": 1, "host": "localhost"}

    benchmark_path = os.path.join(args_cli.root_dir, "task", BENCHMARK)
    env_cfg = OmegaConf.create(
        {
            "sim": load_yaml(os.path.join(ENV_CONFIG_PATH, "sim", eval_cfg["config"]["sim"] + ".yml")),
            "scene": load_yaml(os.path.join(ENV_CONFIG_PATH, "scene", eval_cfg["config"]["scene"] + ".yml")),
            "camera": load_yaml(os.path.join(ENV_CONFIG_PATH, "camera", eval_cfg["config"]["camera"] + ".yml")),
            "robot": load_yaml(os.path.join(ENV_CONFIG_PATH, "robot", eval_cfg["config"]["robot"] + ".yml")),
            "task_env": load_yaml(
                task_registry.task_config_path(os.path.join(benchmark_path, "config"), args_cli.task_name)
            ),
            "eval_cfg": eval_cfg,
            "deploy_cfg": deploy_cfg,
        }
    )
    OmegaConf.update(env_cfg, "sim.scene.num_envs", 1, force_add=True)
    OmegaConf.update(env_cfg, "eval_cfg.num_envs", 1, force_add=True)
    env_cfg = process_randomization(env_cfg)
    env_cfg, eval_num = process_config(env_cfg, task_name=args_cli.task_name)
    eval_cfg["eval_num"] = eval_num
    OmegaConf.update(
        env_cfg, "camera.default_frequency", eval_cfg["observation"].get("collect_freq", 0), force_add=True
    )
    env_cfg.sim.seed = [0]

    # stub 掉 policy client 后再创建 env
    eval_env_mod.WsModelClient = _StubModelClient
    env = eval_env_mod.create_eval_env(env_cfg, simulation_app)
    env._stream_vision = lambda *a, **k: None  # 视频不落盘  # noqa: SLF001
    return env


class SimServer:
    """单环境(num_envs=1)的 reset/step 请求处理。"""

    def __init__(self, env):
        self.env = env
        # eval layout 轮转表(SeedManager 已在 env 初始化时加载)
        self._layout_ids = list(env.seed_manager.seed_list)
        self._layout_ptr = 0

    def _next_seed(self) -> int:
        sid = self._layout_ids[self._layout_ptr % len(self._layout_ids)]
        self._layout_ptr += 1
        return sid

    @staticmethod
    def _clean_obs(obs: dict) -> dict:
        """只保留传输必需字段:相机 color、state、instruction。"""
        vision = {cam: np.ascontiguousarray(d["color"]) for cam, d in obs["vision"].items() if "color" in d}
        state = {k: np.asarray(v, dtype=np.float32) for k, v in obs["state"].items()}
        return {"vision": vision, "state": state, "instruction": obs.get("instruction", "")}

    def hello(self) -> dict:
        info = self.env.robot_action_dim_info
        sample = self._clean_obs(self.env.get_obs())
        return {
            "task_name": self.env.task_name,
            "arm_dim": list(info["arm_dim"]),
            "ee_dim": list(info["ee_dim"]),
            "step_lim": int(self.env.step_lim),
            "cameras": list(sample["vision"].keys()),
            "state_keys": list(sample["state"].keys()),
            "instruction": sample["instruction"],
        }

    def reset(self, seed=None) -> dict:
        last_err = None
        for _ in range(args_cli.max_reset_retries):
            sid = int(seed) if seed is not None else self._next_seed()
            try:
                self.env.reset(seed=[sid])
                break
            except UnStableError as e:  # 场景不稳定,换下一个 layout
                last_err = e
                print(f"[sim_server] layout {sid} unstable, trying next: {e}", flush=True)
        else:
            raise RuntimeError(f"reset failed after {args_cli.max_reset_retries} retries: {last_err}")
        self.env.run_reward()
        if hasattr(self.env, "get_score"):
            self.env.get_score()
        return {"obs": self._clean_obs(self.env.get_obs()), "seed": sid}

    def step(self, action: dict, *, want_obs: bool) -> dict:
        self.env.take_action(action)
        done = bool(self.env.end_flag[0])
        success = bool(self.env.success[0]) and done
        score = None
        if hasattr(self.env, "get_score"):
            try:
                score = float(self.env.reward_manager.get_score()[0]) / 100.0
            except Exception:
                score = None
        result = {
            "reward": 1.0 if success else 0.0,
            "done": done,
            "success": success,
            "score": score,
            "step_count": int(self.env.take_action_cnt[0]),
        }
        # done 时总是带上末帧观测
        if want_obs or done:
            result["obs"] = self._clean_obs(self.env.get_obs())
        return result


def _handle(server: SimServer, req: dict) -> dict:
    op = req.get("op")
    if op == "hello":
        return server.hello()
    if op == "reset":
        return server.reset(seed=req.get("seed"))
    if op == "step":
        action = {k: np.asarray(v, dtype=np.float32) for k, v in req["action"].items()}
        return server.step(action, want_obs=bool(req.get("want_obs", False)))
    if op == "observe":
        # 不执行动作,只取当前观测(末帧补帧用)
        return {"obs": SimServer._clean_obs(server.env.get_obs())}  # noqa: SLF001
    if op == "close":
        return {}
    raise ValueError(f"unknown op: {op}")


def main() -> None:
    listen = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    listen.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    listen.bind((args_cli.host, args_cli.port))
    listen.listen(1)
    print(f"[sim_server] listening on {args_cli.host}:{args_cli.port}", flush=True)

    env = _build_env()
    server = SimServer(env)
    print(f"[sim_server] env ready: task={args_cli.task_name} step_lim={env.step_lim}", flush=True)

    closing = False
    while not closing:
        conn, addr = listen.accept()
        print(f"[sim_server] client connected: {addr}", flush=True)
        conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        try:
            while True:
                req = ipc.recv_msg(conn)
                try:
                    result = _handle(server, req)
                    resp = {"ok": True, "result": result}
                except Exception:
                    resp = {"ok": False, "error": traceback.format_exc()}
                ipc.send_msg(conn, resp)
                if req.get("op") == "close":
                    closing = True
                    break
        except ConnectionError:
            print("[sim_server] client disconnected, waiting for reconnect", flush=True)
        finally:
            conn.close()

    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
