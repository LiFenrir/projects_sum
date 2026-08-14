"""训练中热启动评估:内存 Policy 组装 + 进程内 WS 服务器 + 信号文件协调."""

import asyncio
import json
import logging
import socket
import threading
import time
from pathlib import Path

import torch

import openpi.transforms as transforms
from openpi.policies import policy as _policy

_logger = logging.getLogger(__name__)

_READY_FILE = ".eval_ready"
_DONE_FILE = ".eval_done"
_LOG_CSV = ".eval_log.csv"


def build_policy_from_memory(
    train_config,
    model,
    *,
    norm_stats,
    pytorch_device: str = "cuda",
    action_chunk: int | None = None,
):
    """用内存模型组装 Policy,transforms 链与 create_trained_policy 一致(不含磁盘加载)."""
    data_config = train_config.data.create(train_config.assets_dirs, train_config.model)
    return _policy.Policy(
        model,
        transforms=[
            transforms.InjectDefaultPrompt(None),
            *data_config.data_transforms.inputs,
            transforms.Normalize(norm_stats, use_quantiles=data_config.use_quantile_norm),
            *data_config.model_transforms.inputs,
        ],
        output_transforms=[
            *data_config.model_transforms.outputs,
            transforms.Unnormalize(norm_stats, use_quantiles=data_config.use_quantile_norm),
            *data_config.data_transforms.outputs,
        ],
        sample_kwargs={},
        is_pytorch=True,
        pytorch_device=pytorch_device,
        action_chunk=action_chunk,
    )


def run_eval_server(policy, env_cfg_type: str, action_type: str, host: str = "127.0.0.1"):
    """起 XPolicyLab WS 服务器线程,返回 (thread, port)."""
    from XPolicyLab.policy.innov_pi05.model import Model

    # 动态取空闲端口
    with socket.socket() as sock:
        sock.bind((host, 0))
        port = sock.getsockname()[1]

    model = Model(
        {"task_name": None, "env_cfg_type": env_cfg_type, "action_type": action_type},
        injected_policy=policy,
    )
    loop = asyncio.new_event_loop()

    def _run() -> None:
        from client_server.ws.model_server import PolicyServer, PolicyServerConfig

        asyncio.set_event_loop(loop)
        server = PolicyServer(model, PolicyServerConfig(host=host, port=port))
        loop.run_until_complete(server.start())
        _logger.info("[EvalServer] listening on %s:%d", host, port)
        try:
            loop.run_until_complete(server.serve_forever())
        finally:
            loop.run_until_complete(server.stop())

    thread = threading.Thread(target=_run, name="eval-ws-server", daemon=True)
    thread.start()
    return thread, port


def run_eval_hotstart(model, step: int, config, data_config) -> None:
    """训练 Hook 入口:组装内存 Policy → 起服务器 → 等 eval_client.sh 完成(或超时)."""
    ckpt_root = Path(config.checkpoint_dir)

    # unwrap DDP
    raw_model = model.module if isinstance(model, torch.nn.parallel.DistributedDataParallel) else model
    raw_model.eval()

    norm_stats = data_config.norm_stats
    policy = build_policy_from_memory(
        config,
        raw_model,
        norm_stats=norm_stats,
        pytorch_device=str(next(raw_model.parameters()).device),
        action_chunk=getattr(config.model, "action_horizon", None),
    )

    thread, port = run_eval_server(policy, "arx_x5", "joint")

    ready_path = ckpt_root / _READY_FILE
    done_path = ckpt_root / _DONE_FILE
    done_path.unlink(missing_ok=True)
    ready_path.write_text(f"port={port}\nstep={step}\ntask={config.eval_task}\n", encoding="utf-8")
    _logger.info("[EvalHook] ready signal written: step=%d port=%d task=%s", step, port, config.eval_task)

    timeout = float(getattr(config, "eval_timeout", 1800.0))
    deadline = time.monotonic() + timeout
    status = "timeout"
    success_rate = -1.0
    eval_time = -1.0
    while time.monotonic() < deadline:
        if done_path.exists():
            payload = json.loads(done_path.read_text(encoding="utf-8"))
            status = payload.get("status", "failed")
            success_rate = float(payload.get("success_rate", -1.0))
            eval_time = float(payload.get("eval_time", -1.0))
            break
        time.sleep(5)

    # 追加结果 CSV
    with open(ckpt_root / _LOG_CSV, "a", encoding="utf-8") as f:
        f.write(f"{step},{success_rate},{eval_time},{status}\n")

    ready_path.unlink(missing_ok=True)
    done_path.unlink(missing_ok=True)
    raw_model.train()
    _logger.info(
        "[EvalHook] finished: step=%d status=%s success_rate=%.4f eval_time=%.1f",
        step, status, success_rate, eval_time,
    )
