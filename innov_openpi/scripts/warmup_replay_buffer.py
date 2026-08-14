"""使用离线专家数据预热 ReplayBuffer，供 Stage 2 在线 RL 训练使用。

从 LeRobot 专家示范数据集加载 episode，通过冻结的 VLA + RL Token 模型
提取 RL state 和 VLA 参考动作，按 chunk 切片专家动作，存储到 ReplayBuffer
并保存为 warmup_buffer.pt，可直接被 ``OnlineRLTrainer._load_warmup_buffer()`` 加载。

用法::

    python scripts/warmup_replay_buffer.py \\
        --config configs/rlt/stage2_online_rl.yaml \\
        --vla-checkpoint-dir checkpoints/my_run/20000 \\
        --rl-token-checkpoint checkpoints/rl_token/my_run/rl_token_step5000.pt \\
        --dataset-repo-id lerobot/s1_bimanual_0420 \\
        --output checkpoints/online_rl/my_run/warmup_buffer.pt

也支持纯 CLI（无需 YAML config）::

    python scripts/warmup_replay_buffer.py \\
        --vla-checkpoint-dir checkpoints/my_run/20000 \\
        --vla-config-name configs/bi_s1/pi05_finetune.yaml \\
        --rl-token-checkpoint checkpoints/rl_token/my_run/rl_token_step5000.pt \\
        --dataset-repo-id lerobot/s1_bimanual_0420 \\
        --action-dim 7 \\
        --chunk-length 10 \\
        --embedding-dim 2048 \\
        --output /tmp/warmup_buffer.pt
"""

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from numpy.typing import NDArray

from rlt.training.config import OnlineRLTrainConfig
from rlt.training.replay_buffer import ReplayBuffer
from rlt.utils.checkpoint import load_rl_token_model
from rlt.utils.config_loader import load_config_with_cli
from openpi.training.vla_wrapper import VLAWrapper

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(message)s")
log = logging.getLogger(__name__)


def _build_obs_from_frame(
    frame: dict[str, Any],
    task_prompt: str | None = None,
) -> dict[str, Any]:
    """LeRobot 帧格式 → VLAWrapper.preprocess_obs 期望格式。

    Input:  {"observation.state": tensor[d], "observation.images.cam": tensor[C,H,W], ...}
    Output: {"state": array[d], "images": {"cam": array[H,W,C]}, "prompt": str}
    """
    val = frame["observation.state"]
    state = val.numpy() if hasattr(val, "numpy") else np.asarray(val, dtype=np.float32)

    images: dict[str, np.ndarray] = {}
    for key, val in frame.items():
        if key.startswith("observation.images."):
            cam_name = key.replace("observation.images.", "")
            images[cam_name] = val.numpy() if hasattr(val, "numpy") else np.asarray(val)

    prompt = task_prompt
    if prompt is None:
        task = frame.get("task", None)
        if hasattr(task, "item"):
            task = task.item()
        if task and isinstance(task, str):
            prompt = task

    obs: dict[str, Any] = {"state": state, "images": images}
    if prompt is not None:
        obs["prompt"] = prompt
    return obs


@torch.no_grad()
def extract_rl_state(
    obs: dict[str, Any],
    vla: VLAWrapper,
    rl_token_model: Any,
    chunk_length: int,
    action_dim: int,
) -> tuple[NDArray, NDArray]:
    """从观测中提取 RL state x = cat(z_rl, s^p) 和 VLA 参考动作 chunk。

    与 ``RolloutWorker._extract_rl_state`` 功能等价，但不依赖 RolloutWorker。

    Args:
        obs: VLAWrapper.preprocess_obs 期望的观测字典。
        vla: 冻结的 VLA wrapper。
        rl_token_model: 冻结的 RL Token 模型。
        chunk_length: 动作 chunk 长度 C。
        action_dim: 单步动作维度 d。

    Returns:
        x: RL state [state_dim]。
        a_tilde_flat: VLA 参考动作 chunk（扁平）[C*d]。
    """
    vla_input = vla.preprocess_obs(obs)

    # 提取 prefix embeddings → RL Token 编码 → z_rl
    z, pad_mask = vla.extract_embeddings(vla_input)
    z_rl = rl_token_model.encode(z, pad_mask)  # [1, D]

    # VLA 参考动作 chunk（raw robot space，经过 unnormalize）
    a_tilde = vla.get_rl_chunk_reference(vla_input, chunk_length)  # [1, C, action_dim]
    a_tilde_flat = a_tilde.reshape(1, -1)  # [1, C*d]

    # 本体感觉状态 s^p（截断到 action_dim 以去除 padding）
    s_p = vla_input.state[:, :action_dim].to(dtype=torch.float32, device=vla.device)  # [1, d]

    # RL state x = cat(z_rl, s^p)
    x = torch.cat([z_rl, s_p], dim=-1)  # [1, D + d]

    return (
        x.squeeze(0).cpu().numpy(),
        a_tilde_flat.squeeze(0).cpu().numpy(),
    )


def process_episode(
    episode_frames: list[dict[str, Any]],
    episode_meta_entry: dict[str, Any] | None,
    vla: VLAWrapper,
    rl_token_model: Any,
    buffer: ReplayBuffer,
    chunk_length: int,
    action_dim: int,
    stride: int = 2,
    task_prompt: str | None = None,
    reward_mode: str = "sparse",
    verbose: bool = False,
) -> int:
    """处理一个 episode，提取 transitions 并存入 ReplayBuffer。

    Args:
        episode_frames: episode 的所有帧（LeRobot 格式）。
        episode_meta_entry: ds.meta.episodes 中该 episode 的元数据。
        vla: 冻结的 VLA wrapper。
        rl_token_model: 冻结的 RL Token 模型。
        buffer: ReplayBuffer 实例。
        chunk_length: 动作 chunk 长度 C。
        action_dim: 单步动作维度 d。
        stride: 子采样步长。
        task_prompt: 任务 prompt 覆盖。
        reward_mode: 奖励模式。
        verbose: 是否打印进度。

    Returns:
        实际存入 buffer 的 transition 数量。
    """
    episode_len = len(episode_frames)
    if episode_len <= chunk_length:
        return 0

    success = False
    if episode_meta_entry is not None:
        success = bool(
            episode_meta_entry.get("success", False)
            or episode_meta_entry.get("is_success", False)
        )

    # 预计算所有帧的 RL state 和 VLA 参考动作（VLA 推理瓶颈）
    xs: list[NDArray] = []
    a_tildes: list[NDArray] = []

    if verbose:
        log.info("  提取 RL state (VLA 推理)...")

    for frame in episode_frames:
        obs = _build_obs_from_frame(frame, task_prompt)
        x, a_tilde_flat = extract_rl_state(obs, vla, rl_token_model, chunk_length, action_dim)
        xs.append(x)
        a_tildes.append(a_tilde_flat)

    # 构建 chunk-level transitions
    chunk_xs: list[NDArray] = []
    chunk_actions: list[NDArray] = []
    chunk_a_tildes: list[NDArray] = []
    chunk_rewards: list[NDArray] = []
    chunk_next_xs: list[NDArray] = []
    chunk_dones: list[NDArray] = []

    max_t = episode_len - chunk_length
    for t in range(max_t):
        # RL state 和参考动作
        chunk_xs.append(xs[t])
        chunk_a_tildes.append(a_tildes[t])

        # 专家动作 chunk: 从帧 t 到 t+chunk_length-1 的 action
        expert_actions = []
        for k in range(chunk_length):
            frame = episode_frames[t + k]
            action = frame.get("action")
            if hasattr(action, "numpy"):
                action = action.numpy()
            expert_actions.append(np.asarray(action, dtype=np.float32))
        a_chunk = np.stack(expert_actions, axis=0)  # [C, d]
        chunk_actions.append(a_chunk.reshape(-1))   # [C*d]

        # Next RL state（chunk 之后的帧）
        chunk_next_xs.append(xs[t + chunk_length])

        # Done: chunk 结束后是否到达 episode 末尾
        done = float(t + chunk_length >= episode_len)
        chunk_dones.append(np.array([done], dtype=np.float32))

        # 奖励
        if reward_mode == "zero":
            step_rewards = np.zeros(chunk_length, dtype=np.float32)
        elif reward_mode == "sparse":
            step_rewards = np.zeros(chunk_length, dtype=np.float32)
            if success and t + chunk_length >= episode_len:
                step_rewards[-1] = 1.0
        elif reward_mode == "from_dataset":
            # 尝试从帧的 reward 字段读取（LeRobot 数据集一般不包含）
            step_rewards = np.zeros(chunk_length, dtype=np.float32)
            for k in range(chunk_length):
                r = episode_frames[t + k].get("reward", 0.0)
                if hasattr(r, "item"):
                    r = r.item()
                step_rewards[k] = float(r)
        else:
            raise ValueError(f"不支持的 reward_mode: {reward_mode}")

        chunk_rewards.append(step_rewards)

    # 批量写入 buffer（使用 stride 子采样，与在线训练一致）
    n = len(chunk_xs)
    stored = buffer.add_episode_strided(
        np.stack(chunk_xs),
        np.stack(chunk_actions),
        np.stack(chunk_a_tildes),
        np.stack(chunk_rewards),
        np.stack(chunk_next_xs),
        np.stack(chunk_dones),
        stride=stride,
    )

    if verbose:
        log.info("  生成 %d 个 chunk transitions → stride=%d → 存入 %d 个", n, stride, stored)

    return stored


def main() -> None:
    parser = argparse.ArgumentParser(
        description="使用离线 LeRobot 专家数据预热 ReplayBuffer",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 最小用法
  python scripts/warmup_replay_buffer.py \\
      --config configs/rlt/stage2_online_rl.yaml \\
      --vla-checkpoint-dir checkpoints/my_run/20000 \\
      --rl-token-checkpoint checkpoints/rl_token/my_run/rl_token_step5000.pt \\
      --dataset-repo-id lerobot/s1_bimanual_0420 \\
      --output warmup_buffer.pt

  # 限制 episode 数量
  python scripts/warmup_replay_buffer.py \\
      --config configs/rlt/stage2_online_rl.yaml \\
      --vla-checkpoint-dir checkpoints/my_run/20000 \\
      --rl-token-checkpoint checkpoints/rl_token/my_run/rl_token_step5000.pt \\
      --dataset-repo-id lerobot/s1_bimanual_0420 \\
      --max-episodes 50 \\
      --output warmup_buffer.pt
        """,
    )
    parser.add_argument("--config", type=str, default=None,
                        help="Stage 2 YAML 配置文件路径")
    parser.add_argument("--vla-checkpoint-dir", type=str, required=True,
                        help="VLA checkpoint 目录")
    parser.add_argument("--vla-config-name", type=str, default="configs/bi_s1/pi05_finetune.yaml",
                        help="VLA 配置名（configs/bi_s1/pi05_finetune.yaml）")
    parser.add_argument("--rl-token-checkpoint", type=str, required=True,
                        help="Stage 1 RL Token checkpoint (.pt)")
    parser.add_argument("--dataset-repo-id", type=str, required=True,
                        help="LeRobot 数据集 repo_id 或本地路径")
    parser.add_argument("--output", type=str, required=True,
                        help="输出 warmup_buffer.pt 路径")
    parser.add_argument("--task-prompt", type=str, default=None,
                        help="任务 prompt（默认使用数据集 task 字段）")
    parser.add_argument("--action-dim", type=int, default=None,
                        help="动作维度（默认从 config 读取）")
    parser.add_argument("--chunk-length", type=int, default=None,
                        help="Chunk 长度 C（默认从 config 读取）")
    parser.add_argument("--embedding-dim", type=int, default=None,
                        help="Embedding 维度（默认从 config 读取）")
    parser.add_argument("--buffer-capacity", type=int, default=100000,
                        help="ReplayBuffer 容量")
    parser.add_argument("--stride", type=int, default=2,
                        help="子采样步长")
    parser.add_argument("--max-episodes", type=int, default=None,
                        help="最大处理的 episode 数")
    parser.add_argument("--reward-mode", type=str, default="sparse",
                        choices=["zero", "sparse", "from_dataset"],
                        help="奖励模式: zero(全零), sparse(成功=1), from_dataset(从数据读取)")
    parser.add_argument("--device", type=str, default="cuda",
                        help="推理设备")
    parser.add_argument("--verbose", action="store_true",
                        help="打印详细进度")

    args = parser.parse_args()

    # ── 加载配置 ──────────────────────────────────────────────
    config: OnlineRLTrainConfig | None = None
    if args.config is not None:
        config = load_config_with_cli(OnlineRLTrainConfig, yaml_path=args.config, cli_args=[])
        log.info("从 YAML 加载配置: %s", args.config)

    # 维度参数优先级: CLI > config > 默认
    action_dim = args.action_dim or (config.action_dim if config else 8)
    chunk_length = args.chunk_length or (config.chunk_length if config else 10)
    embedding_dim = args.embedding_dim or (config.rl_arch.embedding_dim if config else 2048)
    state_dim = embedding_dim + action_dim
    action_chunk_dim = chunk_length * action_dim
    buffer_capacity = args.buffer_capacity

    log.info("参数: action_dim=%d, chunk_length=%d, embedding_dim=%d, state_dim=%d",
             action_dim, chunk_length, embedding_dim, state_dim)

    # ── 加载模型 ──────────────────────────────────────────────
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    log.info("设备: %s", device)

    log.info("加载 VLA: config=%s, checkpoint=%s", args.vla_config_name, args.vla_checkpoint_dir)
    vla = VLAWrapper(
        checkpoint_path=args.vla_checkpoint_dir,
        config_name=args.vla_config_name,
        device=device,
    )

    log.info("加载 RL Token 模型: %s", args.rl_token_checkpoint)
    rl_token_model = load_rl_token_model(args.rl_token_checkpoint, device=device)

    # 如果 Stage 1 是联合训练，恢复微调后的 VLA 权重
    stage1_ckpt = torch.load(args.rl_token_checkpoint, map_location="cpu", weights_only=False)
    if "vla_model" in stage1_ckpt:
        vla.pi0.load_state_dict(stage1_ckpt["vla_model"])
        log.info("已恢复 Stage 1 联合微调的 VLA 权重")
    del stage1_ckpt
    torch.cuda.empty_cache()

    # ── 加载数据集 ────────────────────────────────────────────
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.datasets.video_utils import decode_video_frames

    num_episodes = args.max_episodes
    ds = LeRobotDataset(
        repo_id=args.dataset_repo_id,
        episodes=list(range(num_episodes)) if num_episodes else None,
    )
    num_episodes = ds.num_episodes
    video_keys = ds.meta.video_keys
    log.info("数据集: %d 帧 (%d episodes), 摄像头: %s", len(ds), num_episodes, video_keys)

    # ── 创建 ReplayBuffer ────────────────────────────────────
    buffer = ReplayBuffer(
        capacity=buffer_capacity,
        state_dim=state_dim,
        action_chunk_dim=action_chunk_dim,
        chunk_length=chunk_length,
    )

    # ── 处理 episodes ─────────────────────────────────────────
    total_stored = 0
    ep_data_idx = ds.episode_data_index

    for ep_idx in range(num_episodes):
        start = int(ep_data_idx["from"][ep_idx].item())
        end = int(ep_data_idx["to"][ep_idx].item())
        ep_len = end - start
        log.info("Episode %d/%d: %d 帧", ep_idx + 1, num_episodes, ep_len)

        # 1) hf_dataset 批量取元数据（action, state, timestamp，不含图片）
        hf_batch = ds.hf_dataset[start:end]
        hf_timestamps = torch.stack(hf_batch["timestamp"]).tolist()

        # 2) 批量解码整个 episode 的视频帧
        t0 = time.time()
        video_frames: dict[str, torch.Tensor] = {}
        for key in video_keys:
            video_path = ds.root / ds.meta.get_video_file_path(ep_idx, key)
            video_frames[key] = decode_video_frames(
                video_path, hf_timestamps, ds.tolerance_s, ds.video_backend
            )  # [T, 3, H, W]
        log.info("  视频解码: %.1fs", time.time() - t0)

        # 3) 组装帧列表
        episode_frames: list[dict[str, Any]] = []
        task = ds.meta.tasks[int(hf_batch["task_index"][0].item())]
        for t in range(ep_len):
            frame: dict[str, Any] = {"task": task}
            frame["action"] = hf_batch["action"][t]
            frame["observation.state"] = hf_batch["observation.state"][t]
            for key in video_keys:
                frame[key] = video_frames[key][t]
            episode_frames.append(frame)

        episode_meta_entry = ds.meta.episodes[ep_idx] if ep_idx < len(ds.meta.episodes) else None

        ep_start = time.time()
        stored = process_episode(
            episode_frames=episode_frames,
            episode_meta_entry=episode_meta_entry,
            vla=vla,
            rl_token_model=rl_token_model,
            buffer=buffer,
            chunk_length=chunk_length,
            action_dim=action_dim,
            stride=args.stride,
            task_prompt=args.task_prompt,
            reward_mode=args.reward_mode,
            verbose=args.verbose,
        )
        elapsed = time.time() - ep_start
        total_stored += stored

        log.info("  存入 %d transitions (%.1fs, buffer 当前 %d/%d)",
                 stored, elapsed, buffer.size, buffer.capacity)

        if buffer.size >= buffer.capacity:
            log.warning("ReplayBuffer 已满 (%d/%d)，停止处理", buffer.size, buffer.capacity)
            break

    # ── 保存 ──────────────────────────────────────────────────
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(buffer.state_dict(), output_path)

    log.info("=" * 60)
    log.info("完成! 共存入 %d transitions → %s", total_stored, output_path)
    log.info("Buffer 大小: %d/%d", buffer.size, buffer.capacity)
    log.info("可在 Stage 2 训练中使用: --warmup-buffer %s", output_path)


if __name__ == "__main__":
    main()
