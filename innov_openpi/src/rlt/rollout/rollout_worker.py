"""Rollout worker for online RL data collection.

Orchestrates environment interaction, VLA embedding extraction, RL token
encoding, actor inference, and replay buffer storage.  Supports both
VLA-only warmup rollouts and full RL episode collection with optional
human intervention.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)

import numpy as np
import torch
from numpy.typing import NDArray

from rlt.models.actor import Actor
from rlt.models.rl_token import RLTokenModel
from rlt.rollout.intervention import InterventionManager, InterventionResult
from rlt.rollout.keyboard_ctrl import KeyboardCtrl
from rlt.rollout.switch_recorder import SwitchPointRecorder
from rlt.training.replay_buffer import ReplayBuffer
from openpi.training.vla_wrapper import VLAWrapper


@dataclass
class EpisodeStats:
    """Statistics for a single collected episode."""

    total_reward: float = 0.0
    num_chunks: int = 0
    num_steps: int = 0
    done: bool = False
    interventions: int = 0
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class _ChunkData:
    """单个 chunk 的原始数据,缓存用于 stride 错位窗口组装。

    起点 0 的 transition 在下一 chunk 边界落盘(next_x = 下一起点状态);
    错位窗口在下一 chunk 执行完成(逐步数据齐全)后组装。
    """

    x: NDArray                    # [state_dim] 起点 RL 状态
    x_obs: dict[str, Any]         # 起点原始观测(合并观测序列用)
    a_tilde: NDArray              # [C*d] 存入 buffer 的参考(干预时=人类动作)
    a_tilde_full: NDArray         # [H, d] VLA 完整参考块
    a_flat: NDArray               # [C*d] 执行动作
    rewards: NDArray              # [C] 逐步奖励
    done: bool
    intervention: bool
    skip: bool
    obs_history: list[dict[str, Any]] | None = None   # 错位起点观测(不含起点)
    actions_history: NDArray | None = None            # [executed, d] 实际执行
    rewards_history: NDArray | None = None            # [executed] 逐步奖励
    done_history: NDArray | None = None               # [executed] 逐步终止


class RolloutWorker:
    """Collects environment rollouts for online RL training.

    During warmup, runs the VLA-only policy and stores transitions.
    During RL training, uses the actor conditioned on z_rl and VLA
    reference actions, with optional human intervention override.

    Args:
        env: Chunk-level environment wrapper.
        vla: Frozen VLA wrapper for embeddings and reference actions.
        rl_token_model: Frozen RL token encoder (Stage 1 output).
        actor: RL actor network.
        replay_buffer: Buffer to store transitions.
        intervention_mgr: Human intervention manager.
        chunk_length: C, number of steps per action chunk.
        action_dim: Dimension of a single-step action.
        device: Torch device for model inference.
        critical_phase_only: 阶段 RL 模式(episode 从 VLA 开始,rl_active 初始 False)。
        obs_stride: stride 子采样步长(论文 stride=2;1 = 禁用)。
        ctrl: 共享键盘监听(检测 t 切换 rl_active);None 时不支持切换。
        switch_recorder: 切换点记录器(分类器训练数据);None 时不记录。
        skip_chunks_after_reset: episode 开始跳过前 N 个 chunk(远端流水线排空用,本地直连为 0)。
    """

    def __init__(
        self,
        env: Any,
        vla: VLAWrapper,
        rl_token_model: RLTokenModel,
        actor: Actor,
        replay_buffer: ReplayBuffer,
        intervention_mgr: InterventionManager,
        chunk_length: int,
        action_dim: int,
        device: torch.device | str = "cuda",
        critical_phase_only: bool = False,
        obs_stride: int = 2,
        ctrl: KeyboardCtrl | None = None,
        switch_recorder: SwitchPointRecorder | None = None,
        skip_chunks_after_reset: int = 0,
    ) -> None:
        self.env = env
        self.vla = vla
        self.rl_token_model = rl_token_model
        self.actor = actor
        self.replay_buffer = replay_buffer
        self.intervention_mgr = intervention_mgr
        self.chunk_length = chunk_length
        self.action_dim = action_dim
        self.device = torch.device(device)
        self.critical_phase_only = critical_phase_only
        self.obs_stride = max(1, obs_stride)
        self.ctrl = ctrl
        self.switch_recorder = switch_recorder
        self.skip_chunks_after_reset = skip_chunks_after_reset
        # 阶段 RL 模式从 VLA 开始(关键阶段才切到 RL);全局模式全程 RL
        self._rl_active = not critical_phase_only

        self._action_chunk_dim = chunk_length * action_dim

    def _obs_to_vla_input(self, obs: dict[str, Any]) -> Any:
        """Prepare observation dict for VLA inference.

        Uses ``VLAWrapper.preprocess_obs`` if available (real VLA), which
        applies the full OpenPI transform chain and returns an
        ``Observation``.  Falls back to simple batch-wrapping for tests
        with a mock VLA.
        """
        if hasattr(self.vla, "preprocess_obs"):
            return self.vla.preprocess_obs(obs)

        # Fallback: simple batch-wrap (for tests with mock VLA)
        batched: dict[str, Any] = {}
        for key, val in obs.items():
            arr = np.asarray(val)
            batched[key] = arr[np.newaxis]  # add batch dim
        return batched

    @torch.no_grad()
    def _extract_rl_state(self, obs: dict[str, Any]) -> tuple[NDArray, NDArray, NDArray]:
        """Extract RL state x = cat(z_rl, s^p) and VLA reference chunks.

        Returns:
            x: RL state [state_dim] as numpy array.
            a_tilde_flat: Flattened first-C-step VLA reference [action_chunk_dim].
            a_tilde_full: Full H-step VLA reference [H, d] (stride windows slice from it).
        """
        vla_input = self._obs_to_vla_input(obs)

        # Extract VLA embeddings and encode into z_rl
        z, pad_mask = self.vla.extract_embeddings(vla_input)
        z_rl = self.rl_token_model.encode(z, pad_mask)  # [1, D]

        # Full H-step VLA reference (sample once, slice first C steps for the actor)
        a_tilde_full = self.vla.sample_reference_actions(vla_input)  # [1, H, d]
        a_tilde = a_tilde_full[:, : self.chunk_length, :]  # [1, C, d]
        a_tilde_flat = a_tilde.reshape(1, -1)  # [1, C*d]

        # Proprioceptive state s^p from the preprocessed VLA observation.
        # DroidInputs merges joint_pos + gripper into state, then
        # PadStatesAndActions zero-pads to the VLA's internal width.
        # Slice to action_dim to drop the padding.
        s_p = vla_input.state[:, :self.action_dim].to(dtype=torch.float32, device=self.device)  # [1, d]

        # RL state: x = cat(z_rl, s^p)
        x = torch.cat([z_rl, s_p], dim=-1)  # [1, state_dim]

        return (
            x.squeeze(0).cpu().numpy(),
            a_tilde_flat.squeeze(0).cpu().numpy(),
            a_tilde_full.squeeze(0).cpu().numpy(),  # [H, d]
        )

    @torch.no_grad()
    def _get_warmup_action(self, obs: dict[str, Any]) -> NDArray:
        """Get action from VLA-only policy for warmup / RL OFF 执行。

        Returns:
            action_chunk: [C, action_dim] numpy array.
        """
        vla_input = self._obs_to_vla_input(obs)
        a_tilde = self.vla.sample_reference_actions(vla_input)  # [1, H, d]
        return a_tilde[:, : self.chunk_length, :].squeeze(0).cpu().numpy()  # [C, d]

    @torch.no_grad()
    def _get_actor_action(self, x: NDArray, a_tilde_flat: NDArray) -> NDArray:
        """从 RL actor 获取动作（含探索噪声）。

        Actor.forward 返回确定性 μ；探索噪声在此处显式添加，与 TD3 的
        actor loss 计算解耦。

        Args:
            x: RL state [state_dim]。
            a_tilde_flat: VLA 参考动作块 [action_chunk_dim]。

        Returns:
            action_chunk: [C, action_dim] numpy 数组。
        """
        x_t = torch.as_tensor(x, dtype=torch.float32, device=self.device).unsqueeze(0)
        a_tilde_t = torch.as_tensor(a_tilde_flat, dtype=torch.float32, device=self.device).unsqueeze(0)

        mu = self.actor(x_t, a_tilde_t)  # [1, C*d] 确定性输出
        if self.actor.sigma > 0:
            noise = torch.randn_like(mu) * self.actor.sigma
            mu = (mu + noise).clamp(-1.0, 1.0)
        return mu.squeeze(0).cpu().numpy().reshape(self.chunk_length, self.action_dim)

    def collect_warmup(self, num_chunks: int) -> int:
        """Run VLA-only policy and store transitions in the replay buffer.

        Collects ``num_chunks`` chunk-level transitions across potentially
        multiple episodes (auto-resets on termination).

        Args:
            num_chunks: Number of chunk-level transitions to collect.

        Returns:
            Total number of transitions stored.
        """
        stored = 0
        obs = self.env.reset()

        for _ in range(num_chunks):
            # Get VLA reference action (used as both executed and reference)
            action_chunk = self._get_warmup_action(obs)  # [C, action_dim]

            # Build RL state for this observation
            x, a_tilde_flat, _ = self._extract_rl_state(obs)
            a_flat = action_chunk.reshape(-1)  # [C*d]

            # Step environment
            next_obs, rewards, done, _info = self.env.step(action_chunk)

            # Build next RL state
            next_x, _, _ = self._extract_rl_state(next_obs)

            # Store transition
            self.replay_buffer.add(
                x=x,
                a=a_flat,
                a_tilde=a_tilde_flat,
                rewards=rewards,
                next_x=next_x,
                done=float(done),
            )
            stored += 1

            if done:
                obs = self.env.reset()
            else:
                obs = next_obs

        return stored

    def collect_episode(self, store_transitions: bool = True) -> EpisodeStats:
        """Collect a single RL episode using the actor policy.

        When ``critical_phase_only`` is True, transitions are only
        stored when the Robot PC has RL toggled ON (``rl_active=True``).
        RL OFF chunks are skipped entirely.

        Transition storage (论文 Algorithm 1 + Subsampling Action Chunks):
        - 起点 0 的 transition 在下一 chunk 边界落盘(next_x = 下一起点状态)
        - stride 错位窗口(<x_2, a_2:C+2>, <x_4, a_4:C+4> ...)在下一 chunk 执行
          完成(逐步数据齐全)后组装落盘
        - 干预 chunk 的参考动作替换为人类动作(ã ← a_human,算法第 11 行)

        Args:
            store_transitions: Whether to add transitions to the replay
                buffer.  Set to ``False`` during evaluation.

        Returns:
            Episode statistics.
        """
        stats = EpisodeStats()
        obs = self.env.reset()

        # 远端流水线排空(remote 模式)用;本地直接控制为 0
        skip_remaining = self.skip_chunks_after_reset

        prev: _ChunkData | None = None

        while True:
            # Extract RL state and VLA reference
            x, a_tilde_flat, a_tilde_full = self._extract_rl_state(obs)
            vla_action = a_tilde_flat.reshape(self.chunk_length, self.action_dim)

            # Set VLA reference on remote env so Robot PC can choose
            if hasattr(self.env, "set_vla_action"):
                self.env.set_vla_action(vla_action)

            # 阶段/全局 RL 切换:按 t 键切换 rl_active,并记录切换点(分类器训练数据)
            if self.ctrl is not None:
                sig = self.ctrl.poll()
                if sig == "t":
                    self._rl_active = not self._rl_active
                    self._record_switch_point(x, a_tilde_flat, obs)
                    logger.info("RL mode toggled %s", "ON" if self._rl_active else "OFF")

            # Check for human intervention
            intervention: InterventionResult | None = None
            if self.intervention_mgr.check_intervention():
                intervention = self.intervention_mgr.get_human_action(
                    self.action_dim, self.chunk_length
                )

            if intervention is not None:
                action_chunk = intervention.action_chunk
                next_obs = intervention.next_obs
                rewards = intervention.rewards
                done = intervention.done
                info = intervention.info
                stats.interventions += 1
            else:
                # RL OFF(阶段模式关键阶段外):VLA 参考直接执行,不训练
                if self._rl_active:
                    action_chunk = self._get_actor_action(x, a_tilde_flat)
                else:
                    action_chunk = self._get_warmup_action(obs)
                next_obs, rewards, done, info = self.env.step(action_chunk)

            # 远程人类干预：单帧累积为 chunk
            is_intervention = intervention is not None
            if info.get("intervention"):
                is_intervention = True
                stats.interventions += 1
                human_frames: list[NDArray] = [np.asarray(info["action"], dtype=np.float32)]

                # 循环收集直至攒满 chunk_length 帧或 episode 终止
                while len(human_frames) < self.chunk_length and not done:
                    placeholder = self._get_actor_action(x, a_tilde_flat)
                    next_obs, frame_r, done, info = self.env.step(placeholder)
                    if info.get("intervention"):
                        human_frames.append(np.asarray(info["action"], dtype=np.float32))
                    if done:
                        rewards = frame_r  # 终止帧携带奖励信号

                # 不足 C 帧时末尾帧补齐
                while len(human_frames) < self.chunk_length:
                    human_frames.append(human_frames[-1].copy())

                action_chunk = np.stack(human_frames[: self.chunk_length])
                info["steps_executed"] = min(len(human_frames), self.chunk_length)

            a_flat = action_chunk.reshape(-1)
            # 本地键盘控制优先;remote env 兼容(无 ctrl 时用 info 标志)
            rl_active = self._rl_active if self.ctrl is not None else info.get("rl_active", True)

            # == 上一 chunk 落盘(终点 = 本 chunk 起点 x + 本 chunk 逐步历史)==
            # 终止 chunk (done=True) 也存入,它携带 reward 信号。
            if store_transitions and prev is not None and not prev.skip:
                self.replay_buffer.add(
                    x=prev.x, a=prev.a_flat, a_tilde=prev.a_tilde,
                    rewards=prev.rewards, next_x=x, done=float(prev.done),
                )
                # stride 错位窗口(需本 chunk 逐步数据作终点)
                if (self.obs_stride > 1 and prev.obs_history is not None
                        and not prev.done and not prev.intervention):
                    self._store_strided(prev, cur_obs=obs, cur_info=info)

            # 缓存本 chunk 数据(下轮循环或 done 后落盘)
            # RL OFF 阶段不存(论文:只训练关键阶段 transition);干预 chunk 恒存
            skip_store = (skip_remaining > 0) or (not rl_active and not is_intervention)
            if skip_remaining > 0:
                skip_remaining -= 1
            prev = _ChunkData(
                x=x,
                x_obs=obs,
                # 干预 chunk:参考替换为人类动作(论文算法第 11 行)
                a_tilde=a_flat if is_intervention else a_tilde_flat,
                a_tilde_full=a_tilde_full,
                a_flat=a_flat,
                rewards=rewards,
                done=done,
                intervention=is_intervention,
                skip=skip_store,
                obs_history=info.get("obs_history"),
                actions_history=info.get("actions_history"),
                rewards_history=info.get("rewards_history"),
                done_history=info.get("done_history"),
            )

            # Update stats
            stats.total_reward += float(rewards.sum())
            stats.num_chunks += 1
            stats.num_steps += info.get("steps_executed", self.chunk_length)

            if done:
                stats.done = True
                stats.extra = info
                break

            obs = next_obs

        # done 的最后一 chunk 落盘(终点状态需提取)
        if store_transitions and prev is not None and not prev.skip:
            next_x, _, _ = self._extract_rl_state(next_obs)
            self.replay_buffer.add(
                x=prev.x, a=prev.a_flat, a_tilde=prev.a_tilde,
                rewards=prev.rewards, next_x=next_x, done=1.0,
            )

        if self.switch_recorder is not None:
            self.switch_recorder.end_episode()

        return stats

    def _record_switch_point(self, x: NDArray, a_tilde: NDArray, obs: dict[str, Any]) -> None:
        """记录切换点:z_rl 特征 + 参考动作 ã + 切换方向标签(1 = 切到 RL)。"""
        if self.switch_recorder is None:
            return
        z_rl = x[: len(x) - self.action_dim]  # x = cat(z_rl, s^p)
        self.switch_recorder.record(
            z_rl, 1 if self._rl_active else 0,
            obs if self.switch_recorder.save_raw_obs else None,
            a_tilde,
        )

    @torch.no_grad()
    def _extract_x_batch(self, obs_list: list[dict[str, Any]]) -> NDArray:
        """批量提取 RL 状态 x = cat(z_rl, s^p),一次 VLA 前向。

        Args:
            obs_list: 观测 dict 列表(直接格式 state/images/prompt)。

        Returns:
            xs: RL 状态 [N, state_dim]。
        """
        vla_input = self.vla.preprocess_obs_batch(obs_list)
        z, pad_mask = self.vla.extract_embeddings(vla_input)
        z_rl = self.rl_token_model.encode(z, pad_mask)  # [N, D]
        s_p = vla_input.state[:, :self.action_dim].to(dtype=torch.float32, device=self.device)
        return torch.cat([z_rl, s_p], dim=-1).cpu().numpy()  # [N, state_dim]

    def _store_strided(self, prev: _ChunkData, cur_obs: dict[str, Any], cur_info: dict[str, Any]) -> None:
        """组装上一 chunk 的 stride 错位窗口并写入 buffer(论文 stride=2 子采样)。

        窗口 <x_k, a_k:k+C, ã_k:k+C, r_k:k+C, x_{k+C}>(起点 k ∈ {s, ..., C-s}):
        - x_k: prev 错位观测(批量提取 z_rl)
        - a_k:k+C: prev 实际执行动作 ∥ cur 前 k 帧
        - ã_k:k+C: 从 prev 的 H 长 VLA 参考切出(无需跨 chunk)
        - r/done: 逐步序列拼接

        Args:
            prev: 上一 chunk 缓存(数据源)。
            cur_obs: 当前 chunk 起点观测(合并序列中 prev 终点的位置)。
            cur_info: 当前 chunk 的 env step info(含逐步历史)。
        """
        stride = self.obs_stride
        C = self.chunk_length
        n_slots = C // stride

        prev_obs_h = prev.obs_history
        cur_obs_h = cur_info.get("obs_history")
        cur_acts = cur_info.get("actions_history")
        cur_rews = cur_info.get("rewards_history")
        cur_dones = cur_info.get("done_history")
        # 数据不完整(干预/提前终止/旧 env 无历史)则跳过
        if (not prev_obs_h or not cur_obs_h or len(prev_obs_h) < n_slots - 1
                or len(cur_obs_h) < n_slots - 1
                or prev.actions_history is None or cur_acts is None
                or len(prev.actions_history) < C or len(cur_acts) < C):
            return

        # 合并逐步序列:观测每 stride 步一点,动作/奖励/完成逐帧
        obs_series = [prev.x_obs, *prev_obs_h, cur_obs, *cur_obs_h]  # 2*n_slots 点
        acts = np.concatenate([prev.actions_history, cur_acts], axis=0)  # [2C, d]
        rews = np.concatenate([prev.rewards_history, cur_rews], axis=0)  # [2C]
        dones = np.concatenate([prev.done_history, cur_dones], axis=0)  # [2C]

        # 错位观测(除两个起点):一次 batch 前向提取
        to_extract = obs_series[1:n_slots] + obs_series[n_slots + 1:]
        xs = self._extract_x_batch(to_extract)  # [2*(n_slots-1), state_dim]

        for j in range(1, n_slots):
            k = j * stride
            if k + C > len(prev.a_tilde_full):
                continue  # 参考窗口越界(H 不足)
            x_k = xs[j - 1]                  # 起点:prev 错位段
            x_end = xs[j + n_slots - 2]      # 终点:cur 错位段
            a_win = acts[k:k + C].reshape(-1)
            ref = prev.a_tilde_full[k:k + C].reshape(-1)
            r_win = rews[k:k + C]
            done_flag = float(dones[k + C - 1])
            self.replay_buffer.add(
                x=x_k, a=a_win, a_tilde=ref, rewards=r_win,
                next_x=x_end, done=done_flag,
            )
