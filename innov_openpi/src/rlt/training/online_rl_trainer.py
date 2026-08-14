"""Stage 2 trainer: Online RL with frozen VLA + RL token (Algorithm 1).

Implements the full online RL loop from the paper:
1. Warmup: fill replay buffer with VLA-only rollouts
2. Main loop: collect episode → update critic/actor (UTD ratio G)
3. TD3-style updates: twin Q-critics, delayed actor, Polyak targets
"""

from __future__ import annotations

import copy
import logging
import threading
import time
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist

from rlt.models.actor import Actor
from rlt.models.critic import TwinQCritic
from rlt.models.rl_token import RLTokenModel
from rlt.rollout.intervention import InterventionManager
from rlt.rollout.rollout_worker import RolloutWorker
from rlt.rollout.switch_recorder import SwitchPointRecorder
from rlt.training.config import OnlineRLTrainConfig
from rlt.training.ddp_utils import is_main_process
from rlt.training.replay_buffer import ReplayBuffer
from rlt.training.td3_utils import actor_loss, compute_td_target, critic_loss
from rlt.utils import display
from openpi.training.vla_wrapper import VLAWrapper

logger = logging.getLogger(__name__)


class OnlineRLTrainer:
    """Stage 2: Online RL training with Algorithm 1.

    Loads frozen VLA + frozen RL token model, creates Actor + TwinQCritic +
    optimizers + ReplayBuffer + RolloutWorker, and runs the online RL loop.

    Args:
        config: Stage 2 training hyperparameters.
        vla: Frozen VLA wrapper (pre-loaded).
        rl_token_model: Frozen RL token model (loaded from Stage 1 checkpoint).
        device: Torch device for training.
    """

    def __init__(
        self,
        config: OnlineRLTrainConfig,
        vla: VLAWrapper,
        rl_token_model: RLTokenModel,
        device: torch.device | str = "cuda",
    ) -> None:
        self.config = config
        self.device = torch.device(device)

        # Frozen components
        self.vla = vla
        self.rl_token_model = rl_token_model
        self.rl_token_model.eval()
        for param in self.rl_token_model.parameters():
            param.requires_grad_(False)

        # Trainable actor
        self.actor = Actor(
            state_dim=config.state_dim,
            action_chunk_dim=config.action_chunk_dim,
            hidden_dim=config.rl_arch.mlp_hidden_dim,
            num_hidden_layers=config.rl_arch.mlp_num_hidden_layers,
            sigma=config.rl_arch.actor_noise_sigma,
            ref_dropout=config.rl_arch.ref_action_dropout,
        ).to(self.device)

        # Trainable twin Q-critic
        self.critic = TwinQCritic(
            state_dim=config.state_dim,
            action_chunk_dim=config.action_chunk_dim,
            hidden_dim=config.rl_arch.mlp_hidden_dim,
            num_hidden_layers=config.rl_arch.mlp_num_hidden_layers,
        ).to(self.device)

        # Optimizers
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=config.optimizer.actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=config.optimizer.critic_lr)

        # Replay buffer
        self.replay_buffer = ReplayBuffer(
            capacity=config.buffer.capacity,
            state_dim=config.state_dim,
            action_chunk_dim=config.action_chunk_dim,
            chunk_length=config.chunk_length,
        )

        # Counters
        self._total_env_steps = 0
        self._total_updates = 0
        self._total_episodes = 0

        # 异步收集/训练:行为 actor 副本(采集推理用)+ 线程同步原语
        self.actor_policy = copy.deepcopy(self.actor)
        self.actor_policy.eval()
        self._stats_lock = threading.Lock()
        self._stop = threading.Event()
        self._new_episode = threading.Event()
        self._episode_stats: list[Any] = []

    def _create_rollout_worker(
        self,
        env: Any,
        intervention_mgr: InterventionManager | None = None,
    ) -> RolloutWorker:
        """Create a rollout worker wired to this trainer's components."""
        switch_dir = self.config.checkpoint.switch_data_dir or None
        return RolloutWorker(
            env=env,
            vla=self.vla,
            rl_token_model=self.rl_token_model,
            actor=self.actor_policy,  # 行为副本:训练线程更新在线 actor 后定期同步
            replay_buffer=self.replay_buffer,
            intervention_mgr=intervention_mgr or InterventionManager(),
            chunk_length=self.config.chunk_length,
            action_dim=self.config.action_dim,
            device=self.device,
            critical_phase_only=self.config.critical_phase_only,
            obs_stride=self.config.obs_stride,
            ctrl=getattr(env, "ctrl", None),
            switch_recorder=SwitchPointRecorder(
                save_dir=switch_dir,
                save_raw_obs=self.config.checkpoint.switch_save_raw_obs,
            )
            if switch_dir
            else None,
        )

    def _update_step(self, update_idx: int) -> dict[str, float]:
        """Run one TD3 update step.

        Critic is updated every call.  Actor is updated every
        ``critic_updates_per_actor`` calls.

        Args:
            update_idx: 0-based index within the current UTD batch.

        Returns:
            Dict of logged metrics.
        """
        cfg = self.config
        batch = self.replay_buffer.sample(batch_size=cfg.buffer.batch_size, device=str(self.device))

        x = batch["x"]
        a = batch["a"]
        a_tilde = batch["a_tilde"]
        rewards = batch["rewards"]
        next_x = batch["next_x"]
        dones = batch["dones"]

        # --- Critic update (every step) ---
        # Compute TD target
        # For next_a_tilde, we use a_tilde from the batch as an approximation
        # (the true next reference would require a VLA call, which is expensive)
        td_target = compute_td_target(
            rewards=rewards,
            dones=dones,
            next_x=next_x,
            next_a_tilde=a_tilde,  # approximate next reference
            actor=self.actor,
            critic=self.critic,
            gamma=cfg.td3.gamma,
            chunk_length=cfg.chunk_length,
            target_noise_sigma=cfg.td3.target_noise_sigma,
            target_noise_clip=cfg.td3.target_noise_clip,
        )

        q1, q2 = self.critic(x, a)
        c_loss = critic_loss(q1, q2, td_target)

        self.critic_optimizer.zero_grad()
        c_loss.backward()
        self.critic_optimizer.step()

        metrics: dict[str, float] = {
            "critic_loss": c_loss.item(),
            "q1_mean": q1.mean().item(),
            "q2_mean": q2.mean().item(),
        }

        # --- Actor update (delayed) ---
        if update_idx % cfg.td3.critic_updates_per_actor == 0:
            self.actor.train()
            a_actor = self.actor(x, a_tilde)
            q_value = self.critic.q_min(x, a_actor)
            a_loss = actor_loss(q_value, a_actor, a_tilde, cfg.td3.bc_regularizer_beta)

            self.actor_optimizer.zero_grad()
            a_loss.backward()
            self.actor_optimizer.step()

            metrics["actor_loss"] = a_loss.item()

        # --- Polyak target update ---
        self.critic.update_targets(cfg.td3.tau)

        self._total_updates += 1
        return metrics

    def train(
        self,
        env: Any,
        intervention_mgr: InterventionManager | None = None,
        log_fn: Any | None = None,
    ) -> None:
        """Run the full online RL training loop (Algorithm 1).

        Args:
            env: Chunk-level environment wrapper.
            intervention_mgr: Optional human intervention manager.
            log_fn: Optional callable ``log_fn(metrics_dict)`` for logging.
        """
        cfg = self.config
        main = is_main_process()
        worker = self._create_rollout_worker(env, intervention_mgr)
        train_display = display.TrainingDisplay(window_size=20) if main else None
        train_start = time.time()

        if main:
            display.training_start({
                "Task": cfg.env.task_prompt or "(not set)",
                "Max env steps": f"{cfg.max_env_steps:,}",
                "UTD ratio": str(cfg.td3.utd_ratio),
                "Chunk length": str(cfg.chunk_length),
                "Action dim": str(cfg.action_dim),
                "Run name": cfg.checkpoint.run_name,
            })

        # Phase 1: Warmup with VLA-only policy (skip if buffer already has data)
        if self.replay_buffer.size > 0:
            if main:
                logger.info(
                    "Skipping warmup — replay buffer already has %d transitions (resumed from checkpoint)",
                    self.replay_buffer.size,
                )
        elif cfg.checkpoint.warmup_buffer:
            self._load_warmup_buffer(cfg.checkpoint.warmup_buffer)
        else:
            if main:
                display.warmup_start(cfg.buffer.warmup_steps)
            stored = 0
            obs = env.reset()
            for i in range(cfg.buffer.warmup_steps):
                action_chunk = worker._get_warmup_action(obs)
                x, a_tilde_flat, _ = worker._extract_rl_state(obs)
                a_flat = action_chunk.reshape(-1)
                next_obs, rewards, done, _info = env.step(action_chunk)
                next_x, _, _ = worker._extract_rl_state(next_obs)
                self.replay_buffer.add(
                    x=x, a=a_flat, a_tilde=a_tilde_flat,
                    rewards=rewards, next_x=next_x, done=float(done),
                )
                stored += 1
                self._total_env_steps += cfg.chunk_length
                if main:
                    display.warmup_progress(i + 1, cfg.buffer.warmup_steps)
                if done:
                    obs = env.reset()
                else:
                    obs = next_obs
            if main:
                display.warmup_done(stored, self.replay_buffer.size)
            self._save_warmup_buffer()

        # Phase 2: Online RL loop (同步或异步)
        self._sync_actor_policy()
        if hasattr(env, '_display_episode_num'):
            env._display_episode_num = self._total_episodes + 1

        if cfg.async_collection:
            collector = threading.Thread(
                target=self._collect_loop, args=(worker,), daemon=True, name="rlt-collector"
            )
            collector.start()
            if main:
                logger.info("Async mode: rollout collector thread started")
            try:
                self._train_loop(train_display, log_fn, main)
            finally:
                self._stop.set()
                collector.join(timeout=5.0)
                if collector.is_alive():
                    logger.warning("Rollout collector did not exit cleanly")
        else:
            # 同步模式:收集 1 episode → UTD 更新
            while self._total_env_steps < cfg.max_env_steps:
                if hasattr(env, '_display_episode_num'):
                    env._display_episode_num = self._total_episodes + 1
                self.actor.eval()
                stats = worker.collect_episode()
                self._total_episodes += 1
                self._total_env_steps += stats.num_steps
                self._handle_episodes([stats], train_display, log_fn, main)
                self._run_updates_and_save(train_display, log_fn, main)

        # Final save
        ckpt_path = self.save()
        if main and ckpt_path is not None:
            display.checkpoint_saved(str(ckpt_path))
            display.training_done(
                self._total_episodes,
                self._total_env_steps,
                self._total_updates,
                time.time() - train_start,
            )

    # ------------------------------------------------------------------
    # 异步模式:采集线程 + 训练主线程
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _sync_actor_policy(self) -> None:
        """在线 actor 权重 → 行为 actor(采集线程推理用)。"""
        self.actor_policy.load_state_dict(self.actor.state_dict())

    def _collect_loop(self, worker: RolloutWorker) -> None:
        """采集线程:持续收集 episode 写入共享 replay buffer。"""
        while not self._stop.is_set():
            try:
                stats = worker.collect_episode()
            except Exception:
                logger.exception("Rollout collector crashed; stopping training")
                self._stop.set()
                self._new_episode.set()
                break
            with self._stats_lock:
                self._total_episodes += 1
                self._total_env_steps += stats.num_steps
                self._episode_stats.append(stats)
            if hasattr(worker.env, "_display_episode_num"):
                with self._stats_lock:
                    worker.env._display_episode_num = self._total_episodes + 1
            self._new_episode.set()

    def _train_loop(
        self,
        train_display: Any,
        log_fn: Any | None,
        main: bool,
    ) -> None:
        """训练主线程:等待新 episode → UTD 更新 → 同步行为 actor。"""
        cfg = self.config
        while not self._stop.is_set():
            with self._stats_lock:
                if self._total_env_steps >= cfg.max_env_steps:
                    break
            if not self._new_episode.wait(timeout=1.0):
                continue
            self._new_episode.clear()
            with self._stats_lock:
                episodes = self._episode_stats
                self._episode_stats = []
            if not episodes:
                continue
            self._handle_episodes(episodes, train_display, log_fn, main)
            self._run_updates_and_save(train_display, log_fn, main)

    def _handle_episodes(
        self,
        episodes: list[Any],
        train_display: Any,
        log_fn: Any | None,
        main: bool,
    ) -> None:
        """显示与记录一批 episode 统计。"""
        for stats in episodes:
            success = stats.extra.get("success", False)
            if train_display is not None:
                train_display.record_episode(success, stats.total_reward)
            if main:
                display.episode_result(
                    episode_num=self._total_episodes,
                    total_reward=stats.total_reward,
                    success=success,
                    num_chunks=stats.num_chunks,
                    num_steps=stats.num_steps,
                    interventions=stats.interventions,
                )
            episode_metrics = {
                "episode_reward": stats.total_reward,
                "episode_success": int(success),
                "episode_chunks": stats.num_chunks,
                "episode_steps": stats.num_steps,
                "episode_interventions": stats.interventions,
                "total_env_steps": self._total_env_steps,
                "total_episodes": self._total_episodes,
                "buffer_size": self.replay_buffer.size,
            }
            if log_fn is not None:
                log_fn(episode_metrics)

    def _run_updates_and_save(
        self,
        train_display: Any,
        log_fn: Any | None,
        main: bool,
    ) -> None:
        """UTD 更新 → 显示/日志 → 权重同步 → 定期保存。"""
        cfg = self.config
        update_metrics: dict[str, float] = {}
        for g in range(cfg.td3.utd_ratio):
            update_metrics = self._update_step(g)
        self._sync_actor_policy()

        if train_display is not None:
            train_display.print_summary(
                total_episodes=self._total_episodes,
                total_env_steps=self._total_env_steps,
                max_env_steps=cfg.max_env_steps,
                buffer_size=self.replay_buffer.size,
                critic_loss=update_metrics.get("critic_loss", 0.0),
                actor_loss=update_metrics.get("actor_loss"),
                q_mean=update_metrics.get("q1_mean", 0.0),
            )
        if log_fn is not None:
            log_fn(update_metrics)

        if self._total_episodes % cfg.checkpoint.save_every == 0:
            ckpt_path = self.save()
            if main and ckpt_path is not None:
                display.checkpoint_saved(str(ckpt_path))

    def _save_warmup_buffer(self) -> Path:
        """Save the replay buffer as a standalone file after warmup."""
        ckpt = self.config.checkpoint
        save_dir = Path(ckpt.save_dir) / ckpt.run_name
        save_dir.mkdir(parents=True, exist_ok=True)
        buf_path = save_dir / "warmup_buffer.pt"
        torch.save(self.replay_buffer.state_dict(), buf_path)
        logger.info("Saved warmup buffer (%d transitions) to %s", self.replay_buffer.size, buf_path)
        return buf_path

    def _load_warmup_buffer(self, path: str) -> None:
        """Load a standalone warmup buffer file into the replay buffer."""
        buf_state = torch.load(path, map_location="cpu", weights_only=False)
        self.replay_buffer.load_state_dict(buf_state)
        logger.info("Loaded warmup buffer (%d transitions) from %s", self.replay_buffer.size, path)

    def save(self, path: str | None = None, save_buffer: bool = True) -> Path | None:
        """Save actor, critic, optimizer, and replay buffer states.

        When DDP is active, only rank 0 writes the checkpoint.

        Args:
            path: Override save path. Defaults to config.checkpoint.save_dir.
            save_buffer: Whether to include replay buffer in the checkpoint.
                Disable for eval-only checkpoints to save disk space.

        Returns:
            Path to the saved checkpoint, or ``None`` on non-zero ranks.
        """
        if not is_main_process():
            if dist.is_initialized():
                dist.barrier()
            return None

        ckpt_cfg = self.config.checkpoint
        save_dir = Path(path or ckpt_cfg.save_dir) / ckpt_cfg.run_name
        save_dir.mkdir(parents=True, exist_ok=True)
        with self._stats_lock:
            total_env_steps = self._total_env_steps
            total_updates = self._total_updates
            total_episodes = self._total_episodes
        ckpt_path = save_dir / f"online_rl_ep{total_episodes}.pt"
        payload: dict[str, Any] = {
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict(),
            "actor_optimizer": self.actor_optimizer.state_dict(),
            "critic_optimizer": self.critic_optimizer.state_dict(),
            "total_env_steps": total_env_steps,
            "total_updates": total_updates,
            "total_episodes": total_episodes,
            "config": self.config,
        }
        if save_buffer:
            payload["replay_buffer"] = self.replay_buffer.state_dict()
        torch.save(payload, ckpt_path)
        logger.info("Saved checkpoint to %s (buffer=%s)", ckpt_path, save_buffer)

        if dist.is_initialized():
            dist.barrier()
        return ckpt_path

    def load(self, ckpt_path: str) -> None:
        """Load actor, critic, optimizer, and replay buffer from checkpoint.

        Args:
            ckpt_path: Path to a saved checkpoint file.
        """
        ckpt = torch.load(ckpt_path, map_location=self.device, weights_only=False)
        self.actor.load_state_dict(ckpt["actor"])
        self.critic.load_state_dict(ckpt["critic"])
        self.actor_optimizer.load_state_dict(ckpt["actor_optimizer"])
        self.critic_optimizer.load_state_dict(ckpt["critic_optimizer"])
        self._total_env_steps = ckpt["total_env_steps"]
        self._total_updates = ckpt["total_updates"]
        self._total_episodes = ckpt["total_episodes"]

        if "replay_buffer" in ckpt:
            self.replay_buffer.load_state_dict(ckpt["replay_buffer"])
            logger.info(
                "Restored replay buffer (%d transitions)", self.replay_buffer.size
            )

        logger.info(
            "Loaded checkpoint from %s (episode %d, step %d)",
            ckpt_path,
            self._total_episodes,
            self._total_env_steps,
        )
