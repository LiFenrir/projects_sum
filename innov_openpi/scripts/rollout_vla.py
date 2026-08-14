"""Roll out a fine-tuned VLA on a real robot (no RL actor, no RL token).

Loads the PI-0.5 model, creates the environment, and runs episodes using
only the VLA's sampled action chunks.  Useful for evaluating a
fine-tuned VLA before or without Stage 2 RL training.

Usage:
    python scripts/rollout_vla.py --help
    python scripts/rollout_vla.py \
        --env-factory rlt_openpi.envs.franka.env_factory.make_franka_env \
        --vla-config-name pi05_droid_finetune \
        --vla-checkpoint-dir /path/to/model.safetensors \
        --task-prompt "stack the three blocks on the tray" \
        --num-episodes 10
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import torch
import tyro

from rlt.rollout.factory import make_env
from openpi.training.vla_wrapper import VLAWrapper

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(message)s")
log = logging.getLogger(__name__)


@dataclass
class RolloutConfig:
    """VLA-only rollout configuration."""

    vla_config_name: str = "pi05_droid_finetune"
    vla_checkpoint_dir: str = ""
    stage1_checkpoint: str = ""  # Stage 1 .pt checkpoint with fine-tuned VLA weights
    env_factory: str = ""
    task_prompt: str = ""
    action_dim: int = 8
    chunk_length: int = 10
    num_episodes: int = 10
    save_dir: str = "results"  # Directory to save results JSON
    device: str = "cuda"


def main(config: RolloutConfig) -> None:
    """Run VLA-only rollout episodes."""
    log.info("Rollout config: %s", config)

    if not config.vla_checkpoint_dir:
        log.error("--vla-checkpoint-dir is required.")
        raise SystemExit(1)
    if not config.env_factory:
        log.error("--env-factory is required.")
        raise SystemExit(1)

    # Load VLA
    log.info("Loading VLA: config=%s, checkpoint=%s", config.vla_config_name, config.vla_checkpoint_dir)
    vla = VLAWrapper(
        checkpoint_path=config.vla_checkpoint_dir,
        config_name=config.vla_config_name,
        device=config.device,
    )

    if config.stage1_checkpoint:
        ckpt = torch.load(config.stage1_checkpoint, map_location=config.device, weights_only=False)
        if "vla_model" in ckpt:
            vla.pi0.load_state_dict(ckpt["vla_model"])
            log.info("Loaded fine-tuned VLA weights from %s", config.stage1_checkpoint)
        else:
            log.warning("No vla_model key in %s; using base VLA weights", config.stage1_checkpoint)
        del ckpt

    # Create environment
    env = make_env(
        config.env_factory,
        action_dim=config.action_dim,
        chunk_length=config.chunk_length,
        task_prompt=config.task_prompt,
    )
    log.info("Environment created: action_dim=%d, chunk_length=%d", env.action_dim, env.chunk_length)

    episodes = []

    for ep in range(config.num_episodes):
        obs = env.reset()
        episode_reward = 0.0
        episode_chunks = 0

        while True:
            with torch.no_grad():
                vla_input = vla.preprocess_obs(obs)
                action_chunk = vla.get_rl_chunk_reference(vla_input, config.chunk_length)
                action_chunk = action_chunk.squeeze(0).cpu().numpy()  # [C, action_dim]

            next_obs, chunk_rewards, done, info = env.step(action_chunk)
            episode_reward += float(chunk_rewards.sum())
            episode_chunks += 1

            if done:
                success = info.get("success", False)
                episodes.append({
                    "episode": ep,
                    "reward": episode_reward,
                    "success": success,
                    "num_chunks": episode_chunks,
                    "num_steps": info.get("steps_executed", episode_chunks * config.chunk_length),
                })
                log.info(
                    "Episode %d/%d: chunks=%d, reward=%.3f, success=%s",
                    ep + 1, config.num_episodes, episode_chunks, episode_reward, success,
                )
                break

            obs = next_obs

    num_success = sum(e["success"] for e in episodes)
    success_rate = num_success / len(episodes) if episodes else 0.0
    mean_reward = sum(e["reward"] for e in episodes) / len(episodes) if episodes else 0.0

    log.info(
        "Done. Success rate: %.1f%% (%d/%d), mean reward: %.3f",
        100 * success_rate, num_success, len(episodes), mean_reward,
    )

    results = {
        "checkpoint": config.vla_checkpoint_dir,
        "vla_config_name": config.vla_config_name,
        "eval_timestamp": datetime.now().isoformat(),
        "num_episodes": len(episodes),
        "success_rate": success_rate,
        "mean_reward": mean_reward,
        "episodes": episodes,
    }
    save_dir = Path(config.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    results_path = save_dir / f"eval_vla_{len(episodes)}ep_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    results_path.write_text(json.dumps(results, indent=2))
    log.info("Results saved to %s", results_path)


if __name__ == "__main__":
    main(tyro.cli(RolloutConfig))
