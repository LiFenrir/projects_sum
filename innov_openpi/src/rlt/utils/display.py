"""Rich terminal display for real-robot online RL training.

Provides formatted, at-a-glance status output so the human operator
standing at the robot always knows: what phase we're in, how training
is progressing, and what action is expected from them.
"""

from __future__ import annotations

import shutil
import sys
import time
from collections import deque
from dataclasses import dataclass, field


BOLD = "\033[1m"
DIM = "\033[2m"
RESET = "\033[0m"
GREEN = "\033[32m"
RED = "\033[31m"
YELLOW = "\033[33m"
CYAN = "\033[36m"
MAGENTA = "\033[35m"
WHITE = "\033[37m"
BG_GREEN = "\033[42m"
BG_RED = "\033[41m"
BG_YELLOW = "\033[43m"
BG_CYAN = "\033[46m"
BG_MAGENTA = "\033[45m"


def _term_width() -> int:
    return shutil.get_terminal_size((80, 24)).columns


def _bar(label: str, bg: str = BG_CYAN) -> str:
    w = _term_width()
    pad = w - len(label) - 2
    return f"\n{bg}{BOLD} {label}{' ' * max(pad, 0)} {RESET}"


def _kv(key: str, value: str, color: str = WHITE) -> str:
    return f"  {DIM}{key:<22}{RESET} {color}{value}{RESET}"


# ------------------------------------------------------------------
# Warmup progress
# ------------------------------------------------------------------


def warmup_start(total_chunks: int) -> None:
    print(_bar("WARMUP", BG_MAGENTA))
    print(_kv("Phase", "Collecting VLA-only rollouts"))
    print(_kv("Target chunks", str(total_chunks)))
    print(_kv("Controls",
              f"{BOLD}[S]{RESET}/{BOLD}[Space]{RESET}=success   "
              f"{BOLD}[F]{RESET}=failure"))
    print()


def warmup_progress(current: int, total: int) -> None:
    pct = current / total if total > 0 else 0
    w = min(_term_width() - 20, 40)
    filled = int(w * pct)
    bar = f"{'█' * filled}{'░' * (w - filled)}"
    sys.stdout.write(f"\r  {CYAN}Warmup {bar} {current}/{total} ({pct:.0%}){RESET}")
    sys.stdout.flush()


def warmup_done(stored: int, buffer_size: int) -> None:
    print()
    print(_kv("Stored", f"{stored} transitions"))
    print(_kv("Buffer size", str(buffer_size)))
    print(f"{GREEN}{BOLD}  ✓ Warmup complete{RESET}\n")


# ------------------------------------------------------------------
# Episode lifecycle
# ------------------------------------------------------------------


def episode_reset(episode_num: int) -> None:
    """Printed by RobotEnv.reset() before blocking for Enter."""
    print(_bar(f"EPISODE {episode_num}", BG_CYAN))
    print(_kv("Action required", f"{YELLOW}Set up the scene, press Enter{RESET}"))


def episode_running(episode_num: int) -> None:
    """Printed by RobotEnv.reset() after Enter is pressed."""
    print()
    print(f"  {GREEN}{BOLD}▶ RUNNING{RESET}  —  "
          f"{BOLD}[S]{RESET}/{BOLD}[Space]{RESET}=success   "
          f"{BOLD}[F]{RESET}=failure")
    print()


def episode_result(
    episode_num: int,
    total_reward: float,
    success: bool,
    num_chunks: int,
    num_steps: int,
    interventions: int,
) -> None:
    if success:
        tag = f"{BG_GREEN}{BOLD} SUCCESS {RESET}"
    else:
        tag = f"{BG_RED}{BOLD} FAILURE {RESET}"

    print()
    print(f"  Result: {tag}  reward={total_reward:.3f}  "
          f"chunks={num_chunks}  steps={num_steps}  interventions={interventions}")


# ------------------------------------------------------------------
# Training summary (printed after each episode's gradient updates)
# ------------------------------------------------------------------


@dataclass
class TrainingDisplay:
    """Accumulates and displays running training statistics.

    Tracks a sliding window of recent episode outcomes for the running
    success rate and renders a compact summary block after each episode.
    """

    window_size: int = 20
    _recent_success: deque = field(default_factory=lambda: deque(maxlen=20))
    _recent_rewards: deque = field(default_factory=lambda: deque(maxlen=20))
    _total_success: int = 0
    _total_episodes: int = 0
    _start_time: float = field(default_factory=time.time)

    def __post_init__(self) -> None:
        self._recent_success = deque(maxlen=self.window_size)
        self._recent_rewards = deque(maxlen=self.window_size)

    def record_episode(self, success: bool, reward: float) -> None:
        self._recent_success.append(int(success))
        self._recent_rewards.append(reward)
        self._total_success += int(success)
        self._total_episodes += 1

    def print_summary(
        self,
        total_episodes: int,
        total_env_steps: int,
        max_env_steps: int,
        buffer_size: int,
        critic_loss: float,
        actor_loss: float | None,
        q_mean: float,
    ) -> None:
        elapsed = time.time() - self._start_time
        elapsed_str = _fmt_duration(elapsed)
        pct = total_env_steps / max_env_steps if max_env_steps > 0 else 0

        recent_n = len(self._recent_success)
        recent_rate = sum(self._recent_success) / recent_n if recent_n > 0 else 0
        total_rate = self._total_success / self._total_episodes if self._total_episodes > 0 else 0
        recent_reward = sum(self._recent_rewards) / recent_n if recent_n > 0 else 0

        rate_color = GREEN if recent_rate >= 0.5 else YELLOW if recent_rate >= 0.2 else RED

        # Compact progress bar
        w = min(_term_width() - 30, 30)
        filled = int(w * pct)
        prog_bar = f"{'█' * filled}{'░' * (w - filled)}"

        print(f"\n{DIM}{'─' * _term_width()}{RESET}")
        print(_kv("Progress", f"{prog_bar} {total_env_steps}/{max_env_steps} steps ({pct:.0%})"))
        print(_kv("Episodes", f"{total_episodes}  ({elapsed_str} elapsed)"))
        print(_kv("Buffer", str(buffer_size)))
        print(_kv(f"Success (last {recent_n})", f"{rate_color}{BOLD}{recent_rate:.0%}{RESET}"))
        print(_kv("Success (all)", f"{total_rate:.0%} ({self._total_success}/{self._total_episodes})"))
        print(_kv(f"Avg reward (last {recent_n})", f"{recent_reward:.3f}"))
        print(_kv("Critic loss", f"{critic_loss:.6f}"))
        actor_str = f"{actor_loss:.6f}" if actor_loss is not None else "—"
        print(_kv("Actor loss", actor_str))
        print(_kv("Q mean", f"{q_mean:.4f}"))
        print(f"{DIM}{'─' * _term_width()}{RESET}\n")


def training_start(config_summary: dict[str, str]) -> None:
    print(_bar("ONLINE RL TRAINING", BG_CYAN))
    for k, v in config_summary.items():
        print(_kv(k, str(v)))
    print()


def training_done(total_episodes: int, total_steps: int, total_updates: int, elapsed: float) -> None:
    print(_bar("TRAINING COMPLETE", BG_GREEN))
    print(_kv("Episodes", str(total_episodes)))
    print(_kv("Env steps", str(total_steps)))
    print(_kv("Gradient updates", str(total_updates)))
    print(_kv("Wall time", _fmt_duration(elapsed)))
    print()


def checkpoint_saved(path: str) -> None:
    print(f"  {MAGENTA}💾 Checkpoint saved → {path}{RESET}")


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------


def _fmt_duration(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    if h > 0:
        return f"{h}h {m}m {s}s"
    if m > 0:
        return f"{m}m {s}s"
    return f"{s}s"
