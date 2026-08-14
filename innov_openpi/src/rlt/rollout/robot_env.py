"""Real-robot environment for online RL training.

Provides ``RobotEnv``, a chunk-level environment that connects to any
robot through three user-supplied callables (``step_fn``, ``reset_fn``,
``get_obs_fn``).  No dependency on any specific robot stack (DROID,
polymetis, ROS, etc.) — the wiring happens in the user's launch script.

Human reward (success/failure) is collected via
:class:`~rlt.rollout.reward.HumanReward` using instant
keypress detection (no Enter needed) during episodes.

Usage (with DROID)::

    from droid.robot_env import RobotEnv as DroidEnv

    droid = DroidEnv(action_space="cartesian_velocity", control_hz=15)

    def get_obs():
        obs = droid.get_observation()
        return {
            "observation/joint_position": np.array(
                obs["robot_state"]["joint_positions"], dtype=np.float32,
            ),
            "observation/gripper_position": np.array(
                [obs["robot_state"]["gripper_position"]], dtype=np.float32,
            ),
            "observation/exterior_image_1_left": obs["image"]["39790647_left"],
            "observation/wrist_image_left": obs["image"]["15850436_left"],
            "observation/exterior_image_2_left": obs["image"]["35840217_left"],
            "prompt": "stack the three blocks on the tray",
        }

    env = RobotEnv(
        step_fn=droid.step,
        reset_fn=droid.reset,
        get_obs_fn=get_obs,
        action_dim=7,
        chunk_length=10,
        control_hz=15,
    )
"""

from __future__ import annotations

import logging
import time
from typing import Any, Callable

import numpy as np
from numpy.typing import NDArray

from rlt.rollout.keyboard_ctrl import KeyboardCtrl
from rlt.utils import display

logger = logging.getLogger(__name__)


class RobotEnv:
    """Chunk-level environment for real robot online RL.

    Robot-agnostic: connects to any robot through three callables.
    Provides the same interface as ``SimEnv`` (``reset``, ``step``,
    ``action_dim``, ``chunk_length``) so it works with ``RolloutWorker``
    and ``OnlineRLTrainer``.

    Args:
        step_fn: Callable that sends a single action ``[action_dim]`` to
            the robot.  Signature: ``step_fn(action: np.ndarray) -> Any``.
        reset_fn: Callable that resets the robot to a home pose.
            Signature: ``reset_fn() -> Any``.
        get_obs_fn: Callable that returns an observation dict with at
            least a ``"state"`` key (proprioceptive, ``np.float32``).
            Camera images and ``"prompt"`` should also be included for
            VLA embedding extraction.
            Signature: ``get_obs_fn() -> dict[str, Any]``.
        action_dim: Dimension of a single-step action (e.g. 7 for
            cartesian velocity + gripper).
        chunk_length: C, number of single-step actions per chunk.
        control_hz: Robot control frequency in Hz.
        max_episode_chunks: Maximum chunks per episode before forced
            termination.
        obs_stride: 每 N 步采集一次中间观测(供 stride 子采样,论文 stride=2)。
        ctrl: 共享键盘监听(worker/intervention 同实例);None 时内部创建。
    """

    def __init__(
        self,
        step_fn: Callable[[NDArray], Any],
        reset_fn: Callable[[], Any],
        get_obs_fn: Callable[[], dict[str, Any]],
        action_dim: int = 7,
        chunk_length: int = 10,
        control_hz: int = 15,
        max_episode_chunks: int = 50,
        obs_stride: int = 2,
        ctrl: KeyboardCtrl | None = None,
    ) -> None:
        self._step_fn = step_fn
        self._reset_fn = reset_fn
        self._get_obs_fn = get_obs_fn
        self._action_dim = action_dim
        self._chunk_length = chunk_length
        self._control_period = 1.0 / control_hz
        self._max_episode_chunks = max_episode_chunks
        self._obs_stride = max(1, obs_stride)

        self._chunk_count = 0
        self.ctrl = ctrl or KeyboardCtrl()
        self._display_episode_num: int = 0

    @property
    def action_dim(self) -> int:
        return self._action_dim

    @property
    def chunk_length(self) -> int:
        return self._chunk_length

    def reset(self, **kwargs: Any) -> dict[str, Any]:
        """Reset robot to home position and wait for scene setup.

        Returns:
            Observation dict with ``"state"``, camera images, and ``"prompt"``.
        """
        self.ctrl.stop()
        self._reset_fn()
        self._chunk_count = 0

        display.episode_reset(self._display_episode_num)
        input("")

        self.ctrl.start()
        display.episode_running(self._display_episode_num)

        return self._get_obs_fn()

    def step(
        self, action_chunk: NDArray
    ) -> tuple[dict[str, Any], NDArray, bool, dict[str, Any]]:
        """Execute C single-step actions on the robot.

        Args:
            action_chunk: Actions to execute, shape ``[C, action_dim]``.

        Returns:
            next_obs: Observation dict after the last step.
            rewards: Per-step rewards ``[C]``, only non-zero at terminal
                step (success=+1, failure=0).
            done: Whether the episode ended.
            info: Contains ``"success"`` key on termination, plus stride
                subsampling history (``obs_history``/``actions_history``/
                ``rewards_history``/``done_history``) for stride-2 chunk
                reconstruction.
        """
        C = self._chunk_length
        stride = self._obs_stride
        rewards = np.zeros(C, dtype=np.float32)
        done = False
        info: dict[str, Any] = {}

        obs_history: list[dict[str, Any]] = []
        actions_history: list[NDArray] = []
        rewards_history: list[float] = []
        done_history: list[float] = []

        for k in range(C):
            t_start = time.time()

            self._step_fn(action_chunk[k])
            actions_history.append(np.asarray(action_chunk[k], dtype=np.float32))
            rewards_history.append(0.0)
            done_history.append(0.0)

            # Check for human signal between steps
            signal = self.ctrl.check()
            if signal is not None:
                if signal == "s":
                    rewards[k] = 1.0
                    rewards_history[-1] = 1.0
                    done = True
                    done_history[-1] = 1.0
                    info["success"] = True
                    logger.info("Human signal: SUCCESS")
                elif signal == "f":
                    done = True
                    done_history[-1] = 1.0
                    info["success"] = False
                    logger.info("Human signal: FAILURE")
                if done:
                    break

            # Stride subsampling: capture intermediate observations every
            # `stride` steps. The final step (k+1 == C) is excluded — that
            # observation equals the next chunk's start, which the caller
            # already holds.
            if (k + 1) % stride == 0 and k + 1 < C:
                obs_history.append(self._get_obs_fn())

            # Enforce control frequency
            elapsed = time.time() - t_start
            sleep_time = self._control_period - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

        info["steps_executed"] = k + 1
        info["obs_history"] = obs_history
        info["actions_history"] = np.stack(actions_history) if actions_history else None
        info["rewards_history"] = np.asarray(rewards_history, dtype=np.float32)
        info["done_history"] = np.asarray(done_history, dtype=np.float32)
        self._chunk_count += 1

        # Timeout: force episode end after max chunks
        if not done and self._chunk_count >= self._max_episode_chunks:
            done = True
            info["success"] = False
            info["timeout"] = True
            logger.info("Episode timed out after %d chunks", self._chunk_count)

        # Restore terminal when episode ends
        if done:
            self.ctrl.stop()

        obs = self._get_obs_fn()
        return obs, rewards, done, info
