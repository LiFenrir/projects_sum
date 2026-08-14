"""切换点数据记录器:供训练切换分类器(论文:预测何时交给 RL 策略)。

每次 rl_active 切换时记录该时刻的 VLA 抽象特征(z_rl)、参考动作(ã)与切换方向标签:
- label=1: 切到 RL(阶段 RL 接管关键阶段)
- label=0: 切回 VLA

每 episode 落盘一个 npz 分片(``ep_%06d.npz``):
    features [N, D]    z_rl 特征
    a_tilde  [N, C*d]  切换时刻的 VLA 参考动作(论文切换 head 输入 z + ã)
    labels   [N]       切换方向
    obs      [N]       (可选)原始观测 dict,默认关闭(图像大)
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


class SwitchPointRecorder:
    """切换点数据采集与分片落盘。"""

    def __init__(self, save_dir: str | None = None, *, save_raw_obs: bool = False) -> None:
        self._save_dir = Path(save_dir) if save_dir else None
        self.save_raw_obs = save_raw_obs
        self._pending: list[dict] = []
        self._episode = 0

    def record(self, feature: np.ndarray, label: int, obs: dict | None = None, a_tilde: np.ndarray | None = None) -> None:
        """记录一个切换点。

        Args:
            feature: z_rl 抽象特征 [D]。
            label: 切换后状态(1 = 切到 RL,0 = 切回 VLA)。
            obs: 原始观测 dict(仅当 save_raw_obs 时保存)。
            a_tilde: VLA 参考动作 [C*d](与 actor 的 ã 输入对齐)。
        """
        self._pending.append({
            "feature": np.asarray(feature, dtype=np.float32),
            "label": int(label),
            "obs": obs,
            "a_tilde": np.asarray(a_tilde, dtype=np.float32) if a_tilde is not None else None,
        })

    def end_episode(self) -> None:
        """episode 结束:把本 episode 切换点落盘为 npz 分片。"""
        if not self._pending:
            return
        if self._save_dir is None:
            self._pending.clear()
            return
        self._save_dir.mkdir(parents=True, exist_ok=True)
        features = np.stack([p["feature"] for p in self._pending])
        labels = np.array([p["label"] for p in self._pending], dtype=np.int64)
        payload: dict = {"features": features, "labels": labels}
        # 兼容旧调用(未提供 ã):全为 None 时不写该键
        if any(p["a_tilde"] is not None for p in self._pending):
            payload["a_tilde"] = np.stack([p["a_tilde"] for p in self._pending])
        if self.save_raw_obs:
            payload["obs"] = np.asarray([p["obs"] for p in self._pending], dtype=object)
        path = self._save_dir / f"ep_{self._episode:06d}.npz"
        np.savez(path, **payload)
        logger.info("Saved %d switch points to %s", len(self._pending), path)
        self._pending = []
        self._episode += 1
