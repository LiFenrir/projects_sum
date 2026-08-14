"""cbreak 键盘监听,事件队列分发(s/f/t/i 键)。

单一实例供 env(RobotEnv 检测 s/f)、rollout_worker(检测 t 切换 RL 模式)、
intervention(检测 i 进入示教)共享,避免多个 cbreak 实例冲突。

按键语义:
- ``s``: 成功(success,reward +1 且终止 episode)
- ``f``: 失败(failure,终止 episode)
- ``t``: 切换 RL 模式(阶段 RL ↔ 全局 RL 的执行/存储开关)
- ``i``: 切换本体示教(由 InterventionManager 消费)
- ``\\x1b``/``\\x03``: 停止(返回 ``"stop"``)
"""

from __future__ import annotations

import logging
import select
import sys
import termios
import threading
import tty

logger = logging.getLogger(__name__)

_SIGNAL_KEYS = {"s": "s", "f": "f", "t": "t", "i": "i", "\x1b": "stop", "\x03": "stop"}
_REWARD_KEYS = frozenset({"s", "f"})


class KeyboardCtrl:
    """非阻塞键盘事件队列(线程安全,单实例共享)。"""

    def __init__(self) -> None:
        self._queue: list[str] = []
        self._lock = threading.Lock()
        self._old_settings: list | None = None
        self._raw = False

    def start(self) -> None:
        """进入 cbreak 模式,按键即时生效。"""
        try:
            self._old_settings = termios.tcgetattr(sys.stdin)
            tty.setcbreak(sys.stdin.fileno())
            self._raw = True
        except (termios.error, OSError):
            self._raw = False
            logger.warning("Raw terminal mode unavailable, falling back to line input (type + Enter)")

    def stop(self) -> None:
        """恢复终端设置并清空事件队列。"""
        with self._lock:
            self._queue.clear()
        if self._raw and self._old_settings is not None:
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self._old_settings)
            self._raw = False
            self._old_settings = None

    def _pump(self) -> None:
        """把 stdin 上可读的按键压入事件队列。"""
        if not self._raw:
            return
        with self._lock:
            while select.select([sys.stdin], [], [], 0)[0]:
                ch = sys.stdin.read(1).lower()
                if ch in _SIGNAL_KEYS:
                    self._queue.append(_SIGNAL_KEYS[ch])

    def poll(self) -> str | None:
        """取下一个待处理事件(瞬时,消费后不重复)。"""
        self._pump()
        with self._lock:
            return self._queue.pop(0) if self._queue else None

    def check(self) -> str | None:
        """兼容 HumanReward:只消费 s/f 奖励信号,其他事件留给 poll。"""
        self._pump()
        with self._lock:
            for i, sig in enumerate(self._queue):
                if sig in _REWARD_KEYS:
                    return self._queue.pop(i)
            return None
