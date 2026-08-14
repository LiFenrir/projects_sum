"""Unified logger wrapping wandb and stdout."""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class LoggerConfig:
    """Logger configuration.

    Args:
        project: wandb project name.
        enabled: Whether wandb logging is active.
    """

    project: str = "rlt-openpi"
    enabled: bool = True


class Logger:
    """Metric logger wrapping wandb + stdout.

    Args:
        config: Logger configuration.
        run_config: Training config dict to log as wandb run config.
    """

    def __init__(
        self,
        config: LoggerConfig,
        run_config: dict[str, Any] | None = None,
        run_name: str | None = None,
    ) -> None:
        self.config = config
        self._wandb_run = None

        if config.enabled:
            try:
                import wandb

                self._wandb_run = wandb.init(
                    project=config.project,
                    name=run_name,
                    config=run_config or {},
                )
                logger.info("wandb run initialized: %s", self._wandb_run.url)
            except Exception:
                logger.warning("wandb init failed, falling back to stdout only", exc_info=True)

    def log(self, metrics: dict[str, Any], step: int | None = None) -> None:
        """Log a dict of metrics.

        Args:
            metrics: Key-value pairs to log.
            step: Optional global step for wandb x-axis.
        """
        if self._wandb_run is not None:
            self._wandb_run.log(metrics, step=step)

    def finish(self) -> None:
        """Finalize the wandb run."""
        if self._wandb_run is not None:
            self._wandb_run.finish()

    @staticmethod
    def from_train_config(train_config: Any) -> Logger:
        """从训练配置 dataclass 创建 Logger。

        支持新旧两种格式：
        - 新版: ``config.wandb.project`` / ``config.wandb.enabled``
        - 旧版: ``config.wandb_project`` / ``config.wandb_enabled``（getattr 兜底）
        """
        cfg_dict = asdict(train_config) if hasattr(train_config, "__dataclass_fields__") else {}

        # 新版嵌套格式优先
        if hasattr(train_config, "wandb") and hasattr(train_config.wandb, "project"):
            project = train_config.wandb.project
            enabled = train_config.wandb.enabled
        else:
            # 旧版平铺格式（向后兼容）
            project = getattr(train_config, "wandb_project", "rlt-openpi")
            enabled = getattr(train_config, "wandb_enabled", True)

        logger_config = LoggerConfig(project=project, enabled=enabled)

        # run_name: 新版在 checkpoint.run_name，旧版在 run_name
        run_name = None
        if hasattr(train_config, "checkpoint") and hasattr(train_config.checkpoint, "run_name"):
            run_name = train_config.checkpoint.run_name
        if not run_name:
            run_name = getattr(train_config, "run_name", None) or None

        return Logger(config=logger_config, run_config=cfg_dict, run_name=run_name)
