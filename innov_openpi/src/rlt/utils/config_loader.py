"""Lightweight YAML config loading for RLT scripts.

Converts a YAML file to a dataclass instance without requiring the
``_target_`` registry used by the openpi config system.  Designed for
the flat dataclass configs used by ``train_rl_token.py`` and
``train_online_rl.py``.

Usage::

    from rlt.utils.config_loader import load_config_with_cli

    config = load_config_with_cli(MyConfig, yaml_path="configs/rlt/stage1.yaml")
"""

from __future__ import annotations

import dataclasses
import logging
import sys
import typing
from typing import Any, TypeVar

import yaml

T = TypeVar("T")

logger = logging.getLogger(__name__)


def _dict_to_dataclass(data: dict[str, Any], cls: type[T]) -> T:
    """Recursively convert a dict to a dataclass instance.

    Handles nested dataclasses and simple types. Unknown keys are
    silently ignored.
    """
    if not dataclasses.is_dataclass(cls):
        return data

    # Use get_type_hints() to resolve string annotations caused by
    # ``from __future__ import annotations``.
    try:
        field_types = typing.get_type_hints(cls)
    except Exception:
        field_types = {f.name: f.type for f in dataclasses.fields(cls)}

    kwargs: dict[str, Any] = {}

    for key, value in data.items():
        if key not in field_types:
            logger.warning("Unknown field '%s' in config, ignoring", key)
            continue
        target_type = field_types[key]

        # Handle Optional / Union with None
        origin = getattr(target_type, "__origin__", None)
        if origin is not None:
            args = getattr(target_type, "__args__", ())
            non_none = [a for a in args if a is not type(None)]  # noqa: E721
            if len(non_none) == 1:
                target_type = non_none[0]

        if dataclasses.is_dataclass(target_type) and isinstance(value, dict):
            kwargs[key] = _dict_to_dataclass(value, target_type)
        else:
            kwargs[key] = value

    return cls(**kwargs)


def load_config_from_yaml(yaml_path: str, config_cls: type[T]) -> T:
    """Load a dataclass config from a YAML file.

    Supports both old flat-format YAML (auto-migrated) and new nested format.

    Args:
        yaml_path: Path to the YAML file.
        config_cls: Target dataclass type.

    Returns:
        Populated config instance.
    """
    with open(yaml_path) as f:
        raw = yaml.safe_load(f)

    if not isinstance(raw, dict):
        raise ValueError(f"YAML config must be a dict, got {type(raw).__name__}")

    # Auto-detect old format and migrate
    from rlt.training.config import migrate_online_rl_config, migrate_rl_token_config
    from rlt.training.config import OnlineRLTrainConfig as _ORC, RLTokenTrainConfig as _RTC

    if config_cls is _ORC:
        migrated = migrate_online_rl_config(raw)
        if isinstance(migrated, _ORC):
            return migrated
        raw = migrated  # dict (new nested format)
    elif config_cls is _RTC:
        migrated = migrate_rl_token_config(raw)
        if isinstance(migrated, _RTC):
            return migrated
        raw = migrated  # dict (new nested format)

    return _dict_to_dataclass(raw, config_cls)


def load_config_with_cli(
    config_cls: type[T],
    *,
    yaml_path: str | None = None,
    cli_args: list[str] | None = None,
) -> T:
    """Load config from optional YAML, with CLI overrides.

    Resolution order (last wins):
    1. Dataclass defaults
    2. YAML file values (if ``--config`` is provided)
    3. CLI arguments

    Args:
        config_cls: Target dataclass type.
        yaml_path: Optional path to a YAML config file.
        cli_args: CLI arguments (defaults to ``sys.argv[1:]``).

    Returns:
        Merged config instance.
    """
    import tyro

    if cli_args is None:
        cli_args = sys.argv[1:]

    # Start from YAML if provided, otherwise use defaults
    if yaml_path is not None:
        logger.info("Loading config from %s", yaml_path)
        base = load_config_from_yaml(yaml_path, config_cls)
    else:
        base = config_cls()

    # Apply CLI overrides via tyro
    if cli_args:
        sys.argv = [sys.argv[0], *cli_args]
        return tyro.cli(config_cls, default=base)

    return base
