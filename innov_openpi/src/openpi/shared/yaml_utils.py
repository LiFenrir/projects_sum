"""YAML configuration utilities for openpi training.

Provides helpers to load dataclass-based configs from YAML files,
optionally starting from a named preset and applying selective overrides.

Usage::

    from openpi.shared.yaml_utils import load_yaml_overrides

    # Load a YAML file and merge with a preset config
    config = load_yaml_overrides("examples/bi_s1/sft.yaml", config_cls=TrainConfig, registry=_CONFIGS_DICT)
"""

from __future__ import annotations

import dataclasses
import json
import logging
import os
import re
from pathlib import Path
from typing import Any, TypeVar

import yaml

T = TypeVar("T")

logger = logging.getLogger(__name__)


def _envvar_interpolate(value: str) -> str:
    """Replace ``${VAR:default}`` or ``${VAR}`` with environment variable values."""

    def _replace(m: re.Match) -> str:
        expr = m.group(1)
        if ":" in expr:
            var, default = expr.split(":", 1)
            return os.environ.get(var, default)
        return os.environ.get(expr, "")

    return re.sub(r"\$\{([^}]+)\}", _replace, value)


def _interpolate(obj: Any) -> Any:
    """Recursively interpolate environment variables in strings."""
    if isinstance(obj, str):
        return _envvar_interpolate(obj)
    if isinstance(obj, dict):
        return {k: _interpolate(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_interpolate(v) for v in obj]
    return obj


def _deep_merge(base: dict, override: dict) -> dict:
    """Deep-merge override into base dict. Lists are replaced, not concatenated."""
    result = dict(base)
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def _dict_to_dataclass(data: dict, cls: type, *, class_registry: dict[str, type] | None = None) -> Any:
    """Recursively convert a dict to a dataclass instance.

    Handles nested dataclasses, enums, and simple types. Unknown fields
    (not in the dataclass) are silently dropped.

    When a dict contains a ``_target_`` key, it specifies the concrete class
    name to instantiate (looked up from ``class_registry``). This allows YAML
    files to specify concrete subclasses for abstract-typed fields.

    Args:
        data: The dict to convert.
        cls: The target dataclass type.
        class_registry: Optional mapping from class name to type, used to
            resolve ``_target_`` fields.
    """
    # For non-dataclass types (Protocol, ABC), try to resolve _target_ via class_registry
    if not dataclasses.is_dataclass(cls):
        if "_target_" in data and class_registry is not None:
            target_name = data.pop("_target_")
            if target_name in class_registry:
                target_cls = class_registry[target_name]
                if dataclasses.is_dataclass(target_cls):
                    return _dict_to_dataclass(data, target_cls, class_registry=class_registry)
                else:
                    return target_cls(**data)
            logger.warning(
                "_target_='%s' not found in class registry. Available: %s",
                target_name,
                ", ".join(sorted(class_registry.keys())),
            )
        return data

    field_types = {f.name: f.type for f in dataclasses.fields(cls)}
    kwargs: dict[str, Any] = {}

    # Resolve _target_ if present — it specifies the concrete class
    target_cls = cls
    if "_target_" in data and class_registry is not None:
        target_name = data.pop("_target_")
        if target_name in class_registry:
            target_cls = class_registry[target_name]
        else:
            logger.warning(
                "_target_='%s' not found in class registry. Available: %s",
                target_name,
                ", ".join(sorted(class_registry.keys())),
            )

    # Re-resolve field_types from the resolved target class (not the base cls)
    field_types = {f.name: f.type for f in dataclasses.fields(target_cls)}

    for key, value in data.items():
        if key not in field_types:
            continue
        target_type = field_types[key]
        # Handle Optional / Union with None
        origin = getattr(target_type, "__origin__", None)
        if origin is not None:
            args = getattr(target_type, "__args__", ())
            # Union with NoneType -> unwrap the non-None type
            non_none = [a for a in args if a is not type(None)]  # noqa: E721
            if len(non_none) == 1:
                target_type = non_none[0]

        if dataclasses.is_dataclass(target_type) and isinstance(value, dict):
            kwargs[key] = _dict_to_dataclass(value, target_type, class_registry=class_registry)
        elif isinstance(target_type, type) and issubclass(target_type, dict) and isinstance(value, dict):
            # Dict fields like camera_map
            kwargs[key] = value
        elif isinstance(target_type, type) and isinstance(value, dict):
            # Non-dataclass types (Protocols, ABCs) that may contain _target_ resolution
            kwargs[key] = _dict_to_dataclass(value, target_type, class_registry=class_registry)
        else:
            kwargs[key] = value

    return target_cls(**kwargs)


def load_yaml_overrides(
    yaml_path: str | Path,
    *,
    config_cls: type[T],
    registry: dict[str, T] | None = None,
    class_registry: dict[str, type] | None = None,
) -> tuple[T | None, dict[str, Any]]:
    """Load a YAML config file and return (preset_config, override_dict).

    The YAML file can optionally specify a ``preset`` top-level key that
    references a named config in the registry. All other keys are treated
    as field overrides on top of the preset.

    Environment variable interpolation is supported via ``${VAR}`` and
    ``${VAR:default}`` syntax.

    Args:
        yaml_path: Path to the YAML file.
        config_cls: The dataclass type for validation / nested conversion.
        registry: Optional dict of preset name -> config instance.

    Returns:
        Tuple of (preset_config_or_None, override_dict).
        Caller is responsible for merging via ``dataclasses.replace``.
    """
    with open(yaml_path) as f:
        raw = yaml.safe_load(f)

    if not isinstance(raw, dict):
        raise ValueError(f"YAML config must be a dict, got {type(raw).__name__}")

    raw = _interpolate(raw)

    preset_name = raw.pop("preset", None)
    preset_config = None
    if preset_name:
        if registry is None:
            raise ValueError(
                f"YAML specifies preset='{preset_name}' but no registry was provided."
            )
        if preset_name not in registry:
            available = ", ".join(sorted(registry.keys()))
            raise KeyError(
                f"Preset '{preset_name}' not found in registry. Available: {available}"
            )
        preset_config = registry[preset_name]
        logger.info("Loaded preset config: %s", preset_name)

    # Convert nested dicts to dataclass-compatible form for merging
    converted = _dict_to_dataclass(raw, config_cls, class_registry=class_registry) if dataclasses.is_dataclass(config_cls) else raw

    return preset_config, converted if isinstance(converted, dict) else {
        f.name: getattr(converted, f.name)
        for f in dataclasses.fields(converted)
        if f.name in raw
    }


def merge_config(
    preset: T,
    overrides: dict[str, Any],
) -> T:
    """Merge override dict into a preset dataclass instance.

    Handles nested dataclass fields via deep-merge. Override values that
    are already dataclass instances are converted to dict for merging and
    then reconverted.

    Args:
        preset: The base config to start from.
        overrides: Dict of field_name -> value to override.

    Returns:
        A new instance with overrides applied.
    """
    # Convert preset to dict
    preset_dict = dataclasses.asdict(preset)

    # Recursively convert any nested dataclass values in overrides to dicts
    def _to_dicts(obj: Any) -> Any:
        if dataclasses.is_dataclass(obj):
            return dataclasses.asdict(obj)
        if isinstance(obj, dict):
            return {k: _to_dicts(v) for k, v in obj.items()}
        return obj

    overrides_dict = {k: _to_dicts(v) for k, v in overrides.items()}

    # Deep merge
    merged = _deep_merge(preset_dict, overrides_dict)

    # Convert back to dataclass
    return _dict_to_dataclass(merged, type(preset))


def load_config(
    yaml_path: str | Path,
    *,
    config_cls: type[T],
    registry: dict[str, T] | None = None,
    class_registry: dict[str, type] | None = None,
) -> T:
    """Load a complete config from YAML, merging with an optional preset.

    This is the main entry point. It combines :func:`load_yaml_overrides`
    and :func:`merge_config` into a single call.

    Example YAML::

        preset: bi_s1_pi05_sft
        batch_size: 64
        num_train_steps: 10000
        data:
          repo_id: my_org/my_dataset

    Example YAML without preset (pure YAML config)::

        name: bi_s1_pi05_sft
        model:
          _target_: Pi0Config
          pi05: true
          action_dim: 32
        data:
          _target_: LeRobotDataConfig
          repo_id: my_org/my_dataset

    Args:
        yaml_path: Path to the YAML file.
        config_cls: Target dataclass type.
        registry: Optional dict of preset name -> config instance.
        class_registry: Optional dict of class name -> type for resolving
            ``_target_`` fields in the YAML.

    Returns:
        Fully merged config instance.
    """
    preset, overrides = load_yaml_overrides(yaml_path, config_cls=config_cls, registry=registry, class_registry=class_registry)
    if preset is None:
        # No preset — convert the YAML dict directly to the dataclass
        with open(yaml_path) as f:
            raw = _interpolate(yaml.safe_load(f))
        return _dict_to_dataclass(raw, config_cls, class_registry=class_registry)
    return merge_config(preset, overrides)
