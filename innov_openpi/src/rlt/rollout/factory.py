"""Dynamic factories for pluggable environments and intervention managers.

Users provide a Python import path to a factory function.  The same
pattern is used for both the environment and the intervention manager.

Example usage::

    env = make_env(
        "rlt_openpi.envs.franka.env_factory.make_franka_env",
        action_dim=7,
        chunk_length=10,
        task_prompt="pick up the cup",
    )

    intervention = make_intervention(
        "rlt_openpi.envs.franka.intervention.make_vr_intervention",
        env=env,
    )
"""

from __future__ import annotations

import importlib
import logging
import sys
from pathlib import Path
from typing import Any

from rlt.rollout.intervention import InterventionManager

logger = logging.getLogger(__name__)


def _ensure_project_root_on_path() -> None:
    project_root = str(Path(__file__).resolve().parents[4])
    if project_root not in sys.path:
        sys.path.insert(0, project_root)


def _import_factory(import_path: str) -> Any:
    """Dynamically import a factory function from a dotted path."""
    _ensure_project_root_on_path()
    module_path, func_name = import_path.rsplit(".", 1)
    module = importlib.import_module(module_path)
    return getattr(module, func_name)


def make_env(env_factory: str, **kwargs: Any) -> Any:
    """Import and call a user-provided env factory function.

    Args:
        env_factory: Dotted Python import path to a callable, e.g.
            ``"my_package.envs.make_franka_env"``.
        **kwargs: Forwarded to the factory function (typically
            ``action_dim``, ``chunk_length``, ``task_prompt``).

    Returns:
        An env object with ``reset()``, ``step(action_chunk)``,
        ``action_dim``, and ``chunk_length``.
    """
    factory_fn = _import_factory(env_factory)
    logger.info("Creating env via %s", env_factory)
    return factory_fn(**kwargs)


def make_intervention(
    intervention_factory: str, *, env: Any, **kwargs: Any
) -> InterventionManager:
    """Import and call a user-provided intervention factory function.

    Args:
        intervention_factory: Dotted Python import path to a callable,
            e.g. ``"rlt_openpi.envs.franka.intervention.make_vr_intervention"``.
        env: The environment object (passed as the first argument to the
            factory so it can access robot handles).
        **kwargs: Forwarded to the factory function.

    Returns:
        An ``InterventionManager`` instance.
    """
    factory_fn = _import_factory(intervention_factory)
    logger.info("Creating intervention manager via %s", intervention_factory)
    return factory_fn(env=env, **kwargs)
