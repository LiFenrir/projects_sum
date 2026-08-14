"""Training configuration for innov_openpi.

Configuration is loaded from YAML files. Each YAML file is a complete
description of a training or inference run.

Usage::

    python scripts/train_pytorch.py --config configs/bi_s1/pi05_finetune.yaml
    python scripts/train_pytorch.py --config configs/bi_s1/pi05_finetune.yaml --batch-size 64
"""

import abc
from collections.abc import Sequence
import dataclasses
import difflib
import logging
import pathlib
from typing import Any, Literal, Protocol, TypeAlias

import etils.epath as epath
import tyro
from typing_extensions import override

import openpi.models.model as _model
import openpi.models.pi0_config as pi0_config
import openpi.models.pi0_fast as pi0_fast
import openpi.models.tokenizer as _tokenizer
import openpi.policies.lerobot_policy as lerobot_policy
import openpi.shared.download as _download
import openpi.shared.normalize as _normalize
import openpi.training.optimizer as _optimizer
import openpi.training.weight_loaders as weight_loaders
import openpi.transforms as _transforms

ModelType: TypeAlias = _model.ModelType


@dataclasses.dataclass(frozen=True)
class AssetsConfig:
    """Determines the location of assets (e.g., norm stats) that will be used to set up the data pipeline.

    These assets will be replicated inside the checkpoint under the ``assets/asset_id`` directory.

    This can be used to load assets from a different checkpoint (e.g., base model checkpoint) or some other
    centralized location. For example, to load the norm stats for the Trossen robot from the base model checkpoint
    during fine-tuning, use::

        AssetsConfig(
            assets_dir="gs://openpi-assets/checkpoints/pi0_base/assets",
            asset_id="trossen",
        )
    """

    # Assets directory. If not provided, the config assets_dirs will be used.
    assets_dir: str | None = None

    # Asset id. If not provided, the repo id will be used.
    asset_id: str | None = None


@dataclasses.dataclass(frozen=True)
class DataConfig:
    # LeRobot repo id. If None, fake data will be created.
    repo_id: str | None = None
    # Directory within the assets directory containing the data assets.
    asset_id: str | None = None
    # Contains precomputed normalization stats. If None, normalization will not be performed.
    norm_stats: dict[str, _transforms.NormStats] | None = None

    # Used to adopt the inputs from a dataset specific format to a common format
    # which is expected by the data transforms.
    repack_transforms: _transforms.Group = dataclasses.field(default_factory=_transforms.Group)
    # Data transforms, typically include robot specific transformations. Will be applied
    # before the data is normalized.
    data_transforms: _transforms.Group = dataclasses.field(default_factory=_transforms.Group)
    # Model specific transforms. Will be applied after the data is normalized.
    model_transforms: _transforms.Group = dataclasses.field(default_factory=_transforms.Group)
    # If true, will use quantile normalization. Otherwise, normal z-score normalization will be used.
    use_quantile_norm: bool = False

    # Names of keys that will be used by the data loader to generate the action sequence.
    action_sequence_keys: Sequence[str] = ("actions",)

    # If true, will use the LeRobot dataset task to define the prompt.
    prompt_from_task: bool = False


class GroupFactory(Protocol):
    def __call__(self, model_config: _model.BaseModelConfig) -> _transforms.Group:
        """Create a group."""


@dataclasses.dataclass(frozen=True)
class ModelTransformFactory(GroupFactory):
    """Creates model transforms for standard pi0 models."""

    # If provided, will determine the default prompt that be used by the model.
    default_prompt: str | None = None

    def __call__(self, model_config: _model.BaseModelConfig) -> _transforms.Group:
        match model_config.model_type:
            case _model.ModelType.PI0:
                return _transforms.Group(
                    inputs=[
                        _transforms.InjectDefaultPrompt(self.default_prompt),
                        _transforms.ResizeImages(224, 224),
                        _transforms.TokenizePrompt(
                            _tokenizer.PaligemmaTokenizer(model_config.max_token_len),
                        ),
                        _transforms.PadStatesAndActions(model_config.action_dim),
                    ],
                )
            case _model.ModelType.PI05:
                assert isinstance(model_config, pi0_config.Pi0Config)
                return _transforms.Group(
                    inputs=[
                        _transforms.InjectDefaultPrompt(self.default_prompt),
                        _transforms.ResizeImages(224, 224),
                        _transforms.TokenizePrompt(
                            _tokenizer.PaligemmaTokenizer(model_config.max_token_len),
                            discrete_state_input=model_config.discrete_state_input,
                        ),
                        _transforms.PadStatesAndActions(model_config.action_dim),
                    ],
                )
            case _model.ModelType.PI0_FAST:
                tokenizer_cls = (
                    _tokenizer.FASTTokenizer
                    if model_config.fast_model_tokenizer is None
                    else model_config.fast_model_tokenizer
                )
                tokenizer_kwargs = (
                    {} if model_config.fast_model_tokenizer_kwargs is None else model_config.fast_model_tokenizer_kwargs
                )
                return _transforms.Group(
                    inputs=[
                        _transforms.InjectDefaultPrompt(self.default_prompt),
                        _transforms.ResizeImages(224, 224),
                        _transforms.TokenizeFASTInputs(
                            tokenizer_cls(model_config.max_token_len, **tokenizer_kwargs),
                        ),
                    ],
                    outputs=[
                        _transforms.ExtractFASTActions(
                            tokenizer_cls(model_config.max_token_len, **tokenizer_kwargs),
                            action_horizon=model_config.action_horizon,
                            action_dim=model_config.action_dim,
                        )
                    ],
                )


@dataclasses.dataclass(frozen=True)
class DataConfigFactory(abc.ABC):
    # The LeRobot repo id.
    repo_id: str = tyro.MISSING
    # Determines how the assets will be loaded.
    assets: AssetsConfig = dataclasses.field(default_factory=AssetsConfig)
    # Base config that will be updated by the factory.
    base_config: tyro.conf.Suppress[DataConfig | None] = None

    @abc.abstractmethod
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        """Create a data config."""

    def create_base_config(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        repo_id = self.repo_id if self.repo_id is not tyro.MISSING else None
        asset_id = self.assets.asset_id or repo_id
        return dataclasses.replace(
            self.base_config or DataConfig(),
            repo_id=repo_id,
            asset_id=asset_id,
            norm_stats=self._load_norm_stats(epath.Path(self.assets.assets_dir or assets_dirs), asset_id),
            use_quantile_norm=model_config.model_type != ModelType.PI0,
        )

    def _load_norm_stats(self, assets_dir: epath.Path, asset_id: str | None) -> dict[str, _transforms.NormStats] | None:
        if asset_id is None:
            return None
        try:
            data_assets_dir = str(assets_dir / asset_id)
            norm_stats = _normalize.load(_download.maybe_download(data_assets_dir))
            logging.info(f"Loaded norm stats from {data_assets_dir}")
            return norm_stats
        except FileNotFoundError:
            logging.info(f"Norm stats not found in {data_assets_dir}, skipping.")
        return None


@dataclasses.dataclass(frozen=True)
class LeRobotDataConfig(DataConfigFactory):
    """Data config for LeRobot-compatible robots (bi_s1, arx_x5, s1, so100, etc.).

    Supports multiple robot types. ``assets.asset_id`` specifies the norm stats
    directory (e.g., "bi_s1", "arx_x5"), which is computed before training and
    stored under the assets directory. Camera mapping is configurable via
    ``camera_map``, which maps from robodeploy camera names to OpenPI model
    image slots. Example::

        {"front": "base_0_rgb", "left_wrist": "left_wrist_0_rgb", "right_wrist": "right_wrist_0_rgb"}

    Any camera not listed in the map is assigned to the next available model slot
    in sorted alphabetical order.

    The model always outputs 32-dim actions; the output transform truncates to
    the robot-specific ``action_dim`` (e.g., 14 for bi_s1).
    """

    # Robot type identifier (e.g., "bi_s1", "arx_x5").
    robot_type: str = "bi_s1"

    # Mapping from robodeploy camera names to OpenPI model image slots.
    camera_map: dict[str, str] = dataclasses.field(default_factory=dict)

    broadcast_base: bool = False
    # If provided, will be injected into the input data if the "prompt" key is not present.
    default_prompt: str | None = None
    # If true, will convert joint dimensions to deltas with respect to the current state.
    # Gripper dimensions will remain in absolute values.
    use_delta_joint_actions: bool = False

    # Robot-specific action dimension for output truncation. Model always uses 32-dim.
    action_dim: int = 14

    # Repack transforms — dynamically built from camera_map in create().
    repack_transforms: tyro.conf.Suppress[_transforms.Group] = dataclasses.field(
        default=_transforms.Group()
    )
    # Action keys that will be used to read the action sequence from the dataset.
    action_sequence_keys: Sequence[str] = ("action",)

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        data_transforms = _transforms.Group(
            inputs=[lerobot_policy.LeRobotInputs(camera_map=self.camera_map, default_prompt=self.default_prompt)],
            outputs=[lerobot_policy.LeRobotOutputs(action_dim=self.action_dim)],
        )
        if self.use_delta_joint_actions:
            delta_action_mask = _transforms.make_bool_mask(6, -1, 6, -1)
            data_transforms = data_transforms.push(
                inputs=[_transforms.DeltaActions(delta_action_mask)],
                outputs=[_transforms.AbsoluteActions(delta_action_mask)],
            )

        model_transforms = ModelTransformFactory(default_prompt=self.default_prompt)(model_config)

        # Dynamically build images repack from camera_map keys
        images_repack = {cam: f"observation.images.{cam}" for cam in self.camera_map}
        repack_transforms = dataclasses.replace(
            self.repack_transforms,
            inputs=[
                _transforms.RepackTransform(
                    {
                        "images": images_repack,
                        "state": "observation.state",
                        "actions": "action",
                    }
                )
            ],
        )

        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            repack_transforms=repack_transforms,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
            action_sequence_keys=self.action_sequence_keys,
        )


@dataclasses.dataclass(frozen=True)
class TrainConfig:
    # Name of the config. Must be unique.
    name: str = "innov_openpi"
    # Project name.
    project_name: str = "innov_openpi"
    # Experiment name. Will be used to name the metadata and checkpoint directories.
    exp_name: str = tyro.MISSING

    # Defines the model config.
    model: _model.BaseModelConfig = dataclasses.field(default_factory=pi0_config.Pi0Config)

    # A weight loader can optionally load (possibly partial) weights from disk after the model is initialized.
    weight_loader: weight_loaders.WeightLoader = dataclasses.field(default_factory=weight_loaders.NoOpWeightLoader)

    # Optional path to a PyTorch checkpoint to load weights from.
    pytorch_weight_path: str | None = None

    # Precision for PyTorch training.
    pytorch_training_precision: Literal["bfloat16", "float32"] = "bfloat16"

    lr_schedule: _optimizer.LRScheduleConfig = dataclasses.field(default_factory=_optimizer.CosineDecaySchedule)
    optimizer: _optimizer.OptimizerConfig = dataclasses.field(default_factory=_optimizer.AdamW)
    ema_decay: float | None = 0.99

    # Determines the data to be trained on.
    data: DataConfigFactory = dataclasses.field(default_factory=LeRobotDataConfig)

    # Base directory for config assets (e.g., norm stats).
    assets_base_dir: str = "./assets"
    # Base directory for checkpoints.
    checkpoint_base_dir: str = "./checkpoints"

    # Random seed that will be used by random generators during training.
    seed: int = 42
    # Global batch size.
    batch_size: int = 32
    # Number of workers to use for the data loader.
    num_workers: int = 2
    # Number of train steps (batches) to run.
    num_train_steps: int = 30_000

    # How often (in steps) to log training metrics.
    log_interval: int = 100
    # How often (in steps) to save checkpoints.
    save_interval: int = 1000
    # If set, any existing checkpoints matching step % keep_period == 0 will not be deleted.
    keep_period: int | None = 5000

    # If true, will overwrite the checkpoint directory if it already exists.
    overwrite: bool = False
    # If true, will resume training from the last checkpoint.
    resume: bool = False

    # If true, will enable wandb logging.
    wandb_enabled: bool = True

    # Used to pass metadata to the policy server.
    policy_metadata: dict[str, Any] | None = None

    # If the value is greater than 1, FSDP will be enabled and shard across number of specified devices.
    fsdp_devices: int = 1

    # 训练中热启动评估(0 = 关闭)
    eval_every: int = 0
    eval_task: str = "stack_bowls"
    eval_gpu: int = 1
    eval_timeout: float = 1800.0

    @property
    def assets_dirs(self) -> pathlib.Path:
        """Get the assets directory for this config."""
        return pathlib.Path(self.assets_base_dir).resolve()

    @property
    def checkpoint_dir(self) -> pathlib.Path:
        """Get the checkpoint directory for this config."""
        if not self.exp_name:
            raise ValueError("--exp_name must be set")
        return (pathlib.Path(self.checkpoint_base_dir) / self.name / self.exp_name).resolve()

    def __post_init__(self) -> None:
        if self.resume and self.overwrite:
            raise ValueError("Cannot resume and overwrite at the same time.")


# Registry of class names → types for resolving _target_ fields in YAML configs.
_CLASS_REGISTRY: dict[str, type] = {
    # Model configs
    "Pi0Config": pi0_config.Pi0Config,
    "Pi0FASTConfig": pi0_fast.Pi0FASTConfig,
    # Data configs
    "LeRobotDataConfig": LeRobotDataConfig,
    # Weight loaders
    "CheckpointWeightLoader": weight_loaders.CheckpointWeightLoader,
    "NoOpWeightLoader": weight_loaders.NoOpWeightLoader,
    # LR schedules
    "CosineDecaySchedule": _optimizer.CosineDecaySchedule,
    # Optimizers
    "AdamW": _optimizer.AdamW,
}


def load_config(yaml_path: str) -> TrainConfig:
    """Load a complete training config from a YAML file.

    The YAML file contains a complete configuration. All top-level keys map to
    ``TrainConfig`` fields, with nested dataclass conversion handled automatically.

    Use ``_target_`` fields to specify concrete class names for abstract-typed
    fields (e.g. ``model._target_: Pi0Config``, ``data._target_: LeRobotDataConfig``).

    Usage::

        config = load_config("configs/bi_s1/pi05_finetune.yaml")

    Args:
        yaml_path: Path to the YAML configuration file.

    Returns:
        Fully populated TrainConfig instance.
    """
    from openpi.shared.yaml_utils import load_config as _load_yaml

    return _load_yaml(yaml_path, config_cls=TrainConfig, class_registry=_CLASS_REGISTRY)


def load_config_with_cli_overrides(yaml_path: str, cli_args: list[str] | None = None) -> TrainConfig:
    """Load config from YAML, optionally applying CLI field overrides.

    Args:
        yaml_path: Path to the YAML configuration file.
        cli_args: Additional CLI arguments for field overrides (e.g. ``["--batch-size", "64"]``).

    Returns:
        Merged TrainConfig instance.
    """
    import sys

    config = load_config(yaml_path)

    if cli_args:
        sys.argv = [sys.argv[0], *cli_args]
        return tyro.cli(TrainConfig, default=config)

    return config


def get_config(config_name: str) -> TrainConfig:
    """Get a config by YAML file path or config name.

    This is a compatibility function — prefer :func:`load_config` for new code.

    Args:
        config_name: Path to a YAML config file.

    Returns:
        TrainConfig instance.
    """
    import os

    if os.path.isfile(config_name):
        return load_config(config_name)

    raise ValueError(
        f"Config '{config_name}' not found as a file. "
        f"Use the full path to a YAML config file, e.g. 'configs/bi_s1/pi05_finetune.yaml'."
    )
