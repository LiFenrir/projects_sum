"""层次化配置 dataclass（RL Token Stage 1 + Online RL Stage 2）。

每个 Stage 的配置按语义分组为子 dataclass，替代原有平铺结构。
提供迁移函数兼容旧版 checkpoint 和 YAML 格式。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any


# ════════════════════════════════════════════════════════════════════════════
# 共享子配置
# ════════════════════════════════════════════════════════════════════════════


@dataclass
class WandbConfig:
    """wandb 日志配置（Stage 1/2 共用）。"""

    project: str = "rlt-openpi"
    enabled: bool = True


# ════════════════════════════════════════════════════════════════════════════
# Stage 2（Online RL）子配置
# ════════════════════════════════════════════════════════════════════════════


@dataclass
class RLArchConfig:
    """Actor-Critic 网络架构。"""

    mlp_hidden_dim: int = 256
    mlp_num_hidden_layers: int = 2
    embedding_dim: int = 2048
    actor_noise_sigma: float = 0.1
    ref_action_dropout: float = 0.5


@dataclass
class TD3Config:
    """TD3 算法超参。"""

    gamma: float = 0.99
    tau: float = 0.005
    utd_ratio: int = 5
    bc_regularizer_beta: float = 0.5
    critic_updates_per_actor: int = 2
    target_noise_sigma: float = 0.2
    target_noise_clip: float = 0.5


@dataclass
class OptimizerConfig:
    """优化器学习率。"""

    actor_lr: float = 3e-4
    critic_lr: float = 3e-4


@dataclass
class BufferConfig:
    """Replay Buffer 配置。"""

    capacity: int = 100_000
    batch_size: int = 256
    warmup_steps: int = 1000


@dataclass
class EnvConfig:
    """环境与任务配置。"""

    env_factory: str = ""
    intervention_factory: str = ""
    task_prompt: str = ""
    max_episode_chunks: int = 150
    # 透传给 env factory 的额外 kwargs(如 robodojo 的 task_name/root/python)
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class CheckpointConfig:
    """Stage 2 检查点与日志配置。"""

    rl_token_checkpoint: str = ""
    vla_checkpoint_dir: str = ""
    vla_config_name: str = "pi05_droid_finetune"
    resume_checkpoint: str = ""
    warmup_buffer: str = ""
    save_dir: str = "checkpoints/online_rl"
    run_name: str = ""
    save_every: int = 50
    log_every: int = 1
    print_every: int = 100
    # 切换点数据(分类器训练):空 = 默认 {save_dir}/{run_name}/switch_data
    switch_data_dir: str = ""
    switch_save_raw_obs: bool = False

    def __post_init__(self) -> None:
        if not self.run_name:
            self.run_name = datetime.now().strftime("run_%Y%m%d_%H%M%S")
        if not self.switch_data_dir:
            self.switch_data_dir = str(Path(self.save_dir) / self.run_name / "switch_data")


# ════════════════════════════════════════════════════════════════════════════
# Stage 1（RL Token）子配置
# ════════════════════════════════════════════════════════════════════════════


@dataclass
class RLTokenArchConfig:
    """RL Token encoder-decoder 架构。"""

    embedding_dim: int = 2048
    encoder_layers: int = 2
    encoder_heads: int = 8
    decoder_layers: int = 2
    decoder_heads: int = 8


@dataclass
class RLTokenTrainingConfig:
    """Stage 1 训练超参。"""

    num_train_steps: int = 5000
    batch_size: int = 32
    peak_lr: float = 1e-4
    weight_decay: float = 1e-5
    warmup_steps: int = 500
    decay_steps: int = 5000
    decay_lr: float = 1e-5
    max_grad_norm: float = 1.0
    vla_finetune_alpha: float = 0.0
    vla_learning_rate: float = 1e-5
    gradient_checkpointing: bool = True


@dataclass
class RLTokenCheckpointConfig:
    """Stage 1 检查点配置。"""

    vla_checkpoint_dir: str = ""
    vla_config_name: str = "pi05_droid_finetune"
    resume_checkpoint: str = ""
    save_dir: str = "checkpoints/rl_token"
    run_name: str = ""
    save_every: int = 1000
    log_every: int = 1
    print_every: int = 100

    def __post_init__(self) -> None:
        if not self.run_name:
            self.run_name = datetime.now().strftime("run_%Y%m%d_%H%M%S")


# ════════════════════════════════════════════════════════════════════════════
# 主配置类
# ════════════════════════════════════════════════════════════════════════════


@dataclass
class RLTokenTrainConfig:
    """Stage 1: RL token encoder-decoder 训练配置。"""

    arch: RLTokenArchConfig = field(default_factory=RLTokenArchConfig)
    training: RLTokenTrainingConfig = field(default_factory=RLTokenTrainingConfig)
    checkpoint: RLTokenCheckpointConfig = field(default_factory=RLTokenCheckpointConfig)
    wandb: WandbConfig = field(default_factory=WandbConfig)

    # Stage 1 也接收 learning_rate 作为 peak_lr 的别名
    def __post_init__(self) -> None:
        # 当通过旧版 YAML 加载时，migration 可能设置 training.learning_rate 而非 peak_lr
        pass


@dataclass
class OnlineRLTrainConfig:
    """Stage 2: Online RL 训练配置（Algorithm 1）。"""

    # ── 动作空间（顶层，无自然子组）──
    action_dim: int = 8
    chunk_length: int = 10  # 论文 C=10(RL chunk,50Hz 下 0.2s)
    vla_action_horizon: int = 50  # 论文 H=50(VLA 参考采样 1s)
    max_env_steps: int = 100_000
    critical_phase_only: bool = False
    obs_stride: int = 2  # stride 子采样步长(论文 stride=2;1 = 禁用)
    async_collection: bool = True  # 异步:采集线程与训练线程并行(论文异步 rollout/learning)

    # ── 子配置 ──
    rl_arch: RLArchConfig = field(default_factory=RLArchConfig)
    td3: TD3Config = field(default_factory=TD3Config)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    buffer: BufferConfig = field(default_factory=BufferConfig)
    env: EnvConfig = field(default_factory=EnvConfig)
    checkpoint: CheckpointConfig = field(default_factory=CheckpointConfig)
    wandb: WandbConfig = field(default_factory=WandbConfig)

    @property
    def state_dim(self) -> int:
        """RL state 维度: z_rl + s^p。"""
        return self.rl_arch.embedding_dim + self.action_dim

    @property
    def action_chunk_dim(self) -> int:
        """展平动作块维度: C * d。"""
        return self.chunk_length * self.action_dim


# ════════════════════════════════════════════════════════════════════════════
# 迁移函数：旧平铺格式 → 新嵌套格式
# ════════════════════════════════════════════════════════════════════════════

# 旧 OnlineRLTrainConfig 平铺字段（用于检测旧格式）
_ONLINE_RL_OLD_FLAT_KEYS = frozenset({
    "embedding_dim", "mlp_hidden_dim", "mlp_num_hidden_layers",
    "actor_noise_sigma", "ref_action_dropout",
    "gamma", "tau", "utd_ratio", "bc_regularizer_beta",
    "critic_updates_per_actor", "target_noise_sigma", "target_noise_clip",
    "actor_lr", "critic_lr",
    "buffer_capacity", "batch_size", "warmup_steps",
    "env_factory", "intervention_factory", "task_prompt", "max_episode_chunks",
    "rl_token_checkpoint", "vla_checkpoint_dir", "vla_config_name",
    "resume_checkpoint", "warmup_buffer",
    "save_dir", "run_name", "save_every", "log_every", "print_every",
    "wandb_project", "wandb_enabled",
})

# 旧 RLTokenTrainConfig 平铺字段
_RL_TOKEN_OLD_FLAT_KEYS = frozenset({
    "embedding_dim", "encoder_layers", "encoder_heads",
    "decoder_layers", "decoder_heads",
    "num_train_steps", "batch_size", "learning_rate",
    "weight_decay", "warmup_steps", "max_grad_norm",
    "peak_lr", "decay_steps", "decay_lr",
    "vla_finetune_alpha", "vla_learning_rate", "gradient_checkpointing",
    "vla_checkpoint_dir", "vla_config_name", "resume_checkpoint",
    "save_dir", "run_name", "save_every", "log_every", "print_every",
    "wandb_project", "wandb_enabled",
})


def _extract_fields(obj: Any) -> dict[str, Any]:
    """从 dataclass 实例或 dict 中提取字段 dict。"""
    if isinstance(obj, dict):
        return obj
    if hasattr(obj, "__dataclass_fields__"):
        return {k: v for k, v in vars(obj).items() if not k.startswith("_")}
    raise TypeError(f"Expected dataclass or dict, got {type(obj)}")


def _is_old_online_rl_format(d: dict[str, Any]) -> bool:
    """检测是否为旧版平铺 OnlineRLTrainConfig 格式（无 rl_arch/td3 等子键）。"""
    has_new_keys = {"rl_arch", "td3", "optimizer", "buffer", "env", "checkpoint", "wandb"} & d.keys()
    if has_new_keys:
        return False
    return bool(_ONLINE_RL_OLD_FLAT_KEYS & d.keys())


def _is_old_rl_token_format(d: dict[str, Any]) -> bool:
    """检测是否为旧版平铺 RLTokenTrainConfig 格式（无 arch/training 等子键）。"""
    has_new_keys = {"arch", "training", "checkpoint", "wandb"} & d.keys()
    if has_new_keys:
        return False
    return bool(_RL_TOKEN_OLD_FLAT_KEYS & d.keys())


def migrate_online_rl_config(config: Any) -> OnlineRLTrainConfig:
    """将旧版平铺格式的 OnlineRLTrainConfig 转换为新嵌套格式。

    兼容：
    - 旧版 YAML dict（加载 YAML 后得到）
    - 旧版 dataclass 实例（从 pickle checkpoint 加载后得到）
    - 新版 OnlineRLTrainConfig（直接返回）
    """
    # 已是新版格式
    if isinstance(config, OnlineRLTrainConfig):
        if hasattr(config, "rl_arch") and isinstance(config.rl_arch, RLArchConfig):
            return config

    d = _extract_fields(config)

    # 不是旧格式 → 新版嵌套 dict，交由 _dict_to_dataclass 构造
    if not _is_old_online_rl_format(d):
        return config if isinstance(config, dict) else d

    # 旧格式 → 逐字段迁移
    return OnlineRLTrainConfig(
        action_dim=d.get("action_dim", 8),
        chunk_length=d.get("chunk_length", 10),
        vla_action_horizon=d.get("vla_action_horizon", 50),
        max_env_steps=d.get("max_env_steps", 100_000),
        critical_phase_only=d.get("critical_phase_only", False),
        obs_stride=d.get("obs_stride", 2),
        async_collection=d.get("async_collection", True),
        rl_arch=RLArchConfig(
            embedding_dim=d.get("embedding_dim", 2048),
            mlp_hidden_dim=d.get("mlp_hidden_dim", 256),
            mlp_num_hidden_layers=d.get("mlp_num_hidden_layers", 2),
            actor_noise_sigma=d.get("actor_noise_sigma", 0.1),
            ref_action_dropout=d.get("ref_action_dropout", 0.5),
        ),
        td3=TD3Config(
            gamma=d.get("gamma", 0.99),
            tau=d.get("tau", 0.005),
            utd_ratio=d.get("utd_ratio", 5),
            bc_regularizer_beta=d.get("bc_regularizer_beta", 0.5),
            critic_updates_per_actor=d.get("critic_updates_per_actor", 2),
            target_noise_sigma=d.get("target_noise_sigma", 0.2),
            target_noise_clip=d.get("target_noise_clip", 0.5),
        ),
        optimizer=OptimizerConfig(
            actor_lr=d.get("actor_lr", 3e-4),
            critic_lr=d.get("critic_lr", 3e-4),
        ),
        buffer=BufferConfig(
            capacity=d.get("buffer_capacity", 100_000),
            batch_size=d.get("batch_size", 256),
            warmup_steps=d.get("warmup_steps", 1000),
        ),
        env=EnvConfig(
            env_factory=d.get("env_factory", ""),
            intervention_factory=d.get("intervention_factory", ""),
            task_prompt=d.get("task_prompt", ""),
            max_episode_chunks=d.get("max_episode_chunks", 150),
        ),
        checkpoint=CheckpointConfig(
            rl_token_checkpoint=d.get("rl_token_checkpoint", ""),
            vla_checkpoint_dir=d.get("vla_checkpoint_dir", ""),
            vla_config_name=d.get("vla_config_name", "pi05_droid_finetune"),
            resume_checkpoint=d.get("resume_checkpoint", ""),
            warmup_buffer=d.get("warmup_buffer", ""),
            save_dir=d.get("save_dir", "checkpoints/online_rl"),
            run_name=d.get("run_name", ""),
            save_every=d.get("save_every", 50),
            log_every=d.get("log_every", 1),
            print_every=d.get("print_every", 100),
            switch_data_dir=d.get("switch_data_dir", ""),
            switch_save_raw_obs=d.get("switch_save_raw_obs", False),
        ),
        wandb=WandbConfig(
            project=d.get("wandb_project", "rlt-openpi"),
            enabled=d.get("wandb_enabled", True),
        ),
    )


def migrate_rl_token_config(config: Any) -> RLTokenTrainConfig:
    """将旧版平铺格式的 RLTokenTrainConfig 转换为新嵌套格式。"""
    if isinstance(config, RLTokenTrainConfig):
        if hasattr(config, "arch") and isinstance(config.arch, RLTokenArchConfig):
            return config

    d = _extract_fields(config)

    # 不是旧格式 → 新版嵌套 dict，交由 _dict_to_dataclass 构造
    if not _is_old_rl_token_format(d):
        return config if isinstance(config, dict) else d

    lr = d.get("learning_rate", 1e-4)
    return RLTokenTrainConfig(
        arch=RLTokenArchConfig(
            embedding_dim=d.get("embedding_dim", 2048),
            encoder_layers=d.get("encoder_layers", 2),
            encoder_heads=d.get("encoder_heads", 8),
            decoder_layers=d.get("decoder_layers", 2),
            decoder_heads=d.get("decoder_heads", 8),
        ),
        training=RLTokenTrainingConfig(
            num_train_steps=d.get("num_train_steps", 5000),
            batch_size=d.get("batch_size", 32),
            peak_lr=d.get("peak_lr", lr),
            weight_decay=d.get("weight_decay", 1e-5),
            warmup_steps=d.get("warmup_steps", 500),
            decay_steps=d.get("decay_steps", 5000),
            decay_lr=d.get("decay_lr", 1e-5),
            max_grad_norm=d.get("max_grad_norm", 1.0),
            vla_finetune_alpha=d.get("vla_finetune_alpha", 0.0),
            vla_learning_rate=d.get("vla_learning_rate", 1e-5),
            gradient_checkpointing=d.get("gradient_checkpointing", True),
        ),
        checkpoint=RLTokenCheckpointConfig(
            vla_checkpoint_dir=d.get("vla_checkpoint_dir", ""),
            vla_config_name=d.get("vla_config_name", "pi05_droid_finetune"),
            resume_checkpoint=d.get("resume_checkpoint", ""),
            save_dir=d.get("save_dir", "checkpoints/rl_token"),
            run_name=d.get("run_name", ""),
            save_every=d.get("save_every", 1000),
            log_every=d.get("log_every", 1),
            print_every=d.get("print_every", 100),
        ),
        wandb=WandbConfig(
            project=d.get("wandb_project", "rlt-openpi"),
            enabled=d.get("wandb_enabled", True),
        ),
    )
