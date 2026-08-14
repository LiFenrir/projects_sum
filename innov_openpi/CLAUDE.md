# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 项目概述

innov_openpi 是基于 OpenPI 框架的 VLA（Vision-Language-Action）机器人训练项目，包含三条训练流水线：

- **SFT 微调**：在示范数据上微调 π₀.₅ VLA 模型（PyTorch 实现）
- **RL Token 训练（Stage 1）**：信息瓶颈编码器-解码器，将 VLA 嵌入压缩为紧凑状态表示
- **在线 RL 训练（Stage 2）**：TD3 算法（双 Q-critic + BC 正则化），在线优化策略

模型架构：π₀.₅ = PaliGemma 2B（视觉-语言编码）+ Gemma 300M（动作专家），通过 flow matching 生成动作。

## 常用命令

### 安装与环境

```bash
conda create -n innov_openpi python=3.11
conda activate innov_openpi
pip install -e .
# 必须：应用 transformers 补丁（修改了 SigLIP）
cp -r ./src/openpi/models_pytorch/transformers_replace/* $CONDA_PREFIX/lib/python3.11/site-packages/transformers/
```

### 代码质量

```bash
ruff check .          # lint 检查
ruff format .         # 格式化（行宽 120）
pytest src/rlt/tests/ # 运行测试（仅 rlt 模块有测试）
```

### SFT 微调

```bash
# 计算归一化统计量（训练前必须）
python scripts/compute_norm_stats.py --config configs/bi_s1/pi05_finetune.yaml

# 单卡训练
python scripts/train_pytorch.py --config configs/bi_s1/pi05_finetune.yaml --exp_name my_run

# 多卡 DDP
torchrun --standalone --nnodes=1 --nproc_per_node=4 \
    scripts/train_pytorch.py --config configs/bi_s1/pi05_finetune.yaml --exp_name my_run
```

### RL Token 训练（Stage 1）

```bash
# 冻结 VLA，仅训练 RL Token
python scripts/train_rl_token.py --config configs/rlt/stage1_rl_token.yaml \
    --train.vla_checkpoint_dir checkpoints/my_run/20000

# 联合训练（同时微调 VLA 和 RL Token）
python scripts/train_rl_token.py --config configs/rlt/stage1_rl_token.yaml \
    --train.vla_checkpoint_dir checkpoints/my_run/20000 \
    --train.vla_finetune_alpha 0.5
```

### 在线 RL 训练（Stage 2）

```bash
python scripts/train_online_rl.py --config configs/rlt/stage2_online_rl.yaml \
    --vla_checkpoint_dir checkpoints/my_run/20000 \
    --rl_token_checkpoint checkpoints/rl_token/my_run/rl_token_step5000.pt \
    --env_factory my_package.envs.make_env \
    --task_prompt "stack the three blocks"
```

### 推理与服务

```bash
# 启动策略服务器
python scripts/serve_policy.py policy:checkpoint \
    --policy.config=configs/bi_s1/pi05_inference.yaml \
    --policy.dir=checkpoints/my_run/20000

# VLA rollout 评估
python scripts/rollout_vla.py \
    --vla-config-name configs/bi_s1/pi05_finetune.yaml \
    --vla-checkpoint-dir checkpoints/my_run/20000 \
    --env-factory my_package.envs.make_env \
    --num-episodes 10
```

### RECAP 模块（价值模型 + 优势计算）

```bash
python scripts/recap/train_value_sft.py --config configs/recap/recap_value_sft.yaml
python scripts/recap/compute_returns.py --config configs/recap/recap_compute_returns.yaml
python scripts/recap/compute_advantages.py --config configs/recap/recap_compute_advantages.yaml --value_checkpoint <path>
python scripts/recap/train_cfg_sft.py --config configs/recap/recap_cfg_sft.yaml
```

## 代码架构

### 三层包结构

```
src/
├── openpi/          # 核心 VLA 框架（基于 Physical-Intelligence/openpi）
│   ├── models/           # JAX 模型实现（Gemma, SigLIP, Pi0, Pi0-FAST, LoRA）
│   ├── models_pytorch/   # PyTorch 模型实现（PI0Pytorch, 训练用）
│   │   └── transformers_replace/  # 修改版 transformers（SigLIP 补丁）
│   ├── training/         # SFT 训练循环、数据加载、检查点、配置系统
│   ├── policies/         # 策略抽象 + LeRobot 机器人数据转换
│   ├── serving/          # WebSocket 策略服务器
│   ├── shared/           # 归一化、图像工具、YAML、下载等工具
│   └── transforms.py     # 数据变换管道（组合模式）
│
├── rlt/              # RL Token 训练模块
│   ├── models/           # RLTokenModel（编码器-解码器）、Actor、TwinQCritic
│   ├── training/         # RLTokenTrainer（Stage 1）、OnlineRLTrainer（Stage 2）
│   ├── rollout/          # RolloutWorker、环境包装器、人类干预、奖励函数
│   ├── policies/         # RL 特定策略配置工厂
│   ├── utils/            # 日志、检查点、终端显示
│   └── tests/            # 单元测试（actor/critic、rl_token、replay_buffer 等）
│
└── recap/            # RECAP 模块（Reward-Conditioned Action Prediction）
    ├── models/value_critic/  # ValueCriticModel（基于 Gemma 的价值函数）
    ├── training/             # ValueTrainer、CFGTrainer
    └── data/                 # 机器人特定数据配置（bi_s1, libero, franka）
```

### 训练流水线架构（数据流）

```
SFT (Stage 0):
  示范数据 → LeRobot Dataset → DataLoader → PI0Pytorch.forward()
  → Flow Matching Loss → 反向传播 → 更新 VLA

Stage 1 (RL Token):
  示范数据 → VLA.extract_embeddings() → z (prefix embeddings)
  → RLTokenModel (encoder→bottleneck→decoder) → L_ro 重构损失
  可选: + α * L_vla (联合微调 VLA)

Stage 2 (Online RL):
  env obs → VLA.preprocess_obs() → VLA 推理 → 参考动作 a_tilde
  → RLTokenModel.encode() → 压缩状态 z_rl（RL token）
  → Actor(z_rl, a_tilde) → 动作 a → 环境执行 → (r, next_obs)
  → ReplayBuffer → TD3 更新 (critic + actor with BC regularization)
```

### YAML 配置系统

所有训练配置使用 YAML + `_target_` 模式实现类的声明式实例化：

```yaml
model:
  _target_: Pi0Config       # 类名，通过注册表解析
  pi05: true
  action_dim: 32
data:
  _target_: LeRobotDataConfig
  repo_id: your_hf_username/my_dataset
weight_loader:
  _target_: CheckpointWeightLoader
  params_path: gs://openpi-assets/checkpoints/pi05_base/params
```

`_target_` 的解析逻辑在 `src/openpi/training/config.py` 的 `TYPE_REGISTRY` 中。

### 关键模型组件

- **PI0Pytorch** (`src/openpi/models_pytorch/pi0_pytorch.py`)：核心 VLA 模型。包含：
  - `embed_prefix()`：SigLIP 编码图像 + Gemma 嵌入文本 → prefix embeddings
  - `embed_suffix()`：状态 + 噪声动作 + 时间步 → suffix embeddings（动作专家输入）
  - `forward()`：Flow matching 训练前向（噪声 → 速度场预测）
  - `sample_actions()`：推理时采样动作（从噪声迭代去噪）
  - `extract_prefix_embeddings()`：提取 prefix 嵌入供 RL Token 使用
  - `forward_with_prefix_embeddings()`：单次前向同时返回 prefix 嵌入和 VLA loss

- **VLAWrapper** (`src/openpi/training/vla_wrapper.py`)：封装 PI0Pytorch，管理输入/输出变换（归一化/反归一化），提供 `extract_embeddings()`、`sample_reference_actions()`、`compute_vla_loss_with_embeddings()` 等高层 API

- **RLTokenModel** (`src/rlt/models/rl_token.py`)：Transformer 编码器-解码器。编码器将 VLA prefix 嵌入压缩为单个 RL token；解码器从 RL token 重建完整 prefix 嵌入

### 关键注意事项

- **transformers 补丁是必须的**：`src/openpi/models_pytorch/transformers_replace/` 包含修改过的 SigLIP 实现。PI0Pytorch 初始化时会自检（`check_whether_transformers_replace_is_installed_correctly()`），失败则抛错。

- **归一化统计量**：训练前必须运行 `compute_norm_stats.py` 生成 norm stats。VLAWrapper 加载时优先从 checkpoint 的 `assets/` 目录加载 norm stats，失败则回退到配置中的 assets 目录。

- **DDP vs FSDP**：DDP（通过 torchrun）是主要多 GPU 方案。FSDP（`fsdp_devices > 1`）仅在单卡放不下模型时使用。

- **环境变量**：
  - `OPENPI_DATA_HOME`：模型检查点缓存目录，默认 `~/.cache/openpi`
  - `XLA_PYTHON_CLIENT_MEM_FRACTION=0.9`：GPU 显存不足时设置

- **LeRobot 数据集格式**：所有训练数据使用 LeRobot 格式。`LeRobotInputs`/`LeRobotOutputs` 负责 robodeploy 格式 ↔ OpenPI 内部格式的转换。Camera mapping 通过 `camera_map` 配置。

- **wandb 日志**：所有训练脚本默认启用 wandb（`wandb_enabled: true`），项目名 `rlt-openpi`（RL 流水线）或由 config 指定。
