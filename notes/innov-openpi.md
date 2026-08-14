---
title: innov_openpi
description: "fork 自 OpenPI 的 π₀.₅ VLA 训练项目:SFT / RL Token / TD3 在线 RL 三条流水线 + RECAP 价值模型 + WebSocket 策略服务。"
tags: [project, own, vla, openpi, rl, flow-matching]
created: 2026-07-16
---

# innov_openpi

仓库:`/home/kemove/INNOV/projects/innov_openpi`(fork 自 Physical-Intelligence/openpi)。面向自研机器人(bi_s1 / Innov Arm)的 VLA 训练与在线 RL 项目。

## 背景与动机

OpenPI 官方仓库只提供 π₀/π₀.₅ 的预训练权重与 完整的JAX 微调管线, pytorch生态不完整、在线 RL 能力、与自研真机数据链路的对接。

innov_openpi 在 fork 基础上做了三件加法:**PyTorch 版 Lora微调**、**RLT复现**、**RECAP 价值模型模块**,**rtc复现**数据格式与 [[06_Projects/own/robodeploy|robodeploy]] 采集端直接互通。

## 核心功能

- **SFT 微调** — π₀.₅ VLA(PyTorch 实现)在示范数据上微调,支持 DDP 多卡
- **RL Token 训练(Stage 1)** — 信息瓶颈编码器-解码器,把 VLA 嵌入压缩为紧凑状态表示
- **在线 RL(Stage 2)** — TD3(双 Q-critic + BC 正则化)在线优化策略
- **RECAP 价值模型** — 基于 Gemma 的 ValueCritic,奖励条件化动作预测与优势计算
- **策略服务** — WebSocket 策略服务器,供 robodeploy 远程推理调用

模型架构:π₀.₅ = PaliGemma 2B(视觉-语言编码)+ Gemma 300M(动作专家),flow matching 生成动作。

## 技术方案

### 代码结构

```
src/
├── openpi/          # 核心 VLA 框架(fork 自 openpi)
│   ├── models/           # JAX 模型(Gemma/SigLIP/Pi0/Pi0-FAST/LoRA)
│   ├── models_pytorch/   # PyTorch 实现(PI0Pytorch,训练用)
│   │   └── transformers_replace/  # 修改版 transformers(SigLIP 补丁)
│   ├── training/         # SFT 训练循环/数据/checkpoint/config(_target_ 注册表)
│   ├── policies/         # 策略抽象 + LeRobot 数据转换
│   ├── serving/          # WebSocket 策略服务器
│   └── shared/           # 归一化/图像/YAML/下载工具
├── rlt/              # RL Token 模块
│   ├── models/           # RLTokenModel(编解码)、Actor、TwinQCritic
│   ├── training/         # RLTokenTrainer(Stage 1)、OnlineRLTrainer(Stage 2)
│   └── rollout/          # RolloutWorker、环境包装、人类干预、奖励
└── recap/            # RECAP 模块(Reward-Conditioned Action Prediction)
    ├── models/value_critic/  # ValueCriticModel(基于 Gemma 的价值函数)
    ├── training/             # ValueTrainer、CFGTrainer
    └── data/                 # 机器人数据配置(bi_s1/libero/franka)
```

### 训练流水线数据流

```
SFT:   示范数据 → LeRobotDataset → PI0Pytorch.forward() → Flow Matching Loss
Stage1: VLA.extract_embeddings() → RLTokenModel(encoder→bottleneck→decoder) → L_ro 重构损失
        (+ α·L_vla 联合微调 VLA,α = vla_finetune_alpha)
Stage2: obs → VLA 推理得参考动作 a_tilde → RLTokenModel.encode() → z_rl
        → Actor(z_rl, a_tilde) → 动作 → 环境 → (r, next) → ReplayBuffer → TD3 更新
```

## 环境安装

```bash
# 1. 创建虚拟环境(conda,Python 3.11)
conda create -n innov_openpi python=3.11
conda activate innov_openpi

# 2. 安装
pip install -e .

# 3. 应用 transformers 补丁(必须,SigLIP 修改;启动自检失败会抛错)
cp -r ./src/openpi/models_pytorch/transformers_replace/* $CONDA_PREFIX/lib/python3.11/site-packages/transformers/

# 4. 下载预训练检查点
gsutil cp -r gs://openpi-assets/checkpoints/pi05_base ./checkpoints/
```

- 检查点默认缓存 `~/.cache/openpi`,用 `OPENPI_DATA_HOME` 覆盖;显存不足设 `XLA_PYTHON_CLIENT_MEM_FRACTION=0.9`
- 数据全部 LeRobot 格式,`LeRobotInputs/Outputs` 负责 robodeploy 格式 ↔ OpenPI 内部格式转换

## 配置机制

- 全部走 **YAML + `_target_` 注册表**:`load_config()`(`training/config.py`)按 `_target_` 字段把 YAML 段实例化为对应 dataclass(如 `model._target_: Pi0Config`、`data._target_: LeRobotDataConfig`)
- **CLI 覆盖**:任何字段可用点号覆盖,`--batch-size 64`、`--train.training.vla-finetune-alpha 1.0`
- 配置目录按机器人/模块组织:`configs/{bi_s1, arx, innov_arm, robodojo, rlt, recap}/`

## 功能使用

### 功能 1:LoRA 微调

配置:`configs/<robot>/pi05_finetune_<robot>_lora.yaml`(如 `configs/innov_arm/pi05_finetune_innov_arm_lora.yaml`)。仅 VLM 语言模型注入 LoRA(rank=16),冻结 VLM 主干,vision tower / action expert / 投影层照常训练;lora 参数按 normal(stddev=0.01) 随机初始化。

SFT 通用配置字段(LoRA/全参共用):

| 段 | 关键字段 | 说明 |
| --- | --- | --- |
| 顶层 | `name` / `exp_name` | 配置名 / 实验名(checkpoint 输出 `checkpoints/<exp_name>/<step>`) |
| `model` | `pi05: true`、`action_dim`、`action_horizon: 50`、`max_token_len` | π₀.₅ 开关与动作维度/时域 |
| `model` | `paligemma_variant` | `gemma_2b_lora`(LoRA)或 `gemma_2b`(全参) |
| `model` | `enable_blur_aug` | 中值+运动模糊增强(对齐 RISE,各 p=0.1) |
| `model.rtc_config` | `enabled` / `prefix_attention_schedule` / `execution_horizon` | RTC 推理时 chunk 平滑 |
| `data` | `repo_id` | LeRobot 数据集路径(本地绝对路径或 HF repo id) |
| `data` | `robot_type` / `default_prompt` | 机器人类型 / 默认任务指令 |
| `data.camera_map` | `front: base_0_rgb` 等 | 模型输入相机名 → 数据集相机键映射 |
| `data.assets` | `asset_id` | norm stats 资产标识 |
| `weight_loader` | `params_path` | base 权重路径(不含 lora 参数) |
| `lr_schedule` | `warmup_steps` / `peak_lr` / `decay_steps` / `decay_lr` | cosine 衰减;peak 2.5e-5 → 2.5e-6(对齐 JAX 配置) |
| `optimizer` | `AdamW`,`clip_gradient_norm: 1.0` | 权重衰减 1e-10 |
| 训练 | `batch_size`(**全局**,DDP 时每卡 batch/world_size) / `num_train_steps` / `save_interval` / `keep_period` / `fsdp_devices` | FSDP 仅单卡放不下时 >1 |
| `wandb_enabled` | true/false | wandb 开关 |

**换机器人/数据集只改四处**:`data.repo_id`、`data.robot_type`、`data.camera_map`、`data.assets.asset_id`。

```bash
# 先算归一化统计
python scripts/compute_norm_stats.py --config configs/innov_arm/pi05_finetune_innov_arm_lora.yaml

# 单卡
python scripts/train_pytorch.py --config configs/innov_arm/pi05_finetune_innov_arm_lora.yaml --exp_name my_run

# 多卡 DDP
torchrun --standalone --nnodes=1 --nproc_per_node=4 \
    scripts/train_pytorch.py --config configs/innov_arm/pi05_finetune_innov_arm_lora.yaml --exp_name my_run
```

### 功能 2:全参微调

配置:`configs/<robot>/pi05_finetune_<robot>.yaml`(如 `configs/innov_arm/pi05_finetune_innov_arm.yaml`)。与 LoRA 配置的差异仅一处:`paligemma_variant: gemma_2b`(全量训练 VLM),其余字段相同。

```bash
python scripts/compute_norm_stats.py --config configs/innov_arm/pi05_finetune_innov_arm.yaml
torchrun --standalone --nnodes=1 --nproc_per_node=4 \
    scripts/train_pytorch.py --config configs/innov_arm/pi05_finetune_innov_arm.yaml --exp_name my_run
```

### 功能 3:RLT(RL Token + 在线 RL)

**① Stage 1 — RL Token 训练**(`configs/rlt/stage1_rl_token.yaml`):

- `train.arch` — 编解码器结构:`embedding_dim: 2048`、encoder/decoder 各 2 层 8 头
- `train.training` — 训练超参 + **联合训练开关**:`vla_finetune_alpha`(0=冻结 VLA,>0 联合微调,配 `vla_learning_rate`)、`gradient_checkpointing`
- `train.checkpoint` — `vla_checkpoint_dir`(SFT 产物,必填)、`vla_config_name`(指回 SFT yaml 以复用 camera_map/norm stats)、`save_every`
- 顶层 `repo_id` — 数据集;`num_workers`

```bash
# 冻结 VLA,仅训练 RL Token
python scripts/train_rl_token.py --config configs/rlt/stage1_rl_token.yaml \
    --train.vla_checkpoint_dir checkpoints/my_run/20000

# 联合训练(同时微调 VLA 和 RL Token)
python scripts/train_rl_token.py --config configs/rlt/stage1_rl_token.yaml \
    --train.vla_checkpoint_dir checkpoints/my_run/20000 \
    --train.vla_finetune_alpha 0.5
```

**② Stage 2 — 在线 RL**(`configs/rlt/stage2_online_rl.yaml`):

| 段 | 关键字段 | 说明 |
| --- | --- | --- |
| 动作空间 | `action_dim: 14`(双臂;单臂 7)、`chunk_length: 10`(RL chunk)、`vla_action_horizon: 50` | 对齐论文 C=10、H=50 |
| 顶层开关 | `critical_phase_only`(阶段 RL vs 全局 RL)、`obs_stride: 2`、`async_collection`(异步双线程) | |
| `rl_arch` | `embedding_dim: 2048`、`mlp_hidden_dim: 256`、`actor_noise_sigma: 0.1`、`ref_action_dropout: 0.5` | Actor/Critic 结构 |
| `td3` | `gamma`、`tau`、`utd_ratio: 5`、`bc_regularizer_beta: 0.5`、`critic_updates_per_actor: 2` | TD3 超参 |
| `buffer` | `capacity` / `batch_size: 256` / `warmup_steps: 1000` | Replay Buffer |
| `env` | `env_factory` / `intervention_factory` / `task_prompt` / `max_episode_chunks` | 真机环境入口 |
| `checkpoint` | `rl_token_checkpoint`(Stage 1 产物,必填)、`vla_checkpoint_dir`(SFT 产物,必填)、`vla_config_name`、`warmup_buffer` | |

真机串口/相机**不写进 yaml**,走环境变量:`INNOV_ARM_ROBOT_TYPE`、`INNOV_ARM_LEFT_PORT`/`INNOV_ARM_RIGHT_PORT`(单臂 `INNOV_ARM_PORT`)、`INNOV_ARM_CAMERAS`(JSON,front 双相机垂直拼接)。

```bash
python scripts/train_online_rl.py --config configs/rlt/stage2_online_rl.yaml \
    --vla_checkpoint_dir checkpoints/my_run/20000 \
    --rl_token_checkpoint checkpoints/rl_token/my_run/rl_token_step5000.pt \
    --env_factory my_package.envs.make_env --task_prompt "stack the three blocks"
```

### 功能 4:RECAP 价值模型

配置:`configs/recap/*.yaml`:

- 价值离散化:`num_bins: 201`、`v_min: -1.0`、`v_max: 0.0`、`normalize_to_minus_one_zero`
- 模型:`critic_expert_variant: gemma_100m`、`siglip_path` / `gemma3_path` / `tokenizer_path`、`freeze_vlm`
- 数据:`train_data_paths` / `eval_data_paths` 列表(每条含 `dataset_path` / `robot_type` / `type` / `weight`)、`camera_map`、`balance_weights`
- 训练:`micro_batch_size` vs `global_batch_size`(梯度累积)、`lr` / `value_lr` 双学习率、`lr_scheduler`

```bash
python scripts/recap/train_value_sft.py --config configs/recap/recap_value_sft.yaml
python scripts/recap/compute_advantages.py --config configs/recap/recap_compute_advantages.yaml --value_checkpoint <path>
```

### 策略服务与 rollout

```bash
# WebSocket 策略服务器(供 robodeploy 连接)
python scripts/serve_policy.py policy:checkpoint \
    --policy.config=configs/bi_s1/pi05_inference.yaml --policy.dir=checkpoints/my_run/20000

# 开环评测
python scripts/rollout_vla.py --vla-config-name configs/bi_s1/pi05_finetune.yaml \
    --vla-checkpoint-dir ... --env-factory ... --num-episodes 10
```

## 关键注意

- **transformers 补丁必装**:`PI0Pytorch` 初始化时 `check_whether_transformers_replace_is_installed_correctly()` 自检
- **归一化统计**:norm stats 优先从 checkpoint `assets/` 加载,失败回退配置路径;`norm_stats.json` 的 q01/q99/std 异常是 loss 发散排查点
- **并行方案**:DDP(torchrun)为主;FSDP(`fsdp_devices > 1`)仅单卡放不下时用
- wandb 默认开启(`wandb_enabled: true`,项目名 `rlt-openpi`)

## 后期计划

- 完善复现的rlt, rtc, recap， robodojo仿真接入等
