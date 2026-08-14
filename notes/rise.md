---
title: "RISE 复现指南"
description: "OpenDriveLab RISE(组合世界模型 + 想象中 RL 自改进策略)复现:离线策略/价值 → 动力学模型 → 在线 RL 三阶段指南。"
tags: [reproduction, vla, world-model, rl, rise]
created: 2026-08-14
---

# RISE 复现指南

复现对象:`/home/kemove/INNOV/projects/RISE`(OpenDriveLab,[arXiv 2602.11075](https://arxiv.org/abs/2602.11075),RSS 2026)。

## 项目介绍

RISE 把世界模型变成真实操作的学习环境:通过想象中 rollout 引导策略提升,避免真实交互的硬件成本与复位。

- **组合世界模型** — 可控多视角动力学模型 + 进度(progress)价值模型,得到信息量更大的 advantage
- **想象中 RL** — imaginary rollouts 自举策略,无需真实世界交互
- **真实操作收益** — 动态砖块分拣 +35%、背包打包 +45%、合盖 +35%(对比基线)

仓库结构:

```
RISE/
├── policy_and_value/
│   ├── policy_offline_and_value/   # 离线策略 + 价值训练(基于 OpenPI)
│   └── policy_online/              # 在线 RL(基于 RLinf)
├── dynamics/dynamics_model/        # 动作条件动力学模型(LTX-Video 骨干)
├── deploy/                         # Piper 部署与数据采集
└── docs/                           # 各阶段官方文档
```

## 复现目标

- 对象:[OpenDriveLab/RISE](https://arxiv.org/abs/2602.11075)
- 任务:跑通"离线策略 → 价值标注 → 动力学模型 → 想象中在线 RL"全链路
- 目标指标:在线 RL 后任务成功率相对离线策略提升(论文 +35%~45%)

## 前置要求

- GPU:多卡(官方示例 8 卡;在线 RL 需给 env/rollout/actor 分配 GPU)
- CUDA 12.4(torchvision 0.21.0 cu124)、Python 3.11
- 网络:需访问 HuggingFace(LTX-Video、RISE_Assets)

## 环境搭建

### 1. 创建环境并安装

```bash
conda create -n rise python=3.11.14 -y
conda activate rise

cd /home/kemove/INNOV/projects/RISE
bash install.sh
```

`install.sh` 实际做的事(排障时按此拆解):

1. `policy_offline_and_value`:torchvision 0.21.0(cu124 强制重装)→ legacy-resolver 可编辑安装 → 指定 commit 的 lerobot → datasets 3.6.0 / kornia → **拷贝 transformers_replace 补丁**(同 openpi 的 SigLIP 修改)
2. `policy_online`:`pip install rlinf[embodied]`
3. `mini_lerobot`(数据转换用)与 `dynamics` 可编辑安装
4. 最后 `pip install torchcodec==0.2`

### 2. 下载权重

```bash
cd dynamics/dynamics_model
./download.sh    # LTX-Video 骨干(text encoder / tokenizer / VAE)→ checkpoints/
```

- 预训练动力学模型:[OpenDriveLab-org/RISE_Assets](https://huggingface.co/OpenDriveLab-org/RISE_Assets)(Galaxea Open World + AgiBot World Alpha 联合预训练),放入同目录并在配置里改 `pretrained_model_name_or_path`
- 数据/checkpoint 许可为 CC BY-NC-SA 4.0,代码 Apache 2.0

### 3. 冒烟验证

```bash
cd policy_and_value/policy_offline_and_value
python scripts/compute_norm_stats_fast.py --config-name Compute_norm   # 需先备好数据(Step 0)
```

## 复现步骤

### Step 0:数据准备(LeRobot 格式)

训练全链路直接使用 **LeRobot 格式**数据。数据集按标准布局组织:

```text
<dataset_name>/
├── data/chunk-000/episode_*.parquet
├── meta/{info.json, episodes.jsonl, episodes_stats.jsonl, tasks.jsonl}
└── videos/chunk-000/*.mp4
```

- 动力学模型训练要求三视角(1 head + 2 wrist),视频建议预缩到 256×192(见功能 3)
- 现成的 LeRobot 数据集可直接用;自采数据(如经 [[06_Projects/own/robodeploy|robodeploy]] 采集)本身即 LeRobot 格式

### 功能 1:标准 π₀.₅ 离线策略

配置入口:`policy_offline_and_value/src/openpi_value/training/config.py` 的 **`LeRobot_pi05_finetune`**(Python 代码注册式配置,非 yaml)。复现到自己的数据要改的字段:

| 字段 | 说明 |
| --- | --- |
| `data.repo_id` | LeRobot 数据集路径 |
| `data.robot_type` | 机器人类型(如 `innov_arm`) |
| `data.camera_map` | 模型输入相机名 → 数据集键:`front/left_wrist/right_wrist` |
| `data.default_prompt` | 任务指令 |
| `data.assets` | `assets_dir` + `asset_id`,norm stats 输出位置 |
| `pytorch_weight_path` | π₀.₅ base 权重路径 |
| `num_train_steps` / `save_interval` / `keep_period` / `batch_size` | 训练步数与 checkpoint 频率;`batch_size` 为全局 batch |
| `model.advantage_bins: 10` / `apply_blur_visual_aug: true` | π₀.₅ 默认带 advantage 分箱与模糊增强,标准 SFT 不用动 |

```bash
cd policy_and_value/policy_offline_and_value
python scripts/compute_norm_stats_fast.py --config-name Compute_norm   # 先算归一化统计(同步改 Compute_norm 的 repo_id/assets)
bash train.sh LeRobot_pi05_finetune 2          # 用法:train.sh <配置名> <GPU数>;续训加 --resume
```

### 功能 2:带价值模型的离线 π₀.₅

分两步:先训**价值模型**,标注数据后再训 **advantage 条件化策略**。

**① 价值模型(`value_release`)** — 在 π₀.₅ 上加价值头,关键 model 参数:

| 参数 | 值 | 说明 |
| --- | --- | --- |
| `with_value_head` | true | 启用价值头 |
| `loss_action_weight` / `loss_value_weight` | 0 / 1 | 纯价值训练,不训动作 |
| `p_mask_ego_state` | 1.0 | 屏蔽本体状态,价值只看视觉 |
| `p_with_progress_loss` | 1.0 | progress + value 双损失 |
| `value_TD_learning` / `value_TD_TAU` / `value_gamma` | true / 0.01 / 0.995 | TD 学习(目标网络软更新) |
| `value_failure_reward` / `value_terminal_window` | -0.6 / 10 | 失败惩罚与终止窗口 |

数据配置差异:`prompt_from_task: true`,repack 额外读 `fut_1_*`(未来帧,TD target 用)与 `prompt`。

```bash
bash train.sh value_release 8
# 逐帧标注 value/advantage 回数据集(后续 advantage conditioning 的前置)
bash label_value.sh vis_value_release_joint_T /path/to/checkpoints/value_release_joint/<exp>/<step>
bash vis_value.sh vis_value_release_joint_T /path/to/checkpoints/...   # 可视化检查标注质量
```

**② advantage 条件化策略(`Policy_offline_release`)** — 与功能 1 相同配置,两处差异:

- `model.advantage_bins: 10` — advantage 离散化为 10 个 token 拼进策略输入(RECAP 式条件化)
- repack 增加 `"action_advantage": "action_advantage"` — 读 ① 标注的字段

```bash
bash train.sh Policy_offline_release 8
```

### 功能 3:WM + VLA 在线强化

**① 动力学模型微调**(LTX-Video 骨干,动作条件视频预测):

```bash
cd dynamics/dynamics_model
cp -r /path/to/lerobot_dataset dataset/          # 数据集放进 dataset/(三视角:1 head + 2 wrist)
./preprocess.sh <dataset_name>                   # 视频缩到 256×192(ffmpeg 中心 padding)→ videos_small/
python norm.py --datasets <dataset> --save-config data/utils/action_norm.json   # 动作归一化统计
bash train_task_centric.sh
```

`configs/ltx_model/finetune.yaml` 要改的字段:

| 字段 | 说明 |
| --- | --- |
| `pretrained_model_name_or_path` | LTX 骨干目录(download.sh 产物) |
| `diffusion_model.model_path` | 预训练动力学 checkpoint(RISE_Assets 下载) |
| `data.train.data_roots` / `data.val.data_roots` | 数据集路径 |

**② 想象中在线 RL**(基于 RLinf,`policy_online/examples/embodiment/config/rl_release.yaml`):

```bash
bash policy_and_value/policy_online/examples/embodiment/run_embodiment.sh rl_release
```

关键配置修改:

| 段 | 参数 | 说明 |
| --- | --- | --- |
| `cluster` | `component_placement` | env/rollout/actor 的 GPU 分配(如 `env: 0-1, rollout: 0-1, actor: 0-1` 共享;或各独占) |
| `algorithm` | `policy_config_name` | **必须与离线训练配置名严格一致**(如 `Policy_offline_release`) |
| `algorithm` | `num_group_envs: 64` / `n_chunk_steps: 2` | 并行想象环境数 / 每 episode 的 chunk 数 |
| `algorithm` | `adv_type: gae`、`gamma: 0.99`、`gae_lambda: 0.95` | advantage 估计 |
| `algorithm` | `with_advantage_condition: true` | 配合功能 2 的条件化策略 |
| `rollout` | `model_dir` | 功能 1/2 的离线策略 checkpoint(初始化) |
| `actor.model` | `action_dim: 32` / `num_action_chunks: 50` | 动作维度与 chunk 长度 |
| `actor.model.openpi` | `dynamics_model_config` | 指 ① 的 `configs/ltx_model/infer.yaml` |
| `actor.model.openpi` | `reward_model_config: value_release` / `reward_model_ckpt` | 功能 2 的价值模型配置名与 checkpoint |
| `actor.model.openpi` | `advantage_scale: 5` / `wm_action_interval: 2` | advantage 权重 / WM 动作间隔 |
| `actor.model.openpi` | `use_torch_compile: true` | 首次启动 ~10 min 建图;调试可关 |
| `actor.optim` | `lr: 1e-5` / `value_lr: 1e-4` / `clip_grad: 0.2` | 学习率 |
| `runner` | `resume_dir` | 续训指到 `checkpoints/global_step_<N>` |

**③ 部署转换**:在线 RL 产出 Distributed Checkpoint(`.dcp`),部署前转 `.pt`:

```bash
python toolkits/ckpt_convertor/convert_dcp_to_state_dict.py \
    --dcp_path <ckpt_dir> --output_path <输出目录>
```

## 坑点记录

| 问题 | 根因 | 解决 |
|------|------|------|
| 安装报错/依赖解析慢 | install.sh 用 legacy-resolver + 强制重装 torchvision | 不要自己换安装顺序,直接 `bash install.sh` |
| 策略初始化自检失败 | transformers 补丁没拷贝(SigLIP 修改) | install.sh 中的 `cp transformers_replace/*` 步骤不能省 |
| 在线 RL advantage 异常 | 离线/在线配置不一致 | `algorithm.policy_config_name` 与离线训练严格对齐 |
| 价值标注缺失 | 跳过 Step 2 的 label_value | advantage conditioning 必须先标注 |
| 动力学训练慢/爆显存 | 视频未预缩放 | 先 `preprocess.sh` 缩到 256×192 再训 |
| 首次启动卡住 | torch compile 建图 + 模型加载 | 属正常(~10 min);调试关 torch_compile |

## 结果与结论

- 待补充(复现进行中)

## 经验

- **三阶段产物链要理清**:离线策略 checkpoint + 价值模型 checkpoint + 动力学模型 checkpoint,在线 RL 三者都要
- **transformers 补丁是 openpi 系项目共性坑**:RISE 与 innov_openpi 同款,装完务必确认补丁已覆盖
- **数据格式全程 LeRobot**:离线/动力学/在线三阶段共用同一格式,准备一次即可
