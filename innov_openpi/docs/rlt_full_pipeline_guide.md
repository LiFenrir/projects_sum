# RLT 全流程操作文档(innov_arm)

## 概述

RLT(RL Token)流水线在预训练 VLA(pi05)基础上,通过三个阶段训练出可在线优化的机器人操作策略:

```
Stage 0: SFT 微调        → 任务演示数据上微调 VLA,得到任务初始策略
Stage 1: RL Token 训练   → 训练信息瓶颈编码器-解码器,把 VLA 内部嵌入压缩为 RL token z_rl
Stage 2: 在线 RL (RLT)   → 冻结 VLA + RL token,训练轻量 Actor-Critic 在线精调策略
```

与论文(RL Token: Bootstrapping Online RL with VLA Models)的机制对应:

| 论文机制 | 本项目实现 |
|---------|-----------|
| RL token 瓶颈压缩(Eq.1/2) | `RLTokenEncoder/Decoder`,Stage 1 训练 |
| 稀疏二元奖励 r_T = 1 | 键盘 `s`(成功)/`f`(失败) |
| 人类干预 + 干预时 ã ← a_human | 本体复合夹爪示教(按 `i`),干预 chunk 参考替换为人类动作 |
| 阶段 RL(关键阶段)/全局 RL | 按 `t` 切换 rl_active;`critical_phase_only` 配置 |
| 自动预测切换时机(分类器) | 切换点数据记录(z_rl + ã 特征 + 标签),供训练切换分类器 |
| 动作块 H=50,RL chunk C=10 | `vla_action_horizon: 50`、`chunk_length: 10` |
| stride=2 子采样(≈25 samples/s) | `obs_stride: 2`,每 chunk 5 个 transition |
| 异步 rollout 与 learning | `async_collection: true`,采集线程 + 训练线程并行 |

## 架构总览

**单机直接控制**:innov_arm 真机与训练运行在同一进程(训练机串口直连机械臂 + USB 相机)。机器人驱动直接调用 robodeploy 库的 innov_arm 驱动类。

```
训练机 (GPU + innov_arm 真机直连, 单进程)
────────────────────────────────────────────────────────
        ┌──────────── 键盘 KeyboardCtrl (cbreak 单实例) ────────────┐
        │  s/f → RobotEnv 奖励信号        t → RolloutWorker 切换 RL  │
        │  i   → BodyTeachingIntervention 本体示教开关              │
        └───────────────────────────┬───────────────────────────────┘
                                    ▼
[采集线程] RolloutWorker (持续收集 episode)
│
│  每个 chunk 边界 (C=10, 50Hz 下 0.2s):
│    obs ──→ VLA (冻结) ──→ a_tilde 参考动作 (H=50)
│    obs ──→ VLA ──→ RLTokenModel (冻结) ──→ z_rl
│    x = cat(z_rl, s_p)                     │
│                                          ▼ rl_active?
│    rl_active=True  → ActorPolicy(x, ã) → a (μ=ã+残差 + 探索噪声)
│    rl_active=False → 直接执行 a_tilde (VLA 参考, 不训练)
│                                          ▼
│    env.step(a) ──→ innov_arm 驱动 (串口, control 模式) 逐步执行
│      ├─ 每 stride=2 步采集中间 obs → 错位窗口组装
│      └─ 示教中 (i 键): set_mode("collect") 重力补偿 → get_action() 关节位置
│                                          ▼
│    ┌─ 落盘 (每 chunk 5 条 ≈25 samples/s) ─┐
│    │  起点0 transition + 错位窗口          │
│    │  <x_2,a_2:C+2>, <x_4,a_4:C+4>, ...   │
│    └──────────────────┬───────────────────┘
│                       ▼
│    ReplayBuffer (线程安全, CPU)      t 键切换时 → switch_data/*.npz (z_rl+ã+标签)
│
[训练线程] TD3 更新 (每 episode, UTD=5)
│    batch = buffer.sample(256)
│    L_Q = MSE(Q1,td) + MSE(Q2,td)      td = Σγᵏrₖ + γᶜ(1-d)·min Q_target(x',a')
│    L_π = -Q(x,μ) + β·‖μ-ã‖²           (2 次 critic / 1 次 actor)
│    Polyak 目标更新 → 在线 actor 权重同步到 ActorPolicy
│
└─ 每 50 episode → checkpoint (actor/critic/buffer) + switch_data 分片
```

### 组件与数据流

| 组件 | 角色 | 输入 → 输出 |
|------|------|-------------|
| `KeyboardCtrl` | 键盘事件分发(线程安全队列) | s/f → `RobotEnv`;t → `RolloutWorker`;i → `InterventionManager` |
| innov_arm 驱动 | 硬件接口(robodeploy 库) | `send_action`/`get_observation`/`set_mode(control↔collect)` |
| `VLAWrapper` | 冻结感知 + 参考采样 | obs → z 嵌入 / a_tilde [H=50, d] |
| `RLTokenModel` | 冻结信息瓶颈 | z → z_rl [2048] |
| `Actor`(在线 + 行为副本) | 策略(残差式,零初始化) | (x, ã) → μ = ã + residual |
| `TwinQCritic` | 双 Q + 目标网络 | (x, a) → Q₁, Q₂ |
| `ReplayBuffer` | 经验池(锁保护) | stride 子采样,每 chunk 5 条 |
| `SwitchPointRecorder` | 切换点数据 | 按 t 时刻 z_rl + ã + 标签 → npz 分片 |

**数据流(一个 chunk 周期)**:

```
obs ─→ VLA 嵌入 z ─→ RLToken ─→ z_rl ┐
obs ─→ VLA 采样 ã_1:50 ──────────────┤→ x = [z_rl, s^p]
                                     ├→ rl_active? → a = ActorPolicy(x, ã) + 噪声
                                     │              → a = ã (VLA 直接控制)
                                     ▼
env.step(a) 逐步执行 ─→ 逐步 obs/动作/奖励/终止 ─→ stride=2 错位窗口
                                     ▼
                          ReplayBuffer (起点0 + 4 错位窗口)
                                     ▼
TD3 更新 (critic 每次 / actor 每 2 次) ─→ Polyak ─→ 权重同步 ActorPolicy
```

**关键设计**:
- **无桥接**:驱动、相机、键盘、训练全部在同一进程,无网络依赖,无中间协议层
- **异步双线程**:采集不等待训练,训练不阻塞采集(`async_collection: true`)
- **键盘单一实例**:cbreak 事件队列统一分发,env/worker/intervention 共享,避免终端模式冲突
- **行为/在线 actor 分离**:采集用行为副本(权重每 episode 同步),训练在线 actor,避免读写竞争

### 按键功能表

训练过程中通过键盘实时交互(无 Enter,即时生效)。所有按键由单一 `KeyboardCtrl` 实例监听,按消费方分发:

| 按键 | 名称 | 消费方 | 触发时机 | 功能说明 |
|------|------|--------|---------|---------|
| `s` | 成功 | `RobotEnv`(逐步循环内每步检测) | 任意时刻 | 当前 episode 成功:奖励 +1(稀疏二元奖励,论文同款),立即终止 episode |
| `f` | 失败 | `RobotEnv`(逐步循环内每步检测) | 任意时刻 | 当前 episode 失败:奖励 0,立即终止 episode |
| `t` | 切换 RL 模式 | `RolloutWorker`(每 chunk 边界) | 任意时刻 | `rl_active` 取反:RL actor 执行 ↔ VLA 参考执行;**同时记录切换点数据**(z_rl + ã + 标签 → switch_data/*.npz) |
| `i` | 本体示教开关 | `BodyTeachingIntervention`(每 chunk 边界 + 示教循环内) | 任意时刻 | 进入/退出重力补偿示教:进入后驱动切到 `collect` 模式,人可拖拽机械臂/复合夹爪;退出恢复 `control` 模式 |

**阶段 RL 模式(t 键使用流程)**:

```
episode 开始 → VLA 自动执行(非关键阶段,数据不入 buffer)
  接近关键阶段 → 按 t 切到 RL(记录切换点 label=1,RL 数据开始入 buffer)
  关键阶段完成 → 按 t 切回 VLA(记录切换点 label=0)
  episode 结束 → 按 s(成功)/ f(失败)
```

**注意事项**:
- `s`/`f` 在机械臂逐步执行间隙检测,按键后当前步立即终止,无需等待 chunk 执行完
- `t`/`i` 在 chunk 边界生效(执行前);示教中再按 `i` 可提前结束示教(不足 chunk 的帧自动补齐)
- 全局 RL 模式(`--no-critical-phase-only`)下按 `t` 同样有效:切到 VLA 执行且不存数据
- 终端 `Ctrl-C` 终止训练(优雅停止:保存检查点、退出采集线程)

## 前置条件

### 硬件

| 组件 | 要求 |
|------|------|
| 机器人 | innov_arm(单臂 `innov_arm_v1` 7 DOF / 双臂 `bi_innov_arm_v1` 14 DOF) |
| 相机 | ≥1 个(推荐 3 个:front / left_wrist / right_wrist,与 SFT 数据 camera_map 一致) |
| GPU | RTX 4090 24GB 或以上(显存 ≈11-14 GB) |
| 串口 | 机械臂串口设备(如 /dev/ttyUSB0,/dev/ttyUSB1) |

### 软件环境

```bash
conda activate openpi        # 或你的 innov_openpi 环境
cd /home/kemove/INNOV/projects/innov_openpi
pip install -e .
# robodeploy 库(提供 innov_arm 驱动,必须可 import)
cd /home/kemove/INNOV/projects/robodeploy && pip install -e .
```

### 模型与数据(检查点目录约定)

```
/home/kemove/INNOV/datasets/innov_arm/
├── innov_0730_0731_3cam/      # 双臂 3 相机 LeRobot 数据集 (14 DOF)
└── innov_0722_backup/         # 纸杯任务数据集

checkpoints/
├── innov_arm_pi05_sft/<exp>/<step>/    # Stage 0 输出 (含 assets/norm_stats)
├── rl_token/<stage1_run>/              # Stage 1 输出
└── online_rl/<run>/                    # Stage 2 输出
```

---

## Stage 0: SFT 微调

```bash
# 1. 计算归一化统计量(训练前必须)
python scripts/compute_norm_stats.py --config configs/innov_arm/pi05_finetune_innov_arm.yaml

# 2. 单卡训练
python scripts/train_pytorch.py \
    --config configs/innov_arm/pi05_finetune_innov_arm.yaml \
    --exp_name innov_arm_sft_001
```

输出:`checkpoints/innov_arm_pi05_sft/innov_arm_sft_001/<step>/`(含 `model.safetensors` + `assets/` norm_stats)。记下最终步数(如 `24000`)。

> 说明:模型 `action_dim: 32` 是 pi05 内部 pad 宽度;真机/数据集实际为 14 维(双臂 6 关节 + 夹爪 × 2),由 `LeRobotOutputs(action_dim=14)` 在推理链中截断。

---

## Stage 1: RL Token 训练

```bash
# 冻结 VLA,仅训练 RL token(L_ro 重构损失)
python scripts/train_rl_token.py \
    --config configs/rlt/stage1_rl_token.yaml \
    --repo-id /home/kemove/INNOV/datasets/innov_arm/innov_0730_0731_3cam \
    --train.checkpoint.vla-config-name configs/innov_arm/pi05_finetune_innov_arm.yaml \
    --train.checkpoint.vla-checkpoint-dir checkpoints/innov_arm_pi05_sft/innov_arm_sft_001/24000 \
    --train.checkpoint.run-name innov_arm_stage1 \
    --train.training.vla-finetune-alpha 0.0
```

可选:**联合微调 VLA**(α 控制,同时优化 L_ro + α·L_vla):

```bash
    --train.training.vla-finetune-alpha 0.5
```

输出:`checkpoints/rl_token/innov_arm_stage1/rl_token_step<step>.pt`(建议用最终步数,如 `rl_token_step5000.pt`)。

---

## Stage 2: 在线 RL 真机操作

### 环境变量(机器人参数通道:环境变量 > 工厂 kwargs)

| 环境变量 | 说明 | 示例 |
|---------|------|------|
| `INNOV_ARM_ROBOT_TYPE` | 单臂/双臂 | `bi_innov_arm_v1`(默认)或 `innov_arm_v1` |
| `INNOV_ARM_LEFT_PORT` / `INNOV_ARM_RIGHT_PORT` | 双臂串口 | `/dev/ttyUSB0` / `/dev/ttyUSB1` |
| `INNOV_ARM_PORT` | 单臂串口 | `/dev/ttyUSB0` |
| `INNOV_ARM_CAMERAS` | 相机 JSON 配置 | 见下 |
| `INNOV_ARM_CONTROL_HZ` | 控制频率(默认 15) | `15` |

相机 JSON(相机 key 必须与 SFT 配置 `camera_map` 一致:`front`/`left_wrist`/`right_wrist`):

```json
{
  "front":       {"type": "realsense", "width": 960, "height": 848, "fps": 30},
  "left_wrist":  {"type": "realsense", "width": 640, "height": 480, "fps": 30},
  "right_wrist": {"type": "realsense", "width": 640, "height": 480, "fps": 30}
}
```

> opencv 相机:`{"type": "opencv", "source": "/dev/video0", "width": 848, "height": 480, "fps": 30}`。

### 启动训练(阶段 RL 模式)

```bash
CUDA_VISIBLE_DEVICES=1 \
INNOV_ARM_ROBOT_TYPE=bi_innov_arm_v1 \
INNOV_ARM_LEFT_PORT=/dev/ttyUSB0 \
INNOV_ARM_RIGHT_PORT=/dev/ttyUSB1 \
INNOV_ARM_CAMERAS='{"front":{"type":"realsense","width":960,"height":848,"fps":30},"left_wrist":{"type":"realsense","width":640,"height":480,"fps":30},"right_wrist":{"type":"realsense","width":640,"height":480,"fps":30}}' \
python scripts/train_online_rl.py \
    --config configs/rlt/stage2_online_rl.yaml \
    --env-factory rlt.rollout.innov_arm_env.make_innov_arm_env \
    --intervention-factory rlt.rollout.innov_arm_env.make_body_teaching_intervention \
    --checkpoint.vla-config-name configs/innov_arm/pi05_finetune_innov_arm.yaml \
    --checkpoint.vla-checkpoint-dir checkpoints/innov_arm_pi05_sft/innov_arm_sft_001/24000 \
    --checkpoint.rl-token-checkpoint checkpoints/rl_token/innov_arm_stage1/rl_token_step5000.pt \
    --action-dim 14 \
    --chunk-length 10 \
    --critical-phase-only \
    --env.task-prompt "Pick up the block in front and place it into the black box in the middle."
```

启动流程:
1. 驱动连接(control 模式)→ 相机连接
2. 加载冻结 VLA + RL token → 创建 Actor/Critic
3. **Warmup**:VLA-only 策略收集 `warmup_steps`(默认 1000)chunk 预填充 buffer,完成后自动保存 `warmup_buffer.pt`
4. 进入异步双线程循环(采集线程持续收集 episode,训练线程每 episode 做 UTD=5 更新)

> 全局 RL 模式:`--no-critical-phase-only`(默认),episode 全程 RL 策略控制;阶段模式 `--critical-phase-only` 时 episode 从 VLA 参考开始,按 `t` 切到 RL。
> 布尔参数使用 tyro flag 风格:`--x` 开启、`--no-x` 关闭。

### 按键操作

> 完整按键功能(消费方、触发时机、注意事项)见架构总览章节的 [按键功能表](#按键功能表)。

**阶段 RL 工作流(推荐)**:
1. `reset` 后 VLA 自动执行(非关键阶段)
2. 接近关键阶段(高精度段)时按 `t` → 切到 RL 策略,RL 数据开始存入 buffer
3. 关键阶段结束按 `t` → 切回 VLA(该时刻也记录为切换点,label=0)
4. episode 结束按 `s`/`f`

### 本体复合夹爪示教

按 `i` 进入示教:驱动切换到**重力补偿模式**(人可自由拖拽机械臂与复合夹爪),示教系统每 `teach_hz`(默认 15Hz)读取关节位置攒满一个 chunk(10 帧);再按 `i` 或攒满自动退出,恢复位置控制。

- 示教 chunk 存入 buffer 时 **参考动作替换为人类动作**(ã ← a_human,论文算法第 11 行),BC 正则把 actor 拉向人类纠正
- 示教数据可用于:在线纠正策略 / 失败后示范正确动作
- 每次示教建议连贯拖拽一个完整动作段;不足 chunk 的帧会自动补齐

### 训练过程详解

**Warmup(同步)**:VLA 参考执行,存 `(x, a_tilde, a_tilde, r, next_x, done)`,无 RL 更新。

**在线循环(异步双线程)**:

```
[采集线程] 循环:
  obs → z_rl = RLToken.encode(VLA.embed(obs))
      → a_tilde = VLA 参考 (H=50)
  rl_active? → a = ActorPolicy(x, a_tilde) + 探索噪声   | 否则执行 a_tilde
  env.step(a) → 逐步执行,每 2 步采集中间 obs(stride 子采样)
  落盘:起点 0 transition + 错位窗口 <x_2, a_2:C+2>, <x_4, ...>(每 chunk 5 条 ≈25 samples/s)

[训练线程] 每 episode:
  for g in range(UTD=5):   # 2 次 critic 更新 / 1 次 actor 更新
    batch = buffer.sample(256)
    td = Σγᵏrₖ + γᶜ(1-d)·min Q_target(x', a')
    L_Q = MSE(Q1,td) + MSE(Q2,td)              # TD3 双 Q + Polyak 目标
    L_π = -Q(x, μ) + β·‖μ - ã‖²                 # β=0.5 BC 正则
  同步在线 actor 权重 → ActorPolicy(行为副本)
```

### 训练输出

```
checkpoints/online_rl/<run_name>/
├── warmup_buffer.pt              (Warmup 完成后)
├── online_rl_ep50.pt             (每 50 episode)
├── ...
└── switch_data/                  (切换点数据,分类器训练用)
    ├── ep_000000.npz
    ├── ep_000001.npz
    └── ...
```

---

## 切换点数据与切换分类器

每次按 `t` 切换时,记录**该时刻的状态**用于训练"是否应切换到 RL"的二分类器(论文:让模型自动预测切换时机,实现测试时自动接管)。

### 数据格式(npz 分片,每 episode 一个文件)

```python
data = np.load("switch_data/ep_000042.npz")
data["features"]   # [N, 2048]   z_rl 抽象特征(VLA 嵌入经 RL token 编码)
data["a_tilde"]    # [N, C*d]    切换时刻的 VLA 参考动作(论文切换 head 输入 z + ã)
data["labels"]     # [N]         1 = 切到 RL(进入关键阶段), 0 = 切回 VLA
data["obs"]        # [N](可选)   原始观测 dict(默认关闭,图像大)
```

- `features` 即 RL 状态 x 的前 `embedding_dim` 维(z_rl),与 Actor 输入一致;`a_tilde` 即 Actor 的参考输入 —— 分类器输入建议 `concat(z_rl, a_tilde)`,与论文切换 head 对齐
- 启用原始观测保存:`--checkpoint.switch-save-raw-obs`
- 保存目录配置:`--checkpoint.switch-data-dir <path>`(默认 `{save_dir}/{run_name}/switch_data`)

### 训练分类器(示例思路)

```python
import numpy as np
from sklearn.linear_model import LogisticRegression

X, y = [], []
for f in sorted(glob.glob("checkpoints/online_rl/<run>/switch_data/ep_*.npz")):
    d = np.load(f)
    X.append(np.concatenate([d["features"], d["a_tilde"]], axis=-1)); y.append(d["labels"])
X, y = np.concatenate(X), np.concatenate(y)
clf = LogisticRegression(max_iter=1000).fit(X, y)
# 部署:推理时刻预测 P(switch_to_rl | z_rl, ã) 超过阈值 → 自动按 t 切换
```

> 数据量小(每 episode 0-2 个切换点)是正常现象;多收集几个训练日后再训练分类器,或使用 z_rl + 状态的低维特征。

---

## 检查点与恢复

### 恢复训练

```bash
# 从 ep50 恢复(跳过 warmup)
... 同上启动命令 ... \
    --checkpoint.resume-checkpoint checkpoints/online_rl/<run_name>/online_rl_ep50.pt

# 输出:
# Skipping warmup — replay buffer already has N transitions
# Loaded checkpoint ... (episode 50, step 5000)
```

### 从预录 warmup 开始

```bash
    --checkpoint.warmup-buffer checkpoints/online_rl/<run_name>/warmup_buffer.pt
```

---

## 显存与硬件

| 组件 | 显存 |
|------|------|
| VLA(PaliGemma 2B + SigLIP + Gemma 300M) | ~5.4 GB(bfloat16,冻结) |
| RLTokenModel | ~0.1 GB(冻结) |
| Actor + TwinQCritic + ActorPolicy | < 0.02 GB(可训练) |
| VLA 推理激活值 | ~5-7 GB |
| **总计** | **≈ 11-14 GB** |

- RTX 4090(24GB)单卡足够;异步模式下采集(推理)与训练(小网络)并行,共享 GPU,无需额外显存
- ReplayBuffer 在 CPU(100k transitions ≈ 580 MB)
- 机械臂串口波特率与驱动默认一致;相机 USB 带宽足够(3 个 640p+ 相机建议 USB3.0)

## 耗时估算

| 阶段 | 时间 |
|------|------|
| 模型加载 | ~30 秒 |
| Warmup 1000 chunk | 20-40 分钟(VLA 推理是瓶颈,每 chunk 2 次 VLA 前向) |
| 在线训练(异步,采集不等待) | 每 episode ≈ 1-3 分钟(含人工重置);提升 1-5 小时数据量 |
| **实际总耗时** | **1-3 天**,建议分多次训练(每次 `--resume-checkpoint` 续跑) |

## 常见问题

### Q: 训练中断后怎么恢复?
```bash
ls -t checkpoints/online_rl/<run_name>/online_rl_ep*.pt | head -1
# 加上 --checkpoint.resume-checkpoint <最新检查点> 重新启动
```

### Q: 串口打不开 / 驱动连接失败?
- 确认串口设备存在:`ls -l /dev/ttyUSB*`;权限问题加当前用户到 `dialout` 组
- 确认 `INNOV_ARM_ROBOT_TYPE` 与串口参数一一对应(双臂需要两个串口)

### Q: 相机没有图像?
- 检查 `INNOV_ARM_CAMERAS` JSON 的相机 key 是否与 SFT 配置 `camera_map` 一致(front/left_wrist/right_wrist)
- realsense 需指定正确的 serial 或名称;opencv 相机确认 `/dev/video*` 存在且未被占用

### Q: 训练很久没有提升怎么办?
- 确认 buffer 有数据(异步模式训练线程只在有新 episode 时更新;看终端 `buffer_size` 是否增长)
- 降低 BC 正则:`--td3.bc-regularizer-beta 0.1`(actor 零初始化,初始完全复现 VLA 参考)
- 增大探索:`--rl-arch.actor-noise-sigma 0.2`
- 检查是否处于阶段模式且忘了按 `t`(RL OFF 期间不存数据)

### Q: 如何评估训练效果?
```bash
# 回放模式:加载 checkpoint,只收集不更新
python scripts/evaluate.py \
    --env-factory rlt.rollout.innov_arm_env.make_innov_arm_env \
    --checkpoint checkpoints/online_rl/<run_name>/online_rl_ep200.pt \
    --num-episodes 10
```

### Q: 切换点数据为空?
- 阶段模式未按过 `t`(全程 VLA 或全程 RL)→ 无切换事件
- 全局模式默认也会记录(按 t 即切换);确认 switch_data 目录下 npz 已生成
- 想要更多切换点:关键阶段边界多按几次 `t`(来回切换)

### Q: 异步模式下想回到同步?
`--no-async-collection`(收集 1 个 episode → 立即更新,适合排查问题)。
