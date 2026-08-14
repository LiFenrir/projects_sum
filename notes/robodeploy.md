---
title: robodeploy
description: "fork 自 LeRobot 的具身智能部署工具包:砍掉训练与模型库,只保留数据采集与部署,Robot/Teleoperator/Policy 三层可插拔注册机制。"
tags: [project, own, deployment, real-robot, lerobot, openpi]
created: 2026-07-16
---

# robodeploy

仓库:`/home/kemove/INNOV/projects/robodeploy`(Apache-2.0,fork 自 LeRobot)。面向具身智能真机部署场景的轻量化工具包。

## 背景与动机

LeRobot 体系庞大、功能复杂:训练管线、模型库、HF Hub 集成、仿真环境全部耦合在一起,真机部署时只需要其中一小部分,却要背下整套依赖和心智负担。

robodeploy 的思路是**做减法**:fork LeRobot 后砍掉训练与模型库,只保留两件事——**数据采集**和**部署推理**,并针对真机场景重做前端(WebUI/Qt)与策略接入。

## 核心功能

- **多机器人支持** — S1(单/双臂)、Innov Arm、ARX X5 等
- **多模式数据采集** — teleop(纯遥操作)/ policy(纯推理)/ mixed(P 键热切换,DAgger 式纠偏)
- **策略推理接入** — OpenPI WebSocket 客户端,action chunk 时序平滑(server 侧联合部署见 [[innov-openpi#与 robodeploy 联合部署]])
- **相机集成** — OpenCV USB 相机、Intel RealSense 深度相机
- **双前端** — WebUI(FastAPI,远程监控)与 Qt 桌面 UI(相机调试/采集/数据检查/部署四 tab)

## 三层注册机制(draccus.ChoiceRegistry)

三层接口都基于 `draccus.ChoiceRegistry` 的**配置注册**模式:新设备只需"建目录 + 继承基类 + `@register_subclass` 装饰 Config",即自动被 CLI 发现,无需改动任何框架代码。

| 层 | 目录 | 契约 | 实例化 |
| --- | --- | --- | --- |
| **Robot** | `src/robodeploy/robots/` | `get_observation()` → dict、`send_action()`、`connect/disconnect` | `make_robot_from_config(cfg)` |
| **Teleoperator** | `src/robodeploy/teleoperators/` | `get_action()` → dict、`connect/disconnect` | `make_teleoperator_from_config(cfg)` |
| **Policy** | `src/robodeploy/policy_clients/` | `BasePolicy.infer(obs)` → action dict、`reset()` | `make_policy_client_from_config(cfg)` |

### 机器人注册

`robots/` 下每种机器人一个子包(如 `bi_s1_follower/`、`arx_x5/`):

1. Config 继承 `RobotConfig`(本身是 `draccus.ChoiceRegistry`),加 `@RobotConfig.register_subclass("my_robot")`
2. 实现 `Robot` 抽象类:`connect/disconnect`、`get_observation`、`send_action`、标定读写
3. 之后即可 `--robot.type=my_robot` 直接使用

### 遥操作设备注册

`teleoperators/` 同构:Config 继承 `TeleoperatorConfig` + `register_subclass`,实现 `get_action()` 返回目标关节 dict。leader-follower 主从臂(如 `bi_s1_leader`)即为一类 teleoperator。

### Policy 客户端注册

`policy_clients/` 下每个推理后端一个子包(`openpi/`、`lingbot/`、`lerobot_server/`):

1. Config 继承 `PolicyClientConfig`,加 `@PolicyClientConfig.register_subclass("my_policy")`
2. 实现 `PolicyClient`:`infer(obs) → action dict`、`reset()`
3. 之后 `--policy.type=my_policy --policy.host=... --policy.port=...` 接入 mixed/policy 模式

## 环境安装

```bash
# 1. 创建虚拟环境(uv,Python 3.10+)
cd /home/kemove/INNOV/projects/robodeploy
uv venv --python 3.10
source .venv/bin/activate

# 2. 安装(一次性全量,不区分 extra)
uv pip install -e .
```


## 功能使用

### 串口与相机查找

```bash
python -m robodeploy.find_port        # 列出串口设备,插拔一下定位机械臂端口(如 /dev/ttyACM0)
python -m robodeploy.find_cameras     # 枚举可用相机并预览
```

### 数据采集(30fps 控制环)

采集脚本分两个版本:

- **主从臂版 `record_dataset.py`** — S1 等 leader-follower 体系,follower 机器人 + leader 遥操设备(`--robot.type` + `--teleop.type`)
- **复合夹爪版 `record_body_teaching.py`** — Innov Arm / ARX X5 等,本体示教(重力补偿手把手拖动),无独立 teleoperator,机器人自身提供 `get_action()`;支持 RTC(`--use_rtc`)替代 StreamBuffer

```bash
# S1 双臂主从采集(主从臂版)
python -m robodeploy.scripts.record_dataset \
    --robot.type=bi_s1_follower --robot.left_arm_port=/dev/ttyUSB0 --robot.right_arm_port=/dev/ttyUSB1 \
    --teleop.type=bi_s1_leader --teleop.left_arm_port=/dev/ttyUSB2 --teleop.right_arm_port=/dev/ttyUSB3 \
    --control_mode teleop --task="pick and place"

# Innov Arm 双臂本体示教(复合夹爪版,mode=collect,无需主臂)
python -m robodeploy.scripts.record_body_teaching \
    --robot.type=bi_innov_arm_v1 \
    --robot.left_port=/dev/ttyACM0 --robot.right_port=/dev/ttyACM1 --robot.mode=collect \
    --task="pick and place"

# Innov Arm 策略推理部署(mode=control:位置控制)
python -m robodeploy.scripts.record_body_teaching \
    --robot.type=innov_arm_v1 --robot.port=/dev/ttyACM0 --robot.mode=control \
    --policy.type=openpi --policy.host=localhost --policy.port=8000 \
    --task="pick and place"

# mixed 模式:mode=collect + policy 参数,运行中按 P 键热切换(DAgger 式纠偏)
```

每 tick 三阶段:`robot.get_obs()` → 读示教位置(teleop/本体)或 `stream_buffer.pop_next_action()`(policy) → `send_action` + `dataset.add_frame`。

- mixed 模式 policy→teleop 切换时 `interpolate_leader_to_follower()` cosine 混频平滑
- **StreamActionBuffer**(`utils/stream_buffer.py`):重叠 action chunk 线性交叉淡入淡出;调参 `latency_k`、`min_smooth_steps`
- 两版均支持:`--use_rtc`(RTC 收发驱动,`rtc_execution_horizon`)、`warmup_rounds` 推理预热、`action_smooth_max_step` 单步限幅

#### 按键操作(两版相同)

控制指令有两种等价入口:**终端按键**或**前端按钮**(WebUI 网页 / Qt 界面,内部走同一 `pending_ref` 命令通道,主循环同步执行):

| 终端按键 | WebUI/Qt 按钮 | 功能 |
| --- | --- | --- |
| `Enter` | — | 启动控制环 |
| `R` | 录制开关 | 开始/停止录制当前 episode |
| `S` | 保存(带 label) | 结束并保存 episode,随后标注:`1` 成功 / `0` 失败 / `2` 丢弃(写入 `is_failure_data`) |
| `P` / `Tab` | 模式切换 | mixed 模式热切换 示教 ↔ 推理(DAgger 纠偏;固定模式无效) |
| `Z` | 回零 | 机械臂回零位(录制中不可用) |
| `Esc` / `Ctrl+C` | 停止 | 退出 |

episode 达到 `--episode_time_s` 自动停止录制并弹出标注提示。主从臂版 P 切回 teleop 时 leader 余弦插值对齐 follower;复合夹爪版 P 切换时同步切硬件模式(`set_mode("collect")` 重力补偿 ↔ `set_mode("control")` 位置控制)。注意复合夹爪版仅支持键盘 + WebUI,无 Qt 前端。

### DAgger(mixed 模式)

robodeploy 的 DAgger 即采集脚本的 mixed 模式(`--control_mode mixed`):策略自己开,人盯场,跑偏时按 `P` 一键接管纠偏,纠偏段一并录制,聚合重训——训练分布覆盖策略实际访问的状态(含犯错后的恢复轨迹),缓解协变量偏移。

机制要点(逻辑在 `record_loop()`):

- **热切换**:P/Tab 或前端按钮即时切 示教 ↔ 推理,不停录制、不重启进程;初始模式由 `--control_mode_initial` 决定
- **切换平滑**:主从臂版 policy→teleop 时 `interpolate_leader_to_follower()` 把 leader 余弦插值对齐 follower;复合夹爪版直接切硬件 collect/control 模式;两版切换时均清空 StreamActionBuffer/ActionQueue,防止执行残留 chunk
- **逐帧来源标注**:`is_infer_data` 1=策略动作 / 0=人工动作,训练端可筛选加权;配合 `is_failure_data` 保存标注
- **推理与录制耦合**:推理线程仅在 recording=ON 且 mode=POLICY 时请求 server——策略跑的过程本身就是在采集;一次会话同时产出策略成功段、人工纠偏段、失败段,正好是一轮 DAgger 迭代所需的全部数据

```bash
python -m robodeploy.scripts.record_dataset \
    --robot.type=bi_s1_follower --teleop.type=bi_s1_leader \
    --policy.type=openpi --policy.host=localhost --policy.port=8000 \
    --control_mode mixed --control_mode_initial policy \
    --task="pick and place"
```

### 便捷启动脚本(examples/)

`examples/` 按机型把长串 draccus 命令封装成 shell 脚本:`s1/`(record_dataset.sh / record_dataset_rtc.sh / rlt_bridge.sh)、`arx/`(record_arx_bimanual.sh)、`innov/`(record_innov.sh / inference.sh + 归零/主从/关节极值等工具)。日常采集复制对应机型脚本改头部变量即可,不用手敲 `python -m robodeploy.scripts.xxx`。

以 `examples/innov/record_innov.sh` 为例,脚本结构三段式:

**① 快速配置区(文件头部,唯一需要改的部分)**:

```bash
ROBOT_TYPE="bi_innov_arm_v1"
LEFT_PORT="/dev/ttyACM1"  RIGHT_PORT="/dev/ttyACM0"
ROBOT_MODE="control"          # collect=重力补偿示教  control=位置控制(推理时用)
OPENPI_HOST="192.168.200.203" OPENPI_PORT=8000
CAMERA_CONFIG='{"front":{"type":"intelrealsense","serial_number_or_name":"135122077817",...}, ...}'  # 严格 JSON,RealSense 按序列号
REPO_ID="innov/innov_$(date +%m%d_%H%M)"   # 自动带时间戳
TASK="${TASK:-...}"           # 支持环境变量覆盖:TASK="xxx" bash record_innov.sh
USE_RTC=true  RTC_EXECUTION_HORIZON=15  WARMUP_ROUNDS=10
USE_TEMPORAL_SMOOTHING=true  INFERENCE_RATE=5.0  LATENCY_K=16  MIN_SMOOTH_STEPS=8  # 仅非 RTC 生效,附整定注释
```

**② 交互流程**:`sudo chmod 777` 串口授权 → 菜单选控制模式(1 collect 纯示教 / 2 policy 纯推理 / 3 mixed P 键切换)→ 打印完整配置表 → Enter 确认启动。选 collect 时自动裁掉 policy/RTC 参数;RTC 开启时自动关 StreamBuffer(`--use_temporal_smoothing false`),两者互斥。

**③ 最终拼装**:配置区变量翻译成 draccus 参数(`--robot.left_port`、`--policy.host`、`--use_rtc`…)调 `record_body_teaching.py`,并提示当前模式可用按键(mixed 多一个 P)。

### 数据集后端

- **LeRobotDataset** — parquet + 视频(PNG→MP4),HF 兼容,可直接给训练端用
- **LeRobotDatasetNPY** — O(1) RAM 写原始 .npy 帧,episode 保存后子进程异步编码 NPY→MP4,适合长时间采集

### WebUI(远程监控)

```bash
python -m robodeploy.webui            # http://localhost:5000
```

FastAPI 后台线程挂在采集进程里,纯 WebSocket 通信:JSON 控制指令 + 二进制视频帧,浏览器实时看画面与控制采集。

### Qt 桌面 UI

```bash
robodeploy-qt                          # 或 record_dataset --front_end=qt
```

四个 tab:

- **相机调试** — 枚举/预览相机,导出内参
- **数据采集** — 图形化启动/停止录制
- **数据检查** — 浏览/重放/校验已采数据,脚本处理与 v3.0 导出
- **部署** — 策略连接测试 + 一键启动推理

## 后期计划

- 维护升级网页前端和QT前端
