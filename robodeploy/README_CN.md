# robodeploy 中文说明

> 机器人部署与数据采集工具包，基于 LeRobotdataset 数据集，适用于真实机器人硬件控制、遥操作录制和策略推理。

## 功能介绍

- **多机器人支持** — S1（单臂/双臂）、Innov Arm、ARX X5
- **双臂遥操作** — 支持单臂 / 双臂 leader-follower 遥操作
- **多模式数据采集** — 纯遥操作、纯策略推理、混合模式（P 键热切换）
- **摄像头集成** — OpenCV USB 摄像头、Intel RealSense 深度摄像头
- **策略推理** — OpenPI WebSocket 客户端，支持时序平滑与 DAgger 对齐
- **Web 管理界面** — FastAPI WebSocket 实时监控与控制面板

## 项目结构

```
robodeploy/
├── src/robodeploy/         # 核心包：cameras, datasets, motors, robots, teleoperators, webui, scripts
├── scripts/                   # 硬件部署/测试工具（回放、推理检查、RTC 冒烟测试）
├── examples/                  # 使用示例
└── pyproject.toml             # 项目元数据与依赖配置
```

## 快速开始

### 安装

```bash
pip install -e ".[dev,test]"           # 基础 + 开发/测试
pip install -e ".[feetech]"            # Feetech 舵机驱动
pip install -e ".[dynamixel]"          # Dynamixel 舵机驱动
pip install -e ".[intelrealsense]"     # Intel RealSense 摄像头驱动
pip install -e ".[innov]"              # Innov Arm（mujoco 渲染）
pip install -e ".[qt]"                 # Qt 桌面界面（PyQt6）
pip install -e ".[all]"                # 全部
```

### 代码检查与格式化

```bash
ruff check src/                        # 代码检查
ruff format src/                       # 代码格式化
```

(测试套件 `tests/` 已移除,不再有 pytest 命令;dev/test 依赖保留在 pyproject.toml 中备用)

### 启动 Web 界面

```bash
python -m robodeploy.webui           # 默认地址 http://localhost:5000
```

### 数据采集

采集脚本分两个版本，均使用 NPY 存储后端（O(1) 内存），30fps 控制环：

| | 主从臂版 `record_dataset.py` | 复合夹爪版 `record_body_teaching.py` |
|---|---|---|
| 适用机器人 | S1 等 leader-follower 体系 | Innov Arm / ARX X5（本体示教） |
| 示教来源 | 独立 teleoperator（`--teleop.type`，leader 臂） | 无 teleop，机器人自带 `get_action()`，`--robot.mode=collect` 重力补偿下手把手拖动 |
| 录制帧的 action | teleop 模式：leader 臂目标关节位置；policy 模式：推理动作 | collect 模式：本体当前位置（人拖动产生，state/action 同源）；policy 模式：推理动作 |
| 控制模式 | `teleop` / `policy` / `mixed` | `collect` / `policy` / `mixed` |
| P 键切换行为 | policy→teleop 时 `interpolate_leader_to_follower()` 把 leader 余弦插值对齐 follower，平滑接管 | 同时切换硬件模式：`set_mode("collect")` 开重力补偿 / `set_mode("control")` 位置控制 |
| 推理输入预处理 | 夹爪二值化（关节索引 6/13，<0.2→0）；`front_1` 旋转 180° 后与 `front` 上下拼接 | 不二值化；`front_1` 不旋转直接拼接 |
| 前端 | 键盘 / WebUI / Qt（`--front_end=qt`） | 键盘 / WebUI |

两版录制帧均额外写入 `is_infer_data`（该帧动作是否来自策略推理）和 `is_failure_data`（由保存时的成功/失败标注决定）。

```bash
# 主从臂版：双臂 S1 — 纯遥操作
python -m robodeploy.scripts.record_dataset \
    --robot.type=bi_s1_follower \
    --robot.left_arm_port=/dev/ttyUSB0 --robot.right_arm_port=/dev/ttyUSB1 \
    --teleop.type=bi_s1_leader \
    --teleop.left_arm_port=/dev/ttyUSB2 --teleop.right_arm_port=/dev/ttyUSB3 \
    --control_mode teleop \
    --task="pick and place"

# 主从臂版：混合模式 — 策略推理 + 遥操作（P 键切换）
python -m robodeploy.scripts.record_dataset \
    --robot.type=bi_s1_follower \
    --teleop.type=bi_s1_leader \
    --policy.type=openpi \
    --policy.host=localhost --policy.port=8000 \
    --task="pick and place"

# 复合夹爪版：Innov Arm 双臂本体示教（mode=collect，无需主臂）
python -m robodeploy.scripts.record_body_teaching \
    --robot.type=bi_innov_arm_v1 \
    --robot.left_port=/dev/ttyACM0 --robot.right_port=/dev/ttyACM1 \
    --robot.mode=collect \
    --task="pick and place"

# 复合夹爪版：策略推理部署（mode=control 位置控制），支持 --use_rtc
python -m robodeploy.scripts.record_body_teaching \
    --robot.type=innov_arm_v1 --robot.port=/dev/ttyACM0 --robot.mode=control \
    --policy.type=openpi --policy.host=localhost --policy.port=8000 \
    --task="pick and place"
```

#### 按键操作（两版相同）

控制指令有两种等价入口：**终端按键**或**前端按钮**（WebUI 网页 / Qt 界面，内部走同一命令通道，主循环同步执行）：

| 终端按键 | WebUI/Qt 按钮 | 功能 |
|---|---|---|
| `Enter` | — | 启动控制环 |
| `R` | 录制开关 | 开始 / 停止录制当前 episode |
| `S` | 保存（带 label） | 结束并保存 episode，随后标注：`1` 成功 / `0` 失败 / `2` 丢弃 |
| `P` 或 `Tab` | 模式切换 | mixed 模式下热切换 示教 ↔ 推理（DAgger 式纠偏；固定模式下无效） |
| `Z` | 回零 | 机械臂回零位（录制中不可用） |
| `Esc` / `Ctrl+C` | 停止 | 退出 |

episode 达到 `--episode_time_s` 时长后自动停止录制并弹出成功/失败标注提示。注意复合夹爪版仅支持键盘 + WebUI，无 Qt 前端。

详情见 [scripts/README.md](scripts/README.md)。

## 技术栈

| 组件 | 方案 |
|------|------|
| 语言 | Python 3.10+ |
| 机器学习 | PyTorch, HuggingFace datasets & hub |
| 视觉 | OpenCV, Intel RealSense |
| 配置 | draccus（数据类驱动的 CLI 解析） |
| Web 界面 | FastAPI / uvicorn |
| 视频编码 | PyAV (av) |
| 串口通信 | pyserial, Dynamixel SDK, Feetech SDK |
| 策略推理 | OpenPI WebSocket 客户端, msgpack |

## 代码规范

- **双引号** (`"`) 用于所有 Python 字符串（ruff 强制）
- **行宽 110 字符**
- **Google 风格 docstring**
- **Apache 2.0 协议头** 在每个 `.py` 文件
- **`src` 布局** — 导入路径 `from robodeploy import ...`

## 注意事项

- 硬件测试会自动跳过 — `conftest.py` 检测到未连接物理电机/摄像头时跳过
- `torchcodec` 在 Windows / ARM Linux / macOS x86_64 上不可用，此为预期行为
- `[feetech]`、`[dynamixel]`、`[intelrealsense]` 是可选驱动，不在默认安装中
- 顶层 `scripts/` 不属于 Python 包，是独立的硬件部署/测试入口。数据采集与数据集处理脚本位于 `src/robodeploy/scripts/`（通过 `python -m robodeploy.scripts.<module>` 调用）

## 开源协议

Apache 2.0 — 基于 [LeRobot](https://github.com/huggingface/lerobot) 改造。
