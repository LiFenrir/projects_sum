# RoboDojo 仿真接入 rlt 方案

> **状态:方案 B(跨进程)已实现。** 实现与下文设计略有出入,以代码为准:
> - server 复用 `create_eval_env`(monkeypatch `WsModelClient` 为 stub + `_stream_vision` 置空),而非手写 task 类封装
> - 协议新增 `observe` op(不执行动作只取观测,用于补末帧)
> - 序列化为自研 ndarray↔bytes 钩子(msgpack 两边环境均有,无需 msgpack-numpy)
>
> 已实现文件:
> - `src/rlt/rollout/ipc.py` — length-prefix TCP + msgpack(numpy) 编解码,server/client 共用
> - `src/rlt/rollout/robodojo_sim_server.py` — RoboDojo 侧 server(Isaac Sim 环境运行)
> - `src/rlt/rollout/robodojo_env.py` — rlt 侧 `RoboDojoEnv` + `make_robodojo_env` factory
> - `configs/rlt/stage2_online_rl_robodojo.yaml` — 示例配置(`env.extra` 透传 task_name 等)
> - `src/rlt/tests/test_robodojo_env.py` — 6 个单测(fake server,含崩溃重启)
> - `src/rlt/training/config.py` — `EnvConfig.extra: dict` 透传字段;`scripts/train_online_rl.py` 转发

## 目标

在 `src/rlt` 在线 RL 流水线（Stage 2 / rollout 评估）中接入 RoboDojo 仿真（`/home/kemove/INNOV/sim/RoboDojo`，Isaac Sim 后端），支持通过配置指定 task，复用现有 `RolloutWorker` / `OnlineRLTrainer` / `rollout_vla.py` 全链路。

## 关键约束（调研结论）

1. **进程必须隔离**：RoboDojo 依赖 Isaac Sim 专用 Python 环境（自带 torch/omni 全家桶），与 innov_openpi 的 conda 环境（打过 transformers 补丁）不能同进程混跑。→ 跨进程架构。
2. **RoboDojo 原生方向是"sim 当 WS 客户端、policy 当 server"**（eval_client → XPolicyLab），且 `create_eval_env` 强制连 policy server。但 rlt 的 RolloutWorker 是主动驱动方（同步 reset/step 并需要逐步 reward），方向相反。→ 不走 eval_client，直接实例化 task 类，自建"sim 当 server"的薄服务。
3. **奖励可 step 级获取**：`reward_manager.get_reward(final_check=False)` 给 binary success（0/1），`get_score()` 给 0–100 过程分，`is_episode_end()` 给 done + step_lim 超时。稀疏 binary 与现有 `RobotEnv` 的人工 's'/'f' 奖励语义完全一致，TD3 侧零改动。
4. **动作是绝对目标值**：单步 action dict `{left/right_arm_joint_state(6), left/right_ee_joint_state(1)}` 共 14-D（joint 模式；ee 模式 16-D），env 内部从当前状态线性插值执行（`sim.dt=0.004`、`collect_freq=25Hz`，1 个 action ≈ 10 个物理步）。
5. **task 指定机制**：`task/RoboDojo/task_registry.load_task_class(task_name)` 动态导入 `task/RoboDojo/tasks/{task_name}.py`；每个 task 有 `config/{task_name}.yml`；instruction 由 task 类 `gen_instruction()` 生成。`_task.yml` 注册了 37 个 task，绝大多数 embodiment 为 `dual_x5`（双臂 ARX X5）。
6. **Isaac Sim 会崩**（PhysX SIGABRT/SIGSEGV），官方 eval_policy.sh 靠 bash 层重启 + run manifest 恢复。我们的 server 也要带自动重启。

## 架构

```
innov_openpi 进程 (conda: innov_openpi)          RoboDojo 进程 (Isaac Sim python)
┌──────────────────────────────┐               ┌─────────────────────────────┐
│ RolloutWorker                │               │ robodojo_sim_server.py      │
│   └ RoboDojoEnv (client)     │◄── msgpack ──►│   └ task_class(env_cfg,app) │
│       reset()/step(chunk)    │    TCP 请求-应答│       get_obs/take_action   │
└──────────────────────────────┘               │       reward_manager        │
                                               └─────────────────────────────┘
```

- 通信：length-prefix + msgpack（含 numpy 扩展），同步请求-应答。消息仅 3 种：`reset(seed)` / `step(action_chunk[C,14])` / `close`；step 响应带 `obs, rewards[C], done, info{success, score}`。msgpack 两边环境都有（XPolicyLab 已用）。
- server 进程由 `RoboDojoEnv.__init__` 用 `subprocess.Popen` 拉起（可指定 RoboDojo 环境 python 路径、`CUDA_VISIBLE_DEVICES`），等 ready 握手；检测到崩溃（连接断开/进程退出码 99/134/139）自动 respawn 并重新 reset，对齐官方重试策略。

## 改动清单

### 1. 新增 `src/rlt/rollout/robodojo_sim_server.py`（在 RoboDojo 的 python 下运行）

- argparse：`--task_name --env_cfg_type(默认 arx_x5) --port --seed --device_id --step_lim --num_envs 1`。
- 复刻 `src/eval_client/main.py` 的 env_cfg 组装逻辑（OmegaConf: sim/scene/camera/robot/task_env），`task_registry.load_task_class(task_name)` 实例化。
- obs/动作执行逻辑借用 `EvalEnv.get_obs` / `take_action`（去掉 WsModelClient；可将 EvalEnv 相关方法抽成无 client 的 mixin 或直接调底层 `obs_manager` / `robot_manager.control_manager`）。
- 每执行一步 action 后调 `reward_manager.get_reward(final_check=False)` 与 `is_episode_end()` 组装 reward/done。
- 该文件放在 rlt 仓库内，启动时 `PYTHONPATH=$ROBODOJO_ROOT` 即可 import `env`/`task`/`utils`。

### 2. 新增 `src/rlt/rollout/robodojo_env.py`（client + 环境适配）

- `RoboDojoEnv`：实现与 `SimEnv` 相同的 chunk 级接口（`reset()` / `step(action_chunk)` / `action_dim` / `chunk_length`），含 stride 子采样 history（直接参考 `SimEnv.step` 的结构）、`max_episode_chunks` 超时、崩溃自动重启。
- obs 映射（RoboDojo → openpi 格式）：
  - `state` ← `left/right_arm_joint_state + left/right_ee_joint_state` 拼接（14-D）
  - `images` ← 相机映射可配：`cam_head→front`、`cam_left_wrist→left_wrist`、`cam_right_wrist→right_wrist`（默认，与 innov_arm 数据集键一致）
  - `prompt` ← obs 的 `instruction`（task 指定后由 task 类自动生成；`task_prompt` 参数可覆盖）
- action 映射：`[C,14]` → 每步拆成 4 键 dict 发给 server。
- 工厂 `make_robodojo_env(action_dim, chunk_length, task_prompt, max_episode_chunks, **kwargs)`，签名对齐 `make_innov_arm_env`；kwargs/环境变量（`ROBODOJO_*`）双通道：`robodojo_root`、`python`(RoboDojo 环境解释器)、`task_name`、`env_cfg_type`、`seed`、`step_lim`、`reward_mode(sparse|score)`。

### 3. 配置与接线

- 新增 `configs/rlt/stage2_online_rl_robodojo.yaml`：`env.env_factory: rlt.rollout.robodojo_env.make_robodojo_env`，`intervention_factory` 置空（仿真不需要人工干预；reward 来自 reward_manager）。
- `scripts/train_online_rl.py` 目前只转发 4 个固定 kwargs 给 factory → 在 `EnvConfig` 加 `extra: dict` 透传字段（最小改动，同时惠及后续其他 env）；task_name 等也可走 `ROBODOJO_*` 环境变量（与 innov_arm 的 `INNOV_ARM_*` 惯例一致）。
- `rollout_vla.py` 同一 factory 直接可做仿真开环评估，无需改动。

### 4. 测试

- `src/rlt/tests/test_robodojo_env.py`：mock socket server 验证 obs/action 映射、chunk 语义、奖励与 done、崩溃重连逻辑（不依赖 Isaac Sim）。
- 联调冒烟：`stack_blocks` task，VLA warmup 模式跑 1 个 episode 验证全链路。

## 注意事项 / 风险

1. **Embodiment 差异**：RoboDojo 任务是双臂 ARX X5（14-D），与 SFT 数据的 bi_innov_arm_v1（也是 14-D）维度相同但机械臂/相机/动力学不同。直接用 innov_arm 微调的 VLA 大概率失败 → 建议先在 RoboDojo 示范数据（官方提供下载）上做 SFT，再进 Stage 1/2。norm stats 也需对 RoboDojo 数据重算。
2. **GPU 资源**：Isaac Sim 与 VLA 推理同机需 2 张卡（或单卡分时，`CUDA_VISIBLE_DEVICES` 隔离）。
3. **奖励稀疏**：binary success 与真机人工打标语义一致，TD3 可直接训；若收敛困难，后续可加 `reward_mode: score`（get_score 增量做 dense shaping），属于第二步优化。
4. **多环境并行**：首版 num_envs=1 单 server；后续可开多个 server 进程（不同 port/GPU）配合 `async_collection` 扩展。
5. **官方评测路径另议**：若目标是在 RoboDojo leaderboard 上跑分，应走 XPolicyLab policy server 路径（把 `serve_policy.py` 包成 policy 插件），与本方案的 RL 训练接入相互独立，可后续补。

## 实施顺序

1. `robodojo_sim_server.py` + `robodojo_env.py` 最小链路（sparse reward，单 task）
2. mock 单测 + `stack_blocks` 冒烟
3. `EnvConfig.extra` 透传 + 示例 yaml
4. （可选）score dense reward、多 server 并行、XPolicyLab 评测插件
