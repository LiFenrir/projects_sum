---
name: data-pipeline
description: >
  LeRobot 数据处理与质量检查流水线。触发条件：用户提到处理/合并/清洗/检查
  LeRobot 数据集、bi_s1 数据处理、堆叠摄像头(stack front)、夹爪二值化、
  镜像增强(space mirror)、数据质量检查(视频有效性/帧对齐/index连续性)、
  统计值重算等。覆盖所有 robot_type（bi_s1、S1、SO100、Aloha 等）。
---

# LeRobot 数据处理流水线

## 核心流程

```
合并源数据 → 堆叠摄像头(按需) → 夹爪二值化 → 筛选+镜像增强 → 删除异常 → 检查 → 重算stats
```

## 脚本路径

所有脚本在 `robodeploy/src/robodeploy/scripts/` 下：

| 脚本 | 用途 |
|------|------|
| `merge_lerobot_datasets.py` | 合并多个 LeRobot 数据集 |
| `stack_front_cameras.py` | 堆叠 front+front_1 摄像头（双目前端机器人） |
| `binarize_gripper.py` | 夹爪二值化（阈值 0.2） |
| `filter_lerobot_dataset.py` | 按成功/失败/推理类型筛选 |
| `space_mirroring.py` | 空间镜像增强（翻转+左右互换+合并） |
| `delete_episodes.py` | 删除指定 episode 并重建索引 |
| `regenerate_stats.py` | 重新计算 episodes_stats.jsonl |

检查脚本（bundled）：`.claude/skills/data-pipeline/scripts/inspect_dataset.py`

---

## 1. 数据处理步骤

### 1.1 合并数据集 (`merge_lerobot_datasets.py`)

将多个 LeRobot v2.1 数据集合并为一个，自动处理 episode_index/index/task_index 重映射。

```bash
python robodeploy/src/robodeploy/scripts/merge_lerobot_datasets.py \
    --datasets <ds1> <ds2> ... \
    --output_dir <父目录> \
    --repo_id <输出名称>
```

- `--datasets`：源数据集路径列表
- `--output_dir`：输出目录的父路径（如 `robodeploy/outputs/s1`）
- `--repo_id`：输出目录名称（如 `bi_s1_rollout`）
- `--expert-data`：专家数据（无 is_failure_data/is_infer_data 列时用）
- `--defaults`：JSON 格式的缺失列默认值

**重要**：此脚本要求两个数据集 features 完全一致，否则需要用 `--defaults` 填充缺失列。

### 1.2 堆叠前端摄像头 (`stack_front_cameras.py`)

适用于有 front + front_1 双摄像头布局的机器人（如 bi_s1）。将 front_1 旋转 180° 后垂直堆叠到 front 下方。

```bash
python robodeploy/src/robodeploy/scripts/stack_front_cameras.py \
    --src-path <输入数据集> \
    --tgt-path <输出数据集> \
    --num-workers 8
```

- 输出数据集从 4 视图变为 3 视图：front(960×848)、left_wrist、right_wrist
- Parquet 数据直接复制，不修改
- 使用 ffmpeg libx264 编码（另可选 libsvtav1 的 `--codec`）

**其他机器人**：只有 front 但无 front_1 的单摄像头机器人跳过此步。

### 1.3 夹爪二值化 (`binarize_gripper.py`)

将 action 和 observation.state 中的夹爪维度以 0.2 阈值二值化：≥0.2 → 1（开），<0.2 → 0（闭）。

```bash
python robodeploy/src/robodeploy/scripts/binarize_gripper.py \
    --dataset <数据集路径> \
    --backup-suffix bak  # 可选：修改前备份
```

**注意**：脚本内置 `GRIPPER_INDICES = [6, 13]`（适配 14 维双臂动作：7左+7右，第 6/13 维为夹爪）。不同机器人需修改此数组。

### 1.4 筛选数据 (`filter_lerobot_dataset.py`)

按 is_failure_data 和 is_infer_data 筛选 episode。

```bash
# 只保留成功数据
python robodeploy/src/robodeploy/scripts/filter_lerobot_dataset.py \
    --dataset <源数据集> \
    --is-failure false \
    --output_dir <父目录> \
    --repo_id <名称>

# 只保留纯推理数据
python robodeploy/src/robodeploy/scripts/filter_lerobot_dataset.py \
    --dataset <源数据集> \
    --is-infer true \
    --output_dir <父目录> \
    --repo_id <名称>
```

筛选选项：
- `--is-failure true/false`：按失败/成功筛选
- `--is-infer true/false/mixed`：按纯推理/纯遥操作/DAGGER筛选

### 1.5 镜像增强 (`space_mirroring.py`)

对双臂机器人数据做空间镜像增强：水平翻转视频 + 交换左右臂数据 + 合并。

```bash
# 两步式：先创建镜像，再合并
python robodeploy/src/robodeploy/scripts/space_mirroring.py create-mirror \
    --src-path <源数据集> \
    --tgt-path <镜像输出> \
    --flip-views observation.images.front \
    --swap-left-view observation.images.left_wrist \
    --swap-right-view observation.images.right_wrist \
    --num-workers 8

python robodeploy/src/robodeploy/scripts/merge_lerobot_datasets.py \
    --datasets <原始全量> <镜像成功数据> \
    --output_dir <父目录> \
    --repo_id <最终名称>

# 一步式：创建镜像并合并
python robodeploy/src/robodeploy/scripts/space_mirroring.py full \
    --src-path <源> --mirror-path <镜像> --merge-path <输出目录> \
    --repo-id <名称> \
    --flip-views observation.images.front \
    --swap-left-view observation.images.left_wrist \
    --swap-right-view observation.images.right_wrist
```

**典型流程**：筛选成功数据 → 镜像成功数据 → 合并回原始全量。最终 episode 数 = 原始 + 成功镜像。

**关键参数**：
- `--flip-views`：仅水平翻转的视图（如 front）
- `--swap-left-view`/`--swap-right-view`：需要左右互换且 180°旋转的视图（如 wrist）
- 脚本内置 `swap_arms_in_array()` 会交换左右 7 维臂数据并取反 j1/j5/j6

### 1.6 删除异常 episode (`delete_episodes.py`)

删除指定 episode，自动重建索引、重排文件、更新元数据。操作前会创建时间戳备份。

```bash
python robodeploy/src/robodeploy/scripts/delete_episodes.py \
    <数据集路径> <ep_idx1> <ep_idx2> ...
```

### 1.7 重算统计值 (`regenerate_stats.py`)

重新计算所有 episode 的 action/observation.state 统计值并写入 episodes_stats.jsonl。

```bash
python robodeploy/src/robodeploy/scripts/regenerate_stats.py \
    --dataset <数据集路径>
```

**注意**：每次修改 parquet 数据后（二值化、删除episode、合并）都应重新运行此脚本。

---

## 2. 数据质量检查

### 2.1 检查脚本 (`inspect_dataset.py`)

一站式检查，覆盖四个维度：

```bash
python .claude/skills/data-pipeline/scripts/inspect_dataset.py <数据集路径>

# JSON 输出（便于程序处理）
python .claude/skills/data-pipeline/scripts/inspect_dataset.py <数据集路径> --json

# 自动删除有视频问题的 episode
python .claude/skills/data-pipeline/scripts/inspect_dataset.py <数据集路径> --fix-broken
```

检查维度：

| 维度 | 检查内容 | 通过标准 |
|------|---------|---------|
| **视频有效性** | ffprobe 读取每个 mp4 的 `nb_frames` | 全部可读，无缺失/损坏 |
| **帧对齐** | parquet 行数 vs 视频帧数，各视图间一致性 | delta=0，视图间一致 |
| **index 连续** | 全局 index 从 0 开始连续无断点 | 无 gap |
| **数据分布** | 成功/失败/推理类型统计 | 供人工判断 |

### 2.2 手动检查 index 连续性

如果需要快速验证单个数据集的 index 列：

```python
import pyarrow.parquet as pq
from pathlib import Path

prev = -1
for ep in range(287):  # 替换为实际 episode 数
    idx = pq.read_table(f"data/chunk-000/episode_{ep:06d}.parquet",
                         columns=["index"]).column("index").to_numpy()
    if idx[0] != prev + 1:
        print(f"GAP at ep={ep}: expected {prev+1}, got {idx[0]}")
    prev = idx[-1]
print(f"Last index: {prev}")
```

---

## 3. 典型处理模式

### 模式 A：合并 + 质量检查

适用于已有多个数据集要合并成一个，无需图像处理。

```
merge → inspect → regenerate_stats
```

### 模式 B：全流程（bi_s1 双臂数据）

适用于从原始 bi_s1 数据到最终训练数据集的完整流程。

```
merge → stack_front → binarize_gripper → filter(success) → mirror(success) → merge → delete_broken → inspect → regenerate_stats
```

### 模式 C：镜像增强已有数据集

```
filter(success) → mirror(success) → merge(original + mirrored) → inspect
```

---

## 4. 机器人特定参数

### bi_s1 (Agilex Piper 双臂, 14D action)

- 摄像头：front, front_1, left_wrist, right_wrist（需 stack_front）
- 夹爪索引：[6, 13]
- 臂维度：左7 + 右7
- 镜像取反：j1(0), j5(4), j6(5)，以及对称的 +7 偏移

### 单臂机器人 (SO100, Koch, Aloha 等)

- 不需要 stack_front
- 夹爪索引取决于具体机器人
- 镜像逻辑不同（单臂不交换）

### 适配其他机器人

修改 `binarize_gripper.py` 的 `GRIPPER_INDICES` 和 `space_mirroring.py` 的 `--flip-views`/`--swap-*` 参数即可。

---

## 重要提醒

- **顺序敏感**：先 binarize 再 mirror，否则镜像数据的夹爪值未被二值化
- **先检查再重算**：检查通过后再运行 regenerate_stats，避免基于异常数据计算统计值
- **备份策略**：delete_episodes 会自动备份，其他脚本原地修改前建议手动备份
- **脚本 import 修复**：`regenerate_stats.py` 的 sys.path 应为 `parents[3]` 而非 `parent`（已修复）
- **视频编码**：stack_front 默认输出 libx264，space_mirroring 默认输出 libsvtav1（AV1）。跨脚本处理时注意编码一致性
