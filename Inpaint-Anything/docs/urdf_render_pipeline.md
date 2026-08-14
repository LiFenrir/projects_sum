# URDF 渲染修复 Pipeline 交接文档

> 状态：待实施（前置工作已完成）
> 更新日期：2026-08-05

## 1. 背景与目标

**业务场景**：将"人工示教"视频（操作员手抓机械臂进行拖动示教，kinesthetic teaching）转换为"机械臂自主操作"的干净视频，供 Human-to-Robot 数据管线使用。画面中操作员的身体（手、前臂、头发、脸、衣服）必须抹除，机械臂和桌面交互场景必须保留。

**当前痛点**：
1. 操作员全程握持机械臂，被遮挡的连杆像素在任何一帧都不存在 → ProPainter 等视频修复算法无法补全，输出中机械臂残缺/模糊
2. 检测存在两类错误：机械臂被误检为目标（夹爪→hand、臂杆→whole hand and forearm）；操作员的脸/躯干/衣服不在现有四类标注内，漏检

**目标方案**：URDF 纯渲染修复 —— 用逐帧关节角驱动机械臂 URDF 模型渲染，直接合成进被遮挡区域，替代 ProPainter 对机械臂区域的（不可能完成的）修复。

## 2. 已确认的前提条件

- ✅ 数据集带有**逐帧关节角**（遥操作数据，非直接拖动录制）
- ✅ 相机录制时**基本固定**（外参可视为常量，一次标定全程复用）
- ❓ 关节角数据的路径/格式/与视频帧的时间对齐方式 —— **待确认**
- ❓ 机械臂 URDF 文件（含 mesh）—— **待提供**
- ❓ 相机内参（焦距/主点/畸变）—— **待确认是否有标定值**

## 3. 前置工作（已完成）

### 3.1 SAM3 图像模型微调

- 仓库：`/home/kemove/INNOV/projects/sam3_repo`
- 训练配置：`sam3/train/configs/forearm_ft.yaml`，50 epochs，840M 全参数微调，约 2 小时
- 权重：`/home/kemove/INNOV/datasets/sam3_ft_logs/checkpoints/model_{5..50}.pt`
- 验证集指标（model_40，推荐）：bbox AP 0.915 / mask AP 0.918，AP50 0.98
- 检测类别：`hair, hand, forearm, whole hand and forearm`
- 注意：微调权重仅含 `detector.*` 键（1132 个），无 tracker 部分

### 3.2 处理 Pipeline（Inpaint-Anything 仓库）

仓库：`/home/kemove/INNOV/projects/Inpaint-Anything`

| 文件 | 说明 |
|---|---|
| `remove_hands_perframe.py` | **当前主 pipeline**：逐帧检测 + ProPainter 修复 |
| `remove_hands.py` | 旧版：tracker 传播方案（实测覆盖率低，已弃用） |
| `sam_segment.py` | `Sam3PredictorAdapter.predict_texts()` 多提示词接口 |
| `sam3_video_track.py` | 视频模型加载（strict=False，基础 sam3.pt 自带多余的 semantic_seg_head 键） |

**实测对比（episode_000000，1103 帧）**：逐帧版 mask 覆盖率 13.2% vs tracker 版 3.1%，已确定走逐帧路线。

当前运行命令：

```bash
cd /home/kemove/INNOV/projects/Inpaint-Anything && python remove_hands_perframe.py \
  --input_dir /home/kemove/INNOV/datasets/innov_arm/innov_0730_merged/videos/chunk-000/observation.images.front \
  --output_dir /home/kemove/INNOV/datasets/innov_arm/innov_0730_merged/videos/chunk-000/front_clean_perframe \
  --sam_ckpt /home/kemove/INNOV/model/sam3_forearm_checkpoints/model_40.pt \
  --save_masks --fp16 --skip_existing
```

### 3.3 数据位置

- 输入视频：`/home/kemove/INNOV/datasets/innov_arm/innov_0730_merged/videos/chunk-000/observation.images.front/episode_*.mp4`（301 集，每集约 1103 帧）
- 测试产物：`/home/kemove/INNOV/datasets/front_clean_test/`（抽帧、叠加图、对比图）
- 微调训练集：`/home/kemove/INNOV/datasets/sam_datasets_v2/`（labelme→COCO，`scripts/convert_labelme_to_sam.py`）

## 4. 目标 Pipeline 架构

```
每帧输出 = 原图(未 mask 区域)
         + URDF 渲染机械臂(mask ∩ 机械臂区域)      ← 本次新增
         + ProPainter(mask 其余区域:桌面/物体/背景)  ← 现有
```

处理流程：

1. **逐帧检测**（现有）：微调 SAM3 图像模型，多提示词合并出"操作员"二值 mask
2. **机械臂区域判定**（新增）：渲染该帧关节角下的机械臂投影，与检测 mask 求交 → mask ∩ 机械臂投影 区域由渲染像素填充
3. **渲染层**（新增）：
   - 读取该帧关节角 → 正运动学 → URDF mesh 渲染 → 投影到相机视角
   - 外观处理：渲染像素向真实机械臂可见区域做颜色匹配，边缘 feathering/泊松融合，消除 CG 接缝
4. **ProPainter 层**（现有）：mask 剩余区域（被操作员遮挡的桌面、物体、背景）
5. **合成输出**（新增）：叠加顺序 底板=原图 → ProPainter 修复层 → 渲染机械臂层（最上层，机械臂遮挡关系以深度为准时可用深度测试）

### 叠加顺序注意

正确顺序取决于遮挡关系：操作员的手在机械臂**前面**，所以抹除人手后露出的机械臂区域直接填渲染像素即可；机械臂挡住背景的区域，渲染像素也同时覆盖了"本应被机械臂挡住、但 mask 只标了人"的边界——建议用渲染深度做 z-buffer 合成，避免边界穿帮。

## 5. 实施步骤与验收标准

### 步骤 1：渲染对齐验证（最高风险，先行）

**任务**：取一帧机械臂无遮挡/少遮挡的图像 + 该帧关节角，渲染机械臂线框叠加到原图，验证几何对齐。

**需要输入**：关节角数据路径与格式、URDF、相机内参、初始外参（无标定值则手调）。

**外参标定方法**：相机固定 → 外参恒定。选一帧机械臂完整可见的帧，优化/手调 6DoF 外参使渲染投影与图像机械臂重合，全数据集复用。

**验收标准**：静止帧渲染投影与真实机械臂边缘偏差 ≤ 2~3 像素；运动帧无可见时间延迟错位（若有关节角-视频时间戳延迟，需估计并补偿）。

### 步骤 2：渲染质量调优

- 颜色匹配：以每帧真实机械臂可见像素为参照做直方图/均值方差匹配
- 边缘融合：mask 边界 3~5 像素 feathering
- **验收标准**：抽帧肉眼检查，渲染区域与真实画面无明显色差、无边缘断裂感

### 步骤 3：接入 pipeline

在 `remove_hands_perframe.py` 的 `process_episode` 中，检测 mask 之后、ProPainter 之前插入渲染层；渲染覆盖的区域从 ProPainter 的输入 mask 中剔除。

- **验收标准**：episode_000000 输出视频中，被握持的连杆完整、无 ProPainter 涂抹痕迹

### 步骤 4（可并行）：检测标注改造

- 四类合并为单类 `person`（操作员手/脸/头发/衣服），提示词用语义先验好的词
- 补充 hard negative：纯机械臂无人帧标注为空样本
- 从 model_40 继续微调，无需从头训练
- **验收标准**：innov_arm 测试帧上机械臂误检消失，操作员身体各部位检出完整

### 步骤 5：全量批处理

301 集，逐帧检测约 0.5s/帧 + ProPainter，单集约 8~9 分钟，全量约 40+ 小时；`--skip_existing` 支持断点续跑。

## 6. 风险清单

| 风险 | 影响 | 缓解 |
|---|---|---|
| 外参标定不准 | 渲染机械臂与真实位置错位 | 多帧联合优化外参；用机械臂可见部分做逐帧校验 |
| 关节角-视频时间戳有延迟 | 运动帧渲染错位 | 在匀速运动段估计固定延迟补偿 |
| 渲染 CG 质感与真实画面不融合 | 接缝明显 | 颜色匹配 + 泊松融合；下游若为模型训练可接受适度差异 |
| 相机"基本固定"但有微抖 | 外参漂移 | 抽多帧校验；必要时按 episode 分别标定 |
| 检测帧间抖动 | 修复后视频闪烁 | mask 时序平滑（相邻帧 IoU 加权）；或后续视频微调 |

## 7. 后续可选方向

- **视频微调**：单帧检测帧间抖动若不可接受，可做 masklet 标注（YTVIS 格式）微调 tracker。路线：现有 pipeline 输出当预标注 → labelme 修错帧 → 改造 `convert_labelme_to_sam.py` 输出 YTVIS → 从 model_40 续训。数据量：20~50 个 100~300 帧 clip 起步
- **模型精度**：已评估过 bf16→fp32 无实质收益（误检是语义问题非数值问题），不要走这个方向

## 8. 关键文件索引

```
sam3_repo/
  scripts/test_frames.py               # 单帧检测测试(微调模型,四类叠加)
  scripts/visualize_ckpts.py           # 多 checkpoint 对比
  scripts/convert_labelme_to_sam.py    # labelme→COCO 转换
  sam3/train/configs/forearm_ft.yaml   # 微调配置

Inpaint-Anything/
  remove_hands_perframe.py             # 主 pipeline(渲染层将接入此处)
  sam_segment.py                       # predict_texts 多提示词接口
  pretrained_models/sam3.pt            # 基础权重

数据/
  model/sam3_forearm_checkpoints/model_40.pt  # 微调权重(推荐)
  innov_arm/.../observation.images.front/  # 输入视频 301 集
  sam3/forearm_dataset_v2/                 # 前臂微调 COCO 数据
  sam3/forearm_logs/                       # 前臂训练日志
```
