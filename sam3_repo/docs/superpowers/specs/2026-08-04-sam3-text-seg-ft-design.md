# SAM3 Text Segmentation Fine-tuning 设计文档

## 概述

Labelme polygon → COCO JSON → Sam3ImageDataset → facebook/sam3 fine-tuning → text→mask 推理

类别: hand, forearm, whole hand and forearm, hair

## 数据流

```
raw_dataset/                    sam3_dataset/                  split
frame_0001.png                  images/                        ├── train_annotations.json (80%)
frame_0001.json                  frame_0001.png                └── val_annotations.json (20%)
frame_0002.png                  frame_0002.png
frame_0002.json        →        masks/
...                              frame_0001_000.png
                                 frame_0001_001.png
                                 ...
                                annotations.json (全量 COCO)
```

## 模块设计

### 1. convert_labelme_to_sam.py — 重构

**输入**: raw_dataset/ (frame_*.png + frame_*.json)
**输出**: COCO JSON format annotations.json + images/ + masks/

COCO JSON 结构:
```json
{
  "images": [{"id": 1, "file_name": "frame_0001.png", "width": 1920, "height": 1080}],
  "annotations": [
    {"id": 1, "image_id": 1, "category_id": 1, "segmentation": [...], "bbox": [x,y,w,h], "area": ..., "iscrowd": 0}
  ],
  "categories": [
    {"id": 1, "name": "hand"},
    {"id": 2, "name": "forearm"},
    {"id": 3, "name": "whole hand and forearm"},
    {"id": 4, "name": "hair"}
  ]
}
```

- polygon points → COCO segmentation (polygon 格式, DecodeRle transform 支持)
- polygon points → COCO bbox (min/max xy → XYWH)
- mask 保存为 PNG 备查（mask/ 目录），COCO segmentation 存 polygon 即可
- label 映射: `hand_and_forearm` → `whole hand and forearm`（禁止 hand_and_forearm 作为 text prompt）

### 2. Dataset — 不修改 sam3_text_dataset.py

直接使用 `Sam3ImageDataset` + COCO JSON，无需自定义 Dataset。
`Sam3ImageDataset` 内部流程:
- `CustomCocoDetectionAPI._load_datapoint()` → 构造 Datapoint(FindQuery, Image, Object)
- category name → text prompt
- 支持 text-only query（无需 bbox input）

### 3. split_dataset.py — 新建

**规则**: 按 image_id 维度划分，非 annotation 维度
**比例**: train 80% / val 20%
**输入**: annotations.json（全量）
**输出**: train_annotations.json / val_annotations.json
**逻辑**: 收集所有 image_id → shuffle → split → 按 split 过滤 annotations

### 4. forearm_ft.yaml — 重写

基于 `roboflow_v100_full_ft_100_images.yaml`，关键配置:

**模型**:
- checkpoint: facebook/sam3（从 HF 下载或本地 sam3.pt）
- freeze_image_encoder: true（lr=0 for backbone.vision_backbone.*）

**训练参数**:
- resolution: 1008（≈1024，SAM3 默认）
- batch_size: 4/GPU, global batch 8（OOM 则 gradient_accumulation_steps=2）
- epochs: 100
- val_epoch_freq: 1
- enable_segmentation: true

**Optimizer**:
- AdamW, weight_decay: 0.05
- text encoder lr: 1e-5
- transformer/decoder lr: 1e-5
- image encoder lr: 0（冻结）

**Scheduler**: cosine decay, warmup 5 epochs

**AMP**: BF16

**Gradient clip**: max_norm=1.0

**Augmentation**:
- RandomHorizontalFlip(p=0.5)
- RandomResize(scale 0.8~1.2)
- ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2)

**Loss**: BCE + Dice（Sam3LossWrapper 内置 loss_mask + loss_dice）

**Checkpoint**:
- model_{epoch}.pt: 仅模型权重（推理用，key 名与 sam3.pt 一致）
- checkpoint.pt: 最新训练状态（断点续训，覆盖旧 epoch）
- save_freq: 5（每 5 epochs 保存 model_{epoch}.pt）

### 5. 类别采样

在 dataset 层用 repeat_factors 实现: weight = 1/sqrt(class_count)

### 6. evaluate.py — 新建

- 加载 model_{epoch}.pt + Sam3Processor
- 逐图推理，text prompt → mask
- 计算 per-class IoU/Dice + overall mIoU/mean Dice
- 按类别分别报告: hand, forearm, whole hand and forearm, hair

## 实现顺序

1. convert_labelme_to_sam.py — COCO JSON 输出
2. split_dataset.py — train/val split
3. forearm_ft.yaml — 训练配置
4. evaluate.py — 评估脚本
5. trainer.py checkpoint 拆分 — model_{epoch}.pt / checkpoint.pt
