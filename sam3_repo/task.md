SAM3 Text Segmentation Fine-tuning Task Specification
0. 项目目标

实现：

Labelme polygon annotations
        ↓
SAM3 text segmentation dataset
        ↓
facebook/sam3 fine-tuning
        ↓
wo1
Input:
    image
    text prompt

Output:
    segmentation mask

类别：

hand
forearm
whole hand and forearm
hair
Task 1：数据转换
修改
scripts/convert_labelme_to_sam.py
输入
raw_dataset/

frame_0001.png
frame_0001.json

frame_0002.png
frame_0002.json
...
输出
sam3_dataset/

images/

masks/

train_annotations.json

val_annotations.json
Label 映射

必须：

{
    "hand":
        "hand",

    "forearm":
        "forearm",

    "hand_and_forearm":
        "whole hand and forearm",

    "hair":
        "hair"
}

禁止：

hand_and_forearm

作为 text prompt。

Annotation 格式

采用：

{
    "image":
        "frame_0001.png",

    "instances":

    [
        {
            "text":
                "hand",

            "mask":
                "masks/frame_0001_000.png"
        }
    ]
}
Task 2：Dataset 实现

修改 sam3/train/data/sam3_text_dataset.py

要求：

参考官方 dataset 实现，不自行设计数据格式。

输出：

{
    image,
    text_prompt,
    mask,
    metadata
}

支持：

一个 image 多个 instance
polygon → binary mask
text prompt supervision
Task 3：Train / Val Split

新增：

scripts/split_dataset.py

规则：

按 image 划分

不是 annotation。

比例：

train:
80%

val:
20%

例如：

300 images

train:
240

val:
60

然后生成：

train_annotations.json

val_annotations.json
Task 4：Fine-tuning 配置

修改： sam3/train/configs/forearm_ft.yaml
Model 配置

checkpoint:

model:
  name: facebook/sam3
第一阶段训练策略
Freeze

冻结：

image encoder

原因：

数据量小。

训练：

text encoder

prompt fusion module

mask decoder

参数：

model:
  freeze_image_encoder: true
Task 5：训练超参数
Hardware

目标：

GPU:
2 x 48GB

训练方式：

DDP

启动：

python train.py \
-c sam3/train/configs/forearm_ft.yaml \
--num-gpus 2
Batch Size

每 GPU：

batch_size = 4

global batch：

8

如果显存允许：

提升：

batch_size=8/GPU

最大：

global batch=16
Image Resolution

保持 SAM3 默认输入：

1024 × 1024

不要降低。

原因：

人体局部：

hand
hair

包含细粒度边界。

Optimizer

AdamW：

optimizer:
  type: AdamW

  weight_decay: 0.05
Learning Rate

不同模块：

Text encoder
1e-5
Fusion module
1e-5
Mask decoder
1e-5
Image encoder
0

冻结。

配置：

optimizer:
  param_groups:

    text_encoder:
      lr: 1e-5

    fusion:
      lr: 1e-5

    mask_decoder:
      lr: 1e-5
Scheduler

采用：

cosine decay

配置：

scheduler:
  type: cosine

  warmup_epochs: 5
Training Epoch

第一阶段：

100 epochs

保存：

每：

5 epochs

checkpoint。

Mixed Precision

开启：

FP16 / BF16

优先：

BF16

如果 GPU 支持。

Gradient

开启：

gradient clipping

参数：

max_norm=1.0
Data Augmentation

开启：

几何
RandomHorizontalFlip
p=0.5
RandomResize
scale:
0.8~1.2
颜色
ColorJitter:

brightness=0.2

contrast=0.2

saturation=0.2

禁止：

large crop

elastic deformation
Task 6：类别采样策略

由于：

hair

可能较少。

启用：

class-balanced sampler

目标：

每个 text prompt 出现概率接近。

类别：

hand
forearm
whole hand and forearm
hair

采样权重：

按 instance 数量 inverse frequency：

公式：

weight = 1 / sqrt(class_count)

不要：

1/class_count

避免过采样。

Task 7：Loss

使用：

Binary Cross Entropy
+
Dice Loss

权重：

loss:

bce_weight:
0.5

dice_weight:
0.5
Task 8：Evaluation

新增：

scripts/evaluate.py

指标：

必须输出：

Overall
mIoU

mean Dice
Per class
hand:

IoU
Dice


forearm:

IoU
Dice


whole hand and forearm:

IoU
Dice


hair:

IoU
Dice
Task 9：第二阶段 Fine-tuning

如果：

mean IoU < 0.85

执行第二阶段。

修改：

解冻：

image encoder
最后4个block

参数：

Image encoder:

lr=1e-6

其他：

lr=5e-6

训练：

30 epochs
Task 10：实验记录

保存：

experiments/

hand_forearm_v1/

config.yaml

checkpoint/

metrics.json

tensorboard/

记录：

train loss
val IoU
val Dice
per-class metrics
最终验收标准

训练流程必须支持：

python train.py \
-c configs/hand_forearm_ft.yaml \
--num-gpus 2

输入：

image:
frame_xxxx.png

text:
"hand"

输出：

mask

验证：

hand IoU

forearm IoU

whole hand and forearm IoU

hair IoU