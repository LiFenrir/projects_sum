---
title: "SAM3 + Inpaint-Anything 视频修复复现总结"
description: "SAM3 微调检测手部 + DiffuEraser 视频修复抹除:复现成功。面向新人的上手指南,含完整环境安装与分步复现流程。"
tags: [reproduction, sam3, video-inpainting, diffuseraser, ego2robot]
created: 2026-08-14
---

# SAM3 + Inpaint-Anything 视频修复复现总结

复现对象:`sam3_repo`(SAM3 微调)+ `Inpaint-Anything`(逐帧检测 + DiffuEraser 视频修复)。结论:**成功** —— 微调 SAM3 逐帧 grounding 手/前臂,DiffuEraser 时序一致修复

## 项目介绍

**SAM3**([facebook/sam3](https://github.com/facebookresearch/sam3))— Meta 第三代 Segment Anything 模型,支持文本/point/box prompt 的可提示分割,覆盖图像与视频。本复现用其**图像模型 + 官方 Hydra 微调管线**,在自标注数据上微调"手/前臂"检测。

**Inpaint-Anything**([geekyutao/Inpaint-Anything](https://github.com/geekyutao/Inpaint-Anything))— 图像/视频/3D 场景的目标抹除、填充、替换工具箱,模型栈为"分割(SAM3)+ 跟踪 + 修复(LaMa/DiffuEraser/SD)"可插拔组合。本地 fork 在此基础上扩展了:

- `remove_hands_perframe.py` — 微调 SAM3 逐帧检测 + DiffuEraser 修复的端到端脚本
- `diffueraser_video_inpaint.py` — DiffuEraser 适配器(运行时注入源码,避开依赖冲突)
- `sam3_utils.py` / `sam_segment.py` — SAM3 权重加载与多提示词预测封装

两仓库分工:SAM3 侧负责**检测能力**(微调出认得"手"的模型),Inpaint-Anything 侧负责**抹除执行**(逐帧 mask → 视频修复)。

## 复现目标

- 对象:[facebook/sam3](https://github.com/facebookresearch/sam3) 微调管线 + [Inpaint-Anything](https://github.com/geekyutao/Inpaint-Anything)(DiffuEraser 修复端)
- 任务:第一人称机器人数据中抹除"手/前臂",输出时序一致的 clean plate 视频
- 目标指标:val per-class IoU/Dice 可用(evaluate.py);修复视频 mask 外像素与原图逐位一致、无闪烁

## 前置要求

- GPU:单卡 ≥ 24GB 推荐(DiffuEraser 默认 960 边长 ≈ 20GB 显存;)
- CUDA ≥ 12.6、Python ≥ 3.12(SAM3 的硬性下限)

## 环境搭建

### 1. 目录布局

两个仓库是平级 checkout,`Inpaint-Anything` 内部再挂 ProPainter 源码:

```
projects/
├── sam3_repo/            # SAM3(目录绝不能命名为 sam3,会遮蔽已安装的包)
└── Inpaint-Anything/
    └── propainter/       # ProPainter 源码 checkout(非 pip 包)
```

```bash
cd /home/kemove/INNOV/projects
git clone https://github.com/facebookresearch/sam3.git sam3_repo
git clone <Inpaint-Anything 仓库> Inpaint-Anything
cd Inpaint-Anything
git clone https://github.com/sczhou/ProPainter.git propainter
```

### 2. 创建环境并安装

```bash
# 单一环境即可(SAM3 + Inpaint-Anything + DiffuEraser 推理全在这一个 env)
conda create -n inpaint python=3.12 -y
conda activate inpaint

cd /home/kemove/INNOV/projects/sam3_repo
pip install -e ".[train]"        # 可编辑安装,含训练依赖(Hydra 等)

cd /home/kemove/INNOV/projects/Inpaint-Anything
pip install -r requirements.txt  # LaMa / SD / 视频 / UI 等其余依赖
```

**不要执行** `pip install -r diffueraser/requirements.txt` —— 它固定 torch==2.3.1 / diffusers==0.29.2,会把 SAM3 需要的 torch≥2.7 顶掉。DiffuEraser 源码由适配器(`diffueraser_video_inpaint.py`)在运行时 `sys.path` 注入,直接用主环境的 diffusers。

### 3. 下载权重

```bash
cd /home/kemove/INNOV/projects/Inpaint-Anything

# SAM3 是 HF gated 仓库,先过授权:
#   1) 浏览器打开 https://huggingface.co/facebook/sam3 点 Agree
#   2) https://huggingface.co/settings/tokens 建 read token
#   3) hf auth login(或 export HF_TOKEN=hf_xxx)
python script/download_weights.py                # 默认权重 → ./pretrained_models(sam3.pt、LaMa、ProPainter)
python script/download_weights.py --only diffueraser   # DiffuEraser → diffueraser/weights/
```

- HF 不可达时脚本自动回退 ModelScope 镜像
- 手动下载的 `sam3.pt` 放 `./pretrained_models/sam3.pt`
- **不要**用 `sam3.1_multiplex.pt`:point/box prompt 会静默失效
- ProPainter prior 权重(`ProPainter.pth` 等)缺失时首次运行自动从 GitHub release 下载

### 4. 冒烟验证

```bash
bash script/remove_anything.sh   # 用 example/ 样例跑图像抹除,确认环境 OK
```

## 复现步骤

### Step 0:准备标注数据(Labelme)
```bash
# 安装labelme
pip install labelme

# 启动
labelme
```

用 Labelme 标注目标(手/前臂等多边形),目录按图片组织。**categories 名就是 text prompt**(如 `hand`、`forearm`)。

### Step 1:Labelme → COCO → 划分

```bash
cd /home/kemove/INNOV/projects/sam3_repo
python scripts/convert_labelme_to_sam.py --input <labelme_root> --output <coco_dir>
python scripts/split_dataset.py --input <coco_dir>/annotations.json
# 输出 annotations.json + train/val_annotations.json(按 image_id 80/20,seed=42)
```

### Step 2:SAM3 微调

```bash
# 参考配置 sam3/train/configs/forearm_ft.yaml,改 paths.sam3_pretrained 与数据路径
python sam3/train/train.py -c configs/forearm_ft.yaml --use-cluster 0 --num-gpus 2
```

- 走 Hydra 官方管线(`train.py` → `trainer.py`);**不要用** `sam3/train/sam3_finetune.py`(HF transformers 独立草稿)
- `--use-cluster 0` 本地多进程;`--use-cluster 1` 走 Submitit 集群
- 关键超参:分辨率 1008、bf16 AMP、`lr_vision_backbone: 0.0`(lr=0 冻结 image encoder)、transformer/language lr 2.5e-5、50 epochs
- 产出两类 checkpoint,**推理用 `model_{epoch}.pt`**(每 5 epoch 保存),`checkpoint.pt` 仅续训

### Step 3:评测

```bash
python scripts/evaluate.py --checkpoint <coco_dir>/model_40.pt --data <coco_dir>/val_annotations.json
# 输出 per-class IoU/Dice;注意脚本内 MODEL_CHECKPOINT 有硬编码默认值,按实际 epoch 传参
```

### Step 4:逐帧检测 + DiffuEraser 视频修复

```bash
cd /home/kemove/INNOV/projects/Inpaint-Anything
python remove_hands_perframe.py \
    --input_dir /data/clips \
    --output_dir /data/clips_clean \
    --sam_ckpt /path/to/model_40.pt \
    --targets "hand" "forearm" \
    --text_confidence 0.4 \
    --inpaint_model diffueraser \
    --diffueraser_steps 2 \
    --max_img_size 960 \
    --save_masks --skip_existing
```

设计取舍:检测端用微调**图像模型**逐帧 grounding(不走视频跟踪,精度高、可并行),修复端用**视频扩散模型**保证时序一致。

### Step 5:检查结果

```bash
# 原图/结果逐帧上下拼接,目检 3 个 episode
python remove_hands_perframe.py ... --stack_compare --limit 3
```

- 输出:`<output_dir>/<episode>.mp4` + `_masks/` + `manifest.json`(状态/coverage/耗时,增量写盘)
- 大规模:`--num_shards 8 --shard_index <i>` 分片并行(2 进程/GPU),`--skip_existing` 断点续跑

## 坑点记录

| 问题                    | 根因                                                                     | 解决                                                 |
| --------------------- | ---------------------------------------------------------------------- | -------------------------------------------------- |
| 微调权重加载失败              | `model_{epoch}.pt` 有 `"model"` 包装 + `detector.` 前缀;`checkpoint.pt` 是裸键 | 推理用 `model_{epoch}.pt`,加载逻辑已按格式分发                  |
| point/box prompt 静默失效 | 换成 `sam3.1_multiplex.pt`                                               | 一律用基础 `sam3.pt`                                    |
| `import sam3` 冲突/找不到  | 源码目录命名为 `sam3` 遮蔽已安装包                                                  | 目录必须叫 `sam3_repo`                                  |
| 环境装坏,torch 版本被降级      | 装了 `diffueraser/requirements.txt`(torch 2.3.1)                         | 不装它;DiffuEraser 由适配器运行时注入                          |
| `set_image` 检测异常      | Adapter 要求 PIL,传 ndarray 会被 `shape[-2:]` 取错维度                          | 传 `Image.fromarray(frame)`                         |
| DiffuEraser OOM       | 默认 960 ≈ 20GB 显存                                                       | 降 `--max_img_size 640`(≈12GB);`--fp16`             |
| 修复结果发灰/异常             | SAM3 predictor 遗留全局 bfloat16 autocast                                  | DiffuEraser/ProPainter 推理包 `no_autocast()`(适配器已处理) |
| mask 外像素被改动           | 开了 `--diffueraser_soft_blend`                                          | 默认外层硬切合成可保证 mask 外逐位一致                             |

## 结果与结论

- 微调收敛正常(50 epochs,cosine + warmup;freeze vision backbone 用 lr=0 实现)
- 逐帧检测 + mask 平滑/膨胀 + DiffuEraser 2-step PCM,修复视频时序稳定,mask 外无损
- 已跑通大规模分片并行(2 进程/GPU)与断点续跑

## 经验

- **检测与修复解耦**:单帧 grounding 精度 > 跟踪速度;时序一致性交给视频修复模型负责
- **checkpoint 契约要写清**:同一训练管线产出两种格式(续训 vs 推理),加载端必须按格式分发处理
- **多提示词并集 + 平滑/膨胀**:partially-visible 目标(手出画)用更低 confidence(0.4)+ gap_fill 补漏检帧
- **跨仓库集成本质是数据契约**:mask 格式(0/255、膨胀量、反相时机)、帧对齐、分辨率缩放链路要对齐
