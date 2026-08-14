#!/usr/bin/env python3
"""
SAM3 文本分割评估脚本。

输入: COCO JSON (val_annotations.json) + model checkpoint
输出: per-class IoU/Dice + overall mIoU/mean Dice
"""

import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from pycocotools import mask as mask_utils
from tqdm import tqdm

from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor

# =========================
# 配置
# =========================

VAL_ANN = Path(
    "/home/kemove/INNOV/datasets/sam_datasets_v2/val_annotations.json"
)
DATASET_ROOT = Path("/home/kemove/INNOV/datasets/sam_datasets_v2")
MODEL_CHECKPOINT = Path(
    "/home/kemove/INNOV/datasets/sam3_ft_logs/checkpoints/model_100.pt"
)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CONFIDENCE_THRESHOLD = 0.5


def polygon_to_mask(polygon, height, width):
    """COCO polygon → binary mask (numpy)."""
    if isinstance(polygon, dict):
        # RLE format
        return mask_utils.decode(polygon)
    # polygon format: [[x1,y1,x2,y2,...]]
    rles = mask_utils.frPyObjects(polygon, height, width)
    rle = mask_utils.merge(rles)
    return mask_utils.decode(rle)


def compute_iou(pred, gt):
    """二值 mask 的 IoU."""
    intersection = np.logical_and(pred, gt).sum()
    union = np.logical_or(pred, gt).sum()
    if union == 0:
        return float("nan")
    return intersection / union


def compute_dice(pred, gt):
    """二值 mask 的 Dice."""
    intersection = np.logical_and(pred, gt).sum()
    total = pred.sum() + gt.sum()
    if total == 0:
        return float("nan")
    return 2 * intersection / total


def evaluate():
    # 加载数据
    with open(VAL_ANN, "r") as f:
        coco = json.load(f)

    cat_id_to_name = {c["id"]: c["name"] for c in coco["categories"]}
    categories = sorted(cat_id_to_name.values())

    # 按 image_id 分组 annotations
    img_id_to_anns = defaultdict(list)
    for ann in coco["annotations"]:
        img_id_to_anns[ann["image_id"]].append(ann)

    img_id_to_info = {img["id"]: img for img in coco["images"]}
    image_ids = sorted(img_id_to_info.keys())

    print(f"Images: {len(image_ids)}")
    print(f"Annotations: {len(coco['annotations'])}")
    print(f"Categories: {categories}")

    # 加载模型
    print(f"\nLoading model from {MODEL_CHECKPOINT}...")
    model = build_sam3_image_model(
        bpe_path="/home/kemove/INNOV/projects/sam3_repo/sam3/assets/bpe_simple_vocab_16e6.txt.gz",
        device=DEVICE,
        eval_mode=True,
        enable_segmentation=True,
        checkpoint_path=None,
        load_from_HF=False,
    )
    checkpoint = torch.load(MODEL_CHECKPOINT, map_location="cpu")
    # checkpoint 可能是完整训练状态或纯 model state_dict
    if isinstance(checkpoint, dict):
        state_dict = checkpoint.get("model", checkpoint)
        model.load_state_dict(state_dict, strict=False)
    model.to(DEVICE)
    model.eval()

    processor = Sam3Processor(
        model,
        device=DEVICE,
        confidence_threshold=CONFIDENCE_THRESHOLD,
    )

    # 累积指标
    per_class_intersection = defaultdict(float)
    per_class_union = defaultdict(float)
    per_class_dice_sum = defaultdict(float)
    per_class_count = defaultdict(int)

    for img_id in tqdm(image_ids, desc="Evaluating"):
        img_info = img_id_to_info[img_id]
        anns = img_id_to_anns.get(img_id, [])
        if not anns:
            continue

        # 加载 image
        image_path = DATASET_ROOT / img_info["file_name"]
        image = Image.open(image_path).convert("RGB")
        width, height = image.size

        # 按类别分组 GT annotations
        cat_to_gt_anns = defaultdict(list)
        for ann in anns:
            cat_to_gt_anns[ann["category_id"]].append(ann)

        # 对每个类别分别推理 + 评估
        present_cat_ids = list(cat_to_gt_anns.keys())

        # set_image 一次
        state = processor.set_image(image)

        for cat_id in present_cat_ids:
            text_prompt = cat_id_to_name[cat_id]
            gt_anns_for_cat = cat_to_gt_anns[cat_id]

            # 构建 GT mask (OR 所有该类别实例)
            gt_mask = np.zeros((height, width), dtype=np.uint8)
            for ann in gt_anns_for_cat:
                segm = ann["segmentation"]
                inst_mask = polygon_to_mask(segm, height, width)
                gt_mask = np.logical_or(gt_mask, inst_mask)

            # 推理
            state = processor.reset_all_prompts(state)
            state = processor.set_text_prompt(text_prompt, state)

            pred_masks = state.get("masks")
            if pred_masks is None or pred_masks.numel() == 0:
                # 无预测 → pred 全 0
                pred_mask = np.zeros((height, width), dtype=np.uint8)
            else:
                # OR 所有预测 mask
                pred_mask_np = pred_masks.cpu().numpy()
                if pred_mask_np.ndim == 3:
                    pred_mask_np = pred_mask_np.squeeze(1)
                pred_mask = np.any(pred_mask_np, axis=0).astype(np.uint8)

            iou = compute_iou(pred_mask, gt_mask)
            dice = compute_dice(pred_mask, gt_mask)

            if not np.isnan(iou):
                class_name = cat_id_to_name[cat_id]
                per_class_intersection[class_name] += np.logical_and(
                    pred_mask, gt_mask
                ).sum()
                per_class_union[class_name] += np.logical_or(
                    pred_mask, gt_mask
                ).sum()
                per_class_dice_sum[class_name] += dice
                per_class_count[class_name] += 1

    # 输出结果
    print("\n" + "=" * 60)
    print("Per-class Results")
    print("=" * 60)
    print(f"{'Class':<25} {'IoU':>8} {'Dice':>8} {'Samples':>8}")
    print("-" * 55)

    all_ious = []
    all_dices = []
    for cat_name in categories:
        count = per_class_count.get(cat_name, 0)
        if count > 0:
            iou = per_class_intersection[cat_name] / per_class_union[cat_name]
            dice = per_class_dice_sum[cat_name] / count
        else:
            iou = float("nan")
            dice = float("nan")
        print(f"{cat_name:<25} {iou:8.4f} {dice:8.4f} {count:>8}")
        if not np.isnan(iou):
            all_ious.append(iou)
            all_dices.append(dice)

    print("-" * 55)
    if all_ious:
        print(f"{'Overall':<25} {np.mean(all_ious):8.4f} {np.mean(all_dices):8.4f}")
    print("=" * 60)


if __name__ == "__main__":
    evaluate()
