#!/usr/bin/env python3
"""
对比 model_25~model_50 共 6 个 checkpoint 的分割效果。

输入: 随机抽取的测试图片 + 文本提示词
输出: 每张图一行、每个 checkpoint 一列的 mask 叠加对比图
"""

import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor

# =========================
# 配置
# =========================

CKPT_DIR = Path("/home/kemove/INNOV/datasets/innov_arm/sam3_ft_logs_v2/checkpoints")
EPOCHS = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50]
IMAGE_DIR = Path("/home/kemove/INNOV/datasets/sam3/innov_labelme_sampled_part2")
OUTPUT_DIR = Path("/home/kemove/INNOV/datasets/innov_arm/sam3_ft_logs_v2/vis_part2_operator")

# 只从该 episode 编号之后抽图
EPISODE_AFTER = 190

PROMPTS = ["operator"]
# 每个提示词对应的叠加颜色 (R, G, B)
COLORS = [(255, 60, 60)]

NUM_IMAGES = 4
RANDOM_SEED = 42
CONFIDENCE_THRESHOLD = 0.5
ALPHA = 0.5  # mask 叠加不透明度


def overlay_masks(image, masks_per_prompt):
    """在原图上叠加各提示词的 mask,返回 RGB numpy 图。

    image: PIL Image; masks_per_prompt: {prompt: (H, W) bool 数组 或 None}
    """
    base = np.asarray(image).astype(np.float32)
    for prompt, mask in masks_per_prompt.items():
        if mask is None or not mask.any():
            continue
        color = np.array(COLORS[PROMPTS.index(prompt)], dtype=np.float32)
        base[mask] = base[mask] * (1 - ALPHA) + color * ALPHA
    return base.astype(np.uint8)


def run_checkpoint(ckpt_path, images):
    """加载一个 checkpoint,对所有图片跑所有提示词。

    返回 {(img_name, prompt): mask 或 None}
    """
    print(f"\n===== Loading {ckpt_path.name} =====")
    model = build_sam3_image_model(
        checkpoint_path=str(ckpt_path),
        load_from_HF=False,
        eval_mode=True,
        enable_segmentation=True,
    )
    processor = Sam3Processor(
        model, device="cuda", confidence_threshold=CONFIDENCE_THRESHOLD
    )

    results = {}
    for img_name, image in images:
        with torch.autocast("cuda", dtype=torch.bfloat16):
            state = processor.set_image(image)
            for prompt in PROMPTS:
                processor.reset_all_prompts(state)
                state = processor.set_text_prompt(prompt, state)
                masks = state.get("masks")
                if masks is None or masks.numel() == 0:
                    results[(img_name, prompt)] = None
                    continue
                mask_np = masks.cpu().numpy()
                # 统一为 (N, H, W) 后对所有实例取并集
                mask_np = mask_np.reshape(-1, *mask_np.shape[-2:])
                results[(img_name, prompt)] = np.any(mask_np, axis=0)
        print(f"  {img_name}: done")
    return results


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # 随机抽图(固定种子保证可复现), 只保留 episode 编号大于 EPISODE_AFTER 的
    all_frames = [
        p
        for p in sorted(IMAGE_DIR.glob("*.png"))
        if int(p.name.split("_")[1]) > EPISODE_AFTER
    ]
    random.seed(RANDOM_SEED)
    picked = random.sample(all_frames, NUM_IMAGES)
    images = [(p.name, Image.open(p).convert("RGB")) for p in picked]
    print(f"Picked images: {[n for n, _ in images]}")

    # 逐 checkpoint 推理
    all_results = {}  # {epoch: {(img_name, prompt): mask}}
    for epoch in EPOCHS:
        ckpt = CKPT_DIR / f"model_{epoch}.pt"
        all_results[epoch] = run_checkpoint(ckpt, images)

    # 拼图: 每行一张测试图, 每列 [原图] + 各 checkpoint
    label_h = 28  # 顶部标签高度
    for img_name, image in images:
        w, h = image.size
        cols = [np.asarray(image)]
        for epoch in EPOCHS:
            masks_per_prompt = {
                p: all_results[epoch][(img_name, p)] for p in PROMPTS
            }
            cols.append(overlay_masks(image, masks_per_prompt))
        row = np.concatenate(
            [np.pad(c, ((label_h, 0), (0, 0), (0, 0)), constant_values=255) for c in cols],
            axis=1,
        )
        out = Image.fromarray(row)
        # 写列标签
        draw = ImageDraw.Draw(out)
        labels = ["original"] + [f"ep{e}" for e in EPOCHS]
        for i, label in enumerate(labels):
            draw.text((i * w + 6, 6), label, fill=(0, 0, 0))
        out_path = OUTPUT_DIR / f"{Path(img_name).stem}_compare.png"
        out.save(out_path)
        print(f"Saved: {out_path}")

    # 图例
    legend = ", ".join(f"{p}={c}" for p, c in zip(PROMPTS, COLORS))
    print(f"\nColors: {legend}")
    print(f"All results in {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
