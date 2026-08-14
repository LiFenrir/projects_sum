#!/usr/bin/env python3
"""
用微调图像模型对抽帧图片做四类分割测试。

输入: 帧图片目录; 输出: mask 叠加图目录(每帧一张,左上角标注检出情况)
"""

from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor

# =========================
# 配置
# =========================

CKPT = "/home/kemove/INNOV/datasets/sam3_ft_logs/checkpoints/model_40.pt"
FRAME_DIR = Path("/home/kemove/INNOV/datasets/front_clean_test/frames_ep0")
OUTPUT_DIR = Path("/home/kemove/INNOV/datasets/front_clean_test/overlay_ep0")

PROMPTS = ["hair", "hand", "forearm", "whole hand and forearm"]
# 每个提示词对应的叠加颜色 (R, G, B)
COLORS = [(255, 60, 60), (60, 120, 255), (255, 210, 40), (60, 220, 60)]

CONFIDENCE_THRESHOLD = 0.5
ALPHA = 0.5  # mask 叠加不透明度


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    model = build_sam3_image_model(
        checkpoint_path=CKPT,
        load_from_HF=False,
        eval_mode=True,
        enable_segmentation=True,
    )
    processor = Sam3Processor(
        model, device="cuda", confidence_threshold=CONFIDENCE_THRESHOLD
    )

    frames = sorted(FRAME_DIR.glob("*.png"))
    print(f"{len(frames)} frames")
    for i, fp in enumerate(frames):
        image = Image.open(fp).convert("RGB")
        base = np.asarray(image).astype(np.float32)
        labels = []
        with torch.autocast("cuda", dtype=torch.bfloat16):
            state = processor.set_image(image)
            for prompt in PROMPTS:
                processor.reset_all_prompts(state)
                state = processor.set_text_prompt(prompt, state)
                masks = state.get("masks")
                if masks is None or masks.numel() == 0:
                    continue
                mask_np = masks.cpu().numpy().reshape(-1, *masks.shape[-2:])
                union = np.any(mask_np, axis=0)
                scores = state["scores"].float().cpu().numpy()
                labels.append(f"{prompt}:{len(scores)}@{scores.max():.2f}")
                color = np.array(COLORS[PROMPTS.index(prompt)], dtype=np.float32)
                base[union] = base[union] * (1 - ALPHA) + color * ALPHA

        out = Image.fromarray(base.astype(np.uint8))
        ImageDraw.Draw(out).text((6, 6), " | ".join(labels) or "none",
                                 fill=(255, 255, 255))
        out.save(OUTPUT_DIR / fp.name)
        if (i + 1) % 50 == 0:
            print(f"{i + 1}/{len(frames)}")

    print(f"Done -> {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
