"""DiffuEraser 视频修复适配器,接口与 propainter_video_inpaint 对齐。

两段式: 先用本仓库的 ProPainter 生成 prior,再由 DiffuEraser 扩散细化。
上游是文件接口(mp4 进 mp4 出),这里替换其模块内的 read_video/read_mask/
read_priori 与 cv2.VideoWriter 实现内存输入输出,不改动上游源码。
"""
import sys
from pathlib import Path
from typing import List

import cv2
import numpy as np
import torch
from PIL import Image
from diffusers.schedulers import TCDScheduler

from propainter_video_inpaint import (
    build_propainter_model,
    inpaint_video_with_builded_propainter,
)
from sam3_utils import no_autocast

DIFFUERASER_DIR = Path(__file__).resolve().parent / "diffueraser"
DEFAULT_DIFFUERASER_WEIGHTS = str(DIFFUERASER_DIR / "weights")

PCM_CKPTS = {2: "2-Step", 4: "4-Step", 8: "8-Step", 16: "16-Step"}


def _import_diffueraser():
    if str(DIFFUERASER_DIR) not in sys.path:
        sys.path.insert(0, str(DIFFUERASER_DIR))
    import diffueraser.diffueraser as de_module
    return de_module


def build_diffueraser_model(weights_dir=None, device="cuda", steps=2,
                            propainter_ckpt=None):
    """加载 DiffuEraser 与 prior 用的 ProPainter。steps 为 PCM 去噪步数。"""
    de_module = _import_diffueraser()
    w = Path(weights_dir or DEFAULT_DIFFUERASER_WEIGHTS)
    ckpt = PCM_CKPTS[steps]
    # loaded 标记跳过其内部相对路径 weights/PCM_Weights 的加载,改用绝对路径
    de = de_module.DiffuEraser(
        device, str(w / "stable-diffusion-v1-5"), str(w / "sd-vae-ft-mse"),
        str(w / "diffuEraser"), ckpt=ckpt, loaded=ckpt + "sd15")
    weight_name = de_module.checkpoints[ckpt][0].format("sd15")
    de.pipeline.load_lora_weights(
        str(w / "PCM_Weights"), weight_name=weight_name, subfolder="sd15")
    de.pipeline.scheduler = TCDScheduler(
        num_train_timesteps=1000, beta_start=0.00085, beta_end=0.012,
        beta_schedule="scaled_linear", timestep_spacing="trailing")

    propainter = build_propainter_model(propainter_ckpt, device)
    return {"diffueraser": de, "propainter": propainter,
            "de_module": de_module}


@torch.no_grad()
def inpaint_video_with_builded_diffueraser(model, frames, masks, **kwargs):
    with no_autocast():
        return _inpaint_video(model, frames, masks, **kwargs)


def _inpaint_video(model, frames: List[Image.Image],
                   masks: List[Image.Image], device="cuda",
                   max_img_size=960, mask_dilation_iter=4,
                   prior_fp16=False, subvideo_length=80,
                   seed=None, feather=0,
                   soft_blend=False) -> List[Image.Image]:
    """frames/masks 为原分辨率 PIL 列表,返回等长同尺寸 PIL 列表。

    feather 为最终合成边界的奇数高斯核大小,0 = 硬切。
    soft_blend=True 时直接输出上游软合成帧,跳过外层合成(mask 外不再是
    逐像素原图)。
    """
    de = model["diffueraser"]
    de_module = model["de_module"]

    n = len(frames)
    pad = max(0, 22 - n)  # 上游要求至少 22 帧,不足时重复末帧,输出裁回
    if pad:
        frames = list(frames) + [frames[-1]] * pad
        masks = list(masks) + [masks[-1]] * pad

    prioris = inpaint_video_with_builded_propainter(
        model["propainter"], frames, masks, device=device,
        fp16=prior_fp16, subvideo_length=subvideo_length)

    fps = 30  # 占位: 上游三个读函数的 fps 一致性检查,替换后无实际作用

    def fake_read_video(_path, _video_length, nframes, max_img_size_):
        proc = [f.convert("RGB") for f in frames]
        w0, h0 = proc[0].size
        if max(w0, h0) > max_img_size_:
            ratio = max(w0, h0) / max_img_size_
            rs = (int(w0 / ratio), int(h0 / ratio))
            img_size = (rs[0] - rs[0] % 8, rs[1] - rs[1] % 8)
        else:
            img_size = (w0 - w0 % 8, h0 - h0 % 8)
        if img_size != (w0, h0):
            proc = de_module.resize_frames(proc, img_size)
        n_total = len(proc)
        n_clip = int(np.ceil(n_total / nframes))
        return proc, fps, img_size, n_clip, n_total

    def fake_read_mask(_path, _fps, n_total_frames, img_size, dil, frames_r):
        out_masks, masked_images = [], []
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        for idx, m in enumerate(masks[:n_total_frames]):
            m = m.convert("L")
            if m.size != img_size:
                m = m.resize(img_size, Image.NEAREST)
            mm = (np.array(m) > 0).astype(np.uint8)
            mm = cv2.erode(mm, kernel, iterations=1)
            mm = cv2.dilate(mm, kernel, iterations=dil)
            mask_img = Image.fromarray(mm * 255)
            out_masks.append(mask_img)
            masked = (np.array(frames_r[idx]).astype(np.float32)
                      * (1 - np.array(mask_img)[:, :, None] / 255.))
            masked_images.append(Image.fromarray(masked.astype(np.uint8)))
        return out_masks, masked_images

    def fake_read_priori(_path, _fps, n_total_frames, img_size):
        out = []
        for p in prioris[:n_total_frames]:
            if p.size != img_size:
                p = p.resize(img_size)
            out.append(p)
        return out

    collected = []

    class FrameCollector:
        def __init__(self, _path, _fourcc, _fps, _size):
            pass

        def write(self, bgr):
            collected.append(Image.fromarray(
                np.ascontiguousarray(bgr[..., ::-1])))

        def release(self):
            pass

    saved = (de_module.read_video, de_module.read_mask,
             de_module.read_priori, cv2.VideoWriter)
    de_module.read_video = fake_read_video
    de_module.read_mask = fake_read_mask
    de_module.read_priori = fake_read_priori
    cv2.VideoWriter = FrameCollector
    try:
        de.forward(None, None, None, "dummy.mp4", max_img_size=max_img_size,
                   video_length=len(frames),
                   mask_dilation_iter=mask_dilation_iter, seed=seed)
    finally:
        (de_module.read_video, de_module.read_mask,
         de_module.read_priori, cv2.VideoWriter) = saved

    # 合成回原分辨率,mask 外像素保持原样
    results = []
    for idx in range(n):
        comp = np.array(collected[idx])
        out_size = frames[idx].size
        if comp.shape[:2][::-1] != out_size:
            comp = cv2.resize(comp, out_size, interpolation=cv2.INTER_CUBIC)
        if soft_blend:
            # 单一融合: 直接用上游软合成结果,跳过外层硬切
            results.append(Image.fromarray(comp))
            continue
        orig = np.array(frames[idx].convert("RGB"))
        m = np.array(masks[idx].convert("L").resize(out_size, Image.NEAREST))
        m = (m > 25).astype(np.uint8)[:, :, None]
        if feather:
            # 羽化边界: 缓解硬切接缝与编码振铃
            a = cv2.GaussianBlur(m[..., 0].astype(np.float32),
                                 (feather, feather), 0)[..., None]
            out = comp * a + orig * (1 - a)
        else:
            out = comp * m + orig * (1 - m)
        results.append(Image.fromarray(out.astype(np.uint8)))
    return results
