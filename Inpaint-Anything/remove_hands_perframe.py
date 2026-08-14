"""逐帧检测的手部抹除:不经过 SAM3 视频跟踪,每帧用微调图像模型独立检测。

流程: 读帧 -> 逐帧 predict_texts 多提示词合并 mask -> 修复 -> 编码输出。
检测原理是单帧 grounding,与视频模型的 n 帧预测 n+t 不同,
因此直接使用微调的图像模型,tracker 不参与。
修复默认用 ProPainter(视频模型),--inpaint_model diffueraser 切换为
DiffuEraser 视频扩散(质量/时序最优,自带 ProPainter prior),flux/lama/sdxl
为逐帧图像修复。

复用 remove_hands 的 episode 发现、帧读取、mask 平滑、修复与写出函数。

Example:
    python remove_hands_perframe.py \
        --input_dir /data/clips \
        --output_dir /data/clips_clean \
        --sam_ckpt /path/to/model_40.pt \
        --save_masks
"""
import argparse
import json
import random
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from propainter_video_inpaint import (
    DEFAULT_PROPAINTER_CKPT_DIR,
    build_propainter_model,
)
from diffueraser_video_inpaint import (
    DEFAULT_DIFFUERASER_WEIGHTS,
    build_diffueraser_model,
    inpaint_video_with_builded_diffueraser,
)
from lama_inpaint import build_lama_model, inpaint_img_with_builded_lama
from remove_hands import (
    DEFAULT_TARGETS,
    TARGETS_WITH_OBJECT,
    Episode,
    _inpaint_chunked,
    _read_frames,
    _smooth_masks,
    _write_masks,
    _write_output,
    discover_episodes,
)
from sam3_utils import DEFAULT_SAM3_CKPT, build_sam3_image_model
from sam_segment import Sam3PredictorAdapter
from stable_diffusion_inpaint import build_sd_inpaint_pipe, fill_img_with_sd
from utils import dilate_mask


def detect_masks_perframe(segmentor, frames: list, targets, confidence: float,
                          dilate_kernel: int, gap_fill: int) -> list:
    """逐帧检测并合并多提示词 mask,返回与 frames 等长的 uint8 mask 列表。"""
    raw_masks = []
    for i, frame in enumerate(frames):
        segmentor.set_image(np.array(frame))
        try:
            masks, _, _, _ = segmentor.predict_texts(
                targets, confidence_threshold=confidence)
        finally:
            segmentor.reset_image()
        union = (np.any(masks, axis=0) if len(masks)
                 else np.zeros((frame.height, frame.width), dtype=bool))
        raw_masks.append(union)
        if (i + 1) % 100 == 0:
            print(f"    detect {i + 1}/{len(frames)}", flush=True)

    raw_masks = _smooth_masks(raw_masks, gap_fill)

    pil_masks = []
    for m in raw_masks:
        m8 = m.astype(np.uint8)
        if dilate_kernel:
            m8 = dilate_mask(m8, dilate_kernel)
        pil_masks.append(Image.fromarray(np.uint8(m8 * 255)))
    return pil_masks


def _inpaint_perframe(painter, frames, masks, args) -> list:
    """逐帧图像修复,返回与 frames 等长的 PIL 列表。painter 由 inpaint_model 决定。"""
    out = []
    for i, (frame, mask) in enumerate(zip(frames, masks)):
        img = np.array(frame)
        m = np.array(mask)
        if args.inpaint_model == "lama":
            filled = inpaint_img_with_builded_lama(
                painter, img, m, device=args.device)
        else:
            filled = fill_img_with_sd(
                img, m, args.remove_prompt, device=args.device,
                model_id=args.inpaint_model)
        out.append(Image.fromarray(filled))
        if (i + 1) % 10 == 0 or i + 1 == len(frames):
            print(f"    inpaint {i + 1}/{len(frames)}", flush=True)
    return out


def process_episode(models, ep: Episode, args) -> dict:
    """抹除一个 episode 的目标,返回 manifest 条目。"""
    started = time.time()
    out_dir = Path(args.output_dir)
    dst = out_dir / (ep.name + (".mp4" if ep.is_video else ""))
    mask_dst = Path(args.mask_dir or (out_dir / "_masks")) / ep.name

    entry: dict = {"name": ep.name, "source": str(ep.source),
                   "output": str(dst)}

    if args.skip_existing and dst.exists():
        entry.update(status="skipped")
        return entry

    frames = _read_frames(ep)
    if args.start_frame or args.max_frames:
        end = args.start_frame + (args.max_frames or len(frames))
        frames = frames[args.start_frame:end]
    entry["frames"] = len(frames)
    if not frames:
        entry.update(status="empty")
        return entry

    pil_masks = detect_masks_perframe(
        models["segmentor"], frames, args.targets, args.text_confidence,
        args.dilate_kernel_size, args.mask_gap_fill)

    coverage = float(np.mean([np.mean(np.array(m) > 0) for m in pil_masks]))
    entry["mask_coverage"] = round(coverage, 5)
    if coverage < args.min_coverage:
        entry.update(status="low_coverage")
        if not args.keep_low_coverage:
            return entry

    if args.inpaint_model == "propainter":
        out_frames = _inpaint_chunked(models["painter"], frames, pil_masks,
                                      args)
    elif args.inpaint_model == "diffueraser":
        def _diffueraser_fn(fr, mk):
            return inpaint_video_with_builded_diffueraser(
                models["painter"], fr, mk, device=args.device,
                max_img_size=args.max_img_size,
                mask_dilation_iter=args.diffueraser_mask_dilation,
                prior_fp16=args.fp16,
                subvideo_length=args.subvideo_length,
                seed=args.seed, feather=args.diffueraser_feather,
                soft_blend=args.diffueraser_soft_blend)
        out_frames = _inpaint_chunked(models["painter"], frames, pil_masks,
                                      args, inpaint_fn=_diffueraser_fn)
    else:
        out_frames = _inpaint_perframe(models["painter"], frames, pil_masks,
                                       args)

    if args.stack_compare:
        # 原图在上、修复结果在下,逐帧垂直拼接
        out_frames = [
            Image.fromarray(
                np.concatenate([np.array(orig), np.array(out)], axis=0))
            for orig, out in zip(frames, out_frames)
        ]

    _write_output(out_frames, ep, dst, args)
    if args.save_masks:
        _write_masks(pil_masks, mask_dst, args.mask_format)
        entry["masks"] = str(mask_dst)

    entry.update(status="ok", seconds=round(time.time() - started, 1))
    return entry


def build_models(args):
    # 微调图像模型: 只做单帧文本 grounding,不需要 tracker
    model = build_sam3_image_model(args.sam_ckpt, device=args.device,
                                   enable_inst_interactivity=False)
    segmentor = Sam3PredictorAdapter(model, device=args.device)
    if args.inpaint_model == "propainter":
        painter = build_propainter_model(args.propainter_ckpt,
                                         device=args.device)
    elif args.inpaint_model == "diffueraser":
        painter = build_diffueraser_model(
            args.diffueraser_weights, device=args.device,
            steps=args.diffueraser_steps,
            propainter_ckpt=args.propainter_ckpt)
    elif args.inpaint_model == "lama":
        painter = build_lama_model(args.lama_config, args.lama_ckpt,
                                   device=args.device)
    else:
        painter = build_sd_inpaint_pipe(args.inpaint_model,
                                        device=args.device)
    return {"segmentor": segmentor, "painter": painter}


def setup_args(parser):
    parser.add_argument(
        "--input_dir", type=str, required=True,
        help="Dataset root, a single episode folder, or a single video file.",
    )
    parser.add_argument(
        "--output_dir", type=str, required=True,
        help="Where to write clean plates, mirroring the input layout.",
    )
    parser.add_argument(
        "--mode", type=str, default="keep_object",
        choices=["keep_object", "remove_object"],
        help="'keep_object' erases only the hands and forearms, leaving the "
             "manipulated object in place. 'remove_object' erases the held "
             "object too. Default: keep_object",
    )
    parser.add_argument(
        "--targets", type=str, nargs="+", default=None,
        help=f"Text prompts detected per frame; masks are merged. Defaults "
             f"to {DEFAULT_TARGETS!r} ({TARGETS_WITH_OBJECT!r} in "
             f"remove_object mode).",
    )
    parser.add_argument(
        "--text_confidence", type=float, default=0.4,
        help="Detection threshold. Default: 0.4",
    )
    parser.add_argument(
        "--dilate_kernel_size", type=int, default=12,
        help="Grow the mask to swallow contact shadows and soft edges.",
    )
    parser.add_argument(
        "--mask_gap_fill", type=int, default=2,
        help="Fill single-frame detection dropouts from neighbours within "
             "this radius. 0 disables.",
    )
    parser.add_argument(
        "--min_coverage", type=float, default=0.0005,
        help="Flag episodes whose average mask covers less than this fraction "
             "of the frame; usually means detection failed.",
    )
    parser.add_argument(
        "--keep_low_coverage", action="store_true",
        help="Process low-coverage episodes anyway instead of flagging only.",
    )
    parser.add_argument(
        "--save_masks", action="store_true",
        help="Export the masks alongside the clean plates.",
    )
    parser.add_argument(
        "--mask_dir", type=str, default=None,
        help="Where masks go. Default: <output_dir>/_masks",
    )
    parser.add_argument(
        "--mask_format", type=str, default="png", choices=["png", "npz"],
        help="Lossless per-frame PNGs, or one compressed npz per episode.",
    )
    parser.add_argument(
        "--chunk_size", type=int, default=300,
        help="Inpaint in chunks of this many frames to bound GPU memory. "
             "0 processes the episode in one pass. Default: 300",
    )
    parser.add_argument(
        "--chunk_overlap", type=int, default=10,
        help="Warm-up frames shared between chunks.",
    )
    parser.add_argument(
        "--subvideo_length", type=int, default=80,
        help="ProPainter's internal sub-video length.",
    )
    parser.add_argument(
        "--fp16", action="store_true",
        help="Half precision for ProPainter.",
    )
    parser.add_argument(
        "--output_format", type=str, default="same",
        choices=["same", "png", "jpg"],
        help="Frame output format for frame-folder episodes. Default: same.",
    )
    parser.add_argument("--jpeg_quality", type=int, default=95,
                        help="Quality for JPEG frame output.")
    parser.add_argument("--fps", type=int, default=30,
                        help="Fallback FPS when the source has no metadata.")
    parser.add_argument("--video_quality", type=int, default=8,
                        help="imageio quality for mp4 output (0-10).")
    parser.add_argument(
        "--skip_existing", action="store_true",
        help="Resume: skip episodes whose output already exists.",
    )
    parser.add_argument("--limit", type=int, default=0,
                        help="Only process the first N episodes. 0 = all.")
    parser.add_argument("--start_frame", type=int, default=0,
                        help="Start from this frame index within each episode.")
    parser.add_argument("--max_frames", type=int, default=0,
                        help="Process at most N frames per episode. 0 = all.")
    parser.add_argument(
        "--sample", type=int, default=0,
        help="Randomly process N episodes. Fixed --seed keeps the selection "
             "identical across runs (e.g. when comparing checkpoints). 0 = all.")
    parser.add_argument(
        "--num_shards", type=int, default=1,
        help="Split the episode list into N disjoint shards so multiple "
             "processes (e.g. 2 per GPU) can run in parallel. Default: 1.",
    )
    parser.add_argument(
        "--shard_index", type=int, default=0,
        help="Which shard this process handles (0-based). Sharded runs write "
             "manifest_shard<index>.json instead of manifest.json.",
    )
    parser.add_argument(
        "--shard_skip", type=int, default=0,
        help="Skip the first N episodes of this shard's list, e.g. to take "
             "only its tail while another process works the front. Default: 0.",
    )
    parser.add_argument(
        "--manifest_tag", type=str, default="",
        help="Extra suffix for the manifest filename so a helper process "
             "doesn't overwrite the main shard's manifest.",
    )
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for --sample.")
    parser.add_argument(
        "--stack_compare", action="store_true",
        help="Write each output frame as the original (top) stacked above the "
             "inpainted result (bottom) for side-by-side inspection.")
    parser.add_argument(
        "--exclude", nargs="+",
        default=["_masks", "_plates", "comparison"],
        help="Directory names to skip during discovery. Names starting with "
             "'_' are always skipped.")
    parser.add_argument(
        "--sam_ckpt", type=str, default=DEFAULT_SAM3_CKPT,
        help="微调图像模型 checkpoint(仅图像检测,tracker 不参与)。",
    )
    parser.add_argument(
        "--propainter_ckpt", type=str, default=DEFAULT_PROPAINTER_CKPT_DIR,
        help="Directory holding the ProPainter checkpoints.",
    )
    parser.add_argument(
        "--inpaint_model", type=str, default="propainter",
        help="What fills the hole: 'propainter' (video model, the default), "
             "'diffueraser' (video diffusion, best temporal+quality, runs "
             "ProPainter as its prior), 'flux' (FLUX.1-Fill per-frame, "
             "~20GB VRAM), 'lama' (fast per-frame), 'sdxl', or any "
             "diffusers inpainting model id.",
    )
    parser.add_argument(
        "--diffueraser_weights", type=str,
        default=DEFAULT_DIFFUERASER_WEIGHTS,
        help="DiffuEraser weights root (diffuEraser/, stable-diffusion-v1-5/, "
             "PCM_Weights/, sd-vae-ft-mse/ inside).",
    )
    parser.add_argument(
        "--diffueraser_steps", type=int, default=2,
        choices=[2, 4, 8, 16],
        help="PCM denoising steps for DiffuEraser. More is slower and "
             "slightly better. Default: 2",
    )
    parser.add_argument(
        "--max_img_size", type=int, default=960,
        help="DiffuEraser processes at this max side length and the result "
             "is upscaled back. 960 needs ~20GB, 1280 ~33GB, 640 ~12GB.",
    )
    parser.add_argument(
        "--diffueraser_mask_dilation", type=int, default=4,
        help="Extra mask dilation inside DiffuEraser (3x3 iterations), on "
             "top of --dilate_kernel_size. Default: 4",
    )
    parser.add_argument(
        "--diffueraser_feather", type=int, default=0,
        help="Odd Gaussian kernel size for feathering the final composite "
             "edge (softens the hard seam and its encoder ringing). "
             "0 = hard cut. Default: 0",
    )
    parser.add_argument(
        "--diffueraser_soft_blend", action="store_true",
        help="Use DiffuEraser's own soft-blended output as-is (single "
             "fusion, ~10px soft band beyond the mask); skips our outer "
             "composite, so pixels outside the mask are no longer "
             "bit-identical to the source.",
    )
    parser.add_argument(
        "--remove_prompt", type=str, default="",
        help="Prompt used by diffusion inpainters. Empty means 'just "
             "continue the background'. Ignored by propainter/lama.",
    )
    parser.add_argument(
        "--lama_config", type=str,
        default="./lama/configs/prediction/default.yaml",
        help="LaMa config, only used with --inpaint_model lama.",
    )
    parser.add_argument(
        "--lama_ckpt", type=str, default="./pretrained_models/big-lama",
        help="LaMa checkpoint dir, only used with --inpaint_model lama.",
    )
    parser.add_argument("--dry_run", action="store_true",
                        help="List the episodes that would be processed and exit.")
    parser.add_argument("--traceback", action="store_true",
                        help="Print full tracebacks for failed episodes.")


def main():
    parser = argparse.ArgumentParser(
        description="Per-frame hand removal with a finetuned SAM3 image model.")
    setup_args(parser)
    args = parser.parse_args(sys.argv[1:])
    args.device = "cuda" if torch.cuda.is_available() else "cpu"
    if args.targets is None:
        args.targets = (TARGETS_WITH_OBJECT if args.mode == "remove_object"
                        else DEFAULT_TARGETS)

    episodes = discover_episodes(Path(args.input_dir), tuple(args.exclude))
    if args.limit:
        episodes = episodes[:args.limit]
    if args.sample and args.sample < len(episodes):
        rng = random.Random(args.seed)
        episodes = sorted(rng.sample(episodes, args.sample),
                          key=lambda e: e.name)
    if not 0 <= args.shard_index < args.num_shards:
        parser.error("--shard_index must be in [0, --num_shards)")
    if args.num_shards > 1:
        # 按名称排序后取模分片,保证多进程间不相交且可复现
        episodes = sorted(episodes, key=lambda e: e.name)
        episodes = episodes[args.shard_index::args.num_shards]
    if args.shard_skip:
        episodes = episodes[args.shard_skip:]
    shard_note = (f"  shard {args.shard_index}/{args.num_shards}"
                  if args.num_shards > 1 else "")
    if args.shard_skip:
        shard_note += f" skip {args.shard_skip}"
    print(f"Found {len(episodes)} episode(s) under {args.input_dir}{shard_note}")
    print(f"Erasing: {args.targets!r}  Inpainter: {args.inpaint_model}")

    if args.dry_run:
        for ep in episodes:
            kind = "video" if ep.is_video else f"{len(ep.frame_names)} frames"
            print(f"  {ep.name}  ({kind})")
        return

    if not episodes:
        return

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    models = build_models(args)

    manifest_path = (out_dir / f"manifest_shard{args.shard_index}{args.manifest_tag}.json"
                     if args.num_shards > 1
                     else out_dir / f"manifest{args.manifest_tag}.json")

    manifest = []
    counts = {}
    for i, ep in enumerate(episodes, 1):
        print(f"[{i}/{len(episodes)}] {ep.name}", flush=True)
        try:
            entry = process_episode(models, ep, args)
        except Exception as e:
            entry = {"name": ep.name, "source": str(ep.source),
                     "status": "error", "error": f"{type(e).__name__}: {e}"}
            print(f"    failed: {entry['error']}", flush=True)
            if args.traceback:
                traceback.print_exc()
        counts[entry["status"]] = counts.get(entry["status"], 0) + 1
        manifest.append(entry)
        manifest_path.write_text(
            json.dumps({"targets": args.targets, "episodes": manifest},
                       indent=2))

    print("\nDone. " + ", ".join(f"{k}: {v}" for k, v in sorted(counts.items())))
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
