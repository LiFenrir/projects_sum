# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project overview

Inpaint-Anything removes, fills, or replaces objects in images, videos, and 3D scenes. It combines a segmentation model (SAM 3 by default), a tracker for video/multi-view propagation, and one of several inpainting models. The repo is research/demonstration code rather than a packaged library; the entry points are top-level Python scripts and a small Gradio app.

## Environment and dependencies

- Python ≥ 3.12, PyTorch ≥ 2.7, CUDA ≥ 12.6 (SAM 3's requirements).
- SAM 3 is not on PyPI; install from a source checkout (the directory must not be named `sam3`, or it shadows the installed package):
  ```bash
  git clone https://github.com/facebookresearch/sam3.git sam3_repo
  python -m pip install -e ./sam3_repo
  ```
- ProPainter is used as a source checkout, not a package:
  ```bash
  git clone https://github.com/sczhou/ProPainter.git propainter
  ```
- Install everything else with:
  ```bash
  python -m pip install -r requirements.txt
  ```
- Optional: keep the legacy SAM 1 / MobileSAM backend available:
  ```bash
  python -m pip install -e segment_anything
  ```

## Weights

Run `python script/download_weights.py` to download the default models into `./pretrained_models`. Defaults include SAM 3, LaMa (`big-lama/`), and ProPainter. `--only flux` adds FLUX.1-Fill, `--only diffueraser` adds DiffuEraser (into `diffueraser/weights/`), `--all` also adds legacy STTN and OSTrack weights.

SAM 3 is gated on HuggingFace: accept the licence at https://huggingface.co/facebook/sam3, create a read token, then `hf auth login` or `export HF_TOKEN=hf_xxx` before running `python script/download_weights.py --only sam3`. The script falls back to a ModelScope mirror if HuggingFace is unreachable. Place a manually downloaded `sam3.pt` at `./pretrained_models/sam3.pt` (or pass `--sam_ckpt`). Do not use SAM 3.1 multiplex checkpoints (`sam3.1_multiplex.pt`); point/box prompts will silently fail to load.

## Common commands

Run the included example scripts for each pipeline:

```bash
bash script/remove_anything.sh           # image object removal
bash script/fill_anything.sh             # text-guided fill
bash script/replace_anything.sh          # text-guided background replace
bash script/remove_anything_video.sh     # video object removal
bash script/remove_anything_3d.sh        # 3D scene object removal
```

Launch the local Gradio UI:

```bash
cd app && python app.py                  # http://localhost:7860
```

There is no test suite, lint command, or build step. Verify changes by running the relevant example script or the app with the sample inputs under `example/`.

## Model stack and flags

| Stage | Default | Alternatives | Flag |
| --- | --- | --- | --- |
| Segmentation | SAM 3 | SAM 1 (`vit_h/l/b`), MobileSAM (`vit_t`) | `--sam_model_type` |
| Video / multi-view tracking | SAM 3 video predictor | OSTrack (legacy) | `--tracker` |
| Image removal | LaMa | FLUX.1-Fill, SDXL | `--inpaint_model` |
| Text-guided fill/replace | SDXL Inpainting | FLUX.1-Fill, SD 1.5 | `--sd_model` |
| Video inpainting | ProPainter | DiffuEraser (robotics scripts), STTN (legacy), per-frame LaMa | `--vi_model` / `--inpaint_model` |
| 3D novel-view synthesis | NeRF | — | — |

- `--text_select "phrase"` replaces point prompts on any pipeline when using SAM 3. For images, all matching instances are merged; for video/3D, only the highest-scoring instance is kept.
- `--coords_type click` lets the user click on an image with a display device; `key_in` takes `--point_coords`.
- Video: `--vi_fp16` reduces ProPainter memory. `--tracker ostrack` restores the legacy tracker and needs `--tracker_ckpt vitb_384_mae_ce_32x4_ep300` (a config name resolved to `./pytracking/pretrain/<name>.pth`).

## High-level architecture

### Image pipelines

Top-level scripts drive the same building blocks:

- `sam_segment.py` — builds the SAM model (`build_sam_model`) and adapts SAM 3 for image/text prompts (`Sam3PredictorAdapter`).
- `lama_inpaint.py` — builds LaMa (`build_lama_model`) and runs image inpainting (`inpaint_img_with_builded_lama`).
- `stable_diffusion_inpaint.py` — text-guided fill/replace via `diffusers` (SDXL / FLUX.1-Fill).

`remove_anything.py` → segment with SAM, optionally dilate the mask, then inpaint with LaMa (default) or a diffusion model.  
`fill_anything.py` / `replace_anything.py` → segment the target region, then call `fill_img_with_sd` / `replace_img_with_sd` with a text prompt. The difference is which pixels are masked: fill keeps the prompt inside the masked object; replace keeps the object and regenerates the background.

### Video pipeline

`remove_anything_video.py` instantiates `RemoveAnythingVideo`:

- Builds a tracker (`build_sam3_video_tracker` in `sam3_video_track.py`, or the legacy OSTrack path).
- Builds an inpainter (`build_propainter_model`, `build_sttn_model`, or per-frame LaMa).
- Seeds the track from a point, box, text phrase (`forward_segmentor_text`), or explicit `mask_idx`.
- `track_masks_in_video` propagates the first-frame prompt through the video and returns per-frame boolean masks.
- `forward_inpainter` feeds frames and masks to the selected video inpainter and returns the completed frames.

`propainter_video_inpaint.py` adapts the ProPainter source checkout for the repo's image/mask format; `sttn_video_inpaint.py` does the same for STTN.

### 3D pipeline

`remove_anything_3d.py` uses the same tracker/inpainter combination to erase an object from every source view of an LLFF scene, then trains a NeRF (`nerf/run_nerf.py`) on the erased views to synthesize novel views. The erased images are written to `<scene>/images_remove_<factor>/removed_with_mask_<dilate>/`, and `nerf/load_llff.py` reads from that directory. Remember to point `datadir` in `nerf/configs/<scene>.txt` at the actual scene directory.

### Robotics batch engine

`remove_hands.py` implements the hand-removal-and-inpainting stage of Human-to-Robot synthesis pipelines. It walks `--input_dir` recursively, detects hands (or a custom `--target`) with SAM 3 text prompts, tracks them through each episode with SAM 3 video, and inpaints with ProPainter. It exports clean plates, masks, and a `manifest.json`. Use `--dry_run` and `--limit` to sanity-check before a full pass; `--skip_existing` resumes interrupted runs.

`remove_hands_perframe.py` is the per-frame variant: detection runs independently on every frame with a finetuned SAM 3 image model (no tracker). Its `--inpaint_model` selects the inpainter: `propainter` (default), `diffueraser` (video diffusion with a ProPainter prior, best quality + temporal consistency; tune with `--diffueraser_steps`, `--max_img_size`, `--diffueraser_mask_dilation`), or per-frame image models (`flux`, `lama`, `sdxl`). The DiffuEraser adapter (`diffueraser_video_inpaint.py`) wraps the `diffueraser/` source checkout with an in-memory API and reuses this repo's ProPainter for the prior.

### Web app

`app/app.py` is a Gradio UI that reuses the same model builders and inference functions. It builds SAM and LaMa once at startup and holds them in a global `model` dict.

### Shared utilities

- `sam3_utils.py` — checkpoint resolution, autocast helpers, and the SAM 3.1 multiplex warning.
- `utils/utils.py` and `utils/mask_processing.py` — mask dilation/erosion, point/mask visualization, and I/O helpers.
- `utils/frames2video.py`, `utils/video2frames.py` — format conversions used by scripts and the app.

## Important implementation notes

- SAM 3's video predictor sets a global bfloat16 autocast in its constructor. `sam3_utils.no_autocast()` is used around non-SAM inference paths (LaMa, diffusion, ProPainter) to prevent silent bfloat16 execution.
- The video tracker needs a path to a video file or a folder of frames. `sam3_video_track._as_frame_folder` decodes videos to temporary JPEG frames because SAM 3's mp4 decoding path depends on `decord`, which is not guaranteed in this environment.
- Many example scripts and argument parsers use relative paths; run scripts from the repository root unless a script explicitly changes directory.
