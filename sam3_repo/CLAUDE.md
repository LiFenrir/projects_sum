# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository overview

This is the official PyTorch implementation of **SAM 3** (Segment Anything Model 3), a unified foundation model for promptable image and video segmentation. The codebase uses `setuptools`, is formatted with `ufmt`/`black`/`ruff-api`, and uses Hydra for training/evaluation configuration. The index in `.codegraph/` is a pre-built CodeGraph knowledge graph; run `codegraph sync` after significant file changes to keep it up to date.

## Common commands

### Environment setup

```bash
# Basic editable install
pip install -e .

# Development + training dependencies (required for scripts/train/eval work)
pip install -e ".[train,dev]"

# Notebook/example dependencies
pip install -e ".[notebooks]"

# Optional inference speed-ups (CUDA 12.8 wheels shown in README)
pip install einops ninja && pip install flash-attn-3 --no-deps --index-url https://download.pytorch.org/whl/cu128
pip install git+https://github.com/ronghanghu/cc_torch.git
```

The public model checkpoints are hosted on Hugging Face and require accepted access plus authentication (`huggingface-cli login` or `HF_TOKEN`).

### Formatting and linting

The project uses `ufmt` with `black==24.2.0`, `usort==1.0.2`, and `ruff-api==0.1.0`. The CI workflow (`.github/workflows/format.yml`) checks `sam3 scripts`.

```bash
# Format the main code directories
ufmt format sam3 scripts

# Check formatting without modifying files
ufmt check sam3 scripts
```

`pyproject.toml` also configures `mypy`, but there is no enforced CI type-check step.

### Tests

Only a small test suite exists today in `test/` (note: the directory is singular, while `pyproject.toml` sets `testpaths = ["tests"]`).

```bash
# Run the existing tests
pytest test/

# Run a single test file
pytest test/test_io_utils.py

# Run a single test class/method
pytest test/test_io_utils.py::TestLoadVideoFramesRouting::test_mp4_extension_routes_to_video_loader
```

### Training and evaluation

Training/evaluation are driven by `sam3/train/train.py` with Hydra configs located under `sam3/train/configs`.

```bash
# Run a single-node evaluation on N GPUs
python sam3/train/train.py \
  -c configs/gold_image_evals/sam3_gold_image_crowded.yaml \
  --use-cluster 0 \
  --num-gpus N
```

- Hydra initializes from the `sam3.train` package, so the `--config` argument is relative to `sam3/train/` (e.g., `configs/gold_image_evals/...`).
- Set `--use-cluster 1` to launch via SubmitIt/SLURM.
- Override dataset/checkpoint paths in `sam3/train/configs/eval_base.yaml` or in a derived config; `checkpoint_path: null` downloads from Hugging Face.

For offline SA-Co/Gold scoring after predictions have been dumped:

```bash
python scripts/eval/gold/eval_sam3.py \
  --gt-folder <YOUR_GOLD_GT_DIR> \
  --pred-folder <PREDICTION_ROOT>
```

### Image inference (Python API)

```python
from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor

model = build_sam3_image_model()
processor = Sam3Processor(model)
state = processor.set_image(image)
state = processor.set_text_prompt(prompt="a player in white", state=state)
masks, boxes, scores = state["masks"], state["boxes"], state["scores"]
```

`Sam3Processor` also supports `add_geometric_prompt` for box prompts and `set_confidence_threshold`.

### Video inference (request-based API)

```python
from sam3.model_builder import build_sam3_video_predictor

predictor = build_sam3_video_predictor()
resp = predictor.handle_request({"type": "start_session", "resource_path": video_path})
session_id = resp["session_id"]
resp = predictor.handle_request({
    "type": "add_prompt",
    "session_id": session_id,
    "frame_index": 0,
    "text": "YOUR_TEXT_PROMPT",
})

# Stream propagation forward/backward
for frame in predictor.handle_stream_request({
    "type": "propagate_in_video",
    "session_id": session_id,
    "propagation_direction": "both",
}):
    ...
```

`build_sam3_video_predictor` returns a `Sam3VideoPredictorMultiGPU`, which can be constructed with `gpus_to_use=[0, 1, ...]` to run across multiple GPUs via spawned worker processes and NCCL.

## High-level architecture

### Model builder and components

`sam3/model_builder.py` is the central factory. Important public functions:

- `build_sam3_image_model` — builds the image grounding/detection/segmentation model.
- `build_sam3_video_model` — builds the video model by composing a detector and a tracker.
- `build_sam3_video_predictor` — wraps the video model in the multi-GPU request/response predictor.

The image model (`Sam3Image`, `sam3/model/sam3_image.py`) is a DETR-style architecture with the following stages:

1. **Shared visual-language backbone** (`SAM3VLBackbone`, `sam3/model/vl_combiner.py`):
   - Visual branch: a ViT trunk (`sam3/model/vitdet.py`) plus a feature-pyramid neck (`Sam3DualViTDetNeck`, `sam3/model/necks.py`).
   - Text branch: a BPE tokenizer (`sam3/model/tokenizer_ve.py`) and transformer text encoder (`VETextEncoder`, `sam3/model/text_encoder_ve.py`).
2. **Geometry prompt encoder** (`SequenceGeometryEncoder`, `sam3/model/geometry_encoders.py`) for boxes and points.
3. **Transformer encoder-decoder** (`sam3/model/encoder.py`, `sam3/model/decoder.py`). The decoder supports a **presence token** that helps disambiguate similar text prompts.
4. **Segmentation head** (`UniversalSegmentationHead`, `sam3/model/maskformer_segmentation.py`).
5. **Scoring** via dot-product scoring (`DotProductScoring`, `sam3/model/model_misc.py`) or a learned class embed.

`Sam3Processor` (`sam3/model/sam3_image_processor.py`) is the user-facing wrapper that handles preprocessing and calls `model.forward_grounding`.

### Video pipeline: decoupled detector + tracker

The video model (`Sam3VideoInferenceWithInstanceInteractivity`, `sam3/model/sam3_video_inference.py`) combines:

- A **detector** (`Sam3ImageOnVideoMultiGPU`) run on individual frames for open-vocabulary text/box/point grounding.
- A **tracker** (`Sam3TrackerPredictor`, `sam3/model/sam3_tracking_predictor.py`) inherited from the SAM 2 transformer encoder-decoder architecture for temporal propagation and interactive refinement.

At runtime, the detector produces per-frame detections, which are associated into tracklets through an IoU/association heuristic (`assoc_iou_thresh`, `new_det_thresh`, etc.). The tracker then propagates and refines masks across frames. `Sam3BasePredictor` (`sam3/model/sam3_base_predictor.py`) defines the session API (`start_session`, `add_prompt`, `propagate_in_video`, `remove_object`, `close_session`) that both single- and multi-GPU predictors implement.

The multi-object tracking implementation uses a **dynamic multiplex state** (`VideoTrackingDynamicMultiplex`, `sam3/model/video_tracking_multiplex.py`) to keep multiple object slots in a shared memory representation and demux them when needed for single-object interaction.

### Interactive instance segmentation

When `build_sam3_image_model` is called with `enable_inst_interactivity=True`, it constructs a SAM 2-style tracker via `build_tracker` and wraps it in `SAM3InteractiveImagePredictor` (`sam3/model/sam1_task_predictor.py`), enabling point/box prompts for single-instance masks (the original SAM/SAM 2 task) on top of the open-vocabulary detector.

### Training stack

`sam3/train/trainer.py` is the main training loop. It is configured through Hydra YAML files in `sam3/train/configs/` and orchestrated by `sam3/train/train.py`. Key subsystems:

- **Data**: `sam3/train/data/` — image/video PyTorch datasets and COCO JSON loaders.
- **Losses**: `sam3/train/loss/` — focal loss, mask sampling, and the combined SAM 3 loss.
- **Optimization**: `sam3/train/optim/` — AdamW + layer-decay parameter groups and inverse-square-root schedulers.
- **Distributed**: DDP via `torch.distributed` with SubmitIt support for SLURM.

### Evaluation

- `sam3/eval/` implements COCO-style evaluation, cgF1 computation (`cgf1_eval.py`), and SA-Co video evaluation (`saco_veval_eval.py`).
- `scripts/eval/gold/eval_sam3.py` is the offline scorer for SA-Co/Gold.
- `scripts/eval/silver/` and `scripts/eval/veval/` contain dataset download/preprocessing helpers and READMEs.

### Performance notes

- `model_builder.py` enables TensorFloat-32 on Ampere+ GPUs at import.
- `compile=True` / `compile_mode="default"` enables `torch.compile` on parts of the model.
- Optional Flash Attention 3 (`use_fa3`) and custom kernels (`cc_torch`) can be installed for faster inference.
- `sam3/perflib/` contains fused/compiled helpers, connected-components, NMS, and mask operations used in the hot path.

## Code style notes

- Files are marked `# pyre-unsafe`; the project is typed in many places but relies on inline `# pyre-fixme` annotations rather than strict CI enforcement.
- Keep imports at module scope; inline imports are used sparingly in the existing code (e.g., inside methods to avoid circular imports), but prefer module-level imports for new code.
- The public API surface is small: `build_sam3_image_model`, `build_sam3_video_predictor`, and `Sam3Processor` are the main user entry points.
