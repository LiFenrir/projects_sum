#!/usr/bin/env bash
set -euo pipefail

# ARX 双阶段 SFT 训练脚本 (DDP)
# 依次训练 configs/arx 下的两个配置：
#   1. pi05_finetune.yaml       → 原始数据集 arx_0703_1521
#   2. pi05_finetune_space.yaml → 合并数据集 arx_0703_1521_merged
#
# 用法：
#   ./scripts/train_arx.sh              # 默认 4 GPU
#   ./scripts/train_arx.sh 8            # 指定 8 GPU

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

NPROC=${1:-2}

echo "=========================================="
echo "ARX SFT Training Pipeline (DDP, ${NPROC} GPU)"
echo "=========================================="

# ── Stage 1: pi05_finetune ──
echo ""
echo "[Stage 1/2] Training with pi05_finetune.yaml"
echo "------------------------------------------"
torchrun --standalone --nnodes=1 --nproc_per_node="${NPROC}" \
    "${PROJECT_DIR}/scripts/train_pytorch.py" \
    --config "${PROJECT_DIR}/configs/arx/pi05_finetune.yaml" \
    --exp_name arx_001

echo ""
echo "[Stage 1/2] Done."

# ── Stage 2: pi05_finetune_space ──
echo ""
echo "[Stage 2/2] Training with pi05_finetune_space.yaml"
echo "------------------------------------------"
torchrun --standalone --nnodes=1 --nproc_per_node="${NPROC}" \
    "${PROJECT_DIR}/scripts/train_pytorch.py" \
    --config "${PROJECT_DIR}/configs/arx/pi05_finetune_space.yaml" \
    --exp_name arx_space_001

echo ""
echo "=========================================="
echo "All stages completed."
echo "=========================================="
