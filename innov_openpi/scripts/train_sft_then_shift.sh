#!/bin/bash
# SFT → Shifted 串联训练脚本
# 先训练 bi_s1_sft，完成后自动训练 bi_s1_sft_shifted
# 使用 DDP 多卡训练 (2× RTX 4090)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

# ── 配置 ────────────────────────────────────────────────
NPROC=2                          # GPU 数量
CONFIG_SFT="configs/bi_s1/pi05_finetune_sft.yaml"
CONFIG_SHIFT="configs/bi_s1/pi05_finetune_sft_shifted.yaml"

# ── 可选：自定义实验名 ──────────────────────────────────
EXP_NAME_SFT="${1:-bi_s1_sft_run}"
EXP_NAME_SHIFT="${2:-bi_s1_sft_shifted_run}"

cd "$PROJECT_DIR"

echo "========================================="
echo "  Stage 1: SFT 训练"
echo "  配置: $CONFIG_SFT"
echo "  实验: $EXP_NAME_SFT"
echo "  GPU:  $NPROC"
echo "========================================="

torchrun --standalone --nnodes=1 --nproc_per_node="$NPROC" \
    scripts/train_pytorch.py \
    --config "$CONFIG_SFT" \
    --exp_name "$EXP_NAME_SFT"

echo ""
echo "========================================="
echo "  SFT 训练完成 ✓"
echo "========================================="
echo ""
echo "========================================="
echo "  Stage 2: Shifted 训练"
echo "  配置: $CONFIG_SHIFT"
echo "  实验: $EXP_NAME_SHIFT"
echo "  GPU:  $NPROC"
echo "========================================="

torchrun --standalone --nnodes=1 --nproc_per_node="$NPROC" \
    scripts/train_pytorch.py \
    --config "$CONFIG_SHIFT" \
    --exp_name "$EXP_NAME_SHIFT"

echo ""
echo "========================================="
echo "  全部训练完成 ✓"
echo "========================================="
