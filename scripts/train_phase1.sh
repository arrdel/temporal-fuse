#!/bin/bash
# TemporalFusion Phase 1: Single-GPU training on ActivityNet C3D features
# Run inside tmux session: temporalfusion

set -euo pipefail

PROJECT_DIR=/home/achinda1/projects/hierarchical-vlm
cd "${PROJECT_DIR}"

# Activate conda env
eval "$(conda shell.bash hook)"
conda activate temporalfusion

export PYTHONPATH="${PROJECT_DIR}:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES=0

echo "=========================================="
echo "TemporalFusion Phase 1 Training"
echo "=========================================="
echo "Date: $(date)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader -i 0)"
echo "=========================================="

python -m temporalfusion.train \
    --config configs/train_activitynet.yaml \
    --run_name "phase1_activitynet_$(date +%Y%m%d_%H%M%S)" \
    --epochs 50 \
    --batch_size 32 \
    --lr 1e-4 \
    --num_workers 4 \
    --log_every 10 \
    --eval_every 1 \
    --save_every 5

echo "Phase 1 training complete!"
