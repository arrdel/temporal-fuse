#!/bin/bash
# ==========================================================================
# Ablation Study Runner
# Runs each ablation variant as a separate torchrun job, then evaluates all.
# This avoids NCCL deadlocks by keeping DDP training and evaluation separate.
#
# Usage:
#   bash scripts/run_ablation_study.sh thumos14
#   bash scripts/run_ablation_study.sh charades
# ==========================================================================

set -e

DATASET=${1:-thumos14}
PROJ_DIR="/home/achinda1/projects/hierarchical-vlm"
cd "$PROJ_DIR"
export PYTHONPATH="$PROJ_DIR"

echo "============================================="
echo "Ablation Study: ${DATASET}"
echo "============================================="

if [ "$DATASET" = "thumos14" ]; then
    CONFIG="configs/train_thumos14.yaml"
    EPOCHS=200
elif [ "$DATASET" = "charades" ]; then
    CONFIG="configs/train_charades.yaml"
    EPOCHS=100
else
    echo "Unknown dataset: $DATASET"
    exit 1
fi

# Define ablation variants: name lambda_tc lambda_reg lambda_cs
declare -a VARIANTS=(
    "full:1.0:0.1:0.5"
    "no_temporal:0.0:0.1:0.5"
    "no_collapse:1.0:0.0:0.5"
    "no_crossscale:1.0:0.1:0.0"
    "cls_only:0.0:0.0:0.0"
)

# Phase 1: Train each variant (skip if checkpoint exists)
echo ""
echo "===== Phase 1: Training ====="
for variant_str in "${VARIANTS[@]}"; do
    IFS=':' read -r NAME LTC LREG LCS <<< "$variant_str"
    RUN_NAME="ablation_${DATASET}_${NAME}"
    CKPT_PATH="runs/${RUN_NAME}/best_model.pt"

    if [ -f "$CKPT_PATH" ]; then
        echo "[${NAME}] Checkpoint exists, skipping training."
        continue
    fi

    echo ""
    echo "===== Training: ${NAME} (tc=${LTC}, reg=${LREG}, cs=${LCS}) ====="
    
    torchrun --nproc_per_node=8 --master_port=29500 \
        -m temporalfusion.training \
        --config "$CONFIG" \
        --lambda_tc "$LTC" \
        --lambda_reg "$LREG" \
        --lambda_cs "$LCS" \
        --run_name "$RUN_NAME" \
        --epochs "$EPOCHS" \
        --wandb_project "temporalfusion" \
        --amp 1 \
        --cudnn_benchmark 1

    echo "[${NAME}] Training complete."
done

echo ""
echo "===== Phase 1 Complete: All variants trained ====="

# Phase 2: Evaluate all checkpoints (single GPU, no DDP)
echo ""
echo "===== Phase 2: Evaluation ====="

python3 scripts/eval_ablations.py --dataset "$DATASET"

echo ""
echo "===== Ablation Study Complete ====="
