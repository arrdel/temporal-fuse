#!/usr/bin/env bash
# ============================================================================
# TemporalFusion: Master experiment runner
# 
# Runs all experiments:
#   1. Baselines (DirectTransformer, TSN, MeanPool)
#   2. Ablations (Phase 1 only, Phase 1+2, Full)
#   3. Full model training
#   4. Evaluation on all metrics
# ============================================================================

set -euo pipefail

# Configuration
DATA_ROOT="./data/activitynet"
TRAIN_DIR="$DATA_ROOT/c3d_features/train"
VAL_DIR="$DATA_ROOT/c3d_features/val"
ANNOTATIONS="$DATA_ROOT/gt.json"
OUTPUT_DIR="./runs"
EVAL_DIR="./eval_results"

EPOCHS=50
BATCH_SIZE=32
LR=1e-4
NUM_WORKERS=4
MAX_FRAMES=512

export WANDB_PROJECT=temporalfusion

mkdir -p "$OUTPUT_DIR" "$EVAL_DIR"

echo "============================================="
echo "TemporalFusion — Experiment Suite"
echo "============================================="
echo "Train dir: $TRAIN_DIR"
echo "Val dir:   $VAL_DIR"
echo "Epochs:    $EPOCHS"
echo "============================================="

# ---------------------------------------------------------------------------
# 1. Train Baselines
# ---------------------------------------------------------------------------
echo ""
echo ">>> [1/4] Training baselines..."

# Direct Transformer baseline
echo "  Training: DirectTransformer"
python -m temporalfusion.train \
    --train_dir "$TRAIN_DIR" --val_dir "$VAL_DIR" --annotations "$ANNOTATIONS" \
    --batch_size $BATCH_SIZE --epochs $EPOCHS --lr $LR \
    --num_workers $NUM_WORKERS --max_frames $MAX_FRAMES \
    --lambda_tc 0.0 --lambda_reg 0.0 --lambda_cs 0.0 \
    --num_hierarchy_levels 0 \
    --run_name "baseline_direct_transformer" \
    --output_dir "$OUTPUT_DIR"

# ---------------------------------------------------------------------------
# 2. Ablation: Phase 1 only (temporal contrastive + collapse prevention)
# ---------------------------------------------------------------------------
echo ""
echo ">>> [2/4] Ablation: Phase 1 only..."

python -m temporalfusion.train \
    --train_dir "$TRAIN_DIR" --val_dir "$VAL_DIR" --annotations "$ANNOTATIONS" \
    --batch_size $BATCH_SIZE --epochs $EPOCHS --lr $LR \
    --num_workers $NUM_WORKERS --max_frames $MAX_FRAMES \
    --lambda_tc 1.0 --lambda_reg 0.1 --lambda_cs 0.0 \
    --num_hierarchy_levels 0 \
    --run_name "ablation_phase1_only" \
    --output_dir "$OUTPUT_DIR"

# ---------------------------------------------------------------------------
# 3. Ablation: Phase 1+2 (+ hierarchical aggregation)
# ---------------------------------------------------------------------------
echo ""
echo ">>> [3/4] Ablation: Phase 1+2..."

python -m temporalfusion.train \
    --train_dir "$TRAIN_DIR" --val_dir "$VAL_DIR" --annotations "$ANNOTATIONS" \
    --batch_size $BATCH_SIZE --epochs $EPOCHS --lr $LR \
    --num_workers $NUM_WORKERS --max_frames $MAX_FRAMES \
    --lambda_tc 1.0 --lambda_reg 0.1 --lambda_cs 0.5 \
    --num_hierarchy_levels 4 \
    --run_name "ablation_phase1_phase2" \
    --output_dir "$OUTPUT_DIR"

# ---------------------------------------------------------------------------
# 4. Full model
# ---------------------------------------------------------------------------
echo ""
echo ">>> [4/4] Full TemporalFusion model..."

python -m temporalfusion.train \
    --train_dir "$TRAIN_DIR" --val_dir "$VAL_DIR" --annotations "$ANNOTATIONS" \
    --batch_size $BATCH_SIZE --epochs $EPOCHS --lr $LR \
    --num_workers $NUM_WORKERS --max_frames $MAX_FRAMES \
    --lambda_tc 1.0 --lambda_reg 0.1 --lambda_cs 0.5 \
    --num_hierarchy_levels 4 \
    --run_name "full_temporalfusion" \
    --output_dir "$OUTPUT_DIR"

# ---------------------------------------------------------------------------
# 5. Evaluate all models
# ---------------------------------------------------------------------------
echo ""
echo ">>> Evaluating all models..."

for run in baseline_direct_transformer ablation_phase1_only ablation_phase1_phase2 full_temporalfusion; do
    CKPT="$OUTPUT_DIR/$run/best_model.pt"
    if [ -f "$CKPT" ]; then
        echo "  Evaluating: $run"
        python -m temporalfusion.evaluate \
            --checkpoint "$CKPT" \
            --val_dir "$VAL_DIR" \
            --annotations "$ANNOTATIONS" \
            --output "$EVAL_DIR/${run}_results.json" \
            --max_frames $MAX_FRAMES
    else
        echo "  SKIP: $run (no checkpoint found)"
    fi
done

echo ""
echo "============================================="
echo "All experiments complete!"
echo "Results in: $EVAL_DIR/"
echo "============================================="
