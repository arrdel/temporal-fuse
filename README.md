# TemporalFusion: Temporal Contrastive Learning with Hierarchical Feature Aggregation for Video Understanding

Adele Chinda, Desire Emeka, Maryam Koya, Nita Ngozi Ezekwem

[![Code](https://img.shields.io/badge/💻-Code-black)](https://github.com/arrdel/temporal-fuse)
[![W&B Logs](https://img.shields.io/badge/📊-W&B_Logs-orange)](https://wandb.ai/el_chindah1/temporalfusion)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

## Overview

**TemporalFusion** addresses the challenge of learning temporally consistent video representations. We combine a temporal contrastive loss, collapse prevention regularization, and cross-scale consistency within a hierarchical aggregation framework. The model operates on pre-extracted features (I3D for THUMOS-14, VGG-16 for Charades) and is evaluated on action classification and temporal consistency.

### Key Components

| Component | Description |
|-----------|-------------|
| **Temporal Contrastive Loss** | Enforces similarity between adjacent frame representations |
| **Collapse Prevention** | Variance regularization to prevent representation degeneration |
| **Cross-Scale Consistency** | Aligns representations across hierarchical pooling levels |
| **Hierarchical Aggregation** | 4-level attention-weighted temporal pyramid (T → T/16) |

## Results

### Main Comparison — THUMOS-14 (Single-Label, 20 Classes)

| Model | Params | Top-1 (%) | Top-5 (%) | TC |
|-------|--------|-----------|-----------|------|
| MeanPool | 0.05M | 87.74 | 98.11 | 0.806 |
| TemporalSegment (TSN) | 2.1M | 36.32 | 66.98 | 0.806 |
| DirectTransformer | 77.7M | 90.09 | 97.64 | 0.999 |
| **TemporalFusion (Ours)** | **77.9M** | **89.62** | **98.11** | **0.989** |

### Main Comparison — Charades (Multi-Label, 157 Classes)

| Model | Params | mAP (%) | TC |
|-------|--------|---------|----|
| MeanPool | 0.7M | 14.28 | 0.671 |
| TemporalSegment (TSN) | 8.7M | 14.66 | 0.671 |
| DirectTransformer | 79.9M | 17.63 | 0.999 |
| **TemporalFusion (Ours)** | **79.9M** | **16.72** | **0.999** |

### Ablation Study — THUMOS-14

| Variant | Top-1 (%) | TC |
|---------|-----------|------|
| Full Model | 87.74 | 0.989 |
| w/o Temporal Contrastive | 92.92 | 0.862 |
| w/o Collapse Prevention | 91.04 | 0.989 |
| w/o Cross-Scale | 90.57 | 0.999 |
| Classification Only | 88.68 | 0.955 |

### Ablation Study — Charades

| Variant | mAP (%) | TC |
|---------|---------|------|
| Full Model | 16.03 | 0.999 |
| w/o Temporal Contrastive | 17.25 | 0.947 |
| w/o Collapse Prevention | 16.83 | 0.999 |
| w/o Cross-Scale | 13.11 | 1.000 |
| Classification Only | 13.04 | 0.982 |

## Architecture

```
Video Features (B, T, D)
    │
    ▼
Feature Projection (Linear + LN + GELU)
    │  + Sinusoidal Positional Encoding
    ▼
Transformer Encoder (6L × 8H, Pre-LN)
    │  ← L_tc (Temporal Contrastive) + L_reg (Collapse Prevention)
    ▼
Hierarchical Aggregation (4 levels, attention-weighted pairs)
    │  ← L_cs (Cross-Scale Consistency)
    ▼
Classifier (LN → Linear → C classes)
       ← L_cls (Cross-Entropy / BCE)
```

## Project Structure

```
temporalfusion/             # Core package
├── model.py               # TemporalFusion model architecture
├── losses.py              # Loss functions (temporal, collapse, cross-scale)
├── data.py                # Dataset loaders (THUMOS-14, Charades)
├── training.py            # Training loop with DDP support
├── train.py               # Training entry point
├── evaluate.py            # Evaluation metrics (accuracy, mAP, TC)
├── baselines.py           # Baseline models (MeanPool, TSN, DirectTransformer)
├── train_baselines.py     # Baseline training script
├── run_ablations.py       # Ablation study runner
└── utils.py               # Utilities

configs/                    # Training configurations
├── train_thumos14.yaml    # THUMOS-14 config
├── train_charades.yaml    # Charades config
└── train_activitynet.yaml # ActivityNet config

scripts/                    # Experiment scripts
├── run_experiments.sh     # Full experiment pipeline
├── run_baselines.py       # Run all baselines
├── run_ablation_study.sh  # Run ablation study
├── eval_ablations.py      # Evaluate ablation results
├── generate_paper_figures.py
└── generate_improved_figures.py

tests/
└── test_temporalfusion.py # Unit tests
```

## Installation

```bash
# Clone
git clone https://github.com/arrdel/temporal-fuse.git
cd temporal-fuse

# Create environment
conda create -n temporalfusion python=3.10 -y
conda activate temporalfusion

# Install dependencies
pip install -r requirements.txt
pip install -e .
```

## Training

```bash
# Single GPU — THUMOS-14
python -m temporalfusion.train --config configs/train_thumos14.yaml

# 8-GPU DDP — THUMOS-14
torchrun --nproc_per_node=8 -m temporalfusion.train \
    --config configs/train_thumos14.yaml

# 8-GPU DDP — Charades
torchrun --nproc_per_node=8 -m temporalfusion.train \
    --config configs/train_charades.yaml
```

### Baselines & Ablations

```bash
# Run all baselines on both datasets
python scripts/run_baselines.py

# Run full ablation study
bash scripts/run_ablation_study.sh
```

## Evaluation

```bash
# Evaluate a trained model
python -m temporalfusion.evaluate \
    --checkpoint runs/<run_dir>/best_model.pt \
    --config configs/train_thumos14.yaml

# Evaluate ablation results
python scripts/eval_ablations.py
```

## Hardware

All experiments were conducted on **8× NVIDIA RTX A6000** (48 GB each) using PyTorch DDP with mixed precision (AMP). Total compute: ~60 GPU-hours.

## Citation

<!-- ```bibtex
@inproceedings{chinda2026temporalfusion,
  title={TemporalFusion: Temporal Contrastive Learning with Hierarchical 
         Feature Aggregation for Video Understanding},
  author={Chinda, Adele and Emeka, Desire and Koya, Maryam and Ezekwem, Nita Ngozi},
  booktitle={Proceedings of the British Machine Vision Conference (BMVC)},
  year={2026}
}
``` -->

## License

MIT License — See [LICENSE](LICENSE) for details.
