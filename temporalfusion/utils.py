"""Utility functions for TemporalFusion."""

import random
import numpy as np
import torch
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def set_seed(seed: int = 42):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def count_parameters(model: torch.nn.Module) -> dict:
    """Count total and trainable parameters."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total": total, "trainable": trainable, "total_M": total / 1e6, "trainable_M": trainable / 1e6}


def save_checkpoint(model, optimizer, epoch, metrics, path):
    """Save training checkpoint."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    state = {
        "model": model.state_dict() if not hasattr(model, "module") else model.module.state_dict(),
        "optimizer": optimizer.state_dict(),
        "epoch": epoch,
        "metrics": metrics,
    }
    torch.save(state, path)
    logger.info(f"Checkpoint saved to {path}")
