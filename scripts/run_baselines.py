#!/usr/bin/env python3
"""
Train and evaluate all baselines on the same data for comparison.

Produces a JSON results file with:
  - TemporalFusion (ours) — loaded from checkpoint
  - DirectTransformerBaseline
  - TemporalSegmentBaseline (TSN)
  - MeanPoolBaseline
"""

import argparse
import json
import logging
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, top_k_accuracy_score
from tqdm import tqdm

from temporalfusion.model import TemporalFusionModel
from temporalfusion.baselines import (
    DirectTransformerBaseline,
    TemporalSegmentBaseline,
    MeanPoolBaseline,
)
from temporalfusion.data import ActivityNetFeaturesDataset, collate_features

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def train_baseline(model, train_loader, val_loader, device, epochs=30,
                   lr=5e-4, label_smoothing=0.1, name="baseline"):
    """Train a baseline model and return best val accuracy."""
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

    best_val_acc = 0.0
    best_state = None
    total_params = sum(p.numel() for p in model.parameters())

    logger.info(f"Training {name} ({total_params:,} params) for {epochs} epochs...")

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        n_batches = 0

        for batch in train_loader:
            features = batch["features"].to(device)
            masks = batch["masks"].to(device)
            labels = batch["labels"].to(device)

            output = model(features, masks)
            loss = criterion(output["logits"], labels)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        scheduler.step()

        # Validate
        val_acc, val_loss = evaluate_model(model, val_loader, device, criterion)
        avg_loss = total_loss / max(n_batches, 1)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

        if (epoch + 1) % 5 == 0 or epoch == 0:
            logger.info(f"  [{name}] epoch {epoch:3d} | loss {avg_loss:.4f} | val_acc {val_acc:.4f}")

    # Restore best
    if best_state is not None:
        model.load_state_dict(best_state)

    return best_val_acc


@torch.no_grad()
def evaluate_model(model, loader, device, criterion=None):
    """Evaluate model, return (accuracy, loss)."""
    model.eval()
    all_preds, all_labels, all_probs = [], [], []
    total_loss = 0.0
    n = 0

    for batch in loader:
        features = batch["features"].to(device)
        masks = batch["masks"].to(device)
        labels = batch["labels"].to(device)

        output = model(features, masks)
        logits = output["logits"]

        if criterion is not None:
            total_loss += criterion(logits, labels).item()
            n += 1

        probs = F.softmax(logits, dim=-1).cpu().numpy()
        preds = logits.argmax(dim=-1).cpu().numpy()
        lbls = labels.cpu().numpy()

        for i in range(len(lbls)):
            all_preds.append(preds[i])
            all_labels.append(lbls[i])
            all_probs.append(probs[i])

    all_labels = np.array(all_labels)
    all_preds = np.array(all_preds)
    all_probs = np.stack(all_probs)

    acc = accuracy_score(all_labels, all_preds)
    avg_loss = total_loss / max(n, 1)
    return acc, avg_loss


@torch.no_grad()
def full_eval(model, loader, device):
    """Full evaluation returning top1, top5, temporal consistency."""
    model.eval()
    all_preds, all_labels, all_probs = [], [], []
    all_tc = []

    for batch in loader:
        features = batch["features"].to(device)
        masks = batch["masks"].to(device)
        labels = batch["labels"]

        output = model(features, masks)
        logits = output["logits"].cpu()
        probs = F.softmax(logits, dim=-1).numpy()

        for i in range(len(labels)):
            if labels[i] >= 0:
                all_preds.append(logits[i].argmax().item())
                all_labels.append(labels[i].item())
                all_probs.append(probs[i])

        # Temporal consistency from frame features
        h = output.get("frame_features")
        if h is not None:
            h = F.normalize(h, dim=-1)
            cos_sim = (h[:, :-1] * h[:, 1:]).sum(dim=-1)
            pair_mask = masks[:, :-1] * masks[:, 1:]
            for b in range(cos_sim.size(0)):
                valid = pair_mask[b].bool()
                if valid.any():
                    all_tc.append(cos_sim[b][valid].cpu().mean().item())

    all_labels = np.array(all_labels)
    all_preds = np.array(all_preds)
    all_probs = np.stack(all_probs)

    top1 = accuracy_score(all_labels, all_preds)
    num_classes = all_probs.shape[1]
    top5 = top_k_accuracy_score(all_labels, all_probs, k=min(5, num_classes)) if num_classes >= 5 else top1
    tc = float(np.mean(all_tc)) if all_tc else 0.0

    return {
        "top1_accuracy": round(float(top1), 4),
        "top5_accuracy": round(float(top5), 4),
        "temporal_consistency": round(tc, 4),
        "num_samples": len(all_labels),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, help="Path to TemporalFusion best checkpoint")
    parser.add_argument("--train_dir", required=True)
    parser.add_argument("--val_dir", required=True)
    parser.add_argument("--annotations", default=None)
    parser.add_argument("--output", default="./eval_results/baseline_comparison.json")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--feature_dim", type=int, default=2048)
    parser.add_argument("--hidden_dim", type=int, default=1024)
    parser.add_argument("--num_heads", type=int, default=8)
    parser.add_argument("--num_layers", type=int, default=6)
    parser.add_argument("--num_classes", type=int, default=200)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Data
    train_ds = ActivityNetFeaturesDataset(args.train_dir, args.annotations, "training", max_frames=512)
    val_ds = ActivityNetFeaturesDataset(args.val_dir, args.annotations, "val", max_frames=512)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, collate_fn=collate_features,
                              pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, collate_fn=collate_features,
                            pin_memory=True)

    results = {}

    # ========== 1. TemporalFusion (Ours) ==========
    logger.info("=" * 60)
    logger.info("Evaluating TemporalFusion (Ours)")
    logger.info("=" * 60)
    model_ours = TemporalFusionModel(
        feature_dim=args.feature_dim, hidden_dim=args.hidden_dim,
        num_heads=args.num_heads, num_layers=args.num_layers,
        num_classes=args.num_classes, num_hierarchy_levels=4,
    ).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    model_ours.load_state_dict(ckpt["model"])
    params_ours = sum(p.numel() for p in model_ours.parameters())

    t0 = time.time()
    results["TemporalFusion (Ours)"] = full_eval(model_ours, val_loader, device)
    results["TemporalFusion (Ours)"]["params"] = params_ours
    results["TemporalFusion (Ours)"]["eval_time_sec"] = round(time.time() - t0, 1)
    logger.info(f"  -> Top-1: {results['TemporalFusion (Ours)']['top1_accuracy']:.4f}")
    del model_ours
    torch.cuda.empty_cache()

    # ========== 2. DirectTransformerBaseline ==========
    logger.info("=" * 60)
    logger.info("Training DirectTransformerBaseline")
    logger.info("=" * 60)
    model_dt = DirectTransformerBaseline(
        feature_dim=args.feature_dim, hidden_dim=args.hidden_dim,
        num_heads=args.num_heads, num_layers=args.num_layers,
        num_classes=args.num_classes,
    )
    params_dt = sum(p.numel() for p in model_dt.parameters())

    t0 = time.time()
    train_baseline(model_dt, train_loader, val_loader, device, epochs=args.epochs, name="DirectTransformer")
    train_time_dt = time.time() - t0
    results["DirectTransformer"] = full_eval(model_dt, val_loader, device)
    results["DirectTransformer"]["params"] = params_dt
    results["DirectTransformer"]["train_time_sec"] = round(train_time_dt, 1)
    logger.info(f"  -> Top-1: {results['DirectTransformer']['top1_accuracy']:.4f}")
    del model_dt
    torch.cuda.empty_cache()

    # ========== 3. TemporalSegmentBaseline (TSN) ==========
    logger.info("=" * 60)
    logger.info("Training TemporalSegmentBaseline (TSN)")
    logger.info("=" * 60)
    model_tsn = TemporalSegmentBaseline(
        feature_dim=args.feature_dim, num_segments=8,
        num_classes=args.num_classes,
    )
    params_tsn = sum(p.numel() for p in model_tsn.parameters())

    t0 = time.time()
    train_baseline(model_tsn, train_loader, val_loader, device, epochs=args.epochs, name="TSN")
    train_time_tsn = time.time() - t0
    results["TSN (Segments)"] = full_eval(model_tsn, val_loader, device)
    results["TSN (Segments)"]["params"] = params_tsn
    results["TSN (Segments)"]["train_time_sec"] = round(train_time_tsn, 1)
    logger.info(f"  -> Top-1: {results['TSN (Segments)']['top1_accuracy']:.4f}")
    del model_tsn
    torch.cuda.empty_cache()

    # ========== 4. MeanPoolBaseline ==========
    logger.info("=" * 60)
    logger.info("Training MeanPoolBaseline")
    logger.info("=" * 60)
    model_mp = MeanPoolBaseline(
        feature_dim=args.feature_dim, num_classes=args.num_classes,
    )
    params_mp = sum(p.numel() for p in model_mp.parameters())

    t0 = time.time()
    train_baseline(model_mp, train_loader, val_loader, device, epochs=args.epochs, name="MeanPool")
    train_time_mp = time.time() - t0
    results["MeanPool"] = full_eval(model_mp, val_loader, device)
    results["MeanPool"]["params"] = params_mp
    results["MeanPool"]["train_time_sec"] = round(train_time_mp, 1)
    logger.info(f"  -> Top-1: {results['MeanPool']['top1_accuracy']:.4f}")
    del model_mp
    torch.cuda.empty_cache()

    # ========== Summary ==========
    logger.info("\n" + "=" * 80)
    logger.info("BASELINE COMPARISON RESULTS")
    logger.info("=" * 80)
    logger.info(f"{'Model':<30} {'Top-1':>8} {'Top-5':>8} {'TC':>8} {'Params':>12}")
    logger.info("-" * 80)
    for name, res in results.items():
        logger.info(
            f"{name:<30} {res['top1_accuracy']:8.4f} {res['top5_accuracy']:8.4f} "
            f"{res['temporal_consistency']:8.4f} {res['params']:>12,}"
        )
    logger.info("=" * 80)

    # Save
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Results saved to {args.output}")


if __name__ == "__main__":
    main()
