#!/usr/bin/env python3
"""Train and evaluate ALL baselines on 8 GPUs (DDP), then produce comparison table.

Usage:
    torchrun --nproc_per_node=8 -m temporalfusion.train_baselines \
        --config configs/train_activitynet.yaml

Each baseline is trained sequentially (full DDP training), then evaluated.
Results are saved to eval_results/baseline_comparison.json.
"""

import argparse
import json
import logging
import math
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Tuple

import yaml
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim.lr_scheduler import LambdaLR
from torch.cuda.amp import GradScaler, autocast
from sklearn.metrics import accuracy_score, top_k_accuracy_score
import wandb
from tqdm import tqdm

from temporalfusion.baselines import (
    DirectTransformerBaseline,
    TemporalSegmentBaseline,
    MeanPoolBaseline,
)
from temporalfusion.data import build_dataloaders

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("baselines")

# ========================================================================
# Defaults
# ========================================================================
DEFAULTS = dict(
    dataset="activitynet",
    train_dir="./data/activitynet/c3d_features/train",
    val_dir="./data/activitynet/c3d_features/val",
    annotations="./data/activitynet/gt.json",
    max_frames=512,
    feature_dim=2048,
    hidden_dim=1024,
    num_heads=8,
    num_layers=6,
    num_classes=200,
    batch_size=32,
    epochs=50,
    lr=3e-4,
    weight_decay=1e-4,
    warmup_epochs=5,
    num_workers=6,
    grad_accum_steps=2,
    label_smoothing=0.1,
    multi_label=0,
    amp=1,
    cudnn_benchmark=1,
    output_dir="./runs",
    seed=42,
    wandb_project="temporalfusion",
    log_every=20,
    eval_every=1,
    save_every=10,
)


# ========================================================================
# Argument parsing (mirrors training.py style)
# ========================================================================
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Baseline training + evaluation")
    p.add_argument("--config", type=str, default=None)
    for key, val in DEFAULTS.items():
        ty = type(val) if val is not None else str
        p.add_argument(f"--{key}", type=ty, default=None)
    p.add_argument("--local_rank", type=int, default=-1)
    cli = p.parse_args()

    merged = dict(DEFAULTS)
    if cli.config is not None:
        with open(cli.config) as fh:
            cfg = yaml.safe_load(fh)
        _flat = {}
        _flat["dataset"] = cfg.get("dataset", merged["dataset"])
        _MAP = dict(
            data=["train_dir", "val_dir", "annotations", "max_frames"],
            model=["feature_dim", "hidden_dim", "num_heads", "num_layers",
                    "num_classes"],
            training=["batch_size", "epochs", "lr", "weight_decay",
                       "warmup_epochs", "num_workers", "seed",
                       "label_smoothing", "grad_accum_steps", "multi_label"],
            performance=["amp", "cudnn_benchmark"],
            logging=["wandb_project", "log_every", "eval_every",
                      "save_every", "output_dir"],
        )
        for section, keys in _MAP.items():
            sub = cfg.get(section, {})
            for k in keys:
                if k in sub:
                    _flat[k] = sub[k]
        merged.update(_flat)

    for key in DEFAULTS:
        cli_val = getattr(cli, key, None)
        if cli_val is not None:
            merged[key] = cli_val
    for k, v in merged.items():
        setattr(cli, k, v)
    return cli


# ========================================================================
# DDP helpers
# ========================================================================
def setup_distributed():
    if "WORLD_SIZE" in os.environ and int(os.environ["WORLD_SIZE"]) > 1:
        dist.init_process_group(backend="nccl")
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        world_size = dist.get_world_size()
        torch.cuda.set_device(local_rank)
        return local_rank, world_size, True
    device_id = 0 if torch.cuda.is_available() else -1
    if device_id >= 0:
        torch.cuda.set_device(device_id)
    return device_id, 1, False


def cleanup():
    if dist.is_initialized():
        dist.destroy_process_group()


def barrier():
    if dist.is_initialized():
        dist.barrier()


# ========================================================================
# Simple classification-only loss for baselines
# ========================================================================
class BaselineLoss(nn.Module):
    def __init__(self, num_classes: int = 200, label_smoothing: float = 0.1,
                 multi_label: bool = False):
        super().__init__()
        self.multi_label = multi_label
        if multi_label:
            self.cls = nn.BCEWithLogitsLoss()
        else:
            self.cls = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

    def forward(self, model_output, labels=None, **kwargs):
        losses = {}
        total = torch.tensor(0.0, device=model_output["logits"].device)
        if labels is not None:
            losses["cls"] = self.cls(model_output["logits"], labels)
            total = total + losses["cls"]
        losses["total"] = total
        return losses


# ========================================================================
# Training one epoch
# ========================================================================
def train_one_epoch(model, criterion, optimizer, loader, device, epoch,
                    scaler, use_amp, grad_accum, log_every, per_gpu_bs):
    model.train()
    running_loss = 0.0
    comp = {}
    n = 0
    t0 = time.time()
    is_main = int(os.environ.get("LOCAL_RANK", 0)) <= 0
    bar = tqdm(loader, desc=f"  Epoch {epoch}", disable=not is_main, ncols=120)
    optimizer.zero_grad(set_to_none=True)

    for step, batch in enumerate(bar):
        feat = batch["features"].to(device, non_blocking=True)
        mask = batch["masks"].to(device, non_blocking=True)
        lbl = batch["labels"].to(device, non_blocking=True)
        is_ml = batch.get("multi_label", False)

        if is_ml:
            has_lbl = lbl.sum() > 0
        else:
            has_lbl = (lbl >= 0).any()

        with autocast(enabled=use_amp):
            out = model(feat, mask)
            losses = criterion(out, labels=lbl if has_lbl else None)
            loss = losses["total"] / grad_accum

        if scaler.is_enabled():
            scaler.scale(loss).backward()
        else:
            loss.backward()

        running_loss += losses["total"].item()
        for k, v in losses.items():
            comp[k] = comp.get(k, 0.0) + v.item()
        n += 1

        if (step + 1) % grad_accum == 0:
            if scaler.is_enabled():
                scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            if scaler.is_enabled():
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        if n % log_every == 0 and is_main:
            dt = time.time() - t0
            sps = n * per_gpu_bs / max(dt, 1e-6)
            bar.set_postfix(loss=f"{running_loss/n:.4f}", sps=f"{sps:.0f}")

    # flush remaining grads
    if n % grad_accum != 0:
        if scaler.is_enabled():
            scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        if scaler.is_enabled():
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()
        optimizer.zero_grad(set_to_none=True)

    dt = time.time() - t0
    metrics = {k: v / max(n, 1) for k, v in comp.items()}
    metrics["epoch_sec"] = dt
    metrics["samples_per_sec"] = n * per_gpu_bs / max(dt, 1e-6)
    return metrics


# ========================================================================
# Validation
# ========================================================================
@torch.no_grad()
def validate(model, criterion, loader, device, use_amp=False):
    model.eval()
    tot_loss = 0.0
    correct = total = 0
    n = 0
    # Multi-label support
    all_logits_ml = []
    all_labels_ml = []
    is_ml = False

    for batch in loader:
        feat = batch["features"].to(device, non_blocking=True)
        mask = batch["masks"].to(device, non_blocking=True)
        lbl = batch["labels"].to(device, non_blocking=True)
        is_ml = batch.get("multi_label", False)

        with autocast(enabled=use_amp):
            out = model(feat, mask)
            if is_ml:
                has_lbl = lbl.sum() > 0
            else:
                has_lbl = (lbl >= 0).any()
            losses = criterion(out, labels=lbl if has_lbl else None)

        tot_loss += losses["total"].item()
        n += 1

        if is_ml:
            all_logits_ml.append(out["logits"].float().cpu())
            all_labels_ml.append(lbl.float().cpu())
        else:
            if has_lbl:
                valid = lbl >= 0
                preds = out["logits"][valid].argmax(-1)
                correct += (preds == lbl[valid]).sum().item()
                total += valid.sum().item()

    result = dict(val_loss=tot_loss / max(n, 1))

    if is_ml and all_logits_ml:
        from sklearn.metrics import average_precision_score
        import warnings
        logits_cat = torch.cat(all_logits_ml, dim=0).numpy()
        labels_cat = torch.cat(all_labels_ml, dim=0).numpy()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            mAP = float(average_precision_score(labels_cat, logits_cat, average="macro"))
        result["val_mAP"] = mAP
        result["val_accuracy"] = mAP  # also store as accuracy for compat
    else:
        result["val_accuracy"] = correct / max(total, 1)

    return result


# ========================================================================
# Full evaluation (temporal consistency, feature quality, classification)
# ========================================================================
@torch.no_grad()
def full_evaluate(model, loader, device, use_amp=False, multi_label=False):
    """Run comprehensive evaluation on a baseline model."""
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []
    # Multi-label
    all_logits_ml = []
    all_labels_ml = []
    tc_sim = 0.0
    tc_pairs = 0.0
    class_features = {}

    for batch in tqdm(loader, desc="  Evaluating", disable=int(os.environ.get("LOCAL_RANK", 0)) > 0):
        feat = batch["features"].to(device, non_blocking=True)
        mask = batch["masks"].to(device, non_blocking=True)
        labels = batch["labels"]

        with autocast(enabled=use_amp):
            out = model(feat, mask)

        logits = out["logits"].float().cpu()
        frame_feat = out["frame_features"].float()

        # temporal consistency
        h = F.normalize(frame_feat, dim=-1)
        cs = (h[:, :-1] * h[:, 1:]).sum(-1)
        pm = mask[:, :-1] * mask[:, 1:]
        tc_sim += (cs * pm).sum().item()
        tc_pairs += pm.sum().item()

        if multi_label:
            all_logits_ml.append(logits)
            all_labels_ml.append(labels.float())
            # feature quality: use primary class (argmax of label vector) for grouping
            video_repr = out["video_repr"].cpu().numpy()
            for i in range(labels.shape[0]):
                lbl_vec = labels[i]
                if lbl_vec.sum() > 0:
                    primary_cls = int(lbl_vec.argmax().item())
                    class_features.setdefault(primary_cls, []).append(video_repr[i])
        else:
            probs = F.softmax(logits, dim=-1).numpy()
            # classification
            for i, lbl in enumerate(labels.numpy()):
                if lbl >= 0:
                    all_preds.append(logits[i].argmax().item())
                    all_labels.append(lbl)
                    all_probs.append(probs[i])

            # feature quality (video-level repr)
            video_repr = out["video_repr"].cpu().numpy()
            for i, lbl in enumerate(labels.numpy()):
                if lbl >= 0:
                    class_features.setdefault(int(lbl), []).append(video_repr[i])

    tc_mean = tc_sim / max(tc_pairs, 1)

    if multi_label:
        from sklearn.metrics import average_precision_score
        import warnings
        logits_cat = torch.cat(all_logits_ml, dim=0).numpy()
        labels_cat = torch.cat(all_labels_ml, dim=0).numpy()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            mAP = float(average_precision_score(labels_cat, logits_cat, average="macro"))
        classification_results = {
            "mAP": mAP,
            "top1_accuracy": mAP,  # compat key
            "top5_accuracy": mAP,  # compat key
            "num_samples": logits_cat.shape[0],
        }
    else:
        all_labels_np = np.array(all_labels)
        all_preds_np = np.array(all_preds)
        all_probs_np = np.stack(all_probs) if all_probs else np.empty((0, 200))

        top1 = float(accuracy_score(all_labels_np, all_preds_np))
        nc = all_probs_np.shape[1] if all_probs_np.ndim == 2 else 200
        all_class_labels = list(range(nc))
        top5 = float(top_k_accuracy_score(
            all_labels_np, all_probs_np, k=min(5, nc), labels=all_class_labels
        )) if nc >= 5 else top1
        classification_results = {
            "top1_accuracy": top1,
            "top5_accuracy": top5,
            "num_samples": len(all_labels),
        }

    # feature quality
    intra_dists = []
    class_centroids = {}
    for c, feats in class_features.items():
        feats_arr = np.stack(feats)
        centroid = feats_arr.mean(axis=0)
        class_centroids[c] = centroid
        if len(feats_arr) > 1:
            fn = feats_arr / (np.linalg.norm(feats_arr, axis=1, keepdims=True) + 1e-8)
            sm = fn @ fn.T
            mask_tri = np.triu(np.ones_like(sm), k=1).astype(bool)
            intra_dists.append(1.0 - sm[mask_tri].mean())
    if class_centroids:
        centroids = np.stack(list(class_centroids.values()))
        cn = centroids / (np.linalg.norm(centroids, axis=1, keepdims=True) + 1e-8)
        inter_sim = cn @ cn.T
        inter_mask = np.triu(np.ones_like(inter_sim), k=1).astype(bool)
        inter = float(1.0 - inter_sim[inter_mask].mean()) if inter_mask.any() else 0.0
    else:
        inter = 0.0
    intra = float(np.mean(intra_dists)) if intra_dists else 0.0

    nparams = sum(p.numel() for p in model.parameters())
    raw = model.module if hasattr(model, "module") else model
    nparams = sum(p.numel() for p in raw.parameters())

    return {
        "temporal_consistency": {
            "temporal_consistency_mean": tc_mean,
        },
        "feature_quality": {
            "intra_class_distance": intra,
            "inter_class_distance": inter,
            "separation": inter - intra,
            "num_classes_evaluated": len(class_features),
        },
        "classification": classification_results,
        "model_info": {
            "total_params": nparams,
        },
    }


# ========================================================================
# Train one baseline end-to-end
# ========================================================================
def train_baseline(
    name: str,
    model: nn.Module,
    args,
    device: torch.device,
    local_rank: int,
    world_size: int,
    is_ddp: bool,
    is_main: bool,
):
    """Train a single baseline model, evaluate, return results dict."""

    eff_bs = args.batch_size * world_size * args.grad_accum_steps
    base_scaled_lr = args.lr * (eff_bs / 32.0)
    # DirectTransformer needs a gentler LR -- pure CE without auxiliary
    # stabilizing losses (temporal contrastive, collapse prevention)
    lr_scale = {"DirectTransformer": 0.1}.get(name, 1.0)
    scaled_lr = base_scaled_lr * lr_scale
    use_amp = bool(args.amp)

    model = model.to(device)
    nparams = sum(p.numel() for p in model.parameters())

    if is_main:
        log.info("=" * 70)
        log.info(f"BASELINE: {name}")
        log.info(f"  Params        : {nparams:,} ({nparams/1e6:.1f}M)")
        log.info(f"  GPUs          : {world_size}")
        log.info(f"  Per-GPU BS    : {args.batch_size}")
        log.info(f"  Grad accum    : {args.grad_accum_steps}")
        log.info(f"  Effective BS  : {eff_bs}")
        log.info(f"  Scaled LR     : {scaled_lr:.6f}")
        log.info(f"  Epochs        : {args.epochs}")
        log.info("=" * 70)

    if is_ddp:
        model = DDP(model, device_ids=[local_rank], find_unused_parameters=False)

    # Reset label map to avoid stale state across baselines
    from temporalfusion.data import ActivityNetFeaturesDataset, THUMOS14FeaturesDataset
    ActivityNetFeaturesDataset._label_map = None
    THUMOS14FeaturesDataset._label_map = None

    train_loader, val_loader = build_dataloaders(
        dataset_name=args.dataset,
        train_dir=args.train_dir,
        val_dir=args.val_dir,
        annotations_file=args.annotations,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        max_frames=args.max_frames,
        distributed=is_ddp,
    )
    if is_main:
        log.info(f"  train: {len(train_loader.dataset)} videos ({len(train_loader)} batches/gpu)")
        log.info(f"  val  : {len(val_loader.dataset)} videos ({len(val_loader)} batches/gpu)")

    optimizer = torch.optim.AdamW(model.parameters(), lr=scaled_lr,
                                  weight_decay=args.weight_decay)
    wu = args.warmup_epochs
    tot = args.epochs
    scheduler = LambdaLR(optimizer, lambda ep: (
        (ep + 1) / max(1, wu) if ep < wu
        else max(0.01, 0.5 * (1.0 + math.cos(math.pi * (ep - wu) / max(1, tot - wu))))
    ))
    criterion = BaselineLoss(num_classes=args.num_classes,
                             label_smoothing=args.label_smoothing,
                             multi_label=bool(args.multi_label)).to(device)
    scaler = GradScaler(enabled=use_amp)

    # run directory
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_name = name.lower().replace(" ", "_")
    run_dir = Path(args.output_dir) / f"baseline_{safe_name}_{ts}"
    if is_main:
        run_dir.mkdir(parents=True, exist_ok=True)

    # wandb
    if is_main:
        wandb.init(
            project=args.wandb_project,
            name=f"baseline_{safe_name}",
            config={"baseline": name, "params": nparams, "eff_bs": eff_bs,
                    "scaled_lr": scaled_lr, "epochs": args.epochs,
                    "world_size": world_size},
            reinit=True,
        )

    best_val = float("inf")
    t_total = time.time()

    for epoch in range(args.epochs):
        if is_ddp:
            train_loader.sampler.set_epoch(epoch)

        tm = train_one_epoch(
            model, criterion, optimizer, train_loader, device, epoch,
            scaler, use_amp, args.grad_accum_steps, args.log_every,
            args.batch_size,
        )
        scheduler.step()

        if is_main:
            lr_now = optimizer.param_groups[0]["lr"]
            log.info(
                f"  [{name}] Epoch {epoch:3d} | "
                f"loss {tm['total']:.4f} | "
                f"lr {lr_now:.6f} | "
                f"{tm['samples_per_sec']:.0f} s/s | "
                f"{tm['epoch_sec']:.1f}s"
            )
            wandb.log({
                **{f"train/{k}": v for k, v in tm.items()},
                "lr": lr_now, "epoch": epoch,
            }, step=epoch)

        if epoch % args.eval_every == 0 or epoch == args.epochs - 1:
            vm = validate(model, criterion, val_loader, device, use_amp)
            if is_main:
                metric_name = "mAP" if bool(args.multi_label) else "acc"
                metric_val = vm.get("val_mAP", vm.get("val_accuracy", 0))
                log.info(
                    f"  [{name}]    val | "
                    f"loss {vm['val_loss']:.4f} | "
                    f"{metric_name} {metric_val:.4f}"
                )
                wandb.log({f"val/{k}": v for k, v in vm.items()}, step=epoch)
                if vm["val_loss"] < best_val:
                    best_val = vm["val_loss"]
                    raw = model.module if is_ddp else model
                    torch.save(dict(
                        model=raw.state_dict(),
                        epoch=epoch, val_metrics=vm,
                    ), run_dir / "best_model.pt")
                    log.info(f"  >> saved best (val_loss={best_val:.4f})")

        if is_main and (epoch % args.save_every == 0 or epoch == args.epochs - 1):
            raw = model.module if is_ddp else model
            torch.save(dict(model=raw.state_dict(), epoch=epoch),
                       run_dir / f"ckpt_ep{epoch}.pt")

    train_time = time.time() - t_total

    # ---- full evaluation on best checkpoint (rank 0 only, full val set) ----
    if is_main:
        log.info(f"  [{name}] Running full evaluation on best checkpoint ...")

    barrier()

    if is_main:
        # Build a fresh non-distributed val loader to evaluate on ALL samples
        from temporalfusion.data import (
            ActivityNetFeaturesDataset, THUMOS14FeaturesDataset,
            CharadesFeaturesDataset, collate_features,
        )
        from pathlib import Path as _Path
        from torch.utils.data import DataLoader as DL
        ActivityNetFeaturesDataset._label_map = None  # reset for clean load
        THUMOS14FeaturesDataset._label_map = None
        if args.dataset == "thumos14":
            eval_ds = THUMOS14FeaturesDataset(
                args.val_dir, args.annotations, "test", args.max_frames
            )
        elif args.dataset == "charades":
            ann_dir = _Path(args.annotations) if args.annotations else None
            test_csv = str(ann_dir / "Charades_v1_test.csv") if ann_dir else None
            eval_ds = CharadesFeaturesDataset(
                args.val_dir, test_csv, "test", args.max_frames
            )
        else:
            eval_ds = ActivityNetFeaturesDataset(
                args.val_dir, args.annotations, "val", args.max_frames
            )
        eval_loader = DL(
            eval_ds, batch_size=args.batch_size, shuffle=False,
            num_workers=args.num_workers, collate_fn=collate_features,
            pin_memory=True,
        )
        # load best checkpoint on the raw (unwrapped) model
        best_ckpt = run_dir / "best_model.pt"
        raw = model.module if is_ddp else model
        if best_ckpt.exists():
            ckpt = torch.load(best_ckpt, map_location=device)
            raw.load_state_dict(ckpt["model"])

        eval_results = full_evaluate(raw, eval_loader, device, use_amp,
                                      multi_label=bool(args.multi_label))
        eval_results["training_time_sec"] = train_time
        eval_results["training_epochs"] = args.epochs

        # save per-baseline results
        out_path = Path("eval_results") / f"baseline_{safe_name}.json"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(eval_results, f, indent=2)
        log.info(f"  [{name}] Results saved to {out_path}")
        if bool(args.multi_label):
            log.info(
                f"  [{name}] mAP={eval_results['classification'].get('mAP', 0):.4f} | "
                f"tc={eval_results['temporal_consistency']['temporal_consistency_mean']:.4f}"
            )
        else:
            log.info(
                f"  [{name}] top1={eval_results['classification']['top1_accuracy']:.4f} | "
                f"top5={eval_results['classification']['top5_accuracy']:.4f} | "
                f"tc={eval_results['temporal_consistency']['temporal_consistency_mean']:.4f}"
            )
        wandb.finish()
    else:
        eval_results = {}

    barrier()

    # unwrap DDP before returning
    if is_ddp:
        model = model.module

    return eval_results


# ========================================================================
# Main
# ========================================================================
def main():
    args = parse_args()
    local_rank, world_size, is_ddp = setup_distributed()
    device = torch.device(f"cuda:{local_rank}" if local_rank >= 0 else "cpu")
    is_main = local_rank <= 0

    torch.manual_seed(args.seed + local_rank)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed + local_rank)
    if args.cudnn_benchmark:
        torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    if is_main:
        log.info("#" * 70)
        log.info("#  BASELINE TRAINING & EVALUATION SUITE")
        log.info(f"#  GPUs: {world_size} | Dataset: {args.dataset}")
        log.info("#" * 70)

    # define all baselines
    baseline_configs = [
        (
            "DirectTransformer",
            lambda: DirectTransformerBaseline(
                feature_dim=args.feature_dim,
                hidden_dim=args.hidden_dim,
                num_heads=args.num_heads,
                num_layers=args.num_layers,
                num_classes=args.num_classes,
            ),
        ),
        (
            "TemporalSegment",
            lambda: TemporalSegmentBaseline(
                feature_dim=args.feature_dim,
                num_segments=8,
                num_classes=args.num_classes,
            ),
        ),
        (
            "MeanPool",
            lambda: MeanPoolBaseline(
                feature_dim=args.feature_dim,
                num_classes=args.num_classes,
            ),
        ),
    ]

    all_results = {}

    for name, model_fn in baseline_configs:
        model = model_fn()
        results = train_baseline(
            name=name,
            model=model,
            args=args,
            device=device,
            local_rank=local_rank,
            world_size=world_size,
            is_ddp=is_ddp,
            is_main=is_main,
        )
        all_results[name] = results
        # free GPU memory between baselines
        del model
        torch.cuda.empty_cache()

    # ---- load TemporalFusion results for comparison ----
    tf_candidates = [
        Path(f"eval_results/{args.dataset}_full_eval.json"),
        Path("eval_results/activitynet_full_eval.json"),
        Path("eval_results/thumos14_full_eval.json"),
    ]
    for tf_path in tf_candidates:
        if tf_path.exists():
            with open(tf_path) as f:
                all_results["TemporalFusion (Ours)"] = json.load(f)
            break

    # ---- save combined comparison ----
    if is_main:
        out = Path("eval_results/baseline_comparison.json")
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w") as f:
            json.dump(all_results, f, indent=2)
        log.info(f"\nCombined results saved to {out}")

        # ---- print comparison table ----
        is_ml = bool(args.multi_label)
        log.info("")
        log.info("=" * 90)
        if is_ml:
            log.info(f"{'Model':<25} {'Params':>10} {'mAP':>8} {'TC':>8} {'Separation':>12}")
        else:
            log.info(f"{'Model':<25} {'Params':>10} {'Top-1':>8} {'Top-5':>8} {'TC':>8} {'Separation':>12}")
        log.info("-" * 90)
        for mname, res in all_results.items():
            cls = res.get("classification", {})
            tc = res.get("temporal_consistency", {})
            fq = res.get("feature_quality", {})
            mi = res.get("model_info", {})
            npar = mi.get("total_params", mi.get("total", 0))
            pstr = f"{npar/1e6:.1f}M" if npar > 0 else "N/A"
            if is_ml:
                log.info(
                    f"{mname:<25} {pstr:>10} "
                    f"{cls.get('mAP', cls.get('top1_accuracy', 0))*100:>7.2f}% "
                    f"{tc.get('temporal_consistency_mean', 0):>7.4f} "
                    f"{fq.get('separation', 0):>11.4f}"
                )
            else:
                log.info(
                    f"{mname:<25} {pstr:>10} "
                    f"{cls.get('top1_accuracy', 0)*100:>7.2f}% "
                    f"{cls.get('top5_accuracy', 0)*100:>7.2f}% "
                    f"{tc.get('temporal_consistency_mean', 0):>7.4f} "
                    f"{fq.get('separation', 0):>11.4f}"
                )
        log.info("=" * 90)

    cleanup()


if __name__ == "__main__":
    main()
