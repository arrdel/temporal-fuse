"""
Ablation study for TemporalFusion.

Runs all ablation variants on a given dataset using 8-GPU DDP training,
then evaluates each and produces a comparison table.

Ablation variants:
  1. Full model (all losses)
  2. w/o Temporal Contrastive (lambda_tc=0)
  3. w/o Collapse Prevention (lambda_reg=0)
  4. w/o Cross-Scale Consistency (lambda_cs=0)
  5. Classification Only (lambda_tc=0, lambda_reg=0, lambda_cs=0)

Usage:
  torchrun --nproc_per_node=8 -m temporalfusion.run_ablations \
      --config configs/train_thumos14.yaml

  torchrun --nproc_per_node=8 -m temporalfusion.run_ablations \
      --config configs/train_charades.yaml
"""

import argparse
import copy
import json
import logging
import math
import os
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import yaml
from sklearn.metrics import average_precision_score
from torch.cuda.amp import GradScaler, autocast
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from tqdm import tqdm

import wandb

from temporalfusion.data import build_dataloaders, collate_features
from temporalfusion.losses import TemporalFusionLoss
from temporalfusion.model import TemporalFusionModel

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("ablation")

# -------------------------------------------------------------------------
# Ablation variants
# -------------------------------------------------------------------------
ABLATIONS = {
    "full": {
        "description": "Full model (all losses)",
        "overrides": {},
    },
    "no_temporal": {
        "description": "w/o Temporal Contrastive Loss",
        "overrides": {"lambda_tc": 0.0},
    },
    "no_collapse": {
        "description": "w/o Collapse Prevention",
        "overrides": {"lambda_reg": 0.0},
    },
    "no_crossscale": {
        "description": "w/o Cross-Scale Consistency",
        "overrides": {"lambda_cs": 0.0},
    },
    "cls_only": {
        "description": "Classification Only (no auxiliary losses)",
        "overrides": {"lambda_tc": 0.0, "lambda_reg": 0.0, "lambda_cs": 0.0},
    },
}


# -------------------------------------------------------------------------
# Config loading (reused from training.py)
# -------------------------------------------------------------------------
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
    num_hierarchy_levels=4,
    batch_size=64,
    epochs=50,
    lr=1e-4,
    weight_decay=1e-4,
    warmup_epochs=5,
    num_workers=6,
    grad_accum_steps=1,
    multi_label=0,
    lambda_tc=1.0,
    lambda_reg=0.1,
    lambda_cs=0.5,
    lambda_vl=0.1,
    collapse_tau=0.1,
    label_smoothing=0.1,
    amp=1,
    compile_model=0,
    cudnn_benchmark=1,
    output_dir="./runs",
    run_name=None,
    seed=42,
    wandb_project="temporalfusion",
    log_every=5,
    eval_every=1,
    save_every=10,
    resume=None,
)

_MAP = dict(
    data=["train_dir", "val_dir", "annotations", "max_frames"],
    model=["feature_dim", "hidden_dim", "num_heads", "num_layers",
            "num_classes", "num_hierarchy_levels"],
    training=["batch_size", "epochs", "lr", "weight_decay",
              "warmup_epochs", "num_workers", "seed",
              "label_smoothing", "grad_accum_steps", "multi_label"],
    loss=["lambda_tc", "lambda_reg", "lambda_cs", "lambda_vl",
          "collapse_tau"],
    performance=["amp", "compile_model", "cudnn_benchmark"],
    logging=["wandb_project", "log_every", "eval_every",
             "save_every", "output_dir"],
)


def load_config(config_path: str) -> dict:
    merged = dict(DEFAULTS)
    if config_path:
        with open(config_path) as fh:
            cfg = yaml.safe_load(fh)
        flat = {}
        flat["dataset"] = cfg.get("dataset", merged["dataset"])
        for section, keys in _MAP.items():
            sub = cfg.get(section, {})
            if isinstance(sub, dict):
                for k in keys:
                    if k in sub:
                        flat[k] = sub[k]
        merged.update(flat)
    return merged


# -------------------------------------------------------------------------
# DDP setup
# -------------------------------------------------------------------------
def setup_distributed():
    if "RANK" in os.environ:
        dist.init_process_group("nccl", timeout=timedelta(minutes=120))
        local_rank = int(os.environ["LOCAL_RANK"])
        world_size = dist.get_world_size()
        torch.cuda.set_device(local_rank)
        return local_rank, world_size, True
    return 0, 1, False


def cleanup():
    if dist.is_initialized():
        dist.destroy_process_group()


# -------------------------------------------------------------------------
# Training loop for one ablation variant
# -------------------------------------------------------------------------
def train_one_epoch(model, criterion, optimizer, loader, device, epoch,
                    scaler, use_amp, grad_accum, log_every, batch_size,
                    multi_label=False):
    model.train()
    totals = {}
    n_batches = 0
    optimizer.zero_grad(set_to_none=True)
    t0 = time.time()
    n_samples = 0

    pbar = tqdm(loader, desc=f"Epoch {epoch}", leave=True,
                disable=(int(os.environ.get("LOCAL_RANK", 0)) != 0))

    for i, batch in enumerate(pbar):
        feat = batch["features"].to(device, non_blocking=True)
        mask = batch["masks"].to(device, non_blocking=True)
        lbl = batch["labels"].to(device, non_blocking=True)

        with autocast(enabled=use_amp):
            out = model(feat, mask)
            if multi_label:
                has_lbl = lbl.sum() > 0
            else:
                has_lbl = (lbl >= 0).any()
            losses = criterion(out, labels=lbl if has_lbl else None, mask=mask)
            loss = losses["total"] / grad_accum

        scaler.scale(loss).backward()

        if (i + 1) % grad_accum == 0 or (i + 1) == len(loader):
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        n_batches += 1
        n_samples += feat.size(0)
        for k, v in losses.items():
            totals[k] = totals.get(k, 0.0) + v.item()

        if n_batches % log_every == 0:
            pbar.set_postfix(loss=f"{totals['total']/n_batches:.4f}",
                             sps=f"{n_samples/(time.time()-t0):.0f}")

    elapsed = time.time() - t0
    metrics = {k: v / max(n_batches, 1) for k, v in totals.items()}
    metrics["epoch_sec"] = elapsed
    metrics["samples_per_sec"] = n_samples / max(elapsed, 1e-6)
    return metrics


@torch.no_grad()
def evaluate(model, criterion, loader, device, use_amp=False, multi_label=False,
             num_classes=20):
    model.eval()
    tot_loss = 0.0
    correct = total = 0
    n = 0
    tc_sim = tc_pairs = 0.0
    all_preds, all_labels = [], []

    for batch in loader:
        feat = batch["features"].to(device, non_blocking=True)
        mask = batch["masks"].to(device, non_blocking=True)
        lbl = batch["labels"].to(device, non_blocking=True)

        with autocast(enabled=use_amp):
            out = model(feat, mask)
            if multi_label:
                has_lbl = lbl.sum() > 0
            else:
                has_lbl = (lbl >= 0).any()
            losses = criterion(out, labels=lbl if has_lbl else None, mask=mask)

        tot_loss += losses["total"].item()
        n += 1

        if multi_label:
            probs = torch.sigmoid(out["logits"]).cpu().numpy()
            all_preds.append(probs)
            all_labels.append(lbl.cpu().numpy())
        else:
            if has_lbl:
                valid = lbl >= 0
                preds = out["logits"][valid].argmax(-1)
                correct += (preds == lbl[valid]).sum().item()
                total += valid.sum().item()

        h = F.normalize(out["frame_features"].float(), dim=-1)
        cs = (h[:, :-1] * h[:, 1:]).sum(-1)
        pm = mask[:, :-1] * mask[:, 1:]
        tc_sim += (cs * pm).sum().item()
        tc_pairs += pm.sum().item()

    result = dict(
        val_loss=tot_loss / max(n, 1),
        val_temporal_consistency=tc_sim / max(tc_pairs, 1),
    )

    if multi_label:
        all_preds = np.concatenate(all_preds, axis=0)
        all_labels = np.concatenate(all_labels, axis=0)
        aps = []
        for c in range(num_classes):
            if all_labels[:, c].sum() > 0:
                aps.append(average_precision_score(all_labels[:, c], all_preds[:, c]))
        result["val_mAP"] = float(np.mean(aps)) if aps else 0.0
        result["val_accuracy"] = result["val_mAP"]
    else:
        result["val_accuracy"] = correct / max(total, 1)

    return result


# -------------------------------------------------------------------------
# Full evaluation on the complete test set (rank 0 only, no DDP sampler)
# -------------------------------------------------------------------------
def full_evaluate_model(model, args, device, multi_label=False):
    """Run evaluation on the full test set without DistributedSampler."""
    from temporalfusion.data import (
        ActivityNetFeaturesDataset,
        THUMOS14FeaturesDataset,
    )

    if args["dataset"] == "thumos14":
        ds = THUMOS14FeaturesDataset(
            feature_dir=args["val_dir"],
            annotations_file=args["annotations"],
            split="test",
            max_frames=args["max_frames"],
        )
    elif args["dataset"] == "charades":
        from temporalfusion.data import CharadesFeaturesDataset
        ann_dir = args["annotations"]
        split = "test"
        csv_name = f"Charades_v1_{split}.csv"
        csv_path = os.path.join(ann_dir, csv_name) if os.path.isdir(ann_dir) else ann_dir
        ds = CharadesFeaturesDataset(
            feature_dir=args["val_dir"],
            annotations_csv=csv_path,
            split=split,
            max_frames=args["max_frames"],
        )
    else:
        ds = ActivityNetFeaturesDataset(
            feature_dir=args["val_dir"],
            annotations_file=args["annotations"],
            split="validation",
            max_frames=args["max_frames"],
        )

    loader = DataLoader(
        ds, batch_size=32, shuffle=False, num_workers=4,
        collate_fn=collate_features, pin_memory=True,
    )

    model.eval()
    all_preds, all_labels = [], []
    tc_sim = tc_pairs = 0.0
    correct = total_count = 0

    with torch.no_grad():
        for batch in tqdm(loader, desc="Full eval", disable=False):
            feat = batch["features"].to(device)
            mask = batch["masks"].to(device)
            lbl = batch["labels"]
            out = model(feat, mask)

            if multi_label:
                probs = torch.sigmoid(out["logits"]).cpu().numpy()
                all_preds.append(probs)
                all_labels.append(lbl.numpy())
            else:
                valid = lbl >= 0
                if valid.any():
                    preds = out["logits"][valid].cpu().argmax(-1)
                    correct += (preds == lbl[valid]).sum().item()
                    total_count += valid.sum().item()

            h = F.normalize(out["frame_features"].float(), dim=-1)
            cs = (h[:, :-1] * h[:, 1:]).sum(-1)
            pm = mask[:, :-1] * mask[:, 1:]
            tc_sim += (cs * pm).sum().item()
            tc_pairs += pm.sum().item()

    results = {
        "temporal_consistency": float(tc_sim / max(tc_pairs, 1)),
        "num_samples": len(ds),
    }

    if multi_label:
        all_preds = np.concatenate(all_preds, axis=0)
        all_labels = np.concatenate(all_labels, axis=0)
        aps = []
        for c in range(args["num_classes"]):
            if all_labels[:, c].sum() > 0:
                aps.append(average_precision_score(all_labels[:, c], all_preds[:, c]))
        results["mAP"] = float(np.mean(aps)) if aps else 0.0
    else:
        from sklearn.metrics import top_k_accuracy_score
        results["top1_accuracy"] = correct / max(total_count, 1)

        # Recompute for top-5 with full probabilities
        all_preds2, all_labels2 = [], []
        model.eval()
        with torch.no_grad():
            for batch in loader:
                feat = batch["features"].to(device)
                mask = batch["masks"].to(device)
                lbl = batch["labels"]
                out = model(feat, mask)
                valid = lbl >= 0
                if valid.any():
                    all_preds2.append(out["logits"][valid].cpu().numpy())
                    all_labels2.append(lbl[valid].numpy())

        all_preds2 = np.concatenate(all_preds2, axis=0)
        all_labels2 = np.concatenate(all_labels2, axis=0)
        k = min(5, args["num_classes"])
        results["top5_accuracy"] = float(
            top_k_accuracy_score(
                all_labels2, all_preds2, k=k,
                labels=list(range(args["num_classes"]))
            )
        )

    return results


# -------------------------------------------------------------------------
# Train one ablation variant
# -------------------------------------------------------------------------
def run_ablation(ablation_name, ablation_cfg, base_args, local_rank,
                 world_size, is_ddp, device, is_main):
    """Train + evaluate one ablation variant."""

    args = copy.deepcopy(base_args)
    # Apply ablation overrides
    for k, v in ablation_cfg["overrides"].items():
        args[k] = v

    multi_label = bool(args.get("multi_label", 0))
    use_amp = bool(args["amp"])

    # Effective batch & LR
    eff_bs = args["batch_size"] * world_size * args["grad_accum_steps"]
    scaled_lr = args["lr"] * (eff_bs / 32.0)

    run_name = f"ablation_{args['dataset']}_{ablation_name}"
    run_dir = Path(args["output_dir"]) / run_name

    if is_main:
        run_dir.mkdir(parents=True, exist_ok=True)
        log.info(f"\n{'='*70}")
        log.info(f"ABLATION: {ablation_name} -- {ablation_cfg['description']}")
        log.info(f"{'='*70}")
        log.info(f"  lambda_tc  = {args['lambda_tc']}")
        log.info(f"  lambda_reg = {args['lambda_reg']}")
        log.info(f"  lambda_cs  = {args['lambda_cs']}")
        log.info(f"  Scaled LR  = {scaled_lr:.6f}")
        log.info(f"  Eff BS     = {eff_bs}")

        wandb.init(
            project=args["wandb_project"],
            name=run_name,
            config={**args, "ablation": ablation_name, "eff_bs": eff_bs,
                    "scaled_lr": scaled_lr},
            reinit=True,
        )

    # Data
    train_loader, val_loader = build_dataloaders(
        dataset_name=args["dataset"],
        train_dir=args["train_dir"],
        val_dir=args["val_dir"],
        annotations_file=args["annotations"],
        batch_size=args["batch_size"],
        num_workers=args["num_workers"],
        max_frames=args["max_frames"],
        distributed=is_ddp,
    )

    if is_main:
        log.info(f"  train: {len(train_loader.dataset)} videos ({len(train_loader)} batches/gpu)")
        log.info(f"  val  : {len(val_loader.dataset)} videos ({len(val_loader)} batches/gpu)")

    # Model
    model = TemporalFusionModel(
        feature_dim=args["feature_dim"],
        hidden_dim=args["hidden_dim"],
        num_heads=args["num_heads"],
        num_layers=args["num_layers"],
        num_classes=args["num_classes"],
        num_hierarchy_levels=args["num_hierarchy_levels"],
        max_seq_len=5000,
    ).to(device)

    if is_ddp:
        model = DDP(model, device_ids=[local_rank], find_unused_parameters=False)

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=scaled_lr, weight_decay=args["weight_decay"],
    )

    # Scheduler
    wu = args["warmup_epochs"]
    tot = args["epochs"]
    scheduler = LambdaLR(optimizer, lambda ep: (
        (ep + 1) / max(1, wu) if ep < wu
        else max(0.01, 0.5 * (1.0 + math.cos(
            math.pi * (ep - wu) / max(1, tot - wu))))
    ))

    # Loss
    criterion = TemporalFusionLoss(
        lambda_tc=args["lambda_tc"],
        lambda_reg=args["lambda_reg"],
        lambda_cs=args["lambda_cs"],
        lambda_vl=args.get("lambda_vl", 0.0),
        collapse_tau=args["collapse_tau"],
        num_classes=args["num_classes"],
        label_smoothing=args.get("label_smoothing", 0.1),
        multi_label=multi_label,
    ).to(device)

    scaler = GradScaler(enabled=use_amp)

    # Training loop
    best_val = float("inf")
    best_epoch = 0
    t_start = time.time()

    for epoch in range(args["epochs"]):
        if is_ddp:
            train_loader.sampler.set_epoch(epoch)

        tm = train_one_epoch(
            model, criterion, optimizer, train_loader, device, epoch,
            scaler, use_amp, args["grad_accum_steps"], args["log_every"],
            args["batch_size"], multi_label=multi_label,
        )
        scheduler.step()

        if is_main and epoch % args["eval_every"] == 0:
            lr_now = optimizer.param_groups[0]["lr"]
            vm = evaluate(model, criterion, val_loader, device, use_amp,
                          multi_label=multi_label, num_classes=args["num_classes"])

            metric_str = f"loss {vm['val_loss']:.4f}"
            if multi_label:
                metric_str += f" | mAP {vm.get('val_mAP', 0):.4f}"
            else:
                metric_str += f" | acc {vm['val_accuracy']:.4f}"
            metric_str += f" | tc {vm['val_temporal_consistency']:.4f}"

            log.info(
                f"  [{ablation_name}] Epoch {epoch:3d} | "
                f"train_loss {tm['total']:.4f} | {metric_str} | "
                f"lr {lr_now:.6f}"
            )

            wandb.log({
                **{f"train/{k}": v for k, v in tm.items()},
                **{f"val/{k}": v for k, v in vm.items()},
                "lr": lr_now, "epoch": epoch,
            }, step=epoch)

            if vm["val_loss"] < best_val:
                best_val = vm["val_loss"]
                best_epoch = epoch
                raw = model.module if is_ddp else model
                torch.save(dict(
                    model=raw.state_dict(),
                    epoch=epoch,
                    val_metrics=vm,
                    args=args,
                    ablation=ablation_name,
                ), run_dir / "best_model.pt")

    train_time = time.time() - t_start

    # Synchronize before evaluation so all ranks are at the same point
    if is_ddp:
        dist.barrier()

    # Full evaluation on rank 0 (non-distributed)
    results = {}
    if is_main:
        log.info(f"  [{ablation_name}] Running full evaluation (best epoch {best_epoch}) ...")
        ckpt = torch.load(run_dir / "best_model.pt", map_location=device)
        raw = model.module if is_ddp else model
        raw.load_state_dict(ckpt["model"])

        results = full_evaluate_model(raw, args, device, multi_label=multi_label)
        results["training_time_sec"] = train_time
        results["best_epoch"] = best_epoch
        results["ablation"] = ablation_name
        results["description"] = ablation_cfg["description"]
        results["lambda_tc"] = args["lambda_tc"]
        results["lambda_reg"] = args["lambda_reg"]
        results["lambda_cs"] = args["lambda_cs"]

        out_path = f"eval_results/ablation_{args['dataset']}_{ablation_name}.json"
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2)

        if multi_label:
            log.info(f"  [{ablation_name}] mAP={results['mAP']:.4f} | "
                     f"tc={results['temporal_consistency']:.4f}")
        else:
            log.info(f"  [{ablation_name}] top1={results['top1_accuracy']:.4f} | "
                     f"top5={results.get('top5_accuracy', 0):.4f} | "
                     f"tc={results['temporal_consistency']:.4f}")

        wandb.finish()

    # Synchronize after evaluation before proceeding to next ablation
    if is_ddp:
        dist.barrier()

    return results


# -------------------------------------------------------------------------
# Main
# -------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--ablations", type=str, default=None,
                        help="Comma-separated list of ablation names to run. "
                             "Default: all")
    parser.add_argument("--local_rank", type=int, default=-1)
    cli_args = parser.parse_args()

    local_rank, world_size, is_ddp = setup_distributed()
    device = torch.device(f"cuda:{local_rank}" if local_rank >= 0 else "cpu")
    is_main = local_rank <= 0

    base_args = load_config(cli_args.config)

    # Reproducibility
    torch.manual_seed(base_args["seed"] + local_rank)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(base_args["seed"] + local_rank)
    if base_args.get("cudnn_benchmark"):
        torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    # Select ablations
    if cli_args.ablations:
        ablation_names = [a.strip() for a in cli_args.ablations.split(",")]
    else:
        ablation_names = list(ABLATIONS.keys())

    if is_main:
        log.info(f"Ablation Study on {base_args['dataset']}")
        log.info(f"Variants: {ablation_names}")
        log.info(f"GPUs: {world_size}")
        os.makedirs("eval_results", exist_ok=True)

    all_results = {}
    for name in ablation_names:
        if name not in ABLATIONS:
            if is_main:
                log.warning(f"Unknown ablation '{name}', skipping")
            continue

        results = run_ablation(
            name, ABLATIONS[name], base_args,
            local_rank, world_size, is_ddp, device, is_main,
        )
        all_results[name] = results

    # Print summary table
    if is_main and all_results:
        multi_label = bool(base_args.get("multi_label", 0))
        dataset = base_args["dataset"]

        combined_path = f"eval_results/ablation_{dataset}_all.json"
        with open(combined_path, "w") as f:
            json.dump(all_results, f, indent=2)

        log.info(f"\n{'='*90}")
        log.info(f"ABLATION STUDY RESULTS -- {dataset.upper()}")
        log.info(f"{'='*90}")

        if multi_label:
            log.info(f"{'Variant':<30} {'tc':>6} {'reg':>6} {'cs':>6}   "
                     f"{'mAP':>8}   {'TC':>8}")
            log.info("-" * 90)
            for name in ablation_names:
                if name not in all_results or not all_results[name]:
                    continue
                r = all_results[name]
                log.info(
                    f"{ABLATIONS[name]['description']:<30} "
                    f"{r.get('lambda_tc', '?'):>6} {r.get('lambda_reg', '?'):>6} "
                    f"{r.get('lambda_cs', '?'):>6}   "
                    f"{r.get('mAP', 0)*100:>7.2f}%   "
                    f"{r.get('temporal_consistency', 0):>8.4f}"
                )
        else:
            log.info(f"{'Variant':<30} {'tc':>6} {'reg':>6} {'cs':>6}   "
                     f"{'Top-1':>8} {'Top-5':>8}   {'TC':>8}")
            log.info("-" * 90)
            for name in ablation_names:
                if name not in all_results or not all_results[name]:
                    continue
                r = all_results[name]
                log.info(
                    f"{ABLATIONS[name]['description']:<30} "
                    f"{r.get('lambda_tc', '?'):>6} {r.get('lambda_reg', '?'):>6} "
                    f"{r.get('lambda_cs', '?'):>6}   "
                    f"{r.get('top1_accuracy', 0)*100:>7.2f}% "
                    f"{r.get('top5_accuracy', 0)*100:>7.2f}%   "
                    f"{r.get('temporal_consistency', 0):>8.4f}"
                )

        log.info(f"{'='*90}")
        log.info(f"Results saved to {combined_path}")

    cleanup()


if __name__ == "__main__":
    main()
