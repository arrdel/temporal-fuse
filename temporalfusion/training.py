#!/usr/bin/env python3
"""TemporalFusion multi-GPU training script.

Usage (8-GPU DDP):
    torchrun --nproc_per_node=8 -m temporalfusion.training \
        --config configs/train_activitynet.yaml

Usage (single GPU):
    python -m temporalfusion.training \
        --config configs/train_activitynet.yaml --batch_size 32
"""

import argparse
import json
import logging
import math
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

import yaml
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim.lr_scheduler import LambdaLR
from torch.cuda.amp import GradScaler, autocast
import wandb
from tqdm import tqdm

from temporalfusion.model import TemporalFusionModel
from temporalfusion.losses import TemporalFusionLoss
from temporalfusion.data import build_dataloaders

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("temporalfusion")

# ---------------------------------------------------------------------------
# Default hyper-parameters (overridden by config yaml then CLI)
# ---------------------------------------------------------------------------
DEFAULTS = dict(
    # data
    dataset="activitynet",
    train_dir="./data/activitynet/c3d_features/train",
    val_dir="./data/activitynet/c3d_features/val",
    annotations="./data/activitynet/gt.json",
    max_frames=512,
    # model
    feature_dim=2048,
    hidden_dim=1024,
    num_heads=8,
    num_layers=6,
    num_classes=200,
    num_hierarchy_levels=4,
    # training
    batch_size=64,
    epochs=50,
    lr=1e-4,
    weight_decay=1e-4,
    warmup_epochs=5,
    num_workers=6,
    grad_accum_steps=1,
    multi_label=0,
    # loss
    lambda_tc=1.0,
    lambda_reg=0.1,
    lambda_cs=0.5,
    lambda_vl=0.1,
    collapse_tau=0.1,
    label_smoothing=0.1,
    # performance
    amp=1,
    compile_model=0,
    cudnn_benchmark=1,
    # infra
    output_dir="./runs",
    run_name=None,
    seed=42,
    wandb_project="temporalfusion",
    log_every=5,
    eval_every=1,
    save_every=10,
    resume=None,
)

# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, default=None)
    for key, val in DEFAULTS.items():
        ty = type(val) if val is not None else str
        p.add_argument(f"--{key}", type=ty, default=None)
    p.add_argument("--local_rank", type=int, default=-1)
    cli = p.parse_args()

    # layer 1: hard-coded defaults
    merged = dict(DEFAULTS)

    # layer 2: yaml config
    if cli.config is not None:
        with open(cli.config) as fh:
            cfg = yaml.safe_load(fh)
        _flat = {}
        _flat["dataset"] = cfg.get("dataset", merged["dataset"])
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
        for section, keys in _MAP.items():
            sub = cfg.get(section, {})
            for k in keys:
                if k in sub:
                    _flat[k] = sub[k]
        merged.update(_flat)

    # layer 3: explicit CLI overrides
    for key in DEFAULTS:
        cli_val = getattr(cli, key, None)
        if cli_val is not None:
            merged[key] = cli_val

    # write back into namespace
    for k, v in merged.items():
        setattr(cli, k, v)
    return cli


# ---------------------------------------------------------------------------
# Distributed helpers
# ---------------------------------------------------------------------------

def setup_distributed():
    """Returns (local_rank, world_size, is_ddp)."""
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


# ---------------------------------------------------------------------------
# One epoch of training
# ---------------------------------------------------------------------------

def train_one_epoch(
    model: nn.Module,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    loader,
    device: torch.device,
    epoch: int,
    scaler: GradScaler,
    use_amp: bool,
    grad_accum: int,
    log_every: int,
    per_gpu_bs: int,
) -> Dict[str, float]:
    model.train()
    running_loss = 0.0
    comp = {}
    n = 0
    t0 = time.time()
    is_main = int(os.environ.get("LOCAL_RANK", 0)) <= 0
    bar = tqdm(loader, desc=f"Epoch {epoch}", disable=not is_main, ncols=130)

    optimizer.zero_grad(set_to_none=True)

    for step, batch in enumerate(bar):
        feat = batch["features"].to(device, non_blocking=True)
        mask = batch["masks"].to(device, non_blocking=True)
        lbl = batch["labels"].to(device, non_blocking=True)
        is_ml = batch.get("multi_label", False)
        has_lbl = lbl.sum() > 0 if is_ml else (lbl >= 0).any()

        with autocast(enabled=use_amp):
            out = model(feat, mask)
            losses = criterion(out, labels=lbl if has_lbl else None, mask=mask)
            loss = losses["total"] / grad_accum

        if scaler.is_enabled():
            scaler.scale(loss).backward()
        else:
            loss.backward()

        running_loss += losses["total"].item()
        for k, v in losses.items():
            comp[k] = comp.get(k, 0.0) + v.item()
        n += 1

        # optimiser step every grad_accum micro-batches
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
            sps = n * per_gpu_bs / dt
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
    metrics["samples_per_sec"] = n * per_gpu_bs / dt
    return metrics


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate(model, criterion, loader, device, use_amp=False):
    model.eval()
    tot_loss = 0.0
    correct = total = 0
    n = 0
    tc_sim = tc_pairs = 0.0
    # For multi-label mAP
    all_logits = []
    all_labels = []
    is_multi_label = False

    for batch in loader:
        feat = batch["features"].to(device, non_blocking=True)
        mask = batch["masks"].to(device, non_blocking=True)
        lbl = batch["labels"].to(device, non_blocking=True)
        is_ml = batch.get("multi_label", False)
        is_multi_label = is_ml

        with autocast(enabled=use_amp):
            out = model(feat, mask)
            has_lbl = lbl.sum() > 0 if is_ml else (lbl >= 0).any()
            losses = criterion(out, labels=lbl if has_lbl else None, mask=mask)

        tot_loss += losses["total"].item()
        n += 1

        if has_lbl:
            if is_ml:
                all_logits.append(out["logits"].float().cpu())
                all_labels.append(lbl.float().cpu())
            else:
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

    if is_multi_label and all_logits:
        logits_cat = torch.cat(all_logits, dim=0).numpy()
        labels_cat = torch.cat(all_labels, dim=0).numpy()
        # Compute per-class AP and mAP
        from sklearn.metrics import average_precision_score
        try:
            mAP = average_precision_score(labels_cat, logits_cat, average="macro")
        except ValueError:
            mAP = 0.0
        result["val_mAP"] = mAP
        result["val_accuracy"] = mAP  # alias for logging consistency
    else:
        result["val_accuracy"] = correct / max(total, 1)

    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    local_rank, world_size, is_ddp = setup_distributed()
    device = torch.device(f"cuda:{local_rank}" if local_rank >= 0 else "cpu")
    is_main = local_rank <= 0

    # reproducibility
    torch.manual_seed(args.seed + local_rank)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed + local_rank)

    # perf knobs
    if args.cudnn_benchmark:
        torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    # effective batch & LR scaling
    eff_bs = args.batch_size * world_size * args.grad_accum_steps
    scaled_lr = args.lr * (eff_bs / 32.0)

    if is_main:
        log.info("=" * 70)
        log.info("TemporalFusion Training")
        log.info("=" * 70)
        log.info(f"  GPUs            : {world_size}")
        log.info(f"  Per-GPU BS      : {args.batch_size}")
        log.info(f"  Grad accum      : {args.grad_accum_steps}")
        log.info(f"  Effective BS    : {eff_bs}")
        log.info(f"  Base LR         : {args.lr}")
        log.info(f"  Scaled LR       : {scaled_lr:.6f}")
        log.info(f"  AMP             : {'ON' if args.amp else 'OFF'}")
        log.info(f"  compile         : {'ON' if args.compile_model else 'OFF'}")
        log.info(f"  Epochs          : {args.epochs}")
        log.info("=" * 70)

    # run dir
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = args.run_name or f"{args.dataset}_{world_size}gpu_{ts}"
    run_dir = Path(args.output_dir) / run_name
    if is_main:
        run_dir.mkdir(parents=True, exist_ok=True)
        with open(run_dir / "args.json", "w") as fh:
            json.dump(
                {**vars(args), "eff_bs": eff_bs,
                 "scaled_lr": scaled_lr, "world_size": world_size},
                fh, indent=2,
            )

    # wandb
    if is_main:
        wandb.init(
            project=args.wandb_project,
            name=run_name,
            config={**vars(args), "eff_bs": eff_bs,
                    "scaled_lr": scaled_lr, "world_size": world_size},
        )

    # data
    if is_main:
        log.info("Loading data ...")
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
        log.info(f"  train : {len(train_loader.dataset)} videos  "
                 f"({len(train_loader)} batches/gpu)")
        log.info(f"  val   : {len(val_loader.dataset)} videos  "
                 f"({len(val_loader)} batches/gpu)")

    # model
    model = TemporalFusionModel(
        feature_dim=args.feature_dim,
        hidden_dim=args.hidden_dim,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        num_classes=args.num_classes,
        num_hierarchy_levels=args.num_hierarchy_levels,
    ).to(device)

    if is_main:
        npar = model.count_parameters()
        log.info(f"  params: {npar['total']:,} ({npar['total']/1e6:.1f}M)")

    if args.compile_model:
        if is_main:
            log.info("  torch.compile ...")
        model = torch.compile(model)

    if is_ddp:
        model = DDP(model, device_ids=[local_rank], find_unused_parameters=False)

    # optimiser
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=scaled_lr, weight_decay=args.weight_decay,
    )

    # cosine schedule with linear warmup
    wu = args.warmup_epochs
    tot = args.epochs
    scheduler = LambdaLR(optimizer, lambda ep: (
        (ep + 1) / max(1, wu) if ep < wu
        else max(0.01, 0.5 * (1.0 + math.cos(math.pi * (ep - wu) / max(1, tot - wu))))
    ))

    # loss
    is_multi_label = bool(args.multi_label)
    criterion = TemporalFusionLoss(
        lambda_tc=args.lambda_tc, lambda_reg=args.lambda_reg,
        lambda_cs=args.lambda_cs, lambda_vl=args.lambda_vl,
        collapse_tau=args.collapse_tau, num_classes=args.num_classes,
        label_smoothing=args.label_smoothing,
        multi_label=is_multi_label,
    ).to(device)

    # AMP scaler
    use_amp = bool(args.amp)
    scaler = GradScaler(enabled=use_amp)

    # resume
    start_epoch = 0
    if args.resume and Path(args.resume).exists():
        ckpt = torch.load(args.resume, map_location=device)
        raw = model.module if is_ddp else model
        raw.load_state_dict(ckpt["model"])
        if "optimizer" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer"])
        if "scaler" in ckpt and ckpt["scaler"] is not None:
            scaler.load_state_dict(ckpt["scaler"])
        start_epoch = ckpt.get("epoch", -1) + 1
        if is_main:
            log.info(f"  resumed from epoch {start_epoch - 1}")

    # ---------------------------------------------------------------- loop
    best_val = float("inf")
    for epoch in range(start_epoch, args.epochs):
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
                f"Epoch {epoch:3d} | "
                f"loss {tm['total']:.4f} | "
                f"cls {tm.get('cls',0):.4f} | "
                f"tc {tm.get('temporal',0):.4f} | "
                f"lr {lr_now:.6f} | "
                f"{tm['samples_per_sec']:.0f} s/s | "
                f"{tm['epoch_sec']:.1f}s"
            )
            wandb.log({
                **{f"train/{k}": v for k, v in tm.items()},
                "lr": lr_now, "epoch": epoch,
            }, step=epoch)

        if epoch % args.eval_every == 0:
            vm = evaluate(model, criterion, val_loader, device, use_amp)
            if is_main:
                metric_str = f"mAP {vm['val_mAP']:.4f}" if "val_mAP" in vm else f"acc {vm['val_accuracy']:.4f}"
                log.info(
                    f"       val | "
                    f"loss {vm['val_loss']:.4f} | "
                    f"{metric_str} | "
                    f"tc {vm['val_temporal_consistency']:.4f}"
                )
                wandb.log({f"val/{k}": v for k, v in vm.items()}, step=epoch)
                if vm["val_loss"] < best_val:
                    best_val = vm["val_loss"]
                    raw = model.module if is_ddp else model
                    torch.save(dict(
                        model=raw.state_dict(),
                        optimizer=optimizer.state_dict(),
                        scaler=scaler.state_dict(),
                        epoch=epoch, val_metrics=vm,
                        args=vars(args),
                    ), run_dir / "best_model.pt")
                    log.info(f"  >> saved best (val_loss={best_val:.4f})")

        if is_main and (epoch % args.save_every == 0 or epoch == args.epochs - 1):
            raw = model.module if is_ddp else model
            torch.save(dict(
                model=raw.state_dict(),
                optimizer=optimizer.state_dict(),
                scaler=scaler.state_dict(),
                epoch=epoch,
            ), run_dir / f"ckpt_ep{epoch}.pt")

    if is_main:
        wandb.finish()
        log.info(f"Done. Best val_loss = {best_val:.4f}")
    cleanup()


if __name__ == "__main__":
    main()
