"""""""""

Training loop for TemporalFusion -- optimized for multi-GPU DDP.

Training loop for TemporalFusion — optimized for multi-GPU DDP.Training loop for TemporalFusion.

Launch with torchrun:

  torchrun --nproc_per_node=8 -m temporalfusion.train \\

      --config configs/train_activitynet.yaml

Launch with torchrun:Supports:

Supports:

  - 8-GPU DDP with NCCL backend  torchrun --nproc_per_node=8 -m temporalfusion.train --config configs/train_activitynet.yaml  - Single GPU and multi-GPU (DDP) training

  - Automatic Mixed Precision (AMP) with GradScaler

  - torch.compile for fused kernels  - W&B logging

  - Linear LR scaling with warmup

  - Gradient accumulationSupports:  - Phased training (Phase 1 → 2 → 3)

  - W&B logging (rank 0 only)

  - Checkpoint saving / resuming  - 8-GPU DDP with NCCL backend  - Checkpoint saving / resuming

"""

  - Automatic Mixed Precision (AMP) with GradScaler"""

import argparse

import json  - torch.compile for fused kernels

import logging

import math  - Linear LR scaling with warmupimport argparse

import os

import sys  - Gradient accumulationimport json

import time

from datetime import datetime  - W&B logging (rank 0 only)import logging

from pathlib import Path

from typing import Dict, Optional  - Checkpoint saving / resumingimport math



import torch"""import os

import torch.nn as nn

import torch.nn.functional as Fimport sys

import torch.distributed as dist

from torch.nn.parallel import DistributedDataParallel as DDPimport argparseimport time

from torch.utils.data.distributed import DistributedSampler

from torch.optim.lr_scheduler import LambdaLRimport jsonfrom datetime import datetime

from torch.cuda.amp import GradScaler, autocast

import wandbimport loggingfrom pathlib import Path

from tqdm import tqdm

import mathfrom typing import Dict, Optional

from temporalfusion.model import TemporalFusionModel

from temporalfusion.losses import TemporalFusionLossimport os

from temporalfusion.data import build_dataloaders

from torch.utils.data import DataLoaderimport sysimport torch



logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")import timeimport torch.nn as nn

logger = logging.getLogger(__name__)

from datetime import datetimeimport torch.distributed as dist



def parse_args():from pathlib import Pathfrom torch.nn.parallel import DistributedDataParallel as DDP

    p = argparse.ArgumentParser(description="TemporalFusion Training")

    p.add_argument("--config", type=str, default=None)from typing import Dict, Optionalfrom torch.utils.data.distributed import DistributedSampler

    # Data

    p.add_argument("--dataset", type=str, default=None, choices=["activitynet"])from torch.optim.lr_scheduler import LambdaLR

    p.add_argument("--train_dir", type=str, default=None)

    p.add_argument("--val_dir", type=str, default=None)import torchimport wandb

    p.add_argument("--annotations", type=str, default=None)

    p.add_argument("--max_frames", type=int, default=None)import torch.nn as nnfrom tqdm import tqdm

    # Model

    p.add_argument("--feature_dim", type=int, default=None)import torch.nn.functional as F

    p.add_argument("--hidden_dim", type=int, default=None)

    p.add_argument("--num_heads", type=int, default=None)import torch.distributed as distfrom temporalfusion.model import TemporalFusionModel

    p.add_argument("--num_layers", type=int, default=None)

    p.add_argument("--num_classes", type=int, default=None)from torch.nn.parallel import DistributedDataParallel as DDPfrom temporalfusion.losses import TemporalFusionLoss

    p.add_argument("--num_hierarchy_levels", type=int, default=None)

    # Trainingfrom torch.utils.data.distributed import DistributedSamplerfrom temporalfusion.data import build_dataloaders, collate_features, ActivityNetFeaturesDataset

    p.add_argument("--batch_size", type=int, default=None)

    p.add_argument("--epochs", type=int, default=None)from torch.optim.lr_scheduler import LambdaLRfrom torch.utils.data import DataLoader

    p.add_argument("--lr", type=float, default=None)

    p.add_argument("--weight_decay", type=float, default=None)from torch.cuda.amp import GradScaler, autocast

    p.add_argument("--warmup_epochs", type=int, default=None)

    p.add_argument("--num_workers", type=int, default=None)import wandblogging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    p.add_argument("--grad_accum_steps", type=int, default=None)

    # Lossfrom tqdm import tqdmlogger = logging.getLogger(__name__)

    p.add_argument("--lambda_tc", type=float, default=None)

    p.add_argument("--lambda_reg", type=float, default=None)

    p.add_argument("--lambda_cs", type=float, default=None)

    p.add_argument("--lambda_vl", type=float, default=None)from temporalfusion.model import TemporalFusionModel

    p.add_argument("--collapse_tau", type=float, default=None)

    p.add_argument("--label_smoothing", type=float, default=None)from temporalfusion.losses import TemporalFusionLossdef parse_args():

    # Performance

    p.add_argument("--amp", type=int, default=None)from temporalfusion.data import build_dataloaders    p = argparse.ArgumentParser(description="TemporalFusion Training")

    p.add_argument("--compile_model", type=int, default=None)

    p.add_argument("--cudnn_benchmark", type=int, default=None)from torch.utils.data import DataLoader

    # Infra

    p.add_argument("--output_dir", type=str, default=None)    # Config file (overrides defaults; CLI args override config)

    p.add_argument("--run_name", type=str, default=None)

    p.add_argument("--seed", type=int, default=None)logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")    p.add_argument("--config", type=str, default=None, help="Path to YAML config file")

    p.add_argument("--wandb_project", type=str, default=None)

    p.add_argument("--log_every", type=int, default=None)logger = logging.getLogger(__name__)

    p.add_argument("--eval_every", type=int, default=None)

    p.add_argument("--save_every", type=int, default=None)    # Data

    p.add_argument("--resume", type=str, default=None)

    # DDP    p.add_argument("--dataset", type=str, default="activitynet", choices=["activitynet"])

    p.add_argument("--local_rank", type=int, default=-1)

# ============================================================================    p.add_argument("--train_dir", type=str, default=None, help="Path to training features dir")

    args = p.parse_args()

# Argument parsing with YAML config support    p.add_argument("--val_dir", type=str, default=None, help="Path to validation features dir")

    import yaml

    defaults = dict(# ============================================================================    p.add_argument("--annotations", type=str, default=None, help="Path to annotations JSON")

        dataset="activitynet",

        train_dir="./data/activitynet/c3d_features/train",    p.add_argument("--max_frames", type=int, default=None)

        val_dir="./data/activitynet/c3d_features/val",

        annotations="./data/activitynet/gt.json",def parse_args():

        max_frames=512,

        feature_dim=2048, hidden_dim=1024, num_heads=8, num_layers=6,    p = argparse.ArgumentParser(description="TemporalFusion Training")    # Model

        num_classes=200, num_hierarchy_levels=4,

        batch_size=64, epochs=50, lr=1e-4, weight_decay=1e-4,    p.add_argument("--feature_dim", type=int, default=None)

        warmup_epochs=5, num_workers=6, grad_accum_steps=1,

        lambda_tc=1.0, lambda_reg=0.1, lambda_cs=0.5, lambda_vl=0.1,    p.add_argument("--config", type=str, default=None, help="Path to YAML config file")    p.add_argument("--hidden_dim", type=int, default=None)

        collapse_tau=0.1, label_smoothing=0.1,

        amp=1, compile_model=0, cudnn_benchmark=1,    p.add_argument("--num_heads", type=int, default=None)

        output_dir="./runs", run_name=None, seed=42,

        wandb_project="temporalfusion", log_every=5,    # Data    p.add_argument("--num_layers", type=int, default=None)

        eval_every=1, save_every=5, resume=None,

    )    p.add_argument("--dataset", type=str, default=None, choices=["activitynet"])    p.add_argument("--num_classes", type=int, default=None)



    if args.config is not None:    p.add_argument("--train_dir", type=str, default=None)    p.add_argument("--num_hierarchy_levels", type=int, default=None)

        with open(args.config, "r") as f:

            cfg = yaml.safe_load(f)    p.add_argument("--val_dir", type=str, default=None)

        flat = {}

        flat["dataset"] = cfg.get("dataset", defaults["dataset"])    p.add_argument("--annotations", type=str, default=None)    # Training

        for section, keys in [

            ("data", ["train_dir", "val_dir", "annotations", "max_frames"]),    p.add_argument("--max_frames", type=int, default=None)    p.add_argument("--batch_size", type=int, default=None)

            ("model", ["feature_dim", "hidden_dim", "num_heads", "num_layers",

                        "num_classes", "num_hierarchy_levels"]),    p.add_argument("--epochs", type=int, default=None)

            ("training", ["batch_size", "epochs", "lr", "weight_decay", "warmup_epochs",

                          "num_workers", "seed", "label_smoothing", "grad_accum_steps"]),    # Model    p.add_argument("--lr", type=float, default=None)

            ("loss", ["lambda_tc", "lambda_reg", "lambda_cs", "lambda_vl", "collapse_tau"]),

            ("performance", ["amp", "compile_model", "cudnn_benchmark"]),    p.add_argument("--feature_dim", type=int, default=None)    p.add_argument("--weight_decay", type=float, default=None)

        ]:

            sub = cfg.get(section, {})    p.add_argument("--hidden_dim", type=int, default=None)    p.add_argument("--warmup_epochs", type=int, default=None)

            for k in keys:

                if k in sub:    p.add_argument("--num_heads", type=int, default=None)    p.add_argument("--num_workers", type=int, default=None)

                    flat[k] = sub[k]

        log_cfg = cfg.get("logging", {})    p.add_argument("--num_layers", type=int, default=None)

        for k in ["wandb_project", "log_every", "eval_every", "save_every", "output_dir"]:

            if k in log_cfg:    p.add_argument("--num_classes", type=int, default=None)    # Loss weights

                flat[k] = log_cfg[k]

        defaults.update(flat)    p.add_argument("--num_hierarchy_levels", type=int, default=None)    p.add_argument("--lambda_tc", type=float, default=None)



    for k, v in defaults.items():    p.add_argument("--lambda_reg", type=float, default=None)

        if getattr(args, k, None) is None:

            setattr(args, k, v)    # Training    p.add_argument("--lambda_cs", type=float, default=None)

    return args

    p.add_argument("--batch_size", type=int, default=None, help="Per-GPU batch size")    p.add_argument("--lambda_vl", type=float, default=None)



def setup_distributed():    p.add_argument("--epochs", type=int, default=None)    p.add_argument("--collapse_tau", type=float, default=None)

    if "WORLD_SIZE" in os.environ and int(os.environ["WORLD_SIZE"]) > 1:

        dist.init_process_group(backend="nccl")    p.add_argument("--lr", type=float, default=None, help="Base LR (auto-scaled for multi-GPU)")    p.add_argument("--label_smoothing", type=float, default=None)

        local_rank = int(os.environ.get("LOCAL_RANK", 0))

        world_size = dist.get_world_size()    p.add_argument("--weight_decay", type=float, default=None)

        torch.cuda.set_device(local_rank)

        return local_rank, world_size, True    p.add_argument("--warmup_epochs", type=int, default=None)    # Infrastructure

    else:

        device_id = 0 if torch.cuda.is_available() else -1    p.add_argument("--num_workers", type=int, default=None, help="DataLoader workers per GPU")    p.add_argument("--output_dir", type=str, default=None)

        if device_id >= 0:

            torch.cuda.set_device(device_id)    p.add_argument("--grad_accum_steps", type=int, default=None, help="Gradient accumulation steps")    p.add_argument("--run_name", type=str, default=None)

        return device_id, 1, False

    p.add_argument("--seed", type=int, default=None)



def cleanup_distributed():    # Loss weights    p.add_argument("--wandb_project", type=str, default=None)

    if dist.is_initialized():

        dist.destroy_process_group()    p.add_argument("--lambda_tc", type=float, default=None)    p.add_argument("--log_every", type=int, default=None)



    p.add_argument("--lambda_reg", type=float, default=None)    p.add_argument("--eval_every", type=int, default=None)

def train_one_epoch(model, criterion, optimizer, loader, device, epoch,

                    scaler, use_amp, grad_accum_steps, args):    p.add_argument("--lambda_cs", type=float, default=None)    p.add_argument("--save_every", type=int, default=None)

    model.train()

    total_loss = 0.0    p.add_argument("--lambda_vl", type=float, default=None)    p.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from")

    loss_components = {}

    num_batches = 0    p.add_argument("--collapse_tau", type=float, default=None)

    t0 = time.time()

    is_main = int(os.environ.get("LOCAL_RANK", 0)) <= 0    p.add_argument("--label_smoothing", type=float, default=None)    # DDP

    pbar = tqdm(loader, desc=f"Epoch {epoch}", disable=(not is_main), ncols=120)

    p.add_argument("--local_rank", type=int, default=-1)

    optimizer.zero_grad(set_to_none=True)

    # Performance

    for step, batch in enumerate(pbar):

        features = batch["features"].to(device, non_blocking=True)    p.add_argument("--amp", type=int, default=None, help="Use automatic mixed precision (0/1)")    args = p.parse_args()

        masks = batch["masks"].to(device, non_blocking=True)

        labels = batch["labels"].to(device, non_blocking=True)    p.add_argument("--compile_model", type=int, default=None, help="Use torch.compile (0/1)")

        has_labels = (labels >= 0).any()

    p.add_argument("--cudnn_benchmark", type=int, default=None, help="Enable cudnn benchmark (0/1)")    # Load config file and merge

        with autocast(enabled=use_amp):

            output = model(features, masks)    import yaml

            losses = criterion(output, labels=labels if has_labels else None, mask=masks)

            loss = losses["total"] / grad_accum_steps    # Infrastructure    defaults = {



        if scaler is not None:    p.add_argument("--output_dir", type=str, default=None)        "dataset": "activitynet",

            scaler.scale(loss).backward()

        else:    p.add_argument("--run_name", type=str, default=None)        "train_dir": "./data/activitynet/c3d_features/train",

            loss.backward()

    p.add_argument("--seed", type=int, default=None)        "val_dir": "./data/activitynet/c3d_features/val",

        total_loss += losses["total"].item()

        for k, v in losses.items():    p.add_argument("--wandb_project", type=str, default=None)        "annotations": "./data/activitynet/gt.json",

            loss_components[k] = loss_components.get(k, 0.0) + v.item()

        num_batches += 1    p.add_argument("--log_every", type=int, default=None)        "max_frames": 512,



        if (step + 1) % grad_accum_steps == 0:    p.add_argument("--eval_every", type=int, default=None)        "feature_dim": 2048,

            if scaler is not None:

                scaler.unscale_(optimizer)    p.add_argument("--save_every", type=int, default=None)        "hidden_dim": 1024,

                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

                scaler.step(optimizer)    p.add_argument("--resume", type=str, default=None)        "num_heads": 8,

                scaler.update()

            else:        "num_layers": 6,

                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

                optimizer.step()    # DDP (set automatically by torchrun)        "num_classes": 200,

            optimizer.zero_grad(set_to_none=True)

    p.add_argument("--local_rank", type=int, default=-1)        "num_hierarchy_levels": 4,

        if num_batches % args.log_every == 0 and is_main:

            elapsed = time.time() - t0        "batch_size": 32,

            sps = num_batches * features.size(0) / elapsed

            pbar.set_postfix(loss=f"{total_loss / num_batches:.4f}", sps=f"{sps:.0f}")    args = p.parse_args()        "epochs": 50,



    # Handle leftover gradients        "lr": 1e-4,

    if num_batches % grad_accum_steps != 0:

        if scaler is not None:    # ----- Defaults -----        "weight_decay": 1e-4,

            scaler.unscale_(optimizer)

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)    import yaml        "warmup_epochs": 5,

            scaler.step(optimizer)

            scaler.update()    defaults = {        "num_workers": 4,

        else:

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)        "dataset": "activitynet",        "lambda_tc": 1.0,

            optimizer.step()

        optimizer.zero_grad(set_to_none=True)        "train_dir": "./data/activitynet/c3d_features/train",        "lambda_reg": 0.1,



    elapsed = time.time() - t0        "val_dir": "./data/activitynet/c3d_features/val",        "lambda_cs": 0.5,

    metrics = {k: v / max(num_batches, 1) for k, v in loss_components.items()}

    metrics["epoch_time_sec"] = elapsed        "annotations": "./data/activitynet/gt.json",        "lambda_vl": 0.1,

    metrics["samples_per_sec"] = num_batches * args.batch_size / elapsed

    return metrics        "max_frames": 512,        "collapse_tau": 0.1,



        "feature_dim": 2048,        "label_smoothing": 0.1,

@torch.no_grad()

def evaluate(model, criterion, loader, device, use_amp=False):        "hidden_dim": 1024,        "output_dir": "./runs",

    model.eval()

    total_loss = 0.0        "num_heads": 8,        "run_name": None,

    correct = 0

    total_samples = 0        "num_layers": 6,        "seed": 42,

    num_batches = 0

    total_temporal_sim = 0.0        "num_classes": 200,        "wandb_project": "temporalfusion",

    total_temporal_pairs = 0

        "num_hierarchy_levels": 4,        "log_every": 20,

    for batch in loader:

        features = batch["features"].to(device, non_blocking=True)        "batch_size": 64,        "eval_every": 1,

        masks = batch["masks"].to(device, non_blocking=True)

        labels = batch["labels"].to(device, non_blocking=True)        "epochs": 50,        "save_every": 5,



        with autocast(enabled=use_amp):        "lr": 1e-4,        "resume": None,

            output = model(features, masks)

            has_labels = (labels >= 0).any()        "weight_decay": 1e-4,    }

            losses = criterion(output, labels=labels if has_labels else None, mask=masks)

        "warmup_epochs": 5,

        total_loss += losses["total"].item()

        num_batches += 1        "num_workers": 6,    if args.config is not None:



        if has_labels:        "grad_accum_steps": 1,        with open(args.config, "r") as f:

            valid = labels >= 0

            preds = output["logits"][valid].argmax(dim=-1)        "lambda_tc": 1.0,            cfg = yaml.safe_load(f)

            correct += (preds == labels[valid]).sum().item()

            total_samples += valid.sum().item()        "lambda_reg": 0.1,        # Flatten nested config



        h = F.normalize(output["frame_features"].float(), dim=-1)        "lambda_cs": 0.5,        flat = {}

        cos_sim = (h[:, :-1] * h[:, 1:]).sum(dim=-1)

        pair_mask = masks[:, :-1] * masks[:, 1:]        "lambda_vl": 0.1,        flat["dataset"] = cfg.get("dataset", defaults["dataset"])

        total_temporal_sim += (cos_sim * pair_mask).sum().item()

        total_temporal_pairs += pair_mask.sum().item()        "collapse_tau": 0.1,        data_cfg = cfg.get("data", {})



    return {        "label_smoothing": 0.1,        flat["train_dir"] = data_cfg.get("train_dir", defaults["train_dir"])

        "val_loss": total_loss / max(num_batches, 1),

        "val_accuracy": correct / max(total_samples, 1),        "amp": 1,        flat["val_dir"] = data_cfg.get("val_dir", defaults["val_dir"])

        "val_temporal_consistency": total_temporal_sim / max(total_temporal_pairs, 1),

    }        "compile_model": 0,        flat["annotations"] = data_cfg.get("annotations", defaults["annotations"])



        "cudnn_benchmark": 1,        flat["max_frames"] = data_cfg.get("max_frames", defaults["max_frames"])

def main():

    args = parse_args()        "output_dir": "./runs",        model_cfg = cfg.get("model", {})

    local_rank, world_size, is_distributed = setup_distributed()

    device = torch.device(f"cuda:{local_rank}" if local_rank >= 0 else "cpu")        "run_name": None,        for k in ["feature_dim", "hidden_dim", "num_heads", "num_layers", "num_classes",

    is_main = local_rank <= 0

        "seed": 42,                   "num_hierarchy_levels"]:

    torch.manual_seed(args.seed + local_rank)

    if torch.cuda.is_available():        "wandb_project": "temporalfusion",            flat[k] = model_cfg.get(k, defaults[k])

        torch.cuda.manual_seed_all(args.seed + local_rank)

        "log_every": 10,        train_cfg = cfg.get("training", {})

    if args.cudnn_benchmark:

        torch.backends.cudnn.benchmark = True        "eval_every": 1,        for k in ["batch_size", "epochs", "lr", "weight_decay", "warmup_epochs",

    torch.backends.cuda.matmul.allow_tf32 = True

    torch.backends.cudnn.allow_tf32 = True        "save_every": 5,                   "num_workers", "seed", "label_smoothing"]:



    effective_batch_size = args.batch_size * world_size * args.grad_accum_steps        "resume": None,            flat[k] = train_cfg.get(k, defaults[k])

    lr_scale = effective_batch_size / 32.0

    scaled_lr = args.lr * lr_scale    }        loss_cfg = cfg.get("loss", {})



    if is_main:        for k in ["lambda_tc", "lambda_reg", "lambda_cs", "lambda_vl", "collapse_tau"]:

        logger.info("=" * 70)

        logger.info("TemporalFusion Training -- Multi-GPU Optimized")    # ----- Load YAML config -----            flat[k] = loss_cfg.get(k, defaults[k])

        logger.info("=" * 70)

        logger.info(f"  World size:         {world_size} GPUs")    if args.config is not None:        log_cfg = cfg.get("logging", {})

        logger.info(f"  Per-GPU batch size: {args.batch_size}")

        logger.info(f"  Grad accum steps:   {args.grad_accum_steps}")        with open(args.config, "r") as f:        flat["wandb_project"] = log_cfg.get("wandb_project", defaults["wandb_project"])

        logger.info(f"  Effective batch:    {effective_batch_size}")

        logger.info(f"  Base LR:            {args.lr}")            cfg = yaml.safe_load(f)        flat["log_every"] = log_cfg.get("log_every", defaults["log_every"])

        logger.info(f"  Scaled LR:          {scaled_lr:.6f}")

        logger.info(f"  AMP:                {'ON' if args.amp else 'OFF'}")        flat = {}        flat["eval_every"] = log_cfg.get("eval_every", defaults["eval_every"])

        logger.info(f"  torch.compile:      {'ON' if args.compile_model else 'OFF'}")

        logger.info(f"  Epochs:             {args.epochs}")        flat["dataset"] = cfg.get("dataset", defaults["dataset"])        flat["save_every"] = log_cfg.get("save_every", defaults["save_every"])

        logger.info("=" * 70)

        for section, keys in [        flat["output_dir"] = log_cfg.get("output_dir", defaults["output_dir"])

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    run_name = args.run_name or f"{args.dataset}_{world_size}gpu_{ts}"            ("data", ["train_dir", "val_dir", "annotations", "max_frames"]),        flat["run_name"] = defaults["run_name"]

    run_dir = Path(args.output_dir) / run_name

    if is_main:            ("model", ["feature_dim", "hidden_dim", "num_heads", "num_layers",        flat["resume"] = defaults["resume"]

        run_dir.mkdir(parents=True, exist_ok=True)

        with open(run_dir / "args.json", "w") as f:                        "num_classes", "num_hierarchy_levels"]),        defaults.update(flat)

            json.dump({**vars(args), "effective_batch_size": effective_batch_size,

                        "scaled_lr": scaled_lr, "world_size": world_size}, f, indent=2)            ("training", ["batch_size", "epochs", "lr", "weight_decay", "warmup_epochs",



    if is_main:                          "num_workers", "seed", "label_smoothing", "grad_accum_steps"]),    # Apply defaults, then override with any CLI args that were explicitly set

        wandb.init(project=args.wandb_project, name=run_name,

                   config={**vars(args), "effective_batch_size": effective_batch_size,            ("loss", ["lambda_tc", "lambda_reg", "lambda_cs", "lambda_vl", "collapse_tau"]),    for k, v in defaults.items():

                           "scaled_lr": scaled_lr, "world_size": world_size})

            ("performance", ["amp", "compile_model", "cudnn_benchmark"]),        if getattr(args, k, None) is None:

    if is_main:

        logger.info("Building data loaders...")        ]:            setattr(args, k, v)

    train_loader, val_loader = build_dataloaders(

        dataset_name=args.dataset,            sub = cfg.get(section, {})

        train_dir=args.train_dir,

        val_dir=args.val_dir,            for k in keys:    return args

        annotations_file=args.annotations,

        batch_size=args.batch_size,                if k in sub:

        num_workers=args.num_workers,

        max_frames=args.max_frames,                    flat[k] = sub[k]

        distributed=is_distributed,

    )        log_cfg = cfg.get("logging", {})def setup_distributed():

    if is_main:

        logger.info(f"  Train: {len(train_loader.dataset)} samples ({len(train_loader)} batches/GPU)")        for k in ["wandb_project", "log_every", "eval_every", "save_every", "output_dir"]:    """Initialize DDP if WORLD_SIZE > 1."""

        logger.info(f"  Val:   {len(val_loader.dataset)} samples ({len(val_loader)} batches/GPU)")

            if k in log_cfg:    if "WORLD_SIZE" in os.environ and int(os.environ["WORLD_SIZE"]) > 1:

    model = TemporalFusionModel(

        feature_dim=args.feature_dim,                flat[k] = log_cfg[k]        dist.init_process_group(backend="nccl")

        hidden_dim=args.hidden_dim,

        num_heads=args.num_heads,        defaults.update(flat)        local_rank = int(os.environ.get("LOCAL_RANK", 0))

        num_layers=args.num_layers,

        num_classes=args.num_classes,        torch.cuda.set_device(local_rank)

        num_hierarchy_levels=args.num_hierarchy_levels,

    ).to(device)    # Apply: defaults -> config -> CLI overrides        return local_rank, True



    if is_main:    for k, v in defaults.items():    else:

        params = model.count_parameters()

        logger.info(f"  Model: {params['total']:,} params ({params['total']/1e6:.1f}M)")        if getattr(args, k, None) is None:        device_id = 0 if torch.cuda.is_available() else -1



    if args.compile_model:            setattr(args, k, v)        if device_id >= 0:

        if is_main:

            logger.info("  Compiling model with torch.compile...")            torch.cuda.set_device(device_id)

        model = torch.compile(model)

    return args        return device_id, False

    if is_distributed:

        model = DDP(model, device_ids=[local_rank], find_unused_parameters=False)



    optimizer = torch.optim.AdamW(model.parameters(), lr=scaled_lr, weight_decay=args.weight_decay)



    warmup_epochs = args.warmup_epochs# ============================================================================def train_one_epoch(

    total_epochs = args.epochs

# Distributed setup    model: nn.Module,

    def lr_lambda(epoch):

        if epoch < warmup_epochs:# ============================================================================    criterion: TemporalFusionLoss,

            return float(epoch + 1) / float(max(1, warmup_epochs))

        progress = float(epoch - warmup_epochs) / float(max(1, total_epochs - warmup_epochs))    optimizer: torch.optim.Optimizer,

        return max(1e-2, 0.5 * (1.0 + math.cos(math.pi * progress)))

def setup_distributed():    loader: DataLoader,

    scheduler = LambdaLR(optimizer, lr_lambda)

    """Initialize DDP. Returns (local_rank, world_size, is_distributed)."""    device: torch.device,

    criterion = TemporalFusionLoss(

        lambda_tc=args.lambda_tc, lambda_reg=args.lambda_reg,    if "WORLD_SIZE" in os.environ and int(os.environ["WORLD_SIZE"]) > 1:    epoch: int,

        lambda_cs=args.lambda_cs, lambda_vl=args.lambda_vl,

        collapse_tau=args.collapse_tau, num_classes=args.num_classes,        dist.init_process_group(backend="nccl")    args,

        label_smoothing=args.label_smoothing,

    ).to(device)        local_rank = int(os.environ.get("LOCAL_RANK", 0))    scheduler=None,



    use_amp = bool(args.amp)        world_size = dist.get_world_size()) -> Dict[str, float]:

    scaler = GradScaler(enabled=use_amp)

        torch.cuda.set_device(local_rank)    model.train()

    start_epoch = 0

    if args.resume and Path(args.resume).exists():        return local_rank, world_size, True    total_loss = 0.0

        ckpt = torch.load(args.resume, map_location=device)

        raw_model = model.module if is_distributed else model    else:    loss_components = {}

        raw_model.load_state_dict(ckpt.get("model", ckpt))

        if "optimizer" in ckpt:        device_id = 0 if torch.cuda.is_available() else -1    num_batches = 0

            optimizer.load_state_dict(ckpt["optimizer"])

        if "scaler" in ckpt and scaler is not None:        if device_id >= 0:

            scaler.load_state_dict(ckpt["scaler"])

        start_epoch = ckpt.get("epoch", 0) + 1            torch.cuda.set_device(device_id)    pbar = tqdm(loader, desc=f"Train Epoch {epoch}", disable=(args.local_rank > 0))

        if is_main:

            logger.info(f"  Resumed from epoch {start_epoch - 1}")        return device_id, 1, False    for batch in pbar:



    best_val_loss = float("inf")        features = batch["features"].to(device)

    for epoch in range(start_epoch, args.epochs):

        if is_distributed:        masks = batch["masks"].to(device)

            train_loader.sampler.set_epoch(epoch)

def cleanup_distributed():        labels = batch["labels"].to(device)

        train_metrics = train_one_epoch(

            model, criterion, optimizer, train_loader, device, epoch,    if dist.is_initialized():

            scaler=scaler, use_amp=use_amp,

            grad_accum_steps=args.grad_accum_steps, args=args,        dist.destroy_process_group()        # Skip batches where all labels are -1 (no supervision)

        )

        scheduler.step()        has_labels = (labels >= 0).any()



        if is_main:

            current_lr = optimizer.param_groups[0]["lr"]

            logger.info(# ============================================================================        optimizer.zero_grad()

                f"Epoch {epoch:3d} | loss={train_metrics['total']:.4f} | "

                f"cls={train_metrics.get('cls', 0):.4f} | "# Training        output = model(features, masks)

                f"tc={train_metrics.get('temporal', 0):.4f} | "

                f"lr={current_lr:.6f} | "# ============================================================================        losses = criterion(

                f"{train_metrics['samples_per_sec']:.0f} samp/s | "

                f"{train_metrics['epoch_time_sec']:.1f}s"            output,

            )

            wandb.log({def train_one_epoch(            labels=labels if has_labels else None,

                **{f"train/{k}": v for k, v in train_metrics.items()},

                "lr": current_lr, "epoch": epoch,    model: nn.Module,            mask=masks,

            }, step=epoch)

    criterion: TemporalFusionLoss,        )

        if epoch % args.eval_every == 0:

            val_metrics = evaluate(model, criterion, val_loader, device, use_amp=use_amp)    optimizer: torch.optim.Optimizer,

            if is_main:

                logger.info(    loader: DataLoader,        losses["total"].backward()

                    f"         val | loss={val_metrics['val_loss']:.4f} | "

                    f"acc={val_metrics['val_accuracy']:.4f} | "    device: torch.device,        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

                    f"tc={val_metrics['val_temporal_consistency']:.4f}"

                )    epoch: int,        optimizer.step()

                wandb.log({f"val/{k}": v for k, v in val_metrics.items()}, step=epoch)

                if val_metrics["val_loss"] < best_val_loss:    scaler: Optional[GradScaler],

                    best_val_loss = val_metrics["val_loss"]

                    raw_model = model.module if is_distributed else model    use_amp: bool,        total_loss += losses["total"].item()

                    torch.save({

                        "model": raw_model.state_dict(),    grad_accum_steps: int,        for k, v in losses.items():

                        "optimizer": optimizer.state_dict(),

                        "scaler": scaler.state_dict() if scaler else None,    args,            loss_components[k] = loss_components.get(k, 0.0) + v.item()

                        "epoch": epoch, "val_metrics": val_metrics,

                        "args": vars(args),) -> Dict[str, float]:        num_batches += 1

                    }, run_dir / "best_model.pt")

                    logger.info(f"  -> New best model (val_loss={best_val_loss:.4f})")    model.train()



        if is_main and (epoch % args.save_every == 0 or epoch == args.epochs - 1):    total_loss = 0.0        if num_batches % args.log_every == 0:

            raw_model = model.module if is_distributed else model

            torch.save({    loss_components: Dict[str, float] = {}            pbar.set_postfix(loss=f"{total_loss / num_batches:.4f}")

                "model": raw_model.state_dict(),

                "optimizer": optimizer.state_dict(),    num_batches = 0

                "scaler": scaler.state_dict() if scaler else None,

                "epoch": epoch,    t0 = time.time()    return {k: v / max(num_batches, 1) for k, v in loss_components.items()}

            }, run_dir / f"checkpoint_epoch{epoch}.pt")



    if is_main:

        wandb.finish()    is_main = int(os.environ.get("LOCAL_RANK", 0)) <= 0

        logger.info(f"Training complete. Best val_loss: {best_val_loss:.4f}")

    cleanup_distributed()    pbar = tqdm(loader, desc=f"Epoch {epoch}", disable=(not is_main), ncols=120)@torch.no_grad()



def evaluate(

if __name__ == "__main__":

    main()    optimizer.zero_grad(set_to_none=True)    model: nn.Module,


    criterion: TemporalFusionLoss,

    for step, batch in enumerate(pbar):    loader: DataLoader,

        features = batch["features"].to(device, non_blocking=True)    device: torch.device,

        masks = batch["masks"].to(device, non_blocking=True)) -> Dict[str, float]:

        labels = batch["labels"].to(device, non_blocking=True)    model.eval()

        has_labels = (labels >= 0).any()    total_loss = 0.0

    correct = 0

        # Forward with AMP    total_samples = 0

        with autocast(enabled=use_amp):    num_batches = 0

            output = model(features, masks)

            losses = criterion(    # Temporal consistency tracking

                output,    total_temporal_sim = 0.0

                labels=labels if has_labels else None,    total_temporal_pairs = 0

                mask=masks,

            )    for batch in loader:

            loss = losses["total"] / grad_accum_steps        features = batch["features"].to(device)

        masks = batch["masks"].to(device)

        # Backward        labels = batch["labels"].to(device)

        if scaler is not None:

            scaler.scale(loss).backward()        output = model(features, masks)

        else:        has_labels = (labels >= 0).any()

            loss.backward()        losses = criterion(output, labels=labels if has_labels else None, mask=masks)



        # Accumulate loss stats        total_loss += losses["total"].item()

        total_loss += losses["total"].item()        num_batches += 1

        for k, v in losses.items():

            loss_components[k] = loss_components.get(k, 0.0) + v.item()        # Classification accuracy

        num_batches += 1        if has_labels:

            valid = labels >= 0

        # Optimizer step every grad_accum_steps            preds = output["logits"][valid].argmax(dim=-1)

        if (step + 1) % grad_accum_steps == 0:            correct += (preds == labels[valid]).sum().item()

            if scaler is not None:            total_samples += valid.sum().item()

                scaler.unscale_(optimizer)

                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)        # Temporal consistency

                scaler.step(optimizer)        h = torch.nn.functional.normalize(output["frame_features"], dim=-1)

                scaler.update()        cos_sim = (h[:, :-1] * h[:, 1:]).sum(dim=-1)  # (B, T-1)

            else:        pair_mask = masks[:, :-1] * masks[:, 1:]

                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)        total_temporal_sim += (cos_sim * pair_mask).sum().item()

                optimizer.step()        total_temporal_pairs += pair_mask.sum().item()

            optimizer.zero_grad(set_to_none=True)

    metrics = {

        if num_batches % args.log_every == 0 and is_main:        "val_loss": total_loss / max(num_batches, 1),

            elapsed = time.time() - t0        "val_accuracy": correct / max(total_samples, 1),

            samples_per_sec = num_batches * features.size(0) / elapsed        "val_temporal_consistency": total_temporal_sim / max(total_temporal_pairs, 1),

            pbar.set_postfix(    }

                loss=f"{total_loss / num_batches:.4f}",    return metrics

                sps=f"{samples_per_sec:.0f}",

            )

def main():

    # Handle leftover gradients if total steps not divisible by grad_accum_steps    args = parse_args()

    if num_batches % grad_accum_steps != 0:    local_rank, is_distributed = setup_distributed()

        if scaler is not None:    device = torch.device(f"cuda:{local_rank}" if local_rank >= 0 else "cpu")

            scaler.unscale_(optimizer)    is_main = local_rank <= 0

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            scaler.step(optimizer)    # Seed

            scaler.update()    torch.manual_seed(args.seed)

        else:    if torch.cuda.is_available():

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)        torch.cuda.manual_seed_all(args.seed)

            optimizer.step()

        optimizer.zero_grad(set_to_none=True)    # Run directory

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    elapsed = time.time() - t0    run_name = args.run_name or f"{args.dataset}_{ts}"

    metrics = {k: v / max(num_batches, 1) for k, v in loss_components.items()}    run_dir = Path(args.output_dir) / run_name

    metrics["epoch_time_sec"] = elapsed    if is_main:

    metrics["samples_per_sec"] = num_batches * args.batch_size / elapsed        run_dir.mkdir(parents=True, exist_ok=True)

    return metrics        with open(run_dir / "args.json", "w") as f:

            json.dump(vars(args), f, indent=2)



@torch.no_grad()    # W&B

def evaluate(    if is_main:

    model: nn.Module,        wandb.init(project=args.wandb_project, name=run_name, config=vars(args))

    criterion: TemporalFusionLoss,

    loader: DataLoader,    # Data

    device: torch.device,    if is_main:

    use_amp: bool = False,        logger.info(f"Building data loaders for {args.dataset}...")

) -> Dict[str, float]:    train_loader, val_loader = build_dataloaders(

    model.eval()        dataset_name=args.dataset,

    total_loss = 0.0        train_dir=args.train_dir,

    correct = 0        val_dir=args.val_dir,

    total_samples = 0        annotations_file=args.annotations,

    num_batches = 0        batch_size=args.batch_size,

    total_temporal_sim = 0.0        num_workers=args.num_workers,

    total_temporal_pairs = 0        max_frames=args.max_frames,

    )

    for batch in loader:    if is_main:

        features = batch["features"].to(device, non_blocking=True)        logger.info(f"Train: {len(train_loader.dataset)} samples, Val: {len(val_loader.dataset)} samples")

        masks = batch["masks"].to(device, non_blocking=True)

        labels = batch["labels"].to(device, non_blocking=True)    # Model

    model = TemporalFusionModel(

        with autocast(enabled=use_amp):        feature_dim=args.feature_dim,

            output = model(features, masks)        hidden_dim=args.hidden_dim,

            has_labels = (labels >= 0).any()        num_heads=args.num_heads,

            losses = criterion(output, labels=labels if has_labels else None, mask=masks)        num_layers=args.num_layers,

        num_classes=args.num_classes,

        total_loss += losses["total"].item()        num_hierarchy_levels=args.num_hierarchy_levels,

        num_batches += 1    ).to(device)



        if has_labels:    if is_main:

            valid = labels >= 0        params = model.count_parameters()

            preds = output["logits"][valid].argmax(dim=-1)        logger.info(f"Model: {params['total']:,} total params, {params['trainable']:,} trainable")

            correct += (preds == labels[valid]).sum().item()

            total_samples += valid.sum().item()    if is_distributed:

        model = DDP(model, device_ids=[local_rank])

        h = F.normalize(output["frame_features"].float(), dim=-1)

        cos_sim = (h[:, :-1] * h[:, 1:]).sum(dim=-1)    # Optimizer & scheduler (epoch-level cosine annealing with linear warmup)

        pair_mask = masks[:, :-1] * masks[:, 1:]    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

        total_temporal_sim += (cos_sim * pair_mask).sum().item()    warmup_epochs = args.warmup_epochs

        total_temporal_pairs += pair_mask.sum().item()    total_epochs = args.epochs



    metrics = {    def lr_lambda(epoch):

        "val_loss": total_loss / max(num_batches, 1),        if epoch < warmup_epochs:

        "val_accuracy": correct / max(total_samples, 1),            return float(epoch + 1) / float(max(1, warmup_epochs))

        "val_temporal_consistency": total_temporal_sim / max(total_temporal_pairs, 1),        progress = float(epoch - warmup_epochs) / float(max(1, total_epochs - warmup_epochs))

    }        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return metrics

    scheduler = LambdaLR(optimizer, lr_lambda)



# ============================================================================    # Loss

# Main    criterion = TemporalFusionLoss(

# ============================================================================        lambda_tc=args.lambda_tc,

        lambda_reg=args.lambda_reg,

def main():        lambda_cs=args.lambda_cs,

    args = parse_args()        lambda_vl=args.lambda_vl,

    local_rank, world_size, is_distributed = setup_distributed()        collapse_tau=args.collapse_tau,

    device = torch.device(f"cuda:{local_rank}" if local_rank >= 0 else "cpu")        num_classes=args.num_classes,

    is_main = local_rank <= 0        label_smoothing=args.label_smoothing,

    ).to(device)

    # Seed (offset by rank for data diversity)

    torch.manual_seed(args.seed + local_rank)    # Resume

    if torch.cuda.is_available():    start_epoch = 0

        torch.cuda.manual_seed_all(args.seed + local_rank)    if args.resume and Path(args.resume).exists():

        ckpt = torch.load(args.resume, map_location=device)

    # Performance flags        model_state = ckpt.get("model", ckpt)

    if args.cudnn_benchmark:        if is_distributed:

        torch.backends.cudnn.benchmark = True            model.module.load_state_dict(model_state)

    torch.backends.cuda.matmul.allow_tf32 = True        else:

    torch.backends.cudnn.allow_tf32 = True            model.load_state_dict(model_state)

        if "optimizer" in ckpt:

    # Effective batch size and linear LR scaling            optimizer.load_state_dict(ckpt["optimizer"])

    effective_batch_size = args.batch_size * world_size * args.grad_accum_steps        start_epoch = ckpt.get("epoch", 0)

    lr_scale = effective_batch_size / 32.0  # scale relative to BS=32        if is_main:

    scaled_lr = args.lr * lr_scale            logger.info(f"Resumed from epoch {start_epoch}")



    if is_main:    # Training loop

        logger.info("=" * 70)    best_val_loss = float("inf")

        logger.info("TemporalFusion Training — Multi-GPU Optimized")    for epoch in range(start_epoch, args.epochs):

        logger.info("=" * 70)        if is_distributed:

        logger.info(f"  World size:         {world_size} GPUs")            train_loader.sampler.set_epoch(epoch)

        logger.info(f"  Per-GPU batch size: {args.batch_size}")

        logger.info(f"  Grad accum steps:   {args.grad_accum_steps}")        train_metrics = train_one_epoch(model, criterion, optimizer, train_loader, device, epoch, args)

        logger.info(f"  Effective batch:    {effective_batch_size}")        scheduler.step()

        logger.info(f"  Base LR:            {args.lr}")

        logger.info(f"  Scaled LR:          {scaled_lr:.6f}")        if is_main:

        logger.info(f"  AMP:                {'ON' if args.amp else 'OFF'}")            logger.info(f"Epoch {epoch} train — " + ", ".join(f"{k}: {v:.4f}" for k, v in train_metrics.items()))

        logger.info(f"  torch.compile:      {'ON' if args.compile_model else 'OFF'}")            current_lr = optimizer.param_groups[0]["lr"]

        logger.info(f"  Epochs:             {args.epochs}")            wandb.log({**{f"train/{k}": v for k, v in train_metrics.items()}, "lr": current_lr}, step=epoch)

        logger.info("=" * 70)

        # Evaluation

    # Run directory        if epoch % args.eval_every == 0:

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")            val_metrics = evaluate(model, criterion, val_loader, device)

    run_name = args.run_name or f"{args.dataset}_{world_size}gpu_{ts}"            if is_main:

    run_dir = Path(args.output_dir) / run_name                logger.info(f"Epoch {epoch} val   — " + ", ".join(f"{k}: {v:.4f}" for k, v in val_metrics.items()))

    if is_main:                wandb.log({f"val/{k}": v for k, v in val_metrics.items()}, step=epoch)

        run_dir.mkdir(parents=True, exist_ok=True)

        with open(run_dir / "args.json", "w") as f:                if val_metrics["val_loss"] < best_val_loss:

            json.dump({**vars(args), "effective_batch_size": effective_batch_size,                    best_val_loss = val_metrics["val_loss"]

                        "scaled_lr": scaled_lr, "world_size": world_size}, f, indent=2)                    save_model = model.module if is_distributed else model

                    torch.save(

    # W&B (rank 0 only)                        {"model": save_model.state_dict(), "optimizer": optimizer.state_dict(), "epoch": epoch, "val_metrics": val_metrics},

    if is_main:                        run_dir / "best_model.pt",

        wandb.init(                    )

            project=args.wandb_project, name=run_name,                    logger.info(f"  → Saved best model (val_loss={best_val_loss:.4f})")

            config={**vars(args), "effective_batch_size": effective_batch_size,

                    "scaled_lr": scaled_lr, "world_size": world_size},        # Periodic checkpoint

        )        if is_main and epoch % args.save_every == 0:

            save_model = model.module if is_distributed else model

    # Data            torch.save(

    if is_main:                {"model": save_model.state_dict(), "optimizer": optimizer.state_dict(), "epoch": epoch},

        logger.info("Building data loaders...")                run_dir / f"checkpoint_epoch{epoch}.pt",

    train_loader, val_loader = build_dataloaders(            )

        dataset_name=args.dataset,

        train_dir=args.train_dir,    if is_main:

        val_dir=args.val_dir,        wandb.finish()

        annotations_file=args.annotations,        logger.info(f"Training complete. Best val_loss: {best_val_loss:.4f}")

        batch_size=args.batch_size,

        num_workers=args.num_workers,

        max_frames=args.max_frames,if __name__ == "__main__":

        distributed=is_distributed,    main()

    )
    if is_main:
        logger.info(f"  Train: {len(train_loader.dataset)} samples ({len(train_loader)} batches/GPU)")
        logger.info(f"  Val:   {len(val_loader.dataset)} samples ({len(val_loader)} batches/GPU)")

    # Model
    model = TemporalFusionModel(
        feature_dim=args.feature_dim,
        hidden_dim=args.hidden_dim,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        num_classes=args.num_classes,
        num_hierarchy_levels=args.num_hierarchy_levels,
    ).to(device)

    if is_main:
        params = model.count_parameters()
        logger.info(f"  Model: {params['total']:,} params ({params['total']/1e6:.1f}M)")

    # torch.compile (PyTorch 2.x)
    if args.compile_model:
        if is_main:
            logger.info("  Compiling model with torch.compile...")
        model = torch.compile(model)

    # Wrap in DDP
    if is_distributed:
        model = DDP(model, device_ids=[local_rank], find_unused_parameters=False)

    # Optimizer with scaled LR
    optimizer = torch.optim.AdamW(model.parameters(), lr=scaled_lr, weight_decay=args.weight_decay)

    # Scheduler: linear warmup -> cosine decay
    warmup_epochs = args.warmup_epochs
    total_epochs = args.epochs

    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return float(epoch + 1) / float(max(1, warmup_epochs))
        progress = float(epoch - warmup_epochs) / float(max(1, total_epochs - warmup_epochs))
        return max(1e-2, 0.5 * (1.0 + math.cos(math.pi * progress)))

    scheduler = LambdaLR(optimizer, lr_lambda)

    # Loss
    criterion = TemporalFusionLoss(
        lambda_tc=args.lambda_tc,
        lambda_reg=args.lambda_reg,
        lambda_cs=args.lambda_cs,
        lambda_vl=args.lambda_vl,
        collapse_tau=args.collapse_tau,
        num_classes=args.num_classes,
        label_smoothing=args.label_smoothing,
    ).to(device)

    # AMP scaler
    use_amp = bool(args.amp)
    scaler = GradScaler(enabled=use_amp)

    # Resume
    start_epoch = 0
    if args.resume and Path(args.resume).exists():
        ckpt = torch.load(args.resume, map_location=device)
        model_state = ckpt.get("model", ckpt)
        raw_model = model.module if is_distributed else model
        raw_model.load_state_dict(model_state)
        if "optimizer" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer"])
        if "scaler" in ckpt and scaler is not None:
            scaler.load_state_dict(ckpt["scaler"])
        start_epoch = ckpt.get("epoch", 0) + 1
        if is_main:
            logger.info(f"  Resumed from epoch {start_epoch - 1}")

    # ---- Training loop ----
    best_val_loss = float("inf")
    for epoch in range(start_epoch, args.epochs):
        if is_distributed:
            train_loader.sampler.set_epoch(epoch)

        train_metrics = train_one_epoch(
            model, criterion, optimizer, train_loader, device, epoch,
            scaler=scaler, use_amp=use_amp,
            grad_accum_steps=args.grad_accum_steps, args=args,
        )
        scheduler.step()

        if is_main:
            current_lr = optimizer.param_groups[0]["lr"]
            logger.info(
                f"Epoch {epoch:3d} | "
                f"loss={train_metrics['total']:.4f} | "
                f"cls={train_metrics.get('cls', 0):.4f} | "
                f"tc={train_metrics.get('temporal', 0):.4f} | "
                f"lr={current_lr:.6f} | "
                f"{train_metrics['samples_per_sec']:.0f} samp/s | "
                f"{train_metrics['epoch_time_sec']:.1f}s"
            )
            wandb.log({
                **{f"train/{k}": v for k, v in train_metrics.items()},
                "lr": current_lr,
                "epoch": epoch,
            }, step=epoch)

        # Evaluation
        if epoch % args.eval_every == 0:
            val_metrics = evaluate(model, criterion, val_loader, device, use_amp=use_amp)

            if is_main:
                logger.info(
                    f"         val | "
                    f"loss={val_metrics['val_loss']:.4f} | "
                    f"acc={val_metrics['val_accuracy']:.4f} | "
                    f"tc={val_metrics['val_temporal_consistency']:.4f}"
                )
                wandb.log({f"val/{k}": v for k, v in val_metrics.items()}, step=epoch)

                if val_metrics["val_loss"] < best_val_loss:
                    best_val_loss = val_metrics["val_loss"]
                    raw_model = model.module if is_distributed else model
                    torch.save({
                        "model": raw_model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "scaler": scaler.state_dict() if scaler else None,
                        "epoch": epoch,
                        "val_metrics": val_metrics,
                        "args": vars(args),
                    }, run_dir / "best_model.pt")
                    logger.info(f"  -> New best model (val_loss={best_val_loss:.4f})")

        # Periodic checkpoint
        if is_main and (epoch % args.save_every == 0 or epoch == args.epochs - 1):
            raw_model = model.module if is_distributed else model
            torch.save({
                "model": raw_model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scaler": scaler.state_dict() if scaler else None,
                "epoch": epoch,
            }, run_dir / f"checkpoint_epoch{epoch}.pt")

    # Cleanup
    if is_main:
        wandb.finish()
        logger.info(f"Training complete. Best val_loss: {best_val_loss:.4f}")
    cleanup_distributed()


if __name__ == "__main__":
    main()
