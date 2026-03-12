"""
Evaluate all ablation checkpoints on a single GPU.
Produces a JSON results file and prints a comparison table.

Usage:
    python3 scripts/eval_ablations.py --dataset thumos14
    python3 scripts/eval_ablations.py --dataset charades
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import average_precision_score
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from temporalfusion.data import (
    THUMOS14FeaturesDataset,
    CharadesFeaturesDataset,
    ActivityNetFeaturesDataset,
    collate_features,
)
from temporalfusion.model import TemporalFusionModel


VARIANTS = ["full", "no_temporal", "no_collapse", "no_crossscale", "cls_only"]

DATASET_CONFIGS = {
    "thumos14": dict(
        feature_dim=2048,
        hidden_dim=1024,
        num_heads=8,
        num_layers=6,
        num_classes=20,
        num_hierarchy_levels=4,
        max_seq_len=5000,
        multi_label=False,
    ),
    "charades": dict(
        feature_dim=4096,
        hidden_dim=1024,
        num_heads=8,
        num_layers=6,
        num_classes=157,
        num_hierarchy_levels=4,
        max_seq_len=5000,
        multi_label=True,
    ),
}


def build_eval_loader(dataset_name: str):
    if dataset_name == "thumos14":
        ds = THUMOS14FeaturesDataset(
            feature_dir="./data/thumos14/i3d_features/test",
            annotations_file="./data/thumos14/gt.json",
            split="test",
            max_frames=2048,
        )
    elif dataset_name == "charades":
        ds = CharadesFeaturesDataset(
            feature_dir="./data/charades/features/test",
            annotations_csv="./data/charades/Charades/Charades_v1_test.csv",
            split="test",
            max_frames=512,
        )
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    loader = DataLoader(
        ds,
        batch_size=32,
        shuffle=False,
        num_workers=4,
        collate_fn=collate_features,
        pin_memory=True,
    )
    return ds, loader


@torch.no_grad()
def evaluate_checkpoint(ckpt_path, dataset_name, device):
    cfg = DATASET_CONFIGS[dataset_name]
    multi_label = cfg.pop("multi_label", False)

    model = TemporalFusionModel(**{k: v for k, v in cfg.items()}).to(device)
    cfg["multi_label"] = multi_label  # restore

    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    epoch = ckpt.get("epoch", -1)
    val_metrics = ckpt.get("val_metrics", {})

    ds, loader = build_eval_loader(dataset_name)
    num_classes = cfg["num_classes"]

    all_preds = []
    all_labels = []
    tc_sim = 0.0
    tc_pairs = 0.0

    for batch in tqdm(loader, desc=f"Eval {Path(ckpt_path).parent.name}", leave=False):
        feat = batch["features"].to(device, non_blocking=True)
        mask = batch["masks"].to(device, non_blocking=True)
        lbl = batch["labels"]

        out = model(feat, mask)

        if multi_label:
            probs = torch.sigmoid(out["logits"]).cpu().numpy()
        else:
            probs = out["logits"].cpu().numpy()
        all_preds.append(probs)
        all_labels.append(lbl.numpy() if isinstance(lbl, torch.Tensor) else np.array(lbl))

        # temporal consistency
        h = F.normalize(out["frame_features"].float(), dim=-1)
        cs = (h[:, :-1] * h[:, 1:]).sum(-1)
        pm = mask[:, :-1] * mask[:, 1:]
        tc_sim += (cs * pm).sum().item()
        tc_pairs += pm.sum().item()

    all_preds = np.concatenate(all_preds, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    tc = tc_sim / max(tc_pairs, 1)

    results = {
        "temporal_consistency": float(tc),
        "num_samples": len(all_preds),
        "best_epoch": int(epoch),
    }

    if multi_label:
        aps = []
        for c in range(num_classes):
            if all_labels[:, c].sum() > 0:
                ap = average_precision_score(all_labels[:, c], all_preds[:, c])
                aps.append(ap)
        results["mAP"] = float(np.mean(aps))
        results["num_classes_evaluated"] = len(aps)
    else:
        preds_cls = all_preds.argmax(axis=1)
        top1 = (preds_cls == all_labels).mean()
        # top5
        top5_preds = np.argsort(all_preds, axis=1)[:, -5:]
        top5 = np.mean([all_labels[i] in top5_preds[i] for i in range(len(all_labels))])
        results["top1_accuracy"] = float(top1)
        results["top5_accuracy"] = float(top5)

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, choices=["thumos14", "charades"])
    args = parser.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    runs_dir = Path("./runs")
    out_file = Path(f"./eval_results/ablation_{args.dataset}.json")
    out_file.parent.mkdir(parents=True, exist_ok=True)

    all_results = {}
    for variant in VARIANTS:
        run_name = f"ablation_{args.dataset}_{variant}"
        ckpt_path = runs_dir / run_name / "best_model.pt"

        if not ckpt_path.exists():
            print(f"[{variant}] Checkpoint not found: {ckpt_path} -- SKIPPING")
            continue

        print(f"\n[{variant}] Evaluating {ckpt_path} ...")
        results = evaluate_checkpoint(str(ckpt_path), args.dataset, device)
        all_results[variant] = results
        print(f"  -> {results}")

    # Save JSON
    with open(out_file, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {out_file}")

    # Print table
    is_multi = DATASET_CONFIGS[args.dataset].get("multi_label", False)
    print()
    print("=" * 80)
    if is_multi:
        print(f"{'Variant':<25} {'mAP':>8} {'TC':>8} {'Epoch':>6}")
    else:
        print(f"{'Variant':<25} {'Top-1':>8} {'Top-5':>8} {'TC':>8} {'Epoch':>6}")
    print("-" * 80)

    variant_labels = {
        "full": "Full Model",
        "no_temporal": "w/o Temporal Contr.",
        "no_collapse": "w/o Collapse Prev.",
        "no_crossscale": "w/o Cross-Scale",
        "cls_only": "Classification Only",
    }

    for variant in VARIANTS:
        if variant not in all_results:
            continue
        r = all_results[variant]
        label = variant_labels[variant]
        if is_multi:
            print(f"{label:<25} {r['mAP']*100:>7.2f}% {r['temporal_consistency']:>8.4f} {r['best_epoch']:>6d}")
        else:
            print(f"{label:<25} {r['top1_accuracy']*100:>7.2f}% {r['top5_accuracy']*100:>7.2f}% {r['temporal_consistency']:>8.4f} {r['best_epoch']:>6d}")
    print("=" * 80)


if __name__ == "__main__":
    main()
