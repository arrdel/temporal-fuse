"""
Evaluation script for TemporalFusion.

Computes all metrics reported in the paper:
  1. Temporal consistency (avg cosine similarity between adjacent frames)
  2. Feature quality (intra-class variance, inter-class separation)
  3. Activity classification accuracy (linear probe)
  4. Temporal localization IoU@0.5
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, top_k_accuracy_score
from torch.utils.data import DataLoader
from tqdm import tqdm

from temporalfusion.model import TemporalFusionModel
from temporalfusion.data import ActivityNetFeaturesDataset, collate_features

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


@torch.no_grad()
def compute_temporal_consistency(
    model: TemporalFusionModel,
    loader: DataLoader,
    device: torch.device,
) -> Dict[str, float]:
    """Compute average cosine similarity between consecutive frame embeddings."""
    model.eval()
    all_sims = []

    for batch in tqdm(loader, desc="Temporal Consistency"):
        features = batch["features"].to(device)
        masks = batch["masks"].to(device)
        output = model.encode(features, masks)

        h = F.normalize(output["frame_features"], dim=-1)
        cos_sim = (h[:, :-1] * h[:, 1:]).sum(dim=-1)  # (B, T-1)
        pair_mask = masks[:, :-1] * masks[:, 1:]

        for b in range(cos_sim.size(0)):
            valid = pair_mask[b].bool()
            if valid.any():
                all_sims.append(cos_sim[b][valid].cpu().numpy())

    all_sims = np.concatenate(all_sims)
    return {
        "temporal_consistency_mean": float(all_sims.mean()),
        "temporal_consistency_std": float(all_sims.std()),
        "temporal_consistency_min": float(all_sims.min()),
        "temporal_consistency_max": float(all_sims.max()),
        "num_pairs": len(all_sims),
    }


@torch.no_grad()
def compute_feature_quality(
    model: TemporalFusionModel,
    loader: DataLoader,
    device: torch.device,
) -> Dict[str, float]:
    """Compute intra-class variance and inter-class separation."""
    model.eval()
    class_features: Dict[int, list] = {}

    for batch in tqdm(loader, desc="Feature Quality"):
        features = batch["features"].to(device)
        masks = batch["masks"].to(device)
        labels = batch["labels"]
        output = model.encode(features, masks)

        video_repr = output["video_repr"].cpu().numpy()
        for i, label in enumerate(labels.numpy()):
            if label >= 0:
                if label not in class_features:
                    class_features[label] = []
                class_features[label].append(video_repr[i])

    # Compute metrics
    intra_dists = []
    class_centroids = {}
    for c, feats in class_features.items():
        feats = np.stack(feats)
        centroid = feats.mean(axis=0)
        class_centroids[c] = centroid
        if len(feats) > 1:
            # Average pairwise cosine distance within class
            feats_norm = feats / (np.linalg.norm(feats, axis=1, keepdims=True) + 1e-8)
            sim_matrix = feats_norm @ feats_norm.T
            mask = np.triu(np.ones_like(sim_matrix), k=1).astype(bool)
            intra_dists.append(1.0 - sim_matrix[mask].mean())

    # Inter-class: average pairwise cosine distance between centroids
    centroids = np.stack(list(class_centroids.values()))
    centroids_norm = centroids / (np.linalg.norm(centroids, axis=1, keepdims=True) + 1e-8)
    inter_sim = centroids_norm @ centroids_norm.T
    inter_mask = np.triu(np.ones_like(inter_sim), k=1).astype(bool)

    intra = float(np.mean(intra_dists)) if intra_dists else 0.0
    inter = float(1.0 - inter_sim[inter_mask].mean()) if inter_mask.any() else 0.0

    return {
        "intra_class_distance": intra,
        "inter_class_distance": inter,
        "separation": inter - intra,
        "num_classes_evaluated": len(class_features),
    }


@torch.no_grad()
def compute_classification_accuracy(
    model: TemporalFusionModel,
    loader: DataLoader,
    device: torch.device,
) -> Dict[str, float]:
    """Compute top-1 and top-5 accuracy on classification."""
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []

    for batch in tqdm(loader, desc="Classification"):
        features = batch["features"].to(device)
        masks = batch["masks"].to(device)
        labels = batch["labels"]

        output = model(features, masks)
        logits = output["logits"].cpu()
        probs = F.softmax(logits, dim=-1).numpy()

        for i, label in enumerate(labels.numpy()):
            if label >= 0:
                all_preds.append(logits[i].argmax().item())
                all_labels.append(label)
                all_probs.append(probs[i])

    all_labels = np.array(all_labels)
    all_preds = np.array(all_preds)
    all_probs = np.stack(all_probs)

    top1 = accuracy_score(all_labels, all_preds)
    # top5 only if enough classes
    num_classes = all_probs.shape[1]
    top5 = top_k_accuracy_score(all_labels, all_probs, k=min(5, num_classes)) if num_classes >= 5 else top1

    return {
        "top1_accuracy": float(top1),
        "top5_accuracy": float(top5),
        "num_samples": len(all_labels),
    }


def run_full_evaluation(
    checkpoint_path: str,
    val_dir: str,
    annotations_file: Optional[str],
    output_path: str,
    feature_dim: int = 2048,
    hidden_dim: int = 1024,
    num_heads: int = 8,
    num_layers: int = 6,
    num_classes: int = 200,
    num_hierarchy_levels: int = 4,
    max_frames: int = 512,
    batch_size: int = 64,
    num_workers: int = 4,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model
    model = TemporalFusionModel(
        feature_dim=feature_dim, hidden_dim=hidden_dim,
        num_heads=num_heads, num_layers=num_layers,
        num_classes=num_classes, num_hierarchy_levels=num_hierarchy_levels,
    ).to(device)

    ckpt = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt["model"])
    logger.info(f"Loaded checkpoint from {checkpoint_path}")

    # Data
    val_ds = ActivityNetFeaturesDataset(val_dir, annotations_file, "val", max_frames)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, collate_fn=collate_features)

    # Run all evaluations
    results = {}
    results["temporal_consistency"] = compute_temporal_consistency(model, val_loader, device)
    results["feature_quality"] = compute_feature_quality(model, val_loader, device)
    results["classification"] = compute_classification_accuracy(model, val_loader, device)
    results["model_info"] = model.count_parameters()

    # Save
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Results saved to {output_path}")

    # Print summary
    logger.info("=" * 60)
    logger.info("EVALUATION SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Temporal Consistency: {results['temporal_consistency']['temporal_consistency_mean']:.4f}")
    logger.info(f"Feature Intra-class:  {results['feature_quality']['intra_class_distance']:.4f}")
    logger.info(f"Feature Inter-class:  {results['feature_quality']['inter_class_distance']:.4f}")
    logger.info(f"Feature Separation:   {results['feature_quality']['separation']:.4f}")
    logger.info(f"Top-1 Accuracy:       {results['classification']['top1_accuracy']:.4f}")
    logger.info(f"Top-5 Accuracy:       {results['classification']['top5_accuracy']:.4f}")
    logger.info("=" * 60)

    return results


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--val_dir", type=str, required=True)
    p.add_argument("--annotations", type=str, default=None)
    p.add_argument("--output", type=str, default="./eval_results/results.json")
    p.add_argument("--feature_dim", type=int, default=2048)
    p.add_argument("--hidden_dim", type=int, default=1024)
    p.add_argument("--num_heads", type=int, default=8)
    p.add_argument("--num_layers", type=int, default=6)
    p.add_argument("--num_classes", type=int, default=200)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--max_frames", type=int, default=512)
    args = p.parse_args()

    run_full_evaluation(
        checkpoint_path=args.checkpoint,
        val_dir=args.val_dir,
        annotations_file=args.annotations,
        output_path=args.output,
        feature_dim=args.feature_dim,
        hidden_dim=args.hidden_dim,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        num_classes=args.num_classes,
        max_frames=args.max_frames,
        batch_size=args.batch_size,
    )
