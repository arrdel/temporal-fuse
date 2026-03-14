#!/usr/bin/env python3
"""
Prepare Charades features for TemporalFusion training.

The official Charades VGG features are per-frame .txt files organized as:
  Charades_v1_features_rgb/<video_id>/<video_id>-<frame>.txt
Each .txt file has 4096 space-separated float values (VGG-16 fc7 features).

This script aggregates per-frame features into per-video .npy files of shape (T, 4096).
"""

import csv
import os
import sys
from pathlib import Path

import numpy as np

BASE = Path("/home/achinda1/projects/hierarchical-vlm/data/charades")
FEAT_RAW = BASE / "Charades_v1_features_rgb"
ANNO_DIR = BASE / "Charades"
TRAIN_CSV = ANNO_DIR / "Charades_v1_train.csv"
TEST_CSV = ANNO_DIR / "Charades_v1_test.csv"
FEATURE_DIR = BASE / "features"
TRAIN_OUT = FEATURE_DIR / "train"
TEST_OUT = FEATURE_DIR / "test"


def get_video_ids(csv_path):
    """Parse CSV and return set of video IDs."""
    ids = set()
    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            ids.add(row["id"])
    return ids


def aggregate_video_features(video_dir):
    """Load all per-frame .txt files in a video directory and stack into (T, 4096)."""
    txt_files = sorted(video_dir.glob("*.txt"))
    if not txt_files:
        return None

    frames = []
    for tf in txt_files:
        try:
            feat = np.loadtxt(str(tf), dtype=np.float32)
            if feat.shape == (4096,):
                frames.append(feat)
        except Exception:
            continue

    if not frames:
        return None

    return np.stack(frames, axis=0)  # (T, 4096)


def main():
    print("=" * 60)
    print("Charades Feature Preparation")
    print("=" * 60)

    if not FEAT_RAW.exists():
        print(f"ERROR: {FEAT_RAW} not found. Extract the tarball first.")
        sys.exit(1)

    train_ids = get_video_ids(TRAIN_CSV)
    test_ids = get_video_ids(TEST_CSV)
    print(f"Train videos (annotations): {len(train_ids)}")
    print(f"Test videos (annotations): {len(test_ids)}")

    # List all video directories
    video_dirs = sorted([d for d in FEAT_RAW.iterdir() if d.is_dir()])
    print(f"Video directories found: {len(video_dirs)}")

    TRAIN_OUT.mkdir(parents=True, exist_ok=True)
    TEST_OUT.mkdir(parents=True, exist_ok=True)

    train_count = test_count = skip_count = 0
    total = len(video_dirs)

    for i, vdir in enumerate(video_dirs):
        video_id = vdir.name

        # Aggregate per-frame features
        features = aggregate_video_features(vdir)
        if features is None:
            skip_count += 1
            continue

        # Save to appropriate split
        if video_id in train_ids:
            np.save(TRAIN_OUT / f"{video_id}.npy", features)
            train_count += 1
        elif video_id in test_ids:
            np.save(TEST_OUT / f"{video_id}.npy", features)
            test_count += 1
        else:
            skip_count += 1

        if (i + 1) % 500 == 0:
            print(f"  [{i+1}/{total}] train={train_count}, test={test_count}, skip={skip_count}")

    print(f"\nDone!")
    print(f"  Train: {train_count} videos -> {TRAIN_OUT}")
    print(f"  Test:  {test_count} videos -> {TEST_OUT}")
    print(f"  Skipped: {skip_count}")

    # Verify a sample
    if train_count > 0:
        sample = next(TRAIN_OUT.glob("*.npy"))
        arr = np.load(sample)
        print(f"\nSample: {sample.name}")
        print(f"  Shape: {arr.shape} (T={arr.shape[0]}, D={arr.shape[1]})")
        print(f"  Mean: {arr.mean():.4f}, Std: {arr.std():.4f}")
        print(f"  Min: {arr.min():.4f}, Max: {arr.max():.4f}")


if __name__ == "__main__":
    main()
