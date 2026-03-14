#!/usr/bin/env python3
"""
Prepare Charades features for TemporalFusion training.

Steps:
1. Extract the tar.gz if not already extracted
2. Convert .mat/.csv features to .npy format
3. Organize into train/test splits based on annotations
"""

import csv
import os
import sys
import shutil
import tarfile
from pathlib import Path

import numpy as np

BASE = Path("/home/achinda1/projects/hierarchical-vlm/data/charades")
TAR_FILE = BASE / "Charades_v1_features_rgb.tar.gz"
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


def extract_tarball():
    """Extract tar.gz and return the extraction directory."""
    extract_dir = BASE / "Charades_v1_features_rgb"
    if extract_dir.exists() and any(extract_dir.iterdir()):
        print(f"Already extracted to {extract_dir}")
        return extract_dir

    print(f"Extracting {TAR_FILE} ...")
    with tarfile.open(TAR_FILE, "r:gz") as tar:
        tar.extractall(path=BASE)
    print("Extraction complete.")

    # Find the actual feature directory (may be nested)
    candidates = [
        extract_dir,
        BASE / "Charades_v1_rgb",
        BASE / "Charades_rgb",
    ]
    for c in candidates:
        if c.exists():
            return c

    # Check what was extracted
    for item in sorted(BASE.iterdir()):
        if item.is_dir() and item.name not in ["Charades", "features"]:
            return item

    raise FileNotFoundError("Cannot find extracted features directory")


def convert_feature(src_path):
    """Load a feature file and return as numpy array (T, D)."""
    suffix = src_path.suffix.lower()

    if suffix == ".npy":
        return np.load(src_path).astype(np.float32)
    elif suffix == ".mat":
        try:
            import scipy.io as sio
            mat = sio.loadmat(str(src_path))
            for key in ["feat", "features", "x", "data"]:
                if key in mat:
                    return mat[key].astype(np.float32)
            for key, val in mat.items():
                if not key.startswith("_") and isinstance(val, np.ndarray):
                    arr = val.astype(np.float32)
                    if arr.ndim >= 2:
                        return arr
        except Exception as e:
            print(f"  Warning: failed to load {src_path}: {e}")
            return None
    elif suffix == ".txt" or suffix == ".csv":
        try:
            arr = np.loadtxt(str(src_path), dtype=np.float32, delimiter=",")
            if arr.ndim == 1:
                arr = arr.reshape(1, -1)
            return arr
        except Exception:
            try:
                arr = np.loadtxt(str(src_path), dtype=np.float32)
                if arr.ndim == 1:
                    arr = arr.reshape(1, -1)
                return arr
            except Exception as e:
                print(f"  Warning: failed to load {src_path}: {e}")
                return None
    else:
        print(f"  Skipping unsupported format: {suffix}")
        return None

    return None


def organize_features(feat_dir):
    """Convert and organize features into train/test splits."""
    train_ids = get_video_ids(TRAIN_CSV)
    test_ids = get_video_ids(TEST_CSV)
    print(f"Train videos: {len(train_ids)}, Test videos: {len(test_ids)}")

    TRAIN_OUT.mkdir(parents=True, exist_ok=True)
    TEST_OUT.mkdir(parents=True, exist_ok=True)

    # Find all feature files
    feature_files = []
    for ext in ["*.mat", "*.npy", "*.txt", "*.csv"]:
        feature_files.extend(feat_dir.rglob(ext))
    feature_files = sorted(feature_files)
    print(f"Found {len(feature_files)} feature files in {feat_dir}")

    if not feature_files:
        print("ERROR: No feature files found! Listing directory:")
        for item in sorted(feat_dir.iterdir())[:20]:
            print(f"  {item}")
        return

    # Sample one file to check format
    sample = feature_files[0]
    print(f"Sample file: {sample.name} ({sample.suffix})")
    arr = convert_feature(sample)
    if arr is not None:
        print(f"  Shape: {arr.shape}, dtype: {arr.dtype}")
        print(f"  Mean: {arr.mean():.4f}, Std: {arr.std():.4f}")

    train_count = test_count = skip_count = 0
    for fpath in feature_files:
        video_id = fpath.stem
        arr = convert_feature(fpath)
        if arr is None:
            skip_count += 1
            continue

        if arr.ndim == 1:
            arr = arr.reshape(1, -1)

        if video_id in train_ids:
            np.save(TRAIN_OUT / f"{video_id}.npy", arr)
            train_count += 1
        elif video_id in test_ids:
            np.save(TEST_OUT / f"{video_id}.npy", arr)
            test_count += 1
        else:
            # Some features might not have matching annotations
            skip_count += 1

        if (train_count + test_count) % 500 == 0:
            print(f"  Processed: {train_count} train, {test_count} test, {skip_count} skipped")

    print(f"\nDone! Train: {train_count}, Test: {test_count}, Skipped: {skip_count}")
    print(f"  Train dir: {TRAIN_OUT}")
    print(f"  Test dir:  {TEST_OUT}")

    # Verify dimensions
    if train_count > 0:
        sample_file = next(TRAIN_OUT.glob("*.npy"))
        s = np.load(sample_file)
        print(f"\nFeature verification:")
        print(f"  Shape: {s.shape} (T={s.shape[0]}, D={s.shape[1]})")
        print(f"  Dtype: {s.dtype}")


if __name__ == "__main__":
    print("=" * 60)
    print("Charades Feature Preparation")
    print("=" * 60)

    if not TAR_FILE.exists():
        print(f"ERROR: {TAR_FILE} not found. Download it first.")
        sys.exit(1)

    feat_dir = extract_tarball()
    print(f"Feature directory: {feat_dir}")

    organize_features(feat_dir)
