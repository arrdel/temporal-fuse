"""
Dataset loaders for TemporalFusion.

Supports:
  - ActivityNet-1.3 (pre-extracted C3D/I3D features)
  - THUMOS-14 (pre-extracted I3D features, 20 action classes)
  - Charades (pre-extracted VGG/I3D features, 157 classes, multi-label)

All datasets produce a common dict format:
  {
    'video_id': str,
    'features': Tensor (T, D),
    'num_frames': int,
    'label': int or Tensor (class index or multi-hot vector),
    'annotations': list of dicts (optional),
  }
"""

import csv
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# ActivityNet-1.3 pre-extracted features
# ---------------------------------------------------------------------------

class ActivityNetFeaturesDataset(Dataset):
    """ActivityNet-1.3 with pre-extracted C3D/CLIP features."""

    # Map from ActivityNet class names → integer labels (built on first load)
    _label_map: Optional[Dict[str, int]] = None

    def __init__(
        self,
        feature_dir: str,
        annotations_file: Optional[str] = None,
        split: str = "train",
        max_frames: int = 512,
        normalize: bool = True,
    ):
        self.feature_dir = Path(feature_dir)
        self.split = split
        self.max_frames = max_frames
        self.normalize = normalize
        self.annotations: Dict[str, Any] = {}

        # Discover feature files
        self.feature_files = sorted(list(self.feature_dir.glob("*.npy")))
        logger.info(f"[ActivityNet/{split}] Found {len(self.feature_files)} feature files")

        # Load annotations
        if annotations_file and Path(annotations_file).exists():
            with open(annotations_file, "r") as f:
                gt = json.load(f)
            self.annotations = gt.get("database", gt)
            logger.info(f"[ActivityNet/{split}] Loaded annotations for {len(self.annotations)} videos")

            # Build label map once
            if ActivityNetFeaturesDataset._label_map is None:
                labels = set()
                for v in self.annotations.values():
                    for ann in v.get("annotations", []):
                        labels.add(ann.get("label", "unknown"))
                ActivityNetFeaturesDataset._label_map = {l: i for i, l in enumerate(sorted(labels))}
                logger.info(f"[ActivityNet] Built label map with {len(ActivityNetFeaturesDataset._label_map)} classes")

    def __len__(self) -> int:
        return len(self.feature_files)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        fpath = self.feature_files[idx]
        video_id = fpath.stem

        features = np.load(fpath).astype(np.float32)
        if features.ndim == 1:
            features = features.reshape(1, -1)

        T, D = features.shape
        if T > self.max_frames:
            indices = np.linspace(0, T - 1, self.max_frames, dtype=np.int64)
            features = features[indices]
            T = self.max_frames

        if self.normalize:
            norms = np.linalg.norm(features, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            features = features / norms

        # Label
        label = -1
        annotations = []
        if video_id in self.annotations:
            anns = self.annotations[video_id].get("annotations", [])
            annotations = anns
            if anns and self._label_map:
                label = self._label_map.get(anns[0].get("label", ""), -1)

        return {
            "video_id": video_id,
            "features": torch.from_numpy(features),
            "num_frames": T,
            "label": label,
            "annotations": annotations,
        }


# ---------------------------------------------------------------------------
# THUMOS-14 pre-extracted I3D features
# ---------------------------------------------------------------------------

class THUMOS14FeaturesDataset(Dataset):
    """THUMOS-14 with pre-extracted I3D features (2048-D).

    Standard protocol: 'validation' split (200 videos) = training,
                       'test' split (212 videos) = evaluation.
    Multi-label videos use the dominant (most frequent) action class.
    """

    _label_map: Optional[Dict[str, int]] = None

    def __init__(
        self,
        feature_dir: str,
        annotations_file: Optional[str] = None,
        split: str = "train",
        max_frames: int = 2048,
        normalize: bool = True,
    ):
        self.feature_dir = Path(feature_dir)
        self.split = split
        self.max_frames = max_frames
        self.normalize = normalize
        self.annotations: Dict[str, Any] = {}

        # Discover feature files
        self.feature_files = sorted(list(self.feature_dir.glob("*.npy")))
        logger.info(f"[THUMOS14/{split}] Found {len(self.feature_files)} feature files")

        # Load annotations
        if annotations_file and Path(annotations_file).exists():
            with open(annotations_file, "r") as f:
                gt = json.load(f)
            self.annotations = gt.get("database", gt)
            logger.info(
                f"[THUMOS14/{split}] Loaded annotations for "
                f"{len(self.annotations)} videos"
            )

            # Build label map once from ALL annotations
            if THUMOS14FeaturesDataset._label_map is None:
                labels = set()
                for v in self.annotations.values():
                    for ann in v.get("annotations", []):
                        labels.add(ann.get("label", "unknown"))
                THUMOS14FeaturesDataset._label_map = {
                    l: i for i, l in enumerate(sorted(labels))
                }
                logger.info(
                    f"[THUMOS14] Built label map with "
                    f"{len(THUMOS14FeaturesDataset._label_map)} classes"
                )

    def __len__(self) -> int:
        return len(self.feature_files)

    def _dominant_label(self, anns: List[Dict]) -> int:
        """Return the most frequent label index for a video."""
        if not anns or not self._label_map:
            return -1
        from collections import Counter
        counts = Counter(a.get("label", "") for a in anns)
        dominant = counts.most_common(1)[0][0]
        return self._label_map.get(dominant, -1)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        fpath = self.feature_files[idx]
        video_id = fpath.stem

        features = np.load(fpath).astype(np.float32)
        if features.ndim == 1:
            features = features.reshape(1, -1)

        T, D = features.shape
        if T > self.max_frames:
            indices = np.linspace(0, T - 1, self.max_frames, dtype=np.int64)
            features = features[indices]
            T = self.max_frames

        if self.normalize:
            norms = np.linalg.norm(features, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            features = features / norms

        label = -1
        annotations = []
        if video_id in self.annotations:
            anns = self.annotations[video_id].get("annotations", [])
            annotations = anns
            label = self._dominant_label(anns)

        return {
            "video_id": video_id,
            "features": torch.from_numpy(features),
            "num_frames": T,
            "label": label,
            "annotations": annotations,
        }


# ---------------------------------------------------------------------------
# Charades (multi-label, 157 action classes)
# ---------------------------------------------------------------------------

class CharadesFeaturesDataset(Dataset):
    """Charades with pre-extracted features (VGG 4096-D or I3D 2048-D).

    Multi-label dataset: each video has multiple action classes.
    Labels are returned as multi-hot vectors of shape (157,).
    Features can be .mat (scipy), .npy, or .txt files.
    """

    NUM_CLASSES = 157

    def __init__(
        self,
        feature_dir: str,
        annotations_csv: Optional[str] = None,
        split: str = "train",
        max_frames: int = 512,
        normalize: bool = True,
    ):
        self.feature_dir = Path(feature_dir)
        self.split = split
        self.max_frames = max_frames
        self.normalize = normalize
        self.annotations: Dict[str, List[str]] = {}  # video_id -> list of class ids

        # Discover feature files (.mat, .npy, .txt)
        self.feature_files = []
        for ext in ["*.npy", "*.mat", "*.txt"]:
            self.feature_files.extend(self.feature_dir.glob(ext))
        self.feature_files = sorted(self.feature_files)
        logger.info(
            f"[Charades/{split}] Found {len(self.feature_files)} feature files"
        )

        # Parse annotations CSV
        if annotations_csv and Path(annotations_csv).exists():
            with open(annotations_csv, "r") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    vid = row["id"]
                    actions_str = row.get("actions", "").strip()
                    classes = set()
                    if actions_str:
                        for seg in actions_str.split(";"):
                            parts = seg.strip().split()
                            if parts:
                                classes.add(parts[0])
                    self.annotations[vid] = sorted(classes)
            logger.info(
                f"[Charades/{split}] Loaded annotations for "
                f"{len(self.annotations)} videos"
            )

    def __len__(self) -> int:
        return len(self.feature_files)

    def _load_features(self, fpath: Path) -> np.ndarray:
        """Load features from various formats."""
        suffix = fpath.suffix.lower()
        if suffix == ".npy":
            return np.load(fpath).astype(np.float32)
        elif suffix == ".mat":
            try:
                import scipy.io as sio
                mat = sio.loadmat(str(fpath))
                # Common key names in Charades .mat files
                for key in ["feat", "features", "x", "data"]:
                    if key in mat:
                        return mat[key].astype(np.float32)
                # Fallback: use the first non-metadata key
                for key, val in mat.items():
                    if not key.startswith("_") and isinstance(val, np.ndarray):
                        return val.astype(np.float32)
            except Exception:
                pass
            raise ValueError(f"Cannot parse .mat file: {fpath}")
        elif suffix == ".txt":
            return np.loadtxt(str(fpath), dtype=np.float32)
        else:
            raise ValueError(f"Unsupported feature format: {suffix}")

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        fpath = self.feature_files[idx]
        video_id = fpath.stem

        features = self._load_features(fpath)
        if features.ndim == 1:
            features = features.reshape(1, -1)

        T, D = features.shape
        if T > self.max_frames:
            indices = np.linspace(0, T - 1, self.max_frames, dtype=np.int64)
            features = features[indices]
            T = self.max_frames

        if self.normalize:
            norms = np.linalg.norm(features, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            features = features / norms

        # Multi-hot label vector
        label = torch.zeros(self.NUM_CLASSES, dtype=torch.float32)
        if video_id in self.annotations:
            for cls_id in self.annotations[video_id]:
                # cls_id is like "c000", "c001", ...
                idx_cls = int(cls_id[1:])  # strip 'c' prefix
                if 0 <= idx_cls < self.NUM_CLASSES:
                    label[idx_cls] = 1.0

        return {
            "video_id": video_id,
            "features": torch.from_numpy(features),
            "num_frames": T,
            "label": label,
            "multi_label": True,
            "annotations": self.annotations.get(video_id, []),
        }


# ---------------------------------------------------------------------------
# Collate
# ---------------------------------------------------------------------------

def collate_features(batch: List[Dict]) -> Dict[str, Any]:
    """Collate variable-length feature sequences with padding.
    
    Handles both single-label (int) and multi-label (tensor) formats.
    """
    video_ids = [b["video_id"] for b in batch]
    num_frames = [b["num_frames"] for b in batch]
    annotations = [b["annotations"] for b in batch]

    is_multi_label = batch[0].get("multi_label", False)

    if is_multi_label:
        labels = torch.stack([b["label"] for b in batch])  # (B, C) multi-hot
    else:
        labels = torch.tensor([b["label"] for b in batch], dtype=torch.long)

    max_len = max(num_frames)
    D = batch[0]["features"].shape[1]

    padded = torch.zeros(len(batch), max_len, D)
    masks = torch.zeros(len(batch), max_len)
    for i, b in enumerate(batch):
        t = b["num_frames"]
        padded[i, :t] = b["features"]
        masks[i, :t] = 1.0

    return {
        "video_ids": video_ids,
        "features": padded,
        "masks": masks,
        "labels": labels,
        "num_frames": num_frames,
        "annotations": annotations,
        "multi_label": is_multi_label,
    }


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def build_dataloaders(
    dataset_name: str,
    train_dir: str,
    val_dir: str,
    annotations_file: Optional[str] = None,
    batch_size: int = 32,
    num_workers: int = 4,
    max_frames: int = 512,
    distributed: bool = False,
) -> Tuple[DataLoader, DataLoader]:
    """Create train/val DataLoaders for a given dataset.
    
    Args:
        distributed: If True, use DistributedSampler for DDP training.
    """

    if dataset_name == "activitynet":
        train_ds = ActivityNetFeaturesDataset(train_dir, annotations_file, "train", max_frames)
        val_ds = ActivityNetFeaturesDataset(val_dir, annotations_file, "val", max_frames)
    elif dataset_name == "thumos14":
        train_ds = THUMOS14FeaturesDataset(train_dir, annotations_file, "train", max_frames)
        val_ds = THUMOS14FeaturesDataset(val_dir, annotations_file, "test", max_frames)
    elif dataset_name == "charades":
        # For Charades, annotations_file should be the directory containing
        # Charades_v1_train.csv and Charades_v1_test.csv
        ann_dir = Path(annotations_file) if annotations_file else None
        train_csv = str(ann_dir / "Charades_v1_train.csv") if ann_dir else None
        test_csv = str(ann_dir / "Charades_v1_test.csv") if ann_dir else None
        train_ds = CharadesFeaturesDataset(train_dir, train_csv, "train", max_frames)
        val_ds = CharadesFeaturesDataset(val_dir, test_csv, "test", max_frames)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    if distributed:
        from torch.utils.data.distributed import DistributedSampler
        train_sampler = DistributedSampler(train_ds, shuffle=True)
        val_sampler = DistributedSampler(val_ds, shuffle=False)
        train_shuffle = False  # sampler handles shuffling
        val_shuffle = False
    else:
        train_sampler = None
        val_sampler = None
        train_shuffle = True
        val_shuffle = False

    use_persistent = num_workers > 0

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=train_shuffle,
        sampler=train_sampler,
        num_workers=num_workers, collate_fn=collate_features,
        pin_memory=True, drop_last=True,
        persistent_workers=use_persistent,
        prefetch_factor=4 if use_persistent else None,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=val_shuffle,
        sampler=val_sampler,
        num_workers=num_workers, collate_fn=collate_features,
        pin_memory=True,
        persistent_workers=use_persistent,
        prefetch_factor=4 if use_persistent else None,
    )
    return train_loader, val_loader
