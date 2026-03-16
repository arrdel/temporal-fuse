"""
Tests for TemporalFusion model, losses, and data pipeline.
"""

import pytest
import torch
import numpy as np
import tempfile
from pathlib import Path

from temporalfusion.model import TemporalFusionModel, HierarchicalTemporalAggregator
from temporalfusion.losses import (
    TemporalContrastiveLoss,
    CollapsePreventionLoss,
    CrossScaleConsistencyLoss,
    TemporalFusionLoss,
)
from temporalfusion.baselines import DirectTransformerBaseline, TemporalSegmentBaseline, MeanPoolBaseline


# ---- Model Tests ----

class TestTemporalFusionModel:
    def test_forward_shape(self):
        model = TemporalFusionModel(feature_dim=512, hidden_dim=256, num_heads=4, num_layers=2, num_classes=10)
        x = torch.randn(2, 32, 512)
        mask = torch.ones(2, 32)
        out = model(x, mask)
        assert out["frame_features"].shape == (2, 32, 256)
        assert out["video_repr"].shape == (2, 256)
        assert out["logits"].shape == (2, 10)
        assert len(out["hierarchy"]) > 1

    def test_forward_no_mask(self):
        model = TemporalFusionModel(feature_dim=512, hidden_dim=256, num_heads=4, num_layers=2, num_classes=10)
        x = torch.randn(2, 32, 512)
        out = model(x)
        assert out["logits"].shape == (2, 10)

    def test_variable_length(self):
        model = TemporalFusionModel(feature_dim=512, hidden_dim=256, num_heads=4, num_layers=2, num_classes=10)
        x = torch.randn(2, 64, 512)
        mask = torch.ones(2, 64)
        mask[1, 40:] = 0  # second sample is shorter
        out = model(x, mask)
        assert out["logits"].shape == (2, 10)

    def test_parameter_count(self):
        model = TemporalFusionModel(feature_dim=2048, hidden_dim=1024, num_heads=8, num_layers=6, num_classes=200)
        params = model.count_parameters()
        assert params["total"] > 0
        assert params["trainable"] == params["total"]

    def test_vl_projection(self):
        model = TemporalFusionModel(feature_dim=512, hidden_dim=256, num_heads=4, num_layers=2, enable_vl=True, vl_embed_dim=128)
        x = torch.randn(2, 32, 512)
        out = model.encode(x)
        assert "vl_embed" in out
        assert out["vl_embed"].shape == (2, 128)
        # Should be normalized
        norms = out["vl_embed"].norm(dim=-1)
        assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)


class TestHierarchicalAggregator:
    def test_pooling_reduces_length(self):
        agg = HierarchicalTemporalAggregator(d_model=128, num_levels=3)
        x = torch.randn(2, 64, 128)
        hierarchy, masks, video_repr = agg(x)
        # Should have 4 levels: 64, 32, 16, 8
        assert len(hierarchy) == 4
        assert hierarchy[0].shape[1] == 64
        assert hierarchy[1].shape[1] == 32
        assert hierarchy[2].shape[1] == 16
        assert hierarchy[3].shape[1] == 8
        assert video_repr.shape == (2, 128)


# ---- Loss Tests ----

class TestLosses:
    def test_temporal_contrastive(self):
        loss_fn = TemporalContrastiveLoss()
        h = torch.randn(4, 32, 128)
        mask = torch.ones(4, 32)
        loss = loss_fn(h, mask)
        assert loss.ndim == 0
        assert loss.item() >= 0

    def test_collapse_prevention(self):
        loss_fn = CollapsePreventionLoss(tau=0.1)
        # Collapsed features (all same)
        h_collapsed = torch.ones(4, 32, 128) * 0.5
        loss_collapsed = loss_fn(h_collapsed)
        # Diverse features
        h_diverse = torch.randn(4, 32, 128)
        loss_diverse = loss_fn(h_diverse)
        # Collapsed should have higher regularization loss
        assert loss_collapsed > loss_diverse

    def test_cross_scale_consistency(self):
        loss_fn = CrossScaleConsistencyLoss()
        hierarchy = [torch.randn(2, 32, 128), torch.randn(2, 16, 128), torch.randn(2, 8, 128)]
        masks = [None, None, None]
        loss = loss_fn(hierarchy, masks)
        assert loss.ndim == 0
        assert loss.item() >= 0

    def test_combined_loss(self):
        model = TemporalFusionModel(feature_dim=512, hidden_dim=256, num_heads=4, num_layers=2, num_classes=10)
        criterion = TemporalFusionLoss(num_classes=10)
        x = torch.randn(4, 32, 512)
        mask = torch.ones(4, 32)
        labels = torch.randint(0, 10, (4,))
        output = model(x, mask)
        losses = criterion(output, labels=labels, mask=mask)
        assert "total" in losses
        assert "temporal" in losses
        assert "collapse_reg" in losses
        assert losses["total"].requires_grad


# ---- Baseline Tests ----

class TestBaselines:
    def test_direct_transformer(self):
        model = DirectTransformerBaseline(feature_dim=512, hidden_dim=256, num_heads=4, num_layers=2, num_classes=10)
        x = torch.randn(2, 32, 512)
        mask = torch.ones(2, 32)
        out = model(x, mask)
        assert out["logits"].shape == (2, 10)

    def test_tsn_baseline(self):
        model = TemporalSegmentBaseline(feature_dim=512, num_segments=4, num_classes=10)
        x = torch.randn(2, 32, 512)
        out = model(x)
        assert out["logits"].shape == (2, 10)

    def test_mean_pool_baseline(self):
        model = MeanPoolBaseline(feature_dim=512, num_classes=10)
        x = torch.randn(2, 32, 512)
        out = model(x)
        assert out["logits"].shape == (2, 10)


# ---- Data Tests ----

class TestData:
    def test_collate(self):
        from temporalfusion.data import collate_features
        batch = [
            {"video_id": "v1", "features": torch.randn(20, 512), "num_frames": 20, "label": 0, "annotations": []},
            {"video_id": "v2", "features": torch.randn(30, 512), "num_frames": 30, "label": 1, "annotations": []},
        ]
        collated = collate_features(batch)
        assert collated["features"].shape == (2, 30, 512)
        assert collated["masks"].shape == (2, 30)
        assert collated["masks"][0, 20:].sum() == 0  # padded
        assert collated["masks"][1].sum() == 30  # full

    def test_dataset_with_synthetic_features(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create synthetic feature files
            for i in range(5):
                feats = np.random.randn(50, 512).astype(np.float32)
                np.save(Path(tmpdir) / f"video_{i:03d}.npy", feats)

            from temporalfusion.data import ActivityNetFeaturesDataset
            ds = ActivityNetFeaturesDataset(tmpdir, split="test", max_frames=32)
            assert len(ds) == 5
            item = ds[0]
            assert item["features"].shape == (32, 512)
            assert item["num_frames"] == 32


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
