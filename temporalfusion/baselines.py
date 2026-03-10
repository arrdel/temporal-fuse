"""
Baseline implementations for comparison.

Implements the baseline models described in the paper:
  1. DirectTransformer — standard transformer without hierarchy or contrastive loss
  2. TemporalSegmentBaseline — TSN-style segment averaging
  3. MeanPoolBaseline — simple mean pooling (sanity check)
"""

from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class DirectTransformerBaseline(nn.Module):
    """Standard Transformer encoder trained with classification loss only.
    
    No hierarchical aggregation, no temporal contrastive loss, no collapse prevention.
    Serves as the primary architectural baseline.
    """

    def __init__(
        self,
        feature_dim: int = 2048,
        hidden_dim: int = 1024,
        num_heads: int = 8,
        num_layers: int = 6,
        num_classes: int = 200,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.feature_proj = nn.Linear(feature_dim, hidden_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=num_heads,
            dim_feedforward=hidden_dim * 4, dropout=dropout,
            activation="gelu", batch_first=True, norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.classifier = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, features: torch.Tensor, mask: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        x = self.feature_proj(features)
        src_key_padding_mask = (mask == 0) if mask is not None else None
        x = self.transformer(x, src_key_padding_mask=src_key_padding_mask)

        # Mean pool for video representation
        if mask is not None:
            m = mask.unsqueeze(-1)
            video_repr = (x * m).sum(dim=1) / m.sum(dim=1).clamp(min=1)
        else:
            video_repr = x.mean(dim=1)

        logits = self.classifier(video_repr)
        return {"frame_features": x, "video_repr": video_repr, "logits": logits}


class TemporalSegmentBaseline(nn.Module):
    """TSN-style: divide video into N segments, pool each, concatenate, classify.
    
    Represents classical temporal modeling approaches.
    """

    def __init__(
        self,
        feature_dim: int = 2048,
        num_segments: int = 8,
        num_classes: int = 200,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.num_segments = num_segments
        self.segment_fc = nn.Sequential(
            nn.Linear(feature_dim, feature_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.classifier = nn.Sequential(
            nn.Linear(feature_dim // 2, num_classes),
        )

    def forward(self, features: torch.Tensor, mask: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        B, T, D = features.shape
        seg_len = T // self.num_segments

        segment_feats = []
        for s in range(self.num_segments):
            start = s * seg_len
            end = start + seg_len if s < self.num_segments - 1 else T
            seg = features[:, start:end].mean(dim=1)
            segment_feats.append(seg)

        # Average segment features (TSN consensus)
        video_repr = torch.stack(segment_feats, dim=1).mean(dim=1)
        video_repr = self.segment_fc(video_repr)
        logits = self.classifier(video_repr)

        return {"frame_features": features, "video_repr": video_repr, "logits": logits}


class MeanPoolBaseline(nn.Module):
    """Simple mean pooling baseline (sanity check)."""

    def __init__(self, feature_dim: int = 2048, num_classes: int = 200, dropout: float = 0.1):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Dropout(dropout),
            nn.Linear(feature_dim, num_classes),
        )

    def forward(self, features: torch.Tensor, mask: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        if mask is not None:
            m = mask.unsqueeze(-1)
            video_repr = (features * m).sum(dim=1) / m.sum(dim=1).clamp(min=1)
        else:
            video_repr = features.mean(dim=1)

        logits = self.classifier(video_repr)
        return {"frame_features": features, "video_repr": video_repr, "logits": logits}
