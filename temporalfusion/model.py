"""
TemporalFusion Model Architecture.

Core model combining:
  - Feature projection from backbone features to latent space
  - Temporal Transformer encoder with configurable attention
  - Hierarchical temporal aggregation via attention-weighted pooling
  - Optional vision-language projection head
"""

import math
from typing import Optional, Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat


class TemporalPositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for temporal sequences."""

    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, T, D)"""
        x = x + self.pe[:, : x.size(1), :]
        return self.dropout(x)


class HierarchicalPoolingLevel(nn.Module):
    """Single level of attention-weighted hierarchical pooling.
    
    Reduces temporal resolution by factor of 2 using learned attention weights.
    """

    def __init__(self, d_model: int):
        super().__init__()
        self.attn_weight = nn.Linear(d_model, 1)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (B, T, D)
            mask: (B, T) — 1 for valid, 0 for padding
        Returns:
            pooled: (B, T//2, D)
            new_mask: (B, T//2)
        """
        B, T, D = x.shape
        # Pad to even length if needed
        if T % 2 != 0:
            x = F.pad(x, (0, 0, 0, 1))  # pad last temporal position
            if mask is not None:
                mask = F.pad(mask, (0, 1), value=0.0)
            T = T + 1

        x_pairs = rearrange(x, "b (t two) d -> b t two d", two=2)  # (B, T//2, 2, D)
        attn_logits = self.attn_weight(x_pairs).squeeze(-1)  # (B, T//2, 2)

        if mask is not None:
            mask_pairs = rearrange(mask, "b (t two) -> b t two", two=2)
            attn_logits = attn_logits.masked_fill(mask_pairs == 0, float("-inf"))
            new_mask = mask_pairs.sum(dim=-1).clamp(max=1)  # (B, T//2)
        else:
            new_mask = None

        attn_weights = F.softmax(attn_logits, dim=-1)  # (B, T//2, 2)
        attn_weights = torch.nan_to_num(attn_weights, 0.0)
        pooled = (attn_weights.unsqueeze(-1) * x_pairs).sum(dim=2)  # (B, T//2, D)
        pooled = self.norm(pooled)
        return pooled, new_mask


class HierarchicalTemporalAggregator(nn.Module):
    """Multi-level hierarchical temporal aggregation.
    
    Creates a temporal feature pyramid from full resolution down to a single token,
    enforcing cross-scale consistency at each level.
    """

    def __init__(self, d_model: int, num_levels: int = 4):
        super().__init__()
        self.num_levels = num_levels
        self.levels = nn.ModuleList([HierarchicalPoolingLevel(d_model) for _ in range(num_levels)])

    def forward(
        self, x: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> Tuple[list, list, torch.Tensor]:
        """
        Args:
            x: (B, T, D) — full-resolution features
            mask: (B, T)
        Returns:
            hierarchy: list of (B, T_l, D) at each level
            masks: list of (B, T_l) at each level
            video_repr: (B, D) — global video representation (mean of last level)
        """
        hierarchy = [x]
        masks = [mask]
        current = x
        current_mask = mask

        for level in self.levels:
            current, current_mask = level(current, current_mask)
            hierarchy.append(current)
            masks.append(current_mask)

        # Global video representation: mean pool the last level
        last = hierarchy[-1]
        if masks[-1] is not None:
            m = masks[-1].unsqueeze(-1)
            video_repr = (last * m).sum(dim=1) / m.sum(dim=1).clamp(min=1)
        else:
            video_repr = last.mean(dim=1)

        return hierarchy, masks, video_repr


class TemporalFusionModel(nn.Module):
    """
    Full TemporalFusion architecture.

    Phase 1: Temporal Transformer encoder + contrastive heads
    Phase 2: + Hierarchical temporal aggregation
    Phase 3: + Vision-language projection (optional)
    """

    def __init__(
        self,
        feature_dim: int = 2048,
        hidden_dim: int = 1024,
        num_heads: int = 8,
        num_layers: int = 6,
        ff_mult: int = 4,
        dropout: float = 0.1,
        max_seq_len: int = 5000,
        num_hierarchy_levels: int = 4,
        num_classes: int = 200,
        vl_embed_dim: int = 256,
        enable_vl: bool = False,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.enable_vl = enable_vl

        # --- Feature projection ---
        self.feature_proj = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # --- Positional encoding ---
        self.pos_enc = TemporalPositionalEncoding(hidden_dim, max_seq_len, dropout)

        # --- Temporal Transformer Encoder ---
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * ff_mult,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,  # pre-LN for training stability
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # --- Hierarchical Aggregation (Phase 2) ---
        self.hierarchy = HierarchicalTemporalAggregator(hidden_dim, num_hierarchy_levels)

        # --- Classification head ---
        self.classifier = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

        # --- Vision-Language projection (Phase 3, optional) ---
        if enable_vl:
            self.vl_proj = nn.Sequential(
                nn.Linear(hidden_dim, vl_embed_dim),
                nn.LayerNorm(vl_embed_dim),
            )
        else:
            self.vl_proj = None

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def encode(
        self, features: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Encode video features through all phases.

        Args:
            features: (B, T, feature_dim) — pre-extracted frame features
            mask: (B, T) — 1.0 for valid frames, 0.0 for padding
        Returns:
            dict with keys:
                'frame_features': (B, T, D) — per-frame hidden states
                'hierarchy': list of tensors at each pooling level
                'video_repr': (B, D) — global video representation
                'vl_embed': (B, vl_embed_dim) — VL projection (if enabled)
        """
        # Project features
        x = self.feature_proj(features)  # (B, T, hidden_dim)
        x = self.pos_enc(x)

        # Transformer encoding
        src_key_padding_mask = (mask == 0) if mask is not None else None
        x = self.transformer(x, src_key_padding_mask=src_key_padding_mask)  # (B, T, D)

        # Hierarchical aggregation
        hierarchy, hier_masks, video_repr = self.hierarchy(x, mask)

        out = {
            "frame_features": x,
            "hierarchy": hierarchy,
            "hier_masks": hier_masks,
            "video_repr": video_repr,
        }

        # VL projection
        if self.enable_vl and self.vl_proj is not None:
            out["vl_embed"] = F.normalize(self.vl_proj(video_repr), dim=-1)

        return out

    def forward(
        self, features: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """Full forward pass returning encoded features + classification logits."""
        encoded = self.encode(features, mask)
        encoded["logits"] = self.classifier(encoded["video_repr"])
        return encoded

    def count_parameters(self) -> Dict[str, int]:
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {"total": total, "trainable": trainable}
