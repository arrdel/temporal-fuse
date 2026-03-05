"""
Loss functions for TemporalFusion.

Phase 1: Temporal contrastive loss + collapse-prevention regularizer
Phase 2: Cross-scale consistency loss
Phase 3: Vision-language contrastive loss (InfoNCE)
"""

from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class TemporalContrastiveLoss(nn.Module):
    """Temporal contrastive loss: enforce adjacent-frame similarity.
    
    L_temporal = (1 / (T-1)) * sum_{i=1}^{T-1} (1 - cos(h_i, h_{i+1}))
    """

    def forward(self, frame_features: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            frame_features: (B, T, D)
            mask: (B, T) — 1 for valid, 0 for padding
        """
        h = F.normalize(frame_features, dim=-1)
        cos_sim = (h[:, :-1] * h[:, 1:]).sum(dim=-1)  # (B, T-1)

        if mask is not None:
            # Only count pairs where both frames are valid
            pair_mask = mask[:, :-1] * mask[:, 1:]  # (B, T-1)
            loss = ((1.0 - cos_sim) * pair_mask).sum() / pair_mask.sum().clamp(min=1)
        else:
            loss = (1.0 - cos_sim).mean()

        return loss


class CollapsePreventionLoss(nn.Module):
    """Batch-level variance regularization to prevent representation collapse.
    
    L_reg = max(0, tau - std_batch(H))
    
    Maintains representation diversity by ensuring batch-level feature
    standard deviation stays above threshold tau.
    """

    def __init__(self, tau: float = 0.1):
        super().__init__()
        self.tau = tau

    def forward(self, frame_features: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            frame_features: (B, T, D)
            mask: (B, T)
        """
        if mask is not None:
            # Gather valid features
            valid = frame_features[mask.bool()]  # (N_valid, D)
        else:
            valid = frame_features.reshape(-1, frame_features.size(-1))

        if valid.size(0) < 2:
            return torch.tensor(0.0, device=frame_features.device)

        std = valid.std(dim=0).mean()
        loss = F.relu(self.tau - std)
        return loss


class CrossScaleConsistencyLoss(nn.Module):
    """Enforce information preservation across hierarchical levels.
    
    L_consistency^(l) = || h^(l-1)_down - h^(l) ||_2^2
    
    Where h^(l-1)_down is level l-1 downsampled to match level l's resolution.
    """

    def forward(self, hierarchy: List[torch.Tensor], masks: List[Optional[torch.Tensor]]) -> torch.Tensor:
        """
        Args:
            hierarchy: list of (B, T_l, D) tensors from finest to coarsest
            masks: corresponding masks
        """
        total_loss = torch.tensor(0.0, device=hierarchy[0].device)
        num_pairs = 0

        for i in range(1, len(hierarchy)):
            finer = hierarchy[i - 1]  # (B, T_{l-1}, D)
            coarser = hierarchy[i]    # (B, T_l, D)

            # Adaptive average pooling to match coarser resolution
            B, T_fine, D = finer.shape
            T_coarse = coarser.shape[1]
            finer_down = F.adaptive_avg_pool1d(
                finer.transpose(1, 2), T_coarse
            ).transpose(1, 2)  # (B, T_coarse, D)

            mse = F.mse_loss(finer_down, coarser)
            total_loss = total_loss + mse
            num_pairs += 1

        return total_loss / max(num_pairs, 1)


class VisionLanguageContrastiveLoss(nn.Module):
    """InfoNCE contrastive loss for vision-language alignment.
    
    Aligns video embeddings with text embeddings using symmetric contrastive loss.
    """

    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = nn.Parameter(torch.tensor(temperature).log())

    def forward(self, video_embed: torch.Tensor, text_embed: torch.Tensor) -> torch.Tensor:
        """
        Args:
            video_embed: (B, D) — L2-normalized video embeddings
            text_embed: (B, D) — L2-normalized text embeddings
        """
        temperature = self.temperature.exp().clamp(min=0.01, max=100.0)
        logits = torch.matmul(video_embed, text_embed.t()) / temperature  # (B, B)
        labels = torch.arange(logits.size(0), device=logits.device)
        loss_v2t = F.cross_entropy(logits, labels)
        loss_t2v = F.cross_entropy(logits.t(), labels)
        return (loss_v2t + loss_t2v) / 2


class TemporalFusionLoss(nn.Module):
    """Combined loss for TemporalFusion training.
    
    L_total = L_cls + lambda_tc * L_temporal + lambda_reg * L_reg
              + lambda_cs * L_consistency + lambda_vl * L_vl
    """

    def __init__(
        self,
        lambda_tc: float = 1.0,
        lambda_reg: float = 0.1,
        lambda_cs: float = 0.5,
        lambda_vl: float = 0.1,
        collapse_tau: float = 0.1,
        vl_temperature: float = 0.07,
        num_classes: int = 200,
        label_smoothing: float = 0.1,
        multi_label: bool = False,
    ):
        super().__init__()
        self.lambda_tc = lambda_tc
        self.lambda_reg = lambda_reg
        self.lambda_cs = lambda_cs
        self.lambda_vl = lambda_vl
        self.multi_label = multi_label

        if multi_label:
            self.cls_loss = nn.BCEWithLogitsLoss()
        else:
            self.cls_loss = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
        self.temporal_loss = TemporalContrastiveLoss()
        self.collapse_loss = CollapsePreventionLoss(tau=collapse_tau)
        self.consistency_loss = CrossScaleConsistencyLoss()
        self.vl_loss = VisionLanguageContrastiveLoss(temperature=vl_temperature)

    def forward(
        self,
        model_output: Dict[str, torch.Tensor],
        labels: Optional[torch.Tensor] = None,
        text_embed: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute combined loss.

        Args:
            model_output: dict from TemporalFusionModel.forward()
            labels: (B,) class labels (optional — if None, skip cls loss)
            text_embed: (B, D) text embeddings (optional — if None, skip VL loss)
            mask: (B, T) attention mask
        Returns:
            dict with 'total' and individual loss components
        """
        losses = {}
        total = torch.tensor(0.0, device=model_output["frame_features"].device)

        # Classification loss
        if labels is not None and "logits" in model_output:
            if self.multi_label:
                # labels is (B, C) multi-hot float tensor
                losses["cls"] = self.cls_loss(model_output["logits"], labels)
            else:
                # labels is (B,) integer tensor
                losses["cls"] = self.cls_loss(model_output["logits"], labels)
            total = total + losses["cls"]

        # Temporal contrastive loss
        losses["temporal"] = self.temporal_loss(model_output["frame_features"], mask)
        total = total + self.lambda_tc * losses["temporal"]

        # Collapse prevention
        losses["collapse_reg"] = self.collapse_loss(model_output["frame_features"], mask)
        total = total + self.lambda_reg * losses["collapse_reg"]

        # Cross-scale consistency
        if "hierarchy" in model_output and len(model_output["hierarchy"]) > 1:
            losses["consistency"] = self.consistency_loss(
                model_output["hierarchy"], model_output.get("hier_masks", [None] * len(model_output["hierarchy"]))
            )
            total = total + self.lambda_cs * losses["consistency"]

        # Vision-language loss
        if text_embed is not None and "vl_embed" in model_output:
            losses["vl"] = self.vl_loss(model_output["vl_embed"], text_embed)
            total = total + self.lambda_vl * losses["vl"]

        losses["total"] = total
        return losses
