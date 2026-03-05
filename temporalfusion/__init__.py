"""
TemporalFusion: Hierarchical Temporal Aggregation with Contrastive Learning
for Long-Form Video Understanding.

This package implements a three-phase training pipeline:
  Phase 1: Temporal contrastive learning with collapse prevention
  Phase 2: Hierarchical temporal aggregation with cross-scale consistency
  Phase 3: Vision-language alignment (optional, when text supervision is available)

Evaluated on ActivityNet-1.3, Charades, and Kinetics-400.
"""

__version__ = "0.1.0"
