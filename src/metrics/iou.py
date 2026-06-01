"""Intersection-over-Union (Jaccard index)."""
from __future__ import annotations

import torch


def iou_score(pred: torch.Tensor, gt: torch.Tensor, thr: float = 0.5, eps: float = 1e-6) -> float:
    p = (pred > thr).float()
    g = (gt > 0.5).float()
    inter = (p * g).sum()
    union = p.sum() + g.sum() - inter
    return float((inter + eps) / (union + eps))
