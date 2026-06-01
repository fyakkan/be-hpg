"""Dice similarity coefficient (primary metric)."""
from __future__ import annotations

import torch


def dice_coef(pred: torch.Tensor, gt: torch.Tensor, thr: float = 0.5, eps: float = 1e-6) -> float:
    p = (pred > thr).float()
    g = (gt > 0.5).float()
    inter = (p * g).sum()
    return float((2 * inter + eps) / (p.sum() + g.sum() + eps))
