"""Pixel binary cross-entropy on the query mask, with optional class balancing.

Abdominal organs occupy only a few percent of each slice, so plain BCE has a trivial
"predict all background" minimum. A foreground `pos_weight` (balanced per batch) counters
this; it is applied to all models for a fair comparison.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def balanced_pos_weight(target: torch.Tensor, max_pw: float = 20.0) -> torch.Tensor:
    """neg/pos ratio of the target, clamped to [1, max_pw]; 1.0 when there is no foreground."""
    pos = target.float().sum()
    if float(pos) <= 0:
        return torch.tensor(1.0, device=target.device)
    neg = target.numel() - pos
    return torch.clamp(neg / pos, 1.0, max_pw)


def bce_loss(logits: torch.Tensor, target: torch.Tensor, pos_weight=None) -> torch.Tensor:
    pw = None if pos_weight is None else torch.as_tensor(
        pos_weight, device=logits.device, dtype=logits.dtype)
    return F.binary_cross_entropy_with_logits(logits, target.float(), pos_weight=pw)
