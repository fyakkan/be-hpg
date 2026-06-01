"""Boundary loss: weighted BCE evaluated on Sobel-derived ground-truth boundary pixels.

Total objective: L_total = L_BCE + lambda * L_boundary  (proposal eq. in Methods).
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

_SOBEL_X = torch.tensor([[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]]).view(1, 1, 3, 3)


def sobel_boundary(mask: torch.Tensor) -> torch.Tensor:
    """Binary boundary map (B,1,H,W) from a binary mask via Sobel gradient magnitude."""
    m = mask.float()
    kx = _SOBEL_X.to(m.device, m.dtype)
    ky = kx.transpose(2, 3)
    gx = F.conv2d(m, kx, padding=1)
    gy = F.conv2d(m, ky, padding=1)
    grad = torch.sqrt(gx * gx + gy * gy + 1e-12)
    return (grad > 1e-6).float()


def boundary_loss(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Mean BCE computed over the GT boundary pixels only."""
    b = sobel_boundary(target)
    bce = F.binary_cross_entropy_with_logits(logits, target.float(), reduction="none")
    return (bce * b).sum() / b.sum().clamp_min(1.0)


def total_loss(logits: torch.Tensor, target: torch.Tensor, lam: float = 0.3,
               balanced: bool = True, max_pw: float = 20.0):
    """Return (loss, parts dict). lam<=0 disables the boundary term.

    `balanced` applies a per-batch foreground pos_weight to the main BCE (counters the
    background-collapse failure mode); applied identically to every model for fairness.
    """
    from .bce import balanced_pos_weight, bce_loss

    pw = balanced_pos_weight(target, max_pw) if balanced else None
    l_bce = bce_loss(logits, target, pos_weight=pw)
    if lam <= 0:
        zero = torch.zeros((), device=logits.device)
        return l_bce, {"bce": l_bce.detach(), "boundary": zero, "total": l_bce.detach()}
    l_bnd = boundary_loss(logits, target)
    total = l_bce + lam * l_bnd
    return total, {"bce": l_bce.detach(), "boundary": l_bnd.detach(), "total": total.detach()}
