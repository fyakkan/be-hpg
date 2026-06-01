"""Shared model ops: feature/mask alignment, masked prototypes, boundary band, hard prototypes."""
from __future__ import annotations

import torch
import torch.nn.functional as F


def resize_mask_to(feat: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Downsample a (B,1,H,W) mask to feat's (h,w) with nearest interpolation."""
    h, w = feat.shape[-2:]
    if mask.shape[-2:] == (h, w):
        return mask.float()
    return F.interpolate(mask.float(), size=(h, w), mode="nearest")


def l2norm(x: torch.Tensor, dim: int = 1) -> torch.Tensor:
    return F.normalize(x, p=2, dim=dim)


def masked_prototype(feat: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Masked global average pooling: (B,C,h,w) x (B,1,h,w) -> a single (C,) prototype.

    Pools across the whole batch jointly (e.g. all K support images at once).
    """
    num = (feat * mask).sum(dim=(0, 2, 3))            # (C,)
    den = mask.sum(dim=(0, 2, 3)).clamp_min(1e-6)     # (1,)
    return num / den


def boundary_band(mask: torch.Tensor, kernel: int = 3) -> torch.Tensor:
    """dilation - erosion of a binary (B,1,h,w) mask via max/min pooling -> band in {0,1}."""
    pad = kernel // 2
    dil = F.max_pool2d(mask, kernel, stride=1, padding=pad)
    ero = -F.max_pool2d(-mask, kernel, stride=1, padding=pad)
    return (dil - ero).clamp(0, 1)


def erode(mask: torch.Tensor, kernel: int = 3) -> torch.Tensor:
    """Morphological erosion of a binary mask via min pooling."""
    pad = kernel // 2
    return -F.max_pool2d(-mask, kernel, stride=1, padding=pad)


def gather_tokens(feat: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Collect feature vectors where mask > 0.5 across the batch -> (N, C) (differentiable)."""
    c = feat.shape[1]
    f = feat.permute(0, 2, 3, 1).reshape(-1, c)
    return f[mask.reshape(-1) > 0.5]


def hard_prototypes(tokens: torch.Tensor, k: int, seed: int = 0) -> torch.Tensor | None:
    """Mini-batch k-means hard prototypes from (N,C) tokens.

    k-means *assignment* is non-differentiable (run on detached CPU tensors); each returned
    centroid is the differentiable mean of its assigned tokens, so gradients still reach the
    encoder. Returns (k_eff, C), or None when there are no tokens.
    """
    n = tokens.shape[0]
    if n == 0:
        return None
    k_eff = min(k, n)
    if k_eff == 1:
        return tokens.mean(0, keepdim=True)
    from sklearn.cluster import MiniBatchKMeans

    with torch.no_grad():
        x = tokens.detach().to(torch.float32).cpu().numpy()
        labels = MiniBatchKMeans(n_clusters=k_eff, n_init=3, random_state=seed).fit_predict(x)
    labels_t = torch.as_tensor(labels, device=tokens.device)
    protos = [tokens[labels_t == c].mean(0) for c in range(k_eff) if bool((labels_t == c).any())]
    return torch.stack(protos)
