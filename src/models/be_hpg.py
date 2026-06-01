"""Proposed — Boundary-Enhanced Hard Prototype Generation (BE-HPG).

SSP module: boundary band = dilation - erosion of the support mask at feature resolution;
tokens in the band are clustered by mini-batch k-means into HARD prototypes (plus an optional
FG-interior prototype). Query pixels are scored by three cosine-similarity channels fed to a
1x1 conv head: (i) the global FG prototype (reliable region signal), (ii) the max similarity to
the SSP hard/boundary prototypes (the boundary-enhanced contribution), and (iii) the BG
prototype. The head is initialized to the cosine-difference prior temp*(sim_global - sim_bg),
so it starts as a calibrated prototypical classifier and learns to add the boundary signal.
Trained with BCE + lambda * boundary loss.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .backbones import DenseBackbone
from .common import (boundary_band, erode, gather_tokens, hard_prototypes, l2norm,
                     masked_prototype, resize_mask_to)


class BEHPG(nn.Module):
    def __init__(self, backbone: str = "mobilevit_s", embed_dim: int = 256, stride: int = 8,
                 pretrained: bool = True, boundary_kernel: int = 3, kmeans_clusters: int = 5,
                 use_fg_interior: bool = True, use_bg: bool = True, use_ssp: bool = True,
                 seed: int = 0, temperature: float = 20.0):
        super().__init__()
        self.encoder = DenseBackbone(backbone, embed_dim, stride, pretrained)
        self.boundary_kernel = boundary_kernel
        self.kmeans_clusters = kmeans_clusters
        self.use_fg_interior = use_fg_interior
        self.use_bg = use_bg
        self.use_ssp = use_ssp                 # False -> plain global prototypes (SSP-vs-GAP ablation)
        self.seed = seed

        n_ch = 1 + (1 if use_ssp else 0) + (1 if use_bg else 0)
        self.head = nn.Conv2d(n_ch, 1, kernel_size=1)
        with torch.no_grad():                  # cosine-difference prior: +temp*global, 0*hard, -temp*bg
            w = [temperature] + ([0.0] if use_ssp else []) + ([-temperature] if use_bg else [])
            self.head.weight.copy_(torch.tensor(w).view(1, n_ch, 1, 1))
            self.head.bias.zero_()

    def hard_fg_prototypes(self, sf: torch.Tensor, m: torch.Tensor):
        """SSP boundary-band hard prototypes (+ optional FG interior). (P,C) normalized or None."""
        protos = []
        hp = hard_prototypes(gather_tokens(sf, boundary_band(m, self.boundary_kernel)),
                             self.kmeans_clusters, seed=self.seed)
        if hp is not None:
            protos.append(l2norm(hp, dim=1))
        if self.use_fg_interior:
            interior = erode(m, self.boundary_kernel)
            if bool(interior.sum() > 0):
                protos.append(l2norm(masked_prototype(sf, interior), dim=0).unsqueeze(0))
        return torch.cat(protos, dim=0) if protos else None

    def forward(self, sup_img, sup_mask, qry_img):
        H, W = qry_img.shape[-2:]
        sf = l2norm(self.encoder(sup_img))               # (K,C,h,w)
        qf = l2norm(self.encoder(qry_img))               # (Q,C,h,w)
        m = resize_mask_to(sf, sup_mask)                 # (K,1,h,w)

        p_fg = l2norm(masked_prototype(sf, m), dim=0)
        sim_global = torch.einsum("qchw,c->qhw", qf, p_fg)
        chans = [sim_global]
        if self.use_ssp:
            protos = self.hard_fg_prototypes(sf, m)
            if protos is not None:
                chans.append(torch.einsum("qchw,pc->pqhw", qf, protos).max(dim=0).values)
            else:
                chans.append(sim_global)                 # fallback when no FG tokens
        if self.use_bg:
            p_bg = l2norm(masked_prototype(sf, 1.0 - m), dim=0)
            chans.append(torch.einsum("qchw,c->qhw", qf, p_bg))

        logits = self.head(torch.stack(chans, dim=1))    # (Q,1,h,w)
        return F.interpolate(logits, size=(H, W), mode="bilinear", align_corners=False)
