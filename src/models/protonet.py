"""Baseline 1 — Prototypical Network (Snell et al., 2017).

Masked global average pooling builds FG/BG prototypes; query pixels are scored by cosine
similarity to those prototypes (temperature-scaled). Output is raw logits (use BCEWithLogits).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .backbones import DenseBackbone
from .common import l2norm, masked_prototype, resize_mask_to


class ProtoNet(nn.Module):
    def __init__(self, backbone: str = "resnet50", embed_dim: int = 256, stride: int = 8,
                 pretrained: bool = True, temperature: float = 20.0, use_bg: bool = True):
        super().__init__()
        self.encoder = DenseBackbone(backbone, embed_dim, stride, pretrained)
        self.use_bg = use_bg
        self.log_temp = nn.Parameter(torch.tensor(float(temperature)).log())

    def forward(self, sup_img, sup_mask, qry_img):
        H, W = qry_img.shape[-2:]
        sf = l2norm(self.encoder(sup_img))            # (K,C,h,w)
        qf = l2norm(self.encoder(qry_img))            # (Q,C,h,w)
        m = resize_mask_to(sf, sup_mask)              # (K,1,h,w)

        p_fg = l2norm(masked_prototype(sf, m), dim=0)
        sim_fg = torch.einsum("qchw,c->qhw", qf, p_fg)
        temp = self.log_temp.exp()
        if self.use_bg:
            p_bg = l2norm(masked_prototype(sf, 1.0 - m), dim=0)
            sim_bg = torch.einsum("qchw,c->qhw", qf, p_bg)
            logits = temp * (sim_fg - sim_bg)
        else:
            logits = temp * sim_fg
        return F.interpolate(logits.unsqueeze(1), size=(H, W), mode="bilinear", align_corners=False)
