"""Baseline 2 — Meta-RCNN-INSPIRED segmentation baseline.

DISCLOSURE: Meta R-CNN (Yan et al., 2019) is a detector. We adapt its core idea —
support-conditioned, class-attentive channel reweighting of features — to dense
segmentation: a reweighting vector from masked support pooling gates the query feature
channels, then a small conv head predicts the mask. This is NOT the original detection
architecture, and the adaptation is stated plainly in the paper.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .backbones import DenseBackbone
from .common import masked_prototype, resize_mask_to


class MetaReweight(nn.Module):
    def __init__(self, backbone: str = "resnet50", embed_dim: int = 256, stride: int = 8,
                 pretrained: bool = True, hidden: int = 256):
        super().__init__()
        self.encoder = DenseBackbone(backbone, embed_dim, stride, pretrained)
        self.reweight = nn.Sequential(
            nn.Linear(embed_dim, hidden), nn.ReLU(inplace=True),
            nn.Linear(hidden, embed_dim), nn.Sigmoid(),
        )
        self.seg_head = nn.Sequential(
            nn.Conv2d(embed_dim, hidden, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(hidden, 1, 1),
        )

    def forward(self, sup_img, sup_mask, qry_img):
        H, W = qry_img.shape[-2:]
        sf = self.encoder(sup_img)                    # (K,C,h,w)
        qf = self.encoder(qry_img)                    # (Q,C,h,w)
        m = resize_mask_to(sf, sup_mask)
        v = masked_prototype(sf, m)                   # (C,) support-conditioned vector
        w = self.reweight(v)                          # (C,) channel gate in [0,1]
        qf_mod = qf * w.view(1, -1, 1, 1)
        logits = self.seg_head(qf_mod)                # (Q,1,h,w)
        return F.interpolate(logits, size=(H, W), mode="bilinear", align_corners=False)
