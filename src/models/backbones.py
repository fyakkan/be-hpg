"""timm backbones -> dense feature maps projected to a common embedding dim.

CNNs (resnet50, mobilevit_s) use `features_only` and pick the stage matching the requested
stride; ViTs (deit_tiny_patch16_224) use `forward_features` and reshape patch tokens to a grid.
Grayscale (1-channel) medical slices are repeated to 3 channels for the ImageNet-pretrained nets.
"""
from __future__ import annotations

import timm
import torch
import torch.nn as nn


class DenseBackbone(nn.Module):
    def __init__(self, name: str = "resnet50", embed_dim: int = 256, stride: int = 8,
                 pretrained: bool = True):
        super().__init__()
        self.name = name
        # Pure ViTs (deit_*, vit_*) need token-grid reshaping; MobileViT is a hybrid that
        # exposes features_only, so match by prefix to avoid catching "mobilevit".
        self.is_vit = name.startswith("vit") or name.startswith("deit")
        if self.is_vit:
            self.backbone = timm.create_model(name, pretrained=pretrained, num_classes=0)
            self.patch = 16
            self.stride = self.patch
            self.n_prefix = getattr(self.backbone, "num_prefix_tokens", 1)
            out_chs = self.backbone.num_features
        else:
            self.backbone = timm.create_model(name, pretrained=pretrained, features_only=True)
            reductions = list(self.backbone.feature_info.reduction())
            self.idx = (reductions.index(stride) if stride in reductions
                        else min(range(len(reductions)), key=lambda i: abs(reductions[i] - stride)))
            self.stride = reductions[self.idx]
            out_chs = list(self.backbone.feature_info.channels())[self.idx]
        self.proj = nn.Conv2d(out_chs, embed_dim, kernel_size=1)
        self.embed_dim = embed_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)
        if self.is_vit:
            tokens = self.backbone.forward_features(x)        # (B, n_prefix + h*w, C)
            grid = tokens[:, self.n_prefix:, :]
            hw = grid.shape[1]
            h = w = int(round(hw ** 0.5))
            feat = grid.transpose(1, 2).reshape(x.shape[0], -1, h, w)
        else:
            feat = self.backbone(x)[self.idx]                 # (B, C, h, w)
        return self.proj(feat)
