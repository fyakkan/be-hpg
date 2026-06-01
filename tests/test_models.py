"""Phase 2 gate: forward+backward of all three models on CPU with tiny random tensors.

Uses pretrained=False (no network) and asserts output shape, finite loss, and non-zero
gradients reaching the encoder projection.
"""
import pytest
import torch

from src.engine.build import build_model
from src.losses.boundary import total_loss


def _cfg(name, backbone, **model_extra):
    m = {"name": name, "backbone": backbone, "backbone_feature_stride": 8, "pretrained": False}
    m.update(model_extra)
    return {"embed_dim": 64, "seed": 0, "model": m}


_SSP = {"boundary_kernel": 3, "kmeans_clusters": 3,
        "use_fg_interior_prototype": True, "use_bg_prototype": True}

CONFIGS = [
    _cfg("protonet", "resnet50"),
    _cfg("meta_reweight", "resnet50", reweight_hidden=64),
    _cfg("be_hpg", "mobilevit_s", ssp=_SSP),
    _cfg("be_hpg", "deit_tiny_patch16_224", ssp=_SSP),
]
IDS = [f"{c['model']['name']}-{c['model']['backbone']}" for c in CONFIGS]


def _episode(H=224, K=2, Q=2):
    g = torch.Generator().manual_seed(0)
    return (torch.rand(K, 1, H, H, generator=g),
            (torch.rand(K, 1, H, H, generator=g) > 0.7).float(),
            torch.rand(Q, 1, H, H, generator=g),
            (torch.rand(Q, 1, H, H, generator=g) > 0.7).float())


@pytest.mark.parametrize("cfg", CONFIGS, ids=IDS)
def test_forward_backward(cfg):
    torch.manual_seed(0)
    model = build_model(cfg).train()
    sup_img, sup_mask, qry_img, qry_mask = _episode()

    logits = model(sup_img, sup_mask, qry_img)
    assert logits.shape == (2, 1, 224, 224)

    loss, parts = total_loss(logits, qry_mask, lam=0.3)
    assert torch.isfinite(loss)
    loss.backward()

    total_grad = sum(p.grad.abs().sum().item() for p in model.parameters() if p.grad is not None)
    assert total_grad > 0
    assert model.encoder.proj.weight.grad is not None
    assert model.encoder.proj.weight.grad.abs().sum().item() > 0
