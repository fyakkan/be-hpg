"""Phase 2 gate: hard-prototype k-means edge cases + boundary band + BE-HPG robustness."""
import torch

from src.models.be_hpg import BEHPG
from src.models.common import boundary_band, erode, hard_prototypes


def test_hard_prototypes_counts():
    assert hard_prototypes(torch.zeros(0, 4), 3) is None          # no tokens
    assert hard_prototypes(torch.randn(2, 4), 5).shape[0] == 2    # N < k -> N protos
    assert hard_prototypes(torch.randn(20, 4), 3).shape == (3, 4) # N >= k -> k protos


def test_hard_prototypes_are_differentiable():
    t = torch.randn(20, 4, requires_grad=True)
    hard_prototypes(t, 3).sum().backward()
    assert t.grad is not None and t.grad.abs().sum() > 0


def test_boundary_band_and_erode():
    m = torch.zeros(1, 1, 16, 16)
    m[0, 0, 4:12, 4:12] = 1.0
    band = boundary_band(m, 3)
    assert band.sum() > 0
    assert band[0, 0, 8, 8] == 0          # interior not in the band
    assert erode(m, 3).sum() < m.sum()    # erosion shrinks the mask


def test_behpg_head_initialized_to_cosine_prior():
    m = BEHPG(backbone="mobilevit_s", embed_dim=16, stride=8, pretrained=False, temperature=20.0)
    w = m.head.weight.detach().view(-1)        # channels: [global_fg, hard, bg]
    assert m.head.weight.shape == (1, 3, 1, 1)
    assert abs(float(w[0]) - 20.0) < 1e-4       # +temp on global FG
    assert abs(float(w[1])) < 1e-4              # 0 on the hard channel (learned from there)
    assert abs(float(w[2]) + 20.0) < 1e-4       # -temp on BG
    assert float(m.head.bias.detach()) == 0.0


def test_behpg_handles_empty_support_mask():
    torch.manual_seed(0)
    model = BEHPG(backbone="mobilevit_s", embed_dim=64, stride=8, pretrained=False,
                  kmeans_clusters=3).eval()
    out = model(torch.rand(1, 1, 224, 224), torch.zeros(1, 1, 224, 224), torch.rand(1, 1, 224, 224))
    assert out.shape == (1, 1, 224, 224) and torch.isfinite(out).all()
