"""Phase 2 gate: BCE + Sobel boundary loss."""
import torch

from src.losses.bce import bce_loss
from src.losses.boundary import boundary_loss, sobel_boundary, total_loss


def test_bce_finite_and_grad():
    logits = torch.randn(2, 1, 16, 16, requires_grad=True)
    target = (torch.rand(2, 1, 16, 16) > 0.5).float()
    loss = bce_loss(logits, target)
    assert torch.isfinite(loss)
    loss.backward()
    assert logits.grad is not None


def test_sobel_boundary_on_square():
    m = torch.zeros(1, 1, 16, 16)
    m[0, 0, 4:12, 4:12] = 1.0
    b = sobel_boundary(m)
    assert b.sum() > 0
    assert b[0, 0, 8, 8] == 0      # interior is not a boundary
    assert b[0, 0, 0, 0] == 0      # far exterior is not a boundary


def test_boundary_loss_finite_and_grad():
    logits = torch.randn(2, 1, 16, 16, requires_grad=True)
    target = torch.zeros(2, 1, 16, 16)
    target[:, :, 4:12, 4:12] = 1.0
    loss = boundary_loss(logits, target)
    assert torch.isfinite(loss) and loss.item() >= 0
    loss.backward()
    assert logits.grad is not None


def test_balanced_pos_weight():
    from src.losses.bce import balanced_pos_weight, bce_loss

    t = torch.zeros(1, 1, 10, 10)
    t[0, 0, 0, 0] = 1.0                          # 1/100 FG -> neg/pos=99 -> clamped to 20
    assert abs(float(balanced_pos_weight(t, max_pw=20.0)) - 20.0) < 1e-5
    assert float(balanced_pos_weight(torch.zeros(1, 1, 10, 10))) == 1.0   # no FG -> 1

    logits = torch.zeros(1, 1, 10, 10)
    assert bce_loss(logits, t, pos_weight=20.0).item() > bce_loss(logits, t).item()


def test_total_loss_lambda_behaviour():
    logits = torch.randn(2, 1, 16, 16, requires_grad=True)
    target = torch.zeros(2, 1, 16, 16)
    target[:, :, 4:12, 4:12] = 1.0
    l0, p0 = total_loss(logits, target, lam=0.0)
    l1, _ = total_loss(logits, target, lam=0.3)
    assert torch.isclose(l0, p0["bce"])
    assert p0["boundary"].item() == 0.0
    assert l1.item() != l0.item()
