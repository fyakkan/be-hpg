"""Phase 2 gate: Dice/IoU + surface metrics with the both-non-empty rule."""
import numpy as np
import torch

from src.metrics.dice import dice_coef
from src.metrics.iou import iou_score
from src.metrics.postprocess import largest_connected_component
from src.metrics.surface import MetricAccumulator, surface_distances


def test_largest_connected_component():
    m = np.zeros((12, 12))
    m[1:5, 1:5] = 1        # big blob (16 px)
    m[10, 10] = 1          # isolated off-target pixel
    out = largest_connected_component(m)
    assert out[2, 2] == 1 and out[10, 10] == 0          # keeps blob, drops the stray
    assert largest_connected_component(np.zeros((6, 6))).sum() == 0   # empty stays empty


def _block():
    a = torch.zeros(16, 16)
    a[4:12, 4:12] = 1.0
    return a


def test_dice_iou_identical_disjoint_empty():
    a = _block()
    assert dice_coef(a, a) > 0.99 and iou_score(a, a) > 0.99
    b = torch.zeros(16, 16)
    b[0, 0] = 1.0
    assert dice_coef(a, b) < 0.2
    empty = torch.zeros(16, 16)
    assert dice_coef(empty, empty) > 0.99       # both empty -> 1 via eps


def test_surface_distances_and_empty_rule():
    a = _block()
    h, s = surface_distances(a, a)
    assert h == 0 and s == 0
    empty = torch.zeros(16, 16)
    assert surface_distances(a, empty) == (None, None)
    assert surface_distances(empty, a) == (None, None)


def test_accumulator_excludes_empty_surface():
    acc = MetricAccumulator()
    a = _block()
    empty = torch.zeros(16, 16)
    acc.update(a, a)        # both non-empty -> surface counted
    acc.update(empty, a)    # pred empty -> surface excluded, still counts for Dice/IoU
    s = acc.summary()
    assert s["n"] == 2 and s["n_surface"] == 1
    assert abs(s["surface_excluded_frac"] - 0.5) < 1e-9
    assert s["hd95"] is not None
