"""Surface metrics (HD95, ASD) with the both-non-empty rule, plus a metric accumulator.

HD95/ASD are undefined when either mask is empty; such slices are EXCLUDED from the surface
averages and counted, and the excluded fraction is reported (no fabricated penalty value).
"""
from __future__ import annotations

import numpy as np

from .dice import dice_coef
from .iou import iou_score


def _to_numpy_bool(x, thr=0.5):
    if hasattr(x, "detach"):
        x = x.detach().cpu().numpy()
    return np.asarray(x) > thr


def surface_distances(pred, gt):
    """Return (hd95, asd) in pixels, or (None, None) if either mask is empty."""
    p = _to_numpy_bool(pred)
    g = _to_numpy_bool(gt, thr=0.5)
    if p.sum() == 0 or g.sum() == 0:
        return None, None
    from medpy.metric.binary import asd, hd95
    return float(hd95(p, g)), float(asd(p, g))


class MetricAccumulator:
    """Dice/IoU over all slices; HD95/ASD over both-non-empty slices only."""

    def __init__(self):
        self.dice, self.iou, self.hd95, self.asd = [], [], [], []
        self.n = self.n_surface = self.n_excluded = 0

    def update(self, pred, gt):
        self.dice.append(dice_coef(pred, gt))
        self.iou.append(iou_score(pred, gt))
        h, a = surface_distances(pred, gt)
        self.n += 1
        if h is None:
            self.n_excluded += 1
        else:
            self.hd95.append(h)
            self.asd.append(a)
            self.n_surface += 1

    def summary(self) -> dict:
        mean = lambda xs: float(np.mean(xs)) if xs else None
        return {
            "dice": mean(self.dice) or 0.0,
            "iou": mean(self.iou) or 0.0,
            "hd95": mean(self.hd95),
            "asd": mean(self.asd),
            "n": self.n,
            "n_surface": self.n_surface,
            "surface_excluded_frac": (self.n_excluded / self.n) if self.n else 0.0,
        }

    def raw(self) -> dict:
        """Per-slice metric lists, for bootstrap confidence intervals.

        Dice/IoU have `n` entries (all slices); HD95/ASD have `n_surface` entries (the
        both-non-empty slices only). Lists are plain floats so they serialise to JSON.
        """
        return {
            "dice": [float(x) for x in self.dice],
            "iou": [float(x) for x in self.iou],
            "hd95": [float(x) for x in self.hd95],
            "asd": [float(x) for x in self.asd],
        }
