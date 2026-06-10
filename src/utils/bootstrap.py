"""Percentile bootstrap confidence intervals over the fixed test-episode set.

Given the per-slice metric values produced by a single trained model on the fixed eval
episodes, we resample slices with replacement to obtain a non-parametric 95% interval on the
mean. This quantifies the sampling variability of the reported score *under the single seed we
ran* (it is not a cross-seed interval); we state this honestly in the paper.
"""
from __future__ import annotations

import numpy as np


def bootstrap_ci(values, *, n_boot: int = 2000, alpha: float = 0.05, seed: int = 0):
    """Return (mean, lo, hi) for the percentile bootstrap of the mean of `values`.

    `values` is a 1-D sequence of per-slice metric scores. Empty -> (nan, nan, nan).
    """
    v = np.asarray(values, dtype=np.float64)
    if v.size == 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, v.size, size=(n_boot, v.size))
    boot_means = v[idx].mean(axis=1)
    lo, hi = np.percentile(boot_means, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(v.mean()), float(lo), float(hi)


def fmt_ci(values, fmt="{:.3f}", **kw):
    """`mean [lo, hi]` string for a metric's per-slice values (paper/console use)."""
    m, lo, hi = bootstrap_ci(values, **kw)
    return f"{fmt.format(m)} [{fmt.format(lo)}, {fmt.format(hi)}]"
