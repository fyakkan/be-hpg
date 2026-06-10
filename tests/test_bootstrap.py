"""Phase-2 gate for the revision: percentile bootstrap CI on per-slice metrics."""
import math

import numpy as np

from src.utils.bootstrap import bootstrap_ci, fmt_ci


def test_ci_brackets_mean_and_is_ordered():
    rng = np.random.default_rng(0)
    vals = rng.normal(0.9, 0.05, size=300).clip(0, 1)
    m, lo, hi = bootstrap_ci(vals, n_boot=1000, seed=1)
    assert lo < m < hi                      # mean sits inside its own CI
    assert math.isclose(m, float(np.mean(vals)), rel_tol=1e-9)
    assert hi - lo < 0.05                    # n=300 -> tight interval


def test_ci_is_reproducible_with_seed():
    vals = [0.8, 0.85, 0.9, 0.95, 1.0] * 20
    a = bootstrap_ci(vals, n_boot=500, seed=42)
    b = bootstrap_ci(vals, n_boot=500, seed=42)
    assert a == b


def test_constant_values_have_zero_width():
    _, lo, hi = bootstrap_ci([0.5] * 50, n_boot=300, seed=0)
    assert lo == hi == 0.5


def test_empty_returns_nan_and_fmt_runs():
    m, lo, hi = bootstrap_ci([])
    assert all(math.isnan(x) for x in (m, lo, hi))
    s = fmt_ci([0.9, 0.92, 0.94], seed=0)
    assert "[" in s and "]" in s
