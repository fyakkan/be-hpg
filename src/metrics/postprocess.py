"""Prediction post-processing. Largest-connected-component keeps only the biggest 2D blob,
which is anatomically valid for a single organ and removes off-target false positives that
inflate surface metrics (HD95/ASD). Applied identically to every model for a fair comparison.
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage


def largest_connected_component(mask) -> np.ndarray:
    """Return a binary uint8 mask keeping only the largest connected component."""
    m = np.asarray(mask) > 0.5
    if not m.any():
        return m.astype(np.uint8)
    labels, n = ndimage.label(m)
    if n <= 1:
        return m.astype(np.uint8)
    counts = np.bincount(labels.ravel())
    counts[0] = 0  # ignore background
    return (labels == int(counts.argmax())).astype(np.uint8)
