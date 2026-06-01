"""Phase 2 gate: episodic sampler shapes + patient/volume-disjoint support/query."""
import numpy as np
import pytest

from src.data.sampler import EpisodicSampler


def _data(n_vol=4, per=5, H=8):
    imgs, masks, vols = [], [], []
    rng = np.random.default_rng(0)
    for v in range(n_vol):
        for _ in range(per):
            imgs.append(rng.random((H, H)).astype(np.float32))
            masks.append((rng.random((H, H)) > 0.5).astype(np.uint8))
            vols.append(f"v{v}")
    return np.stack(imgs), np.stack(masks), np.array(vols)


def test_sampler_shapes_and_disjoint():
    imgs, masks, vols = _data()
    s = EpisodicSampler(imgs, masks, vols, k_shots=[1, 5], query_size=2, seed=0)
    for _ in range(25):
        b = s.sample()
        assert b["sup_img"].shape == (b["k"], 1, 8, 8)
        assert b["qry_img"].shape == (2, 1, 8, 8)
        sup_vols = {vols[i] for i in b["sup_idx"]}
        assert b["qry_vol"] not in sup_vols      # no leakage


def test_sampler_requires_two_volumes():
    imgs, masks, vols = _data(n_vol=1)
    with pytest.raises(ValueError):
        EpisodicSampler(imgs, masks, vols)
