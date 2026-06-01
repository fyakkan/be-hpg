"""Episodic sampler: 1-way K-shot, support and query drawn from DIFFERENT volumes (no leakage)."""
from __future__ import annotations

import numpy as np
import torch


class EpisodicSampler:
    """Samples episodes from a pool of 2D slices grouped by volume.

    Each episode: K support slices and `query_size` query slices, with the support volume(s)
    disjoint from the query volume. Pass a split's slices (train OR test) so episodes never
    mix splits.
    """

    def __init__(self, images, masks, vol_ids, k_shots=(1, 5), query_size=2, seed=0):
        self.images = images                 # (N,H,W)
        self.masks = masks                   # (N,H,W)
        self.vol_ids = np.asarray(vol_ids).astype(str)
        self.k_shots = list(k_shots)
        self.query_size = query_size
        self.rng = np.random.default_rng(seed)
        self.by_vol: dict[str, list[int]] = {}
        for i, v in enumerate(self.vol_ids):
            self.by_vol.setdefault(v, []).append(i)
        self.vols = [v for v, idx in self.by_vol.items() if idx]
        if len(self.vols) < 2:
            raise ValueError("Episodic sampling needs >= 2 volumes for disjoint support/query.")

    def _stack(self, idxs):
        img = torch.from_numpy(np.stack([self.images[i] for i in idxs]).astype(np.float32))
        msk = torch.from_numpy(np.stack([self.masks[i] for i in idxs]).astype(np.float32))
        return img.unsqueeze(1), msk.unsqueeze(1)        # (n,1,H,W)

    def sample(self, k=None) -> dict:
        k = int(self.rng.choice(self.k_shots)) if k is None else int(k)
        qry_vol = self.vols[int(self.rng.integers(len(self.vols)))]
        q_pool = self.by_vol[qry_vol]
        q_idx = list(self.rng.choice(q_pool, size=min(self.query_size, len(q_pool)),
                                     replace=len(q_pool) < self.query_size))
        s_pool = [i for v in self.vols if v != qry_vol for i in self.by_vol[v]]
        s_idx = list(self.rng.choice(s_pool, size=k, replace=len(s_pool) < k))
        sup_img, sup_mask = self._stack(s_idx)
        qry_img, qry_mask = self._stack(q_idx)
        return {"sup_img": sup_img, "sup_mask": sup_mask, "qry_img": qry_img, "qry_mask": qry_mask,
                "k": k, "qry_vol": qry_vol, "sup_idx": s_idx, "qry_idx": q_idx}


def sampler_from_npz(npz_path, manifest_df=None, split=None, **kwargs) -> EpisodicSampler:
    """Build a sampler from a preprocessed .npz, optionally restricted to a manifest split."""
    data = np.load(npz_path)
    images, masks, vol_ids = data["images"], data["masks"], data["vol_ids"].astype(str)
    if split is not None and manifest_df is not None:
        keep_vols = set(manifest_df.loc[manifest_df["split"] == split, "vol_id"].astype(str))
        sel = np.array([v in keep_vols for v in vol_ids])
        images, masks, vol_ids = images[sel], masks[sel], vol_ids[sel]
    return EpisodicSampler(images, masks, vol_ids, **kwargs)
