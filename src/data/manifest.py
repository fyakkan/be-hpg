"""Slice<->volume manifest and patient/volume-disjoint splits (no leakage).

The manifest records the volume id of every kept slice so the episodic sampler can keep
support and query on different volumes, and so train/val/test are split BY VOLUME.
"""
from __future__ import annotations

import os
from typing import Sequence

import numpy as np
import pandas as pd


def build_manifest(records: Sequence[dict]) -> pd.DataFrame:
    cols = ["global_index", "vol_id", "slice_idx"]
    df = pd.DataFrame(list(records))
    return df.reindex(columns=cols) if len(df) else pd.DataFrame(columns=cols)


def split_volumes(vol_ids: Sequence, fracs: dict, seed: int = 42) -> dict:
    """Deterministically assign each unique volume id to 'train' / 'val' / 'test'."""
    uniq = sorted(set(map(str, vol_ids)))
    rng = np.random.default_rng(seed)
    uniq = [uniq[i] for i in rng.permutation(len(uniq))]
    n = len(uniq)
    n_train = min(int(round(fracs.get("train", 0.6) * n)), n)
    n_val = min(int(round(fracs.get("val", 0.15) * n)), n - n_train)
    train, val = set(uniq[:n_train]), set(uniq[n_train:n_train + n_val])
    return {v: ("train" if v in train else "val" if v in val else "test") for v in uniq}


def assign_splits(df: pd.DataFrame, vol_split: dict) -> pd.DataFrame:
    df = df.copy()
    df["split"] = df["vol_id"].astype(str).map(vol_split)
    return df


def save_manifest(df: pd.DataFrame, path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    df.to_csv(path, index=False)


def load_manifest(path: str) -> pd.DataFrame:
    return pd.read_csv(path)
