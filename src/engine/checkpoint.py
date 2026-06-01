"""Resumable checkpointing: model / optimizer / scheduler / episode / RNG state -> Drive."""
from __future__ import annotations

import os

import numpy as np
import torch


def save_checkpoint(path, model, optimizer=None, scheduler=None, episode=0, extra=None):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    state = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict() if optimizer is not None else None,
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
        "episode": episode,
        "torch_rng": torch.get_rng_state(),
        "numpy_rng": np.random.get_state(),
        "extra": extra or {},
    }
    tmp = path + ".tmp"
    torch.save(state, tmp)
    os.replace(tmp, path)            # atomic: a disconnect mid-write never corrupts the ckpt
    return path


def load_checkpoint(path, model, optimizer=None, scheduler=None, map_location="cpu"):
    state = torch.load(path, map_location=map_location, weights_only=False)
    # Robust to two formats: our wrapped dict, or a raw state_dict (e.g. older runs that
    # did torch.save(model.state_dict())). Raw -> load weights, report episode 0.
    if not (isinstance(state, dict) and "model" in state):
        model.load_state_dict(state)
        return 0, {}
    model.load_state_dict(state["model"])
    if optimizer is not None and state.get("optimizer"):
        optimizer.load_state_dict(state["optimizer"])
    if scheduler is not None and state.get("scheduler"):
        scheduler.load_state_dict(state["scheduler"])
    try:
        torch.set_rng_state(state["torch_rng"])
        np.random.set_state(state["numpy_rng"])
    except Exception:
        pass
    return state.get("episode", 0), state.get("extra", {})
