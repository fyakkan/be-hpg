"""YAML config loader with `base:` deep-merge and dotted-key overrides.

Configs are the single source of truth. A model config may set `base: <file>` (path
relative to its own directory); the base is loaded first and the child is deep-merged on
top. CLI/programmatic overrides are `dotted.key=value` strings applied last.
"""
from __future__ import annotations

import copy
import os
from typing import Any, Iterable

import yaml


def _deep_merge(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in override.items():
        if k == "base":
            continue
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def _coerce(value: str) -> Any:
    """Interpret an override value as a YAML scalar (int/float/bool/None/list/str)."""
    try:
        return yaml.safe_load(value)
    except Exception:
        return value


def _set_dotted(cfg: dict, dotted: str, value: Any) -> None:
    keys = dotted.split(".")
    node = cfg
    for k in keys[:-1]:
        nxt = node.get(k)
        if not isinstance(nxt, dict):
            nxt = {}
            node[k] = nxt
        node = nxt
    node[keys[-1]] = value


def load_config(path: str, overrides: Iterable[str] | None = None) -> dict:
    """Load a YAML config, resolving `base:` inheritance and applying `key=value` overrides."""
    path = os.path.abspath(path)
    with open(path) as f:
        cfg = yaml.safe_load(f) or {}
    base_ref = cfg.get("base")
    if base_ref:
        base_path = os.path.join(os.path.dirname(path), base_ref)
        cfg = _deep_merge(load_config(base_path), cfg)
    cfg.pop("base", None)
    for ov in overrides or []:
        if "=" not in ov:
            raise ValueError(f"override must be key=value, got {ov!r}")
        key, val = ov.split("=", 1)
        _set_dotted(cfg, key.strip(), _coerce(val.strip()))
    return cfg


def get(cfg: dict, dotted: str, default: Any = None) -> Any:
    """Read a nested value by dotted path, returning `default` if absent."""
    node = cfg
    for k in dotted.split("."):
        if not isinstance(node, dict) or k not in node:
            return default
        node = node[k]
    return node
