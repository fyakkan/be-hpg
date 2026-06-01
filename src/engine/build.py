"""Build a model from a (merged) config dict."""
from __future__ import annotations

from ..models.be_hpg import BEHPG
from ..models.meta_reweight import MetaReweight
from ..models.protonet import ProtoNet


def build_model(cfg: dict):
    m = cfg["model"]
    name = m["name"]
    common = dict(
        backbone=m["backbone"],
        embed_dim=cfg.get("embed_dim", 256),
        stride=m.get("backbone_feature_stride", 8),
        pretrained=m.get("pretrained", True),
    )
    if name == "protonet":
        return ProtoNet(**common, temperature=m.get("temperature", 20.0),
                        use_bg=m.get("use_bg_prototype", True))
    if name == "meta_reweight":
        return MetaReweight(**common, hidden=m.get("reweight_hidden", 256))
    if name == "be_hpg":
        ssp = m.get("ssp", {})
        return BEHPG(**common,
                     boundary_kernel=ssp.get("boundary_kernel", 3),
                     kmeans_clusters=ssp.get("kmeans_clusters", 5),
                     use_fg_interior=ssp.get("use_fg_interior_prototype", True),
                     use_bg=ssp.get("use_bg_prototype", True),
                     use_ssp=ssp.get("use_ssp", True),
                     seed=cfg.get("seed", 0),
                     temperature=m.get("temperature", 20.0))
    raise ValueError(f"unknown model name: {name!r}")
