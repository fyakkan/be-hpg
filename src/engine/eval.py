"""Episodic evaluation over a FIXED test-episode set; writes a metrics JSON.

`eval_episodes` is exercised by the CPU smoke tests; `main` evaluates a trained checkpoint at
1-shot and 5-shot on the test split and saves results for the paper.
"""
from __future__ import annotations

import argparse
import json
import os

import torch

from ..metrics.surface import MetricAccumulator
from .train import _move


@torch.no_grad()
def eval_episodes(model, sampler, *, episodes, k_shot=1, device="cpu", threshold=0.5,
                  postprocess=None, return_samples=False):
    """Run `episodes` eval episodes at a fixed K; return the MetricAccumulator summary.

    postprocess="lcc" keeps only the largest connected component of each prediction
    (applied identically to all models). With return_samples=True the per-slice metric
    lists are returned alongside the summary as (summary, raw) for bootstrap CIs.
    """
    model.to(device).eval()
    acc = MetricAccumulator()
    for _ in range(episodes):
        sup_img, sup_mask, qry_img, qry_mask = _move(sampler.sample(k=k_shot), device)
        prob = torch.sigmoid(model(sup_img, sup_mask, qry_img))
        for q in range(prob.shape[0]):
            pred = (prob[q, 0] > threshold).float()
            if postprocess == "lcc":
                from ..metrics.postprocess import largest_connected_component
                pred = torch.from_numpy(largest_connected_component(pred.cpu().numpy())).to(pred)
            acc.update(pred, qry_mask[q, 0])
    if return_samples:
        return acc.summary(), acc.raw()
    return acc.summary()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--episodes", type=int, default=None)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--set", nargs="*", default=[])
    args = ap.parse_args()

    import pandas as pd

    from ..data.sampler import sampler_from_npz
    from ..engine.build import build_model
    from ..engine.checkpoint import load_checkpoint
    from ..utils.config import load_config
    from ..utils.seed import set_seed

    cfg = load_config(args.config, args.set)
    set_seed(cfg["eval"].get("episode_seed", 1234))     # fixes the test-episode set
    task = cfg["data"]["task"]
    npz_dir = cfg["data"]["npz_dir"]
    suffix = "_smoke" if args.smoke else ""
    npz = os.path.join(npz_dir, f"{task}{suffix}.npz")
    manifest = os.path.join(npz_dir, f"manifest{suffix}.csv")
    mdf = pd.read_csv(manifest) if os.path.exists(manifest) else None
    split = None if args.smoke else "test"

    model = build_model(cfg)
    ckpt = os.path.join(cfg["train"]["ckpt_dir"], f"{cfg['model']['name']}_{task}{suffix}.pt")
    if os.path.exists(ckpt):
        load_checkpoint(ckpt, model, map_location=args.device)
        print("loaded checkpoint:", ckpt)
    else:
        print("WARNING: no checkpoint found; evaluating an untrained model:", ckpt)

    episodes = args.episodes or (5 if args.smoke else cfg["eval"]["test_episodes"])
    results = {"model": cfg["model"]["name"], "task": task, "episodes": episodes, "by_shot": {}}
    for k in cfg["eval"]["k_shots"]:
        sampler = sampler_from_npz(npz, mdf, split, k_shots=[k],
                                   query_size=cfg["episode"]["query_size"],
                                   seed=cfg["eval"].get("episode_seed", 1234))
        summ = eval_episodes(model, sampler, episodes=episodes, k_shot=k, device=args.device,
                             threshold=cfg["eval"].get("threshold", 0.5))
        results["by_shot"][f"{k}shot"] = summ
        print(f"{k}-shot:", summ)

    out_dir = cfg["eval"]["results_dir"]
    os.makedirs(out_dir, exist_ok=True)
    out = os.path.join(out_dir, f"{cfg['model']['name']}_{task}{suffix}.json")
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print("saved results:", out)


if __name__ == "__main__":
    main()
