"""Episodic training loop (AdamW + cosine LR, optional AMP, grad clip) + a --smoke CLI.

The reusable `train_episodes` is exercised by the CPU smoke tests; `main` wires it to a
preprocessed .npz on Drive and resumable checkpointing for Colab.
"""
from __future__ import annotations

import argparse
import os

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from ..losses.boundary import total_loss
from .checkpoint import load_checkpoint, save_checkpoint


def _move(batch, device):
    return (batch["sup_img"].to(device), batch["sup_mask"].to(device),
            batch["qry_img"].to(device), batch["qry_mask"].to(device))


def train_episodes(model, sampler, *, episodes, lr=1e-4, weight_decay=0.01, lam=0.0,
                   device="cpu", grad_clip=1.0, amp=False, start_episode=0,
                   optimizer=None, scheduler=None, ckpt_path=None, ckpt_every=0, log_every=0,
                   balanced=True, max_pw=20.0):
    """Train for `episodes` episodes. Returns (history, optimizer, scheduler)."""
    model.to(device).train()
    if optimizer is None:
        optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    if scheduler is None:
        scheduler = CosineAnnealingLR(optimizer, T_max=max(episodes, 1))
    use_amp = bool(amp) and str(device).startswith("cuda") and torch.cuda.is_available()
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp) if use_amp else None

    history = []
    for ep in range(start_episode, episodes):
        sup_img, sup_mask, qry_img, qry_mask = _move(sampler.sample(), device)
        optimizer.zero_grad(set_to_none=True)
        if use_amp:
            with torch.autocast(device_type="cuda"):
                logits = model(sup_img, sup_mask, qry_img)
                loss, parts = total_loss(logits, qry_mask, lam, balanced=balanced, max_pw=max_pw)
            scaler.scale(loss).backward()
            if grad_clip:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(sup_img, sup_mask, qry_img)
            loss, parts = total_loss(logits, qry_mask, lam)
            loss.backward()
            if grad_clip:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
        scheduler.step()
        history.append({k: float(v) for k, v in parts.items()})
        if log_every and (ep + 1) % log_every == 0:
            print(f"[{ep + 1}/{episodes}] total={parts['total']:.4f} bce={parts['bce']:.4f}")
        if ckpt_path and ckpt_every and (ep + 1) % ckpt_every == 0:
            save_checkpoint(ckpt_path, model, optimizer, scheduler, ep + 1)
    if ckpt_path:
        save_checkpoint(ckpt_path, model, optimizer, scheduler, episodes)
    return history, optimizer, scheduler


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--smoke", action="store_true", help="5 episodes on the smoke .npz")
    ap.add_argument("--episodes", type=int, default=None)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--set", nargs="*", default=[], help="config overrides key=value")
    args = ap.parse_args()

    import pandas as pd

    from ..data.sampler import sampler_from_npz
    from ..utils.config import load_config
    from ..utils.seed import set_seed
    from .build import build_model

    cfg = load_config(args.config, args.set)
    set_seed(cfg.get("seed", 42))
    task = cfg["data"]["task"]
    npz_dir = cfg["data"]["npz_dir"]
    suffix = "_smoke" if args.smoke else ""
    npz = os.path.join(npz_dir, f"{task}{suffix}.npz")
    manifest = os.path.join(npz_dir, f"manifest{suffix}.csv")

    mdf = pd.read_csv(manifest) if os.path.exists(manifest) else None
    split = None if args.smoke else "train"     # smoke uses all volumes (only 2 exist)
    sampler = sampler_from_npz(npz, mdf, split, k_shots=cfg["episode"]["k_shot_train"],
                               query_size=cfg["episode"]["query_size"], seed=cfg.get("seed", 42))

    model = build_model(cfg)
    episodes = args.episodes or (5 if args.smoke else cfg["train"]["episodes"])
    ckpt = os.path.join(cfg["train"]["ckpt_dir"], f"{cfg['model']['name']}_{task}{suffix}.pt")
    start = 0
    if cfg["train"].get("resume") and os.path.exists(ckpt) and not args.smoke:
        start, _ = load_checkpoint(ckpt, model, map_location=args.device)
        print(f"resumed from episode {start}")

    print(f"training {cfg['model']['name']} on {task}{suffix}: {episodes} episodes, device={args.device}")
    train_episodes(model, sampler, episodes=episodes, lr=cfg["train"]["lr"],
                   weight_decay=cfg["train"]["weight_decay"], lam=cfg["loss"]["boundary_lambda"],
                   device=args.device, grad_clip=cfg["train"].get("grad_clip", 1.0),
                   amp=cfg.get("amp", True), start_episode=start, ckpt_path=ckpt,
                   ckpt_every=cfg["train"].get("ckpt_every", 0), log_every=max(episodes // 20, 1),
                   balanced=cfg["loss"].get("balanced_bce", True),
                   max_pw=cfg["loss"].get("bce_max_pos_weight", 20.0))
    print("done; checkpoint:", ckpt)


if __name__ == "__main__":
    main()
