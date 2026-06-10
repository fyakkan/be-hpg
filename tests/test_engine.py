"""Phase 2 gate: train loop runs, eval computes metrics, checkpoint round-trips."""
import numpy as np
import torch

from src.data.sampler import EpisodicSampler
from src.engine.checkpoint import load_checkpoint, save_checkpoint
from src.engine.eval import eval_episodes
from src.engine.train import train_episodes
from src.models.protonet import ProtoNet


def _sampler(H=64):
    imgs, masks, vols = [], [], []
    rng = np.random.default_rng(0)
    for v in range(3):
        for _ in range(4):
            imgs.append(rng.random((H, H)).astype(np.float32))
            m = np.zeros((H, H), np.uint8)
            m[10:30, 10:30] = 1
            masks.append(m)
            vols.append(f"v{v}")
    return EpisodicSampler(np.stack(imgs), np.stack(masks), np.array(vols),
                           k_shots=[1], query_size=2, seed=0)


def test_eval_return_samples_with_lcc():
    """Revision path: per-slice metric lists are returned for bootstrap CIs (with LCC)."""
    from src.utils.bootstrap import bootstrap_ci

    torch.manual_seed(0)
    model = ProtoNet(backbone="resnet50", embed_dim=32, stride=8, pretrained=False)
    sampler = _sampler()
    summ, raw = eval_episodes(model, sampler, episodes=3, k_shot=1, device="cpu",
                              postprocess="lcc", return_samples=True)
    assert summ["n"] == 6
    assert len(raw["dice"]) == 6 and len(raw["iou"]) == 6      # all slices
    assert len(raw["hd95"]) == summ["n_surface"]               # both-non-empty only
    m, lo, hi = bootstrap_ci(raw["dice"], n_boot=200, seed=0)
    assert lo <= m <= hi


def test_train_eval_checkpoint(tmp_path):
    torch.manual_seed(0)
    model = ProtoNet(backbone="resnet50", embed_dim=32, stride=8, pretrained=False)
    sampler = _sampler()

    hist, opt, sched = train_episodes(model, sampler, episodes=3, lr=1e-3, lam=0.3, device="cpu")
    assert len(hist) == 3 and all(np.isfinite(h["total"]) for h in hist)

    summ = eval_episodes(model, sampler, episodes=3, k_shot=1, device="cpu")
    assert 0.0 <= summ["dice"] <= 1.0 and summ["n"] == 6     # 3 episodes x 2 query slices

    ckpt = str(tmp_path / "m.pt")
    save_checkpoint(ckpt, model, opt, sched, episode=3)
    model2 = ProtoNet(backbone="resnet50", embed_dim=32, stride=8, pretrained=False)
    ep, _ = load_checkpoint(ckpt, model2)
    assert ep == 3

    raw = str(tmp_path / "raw.pt")             # old-style raw state_dict must still load
    torch.save(model.state_dict(), raw)
    ep_raw, _ = load_checkpoint(raw, model2)
    assert ep_raw == 0
