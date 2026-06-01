"""Phase 1 gate: slice extraction / filtering / windowing / resize / manifest split.

Runs fully on CPU with a tiny synthetic NIfTI (no download). Validates the exact logic
mirrored by notebooks/01_data_prep.ipynb before any Colab run.
"""
import numpy as np
import nibabel as nib

from src.data.preprocess import (
    PreprocessConfig, binarize_label, ct_window_normalize, resize2d,
    load_volume, process_volume, process_task,
)
from src.data.manifest import build_manifest, split_volumes, assign_splits


def _make_volume():
    """16x16x8 volume; FG blocks give kept slices {2, 3} (slice 4 is 0.39% < 1% -> dropped)."""
    rng = np.random.default_rng(0)
    vol = rng.uniform(-500, 500, size=(16, 16, 8)).astype(np.float32)
    lab = np.zeros((16, 16, 8), dtype=np.uint8)
    lab[6:10, 6:10, 2] = 1   # 16/256 = 6.25% FG          -> kept
    lab[6:10, 6:10, 3] = 2   # label 2 also FG (merge test) -> kept
    lab[0:1, 0:1, 4] = 1     # 1/256 = 0.39% < 1%          -> dropped
    return vol, lab


def _write_nifti(path, arr):
    nib.save(nib.Nifti1Image(arr.astype(np.float32), affine=np.eye(4)), str(path))


def test_ct_window_normalize_range():
    x = np.array([[-1000.0, -125.0, 75.0, 275.0, 1000.0]], dtype=np.float32)
    out = ct_window_normalize(x, (-125.0, 275.0))
    assert out.min() >= 0.0 and out.max() <= 1.0
    assert np.isclose(out[0, 0], 0.0)            # -1000 -> clip -125 -> 0
    assert np.isclose(out[0, 4], 1.0)            #  1000 -> clip  275 -> 1
    assert np.isclose(out[0, 2], 0.5, atol=1e-3)  #   75 -> (75+125)/400 = 0.5


def test_binarize_label_merges_fg_labels():
    lab = np.array([0, 1, 2, 3], dtype=np.float32)
    assert binarize_label(lab, [1, 2]).tolist() == [0, 1, 1, 0]


def test_resize_image_and_mask_binary():
    img = np.random.rand(16, 16).astype(np.float32)
    m = (np.random.rand(16, 16) > 0.5).astype(np.uint8)
    assert resize2d(img, 32, is_mask=False).shape == (32, 32)
    m_r = resize2d(m, 32, is_mask=True)
    assert m_r.shape == (32, 32)
    assert set(np.unique(m_r)).issubset({0, 1})


def test_process_volume_fg_filter():
    vol, lab = _make_volume()
    cfg = PreprocessConfig(fg_labels=[1, 2], min_fg_ratio=0.01, size=32)
    imgs, masks, idxs = process_volume(vol, lab, cfg)
    assert idxs == [2, 3]                                   # slice 4 dropped
    assert all(i.shape == (32, 32) for i in imgs)
    assert all(0.0 <= float(i.min()) and float(i.max()) <= 1.0 for i in imgs)
    assert all(set(np.unique(m)).issubset({0, 1}) and m.sum() > 0 for m in masks)


def test_load_volume_roundtrip(tmp_path):
    vol, _ = _make_volume()
    p = tmp_path / "img.nii.gz"
    _write_nifti(p, vol)
    data, _ = load_volume(str(p))
    assert data.shape == (16, 16, 8)


def test_process_task_and_manifest(tmp_path):
    cfg = PreprocessConfig(fg_labels=[1, 2], min_fg_ratio=0.01, size=32)
    files = []
    for k in range(3):
        vol, lab = _make_volume()
        ip, lp = tmp_path / f"img_{k}.nii.gz", tmp_path / f"lab_{k}.nii.gz"
        _write_nifti(ip, vol)
        _write_nifti(lp, lab)
        files.append((str(ip), str(lp), f"spleen_{k}"))

    out_npz = tmp_path / "spleen.npz"
    arrays, manifest = process_task(files, cfg, out_npz=str(out_npz))
    assert arrays["images"].shape == (6, 32, 32)            # 3 vols x 2 kept slices
    assert arrays["masks"].shape == (6, 32, 32)
    assert len(manifest) == 6 and out_npz.exists()

    loaded = np.load(str(out_npz))
    assert loaded["images"].shape == (6, 32, 32)
    assert set(loaded["vol_ids"].tolist()) == {"spleen_0", "spleen_1", "spleen_2"}

    df = build_manifest(manifest)
    assert list(df.columns) == ["global_index", "vol_id", "slice_idx"] and len(df) == 6


def test_split_volumes_disjoint_and_deterministic():
    vols = [f"v{i}" for i in range(10)]
    fracs = {"train": 0.6, "val": 0.2, "test": 0.2}
    split = split_volumes(vols, fracs, seed=42)
    groups = {s: {v for v, ss in split.items() if ss == s} for s in ("train", "val", "test")}
    assert groups["train"] & groups["val"] == set()
    assert groups["train"] & groups["test"] == set()
    assert groups["val"] & groups["test"] == set()
    assert groups["train"] | groups["val"] | groups["test"] == set(vols)
    assert split == split_volumes(vols, fracs, seed=42)       # deterministic


def test_assign_splits_keeps_volumes_intact():
    recs = [{"global_index": i, "vol_id": f"v{i % 3}", "slice_idx": i} for i in range(9)]
    df = build_manifest(recs)
    split = split_volumes(df["vol_id"], {"train": 0.34, "val": 0.33, "test": 0.33}, seed=1)
    df2 = assign_splits(df, split)
    for _, sub in df2.groupby("vol_id"):
        assert sub["split"].nunique() == 1                   # no volume spans two splits
