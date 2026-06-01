"""NIfTI -> 2D axial slice preprocessing for MSD CT tasks.

Per volume:
  1. load with nibabel, reorient to closest canonical (RAS) so axis 2 is axial
  2. binarize the label (foreground = label in `fg_labels`; e.g. pancreas + tumor)
  3. keep an axial slice iff its foreground ratio >= `min_fg_ratio` (measured at native res)
  4. CT HU window: clip to [lo, hi], then min-max normalize to [0, 1]
  5. resize image (bilinear) and mask (nearest) to (size, size)

This tested logic is mirrored by notebooks/01_data_prep.ipynb (data prep runs on Colab).
Heavy deps (nibabel, skimage) are imported lazily so the module stays importable.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Sequence

import numpy as np


# --------------------------------------------------------------------------- I/O
def load_volume(nifti_path: str):
    """Return (data: float32 [H, W, Z], affine), reoriented to closest canonical."""
    import nibabel as nib

    img = nib.as_closest_canonical(nib.load(nifti_path))
    data = np.asanyarray(img.dataobj, dtype=np.float32)
    return data, img.affine


# ------------------------------------------------------------------- transforms
def binarize_label(label: np.ndarray, fg_labels: Sequence[int]) -> np.ndarray:
    """1 where the (rounded) label is one of `fg_labels`, else 0."""
    return np.isin(np.rint(label).astype(np.int64), list(fg_labels)).astype(np.uint8)


def ct_window_normalize(img2d: np.ndarray, hu_window: Sequence[float]) -> np.ndarray:
    """Clip to the HU window then min-max normalize to [0, 1]."""
    lo, hi = float(hu_window[0]), float(hu_window[1])
    x = np.clip(img2d.astype(np.float32), lo, hi)
    return ((x - lo) / max(hi - lo, 1e-6)).astype(np.float32)


def resize2d(arr: np.ndarray, size: int, *, is_mask: bool) -> np.ndarray:
    """Resize a 2D array to (size, size). Nearest+binary for masks, bilinear for images."""
    from skimage.transform import resize as sk_resize

    downsampling = arr.shape[0] > size or arr.shape[1] > size
    out = sk_resize(
        arr.astype(np.float32),
        (size, size),
        order=0 if is_mask else 1,
        mode="edge",
        anti_aliasing=(not is_mask and downsampling),
        preserve_range=True,
    )
    return (out >= 0.5).astype(np.uint8) if is_mask else out.astype(np.float32)


def fg_ratio(mask2d: np.ndarray) -> float:
    return float(mask2d.mean()) if mask2d.size else 0.0


def iter_axial(volume: np.ndarray, mask: np.ndarray, axis: int = 2):
    """Yield (img2d, mask2d, slice_index) along the axial axis."""
    for i in range(volume.shape[axis]):
        yield np.take(volume, i, axis=axis), np.take(mask, i, axis=axis), i


@dataclass
class PreprocessConfig:
    fg_labels: Sequence[int] = (1,)
    min_fg_ratio: float = 0.01
    hu_window: Sequence[float] = (-125.0, 275.0)
    size: int = 224
    axis: int = 2


# ----------------------------------------------------------------- volume/task
def process_volume(volume: np.ndarray, label: np.ndarray, cfg: PreprocessConfig):
    """Return (images[float16], masks[uint8], slice_indices) for KEPT slices of one volume."""
    mask_vol = binarize_label(label, cfg.fg_labels)
    images, masks, idxs = [], [], []
    for img2d, m2d, i in iter_axial(volume, mask_vol, cfg.axis):
        if fg_ratio(m2d) < cfg.min_fg_ratio:
            continue
        images.append(resize2d(ct_window_normalize(img2d, cfg.hu_window), cfg.size, is_mask=False).astype(np.float16))
        masks.append(resize2d(m2d, cfg.size, is_mask=True))
        idxs.append(i)
    return images, masks, idxs


def process_task(volume_files, cfg: PreprocessConfig, out_npz: str | None = None,
                 limit: int | None = None, progress: bool = False):
    """Preprocess a list of (image_path, label_path, vol_id) into stacked arrays + a manifest.

    Returns (arrays_dict, manifest_records). If `out_npz` is set, saves a compressed npz with
    keys: images[float16 N,H,W], masks[uint8 N,H,W], vol_ids[str N], slice_idx[int32 N].
    """
    pairs = volume_files[:limit] if limit else list(volume_files)
    if progress:
        try:
            from tqdm.auto import tqdm
            pairs = tqdm(pairs, desc="volumes")
        except ImportError:
            pass

    imgs_all, masks_all, vol_ids, slice_idxs, manifest = [], [], [], [], []
    for image_path, label_path, vol_id in pairs:
        vol, _ = load_volume(image_path)
        lab, _ = load_volume(label_path)
        imgs, masks, idxs = process_volume(vol, lab, cfg)
        for img, m, si in zip(imgs, masks, idxs):
            manifest.append({"global_index": len(imgs_all), "vol_id": str(vol_id), "slice_idx": int(si)})
            imgs_all.append(img)
            masks_all.append(m)
            vol_ids.append(str(vol_id))
            slice_idxs.append(int(si))

    images = np.stack(imgs_all).astype(np.float16) if imgs_all else np.zeros((0, cfg.size, cfg.size), np.float16)
    masks = np.stack(masks_all).astype(np.uint8) if masks_all else np.zeros((0, cfg.size, cfg.size), np.uint8)
    arrays = {
        "images": images,
        "masks": masks,
        "vol_ids": np.asarray(vol_ids),
        "slice_idx": np.asarray(slice_idxs, dtype=np.int32),
    }
    if out_npz:
        os.makedirs(os.path.dirname(out_npz) or ".", exist_ok=True)
        np.savez_compressed(out_npz, **arrays)
    return arrays, manifest
