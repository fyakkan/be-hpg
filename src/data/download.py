"""Download + extract MSD tasks.

Primary source is the MONAI S3 mirror (VERIFIED LIVE 2026-05-31):
  https://msd-for-monai.s3-us-west-2.amazonaws.com/<TaskXX_Name>.tar
Downloads are resumable (HTTP Range) and persisted under <drive_root>/downloads so a Colab
disconnect never forces a re-download. A `manual` fallback reads a tar the user dropped by
hand (e.g. from medicaldecathlon.com).
"""
from __future__ import annotations

import os
import tarfile
import urllib.request

DEFAULT_S3_BASE = "https://msd-for-monai.s3-us-west-2.amazonaws.com"


def task_url(msd_name: str, s3_base: str = DEFAULT_S3_BASE) -> str:
    return f"{s3_base.rstrip('/')}/{msd_name}.tar"


def _remote_size(url: str) -> int | None:
    try:
        req = urllib.request.Request(url, method="HEAD")
        with urllib.request.urlopen(req) as r:
            cl = r.headers.get("Content-Length")
            return int(cl) if cl is not None else None
    except Exception:
        return None


def download_resumable(url: str, dest: str, chunk: int = 1 << 20, progress: bool = True) -> str:
    """Stream `url` to `dest`, resuming from a partial download via an HTTP Range request."""
    os.makedirs(os.path.dirname(dest) or ".", exist_ok=True)
    total = _remote_size(url)
    have = os.path.getsize(dest) if os.path.exists(dest) else 0
    if total is not None and have == total:
        return dest  # already complete

    req = urllib.request.Request(url)
    if have:
        req.add_header("Range", f"bytes={have}-")
    bar = None
    if progress and total:
        try:
            from tqdm.auto import tqdm
            bar = tqdm(total=total, initial=have, unit="B", unit_scale=True, desc=os.path.basename(dest))
        except ImportError:
            bar = None
    with urllib.request.urlopen(req) as r, open(dest, "ab" if have else "wb") as f:
        while True:
            buf = r.read(chunk)
            if not buf:
                break
            f.write(buf)
            if bar:
                bar.update(len(buf))
    if bar:
        bar.close()
    return dest


def extract_tar(tar_path: str, out_dir: str) -> str:
    """Extract a tar into out_dir (MSD tars contain a single TaskXX_Name/ directory)."""
    os.makedirs(out_dir, exist_ok=True)
    with tarfile.open(tar_path) as t:
        try:
            t.extractall(out_dir, filter="data")  # py>=3.12 safe extraction
        except TypeError:
            t.extractall(out_dir)
    return out_dir


def ensure_task(msd_name: str, drive_root: str, *, source: str = "monai_s3",
                s3_base: str = DEFAULT_S3_BASE, manual_fallback_dir: str | None = None,
                progress: bool = True) -> str:
    """Make sure <drive_root>/raw/<msd_name>/ exists; return that path."""
    downloads = os.path.join(drive_root, "downloads")
    raw = os.path.join(drive_root, "raw")
    extracted = os.path.join(raw, msd_name)
    if os.path.isdir(extracted) and os.listdir(extracted):
        return extracted

    if source == "manual":
        tar_path = os.path.join(manual_fallback_dir or downloads, f"{msd_name}.tar")
        if not os.path.exists(tar_path):
            raise FileNotFoundError(
                f"Manual mode: expected {msd_name}.tar at {tar_path}. Download it from "
                f"medicaldecathlon.com and place it there, then re-run."
            )
    else:
        tar_path = os.path.join(downloads, f"{msd_name}.tar")
        download_resumable(task_url(msd_name, s3_base), tar_path, progress=progress)

    extract_tar(tar_path, raw)
    return extracted
