"""Data pipeline (Phase 1).

Planned modules:
  download.py    verify + fetch MSD tasks (MONAI S3 mirror, manual Drive fallback)
  preprocess.py  NIfTI -> axial slices, HU window -> [0,1], resize 224, >=1% FG filter -> .npz
  manifest.py    build/read slice<->volume manifest CSV; patient-disjoint volume splits
  sampler.py     EpisodicSampler: K-shot support + query, support/query volume-disjoint
"""
