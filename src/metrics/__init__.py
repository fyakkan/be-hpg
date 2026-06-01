"""Metrics (Phase 2).

Planned modules:
  dice.py     Dice similarity coefficient (primary)
  iou.py      Intersection-over-Union (Jaccard)
  surface.py  HD95 + ASD via medpy.metric.binary, averaged only over slices where BOTH
              pred and GT are non-empty; reports the excluded fraction
"""
