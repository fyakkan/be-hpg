"""Losses (Phase 2).

Planned modules:
  bce.py       pixel binary cross-entropy on the query mask
  boundary.py  Sobel-derived GT boundary map + weighted BCE on boundary pixels;
               total = L_BCE + lambda * L_boundary
"""
