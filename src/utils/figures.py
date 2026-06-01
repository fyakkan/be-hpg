"""Generate the BE-HPG pipeline graphical abstract -> paper/figures/be_hpg_pipeline.{pdf,png}.

Run from the repo root:  python -m src.utils.figures
"""
from __future__ import annotations

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

OUT = "paper/figures"

_COLORS = {"in": "#dbeafe", "enc": "#fde68a", "ssp": "#bbf7d0", "proto": "#bbf7d0",
           "match": "#e9d5ff", "head": "#fbcfe8", "out": "#fecaca", "loss": "#f1f5f9"}


def _box(ax, x, y, w, h, text, color, fs=8.5, bold=False):
    ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h,
                                boxstyle="round,pad=0.012,rounding_size=0.02",
                                linewidth=1.1, edgecolor="#334155", facecolor=color))
    ax.text(x, y, text, ha="center", va="center", fontsize=fs,
            fontweight="bold" if bold else "normal", color="#0f172a")


def _arrow(ax, x1, y1, x2, y2):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="-|>", color="#334155", lw=1.3))


def build_pipeline(path_stem=f"{OUT}/be_hpg_pipeline"):
    os.makedirs(OUT, exist_ok=True)
    fig, ax = plt.subplots(figsize=(5.4, 7.6))
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")

    _box(ax, 0.27, 0.95, 0.40, 0.066, "Support slice + mask", _COLORS["in"])
    _box(ax, 0.73, 0.95, 0.40, 0.066, "Query slice", _COLORS["in"])
    _box(ax, 0.50, 0.82, 0.74, 0.085,
         "Shared lightweight ViT backbone\n(MobileViT-S / ViT-Tiny) → dense tokens",
         _COLORS["enc"], bold=True)
    _box(ax, 0.27, 0.635, 0.46, 0.135,
         "SSP module\nboundary band = dilation−erosion\n→ mini-batch $k$-means\n"
         "→ hard boundary prototypes", _COLORS["ssp"])
    _box(ax, 0.73, 0.655, 0.42, 0.095, "Global FG / BG\nprototypes\n(masked pooling)", _COLORS["proto"])
    _box(ax, 0.50, 0.45, 0.78, 0.085,
         "Cosine-similarity matching\nchannels: [global-FG, max-hard, BG]", _COLORS["match"])
    _box(ax, 0.50, 0.305, 0.50, 0.062, "$1{\\times}1$ conv + sigmoid", _COLORS["head"])
    _box(ax, 0.50, 0.175, 0.46, 0.062, "Predicted query mask", _COLORS["out"], bold=True)
    _box(ax, 0.50, 0.055, 0.86, 0.062,
         "Objective: $\\mathcal{L}=\\mathcal{L}_{\\mathrm{BCE}}+\\lambda\\,"
         "\\mathcal{L}_{\\mathrm{boundary}}$  (Sobel GT boundary)", _COLORS["loss"], fs=8)

    _arrow(ax, 0.27, 0.917, 0.40, 0.863)
    _arrow(ax, 0.73, 0.917, 0.60, 0.863)
    _arrow(ax, 0.40, 0.777, 0.30, 0.703)        # backbone -> SSP (support path)
    _arrow(ax, 0.55, 0.777, 0.71, 0.703)        # backbone -> global prototypes
    _arrow(ax, 0.73, 0.777, 0.62, 0.493)        # query features -> matching
    _arrow(ax, 0.29, 0.567, 0.42, 0.493)        # hard prototypes -> matching
    _arrow(ax, 0.71, 0.607, 0.58, 0.493)        # global/bg prototypes -> matching
    _arrow(ax, 0.50, 0.407, 0.50, 0.337)
    _arrow(ax, 0.50, 0.273, 0.50, 0.207)
    _arrow(ax, 0.50, 0.143, 0.50, 0.087)

    plt.tight_layout(pad=0.2)
    fig.savefig(path_stem + ".pdf", bbox_inches="tight")
    fig.savefig(path_stem + ".png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("wrote", path_stem + ".pdf / .png")


if __name__ == "__main__":
    build_pipeline()
