# BE-HPG: Boundary-Enhanced Hard Prototype Generation

**Few-Shot Medical Image Segmentation of Rare Organ Malformations** — a controlled comparison
study of three few-shot 2D segmentation methods on the
[Medical Segmentation Decathlon](http://medicaldecathlon.com/) (MSD) Spleen task, trained and
evaluated on identical episodes.

1. **Prototypical Network** — CNN baseline (ResNet-50, masked global average pooling).
2. **Meta-RCNN-style** — support-conditioned channel-wise feature reweighting + conv head
   (*an adaptation of Meta R-CNN, which is a detector — disclosed in the paper*).
3. **BE-HPG (proposed)** — MobileViT-S backbone, Support Self-Prediction (SSP) boundary-band
   **hard prototypes** via mini-batch k-means, global FG/BG prototypes, cosine matching, a 1×1
   conv head, and a boundary loss (`L = L_BCE + λ·L_boundary`).

The full write-up is in [`paper/main.tex`](paper/main.tex) (IEEEtran conference, 6 pages).

## Headline results (MSD Spleen, 150 test episodes, with largest-connected-component post-processing)

| Method | Dice (1/5-shot) | IoU (1/5) | HD95↓ (1/5) | ASD↓ (1/5) |
|---|---|---|---|---|
| Prototypical Network | 0.896 / 0.887 | 0.815 / 0.800 | 3.8 / 4.1 | 1.5 / 1.7 |
| Meta-RCNN-style | 0.906 / 0.898 | 0.830 / 0.819 | 3.5 / 4.0 | 1.4 / 1.4 |
| **BE-HPG (ours)** | **0.930 / 0.926** | **0.871 / 0.867** | **2.9 / 3.4** | **1.0 / 1.3** |

BE-HPG is best on every metric, including boundary accuracy; the 1-shot Dice gap is supported by
non-overlapping 95% bootstrap confidence intervals. A focused SSP-vs-global ablation is an **honest
negative result**: on this large, well-bounded organ the hard prototypes are inert (Dice 0.930 vs
0.929), so the win is owed to the ViT backbone and boundary-aware training, not the SSP prototypes —
the boundary machinery is expected to matter on the thin/boundary-critical targets this Spleen
proxy cannot test. (Single training seed, 1500 episodes — a compute-limited *proxy* study on a
healthy organ; see the paper's Limitations.)

## Repository layout
```
configs/     YAML configs (single source of truth): base, data, per-model, ablation
src/         data · models · losses · metrics · engine · utils
tests/       CPU smoke tests (pytest) — the bug-prevention gate before any GPU run
notebooks/   Colab: 01_data_prep · 02_train_eval · 03_ablations · 04_eval_postprocess · 05_revision
results/     metrics JSON + qualitative figures from the experiments
paper/       IEEEtran paper (main.tex, refs.bib, tab_*.tex, figures/, IEEEtran.cls)
```

## Reproduce
All GPU work runs on **Google Colab (free T4)**; nothing here needs a local GPU. Every model
passes CPU smoke tests and a 1–2 min dry run before any real run, and all heavy artifacts persist
to Google Drive (`/content/drive/MyDrive/be-hpg/`).

1. **`notebooks/01_data_prep.ipynb`** — download MSD Spleen, slice (≥1% foreground), HU-window
   `[-125,275]`→`[0,1]`, resize 224, patient-disjoint splits → `.npz` + manifest on Drive.
2. **`notebooks/02_train_eval.ipynb`** — train + evaluate all three models (1-/5-shot).
3. **`notebooks/03_ablations.ipynb`** — boundary-loss weight λ ∈ {0, 0.1, 0.3, 0.5}.
4. **`notebooks/04_eval_postprocess.ipynb`** — re-evaluate with largest-connected-component.
5. **`notebooks/05_revision.ipynb`** — 95% bootstrap confidence intervals, the SSP-vs-global
   ablation (`configs/be_hpg_nossp.yaml`), and the pre-LCC off-target failure figure.

Each notebook uses a small `be-hpg-src.zip` of `src/` + `configs/` (uploaded once, cached on Drive).

### Local CPU tests
```bash
python3 -m venv .venv && source .venv/bin/activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
pytest                                   # 37 CPU smoke tests
python -m src.utils.aggregate            # regenerate paper tables from results/
python -m src.utils.figures              # regenerate the pipeline figure
```

## Build the paper
Open `paper/main.tex` in Overleaf (IEEEtran is built in), or locally:
```bash
cd paper && pdflatex main && bibtex main && pdflatex main && pdflatex main
```

## Scope (honest disclosure)
Compute-limited configuration for a tight deadline on free Colab: **Spleen** task, single seed,
1500 training / 150 fixed test episodes, 1- and 5-shot, one boundary-loss ablation. All numbers
come from runs actually executed; exact counts and cuts vs. the original proposal are reported in
the paper.

## Citation
```bibtex
@misc{yakkan2026behpg,
  author = {Furkan Yakkan},
  title  = {Boundary-Enhanced Hard Prototype Generation for Few-Shot Medical Image Segmentation
            of Rare Organ Malformations},
  year   = {2026},
  note   = {Computer Vision term project, Abdullah G\"ul University}
}
```
