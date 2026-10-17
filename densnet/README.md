# DenseNet-121 Bladder Segmentation (Comparative Study Arm)

The deep-learning half of the "Chan-Vese vs DenseNet-121" comparison.
Trains a DenseNet-121 U-Net to segment the bladder, then compares it
against your classical pipeline (auto.py) on the same panels.

## Honest framing (read this)

You have ~18 dual-panel screenshots and NO hand-drawn masks. So this
pipeline:
1. Splits each dual-panel image into LEFT + RIGHT single panels
   (~36 training samples).
2. Uses your classical pipeline's output as the training label
   (pseudo-labels) — there's nothing else to learn from.
3. Fine-tunes an ImageNet-pretrained DenseNet-121.

Consequence: the DenseNet learns to IMITATE the classical pipeline, so
it cannot be MORE accurate than Chan-Vese on segmentation. Its genuine,
reportable win is SPEED (one fast forward pass vs. multi-second
Chan-Vese). Frame the study as "accuracy parity + large speedup", not
"DL beats classical on accuracy". If you later hand-label real masks,
retrain on those for a true accuracy contest.

## Setup (on your GPU machine)

Put these next to your pipeline file `auto.py`:
    step1_generate_masks.py  dataset.py  model.py  train.py  step3_compare.py

```bash
pip install -r requirements.txt
```

## Folder layout

```
<this folder>/
├── auto.py                  ← your classical pipeline (MUST be here)
├── step1_generate_masks.py
├── dataset.py  model.py  train.py  step3_compare.py
└── data/
    ├── raw/                 ← put your DUAL-PANEL images here (img01..img018)
    ├── images/              ← step 1 fills this (split single panels)
    └── masks/               ← step 1 fills this
```

Create `data/raw/` and copy img01..img018 (the dual-panel screenshots)
into it.

## Run order

```bash
# 1. Split dual panels + generate masks from the classical pipeline
python step1_generate_masks.py --images data/raw --out_images data/images --out_masks data/masks

# 2. Train the DenseNet-121 U-Net
python train.py --images data/images --masks data/masks --epochs 60

# 3. Compare DenseNet vs Chan-Vese (Dice, IoU, speed) on the split panels
python step3_compare.py --images data/images --model best_model.pth
```

## If auto.py's function names differ

step1 expects `split_panels(roi)` and `segment_bladder_full(panel, label=, right_margin_frac=)`.
If yours are named differently, edit the two marked lines near the top
of step1_generate_masks.py (SPLIT_FN / SEGMENT_FN).

## Outputs for your report

- `training_log.csv` — per-epoch Dice/IoU (plot as learning curve)
- `best_model.pth` — trained weights
- `comparison_overlays/*.png` — green = DenseNet, orange = Chan-Vese
- step3 prints mean Dice, IoU, and the SPEEDUP factor

## Small-data caveats to expect

- Overfitting: train loss keeps dropping while val Dice plateaus — normal
  at this size; report it.
- Tiny val set (a few panels) → noisy val Dice; report with that caveat.
- Pseudo-label ceiling: if Chan-Vese is wrong on an image, DenseNet
  learns that error too.
