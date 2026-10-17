"""
STEP 1 (dual-panel version) — Generate training masks from the
classical pipeline, automatically splitting each dual-panel screenshot
into its LEFT and RIGHT panels first.

Why this version:
Your source images (img01 ... img018) are dual-panel screenshots — two
bladder views side by side. The DenseNet needs SINGLE-panel inputs, so
this script:
  1. loads each dual-panel image,
  2. splits it into left + right panels (via the classical pipeline's
     split_panels),
  3. runs the classical segmentation on EACH panel,
  4. saves each panel image + its mask as a separate training sample.

This also roughly DOUBLES your effective dataset (~18 dual images ->
~36 single-panel training samples), which matters a lot at this size.

Your main pipeline file is `auto.py`. This script tries to import it as
`auto` first, then falls back to `bladder_volume_v4`. If your function
names differ from split_panels / segment_bladder_full, edit the two
marked lines below.

Usage:
    python step1_generate_masks.py --images data/raw --out_images data/images --out_masks data/masks
"""
import os
import sys
import argparse
import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# --- import your classical pipeline (auto.py preferred) ---
classical = None
for modname in ("auto", "bladder_volume_v4"):
    try:
        classical = __import__(modname)
        print(f"[import] using classical pipeline module: {modname}.py")
        break
    except ImportError:
        continue
if classical is None:
    raise RuntimeError("Could not import auto.py or bladder_volume_v4.py. "
                       "Put this script in the same folder as your pipeline.")

# silence any GUI windows during batch processing
for _fn in ("imshow", "waitKey", "namedWindow", "resizeWindow", "destroyAllWindows", "setMouseCallback"):
    if hasattr(classical, "cv2"):
        setattr(classical.cv2, _fn, (lambda *a, **k: -1))


# ---- EDIT THESE TWO LINES if your function names differ ----
SPLIT_FN = "split_panels"            # expects: panels, was_split = split_panels(roi)
SEGMENT_FN = "segment_bladder_full"  # expects: contour = segment_bladder_full(panel, label=, right_margin_frac=)
# ------------------------------------------------------------


def mask_from_contour(shape, contour):
    m = np.zeros(shape[:2], dtype=np.uint8)
    if contour is not None:
        cv2.drawContours(m, [contour], -1, 255, thickness=-1)
    return m


def generate(images_dir, out_images, out_masks, right_margin_frac_right=0.15):
    os.makedirs(out_images, exist_ok=True)
    os.makedirs(out_masks, exist_ok=True)

    split_panels = getattr(classical, SPLIT_FN)
    segment = getattr(classical, SEGMENT_FN)

    exts = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")
    files = [f for f in sorted(os.listdir(images_dir)) if f.lower().endswith(exts)]
    if not files:
        print(f"No images found in {images_dir}")
        return

    ok, fail = 0, 0
    for fn in files:
        img = cv2.imread(os.path.join(images_dir, fn))
        if img is None:
            print(f"  [skip] unreadable: {fn}")
            continue
        try:
            panels, was_split = split_panels(img)
        except Exception as e:
            print(f"  [skip] {fn}: split failed ({e})")
            continue

        labels = ["LEFT", "RIGHT"] if was_split and len(panels) == 2 else ["SINGLE"]
        for panel, label in zip(panels, labels):
            rmf = right_margin_frac_right if label == "RIGHT" else 0.05
            stem = f"{os.path.splitext(fn)[0]}_{label}"
            try:
                cnt = segment(panel, label=label, right_margin_frac=rmf)
            except Exception as e:
                print(f"  [fail] {stem}: {e}")
                fail += 1
                continue
            if cnt is None:
                print(f"  [fail] {stem}: no bladder found")
                fail += 1
                continue
            mask = mask_from_contour(panel.shape, cnt)
            cv2.imwrite(os.path.join(out_images, stem + ".png"), panel)
            cv2.imwrite(os.path.join(out_masks, stem + "_mask.png"), mask)
            ok += 1
            print(f"  [ok]   {stem}  (bladder area {int(mask.sum()/255)} px)")

    print(f"\nDone. {ok} panel samples generated, {fail} failed.")
    print(f"Training images -> {out_images}")
    print(f"Training masks  -> {out_masks}")
    if fail:
        print("Failed panels had no clean classical segmentation - they're "
              "simply skipped, which is fine.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", default="data/raw",
                    help="folder of DUAL-PANEL source screenshots (img01..img018)")
    ap.add_argument("--out_images", default="data/images",
                    help="where split single-panel images are written")
    ap.add_argument("--out_masks", default="data/masks",
                    help="where masks are written")
    ap.add_argument("--right_margin_frac_right", type=float, default=0.15)
    args = ap.parse_args()
    generate(args.images, args.out_images, args.out_masks, args.right_margin_frac_right)
