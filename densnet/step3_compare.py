"""
STEP 3 — Systematic comparison table:
    OCR ground truth (from the image)  vs  auto.py (Chan-Vese)  vs  DenseNet

For every dual-panel image it prints d1/d2/d3 (mm) and Volume (ml) for
all three, with error-% of each method against the OCR'd truth, and the
Chan-Vese-vs-DenseNet speedup.

OCR NOTE (option B): the "ACTUAL" column is read automatically from the
burned-in measurement table via OCR. Any value OCR was unsure of is
flagged with '*' — eyeball those against the on-screen number, because
OCR of burned-in ultrasound text is not 100% reliable.

Usage:
    python step3_compare.py --images data/raw --model best_model.pth --diagonal_panel RIGHT
"""
import os
import sys
import time
import argparse
import cv2
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# ---- import the classical pipeline (auto.py preferred) ----
classical = None
for _m in ("auto", "bladder_volume_v4"):
    try:
        classical = __import__(_m); break
    except ImportError:
        continue
if classical is None:
    raise RuntimeError("Could not import auto.py / bladder_volume_v4.py — put this script beside it.")
# silence any GUI windows from the classical pipeline during batch runs
for _fn in ("imshow", "waitKey", "namedWindow", "resizeWindow", "destroyAllWindows", "setMouseCallback"):
    if hasattr(classical, "cv2"):
        setattr(classical.cv2, _fn, (lambda *a, **k: -1))

from model import build_model
from ocr_truth import ocr_printed_values

IMG_SIZE = 256
_MEAN = np.array([0.485, 0.456, 0.406]); _STD = np.array([0.229, 0.224, 0.225])

import io
import contextlib


@contextlib.contextmanager
def _quiet():
    """Suppress the classical pipeline's internal prints (the [split],
    [pixel spacing], [Chan-Vese], [orthogonal panel] chatter) so only
    the comparison table is printed. Warnings on stderr still show."""
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        yield


def densenet_mask(model, panel_bgr, device):
    """Full-resolution binary mask from the DenseNet for one panel."""
    h, w = panel_bgr.shape[:2]
    rgb = cv2.cvtColor(panel_bgr, cv2.COLOR_BGR2RGB)
    small = cv2.resize(rgb, (IMG_SIZE, IMG_SIZE)).astype(np.float32) / 255.0
    small = (small - _MEAN) / _STD
    inp = torch.from_numpy(small.transpose(2, 0, 1)).unsqueeze(0).float().to(device)
    with torch.no_grad():
        prob = torch.sigmoid(model(inp))[0, 0].cpu().numpy()
    m = (prob > 0.5).astype(np.uint8) * 255
    return cv2.resize(m, (w, h), interpolation=cv2.INTER_NEAREST)


def largest_contour(mask):
    c, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return max(c, key=cv2.contourArea) if c else None


def method_dims_mm(panels, was_split, segment_fn, mm_per_px, diagonal_panel="RIGHT"):
    """Segment both panels with one method, combine into d1/d2/d3 (mm) +
    volume (ml). Returns (d1, d2, d3, vol, seconds) or None on failure."""
    labels = ["LEFT", "RIGHT"] if was_split else ["SINGLE"]
    contours = {}
    t0 = time.time()
    for panel, label in zip(panels, labels):
        rmf = 0.15 if (was_split and label == "RIGHT") else 0.05
        cnt = segment_fn(panel, label=label, right_margin_frac=rmf)
        if cnt is not None:
            contours[label] = cnt
    elapsed = time.time() - t0

    ortho = "LEFT" if diagonal_panel == "RIGHT" else "RIGHT"
    if not was_split:
        if "SINGLE" not in contours:
            return None
        ortho = diagonal_panel = "SINGLE"
    if ortho not in contours or diagonal_panel not in contours:
        return None

    d1, d2, d3, _ = classical.extract_dimensions(
        contours[ortho], contours[diagonal_panel], mm_per_px)
    vol = 0.52 * (d1 / 10) * (d2 / 10) * (d3 / 10)
    return d1, d2, d3, vol, elapsed


def err_pct(val, truth):
    if truth is None or val is None or truth == 0:
        return None
    return abs(val - truth) / truth * 100.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", default="data/raw")
    ap.add_argument("--model", default="best_model.pth")
    ap.add_argument("--diagonal_panel", default="RIGHT")
    ap.add_argument("--no_dl", action="store_true",
                    help="skip DenseNet (only OCR-truth vs auto.py)")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = None
    if not args.no_dl:
        model = build_model().to(device)
        model.load_state_dict(torch.load(args.model, map_location=device))
        model.eval()

    def dl_segment(panel, label="", right_margin_frac=0.05):
        return largest_contour(densenet_mask(model, panel, device))

    exts = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")
    files = [f for f in sorted(os.listdir(args.images)) if f.lower().endswith(exts)]

    print("Values: d1/d2/d3 in mm, Vol in ml.  '*' = OCR unsure of that actual value (verify by eye).")
    print("err% is each method vs the OCR'd ACTUAL.\n")
    print(f"{'image':16s} | {'ACTUAL d1/d2/d3  Vol':28s} | "
          f"{'auto.py d1/d2/d3  Vol  (Vol err%)':40s} | "
          f"{'DenseNet d1/d2/d3  Vol  (Vol err%)':40s} | {'speed':>6s}")
    print("-" * 150)

    rows = []
    for fn in files:
        img = cv2.imread(os.path.join(args.images, fn))
        if img is None:
            continue
        with _quiet():
            panels, was_split = classical.split_panels(img)
            mmpp = classical.get_pixel_spacing_auto(img, panels[0].shape[0])
            t = ocr_printed_values(img)
            cv_res = method_dims_mm(panels, was_split, classical.segment_bladder_full, mmpp, args.diagonal_panel)
            dl_res = method_dims_mm(panels, was_split, dl_segment, mmpp, args.diagonal_panel) if model else None

        # ACTUAL column (mark OCR-unsure values with *)
        flagset = t.get('flag', '')
        def star(k):
            return '*' if k in flagset else ''
        act = (f"{fmtv(t['d1'])}{star('d1')}/{fmtv(t['d2'])}{star('d2')}/"
               f"{fmtv(t['d3'])}{star('d3')}  {fmtv(t['vol'])}{star('vol')}")

        def method_str(res):
            if res is None:
                return f"{'FAILED':40s}"
            d1, d2, d3, vol, _ = res
            e = err_pct(vol, t['vol'])
            estr = f"({e:.1f}%)" if e is not None else "(n/a)"
            return f"{d1:5.1f}/{d2:5.1f}/{d3:5.1f}  {vol:6.1f}  {estr:8s}"

        speed = ""
        if cv_res and dl_res and dl_res[4] > 0:
            speed = f"{cv_res[4]/dl_res[4]:5.1f}x"

        print(f"{fn:16s} | {act:28s} | {method_str(cv_res):40s} | "
              f"{method_str(dl_res):40s} | {speed:>6s}")
        rows.append((fn, t, cv_res, dl_res))

    # summary
    cv_errs = [err_pct(r[2][3], r[1]['vol']) for r in rows if r[2] and r[1]['vol']]
    dl_errs = [err_pct(r[3][3], r[1]['vol']) for r in rows if r[3] and r[1]['vol']]
    cv_errs = [e for e in cv_errs if e is not None]
    dl_errs = [e for e in dl_errs if e is not None]
    print("\n=== SUMMARY (volume error vs OCR truth) ===")
    if cv_errs:
        print(f"auto.py  : mean {np.mean(cv_errs):.1f}%  (n={len(cv_errs)})")
    if dl_errs:
        print(f"DenseNet : mean {np.mean(dl_errs):.1f}%  (n={len(dl_errs)})")
    speeds = [r[2][4]/r[3][4] for r in rows if r[2] and r[3] and r[3][4] > 0]
    if speeds:
        print(f"DenseNet speedup vs auto.py: mean {np.mean(speeds):.1f}x")
    print("\nReminder: any ACTUAL value with '*' was OCR-uncertain — check it "
          "against the number printed on that scan before trusting its err%.")


def fmtv(v):
    return f"{v:.1f}" if v is not None else "  ?  "


if __name__ == "__main__":
    main()