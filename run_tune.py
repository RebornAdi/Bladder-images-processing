#!/usr/bin/env python3
# =====================================================================
#  run_tune.py  —  auto-tune the classical segmentation parameters
#
#  Sweeps the two parameters that control the main failure modes and
#  reports which setting gives the lowest volume error across your
#  dataset. Guaranteed not to be worse than the current defaults,
#  because the current defaults are one point in the grid.
#
#     ellipse_scale      -> how far the bounded ellipse may extend past
#                           the detected boundary. Too small clips the
#                           bladder (under-measures d1/d2); too large
#                           lets it leak (over-measures).
#     boundary_dilate_px -> slack when intersecting ellipse with Stage-1
#                           boundary.
#
#  USAGE:
#     python run_tune.py <images_dir> <ground_truth.csv> [module] [--limit N] [--diag auto|RIGHT|LEFT]
#
#  EXAMPLES:
#     python run_tune.py ./cropped image_measurements_77.csv --limit 25   # quick sweep on 25 imgs
#     python run_tune.py ./cropped image_measurements_77.csv              # full sweep (slower)
#
#  TIP: start with --limit 20-30 for a fast read, then run the full set
#       on the best 1-2 settings. Chan-Vese is the slow part.
#
#  After it prints the winner, set those values as the defaults in
#  segment_bladder_full(...) in your pipeline.
# =====================================================================
import sys, os, io, csv, contextlib, importlib, itertools
import cv2, numpy as np

for _n in ("imshow", "namedWindow", "resizeWindow", "destroyWindow",
           "destroyAllWindows", "setMouseCallback"):
    setattr(cv2, _n, (lambda *a, **k: None))
cv2.waitKey = lambda *a, **k: 1
cv2.imwrite = lambda *a, **k: True

# ----------------------- the grid to search -------------------------
# edit these lists to widen / narrow the search
ELLIPSE_SCALES     = [1.05, 1.20, 1.35, 1.50]
BOUNDARY_DILATE_PX = [4, 10]
# --------------------------------------------------------------------


def parse_args():
    args = [a for a in sys.argv[1:]]
    diag, limit = "auto", None
    if "--diag" in args:
        i = args.index("--diag"); diag = args[i + 1]; del args[i:i + 2]
    if "--limit" in args:
        i = args.index("--limit"); limit = int(args[i + 1]); del args[i:i + 2]
    images_dir = args[0] if len(args) > 0 else "data/images"
    csv_path   = args[1] if len(args) > 1 else "data/image_measurements_77.csv"
    module     = args[2] if len(args) > 2 else "auto"
    return images_dir, csv_path, module, diag, limit


def green_count(img):
    # look only at the top 72% of the panel, so the GREEN border of the
    # bottom-left measurement table can't be mistaken for the d3 caliper
    h = img.shape[0]
    img = img[:int(h * 0.72)]
    b, g, r = img[:, :, 0].astype(int), img[:, :, 1].astype(int), img[:, :, 2].astype(int)
    return int(((g > 110) & (g - r > 45) & (g - b > 45)).sum())


def pick_diag(A, roi, mode):
    if mode in ("RIGHT", "LEFT"):
        return mode
    try:
        panels, was = A.split_panels(roi)
        if was and len(panels) >= 2:
            return "LEFT" if green_count(panels[0]) > green_count(panels[1]) else "RIGHT"
    except Exception:
        pass
    return "RIGHT"


def find_image(images_dir, name):
    for ext in (".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"):
        p = os.path.join(images_dir, name + ext)
        if os.path.exists(p):
            return p
    return None


def main():
    images_dir, csv_path, module, diag_mode, limit = parse_args()
    A = importlib.import_module(module)
    orig_seg = A.segment_bladder_full      # keep a handle to the real function

    # load GT and resolve image paths up front
    items = []
    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f):
            name = row["names"].strip()
            p = find_image(images_dir, name)
            if p is not None:
                items.append((name, p, float(row["vol"])))
    if limit:
        items = items[:limit]
    print(f"Tuning on {len(items)} images; grid = {len(ELLIPSE_SCALES)}x{len(BOUNDARY_DILATE_PX)} "
          f"= {len(ELLIPSE_SCALES)*len(BOUNDARY_DILATE_PX)} settings.\n")

    def run_once(es, bd):
        # wrap segment_bladder_full to inject the swept parameters
        A.segment_bladder_full = (lambda panel, label="", **kw:
                                  orig_seg(panel, label, ellipse_scale=es,
                                           boundary_dilate_px=bd, **kw))
        errs = []
        for name, path, gv in items:
            roi = cv2.imread(path)
            if roi is None:
                continue
            diag = pick_diag(A, roi, diag_mode)
            try:
                with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                    vol, _ = A.process_image(roi_path=path, diagonal_panel=diag)
                errs.append(abs(vol - gv) / gv * 100.0)
            except Exception:
                errs.append(float("nan"))
        A.segment_bladder_full = orig_seg
        e = np.array(errs, float)
        ok = e[~np.isnan(e)]
        return dict(es=es, bd=bd, n_ok=len(ok), n_fail=int(np.isnan(e).sum()),
                    mean=float(ok.mean()) if len(ok) else 1e9,
                    median=float(np.median(ok)) if len(ok) else 1e9,
                    under20=int((ok < 20).sum()))

    table = []
    for es, bd in itertools.product(ELLIPSE_SCALES, BOUNDARY_DILATE_PX):
        r = run_once(es, bd)
        table.append(r)
        print(f"  ellipse_scale={es:4.2f}  dilate={bd:2d}  ->  "
              f"median={r['median']:6.1f}%  mean={r['mean']:6.1f}%  "
              f"<20%: {r['under20']}/{r['n_ok']}  (fail {r['n_fail']})")

    table.sort(key=lambda r: (r["median"], r["mean"]))
    best = table[0]
    print("\n" + "=" * 60)
    print("BEST SETTING (by median volume error):")
    print(f"   ellipse_scale = {best['es']}")
    print(f"   boundary_dilate_px = {best['bd']}")
    print(f"   -> median {best['median']:.1f}%  mean {best['mean']:.1f}%  "
          f"<20%: {best['under20']}/{best['n_ok']}")
    print("\nSet these as the defaults in segment_bladder_full(...).")
    print("=" * 60)


if __name__ == "__main__":
    main()
