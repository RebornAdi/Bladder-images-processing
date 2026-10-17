#!/usr/bin/env python3
# =====================================================================
#  run_eval.py  —  batch evaluation harness
#
#  Runs your pipeline on every image and compares the computed volume
#  (and d1/d2/d3) against the ground-truth CSV, then prints a summary
#  and writes a per-image results CSV.
#
#  USAGE:
#     python run_eval.py <images_dir> <ground_truth.csv> [module] [--diag auto|RIGHT|LEFT]
#
#  EXAMPLES:
#     python run_eval.py ./cropped image_measurements_77.csv
#     python run_eval.py ./cropped image_measurements_77.csv auto_fixed --diag auto
#
#  NOTES:
#   * Put auto_fixed.py (your fixed pipeline) in the same folder, or pass
#     its module name as the 3rd argument (default: auto_fixed).
#   * Image files must be named to match the CSV 'names' column, e.g.
#     image_001.jpg / image_001.png  for the row "image_001".
#   * --diag auto  (default) picks the diagonal panel per image by which
#     side carries the green d3 caliper — correct for THIS annotated set.
#     For real (un-annotated) images pass --diag RIGHT (or LEFT).
#   * The CSV ground truth is used ONLY for scoring, never fed into the
#     measurement; caliper overlays are still erased by remove_overlays().
# =====================================================================
import sys, os, io, csv, contextlib, importlib
import cv2, numpy as np

# ---- make the pipeline run headless (no windows, no 1-second waits) ----
for _n in ("imshow", "namedWindow", "resizeWindow", "destroyWindow",
           "destroyAllWindows", "setMouseCallback"):
    setattr(cv2, _n, (lambda *a, **k: None))
cv2.waitKey = lambda *a, **k: 1
cv2.imwrite = lambda *a, **k: True          # skip debug-image writes


def parse_args():
    args = [a for a in sys.argv[1:]]
    diag = "auto"
    if "--diag" in args:
        i = args.index("--diag"); diag = args[i + 1]; del args[i:i + 2]
    # defaults match the project structure (run from the project root)
    images_dir = args[0] if len(args) > 0 else "data/images"
    csv_path   = args[1] if len(args) > 1 else "data/image_measurements_77.csv"
    module     = args[2] if len(args) > 2 else "auto"
    return images_dir, csv_path, module, diag


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
    images_dir, csv_path, module, diag_mode = parse_args()
    A = importlib.import_module(module)

    gt = {}
    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f):
            gt[row["names"].strip()] = (float(row["d1"]), float(row["d2"]),
                                        float(row["d3"]), float(row["vol"]))

    results = []
    print(f"Evaluating {len(gt)} images from '{images_dir}' using module '{module}'...\n")
    for i, (name, (g1, g2, g3, gv)) in enumerate(gt.items(), 1):
        path = find_image(images_dir, name)
        if path is None:
            results.append(dict(name=name, status="IMAGE_NOT_FOUND",
                                gt_vol=gv, got_vol=None, vol_err=None, diag="", got=""))
            print(f"[{i:3}/{len(gt)}] {name:12} IMAGE NOT FOUND")
            continue
        roi = cv2.imread(path)
        if roi is None:
            results.append(dict(name=name, status="UNREADABLE",
                                gt_vol=gv, got_vol=None, vol_err=None, diag="", got=""))
            continue
        diag = pick_diag(A, roi, diag_mode)
        buf = io.StringIO()
        try:
            with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
                vol, (d1, d2, d3) = A.process_image(roi_path=path, diagonal_panel=diag)
            ve = abs(vol - gv) / gv * 100.0
            results.append(dict(name=name, status="OK", gt_vol=gv, got_vol=vol,
                                vol_err=ve, diag=diag, got=f"{d1:.0f}/{d2:.0f}/{d3:.0f}"))
            print(f"[{i:3}/{len(gt)}] {name:12} GT={gv:7.1f}  got={vol:7.1f}  err={ve:6.1f}%  ({diag})")
        except Exception as e:
            log = buf.getvalue()
            split = "2 panels" if "[split]" in log else "NOT split (single panel)"
            reasons = [ln.strip() for ln in log.splitlines()
                       if ("WARNING" in ln or "failed" in ln.lower()
                           or "implausible" in ln.lower() or "seed" in ln.lower())]
            tail = " | ".join(reasons[-2:]) if reasons else ""
            why = f"{str(e)[:40]} [{split}] {tail}"
            results.append(dict(name=name, status="FAIL", gt_vol=gv, got_vol=None,
                                vol_err=None, diag=diag, got=why[:160]))
            print(f"[{i:3}/{len(gt)}] {name:12} FAILED: {why[:120]}")

    # ---- write per-image results ----
    out_csv = "eval_results.csv"
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["name", "status", "gt_vol", "got_vol",
                                          "vol_err", "diag", "got"])
        w.writeheader()
        for r in results:
            w.writerow(r)

    # ---- summary ----
    ok = [r for r in results if r["status"] == "OK"]
    errs = np.array([r["vol_err"] for r in ok]) if ok else np.array([])
    print("\n" + "=" * 60)
    print(f"RESULTS  ({len(ok)}/{len(results)} ran successfully)")
    if len(errs):
        print(f"  volume error  mean={errs.mean():6.1f}%   median={np.median(errs):6.1f}%")
        for thr in (10, 20, 30, 50):
            print(f"    under {thr:3d}% error : {int((errs < thr).sum()):3d}/{len(ok)}")
        worst = sorted(ok, key=lambda r: -r["vol_err"])[:10]
        print("\n  worst 10:")
        for r in worst:
            print(f"    {r['name']:12} err={r['vol_err']:6.1f}%  GT={r['gt_vol']:.0f} got={r['got_vol']:.0f} ({r['got']})")
    failed = [r for r in results if r["status"] not in ("OK",)]
    if failed:
        print(f"\n  {len(failed)} FAILED to run:")
        nosplit = sum(1 for r in failed if "NOT split" in str(r.get("got","")))
        print(f"    - panel-split failures (image not split into 2): {nosplit}")
        print(f"    - other segmentation failures:                   {len(failed)-nosplit}")
        print("    details (see eval_results.csv 'got' column for full reason):")
        for r in failed[:20]:
            print(f"      {r['name']:12} {str(r.get('got',''))[:90]}")
    print(f"\n  per-image results written to: {out_csv}")
    print("=" * 60)


if __name__ == "__main__":
    main()