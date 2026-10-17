import cv2
import numpy as np
import pytesseract
import re
import os
import sys
from skimage.segmentation import chan_vese

DEBUG_DIR = "debug_output"

def _show(name, img, max_w=1100, max_h=750):
    """Display an image scaled to fit on screen (prevents huge native-res
    frames from appearing 'zoomed in' / overflowing the display)."""
    h, w = img.shape[:2]
    scale = min(max_w / float(w), max_h / float(h), 1.0)
    disp = cv2.resize(img, (max(1, int(w * scale)), max(1, int(h * scale))),
                      interpolation=cv2.INTER_AREA) if scale < 1.0 else img
    cv2.namedWindow(name, cv2.WINDOW_NORMAL)
    cv2.imshow(name, disp)
    cv2.resizeWindow(name, disp.shape[1], disp.shape[0])



# =========================================================
# TESSERACT PATH (Windows only — Linux/Mac: leave as-is)
# =========================================================
tesseract_path = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
if os.path.exists(tesseract_path):
    pytesseract.pytesseract.tesseract_cmd = tesseract_path


def warn(msg):
    print(f"[WARNING] {msg}", file=sys.stderr)


# =========================================================
# ROI is now always supplied directly via process_image(roi_box=...)
# — auto_detect_roi() was removed since it's no longer used for
# anything; detection is replaced entirely by the caller-supplied box.
# =========================================================


# def split_panels(roi):
#     gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
#     h, w = gray.shape
#     search_start, search_end = int(w * 0.25), int(w * 0.75)
#     col_mean = np.mean(gray[:, search_start:search_end], axis=0)

#     rel_idx = int(np.argmin(col_mean))
#     split_x = search_start + rel_idx
#     dip_depth = col_mean.mean() - col_mean[rel_idx]
#     dip_significance = dip_depth / (col_mean.std() + 1e-6)

#     if dip_significance < 0.8:
#         warn("No clear panel boundary found — treating image as a single panel.")
#         return [roi], False

#     split_x = max(int(w * 0.30), min(split_x, int(w * 0.70)))
#     left = roi[:, :split_x]
#     right = roi[:, split_x:]
#     print(f"[split] x={split_x}  left={left.shape[1]}px  right={right.shape[1]}px")
#     return [left, right], True
def split_panels(roi):
    """Split a dual-panel ultrasound image straight down the middle.
    No dip detection — the two panels are assumed to be equal halves,
    so the split is always at the horizontal centre and always returns
    two panels."""
    h, w = roi.shape[:2]
    split_x = w // 2
    left = roi[:, :split_x]
    right = roi[:, split_x:]
    print(f"[split] x={split_x}  left={left.shape[1]}px  right={right.shape[1]}px")
    return [left, right], True


def remove_overlays(panel):
    """Inpaint the bright YELLOW/GREEN annotation overlays (caliper cross
    & diagonal lines, crosshairs) so they cannot wall off the dark bladder
    lumen during flood-fill / thresholding. Targets the two caliper hues
    specifically (both are low-blue), so a blue image tint is never masked.
    No-op on a clean clinical image with no coloured overlay."""
    if panel.ndim != 3:
        return panel
    b = panel[:, :, 0].astype(int)
    g = panel[:, :, 1].astype(int)
    r = panel[:, :, 2].astype(int)
    yellow = (r > 120) & (g > 120) & (r - b > 50) & (g - b > 50)
    green  = (g > 110) & (g - r > 45) & (g - b > 45)
    mask = ((yellow | green).astype(np.uint8)) * 255
    if cv2.countNonZero(mask) == 0:
        return panel
    mask = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)), 1)
    return cv2.inpaint(panel, mask, 5, cv2.INPAINT_TELEA)


# ---------------------------------------------------------------------
# FIX 1 : grid-consensus scale detection
# ---------------------------------------------------------------------
def _ticks_in_band(gray, x0, x1, min_tick_h=2, max_tick_h=24, bright_thr=80.0):
    """Y-centres of thin bright horizontal marks in a vertical edge band."""
    x0, x1 = max(0, int(x0)), min(gray.shape[1], int(x1))
    if x1 - x0 < 2:
        return []
    prof = gray[:, x0:x1].astype(float).mean(axis=1)
    bright = prof > bright_thr
    cl, st = [], None
    for y in range(len(bright)):
        if bright[y] and st is None:
            st = y
        elif not bright[y] and st is not None:
            cl.append((st, y - 1)); st = None
    if st is not None:
        cl.append((st, len(bright) - 1))
    return [(a + b) / 2.0 for a, b in cl if min_tick_h <= (b - a + 1) <= max_tick_h]


def _consensus(ticks, d, tol_px=6.0):
    """How many DISTINCT grid slots (period d) the ticks occupy, plus span
    and fill fraction. Counting distinct slots (not ticks) stops a cluster
    of marks collapsing onto one slot and inflating the score, and makes
    sub-harmonics fail the fill test."""
    tol = max(tol_px, 0.06 * d)
    best, best_ks = 0, None
    for anchor in ticks:
        k = np.round((ticks - anchor) / d)
        sel = np.abs(ticks - (anchor + k * d)) < tol
        if sel.any():
            ks = np.unique(k[sel])
            if len(ks) > best:
                best, best_ks = len(ks), ks
    if best_ks is None:
        return 0, 0.0, 0.0
    spread = int(best_ks.max() - best_ks.min())
    span = spread * d
    fill = best / (spread + 1) if spread >= 0 else 0.0
    return best, span, fill


def _grid(ticks, min_sp=12.0, max_sp=400.0, min_slots=5):
    """Best regular spacing for a set of tick positions, by grid consensus.
    Candidates are the pairwise gaps; we keep those with >= min_slots
    aligned slots and fill >= 0.70 (rejects half-spacing harmonics, which
    have fill ~0.5), then take max slot count, tie-broken by larger spacing
    (rejects double-spacing harmonics, which have ~half the slots)."""
    ticks = np.array(sorted(ticks), float)
    if len(ticks) < min_slots:
        return None
    cand = set()
    for i in range(len(ticks)):
        for j in range(i + 1, len(ticks)):
            gap = ticks[j] - ticks[i]
            if min_sp <= gap <= max_sp:
                cand.add(round(gap, 1))
    res = []
    for d in sorted(cand):
        n, span, fill = _consensus(ticks, d)
        if n >= min_slots and fill >= 0.70:
            res.append((n, fill, d, span))
    if not res:
        return None
    res.sort(key=lambda r: (r[0], r[2]))
    n, fill, d, span = res[-1]
    return dict(d=d, slots=n, fill=fill, span=span)


def _best_on_side(gray, edge):
    """Auto-locate the ruler on one side by scanning several sub-bands and
    keeping the one with the strongest grid (handles rulers that sit a few
    percent in from the edge, and skips contaminated columns)."""
    h, w = gray.shape
    best = None
    for fs in (0.002, 0.006, 0.010, 0.015, 0.022, 0.030):
        for bw in (0.010, 0.018, 0.028):
            if edge == "left":
                x0, x1 = w * fs, w * (fs + bw)
            else:
                x0, x1 = w * (1 - fs - bw), w * (1 - fs)
            g = _grid(_ticks_in_band(gray, x0, x1))
            if g and (best is None
                      or g["slots"] > best["slots"]
                      or (g["slots"] == best["slots"] and g["span"] > best["span"])):
                best = g; best["edge"] = edge
    return best


def get_scale_mm_per_px(gray, panel_height_px, minor_tick_cm=1.0,
                        depth_range=(4.0, 25.0)):
    """Return (mm_per_px, info) from the depth ruler, or (None, None).
    Tries both edges, prefers a plausible implied scan depth, then the
    strongest grid."""
    cands = []
    for edge in ("left", "right"):
        b = _best_on_side(gray, edge)
        if b:
            mmpp = (minor_tick_cm * 10.0) / b["d"]
            depth = panel_height_px * mmpp / 10.0
            b["mm_per_px"] = mmpp
            b["depth"] = depth
            b["plausible"] = depth_range[0] <= depth <= depth_range[1]
            cands.append(b)
    if not cands:
        return None, None
    usable = [c for c in cands if c["plausible"]] or cands
    usable.sort(key=lambda c: (c["plausible"], c["slots"], c["span"]))
    return usable[-1]["mm_per_px"], usable[-1]


def get_pixel_spacing_auto(image, panel_height_px, depth_cm=None, minor_tick_cm=1.0):
    """Drop-in replacement. Priority: manual depth -> grid-consensus ruler
    detection -> OCR of the 'D ##.#' field -> 14 cm default (unreliable)."""
    if depth_cm is not None:
        mm_per_px = (depth_cm * 10.0) / panel_height_px
        print(f"[pixel spacing] manual depth={depth_cm}cm -> mm_per_px={mm_per_px:.4f}")
        return mm_per_px

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    mm_per_px, info = get_scale_mm_per_px(gray, panel_height_px, minor_tick_cm=minor_tick_cm)
    if mm_per_px is not None:
        print(f"[pixel spacing] {info['edge']} edge grid: {info['d']:.1f}px/cm "
              f"slots={info['slots']} -> mm_per_px={mm_per_px:.4f} "
              f"(implied depth {info['depth']:.1f}cm"
              f"{'' if info['plausible'] else ', IMPLAUSIBLE'})")
        return mm_per_px

    warn("No ruler grid found on either edge — falling back to OCR.")
    h_img, w_img = gray.shape
    depth_found = None
    try:
        for frac in (0.85, 0.78, 0.70, 0.60, 0.50):
            strip = cv2.resize(gray[:, int(w_img * frac):], None, fx=2, fy=2,
                               interpolation=cv2.INTER_CUBIC)
            text = re.sub(r"\s+", " ", pytesseract.image_to_string(strip, config="--psm 6"))
            for m in re.finditer(r"D.{0,6}?(\d{1,2}\.\d)", text):
                v = float(m.group(1))
                if 5.0 <= v <= 30.0:
                    depth_found = v; break
            if depth_found:
                break
    except Exception as e:
        warn(f"OCR failed: {e}")
    if depth_found is None:
        warn("OCR failed too — defaulting to 14.0 cm (UNRELIABLE; pass depth_cm=...).")
        depth_found = 14.0
    mm_per_px = depth_found * 10.0 / panel_height_px
    print(f"[pixel spacing] fallback depth={depth_found}cm -> mm_per_px={mm_per_px:.4f}")
    return mm_per_px


def manual_scale_calibration(image, known_distance_cm=10.0,
                              window_name="Click the two scale-ruler ticks (e.g. 0cm and 10cm), then press any key"):
    """
    Lets you click two points directly on the depth-scale ruler ticks
    visible in the image (e.g. the '0' and '10' marks) and computes
    mm_per_px from the real pixel distance between them — sidesteps
    OCR entirely, which is the right call when the depth text itself
    is too distorted/cropped to read reliably, since the tick marks
    are unambiguous and you can verify the click visually before
    committing to it.

    Click order doesn't matter. Press any key once both points are
    placed to confirm, or Esc to cancel.
    """
    points = []
    display = image.copy()

    def on_click(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and len(points) < 2:
            points.append((x, y))
            cv2.circle(display, (x, y), 6, (0, 0, 255), -1)
            if len(points) == 2:
                cv2.line(display, points[0], points[1], (0, 255, 0), 2)
            _show(window_name, display)

    _show(window_name, display)
    cv2.setMouseCallback(window_name, on_click)
    print(f"[manual calibration] click on the two ruler ticks that are "
          f"{known_distance_cm}cm apart (e.g. the '0' and '{known_distance_cm:.0f}' marks), "
          f"then press any key to confirm...")

    while True:
        key = cv2.waitKey(20) & 0xFF
        if key == 27:  # Esc
            cv2.destroyWindow(window_name)
            raise ValueError("Manual calibration cancelled (Esc pressed) before 2 points were placed.")
        if len(points) == 2 and key != 255:
            break

    cv2.destroyWindow(window_name)

    (x1, y1), (x2, y2) = points
    pixel_dist = float(np.hypot(x2 - x1, y2 - y1))
    if pixel_dist < 1e-3:
        raise ValueError("The two clicked points were on top of each other — calibration failed, try again.")

    known_distance_mm = known_distance_cm * 10.0
    mm_per_px = known_distance_mm / pixel_dist
    print(f"[manual calibration] points={points}  pixel_dist={pixel_dist:.1f}px  "
          f"known={known_distance_mm:.0f}mm  -> mm_per_px={mm_per_px:.4f}")
    return mm_per_px



def _find_seed(panel, zone=(0.25, 0.75, 0.15, 0.75)):
    gray = cv2.cvtColor(panel, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    fx0, fx1, fy0, fy1 = zone
    cx0, cx1, cy0, cy1 = int(w * fx0), int(w * fx1), int(h * fy0), int(h * fy1)
    central = gray[cy0:cy1, cx0:cx1]
    blurred = cv2.GaussianBlur(central, (15, 15), 0)
    min_idx = np.unravel_index(np.argmin(blurred), blurred.shape)
    return (cx0 + min_idx[1], cy0 + min_idx[0])
from scipy.ndimage import distance_transform_edt  # not used directly anymore,
                                                    # kept here only as a reminder
                                                    # of what NOT to use (see notes)
from skimage.segmentation import chan_vese
import time


def _ellipse_level_set(shape, center, axes, angle_deg):
    h, w = shape
    yv, xv = np.mgrid[0:h, 0:w].astype(np.float64)
    theta = np.deg2rad(angle_deg)
    cos_t, sin_t = np.cos(theta), np.sin(theta)
    dx, dy = xv - center[0], yv - center[1]
    xr = dx * cos_t + dy * sin_t
    yr = -dx * sin_t + dy * cos_t
    a, b = axes
    e = (xr / a) ** 2 + (yr / b) ** 2
    return 1.0 - e


def _mask_level_set(mask):
    """Signed-distance level set from an arbitrary binary mask (used
    once the ellipse is no longer a pure analytic ellipse — see
    segment_bladder_full stage 2, where it gets intersected with the
    detected boundary)."""
    from scipy.ndimage import distance_transform_edt
    inside = mask > 0
    dist_in = distance_transform_edt(inside)
    dist_out = distance_transform_edt(~inside)
    return dist_in - dist_out


def segment_bladder_chanvese(panel, label="", right_margin_frac=0.05,
                              core_tol=25, ellipse_scale=1.35,
                              mu=0.1, max_num_iter=250, chanvese_downscale=0.35):
    
    gray = cv2.cvtColor(panel, cv2.COLOR_BGR2GRAY)
    h_img, w_img = gray.shape
    margin_y = int(h_img * 0.08)
    margin_x = int(w_img * 0.05)
    margin_x_right = int(w_img * right_margin_frac)
    crop = gray[margin_y: h_img - margin_y, margin_x: w_img - margin_x_right]
    ch, cw = crop.shape

    seed_full = _find_seed(panel)
    seed_crop = (seed_full[0] - margin_x, seed_full[1] - margin_y)
    if not (0 <= seed_crop[0] < cw and 0 <= seed_crop[1] < ch):
        warn(f"[{label}] seed point outside crop")
        return None
    sx, sy = seed_crop
    near_black = (crop < 25).astype(np.uint8) * 255
    ff_mask = np.zeros((ch + 2, cw + 2), np.uint8)
    flood_src = near_black.copy()
    for (bx, by) in [(0, 0), (cw - 1, 0), (0, ch - 1), (cw - 1, ch - 1)]:
        if flood_src[by, bx] == 255:
            cv2.floodFill(flood_src, ff_mask, (bx, by), 128)
    bg = (flood_src == 128).astype(np.uint8) * 255
    bg = cv2.dilate(bg, np.ones((9, 9), np.uint8), iterations=1)
    masked = crop.copy()
    masked[bg > 0] = 255

    # --- Step 1: conservative core estimate, just to anchor an ellipse. ---
    ff = np.zeros((ch + 2, cw + 2), np.uint8)
    flood = masked.copy()
    cv2.floodFill(flood, ff, (int(sx), int(sy)), 200,
                   loDiff=core_tol, upDiff=core_tol, flags=8 | cv2.FLOODFILL_FIXED_RANGE)
    core = (ff[1:-1, 1:-1] > 0).astype(np.uint8) * 255
    core = cv2.morphologyEx(core, cv2.MORPH_CLOSE,
                             cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)), iterations=1)
    contours, _ = cv2.findContours(core, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) == 0:
        warn(f"[{label}] couldn't find an initial core to fit an ellipse to.")
        return None
    core_cnt = max(contours, key=cv2.contourArea)

    if len(core_cnt) < 5:
        bx, by, bw, bh = cv2.boundingRect(core_cnt)
        center, axes, angle = (bx + bw / 2, by + bh / 2), (bw / 2, bh / 2), 0.0
    else:
        center, axes, angle = cv2.fitEllipse(core_cnt)
        axes = (axes[0] / 2, axes[1] / 2)  # cv2 returns full-length axes
    big_axes = (max(axes[0] * ellipse_scale, 5), max(axes[1] * ellipse_scale, 5))

    ellipse_mask = np.zeros((ch, cw), np.uint8)
    cv2.ellipse(ellipse_mask, (int(center[0]), int(center[1])),
                (int(big_axes[0]), int(big_axes[1])), angle, 0, 360, 255, -1)

    print(f"[{label}] running Chan-Vese ({max_num_iter} iters max, "
          f"{chanvese_downscale:.2f}x scale)...", flush=True)
    t0 = time.time()
    ds = chanvese_downscale
    if ds < 1.0:
        small = cv2.resize(masked, None, fx=ds, fy=ds, interpolation=cv2.INTER_AREA)
    else:
        small = masked
    sh, sw = small.shape
    level_set = _ellipse_level_set((sh, sw), (center[0] * ds, center[1] * ds),
                                    (big_axes[0] * ds, big_axes[1] * ds), angle)
    image_f = small.astype(np.float64) / 255.0
    seg = chan_vese(image_f, mu=mu, lambda1=1.5, lambda2=1.0, tol=1e-3,
                     max_num_iter=max_num_iter, dt=0.5,
                     init_level_set=level_set, extended_output=False)
    seg_mask = seg.astype(np.uint8) * 255
    if ds < 1.0:
        seg_mask = cv2.resize(seg_mask, (cw, ch), interpolation=cv2.INTER_NEAREST)
    print(f"[{label}] Chan-Vese done in {time.time()-t0:.2f}s.", flush=True)

    if seg_mask[sy, sx] == 0:
        seg_mask = 255 - seg_mask
        seg_mask[bg > 0] = 0

    seg_mask = cv2.bitwise_and(seg_mask, ellipse_mask)
    seg_mask = cv2.morphologyEx(seg_mask, cv2.MORPH_CLOSE,
                                 cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)), iterations=1)

    contours, _ = cv2.findContours(seg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) == 0:
        warn(f"[{label}] chan_vese produced an empty segmentation.")
        return None
    seed_pt = (float(sx), float(sy))
    containing = [c for c in contours if cv2.pointPolygonTest(c, seed_pt, False) >= 0]
    cnt = max(containing, key=cv2.contourArea) if containing else max(contours, key=cv2.contourArea)

    area = cv2.contourArea(cnt)
    if not (0.02 * ch * cw < area < 0.6 * ch * cw):
        warn(f"[{label}] chan_vese result area ({area/(ch*cw):.2%} of panel) "
             f"looks implausible — check core_tol/ellipse_scale for this image.")
        return None

    best_cnt = cnt.copy()
    best_cnt[:, :, 0] += margin_x
    best_cnt[:, :, 1] += margin_y

    os.makedirs(DEBUG_DIR, exist_ok=True)
    debug = panel.copy()
    cv2.ellipse(debug, (int(center[0]) + margin_x, int(center[1]) + margin_y),
                (int(big_axes[0]), int(big_axes[1])), angle, 0, 360, (255, 0, 0), 1)
    cv2.drawContours(debug, [best_cnt], -1, (0, 255, 0), 2)
    cv2.circle(debug, seed_full, 5, (0, 0, 255), -1)
    _show(
        f"Segmented {label} (chan-vese)",
        debug
    )

    cv2.waitKey(1000)
    return best_cnt
def segment_bladder(panel, label="", right_margin_frac=0.05):

    gray = cv2.cvtColor(panel, cv2.COLOR_BGR2GRAY)

    h_img, w_img = gray.shape

    margin_y = int(h_img * 0.08)
    margin_x = int(w_img * 0.05)
    margin_x_right = int(w_img * right_margin_frac)

    crop = gray[
        margin_y:h_img - margin_y,
        margin_x:w_img - margin_x_right
    ]

    ch, cw = crop.shape

    # CLAHE-enhanced version, used ONLY for the threshold sweep below —
    # NOT for background exclusion, which deliberately stays on the
    # original `crop`. Validated: CLAHE measurably reduces leakage on
    # blurry images (boundary tracks the real wall instead of escaping
    # past it), with no downside found on sharp images either. Using it
    # for background-exclusion too was tried and rejected — it brightens
    # the dark outside-sector corners enough to break that step's
    # near-black assumption.
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    crop_enh = clahe.apply(crop)

    seed_full = _find_seed(panel)

    seed_crop = (
        seed_full[0] - margin_x,
        seed_full[1] - margin_y
    )

    if not (
        0 <= seed_crop[0] < cw and
        0 <= seed_crop[1] < ch
    ):
        warn(f"[{label}] seed point outside crop")
        return None

    # --------------------------------------------------
    # Background exclusion
    # --------------------------------------------------

    near_black = (crop < 25).astype(np.uint8) * 255

    ff_mask = np.zeros((ch + 2, cw + 2), np.uint8)

    flood_src = near_black.copy()

    for sx, sy in [
        (0, 0),
        (cw - 1, 0),
        (0, ch - 1),
        (cw - 1, ch - 1)
    ]:

        if flood_src[sy, sx] == 255:

            cv2.floodFill(
                flood_src,
                ff_mask,
                (sx, sy),
                128
            )

    bg = np.uint8(flood_src == 128) * 255

    bg = cv2.dilate(
        bg,
        np.ones((9, 9), np.uint8),
        iterations=1
    )

    # --------------------------------------------------
    # Threshold sweep
    # --------------------------------------------------

    best_cnt = None
    best_score = -1

    otsu_val = cv2.threshold(
        crop_enh,
        0,
        255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )[0]

    thresh_vals = [otsu_val, 40, 50, 60, 70, 80]

    for tv in thresh_vals:

        blur = cv2.GaussianBlur(
            crop_enh,
            (21, 21),
            0
        )

        _, thresh = cv2.threshold(
            blur,
            tv,
            255,
            cv2.THRESH_BINARY_INV
        )

        thresh[bg > 0] = 0

        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (7, 7)
        )

        thresh = cv2.morphologyEx(
            thresh,
            cv2.MORPH_CLOSE,
            kernel,
            iterations=1
        )

        # ------------------------------------------
        # REGION GROWING FROM SEED
        # ------------------------------------------

        sx, sy = seed_crop

        if thresh[sy, sx] == 0:
            continue

        flood = thresh.copy()

        seed_mask = np.zeros(
            (ch + 2, cw + 2),
            np.uint8
        )

        cv2.floodFill(
            flood,
            seed_mask,
            (int(sx), int(sy)),
            128
        )

        seed_region = np.uint8(
            flood == 128
        ) * 255

        seed_region = cv2.morphologyEx(
            seed_region,
            cv2.MORPH_OPEN,
            np.ones((5, 5), np.uint8)
        )

        _show(
            f"Seed Region {label}",
            seed_region
        )
        cv2.waitKey(200)

        contours, _ = cv2.findContours(
            seed_region,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        if len(contours) == 0:
            continue

        cnt = max(
            contours,
            key=cv2.contourArea
        )

        area = cv2.contourArea(cnt)

        if area < 0.02 * ch * cw:
            continue

        hull_area = cv2.contourArea(
            cv2.convexHull(cnt)
        )

        solidity = (
            area / hull_area
            if hull_area > 0
            else 0
        )

        score = area * solidity

        if score > best_score:

            best_score = score

            cnt_s = cnt.copy()

            cnt_s[:, :, 0] += margin_x
            cnt_s[:, :, 1] += margin_y

            best_cnt = cnt_s

    if best_cnt is None:

        warn(
            f"[{label}] segmentation failed"
        )

        return None

    debug = panel.copy()

    cv2.drawContours(
        debug,
        [best_cnt],
        -1,
        (0, 255, 0),
        2
    )

    cv2.circle(
        debug,
        seed_full,
        5,
        (0, 0, 255),
        -1
    )

    _show(
        f"Segmented {label}",
        debug
    )

    cv2.waitKey(1500)

    os.makedirs(
        DEBUG_DIR,
        exist_ok=True
    )

    cv2.imwrite(
        os.path.join(
            DEBUG_DIR,
            f"segmented_{label}.png"
        ),
        debug
    )

    return best_cnt


# =========================================================
# THE 3-STAGE PLAN (per the whiteboard sketch):
#
#   Stage 1 — "catch these boundaries first, so detection doesn't go
#              outside": run the existing threshold-sweep +
#              region-growing segmentation (segment_bladder) to get a
#              rough but real boundary of the bladder wall.
#
#   Stage 2 — "the ellipse bounded inside the boundaries we already
#              defined": fit an ellipse to that boundary, but instead
#              of inflating it past the boundary (the old
#              ellipse_scale > 1 approach), intersect it with the
#              boundary mask so the ellipse can never extend past what
#              stage 1 already found. ellipse_scale here is meant to
#              stay close to 1.0 (a little padding only) — the
#              boundary itself does the bounding, not the scale.
#
#   Stage 3 — "use Chan-Vese to bound the bladder": refine within that
#              bounded ellipse with Chan-Vese, exactly as before.
# =========================================================
def segment_bladder_full(panel, label="", right_margin_frac=0.05,
                          ellipse_scale=1.05, boundary_dilate_px=4,
                          mu=0.1, max_num_iter=250, chanvese_downscale=0.35):

    # --- Stage 1: catch the real boundary first ---
    boundary_cnt = segment_bladder(panel, label=label, right_margin_frac=right_margin_frac)
    if boundary_cnt is None:
        warn(f"[{label}] stage 1 (boundary detection) failed — cannot proceed to ellipse/Chan-Vese.")
        return None

    gray = cv2.cvtColor(panel, cv2.COLOR_BGR2GRAY)
    h_img, w_img = gray.shape
    margin_y = int(h_img * 0.08)
    margin_x = int(w_img * 0.05)
    margin_x_right = int(w_img * right_margin_frac)
    crop = gray[margin_y:h_img - margin_y, margin_x:w_img - margin_x_right]
    ch, cw = crop.shape

    seed_full = _find_seed(panel)
    sx, sy = seed_full[0] - margin_x, seed_full[1] - margin_y
    if not (0 <= sx < cw and 0 <= sy < ch):
        warn(f"[{label}] seed point outside crop")
        return None

    near_black = (crop < 25).astype(np.uint8) * 255
    ff_mask = np.zeros((ch + 2, cw + 2), np.uint8)
    flood_src = near_black.copy()
    for (bx, by) in [(0, 0), (cw - 1, 0), (0, ch - 1), (cw - 1, ch - 1)]:
        if flood_src[by, bx] == 255:
            cv2.floodFill(flood_src, ff_mask, (bx, by), 128)
    bg = (flood_src == 128).astype(np.uint8) * 255
    bg = cv2.dilate(bg, np.ones((9, 9), np.uint8), iterations=1)
    masked = crop.copy()
    masked[bg > 0] = 255

    # boundary_cnt came back in PANEL coordinates (segment_bladder adds
    # margin_x/margin_y before returning) — shift into this crop's frame.
    boundary_crop = boundary_cnt.copy()
    boundary_crop[:, :, 0] -= margin_x
    boundary_crop[:, :, 1] -= margin_y

    boundary_mask = np.zeros((ch, cw), np.uint8)
    cv2.drawContours(boundary_mask, [boundary_crop], -1, 255, -1)
    if boundary_dilate_px > 0:
        boundary_mask = cv2.dilate(
            boundary_mask,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (boundary_dilate_px * 2 + 1,) * 2),
            iterations=1,
        )

    # --- Stage 2: ellipse, fit to the boundary and then intersected
    # with it so it can never extend past what stage 1 found ---
    if len(boundary_crop) < 5:
        bx, by, bw, bh = cv2.boundingRect(boundary_crop)
        center, axes, angle = (bx + bw / 2, by + bh / 2), (bw / 2, bh / 2), 0.0
    else:
        center, axes, angle = cv2.fitEllipse(boundary_crop)
        axes = (axes[0] / 2, axes[1] / 2)
    big_axes = (max(axes[0] * ellipse_scale, 5), max(axes[1] * ellipse_scale, 5))

    raw_ellipse_mask = np.zeros((ch, cw), np.uint8)
    cv2.ellipse(raw_ellipse_mask, (int(center[0]), int(center[1])),
                (int(big_axes[0]), int(big_axes[1])), angle, 0, 360, 255, -1)

    # the actual bound: ellipse AND boundary — this is "the ellipse
    # bounded inside the boundaries we already defined"
    ellipse_mask = cv2.bitwise_and(raw_ellipse_mask, boundary_mask)
    if cv2.countNonZero(ellipse_mask) < 0.05 * ch * cw:
        warn(f"[{label}] ellipse/boundary intersection is implausibly small — "
             f"falling back to the raw (unclipped) ellipse for this panel.")
        ellipse_mask = raw_ellipse_mask

    debug_stage2 = panel.copy()
    cv2.drawContours(debug_stage2, [boundary_crop + [margin_x, margin_y]], -1, (0, 165, 255), 2)
    cv2.ellipse(debug_stage2, (int(center[0]) + margin_x, int(center[1]) + margin_y),
                (int(big_axes[0]), int(big_axes[1])), angle, 0, 360, (255, 0, 0), 1)
    overlay_mask_full = np.zeros((h_img, w_img), np.uint8)
    overlay_mask_full[margin_y:h_img - margin_y, margin_x:w_img - margin_x_right] = ellipse_mask
    contours_ell, _ = cv2.findContours(overlay_mask_full, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(debug_stage2, contours_ell, -1, (0, 255, 0), 2)
    _show(f"Stage2 {label} ellipse bounded inside boundary", debug_stage2)
    cv2.waitKey(800)

    # --- Stage 3: Chan-Vese refine within that bounded ellipse ---
    print(f"[{label}] running Chan-Vese ({max_num_iter} iters max, "
          f"{chanvese_downscale:.2f}x scale)...", flush=True)
    t0 = time.time()
    ds = chanvese_downscale
    if ds < 1.0:
        small = cv2.resize(masked, None, fx=ds, fy=ds, interpolation=cv2.INTER_AREA)
        small_ellipse_mask = cv2.resize(ellipse_mask, None, fx=ds, fy=ds, interpolation=cv2.INTER_NEAREST)
    else:
        small = masked
        small_ellipse_mask = ellipse_mask
    level_set = _mask_level_set(small_ellipse_mask)
    image_f = small.astype(np.float64) / 255.0
    seg = chan_vese(image_f, mu=mu, lambda1=1.5, lambda2=1.0, tol=1e-3,
                     max_num_iter=max_num_iter, dt=0.5,
                     init_level_set=level_set, extended_output=False)
    seg_mask = seg.astype(np.uint8) * 255
    if ds < 1.0:
        seg_mask = cv2.resize(seg_mask, (cw, ch), interpolation=cv2.INTER_NEAREST)
    print(f"[{label}] Chan-Vese done in {time.time()-t0:.2f}s.", flush=True)

    if seg_mask[sy, sx] == 0:
        seg_mask = 255 - seg_mask
        seg_mask[bg > 0] = 0

    # final guard: clip to the bounded ellipse again (belt-and-suspenders)
    seg_mask = cv2.bitwise_and(seg_mask, ellipse_mask)
    seg_mask = cv2.morphologyEx(seg_mask, cv2.MORPH_CLOSE,
                                 cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)), iterations=1)

    contours, _ = cv2.findContours(seg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) == 0:
        warn(f"[{label}] chan_vese produced an empty segmentation.")
        return None
    seed_pt = (float(sx), float(sy))
    containing = [c for c in contours if cv2.pointPolygonTest(c, seed_pt, False) >= 0]
    cnt = max(containing, key=cv2.contourArea) if containing else max(contours, key=cv2.contourArea)

    area = cv2.contourArea(cnt)
    if not (0.02 * ch * cw < area < 0.6 * ch * cw):
        warn(f"[{label}] chan_vese result area ({area/(ch*cw):.2%} of panel) "
             f"looks implausible — check ellipse_scale/boundary_dilate_px for this image.")
        return None

    best_cnt = cnt.copy()
    best_cnt[:, :, 0] += margin_x
    best_cnt[:, :, 1] += margin_y

    os.makedirs(DEBUG_DIR, exist_ok=True)
    debug = panel.copy()
    cv2.drawContours(debug, [boundary_cnt], -1, (0, 165, 255), 1)
    cv2.drawContours(debug, contours_ell, -1, (255, 0, 0), 1)
    cv2.drawContours(debug, [best_cnt], -1, (0, 255, 0), 2)
    cv2.circle(debug, seed_full, 5, (0, 0, 255), -1)
    _show(f"Stage3 {label} FINAL (chan-vese)", debug)
    cv2.waitKey(1000)
    cv2.imwrite(os.path.join(DEBUG_DIR, f"segmented_full_{label}.png"), debug)

    return best_cnt

def max_horizontal_distance(contour):

    x, y, w, h = cv2.boundingRect(contour)

    mask = np.zeros((h + 20, w + 20), dtype=np.uint8)

    shifted = contour.copy()
    shifted[:, :, 0] -= (x - 10)
    shifted[:, :, 1] -= (y - 10)

    cv2.drawContours(
        mask,
        [shifted],
        -1,
        255,
        thickness=-1
    )

    best_dist = 0
    best_pair = None

    for row in range(mask.shape[0]):

        xs = np.where(mask[row] > 0)[0]

        if len(xs) < 2:
            continue

        dist = xs[-1] - xs[0]

        if dist > best_dist:

            best_dist = dist

            best_pair = (
                (xs[0] + x - 10, row + y - 10),
                (xs[-1] + x - 10, row + y - 10)
            )

    return best_dist, best_pair

def max_vertical_distance(contour):

    x, y, w, h = cv2.boundingRect(contour)

    mask = np.zeros((h + 20, w + 20), dtype=np.uint8)

    shifted = contour.copy()
    shifted[:, :, 0] -= (x - 10)
    shifted[:, :, 1] -= (y - 10)

    cv2.drawContours(
        mask,
        [shifted],
        -1,
        255,
        thickness=-1
    )

    best_dist = 0
    best_pair = None

    for col in range(mask.shape[1]):

        ys = np.where(mask[:, col] > 0)[0]

        if len(ys) < 2:
            continue

        dist = ys[-1] - ys[0]

        if dist > best_dist:

            best_dist = dist

            best_pair = (
                (col + x - 10, ys[0] + y - 10),
                (col + x - 10, ys[-1] + y - 10)
            )

    return best_dist, best_pair


def max_diagonal_distance(contour):
    """Diagonal (d3) caliper distance, restricted to the top-left ->
    bottom-right orientation:

        .
          .
            .
              .
                .
                  >

    i.e. only point pairs where x and y both increase together are
    considered (a '\\'-shaped diagonal), not the unconstrained Feret
    diameter. The clinical d3 caliper in these scans is always drawn
    this way (from the upper-left edge of the bladder down to its
    lower-right edge) — picking the single longest pairwise distance
    on the hull regardless of direction could instead land on a
    '/'-shaped (top-right to bottom-left) pair, or a near-horizontal
    one, if that happened to be slightly longer, which doesn't match
    what the machine itself measures.
    """
    hull = cv2.convexHull(contour)[:, 0, :].astype(np.float64)
    n = len(hull)
    best_dist, best_pair = 0, None
    for i in range(n):
        for j in range(i + 1, n):
            p1, p2 = hull[i], hull[j]
            if p1[0] <= p2[0] and p1[1] <= p2[1]:
                top_left, bottom_right = p1, p2
            elif p2[0] <= p1[0] and p2[1] <= p1[1]:
                top_left, bottom_right = p2, p1
            else:
                continue  # not a top-left/bottom-right oriented pair
            d = np.linalg.norm(top_left - bottom_right)
            if d > best_dist:
                best_dist, best_pair = d, (top_left, bottom_right)
    if best_pair is None:
        warn("No top-left->bottom-right oriented point pair found on the "
             "hull — falling back to the unconstrained longest pair.")
        for i in range(n):
            for j in range(i + 1, n):
                d = np.linalg.norm(hull[i] - hull[j])
                if d > best_dist:
                    best_dist, best_pair = d, (hull[i], hull[j])
    return best_dist, best_pair


def panel_axes(contour):
    """Kept for backward compatibility / debug visuals: returns
    (major Feret diameter, perpendicular caliper width)."""
    best_dist, best_pair = max_diagonal_distance(contour)
    direction = (best_pair[1] - best_pair[0])
    direction = direction / np.linalg.norm(direction)
    perp = np.array([-direction[1], direction[0]])
    pts = contour[:, 0, :].astype(np.float64)
    proj = pts @ perp
    width = proj.max() - proj.min()
    return best_dist, width, best_pair


def extract_dimensions(orthogonal_cnt, diagonal_cnt, mm_per_px):
    d1_px, d1_pair = max_horizontal_distance(orthogonal_cnt)
    d2_px, d2_pair = max_vertical_distance(orthogonal_cnt)
    d3_px, d3_pair = max_diagonal_distance(diagonal_cnt)
    d1, d2, d3 = d1_px * mm_per_px, d2_px * mm_per_px, d3_px * mm_per_px
    print(f"[orthogonal panel] d1 (horizontal) = {d1:.1f}mm   d2 (vertical) = {d2:.1f}mm")
    print(f"[diagonal panel]    d3 (diagonal)   = {d3:.1f}mm")
    return d1, d2, d3, (d1_pair, d2_pair, d3_pair)

def show_measurements(
        panel,
        contour,
        d1_pair=None,
        d2_pair=None,
        d3_pair=None,
        title="Measurements"
):

    debug = panel.copy()

    cv2.drawContours(
        debug,
        [contour],
        -1,
        (0,255,0),
        2
    )

    if d1_pair:
        cv2.line(
            debug,
            d1_pair[0],
            d1_pair[1],
            (255,0,0),
            3
        )

    if d2_pair:
        cv2.line(
            debug,
            d2_pair[0],
            d2_pair[1],
            (0,0,255),
            3
        )

    if d3_pair:
        p1 = tuple(d3_pair[0].astype(int))
        p2 = tuple(d3_pair[1].astype(int))

        cv2.line(
            debug,
            p1,
            p2,
            (0,255,255),
            3
        )

    _show(title, debug)
    cv2.waitKey(2000)

def combine_panel_dimensions(contours_by_label, mm_per_px, diagonal_panel="RIGHT"):
    labels = list(contours_by_label.keys())

    if len(labels) == 1:
        only_label = labels[0]
        cnt = contours_by_label[only_label]
        d1_px, _ = max_horizontal_distance(cnt)
        d2_px, _ = max_vertical_distance(cnt)
        d3_px, _ = max_diagonal_distance(cnt)
        d1, d2 = d1_px * mm_per_px, d2_px * mm_per_px
        warn("Only one panel/view was found — true 3D volume needs a second, "
             "orthogonal view for the depth axis. Approximating d3 from this "
             "same panel's diagonal; treat the volume as a rough estimate only.")
        return d1, d2, d3_px * mm_per_px

    if diagonal_panel not in contours_by_label:
        warn(f"diagonal_panel='{diagonal_panel}' not found among panels {labels} — "
             f"defaulting to '{labels[-1]}' as the diagonal/depth view.")
        diagonal_panel = labels[-1]
    orthogonal_panel = next(l for l in labels if l != diagonal_panel)

    d1, d2, d3, pairs = extract_dimensions(
        contours_by_label[orthogonal_panel], contours_by_label[diagonal_panel], mm_per_px)
    return d1, d2, d3


# =========================================================
# VOLUME
# =========================================================
def compute_volume(d1_mm, d2_mm, d3_mm):
    d1_cm, d2_cm, d3_cm = d1_mm / 10, d2_mm / 10, d3_mm / 10
    print("\n[REAL DIMENSIONS]")
    print(f"  d1 = {d1_cm:.2f} cm")
    print(f"  d2 = {d2_cm:.2f} cm")
    print(f"  d3 = {d3_cm:.2f} cm")
    volume = 0.52 * d1_cm * d2_cm * d3_cm
    return volume

def show_d1_d2_d3(left_panel,
                  right_panel,
                  left_cnt,
                  right_cnt,
                  d1_pair,
                  d2_pair,
                  d3_pair):

    left_debug = left_panel.copy()

    cv2.drawContours(
        left_debug,
        [left_cnt],
        -1,
        (0,255,0),
        2
    )

    cv2.line(
        left_debug,
        d1_pair[0],
        d1_pair[1],
        (255,0,0),
        3
    )

    cv2.line(
        left_debug,
        d2_pair[0],
        d2_pair[1],
        (0,0,255),
        3
    )

    d1_px = np.linalg.norm(
        np.array(d1_pair[0]) -
        np.array(d1_pair[1])
    )

    d2_px = np.linalg.norm(
        np.array(d2_pair[0]) -
        np.array(d2_pair[1])
    )

    cv2.putText(
        left_debug,
        f"d1={d1_px:.1f}px",
        (20,30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255,0,0),
        2
    )

    cv2.putText(
        left_debug,
        f"d2={d2_px:.1f}px",
        (20,60),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0,0,255),
        2
    )

    _show("D1 + D2", left_debug)

    right_debug = right_panel.copy()

    cv2.drawContours(
        right_debug,
        [right_cnt],
        -1,
        (0,255,0),
        2
    )

    p1 = tuple(d3_pair[0].astype(int))
    p2 = tuple(d3_pair[1].astype(int))

    cv2.line(
        right_debug,
        p1,
        p2,
        (0,255,255),
        3
    )

    d3_px = np.linalg.norm(
        np.array(p1) -
        np.array(p2)
    )

    cv2.putText(
        right_debug,
        f"d3={d3_px:.1f}px",
        (20,30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0,255,255),
        2
    )

    _show("D3", right_debug)

    cv2.waitKey(0)
# =========================================================
# MAIN
# =========================================================
def process_image(image_path=None, roi_box=None, roi_path=None, diagonal_panel="RIGHT",
                   depth_cm=None, manual_calibrate=False, calibration_distance_cm=10.0):
    """
    Two ways to supply the ROI — use exactly one:

    roi_path : path to an already-cropped ROI image (e.g. "roi.png").
        Loaded directly and used as-is — no cropping happens, and
        image_path/roi_box are ignored. Since there's no separate
        full-frame image in this mode, the depth-field OCR (for pixel
        spacing) runs on this same ROI image; if the depth field isn't
        visible within it, get_pixel_spacing_auto() will warn and fall
        back to its default 14cm rather than fail.

    image_path + roi_box : path to the full original screenshot, plus
        (x1, y1, x2, y2) pixel coordinates to crop out of it. Depth-field
        OCR runs on the full image (where the info column actually is).

    Pixel-to-mm scale, three ways (checked in this priority order):
      1. depth_cm=<value>          — skip OCR, use this depth directly.
      2. manual_calibrate=True     — click 2 points on the depth-scale
         ruler ticks yourself (e.g. the '0' and '10' marks); distance
         between them is calibration_distance_cm (default 10cm). Most
         reliable when OCR keeps failing on a given crop/photo.
      3. (default)                 — OCR the depth field automatically.
    """

    if roi_path is not None:
        roi = cv2.imread(roi_path)
        if roi is None:
            raise ValueError(f"ROI image not found: {roi_path}")
        print(f"[roi] loaded pre-cropped ROI directly from '{roi_path}' "
              f"({roi.shape[1]}x{roi.shape[0]}) — image_path/roi_box ignored.")
        image = roi  # used below for depth-field OCR; see docstring note
        x1, y1, x2, y2 = 0, 0, roi.shape[1], roi.shape[0]
        roi_debug = roi.copy()
    else:
        if image_path is None or roi_box is None:
            raise ValueError("Must supply either roi_path, or both image_path and roi_box.")
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Image not found: {image_path}")

        x1, y1, x2, y2 = roi_box
        h_img, w_img = image.shape[:2]
        x1, y1 = max(0, int(x1)), max(0, int(y1))
        x2, y2 = min(w_img, int(x2)), min(h_img, int(y2))
        if x2 <= x1 or y2 <= y1:
            raise ValueError(f"roi_box {roi_box} is invalid/empty after clamping to image bounds {w_img}x{h_img}")
        roi = image[y1:y2, x1:x2]
        print(f"[roi] using supplied roi_box=({x1},{y1},{x2},{y2})")
        roi_debug = image.copy()

    cv2.rectangle(
        roi_debug,
        (x1, y1),
        (x2, y2),
        (0, 255, 0),
        3
    )

    _show("ROI DETECTION", roi_debug)
    cv2.waitKey(1000)

    # =====================================================
    # PANEL SPLIT VISUALIZATION
    # =====================================================
    panels, was_split = split_panels(roi)
    panels = [remove_overlays(p) for p in panels]   # inpaint caliper overlays

    if was_split:
        _show("LEFT PANEL", panels[0])
        cv2.waitKey(1000)

        _show("RIGHT PANEL", panels[1])
        cv2.waitKey(1000)
    else:
        _show("SINGLE PANEL", panels[0])
        cv2.waitKey(1000)

    # =====================================================
    # PIXEL SPACING
    # =====================================================
    panel_height_px = panels[0].shape[0]
    if manual_calibrate:
        mm_per_px = manual_scale_calibration(image, known_distance_cm=calibration_distance_cm)
    else:
        mm_per_px = get_pixel_spacing_auto(image, panel_height_px, depth_cm=depth_cm)

    # =====================================================
    # SEGMENTATION
    # =====================================================
    labels = ["LEFT", "RIGHT"] if was_split else ["SINGLE"]

    contours_by_label = {}

    for panel, label in zip(panels, labels):

        right_margin = 0.15 if (
            was_split and label == "RIGHT"
        ) else 0.05

        cnt = segment_bladder_full(
            panel,
            label,
            right_margin_frac=right_margin
        )

        if cnt is None:
            continue

        contours_by_label[label] = cnt

    if len(contours_by_label) == 0:
        raise ValueError(
            "Bladder segmentation failed on all panels"
        )

    # =====================================================
    # DETERMINE WHICH PANEL PROVIDES d1/d2 AND d3
    # =====================================================
    if diagonal_panel == "RIGHT":

        orthogonal_label = "LEFT"
        diagonal_label = "RIGHT"

    else:

        orthogonal_label = "RIGHT"
        diagonal_label = "LEFT"

    if orthogonal_label not in contours_by_label:
        raise ValueError(
            f"Missing orthogonal panel: {orthogonal_label}"
        )

    if diagonal_label not in contours_by_label:
        raise ValueError(
            f"Missing diagonal panel: {diagonal_label}"
        )

    # =====================================================
    # DIMENSIONS (ONLY CALCULATED ONCE)
    # =====================================================
    d1, d2, d3, pairs = extract_dimensions(
        contours_by_label[orthogonal_label],
        contours_by_label[diagonal_label],
        mm_per_px
    )

    d1_pair, d2_pair, d3_pair = pairs

    # =====================================================
    # VISUALIZE d1 d2 d3
    # =====================================================
    show_d1_d2_d3(
        panels[labels.index(orthogonal_label)],
        panels[labels.index(diagonal_label)],
        contours_by_label[orthogonal_label],
        contours_by_label[diagonal_label],
        d1_pair,
        d2_pair,
        d3_pair
    )

    # =====================================================
    # FINAL VOLUME
    # =====================================================
    volume = compute_volume(d1, d2, d3)

    print(f"\nEstimated Volume: {volume:.2f} ml")

    return volume, (d1, d2, d3)

def _choose_calibration():
    """Ask the user, in the terminal, how to calibrate the pixel->mm scale.
    Returns kwargs to pass straight into process_image()."""
    print("\n=== Calibration mode ===")
    print("  [A] Auto  (detect the depth-scale ruler automatically)  [default]")
    print("  [M] Manual click (click the two ruler ticks yourself, e.g. 0 and 10 cm)")
    print("  [D] Depth value (type the scan depth in cm directly)")
    choice = input("Choose A / M / D  (Enter = A): ").strip().lower()

    if choice == "m":
        d = input("  Known distance between the two ticks you'll click, in cm "
                  "(Enter = 10): ").strip()
        try:
            dist = float(d) if d else 10.0
        except ValueError:
            dist = 10.0
        print(f"  -> Manual click calibration ({dist} cm between the two ticks).")
        print("     A window will open: click the two ruler ticks, then press any key.")
        return dict(manual_calibrate=True, calibration_distance_cm=dist)

    if choice == "d":
        d = input("  Enter the scan depth in cm (e.g. 10 or 12): ").strip()
        try:
            depth = float(d)
            print(f"  -> Using manual depth = {depth} cm.")
            return dict(depth_cm=depth)
        except ValueError:
            print("  (not a number — falling back to Auto.)")
            return dict()

    print("  -> Auto calibration.")
    return dict()


if __name__ == "__main__":
    # Two ways to call this:
    #   python bladder_volume_v4.py roi.png                      <- pre-cropped ROI, feed directly
    #   python bladder_volume_v4.py roi.png RIGHT                <- same, with diagonal_panel override
    #   python bladder_volume_v4.py full.png x1 y1 x2 y2 [RIGHT] <- full image + crop box
    #
    # Calibration: by default you are ASKED (Auto / Manual-click / Depth).
    # To skip the prompt, append a flag:  --manual  |  --auto  |  --depth=12
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python bladder_volume_v4.py roi.png [diagonal_panel] [--manual|--auto|--depth=CM]")
        print("  python bladder_volume_v4.py full_image.png x1 y1 x2 y2 [diagonal_panel] [--manual|--auto|--depth=CM]")
        sys.exit(1)

    # pull out any calibration flag so it doesn't disturb positional parsing
    cal_kwargs = None
    args = []
    for a in sys.argv[1:]:
        al = a.lower()
        if al == "--manual":
            cal_kwargs = dict(manual_calibrate=True, calibration_distance_cm=10.0)
        elif al == "--auto":
            cal_kwargs = dict()
        elif al.startswith("--depth="):
            try:
                cal_kwargs = dict(depth_cm=float(al.split("=", 1)[1]))
            except ValueError:
                pass
        else:
            args.append(a)

    # if no flag was given, ask interactively
    if cal_kwargs is None:
        cal_kwargs = _choose_calibration()

    if len(args) >= 5:
        image_path = args[0]
        roi_box = tuple(int(v) for v in args[1:5])
        diagonal_panel = args[5] if len(args) > 5 else "RIGHT"
        process_image(image_path=image_path, roi_box=roi_box,
                      diagonal_panel=diagonal_panel, **cal_kwargs)
    else:
        roi_path = args[0]
        diagonal_panel = args[1] if len(args) > 1 else "RIGHT"
        process_image(roi_path=roi_path, diagonal_panel=diagonal_panel, **cal_kwargs)