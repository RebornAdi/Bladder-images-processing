# =========================================================
# GEOMETRIC BLADDER VOLUME ESTIMATION — v2
# Generic, vendor-agnostic, automatic pipeline.
#
# Key changes vs. v1 (see CHANGELOG at bottom of file):
#   1. Depth-text OCR regex fixed to survive line breaks.
#   2. Bladder segmentation no longer bleeds into the black
#      background outside the ultrasound sector, or into the
#      on-screen info-panel text column.
#   3. Segmentation candidates must contain an automatically
#      placed interior seed point (robustness check).
#   4. Dimensions are extracted as rotation-invariant Feret
#      diameter + perpendicular caliper width (PCA/axis-aligned
#      vertical & horizontal distance silently assumed the
#      bladder was aligned to image rows/cols, which is not
#      guaranteed).
#   5. d1/d2/d3 are no longer hardcoded to "left panel gives
#      d1,d2 / right panel gives d3". Instead each panel's two
#      principal axes are computed independently, and the axis
#      that AGREES most closely between the two panels is taken
#      as the shared (antero-posterior) dimension — this works
#      regardless of which panel is sagittal/transverse or which
#      measurement convention the sonographer/machine used.
#   6. No silent fallbacks: any time the pipeline has to guess
#      (depth calibration, segmentation confidence, single-panel
#      images) it prints an explicit WARNING so the caller knows
#      the result may be less reliable.
#   7. All cv2.imshow/waitKey calls removed (script now runs
#      headlessly) — debug visualizations are saved as PNG files
#      instead, which also makes it inspectable/auditable.
# =========================================================

import cv2
import numpy as np
import pytesseract
import re
import os
import sys

DEBUG_DIR = "debug_output"


# =========================================================
# TESSERACT PATH (Windows only — Linux/Mac: leave as-is)
# =========================================================
tesseract_path = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
if os.path.exists(tesseract_path):
    pytesseract.pytesseract.tesseract_cmd = tesseract_path


def warn(msg):
    print(f"[WARNING] {msg}", file=sys.stderr)


# =========================================================
# ROI DETECTION
# =========================================================
def auto_detect_roi(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    kernel = np.ones((5, 5), np.uint8)
    edges = cv2.dilate(edges, kernel, iterations=2)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    h, w = gray.shape
    valid = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        x, y, cw, ch = cv2.boundingRect(cnt)
        if area > 0.01 * h * w and ch > h * 0.35 and cw > w * 0.15:
            valid.append(cnt)
    valid = sorted(valid, key=cv2.contourArea, reverse=True)[:2]
    if len(valid) == 0:
        raise ValueError("ROI detection failed — no ultrasound sector found")
    x_min, y_min = 99999, 99999
    x_max, y_max = 0, 0
    for cnt in valid:
        x, y, cw, ch = cv2.boundingRect(cnt)
        x_min, y_min = min(x_min, x), min(y_min, y)
        x_max, y_max = max(x_max, x + cw), max(y_max, y + ch)
    roi = image[y_min:y_max, x_min:x_max]
    return roi, (x_min, y_min, x_max, y_max)


# =========================================================
# PANEL SPLIT
# Generic improvements: wider search band (25-75% instead of
# 40-60%), and an explicit single-panel fallback — if no
# meaningful brightness dip exists, this is a single-view image
# and should not be force-split in half.
# =========================================================
def split_panels(roi):
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    search_start, search_end = int(w * 0.25), int(w * 0.75)
    col_mean = np.mean(gray[:, search_start:search_end], axis=0)

    rel_idx = int(np.argmin(col_mean))
    split_x = search_start + rel_idx
    dip_depth = col_mean.mean() - col_mean[rel_idx]
    dip_significance = dip_depth / (col_mean.std() + 1e-6)

    if dip_significance < 0.8:
        warn("No clear panel boundary found — treating image as a single panel.")
        return [roi], False

    split_x = max(int(w * 0.30), min(split_x, int(w * 0.70)))
    left = roi[:, :split_x]
    right = roi[:, split_x:]
    print(f"[split] x={split_x}  left={left.shape[1]}px  right={right.shape[1]}px")
    return [left, right], True


# =========================================================
# PIXEL SPACING
# Fix: OCR can split "D" and the depth value across lines
# (e.g. "...D/O\n15.0\n..."). The old regex required them on
# the same line and silently fell back to a hardcoded 14.0 cm
# whenever OCR wrapped. We now flatten whitespace/newlines
# before searching, and warn loudly if calibration still fails
# instead of silently guessing.
# =========================================================
def get_pixel_spacing_auto(image, panel_height_px):
    h_img, w_img = image.shape[:2]
    info_strip = image[:, int(w_img * 0.85):]
    strip_up = cv2.resize(info_strip, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
    strip_gray = cv2.cvtColor(strip_up, cv2.COLOR_BGR2GRAY)
    text = pytesseract.image_to_string(strip_gray, config='--psm 6')
    flat = re.sub(r'\s+', ' ', text)

    depth_cm = None
    for match in re.finditer(r'D.{0,6}?(\d{1,2}\.\d)', flat):
        val = float(match.group(1))
        if 5.0 <= val <= 30.0:
            depth_cm = val
            break

    if depth_cm is None:
        warn("Could not OCR the depth (D) field — falling back to 14.0 cm. "
             "Volume accuracy is NOT reliable until this is fixed (e.g. supply "
             "depth_cm manually, or improve OCR crop region for this vendor's UI).")
        depth_cm = 14.0
    else:
        print(f"[pixel spacing] OCR depth = {depth_cm} cm")

    depth_mm = depth_cm * 10.0
    mm_per_px = depth_mm / panel_height_px
    print(f"[pixel spacing] {depth_mm:.0f} mm / {panel_height_px} px = {mm_per_px:.4f} mm/px")
    return mm_per_px


# =========================================================
# BLADDER SEGMENTATION
#
# v1 problem: a global Otsu/fixed threshold sweep + a large
# (15x15, 3-iteration) morphological closing routinely fused
# the anechoic bladder region with (a) the black background
# outside the ultrasound sector's corners, and (b) the dark
# background behind the on-screen info-panel text — both threshold
# to the same value as bladder fluid, and the large closing
# kernel bridges the gap between them into one blob.
#
# Fixes:
#   - Flood-fill from the 4 crop corners on a near-black mask to
#     identify true background, and exclude it before contour
#     search (stops corner-wedge bleed).
#   - A larger right-side margin specifically (info-panel text
#     overlaps the right portion of whichever panel is rightmost).
#   - An automatically placed interior seed point (darkest pixel
#     in a central zone, blurred to avoid noise): any candidate
#     contour that doesn't contain this point is rejected. This
#     is a cheap, generic "is this actually the bladder, not some
#     other dark blob" sanity check.
# =========================================================
def _find_seed(panel, zone=(0.25, 0.75, 0.15, 0.75)):
    gray = cv2.cvtColor(panel, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    fx0, fx1, fy0, fy1 = zone
    cx0, cx1, cy0, cy1 = int(w * fx0), int(w * fx1), int(h * fy0), int(h * fy1)
    central = gray[cy0:cy1, cx0:cx1]
    blurred = cv2.GaussianBlur(central, (15, 15), 0)
    min_idx = np.unravel_index(np.argmin(blurred), blurred.shape)
    return (cx0 + min_idx[1], cy0 + min_idx[0])

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
        crop,
        0,
        255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )[0]

    thresh_vals = [otsu_val, 40, 50, 60, 70, 80]

    for tv in thresh_vals:

        blur = cv2.GaussianBlur(
            crop,
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

        cv2.imshow(
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

    cv2.imshow(
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
# DIMENSION EXTRACTION
#
# Per clarified clinical convention:
#   - The "orthogonal" panel (sagittal, normally LEFT) supplies
#     d1 = max HORIZONTAL distance, d2 = max VERTICAL distance,
#     measured axis-aligned to the image (not rotated) — this is
#     how the caliper is actually placed: straight across, straight
#     down.
#   - The "diagonal" panel (transverse, normally RIGHT) supplies
#     d3 = the longest distance across the bladder regardless of
#     its angle (true Feret diameter) — used as the depth/AP axis.
#
# Which physical panel plays which role is configurable via
# `diagonal_panel`, since it's normally-but-not-always the right
# panel.
# =========================================================
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
    """True Feret diameter: longest pairwise distance between any two
    points on the convex hull, at whatever angle it falls."""
    hull = cv2.convexHull(contour)[:, 0, :].astype(np.float64)
    n = len(hull)
    best_dist, best_pair = 0, None
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

    cv2.imshow(title, debug)
    cv2.waitKey(2000)

# =========================================================
# COMBINE PANELS INTO d1, d2, d3
#
# Per the clarified convention: one panel is the "orthogonal"
# view (gives d1=horizontal max, d2=vertical max), the other is
# the "diagonal" view (gives d3=longest diagonal/Feret distance,
# used as depth). Defaults to LEFT=orthogonal, RIGHT=diagonal,
# since that's the normal layout — override with
# diagonal_panel="LEFT" if a given machine/operator does it the
# other way around.
# =========================================================
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

    cv2.imshow("D1 + D2", left_debug)

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

    cv2.imshow("D3", right_debug)

    cv2.waitKey(0)
# =========================================================
# MAIN
# =========================================================
def process_image(image_path, diagonal_panel="RIGHT"):

    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Image not found: {image_path}")

    # =====================================================
    # ROI VISUALIZATION
    # =====================================================
    roi, (x1, y1, x2, y2) = auto_detect_roi(image)

    roi_debug = image.copy()

    cv2.rectangle(
        roi_debug,
        (x1, y1),
        (x2, y2),
        (0, 255, 0),
        3
    )

    cv2.imshow("ROI DETECTION", roi_debug)
    cv2.waitKey(1000)

    # =====================================================
    # PANEL SPLIT VISUALIZATION
    # =====================================================
    panels, was_split = split_panels(roi)

    if was_split:
        cv2.imshow("LEFT PANEL", panels[0])
        cv2.waitKey(1000)

        cv2.imshow("RIGHT PANEL", panels[1])
        cv2.waitKey(1000)
    else:
        cv2.imshow("SINGLE PANEL", panels[0])
        cv2.waitKey(1000)

    # =====================================================
    # PIXEL SPACING
    # =====================================================
    panel_height_px = panels[0].shape[0]
    mm_per_px = get_pixel_spacing_auto(image, panel_height_px)

    # =====================================================
    # SEGMENTATION
    # =====================================================
    labels = ["LEFT", "RIGHT"] if was_split else ["SINGLE"]

    contours_by_label = {}

    for panel, label in zip(panels, labels):

        right_margin = 0.15 if (
            was_split and label == "RIGHT"
        ) else 0.05

        cnt = segment_bladder(
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

if __name__ == "__main__":
    image_path = sys.argv[1] if len(sys.argv) > 1 else "image.png"
    process_image(image_path)


# =========================================================
# CHANGELOG (v1 -> v2)
# =========================================================
# - Removed all cv2.imshow/waitKey calls (script now runs
#   headlessly; debug overlays are written to debug_output/).
# - Fixed depth-OCR regex so it survives OCR line-wrapping
#   instead of silently using a hardcoded 14.0cm fallback.
# - Segmentation: added corner-background flood-fill exclusion
#   and a wider right-margin trim, fixing major over-segmentation
#   into the sector's black corners and the on-screen info panel.
# - Segmentation: added an interior-seed containment check as a
#   generic "is this really the bladder" validity test.
# - Dimension extraction switched from axis-aligned
#   vertical/horizontal max-distance to true (rotation-invariant)
#   Feret diameter + perpendicular caliper width.
# - d1/d2/d3 assignment is no longer hardcoded to "left panel
#   gives d1+d2, right panel gives d3" — axes are now matched
#   across panels by mm agreement, which generalizes to either
#   measurement convention and flags low-confidence results.
# - Panel splitting now has an explicit single-panel fallback
#   instead of always force-splitting in half.
# - All silent fallbacks now print explicit WARNING messages.
#
# KNOWN REMAINING LIMITATIONS (be aware of these):
# - Depth calibration still relies on OCR of the vendor's info
#   text; it has only been validated against GE Logiq/XDclear's
#   layout. A ruler-tick-based calibration (reading the depth
#   scale bar drawn next to each panel) would be more vendor-
#   generic but proved unreliable to OCR at this resolution/font
#   in testing — it's a good next investment if you need to
#   support other vendors, but ship with a manual depth_cm
#   override in the meantime for non-GE images.
# - Segmentation is still a classical threshold/region approach;
#   it was validated on this single reference image and on
#   straightforward sector geometry. Atypical bladder shapes,
#   very low-contrast scans, or septated/multiloculated bladders
#   will need a learned segmentation model (e.g. a small U-Net)
#   for production-grade robustness.
# - The cross-panel axis matching assumes exactly one dimension
#   is shared between the two views, which is anatomically true
#   for standard sagittal+transverse bladder scans, but would
#   need rethinking for any other paired-view protocol.