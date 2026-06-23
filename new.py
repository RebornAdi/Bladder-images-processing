# =========================================================
# GEOMETRIC BLADDER VOLUME ESTIMATION
# Fully Automatic — tested against GE Logiq / XDclear images
# =========================================================

import cv2
import numpy as np
import pytesseract
import re
import os

# =========================================================
# TESSERACT PATH (Windows only — Linux/Mac: leave as-is)
# =========================================================
tesseract_path = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
if os.path.exists(tesseract_path):
    pytesseract.pytesseract.tesseract_cmd = tesseract_path


# =========================================================
# ROI DETECTION  (unchanged from original)
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
        raise ValueError("ROI detection failed")
    x_min, y_min = 99999, 99999
    x_max, y_max = 0, 0
    for cnt in valid:
        x, y, cw, ch = cv2.boundingRect(cnt)
        x_min = min(x_min, x)
        y_min = min(y_min, y)
        x_max = max(x_max, x + cw)
        y_max = max(y_max, y + ch)
    roi = image[y_min:y_max, x_min:x_max]
    return roi, (x_min, y_min, x_max, y_max)


# =========================================================
# PANEL SPLIT
# FIX: original w//2 assumes equal panel widths — not always true.
# Search for the darkest column only in the central 40–60 % band.
# =========================================================
def split_panels(roi):
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    search_start = int(w * 0.40)
    search_end   = int(w * 0.60)
    col_sum  = np.sum(gray[:, search_start:search_end], axis=0)
    rel_idx  = int(np.argmin(col_sum))
    split_x  = search_start + rel_idx
    split_x  = max(int(w * 0.30), min(split_x, int(w * 0.70)))  # safety clamp
    left  = roi[:, :split_x]
    right = roi[:, split_x:]
    print(f"[split] x={split_x}  left={left.shape[1]}px  right={right.shape[1]}px")
    return left, right


# =========================================================
# PIXEL SPACING
# FIX: original strip-based tick detection pointed at the wrong
# columns (inside the scan image, not the ruler) and assumed 10 mm
# tick pitch which is wrong for GE mixed 5/10 mm rulers.
#
# NEW approach:
#   Primary  — OCR the "D  14.0" depth label from the right-side
#               info panel, then divide by the panel height in pixels.
#   Fallback — assume 14.0 cm (standard abdominal preset on GE).
# =========================================================
def get_pixel_spacing_auto(image, panel_height_px):
    h_img, w_img = image.shape[:2]

    # The right-side info strip contains "D  14.0"
    info_strip = image[:, int(w_img * 0.85):]
    strip_up   = cv2.resize(info_strip, None, fx=2.0, fy=2.0,
                            interpolation=cv2.INTER_CUBIC)
    strip_gray = cv2.cvtColor(strip_up, cv2.COLOR_BGR2GRAY)

    text = pytesseract.image_to_string(strip_gray, config='--psm 6')

    depth_cm = None
    for match in re.finditer(r'D\s+(\d+\.\d)', text):
        val = float(match.group(1))
        if 5.0 <= val <= 30.0:           # sanity: 5–30 cm is realistic
            depth_cm = val
            break

    if depth_cm is None:
        print("[pixel spacing] OCR failed — using fallback 14.0 cm")
        depth_cm = 14.0
    else:
        print(f"[pixel spacing] OCR depth = {depth_cm} cm")

    depth_mm  = depth_cm * 10.0
    mm_per_px = depth_mm / panel_height_px
    print(f"[pixel spacing] {depth_mm:.0f} mm / {panel_height_px} px = {mm_per_px:.4f} mm/px")
    return mm_per_px


# =========================================================
# BLADDER SEGMENTATION
# FIX: original threshold of 36.5 is too low — only captures the
# deepest blacks and produces a broken contour.
#
# NEW approach:
#   - Try Otsu first (adaptive to image brightness/gain).
#   - Also sweep fixed thresholds 40–80 as fallback.
#   - Filter by size (4–55 % of panel), solidity ≥ 0.55,
#     and reward contours near the panel centre.
# =========================================================
def segment_bladder(panel, label=""):
    gray  = cv2.cvtColor(panel, cv2.COLOR_BGR2GRAY)
    h_img, w_img = gray.shape

    # Ignore outermost 8 % top/bottom and 5 % sides (skin / noise)
    margin_y = int(h_img * 0.08)
    margin_x = int(w_img * 0.05)
    crop = gray[margin_y: h_img - margin_y, margin_x: w_img - margin_x]
    ch, cw = crop.shape

    best_cnt   = None
    best_score = 0

    otsu_val = cv2.threshold(crop, 0, 255,
                              cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[0]
    thresh_vals = [otsu_val, 40, 50, 60, 70, 80]

    for tv in thresh_vals:
        blur = cv2.GaussianBlur(crop, (21, 21), 0)
        _, thresh = cv2.threshold(blur, tv, 255, cv2.THRESH_BINARY_INV)

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=3)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN,  kernel, iterations=2)

        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL,
                                        cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if not (0.04 * ch * cw < area < 0.55 * ch * cw):
                continue
            if len(cnt) < 5:
                continue
            hull_area = cv2.contourArea(cv2.convexHull(cnt))
            solidity  = area / hull_area if hull_area > 0 else 0
            if solidity < 0.55:
                continue
            bx, by, bw, bh = cv2.boundingRect(cnt)
            aspect     = min(bw, bh) / max(bw, bh) if max(bw, bh) > 0 else 0
            cx_c       = bx + bw / 2
            cy_c       = by + bh / 2
            norm_dist  = (np.sqrt((cx_c - cw / 2) ** 2 +
                                  (cy_c - ch / 2) ** 2) /
                          np.sqrt(cw ** 2 + ch ** 2))
            score = solidity * aspect * area * (1 - 0.5 * norm_dist)
            if score > best_score:
                best_score = score
                cnt_s = cnt.copy()
                cnt_s[:, :, 0] += margin_x
                cnt_s[:, :, 1] += margin_y
                best_cnt = cnt_s

    if best_cnt is None:
        print(f"[{label}] segmentation failed")
        return None

    debug = panel.copy()
    cv2.drawContours(debug, [best_cnt], -1, (0, 255, 0), 2)
    cv2.imshow(f"Segmented {label}", debug)
    cv2.waitKey(1000)
    return best_cnt


# =========================================================
# d1 = MAX VERTICAL DISTANCE  (unchanged)
# =========================================================
def max_vertical_distance(contour):
    pts    = contour[:, 0, :]
    x_vals = pts[:, 0]
    y_vals = pts[:, 1]
    best_dist, best_pair = 0, None
    for x in np.unique(x_vals):
        ys = y_vals[x_vals == x]
        if len(ys) < 2:
            continue
        dist = np.max(ys) - np.min(ys)
        if dist > best_dist:
            best_dist = dist
            best_pair = ((x, int(np.min(ys))), (x, int(np.max(ys))))
    return best_dist, best_pair


# =========================================================
# d2 = MAX HORIZONTAL DISTANCE  (unchanged)
# =========================================================
def max_horizontal_distance(contour):
    pts    = contour[:, 0, :]
    x_vals = pts[:, 0]
    y_vals = pts[:, 1]
    best_dist, best_pair = 0, None
    for y in np.unique(y_vals):
        xs = x_vals[y_vals == y]
        if len(xs) < 2:
            continue
        dist = np.max(xs) - np.min(xs)
        if dist > best_dist:
            best_dist = dist
            best_pair = ((int(np.min(xs)), y), (int(np.max(xs)), y))
    return best_dist, best_pair


# =========================================================
# d3 = ELLIPSE MAJOR AXIS on the right panel
# FIX: original max_diameter on convex hull extends to panel edges
# when the bladder contour touches the boundary, giving a huge
# over-estimate.  fitEllipse is robust to boundary touching.
# =========================================================
def ellipse_major_axis(contour):
    if len(contour) < 5:
        return 0, None
    ellipse = cv2.fitEllipse(contour)
    (cx, cy), (MA, ma), angle = ellipse
    major = max(MA, ma)
    angle_rad = np.deg2rad(angle)
    ux = np.sin(angle_rad) if MA >= ma else np.cos(angle_rad)
    uy = -np.cos(angle_rad) if MA >= ma else np.sin(angle_rad)
    p1 = (int(cx + (major / 2) * ux), int(cy + (major / 2) * uy))
    p2 = (int(cx - (major / 2) * ux), int(cy - (major / 2) * uy))
    return major, (p1, p2)


# =========================================================
# MEASUREMENTS
# =========================================================
def extract_measurements(left, right, left_cnt, right_cnt):
    d1_px, d1_pair = max_vertical_distance(left_cnt)
    d2_px, d2_pair = max_horizontal_distance(left_cnt)
    d3_px, d3_pair = ellipse_major_axis(right_cnt)

    debug_left = left.copy()
    cv2.drawContours(debug_left, [left_cnt], -1, (0, 255, 0), 2)
    if d1_pair:
        cv2.line(debug_left, d1_pair[0], d1_pair[1], (255, 0, 0), 2)
    if d2_pair:
        cv2.line(debug_left, d2_pair[0], d2_pair[1], (0, 0, 255), 2)

    debug_right = right.copy()
    cv2.drawContours(debug_right, [right_cnt], -1, (0, 255, 0), 2)
    if d3_pair:
        cv2.line(debug_right, d3_pair[0], d3_pair[1], (255, 255, 0), 2)

    cv2.imshow("Left Measurements",  debug_left)
    cv2.imshow("Right Measurements", debug_right)
    cv2.waitKey(1000)

    print("\n[PIXEL DIMENSIONS]")
    print(f"  d1 (Vertical):       {d1_px:.1f} px")
    print(f"  d2 (Horizontal):     {d2_px:.1f} px")
    print(f"  d3 (Ellipse major):  {d3_px:.1f} px")

    return d1_px, d2_px, d3_px


# =========================================================
# PROCESS IMAGE
# =========================================================
def process_image(image):
    roi, (x1, y1, x2, y2) = auto_detect_roi(image)

    vis = image.copy()
    cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.imshow("ROI", vis)
    cv2.waitKey(800)

    left, right = split_panels(roi)

    cv2.imshow("Left Scan",  left)
    cv2.waitKey(600)
    cv2.imshow("Right Scan", right)
    cv2.waitKey(600)

    left_cnt  = segment_bladder(left,  "LEFT")
    right_cnt = segment_bladder(right, "RIGHT")

    if left_cnt is None or right_cnt is None:
        raise ValueError("Bladder segmentation failed")

    d1_px, d2_px, d3_px = extract_measurements(left, right, left_cnt, right_cnt)

    panel_height_px = left.shape[0]
    return d1_px, d2_px, d3_px, panel_height_px


# =========================================================
# VOLUME  (unchanged)
# =========================================================
def compute_volume(d1_px, d2_px, d3_px, pixel_spacing):
    d1_mm = d1_px * pixel_spacing
    d2_mm = d2_px * pixel_spacing
    d3_mm = d3_px * pixel_spacing
    d1_cm = d1_mm / 10
    d2_cm = d2_mm / 10
    d3_cm = d3_mm / 10
    print("\n[REAL DIMENSIONS]")
    print(f"  d1 = {d1_cm:.2f} cm  ({d1_mm:.1f} mm)")
    print(f"  d2 = {d2_cm:.2f} cm  ({d2_mm:.1f} mm)")
    print(f"  d3 = {d3_cm:.2f} cm  ({d3_mm:.1f} mm)")
    volume = 0.52 * d1_cm * d2_cm * d3_cm
    return volume


# =========================================================
# MAIN
# =========================================================
if __name__ == "__main__":
    image_path = r"image.png"

    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Image not found: {image_path}")

    d1_px, d2_px, d3_px, panel_height_px = process_image(image)

    pixel_spacing = get_pixel_spacing_auto(image, panel_height_px)

    volume = compute_volume(d1_px, d2_px, d3_px, pixel_spacing)

    print(f"\nEstimated Volume: {volume:.2f} ml")

    cv2.waitKey(0)
    cv2.destroyAllWindows()