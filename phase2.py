# =========================================================
# GEOMETRIC BLADDER VOLUME ESTIMATION
# Fully Automatic
# No doctor annotations
# No yellow calipers
# Uses contour geometry only
# =========================================================

import cv2
import numpy as np
import os
import pytesseract

# =========================================================
# TESSERACT
# =========================================================

tesseract_path = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

if os.path.exists(tesseract_path):
    pytesseract.pytesseract.tesseract_cmd = tesseract_path

# =========================================================
# ROI DETECTION
# =========================================================

def auto_detect_roi(image):

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    edges = cv2.Canny(gray, 50, 150)

    kernel = np.ones((5,5), np.uint8)

    edges = cv2.dilate(edges, kernel, iterations=2)

    contours, _ = cv2.findContours(
        edges,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    h, w = gray.shape

    valid = []

    for cnt in contours:

        area = cv2.contourArea(cnt)

        x, y, cw, ch = cv2.boundingRect(cnt)

        if area > 20000 and ch > h * 0.4 and cw > w * 0.2:
            valid.append(cnt)

    valid = sorted(
        valid,
        key=cv2.contourArea,
        reverse=True
    )[:2]

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
# AUTOMATIC PIXEL SPACING
# =========================================================

def get_pixel_spacing_auto(image):

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    h, w = gray.shape

    candidate_strips = [
        (int(w * 0.190), int(w * 0.208)),
        (int(w * 0.205), int(w * 0.220)),
        (int(w * 0.215), int(w * 0.230)),
    ]

    best_spacing = None

    for x_start, x_end in candidate_strips:

        strip = gray[:, x_start:x_end]

        row_max = np.max(strip, axis=1)

        threshold = np.percentile(row_max, 85)

        threshold = max(threshold, 160)

        raw_ticks = [
            y for y in range(int(h * 0.1), int(h * 0.85))
            if row_max[y] > threshold
        ]

        if len(raw_ticks) < 4:
            continue

        merged = []

        cluster = [raw_ticks[0]]

        for r in raw_ticks[1:]:

            if r - cluster[-1] <= 8:
                cluster.append(r)

            else:
                merged.append(int(np.mean(cluster)))
                cluster = [r]

        merged.append(int(np.mean(cluster)))

        if len(merged) < 4:
            continue

        gaps = np.diff(merged)

        median_gap = float(np.median(gaps))

        good_gaps = [
            g for g in gaps
            if abs(g - median_gap) < median_gap * 0.25
            and g > 10
        ]

        if len(good_gaps) < 3:
            continue

        avg_gap = float(np.mean(good_gaps))

        spacing = 10.0 / avg_gap

        best_spacing = spacing

        print("\n[AUTO SCALE SUCCESS]")
        print(f"Average Tick Gap: {avg_gap:.2f} px")
        print(f"Pixel Spacing: {spacing:.4f} mm/pixel")

        break

    if best_spacing is None:

        print("[WARN] Auto scaling failed")
        print("[WARN] Using default 0.15 mm/pixel")

        best_spacing = 0.15

    return best_spacing

# =========================================================
# BLADDER SEGMENTATION
# =========================================================

def segment_bladder(image, label=""):

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    blur = cv2.GaussianBlur(gray, (7,7), 0)

    _, thresh = cv2.threshold(
        blur,
        36.5,
        255,
        cv2.THRESH_BINARY_INV
    )

    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (5,5)
    )

    thresh = cv2.morphologyEx(
        thresh,
        cv2.MORPH_CLOSE,
        kernel,
        iterations=2
    )

    contours, _ = cv2.findContours(
        thresh,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    if not contours:
        return None

    h_img, w_img = gray.shape

    cx_img = w_img // 2
    cy_img = h_img // 2

    best_cnt = None
    best_score = 1e9

    for cnt in contours:

        area = cv2.contourArea(cnt)

        if area < 3000:
            continue

        x, y, w, h = cv2.boundingRect(cnt)

        cx = x + w // 2
        cy = y + h // 2

        dist = np.sqrt(
            (cx - cx_img) ** 2 +
            (cy - cy_img) ** 2
        )

        score = dist - 0.001 * area

        if score < best_score:

            best_score = score
            best_cnt = cnt

    if best_cnt is None:
        return None

    debug = image.copy()

    cv2.drawContours(
        debug,
        [best_cnt],
        -1,
        (0,255,0),
        2
    )

    cv2.imshow(f"Segmented {label}", debug)

    cv2.waitKey(1000)

    return best_cnt

# =========================================================
# d1 = MAX VERTICAL DISTANCE
# =========================================================

def max_vertical_distance(contour):

    pts = contour[:, 0, :]

    x_vals = pts[:, 0]
    y_vals = pts[:, 1]

    best_dist = 0

    best_pair = None

    for x in np.unique(x_vals):

        ys = y_vals[x_vals == x]

        if len(ys) < 2:
            continue

        dist = np.max(ys) - np.min(ys)

        if dist > best_dist:

            best_dist = dist

            best_pair = (
                (x, int(np.min(ys))),
                (x, int(np.max(ys)))
            )

    return best_dist, best_pair

# =========================================================
# d2 = MAX HORIZONTAL DISTANCE
# =========================================================

def max_horizontal_distance(contour):

    pts = contour[:, 0, :]

    x_vals = pts[:, 0]
    y_vals = pts[:, 1]

    best_dist = 0

    best_pair = None

    for y in np.unique(y_vals):

        xs = x_vals[y_vals == y]

        if len(xs) < 2:
            continue

        dist = np.max(xs) - np.min(xs)

        if dist > best_dist:

            best_dist = dist

            best_pair = (
                (int(np.min(xs)), y),
                (int(np.max(xs)), y)
            )

    return best_dist, best_pair

# =========================================================
# d3 = TRUE MAXIMUM DIAMETER
# =========================================================

def max_diameter(contour):

    hull = cv2.convexHull(contour)

    pts = hull[:, 0, :]

    best_dist = 0

    best_pair = None

    for i in range(len(pts)):

        for j in range(i + 1, len(pts)):

            p1 = pts[i]
            p2 = pts[j]

            dist = np.linalg.norm(p1 - p2)

            if dist > best_dist:

                best_dist = dist

                best_pair = (
                    tuple(p1),
                    tuple(p2)
                )

    return best_dist, best_pair

# =========================================================
# MEASUREMENTS
# =========================================================

def extract_measurements(left, right, left_cnt, right_cnt):

    # LEFT IMAGE
    d1_px, d1_pair = max_vertical_distance(left_cnt)

    d2_px, d2_pair = max_horizontal_distance(left_cnt)

    # RIGHT IMAGE
    d3_px, d3_pair = max_diameter(right_cnt)

    # -----------------------------------------------------
    # DEBUG DRAWING
    # -----------------------------------------------------

    debug_left = left.copy()

    cv2.drawContours(
        debug_left,
        [left_cnt],
        -1,
        (0,255,0),
        2
    )

    cv2.line(
        debug_left,
        d1_pair[0],
        d1_pair[1],
        (255,0,0),
        2
    )

    cv2.line(
        debug_left,
        d2_pair[0],
        d2_pair[1],
        (0,0,255),
        2
    )

    debug_right = right.copy()

    cv2.drawContours(
        debug_right,
        [right_cnt],
        -1,
        (0,255,0),
        2
    )

    cv2.line(
        debug_right,
        d3_pair[0],
        d3_pair[1],
        (255,255,0),
        2
    )

    cv2.imshow("Left Measurements", debug_left)
    cv2.imshow("Right Measurements", debug_right)

    cv2.waitKey(1000)

    print("\n[PIXEL DIMENSIONS]")
    print(f"d1 (Vertical):   {d1_px:.2f}")
    print(f"d2 (Horizontal): {d2_px:.2f}")
    print(f"d3 (Diameter):   {d3_px:.2f}")

    return d1_px, d2_px, d3_px

# =========================================================
# PROCESS IMAGE
# =========================================================

def process_image(image):

    roi, (x1, y1, x2, y2) = auto_detect_roi(image)

    h, w = roi.shape[:2]

    left = roi[:, :w // 2]

    right = roi[:, w // 2:]

    cv2.imshow("Left Scan", left)
    cv2.imshow("Right Scan", right)

    cv2.waitKey(1000)

    left_cnt = segment_bladder(left, "LEFT")

    right_cnt = segment_bladder(right, "RIGHT")

    if left_cnt is None or right_cnt is None:
        raise ValueError("Bladder segmentation failed")

    d1_px, d2_px, d3_px = extract_measurements(
        left,
        right,
        left_cnt,
        right_cnt
    )

    return d1_px, d2_px, d3_px

# =========================================================
# VOLUME
# =========================================================

def compute_volume(d1_px, d2_px, d3_px, pixel_spacing):

    d1_mm = d1_px * pixel_spacing
    d2_mm = d2_px * pixel_spacing
    d3_mm = d3_px * pixel_spacing

    d1_cm = d1_mm / 10
    d2_cm = d2_mm / 10
    d3_cm = d3_mm / 10

    print("\n[REAL DIMENSIONS]")
    print(f"d1 = {d1_cm:.2f} cm")
    print(f"d2 = {d2_cm:.2f} cm")
    print(f"d3 = {d3_cm:.2f} cm")

    volume = 0.52 * d1_cm * d2_cm * d3_cm

    return volume

# =========================================================
# MAIN
# =========================================================

if __name__ == "__main__":

    image_path = r"img012.jpg"

    image = cv2.imread(image_path)

    if image is None:
        raise ValueError("Image not found")

    # SCALE
    pixel_spacing = get_pixel_spacing_auto(image)

    # PROCESS
    d1_px, d2_px, d3_px = process_image(image)

    # VOLUME
    volume = compute_volume(
        d1_px,
        d2_px,
        d3_px,
        pixel_spacing
    )

    print(f"\nEstimated Volume: {volume:.2f} ml")

    cv2.waitKey(0)

    cv2.destroyAllWindows()