import cv2
import numpy as np
import json


# ─────────────────────────────────────────────
# 1. LOAD PIXEL SPACING (IMPORTANT)
# ─────────────────────────────────────────────
def load_pixel_spacing():
    return 0.1


# ─────────────────────────────────────────────
# 2. DWT CALCULATION
# ─────────────────────────────────────────────
def compute_dwt(mask, pixel_spacing):
    inner = (mask == 1).astype(np.uint8)
    outer = (mask == 2).astype(np.uint8)

    # Distance transform from inner boundary
    dist_map = cv2.distanceTransform(1 - inner, cv2.DIST_L2, 5)

    # Only measure where outer wall exists
    dwt_pixels = dist_map[outer == 1]

    if len(dwt_pixels) == 0:
        return None

    # Take median (robust to noise)
    dwt_px = np.median(dwt_pixels)

    # Convert to mm
    dwt_mm = dwt_px * pixel_spacing

    return dwt_mm


# ─────────────────────────────────────────────
# 3. IPP CALCULATION
# ─────────────────────────────────────────────
def compute_ipp(mask, pixel_spacing):
    prostate = (mask == 3).astype(np.uint8)
    bladder = (mask == 1).astype(np.uint8)

    # Get contours
    contours, _ = cv2.findContours(prostate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if len(contours) == 0:
        return None

    # Largest prostate contour
    cnt = max(contours, key=cv2.contourArea)

    # Find highest protrusion (min y)
    top_point = tuple(cnt[cnt[:, :, 1].argmin()][0])

    # Find bladder neck (approx bottom of bladder)
    bladder_pts = np.column_stack(np.where(bladder == 1))
    neck_y = np.max(bladder_pts[:, 0])  # lowest point

    # Distance in pixels
    ipp_px = neck_y - top_point[1]

    if ipp_px < 0:
        return None

    ipp_mm = ipp_px * pixel_spacing

    return ipp_mm


# ─────────────────────────────────────────────
# 4. BLADDER VOLUME (OPTIONAL)
# ─────────────────────────────────────────────
def compute_volume(mask, pixel_spacing):
    bladder = (mask == 1).astype(np.uint8)

    # Get contour
    contours, _ = cv2.findContours(bladder, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if len(contours) == 0:
        return None

    cnt = max(contours, key=cv2.contourArea)

    # Fit ellipse
    if len(cnt) < 5:
        return None

    ellipse = cv2.fitEllipse(cnt)

    (x, y), (MA, ma), angle = ellipse

    # Convert to cm
    d1 = MA * pixel_spacing / 10
    d2 = ma * pixel_spacing / 10
    d3 = d2  # assume symmetric

    volume = 0.52 * d1 * d2 * d3  # ellipsoid formula

    return volume


# ─────────────────────────────────────────────
# 5. MASTER FUNCTION
# ─────────────────────────────────────────────
def compute_all(mask_path):
    mask = cv2.imread(mask_path, 0)
    cv2.imshow("mask",mask)
    cv2.waitKey(0)
    print(np.unique(mask))
    pixel_spacing = load_pixel_spacing()

    dwt = compute_dwt(mask, pixel_spacing)
    ipp = compute_ipp(mask, pixel_spacing)
    volume = compute_volume(mask, pixel_spacing)

    return {
        "DWT_mm": dwt,
        "IPP_mm": ipp,
        "Volume_ml": volume
    }