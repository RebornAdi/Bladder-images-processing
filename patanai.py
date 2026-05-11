import cv2
import numpy as np
import matplotlib.pyplot as plt

# =========================================================
# BLADDER VOLUME ESTIMATION SYSTEM
# ROI + LEFT/RIGHT DETECTION FROM YOUR CODE
# BLADDER SEGMENTATION FROM IMPROVED PIPELINE
# =========================================================

# ---------------------------------------------------------
# UTILITY
# ---------------------------------------------------------
def distance(p1, p2):
    return np.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)

# ---------------------------------------------------------
# ROI DETECTION  (FROM YOUR WORKING CODE)
# ---------------------------------------------------------
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

        if area > 20000 and ch > h*0.4 and cw > w*0.2:
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

# ---------------------------------------------------------
# MANUAL SCALE CALIBRATION
# ---------------------------------------------------------
points = []
img_display = None

def click_event(event, x, y, flags, param):

    global points, img_display

    if event == cv2.EVENT_LBUTTONDOWN:

        points.append((x, y))

        cv2.circle(
            img_display,
            (x, y),
            5,
            (0,0,255),
            -1
        )

        if len(points) % 2 == 0:

            cv2.line(
                img_display,
                points[-2],
                points[-1],
                (0,255,0),
                2
            )

        cv2.imshow("Calibration", img_display)

def get_pixel_spacing_manual(image):

    global points, img_display

    print("\nCLICK 2 POINTS ON SCALE BAR")

    points = []

    img_display = image.copy()

    cv2.imshow("Calibration", img_display)

    cv2.setMouseCallback(
        "Calibration",
        click_event
    )

    cv2.waitKey(0)

    cv2.destroyAllWindows()

    if len(points) < 2:
        return 0.1

    p1, p2 = points[:2]

    pixel_dist = distance(p1, p2)

    if pixel_dist == 0:
        return 0.1

    # 10 cm scale
    pixel_spacing = 100 / pixel_dist

    print("Pixel spacing:", pixel_spacing)

    return pixel_spacing

# ---------------------------------------------------------
# BLADDER SEGMENTATION (IMPROVED)
# ---------------------------------------------------------
def segment_bladder(image, label=""):

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    blur = cv2.GaussianBlur(gray, (7,7), 0)

    # DARK REGION SEGMENTATION
    _, thresh = cv2.threshold(
        blur,
        35.5,
        255,
        cv2.THRESH_BINARY_INV
    )

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))

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
        return None, None

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

        cx = x + w//2
        cy = y + h//2

        dist = np.sqrt(
            (cx - cx_img)**2 +
            (cy - cy_img)**2
        )

        # prioritize large central dark object
        score = dist - 0.001 * area

        if score < best_score and len(cnt) >= 5:

            best_score = score
            best_cnt = cnt

    if best_cnt is None:
        return None, None

    ellipse = cv2.fitEllipse(best_cnt)

    debug = image.copy()

    cv2.drawContours(
        debug,
        [best_cnt],
        -1,
        (255,0,0),
        2
    )

    cv2.ellipse(
        debug,
        ellipse,
        (0,255,0),
        2
    )

    cv2.imshow(f"Bladder {label}", debug)

    cv2.waitKey(1500)

    return best_cnt, ellipse

# ---------------------------------------------------------
# EXTRACT DIMENSIONS
# ---------------------------------------------------------
def get_dimensions(ellipse):

    (_, _), (major, minor), _ = ellipse

    L = max(major, minor)
    W = min(major, minor)

    return L, W

# ---------------------------------------------------------
# MAIN PIPELINE
# ---------------------------------------------------------
def process_image(image):

    # ROI DETECTION
    roi, (x1, y1, x2, y2) = auto_detect_roi(image)

    vis = image.copy()

    cv2.rectangle(
        vis,
        (x1, y1),
        (x2, y2),
        (0,255,0),
        2
    )

    # -------------------------------------------------
    # LEFT / RIGHT SPLIT
    # (FROM YOUR WORKING CODE)
    # -------------------------------------------------
    h, w = roi.shape[:2]

    left = roi[:, :w//2]
    right = roi[:, w//2:]

    cv2.imshow("Left Scan", left)
    cv2.waitKey(1000)

    cv2.imshow("Right Scan", right)
    cv2.waitKey(1000)

    # -------------------------------------------------
    # BLADDER SEGMENTATION
    # -------------------------------------------------
    left_cnt, left_ellipse = segment_bladder(
        left,
        "LEFT"
    )

    right_cnt, right_ellipse = segment_bladder(
        right,
        "RIGHT"
    )

    if left_ellipse is None or right_ellipse is None:
        raise ValueError("Bladder segmentation failed")

    # LEFT IMAGE
    L_px, W_px = get_dimensions(left_ellipse)

    # RIGHT IMAGE
    H_px, _ = get_dimensions(right_ellipse)

    return L_px, W_px, H_px, vis

# ---------------------------------------------------------
# VOLUME COMPUTATION
# ---------------------------------------------------------
def compute_volume(
    L_px,
    W_px,
    H_px,
    pixel_spacing
):

    print("\nPixel Measurements")
    print("L_px:", L_px)
    print("W_px:", W_px)
    print("H_px:", H_px)

    # px -> cm
    L = (L_px * pixel_spacing) / 10
    W = (W_px * pixel_spacing) / 10
    H = (H_px * pixel_spacing) / 10

    print("\nReal Dimensions")
    print("L:", L, "cm")
    print("W:", W, "cm")
    print("H:", H, "cm")

    volume = 0.52 * L * W * H

    return volume

# ---------------------------------------------------------
# RUN
# ---------------------------------------------------------
if __name__ == "__main__":

    image_path = "img012.jpg"

    image = cv2.imread(image_path)

    if image is None:
        raise ValueError("Image not found")

    # SCALE
    pixel_spacing = get_pixel_spacing_manual(image)

    # PROCESS
    L_px, W_px, H_px, vis = process_image(image)

    # VOLUME
    volume = compute_volume(
        L_px,
        W_px,
        H_px,
        pixel_spacing
    )

    cv2.imshow("Detected ROI", vis)

    print("\nEstimated Volume:", volume, "ml")

    cv2.waitKey(0)

    cv2.destroyAllWindows()