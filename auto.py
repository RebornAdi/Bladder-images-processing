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


def save_calibration_image(image, path=r"D:\Bladder-images-processing\calibration.png"):
    cv2.imwrite(path, image)
    print(f"Calibration image saved to: {path}")
    print("Open this file and manually enter two pixel coordinates for the scale bar.")
    return path


def query_manual_points():
    manual_points = []
    for i in range(2):
        while True:
            raw = input(f"Enter point {i+1} as x,y: ").strip()
            try:
                x_str, y_str = raw.split(",")
                x, y = int(x_str), int(y_str)
                manual_points.append((x, y))
                break
            except Exception:
                print("Invalid entry. Use format x,y like 45,120.")
    return manual_points


def get_pixel_spacing_manual(image, scale_length_mm=100):

    global points, img_display

    print("\nCLICK 2 POINTS ON SCALE BAR")
    print(f"Enter the known scale bar length in millimeters (default {scale_length_mm} mm).")

    user_input = input("Scale length [mm]: ").strip()
    if user_input:
        try:
            scale_length_mm = float(user_input)
        except ValueError:
            print("Invalid scale length input, using default", scale_length_mm, "mm")

    points = []
    img_display = image.copy()

    cv2.namedWindow("Calibration", cv2.WINDOW_NORMAL)
    cv2.imshow("Calibration", img_display)
    cv2.setMouseCallback(
        "Calibration",
        click_event
    )

    print("Click two points on the scale bar. Press ESC to cancel.")

    # Give OpenCV one frame to create the window, then verify visibility.
    if cv2.waitKey(1) & 0xFF == 27:
        pass
    vis = cv2.getWindowProperty("Calibration", cv2.WND_PROP_VISIBLE)
    if vis < 1:
        window_failed = True
    else:
        window_failed = False

    if window_failed:
        cv2.destroyAllWindows()
        save_calibration_image(image)
        points = query_manual_points()
    else:
        while True:
            key = cv2.waitKey(1) & 0xFF
            if len(points) >= 2 or key == 27:
                break
        cv2.destroyAllWindows()

    if len(points) < 2:
        print("Calibration failed or canceled, using default 0.1 mm/pixel.")
        return 0.1

    p1, p2 = points[:2]
    pixel_dist = distance(p1, p2)

    if pixel_dist == 0:
        print("Invalid calibration points, using default 0.1 mm/pixel.")
        return 0.1

    pixel_spacing = float(scale_length_mm) / pixel_dist

    print("Pixel spacing:", pixel_spacing, "mm/pixel")

    return pixel_spacing

# ---------------------------------------------------------
# LINE DETECTION & MEASUREMENT
# ---------------------------------------------------------
def detect_lines(image, min_length=50, max_gap=20, threshold=40):

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5,5), 0)
    edges = cv2.Canny(blur, 50, 150)

    lines = cv2.HoughLinesP(
        edges,
        1,
        np.pi/180,
        threshold,
        minLineLength=min_length,
        maxLineGap=max_gap
    )

    if lines is None:
        return []

    return [tuple(line[0]) for line in lines]


def line_length(line):
    x1, y1, x2, y2 = line
    return np.hypot(x2 - x1, y2 - y1)


def line_angle_deg(line):
    x1, y1, x2, y2 = line
    return np.degrees(np.arctan2(y2 - y1, x2 - x1))


def select_cross_lines(lines):
    if not lines:
        return None, None

    horizontal = [l for l in lines if abs(line_angle_deg(l)) < 20 or abs(abs(line_angle_deg(l)) - 180) < 20]
    vertical = [l for l in lines if abs(abs(line_angle_deg(l)) - 90) < 20]

    if horizontal and vertical:
        h_line = max(horizontal, key=line_length)
        v_line = max(vertical, key=line_length)
        L_px = max(line_length(h_line), line_length(v_line))
        W_px = min(line_length(h_line), line_length(v_line))
        return L_px, W_px

    sorted_lines = sorted(lines, key=line_length, reverse=True)
    if len(sorted_lines) >= 2:
        L_px = line_length(sorted_lines[0])
        W_px = line_length(sorted_lines[1])
        return max(L_px, W_px), min(L_px, W_px)

    return None, None


def select_depth_line(lines):
    if not lines:
        return None

    slanted = [
        l for l in lines
        if 20 < abs(line_angle_deg(l)) < 70 or 110 < abs(line_angle_deg(l)) < 160
    ]
    if slanted:
        return max(slanted, key=line_length)

    return max(lines, key=line_length)


def draw_line_overlay(image, lines, title):
    debug = image.copy()
    for line in lines:
        x1, y1, x2, y2 = line
        cv2.line(debug, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.imshow(title, debug)
    cv2.waitKey(1000)


def measure_scan_lines(left, right, left_ellipse=None, right_ellipse=None):
    left_lines = detect_lines(left)
    right_lines = detect_lines(right)

    L_px, W_px = select_cross_lines(left_lines)
    depth_line = select_depth_line(right_lines)
    H_px = line_length(depth_line) if depth_line is not None else None

    if left_lines:
        draw_line_overlay(left, left_lines, "Detected Left Lines")
    if right_lines:
        draw_line_overlay(right, right_lines, "Detected Right Lines")

    if L_px is None or W_px is None or H_px is None:
        if left_ellipse is not None:
            L_px, W_px = get_dimensions(left_ellipse)
        else:
            raise ValueError("Left scan line measurement failed")

        if right_ellipse is not None:
            H_px, _ = get_dimensions(right_ellipse)
        else:
            raise ValueError("Right scan line measurement failed")

    return L_px, W_px, H_px

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
# REMOVE BLADDER FROM IMAGE
# ---------------------------------------------------------
def remove_bladder_volume(image, contour, fill_value=(0, 0, 0)):

    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    cv2.drawContours(mask, [contour], -1, 255, -1)

    result = image.copy()
    result[mask == 255] = fill_value

    return result

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

    left_clean = remove_bladder_volume(left, left_cnt)
    right_clean = remove_bladder_volume(right, right_cnt)

    cv2.imshow("Left without bladder", left_clean)
    cv2.waitKey(1000)

    cv2.imshow("Right without bladder", right_clean)
    cv2.waitKey(1000)

    L_px, W_px, H_px = measure_scan_lines(
        left,
        right,
        left_ellipse,
        right_ellipse
    )

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

    L_mm = L_px * pixel_spacing
    W_mm = W_px * pixel_spacing
    H_mm = H_px * pixel_spacing

    L_cm = L_mm / 10
    W_cm = W_mm / 10
    H_cm = H_mm / 10

    print("\nReal Dimensions")
    print(f"L: {L_mm:.2f} mm ({L_cm:.2f} cm)")
    print(f"W: {W_mm:.2f} mm ({W_cm:.2f} cm)")
    print(f"H: {H_mm:.2f} mm ({H_cm:.2f} cm)")

    # Bladder volume is computed from the three 2D measurement lines:
    #   - left scan cross-lines → L and W
    #   - right scan slanted depth line → H
    volume = 0.52 * L_cm * W_cm * H_cm

    return volume

# ---------------------------------------------------------
# RUN
# ---------------------------------------------------------
if __name__ == "__main__":

    image_path = r"img012.jpg"

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