import cv2
import numpy as np
import matplotlib.pyplot as plt

points = []
img_display = None


# ------------------ Utility ------------------
def distance(p1, p2):
    return np.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)


def click_event(event, x, y, flags, param):
    global points, img_display

    if event == cv2.EVENT_LBUTTONDOWN:
        points.append((x, y))
        cv2.circle(img_display, (x, y), 5, (0, 0, 255), -1)

        if len(points) % 2 == 0:
            cv2.line(img_display, points[-2], points[-1], (0, 255, 0), 2)

        cv2.imshow("Annotate", img_display)


# ------------------ AUTO ROI ------------------
def auto_detect_roi(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    edges = cv2.Canny(gray, 50, 150)

    kernel = np.ones((5,5), np.uint8)
    edges = cv2.dilate(edges, kernel, iterations=2)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    h, w = gray.shape
    valid = []

    for cnt in contours:
        area = cv2.contourArea(cnt)
        x, y, cw, ch = cv2.boundingRect(cnt)

        if area > 20000 and ch > h*0.4 and cw > w*0.2:
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


# ------------------ SCALE CALIBRATION ------------------
def get_pixel_spacing_manual(image):
    global points, img_display

    print("\nSTEP 0: SCALE CALIBRATION")
    print("Click 2 points on scale bar (0 cm to 10 cm)")

    points = []
    img_display = image.copy()

    cv2.imshow("Scale Calibration", img_display)
    cv2.setMouseCallback("Scale Calibration", click_event)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    if len(points) < 2:
        print("Calibration failed → fallback")
        return 0.1

    p1, p2 = points[0], points[1]

    pixel_dist = distance(p1, p2)

    if pixel_dist == 0:
        print("Invalid scale → fallback")
        return 0.1

    pixel_spacing = 100 / pixel_dist

    print("Pixel spacing:", pixel_spacing, "mm/pixel")

    return float(pixel_spacing)

def auto_measure(image, label=""):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    blur = cv2.GaussianBlur(gray, (5,5), 0)

    edges = cv2.Canny(blur, 40, 120)

    kernel = np.ones((3,3), np.uint8)
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(
        edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    if not contours:
        print(f"{label} fallback")
        h, w = image.shape[:2]
        return w//2, h//2

    h_img, w_img = gray.shape
    cx_img, cy_img = w_img//2, h_img//2

    best_cnt = None
    best_score = 1e9

    for cnt in contours:
        area = cv2.contourArea(cnt)

        if area < 600:   # only remove tiny noise
            continue

        x, y, w, h = cv2.boundingRect(cnt)
        cx = x + w//2
        cy = y + h//2

        dist = np.sqrt((cx - cx_img)**2 + (cy - cy_img)**2)

        # 🔥 balanced scoring (no harsh rejection)
        score = dist + 0.0005 * abs(area - 20000)

        if score < best_score and len(cnt) >= 5:
            best_score = score
            best_cnt = cnt

    if best_cnt is None:
        print(f"{label} fallback used")
        h, w = image.shape[:2]
        return w//2, h//2

    ellipse = cv2.fitEllipse(best_cnt)
    (cx, cy), (major, minor), _ = ellipse

    L = max(major, minor)
    W = min(major, minor)

    # Debug
    debug = image.copy()
    cv2.drawContours(debug, [best_cnt], -1, (255,0,0), 2)
    cv2.ellipse(debug, ellipse, (0,255,0), 2)

    cv2.imshow(f"Edge {label}", debug)
    cv2.waitKey(2000)   # stays longer

    return L, W
# ------------------ MAIN PIPELINE ------------------
def process_and_measure(image):
    roi, (x1, y1, x2, y2) = auto_detect_roi(image)

    vis_img = image.copy()
    cv2.rectangle(vis_img, (x1, y1), (x2, y2), (0, 255, 0), 2)

    h, w = roi.shape[:2]

    left = roi[:, :w//2]
    right = roi[:, w//2:]

    # Show left/right images clearly
    cv2.imshow("Left Image", left)
    cv2.waitKey(1200)

    cv2.imshow("Right Image", right)
    cv2.waitKey(1200)

    plt.figure(figsize=(8, 6))
    plt.subplot(1, 2, 1)
    plt.title("ROI")
    plt.imshow(cv2.cvtColor(roi, cv2.COLOR_BGR2RGB))
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.title("Full Image with ROI")
    plt.imshow(cv2.cvtColor(vis_img, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.show()

    print("\nAUTO MEASUREMENT RUNNING...")

    L_px, W_px = auto_measure(left, "LEFT")
    _, H_px = auto_measure(right, "RIGHT")

    return L_px, W_px, H_px


# ------------------ VOLUME ------------------
def compute_volume(L_px, W_px, H_px, pixel_spacing):

    if pixel_spacing is None:
        print("⚠️ Pixel spacing missing → fallback")
        pixel_spacing = 0.1

    print("\n--- Pixel Measurements ---")
    print("L_px:", L_px)
    print("W_px:", W_px)
    print("H_px:", H_px)

    L = (L_px * pixel_spacing) / 10
    W = (W_px * pixel_spacing) / 10
    H = (H_px * pixel_spacing) / 10

    print("\n--- Real Dimensions (cm) ---")
    print("L:", L)
    print("W:", W)
    print("H:", H)

    volume = 0.52 * L * W * H

    return volume


# ------------------ RUN ------------------
if __name__ == "__main__":
    image_path = "img012.jpg"

    image = cv2.imread(image_path)

    if image is None:
        raise ValueError("Image not found")

    pixel_spacing = get_pixel_spacing_manual(image)

    L_px, W_px, H_px = process_and_measure(image)

    vol = compute_volume(L_px, W_px, H_px, pixel_spacing)

    print("\nEstimated Volume:", vol, "ml")

    cv2.destroyAllWindows()