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
        print("Calibration failed")
        return 0.1

    p1, p2 = points[0], points[1]

    pixel_dist = distance(p1, p2)
    pixel_spacing = 100 / pixel_dist  # 10 cm = 100 mm

    print("Pixel spacing:", pixel_spacing, "mm/pixel")

    return pixel_spacing


# ------------------ MAIN PIPELINE ------------------
def process_and_annotate(image):
    global points, img_display

    # --------- AUTO ROI ----------
    roi, (x1, y1, x2, y2) = auto_detect_roi(image)

    vis_img = image.copy()
    cv2.rectangle(vis_img, (x1, y1), (x2, y2), (0, 255, 0), 2)

    h, w = roi.shape[:2]

    # --------- SPLIT (still simple for now) ----------
    left = roi[:, :w//2]
    right = roi[:, w//2:]

    # --------- VISUAL ----------
    plt.figure(figsize=(8, 6))

    plt.subplot(1, 2, 1)
    plt.title("Detected ROI")
    plt.imshow(cv2.cvtColor(roi, cv2.COLOR_BGR2RGB))
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.title("Full Image with ROI")
    plt.imshow(cv2.cvtColor(vis_img, cv2.COLOR_BGR2RGB))
    plt.axis('off')

    plt.show()

    # --------- LEFT ----------
    print("\nSTEP 1: LEFT IMAGE")
    print("Click 4 points:")
    print("1-2: Length")
    print("3-4: Width")

    points = []
    img_display = left.copy()
    cv2.imshow("Annotate", img_display)
    cv2.setMouseCallback("Annotate", click_event)
    cv2.waitKey(0)

    left_pts = points.copy()

    # --------- RIGHT ----------
    print("\nSTEP 2: RIGHT IMAGE")
    print("Click 2 points:")
    print("Depth")

    points = []
    img_display = right.copy()
    cv2.imshow("Annotate", img_display)
    cv2.setMouseCallback("Annotate", click_event)
    cv2.waitKey(0)

    cv2.destroyAllWindows()

    right_pts = points.copy()

    return left_pts, right_pts


# ------------------ VOLUME ------------------
def compute_volume(left_pts, right_pts, pixel_spacing):

    if len(left_pts) < 4 or len(right_pts) < 2:
        print("Not enough points selected!")
        return None

    L_px = distance(left_pts[0], left_pts[1])
    W_px = distance(left_pts[2], left_pts[3])
    H_px = distance(right_pts[0], right_pts[1])

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
    image_path = r"img012.jpg"

    image = cv2.imread(image_path)

    if image is None:
        raise ValueError("Image not found")

    # STEP 0: SCALE
    pixel_spacing = get_pixel_spacing_manual(image)

    # STEP 1–2: ROI + ANNOTATION
    left_pts, right_pts = process_and_annotate(image)

    # STEP 3: VOLUME
    vol = compute_volume(left_pts, right_pts, pixel_spacing)

    print("\nEstimated Volume:", vol, "ml")