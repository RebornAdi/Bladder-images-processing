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

        # draw point
        cv2.circle(img_display, (x, y), 5, (0, 0, 255), -1)

        # draw line every 2 points
        if len(points) % 2 == 0:
            cv2.line(img_display, points[-2], points[-1], (0, 255, 0), 2)

        cv2.imshow("Annotate", img_display)


# ------------------ AUTO PIXEL SPACING ------------------
def auto_pixel_spacing(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape

    # Crop right-most area (scale region)
    scale_region = gray[:, int(w * 0.9):]

    # Edge detection
    edges = cv2.Canny(scale_region, 50, 150)

    # Sum edges vertically
    row_sum = np.sum(edges, axis=1)

    # Find strong responses (tick marks)
    peaks = np.where(row_sum > np.mean(row_sum))[0]

    if len(peaks) < 2:
        print("⚠️ Scale detection failed, using fallback")
        return 0.1  # fallback

    y1 = peaks[0]
    y2 = peaks[-1]

    pixel_dist = abs(y2 - y1)

    real_distance_mm = 100  # assume 10 cm scale

    pixel_spacing = real_distance_mm / pixel_dist

    print("✅ Auto pixel spacing:", pixel_spacing, "mm/pixel")

    return pixel_spacing


# ------------------ MAIN PIPELINE ------------------
def process_and_annotate(image_path):
    global points, img_display

    image = cv2.imread(image_path)

    # --------- ROI ----------
    x1, y1 = 324, 116
    x2, y2 = 1441, 958

    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

    roi = image[y1:y2, x1:x2]
    h, w = roi.shape[:2]

    # --------- SPLIT ----------
    left = roi[:, :w//2]
    right = roi[:, w//2:]

    # --------- VISUAL ----------
    plt.figure(figsize=(8, 6))
    plt.subplot(1, 2, 1)
    plt.title("ROI")
    plt.imshow(cv2.cvtColor(roi, cv2.COLOR_BGR2RGB))
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.title("Full Image")
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.axis('off')

    plt.show()
    plt.pause(2)
    plt.close()

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
def compute_volume(left_pts, right_pts, image_path):

    image = cv2.imread(image_path)

    # 🔥 AUTO PIXEL SPACING
    pixel_spacing = auto_pixel_spacing(image)

    # Safety check
    if len(left_pts) < 4 or len(right_pts) < 2:
        print("❌ Not enough points selected!")
        return None

    L_px = distance(left_pts[0], left_pts[1])
    W_px = distance(left_pts[2], left_pts[3])
    H_px = distance(right_pts[0], right_pts[1])

    print("\n--- Pixel Measurements ---")
    print("L_px:", L_px)
    print("W_px:", W_px)
    print("H_px:", H_px)

    # px → cm
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
    image_path = "D:\Bladder-images-processing\img012.jpg"

    left_pts, right_pts = process_and_annotate(image_path)

    vol = compute_volume(left_pts, right_pts, image_path)

    print("\n✅ Estimated Volume:", vol, "ml")