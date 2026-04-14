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
def auto_roi(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    edges = cv2.Canny(gray, 30, 120)

    kernel = np.ones((5,5), np.uint8)
    edges = cv2.dilate(edges, kernel, iterations=2)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    h, w = gray.shape
    valid = []

    for cnt in contours:
        area = cv2.contourArea(cnt)
        x, y, cw, ch = cv2.boundingRect(cnt)

        # relaxed filtering
        if area > 10000 and ch > h*0.3:
            valid.append(cnt)

    # fallback (VERY IMPORTANT)
    if len(valid) < 2:
        print("Using fallback ROI")
        return image[int(h*0.2):int(h*0.9), int(w*0.1):int(w*0.9)]

    # take top 2 biggest contours
    valid = sorted(valid, key=cv2.contourArea, reverse=True)[:2]

    x_min = min([cv2.boundingRect(c)[0] for c in valid])
    y_min = min([cv2.boundingRect(c)[1] for c in valid])
    x_max = max([cv2.boundingRect(c)[0] + cv2.boundingRect(c)[2] for c in valid])
    y_max = max([cv2.boundingRect(c)[1] + cv2.boundingRect(c)[3] for c in valid])

    roi = image[y_min:y_max, x_min:x_max]

    return roi

# ------------------ AUTO SPLIT ------------------
def split_image(roi):
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

    col_sum = np.sum(gray, axis=0)

    # find darkest vertical gap
    split_x = np.argmin(col_sum)

    # safety clamp
    split_x = max(50, min(split_x, roi.shape[1] - 50))

    left = roi[:, :split_x]
    right = roi[:, split_x:]

    return left, right

# ------------------ MAIN ------------------
def process_and_annotate(image_path):
    global points, img_display

    image = cv2.imread(image_path)

    if image is None:
        raise ValueError("Image not found!")

    # 🔥 AUTO ROI
    roi = auto_roi(image)

    # 🔥 AUTO SPLIT
    left, right = split_image(roi)

    # -------- visualize --------
    plt.figure(figsize=(8,6))

    plt.subplot(1,2,1)
    plt.title("ROI")
    plt.imshow(cv2.cvtColor(roi, cv2.COLOR_BGR2RGB))
    plt.axis('off')

    plt.subplot(1,2,2)
    plt.title("Split")
    plt.imshow(cv2.cvtColor(np.hstack((left, right)), cv2.COLOR_BGR2RGB))
    plt.axis('off')

    plt.show()

    # -------- LEFT --------
    print("\nSTEP 1: LEFT IMAGE")
    print("Click 4 points: Length & Width")

    points = []
    img_display = left.copy()
    cv2.imshow("Annotate", img_display)
    cv2.setMouseCallback("Annotate", click_event)
    cv2.waitKey(0)

    left_pts = points.copy()

    # -------- RIGHT --------
    print("\nSTEP 2: RIGHT IMAGE")
    print("Click 2 points: Depth")

    points = []
    img_display = right.copy()
    cv2.imshow("Annotate", img_display)
    cv2.setMouseCallback("Annotate", click_event)
    cv2.waitKey(0)

    cv2.destroyAllWindows()

    right_pts = points.copy()

    return left_pts, right_pts

# ------------------ VOLUME ------------------
def compute_volume(left_pts, right_pts):
    if len(left_pts) < 4 or len(right_pts) < 2:
        print("Not enough points")
        return None

    L_px = distance(left_pts[0], left_pts[1])
    W_px = distance(left_pts[2], left_pts[3])
    H_px = distance(right_pts[0], right_pts[1])

    # temporary scaling (same as your approach)
    pixel_spacing = 81.70 / L_px

    L = (L_px * pixel_spacing) / 10
    W = (W_px * pixel_spacing) / 10
    H = (H_px * pixel_spacing) / 10

    print("\nDimensions (cm):")
    print("L:", L)
    print("W:", W)
    print("H:", H)

    volume = 0.52 * L * W * H

    return volume

# ------------------ RUN ------------------
if __name__ == "__main__":
    image_path = r"D:\Bladder-images-processing\img012.jpg"

    left_pts, right_pts = process_and_annotate(image_path)

    vol = compute_volume(left_pts, right_pts)

    if vol:
        print("\nEstimated Volume:", vol, "ml")