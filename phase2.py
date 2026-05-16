import cv2
import numpy as np
import matplotlib.pyplot as plt

# ------------------ AUTO ROI ------------------
def auto_roi(image):
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

    if len(valid) < 2:
        raise ValueError("ROI detection failed")

    x_min, y_min = 99999, 99999
    x_max, y_max = 0, 0

    for cnt in valid:
        x, y, cw, ch = cv2.boundingRect(cnt)
        x_min = min(x_min, x)
        y_min = min(y_min, y)
        x_max = max(x_max, x+cw)
        y_max = max(y_max, y+ch)

    roi = image[y_min:y_max, x_min:x_max]
    return roi


# ------------------ AUTO SPLIT (FIXED) ------------------
def auto_roi(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Step 1: blur
    blur = cv2.GaussianBlur(gray, (5,5), 0)

    # Step 2: adaptive threshold (better than canny alone)
    _, thresh = cv2.threshold(blur, 30, 255, cv2.THRESH_BINARY)

    # Step 3: morphology to merge regions
    kernel = np.ones((15,15), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

    # Step 4: find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if len(contours) == 0:
        print("ROI fallback used")
        h, w = image.shape[:2]
        return image[int(0.1*h):int(0.9*h), int(0.1*w):int(0.9*w)]

    # Step 5: take largest contour
    cnt = max(contours, key=cv2.contourArea)

    x, y, w, h = cv2.boundingRect(cnt)

    # Add margin
    pad = 20
    x1 = max(0, x-pad)
    y1 = max(0, y-pad)
    x2 = min(image.shape[1], x+w+pad)
    y2 = min(image.shape[0], y+h+pad)

    roi = image[y1:y2, x1:x2]

    return roi


# ------------------ AUTO PIXEL SPACING ------------------
def auto_pixel_spacing(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape

    scale_region = gray[:, int(w * 0.9):]

    edges = cv2.Canny(scale_region, 50, 150)
    row_sum = np.sum(edges, axis=1)

    peaks = np.where(row_sum > np.mean(row_sum))[0]

    if len(peaks) < 2:
        print("Scale detection failed → fallback used")
        return 0.1

    y1 = peaks[0]
    y2 = peaks[-1]

    pixel_dist = abs(y2 - y1)
    pixel_spacing = 100 / pixel_dist

    print("Pixel spacing:", pixel_spacing, "mm/pixel")

    return pixel_spacing


# ------------------ AUTO BLADDER MEASUREMENT ------------------
def auto_measure(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    blur = cv2.GaussianBlur(gray, (5,5), 0)

    # bladder = dark → invert threshold
    _, thresh = cv2.threshold(blur, 50, 255, cv2.THRESH_BINARY_INV)

    kernel = np.ones((5,5), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=3)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if len(contours) == 0:
        print("Bladder not detected")
        return None, None

    cnt = max(contours, key=cv2.contourArea)

    x, y, w, h = cv2.boundingRect(cnt)

    # visualize
    vis = image.copy()
    cv2.rectangle(vis, (x,y), (x+w,y+h), (0,255,0), 2)

    plt.imshow(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB))
    plt.title("Auto Detection")
    plt.axis('off')
    plt.show()

    return w, h


# ------------------ MAIN PIPELINE ------------------
def process(image_path):

    image = cv2.imread(image_path)

    if image is None:
        raise ValueError("Image not found")

    # ROI
    roi = auto_roi(image)

    # Split
    left, right = auto_roi(roi)

    # Pixel spacing
    pixel_spacing = auto_pixel_spacing(image)

    # Measurements
    L_px, W_px = auto_measure(left)
    H_px, _ = auto_measure(right)

    if L_px is None or H_px is None:
        print("Measurement failed")
        return

    print("\nPixel values:")
    print("L:", L_px, "W:", W_px, "H:", H_px)

    # Convert to cm
    L = (L_px * pixel_spacing) / 10
    W = (W_px * pixel_spacing) / 10
    H = (H_px * pixel_spacing) / 10

    print("\nDimensions (cm):")
    print("L:", L, "W:", W, "H:", H)

    volume = 0.52 * L * W * H

    print("\nEstimated Volume:", volume, "ml")


# ------------------ RUN ------------------
if __name__ == "__main__":
    image_path = r"D:\Bladder-images-processing\img012.jpg"
    process(image_path)