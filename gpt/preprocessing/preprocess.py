import cv2
import os
import numpy as np

INPUT_DIR = r"C:\Users\Aditya Atul Deshmukh\Desktop\Nidhi_proj\data\raw\mixed"
OUTPUT_DIR = r"C:\Users\Aditya Atul Deshmukh\Desktop\Nidhi_proj\data\processed\images"

os.makedirs(OUTPUT_DIR, exist_ok=True)

def preprocess(img):
    # --- 1. SAFE CROP (reduce UI only, not anatomy) ---
    h, w = img.shape[:2]
    img = img[60:h-200, 20:w-150]   # less aggressive crop

    # --- 2. Convert to grayscale ---
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # --- 3. Resize ---
    # img = cv2.resize(img, (512, 512))

    # --- 4. Speckle noise reduction (better than bilateral here) ---
    img = cv2.fastNlMeansDenoising(img, None, 10, 7, 21)

    # --- 5. Mild CLAHE (not aggressive) ---
    clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8,8))
    img = clahe.apply(img)

    # --- 6. Smooth edges slightly (important for segmentation) ---
    img = cv2.GaussianBlur(img, (3,3), 0)

    # --- 7. Normalize safely ---
    img = img.astype(np.float32) / 255.0

    return (img * 255).astype(np.uint8)


for file in os.listdir(INPUT_DIR):
    path = os.path.join(INPUT_DIR, file)

    img = cv2.imread(path)
    if img is None:
        continue

    processed = preprocess(img)

    cv2.imwrite(os.path.join(OUTPUT_DIR, file), processed)

print("DONE: Clean preprocessing complete")