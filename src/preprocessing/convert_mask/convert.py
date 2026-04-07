import json
import numpy as np
import cv2
import os

def map_label(label):
    label = label.lower()

    if "inner_bladder" in label:
        return 1
    elif "outer_bladder" in label:
        return 2
    elif "ipp" in label:
        return 3
    else:
        return None


def json_to_mask(json_path, output_path):
    with open(json_path) as f:
        data = json.load(f)

    h, w = data["imageHeight"], data["imageWidth"]
    mask = np.zeros((h, w), dtype=np.uint8)

    for shape in data["shapes"]:
        raw_label = shape["label"]
        mapped_label = map_label(raw_label)

        print(f"Processing: {raw_label} → {mapped_label}")

        if mapped_label is None:
            print("Skipped:", raw_label)
            continue

        points = np.array(shape["points"], dtype=np.int32)

        # Fill polygon
        cv2.fillPoly(mask, [points], mapped_label)

    # Optional resize to match your pipeline (512x512)
    mask = cv2.resize(mask, (512, 512), interpolation=cv2.INTER_NEAREST)

    cv2.imwrite(output_path, mask)
    print("Saved mask:", output_path)


# 🔥 RUN THIS
json_to_mask(
    r"C:\Users\Aditya Atul Deshmukh\Desktop\Nidhi_proj\data\raw\mixed\img011.json",
    r"C:\Users\Aditya Atul Deshmukh\Desktop\Nidhi_proj\data\processed\masked\img011.png"
)