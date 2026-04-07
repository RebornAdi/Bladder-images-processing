import torch
import cv2
import numpy as np
from models.unet import UNet

model = UNet()
model.load_state_dict(torch.load("model.pth"))
model.eval()

def predict(image_path):
    img = cv2.imread(image_path, 0)
    img_resized = cv2.resize(img, (512, 512)) / 255.0

    tensor = torch.tensor(img_resized).unsqueeze(0).unsqueeze(0).float()

    with torch.no_grad():
        pred = model(tensor)
        pred = torch.argmax(pred, dim=1).squeeze().numpy()

    return img, pred

def overlay(image, mask):
    color_mask = np.zeros((512, 512, 3), dtype=np.uint8)

    color_mask[mask == 1] = [0, 255, 0]     # bladder inner
    color_mask[mask == 2] = [255, 0, 0]     # bladder outer
    color_mask[mask == 3] = [0, 0, 255]     # prostate

    image = cv2.resize(image, (512, 512))
    image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    overlay = cv2.addWeighted(image, 0.7, color_mask, 0.3, 0)

    return overlay

img, mask = predict("test.png")
result = overlay(img, mask)

cv2.imwrite("output.png", result)