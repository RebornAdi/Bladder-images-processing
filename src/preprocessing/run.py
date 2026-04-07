from measurements import compute_all

result = compute_all(
    mask_path=r"C:\Users\Aditya Atul Deshmukh\Desktop\Nidhi_proj\data\processed\masked\img011.png"
)

print(result)

import numpy as np
import cv2
color = np.zeros((512,512,3), dtype=np.uint8)
mask = cv2.imread(r"C:\Users\Aditya Atul Deshmukh\Desktop\Nidhi_proj\data\processed\masked\img011.png",0)
color[mask==1] = [0,255,0]   # inner
color[mask==2] = [255,0,0]   # outer
color[mask==3] = [0,0,255]   # ipp

cv2.imwrite("mask_debug.png", color)