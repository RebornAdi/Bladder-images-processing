import cv2
import numpy as np
import matplotlib.pyplot as plt
img = cv2.imread("img012.jpg")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

edges = cv2.Canny(gray, 50, 150)

kernel = np.ones((5,5), np.uint8)
edges = cv2.dilate(edges, kernel, iterations=2)

contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

h, w = gray.shape
valid = []

for cnt in contours:
    area = cv2.contourArea(cnt)
    x, y, cw, ch = cv2.boundingRect(cnt)

    # filter conditions (IMPORTANT)
    if area > 20000 and ch > h*0.4 and cw > w*0.2:
        valid.append(cnt)

# take top 2 biggest
valid = sorted(valid, key=cv2.contourArea, reverse=True)[:2]

mask = np.zeros_like(gray)

for cnt in valid:
    cv2.drawContours(mask, [cnt], -1, 255, -1)

result = cv2.bitwise_and(img, img, mask=mask)

plt.imshow(result)
cv2.imshow("Result", result)
cv2.waitKey(0)
cv2.destroyAllWindows()