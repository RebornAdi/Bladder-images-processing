import cv2
import numpy as np

points = []
current_mode = "left"  # left or right

def distance(p1, p2):
    return np.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)

def click_event(event, x, y, flags, param):
    global points, img_display

    if event == cv2.EVENT_LBUTTONDOWN:
        points.append((x, y))

        cv2.circle(img_display, (x, y), 5, (0,0,255), -1)

        if len(points) % 2 == 0:
            cv2.line(img_display, points[-2], points[-1], (0,255,0), 2)

        cv2.imshow("Annotate", img_display)


def annotate_dual(image_path):
    global img_display, points

    img = cv2.imread(image_path)
    h, w = img.shape[:2]

    # split into left & right
    left = img[:, w//5:w//3]
    right = img[:, w//2:]

    print("STEP 1: LEFT IMAGE")
    print("Click 4 points:")
    print("1-2: Length (vertical)")
    print("3-4: Width (horizontal)")

    points = []
    img_display = left.copy()
    cv2.imshow("Annotate", img_display)
    cv2.setMouseCallback("Annotate", click_event)
    cv2.waitKey(0)

    left_points = points.copy()

    print("STEP 2: RIGHT IMAGE")
    print("Click 2 points:")
    print("5-6: Depth (H)")

    points = []
    img_display = right.copy()
    cv2.imshow("Annotate", img_display)
    cv2.setMouseCallback("Annotate", click_event)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    right_points = points.copy()

    return left_points, right_points


def compute_volume(left_pts, right_pts, pixel_spacing=0.1):
    # LEFT
    L_px = distance(left_pts[0], left_pts[1])
    W_px = distance(left_pts[2], left_pts[3])

    # RIGHT
    H_px = distance(right_pts[0], right_pts[1])
    
    
    # convert to mm
    L = L_px * pixel_spacing
    W = W_px * pixel_spacing
    H = H_px * pixel_spacing
    
    # mm → cm
    L /= 10
    W /= 10
    H /= 10
    
    volume = 0.52 * L * W * H

    return volume


if __name__ == "__main__":
    image_path = r"C:\Users\Aditya Atul Deshmukh\Desktop\Nidhi_proj\data\processed\images\img012.jpg"

    left_pts, right_pts = annotate_dual(image_path)

    vol = compute_volume(left_pts, right_pts)

    print("Estimated Volume:", vol, "ml")