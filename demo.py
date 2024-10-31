from sparse import edge
import cv2

eps = 1e-2
img = cv2.imread("data/input/16.png",0)
img = edge(img, eps)
cv2.imwrite("data/output/16.png", img)