import numpy as np
import cv2

image = np.zeros((400, 400, 3), dtype=np.uint8)
image = cv2.rectangle(image, (0, 0), (400, 400), (0, 255, 0), -1)
# image = cv2.circle(image, (200, 200), 150, (0, 0, 255), -1)


def gradient(image, center, radius):
    # cv2.circle(image, center, radius, (0, 0, 0), -1)

    for i in range(1, radius):
        color = 255 - int(i / radius * 255)
        cv2.circle(image, center, i, (0, 0, 255-color), 2)
