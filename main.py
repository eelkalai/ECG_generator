import cv2
import numpy as np

import FormatUtils

# Read the image
if __name__ == '__main__':
    image = cv2.imread('ECG10.jpg')
    height, width, _ = image.shape
    # Define the coordinates of the line
    # For example, let's say the line is a horizontal line at y=100

    points = FormatUtils.loadECGPoints('ECG10.txt')[1:]
    x1, y1 = int(points[0] * width), int(points[1] * height)
    x2, y2 = int(points[2] * width), int(points[3] * height)
    x3, y3 = int(points[4] * width), int(points[5] * height)
    x4, y4 = int(points[6] * width), int(points[7] * height)
    # Create a mask for the diagonal line
    mask = np.zeros_like(image[:, :, 0])

    # Define the size of the blur kernel
    kernel_size = (15, 15)

    cv2.line(mask, (x1, y1), (x2, y2), color=255, thickness=5)
    cv2.line(mask, (x2, y2), (x3, y3), color=255, thickness=5)
    cv2.line(mask, (x3, y3), (x4, y4), color=255, thickness=5)
    cv2.line(mask, (x4, y4), (x1, y1), color=255, thickness=5)

    blurred_region = cv2.GaussianBlur(image, kernel_size, sigmaX=0, sigmaY=0)[y1:y2, x1:x2]
    image[y1:y2, x1:x2] = blurred_region

    blurred_region = cv2.GaussianBlur(image, kernel_size, sigmaX=0, sigmaY=0)[y2:y3, x2:x3]
    image[y2:y3, x2:x3] = blurred_region

    blurred_region = cv2.GaussianBlur(image, kernel_size, sigmaX=0, sigmaY=0)[y3:y4, x3:x4]
    image[y3:y4, x3:x4] = blurred_region

    blurred_region = cv2.GaussianBlur(image, kernel_size, sigmaX=0, sigmaY=0)[y4:y1, x4:x1]
    image[y4:y1, x4:x1] = blurred_region

    # Display the result
    cv2.imshow('Blurred Image', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
