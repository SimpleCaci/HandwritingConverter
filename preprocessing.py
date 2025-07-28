import cv2
import numpy as np
#for filters and preprocessing
def preprocess(image):
    resized = cv2.resize(image, None, fx=2, fy=2, interpolation=cv2.INTER_LINEAR)  
    blurred = cv2.GaussianBlur(resized, (5, 5), 0)  
    adaptive_thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)  
    kernel = np.ones((2, 2), np.uint8)
    image = cv2.morphologyEx(adaptive_thresh, cv2.MORPH_OPEN, kernel)
    return image