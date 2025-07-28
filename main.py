import imageLoader, preprocessing, textExtractor #local files

import numpy as np
import cv2

#pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

image = imageLoader.loadImage("./Input/test.jpg")

image = preprocessing.imagePreprocessing(image)
