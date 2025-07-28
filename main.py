import imageLoader, preprocessing, textExtractor #local files
import debugging

import numpy as np
import cv2, pytesseract
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

#pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

image = imageLoader.loadImage("./Input/test2.jpg")

image = preprocessing.preprocess(image)

#debugging
print("debugging")
debugging.showImage(image, "Processed Image")
text = pytesseract.image_to_string(image, config='--psm 6')
print("Extracted Text:\n", text)

