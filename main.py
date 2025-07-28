# main.py
import imageLoader, preprocessing, textExtractor  # local files
import debugging

import numpy as np
import cv2
import pytesseract

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

image = imageLoader.loadImage("./Input/test3.jpg")

# Optional resize for huge images
max_width = 1200
if image.shape[1] > max_width:
    scale_ratio = max_width / image.shape[1]
    image = cv2.resize(image, None, fx=scale_ratio, fy=scale_ratio)

processed = preprocessing.preprocess(image)

# Show processed image
cv2.imshow("Processed", processed)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Extract and print text
text = textExtractor.extract_text(processed)
print("Extracted Text:\n", text)

# Draw text boxes
results = textExtractor.extract_text_with_boxes(processed)
image_with_boxes = cv2.cvtColor(processed, cv2.COLOR_GRAY2BGR)

for item in results:
    x, y, w, h = item["box"]
    conf = item["conf"]
    cv2.rectangle(image_with_boxes, (x, y), (x + w, y + h), (0, 255, 0), 2)
    cv2.putText(image_with_boxes, f'{conf}', (x, y - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

cv2.imshow("Detected Text Boxes", image_with_boxes)
cv2.waitKey(0)
cv2.destroyAllWindows()