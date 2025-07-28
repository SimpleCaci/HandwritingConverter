import pytesseract
from pytesseract import Output
import cv2

def extract_text(image, config='--psm 11'):
    """Extract plain text from an image."""
    return pytesseract.image_to_string(image, config=config)

def extract_text_with_boxes(image, conf_threshold=30):
    data = pytesseract.image_to_data(image, output_type=Output.DICT)
    results = []
    for i in range(len(data["text"])):
        if int(data["conf"][i]) > conf_threshold:
            results.append({
                "text": data["text"][i],
                "conf": int(data["conf"][i]),
                "box": (data["left"][i], data["top"][i], data["width"][i], data["height"][i])
            })
    return results

