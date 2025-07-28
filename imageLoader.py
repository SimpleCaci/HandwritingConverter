#importations:
import cv2
import os # to find file location
#function imageLOADER
    #loads a specific image

def loadImage(path, grayscale=True):
    #path - str - path to the image
    #grayscale - bool - load the image in grayscale or not

    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")
    
    #sets the image to grayscale loading unless not
    flag = cv2.IMREAD_GRAYSCALE
    if not grayscale:
        flag = cv2.IMREAD_COLOR

    #reads the image
    image = cv2.imread(path,flag)

    if image is None:
        raise FileNotFoundError(f"Image unable to load: {path}")

    return image
