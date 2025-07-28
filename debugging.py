import cv2

def showImage(image, title="image"):
    cv2.imshow(title, image)  
    cv2.waitKey(0)  
    cv2.destroyAllWindows()
    print(image.shape)