import cv2
import numpy as np


# Using Hough Circle Transform to detect balls
def detect_circles(image, dp=1.2, minDist=100, param1=150, param2=60, minRadius=35, maxRadius=1000):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    circles = cv2.HoughCircles(
        gray, 
        cv2.HOUGH_GRADIENT, 
        dp, 
        minDist, 
        param1=param1, 
        param2=param2, 
        minRadius=minRadius, 
        maxRadius=maxRadius
        )
    
    if circles is not None:
        circles = np.round(circles[0, :]).astype("int")
    return circles

# Draw circles on the image
def draw_circles(image, circles):
    image_with_circles = image.copy()
    if circles is not None:
        for (x, y, r) in circles:
            cv2.circle(image_with_circles, (x, y), r, (0, 255, 0), 4)  
    return image_with_circles
