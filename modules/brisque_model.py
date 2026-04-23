import cv2
import numpy as np

def brisque_score(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    lap_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    contrast = gray.std()

    # combine features
    score = (0.6 * (1 / (lap_var + 1)) + 
             0.4 * (1 / (contrast + 1)))

    return float(score * 100)