import cv2
import numpy as np

def niqe_score(img):
    """
    Lightweight NIQE-like proxy
    Lower = worse quality
    """

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # noise estimate
    noise = gray.std()

    # sharpness
    sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()

    # brightness deviation
    brightness = gray.mean()

    score = (
        0.5 * (1 / (sharpness + 1)) +
        0.3 * (noise / 100) +
        0.2 * (abs(brightness - 128) / 128)
    )

    return float(score * 100)


def normalize_niqe(score):
    return max(0, min(1, 1 - score / 100))