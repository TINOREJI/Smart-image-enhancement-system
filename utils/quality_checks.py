import cv2
import numpy as np

def extract_features(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    brightness = gray.mean()
    contrast = gray.std()
    sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()
    color_var = img.var()

    return brightness, contrast, sharpness, color_var


# ---------------- LOW LIGHT DECISION ----------------
def should_brighten(img):
    brightness, contrast, sharpness, color_var = extract_features(img)

    # tuned values
    return (
        brightness < 65 and        # dark
        contrast < 45 and          # low contrast
        color_var > 150 and        # not grayscale
        sharpness > 30             # avoid extreme blur/noise
    )


# ---------------- COLOR CAST ----------------
def has_color_cast(img, threshold=18):
    b, g, r = cv2.split(img)

    mean_b = np.mean(b)
    mean_g = np.mean(g)
    mean_r = np.mean(r)

    max_diff = max(
        abs(mean_r - mean_g),
        abs(mean_r - mean_b),
        abs(mean_g - mean_b)
    )

    return max_diff > threshold


# ---------------- INTENT DETECTION ----------------
def is_artistic_dark(img):
    brightness, contrast, sharpness, _ = extract_features(img)

    return (
        brightness < 65 and
        contrast > 60 and
        sharpness > 90
    )

def is_noisy(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # standard deviation (texture variation)
    std_dev = gray.std()

    # local noise estimation
    lap_var = cv2.Laplacian(gray, cv2.CV_64F).var()

    return std_dev > 25 and lap_var > 120

# ---------------- GOOD IMAGE ----------------
def is_good_image(img):
    brightness, contrast, sharpness, _ = extract_features(img)

    return (
        brightness > 100 and
        contrast > 50 and
        sharpness > 120
    )