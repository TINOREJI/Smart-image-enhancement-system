import cv2
import numpy as np

# ---------------- BASIC ENHANCEMENTS ----------------
def enhance_low_light(img):
    """
    Hybrid low-light enhancement:
    CLAHE + Gamma correction
    """

    # --- CLAHE ---
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=1.8, tileGridSize=(8,8))
    l = clahe.apply(l)

    lab = cv2.merge((l, a, b))
    img_clahe = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    # --- Gamma correction (adaptive) ---
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    brightness = gray.mean()

    # darker image → stronger gamma
    if brightness < 40:
        gamma = 1.45
    elif brightness < 60:
        gamma = 1.2
    else:
        gamma = 1.0

    img_gamma = gamma_correction(img_clahe, gamma)

    return img_gamma

def denoise(img):
    return cv2.fastNlMeansDenoisingColored(img, None, 5, 5, 7, 21)

def sharpen(img):
    kernel = np.array([[0,-0.3,0],
                       [-0.3,2,-0.3],
                       [0,-0.3,0]])
    return cv2.filter2D(img, -1, kernel)

def contrast(img):
    return cv2.convertScaleAbs(img, alpha=1.15, beta=0)

def brighten(img):
    return cv2.convertScaleAbs(img, alpha=1.08, beta=15)

# ---------------- COLOR CORRECTION ----------------

def correct_color(img):
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l = cv2.equalizeHist(l)
    lab = cv2.merge((l, a, b))
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

# ---------------- BLENDING (VERY IMPORTANT) ----------------

def blend_images(original, enhanced, alpha=0.7):
    return cv2.addWeighted(original, alpha, enhanced, 1 - alpha, 0)

def gamma_correction(img, gamma=1.5):
    invGamma = 1.0 / gamma
    table = np.array([
        ((i / 255.0) ** invGamma) * 255 for i in np.arange(256)
    ]).astype("uint8")

    return cv2.LUT(img, table)
def enhance_face_low_light(img):
    """
    Gentle enhancement for faces
    Preserves skin tone
    """

    # --- Convert to LAB ---
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    # --- Mild CLAHE (less aggressive) ---
    clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8,8))
    l = clahe.apply(l)

    lab = cv2.merge((l, a, b))
    img = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    # --- Very mild gamma ---
    gamma = 1.2
    invGamma = 1.0 / gamma

    table = np.array([
        ((i / 255.0) ** invGamma) * 255 for i in np.arange(256)
    ]).astype("uint8")

    img = cv2.LUT(img, table)

    return img

# ---------------- PIPELINE EXECUTION ----------------

def apply_pipeline(img, pipeline):
    out = img.copy()

    for step in pipeline:
        if step == "denoise":
            out = denoise(out)
        elif step == "sharpen":
            out = sharpen(out)
        elif step == "contrast":
            out = contrast(out)
        elif step == "brighten":
            out = brighten(out)

    return out