import cv2
import numpy as np

# ---------------- CORE UTILS ----------------

def get_brightness(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return gray.mean()

def gamma_correction(img, gamma=1.5):
    invGamma = 1.0 / gamma
    table = np.array([
        ((i / 255.0) ** invGamma) * 255 for i in np.arange(256)
    ]).astype("uint8")

    return cv2.LUT(img, table)


# ---------------- LOW LIGHT ----------------

def enhance_low_light(img):
    """
    Adaptive low-light enhancement (balanced)
    """
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    brightness = get_brightness(img)

    # adaptive CLAHE
    clip = 2.0 if brightness < 60 else 1.5
    clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=(8,8))
    l = clahe.apply(l)

    img = cv2.cvtColor(cv2.merge((l,a,b)), cv2.COLOR_LAB2BGR)

    # adaptive gamma
    if brightness < 40:
        gamma = 1.6
    elif brightness < 70:
        gamma = 1.3
    else:
        gamma = 1.0

    return gamma_correction(img, gamma)


def enhance_low_light_strong(img):
    """
    Strong version (for variant generation)
    """
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    l = clahe.apply(l)

    img = cv2.cvtColor(cv2.merge((l,a,b)), cv2.COLOR_LAB2BGR)

    return gamma_correction(img, 1.6)


# ---------------- DENOISE ----------------

def denoise(img):
    return cv2.fastNlMeansDenoisingColored(img, None, 5, 5, 7, 21)

def denoise_strong(img):
    return cv2.fastNlMeansDenoisingColored(img, None, 10, 10, 7, 21)


# ---------------- SHARPEN ----------------

def sharpen(img):
    kernel = np.array([[0,-0.3,0],
                       [-0.3,2,-0.3],
                       [0,-0.3,0]])
    return cv2.filter2D(img, -1, kernel)

def sharpen_strong(img):
    kernel = np.array([[0,-1,0],
                       [-1,5,-1],
                       [0,-1,0]])
    return cv2.filter2D(img, -1, kernel)


# ---------------- CONTRAST / BRIGHTNESS ----------------

def contrast(img):
    return cv2.convertScaleAbs(img, alpha=1.15, beta=0)

def contrast_strong(img):
    return cv2.convertScaleAbs(img, alpha=1.4, beta=0)

def brighten(img):
    return cv2.convertScaleAbs(img, alpha=1.1, beta=20)

def brighten_strong(img):
    return cv2.convertScaleAbs(img, alpha=1.2, beta=35)


# ---------------- COLOR ----------------

def correct_color(img):
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l = cv2.equalizeHist(l)
    return cv2.cvtColor(cv2.merge((l,a,b)), cv2.COLOR_LAB2BGR)


# ---------------- FACE SAFE ----------------

def enhance_face_low_light(img):
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8,8))
    l = clahe.apply(l)

    img = cv2.cvtColor(cv2.merge((l,a,b)), cv2.COLOR_LAB2BGR)

    return gamma_correction(img, 1.2)


# ---------------- BLENDING ----------------

def blend_images(original, enhanced, alpha=0.7):
    return cv2.addWeighted(original, alpha, enhanced, 1 - alpha, 0)


# ---------------- PIPELINE EXECUTION ----------------

def apply_pipeline(img, pipeline):
    out = img.copy()

    for step in pipeline:

        if step == "denoise":
            out = denoise(out)

        elif step == "denoise_strong":
            out = denoise_strong(out)

        elif step == "sharpen":
            out = sharpen(out)

        elif step == "sharpen_strong":
            out = sharpen_strong(out)

        elif step == "contrast":
            out = contrast(out)

        elif step == "contrast_strong":
            out = contrast_strong(out)

        elif step == "brighten":
            out = brighten(out)

        elif step == "brighten_strong":
            out = brighten_strong(out)

        elif step == "low_light":
            out = enhance_low_light(out)

        elif step == "low_light_strong":
            out = enhance_low_light_strong(out)

    return out