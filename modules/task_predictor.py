import cv2
from detectors.face_detector import detect_face
from detectors.ocr_detector import detect_text
from utils.quality_checks import *

def predict_tasks(img):

    tasks = []

    # ---------------- STRONG SIGNALS ----------------
    if detect_face(img):
        tasks.append("face")

    if detect_text(img):
        tasks.append("ocr")

    # ---------------- FEATURES ----------------
    brightness, contrast, sharpness, color_var = extract_features(img)

    # ---------------- INTENT ----------------
    if brightness < 60 and contrast > 60 and sharpness > 100:
        return ["no_change"]

    # ---------------- NOISE FIRST (PRIORITY) ----------------
    noise_flag = is_noisy(img)
    if noise_flag:
        tasks.append("noise")

    # ---------------- BLUR (ONLY IF NOT NOISY) ----------------
    if not noise_flag:
        if sharpness < 65 and "ocr" not in tasks and brightness > 80:
            tasks.append("blur")

    # ---------------- LOW LIGHT ----------------
    if brightness < 70 and contrast < 50 and color_var > 150:
        tasks.append("low_light")

    # ---------------- GRAYSCALE SPECIAL CASE ----------------
    if color_var < 50 and noise_flag:
        # likely medical / grayscale noisy image
        if "noise" not in tasks:
            tasks.append("noise")

    # ---------------- LIMIT ----------------
    tasks = list(dict.fromkeys(tasks))[:3]

    if not tasks:
        tasks.append("natural")

    return tasks