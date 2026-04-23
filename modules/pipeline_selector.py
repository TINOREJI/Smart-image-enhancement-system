import cv2
from modules.enhancement import *
from detectors.ocr_detector import is_rotated, correct_rotation
from utils.quality_checks import *

def process_image(img, tasks):

    pipeline = []

    brightness, contrast, sharpness, color_var = extract_features(img)

    # ---------------- INTENT ----------------
    if is_artistic_dark(img):
        return img, ["no_change"]
    # ---------------- LOW LIGHT ----------------
    if "face" in tasks and "low_light" in tasks:
        # ONLY face-aware enhancement (not both)
        img = enhance_face_low_light(img)
        pipeline.append("face_light")

    elif "low_light" in tasks and should_brighten(img):
        img = enhance_low_light(img)
        pipeline.append("adaptive_light")

    # ---------------- NOISE ----------------
    if "noise" in tasks:
        pipeline.append("denoise")

    # ---------------- OCR ----------------
    if "ocr" in tasks:
        if is_rotated(img):
            img = correct_rotation(img)

        if "contrast" not in pipeline:
            pipeline.append("contrast")

    # ---------------- FACE (SAFE VERSION) ----------------
    if "face" in tasks:
        if "noise" in tasks:
            pipeline.append("denoise")

        if sharpness < 80:
            pipeline.append("sharpen")

    # ---------------- BLUR ----------------
    if "blur" in tasks and sharpness < 60:
        pipeline.append("sharpen")

    # ---------------- COLOR ----------------
    if "color_cast" in tasks:
        img = correct_color(img)

    # ---------------- LIMIT ----------------
    pipeline = list(dict.fromkeys(pipeline))[:3]

    # ---------------- APPLY ----------------
    enhanced = apply_pipeline(img, pipeline)

    # 🔥 better blending for faces
    alpha = 0.85 if "face" in tasks else 0.75
    img = blend_images(img, enhanced, alpha=alpha)

    return img, pipeline