from modules.pipeline_definitions import *
from modules.pipeline_executor import apply_pipeline
from modules.pipeline_selector import process_image

def generate_variants(img, tasks):

    variants = {}

    # ---------------- BASE ----------------
    variants["original"] = img.copy()
    pipelines = []

    # ---------------- TASK PIPELINES ----------------
    if "noise" in tasks:
        pipelines += NOISE_PIPELINES

    if "blur" in tasks:
        pipelines += BLUR_PIPELINES

    if "low_light" in tasks:
        pipelines += LOW_LIGHT_PIPELINES

    if "ocr" in tasks:
        pipelines += OCR_PIPELINES

    if "face" in tasks:
        pipelines += FACE_PIPELINES

    if "color_cast" in tasks:
        pipelines += COLOR_PIPELINES

    # ---------------- COMBINATIONS ----------------
    for combo, combo_pipes in COMBINATIONS.items():
        if all(t in tasks for t in combo):
            pipelines += combo_pipes

    # ---------------- REMOVE DUPLICATES ----------------
    pipelines = list(dict.fromkeys(tuple(p) for p in pipelines))

    # ---------------- LIMIT (IMPORTANT) ----------------
    pipelines = pipelines[:6]

    # ---------------- APPLY ----------------
    for i, pipe in enumerate(pipelines):
        name = "_".join(pipe)
        variants[name] = apply_pipeline(img.copy(), pipe)

    return variants