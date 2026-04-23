from modules.enhancement import *

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

        elif step == "low_light":
            out = enhance_low_light(out)

        elif step == "face_light":
            out = enhance_face_low_light(out)

        elif step == "face_smooth":
            out = denoise(out)

        elif step == "color_correct":
            out = correct_color(out)

    return out