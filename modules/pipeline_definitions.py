# ---------------- PIPELINE DEFINITIONS ----------------

NOISE_PIPELINES = [
    ["denoise"],
    ["denoise", "sharpen"],
]

BLUR_PIPELINES = [
    ["sharpen"],
    ["denoise", "sharpen"],
]

LOW_LIGHT_PIPELINES = [
    ["low_light"],
    ["brighten"],
    ["low_light", "denoise"],
    ["denoise", "low_light"],
]

OCR_PIPELINES = [
    ["contrast"],
    ["contrast", "sharpen"],
    ["sharpen", "contrast"],
]

FACE_PIPELINES = [
    ["face_smooth"],
    ["face_light"],
    ["face_smooth", "face_light"],
]

COLOR_PIPELINES = [
    ["color_correct"],
]

# ---------------- COMBINATIONS ----------------

COMBINATIONS = {
    ("noise", "low_light"): [
        ["denoise", "low_light"],
        ["low_light", "denoise"],
    ],
    ("blur", "noise"): [
        ["denoise", "sharpen"],
    ],
    ("face", "low_light"): [
        ["face_light"],
        ["denoise", "face_light"],
    ],
    ("ocr", "low_light"): [
        ["low_light", "contrast"],
        ["contrast", "sharpen"],
    ],
}