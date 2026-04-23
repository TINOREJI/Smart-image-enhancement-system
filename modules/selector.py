import cv2
from modules.cnn_model import cnn_score
from modules.niqe_model import niqe_score, normalize_niqe


# ---------------- OVERPROCESS PENALTY ----------------
def overprocess_penalty(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    contrast = gray.std()
    sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()

    penalty = 0

    # too high contrast → unnatural
    if contrast > 80:
        penalty += 0.05

    # too sharp → artifacts
    if sharpness > 300:
        penalty += 0.05

    return penalty


# ---------------- MAIN SELECTOR ----------------
def select_best_variant(variants, tasks):

    original = variants["original"]

    # baseline scores
    cnn_orig = cnn_score(original)
    niqe_orig = normalize_niqe(niqe_score(original))

    best_name = None
    best_score = -1e9   # allow negative values
    best_img = None

    print("\n--- VARIANT EVALUATION ---")

    for name, img in variants.items():

        cnn = cnn_score(img)
        niqe = normalize_niqe(niqe_score(img))

        # 🔥 AMPLIFY DIFFERENCES
        cnn_gain = (cnn - cnn_orig) * 3
        niqe_gain = (niqe_orig - niqe) * 2

        # 🔥 PENALTY
        penalty = overprocess_penalty(img)

        # 🔥 BONUS SYSTEM (helps break ties)
        bonus = 0
        if cnn > cnn_orig:
            bonus += 0.02
        if niqe < niqe_orig:
            bonus += 0.02

        # 🔥 TASK-AWARE WEIGHTING
        if "face" in tasks:
            final = 0.7 * cnn_gain + 0.3 * niqe_gain + bonus - penalty

        elif "ocr" in tasks:
            final = 0.5 * cnn_gain + 0.5 * niqe_gain + bonus - penalty

        elif "low_light" in tasks:
            final = 0.6 * cnn_gain + 0.4 * niqe_gain + bonus - penalty

        else:
            final = 0.6 * cnn_gain + 0.4 * niqe_gain + bonus - penalty

        print(
            f"{name:20s} → "
            f"CNN:{cnn:.3f} | NIQE:{niqe:.3f} | "
            f"GAIN(C:{cnn_gain:.3f}, N:{niqe_gain:.3f}) | "
            f"FINAL:{final:.3f}"
        )

        if final > best_score:
            best_score = final
            best_img = img
            best_name = name

    print(f"\n✅ SELECTED: {best_name} (score: {best_score:.3f})\n")

    return best_img, best_name