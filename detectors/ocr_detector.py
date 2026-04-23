import pytesseract
import cv2

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

def detect_text(img):
    text = pytesseract.image_to_string(img)
    return len(text.strip()) > 10

def is_rotated(img):
    try:
        osd = pytesseract.image_to_osd(img)
        return "Rotate: 90" in osd or "Rotate: 270" in osd
    except:
        return False

def correct_rotation(img):
    osd = pytesseract.image_to_osd(img)

    angle = 0
    if "Rotate: 90" in osd:
        angle = 90
    elif "Rotate: 180" in osd:
        angle = 180
    elif "Rotate: 270" in osd:
        angle = 270

    if angle != 0:
        (h, w) = img.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, -angle, 1.0)
        img = cv2.warpAffine(img, M, (w, h))

    return img