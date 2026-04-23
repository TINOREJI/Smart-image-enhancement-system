import cv2

def extract_features(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    return {
        "brightness": round(gray.mean(), 2),
        "sharpness": round(cv2.Laplacian(gray, cv2.CV_64F).var(), 2),
        "contrast": round(gray.std(), 2)
    }