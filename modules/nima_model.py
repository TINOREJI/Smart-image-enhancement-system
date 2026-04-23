import numpy as np
import cv2
from tensorflow.keras.applications.mobilenet import MobileNet, preprocess_input
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.models import Model

# Build NIMA-like model
def build_nima_model():
    base_model = MobileNet(weights='imagenet', include_top=False)

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.5)(x)
    x = Dense(10, activation='softmax')(x)  # NIMA predicts distribution (1–10)

    model = Model(inputs=base_model.input, outputs=x)

    return model


# Load model (no pretrained weights for NIMA → using ImageNet features)
model = build_nima_model()


def nima_score(img):
    """
    Returns score from 1–10
    """

    img = cv2.resize(img, (224, 224))
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img.astype("float32"))

    preds = model.predict(img, verbose=0)[0]

    # expected score
    scores = np.arange(1, 11)
    mean_score = np.sum(preds * scores)

    return mean_score / 10.0  # normalize 0–1