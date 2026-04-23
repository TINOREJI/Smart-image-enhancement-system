import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import numpy as np

# ---------------- MODEL (MATCH TRAINING EXACTLY) ----------------

class IQACNN(nn.Module):
    def __init__(self):
        super(IQACNN, self).__init__()

        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.conv4 = nn.Conv2d(128, 256, 3, padding=1)

        self.pool = nn.MaxPool2d(2, 2)
        self.gap = nn.AdaptiveAvgPool2d((1, 1))

        self.fc1 = nn.Linear(256, 128)
        self.fc2 = nn.Linear(128, 1)

        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = self.pool(F.relu(self.conv4(x)))

        x = self.gap(x)
        x = x.view(x.size(0), -1)

        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = torch.sigmoid(self.fc2(x))

        return x.squeeze()


# ---------------- LOAD MODEL ----------------

model = IQACNN()
model.load_state_dict(torch.load("modules/cnn/iqa_model.pth", map_location="cpu"))
model.eval()


# ---------------- SCORE FUNCTION ----------------

def cnn_score(img):
    """
    img: OpenCV BGR
    return: 0–1 quality score
    """

    img = cv2.resize(img, (224, 224))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    img = img / 255.0
    tensor = torch.tensor(img).permute(2, 0, 1).unsqueeze(0).float()

    with torch.no_grad():
        score = model(tensor).item()

    # safety clamp
    score = max(0, min(1, score))

    return float(score)