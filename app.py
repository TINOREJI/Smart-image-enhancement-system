from flask import Flask, render_template, request
import os
import cv2

from modules.task_predictor import predict_tasks
from modules.features import extract_features
from modules.pipeline_selector import process_image

app = Flask(__name__)

UPLOAD_FOLDER = "static/uploads/"
OUTPUT_FOLDER = "static/outputs/"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":

        file = request.files["image"]

        if file.filename == "":
            return render_template("index.html", error="No file selected")

        filepath = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(filepath)

        img = cv2.imread(filepath)

        if img is None:
            return render_template("index.html", error="Invalid image")

        # 🔥 MULTI TASK DETECTION
        tasks = predict_tasks(img)

        # features
        features = extract_features(img)

        # enhancement
        output_img, pipeline = process_image(img, tasks)

        output_path = os.path.join(OUTPUT_FOLDER, "output.jpg")
        cv2.imwrite(output_path, output_img)

        return render_template(
            "index.html",
            original=filepath,
            output=output_path,
            tasks=tasks,
            pipeline="_".join(pipeline),
            features=features
        )

    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)