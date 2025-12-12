from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os

app = Flask(__name__)

app.config["UPLOAD_FOLDER"] = "static/uploads"
app.config["SAMPLE_FOLDER"] = "static/samples"

model = load_model("fire_smoke_model.h5")

CLASSES = ["Fire", "Non Fire", "Smoke"]
IMG_SIZE = (128, 128)

def predict_image(img_path):
    img = image.load_img(img_path, target_size=IMG_SIZE)
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)[0]
    class_index = np.argmax(prediction)
    confidence = round(np.max(prediction) * 100, 2)

    return CLASSES[class_index], confidence


@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    img_path = None

    # CASE 1: User uploads a file
    if request.method == "POST" and "image" in request.files:
        file = request.files["image"]
        if file.filename != "":
            img_path = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
            file.save(img_path)
            label, conf = predict_image(img_path)
            prediction = f"{label} ({conf}%)"

    # CASE 2: User clicks sample button
    if request.method == "POST" and "sample" in request.form:
        sample_name = request.form["sample"]
        img_path = os.path.join(app.config["SAMPLE_FOLDER"], sample_name)
        label, conf = predict_image(img_path)
        prediction = f"{label} ({conf}%)"

    return render_template("index.html", prediction=prediction, img_path=img_path)


if __name__ == "__main__":
    app.run(debug=True)
