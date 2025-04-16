from flask import Flask, render_template, request
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
import os

app = Flask(__name__)

# Load the trained model
model = tf.keras.models.load_model("models/basil_disease_best_model.keras")

# Class names
class_names = ["Bacterial", "Fungal", "Healthy", "Pest"]

# Remedies for each disease
remedies = {
    "Bacterial": "Use copper-based fungicides and remove infected leaves.",
    "Fungal": "Apply neem oil or baking soda spray to the plant.",
    "Pest": "Use natural insecticides like neem oil or garlic spray.",
    "Healthy": "Your plant is healthy! No action needed."
}

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
def predict():
    if "file" not in request.files:
        return "No file uploaded", 400

    file = request.files["file"]
    if file.filename == "":
        return "No selected file", 400

    filepath = os.path.join("static/uploads", file.filename)
    file.save(filepath)

    # Process image
    img = image.load_img(filepath, target_size=(224, 224))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Predict
    prediction = model.predict(img_array)
    predicted_class = class_names[np.argmax(prediction)]
    remedy = remedies[predicted_class]

    return render_template("result.html", image_path=filepath, disease=predicted_class, remedy=remedy)

if __name__ == "__main__":
    app.run(debug=True)
