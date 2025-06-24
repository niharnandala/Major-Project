# 🌿 Basil Plant Disease Detection Web App

A lightweight Flask-based web application that detects **Basil plant diseases** using a custom-trained **deep learning model** built with TensorFlow. The model can classify images of basil leaves into four categories: **Bacterial**, **Fungal**, **Pest**, or **Healthy**, and recommends a suitable remedy.

---

## 🧠 Features

- ✅ Upload an image of a basil leaf
- 🧪 Classifies it into one of four categories:
  - **Bacterial**
  - **Fungal**
  - **Pest**
  - **Healthy**
- 💡 Provides tailored remedies based on the prediction
- 🔬 Uses a **custom-trained TensorFlow model**
- 🖼️ Displays uploaded image with results for better user understanding

---

## 🏗️ Project Structure

basil-disease-detector/
│
├── app.py # Main Flask app
├── models/
│ └── basil_disease_best_model.keras # Your custom trained model
├── templates/
│ ├── index.html # Upload page
│ └── result.html # Result page with prediction & remedy
├── static/
│ └── uploads/ # Uploaded images saved here
└── requirements.txt # Python dependencies

yaml
Copy
Edit

---

## 🚀 Getting Started

### 1. Clone this repo

```bash
git clone https://github.com/yourusername/basil-disease-detector.git
cd basil-disease-detector
2. Set up environment
bash
Copy
Edit
pip install -r requirements.txt
Or manually install:

bash
Copy
Edit
pip install Flask tensorflow numpy pillow
3. Run the app
bash
Copy
Edit
python app.py
Then open http://127.0.0.1:5000 in your browser.

📸 Sample Use
Open the app in your browser.

Upload a clear image of a basil leaf.

Get the disease classification and suggested treatment.

🧠 Model Info
Trained using TensorFlow/Keras

Input shape: 224x224 RGB images

Normalized pixel values (/255.0)

Custom-trained by NIHAR NANDALA for Basil leaf disease classification

Accuracy: [You can add your model accuracy here]

📌 To-Do Ideas
Deploy to Render, Hugging Face Spaces, or Heroku

Add camera input support

Add Grad-CAM visualizations for model explainability

👨‍🔬 Credits
Model trained and app developed by: NIHAR NANDALA

Powered by Flask, TensorFlow, and Love for Plants 🌿

📄 License
This project is licensed under the MIT License.


---

## ✅ Final Steps (Quick Recap)

1. Create file `LICENSE` → paste the first block above
2. Create or update file `README.md` → paste the second block above
3. In terminal:

```bash
git add LICENSE README.md
git commit -m "Add MIT license and project README"
git push