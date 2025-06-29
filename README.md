# GrainPalette 🧠🌾  
**A Deep Learning Odyssey in Rice Type Classification Through Transfer Learning**

---

## 📌 Overview
GrainPalette is a deep learning-based image classification project aimed at identifying different types of rice grains—Arborio, Basmati, Ipsala, Jasmine, and Karacadag—using transfer learning techniques.

---

## 🚀 Project Objectives
- Build an accurate image classification model for rice types.
- Use Transfer Learning to reduce training time and increase performance.
- Deploy a simple prediction script (with optional GUI) for real-time predictions.

---

## 📁 Project Structure
GrainPalette/
├── data/ # Contains the rice image dataset
│ └── Rice_Image_Dataset_split/
├── notebooks/ # Jupyter notebooks for step-by-step development
├── saved_model/ # Trained model (.h5)
├── app/ # Prediction scripts (CLI or GUI)
├── docs/ # Documentation and final report
├── requirements.txt # List of dependencies
└── README.md # You're here!

---

## 📌 Workflow

1. **Data Collection** – Gather rice images organized by type.
2. **Image Preprocessing** – Resize, normalize, and prepare data generators.
3. **Model Building** – Use MobileNetV2 or VGG16 with transfer learning.
4. **Training and Evaluation** – Train the model and validate accuracy.
5. **Save and Load the Model** – Save model to `.h5` and load for predictions.
6. **Application Building** – Predict using CLI or optional GUI app.

---

## 🛠️ Technologies Used

- Python
- TensorFlow / Keras
- Jupyter Notebook
- NumPy
- Matplotlib
- Pillow (PIL)

---

## 🎯 Final Model Performance

- **Model Type:** Transfer Learning (MobileNetV2 or VGG16)
- **Input Size:** 224 × 224 × 3
- **Validation Accuracy:** 96.29%
- **Framework:** TensorFlow + Keras

---

## ✅ How to Run

### 🧪 Step 1: Setup Environment
```bash
pip install -r requirements.txt


🖼️ Step 2: Run Inference Script
You can test the model on a sample image using this code:

python
Copy code
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np

# Load model
model = load_model("saved_model/rice_classifier.h5")

# Set image path
img_path = "data/Rice_Image_Dataset_split/validation/Basmati/basmati (10014).jpg"

# Preprocess image
img = image.load_img(img_path, target_size=(224, 224))
img_array = image.img_to_array(img) / 255.0
img_array = np.expand_dims(img_array, axis=0)

# Predict
class_labels = ['Arborio', 'Basmati', 'Ipsala', 'Jasmine', 'Karacadag']
pred = model.predict(img_array)
print("Predicted Class:", class_labels[np.argmax(pred)])


🙋 Author
Name: Bitta Saiumesh

Location: Kurnool, Andhra Pradesh, India

Email: saiumeshbitta@gmail.com

Phone: 6304695782

📌 Project Status
🎯 Project Completed
📦 Ready for portfolio, demo, or deployment


---


