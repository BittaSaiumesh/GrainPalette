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

## 📷 Sample Output
Input: 224x224 rice image
Output: Predicted rice type → "Basmati"

yaml
Copy code

---

## ✅ How to Run

### 🧪 Step 1: Setup Environment
```bash
pip install -r requirements.txt