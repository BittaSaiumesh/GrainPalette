import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import base64
import os

# ✅ Load the trained model with corrected path
model = tf.keras.models.load_model("app/rice_classifier_model.h5")

# Class names
class_names = ['Arborio', 'Basmati', 'Ipsala', 'Jasmine', 'Karacadag']

# Set custom background
def set_background(image_file):
    if os.path.exists(image_file):
        with open(image_file, "rb") as image:
            encoded = base64.b64encode(image.read()).decode()
        st.markdown(
            f"""
            <style>
            .stApp {{
                background-image: url("data:image/png;base64,{encoded}");
                background-size: cover;
                background-repeat: no-repeat;
                background-attachment: fixed;
            }}
            </style>
            """,
            unsafe_allow_html=True
        )

# Set background image (optional)
set_background("app/bg.jpg")

# Title
st.markdown("""
    <h1 style='text-align: center; color: white;'>🍚 Rice Type Classifier</h1>
    <p style='text-align: center; color: white;'>Upload a rice image to predict its type with confidence!</p>
""", unsafe_allow_html=True)

# Upload image
uploaded_file = st.file_uploader("📷 Choose a rice image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    try:
        # Open and display image
        img = Image.open(uploaded_file).convert("RGB")
        st.image(img, caption='Uploaded Image', use_container_width=True)

        # Preprocess
        img = img.resize((224, 224))
        img_array = image.img_to_array(img)
        img_array = tf.expand_dims(img_array, 0)

        # Predict
        predictions = model.predict(img_array)
        predicted_class = class_names[np.argmax(predictions)]
        confidence = np.max(predictions)

        # Display result
        st.markdown(f"""
            <div style='
                background-color: rgba(255, 255, 255, 0.9); 
                padding: 20px; 
                border-radius: 15px;
                text-align: center;
                box-shadow: 0 0 10px rgba(0,0,0,0.2);
            '>
                <h2>🔍 Predicted: <span style='color: #2c3e50;'>{predicted_class}</span></h2>
                <p>Confidence: <strong>{confidence:.2%}</strong></p>
            </div>
        """, unsafe_allow_html=True)

        # Show all probabilities
        st.markdown("### 📊 All Class Probabilities:")
        for i, prob in enumerate(predictions[0]):
            st.write(f"**{class_names[i]}**: {prob:.2%}")
    except Exception as e:
        st.error(f"⚠️ Error processing image: {str(e)}")
