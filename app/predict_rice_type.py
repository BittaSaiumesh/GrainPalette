import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np

# Load the saved model
model = tf.keras.models.load_model("rice_type_model.h5")

# Class labels (same order as during training)
class_names = ['Arborio', 'Basmati', 'Ipsala', 'Jasmine', 'Karacadag']

# Load and preprocess image
img_path = r"C:\Users\saium\OneDrive\Desktop\rice_dataset\Rice_Image_Dataset\Ipsala\Ipsala (42).jpg"
  # Change this to your test image path
img = image.load_img(img_path, target_size=(224, 224))
img_array = image.img_to_array(img)
img_array = tf.expand_dims(img_array, 0)  # Make it batch size 1

# Predict
predictions = model.predict(img_array)
predicted_class = class_names[np.argmax(predictions)]

print(f"Predicted rice type: {predicted_class}")
