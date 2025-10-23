from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
import numpy as np

# Classes
classes = ['Arborio', 'Basmati', 'Ipsala', 'Jasmine', 'Karacadag']

# Create a very simple model
model = Sequential([
    Flatten(input_shape=(224,224,3)),
    Dense(128, activation='relu'),
    Dense(len(classes), activation='softmax')
])

# Since we don't have data, just save the untrained model
model.save("rice_model.h5")
print("Dummy rice_model.h5 created successfully!")
