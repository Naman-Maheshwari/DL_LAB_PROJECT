import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image

# Load the saved model
model = load_model('model.h5')

class_labels = ['Non-Autistic', 'Autistic'] 

# Streamlit UI
st.title("Image Classification using CNN")
st.write("Take a picture or upload an image to classify.")

# Use camera input to capture image
img_file = st.camera_input("Take a picture")

# Alternatively, allow users to upload an image
if img_file is None:
    img_file = st.file_uploader("Or upload an image...", type=["jpg", "png", "jpeg"])

if img_file is not None:
    # Load the image and preprocess it
    img = Image.open(img_file)
    st.image(img, caption='Uploaded Image', use_column_width=True)

    # Preprocess the image to the correct format
    img = img.resize((224, 224))  # Resizing to the model's expected input size
    img = np.array(img) / 255.0   # Normalize
    img = np.expand_dims(img, axis=0)  # Add batch dimension

    # Make prediction
    prediction = model.predict(img)
    predicted_class = np.argmax(prediction, axis=1)[0]
    st.write(prediction)
    st.write(f"Prediction: {class_labels[predicted_class]}")
