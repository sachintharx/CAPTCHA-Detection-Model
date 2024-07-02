import os
import streamlit as st
from PIL import Image
import numpy as np
from tensorflow.keras.models import load_model
import tensorflow as tf

# Load the pre-trained model
model = load_model('model.h5')  # Update with your model path

def preprocess_image(image):
    # Resize the image to match model's expected sizing
    image = image.resize((150, 40))
    # Convert image to grayscale
    image = image.convert('L')
    # Convert image to array
    image = np.array(image)
    # Normalize pixel values to between 0 and 1
    image = image / 255.0
    # Expand dimensions to match batch size used during training
    image = np.expand_dims(image, axis=-1)
    image = np.expand_dims(image, axis=0)
    return image

# Define the characters your model can predict
characters = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"

def predict_single(img):
    img = Image.open(img)
    img = preprocess_image(img)
    pred = model.predict(img)
    captcha = ""
    for i in range(5):
        captcha += characters[np.argmax(pred[i])]
    return captcha

st.title('Captcha Detection Model')

uploaded_file = st.file_uploader("Upload a Captcha Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Load and preprocess the image
    image = Image.open(uploaded_file)

    # Display the uploaded image
    st.image(image, caption='Uploaded Captcha Image', use_column_width=True)

    # Predict the captcha text and display result
    captcha = predict_single(uploaded_file)
    st.write(f"Predicted Captcha Text: {captcha}")

# streamlit run captcha_detection.py