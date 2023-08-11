#!/usr/bin/env python
import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras.models import load_model

# Load the pre-trained model
model = load_model('model.h5')
class_names = ['Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy']

# Preprocess the input image
def preprocess_image(image):
    img = image.resize((224, 224))  # Resize to match the input shape expected by the model
    img = np.array(img) / 255.0     # Normalize pixel values to [0, 1]
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    return img

# Function to predict disease
def predict_disease(image):
    preprocessed_img = preprocess_image(image)
    prediction = model.predict(preprocessed_img)
    predicted_class = np.argmax(prediction)
    confidence = prediction[0][predicted_class] * 100
    return class_names[predicted_class], confidence

# Streamlit app
def main():
    st.title("Potato Disease Detection")
    st.write("Upload an image of a potato leaf to detect the disease.")

    uploaded_image = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_image is not None:
        image = Image.open(uploaded_image)
        st.image(image, caption="Uploaded Image", use_column_width=True)

        if st.button("Predict"):
            predicted_class, confidence = predict_disease(image)
            st.write(f"Predicted Disease: {predicted_class}")
            st.write(f"Confidence: {confidence:.2f}%")

if __name__ == "__main__":
    main()
