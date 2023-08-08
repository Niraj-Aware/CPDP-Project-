import streamlit as st
from PIL import Image
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np

# Load the pre-trained model from TensorFlow Hub
model_url = "https://tfhub.dev/google/plant_village/classifier/corn_disease_V1/2"
model = hub.load(model_url)

# Define classes for plant diseases
classes = ["Healthy", "Diseased"]

st.title("Plant Disease Detection")

uploaded_image = st.file_uploader("Choose a plant leaf image...", type=["jpg", "png"])

if uploaded_image is not None:
    image = Image.open(uploaded_image)
    st.image(image, caption='Uploaded Image', use_column_width=True)

    # Preprocess the image for the model
    image = image.resize((224, 224))  # Resize image to match model input size
    image = np.array(image) / 255.0  # Normalize pixel values
    image = np.expand_dims(image, axis=0)  # Add batch dimension

    # Make prediction using the model
    prediction = model.predict(image)

    # Display disease prediction
    st.subheader("Disease Prediction:")
    predicted_class = classes[np.argmax(prediction)]
    confidence = prediction[0][np.argmax(prediction)]
    st.write("Predicted Class:", predicted_class)
    st.write("Confidence:", confidence)
    
