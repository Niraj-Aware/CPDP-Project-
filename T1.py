import streamlit as st
from PIL import Image
import tensorflow as tf
import numpy as np

# Load the pre-trained model for tomato leaf disease detection
tomato_model = tf.keras.models.load_model('v3_pred_tomato_dis.h5')

# Define classes for tomato leaf diseases
tomato_classes = ["Healthy", "Diseased"]

st.title("Tomato Leaf Disease Detection")

uploaded_image = st.file_uploader("Choose a tomato leaf image...", type=["jpg", "png"])

if uploaded_image is not None:
    image = Image.open(uploaded_image)
    st.image(image, caption='Uploaded Image', use_column_width=True)

    # Preprocess the image for the model
    image = image.resize((224, 224))  # Resize image to match model input size
    image = np.array(image) / 255.0  # Normalize pixel values
    image = np.expand_dims(image, axis=0)  # Add batch dimension

    # Make prediction using the model
    prediction = tomato_model.predict(image)

    # Display disease prediction
    st.subheader("Disease Prediction:")
    st.write("Predicted Class:", tomato_classes[np.argmax(prediction)])
    st.write("Confidence:", prediction[0][np.argmax(prediction)])
    
