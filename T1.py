import streamlit as st
from PIL import Image
import tensorflow as tf
import numpy as np

# Load pre-trained models
#tomato_model = tf.keras.models.load_model('tomato_model.h5')
cotton_model = tf.keras.models.load_model('cotton_model.h5')
corn_model = tf.keras.models.load_model('corn_model.h5')

# Define classes for diseases
#tomato_classes = ["Healthy", "Diseased"]
cotton_classes = ["Healthy", "Diseased"]
corn_classes = ["Healthy", "Diseased"]

st.title("Plant Disease Detection")

uploaded_image = st.file_uploader("Choose an image...", type=["jpg", "png"])

if uploaded_image is not None:
    image = Image.open(uploaded_image)
    st.image(image, caption='Uploaded Image', use_column_width=True)

    # Preprocess the image for the models
    image = image.resize((224, 224))  # Resize image to match model input size
    image = np.array(image) / 255.0  # Normalize pixel values
    image = np.expand_dims(image, axis=0)  # Add batch dimension

    # Make predictions using the models
    #tomato_prediction = tomato_model.predict(image)
    cotton_prediction = cotton_model.predict(image)
    corn_prediction = corn_model.predict(image)

    # Display disease predictions
    st.subheader("Disease Predictions:")
    #st.write("Tomato:", tomato_classes[np.argmax(tomato_prediction)])
    st.write("Cotton:", cotton_classes[np.argmax(cotton_prediction)])
    st.write("Corn:", corn_classes[np.argmax(corn_prediction)])
  
