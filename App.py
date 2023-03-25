#!/usr/bin/env python

import streamlit as st
import requests
from io import BytesIO
from PIL import Image
import numpy as np
import tensorflow as tf

# Define function to preprocess input image
def preprocess_image(image):
    # Resize image
    image = image.resize((224,224))
    # Convert image to numpy array
    image = np.array(image)
    # Scale pixel values to range [0, 1]
    image = image / 255.0
    # Expand dimensions to create batch of size 1
    image = np.expand_dims(image, axis=0)
    return image

# Define function to make prediction on input image
def predict(image_url):
    # Fetch image from URL
    response = requests.get(image_url)
    image = Image.open(BytesIO(response.content))
    # Preprocess input image
    image = preprocess_image(image)
    # Load pre-trained model
    model = tf.keras.applications.MobileNetV2()
    # Make prediction using pre-trained model
    predictions = model.predict(image)
    # Decode predictions
    decoded_predictions = tf.keras.applications.mobilenet_v2.decode_predictions(predictions, top=1)
    # Return True if image contains a cotton leaf, else False
    return decoded_predictions[0][0][1] == 'cotton'

# Define Streamlit app
def main():
    # Set app title
    st.title('Cotton Plant Leaf Detection')
    # Set app description
    st.write('This app helps you to detect whether an uploaded image is a cotton plant leaf or not.')
    # Add text input for image URL
    image_url = st.text_input('Enter image URL:')
    # If image URL provided, display it and make prediction
    if image_url != '':
        # Make prediction
        is_cotton = predict(image_url)
        # Display prediction
        if is_cotton:
            st.write('This is a cotton plant leaf.')
        else:
            st.write('This is not a cotton plant leaf. Please try with an appropriate input image.')

# Run Streamlit app
if __name__ == '__main__':
    main()

