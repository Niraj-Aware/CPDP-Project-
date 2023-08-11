#!/usr/bin/env python

import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf

# Load pre-trained model
model = tf.keras.models.load_model('potatoDisease.h5')

# Define labels for prediction output
labels = {
    0: 'Potato___Late_blight',
    1: 'Potato___Early_blight',
    2: 'Potato___healthy',
}

# Define function to preprocess input image
def preprocess_image(image):
    # Resize image
    image = image.resize((150, 150))
    # Convert image to numpy array
    image = np.array(image)
    # Scale pixel values to range [0, 1]
    image = image / 150
    # Expand dimensions to create a batch of size 1
    image = np.expand_dims(image, axis=0)
    return image

# Define function to make a prediction on the input image
def predict(image):
    # Preprocess the input image
    image = preprocess_image(image)
    # Make a prediction using the pre-trained model
    prediction = model.predict(image)
    # Convert the prediction from probabilities to a label
    label = labels[np.argmax(prediction)]
    # Return the label and confidence score
    return label, prediction[0][np.argmax(prediction)]

# Define the Streamlit app
def main():
    # Set app title
    st.title('Potato Disease Detection')
    # Set app description
    st.write('This app helps you to detect the type of disease in a potato plant.')
    # Add file uploader for input image
    uploaded_file = st.file_uploader('Choose an image', type=['jpg', 'jpeg', 'png'])
    # If a file is uploaded, display it and make a prediction
    if uploaded_file is not None:
        # Load the image
        image = Image.open(uploaded_file)
        # Display the image
        st.image(image, caption='Uploaded Image', use_column_width=True)
        # Make a prediction
        label, score = predict(image)
        # Display prediction
        st.write('Prediction: {} (confidence score: {:.2%})'.format(label, score))

if __name__ == '__main__':
    main()
