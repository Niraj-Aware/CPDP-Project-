#!/usr/bin/env python

import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf

# Load pre-trained model
model = tf.keras.models.load_model('tomato.h5')  # Replace with your tomato disease detection model

# Define labels for prediction output
labels = [
    'Tomato_Bacterial_spot',
    'Tomato_Early_blight',
    'Tomato_Late_blight',
    'Tomato_Leaf_Mold',
    'Tomato_Septoria_leaf_spot',
    'Tomato_Spider_mites_Two_spotted_spider_mite',
    'Tomato__Target_Spot',
    'Tomato__Tomato_YellowLeaf__Curl_Virus',
    'Tomato__Tomato_mosaic_virus',
    'Tomato_healthy'
]

# Define function to preprocess input image
def preprocess_image(image):
    # Resize image
    image = image.resize((150, 150))
    # Convert image to numpy array
    image = np.array(image)
    # Scale pixel values to range [0, 1]
    image = image / 255
    # Expand dimensions to create batch of size 1
    image = np.expand_dims(image, axis=0)
    return image

# Define function to make prediction on input image
def predict(image):
    # Preprocess input image
    image = preprocess_image(image)
    # Make prediction using pre-trained model
    prediction = model.predict(image)
    # Convert prediction from probabilities to label
    label = labels[np.argmax(prediction)]
    # Return label and confidence score
    return label, prediction[0][np.argmax(prediction)]

# Define Streamlit app
def main():
    # Set app title
    st.title('Tomato Plant Disease Detection')
    # Set app description
    st.write('This app helps you to detect diseases on tomato plants.')
    # Add file uploader for input image
    uploaded_file = st.file_uploader('Choose an image', type=['jpg', 'jpeg', 'png'])
    # If file uploaded, display it and make prediction
    if uploaded_file is not None:
        # Load image
        image = Image.open(uploaded_file)
        # Display image
        st.image(image, caption='Uploaded Image', use_column_width=True)
        # Make prediction
        label, score = predict(image)
        # Check if the predicted label is one of the disease labels
        if label in labels[:-1]:
            # Display prediction
            st.write('Prediction: {} (confidence score: {:.2%})'.format(label, score))
            # Provide instructions based on prediction
            st.write('Your tomato plant appears to have {}. You should take appropriate measures to treat the disease and prevent its spread.'.format(label))
        elif label == 'Tomato_healthy':
            st.write('Your tomato plant appears to be healthy. To keep it healthy, make sure to provide proper care, water, and nutrients.')
        else:
            st.write('The uploaded image does not match any disease category.')
            
# Run Streamlit app
if __name__ == '__main__':
    main()
