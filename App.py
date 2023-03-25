#!/usr/bin/env python
import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
import requests

# Load pre-trained model
model = tf.keras.models.load_model('v3_pred_cott_dis.h5')

# Define labels for prediction output
labels = ['diseased','healthy']

# Define function to preprocess input image
def preprocess_image(image):
    # Resize image
    image = image.resize((150,150))
    # Convert image to numpy array
    image = np.array(image)
    # Scale pixel values to range [0, 1]
    image = image / 150
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
    st.title('Cotton Plant Disease Detection')
    # Set app description
    st.write('This app helps you to detect whether a cotton plant is healthy or diseased.')
    # Add file uploader for input image
    uploaded_file = st.file_uploader('Choose an image', type=['jpg', 'jpeg', 'png'])
    # If file uploaded, display it and make prediction
    if uploaded_file is not None:
        try:
            # Load image
            image = Image.open(uploaded_file)
            # Check whether the image is a cotton plant image
            response = requests.post(
                "https://api.deepai.org/api/image-similarity",
                data={
                    'image': image,
                    'key': 'YOUR_API_KEY'
                }
            )
            result = response.json()
            if result['output'][0]['similarity'] >= 0.5:
                # Display image
                st.image(image, caption='Uploaded Image', use_column_width=True)
                # Make prediction
                label, score = predict(image)
                # Display prediction
                st.write('Prediction: {} (confidence score: {:.2f})'.format(label, score))
                
                # Provide instructions based on prediction
                if label == 'diseased':
                    st.write('Your cotton plant appears to be diseased. To prevent the spread of disease, you should remove the infected plant and treat the soil. You can also consult a local agricultural expert for advice on how to prevent future outbreaks of disease.')
                else:
                    st.write('Your cotton plant appears to be healthy. To keep it healthy, make sure to provide adequate water and fertilize regularly. You should also control pests and prune and train the plant to promote healthy growth. Harvest at the right time to ensure the highest quality fiber.')
            else:
                st.write('The uploaded image does not seem to be a cotton plant image. Please upload an appropriate image.')
        except:
            st.write('Error: Please try again with an appropriate image.')
            
# Run Streamlit app
if __name__ == '__main__':
    main()
