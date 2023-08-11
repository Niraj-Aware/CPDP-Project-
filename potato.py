import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf

# Load the pre-trained model
model = tf.keras.models.load_model('potato_disease_model.h5')

# Define disease class names
class_names = ['Potato Early Blight', 'Potato Late Blight', 'Healthy']

def preprocess_image(image):
    img = image.resize((150, 150))
    img = np.array(img) / 255.0
    img = np.expand_dims(img, axis=0)
    return img

def predict_disease(image):
    preprocessed_img = preprocess_image(image)
    prediction = model.predict(preprocessed_img)
    predicted_class = np.argmax(prediction)
    confidence = prediction[0][predicted_class] * 100
    return class_names[predicted_class], confidence

# Streamlit app
st.title('Potato Disease Detection')

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)
    st.write("")

    if st.button('Predict'):
        predicted_class, confidence = predict_disease(image)
        st.write(f'Prediction: {predicted_class}')
        st.write(f'Confidence: {confidence:.2f}%')
