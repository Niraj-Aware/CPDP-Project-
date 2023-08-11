import os
import numpy as np
import streamlit as st
from tensorflow.keras.preprocessing import image
import tensorflow as tf
from gevent.pywsgi import WSGIServer

# Load the Keras model
model = tf.keras.models.load_model('PlantDNet.h5', compile=False)
disease_class = ['Pepper__bell___Bacterial_spot', 'Pepper__bell___healthy', 'Potato___Early_blight',
                 'Potato___Late_blight', 'Potato___healthy', 'Tomato_Bacterial_spot', 'Tomato_Early_blight',
                 'Tomato_Late_blight', 'Tomato_Leaf_Mold', 'Tomato_Septoria_leaf_spot',
                 'Tomato_Spider_mites_Two_spotted_spider_mite', 'Tomato__Target_Spot',
                 'Tomato__Tomato_YellowLeaf__Curl_Virus', 'Tomato__Tomato_mosaic_virus', 'Tomato_healthy']

# Streamlit UI
st.title("Plant Disease Prediction")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    st.image(uploaded_file, caption="Uploaded Image.", use_column_width=True)
    st.write("")
    st.write("Classifying...")

    img = image.load_img(uploaded_file, grayscale=False, target_size=(64, 64))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = np.array(x, 'float32')
    x /= 255
    preds = model.predict(x)
    
    ind = np.argmax(preds[0])
    result = disease_class[ind]
    st.write(f"Prediction: {result}")
