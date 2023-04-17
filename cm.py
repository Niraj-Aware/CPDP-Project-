import streamlit as st
from pymongo import MongoClient
import numpy as np
from PIL import Image
import tensorflow as tf

# Connect to MongoDB
client = MongoClient("mongodb+srv://Niraj_Aware:Test123@db.psuxghr.mongodb.net/test")
db = client["user_login"]
collection = db["users"]

# Load pre-trained model
model = tf.keras.models.load_model('v3_pred_cott_dis.h5')

# Define labels for prediction output
labels = {
    0: 'Alternaria Alternata',
    1: 'Anthracnose',
    2: 'Bacterial Blight',
    3: 'Corynespora Leaf Fall',
    4: 'Healthy',
    5: 'Grey Mildew'
}

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

# Define function to authenticate user
def authenticate_user(username, password):
    user = collection.find_one({"username": username, "password": password})
    if user:
        return True
    else:
        return False

# Define Streamlit app
def main():
    # Set app title
    st.title('Cotton Plant Disease Detection Login')

    # Connect to MongoDB
    client = pymongo.MongoClient(MONGO_URI)
    db = client[MONGO_DB]
    users_collection = db['users']

    # Add login form
    st.subheader('Login to access the app')
    username = st.text_input('Username')
    password = st.text_input('Password', type='password')

    if st.button('Login'):
        # Check if username exists in database
        user = users_collection.find_one({'username': username})
        if user:
            # Check if password is correct
            if bcrypt.checkpw(password.encode('utf-8'), user['password']):
                # Store user information in session state
                session_state.user_id = str(user['_id'])
                session_state.username = username
                # Redirect to the next page
                page = st.session_state.get('page', 'home')
                if page == 'home':
                    home_page()
                elif page == 'predict':
                    predict_page()
            else:
                st.error('Incorrect password')
        else:
            st.error('Username not found')
