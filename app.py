import streamlit as st
import tensorflow as tf
from PIL import Image
import cv2
import numpy as np

# Load the pre-trained model with proper caching
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model('EfficientNetB0.h5')
    return model

model = load_model()

# Streamlit app
st.title('Brain Tumor Classification')
st.write('Upload an MRI scan for tumor detection')

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_container_width=True)  # Updated parameter
    
    # Preprocess the image
    opencv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    img = cv2.resize(opencv_image, (150, 150))
    img = img.reshape(1, 150, 150, 3)
    
    # Make prediction
    prediction = model.predict(img)
    p = np.argmax(prediction, axis=1)[0]
    
    # Display results
    if p == 0:
        st.success('The Model predicts that it is a Glioma Tumor')
    elif p == 1:
        st.success('The model predicts that there is no tumor')
    elif p == 2:
        st.success('The Model predicts that it is a Meningioma Tumor')
    else:
        st.success('The Model predicts that it is a Pituitary Tumor')
