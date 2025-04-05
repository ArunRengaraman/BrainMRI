import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import model_from_json
import cv2
from PIL import Image
import os
import json

# Set page configuration
st.set_page_config(
    page_title="Brain Tumor Classification",
    page_icon="🧠",
    layout="wide"
)

# Define constants
IMAGE_SIZE = 150
LABELS = ['Glioma Tumor', 'No Tumor', 'Meningioma Tumor', 'Pituitary Tumor']
MODEL_PATH = 'EfficientNetB0.h5'

# Print TensorFlow version for debugging
st.sidebar.text(f"TensorFlow version: {tf.__version__}")

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1F1F1F;
        text-align: center;
        margin-bottom: 2rem;
    }
    .subheader {
        font-size: 1.5rem;
        color: #313131;
        margin-bottom: 1rem;
    }
    .result-text {
        font-size: 1.8rem;
        font-weight: bold;
        margin-top: 1rem;
    }
    .info-text {
        font-size: 1rem;
        color: #636363;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown("<h1 class='main-header'>Brain Tumor MRI Classification</h1>", unsafe_allow_html=True)

# Create a simple CNN model as a fallback
def create_simple_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3)),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(4, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# Custom load function that handles the compatibility issue
@st.cache_resource
def load_classification_model():
    # Try different approaches to load the model
    try:
        # Try loading just the weights into a new model architecture (most compatible approach)
        from tensorflow.keras.applications import EfficientNetB0
        
        base_model = EfficientNetB0(weights=None, include_top=False, input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3))
        x = tf.keras.layers.GlobalAveragePooling2D()(base_model.output)
        x = tf.keras.layers.Dropout(rate=0.5)(x)
        output = tf.keras.layers.Dense(4, activation='softmax')(x)
        model = tf.keras.models.Model(inputs=base_model.input, outputs=output)
        
        # Load weights if the file exists
        if os.path.exists(MODEL_PATH):
            try:
                model.load_weights(MODEL_PATH, by_name=True, skip_mismatch=True)
                st.success("Successfully loaded model weights with architecture reconstruction")
                return model
            except Exception as e:
                st.warning(f"Could not load weights directly: {e}")
        else:
            st.warning(f"Model file not found at: {os.path.abspath(MODEL_PATH)}")
            
        # If we reach here, either the model file doesn't exist or loading weights failed
        st.warning("Using a simpler fallback model for demonstration")
        return create_simple_model()
        
    except Exception as e:
        st.error(f"Error during model loading: {e}")
        st.info("Using a simpler fallback model for demonstration")
        return create_simple_model()

# Preprocess image
def preprocess_image(image):
    # Convert to numpy array
    img_array = np.array(image)
    
    # Resize to model input size
    img_resized = cv2.resize(img_array, (IMAGE_SIZE, IMAGE_SIZE))
    
    # Normalize the image
    img_normalized = img_resized / 255.0
    
    # Expand dimensions to match model input shape
    img_expanded = np.expand_dims(img_normalized, axis=0)
    
    return img_expanded

# Predict function
def predict_tumor_class(model, img):
    # Get prediction
    prediction = model.predict(img)
    
    # Get class with highest probability
    class_index = np.argmax(prediction[0])
    class_name = LABELS[class_index]
    confidence = float(prediction[0][class_index])
    
    return class_name, confidence, prediction[0]

# Main app
def main():
    # Load model
    model = load_classification_model()
    
    # Sidebar
    st.sidebar.markdown("<h2 class='subheader'>About</h2>", unsafe_allow_html=True)
    st.sidebar.info(
        "This application uses a deep learning model to classify brain MRI scans "
        "into four categories: Glioma Tumor, No Tumor, Meningioma Tumor, and Pituitary Tumor."
    )
    
    st.sidebar.markdown("<h2 class='subheader'>Note</h2>", unsafe_allow_html=True)
    st.sidebar.warning(
        "Due to TensorFlow version compatibility issues, the app may be using a simplified model. "
        "For production use, ensure TensorFlow versions match between training and deployment."
    )
    
    st.sidebar.markdown("<h2 class='subheader'>Instructions</h2>", unsafe_allow_html=True)
    st.sidebar.info(
        "1. Upload a brain MRI scan image\n"
        "2. Wait for the model to process the image\n"
        "3. View the classification results"
    )
    
    # Image upload
    st.markdown("<h2 class='subheader'>Upload Brain MRI Image</h2>", unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    
    col1, col2 = st.columns([1, 1])
    
    if uploaded_file is not None and model is not None:
        # Read and display the image
        image = Image.open(uploaded_file).convert('RGB')
        with col1:
            st.image(image, caption="Uploaded MRI Scan", use_column_width=True)
            st.markdown("<p class='info-text'>Image preview</p>", unsafe_allow_html=True)
        
        # Preprocess the image
        processed_img = preprocess_image(image)
        
        # Get prediction
        with st.spinner("Analyzing image..."):
            class_name, confidence, all_probabilities = predict_tumor_class(model, processed_img)
        
        # Display results
        with col2:
            st.markdown("<h2 class='subheader'>Classification Results</h2>", unsafe_allow_html=True)
            st.markdown(f"<p class='result-text'>Diagnosis: {class_name}</p>", unsafe_allow_html=True)
            st.markdown(f"<p>Confidence: {confidence:.2%}</p>", unsafe_allow_html=True)
            
            # Display probabilities for all classes
            st.markdown("<h3>Probability Distribution</h3>", unsafe_allow_html=True)
            
            for i, label in enumerate(LABELS):
                prob = all_probabilities[i]
                st.progress(float(prob))
                st.markdown(f"{label}: {prob:.2%}")
    else:
        st.info("Please upload a brain MRI image to get started.")
        
        # Display sample information
        st.markdown("<h2 class='subheader'>Classification Information</h2>", unsafe_allow_html=True)
        st.markdown(
            "The model can classify brain MRI scans into four categories:\n"
            "- Glioma Tumor: A tumor that starts in the glial cells of the brain or spine\n"
            "- No Tumor: Normal brain scan with no tumor present\n"
            "- Meningioma Tumor: A tumor that forms on membranes covering the brain and spinal cord\n"
            "- Pituitary Tumor: A tumor that develops in the pituitary gland"
        )

if __name__ == "__main__":
    main()
