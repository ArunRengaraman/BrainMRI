import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import cv2
from PIL import Image
import io

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

# Load the model
@st.cache_resource
def load_classification_model():
    try:
        model = load_model(MODEL_PATH)
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

# Image prediction function
def img_pred(uploaded_file, model):
    # Read and process image
    img = Image.open(io.BytesIO(uploaded_file.read())).convert('RGB')
    opencvImage = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    img = cv2.resize(opencvImage, (IMAGE_SIZE, IMAGE_SIZE))
    img = img.reshape(1, IMAGE_SIZE, IMAGE_SIZE, 3)
    
    # Normalize image
    img = img / 255.0
    
    # Get prediction
    prediction = model.predict(img)
    pred_index = np.argmax(prediction, axis=1)[0]
    
    # Get probabilities for all classes
    probabilities = prediction[0]
    
    # Determine result
    if pred_index == 0:
        result = 'Glioma Tumor'
    elif pred_index == 1:
        result = 'No Tumor'
    elif pred_index == 2:
        result = 'Meningioma Tumor'
    else:
        result = 'Pituitary Tumor'
        
    confidence = float(probabilities[pred_index])
    return result, confidence, probabilities

# Main app
def main():
    # Load model
    model = load_classification_model()
    
    # Sidebar
    st.sidebar.markdown("<h2 class='subheader'>About</h2>", unsafe_allow_html=True)
    st.sidebar.info(
        "This application uses EfficientNetB0 to classify brain MRI scans "
        "into four categories: Glioma Tumor, No Tumor, Meningioma Tumor, and Pituitary Tumor."
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
        # Display the image
        image = Image.open(uploaded_file).convert('RGB')
        uploaded_file.seek(0)  # Reset file pointer after reading
        
        with col1:
            st.image(image, caption="Uploaded MRI Scan", use_column_width=True)
            st.markdown("<p class='info-text'>Image preview</p>", unsafe_allow_html=True)
        
        # Get prediction
        with st.spinner("Analyzing image..."):
            class_name, confidence, probabilities = img_pred(uploaded_file, model)
        
        # Display results
        with col2:
            st.markdown("<h2 class='subheader'>Classification Results</h2>", unsafe_allow_html=True)
            st.markdown(f"<p class='result-text'>Diagnosis: {class_name}</p>", unsafe_allow_html=True)
            st.markdown(f"<p>Confidence: {confidence:.2%}</p>", unsafe_allow_html=True)
            
            # Display probabilities
            st.markdown("<h3>Probability Distribution</h3>", unsafe_allow_html=True)
            for i, label in enumerate(LABELS):
                prob = probabilities[i]
                st.progress(float(prob))
                st.markdown(f"{label}: {prob:.2%}")
                
    elif model is None:
        st.error("Model could not be loaded. Please ensure the model file is available.")
    else:
        st.info("Please upload a brain MRI image to get started.")
        
        # Display sample information
        st.markdown("<h2 class='subheader'>Classification Information</h2>", unsafe_allow_html=True)
        st.markdown(
            "The model can classify brain MRI scans into:\n"
            "- Glioma Tumor: Originates in glial cells\n"
            "- No Tumor: Normal brain scan\n"
            "- Meningioma Tumor: Forms on brain/spinal cord membranes\n"
            "- Pituitary Tumor: Develops in pituitary gland"
        )

if __name__ == "__main__":
    main()
