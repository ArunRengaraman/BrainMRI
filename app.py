import streamlit as st
import tensorflow as tf
from PIL import Image
import cv2
import numpy as np

# Load the pre-trained model with proper caching
@st.cache_resource
def load_model():
    return tf.keras.models.load_model('Densenet121')

model = load_model()

# Custom CSS styling
st.markdown("""
    <style>
    .main {
        background-color: #F5F5F5;
    }
    h1 {
        color: #2E86C1;
        text-align: center;
    }
    .sidebar .sidebar-content {
        background-color: #EBF5FB;
    }
    .st-b7 {
        color: #2E86C1;
    }
    .diagnosis-box {
        padding: 20px;
        border-radius: 10px;
        margin: 20px 0;
        font-size: 24px;
        text-align: center;
    }
    </style>
""", unsafe_allow_html=True)

# Sidebar with information
with st.sidebar:
    st.title("ℹ️ App Information")
    st.markdown("""
    **Brain Tumor Classifier** helps identify potential tumors in MRI scans using AI.
    - Upload an MRI scan in JPG, JPEG, or PNG format
    - The model will analyze the image
    - Results will show tumor type or 'no tumor' detection
    """)
    st.markdown("---")
    st.markdown("**Supported Tumor Types:**")
    st.markdown("- Glioma Tumor\n- Meningioma Tumor\n- Pituitary Tumor")
    st.markdown("---")
    st.markdown("🩺 This tool is for research purposes only. Always consult a medical professional for diagnosis.")

# Main app content
st.title("🧠 Brain Tumor Detection AI")
st.markdown("---")

# File upload section
upload_col, info_col = st.columns([2, 1])
with upload_col:
    uploaded_file = st.file_uploader(
        "Upload MRI Scan", 
        type=["jpg", "jpeg", "png"],
        help="Select a brain MRI scan for analysis"
    )

with info_col:
    st.markdown("### 📌 Instructions")
    st.markdown("1. Upload a brain MRI scan\n2. Wait for analysis\n3. Review results")

st.markdown("---")

if uploaded_file is not None:
    # Image preview section
    with st.spinner("Processing image..."):
        image = Image.open(uploaded_file)
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Original Image**")
            st.image(image, use_container_width=True)
        
        with col2:
            st.markdown("**Processed Image**")
            opencv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            processed_img = cv2.resize(opencv_image, (150, 150))
            st.image(processed_img, use_container_width=True, clamp=True)

    # Prediction and results
    with st.spinner("Analyzing scan..."):
        img_array = processed_img.reshape(1, 150, 150, 3)
        prediction = model.predict(img_array)
        confidence = np.max(prediction)
        p = np.argmax(prediction, axis=1)[0]

    st.markdown("---")
    st.markdown("## 🔍 Analysis Results")

    # Diagnosis display with different colors
    diagnosis_box = st.container()
    with diagnosis_box:
        if p == 0:
            diagnosis = "Glioma Tumor"
            color = "#F1948A"
        elif p == 1:
            diagnosis = "No Tumor Detected"
            color = "#7DCEA0"
        elif p == 2:
            diagnosis = "Meningioma Tumor"
            color = "#85C1E9"
        else:
            diagnosis = "Pituitary Tumor"
            color = "#F7DC6F"
        
        st.markdown(f"""
        <div class="diagnosis-box" style="background-color: {color}30; border: 2px solid {color};">
            <h3 style="color: {color};">Predicted Diagnosis:</h3>
            <h2 style="color: {color};">{diagnosis}</h2>
            <p>Confidence: {confidence*100:.2f}%</p>
        </div>
        """, unsafe_allow_html=True)

    # Additional information
    with st.expander("📚 Technical Details"):
        st.markdown("""
        **Model Architecture:** EfficientNetB0  
        **Input Size:** 150x150 pixels  
        **Classes:** 4 (Glioma, Meningioma, Pituitary, No Tumor)  
        **Accuracy:** [Your Model Accuracy]  
        **Training Data:** [Your Dataset Info]
        """)

    with st.expander("📖 Interpretation Guide"):
        st.markdown("""
        - **Glioma Tumor:** Develops in the brain's glial cells
        - **Meningioma Tumor:** Affects the meninges (brain membranes)
        - **Pituitary Tumor:** Occurs in the pituitary gland
        - **No Tumor:** Healthy brain tissue
        """)

else:
    # Show upload prompt when no file is selected
    st.markdown("""
    <div style="text-align: center; padding: 50px 20px; border: 2px dashed #2E86C1; border-radius: 10px;">
        <h3 style="color: #2E86C1;">⬆️ Upload an MRI Scan to Begin Analysis</h3>
        <p>Supported formats: JPG, JPEG, PNG</p>
    </div>
    """, unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; font-size: 0.9em;">
    <p>This AI diagnostic tool provides preliminary analysis and should not be used as a substitute for professional medical advice.</p>
    <p>Developed with ❤️</p>
</div>
""", unsafe_allow_html=True)
