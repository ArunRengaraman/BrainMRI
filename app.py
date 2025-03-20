import streamlit as st
import numpy as np
import cv2
import os
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image

# Streamlit app configuration
st.set_page_config(
    page_title="Brain Tumor Detection",
    page_icon="🧠",
    layout="wide"
)

# Constants
DEFAULT_IMAGE_SIZE = 224  # Default size, will adjust based on model
CLASSES = ['No Tumor', 'Benign Tumor', 'Malignant Tumor', 'Pituitary Tumor']

# Load models (Ensure correct paths)
MODEL_PATHS = {
    "DensenetModel": "densenet121.h5",  # Update with actual model path
    "EfficientNet": "EfficientNetB0_New.h5"  # Update with actual model path
}

# Verify model file existence
for model_name, path in MODEL_PATHS.items():
    if not os.path.exists(path):
        st.sidebar.error(f"🚨 Model file not found: {path}")

# Sidebar for model selection
st.sidebar.header("Model Selection")
selected_model_name = st.sidebar.selectbox(
    "Choose a model for prediction:",
    list(MODEL_PATHS.keys())
)

# Load the selected model
@st.cache_resource
def load_selected_model(model_name):
    try:
        model = load_model(MODEL_PATHS[model_name])
        st.write(f"Model input shape for {model_name}: {model.input_shape}")  # Debug line
        return model
    except Exception as e:
        st.error(f"⚠️ Error loading model: {e}")
        return None  # Return None to handle errors

model = load_selected_model(selected_model_name)

# Function to preprocess the image
def preprocess_image(image, target_size=(224, 224)):
    """
    Preprocesses an uploaded MRI image for model prediction.

    Args:
        image (numpy.ndarray): The uploaded image in grayscale or RGB format.
        target_size (tuple): The target size for the image.

    Returns:
        numpy.ndarray: The preprocessed image.
    """
    # Handle grayscale or single-channel images
    if len(image.shape) == 2 or (len(image.shape) == 3 and image.shape[2] == 1):
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    else:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Resize and preprocess
    image = cv2.resize(image, target_size)
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    image = image / 255.0  # Normalize pixel values to [0, 1]
    st.write(f"Processed image shape: {image.shape}")  # Debug line
    return image

# App header and introduction
st.title("🧠 Brain Tumor Detection")
st.markdown(""" 
Welcome to the **Brain Tumor Detection App**!  
Upload an **MRI scan** to detect the presence and type of brain tumor using an AI-powered deep learning model.
""")

# Sidebar information
st.sidebar.header("About Brain Tumors")
st.sidebar.write("""
Brain tumors are abnormal growths in the brain and can be categorized as:
- **No Tumor**: No presence of abnormal growth.
- **Benign Tumor**: Non-cancerous growths.
- **Malignant Tumor**: Cancerous tumors requiring immediate attention.
- **Pituitary Tumor**: Tumors affecting the pituitary gland.

MRI scans are the best technique for tumor detection. AI-based models can assist doctors in faster and more accurate diagnostics.
""")

# File uploader
uploaded_file = st.file_uploader("📂 Upload an MRI image (JPEG/PNG)...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Read and decode the image safely
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    # Validate image
    if image is None:
        st.error("⚠️ The uploaded file is not a valid image. Please upload a valid MRI scan.")
        st.stop()
    
    # Check resolution (minimum size will be dynamically set by model)
    if model and (image.shape[0] < model.input_shape[1] or image.shape[1] < model.input_shape[2]):
        st.error(f"⚠️ Please upload an MRI scan with at least {model.input_shape[1]}x{model.input_shape[2]} pixels.")
        st.stop()
    elif not model and (image.shape[0] < DEFAULT_IMAGE_SIZE or image.shape[1] < DEFAULT_IMAGE_SIZE):
        st.error(f"⚠️ Please upload an MRI scan with at least {DEFAULT_IMAGE_SIZE}x{DEFAULT_IMAGE_SIZE} pixels.")
        st.stop()
    
    # Display the uploaded image
    col1, col2 = st.columns([1, 1])
    with col1:
        st.image(image, caption="📷 Uploaded MRI Scan", use_column_width=True)
    
    # Preprocess the image based on model input shape
    if model:
        target_size = model.input_shape[1:3]  # Extract height and width (e.g., (224, 224))
        processed_image = preprocess_image(image, target_size=target_size)
        
        # Make prediction
        try:
            prediction = model.predict(processed_image)
            predicted_class = np.argmax(prediction[0])
            
            # Display results
            with col2:
                st.markdown("## 🏆 Prediction Results")
                st.write(f"### **Model Used:** {selected_model_name}")
                st.write(f"### **Predicted Category:** {CLASSES[predicted_class]}")
                
                # Detailed explanation
                if predicted_class == 0:
                    st.success("No tumor detected! Keep up with regular health check-ups.")
                elif predicted_class == 1:
                    st.warning("A benign tumor detected. Please consult a doctor for further evaluation.")
                elif predicted_class == 2:
                    st.error("A malignant tumor detected. Immediate medical attention is advised.")
                elif predicted_class == 3:
                    st.error("A pituitary tumor detected. Consult a neurologist for further treatment.")
        
        except Exception as e:
            st.error(f"⚠️ Prediction failed: {e}")
    else:
        st.error("⚠️ Model failed to load. Please try again later.")

    # Future enhancements
    st.markdown("---")
    st.markdown("### 🧪 Future Enhancements")
    st.write("Further analysis such as tumor segmentation and 3D visualization can improve diagnostics.")

# Add footer
st.markdown("---")
st.markdown("_This app is for educational purposes only. For medical advice, please consult a professional doctor._")
