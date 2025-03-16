import streamlit as st
import numpy as np
import cv2
import os
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image
import matplotlib.pyplot as plt

# Streamlit app configuration
st.set_page_config(
    page_title="Brain Tumor Detection",
    page_icon="🧠",
    layout="wide"
)

# Constants
IMAGE_SIZE = 224
CLASSES = ['No Tumor', 'Benign Tumor', 'Malignant Tumor', 'Pituitary Tumor']

# Load models (Ensure correct paths)
MODEL_PATHS = {
    "DensenetModel": "densenet121.h5",  # Update with actual model path
    "EfficientNet": "EfficientNetB0.h5"  # Update with actual model path
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
        return load_model(MODEL_PATHS[model_name])
    except Exception as e:
        st.error(f"⚠️ Error loading model: {e}")
        return None

model = load_selected_model(selected_model_name)

# Function to preprocess the image
def preprocess_image(image, target_size=(224, 224)):
    """
    Preprocesses an uploaded MRI image for model prediction.

    Args:
        image (numpy.ndarray): The uploaded image.
        target_size (tuple): The target size for the image.

    Returns:
        numpy.ndarray: The preprocessed image.
    """
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, target_size)
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    image = image / 255.0
    return image

# Grad-CAM function
def get_gradcam_heatmap(model, img_array, layer_name="conv5_block16_concat"):
    """
    Generates a Grad-CAM heatmap for visualizing model focus areas.

    Args:
        model (tf.keras.Model): The trained model.
        img_array (numpy.ndarray): Preprocessed input image.
        layer_name (str): The convolutional layer for Grad-CAM.

    Returns:
        numpy.ndarray: Heatmap of the same size as the input image.
    """
    grad_model = tf.keras.models.Model(
        [model.inputs], [model.get_layer(layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        class_idx = np.argmax(predictions[0])
        loss = predictions[:, class_idx]

    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_outputs = conv_outputs.numpy()[0]
    pooled_grads = pooled_grads.numpy()

    for i in range(pooled_grads.shape[-1]):
        conv_outputs[:, :, i] *= pooled_grads[i]

    heatmap = np.mean(conv_outputs, axis=-1)
    heatmap = np.maximum(heatmap, 0)
    heatmap /= np.max(heatmap)

    return heatmap

# Function to find bounding box from heatmap
def get_bounding_box(heatmap, threshold=0.6):
    heatmap = cv2.resize(heatmap, (224, 224))
    heatmap = np.uint8(255 * heatmap)
    _, thresh = cv2.threshold(heatmap, int(255 * threshold), 255, cv2.THRESH_BINARY)
    
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        x, y, w, h = cv2.boundingRect(max(contours, key=cv2.contourArea))
        return x, y, w, h
    return None

# Function to overlay heatmap and draw bounding box
def overlay_heatmap(img, heatmap, alpha=0.5):
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    overlayed_img = cv2.addWeighted(img, 1 - alpha, heatmap, alpha, 0)

    bbox = get_bounding_box(heatmap)
    if bbox:
        x, y, w, h = bbox
        overlayed_img = cv2.rectangle(overlayed_img, (x, y), (x + w, y + h), (0, 255, 0), 2)  # Green Box
    return overlayed_img

# App header and introduction
st.title("🧠 Brain Tumor Detection")
st.markdown(""" 
Upload an **MRI scan** to detect the presence and type of brain tumor using an AI-powered deep learning model.
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
    elif image.shape[0] < 224 or image.shape[1] < 224:
        st.error("⚠️ Please upload a higher resolution MRI scan (at least 224x224 pixels).")
        st.stop()
    
    # Display the uploaded image
    col1, col2 = st.columns([1, 1])
    with col1:
        st.image(image, caption="📷 Uploaded MRI Scan", use_column_width=True)
    
    # Preprocess the image
    processed_image = preprocess_image(image, target_size=(IMAGE_SIZE, IMAGE_SIZE))
    
    # Ensure model is loaded before predicting
    if model:
        prediction = model.predict(processed_image)
        predicted_class = np.argmax(prediction[0])
        
        # Display results
        with col2:
            st.markdown("## 🏆 Prediction Results")
            st.write(f"### **Model Used:** {selected_model_name}")
            st.write(f"### **Predicted Category:** {CLASSES[predicted_class]}")

        # Generate Grad-CAM
        heatmap = get_gradcam_heatmap(model, processed_image)

        # Convert PIL image to OpenCV format
        img_cv = cv2.resize(image, (224, 224))
        img_cv = cv2.cvtColor(img_cv, cv2.COLOR_RGB2BGR)

        overlayed_img = overlay_heatmap(img_cv, heatmap)

        # Display Grad-CAM results
        st.image(overlayed_img, caption="Grad-CAM Visualization", use_column_width=True)

        # Legend
        st.write("""
        **Legend:**  
        🟩 **Green Box** → Predicted Tumor Region  
        🔴 **Red Areas** → Model's Focused Attention  
        """)

    else:
        st.error("⚠️ Model failed to load. Please try again later.")

# Footer
st.markdown("---")
st.markdown("_This app is for educational purposes only. For medical advice, please consult a professional doctor._")
