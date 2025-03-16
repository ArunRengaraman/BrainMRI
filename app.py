import streamlit as st
import numpy as np
import cv2
import os
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras import backend as K
import tensorflow as tf
from PIL import Image

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
    "DensenetModel": "densenet121.h5",
    "EfficientNet": "EfficientNetB0.h5"
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
        image (numpy.ndarray): The uploaded image in grayscale or RGB format.
        target_size (tuple): The target size for the image.

    Returns:
        numpy.ndarray: The preprocessed image.
    """
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, target_size)
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    image = image / 255.0  # Normalize pixel values to [0, 1]
    return image

# Function to get the Grad-CAM heatmap
def get_gradcam_heatmap(model, img_array, last_conv_layer_name=None):
    """
    Generates a Grad-CAM heatmap for model explainability.

    Args:
        model: Trained model.
        img_array: Preprocessed image.
        last_conv_layer_name: Name of the last convolutional layer.

    Returns:
        numpy.ndarray: Grad-CAM heatmap.
    """
    if last_conv_layer_name is None:
        # Automatically find the last convolutional layer
        for layer in reversed(model.layers):
            if len(layer.output_shape) == 4:  # Look for Conv2D layers
                last_conv_layer_name = layer.name
                break

    if not last_conv_layer_name:
        st.error("⚠️ Could not determine the last convolutional layer for Grad-CAM.")
        return None

    grad_model = tf.keras.models.Model(
        [model.inputs], [model.get_layer(last_conv_layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        class_index = tf.argmax(predictions[0])
        loss = predictions[:, class_index]

    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]

    heatmap = tf.reduce_mean(tf.multiply(pooled_grads, conv_outputs), axis=-1)
    heatmap = np.maximum(heatmap, 0)
    heatmap /= np.max(heatmap)

    return heatmap.numpy()

# App header
st.title("🧠 Brain Tumor Detection")
st.markdown("""
Welcome to the **Brain Tumor Detection App**!  
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

            # Detailed explanation
            if predicted_class == 0:
                st.success("No tumor detected! Keep up with regular health check-ups.")
            elif predicted_class == 1:
                st.warning("A benign tumor detected. Please consult a doctor for further evaluation.")
            elif predicted_class == 2:
                st.error("A malignant tumor detected. Immediate medical attention is advised.")
            elif predicted_class == 3:
                st.error("A pituitary tumor detected. Consult a neurologist for further treatment.")

        # Generate and display Grad-CAM heatmap
        st.markdown("### 🔥 Grad-CAM Visualization")
        heatmap = get_gradcam_heatmap(model, processed_image)

        if heatmap is not None:
            # Resize heatmap to match original image
            heatmap = cv2.resize(heatmap, (image.shape[1], image.shape[0]))
            heatmap = np.uint8(255 * heatmap)
            heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
            superimposed_img = cv2.addWeighted(image, 0.6, heatmap, 0.4, 0)

            # Display Grad-CAM heatmap
            st.image(superimposed_img, caption="Grad-CAM Heatmap", use_column_width=True)
        else:
            st.error("⚠️ Could not generate Grad-CAM heatmap.")

    else:
        st.error("⚠️ Model failed to load. Please try again later.")

# Footer
st.markdown("---")
st.markdown("_This app is for educational purposes only. For medical advice, please consult a professional doctor._")
