import streamlit as st
import tensorflow as tf
from PIL import Image
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Model

# Load the pre-trained models with proper caching
@st.cache_resource
def load_model(model_name):
    model_files = {
     "EfficientNetB0": "EfficientNetB0.h5",
     "ResNet50": "ResNet50.h5",
     "DenseNet121": "DenseNet121.h5",
     "MobileNetV2": "MobileNetV2.h5"
 }
    
    try:
        model_path = model_files[model_name]
        return tf.keras.models.load_model(model_path)
    except KeyError:
        st.error(f"Invalid model name: {model_name}")
        return None
    except OSError as e:
        st.error(f"""
            Model file not found or corrupted: {model_path}
            Error: {str(e)}
            Please ensure model files are in the correct directory
        """)
        return None
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

# Grad-CAM implementation
def grad_cam(model, img_array, layer_name, pred_index=None):
    grad_model = Model(
        inputs=model.inputs,
        outputs=[model.get_layer(layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(predictions[0])
        class_channel = predictions[:, pred_index]

    grads = tape.gradient(class_channel, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()

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
    .heatmap-caption {
        font-size: 0.8em;
        color: #666;
        text-align: center;
        margin-top: -15px;
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

# Model selection
model_options = ["EfficientNetB0", "ResNet50", "DenseNet121", "MobileNetV2"]
selected_model = st.selectbox("Select Model", model_options)

# Define the appropriate layer name for each model
layer_names = {
    "EfficientNetB0": "top_conv",
    "ResNet50": "conv5_block3_out",
    "DenseNet121": "conv5_block16_concat",
    "MobileNetV2": "block_16_project"  # Typical last conv layer for MobileNetV2
}

# Load the selected model
model = load_model(selected_model)

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
    # Image processing and prediction
    with st.spinner("Processing image..."):
        image = Image.open(uploaded_file)
        opencv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        processed_img = cv2.resize(opencv_image, (224, 224))  # Changed to 224x224
        img_array = processed_img.reshape(1, 224, 224, 3)     # Updated shape
        
        # Generate prediction and heatmap
        prediction = model.predict(img_array)
        confidence = np.max(prediction)
        p = np.argmax(prediction, axis=1)[0]
        
        try:
            heatmap = grad_cam(model, img_array, layer_names[selected_model])
            heatmap = cv2.resize(heatmap, (224, 224))         # Updated to 224x224
            heatmap = np.uint8(255 * heatmap)
            heatmap_img = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
            superimposed_img = cv2.addWeighted(processed_img, 0.6, heatmap_img, 0.4, 0)
        except Exception as e:
            st.error(f"Could not generate explanation: {str(e)}")
            superimposed_img = processed_img

    # Image display columns
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("**Original Image**")
        st.image(image, use_container_width=True)
    
    with col2:
        st.markdown("**Processed Image**")
        st.image(processed_img, use_container_width=True, clamp=True)
    
    with col3:
        st.markdown("**Model Attention Map**")
        st.image(superimposed_img, use_container_width=True, clamp=True)
        st.markdown('<div class="heatmap-caption">Red areas show regions influencing prediction</div>', 
                    unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("## 🔍 Analysis Results")

    # Diagnosis display
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

    # Additional information sections
    with st.expander("📚 Technical Details"):
        st.markdown(f"""
        **Model Architecture:** {selected_model}  
        **Input Size:** 150x150 pixels  
        **Classes:** 4 (Glioma, Meningioma, Pituitary, No Tumor)  
        **Explanation Method:** Grad-CAM (Gradient-weighted Class Activation Mapping)  
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

    with st.expander("🔍 How to Read the Heatmap"):
        st.markdown("""
        **The color overlay shows regions that influenced the model's prediction:**
        - 🔴 **Red Areas:** High model attention
        - 🟢 **Green Areas:** Moderate attention
        - 🔵 **Blue Areas:** Low attention
        - The model focuses on biologically relevant patterns
        - Heatmap helps verify model's focus areas
        """)

else:
    # Upload prompt
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
    <p>Developed with ❤️ using Streamlit | Model explainability powered by Grad-CAM</p>
</div>
""", unsafe_allow_html=True)
