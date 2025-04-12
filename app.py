import streamlit as st
import tensorflow as tf
from PIL import Image
import cv2
import numpy as np
import gdown  # Added gdown import
from tensorflow.keras.models import Model
import io
import base64
from fpdf import FPDF
import os
import datetime

# Configure gdown
gdown.download_folder = "cache"

# Load models from Google Drive with caching
@st.cache_resource
def load_model(model_name):
    model_files = {
        "EfficientNetB0": "https://drive.google.com/uc?id=1gev_WR1p0Y9FPu4cpgQUcV-vF-r7cBPo",
        "ResNet50": "https://drive.google.com/uc?id=14hqNm2bNiNo3Q9oMPpXxiEj0-JBuPa7M",
        "DenseNet121": "https://drive.google.com/uc?id=1y28fghxyJ0x81pDz5ck_cW7-BW2-4j74",
        "Xception": "https://drive.google.com/uc?id=1IbnLehzpsceOqE92tNBSnLJotY9mGiDQ",
        "InceptionV3": "https://drive.google.com/uc?id=1g3aL3PYj-wYNhvOe1zKbrzVDyHGjc8lF",
        "MobileNetV2": "https://drive.google.com/uc?id=1TYqJh4zDCT5EqQzgajtT6JceryDGeJYp"
    }

    try:
        model_url = model_files[model_name]
        output_path = f"/tmp/{model_name}.h5"  # Using /tmp for write permissions
        
        # Download model file
        gdown.download(model_url, output_path, quiet=False)
        
        # Load and return model
        return tf.keras.models.load_model(output_path)
        
    except KeyError:
        st.error(f"Invalid model name: {model_name}")
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

# PDF Report Generation Function
def generate_pdf_report(uploaded_image, processed_img, heatmap_img, diagnosis, confidence, model_name):
    class PDF(FPDF):
        def header(self):
            self.set_font('Arial', 'B', 12)
            self.cell(0, 10, 'Brain Tumor Detection AI - Analysis Report', 0, 1, 'C')
            self.ln(5)
            
        def footer(self):
            self.set_y(-15)
            self.set_font('Arial', 'I', 8)
            self.cell(0, 10, f'Page {self.page_no()}/{{nb}}', 0, 0, 'C')
            self.cell(0, 10, f'Generated on {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}', 0, 0, 'R')
    
    # Create PDF object
    pdf = PDF()
    pdf.alias_nb_pages()
    pdf.add_page()
    
    # Add title
    pdf.set_font('Arial', 'B', 16)
    pdf.cell(0, 10, 'Brain MRI Analysis Report', 0, 1, 'C')
    pdf.line(10, 30, 200, 30)
    pdf.ln(10)
    
    # Add report details
    pdf.set_font('Arial', '', 12)
    pdf.cell(0, 10, f'Report Date: {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}', 0, 1)
    pdf.cell(0, 10, f'Model Used: {model_name}', 0, 1)
    pdf.ln(5)
    
    # Add diagnosis section
    pdf.set_font('Arial', 'B', 14)
    pdf.cell(0, 10, 'Diagnosis Results:', 0, 1)
    pdf.set_font('Arial', 'B', 12)
    
    # Set diagnosis color
    if "Glioma" in diagnosis:
        pdf.set_text_color(231, 76, 60)  # Red for Glioma
    elif "No Tumor" in diagnosis:
        pdf.set_text_color(46, 204, 113)  # Green for No Tumor
    elif "Meningioma" in diagnosis:
        pdf.set_text_color(52, 152, 219)  # Blue for Meningioma
    else:
        pdf.set_text_color(241, 196, 15)  # Yellow for Pituitary
    
    pdf.cell(0, 10, f'Predicted Diagnosis: {diagnosis}', 0, 1)
    pdf.set_text_color(0, 0, 0)  # Reset text color to black
    pdf.set_font('Arial', '', 12)
    pdf.cell(0, 10, f'Confidence: {confidence*100:.2f}%', 0, 1)
    pdf.ln(5)
    
    # Save images to temporary files
    temp_dir = "/tmp"
    os.makedirs(temp_dir, exist_ok=True)
    
    # Convert images to PIL format and save
    original_pil = Image.fromarray(np.array(uploaded_image))
    original_path = f"{temp_dir}/original.png"
    original_pil.save(original_path)
    
    processed_pil = Image.fromarray(cv2.cvtColor(processed_img, cv2.COLOR_BGR2RGB))
    processed_path = f"{temp_dir}/processed.png"
    processed_pil.save(processed_path)
    
    heatmap_pil = Image.fromarray(cv2.cvtColor(heatmap_img, cv2.COLOR_BGR2RGB))
    heatmap_path = f"{temp_dir}/heatmap.png"
    heatmap_pil.save(heatmap_path)
    
    # Add images to PDF
    pdf.set_font('Arial', 'B', 14)
    pdf.cell(0, 10, 'Analysis Images:', 0, 1)
    
    # Add original image
    pdf.set_font('Arial', 'B', 12)
    pdf.cell(0, 10, 'Original MRI Scan:', 0, 1)
    pdf.image(original_path, x=50, w=110)
    pdf.ln(5)
    
    # Add processed image
    pdf.cell(0, 10, 'Processed Image:', 0, 1)
    pdf.image(processed_path, x=50, w=110)
    pdf.ln(5)
    
    # Add heatmap image
    pdf.cell(0, 10, 'Model Attention Map:', 0, 1)
    pdf.image(heatmap_path, x=50, w=110)
    pdf.set_font('Arial', 'I', 10)
    pdf.cell(0, 10, 'Red areas show regions influencing prediction', 0, 1, 'C')
    pdf.ln(5)
    
    # Add detailed analysis section
    pdf.add_page()
    pdf.set_font('Arial', 'B', 14)
    pdf.cell(0, 10, 'Detailed Analysis:', 0, 1)
    pdf.set_font('Arial', '', 12)
    
    # Add tumor type information based on diagnosis
    if "Glioma" in diagnosis:
        pdf.multi_cell(0, 10, 'Glioma Tumor: Gliomas are tumors that develop from glial cells in the brain. They can be low-grade (slow growing) or high-grade (fast growing). They typically affect the cerebrum, brain stem, or cerebellum and may cause symptoms including headaches, seizures, and cognitive changes depending on their location and size.')
    elif "Meningioma" in diagnosis:
        pdf.multi_cell(0, 10, 'Meningioma Tumor: Meningiomas arise from the meninges, the membranes that surround the brain and spinal cord. They are typically slow-growing and often benign. Symptoms vary based on tumor location and may include headaches, vision problems, and hearing loss. Many meningiomas can be successfully treated with surgery.')
    elif "Pituitary" in diagnosis:
        pdf.multi_cell(0, 10, 'Pituitary Tumor: Pituitary tumors develop in the pituitary gland at the base of the brain. They can affect hormone production and may cause symptoms including vision problems, headaches, and various hormonal imbalances. Treatment depends on size, type, and hormone activity.')
    else:
        pdf.multi_cell(0, 10, 'No Tumor Detected: The analysis suggests no evidence of tumor in the provided MRI scan. The brain structures appear within normal limits based on the AI model\'s evaluation.')
    
    pdf.ln(5)
    
    # Add model interpretation section
    pdf.set_font('Arial', 'B', 14)
    pdf.cell(0, 10, 'Model Interpretation Guide:', 0, 1)
    pdf.set_font('Arial', '', 12)
    pdf.multi_cell(0, 10, 'The attention map highlights regions that influenced the model\'s prediction. Red areas indicate high model attention, while green and blue areas show moderate and low attention respectively. This visualization helps verify that the model is focusing on biologically relevant patterns in the MRI scan.')
    pdf.ln(5)
    
    # Add disclaimer
    pdf.set_font('Arial', 'B', 12)
    pdf.set_text_color(192, 57, 43)  # Red color for disclaimer
    pdf.cell(0, 10, 'IMPORTANT DISCLAIMER:', 0, 1)
    pdf.set_font('Arial', 'I', 10)
    pdf.multi_cell(0, 10, 'This report is generated by an AI system for research purposes only. It is not a substitute for professional medical diagnosis. Always consult with a qualified healthcare professional for proper diagnosis and treatment decisions.')
    pdf.set_text_color(0, 0, 0)  # Reset text color
    
    # Create binary stream to hold PDF data
    pdf_output = io.BytesIO()
    pdf.output(pdf_output)
    pdf_data = pdf_output.getvalue()
    
    # Clean up temp files
    for file_path in [original_path, processed_path, heatmap_path]:
        if os.path.exists(file_path):
            os.remove(file_path)
    
    return pdf_data

# Function to create a download link
def get_download_link(pdf_data, filename="brain_tumor_analysis_report.pdf"):
    b64_pdf = base64.b64encode(pdf_data).decode()
    href = f'<a href="data:application/pdf;base64,{b64_pdf}" download="{filename}">Download PDF Report</a>'
    return href

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
    .report-btn {
        background-color: #2E86C1;
        color: white;
        font-weight: bold;
        padding: 10px 20px;
        border-radius: 5px;
        text-align: center;
        margin: 20px auto;
        display: block;
    }
    .download-link {
        text-align: center;
        padding: 15px;
        background-color: #E8F8F5;
        border-radius: 5px;
        margin: 20px 0;
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
    - Generate a detailed PDF report
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
model_options = ["EfficientNetB0","ResNet50", "DenseNet121", "Xception", "InceptionV3", "MobileNetV2"]
selected_model = st.selectbox("Select Model", model_options)

# Define the appropriate layer name for each model
layer_names = {
    "EfficientNetB0": "top_conv",
    "ResNet50": "conv5_block3_concat",
    "DenseNet121": "conv5_block16_concat",
    "MobileNetV2": "block_16_project",
    "Xception": "block14_sepconv2_act", 
    "InceptionV3": "mixed10" 
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
    st.markdown("1. Upload a brain MRI scan\n2. Wait for analysis\n3. Review results\n4. Generate PDF report")

st.markdown("---")

# Initialize session state for PDF report
if 'pdf_report' not in st.session_state:
    st.session_state.pdf_report = None

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

    # Report generation button
    if st.button("Generate Report", key="generate_report"):
        with st.spinner("Generating detailed PDF report..."):
            # Generate the PDF report
            pdf_data = generate_pdf_report(
                image, 
                processed_img, 
                superimposed_img, 
                diagnosis, 
                confidence, 
                selected_model
            )
            st.session_state.pdf_report = pdf_data
            
            # Show success message
            st.success("PDF report generated successfully!")
    
    # Show download link if report is generated
    if st.session_state.pdf_report:
        st.markdown(
            f'<div class="download-link">{get_download_link(st.session_state.pdf_report)}</div>', 
            unsafe_allow_html=True
        )

    # Additional information sections
    with st.expander("📚 Technical Details"):
        st.markdown(f"""
        **Model Architecture:** {selected_model}  
        **Input Size:** 224x224 pixels  
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
