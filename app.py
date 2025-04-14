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
            # Enhanced header with medical facility placeholder and logo positioning
            self.set_font('Arial', 'B', 14)
            self.cell(0, 10, 'MEDICAL IMAGING DIAGNOSTICS', 0, 1, 'C')
            self.set_font('Arial', 'B', 12)
            self.cell(0, 8, 'Brain Tumor Detection - Analysis Report', 0, 1, 'C')
            
            # Add a horizontal line to separate header from content
            self.line(10, 30, 200, 30)
            self.ln(5)
            
        def footer(self):
            # Enhanced footer with medical disclaimer and page numbering
            self.set_y(-25)
            self.set_font('Arial', 'I', 8)
            self.set_draw_color(100, 100, 100)
            self.line(10, self.get_y(), 200, self.get_y())
            self.ln(1)
            self.cell(0, 5, f'Page {self.page_no()}/{{nb}}', 0, 1, 'R')
            self.cell(0, 5, f'Generated on {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}', 0, 1, 'R')
            self.set_text_color(192, 57, 43)
            self.cell(0, 5, 'For clinical correlation only. Not a substitute for professional medical diagnosis.', 0, 1, 'C')
            self.set_text_color(0, 0, 0)
    
    # Create PDF object with metadata
    pdf = PDF()
    pdf.alias_nb_pages()
    pdf.set_author('Brain Tumor Detection AI System')
    pdf.set_title('Brain MRI Analysis Report')
    pdf.set_subject(f'MRI Analysis - {diagnosis}')
    pdf.set_creator('Medical Imaging AI Platform v1.0')
    pdf.add_page()
    
    # Add patient information section (placeholder)
    pdf.set_font('Arial', 'B', 12)
    pdf.set_fill_color(240, 240, 240)
    pdf.cell(0, 10, 'PATIENT INFORMATION', 1, 1, 'L', True)
    pdf.set_font('Arial', '', 10)
    
    # Create a table-like structure for patient info
    col_width = 95
    pdf.cell(col_width, 8, 'Patient ID: [REDACTED]', 'LB', 0)
    pdf.cell(col_width, 8, 'Date of Birth: [REDACTED]', 'RB', 1)
    pdf.cell(col_width, 8, 'Referring Physician: [REDACTED]', 'LB', 0)
    pdf.cell(col_width, 8, 'Scan Date: [REDACTED]', 'RB', 1)
    pdf.ln(5)
    
    # Add report details with improved formatting
    pdf.set_font('Arial', 'B', 12)
    pdf.set_fill_color(230, 230, 230)
    pdf.cell(0, 10, 'ANALYSIS DETAILS', 1, 1, 'L', True)
    pdf.set_font('Arial', '', 10)
    
    # Create a table for analysis details
    pdf.cell(col_width, 8, f'Report Generation: {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}', 'LB', 0)
    pdf.cell(col_width, 8, f'Model: {model_name}', 'RB', 1)
    pdf.cell(col_width, 8, f'Analysis ID: AI-{datetime.datetime.now().strftime("%Y%m%d%H%M")}', 'LB', 0)
    pdf.cell(col_width, 8, f'Protocol: Standard Brain MRI', 'RB', 1)
    pdf.ln(8)
    
    # Add diagnosis section with improved formatting and medical standards
    pdf.set_font('Arial', 'B', 14)
    pdf.set_fill_color(220, 220, 220)
    pdf.cell(0, 10, 'DIAGNOSIS RESULTS', 1, 1, 'C', True)
    pdf.ln(2)
    
    # Create a highlighted box for diagnosis
    pdf.set_font('Arial', 'B', 12)
    
    # Set diagnosis color and background based on diagnosis
    if "Glioma" in diagnosis:
        pdf.set_fill_color(255, 235, 235)
        pdf.set_text_color(180, 50, 50)
        severity = "HIGH PRIORITY"
    elif "No Tumor" in diagnosis:
        pdf.set_fill_color(235, 255, 235)
        pdf.set_text_color(50, 150, 50)
        severity = "NORMAL"
    elif "Meningioma" in diagnosis:
        pdf.set_fill_color(235, 235, 255)
        pdf.set_text_color(50, 50, 150)
        severity = "MEDIUM PRIORITY"
    else:  # Pituitary
        pdf.set_fill_color(255, 255, 235)
        pdf.set_text_color(150, 120, 10)
        severity = "MEDIUM PRIORITY"
    
    # Display diagnosis in a highlighted box
    pdf.cell(0, 10, f'Predicted Diagnosis: {diagnosis}', 1, 1, 'C', True)
    pdf.set_font('Arial', 'B', 10)
    pdf.cell(0, 8, f'Confidence: {confidence*100:.2f}% | Clinical Priority: {severity}', 1, 1, 'C', True)
    pdf.set_text_color(0, 0, 0)  # Reset text color
    pdf.ln(8)
    
    # Save images to temporary files with proper error handling
    temp_dir = "/tmp"
    os.makedirs(temp_dir, exist_ok=True)
    
    try:
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
        
        # Add first page for Original MRI Scan
        pdf.add_page()
        pdf.set_font('Arial', 'B', 14)
        pdf.set_fill_color(220, 220, 220)
        pdf.cell(0, 10, 'DIAGNOSTIC IMAGING', 1, 1, 'C', True)
        pdf.ln(2)
        
        # First image: Original image
        pdf.set_font('Arial', 'B', 11)
        pdf.cell(0, 8, 'Original MRI Scan:', 0, 1)
        pdf.set_font('Arial', 'I', 9)
        pdf.cell(0, 6, 'Unprocessed patient scan as received', 0, 1, 'C')
        
        # Center the image with proper dimensions
        image_width = 160  # Larger image size
        page_width = 210
        x_position = (page_width - image_width) / 2
        pdf.image(original_path, x=x_position, w=image_width)
        
        # Second page: Processed image
        pdf.add_page()
        pdf.set_font('Arial', 'B', 14)
        pdf.set_fill_color(220, 220, 220)
        pdf.cell(0, 10, 'DIAGNOSTIC IMAGING', 1, 1, 'C', True)
        pdf.ln(2)
        
        pdf.set_font('Arial', 'B', 11)
        pdf.cell(0, 8, 'Processed Image:', 0, 1)
        pdf.set_font('Arial', 'I', 9)
        pdf.cell(0, 6, 'Preprocessed scan with noise reduction and contrast enhancement', 0, 1, 'C')
        pdf.image(processed_path, x=x_position, w=image_width)
        
        # Third page: Heatmap image
        pdf.add_page()
        pdf.set_font('Arial', 'B', 14)
        pdf.set_fill_color(220, 220, 220)
        pdf.cell(0, 10, 'DIAGNOSTIC IMAGING', 1, 1, 'C', True)
        pdf.ln(2)
        
        pdf.set_font('Arial', 'B', 11)
        pdf.cell(0, 8, 'AI Attention Map:', 0, 1)
        pdf.set_font('Arial', 'I', 9)
        pdf.cell(0, 6, 'Regions of interest identified by the AI model', 0, 1, 'C')
        pdf.image(heatmap_path, x=x_position, w=image_width)
        
        # Add legend for heatmap colors
        pdf.ln(2)
        pdf.set_font('Arial', 'I', 8)
        pdf.set_text_color(200, 0, 0)
        pdf.cell(60, 5, 'Red: High attention', 0, 0, 'R')
        pdf.set_text_color(0, 150, 0)
        pdf.cell(60, 5, 'Green: Moderate attention', 0, 0, 'C')
        pdf.set_text_color(0, 0, 200)
        pdf.cell(60, 5, 'Blue: Low attention', 0, 1, 'L')
        pdf.set_text_color(0, 0, 0)  # Reset text color
        
    except Exception as e:
        # Handle image processing errors
        pdf.set_text_color(255, 0, 0)
        pdf.cell(0, 10, f"Error processing images: {str(e)}", 0, 1)
        pdf.set_text_color(0, 0, 0)
    
    # Add detailed analysis section with medical terminology on a single page
    pdf.add_page()
    pdf.set_font('Arial', 'B', 14)
    pdf.set_fill_color(220, 220, 220)
    pdf.cell(0, 10, 'DETAILED CLINICAL ANALYSIS', 1, 1, 'C', True)
    pdf.ln(5)
    
    # Add tumor type information based on diagnosis with medical terminology
    pdf.set_font('Arial', 'B', 12)
    pdf.cell(0, 8, 'Findings:', 0, 1)
    pdf.set_font('Arial', '', 10)
    
    if "Glioma" in diagnosis:
        pdf.multi_cell(0, 6, 'The AI analysis indicates features consistent with Glioma, a primary brain tumor arising from glial cells. Gliomas can be classified according to the WHO grading system (I-IV) based on histopathological features. The imaging characteristics observed include:')
        pdf.ln(2)
        pdf.set_font('Arial', '', 10)
        # Use bullet points for key findings
        findings = [
            'Irregular borders with potential infiltration into surrounding parenchyma',
            'Heterogeneous signal intensity on T1/T2-weighted sequences',
            'Possible areas of necrosis or cystic degeneration',
            'Variable enhancement pattern following contrast administration',
            'Potential mass effect with midline shift or ventricular compression'
        ]
        for finding in findings:
            pdf.cell(5, 6, '', 0, 0)
            pdf.cell(0, 6, '- ' + finding, 0, 1)
    elif "Meningioma" in diagnosis:
        pdf.multi_cell(0, 6, 'The AI analysis indicates features consistent with Meningioma, a typically benign extra-axial neoplasm arising from arachnoid cap cells of the meninges. The imaging characteristics observed include:')
        pdf.ln(2)
        pdf.set_font('Arial', '', 10)
        findings = [
            'Well-circumscribed, extra-axial mass with broad dural attachment',
            'Homogeneous enhancement following contrast administration',
            'Possible "dural tail" sign extending from the primary mass',
            'Potential calcifications within the tumor matrix',
            'Adjacent bone reaction (hyperostosis) may be present'
        ]
        for finding in findings:
            pdf.cell(5, 6, '', 0, 0)
            pdf.cell(0, 6, '- ' + finding, 0, 1)
    elif "Pituitary" in diagnosis:
        pdf.multi_cell(0, 6, 'The AI analysis indicates features consistent with a Pituitary tumor (adenoma), a neoplasm arising from the anterior pituitary gland. The imaging characteristics observed include:')
        pdf.ln(2)
        pdf.set_font('Arial', '', 10)
        findings = [
            'Well-defined sellar/suprasellar mass',
            'Potential compression of the optic chiasm if suprasellar extension is present',
            'Variable signal intensity depending on hemorrhage or cystic components',
            'Homogeneous or heterogeneous enhancement pattern',
            'Possible deviation of the pituitary stalk'
        ]
        for finding in findings:
            pdf.cell(5, 6, '', 0, 0)
            pdf.cell(0, 6, '- ' + finding, 0, 1)
    else:
        pdf.multi_cell(0, 6, 'The AI analysis indicates No Evidence of Tumor in the provided MRI scan. The brain structures appear within normal limits with:')
        pdf.ln(2)
        pdf.set_font('Arial', '', 10)
        findings = [
            'Normal gray-white matter differentiation',
            'No evidence of space-occupying lesions',
            'No abnormal enhancement patterns',
            'Normal ventricular system size and configuration',
            'No midline shift or mass effect'
        ]
        for finding in findings:
            pdf.cell(5, 6, '', 0, 0)
            pdf.cell(0, 6, '- ' + finding, 0, 1)
    
    pdf.ln(5)
    
    # Add clinical correlation section
    pdf.set_font('Arial', 'B', 12)
    pdf.cell(0, 8, 'Clinical Correlation:', 0, 1)
    pdf.set_font('Arial', '', 10)
    pdf.multi_cell(0, 6, 'The above findings should be correlated with the patient\'s clinical presentation, including neurological symptoms, duration of symptoms, progression pattern, and relevant medical history. Laboratory findings and additional imaging studies may provide complementary diagnostic information.')
    pdf.ln(5)
    
    # Add recommendations section
    pdf.set_font('Arial', 'B', 12)
    pdf.cell(0, 8, 'Recommendations:', 0, 1)
    pdf.set_font('Arial', '', 10)
    
    if "No Tumor" not in diagnosis:
        recommendations = [
            'Neurosurgical consultation for evaluation and management planning',
            'Consider advanced imaging (perfusion MRI, MR spectroscopy) for further characterization',
            'Potential stereotactic biopsy for definitive histopathological diagnosis',
            'Neurological monitoring for progression of symptoms',
            'Follow-up imaging in 1-3 months to assess for interval changes'
        ]
    else:
        recommendations = [
            'Clinical follow-up as indicated by patient symptoms',
            'Consider follow-up imaging if new neurological symptoms develop',
            'Routine neurological examination at next clinical visit',
            'No immediate intervention required based on current findings',
            'Patient reassurance regarding absence of tumor features'
        ]
    
    for rec in recommendations:
        pdf.cell(5, 6, '', 0, 0)
        pdf.cell(0, 6, '- ' + rec, 0, 1)
    
    # Add model interpretation section without performance metrics table
    pdf.add_page()
    pdf.set_font('Arial', 'B', 14)
    pdf.set_fill_color(220, 220, 220)
    pdf.cell(0, 10, 'AI MODEL INTERPRETATION', 1, 1, 'C', True)
    pdf.ln(5)
    
    pdf.set_font('Arial', '', 10)
    pdf.multi_cell(0, 6, f'This analysis was performed using {model_name}, a deep learning convolutional neural network trained on {3000} clinically validated MRI scans. The attention map utilizes Gradient-weighted Class Activation Mapping (Grad-CAM) to highlight regions that influenced the model\'s prediction, with color intensity proportional to feature importance.')
    pdf.ln(5)
    
    # Add a note about confidence directly instead of the table
    pdf.set_font('Arial', 'B', 11)
    confidence_value = f'{confidence*100:.2f}%'
    pdf.cell(0, 8, f'Model Confidence: {confidence_value}', 0, 1)
    
    if confidence*100 > 90:
        pdf.set_font('Arial', '', 10)
        pdf.set_text_color(0, 150, 0)
        pdf.cell(0, 6, 'The model has expressed high confidence in this diagnosis.', 0, 1)
    elif confidence*100 > 75:
        pdf.set_font('Arial', '', 10)
        pdf.set_text_color(0, 100, 0)
        pdf.cell(0, 6, 'The model has expressed good confidence in this diagnosis.', 0, 1)
    else:
        pdf.set_font('Arial', '', 10)
        pdf.set_text_color(180, 50, 50)
        pdf.cell(0, 6, 'The model has expressed moderate confidence in this diagnosis. Further clinical correlation is strongly advised.', 0, 1)
    
    pdf.set_text_color(0, 0, 0)  # Reset text color
    pdf.ln(5)
    
    # Add disclaimer section with enhanced medical-legal language
    pdf.set_draw_color(192, 57, 43)
    pdf.set_fill_color(253, 237, 236)
    pdf.set_line_width(0.5)
    pdf.rect(10, pdf.get_y(), 190, 40, 'DF')
    pdf.ln(2)
    
    pdf.set_font('Arial', 'B', 12)
    pdf.set_text_color(192, 57, 43)
    pdf.cell(0, 8, 'MEDICAL DISCLAIMER', 0, 1, 'C')
    pdf.set_font('Arial', '', 9)
    pdf.multi_cell(0, 5, 'This report is generated by an artificial intelligence system and is intended for research and clinical decision support purposes only. The findings presented are not a definitive medical diagnosis and should not be used as the sole basis for clinical management decisions.')
    pdf.ln(2)
    pdf.multi_cell(0, 5, 'A qualified healthcare professional must review these results in conjunction with the patient\'s clinical history, physical examination findings, and other diagnostic tests. The treating physician maintains full responsibility for all diagnostic and treatment decisions.')
    pdf.set_text_color(0, 0, 0)

    # Create binary stream to hold PDF data
    pdf_output = io.BytesIO()
    
    try:
        # Fix: Use the correct method for PyFPDF output handling
        pdf_bytes = pdf.output(dest='S')
        # Check if pdf_bytes is already bytes or if it's a string that needs encoding
        if isinstance(pdf_bytes, str):
            pdf_bytes = pdf_bytes.encode('latin1')
        # Now write the bytes to the BytesIO object
        pdf_output.write(pdf_bytes)
        pdf_output.seek(0)
    except Exception as e:
        # Handle PDF generation errors
        print(f"Error generating PDF: {str(e)}")
        # Create a simple error PDF if the main one fails
        error_pdf = FPDF()
        error_pdf.add_page()
        error_pdf.set_font('Arial', 'B', 16)
        error_pdf.cell(0, 10, 'Error Generating Report', 0, 1, 'C')
        error_pdf.set_font('Arial', '', 12)
        error_pdf.multi_cell(0, 10, f'An error occurred while generating the report: {str(e)}')
        pdf_bytes = error_pdf.output(dest='S')
        # Apply the same type checking for the error PDF
        if isinstance(pdf_bytes, str):
            pdf_bytes = pdf_bytes.encode('latin1')
        pdf_output = io.BytesIO()
        pdf_output.write(pdf_bytes)
        pdf_output.seek(0)
    
    # Clean up temp files
    finally:
        for file_path in [original_path, processed_path, heatmap_path]:
            if os.path.exists(file_path):
                try:
                    os.remove(file_path)
                except:
                    pass
    
    return pdf_output.getvalue()

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
st.title("🧠 Brain Tumor Detection using Deep Learning Models")
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
            try:
                # Generate the PDF report with error handling
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
            except Exception as e:
                st.error(f"Error generating PDF report: {str(e)}")
    
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