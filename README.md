# Brain Tumor Detection Using Deep Learning

## Table of Contents
1. [Abstract](#1-abstract)  
2. [Introduction](#2-introduction)  
3. [Related Work](#3-related-work)  
4. [Dataset Description](#4-dataset-description)  
5. [Solution Architecture](#5-solution-architecture)  
   5.1 [Data Preprocessing](#51-data-preprocessing)  
   5.2 [Data Augmentation](#52-data-augmentation)  
   5.3 [Data Preparation](#53-data-preparation)  
   5.4 [Training Strategy](#54-training-strategy)  
6. [Prediction Aggregation](#6-prediction-aggregation)  
   6.1 [Bias and Fairness Consideration](#61-bias-and-fairness-consideration)  
7. [Deployment](#7-deployment)  
8. [Result](#8-result)  
9. [References](#9-references)  
10. [Contributors](#10-contributors)

---

## 1. Abstract
Timely and accurate brain tumor diagnosis is essential for improving survival rates. Manual MRI analysis is time-intensive and subject to human error. This project automates brain tumor detection using pretrained CNNs such as Xception, MobileNet, DenseNet, ResNet, EfficientNet, and Inception. The models are fine-tuned on an MRI dataset and evaluated. EfficientNet achieved the highest accuracy (94.87%), demonstrating the feasibility of deep learning in assisting medical professionals.

## 2. Introduction
Brain tumors can severely impact neurological function. Early detection using MRI is effective but demands expert interpretation. CNNs can learn features directly from image data and are especially powerful when paired with transfer learning. Our goal is to use such pretrained architectures to automate and improve tumor detection.

## 3. Related Work
Prior studies demonstrated CNNs outperform traditional ML in MRI classification:
- Hossain et al. (2019): Automatic feature extraction with CNNs.
- Chowdhury et al. (2020): Transfer learning with VGG16/ResNet50.
- Islam et al. (2021): EfficientNet in medical imaging.

These findings support our choice to compare multiple pretrained CNNs for MRI-based tumor classification.

## 4. Dataset Description
We use the Brain Tumor MRI dataset from Kaggle (Masoud Nickparvar), with 7023 MRI images classified into:
- Glioma Tumor  
- Meningioma Tumor  
- Pituitary Tumor  
- No Tumor  

All images are JPEG, resized to 224x224, and the dataset is organized by folders for each class.

## 5. Solution Architecture
The pipeline includes preprocessing, model training, evaluation, and deployment:
- Environment: Google Colab  
- Pretrained CNNs used: Xception, MobileNetV2, DenseNet121, ResNet50, EfficientNetB0, InceptionV3  
- Optimizer: Adam, Loss: Categorical Crossentropy  
- Evaluation: Accuracy, F1-score, Confusion Matrices

### 5.1 Data Preprocessing
- Resizing to 224x224  
- Normalization to [0,1]  
- One-hot encoding of labels  
- 80-20 train-test split

### 5.2 Data Augmentation
Recommended techniques:
- Rotation, Zoom, Flipping, Shearing  
Use `ImageDataGenerator` for implementation.

### 5.3 Data Preparation
- Batch size: 32  
- 10% of training data for validation  
- Proper formatting of training/testing splits

### 5.4 Training Strategy
- Transfer learning with frozen base layers  
- Callbacks: `ModelCheckpoint`, `ReduceLROnPlateau`  
- 6 epochs, default learning rate  
- Evaluation via classification report and confusion matrix

## 6. Prediction Aggregation
Soft voting ensemble using all six models:
- Average softmax probabilities  
- Improves robustness and accuracy  
- EfficientNetB0 and Xception are strong contributors

### 6.1 Bias and Fairness Consideration
- Class imbalance mitigation (weighted loss, oversampling)  
- Representation diversity (scanner settings, demographics)  
- Ensemble models to reduce individual bias  
- Fairness metrics like recall across classes

## 7. Deployment
Deployed using Streamlit:
- Interface for uploading MRI scans and viewing classification  
- Hosted on Streamlit Community Cloud  
- Repository: [GitHub](https://github.com/ArunRengaraman/BrainMRI)  
- App: [Live Demo](https://brainmri-y5zgpn2hrnbrahgwnogy26.streamlit.app/)

Deployment Challenges:
- Free-tier compute limitations  
- Input format assumptions (224x224 RGB)

## 8. Result

| Model         | Accuracy | Precision | Recall  | F1-Score |
|---------------|----------|-----------|---------|----------|
| Xception      | 0.8982   | 0.9081    | 0.8982  | 0.8983   |
| MobileNetV2   | 0.7217   | 0.8534    | 0.7217  | 0.7320   |
| DenseNet121   | 0.8470   | 0.8818    | 0.8470  | 0.8463   |
| ResNet50      | 0.8797   | 0.8969    | 0.8797  | 0.8754   |
| EfficientNetB0| 0.9488   | 0.9561    | 0.9488  | 0.9495   |
| InceptionV3   | 0.8705   | 0.9035    | 0.8705  | 0.8725   |

EfficientNetB0 outperforms others. Ensemble prediction improves classification reliability, especially in low-representation classes.

## 9. References
1. Badža & Barjaktarović (2020), *Applied Sciences*  
2. Ruba et al. (2020), *Biomedical and Pharmacology Journal*  
3. Sultan et al. (2019), *IEEE Access*  
4. Ismael & Abdel-Qader (2018), *IEEE EIT*  
5. Khan et al. (2020), *Mathematical Biosciences and Engineering*  
6. Hatami et al. (2019), *IEEE ITAIC*  
7. Rajinikanth et al. (2020), *Applied Sciences*  
8. Kebir & Mekaoui (2018), *IEEE IC_ASET*  
9. Agrawal & Maan (2020), *Mody Univ. J. Computing*  
10. Wang et al. (2018), *Lecture Notes in Computer Science*

## 9.1 Application Demo

![Kapture 2025-05-12 at 08 42 42](https://github.com/user-attachments/assets/7305cf4d-549c-4b21-b05e-a06b9b6a4c56)

## 10. Contributors
- [@ArunRengaraman](https://github.com/ArunRengaraman) - Arun Rengaraman  
- [@PraveenGanapathy](https://github.com/PraveenGanapathy) - Praveen Ganapathy Ravi
