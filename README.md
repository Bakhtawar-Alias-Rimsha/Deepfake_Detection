# 🧠 Deep Fake Detection using PyTorch and Transfer Learning
#  Deep learning  project
## 🌟 Project Overview

This project implements and compares two distinct Convolutional Neural
Network (CNN) approaches---a custom-built model and a fine-tuned
MobileNetV2 architecture---to solve the binary classification task of
distinguishing between real and deep fake images. The entire pipeline,
from data preprocessing and training to comprehensive evaluation
(including ROC curves and Confusion Matrices), is executed using the
PyTorch framework.

## ✨ Key Features

-   **Transfer Learning:** Implementation of MobileNetV2 with frozen
    base layers and a custom classification head.\
-   **Custom CNN Baseline:** A multi-layered convolutional model built
    from scratch for comparison.\
-   **Robust Evaluation:** Calculation of Accuracy, Precision, Recall,
    F1-Score, and ROC-AUC on a held-out test set.\
-   **Early Stopping:** Implemented to prevent overfitting and optimize
    training time based on validation loss.\
-   **Visualization:** Comprehensive plotting of dataset distribution,
    learning curves (Accuracy/Loss), and evaluation metrics.

------------------------------------------------------------------------

## 📦 Required Packages (`requirements.txt`)

  -------------------------------------------------------------------------
  Package        Version (Example)  Purpose
  -------------- ------------------ ---------------------------------------
  torch          2.0.1              Main deep learning framework

  torchvision    0.15.2             Image datasets and models

  numpy          1.25.0             Numerical operations

  pandas         2.0.3              Data handling for summaries

  matplotlib     3.7.2              Plotting and visualization

  scikit-learn   1.3.0              Metrics (Confusion Matrix, ROC)

  seaborn        0.12.2             Advanced statistical visualizations

  pillow         9.5.0              Image manipulation
  -------------------------------------------------------------------------

------------------------------------------------------------------------

## 📁 Dataset Requirements

Expected root path structure (e.g., `/path/to/Dataset2`)\
Image Size: **224 x 224 pixels**

    Dataset2/
    ├── train/
    │   ├── Fake/
    │   └── Real/
    ├── valid/
    │   ├── Fake/
    │   └── Real/
    └── test/
        ├── Fake/
        └── Real/

------------------------------------------------------------------------

## 📐 Model Architectures

### **1. MobileNetV2 (Transfer Learning)**

-   Loaded with pretrained ImageNet weights\
-   Base layers **frozen**\
-   Custom classification head added\
-   Strategy: *Feature Extraction + Custom Classifier*

### **2. Custom CNN**

A simple, lightweight CNN architecture using:

    Conv2D → ReLU → MaxPool → Dense Layers

Used as a baseline model.

------------------------------------------------------------------------

## 🚀 Execution and Training

The core logic is written inside the notebook
`DeepFake_Detection_Notebook.ipynb`.

### **Key Training Details**

  Configuration    Value
  ---------------- ---------------------
  Loss Function    BCEWithLogitsLoss
  Optimizer        Adam
  Learning Rate    1e-4
  Early Stopping   Patience = 3 epochs

### **Notebook Workflow**

1.  **Cell 1--2:** Imports and Data Loading\
2.  **Cell 3:** Dataset distribution and visualization\
3.  **Cell 4:** Model definitions (CustomCNN + MobileNetV2)\
4.  **Cell 5:** Training loop execution\
5.  **Cell 6:** Plot learning curves\
6.  **Cell 7:** Evaluation on test set\
7.  **Cell 8:** Save best model weights (.pth)

------------------------------------------------------------------------

## 📊 Results and Performance

### **Final Model Comparison (Test Set Metrics)**

  Model         Accuracy   Precision   Recall   F1-Score   AUC Score
  ------------- ---------- ----------- -------- ---------- -----------
  MobileNetV2   0.8582     0.8898  0.8750  0.8824  0.9464

  Custom CNN    0.8825     0.9116  0.8933  0.9024  0.9487

------------------------------------------------------------------------

## 🏆 Best Performing Model

**\[ Custom CNN\]**\
Custom CNN    0.8825
Chosen based on the highest overall Accuracy and ROC-AUC.

------------------------------------------------------------------------

## 🔮 Future Work

-   Fine-tune deeper layers of MobileNetV2\
-   Use stronger augmentations specific to deep fake artifacts\
-   Deploy via lightweight API for real-time inference
