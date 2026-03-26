# 🩺 Pneumonia Detection using Chest X-rays

A deep learning-based system to detect pneumonia from chest X-ray images using Convolutional Neural Networks (CNN) and transfer learning.

---

## 🚀 Problem Statement

Pneumonia is a serious respiratory condition that requires early and accurate detection. Manual diagnosis from chest X-rays can be time-consuming and prone to error, especially in high-volume medical settings.

This project aims to build an automated system to assist in detecting pneumonia from X-ray images.

---

## 🧠 Approach

### 1. Data Preprocessing
- Resized and normalised chest X-ray images
- Handled class imbalance
- Split the dataset into training, validation, and test sets

### 2. Model Development
- Built a **baseline CNN model** for image classification
- Implemented **Transfer Learning using MobileNetV2**
- Fine-tuned top layers to improve performance on medical data

### 3. Regularization & Optimization
- EarlyStopping to prevent overfitting
- ReduceLROnPlateau for adaptive learning rate
- Data augmentation for better generalisation

---

## 📊 Model Evaluation

- Evaluated using:
  - Accuracy
  - Precision
  - Recall
  - F1 Score

### ⚠️ Key Insight
In medical diagnosis, **recall is more important than accuracy**, since missing a pneumonia case (false negative) can be critical.

---

## 🔍 Error Analysis

- Model struggles with:
  - Low-contrast X-ray images
  - Early-stage pneumonia cases
  - Visually ambiguous samples

- Some predictions show **high confidence but incorrect results**, indicating overfitting to certain patterns

---

## 📊 Confusion Matrix

![Baseline](<img width="709" height="407" alt="Screenshot 2026-03-26 123436" src="https://github.com/user-attachments/assets/c579e390-0bc0-4d61-a327-b8bf990b3921" />
)
![MobileNetV2](<img width="645" height="201" alt="Screenshot 2026-03-26 123736" src="https://github.com/user-attachments/assets/47fea996-d71b-4f95-a095-6ff3e6be046d" />
)

### 🔍 Interpretation

- True Positives (Pneumonia correctly detected): High  
- True Negatives (Normal correctly detected): High  

- ⚠️ False Negatives: Critical cases where pneumonia is missed  
- ⚠️ False Positives: Normal cases misclassified as pneumonia  

### Key Insight:
The model shows strong overall performance, but false negatives remain a concern. In medical applications, minimizing false negatives is crucial, as missed diagnoses can have serious consequences.

This indicates the need for:
- Improving sensitivity (recall)
- Better handling of subtle pneumonia cases

## 📈 Results

| Model              | Accuracy |
|-------------------|---------|
| Baseline CNN      | ~88%    |
| MobileNetV2       | ~87%    |

### Insight:
Transfer learning did not significantly outperform the baseline model, possibly due to:
- Domain difference (ImageNet vs X-ray images)
- Limited dataset size

---

## 🛠️ Tech Stack

- Python  
- TensorFlow / Keras  
- NumPy, Matplotlib  
- CNN, Transfer Learning (MobileNetV2)

---

## ⚠️ Limitations

- Limited dataset size affects generalisation  
- Model struggles with subtle pneumonia patterns  
- No deployment (currently notebook-based)

---

## 🚀 Future Improvements

- Use advanced architectures (ResNet, EfficientNet)  
- Improve dataset quality and size  
- Deploy model using Streamlit or FastAPI  
- Perform deeper error analysis on misclassified samples  

---

## 📌 Conclusion

This project demonstrates the application of deep learning in medical imaging, highlighting both the potential and limitations of CNN-based models in real-world scenarios.

---
