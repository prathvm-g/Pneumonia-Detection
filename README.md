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

## 📊 Model Evaluation & Confusion Matrix Analysis

### 🧠 Baseline CNN Results

| Class       | Precision | Recall | F1-Score | Support |
|------------|----------|--------|----------|--------|
| NORMAL      | 0.88     | 0.84   | 0.86     | 234    |
| PNEUMONIA   | 0.91     | 0.93   | 0.92     | 390    |

- **Accuracy:** ~0.90  
- **Macro Avg F1-score:** 0.89  
- **Weighted Avg F1-score:** 0.90  

---

### 🚀 MobileNetV2 Results

| Class       | Precision | Recall | F1-Score | Support |
|------------|----------|--------|----------|--------|
| NORMAL      | 0.98     | 0.68   | 0.80     | 234    |
| PNEUMONIA   | 0.84     | 0.99   | 0.91     | 390    |

- **Accuracy:** ~0.87  
- **Macro Avg F1-score:** 0.85  
- **Weighted Avg F1-score:** 0.87  

---

## 🔍 Key Observations (Critical Insights)

### 1. Baseline CNN Performance
- Balanced performance across both classes  
- Good trade-off between precision and recall  
- More stable and reliable predictions overall  

---

### 2. MobileNetV2 Behavior
- Extremely high **recall for pneumonia (0.99)** → almost all pneumonia cases detected  
- But very low **recall for normal (0.68)** → many normal cases misclassified as pneumonia  

👉 This indicates:
- Model is **biased towards predicting pneumonia**
- Likely due to **class imbalance or overfitting to dominant features**

---

### ⚠️ Medical Insight (IMPORTANT)

- **False Negatives (missing pneumonia)** are dangerous  
- MobileNetV2 minimizes this risk (high recall = good)  
- But increases **False Positives** → unnecessary alarms  

👉 Trade-off:
- **Baseline CNN → Balanced model**
- **MobileNetV2 → Safer but more conservative (over-predicts pneumonia)**

---

## 🧠 Final Conclusion

- Baseline CNN provides **more balanced and stable performance**
- MobileNetV2 prioritizes **sensitivity (recall)** over precision
- In real-world medical use:
  - High recall is preferred  
  - But excessive false positives can reduce usability
    
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


