# 🫁 Pneumonia Detection from Chest X-Rays

---

## 📌 Project Overview

This project develops an **AI-powered system** to detect pneumonia from chest X-ray images
using deep learning. It is a **binary classification** task with two target classes:
`NORMAL` and `PNEUMONIA`.

The project follows a complete, professional ML pipeline:
- Exploratory Data Analysis
- Image Preprocessing & Augmentation
- Baseline CNN trained from scratch
- Transfer Learning with MobileNetV2
- Clinical evaluation with medical metrics
- Grad-CAM visual explainability

> ⚠️ **Disclaimer:** This model is a decision-support tool only.
> It is not a substitute for professional medical diagnosis.

---

## 🎯 Results Summary

| Model | Accuracy | Pneumonia Recall | Normal Precision | F1 Score |
|-------|----------|-----------------|-----------------|----------|
| Baseline CNN | 88% | 0.89 | 0.83 | 0.91 |
| MobileNetV2 (Transfer Learning) | 87% | **0.98** | **0.96** | 0.90 |

> **Key insight:** MobileNetV2 catches 98% of real pneumonia cases —
> missing only 2% vs 11% for the baseline. In healthcare, this difference is critical.

---

## 🗂️ Project Structure
```
pneumonia-detection/
│
├── README.md                              ← You are here
├── requirements.txt                       ← Python dependencies
├── .gitignore                             ← Files excluded from GitHub
│
├── notebooks/
│   ├── task1_load_explore.ipynb           ← Task 1: Load & Explore (2 marks)
│   ├── task2_preprocessing.ipynb          ← Task 2: Preprocessing & Augmentation (4 marks)
│   ├── task3_baseline_cnn.ipynb           ← Task 3: Baseline CNN (4 marks)
│   ├── task4_evaluation.ipynb             ← Task 4: Model Evaluation (2 marks)
│   ├── task5_transfer_learning.ipynb      ← Task 5: Transfer Learning (4 marks)
│   ├── task6_visualise_predictions.ipynb  ← Task 6: Visualise Predictions (2 marks)
│   ├── task8_documentation.ipynb          ← Task 8: Documentation (2 marks)
│   └── optional_gradcam.ipynb             ← Bonus: Grad-CAM Explainability
│
├── src/
│   ├── config.py                          ← Central config (paths, hyperparameters)
│   ├── data_loader.py                     ← Dataset loading & Keras generators
│   ├── model_builder.py                   ← CNN and MobileNetV2 model definitions
│   ├── trainer.py                         ← Training loops and callbacks
│   ├── evaluator.py                       ← Metrics, confusion matrix, plots
│   └── visualiser.py                      ← Prediction & Grad-CAM visualisations
│
└── outputs/                               ← Saved models (git-ignored)
    ├── baseline_cnn.h5
    └── pneumonia_model.h5
```

---

## 📁 Dataset

**Chest X-Ray Images (Pneumonia)** — Kermany et al., Kaggle:
> https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia

| Split | NORMAL | PNEUMONIA | Total |
|-------|--------|-----------|-------|
| Train | 1,341 | 3,875 | 5,216 |
| Val | 8 | 8 | 16 |
| Test | 234 | 390 | 624 |

> ⚠️ Note: Significant class imbalance — 74.29% Pneumonia vs 25.71% Normal in training set.

After downloading, extract so the structure looks like:
```
chest_xray/
├── train/
│   ├── NORMAL/
│   └── PNEUMONIA/
├── val/
│   ├── NORMAL/
│   └── PNEUMONIA/
└── test/
    ├── NORMAL/
    └── PNEUMONIA/
```

---

## ⚙️ Setup

### 1. Clone the repository
```bash
git clone https://github.com/prathvm-g/pneumonia-detection.git
cd pneumonia-detection
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Update dataset path
Open `src/config.py` and update:
```python
BASE_DIR = '/content/drive/MyDrive/chest_xray'  # your path here
```

### 4. Run on Google Colab (recommended)
Each notebook mounts Google Drive automatically.
Run notebooks in order: Task 1 → Task 2 → ... → Task 8 → Grad-CAM

---

### 3. Dataset Setup

Download the dataset from Kaggle:
https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia

Extract and place the `chest_xray` folder in one of these locations:

**Option A: In a `dataset` subfolder (recommended)**
```
Pneumonia Detection/
├── Pneumonia_Image_Detection.ipynb
└── dataset/
    └── chest_xray/
        ├── train/
        ├── val/
        └── test/
```

**Option B: Directly in the project folder**
```
Pneumonia Detection/
├── Pneumonia_Image_Detection.ipynb
└── chest_xray/
    ├── train/
    ├── val/
    └── test/
```

If using Option B, update `BASE_DIR` in the notebook:
```python
BASE_DIR = os.path.join(NOTEBOOK_DIR, 'chest_xray')
```

### 4. Run the notebook

**Using VS Code:**
1. Open the `Pneumonia_Image_Detection.ipynb` file in VS Code
2. Select a Python kernel with TensorFlow installed
3. Run cells in order from top to bottom

**Using Jupyter:**
```bash
jupyter notebook Pneumonia_Image_Detection.ipynb
```

**Using Google Colab (alternative):**
1. Upload the notebook to Google Drive
2. Open with Colab
3. Update `BASE_DIR` to point to your mounted Drive path

---

## 🧪 Tasks

| Steps | Notebook | Description |
|------|----------|-------|-------------|
| 1 | 1_load_explore.ipynb | Load dataset, count images, visualise samples |
| 2 | 2_preprocessing.ipynb | Resize, normalise, augment with ImageDataGenerator |
| 3 | 3_baseline_cnn.ipynb | Build & train baseline CNN from scratch |
| 4 | 4_evaluation.ipynb | Confusion matrix, accuracy, F1, clinical interpretation |
| 5 | 5_transfer_learning.ipynb | MobileNetV2, 2-phase training, model comparison |
| 6 | 6_visualise_predictions.ipynb | Correct/incorrect/hard-case visualisations |
| 8 | 8_documentation.ipynb | Reflections, clinical implications, future work |
| 9 |_gradcam.ipynb | Grad-CAM heatmap overlays for explainability |

---

## 🏗️ Pipeline Architecture
```
Raw X-Ray Images (JPEG)
         │
         ▼
Task 1: Explore ──── Count images, visualise samples, check sizes
         │
         ▼
Task 2: Preprocess ── Resize 224×224, normalise [0,1], augment training
         │
         ▼
Task 3: Baseline CNN ─ 3×(Conv2D→BatchNorm→MaxPool) → Dense → sigmoid
         │
         ▼
Task 4: Evaluate ──── Confusion matrix, recall, F1, clinical analysis
         │
         ▼
Task 5: MobileNetV2 ─ Phase 1 (frozen) → Phase 2 (fine-tune last 30 layers)
         │
         ▼
Task 6: Visualise ─── Correct vs incorrect predictions, hard cases
         │
         ▼
Grad-CAM: Explain ─── Heatmap overlays showing model attention regions
```

---

## 🧠 Key Concepts Learned

### Why CNNs for Medical Images?
CNNs learn spatial hierarchies of features:
- **Early layers:** edges, corners, basic textures
- **Middle layers:** shapes, patterns, organ structures
- **Deep layers:** high-level features like consolidations

### Why Transfer Learning?
Training from scratch on 5,216 images risks overfitting.
MobileNetV2 pretrained on 1.2M ImageNet images already knows
how to detect rich visual features — we just adapt them to X-rays.

### Why Recall over Accuracy in Healthcare?
- **False Negative** (missed pneumonia) → patient goes untreated → potentially fatal ❌
- **False Positive** (false alarm) → extra tests → inconvenient but not dangerous ⚠️
- Therefore: **maximise recall**, accept lower precision if necessary

### Why Grad-CAM?
Doctors need to know WHERE the model looked, not just WHAT it decided.
Grad-CAM makes the model transparent and trustworthy for clinical use.

---

## 🏥 Clinical Relevance

### How this model can help doctors
- **Triage:** Automatically flag high-probability pneumonia for priority review
- **Low-resource settings:** First-line screening where radiologists are scarce
- **Consistency:** Objective assessment without fatigue or shift bias
- **Education:** Grad-CAM overlays help train junior radiologists

### Limitations
- Single-source dataset — may not generalise across hospitals
- Binary only — cannot detect other lung conditions
- Requires clinical validation before real-world deployment
- Black box without Grad-CAM — doctors cannot see reasoning
- Very small validation set (16 images) — unreliable val metrics

---

## 🔮 Future Work

| Priority | Extension |
|----------|-----------|
| High | NIH Chest X-ray Dataset (100k+ images, 14 conditions) |
| High | Grad-CAM integration into clinical dashboard |
| Medium | Multi-label classification (COVID-19, effusion, cardiomegaly) |
| Medium | Ensemble: MobileNetV2 + EfficientNet + DenseNet |
| Medium | Bayesian uncertainty estimation |
| Low | Streamlit/Gradio web app for clinical demonstration |
| Low | DICOM format support for raw medical images |

---

## 📦 Tech Stack

| Tool | Purpose |
|------|---------|
| Python 3.10 | Core language |
| TensorFlow / Keras | Model building and training |
| MobileNetV2 | Pretrained transfer learning base |
| NumPy / Pandas | Data manipulation |
| Matplotlib / Seaborn | Visualisation |
| scikit-learn | Evaluation metrics |
| Pillow / OpenCV | Image processing |
| Google Colab | Training environment (free GPU) |

---

## 👤 Author

**Pratham Gupta**
- GitHub: [@prathvm-g](https://github.com/prathvm-g)

---

⭐ If you found this project useful, please star the repository!
