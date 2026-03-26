
# src/config.py
import os

# Dataset paths
BASE_DIR  = '/content/drive/MyDrive/Dataset/chest_xray'  # your chest_xray path
TRAIN_DIR = os.path.join(BASE_DIR, 'train')
VAL_DIR   = os.path.join(BASE_DIR, 'val')
TEST_DIR  = os.path.join(BASE_DIR, 'test')

# Classes
CLASSES = ['NORMAL', 'PNEUMONIA']

# Image settings
IMG_SIZE   = (224, 224)
BATCH_SIZE = 32

# Training settings
EPOCHS = 15
LR     = 1e-4

# Output paths
OUTPUT_DIR           = 'outputs'
BASELINE_MODEL_PATH  = os.path.join(OUTPUT_DIR, 'baseline_cnn.h5')
TRANSFER_MODEL_PATH  = os.path.join(OUTPUT_DIR, 'pneumonia_model.h5')

os.makedirs(OUTPUT_DIR, exist_ok=True)
