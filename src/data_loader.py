
import os
import numpy as np
from PIL import Image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from src.config import TRAIN_DIR, VAL_DIR, TEST_DIR, IMG_SIZE, BATCH_SIZE, CLASSES

def build_generators():
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=15,
        zoom_range=0.1,
        horizontal_flip=True
    )
    val_test_datagen = ImageDataGenerator(rescale=1./255)

    train_gen = train_datagen.flow_from_directory(
        TRAIN_DIR, target_size=IMG_SIZE,
        batch_size=BATCH_SIZE, class_mode='binary', shuffle=True
    )
    val_gen = val_test_datagen.flow_from_directory(
        VAL_DIR, target_size=IMG_SIZE,
        batch_size=BATCH_SIZE, class_mode='binary', shuffle=False
    )
    test_gen = val_test_datagen.flow_from_directory(
        TEST_DIR, target_size=IMG_SIZE,
        batch_size=BATCH_SIZE, class_mode='binary', shuffle=False
    )
    return train_gen, val_gen, test_gen


def load_images_for_viz(split_dir, n_per_class=20):
    images, labels = [], []
    label_map = {'NORMAL': 0, 'PNEUMONIA': 1}
    for cls in CLASSES:
        folder = os.path.join(split_dir, cls)
        files  = os.listdir(folder)[:n_per_class]
        for f in files:
            img = Image.open(os.path.join(folder, f)).convert('RGB').resize(IMG_SIZE)
            images.append(np.array(img) / 255.0)
            labels.append(label_map[cls])
    return np.array(images), np.array(labels)
