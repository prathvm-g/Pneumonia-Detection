
import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from src.config import TRAIN_DIR, CLASSES

def plot_sample_images(split_dir, n_per_class=5):
    fig, axes = plt.subplots(len(CLASSES), n_per_class,
                             figsize=(3*n_per_class, 3*len(CLASSES)))
    fig.suptitle('Sample Chest X-Ray Images', fontsize=15, fontweight='bold')
    for row, cls in enumerate(CLASSES):
        folder = os.path.join(split_dir, cls)
        files  = os.listdir(folder)[:n_per_class]
        for col, fname in enumerate(files):
            img = Image.open(os.path.join(folder, fname)).convert('RGB')
            axes[row, col].imshow(img)
            color = 'green' if cls == 'NORMAL' else 'red'
            axes[row, col].set_title(cls, color=color, fontweight='bold')
            axes[row, col].axis('off')
    plt.tight_layout()
    plt.show()


def plot_predictions(images, true_labels, pred_labels, probs,
                     n_correct=4, n_wrong=4):
    class_names  = ['NORMAL', 'PNEUMONIA']
    correct_idx  = np.where(true_labels == pred_labels)[0][:n_correct]
    incorrect_idx = np.where(true_labels != pred_labels)[0][:n_wrong]

    fig, axes = plt.subplots(2, max(n_correct, n_wrong),
                             figsize=(16, 8))
    fig.suptitle('Correct (top) vs Incorrect (bottom) Predictions',
                 fontsize=14, fontweight='bold')

    for i, idx in enumerate(correct_idx):
        axes[0, i].imshow(images[idx])
        axes[0, i].set_title(
            f"True: {class_names[true_labels[idx]]}\n"
            f"Pred: {class_names[pred_labels[idx]]} ({probs[idx]:.2f})",
            color='green', fontsize=9)
        axes[0, i].axis('off')

    for i, idx in enumerate(incorrect_idx):
        axes[1, i].imshow(images[idx])
        axes[1, i].set_title(
            f"True: {class_names[true_labels[idx]]}\n"
            f"Pred: {class_names[pred_labels[idx]]} ({probs[idx]:.2f})",
            color='red', fontsize=9)
        axes[1, i].axis('off')

    plt.tight_layout()
    plt.show()
