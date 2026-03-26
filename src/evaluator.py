
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

def evaluate_model(model, test_gen, model_name='Model'):
    test_gen.reset()
    preds      = model.predict(test_gen).flatten()
    y_pred     = (preds > 0.5).astype(int)
    y_true     = test_gen.classes

    acc = accuracy_score(y_true, y_pred)
    print(f'\n{model_name} — Accuracy: {acc:.4f}')
    print(classification_report(y_true, y_pred,
          target_names=['NORMAL', 'PNEUMONIA']))

    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['NORMAL', 'PNEUMONIA'],
                yticklabels=['NORMAL', 'PNEUMONIA'])
    plt.title(f'{model_name} — Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.show()
    return acc, y_true, y_pred, preds


def plot_history(history, title='Training History'):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(title, fontsize=14, fontweight='bold')

    axes[0].plot(history.history['accuracy'],     label='Train')
    axes[0].plot(history.history['val_accuracy'], label='Validation')
    axes[0].set_title('Accuracy')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(history.history['loss'],     label='Train')
    axes[1].plot(history.history['val_loss'], label='Validation')
    axes[1].set_title('Loss')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


def compare_models(results):
    import matplotlib.pyplot as plt
    names  = list(results.keys())
    values = [v * 100 for v in results.values()]

    plt.figure(figsize=(8, 5))
    bars = plt.bar(names, values, color=['#4C72B0', '#DD8452'],
                   width=0.4, edgecolor='white')
    for bar, val in zip(bars, values):
        plt.text(bar.get_x() + bar.get_width()/2,
                 bar.get_height() + 0.3,
                 f'{val:.2f}%', ha='center', fontweight='bold')
    plt.ylim(60, 100)
    plt.title('Model Comparison — Test Accuracy')
    plt.ylabel('Accuracy (%)')
    plt.tight_layout()
    plt.show()
