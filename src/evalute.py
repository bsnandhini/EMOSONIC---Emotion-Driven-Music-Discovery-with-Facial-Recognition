import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.models import load_model
from sklearn.metrics import confusion_matrix, classification_report

from config import (CNN_MODEL_PATH, PLOTS_DIR, NUM_CLASSES,
                    EMOTION_LABELS)
from data_loader import preprocess_data


def main():
    _, _, X_test, y_test = preprocess_data()

    model = load_model(CNN_MODEL_PATH)

    y_pred = model.predict(X_test, verbose=1)
    y_pred_labels = np.argmax(y_pred, axis=1)
    y_true_labels = np.argmax(y_test, axis=1)

    classes = [EMOTION_LABELS[i] for i in range(NUM_CLASSES)]
    cm = confusion_matrix(y_true_labels, y_pred_labels)

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, xticklabels=classes, yticklabels=classes,
                linewidths=1, annot=True, ax=ax, fmt='g', cmap='Blues')
    plt.title('CNN — Confusion Matrix')
    plt.ylabel('True')
    plt.xlabel('Predicted')
    plt.savefig(os.path.join(PLOTS_DIR, 'cnn_confusion_matrix.png'))
    plt.show()

    print("\n Classification Report:")
    print(classification_report(y_true_labels, y_pred_labels,
                                target_names=classes))


if __name__ == "__main__":
    main()