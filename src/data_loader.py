import os
import cv2
import numpy as np
from tensorflow.keras.utils import to_categorical
from config import (TRAIN_DIR, TEST_DIR, IMG_SHAPE, NUM_CLASSES,
                    EMOTION_LABELS)


def load_fer2013_from_folders(train_dir, test_dir):
    X_train, y_train = [], []
    X_test, y_test = [], []

    emotion_order = [
        "angry", "disgust", "fear", "happy",
        "neutral", "sad", "surprise"
    ]

    for emotion in emotion_order:
        folder = os.path.join(train_dir, emotion)
        if not os.path.isdir(folder):
            raise FileNotFoundError(f"Missing emotion folder: {folder}")

        label_idx = emotion_order.index(emotion)

        for img_file in os.listdir(folder):
            img_path = os.path.join(folder, img_file)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            img = cv2.resize(img, (IMG_SHAPE, IMG_SHAPE))
            X_train.append(img)
            y_train.append(label_idx)

    for emotion in emotion_order:
        folder = os.path.join(test_dir, emotion)
        if not os.path.isdir(folder):
            raise FileNotFoundError(f"Missing emotion folder: {folder}")

        label_idx = emotion_order.index(emotion)

        for img_file in os.listdir(folder):
            img_path = os.path.join(folder, img_file)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            img = cv2.resize(img, (IMG_SHAPE, IMG_SHAPE))
            X_test.append(img)
            y_test.append(label_idx)

    return (np.array(X_train), np.array(y_train),
            np.array(X_test), np.array(y_test))


def preprocess_data():
    print("Loading data...")
    X_train, y_train, X_test, y_test = load_fer2013_from_folders(
        TRAIN_DIR, TEST_DIR
    )

    print("X_train:", X_train.shape)
    print(" y_train:", y_train.shape)
    print("X_test: ", X_test.shape)
    print("y_test: ", y_test.shape)

    print("\nTrain class distribution:")
    for i in range(NUM_CLASSES):
        print(f"{EMOTION_LABELS[i]:10s}: {np.sum(y_train == i)}")

    print("\nTest class distribution:")
    for i in range(NUM_CLASSES):
        print(f"{EMOTION_LABELS[i]:10s}: {np.sum(y_test == i)}")

    # Normalize
    X_train = X_train.astype("float32") / 255.0
    X_test = X_test.astype("float32") / 255.0

    # Add channel dimension
    X_train = X_train[..., np.newaxis]
    X_test = X_test[..., np.newaxis]

    # One-hot encode
    y_train = to_categorical(y_train, num_classes=NUM_CLASSES)
    y_test = to_categorical(y_test, num_classes=NUM_CLASSES)

    print("\nFinal X_train shape:", X_train.shape)
    print("Final X_test shape: ", X_test.shape)

    return X_train, y_train, X_test, y_test