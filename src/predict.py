import os
import sys
import cv2
import numpy as np
from tensorflow.keras.models import load_model

from config import (CNN_MODEL_PATH, IMG_SHAPE, EMOTION_LABELS)
from recommend import load_music, recommend_songs


def predict_emotion_from_image(model, image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Cannot read: {image_path}")
    img = cv2.resize(img, (IMG_SHAPE, IMG_SHAPE))
    img = img.astype('float32') / 255.0
    img = img.reshape(1, IMG_SHAPE, IMG_SHAPE, 1)

    preds = model.predict(img, verbose=0)[0]
    idx = int(np.argmax(preds))
    return EMOTION_LABELS[idx], float(preds[idx]), preds


def main():
    # Image path -- terminal la kudutha argument use pannum
    if len(sys.argv) < 2:
        print("Usage: python predict.py <image_path>")
        sys.exit(1)

    img_path = sys.argv[1]

    model = load_model(CNN_MODEL_PATH)
    music_df = load_music()

    emotion, confidence, all_probs = predict_emotion_from_image(model, img_path)
    print(f"\nDetected Emotion: {emotion} ({confidence:.2%})")

    print("\nAll probabilities:")
    for i, prob in enumerate(all_probs):
        print(f"  {EMOTION_LABELS[i]:10s}: {prob:.2%}")

    songs = recommend_songs(music_df, emotion, top_k=5)
    if not songs.empty:
        print(f"\n Top 5 songs for mood '{emotion}':\n")
        print(songs)
    else:
        print("  No songs for this mood.")


if __name__ == "__main__":
    main()