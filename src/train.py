import os
import pandas as pd
import plotly.express as px
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import (EarlyStopping, ReduceLROnPlateau,
                                        ModelCheckpoint)

from config import (BATCH_SIZE, CNN_MODEL_PATH, PLOTS_DIR, EMOTION_LABELS,
                    NUM_CLASSES)
from data_loader import preprocess_data
from model import build_cnn_model


def main():
    X_train, y_train, X_test, y_test = preprocess_data()

    # Data augmentation
    train_datagen = ImageDataGenerator(
        rotation_range=10,
        horizontal_flip=True,
        width_shift_range=0.1,
        height_shift_range=0.1,
        fill_mode='nearest')
    test_datagen = ImageDataGenerator()

    train_flow = train_datagen.flow(X_train, y_train, batch_size=BATCH_SIZE)
    test_flow = test_datagen.flow(X_test, y_test, batch_size=BATCH_SIZE)

    print("Generators ready")

    model = build_cnn_model()
    model.summary()

    callbacks = [
        EarlyStopping(monitor='val_accuracy', min_delta=0.00005, patience=11,
                      verbose=1, restore_best_weights=True),
        ReduceLROnPlateau(monitor='val_accuracy', factor=0.5, patience=7,
                          min_lr=1e-7, verbose=1),
        ModelCheckpoint(CNN_MODEL_PATH, monitor='val_accuracy',
                        save_best_only=True, verbose=1)
    ]

    history = model.fit(
        train_flow,
        epochs=50,
        verbose=1,
        validation_data=test_flow,
        callbacks=callbacks
    )

    model.save(CNN_MODEL_PATH)
    print(f"\n CNN saved: {CNN_MODEL_PATH}")

    # ---- Accuracy plot ----
    acc_df = pd.DataFrame({
        'epochs': range(len(history.history['accuracy'])),
        'accuracy': history.history['accuracy'],
        'val_accuracy': history.history['val_accuracy']
    })
    fig = px.line(acc_df, x='epochs', y=['accuracy', 'val_accuracy'],
                  title='CNN — Training & Validation Accuracy')
    fig.write_html(os.path.join(PLOTS_DIR, 'cnn_accuracy.html'))
    fig.show()

    # ---- Loss plot ----
    loss_df = pd.DataFrame({
        'epochs': range(len(history.history['loss'])),
        'loss': history.history['loss'],
        'val_loss': history.history['val_loss']
    })
    fig = px.line(loss_df, x='epochs', y=['loss', 'val_loss'],
                  title='CNN — Training & Validation Loss')
    fig.write_html(os.path.join(PLOTS_DIR, 'cnn_loss.html'))
    fig.show()


if __name__ == "__main__":
    main()