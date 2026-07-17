"""
Trains the facial-expression CNN per thesis Section 3.5. Exact filter counts
for the conv layers aren't given in the source material (only the qualitative
architecture description) -- reasonable values are used below. If Carmine's
actual layer sizes can be recovered, replace them for a faithful reproduction.

Expects --data_dir/{train,val}/<emotion_name>/*.jpg layout after you've
aggregated RAF-DB + AffectNet + ExpW-F + FER2013 and removed the 'contempt'
class per thesis 3.5. Face-cropping is assumed already done for RAF-DB/
AffectNet/ExpW-F (they ship pre-cropped); FER2013 images are already 48x48
crops. Resize step below handles the 96x96 target.

Target: ~72% validation accuracy (thesis-reported figure).
"""
import argparse

import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
from sklearn.metrics import classification_report, accuracy_score
import numpy as np

from common import EMOTIONS, NUM_CLASSES, VISUAL_IMG_SIZE, RANDOM_SEED


def build_model():
    inp = layers.Input(shape=(VISUAL_IMG_SIZE, VISUAL_IMG_SIZE, 1))
    x = layers.Conv2D(64, 3, activation="relu", padding="same")(inp)
    x = layers.Conv2D(64, 3, activation="relu", padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D(2)(x)
    x = layers.Dropout(0.20)(x)

    x = layers.Conv2D(128, 3, activation="relu", padding="same")(x)
    x = layers.Conv2D(128, 3, activation="relu", padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D(2)(x)
    x = layers.Dropout(0.30)(x)

    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dropout(0.40)(x)
    x = layers.Dense(128, activation="relu")(x)
    out = layers.Dense(NUM_CLASSES, activation="softmax")(x)

    model = models.Model(inp, out)
    opt = tf.keras.optimizers.Nadam(
        learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-7
    )
    model.compile(optimizer=opt, loss="categorical_crossentropy", metrics=["accuracy"])
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True,
                     help="Directory with train/val subfolders, one folder per emotion class")
    ap.add_argument("--out", default="visual_model.h5")
    args = ap.parse_args()

    tf.random.set_seed(RANDOM_SEED)

    datagen_train = tf.keras.preprocessing.image.ImageDataGenerator(
        rescale=1.0 / 255,
        rotation_range=15,
        width_shift_range=0.15,
        height_shift_range=0.15,
        shear_range=0.15,
        zoom_range=0.15,
        horizontal_flip=True,
    )
    datagen_val = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1.0 / 255)

    train_gen = datagen_train.flow_from_directory(
        f"{args.data_dir}/train",
        target_size=(VISUAL_IMG_SIZE, VISUAL_IMG_SIZE),
        color_mode="grayscale",
        classes=EMOTIONS,
        class_mode="categorical",
        batch_size=32,
        seed=RANDOM_SEED,
    )
    val_gen = datagen_val.flow_from_directory(
        f"{args.data_dir}/val",
        target_size=(VISUAL_IMG_SIZE, VISUAL_IMG_SIZE),
        color_mode="grayscale",
        classes=EMOTIONS,
        class_mode="categorical",
        batch_size=32,
        shuffle=False,
    )

    model = build_model()
    model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=150,
        callbacks=[
            callbacks.EarlyStopping(min_delta=5e-5, patience=11, monitor="val_accuracy", restore_best_weights=True),
            callbacks.ReduceLROnPlateau(factor=0.5, patience=7, min_lr=1e-7, monitor="val_accuracy"),
        ],
    )

    val_gen.reset()
    y_pred = model.predict(val_gen).argmax(1)
    y_true = val_gen.classes
    print("Validation accuracy:", accuracy_score(y_true, y_pred))
    print(classification_report(y_true, y_pred, target_names=EMOTIONS))

    model.save(args.out)
    print(f"Saved to {args.out}")


if __name__ == "__main__":
    main()
