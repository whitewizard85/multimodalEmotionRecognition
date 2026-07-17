"""
DIAGNOSTIC SCRIPT -- NOT FOR USE IN THE PAPER.

Companion to test_leaky_hypothesis.py (audio). Tests whether the same
pre-split-augmentation leak that inflated audio accuracy (62% -> 90%) would
also inflate visual accuracy, IF the pipeline were built the naive way
(pre-generate augmented image variants, then split samples into train/test
afterward). The real train_visual.py does NOT have this bug -- Keras's
ImageDataGenerator applies rotation/shift/zoom/flip live, per-epoch, only to
images already sitting in the train/ folder, so it structurally cannot leak
this way. This script deliberately reproduces the naive mistake anyway, to
confirm that reasoning empirically rather than just asserting it.

ANY NUMBER THIS SCRIPT PRODUCES IS NOT VALID FOR THE PAPER.
"""
import argparse
import os
import glob

import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import tensorflow as tf
from tensorflow.keras import callbacks

from common import EMOTIONS, NUM_CLASSES, VISUAL_IMG_SIZE, RANDOM_SEED
from train_visual import build_model


def load_and_resize(path):
    img = Image.open(path).convert("L").resize((VISUAL_IMG_SIZE, VISUAL_IMG_SIZE))
    return np.array(img, dtype=np.float32) / 255.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True,
                     help="The already-built visual_dataset/ directory (train+val "
                          "subfolders will both be pooled and re-split, deliberately "
                          "ignoring the original honest split boundary)")
    ap.add_argument("--n_augmented_per_image", type=int, default=3)
    args = ap.parse_args()

    print("=" * 70)
    print("DIAGNOSTIC RUN -- testing the augmentation-leakage hypothesis for")
    print("the visual modality. This number is NOT valid for the paper.")
    print("=" * 70)

    tf.random.set_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        rotation_range=15, width_shift_range=0.15, height_shift_range=0.15,
        shear_range=0.15, zoom_range=0.15, horizontal_flip=True,
    )

    filepaths, labels = [], []
    for split in ["train", "val"]:
        for emotion in EMOTIONS:
            d = os.path.join(args.data_dir, split, emotion)
            if not os.path.isdir(d):
                continue
            for f in glob.glob(os.path.join(d, "*")):
                filepaths.append(f)
                labels.append(EMOTIONS.index(emotion))
    print(f"Total source images pooled (train+val recombined): {len(filepaths)}")

    print(f"Generating {args.n_augmented_per_image} augmented variants per "
          f"image BEFORE the split (this is the deliberate leak being tested)...")
    X, y, source_file_idx = [], [], []
    for i, (f, label) in enumerate(zip(filepaths, labels)):
        img = load_and_resize(f)
        img_3d = img.reshape(VISUAL_IMG_SIZE, VISUAL_IMG_SIZE, 1)

        X.append(img); y.append(label); source_file_idx.append(i)
        for _ in range(args.n_augmented_per_image):
            aug = datagen.random_transform(img_3d).reshape(VISUAL_IMG_SIZE, VISUAL_IMG_SIZE)
            X.append(aug); y.append(label); source_file_idx.append(i)

        if (i + 1) % 2000 == 0:
            print(f"  {i + 1}/{len(filepaths)} images processed ({len(X)} samples so far)")

    X = np.array(X).reshape(-1, VISUAL_IMG_SIZE, VISUAL_IMG_SIZE, 1).astype(np.float32)
    y_cat = tf.keras.utils.to_categorical(np.array(y), NUM_CLASSES)
    print(f"Total samples (original + augmented, pre-split): {len(X)}")

    indices = np.arange(len(X))
    train_idx, test_idx = train_test_split(
        indices, test_size=0.20, random_state=RANDOM_SEED, stratify=y
    )
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y_cat[train_idx], y_cat[test_idx]

    train_source_files = set(np.array(source_file_idx)[train_idx])
    test_source_files = set(np.array(source_file_idx)[test_idx])
    leaked_files = train_source_files & test_source_files
    print(f"Source images appearing on BOTH sides of the split: "
          f"{len(leaked_files)} / {len(filepaths)} "
          f"({100*len(leaked_files)/len(filepaths):.1f}%) -- this is the leak.")

    model = build_model()
    model.fit(
        X_train, y_train,
        validation_split=0.1,
        epochs=150,
        batch_size=32,
        callbacks=[
            callbacks.EarlyStopping(min_delta=5e-5, patience=11, monitor="val_accuracy", restore_best_weights=True),
            callbacks.ReduceLROnPlateau(factor=0.5, patience=7, min_lr=1e-7, monitor="val_accuracy"),
        ],
    )

    y_pred = model.predict(X_test).argmax(1)
    y_true = y_test.argmax(1)
    print("\n" + "=" * 70)
    print("DIAGNOSTIC (LEAKY) visual test accuracy:", accuracy_score(y_true, y_pred))
    print("REMINDER: this number is NOT valid for the paper.")
    print("=" * 70)
    print(classification_report(y_true, y_pred, target_names=EMOTIONS))


if __name__ == "__main__":
    main()
