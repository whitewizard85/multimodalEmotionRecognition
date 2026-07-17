"""
DIAGNOSTIC SCRIPT -- NOT FOR USE IN THE PAPER.

Tests a specific hypothesis: that the thesis's reported 97% audio accuracy
(vs. our honestly-evaluated ~61-64% across three different feature
representations) might be explained by a common, usually unintentional
pipeline mistake -- applying augmentation BEFORE the train/test split,
so near-duplicate augmented copies of the same underlying recording end up
on both sides of the split. The model doesn't need to generalize to the
test set's actual content in that case; it just needs to recognize a
noisy/pitched/stretched variant of something it already memorized during
training. This inflates reported accuracy without reflecting real
generalization.

This script deliberately reproduces that mistake, on purpose, to see if it
reproduces something closer to 97%. If it does, that's reasonably strong
evidence for the leakage explanation. If it doesn't, the gap has some other
cause and this hypothesis can be ruled out.

ANY NUMBER THIS SCRIPT PRODUCES IS NOT VALID FOR THE PAPER. It exists only
to test a hypothesis about the discrepancy. Do not report this accuracy in
Table 5, the response to reviewers, or anywhere else presented as a real
result -- it is evaluated on a compromised, leaky split by design.
"""
import argparse
import glob
import os

import numpy as np
import librosa
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import tensorflow as tf
from tensorflow.keras import callbacks

from common import EMOTIONS, LABEL2IDX, NUM_CLASSES, RANDOM_SEED
from train_audio import (
    parse_label_from_filename, extract_features_from_audio,
    augment_noise, augment_pitch, augment_stretch, build_model,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ravdess_dir", default=None)
    ap.add_argument("--crema_d_dir", default=None)
    ap.add_argument("--tess_dir", default=None)
    ap.add_argument("--savee_dir", default=None)
    args = ap.parse_args()

    print("=" * 70)
    print("DIAGNOSTIC RUN -- testing the augmentation-leakage hypothesis.")
    print("This number is NOT valid for the paper. See module docstring.")
    print("=" * 70)

    tf.random.set_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    sources = {
        "ravdess": args.ravdess_dir, "crema_d": args.crema_d_dir,
        "tess": args.tess_dir, "savee": args.savee_dir,
    }
    filepaths, labels = [], []
    for dataset, d in sources.items():
        if not d:
            continue
        files = glob.glob(os.path.join(d, "**", "*.wav"), recursive=True)
        for f in files:
            label = parse_label_from_filename(f, dataset=dataset)
            if label is not None and label in LABEL2IDX:
                filepaths.append(f)
                labels.append(LABEL2IDX[label])
    print(f"Total usable files: {len(filepaths)}")

    # THE LEAK: augment every file FIRST, producing original + 3 augmented
    # variants per file as independent "samples" -- then split AFTER, so
    # variants of the same source file can land on opposite sides of the
    # train/test split. This is the mistake being tested.
    print("Extracting features WITH augmentation, BEFORE the train/test "
          "split (this is the deliberate leak being tested)...")
    X, y, source_file_idx = [], [], []
    for i, (f, label) in enumerate(zip(filepaths, labels)):
        y_raw, sr = librosa.load(f, sr=None)
        X.append(extract_features_from_audio(y_raw, sr))
        y.append(label)
        source_file_idx.append(i)
        try:
            X.append(extract_features_from_audio(augment_noise(y_raw), sr))
            y.append(label); source_file_idx.append(i)
            X.append(extract_features_from_audio(augment_pitch(y_raw, sr, n_steps=np.random.uniform(-2, 2)), sr))
            y.append(label); source_file_idx.append(i)
            X.append(extract_features_from_audio(augment_stretch(y_raw, rate=np.random.uniform(0.85, 1.15)), sr))
            y.append(label); source_file_idx.append(i)
        except Exception:
            pass
        if (i + 1) % 1000 == 0:
            print(f"  {i + 1}/{len(filepaths)} files processed ({len(X)} samples so far)")

    X = np.array(X)
    y_cat = tf.keras.utils.to_categorical(np.array(y), NUM_CLASSES)
    print(f"Total samples (original + augmented, pre-split): {len(X)}")

    # Split AFTER augmentation, on sample indices, NOT on source file --
    # this is what allows leakage: augmented siblings of the same original
    # file can end up split across train and test.
    indices = np.arange(len(X))
    train_idx, test_idx = train_test_split(
        indices, test_size=0.20, random_state=RANDOM_SEED, stratify=y
    )
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y_cat[train_idx], y_cat[test_idx]

    # Quantify the actual leakage for the record
    train_source_files = set(np.array(source_file_idx)[train_idx])
    test_source_files = set(np.array(source_file_idx)[test_idx])
    leaked_files = train_source_files & test_source_files
    print(f"Source files appearing on BOTH sides of the split: "
          f"{len(leaked_files)} / {len(filepaths)} "
          f"({100*len(leaked_files)/len(filepaths):.1f}%) -- this is the leak.")

    model = build_model(use_dropout=False)
    model.fit(
        X_train, y_train,
        validation_split=0.1,
        epochs=100,
        batch_size=32,
        callbacks=[
            callbacks.EarlyStopping(patience=15, restore_best_weights=True),
            callbacks.ReduceLROnPlateau(factor=0.5, patience=7),
        ],
    )

    y_pred = model.predict(X_test).argmax(1)
    y_true = y_test.argmax(1)
    print("\n" + "=" * 70)
    print("DIAGNOSTIC (LEAKY) test accuracy:", accuracy_score(y_true, y_pred))
    print("REMINDER: this number is NOT valid for the paper -- it is")
    print("evaluated on a split with deliberate train/test leakage, only")
    print("to test whether leakage explains the thesis's reported 97%.")
    print("=" * 70)
    print(classification_report(y_true, y_pred, target_names=EMOTIONS))


if __name__ == "__main__":
    main()
