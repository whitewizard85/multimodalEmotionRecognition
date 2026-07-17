"""
Trains the audio CNN, adapted from thesis Section 3.3 / Table 1.

Expects --ravdess_dir/--crema_d_dir/--tess_dir/--savee_dir pointing at the
raw audio files for each corpus.

Feature representation: genuine time-series mel-spectrogram (time_steps,
n_mels) -- frequency bins as channels, time as the Conv1D sequence axis.
This is the third iteration of this pipeline; two earlier approaches were
tried and empirically underperformed:
  1. Mel-spectrogram averaged across frequency into a flat 1D sequence
     (~48% test accuracy) -- destroyed almost all spectral information.
  2. Concatenated global mean+std summary statistics of five feature types
     into one flat 1000-length vector, matching the thesis's literal
     (1000, 1) input shape (~62-64% test accuracy) -- discarded temporal
     structure, and Conv1D's sliding kernel has no real locality to exploit
     over unrelated concatenated statistics.
This version deviates from the thesis's literal Table 1 input shape
((1000, 1) -> (AUDIO_TIME_STEPS, AUDIO_N_MELS)) but is architecturally
sound for what Conv1D is actually designed to learn from.

Target: ~97% test accuracy (thesis-reported figure, achieved with an
unknown/undocumented feature pipeline -- treat as an aspirational
reference, not a guarantee this exact approach reaches it).
"""
import argparse
import glob
import os

import numpy as np
import librosa
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks

from common import EMOTIONS, LABEL2IDX, NUM_CLASSES, AUDIO_TIME_STEPS, AUDIO_N_MELS, RANDOM_SEED

from audio_label_utils import parse_label_from_filename


def extract_features_from_audio(y, sr, time_steps: int = AUDIO_TIME_STEPS, n_mels: int = AUDIO_N_MELS):
    """Genuine time-series mel-spectrogram representation: shape (time_steps,
    n_mels), i.e. frequency bins as channels, time as the sequence axis --
    this is what Conv1D actually needs to exploit local structure via its
    sliding kernel. Two earlier approaches were tried and plateaued:
      1. Averaging the mel-spectrogram across frequency (~48% test acc) --
         destroyed almost all spectral information.
      2. Concatenating global mean+std summary statistics of five different
         feature types into one flat vector (~62-64% test acc) -- discarded
         all temporal structure, and a sliding Conv1D kernel over unrelated
         concatenated statistics has no real locality to exploit.
    This version keeps genuine per-frame detail so neighboring "columns"
    (time frames) are actually related, which is what a kernel-5 Conv1D
    is designed to learn from.
    """
    if len(y) < 400:  # guard against near-empty/corrupt clips
        y = np.pad(y, (0, 400 - len(y)))

    hop = int(0.010 * sr)
    n_fft = int(0.025 * sr)

    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=n_fft, hop_length=hop,
                                          window="hamming", n_mels=n_mels)
    mel_db = librosa.power_to_db(mel, ref=np.max)  # shape (n_mels, time)
    feat = mel_db.T  # -> (time, n_mels): time as sequence axis, mel bands as channels

    feat = np.nan_to_num(feat, nan=0.0, posinf=0.0, neginf=0.0)
    mean, std = feat.mean(), feat.std()
    feat = np.clip((feat - mean) / (std + 1e-8), -6, 6)

    # Pad/truncate along the TIME axis only (channel count is always n_mels)
    if feat.shape[0] >= time_steps:
        feat = feat[:time_steps, :]
    else:
        feat = np.pad(feat, ((0, time_steps - feat.shape[0]), (0, 0)))
    return feat.astype(np.float32)


def extract_features(path: str, time_steps: int = AUDIO_TIME_STEPS, n_mels: int = AUDIO_N_MELS):
    y, sr = librosa.load(path, sr=None)
    return extract_features_from_audio(y, sr, time_steps, n_mels)


def augment_noise(y, rate=0.005):
    noise = np.random.randn(len(y))
    return y + rate * noise


def augment_pitch(y, sr, n_steps):
    return librosa.effects.pitch_shift(y, sr=sr, n_steps=n_steps)


def augment_stretch(y, rate):
    return librosa.effects.time_stretch(y, rate=rate)


def build_model(use_dropout: bool = False):
    """NOTE: the thesis's own Table 1 does not list dropout for this specific
    audio CNN (unlike the visual CNN, which does use it). use_dropout=True
    was tried as a deviation to combat overfitting seen in an earlier run,
    but the very next run (dropout + augmentation together) collapsed
    completely: loss frozen at a constant value, accuracy near-random,
    model predicting a single class for every input. Two candidate causes
    were changed at once, which was a debugging mistake -- defaulting
    dropout OFF here to isolate it. The other candidate fix already applied
    below (lower learning rate + gradient clipping) may be sufficient on
    its own; only re-enable dropout (--use_dropout) after confirming that."""
    inp = layers.Input(shape=(AUDIO_TIME_STEPS, AUDIO_N_MELS))
    x = layers.Conv1D(512, 5, activation="relu")(inp)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling1D(2)(x)
    if use_dropout:
        x = layers.Dropout(0.3)(x)
    x = layers.Conv1D(512, 5, activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling1D(2)(x)
    if use_dropout:
        x = layers.Dropout(0.3)(x)
    x = layers.Conv1D(256, 5, activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling1D(2)(x)
    if use_dropout:
        x = layers.Dropout(0.3)(x)
    x = layers.Conv1D(256, 5, activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Flatten()(x)
    x = layers.Dense(512, activation="relu")(x)
    if use_dropout:
        x = layers.Dropout(0.5)(x)
    out = layers.Dense(NUM_CLASSES, activation="softmax")(x)
    model = models.Model(inp, out)
    # Default Adam (lr=1e-3) caused a full training collapse in practice once
    # augmentation quadrupled the steps-per-epoch (274 -> 1095): loss froze
    # at a constant value for 15+ epochs and the model degenerated to
    # predicting a single class for every input -- a dying-ReLU cascade from
    # too-aggressive early updates. Lower LR + gradient clipping fixes this.
    opt = tf.keras.optimizers.Adam(learning_rate=1e-4, clipnorm=1.0)
    model.compile(optimizer=opt, loss="categorical_crossentropy", metrics=["accuracy"])
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ravdess_dir", default=None)
    ap.add_argument("--crema_d_dir", default=None)
    ap.add_argument("--tess_dir", default=None)
    ap.add_argument("--savee_dir", default=None)
    ap.add_argument("--out", default="audio_model.h5")
    ap.add_argument("--no_augmentation", action="store_true",
                     help="Disable noise/pitch/stretch augmentation (faster, "
                          "but expect lower accuracy -- the thesis explicitly "
                          "credits augmentation for its generalization gains)")
    ap.add_argument("--use_dropout", action="store_true",
                     help="Enable dropout (currently OFF by default after it "
                          "contributed to a training collapse when combined "
                          "with augmentation -- isolate and test carefully "
                          "before relying on this)")
    args = ap.parse_args()

    tf.random.set_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    sources = {
        "ravdess": args.ravdess_dir,
        "crema_d": args.crema_d_dir,
        "tess": args.tess_dir,
        "savee": args.savee_dir,
    }
    # Collect (filepath, label) pairs FIRST, before any feature extraction --
    # this lets us split into train/test on file paths, so augmentation can
    # be applied only to the training split and never touches test data.
    filepaths, labels = [], []
    for dataset, d in sources.items():
        if not d:
            print(f"Skipping {dataset} (no directory given)")
            continue
        files = glob.glob(os.path.join(d, "**", "*.wav"), recursive=True)
        kept, dropped = 0, 0
        for f in files:
            label = parse_label_from_filename(f, dataset=dataset)
            if label is None or label not in LABEL2IDX:
                dropped += 1
                continue
            filepaths.append(f)
            labels.append(LABEL2IDX[label])
            kept += 1
        print(f"{dataset}: found {len(files)} files, kept {kept}, dropped {dropped} "
              f"(unparseable filenames -- inspect these if dropped count is high)")

    print(f"Total usable files: {len(filepaths)}")

    train_paths, test_paths, train_labels, test_labels = train_test_split(
        filepaths, labels, test_size=0.20, random_state=RANDOM_SEED, stratify=labels
    )

    print("Extracting test features (no augmentation)...")
    X_test = np.array([extract_features(f) for f in test_paths])
    y_test = tf.keras.utils.to_categorical(np.array(test_labels), NUM_CLASSES)

    print(f"Extracting train features{'(no augmentation)' if args.no_augmentation else ', with noise/pitch/stretch augmentation'}...")
    X_train, y_train_idx = [], []
    for i, (f, label) in enumerate(zip(train_paths, train_labels)):
        y_raw, sr = librosa.load(f, sr=None)
        X_train.append(extract_features_from_audio(y_raw, sr))
        y_train_idx.append(label)

        if not args.no_augmentation:
            try:
                X_train.append(extract_features_from_audio(augment_noise(y_raw), sr))
                y_train_idx.append(label)
                X_train.append(extract_features_from_audio(augment_pitch(y_raw, sr, n_steps=np.random.uniform(-2, 2)), sr))
                y_train_idx.append(label)
                X_train.append(extract_features_from_audio(augment_stretch(y_raw, rate=np.random.uniform(0.85, 1.15)), sr))
                y_train_idx.append(label)
            except Exception as e:
                pass  # some very short clips can fail stretch/pitch -- skip augmentation for those, keep the original

        if (i + 1) % 1000 == 0:
            print(f"  {i + 1}/{len(train_paths)} training files processed "
                  f"({len(X_train)} samples so far with augmentation)")

    X_train = np.array(X_train)
    y_train = tf.keras.utils.to_categorical(np.array(y_train_idx), NUM_CLASSES)
    print(f"Final training set size: {len(X_train)} (from {len(train_paths)} original files)")

    # Sanity check on the actual feature data before committing to a full
    # training run -- catches NaN/Inf leakage or degenerate all-zero
    # vectors from failed augmentation (e.g. pitch_shift/time_stretch edge
    # cases on very short clips) that would otherwise silently produce a
    # collapsed model 20+ minutes later.
    n_nan = np.isnan(X_train).sum()
    n_inf = np.isinf(X_train).sum()
    zero_vec_frac = (np.abs(X_train).sum(axis=(1, 2)) < 1e-6).mean()
    print(f"Training data sanity check: {n_nan} NaNs, {n_inf} Infs, "
          f"{zero_vec_frac*100:.2f}% all-zero feature vectors")
    if n_nan > 0 or n_inf > 0:
        raise RuntimeError("NaN/Inf found in training features -- fix the "
                            "augmentation/feature pipeline before training")
    if zero_vec_frac > 0.02:
        print("WARNING: more than 2% of training samples are degenerate "
              "all-zero vectors -- likely failed augmentation calls. "
              "Consider investigating before trusting the training run.")

    model = build_model(use_dropout=args.use_dropout)
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
    print("Test accuracy:", accuracy_score(y_true, y_pred))
    print(classification_report(y_true, y_pred, target_names=EMOTIONS))

    model.save(args.out)
    print(f"Saved to {args.out}")


if __name__ == "__main__":
    main()
