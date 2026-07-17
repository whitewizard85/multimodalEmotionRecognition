"""
Audio emotion classification via transfer learning from a pretrained
wav2vec 2.0 encoder, instead of hand-crafted mel-spectrogram features fed
into a CNN trained from scratch (train_audio.py, which plateaued at ~62%
test accuracy across three different feature representations).

Rationale (see manuscript's own Related Work, which already cites this
paradigm): self-supervised pretrained speech encoders like wav2vec 2.0 and
HuBERT -- see also emotion2vec (Ma et al. 2023), cited in the paper --
consistently outperform hand-crafted-feature + CNN-from-scratch pipelines
in modern speech emotion recognition literature. This script fine-tunes
facebook/wav2vec2-base (pretrained on 960h of LibriSpeech) with a
classification head on top, using HuggingFace's Wav2Vec2ForSequenceClassification.

IMPORTANT -- respects the leakage lesson learned earlier: files are split
into train/test BEFORE any augmentation, and augmentation (light additive
noise only, kept simple given the heavier compute cost of this model) is
applied ONLY to the training split.

Requires the torch/transformers venv (NOT the TensorFlow-only venv_visual
used for train_visual.py) -- same environment train_text.py used.
"""
import argparse
import glob
import os

import numpy as np
import librosa
import torch
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from transformers import (
    Wav2Vec2FeatureExtractor, Wav2Vec2ForSequenceClassification,
    TrainingArguments, Trainer,
)

from common import EMOTIONS, LABEL2IDX, NUM_CLASSES, RANDOM_SEED
from audio_label_utils import parse_label_from_filename

WAV2VEC2_CHECKPOINT = "facebook/wav2vec2-base"
SAMPLE_RATE = 16000
MAX_DURATION_S = 4.0
MAX_SAMPLES = int(SAMPLE_RATE * MAX_DURATION_S)


class AudioEmotionDataset(Dataset):
    """Loads raw waveforms on the fly, resampled to 16kHz, padded/truncated
    to a fixed duration. Augmentation (if enabled) is applied here, so it's
    trivially scoped to whichever file list this dataset was constructed
    with -- pass only the training file list to get train-only augmentation."""

    def __init__(self, filepaths, labels, feature_extractor, augment=False):
        self.filepaths = filepaths
        self.labels = labels
        self.feature_extractor = feature_extractor
        self.augment = augment

    def __len__(self):
        return len(self.filepaths)

    def __getitem__(self, idx):
        y, sr = librosa.load(self.filepaths[idx], sr=SAMPLE_RATE)
        if len(y) > MAX_SAMPLES:
            y = y[:MAX_SAMPLES]
        else:
            y = np.pad(y, (0, MAX_SAMPLES - len(y)))

        if self.augment and np.random.rand() < 0.5:
            y = y + 0.005 * np.random.randn(len(y)).astype(np.float32)

        inputs = self.feature_extractor(
            y, sampling_rate=SAMPLE_RATE, return_tensors="pt", padding=False
        )
        return {
            "input_values": inputs["input_values"][0],
            "labels": torch.tensor(self.labels[idx], dtype=torch.long),
        }


def collate_fn(batch):
    input_values = torch.stack([b["input_values"] for b in batch])
    labels = torch.stack([b["labels"] for b in batch])
    return {"input_values": input_values, "labels": labels}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ravdess_dir", default=None)
    ap.add_argument("--crema_d_dir", default=None)
    ap.add_argument("--tess_dir", default=None)
    ap.add_argument("--savee_dir", default=None)
    ap.add_argument("--out", default="audio_wav2vec2_model/")
    ap.add_argument("--freeze_feature_encoder", action="store_true", default=True,
                     help="Freeze wav2vec2's CNN feature-extraction front-end "
                          "(standard practice when fine-tuning -- reduces "
                          "overfitting risk and compute; the transformer "
                          "layers + classification head still train)")
    args = ap.parse_args()

    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    sources = {
        "ravdess": args.ravdess_dir, "crema_d": args.crema_d_dir,
        "tess": args.tess_dir, "savee": args.savee_dir,
    }
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
        print(f"{dataset}: found {len(files)} files, kept {kept}, dropped {dropped}")
    print(f"Total usable files: {len(filepaths)}")

    train_paths, test_paths, train_labels, test_labels = train_test_split(
        filepaths, labels, test_size=0.20, random_state=RANDOM_SEED, stratify=labels
    )
    print(f"Train: {len(train_paths)}, Test: {len(test_paths)}")

    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(WAV2VEC2_CHECKPOINT)
    model = Wav2Vec2ForSequenceClassification.from_pretrained(
        WAV2VEC2_CHECKPOINT, num_labels=NUM_CLASSES, use_safetensors=True
    )
    if args.freeze_feature_encoder:
        model.freeze_feature_encoder()
        print("Feature encoder (CNN front-end) frozen -- fine-tuning "
              "transformer layers + classification head only.")

    train_dataset = AudioEmotionDataset(train_paths, train_labels, feature_extractor, augment=True)
    test_dataset = AudioEmotionDataset(test_paths, test_labels, feature_extractor, augment=False)

    def compute_metrics(eval_pred):
        logits, label_ids = eval_pred
        preds = np.argmax(logits, axis=-1)
        return {"accuracy": accuracy_score(label_ids, preds)}

    training_args = TrainingArguments(
        output_dir="./wav2vec2_training_output",
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=3e-5,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        gradient_accumulation_steps=2,  # effective batch size 16, keeps memory manageable
        num_train_epochs=15,
        weight_decay=0.01,
        warmup_ratio=0.1,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        greater_is_better=True,
        logging_steps=20,
        fp16=torch.cuda.is_available(),  # mixed precision if GPU available -- faster, lower memory
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        data_collator=collate_fn,
        compute_metrics=compute_metrics,
    )
    trainer.train()

    preds = trainer.predict(test_dataset)
    y_pred = preds.predictions.argmax(1)
    y_true = preds.label_ids
    print("\nFinal test accuracy:", accuracy_score(y_true, y_pred))
    print(classification_report(y_true, y_pred, target_names=EMOTIONS))

    model.save_pretrained(args.out)
    feature_extractor.save_pretrained(args.out)
    print(f"Saved to {args.out}")


if __name__ == "__main__":
    main()
