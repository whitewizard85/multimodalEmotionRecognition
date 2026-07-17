"""
Runs the trained wav2vec2 audio model on its held-out test set and saves
predicted probabilities + true labels to audio_probs.npz.

Recreates the EXACT same file-level train/test split used during training
(same file collection logic, same random seed) -- this is critical, since
re-splitting differently would either leak training files into "test" or
evaluate on the wrong held-out set entirely.

Run in the torch venv (same one used for train_audio_wav2vec2.py).
"""
import argparse

import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2ForSequenceClassification
import glob
import os

from common import LABEL2IDX, RANDOM_SEED
from audio_label_utils import parse_label_from_filename
from train_audio_wav2vec2 import AudioEmotionDataset, collate_fn


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", default="audio_wav2vec2_model/")
    ap.add_argument("--ravdess_dir", default=None)
    ap.add_argument("--crema_d_dir", default=None)
    ap.add_argument("--tess_dir", default=None)
    ap.add_argument("--savee_dir", default=None)
    ap.add_argument("--out", default="audio_probs.npz")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Recreate the exact same file list + split as train_audio_wav2vec2.py
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

    _, test_paths, _, test_labels = train_test_split(
        filepaths, labels, test_size=0.20, random_state=RANDOM_SEED, stratify=labels
    )
    print(f"Recreated held-out audio test set: {len(test_paths)} files")

    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(args.model_dir)
    model = Wav2Vec2ForSequenceClassification.from_pretrained(args.model_dir, use_safetensors=True)
    model.to(device).eval()

    test_dataset = AudioEmotionDataset(test_paths, test_labels, feature_extractor, augment=False)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, collate_fn=collate_fn)

    all_probs, all_labels = [], []
    with torch.no_grad():
        for batch in test_loader:
            input_values = batch["input_values"].to(device)
            logits = model(input_values=input_values).logits
            probs = torch.softmax(logits, dim=-1).cpu().numpy()
            all_probs.append(probs)
            all_labels.append(batch["labels"].numpy())

    probs = np.concatenate(all_probs, axis=0)
    labels_arr = np.concatenate(all_labels, axis=0)
    print(f"Saved probabilities shape: {probs.shape}, labels shape: {labels_arr.shape}")

    np.savez(args.out, probs=probs, labels=labels_arr)
    print(f"Saved to {args.out}")


if __name__ == "__main__":
    main()
