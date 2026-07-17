"""
Runs the trained text model (BERT or RoBERTa -- same interface, works with
either) on its held-out test set and saves predicted probabilities + true
labels to text_probs.npz.

Recreates the EXACT same train/test split used during training (same data
loading, same random seed) so this is a genuine held-out evaluation, not a
re-shuffled one.

Run in the torch venv.
"""
import argparse

import numpy as np
import torch
import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from common import EMOTIONS, BERT_MAX_LENGTH, RANDOM_SEED
from train_text import load_goemotions, load_dair_ai_emotion


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", default="text_model/",
                     help="Path to either the BERT or RoBERTa saved model "
                          "directory -- AutoTokenizer/AutoModel handle both")
    ap.add_argument("--out", default="text_probs.npz")
    ap.add_argument("--batch_size", type=int, default=32)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Recreate the exact same combined dataset + split as train_text.py /
    # train_text_roberta.py (both use the identical loading + split logic)
    goemo = load_goemotions()
    dair = load_dair_ai_emotion()
    df = pd.concat([goemo, dair], ignore_index=True).drop_duplicates(subset="text")
    _, test_df = train_test_split(
        df, test_size=0.15, random_state=RANDOM_SEED, stratify=df.label
    )
    print(f"Recreated held-out text test set: {len(test_df)} rows")

    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(args.model_dir, use_safetensors=True)
    model.to(device).eval()

    texts = test_df.text.tolist()
    labels = test_df.label.values

    all_probs = []
    with torch.no_grad():
        for i in range(0, len(texts), args.batch_size):
            batch = texts[i:i + args.batch_size]
            enc = tokenizer(batch, padding="max_length", truncation=True,
                             max_length=BERT_MAX_LENGTH, return_tensors="pt").to(device)
            logits = model(**enc).logits
            probs = torch.softmax(logits, dim=-1).cpu().numpy()
            all_probs.append(probs)

    probs = np.concatenate(all_probs, axis=0)
    print(f"Saved probabilities shape: {probs.shape}, labels shape: {labels.shape}")

    np.savez(args.out, probs=probs, labels=labels)
    print(f"Saved to {args.out}")


if __name__ == "__main__":
    main()
