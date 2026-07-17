"""
Generates the figures needed for the revised manuscript directly from the
saved probability files (audio_probs.npz, text_probs.npz, visual_probs.npz)
and the fusion evaluation results -- real confusion matrices and a results
comparison chart, not fabricated or placeholder images.

Only needs numpy/pandas/sklearn/matplotlib -- run in either venv.
"""
import argparse
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

from common import EMOTIONS


def plot_confusion(probs, labels, title, out_path):
    preds = probs.argmax(1)
    cm = confusion_matrix(labels, preds, labels=list(range(len(EMOTIONS))))

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(EMOTIONS)))
    ax.set_yticks(range(len(EMOTIONS)))
    ax.set_xticklabels(EMOTIONS, rotation=45, ha="right")
    ax.set_yticklabels(EMOTIONS)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_title(title)
    for i in range(len(EMOTIONS)):
        for j in range(len(EMOTIONS)):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                     color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=8)
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"Saved {out_path}")


def plot_results_bar(table5_csv, out_path):
    df = pd.read_csv(table5_csv)
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = ["#4C72B0", "#4C72B0", "#4C72B0", "#DD8452"]
    bars = ax.bar(df["Method"], df["Accuracy (%)"], color=colors[:len(df)])
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Unimodal vs. Multimodal Fusion Accuracy")
    ax.set_ylim(0, 100)
    plt.xticks(rotation=20, ha="right")
    for bar, val in zip(bars, df["Accuracy (%)"]):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 1, f"{val:.1f}%",
                 ha="center", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"Saved {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--audio_probs", default="audio_probs.npz")
    ap.add_argument("--text_probs", default="text_probs.npz")
    ap.add_argument("--visual_probs", default="visual_probs.npz")
    ap.add_argument("--table5_csv", default="results/table5_corrected.csv")
    ap.add_argument("--out_dir", default="figures/")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    audio = np.load(args.audio_probs)
    plot_confusion(audio["probs"], audio["labels"], "Audio (wav2vec2) Confusion Matrix",
                    os.path.join(args.out_dir, "confusion_audio.png"))

    text = np.load(args.text_probs)
    plot_confusion(text["probs"], text["labels"], "Text Confusion Matrix",
                    os.path.join(args.out_dir, "confusion_text.png"))

    visual = np.load(args.visual_probs)
    plot_confusion(visual["probs"], visual["labels"], "Visual (ResNet50) Confusion Matrix",
                    os.path.join(args.out_dir, "confusion_visual.png"))

    if os.path.exists(args.table5_csv):
        plot_results_bar(args.table5_csv, os.path.join(args.out_dir, "results_comparison.png"))
    else:
        print(f"WARNING: {args.table5_csv} not found, skipping results bar chart")

    print(f"\nAll figures saved to {args.out_dir}/ -- copy these into your LaTeX project directory")


if __name__ == "__main__":
    main()
