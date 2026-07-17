"""
Runs the actual multimodal fusion evaluation, producing real numbers for a
corrected Table 5, an ablation table (modality subsets), and a
missing-modality robustness test.

Framework-agnostic: takes pre-computed probability arrays from each
modality's own predict_*.py script, rather than loading model objects here
-- avoids ever needing TensorFlow and PyTorch loaded in the same process.

IMPORTANT: this evaluates fusion over SYNTHETICALLY PAIRED same-label
samples across three independently-collected unimodal test sets. There is
no real simultaneous audio+text+face data here. Because samples within a
synthetic triplet come from unrelated underlying instances that only share
a label, whatever makes one modality's sample hard to classify has no
relationship to what makes another modality's sample hard -- unlike real
multimodal data, where a genuinely ambiguous moment tends to be ambiguous
across voice, face, and words together, since all three come from the same
event. This means fusion errors here are artificially decorrelated across
modalities, which is exactly the setup where ensembling produces large
accuracy gains. Part of the fusion improvement reported below may therefore
be a property of the synthetic pairing rather than evidence of how well
fusion would perform on genuinely paired data. Report this explicitly in
the paper (Limitations) rather than presenting the fusion gain at face value.

MULTI-SEED VARIANCE: since the synthetic pairing is itself a random
process (which samples get matched with which, within each class), a
single seed's result could overstate precision. This script repeats the
pairing across multiple seeds and reports mean +/- std for every fusion
and ablation number, rather than a single point estimate. Single-modality
accuracies do not depend on pairing and are computed once.

Expected inputs (each produced by the corresponding predict_*.py script):
  --audio_probs  : .npz with arrays 'probs' (N, 7) float32, 'labels' (N,) int
  --text_probs   : .npz with arrays 'probs' (N, 7) float32, 'labels' (N,) int
  --visual_probs : .npz with arrays 'probs' (N, 7) float32, 'labels' (N,) int
"""
import argparse
import os

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

from common import EMOTIONS, NUM_CLASSES, RANDOM_SEED


def load_probs(path):
    data = np.load(path)
    return data["probs"], data["labels"]


def group_by_label(probs, labels):
    grouped = {c: [] for c in range(NUM_CLASSES)}
    for p, l in zip(probs, labels):
        grouped[int(l)].append(p)
    return {c: np.array(v) for c, v in grouped.items()}


def build_synthetic_triplets(audio_g, text_g, visual_g, seed):
    """Positionally pairs same-label samples across the three modalities'
    probability arrays, truncated to the minimum count per class. Re-running
    with a different seed re-shuffles which specific samples get paired
    together within each class, giving an estimate of how sensitive the
    fusion numbers are to this arbitrary pairing choice."""
    rng = np.random.default_rng(seed)
    triplets = {"audio": [], "text": [], "visual": [], "label": []}
    for c in range(NUM_CLASSES):
        n = min(len(audio_g[c]), len(text_g[c]), len(visual_g[c]))
        if n == 0:
            continue
        a_idx = rng.permutation(len(audio_g[c]))[:n]
        t_idx = rng.permutation(len(text_g[c]))[:n]
        v_idx = rng.permutation(len(visual_g[c]))[:n]
        triplets["audio"].append(audio_g[c][a_idx])
        triplets["text"].append(text_g[c][t_idx])
        triplets["visual"].append(visual_g[c][v_idx])
        triplets["label"].extend([c] * n)
    return (
        np.concatenate(triplets["audio"]),
        np.concatenate(triplets["text"]),
        np.concatenate(triplets["visual"]),
        np.array(triplets["label"]),
    )


def fuse(prob_list):
    """Eq. 1 of the paper: unweighted average of available modality
    probability vectors."""
    return np.mean(np.stack(prob_list, axis=0), axis=0)


def score(fused_probs, labels):
    preds = fused_probs.argmax(1)
    return (
        accuracy_score(labels, preds) * 100,
        f1_score(labels, preds, average="macro"),
    )


def mean_std(values):
    arr = np.array(values)
    return arr.mean(), arr.std()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--audio_probs", required=True)
    ap.add_argument("--text_probs", required=True)
    ap.add_argument("--visual_probs", required=True)
    ap.add_argument("--out", default="results/")
    ap.add_argument("--n_seeds", type=int, default=20,
                     help="Number of different random pairings to average "
                          "over, to report variance rather than a single "
                          "point estimate for the fusion/ablation numbers.")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    print("Loading pre-computed probabilities...")
    audio_probs, y_audio = load_probs(args.audio_probs)
    text_probs, y_text = load_probs(args.text_probs)
    visual_probs, y_visual = load_probs(args.visual_probs)
    print(f"Audio: {audio_probs.shape}, Text: {text_probs.shape}, Visual: {visual_probs.shape}")

    # Single-modality accuracy does not depend on the synthetic pairing --
    # computed once, deterministically.
    single_rows = []
    for name, probs, y in [("Audio only (wav2vec2)", audio_probs, y_audio),
                            ("Text only", text_probs, y_text),
                            ("Visual only (ResNet50)", visual_probs, y_visual)]:
        acc, f1 = score(probs, y)
        single_rows.append({"Method": name, "Accuracy (%)": round(acc, 1), "Macro F1": round(f1, 3)})
        print(f"{name}: accuracy={acc:.1f}%  macro-F1={f1:.3f}")

    audio_g = group_by_label(audio_probs, y_audio)
    text_g = group_by_label(text_probs, y_text)
    visual_g = group_by_label(visual_probs, y_visual)

    print(f"\nRunning {args.n_seeds} different synthetic pairings to estimate "
          f"variance in the fusion/ablation numbers...")

    combo_names = ["Audio+Text+Visual (full fusion)", "Audio+Text", "Audio+Visual", "Text+Visual",
                   "Fusion w/ one modality randomly missing"]
    results = {name: {"acc": [], "f1": []} for name in combo_names}
    n_triplets_per_seed = []

    for seed_offset in range(args.n_seeds):
        seed = RANDOM_SEED + seed_offset
        a, t, v, labels = build_synthetic_triplets(audio_g, text_g, visual_g, seed=seed)
        n_triplets_per_seed.append(len(labels))

        combos = {
            "Audio+Text+Visual (full fusion)": [a, t, v],
            "Audio+Text": [a, t],
            "Audio+Visual": [a, v],
            "Text+Visual": [t, v],
        }
        for name, parts in combos.items():
            fused = fuse(parts)
            acc, f1 = score(fused, labels)
            results[name]["acc"].append(acc)
            results[name]["f1"].append(f1)

        rng = np.random.default_rng(seed)
        drop_choice = rng.integers(0, 3, size=len(labels))
        robust_fused = np.zeros_like(a)
        for i in range(len(labels)):
            parts = [a[i], t[i], v[i]]
            del parts[drop_choice[i]]
            robust_fused[i] = np.mean(parts, axis=0)
        racc, rf1 = score(robust_fused, labels)
        results["Fusion w/ one modality randomly missing"]["acc"].append(racc)
        results["Fusion w/ one modality randomly missing"]["f1"].append(rf1)

    print(f"Synthetic triplet count per seed: {n_triplets_per_seed[0]} "
          f"(constant across seeds -- only which samples are paired changes)")

    ablation_rows = []
    for name in combo_names:
        acc_mean, acc_std = mean_std(results[name]["acc"])
        f1_mean, f1_std = mean_std(results[name]["f1"])
        ablation_rows.append({
            "Method": name,
            "Accuracy (%) mean": round(acc_mean, 1),
            "Accuracy (%) std": round(acc_std, 2),
            "Macro F1 mean": round(f1_mean, 3),
            "Macro F1 std": round(f1_std, 3),
        })
        print(f"{name}: accuracy={acc_mean:.1f}% (+/-{acc_std:.2f})  "
              f"macro-F1={f1_mean:.3f} (+/-{f1_std:.3f})  over {args.n_seeds} pairings")

    full_acc_mean = ablation_rows[0]["Accuracy (%) mean"]
    robust_acc_mean = ablation_rows[4]["Accuracy (%) mean"]
    print(f"\nMissing-modality robustness delta vs full fusion (mean): "
          f"{robust_acc_mean - full_acc_mean:+.1f} pts")

    table5_rows = single_rows + [{
        "Method": ablation_rows[0]["Method"],
        "Accuracy (%)": ablation_rows[0]["Accuracy (%) mean"],
        "Macro F1": ablation_rows[0]["Macro F1 mean"],
    }]
    pd.DataFrame(table5_rows).to_csv(os.path.join(args.out, "table5_corrected.csv"), index=False)
    pd.DataFrame(ablation_rows).to_csv(os.path.join(args.out, "ablation.csv"), index=False)
    print(f"\nSaved results to {args.out}/table5_corrected.csv and {args.out}/ablation.csv "
          f"(ablation.csv now includes mean/std across {args.n_seeds} synthetic pairings)")


if __name__ == "__main__":
    main()
