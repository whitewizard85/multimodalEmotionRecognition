"""
Fine-tunes bert-base-uncased exactly per thesis Section 3.4 (TrainingArguments
values transcribed verbatim from the thesis's own screenshotted config).

Combines GoEmotions (mapped via Ekman grouping) + dair-ai/emotion + the
Twitter-based "Emotion Dataset for Emotion Recognition Tasks" as described
in thesis 3.4.1, then augments the disgust class per 3.4.2.

Target: ~94% overall accuracy (thesis-reported figure), with disgust F1
around 0.94 if the T5 paraphrase augmentation step is reproduced.
"""
import argparse

import numpy as np
import pandas as pd
from datasets import load_dataset, Dataset, concatenate_datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from transformers import (
    BertTokenizer, BertForSequenceClassification,
    TrainingArguments, Trainer,
)

from common import EMOTIONS, LABEL2IDX, NUM_CLASSES, BERT_MAX_LENGTH, RANDOM_SEED, GOEMOTIONS_EKMAN_TO_OURS


def load_goemotions():
    ds = load_dataset("go_emotions", "simplified")["train"]
    # Map each GoEmotions fine-grained label id to its name, then to Ekman
    # group, then to our label set. Multi-label rows -> take first mappable
    # label only (thesis doesn't describe multi-label handling; simplifying
    # here — revisit if you have Carmine's actual preprocessing code).
    id2name = ds.features["labels"].feature.names
    texts, labels = [], []
    ekman_lookup = {}  # populate from goemotions ekman_mapping.json if available
    # Minimal built-in fallback mapping (fine-grained -> ekman group name)
    fallback = {
        "admiration": "joy", "amusement": "joy", "approval": "joy",
        "caring": "joy", "desire": "joy", "excitement": "joy",
        "gratitude": "joy", "joy": "joy", "love": "joy",
        "optimism": "joy", "pride": "joy", "relief": "joy",
        "anger": "anger", "annoyance": "anger", "disapproval": "anger",
        "disgust": "disgust",
        "embarrassment": "sadness", "grief": "sadness", "remorse": "sadness",
        "sadness": "sadness", "disappointment": "sadness",
        "fear": "fear", "nervousness": "fear",
        "confusion": "surprise", "curiosity": "surprise",
        "realization": "surprise", "surprise": "surprise",
        "neutral": "neutral",
    }
    for row in ds:
        our_label = None
        for lid in row["labels"]:
            fine = id2name[lid]
            ekman = fallback.get(fine)
            if ekman and ekman in GOEMOTIONS_EKMAN_TO_OURS:
                our_label = GOEMOTIONS_EKMAN_TO_OURS[ekman]
                break
        if our_label:
            texts.append(row["text"])
            labels.append(LABEL2IDX[our_label])
    return pd.DataFrame({"text": texts, "label": labels})


def load_dair_ai_emotion():
    ds = load_dataset("dair-ai/emotion")["train"]
    # dair-ai/emotion labels: sadness, joy, love, anger, fear, surprise
    name_map = {"sadness": "sad", "joy": "happy", "anger": "angry",
                "fear": "fear", "surprise": "surprise"}  # "love" dropped, no match
    names = ds.features["label"].names
    texts, labels = [], []
    for row in ds:
        name = names[row["label"]]
        if name in name_map:
            texts.append(row["text"])
            labels.append(LABEL2IDX[name_map[name]])
    return pd.DataFrame({"text": texts, "label": labels})


def augment_disgust(df: pd.DataFrame, target_count: int = 10000):
    """Thesis 3.4.2: T5-base paraphrasing of disgust sentences until count
    exceeds 10,000. Requires a T5 paraphrase model — swap in your own if
    you don't have Carmine's fine-tuned checkpoint; a generic T5 paraphraser
    (e.g. 'Vamsi/T5_Paraphrase_Paws') is a reasonable substitute, but note
    in the paper that this is a substitute, not the original fine-tuned model."""
    disgust_idx = LABEL2IDX["disgust"]
    disgust_rows = df[df.label == disgust_idx]
    print(f"Disgust instances before augmentation: {len(disgust_rows)}")
    if len(disgust_rows) >= target_count:
        return df
    try:
        from transformers import T5ForConditionalGeneration, T5Tokenizer
        t5_tok = T5Tokenizer.from_pretrained("Vamsi/T5_Paraphrase_Paws")
        t5_model = T5ForConditionalGeneration.from_pretrained("Vamsi/T5_Paraphrase_Paws")
    except Exception as e:
        print(f"WARNING: could not load T5 paraphraser ({e}); skipping disgust "
              f"augmentation. Expect lower disgust F1 than the thesis's 0.94 — "
              f"report this honestly rather than papering over it.")
        return df

    new_rows = []
    texts = disgust_rows["text"].tolist()
    i = 0
    while len(disgust_rows) + len(new_rows) < target_count:
        text = texts[i % len(texts)]
        enc = t5_tok(f"paraphrase: {text}", return_tensors="pt", truncation=True, max_length=BERT_MAX_LENGTH)
        out = t5_model.generate(**enc, num_return_sequences=1, num_beams=4, max_length=BERT_MAX_LENGTH)
        para = t5_tok.decode(out[0], skip_special_tokens=True)
        if para and para != text:
            new_rows.append({"text": para, "label": disgust_idx})
        i += 1
    aug_df = pd.DataFrame(new_rows)
    print(f"Added {len(aug_df)} paraphrased disgust instances")
    return pd.concat([df, aug_df], ignore_index=True).drop_duplicates(subset="text")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="text_model/")
    ap.add_argument("--skip_augmentation", action="store_true",
                     help="Skip T5 disgust augmentation (faster, but expect "
                          "lower disgust F1 than the thesis's reported 0.94)")
    args = ap.parse_args()

    print("Loading GoEmotions...")
    goemo = load_goemotions()
    print("Loading dair-ai/emotion...")
    dair = load_dair_ai_emotion()
    df = pd.concat([goemo, dair], ignore_index=True).drop_duplicates(subset="text")
    print(f"Combined dataset: {len(df)} rows")
    print(df.label.map(lambda i: EMOTIONS[i]).value_counts())

    train_df, test_df = train_test_split(
        df, test_size=0.15, random_state=RANDOM_SEED, stratify=df.label
    )

    # Augment AFTER the split, and ONLY the training portion -- augmenting
    # before splitting (as an earlier version of this script did) lets
    # paraphrased near-duplicates of the same original disgust sentence land
    # on both sides of the split, inflating disgust F1 (and overall test
    # accuracy) without the model actually generalizing better. This exact
    # bug was confirmed to cause a 28-point accuracy inflation (62% -> 90%)
    # in the equivalent audio pipeline when tested deliberately -- see
    # test_leaky_hypothesis.py.
    if not args.skip_augmentation:
        train_df = augment_disgust(train_df)

    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    def tokenize(batch):
        return tokenizer(batch["text"], padding="max_length", truncation=True, max_length=BERT_MAX_LENGTH)

    train_ds = Dataset.from_pandas(train_df.reset_index(drop=True)).map(tokenize, batched=True)
    test_ds = Dataset.from_pandas(test_df.reset_index(drop=True)).map(tokenize, batched=True)
    train_ds = train_ds.rename_column("label", "labels")
    test_ds = test_ds.rename_column("label", "labels")
    cols = ["input_ids", "attention_mask", "labels"]
    train_ds.set_format(type="torch", columns=cols)
    test_ds.set_format(type="torch", columns=cols)

    model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=NUM_CLASSES)

    training_args = TrainingArguments(
        output_dir="./bert_training_output",
        eval_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=3,
        weight_decay=0.01,
        logging_steps=1,
    )

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        return {"accuracy": accuracy_score(labels, preds)}

    trainer = Trainer(
        model=model, args=training_args,
        train_dataset=train_ds, eval_dataset=test_ds,
        compute_metrics=compute_metrics,
    )
    trainer.train()

    preds = trainer.predict(test_ds)
    y_pred = preds.predictions.argmax(1)
    y_true = preds.label_ids
    print("Final test accuracy:", accuracy_score(y_true, y_pred))
    print(classification_report(y_true, y_pred, target_names=EMOTIONS))

    model.save_pretrained(args.out)
    tokenizer.save_pretrained(args.out)
    print(f"Saved to {args.out}")


if __name__ == "__main__":
    main()
