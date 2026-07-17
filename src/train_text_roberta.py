"""
Fine-tunes roberta-base instead of bert-base-uncased (train_text.py, which
reached ~74% test accuracy), since RoBERTa consistently outperforms BERT on
classification benchmarks in the literature (already cited in the
manuscript's own Related Work -- Liu et al. 2019). This is a smaller,
incremental change compared to the audio/visual rebuilds -- BERT was
already a strong pretrained encoder, so expect a modest gain here, not
another dramatic jump.

Reuses the already leak-fixed data loading (train/test split happens BEFORE
disgust augmentation) from train_text.py directly via import, rather than
duplicating that logic.
"""
import argparse

import numpy as np
from datasets import Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from transformers import (
    RobertaTokenizer, RobertaForSequenceClassification,
    TrainingArguments, Trainer,
)

from common import EMOTIONS, NUM_CLASSES, BERT_MAX_LENGTH, RANDOM_SEED
from train_text import load_goemotions, load_dair_ai_emotion, augment_disgust
import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="text_roberta_model/")
    ap.add_argument("--skip_augmentation", action="store_true")
    ap.add_argument("--num_epochs", type=int, default=3,
                     help="Matches the thesis's original recipe (3) by "
                          "default, for a clean isolated comparison against "
                          "the BERT baseline -- increase if you want to also "
                          "test whether more training helps, as a separate "
                          "follow-up experiment")
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

    if not args.skip_augmentation:
        train_df = augment_disgust(train_df)

    tokenizer = RobertaTokenizer.from_pretrained("roberta-base")

    def tokenize(batch):
        return tokenizer(batch["text"], padding="max_length", truncation=True, max_length=BERT_MAX_LENGTH)

    train_ds = Dataset.from_pandas(train_df.reset_index(drop=True)).map(tokenize, batched=True)
    test_ds = Dataset.from_pandas(test_df.reset_index(drop=True)).map(tokenize, batched=True)
    train_ds = train_ds.rename_column("label", "labels")
    test_ds = test_ds.rename_column("label", "labels")
    cols = ["input_ids", "attention_mask", "labels"]
    train_ds.set_format(type="torch", columns=cols)
    test_ds.set_format(type="torch", columns=cols)

    model = RobertaForSequenceClassification.from_pretrained(
        "roberta-base", num_labels=NUM_CLASSES, use_safetensors=True
    )

    training_args = TrainingArguments(
        output_dir="./roberta_training_output",
        eval_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=args.num_epochs,
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
