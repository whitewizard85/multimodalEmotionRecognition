"""
Facial emotion classification via transfer learning from an ImageNet-
pretrained ResNet50, instead of a custom CNN trained from scratch
(train_visual.py, which plateaued at ~52% validation accuracy with almost
no train/val gap -- a sign of a genuine capacity/data-quality ceiling for
a from-scratch model on this noisy, heterogeneous 4-dataset combination,
not an overfitting problem).

Rationale: this also reconciles an inconsistency flagged early on -- the
original manuscript's Table 5 and Conclusion both label the visual branch
"ResNet-50", while the thesis's prose (Section 3.5) describes a small
custom CNN. Training a small CNN from scratch on combined RAF-DB+AffectNet+
ExpW+FER2013 data is known to underperform transfer learning badly in FER
literature (this is exactly why ResNet/EfficientNet, both cited in the
manuscript's own Related Work, dominate this space) -- so this rebuild is
both better-grounded in the literature AND more consistent with what
Table 5 actually claims to have used.

Two-phase fine-tuning (standard practice, not novel):
  Phase 1: ResNet50 backbone frozen, train only the new classification head.
  Phase 2: unfreeze the top N backbone layers, continue training at a much
           lower learning rate, so the pretrained low/mid-level features
           aren't destroyed by early large gradient updates.

Requires the TensorFlow-only environment (venv_visual), same as train_visual.py.
"""
import argparse

import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
from sklearn.metrics import classification_report, accuracy_score
import numpy as np

from common import EMOTIONS, NUM_CLASSES, RANDOM_SEED

RESNET_IMG_SIZE = 224  # standard ImageNet-pretrained input size


def build_model(fine_tune_layers: int = 0):
    """fine_tune_layers=0 means the backbone is fully frozen (phase 1).
    Pass a positive number to unfreeze that many top layers (phase 2)."""
    base = tf.keras.applications.ResNet50(
        weights="imagenet", include_top=False,
        input_shape=(RESNET_IMG_SIZE, RESNET_IMG_SIZE, 3),
    )
    if fine_tune_layers == 0:
        base.trainable = False
    else:
        base.trainable = True
        for layer in base.layers[:-fine_tune_layers]:
            layer.trainable = False

    x = layers.GlobalAveragePooling2D()(base.output)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dropout(0.4)(x)
    out = layers.Dense(NUM_CLASSES, activation="softmax")(x)
    model = models.Model(base.input, out)
    return model, base


def make_generators(data_dir, batch_size=32):
    train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        preprocessing_function=tf.keras.applications.resnet50.preprocess_input,
        rotation_range=15, width_shift_range=0.15, height_shift_range=0.15,
        shear_range=0.15, zoom_range=0.15, horizontal_flip=True,
    )
    val_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        preprocessing_function=tf.keras.applications.resnet50.preprocess_input,
    )
    train_gen = train_datagen.flow_from_directory(
        f"{data_dir}/train", target_size=(RESNET_IMG_SIZE, RESNET_IMG_SIZE),
        color_mode="rgb", classes=EMOTIONS, class_mode="categorical",
        batch_size=batch_size, seed=RANDOM_SEED,
    )
    val_gen = val_datagen.flow_from_directory(
        f"{data_dir}/val", target_size=(RESNET_IMG_SIZE, RESNET_IMG_SIZE),
        color_mode="rgb", classes=EMOTIONS, class_mode="categorical",
        batch_size=batch_size, shuffle=False,
    )
    return train_gen, val_gen


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--out", default="visual_resnet50_model.h5")
    ap.add_argument("--phase1_epochs", type=int, default=20)
    ap.add_argument("--phase2_epochs", type=int, default=20)
    ap.add_argument("--fine_tune_layers", type=int, default=30,
                     help="Number of top ResNet50 layers to unfreeze in "
                          "phase 2. Set to 0 to skip phase 2 entirely and "
                          "evaluate with the frozen-backbone model only.")
    args = ap.parse_args()

    tf.random.set_seed(RANDOM_SEED)

    train_gen, val_gen = make_generators(args.data_dir)

    print("=" * 70)
    print("PHASE 1: training classification head, backbone frozen")
    print("=" * 70)
    model, base = build_model(fine_tune_layers=0)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="categorical_crossentropy", metrics=["accuracy"],
    )
    model.fit(
        train_gen, validation_data=val_gen, epochs=args.phase1_epochs,
        callbacks=[
            callbacks.EarlyStopping(patience=6, monitor="val_accuracy", restore_best_weights=True),
            callbacks.ReduceLROnPlateau(factor=0.5, patience=3, monitor="val_accuracy"),
        ],
    )

    if args.fine_tune_layers > 0:
        print("=" * 70)
        print(f"PHASE 2: fine-tuning top {args.fine_tune_layers} backbone "
              f"layers at a low learning rate")
        print("=" * 70)
        for layer in base.layers[:-args.fine_tune_layers]:
            layer.trainable = False
        for layer in base.layers[-args.fine_tune_layers:]:
            layer.trainable = True
        # Much lower LR than phase 1 -- large updates here would destroy the
        # pretrained features we're trying to preserve and only lightly adapt
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
            loss="categorical_crossentropy", metrics=["accuracy"],
        )
        model.fit(
            train_gen, validation_data=val_gen, epochs=args.phase2_epochs,
            callbacks=[
                callbacks.EarlyStopping(patience=6, monitor="val_accuracy", restore_best_weights=True),
                callbacks.ReduceLROnPlateau(factor=0.5, patience=3, monitor="val_accuracy"),
            ],
        )

    val_gen.reset()
    y_pred = model.predict(val_gen).argmax(1)
    y_true = val_gen.classes
    print("\nFinal validation accuracy:", accuracy_score(y_true, y_pred))
    print(classification_report(y_true, y_pred, target_names=EMOTIONS))

    model.save(args.out)
    print(f"Saved to {args.out}")


if __name__ == "__main__":
    main()
