"""
Runs the trained ResNet50 visual model on visual_dataset/val/ and saves
predicted probabilities + true labels to visual_probs.npz.

The val/ folder was already a fixed, honest held-out split created by
build_visual_dataset.py before any training happened -- no re-splitting
needed here, just inference.

Run in venv_visual (the TensorFlow-only environment).
"""
import argparse

import numpy as np
import tensorflow as tf

from common import EMOTIONS


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", default="visual_resnet50_model.h5")
    ap.add_argument("--data_dir", default="visual_dataset")
    ap.add_argument("--out", default="visual_probs.npz")
    ap.add_argument("--img_size", type=int, default=224,
                     help="Must match the size the model was trained with "
                          "-- 224 for the ResNet50 model, 96 for the "
                          "original custom CNN")
    args = ap.parse_args()

    model = tf.keras.models.load_model(args.model_path)

    val_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        preprocessing_function=tf.keras.applications.resnet50.preprocess_input,
    )
    val_gen = val_datagen.flow_from_directory(
        f"{args.data_dir}/val", target_size=(args.img_size, args.img_size),
        color_mode="rgb", classes=EMOTIONS, class_mode="categorical",
        batch_size=32, shuffle=False,
    )

    probs = model.predict(val_gen)
    labels = val_gen.classes
    print(f"Saved probabilities shape: {probs.shape}, labels shape: {labels.shape}")

    np.savez(args.out, probs=probs, labels=labels)
    print(f"Saved to {args.out}")


if __name__ == "__main__":
    main()
