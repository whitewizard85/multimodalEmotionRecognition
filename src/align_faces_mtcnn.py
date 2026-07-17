"""
Re-aligns every image in visual_dataset/{train,val}/<emotion>/ using
facenet-pytorch's own MTCNN face detector/aligner, producing crops that
match the alignment convention InceptionResnetV1's VGGFace2 pretrained
weights actually expect. Run this ONCE, then point train_visual_facenet.py
at the output directory instead of the original visual_dataset/.

Rationale: our existing crops come from four different sources with four
different cropping conventions (RAF-DB's own alignment, AffectNet's crops,
our own box-crop from ExpW's raw bounding boxes, FER2013's fixed 48x48
format) -- none of which necessarily match MTCNN-aligned faces. Face
embedding models are known to be sensitive to this mismatch; this was the
leading hypothesis for why the VGGFace2-backbone attempt plateaued at ~30%,
well below even the from-scratch custom CNN's 52%.

If MTCNN fails to detect a face in a given image (a real risk, since many
of our source images are already tightly cropped and may lack the
surrounding context MTCNN's detection cascade needs), the original image is
resized and kept as a fallback rather than dropped -- with the fallback
rate reported, since a high fallback rate would itself be informative.
"""
import argparse
import os

import torch
import numpy as np
from PIL import Image
from facenet_pytorch import MTCNN

from common import EMOTIONS

IMG_SIZE = 160


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_dir", default="visual_dataset")
    ap.add_argument("--output_dir", default="visual_dataset_aligned")
    ap.add_argument("--margin", type=int, default=20,
                     help="Pixels of context added around the detected face "
                          "box before cropping -- helps since our source "
                          "images are often already tightly cropped")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    mtcnn = MTCNN(image_size=IMG_SIZE, margin=args.margin, post_process=False,
                  select_largest=True, device=device)

    stats = {"detected": 0, "fallback": 0, "total": 0}

    for split in ["train", "val"]:
        for emotion in EMOTIONS:
            src_dir = os.path.join(args.input_dir, split, emotion)
            if not os.path.isdir(src_dir):
                continue
            dst_dir = os.path.join(args.output_dir, split, emotion)
            os.makedirs(dst_dir, exist_ok=True)

            files = os.listdir(src_dir)
            for i, fname in enumerate(files):
                stats["total"] += 1
                src_path = os.path.join(src_dir, fname)
                dst_path = os.path.join(dst_dir, fname)
                try:
                    img = Image.open(src_path).convert("RGB")
                    face_tensor = mtcnn(img)
                    if face_tensor is not None:
                        # face_tensor is (3, H, W), float, raw pixel range
                        face_np = face_tensor.permute(1, 2, 0).byte().numpy()
                        Image.fromarray(face_np).save(dst_path)
                        stats["detected"] += 1
                    else:
                        img.resize((IMG_SIZE, IMG_SIZE)).save(dst_path)
                        stats["fallback"] += 1
                except Exception as e:
                    # Corrupt/unreadable image -- fall back rather than crash
                    # the whole preprocessing run over one bad file
                    try:
                        Image.open(src_path).convert("RGB").resize((IMG_SIZE, IMG_SIZE)).save(dst_path)
                        stats["fallback"] += 1
                    except Exception:
                        stats["total"] -= 1  # truly unreadable, skip entirely

                if (i + 1) % 1000 == 0:
                    print(f"  {split}/{emotion}: {i + 1}/{len(files)} processed")

            print(f"{split}/{emotion}: done ({len(files)} files)")

    print("\n" + "=" * 70)
    print(f"Total processed: {stats['total']}")
    print(f"Face detected by MTCNN: {stats['detected']} "
          f"({100*stats['detected']/max(stats['total'],1):.1f}%)")
    print(f"Fallback (no face detected, used plain resize): {stats['fallback']} "
          f"({100*stats['fallback']/max(stats['total'],1):.1f}%)")
    print("=" * 70)
    if stats["fallback"] / max(stats["total"], 1) > 0.3:
        print("WARNING: fallback rate is high (>30%). MTCNN struggled to "
              "detect faces in a large fraction of images -- likely because "
              "many source crops are already very tight. Consider increasing "
              "--margin, or this alignment approach may not help as much as "
              "hoped.")


if __name__ == "__main__":
    main()
