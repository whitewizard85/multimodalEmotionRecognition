"""
Consolidates the four visual datasets (RAF-DB, AffectNet, ExpW, FER2013)
into the train/val/<emotion>/ layout train_visual.py expects, applying the
thesis 3.5 preprocessing steps: contempt-class removal, face-cropping for
ExpW (using its bounding-box annotations), and downsampling of
overrepresented classes (>10,000 instances) with a 90/10 train/val split.

Each source dataset has its own label convention -- do NOT assume the
folder/label names line up with our EMOTIONS list without an explicit
mapping; several of the label mappings below were confirmed from the
user's actual downloaded folder structure and label files (RAF-DB numeric
codes, AffectNet mixed-case names, ExpW's own 0-6 ordering -- which is
DIFFERENT from our common.py EMOTIONS order, so don't reuse that index
mapping here).
"""
import argparse
import os
import shutil
import random
from collections import defaultdict

from PIL import Image

from common import EMOTIONS, RANDOM_SEED

# --- RAF-DB: numbered folders 1-7, official RAF-DB single-label convention ---
RAFDB_MAP = {
    "1": "surprise", "2": "fear", "3": "disgust", "4": "happy",
    "5": "sad", "6": "angry", "7": "neutral",
}

# --- AffectNet (Kaggle mirror): mixed-case folder names, Contempt excluded ---
AFFECTNET_MAP = {
    "happy": "happy", "anger": "angry", "fear": "fear",
    "surprise": "surprise", "neutral": "neutral", "disgust": "disgust",
    "sad": "sad",
    # "contempt" deliberately excluded (thesis 3.5: "Removal of the contempt
    # class due to low representation")
}

# --- ExpW: label.lst codes, per its own readme.txt -- NOT the same order
# as our EMOTIONS list, do not conflate the two ---
EXPW_MAP = {
    "0": "angry", "1": "disgust", "2": "fear", "3": "happy",
    "4": "sad", "5": "surprise", "6": "neutral",
}

MAX_PER_CLASS = 10000  # thesis 3.5: "Downsampling overrepresented classes (>10,000 instances)"


def stage_rafdb(src_root, staging_dir):
    for split in ["train", "test"]:
        split_dir = os.path.join(src_root, split)
        if not os.path.isdir(split_dir):
            continue
        for code, emotion in RAFDB_MAP.items():
            src = os.path.join(split_dir, code)
            if not os.path.isdir(src):
                continue
            dst = os.path.join(staging_dir, emotion)
            os.makedirs(dst, exist_ok=True)
            for fname in os.listdir(src):
                shutil.copy(os.path.join(src, fname), os.path.join(dst, f"rafdb_{split}_{fname}"))
    print("RAF-DB staged.")


def stage_affectnet(src_root, staging_dir):
    for split_name in ["Train", "Test", "train", "test"]:
        split_dir = os.path.join(src_root, split_name)
        if not os.path.isdir(split_dir):
            continue
        for folder in os.listdir(split_dir):
            key = folder.lower()
            if key not in AFFECTNET_MAP:
                if key == "contempt":
                    print(f"Skipping AffectNet '{folder}' (contempt class excluded per thesis 3.5)")
                continue
            emotion = AFFECTNET_MAP[key]
            src = os.path.join(split_dir, folder)
            dst = os.path.join(staging_dir, emotion)
            os.makedirs(dst, exist_ok=True)
            for fname in os.listdir(src):
                shutil.copy(os.path.join(src, fname), os.path.join(dst, f"affectnet_{split_name}_{fname}"))
    print("AffectNet staged.")


def stage_expw(src_root, staging_dir, label_lst_path):
    """ExpW ships uncropped -- crop each face using the box coordinates in
    label.lst (top, left, right, bottom), per thesis's 'ExpW-F' description."""
    images_dir = None
    for root, dirs, files in os.walk(src_root):
        if any(f.lower().endswith(".jpg") for f in files):
            images_dir = root
            break
    if images_dir is None:
        print(f"WARNING: no images found under {src_root} -- did the 7z extraction finish?")
        return

    counts = defaultdict(int)
    with open(label_lst_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 8:
                continue
            fname, face_id, top, left, right, bottom, conf, label_code = parts[:8]
            emotion = EXPW_MAP.get(label_code)
            if emotion is None:
                continue
            src_path = os.path.join(images_dir, fname)
            if not os.path.exists(src_path):
                continue
            try:
                img = Image.open(src_path)
                box = (int(left), int(top), int(right), int(bottom))
                cropped = img.crop(box)
            except Exception as e:
                continue
            dst = os.path.join(staging_dir, emotion)
            os.makedirs(dst, exist_ok=True)
            out_name = f"expw_{face_id}_{fname}"
            cropped.save(os.path.join(dst, out_name))
            counts[emotion] += 1
    print(f"ExpW staged: {dict(counts)}")


def stage_fer2013(src_root, staging_dir):
    for split in ["train", "test"]:
        split_dir = os.path.join(src_root, split)
        if not os.path.isdir(split_dir):
            continue
        for folder in os.listdir(split_dir):
            emotion = folder.lower()
            if emotion not in EMOTIONS:
                print(f"Skipping unrecognized FER2013 folder '{folder}'")
                continue
            src = os.path.join(split_dir, folder)
            dst = os.path.join(staging_dir, emotion)
            os.makedirs(dst, exist_ok=True)
            for fname in os.listdir(src):
                shutil.copy(os.path.join(src, fname), os.path.join(dst, f"fer2013_{split}_{fname}"))
    print("FER2013 staged.")


def finalize_split(staging_dir, out_dir, max_per_class=MAX_PER_CLASS, val_frac=0.10, seed=RANDOM_SEED):
    random.seed(seed)
    for emotion in EMOTIONS:
        src = os.path.join(staging_dir, emotion)
        if not os.path.isdir(src):
            print(f"WARNING: no staged images at all for '{emotion}'")
            continue
        files = os.listdir(src)
        random.shuffle(files)
        if len(files) > max_per_class:
            print(f"Downsampling '{emotion}': {len(files)} -> {max_per_class}")
            files = files[:max_per_class]
        n_val = int(len(files) * val_frac)
        val_files, train_files = files[:n_val], files[n_val:]

        for split, split_files in [("train", train_files), ("val", val_files)]:
            dst_dir = os.path.join(out_dir, split, emotion)
            os.makedirs(dst_dir, exist_ok=True)
            for fname in split_files:
                shutil.copy(os.path.join(src, fname), os.path.join(dst_dir, fname))
        print(f"{emotion}: {len(train_files)} train, {len(val_files)} val")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rafdb_dir", help="e.g. archive/DATASET")
    ap.add_argument("--affectnet_dir", help="e.g. 'archive (1)/archive (3)'")
    ap.add_argument("--expw_dir", help="e.g. expw/origin (extracted images folder)")
    ap.add_argument("--expw_label_lst", help="e.g. expw/label.lst")
    ap.add_argument("--fer2013_dir", help="e.g. fer2013_unzipped")
    ap.add_argument("--staging_dir", default="visual_staging")
    ap.add_argument("--out_dir", default="visual_dataset")
    args = ap.parse_args()

    os.makedirs(args.staging_dir, exist_ok=True)

    if args.rafdb_dir:
        stage_rafdb(args.rafdb_dir, args.staging_dir)
    if args.affectnet_dir:
        stage_affectnet(args.affectnet_dir, args.staging_dir)
    if args.expw_dir and args.expw_label_lst:
        stage_expw(args.expw_dir, args.staging_dir, args.expw_label_lst)
    if args.fer2013_dir:
        stage_fer2013(args.fer2013_dir, args.staging_dir)

    print("\nFinalizing train/val split with downsampling...")
    finalize_split(args.staging_dir, args.out_dir)
    print(f"\nDone. Point train_visual.py --data_dir at: {args.out_dir}")


if __name__ == "__main__":
    main()
