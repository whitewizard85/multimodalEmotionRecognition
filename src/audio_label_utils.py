"""
Audio filename -> emotion label parsing, extracted into its own module with
ZERO heavy dependencies (only os, stdlib) -- specifically so this can be
imported by both train_audio.py (which needs TensorFlow) and predict_audio.py
(which runs in the torch venv and must NOT import TensorFlow at all). The
original design imported this function directly from train_audio.py, which
unconditionally imports tensorflow at module load time -- silently pulling
TensorFlow into the torch-only prediction environment and triggering an
unrelated NumPy/TensorFlow ABI version crash. Keep this module free of any
ML framework imports going forward.
"""
import os

RAVDESS_CODE = {  # RAVDESS 3rd filename field, e.g. 03-01-06-... -> emotion
    "01": "neutral", "02": "neutral", "03": "happy", "04": "sad",
    "05": "angry", "06": "fear", "07": "disgust", "08": "surprise",
}

CREMA_D_CODE = {  # CREMA-D 3rd underscore field, e.g. 1001_DFA_ANG_XX.wav
    "ANG": "angry", "DIS": "disgust", "FEA": "fear", "HAP": "happy",
    "NEU": "neutral", "SAD": "sad",  # CREMA-D has no 'surprise' class
}

SAVEE_CODE = {  # leading letter code before the 2-digit number
    "a": "angry", "d": "disgust", "f": "fear", "h": "happy",
    "n": "neutral", "sa": "sad", "su": "surprise",
}

TESS_CODE = {  # trailing token after the last underscore, before .wav
    "angry": "angry", "disgust": "disgust", "fear": "fear",
    "happy": "happy", "neutral": "neutral", "sad": "sad",
    "ps": "surprise",  # TESS's own name for "pleasant surprise"
}


def parse_label_from_filename(path: str, dataset: str = None):
    """Per-dataset filename parser. Pass --dataset_hint per source folder if
    you keep the four corpora in clearly separate directories (recommended)
    -- this is far more reliable than sniffing the filename pattern, since
    some conventions can collide (e.g. CREMA-D's 'SAD' vs. RAVDESS numeric
    codes). If dataset is None, falls back to pattern-sniffing below."""
    fname = os.path.basename(path)
    stem = fname.rsplit(".", 1)[0]

    if dataset == "ravdess" or (dataset is None and stem.count("-") >= 6):
        parts = stem.split("-")
        if len(parts) >= 3 and parts[2] in RAVDESS_CODE:
            return RAVDESS_CODE[parts[2]]

    if dataset == "crema_d" or (dataset is None and stem.count("_") == 3):
        parts = stem.split("_")
        if len(parts) >= 3 and parts[2].upper() in CREMA_D_CODE:
            return CREMA_D_CODE[parts[2].upper()]

    if dataset == "tess" or (dataset is None and stem.count("_") == 2):
        tail = stem.rsplit("_", 1)[-1].lower()
        if tail in TESS_CODE:
            return TESS_CODE[tail]

    if dataset == "savee" or dataset is None:
        # e.g. "n17", "a06", "sa03", "su12" -- leading letters, then digits
        letters = "".join(ch for ch in stem if ch.isalpha()).lower()
        if letters in SAVEE_CODE:
            return SAVEE_CODE[letters]

    return None
