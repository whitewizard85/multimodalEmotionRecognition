"""
Shared constants for the multimodal emotion recognition reproduction.

Label ordering is fixed to match the visual-model encoding given explicitly
in the thesis (Section 3.5, "Label encoding of emotions"):

    {'angry': 0, 'disgust': 1, 'fear': 2, 'happy': 3,
     'neutral': 4, 'sad': 5, 'surprise': 6}

All three modalities are mapped onto this same 7-class label space so their
probability vectors are directly comparable/fusable via Eq. 1 of the paper.
"""

EMOTIONS = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]
LABEL2IDX = {e: i for i, e in enumerate(EMOTIONS)}
IDX2LABEL = {i: e for e, i in LABEL2IDX.items()}
NUM_CLASSES = len(EMOTIONS)

RANDOM_SEED = 42

# GoEmotions -> Ekman-style 7-class mapping (google-research/goemotions
# ekman_mapping.json convention). "joy" -> "happy" to match our label set.
# CONFIRM against Carmine's actual mapping if it can be recovered; this is
# the standard/likely choice but is not explicitly documented in the thesis.
GOEMOTIONS_EKMAN_TO_OURS = {
    "anger": "angry",
    "disgust": "disgust",
    "fear": "fear",
    "joy": "happy",
    "neutral": "neutral",
    "sadness": "sad",
    "surprise": "surprise",
}

AUDIO_FEATURE_LENGTH = 1000  # thesis 3.3: truncated/padded feature length --
                              # kept for reference; the flat-concatenated-
                              # summary-statistics version of this plateaued
                              # at ~62-64% test accuracy (vs. thesis's 97%),
                              # so train_audio.py now uses a genuine time-
                              # series representation instead (see below).
AUDIO_TIME_STEPS = 300   # ~3 seconds of audio at 10ms hop -- covers most
                          # clips in RAVDESS/CREMA-D/TESS/SAVEE without
                          # excessive padding; adjust if your clips run
                          # longer on average
AUDIO_N_MELS = 128        # standard mel-band count; used as the Conv1D
                          # "channel" dimension so the network sees genuine
                          # local time-frequency structure, not a global
                          # summary stripped of temporal detail
BERT_MAX_LENGTH = 128         # thesis 3.4.2
VISUAL_IMG_SIZE = 96           # thesis 3.5
