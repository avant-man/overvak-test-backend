"""
Fusion module: combines video and audio classification scores via weighted average.

Design notes:
- Video scores (X-CLIP relative-to-uniform) sit in [-1, 1] and are clipped to
  [0, 1] before fusion.  X-CLIP was trained on physical actions; when it does
  not recognise a class it produces a *negative* score.  Allowing that negative
  score into the weighted sum would actively cancel out a strong audio signal —
  a confused video model should abstain, not vote against.
- Crying and Laughing have no visually distinctive motion pattern that X-CLIP
  can reliably detect, so they are classified from audio only (video_weight=0).
- Visually clear physical actions (Climbing, Jumping, Running, …) remain
  video-dominant because X-CLIP is reliable there and audio is less distinctive.

Per-class thresholds replace a single global threshold.  Call
threshold_search() with fused scores and ground-truth labels from the labeled
evaluation videos to find optimal per-class thresholds.
"""

import numpy as np

CLASSES = ["Climbing", "Crying", "Falling", "Hitting", "Jumping", "Laughing", "Pushing", "Running"]

# (video_weight, audio_weight) — must sum to 1.0 per class.
# Crying and Laughing use video_weight=0: X-CLIP cannot reliably detect these
# from visual motion alone, and a negative video score was actively cancelling
# out valid audio evidence.  All other video scores are clipped to [0, 1] in
# fuse() so a confused video model abstains rather than voting against.
WEIGHTS: dict[str, tuple[float, float]] = {
    "Climbing": (0.7, 0.3),  # reduced from 0.8 — give more say to audio scraping/gripping cues
    "Crying":   (0.0, 1.0),
    "Falling":  (0.6, 0.4),
    "Hitting":  (0.5, 0.5),  # balanced: audio can be muffled; video arm-swing is equally reliable
    "Jumping":  (0.8, 0.2),
    "Laughing": (0.0, 1.0),
    "Pushing":  (0.6, 0.4),  # reduced from 0.8 — X-CLIP was producing FPs by over-scoring pushing
    "Running":  (0.7, 0.3),
}

# Per-class detection thresholds on the fused score.
# Video scores are relative-to-uniform in [-1, 1]; audio scores are cosine
# similarities in [0, ~0.4].  The weighted average sits in a similar range.
# A threshold of 0.20 means the fused signal must be meaningfully above the
# baseline — tune with threshold_search() once scores are collected.
THRESHOLDS: dict[str, float] = {
    "Climbing": 0.22,
    "Crying":   0.22,  # pure-audio: tuned to CLAP cosine-similarity scale [0, ~0.4]
    "Falling":  0.22,
    "Hitting":  0.22,
    "Jumping":  0.25,
    "Laughing": 0.22,  # pure-audio: tuned to CLAP cosine-similarity scale [0, ~0.4]
    "Pushing":  0.38,  # raised from 0.25 — was producing FPs at 57%; needs stronger signal
    "Running":  0.22,
}

# Pairs of classes that cannot co-occur in the same clip.
# ("Crying", "Laughing") has been intentionally removed: a video clip can contain
# both at different moments (e.g. one child crying while another laughs), and
# keeping the pair caused valid Crying detections to be dropped whenever Laughing
# also scored above threshold.
MUTUAL_EXCLUSIONS: list[tuple[str, str]] = [
    ("Hitting", "Pushing"),
]


def fuse(
    video_scores: dict[str, float],
    audio_scores: dict[str, float],
) -> tuple[dict[str, float], dict[str, int]]:
    """
    Combine video and audio scores into fused scores and binary predictions.

    Video scores are clipped to [0, 1] before weighting: X-CLIP returns
    relative-to-uniform scores in [-1, 1], and a negative value means the model
    is uncertain, not that the action is absent.  Allowing negative video scores
    into the sum would cancel valid audio evidence.

    Args:
        video_scores: Per-class relative-to-uniform scores from the video classifier.
        audio_scores: Per-class cosine similarities from the audio classifier.

    Returns:
        A tuple of (fused_scores, predictions) where fused_scores maps each class
        to its weighted-average score and predictions maps each class to 0 or 1.
    """
    fused_scores: dict[str, float] = {}
    predictions: dict[str, int] = {}

    for cls in CLASSES:
        v_w, a_w = WEIGHTS[cls]
        v_score = max(video_scores.get(cls, 0.0), 0.0)  # clip: uncertain ≠ absent
        score = v_w * v_score + a_w * audio_scores.get(cls, 0.0)
        fused_scores[cls] = round(score, 4)
        predictions[cls] = int(score >= THRESHOLDS[cls])

    # Suppress the lower-scoring class for mutually exclusive pairs
    for cls_a, cls_b in MUTUAL_EXCLUSIONS:
        if predictions[cls_a] and predictions[cls_b]:
            loser = cls_a if fused_scores[cls_a] < fused_scores[cls_b] else cls_b
            predictions[loser] = 0

    return fused_scores, predictions


def threshold_search(
    scores_by_video: dict[str, dict[str, float]],
    ground_truth: dict[str, dict[str, int]],
    n_steps: int = 200,
    beta: float = 1.0,
) -> dict[str, float]:
    """
    Find per-class thresholds that maximise F-beta on the labeled evaluation set.

    Iterates over a fine grid of candidate thresholds for each class independently
    and picks the one that gives the best F-beta score across all labeled videos.

    Args:
        scores_by_video: {video_id: {class: fused_score}} for all labeled videos.
        ground_truth: {video_id: {class: 0_or_1}} matching scores_by_video keys.
        n_steps: Number of candidate thresholds to try per class (default 200).
        beta: F-beta parameter.  beta=1 → F1, beta=2 → emphasise recall,
              beta=0.5 → emphasise precision.

    Returns:
        Dict mapping each class to its optimal threshold.
    """
    video_ids = list(scores_by_video.keys())
    best_thresholds: dict[str, float] = {}

    for cls in CLASSES:
        cls_scores = np.array([scores_by_video[vid].get(cls, 0.0) for vid in video_ids])
        cls_labels = np.array([ground_truth[vid].get(cls, 0) for vid in video_ids])

        lo = float(cls_scores.min())
        hi = float(cls_scores.max())
        candidates = np.linspace(lo, hi, n_steps)

        best_t = 0.5
        best_score = -1.0

        for t in candidates:
            preds = (cls_scores >= t).astype(int)
            tp = int(np.sum((preds == 1) & (cls_labels == 1)))
            fp = int(np.sum((preds == 1) & (cls_labels == 0)))
            fn = int(np.sum((preds == 0) & (cls_labels == 1)))
            denom = (1 + beta ** 2) * tp + beta ** 2 * fn + fp
            fb = (1 + beta ** 2) * tp / denom if denom > 0 else 0.0
            if fb > best_score:
                best_score = fb
                best_t = float(t)

        best_thresholds[cls] = round(best_t, 4)

    return best_thresholds
