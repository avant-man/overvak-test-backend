"""
Fusion module: combines video and audio classification scores via weighted average.

Per-class weights are tuned so that audio-dominant actions (Crying, Laughing)
rely more on the audio model, while visually clear actions (Climbing, Jumping)
rely more on the video model.
"""

CLASSES = ["Climbing", "Crying", "Falling", "Hitting", "Jumping", "Laughing", "Pushing", "Running"]

# (video_weight, audio_weight) — must sum to 1.0 per class
WEIGHTS: dict[str, tuple[float, float]] = {
    "Climbing": (0.8, 0.2),
    "Crying":   (0.3, 0.7),
    "Falling":  (0.6, 0.4),
    "Hitting":  (0.6, 0.4),
    "Jumping":  (0.8, 0.2),
    "Laughing": (0.3, 0.7),
    "Pushing":  (0.7, 0.3),
    "Running":  (0.7, 0.3),
}

FUSION_THRESHOLD = 0.35


def fuse(
    video_scores: dict[str, float],
    audio_scores: dict[str, float],
) -> tuple[dict[str, float], dict[str, int]]:
    """
    Combine video and audio scores into fused scores and binary predictions.

    Args:
        video_scores: Per-class softmax probabilities from the video classifier.
        audio_scores: Per-class max cosine similarities from the audio classifier.

    Returns:
        A tuple of (fused_scores, predictions) where fused_scores maps each class
        to its weighted-average score and predictions maps each class to 0 or 1.
    """
    fused_scores: dict[str, float] = {}
    predictions: dict[str, int] = {}

    for cls in CLASSES:
        v_w, a_w = WEIGHTS[cls]
        score = v_w * video_scores.get(cls, 0.0) + a_w * audio_scores.get(cls, 0.0)
        fused_scores[cls] = round(score, 4)
        predictions[cls] = int(score >= FUSION_THRESHOLD)

    return fused_scores, predictions
