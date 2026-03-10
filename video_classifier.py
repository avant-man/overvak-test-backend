"""
Video-based zero-shot action classifier using microsoft/xclip-base-patch32.

Samples 8 evenly-spaced frames from an MP4 file, runs a single forward pass
with pre-computed ensembled text embeddings, and produces a relative-to-uniform
score per class that maps to [-1, 1].

Score normalisation: after softmax, each class probability is expressed as how
many times above the uniform baseline (1/N) it sits, capped at 1.0:
    score = min((p - 1/N) / (1/N), 1.0)
A score of 0 means "exactly as likely as random chance"; a score of 1.0 means
"twice the uniform probability or higher".  This puts video scores on the same
[-1, 1] scale as CLAP audio cosine similarities, making fusion weights
directly interpretable.

Key design choices (research-backed):
- Prompt ensembling: 3-5 action-oriented prompts per class are encoded at load
  time, averaged, and re-normalised to form a single robust text embedding per
  class (validated in X-CLIP ablations and ActionCLIP).
- Relative-to-uniform normalisation restores discriminability that raw cosine
  similarities lack (X-CLIP cosine sims cluster near 0.1-0.2 for all classes
  without the learned logit_scale temperature).
"""

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from transformers import XCLIPModel, XCLIPProcessor

CLASSES = ["Climbing", "Crying", "Falling", "Hitting", "Jumping", "Laughing", "Pushing", "Running"]

# 3-5 action-oriented prompt variants per class.
# Embeddings are averaged and re-normalised at load time (prompt ensembling).
VIDEO_PROMPTS: dict[str, list[str]] = {
    "Climbing": [
        "a child climbing up a structure or ladder",
        "a kid scaling a jungle gym or tree",
        "a child pulling themselves upward with hands and feet",
    ],
    "Crying": [
        "a child crying with tears, face contorted in distress",
        "a kid sobbing and weeping visibly",
        "a child in emotional pain with mouth open crying",
    ],
    "Falling": [
        "a child falling down to the ground",
        "a kid tripping and tumbling over",
        "a child losing balance and collapsing to the floor",
    ],
    "Hitting": [
        "a child throwing a single fast punch or slap at another person, arm snapping quickly",
        "a kid striking someone with one rapid blow and then the arm retracts instantly",
        "a child hitting another person with a brief high-speed arm swing — not a slow push",
        "a quick single strike where a child's arm moves fast and retracts immediately",
    ],
    "Jumping": [
        "a child jumping up into the air",
        "a kid leaping off the ground with both feet",
        "a child bouncing and jumping repeatedly",
    ],
    "Laughing": [
        "a child laughing with open mouth, body shaking with joy",
        "a kid giggling and smiling broadly",
        "a child in joyful laughter with visible happiness",
    ],
    "Pushing": [
        "a child pressing both hands against another person and slowly shoving them away",
        "a kid leaning their whole body weight into another person with sustained slow force",
        "a child steadily displacing another person with prolonged arm pressure, not a quick strike",
        "two children struggling with sustained body contact as one pushes the other slowly",
    ],
    "Running": [
        "a child running at full speed",
        "a kid sprinting with fast leg movement",
        "a child racing with rapid strides and pumping arms",
    ],
}

# X-CLIP (xclip-base-patch32) has fixed temporal positional embeddings for
# exactly 8 frames — do NOT change this value or inference will crash.
NUM_FRAMES = 8
THRESHOLD = 0.15  # relative-to-uniform threshold: class must be ≥1.15× the baseline
MODEL_ID = "microsoft/xclip-base-patch32"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"[video_classifier] device: {DEVICE}")
print("[video_classifier] loading X-CLIP model…")
_processor: XCLIPProcessor = XCLIPProcessor.from_pretrained(MODEL_ID)
_model: XCLIPModel = XCLIPModel.from_pretrained(MODEL_ID).to(DEVICE)
_model.eval()
print("[video_classifier] ready.")


def _build_text_embeddings() -> torch.Tensor:
    """
    Pre-compute ensembled, L2-normalised text embeddings for all classes.

    Each class has multiple prompt variants; all are encoded, averaged per class,
    and re-normalised.  Returns a (num_classes, d) tensor in CLASSES order.
    """
    class_embeds: list[torch.Tensor] = []
    for cls in CLASSES:
        prompts = VIDEO_PROMPTS[cls]
        inputs = _processor(text=prompts, return_tensors="pt", padding=True)
        inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
        with torch.no_grad():
            out = _model.get_text_features(**inputs)
            text_feats = out if isinstance(out, torch.Tensor) else out.pooler_output
            text_feats = F.normalize(text_feats, dim=-1)  # (n_prompts, d)
        mean_feat = F.normalize(text_feats.mean(dim=0), dim=-1)  # (d,)
        class_embeds.append(mean_feat)
    return torch.stack(class_embeds, dim=0)  # (num_classes, d)


# Compute once at module load
_text_embeddings: torch.Tensor = _build_text_embeddings()


def _motion_scores(video_path: str, n_samples: int = 64) -> list[tuple[int, float]]:
    """
    Compute per-frame motion scores (mean absolute frame difference) for a
    lightweight grid of n_samples evenly-spaced frames.  Returns a list of
    (frame_index, score) sorted descending by score.

    Used to bias frame selection toward high-motion moments so that brief
    events (e.g. a fall lasting <1 s) are not missed by uniform sampling.
    """
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 1:
        cap.release()
        return []

    step = max(1, total // n_samples)
    sample_indices = list(range(0, total, step))[:n_samples]

    grays: list[tuple[int, np.ndarray]] = []
    for idx in sample_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            continue
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)
        grays.append((idx, gray))
    cap.release()

    scores: list[tuple[int, float]] = []
    for i in range(1, len(grays)):
        idx, g = grays[i]
        _, g_prev = grays[i - 1]
        diff = float(np.mean(np.abs(g - g_prev)))
        scores.append((idx, diff))

    scores.sort(key=lambda x: x[1], reverse=True)
    return scores


def _sample_frames(video_path: str, n: int = NUM_FRAMES) -> list[Image.Image]:
    """
    Sample n RGB frames from a video file using a hybrid strategy:
    - Half the frames are evenly spaced (uniform coverage).
    - Half come from the highest-motion moments detected via frame differencing.

    The motion-biased half ensures brief, high-energy events like falls are
    represented even when uniform sampling would miss them entirely.
    """
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        cap.release()
        raise ValueError(f"Cannot read frame count from {video_path}")

    # Uniform indices — always include first and last frame
    n_uniform = max(2, n // 2)
    uniform_indices = [int(i * (total - 1) / (n_uniform - 1)) for i in range(n_uniform)]

    # Motion-biased indices — top-(n - n_uniform) highest-motion frames
    n_motion = n - n_uniform
    motion_scores = _motion_scores(video_path, n_samples=min(128, total))
    motion_indices = [idx for idx, _ in motion_scores[:n_motion]]

    # Merge, deduplicate, and sort
    all_indices = sorted(set(uniform_indices + motion_indices))

    def _read_frame(cap: cv2.VideoCapture, idx: int) -> Image.Image | None:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            return None
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return Image.fromarray(rgb).resize((224, 224))

    frames: list[Image.Image] = []
    for idx in all_indices:
        img = _read_frame(cap, idx)
        if img is not None:
            frames.append(img)
    cap.release()

    if len(frames) == 0:
        raise ValueError(f"No frames could be read from {video_path}")

    # Sub-sample or pad to exactly n frames
    if len(frames) >= n:
        step = len(frames) / n
        frames = [frames[int(i * step)] for i in range(n)]
    else:
        while len(frames) < n:
            frames.append(frames[-1])

    return frames


def classify_video(video_path: str) -> tuple[dict[str, float], dict[str, int]]:
    """
    Run zero-shot video classification on an MP4 file.

    Scores each class using softmax probabilities normalised relative to the
    uniform baseline, capped at 1.0.  Scores sit in [-1, 1] and are directly
    comparable to CLAP audio cosine similarities used in fusion.

    Args:
        video_path: Path to the MP4 video file.

    Returns:
        A tuple of (scores, predictions) where scores maps each class to its
        relative-to-uniform score in [-1, 1] and predictions maps each class
        to 0 or 1.

    Raises:
        RuntimeError: If inference fails.
    """
    try:
        frames = _sample_frames(video_path)
    except Exception as e:
        raise RuntimeError(f"Frame extraction failed: {e}") from e

    try:
        # Use a dummy single text to satisfy the processor (we override text features)
        inputs = _processor(
            text=["placeholder"],
            images=frames,
            return_tensors="pt",
            padding=True,
        )
        inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
        if inputs["pixel_values"].dim() == 4:
            inputs["pixel_values"] = inputs["pixel_values"].unsqueeze(0)

        with torch.no_grad():
            # Get video embedding
            video_outputs = _model.get_video_features(
                pixel_values=inputs["pixel_values"]
            )
            video_emb = video_outputs if isinstance(video_outputs, torch.Tensor) else video_outputs.pooler_output
            video_emb = F.normalize(video_emb, dim=-1)  # (1, d)

            # Scale cosine sims with logit_scale so softmax is discriminative,
            # then convert to relative-to-uniform scores in [-1, 1].
            logit_scale = _model.logit_scale.exp()
            logits = logit_scale * (video_emb @ _text_embeddings.T)  # (1, N)
            probs = F.softmax(logits, dim=-1).squeeze(0)              # (N,)

    except Exception as e:
        raise RuntimeError(f"X-CLIP inference failed: {e}") from e

    # Normalise each class relative to the uniform baseline (1/N).
    # score = 0 → exactly as likely as random; score = 1.0 → 2× baseline or more.
    # Scores in [-1, 1] match the CLAP cosine-similarity scale used in fusion.
    N = len(CLASSES)
    uniform = 1.0 / N
    scores: dict[str, float] = {
        cls: round(min((probs[i].item() - uniform) / uniform, 1.0), 4)
        for i, cls in enumerate(CLASSES)
    }
    predictions: dict[str, int] = {
        cls: int(scores[cls] >= THRESHOLD) for cls in CLASSES
    }
    return scores, predictions
