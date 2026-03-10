"""
Video-based zero-shot action classifier using microsoft/xclip-base-patch32.

Samples 8 evenly-spaced frames from an MP4 file, runs a single forward pass
with all 8 class prompts, and applies softmax + threshold to produce binary
predictions.
"""

import cv2
import torch
import torch.nn.functional as F
from PIL import Image
from transformers import XCLIPModel, XCLIPProcessor

CLASSES = ["Climbing", "Crying", "Falling", "Hitting", "Jumping", "Laughing", "Pushing", "Running"]

VIDEO_PROMPTS: dict[str, str] = {
    "Climbing": "a child climbing",
    "Crying":   "a child crying",
    "Falling":  "a child falling down",
    "Hitting":  "a child hitting someone",
    "Jumping":  "a child jumping",
    "Laughing": "a child laughing",
    "Pushing":  "a child pushing another person",
    "Running":  "a child running",
}

NUM_FRAMES = 8
THRESHOLD = 0.15
MODEL_ID = "microsoft/xclip-base-patch32"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"[video_classifier] device: {DEVICE}")
print("[video_classifier] loading X-CLIP model…")
_processor: XCLIPProcessor = XCLIPProcessor.from_pretrained(MODEL_ID)
_model: XCLIPModel = XCLIPModel.from_pretrained(MODEL_ID).to(DEVICE)
_model.eval()
print("[video_classifier] ready.")


def _sample_frames(video_path: str, n: int = NUM_FRAMES) -> list[Image.Image]:
    """Sample n evenly-spaced RGB frames from a video file using OpenCV."""
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        cap.release()
        raise ValueError(f"Cannot read frame count from {video_path}")

    indices = [int(i * (total - 1) / (n - 1)) for i in range(n)]
    frames: list[Image.Image] = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            continue
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(rgb).resize((224, 224))
        frames.append(img)
    cap.release()

    if len(frames) == 0:
        raise ValueError(f"No frames could be read from {video_path}")
    # Pad to exactly n frames if some reads failed
    while len(frames) < n:
        frames.append(frames[-1])
    return frames[:n]


def classify_video(video_path: str) -> tuple[dict[str, float], dict[str, int]]:
    """
    Run zero-shot video classification on an MP4 file.

    Args:
        video_path: Path to the MP4 video file.

    Returns:
        A tuple of (scores, predictions) where scores maps each class to its
        softmax probability and predictions maps each class to 0 or 1.

    Raises:
        RuntimeError: If inference fails.
    """
    try:
        frames = _sample_frames(video_path)
    except Exception as e:
        raise RuntimeError(f"Frame extraction failed: {e}") from e

    text_prompts = [VIDEO_PROMPTS[c] for c in CLASSES]

    try:
        inputs = _processor(
            text=text_prompts,
            videos=frames,
            return_tensors="pt",
            padding=True,
        )
        inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = _model(**inputs)
            # logits_per_video: (1, num_classes)
            logits = outputs.logits_per_video  # shape (1, 8)
            probs = F.softmax(logits, dim=-1).squeeze(0)  # shape (8,)
    except Exception as e:
        raise RuntimeError(f"X-CLIP inference failed: {e}") from e

    scores: dict[str, float] = {
        cls: round(probs[i].item(), 4) for i, cls in enumerate(CLASSES)
    }
    predictions: dict[str, int] = {
        cls: int(scores[cls] >= THRESHOLD) for cls in CLASSES
    }
    return scores, predictions
