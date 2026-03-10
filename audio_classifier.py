"""
Audio-based zero-shot action classifier using LAION CLAP (laion/larger_clap_general).

Loads a WAV file at 48 kHz, splits it into 3-second windows with a 1-second hop,
computes cosine similarity between each chunk's audio embedding and per-class text
embeddings, aggregates by max across chunks, and thresholds at 0.25.
"""

import numpy as np
import torch
import torch.nn.functional as F
import librosa
from transformers import ClapModel, ClapProcessor

CLASSES = ["Climbing", "Crying", "Falling", "Hitting", "Jumping", "Laughing", "Pushing", "Running"]

AUDIO_PROMPTS: dict[str, str] = {
    "Crying":   "sound of a child crying or sobbing",
    "Laughing": "sound of children laughing",
    "Falling":  "sound of someone falling or a thud",
    "Hitting":  "sound of hitting or slapping",
    "Pushing":  "sound of children struggling or grunting",
    "Running":  "sound of running footsteps",
    "Jumping":  "sound of jumping and landing",
    "Climbing": "sound of climbing",
}

SAMPLE_RATE = 48_000
WINDOW_SEC = 3
HOP_SEC = 1
THRESHOLD = 0.25
MODEL_ID = "laion/larger_clap_general"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"[audio_classifier] device: {DEVICE}")
print("[audio_classifier] loading CLAP model…")
_processor: ClapProcessor = ClapProcessor.from_pretrained(MODEL_ID)
_model: ClapModel = ClapModel.from_pretrained(MODEL_ID).to(DEVICE)
_model.eval()
print("[audio_classifier] ready.")


def _get_text_embeddings() -> dict[str, torch.Tensor]:
    """Pre-compute and L2-normalise text embeddings for all class prompts."""
    prompts = [AUDIO_PROMPTS[c] for c in CLASSES]
    inputs = _processor(text=prompts, return_tensors="pt", padding=True)
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
    with torch.no_grad():
        text_emb = _model.get_text_features(**inputs)  # (8, d)
        text_emb = F.normalize(text_emb, dim=-1)
    return {cls: text_emb[i] for i, cls in enumerate(CLASSES)}


# Compute once at module load
_text_embeddings: dict[str, torch.Tensor] = _get_text_embeddings()


def _chunk_audio(waveform: np.ndarray, sr: int) -> list[np.ndarray]:
    """Split waveform into overlapping windows of WINDOW_SEC with HOP_SEC step."""
    window_samples = int(WINDOW_SEC * sr)
    hop_samples = int(HOP_SEC * sr)
    chunks: list[np.ndarray] = []

    start = 0
    while start + window_samples <= len(waveform):
        chunks.append(waveform[start : start + window_samples])
        start += hop_samples

    # Include a final partial chunk if any audio remains and it's at least 0.5s
    remaining = waveform[start:]
    if len(remaining) >= int(0.5 * sr):
        padded = np.zeros(window_samples, dtype=waveform.dtype)
        padded[: len(remaining)] = remaining
        chunks.append(padded)

    return chunks


def classify_audio(wav_path: str) -> tuple[dict[str, float], dict[str, int]]:
    """
    Run zero-shot audio classification on a WAV file.

    Args:
        wav_path: Path to the WAV audio file.

    Returns:
        A tuple of (scores, predictions) where scores maps each class to its
        max cosine similarity across chunks and predictions maps each class to
        0 or 1 based on THRESHOLD.

    Raises:
        RuntimeError: If audio loading or inference fails.
    """
    try:
        waveform, _ = librosa.load(wav_path, sr=SAMPLE_RATE, mono=True)
    except Exception as e:
        raise RuntimeError(f"Audio loading failed for {wav_path}: {e}") from e

    chunks = _chunk_audio(waveform, SAMPLE_RATE)
    if not chunks:
        # Audio shorter than window: use the whole file padded
        padded = np.zeros(int(WINDOW_SEC * SAMPLE_RATE), dtype=waveform.dtype)
        padded[: len(waveform)] = waveform
        chunks = [padded]

    # class -> max similarity across all chunks
    max_sims: dict[str, float] = {cls: 0.0 for cls in CLASSES}

    try:
        for chunk in chunks:
            audio_inputs = _processor(
                audios=chunk,
                sampling_rate=SAMPLE_RATE,
                return_tensors="pt",
            )
            audio_inputs = {k: v.to(DEVICE) for k, v in audio_inputs.items()}
            with torch.no_grad():
                audio_emb = _model.get_audio_features(**audio_inputs)  # (1, d)
                audio_emb = F.normalize(audio_emb, dim=-1).squeeze(0)  # (d,)

            for cls in CLASSES:
                sim = torch.dot(audio_emb, _text_embeddings[cls]).item()
                if sim > max_sims[cls]:
                    max_sims[cls] = sim
    except Exception as e:
        raise RuntimeError(f"CLAP inference failed: {e}") from e

    scores: dict[str, float] = {cls: round(max_sims[cls], 4) for cls in CLASSES}
    predictions: dict[str, int] = {
        cls: int(scores[cls] >= THRESHOLD) for cls in CLASSES
    }
    return scores, predictions
