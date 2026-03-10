"""
Audio-based zero-shot action classifier using LAION CLAP (laion/larger_clap_general).

Loads a WAV file at 48 kHz, splits it into 3-second windows with a 1-second hop,
scores each window against per-class ensembled text embeddings, then aggregates
across windows using a per-class strategy:

  - MEAN aggregation (default): averages scores across all windows.  Reduces
    false alarms from a single noisy chunk for sustained actions (Running,
    Climbing, etc.).
  - MAX aggregation (Crying, Laughing): takes the highest per-window score.
    Crying and laughing are intermittent — a child may cry for 5 seconds inside
    a 60-second clip.  Mean aggregation drowns that signal in silence.  Max
    aggregation fires whenever any window has strong evidence.

Key design choices (research-backed):
- Prompt ensembling: multiple acoustically-descriptive prompts per class, averaged
  and re-normalised (Olvera et al. 2024 / TSPE Anand et al. 2025).
- No negation phrases: CLAP ignores "not X" in text (arXiv:2602.21035).
- Per-class aggregation: max for intermittent vocal events; mean for sustained
  physical actions.
- Prompt format: capitalised class name + period works best for LAION-CLAP
  (Olvera et al. 2024 empirical benchmark).
"""

import numpy as np
import torch
import torch.nn.functional as F
import librosa
from transformers import ClapModel, ClapProcessor

CLASSES = ["Climbing", "Crying", "Falling", "Hitting", "Jumping", "Laughing", "Pushing", "Running"]

# Per-class score aggregation strategy across windows.
# "max"  → highest per-window score (best for intermittent vocal events).
# "mean" → average per-window score (best for sustained physical actions).
AGGREGATION: dict[str, str] = {
    "Climbing": "mean",
    "Crying":   "max",
    "Falling":  "mean",
    "Hitting":  "mean",
    "Jumping":  "mean",
    "Laughing": "max",
    "Pushing":  "mean",
    "Running":  "mean",
}

# Acoustically-discriminative prompts per class — no negation phrases.
# Embeddings are averaged and re-normalised at load time (prompt ensembling).
# Crying and Laughing have 5 prompts (vs 3 for others) to improve CLAP recall
# for these harder-to-isolate vocal events.
AUDIO_PROMPTS: dict[str, list[str]] = {
    "Crying": [
        "Crying: sustained wailing and sobbing with irregular breath, rising pitch, wet vocal quality.",
        "A distressed human vocalization of crying with high-pitched exhales and audible sniffles.",
        "The sound of a person crying with repeated sobs, broken inhales, and tearful wailing.",
        "A child crying loudly with prolonged high-pitched wails and gasping breath.",
        "Distressed sobbing and whimpering with irregular inhales and wet vocal quality.",
    ],
    "Laughing": [
        "Laughing: rhythmic explosive bursts of voiced air in a ha-he-ho pattern, ascending tempo.",
        "A joyful human sound of laughter with short staccato vocal bursts and bright upbeat tone.",
        "The sound of a person laughing with repeating rhythmic vocal pulses and rising energy.",
        "A child laughing out loud with rapid bursts of giggles and high-pitched joyful sounds.",
        "Uncontrollable laughter with fast rhythmic vocal outbursts and rising pitch.",
    ],
    "Falling": [
        "Falling: a sudden loud thud as a body hits a hard surface, sharp attack and fast decay.",
        "A percussive impact sound of someone falling, a brief heavy thud followed by silence.",
        "The sound of a heavy fall: a sudden impact thud with fast attack and short resonance.",
    ],
    "Hitting": [
        "Hitting: a single instantaneous crack or slap, an extremely brief transient with near-zero duration.",
        "A sharp isolated punch or slap sound — one sudden loud crack followed by immediate silence.",
        "The sound of a hard single strike: a fast dry crack with no sustained noise before or after.",
        "A quick violent slap or punch impact — one short high-energy burst lasting under 0.1 seconds.",
        "A dull single thud of a fist or hand hitting a body — one brief muffled low-frequency impact.",
        "The sound of a hit on clothing or soft surface — a short muted thump with fast attack and decay.",
    ],
    "Pushing": [
        "Pushing: slow sustained body contact with continuous low rumbling and shuffling friction.",
        "The sound of a prolonged physical struggle with low grunting, scraping, and sliding — no sharp crack.",
        "A sustained low-frequency effort sound with friction and displacement noise lasting several seconds.",
        "Continuous body-to-body contact noise with labored breathing and slow sliding — no sudden impact.",
    ],
    "Running": [
        "Running: rapid rhythmic footsteps with consistent fast tempo and quick cadence.",
        "The sound of fast running footsteps, repeated impacts on a hard or soft surface.",
        "Rapid repeated footfall sounds with short even intervals between each step.",
    ],
    "Jumping": [
        "Jumping: a brief push-off rustle followed by a landing impact thud.",
        "The sound of jumping: a launch sound and then a hard landing thud on the ground.",
        "A periodic bounce sound with rhythmic landing impacts at regular intervals.",
    ],
    "Climbing": [
        "Climbing: shuffling, scraping, and gripping sounds of hands and feet on a surface.",
        "The sound of climbing with friction, creaking of structure, and exertion grunts.",
        "A continuous scraping and rustling sound of limbs moving against a climbing surface.",
        "Repeated hand and foot placements with effort grunts and rhythmic body shifting sounds.",
    ],
}

# Pairs of classes that are mutually exclusive within a single audio window —
# only the higher-scoring one is kept.
# NOTE: ("Crying", "Laughing") has been intentionally removed.  A video clip
# can contain both (different children, or different moments in the same clip).
# Suppressing the lower of the two caused valid Crying detections to be dropped
# whenever Laughing also scored high.
MUTUAL_EXCLUSIONS: list[tuple[str, str]] = [
    ("Hitting", "Pushing"),
]

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
    """
    Pre-compute ensembled, L2-normalised text embeddings for all class prompts.

    Each class has multiple prompts; all are encoded, averaged, then re-normalised
    to produce a single robust embedding per class (prompt ensembling).
    """
    embeddings: dict[str, torch.Tensor] = {}
    for cls in CLASSES:
        prompts = AUDIO_PROMPTS[cls]
        inputs = _processor(text=prompts, return_tensors="pt", padding=True)
        inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
        with torch.no_grad():
            out = _model.get_text_features(**inputs)
            text_emb = out if isinstance(out, torch.Tensor) else out.pooler_output
            text_emb = F.normalize(text_emb, dim=-1)  # (n_prompts, d)
        # Average across prompts, then re-normalise
        mean_emb = text_emb.mean(dim=0)
        embeddings[cls] = F.normalize(mean_emb, dim=-1)
    return embeddings


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

    Scores each 3-second window independently against the ensembled text
    embeddings, then aggregates per class:
      - MEAN: average across all windows (sustained actions — less false alarms).
      - MAX:  best window score (intermittent vocal events like Crying/Laughing —
              preserves a brief but strong burst of evidence).

    Args:
        wav_path: Path to the WAV audio file.

    Returns:
        A tuple of (scores, predictions) where scores maps each class to its
        aggregated cosine similarity and predictions maps each class to 0 or 1.

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

    # chunk_scores[cls] accumulates one cosine-similarity per window
    chunk_scores: dict[str, list[float]] = {cls: [] for cls in CLASSES}

    try:
        for chunk in chunks:
            audio_inputs = _processor(
                audio=chunk,
                sampling_rate=SAMPLE_RATE,
                return_tensors="pt",
            )
            audio_inputs = {k: v.to(DEVICE) for k, v in audio_inputs.items()}
            with torch.no_grad():
                out = _model.get_audio_features(**audio_inputs)
                audio_emb = out if isinstance(out, torch.Tensor) else out.pooler_output
                audio_emb = F.normalize(audio_emb, dim=-1).squeeze(0)  # (d,)
            for cls in CLASSES:
                sim = torch.dot(audio_emb, _text_embeddings[cls]).item()
                chunk_scores[cls].append(sim)
    except Exception as e:
        raise RuntimeError(f"CLAP inference failed: {e}") from e

    # Aggregate per class according to AGGREGATION strategy
    scores: dict[str, float] = {}
    for cls in CLASSES:
        vals = chunk_scores[cls]
        if AGGREGATION.get(cls) == "max":
            scores[cls] = round(max(vals), 4)
        else:
            scores[cls] = round(sum(vals) / len(vals), 4)

    predictions: dict[str, int] = {
        cls: int(scores[cls] >= THRESHOLD) for cls in CLASSES
    }

    # Suppress the lower-scoring class for mutually exclusive pairs
    for cls_a, cls_b in MUTUAL_EXCLUSIONS:
        if predictions[cls_a] and predictions[cls_b]:
            loser = cls_a if scores[cls_a] < scores[cls_b] else cls_b
            predictions[loser] = 0

    return scores, predictions
