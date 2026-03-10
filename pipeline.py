import os
import json
import cv2
import torch
import tempfile
import yt_dlp
from PIL import Image
from transformers import AutoTokenizer, AutoModel
import torchvision.transforms as T
from ground_truth import ACTIONS, get_ground_truth

MODEL_ID = "microsoft/xclip-base-patch32"
CLIP_LEN = 8       # frames per clip
CLIP_STRIDE = 16   # sliding window step (~0.5s at 30fps)
SCORE_THRESHOLD = 0.25  # cosine-sim cutoff for a single clip to "vote"
VOTE_RATIO = 0.10       # action fires if >10% of clips voted for it
BATCH_SIZE = 8

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[pipeline] device: {DEVICE}")
if DEVICE.type == "cuda":
    print(f"[pipeline] GPU: {torch.cuda.get_device_name(0)}")

# two prompts per action so the text embedding is a bit more robust
ACTION_PROMPTS: dict[str, list[str]] = {
    "climbing":  ["a child climbing up a structure",
                  "a kid scaling a tree or jungle gym"],
    "crying":    ["a child crying and sobbing",
                  "a kid weeping with tears on their face"],
    "falling":   ["a child falling down to the ground",
                  "a kid tripping and tumbling over"],
    "hitting":   ["a child hitting or punching someone",
                  "a kid striking another person"],
    "jumping":   ["a child jumping up and down",
                  "a kid leaping through the air"],
    "laughing":  ["a child laughing and giggling",
                  "a kid smiling and having fun"],
    "pushing":   ["a child pushing another person",
                  "a kid shoving someone forcefully"],
    "running":   ["a child running at full speed",
                  "a kid sprinting and racing"],
}

_transform = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

print("[pipeline] loading VideoCLIP…")
_tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
_model = AutoModel.from_pretrained(MODEL_ID).to(DEVICE)
_model.eval()
print("[pipeline] ready.")


def _encode_texts() -> dict[str, torch.Tensor]:
    result = {}
    for action, prompts in ACTION_PROMPTS.items():
        inputs = _tokenizer(
            prompts, return_tensors="pt",
            padding=True, truncation=True, max_length=77
        )
        inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
        with torch.no_grad():
            out = _model.get_text_features(**inputs)
            emb = out.pooler_output if hasattr(out, "pooler_output") else out
            emb = emb / emb.norm(dim=-1, keepdim=True)
            result[action] = emb.mean(dim=0)
    return result


_text_embeddings = _encode_texts()


_COOKIES_JSON = os.path.join(os.path.dirname(__file__), "cookies.json")


def _make_netscape_cookies(json_path: str, out_path: str) -> None:
    """Convert cookies.json to Netscape cookie file format for yt-dlp."""
    with open(json_path, "r", encoding="utf-8") as f:
        cookies = json.load(f)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("# Netscape HTTP Cookie File\n")
        for c in cookies:
            domain = c.get("domain", "")
            include_subdomains = "TRUE" if domain.startswith(".") else "FALSE"
            path = c.get("path", "/")
            secure = "TRUE" if c.get("secure") else "FALSE"
            expires = c.get("expires", "Session")
            try:
                from datetime import datetime, timezone
                exp_ts = int(
                    datetime.fromisoformat(expires.replace("Z", "+00:00"))
                    .timestamp()
                ) if expires != "Session" else 0
            except Exception:
                exp_ts = 0
            name = c.get("name", "")
            value = c.get("value", "")
            f.write(f"{domain}\t{include_subdomains}\t{path}\t{secure}\t{exp_ts}\t{name}\t{value}\n")


def download_video(url: str, out_dir: str) -> str:
    # Use single-file formats only so ffmpeg is not required for merging
    ydl_opts = {
        "format": "best[height<=480]/best",
        "outtmpl": os.path.join(out_dir, "%(id)s.%(ext)s"),
        "quiet": True,
        "no_warnings": True,
        "extractor_args": {"youtube": {"player_client": ["ios", "android", "web"]}},
        "http_headers": {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/124.0.0.0 Safari/537.36"
            ),
        },
    }
    if os.path.exists(_COOKIES_JSON):
        netscape_cookies = os.path.join(out_dir, "cookies.txt")
        _make_netscape_cookies(_COOKIES_JSON, netscape_cookies)
        ydl_opts["cookiefile"] = netscape_cookies
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)
        fname = ydl.prepare_filename(info)

    if not os.path.exists(fname):
        # yt-dlp sometimes remuxes to a different container, try common ones
        base = os.path.splitext(fname)[0]
        for ext in ("mp4", "webm", "mkv"):
            candidate = f"{base}.{ext}"
            if os.path.exists(candidate):
                return candidate
        raise FileNotFoundError(f"downloaded file not found near: {fname}")
    return fname


def extract_clips(video_path: str) -> list[torch.Tensor]:
    cap = cv2.VideoCapture(video_path)
    all_frames: list[torch.Tensor] = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        all_frames.append(_transform(Image.fromarray(rgb)))
    cap.release()

    if not all_frames:
        return []

    max_start = len(all_frames) - CLIP_LEN + 1
    if max_start <= 0:
        return []

    clips = []
    for start in range(0, max_start, CLIP_STRIDE):
        clip = torch.stack(all_frames[start : start + CLIP_LEN], dim=0)  # (T,C,H,W)
        clips.append(clip)
    return clips


def classify_clips(clips: list[torch.Tensor]) -> dict[str, float]:
    if not clips:
        return {a: 0.0 for a in ACTIONS}

    all_vid_emb = []
    for i in range(0, len(clips), BATCH_SIZE):
        batch = torch.stack(clips[i : i + BATCH_SIZE], dim=0).to(DEVICE)  # (B,T,C,H,W)
        with torch.no_grad():
            out = _model.get_video_features(pixel_values=batch)
            emb = out.pooler_output if hasattr(out, "pooler_output") else out
            emb = emb / emb.norm(dim=-1, keepdim=True)
        all_vid_emb.append(emb)

    vid_emb = torch.cat(all_vid_emb, dim=0)  # (N_clips, d)

    vote_ratios: dict[str, float] = {}
    for action in ACTIONS:
        sims = vid_emb @ _text_embeddings[action]
        votes = (sims >= SCORE_THRESHOLD).float().sum().item()
        vote_ratios[action] = votes / len(clips)

    return vote_ratios


def compute_metrics(predictions: dict[str, int], ground_truth: dict[str, int]) -> dict:
    tp = sum(1 for a in ACTIONS if predictions[a] == 1 and ground_truth[a] == 1)
    fp = sum(1 for a in ACTIONS if predictions[a] == 1 and ground_truth[a] == 0)
    fn = sum(1 for a in ACTIONS if predictions[a] == 0 and ground_truth[a] == 1)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    return {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "tp": tp, "fp": fp, "fn": fn,
    }


def run_pipeline(url: str, video_id: str) -> dict:
    with tempfile.TemporaryDirectory() as tmp:
        video_path = download_video(url, tmp)
        clips = extract_clips(video_path)
        vote_ratios = classify_clips(clips)
        predictions = {a: int(vote_ratios[a] >= VOTE_RATIO) for a in ACTIONS}
        detected = [a for a in ACTIONS if predictions[a] == 1]
        gt = get_ground_truth(video_id)
        metrics = compute_metrics(predictions, gt) if gt else None

        return {
            "video_id": video_id,
            "n_clips_sampled": len(clips),
            "device": str(DEVICE),
            "vote_ratios": {a: round(vote_ratios[a], 4) for a in ACTIONS},
            "predictions": predictions,
            "detected_actions": detected,
            "ground_truth": gt,
            "metrics": metrics,
        }
