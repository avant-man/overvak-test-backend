import json
import os
import subprocess
import time
import traceback
from contextlib import asynccontextmanager

import yt_dlp
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from ground_truth import extract_video_id

_COOKIES_JSON = os.path.join(os.path.dirname(__file__), "cookies.json")


def _make_netscape_cookies(json_path: str, out_path: str) -> None:
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
                from datetime import datetime
                exp_ts = int(
                    datetime.fromisoformat(expires.replace("Z", "+00:00")).timestamp()
                ) if expires != "Session" else 0
            except Exception:
                exp_ts = 0
            name = c.get("name", "")
            value = c.get("value", "")
            f.write(f"{domain}\t{include_subdomains}\t{path}\t{secure}\t{exp_ts}\t{name}\t{value}\n")


def _ensure_video_and_audio(url: str, video_path: str, audio_path: str) -> None:
    """Download video (if missing) then extract audio WAV (if missing)."""
    downloads_dir = os.path.dirname(video_path)
    os.makedirs(downloads_dir, exist_ok=True)

    if not os.path.exists(video_path):
        print(f"[main] video not found locally, downloading: {url}")
        ydl_opts = {
            "format": "best[height<=480]/best",
            "outtmpl": os.path.join(downloads_dir, "%(id)s.%(ext)s"),
            "quiet": True,
            "no_warnings": True,
            "extractor_args": {
                "youtube": {
                    "player_client": ["mweb", "tv", "tv_simply", "web_safari"],
                    "getpot_bgutil_script": "/opt/bgutil-provider/server/build/generate_once.js",
                }
            },
        }
        if os.path.exists(_COOKIES_JSON):
            netscape_cookies = os.path.join(downloads_dir, "cookies.txt")
            _make_netscape_cookies(_COOKIES_JSON, netscape_cookies)
            ydl_opts["cookiefile"] = netscape_cookies

        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)
            downloaded = ydl.prepare_filename(info)

        if not os.path.exists(downloaded):
            # yt-dlp sometimes remuxes — try common containers
            base = os.path.splitext(downloaded)[0]
            for ext in ("mp4", "webm", "mkv"):
                candidate = f"{base}.{ext}"
                if os.path.exists(candidate):
                    downloaded = candidate
                    break
            else:
                raise FileNotFoundError(f"yt-dlp finished but downloaded file not found near: {downloaded}")

        # Normalise to .mp4 path expected by the rest of the pipeline
        if downloaded != video_path:
            os.rename(downloaded, video_path)

    if not os.path.exists(audio_path):
        print(f"[main] audio not found locally, extracting from video …")
        result = subprocess.run(
            [
                "ffmpeg", "-y",
                "-i", video_path,
                "-vn",
                "-acodec", "pcm_s16le",
                "-ar", "48000",
                "-ac", "1",
                audio_path,
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        if result.returncode != 0:
            err = result.stderr.decode(errors="replace")
            raise RuntimeError(f"ffmpeg audio extraction failed:\n{err}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Import the classifier modules here so their module-level model loading
    # (which is the slow part) happens at startup, not on the first request.
    import video_classifier  # noqa: F401
    import audio_classifier  # noqa: F401
    yield


app = FastAPI(title="Overvak Action Detection API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")
if os.path.isdir(_STATIC_DIR):
    app.mount("/static", StaticFiles(directory=_STATIC_DIR), name="static")


class AnalyzeRequest(BaseModel):
    youtube_url: str


class AnalyzeResponse(BaseModel):
    video_id: str
    predictions: dict[str, int]
    scores: dict[str, float]
    metrics: dict | None
    processing_time_seconds: float


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/")
def index():
    path = os.path.join(_STATIC_DIR, "index.html")
    if os.path.exists(path):
        return FileResponse(path)
    return {"status": "ok", "message": "Overvak API — no frontend found at static/index.html"}


@app.post("/analyze", response_model=AnalyzeResponse)
def analyze(req: AnalyzeRequest):
    video_id = extract_video_id(req.youtube_url)
    if not video_id:
        raise HTTPException(
            status_code=422,
            detail="Could not parse a YouTube video ID from the provided URL.",
        )

    downloads_dir = os.path.join(os.path.dirname(__file__), "downloads")
    video_path = os.path.join(downloads_dir, f"{video_id}.mp4")
    audio_path = os.path.join(downloads_dir, f"{video_id}.wav")

    try:
        _ensure_video_and_audio(req.youtube_url, video_path, audio_path)
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Failed to acquire video/audio: {e}")

    try:
        t0 = time.perf_counter()

        from video_classifier import classify_video
        from audio_classifier import classify_audio
        from fusion import fuse
        from metrics import compute_metrics

        video_scores, _ = classify_video(video_path)
        audio_scores, _ = classify_audio(audio_path)
        fused_scores, predictions = fuse(video_scores, audio_scores)
        metrics = compute_metrics(video_id, predictions)

        elapsed = round(time.perf_counter() - t0, 2)
    except HTTPException:
        raise
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

    return AnalyzeResponse(
        video_id=video_id,
        predictions=predictions,
        scores=fused_scores,
        metrics=metrics,
        processing_time_seconds=elapsed,
    )
