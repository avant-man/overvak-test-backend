import os
import time
import traceback
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from ground_truth import extract_video_id


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

    if not os.path.exists(video_path):
        raise HTTPException(
            status_code=404,
            detail=f"Video file not found: downloads/{video_id}.mp4",
        )
    if not os.path.exists(audio_path):
        raise HTTPException(
            status_code=404,
            detail=f"Audio file not found: downloads/{video_id}.wav",
        )

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
