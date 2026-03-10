from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import traceback

from ground_truth import extract_video_id
from pipeline import run_pipeline

app = FastAPI(title="Overvak Action Detection API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class AnalyzeRequest(BaseModel):
    url: str


class AnalyzeResponse(BaseModel):
    video_id: str
    n_clips_sampled: int
    device: str
    detected_actions: list[str]
    vote_ratios: dict[str, float]
    predictions: dict[str, int]
    ground_truth: dict[str, int] | None
    metrics: dict | None


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/analyze", response_model=AnalyzeResponse)
def analyze(req: AnalyzeRequest):
    video_id = extract_video_id(req.url)
    if not video_id:
        raise HTTPException(
            status_code=422,
            detail="Could not parse a YouTube video ID from the provided URL."
        )
    try:
        result = run_pipeline(url=req.url, video_id=video_id)
    except FileNotFoundError as e:
        raise HTTPException(status_code=500, detail=f"Download failed: {e}")
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
    return result
