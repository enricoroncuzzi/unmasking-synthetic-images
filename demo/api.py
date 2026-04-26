"""
FastAPI REST API — Unmasking Synthetic Images.

Minimal inference API exposing /health and /predict endpoints.
Runs the same MoEPipeline used by the Gradio demo.

Usage:
    uvicorn demo.api:app --host 0.0.0.0 --port 8000

Docs:
    http://localhost:8000/docs
"""

import io
import sys
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal, Optional

from fastapi import FastAPI, File, HTTPException, UploadFile
from PIL import Image
from pydantic import BaseModel

# Ensure demo/ and repo root are on sys.path so pipeline.py and models/ resolve
# whether this runs as `uvicorn demo.api:app` from repo root or from Docker.
_DEMO_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _DEMO_DIR.parent
sys.path.insert(0, str(_REPO_ROOT))
sys.path.insert(0, str(_DEMO_DIR))

from pipeline import MoEPipeline  # noqa: E402

PIPELINE: Optional[MoEPipeline] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global PIPELINE
    PIPELINE = MoEPipeline(device="cpu", strategy="logit")
    yield
    PIPELINE = None


app = FastAPI(
    title="Unmasking Synthetic Images API",
    description="Forensic MoE system for AI-generated image detection and attribution.",
    version="0.5.0",
    lifespan=lifespan,
)


class PredictionResponse(BaseModel):
    prediction: Literal["real", "synthetic"]
    confidence: float
    alpha_weights: dict[str, float]
    attributed_source: Optional[str]


@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "model_loaded": PIPELINE is not None,
        "strategy": "logit",
    }


@app.post("/predict", response_model=PredictionResponse)
async def predict(file: UploadFile = File(...)):
    if file.content_type not in {"image/jpeg", "image/png"}:
        raise HTTPException(status_code=400, detail="Only JPEG or PNG accepted")
    contents = await file.read()
    if len(contents) > 10 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="File too large (max 10 MB)")
    image = Image.open(io.BytesIO(contents)).convert("RGB")
    return PIPELINE.predict(image)
