import logging
from contextlib import asynccontextmanager
from typing import List, Optional
import cv2
import numpy as np
import torch
from fastapi import FastAPI, File, UploadFile, HTTPException
from pydantic import BaseModel, Field

from inference import AuthenticDetector, get_device

logger = logging.getLogger("authentilens.api")
logging.basicConfig(level=logging.INFO)

class AnalysisResult(BaseModel):
    is_fake: bool = Field(...)
    fake_prob: float = Field(...)
    mask: List[List[float]] = Field(...)
    clf_time: float = Field(...)
    seg_time: float = Field(...)
    total_time: float = Field(...)
    cuda_alloc_mb: float = 0.0
    cuda_res_mb: float = 0.0
    # % of pixels above the segmentation threshold (None if segmentation is unavailable)
    flagged_pct: Optional[float] = None
    # "classifier" or "segmentation_override" (classifier said REAL, segmentation flagged it)
    decision_source: str = "classifier"
    segmentation_available: bool = False

_device = None
_detector = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global _device, _detector
    # Startup
    _device = get_device()
    logger.info(f"Loading Models onto {_device}...")
    _detector = AuthenticDetector(device=_device)
    yield
    # Shutdown
    logger.info("Shutting down...")

app = FastAPI(title="AuthentiLens API", lifespan=lifespan)

@app.get("/health")
async def health_check():
    if _detector is None:
        return {"status": "loading"}
    return {
        "status": "healthy",
        "classifier_checkpoint": str(_detector.classifier_path),
        "segmentation_loaded": _detector.segmentation_available,
    }

@app.post("/analyze", response_model=AnalysisResult)
async def analyze_image(file: UploadFile = File(...)):
    if _detector is None:
        raise HTTPException(status_code=503, detail="Models still loading")

    raw_bytes = await file.read()
    np_arr = np.frombuffer(raw_bytes, dtype=np.uint8)
    img_bgr = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise HTTPException(status_code=400, detail="Invalid image")

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    
    return AnalysisResult(**_detector.analyze(img_rgb))
