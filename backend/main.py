import logging
from typing import List
import cv2
import numpy as np
import torch
from fastapi import FastAPI, File, UploadFile, HTTPException
from pydantic import BaseModel, Field

from inference import AuthenticDetector, get_device

logger = logging.getLogger("aperture.api")
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

app = FastAPI(title="ApertureAuthentiLens API")

_device = None
_detector = None

@app.on_event("startup")
async def _load_models():
    global _device, _detector
    _device = get_device()
    logger.info(f"Loading Models onto {_device}...")
    _detector = AuthenticDetector(device=_device)

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

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
    
    is_fake, fake_prob, mask, clf_time, seg_time, total_time, cuda_alloc_mb, cuda_res_mb = _detector.analyze(img_rgb)
    return AnalysisResult(
        is_fake=is_fake, 
        fake_prob=fake_prob, 
        mask=mask,
        clf_time=clf_time,
        seg_time=seg_time,
        total_time=total_time,
        cuda_alloc_mb=cuda_alloc_mb,
        cuda_res_mb=cuda_res_mb
    )
