"""
============================================================================
ApertureAuthentiLens — FastAPI Backend Server
============================================================================
Exposes a POST /analyze endpoint that:
    1. Accepts an uploaded image (JPEG / PNG).
    2. Preprocesses: decode → BGR→RGB → resize to 224×224 → tensor.
    3. Runs both ML models (UnivFD + ExplainableViT) under
       ``torch.profiler.profile`` to capture execution metrics.
    4. Returns a JSON payload with global score, patch scores,
       spatial attention matrix, and profiling summary.

Health endpoint:  GET /health
============================================================================
"""

from __future__ import annotations

import io
import logging
from typing import Any, Dict, List

import cv2
import numpy as np
import torch
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from inference import ExplainableViTDetector, UnivFDDetector, get_device

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logger = logging.getLogger("aperture.api")
logging.basicConfig(level=logging.INFO)

# ============================================================================
# Pydantic Response Schema
# ============================================================================

class AnalysisResult(BaseModel):
    """Schema for the /analyze endpoint response."""

    global_score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Overall probability that the image is AI-generated.",
    )
    patch_scores: List[float] = Field(
        ...,
        description="Per-patch fake-confidence scores from UnivFD.",
    )
    attention_matrix: List[List[float]] = Field(
        ...,
        description="2-D spatial attention heatmap from ExplainableViT.",
    )
    profiling_summary: str = Field(
        ...,
        description="Formatted torch.profiler output (CPU/CUDA timing).",
    )


# ============================================================================
# FastAPI Application
# ============================================================================

app: FastAPI = FastAPI(
    title="ApertureAuthentiLens API",
    description=(
        "Visual forensic API that localises AI manipulations in images "
        "using zero-shot pre-trained models."
    ),
    version="1.0.0",
)


# ---------------------------------------------------------------------------
# Model initialisation at startup
# ---------------------------------------------------------------------------

# Globals populated once at startup
_device: torch.device | None = None
_univfd: UnivFDDetector | None = None
_explainable: ExplainableViTDetector | None = None


@app.on_event("startup")
async def _load_models() -> None:
    """Load both inference models into memory on server start."""
    global _device, _univfd, _explainable

    logger.info("=== ApertureAuthentiLens Backend — Startup ===")
    _device = get_device()

    logger.info("Initialising UnivFDDetector …")
    _univfd = UnivFDDetector(device=_device)

    logger.info("Initialising ExplainableViTDetector …")
    _explainable = ExplainableViTDetector(device=_device)

    logger.info("=== All models loaded — server ready. ===")


# ============================================================================
# Endpoints
# ============================================================================

@app.get("/health")
async def health_check() -> Dict[str, str]:
    """Simple liveness probe for Docker healthcheck."""
    return {"status": "healthy"}


@app.post("/analyze", response_model=AnalysisResult)
async def analyze_image(file: UploadFile = File(...)) -> AnalysisResult:
    """
    Accept an uploaded image, run both ML models, and return a forensic
    analysis payload.

    Parameters
    ----------
    file : UploadFile
        The image to analyse (JPEG or PNG).

    Returns
    -------
    AnalysisResult
        JSON with global_score, patch_scores, attention_matrix, and
        profiling_summary.
    """
    # ----- Guard: models must be loaded ------------------------------------
    if _univfd is None or _explainable is None or _device is None:
        raise HTTPException(
            status_code=503,
            detail="Models are still loading. Please retry shortly.",
        )

    # =====================================================================
    # 1.  Read & preprocess the uploaded image
    # =====================================================================
    try:
        raw_bytes: bytes = await file.read()
        np_arr: np.ndarray = np.frombuffer(raw_bytes, dtype=np.uint8)
        img_bgr: np.ndarray = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        if img_bgr is None:
            raise ValueError("cv2.imdecode returned None")
    except Exception as exc:
        raise HTTPException(
            status_code=400,
            detail=f"Unable to decode the uploaded image: {exc}",
        )

    # Standardise colour space: BGR → RGB
    img_rgb: np.ndarray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    # Dynamic resize to 224 × 224 (model input size)
    target_size: int = 224
    img_resized: np.ndarray = cv2.resize(
        img_rgb, (target_size, target_size), interpolation=cv2.INTER_LINEAR
    )

    # Convert to PyTorch float tensor (C, H, W) in [0, 1]
    image_tensor: torch.Tensor = (
        torch.from_numpy(img_resized)
        .permute(2, 0, 1)
        .float()
        .div(255.0)
        .to(_device)
    )

    # =====================================================================
    # 2.  Run inference with torch.profiler
    # =====================================================================
    profiling_output: str = ""
    patch_scores: List[float] = []
    attention_matrix: List[List[float]] = []
    global_score: float = 0.0

    # Configure profiler activities based on available device
    activities: List[Any] = [torch.profiler.ProfilerActivity.CPU]
    if torch.cuda.is_available():
        activities.append(torch.profiler.ProfilerActivity.CUDA)

    with torch.profiler.profile(
        activities=activities,
        record_shapes=True,
        profile_memory=True,
        with_stack=False,
    ) as prof:
        # ---- Model 1: UnivFD patch-based scanning -------------------------
        logger.info("Running UnivFD patch-based scanning …")
        patch_scores = _univfd.generate_patch_scores(
            image_tensor=image_tensor,
            patch_size=64,
            stride=32,
        )

        # ---- Model 2: Explainable ViT attention map ------------------------
        logger.info("Running ExplainableViT attention map extraction …")
        attn_map: np.ndarray = _explainable.generate_attention_map(
            image_tensor=image_tensor,
            output_size=(target_size, target_size),
        )
        attention_matrix = attn_map.tolist()

        # ---- Global score (from ExplainableViT classifier) -----------------
        global_score = _explainable.predict_score(image_tensor)

    # =====================================================================
    # 3.  Format profiling summary
    # =====================================================================
    profiling_output = prof.key_averages().table(
        sort_by="cpu_time_total",
        row_limit=20,
    )
    logger.info(f"Profiling summary:\n{profiling_output}")

    # =====================================================================
    # 4.  Return structured response
    # =====================================================================
    return AnalysisResult(
        global_score=global_score,
        patch_scores=patch_scores,
        attention_matrix=attention_matrix,
        profiling_summary=profiling_output,
    )
