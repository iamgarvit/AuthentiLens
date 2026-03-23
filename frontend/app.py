"""
============================================================================
ApertureAuthentiLens — Streamlit Frontend
============================================================================
Interactive web UI that allows users to:

    1. Upload an image (drag-and-drop or file picker).
    2. Send it to the FastAPI backend for forensic analysis.
    3. Visualise results:
       a) Heatmap Overlay  — attention map alpha-composited over the image.
       b) Intensity Histogram — distribution of patch confidence scores.
       c) Global Authenticity Score — colour-coded verdict.

    4. Confidence Threshold slider to dynamically filter the heatmap.

The backend URL is configured via the BACKEND_URL environment variable
(defaults to http://backend:8000 for Docker networking).
============================================================================
"""

from __future__ import annotations

import os
from typing import Any, Dict, List

import cv2
import matplotlib.pyplot as plt
import numpy as np
import requests
import streamlit as st
from PIL import Image

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
BACKEND_URL: str = os.getenv("BACKEND_URL", "http://backend:8000")
ANALYZE_ENDPOINT: str = f"{BACKEND_URL}/analyze"

# ---------------------------------------------------------------------------
# Page configuration
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="ApertureAuthentiLens",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================================
# Custom CSS for a premium, dark-themed look
# ============================================================================
st.markdown(
    """
    <style>
    /* ---------- Global ---------- */
    .stApp {
        background: linear-gradient(135deg, #0f0c29, #1a1a2e, #16213e);
        color: #e0e0e0;
    }

    /* ---------- Header ---------- */
    .main-header {
        text-align: center;
        padding: 1.5rem 0;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2.8rem;
        font-weight: 800;
        letter-spacing: -0.5px;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        text-align: center;
        color: #a0a0b0;
        font-size: 1.1rem;
        margin-bottom: 2rem;
    }

    /* ---------- Score card ---------- */
    .score-card {
        text-align: center;
        padding: 1.5rem;
        border-radius: 16px;
        margin: 1rem 0;
        backdrop-filter: blur(10px);
    }
    .score-card.safe {
        background: rgba(0, 200, 83, 0.12);
        border: 1px solid rgba(0, 200, 83, 0.35);
    }
    .score-card.suspicious {
        background: rgba(255, 193, 7, 0.12);
        border: 1px solid rgba(255, 193, 7, 0.35);
    }
    .score-card.manipulated {
        background: rgba(255, 61, 0, 0.12);
        border: 1px solid rgba(255, 61, 0, 0.35);
    }
    .score-value {
        font-size: 3rem;
        font-weight: 700;
    }
    .score-label {
        font-size: 1rem;
        opacity: 0.8;
        margin-top: 0.3rem;
    }

    /* ---------- Section headers ---------- */
    .section-title {
        font-size: 1.3rem;
        font-weight: 600;
        color: #b0b0d0;
        border-bottom: 2px solid rgba(102, 126, 234, 0.3);
        padding-bottom: 0.5rem;
        margin: 1.5rem 0 1rem 0;
    }

    /* ---------- Profiling expander ---------- */
    .profiling-text {
        font-family: 'Courier New', monospace;
        font-size: 0.75rem;
        white-space: pre;
        overflow-x: auto;
        background: rgba(0,0,0,0.3);
        padding: 1rem;
        border-radius: 8px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================================
# Helper functions
# ============================================================================

def _build_heatmap_overlay(
    original_rgb: np.ndarray,
    attention_matrix: np.ndarray,
    threshold: float,
    alpha: float = 0.55,
) -> np.ndarray:
    """
    Build a heatmap overlay by mapping the attention matrix to the INFERNO
    colourmap and alpha-compositing it over the original image.

    Parameters
    ----------
    original_rgb : np.ndarray
        Original image in RGB, uint8, shape ``(H, W, 3)``.
    attention_matrix : np.ndarray
        2-D float attention map in ``[0, 1]``, shape ``(H', W')``.
    threshold : float
        Confidence threshold — pixels below this value are suppressed.
    alpha : float
        Blending factor for the overlay (0 = invisible, 1 = opaque).

    Returns
    -------
    np.ndarray
        Composited image in RGB, uint8, shape ``(H, W, 3)``.
    """
    h, w = original_rgb.shape[:2]

    # Resize attention map to match original image dimensions
    attn_resized: np.ndarray = cv2.resize(
        attention_matrix, (w, h), interpolation=cv2.INTER_LINEAR
    )

    # Apply threshold — zero out low-confidence regions
    attn_thresholded: np.ndarray = np.where(
        attn_resized >= threshold, attn_resized, 0.0
    ).astype(np.float32)

    # Normalize to [0, 255] for colourmap application
    attn_uint8: np.ndarray = (attn_thresholded * 255).clip(0, 255).astype(np.uint8)

    # Apply INFERNO colourmap (OpenCV uses BGR internally)
    heatmap_bgr: np.ndarray = cv2.applyColorMap(attn_uint8, cv2.COLORMAP_INFERNO)
    heatmap_rgb: np.ndarray = cv2.cvtColor(heatmap_bgr, cv2.COLOR_BGR2RGB)

    # Alpha-composite: overlay = α·heatmap + (1−α)·original
    overlay: np.ndarray = cv2.addWeighted(
        heatmap_rgb, alpha, original_rgb, 1.0 - alpha, 0
    )

    return overlay


def _plot_histogram(patch_scores: List[float]) -> plt.Figure:
    """
    Plot a histogram of patch confidence scores.

    Parameters
    ----------
    patch_scores : List[float]
        Per-patch fake-confidence scores in ``[0, 1]``.

    Returns
    -------
    plt.Figure
        Matplotlib Figure for Streamlit rendering.
    """
    fig, ax = plt.subplots(figsize=(8, 4))

    # Style the plot for dark theme
    fig.patch.set_facecolor("#0f0c29")
    ax.set_facecolor("#1a1a2e")

    ax.hist(
        patch_scores,
        bins=30,
        range=(0.0, 1.0),
        color="#667eea",
        edgecolor="#a78bfa",
        alpha=0.85,
        linewidth=0.8,
    )

    ax.set_xlabel("Confidence Score", color="#b0b0d0", fontsize=11)
    ax.set_ylabel("Frequency (Patches)", color="#b0b0d0", fontsize=11)
    ax.set_title(
        "Patch-Level Manipulation Confidence Distribution",
        color="#e0e0e0",
        fontsize=13,
        fontweight="bold",
        pad=12,
    )

    ax.tick_params(colors="#a0a0b0")
    ax.spines["bottom"].set_color("#404060")
    ax.spines["left"].set_color("#404060")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", alpha=0.15, color="#606080")

    fig.tight_layout()
    return fig


def _get_verdict(score: float) -> tuple[str, str, str]:
    """
    Map a global score to a human-readable verdict, CSS class, and emoji.

    Returns
    -------
    tuple[str, str, str]
        (verdict_text, css_class, emoji)
    """
    if score < 0.35:
        return "Likely Authentic", "safe", "✅"
    elif score < 0.65:
        return "Suspicious — Review Recommended", "suspicious", "⚠️"
    else:
        return "Likely Manipulated", "manipulated", "🚨"


# ============================================================================
# Main UI
# ============================================================================

def main() -> None:
    """Entry point for the Streamlit application."""

    # ----- Header -----------------------------------------------------------
    st.markdown('<h1 class="main-header">🔬 ApertureAuthentiLens</h1>', unsafe_allow_html=True)
    st.markdown(
        '<p class="sub-header">'
        "AI-Powered Visual Forensics — Localise Manipulations in Images"
        "</p>",
        unsafe_allow_html=True,
    )

    # ----- Sidebar controls -------------------------------------------------
    with st.sidebar:
        st.markdown("### ⚙️ Controls")
        threshold: float = st.slider(
            "Confidence Threshold",
            min_value=0.0,
            max_value=1.0,
            value=0.25,
            step=0.01,
            help=(
                "Filter out heatmap regions below this confidence level. "
                "Higher values show only the most suspicious areas."
            ),
        )
        alpha: float = st.slider(
            "Heatmap Opacity",
            min_value=0.0,
            max_value=1.0,
            value=0.55,
            step=0.05,
            help="Controls how opaque the heatmap overlay appears.",
        )
        st.markdown("---")
        st.markdown(
            "**ApertureAuthentiLens** v1.0  \n"
            "Zero-shot forensic analysis powered by  \n"
            "CLIP ViT + EfficientNet-Transformer."
        )

    # ----- File uploader ----------------------------------------------------
    uploaded_file = st.file_uploader(
        "Upload an image for forensic analysis",
        type=["jpg", "jpeg", "png", "bmp", "webp"],
        help="Drag & drop or click to browse. Supported: JPG, PNG, BMP, WebP.",
    )

    if uploaded_file is None:
        # Show placeholder when no image is uploaded
        st.markdown("---")
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.markdown(
                """
                <div style="text-align:center; padding: 4rem 2rem;
                            border: 2px dashed rgba(102,126,234,0.3);
                            border-radius: 16px; margin: 2rem 0;">
                    <p style="font-size: 3rem; margin-bottom: 0.5rem;">📤</p>
                    <p style="color: #b0b0d0; font-size: 1.1rem;">
                        Upload an image above to begin forensic analysis
                    </p>
                    <p style="color: #707090; font-size: 0.85rem; margin-top: 0.5rem;">
                        The system will scan for AI-generated manipulations
                        using patch-based confidence scoring and attention mapping.
                    </p>
                </div>
                """,
                unsafe_allow_html=True,
            )
        return

    # =====================================================================
    # Image uploaded — run analysis
    # =====================================================================

    # Display the original uploaded image
    pil_image: Image.Image = Image.open(uploaded_file)
    original_rgb: np.ndarray = np.array(pil_image.convert("RGB"))

    # ----- Send to backend --------------------------------------------------
    with st.spinner("🔍 Analysing image — running forensic pipeline …"):
        try:
            uploaded_file.seek(0)  # Reset file pointer
            response = requests.post(
                ANALYZE_ENDPOINT,
                files={"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)},
                timeout=300,
            )
            response.raise_for_status()
        except requests.exceptions.ConnectionError:
            st.error(
                "⚠️ Cannot connect to the backend service. "
                "Ensure the backend container is running."
            )
            return
        except requests.exceptions.Timeout:
            st.error("⏱️ Backend request timed out. The image may be too large.")
            return
        except requests.exceptions.HTTPError as exc:
            st.error(f"❌ Backend returned an error: {exc.response.text}")
            return

    result: Dict[str, Any] = response.json()

    # =====================================================================
    # Display results
    # =====================================================================

    # ----- Global Score Card ------------------------------------------------
    global_score: float = result["global_score"]
    verdict_text, css_class, emoji = _get_verdict(global_score)

    st.markdown(
        f"""
        <div class="score-card {css_class}">
            <div class="score-value">{emoji} {global_score:.1%}</div>
            <div class="score-label">{verdict_text}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ----- Two-column layout: Original + Heatmap ---------------------------
    st.markdown('<div class="section-title">🖼️ Visual Analysis</div>', unsafe_allow_html=True)

    col_orig, col_heat = st.columns(2)

    with col_orig:
        st.markdown("**Original Image**")
        st.image(original_rgb, use_container_width=True)

    with col_heat:
        st.markdown("**Manipulation Heatmap Overlay**")
        attention_matrix: np.ndarray = np.array(
            result["attention_matrix"], dtype=np.float32
        )
        overlay: np.ndarray = _build_heatmap_overlay(
            original_rgb=original_rgb,
            attention_matrix=attention_matrix,
            threshold=threshold,
            alpha=alpha,
        )
        st.image(overlay, use_container_width=True)

    # ----- Histogram --------------------------------------------------------
    st.markdown(
        '<div class="section-title">📊 Patch Confidence Distribution</div>',
        unsafe_allow_html=True,
    )
    patch_scores: List[float] = result["patch_scores"]

    if patch_scores:
        fig: plt.Figure = _plot_histogram(patch_scores)
        st.pyplot(fig)
        plt.close(fig)

        # Summary stats
        scores_arr = np.array(patch_scores)
        scol1, scol2, scol3, scol4 = st.columns(4)
        scol1.metric("Patches Analysed", len(patch_scores))
        scol2.metric("Mean Score", f"{scores_arr.mean():.3f}")
        scol3.metric("Max Score", f"{scores_arr.max():.3f}")
        scol4.metric(
            "Suspicious Patches",
            int((scores_arr >= threshold).sum()),
        )
    else:
        st.info("No patch scores were generated.")

    # ----- Profiling --------------------------------------------------------
    with st.expander("🔧 Inference Profiling Details"):
        st.markdown(
            f'<div class="profiling-text">{result["profiling_summary"]}</div>',
            unsafe_allow_html=True,
        )


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    main()
