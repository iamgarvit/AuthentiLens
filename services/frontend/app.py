import os
import cv2
import numpy as np
import requests
import streamlit as st
from PIL import Image

BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")
ANALYZE_ENDPOINT = f"{BACKEND_URL}/analyze"

st.set_page_config(page_title="AI Forensics Pipeline", layout="wide")

st.markdown("## 🔬 Real or Fake Image Detector")
st.markdown("Upload an image to classify. If fake, we will generate a segmentation mask showing tampered areas!")

uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    pil_image = Image.open(uploaded_file)
    original_rgb = np.array(pil_image.convert("RGB"))

    with st.spinner("Analyzing with EfficientNet and DeepLabV3..."):
        try:
            uploaded_file.seek(0)
            res = requests.post(ANALYZE_ENDPOINT, files={"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)})
            res.raise_for_status()
            result = res.json()
        except requests.exceptions.RequestException as e:
            st.error(f"Backend error: {e}")
            st.stop()

    is_fake = result["is_fake"]
    fake_prob = result["fake_prob"]
    mask = result.get("mask", [])
    clf_time = result.get("clf_time", 0.0)
    seg_time = result.get("seg_time", 0.0)
    total_time = result.get("total_time", 0.0)
    cuda_alloc_mb = result.get("cuda_alloc_mb", 0.0)
    cuda_res_mb = result.get("cuda_res_mb", 0.0)
    flagged_pct = result.get("flagged_pct")

    if not result.get("segmentation_available", True):
        st.info("Segmentation model not loaded — showing classification only")

    if not is_fake:
        st.success(f"✅ Image is NOT FAKE! (Confidence it's real: {(1-fake_prob)*100:.1f}%)")
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.image(original_rgb, use_container_width=True)
    else:
        st.warning(f"🚨 Image is FAKE! (Confidence: {fake_prob*100:.1f}%)")
        if result.get("decision_source") == "segmentation_override":
            st.caption(f"The classifier predicted REAL, but segmentation flagged {flagged_pct:.1f}% of pixels "
                       "as inpainted, so the pipeline overrides the verdict to FAKE.")
        col1, col2 = st.columns(2)
        with col1:
            st.image(original_rgb, caption="Original Image", use_container_width=True)
        with col2:
            if mask:
                # mask is a probability map [0, 1] from the model
                mask_np = np.array(mask, dtype=np.float32)
                h, w = original_rgb.shape[:2]
                # Resize probability map back to original image size
                mask_resized = cv2.resize(mask_np, (w, h))

                # Apply thresholding to probability map (matches notebook's predict_single)
                threshold = 0.5
                bool_mask = mask_resized > threshold

                # Create heatmap overlay from probability map (before thresholding)
                # This shows the continuous confidence, not just binary regions
                heatmap = cv2.applyColorMap((mask_resized * 255).astype(np.uint8), cv2.COLORMAP_INFERNO)
                heatmap_rgb = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
                
                # Composite: blend only the regions above threshold
                overlay = original_rgb.copy()
                blended = cv2.addWeighted(heatmap_rgb, 0.6, original_rgb, 0.4, 0)
                overlay[bool_mask] = blended[bool_mask]
                
                st.image(overlay, caption="DeepLabV3 Manipulation Mask", use_container_width=True)

    st.markdown("---")
    st.markdown("### ⏱️ Inference Compute & Memory Metrics")
    m1, m2, m3 = st.columns(3)
    m1.metric("EfficientNet-B0 (Classify)", f"{clf_time*1000:.1f} ms")
    if is_fake:
        m2.metric("DeepLabV3 (Segment)", f"{seg_time*1000:.1f} ms")
    else:
        m2.metric("DeepLabV3 (Segment)", "Skipped")
    m3.metric("Total End-to-End Pipeline", f"{total_time*1000:.1f} ms")
    
    st.markdown("#### ⚡ Hardware Utilization (CUDA)")
    m4, m5 = st.columns(2)
    if cuda_alloc_mb > 0:
        m4.metric("Peak CUDA VRAM Allocated", f"{cuda_alloc_mb:.1f} MB")
        m5.metric("Peak CUDA VRAM Reserved", f"{cuda_res_mb:.1f} MB")
    else:
        m4.metric("Peak CUDA VRAM Allocated", "N/A (Running on CPU)")
        m5.metric("Peak CUDA VRAM Reserved", "N/A (Running on CPU)")

