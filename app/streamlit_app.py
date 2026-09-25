# =============================================================================
# AuthentiLens - Streamlit Demo (Enhanced)
# =============================================================================
# Combines segmentation (DeepLabV3Plus) and classification (ResNet + EfficientNet)
# - Segmentation: Detects AI-inpainted regions
# - Classification: Classifies the overall image (e.g., FAKE/REAL)
#
# Run:
#   streamlit run app/streamlit_app.py --server.port 8501   (from the repo root)
#
# Dependencies:
#   pip install streamlit segmentation-models-pytorch torch torchvision pillow numpy opencv-python
# =============================================================================

import io
import os
import json
import torch
import numpy as np
import streamlit as st
import time
import psutil
from PIL import Image
import matplotlib.colors as mcolors
from datetime import datetime
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repo root
sys.path.insert(0, str(Path(__file__).resolve().parent))      # this directory
from authentilens import models as M
try:
    import cv2
    from utils_noiseprint import generate_noiseprint_like_from_pil
    NOISEPRINT_AVAILABLE = True
    NOISEPRINT_IMPORT_ERROR = ""
except Exception as e:
    cv2 = None
    generate_noiseprint_like_from_pil = None
    NOISEPRINT_AVAILABLE = False
    NOISEPRINT_IMPORT_ERROR = str(e)
# =============================================================================
# CONFIG
# =============================================================================
# Architectures, checkpoint loading, inference and the pipeline decision rule
# all live in authentilens/models.py, so this demo, the FastAPI backend and the
# Hugging Face Space run the same code.

DEVICE = M.get_device()
CLASS_NAMES = M.CLASS_NAMES

# Only offer models whose weights are actually on disk. Missing checkpoints and
# un-pulled Git LFS pointers are hidden rather than listed as broken options.
AVAILABLE_CLASSIFIERS = M.available_classifiers()
AVAILABLE_SEGMENTERS = M.available_segmenters()


# =============================================================================
# SESSION STATE INITIALIZATION
# =============================================================================

if 'results_history' not in st.session_state:
    st.session_state.results_history = []


# =============================================================================
# MODEL LOADING (cached across reruns)
# =============================================================================

@st.cache_resource(show_spinner='Loading segmentation models...')
def load_selected_segmentation_models(selected_model_names: tuple):
    """Load the selected segmenters. Labels without weights are skipped."""
    models = {}
    for label in selected_model_names:
        try:
            models[label] = M.load_segmenter(label, DEVICE)
        except FileNotFoundError:
            continue
    return models


@st.cache_resource(show_spinner='Loading classification models...')
def load_selected_classifier_models(selected_model_names: tuple):
    """Load the selected classifiers. Labels without weights are skipped."""
    models = {}
    for label in selected_model_names:
        try:
            models[label] = M.load_classifier(label, DEVICE)
        except FileNotFoundError:
            continue
    return models


# =============================================================================
# INFERENCE
# =============================================================================

def run_segmentation_inference(model, pil_image: Image.Image, threshold: float) -> tuple:
    """Returns (prob_map [H, W] float32 in [0, 1], binary_map [H, W] uint8, flagged %)."""
    return M.segment(model, pil_image, threshold, DEVICE)


def run_classification_inference(models: dict, pil_image: Image.Image) -> dict:
    """Run every loaded classifier on the image and time each one."""
    results = {}
    for model_name, model in models.items():
        start_t = time.time()
        result = M.classify(model, pil_image, DEVICE)
        result['inference_time'] = time.time() - start_t
        result['confidence'] = result['confidence_pct'] / 100.0
        results[model_name] = result
    return results


# =============================================================================
# NOISEPRINT INFERENCE
# =============================================================================

def run_noiseprint_inference(pil_image: Image.Image) -> dict:
    """
    Run a Noiseprint-like residual analysis and return display-ready artifacts.
    """
    noiseprint_map = generate_noiseprint_like_from_pil(pil_image)
    residual_uint8 = (noiseprint_map * 255).astype(np.uint8)

    heatmap_bgr = cv2.applyColorMap(residual_uint8, cv2.COLORMAP_INFERNO)
    heatmap_rgb = cv2.cvtColor(heatmap_bgr, cv2.COLOR_BGR2RGB)
    original_rgb = np.array(pil_image.convert('RGB'))
    overlay_rgb = cv2.addWeighted(original_rgb, 0.65, heatmap_rgb, 0.35, 0)

    mean_score = float(noiseprint_map.mean())
    std_score = float(noiseprint_map.std())
    max_score = float(noiseprint_map.max())
    high_activity_ratio = float((noiseprint_map > 0.60).mean())

    return {
        'residual_map': noiseprint_map,
        'residual_uint8': residual_uint8,
        'heatmap_rgb': heatmap_rgb,
        'overlay_rgb': overlay_rgb,
        'mean_score': mean_score,
        'std_score': std_score,
        'max_score': max_score,
        'high_activity_ratio': high_activity_ratio,
        'fake_score': (0.50 * mean_score) + (0.30 * std_score) + (0.20 * high_activity_ratio),
    }


# =============================================================================
# VISUALIZATION HELPERS
# =============================================================================

def make_overlay(pil_image: Image.Image, prob_map: np.ndarray, opacity: float, threshold: float) -> Image.Image:
    """Composite a realistic probability heatmap over original image."""
    from scipy.ndimage import gaussian_filter

    # --- 1. Smooth the raw probability map ---
    smoothed = gaussian_filter(prob_map.astype(np.float32), sigma=6)

    # --- 2. Normalize only the above-threshold region ---
    above = smoothed[smoothed >= threshold]
    if above.size > 0:
        p_low, p_high = np.percentile(above, 2), np.percentile(above, 98)
    else:
        p_low, p_high = threshold, 1.0

    normalized = np.clip((smoothed - p_low) / (p_high - p_low + 1e-8), 0, 1)

    # --- 3. Realistic heatmap colormap (deep blue -> cyan -> green -> yellow -> red) ---
    cmap = mcolors.LinearSegmentedColormap.from_list(
        'realistic_heat', [
            (0.00, '#0000ff'),  # deep blue  - very low
            (0.25, '#00cfff'),  # cyan        - low-mid
            (0.50, '#00ff88'),  # green       - mid
            (0.75, '#ffdd00'),  # yellow      - high
            (1.00, '#ff2200'),  # hot red     - very high
        ]
    )

    rgba = cmap(normalized).astype(np.float32)  # (H, W, 4)

    # --- 4. Soft alpha: smooth sigmoid falloff around threshold ---
    # Instead of a hard cutoff, ramp smoothly
    sharpness = 12.0  # higher = sharper edge at threshold
    soft_alpha = 1.0 / (1.0 + np.exp(-sharpness * (smoothed - threshold)))

    # Weight alpha by probability magnitude so bright spots = more opaque
    alpha = soft_alpha * normalized * opacity
    alpha = gaussian_filter(alpha, sigma=3)  # feather the alpha edges
    alpha = np.clip(alpha, 0.0, 1.0).astype(np.float32)

    rgba[:, :, 3] = alpha

    # --- 5. Composite ---
    heat_rgba = Image.fromarray((rgba * 255).astype(np.uint8), mode='RGBA')
    base = pil_image.convert('RGBA')
    composite = Image.alpha_composite(base, heat_rgba)
    return composite.convert('RGB')


def get_prob_map_image(prob_map: np.ndarray) -> Image.Image:
    """Convert probability map to realistic colored heatmap image."""
    from scipy.ndimage import gaussian_filter

    smoothed = gaussian_filter(prob_map.astype(np.float32), sigma=6)

    p_low, p_high = np.percentile(smoothed, 2), np.percentile(smoothed, 98)
    normalized = np.clip((smoothed - p_low) / (p_high - p_low + 1e-8), 0, 1)

    cmap = mcolors.LinearSegmentedColormap.from_list(
        'realistic_heat', [
            (0.00, '#0000ff'),
            (0.25, '#00cfff'),
            (0.50, '#00ff88'),
            (0.75, '#ffdd00'),
            (1.00, '#ff2200'),
        ]
    )

    prob_rgb = (cmap(normalized)[:, :, :3] * 255).astype(np.uint8)
    return Image.fromarray(prob_rgb)


def image_to_bytes(image: Image.Image, format: str = 'PNG') -> bytes:
    """Convert PIL Image to bytes."""
    buffer = io.BytesIO()
    image.save(buffer, format=format)
    return buffer.getvalue()


def numpy_to_uint8_image(arr: np.ndarray) -> Image.Image:
    """Convert numpy array to PIL Image."""
    return Image.fromarray((arr * 255).astype(np.uint8) if arr.max() <= 1.0 else arr.astype(np.uint8))


# =============================================================================
# PAGE CONFIG & STYLING
# =============================================================================

st.set_page_config(
    page_title='AuthentiLens',
    layout='wide',
    initial_sidebar_state='expanded'
)

st.markdown("""
<style>
    .block-container { padding-top: 1.5rem; padding-bottom: 1rem; }
    .stImage img { border-radius: 8px; box-shadow: 0 2px 8px rgba(0,0,0,0.1); }
    h1 { font-size: 2.2rem !important; margin-bottom: 0.5rem; }
    h2 { font-size: 1.5rem !important; margin-top: 1.5rem; }
    .metric-card { 
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        box-shadow: 0 4px 12px rgba(0,0,0,0.15);
    }
    .metric-value { font-size: 2rem; font-weight: bold; }
    .metric-label { font-size: 0.95rem; opacity: 0.9; }
    .confidence-high { color: #00d084; font-weight: bold; }
    .confidence-med { color: #ffa500; font-weight: bold; }
    .confidence-low { color: #ff6b6b; font-weight: bold; }
    .download-btn { margin-top: 0.5rem; }
</style>
""", unsafe_allow_html=True)

st.title('AuthentiLens')
st.markdown(
    '**Detects AI-inpainted images and highlights the regions that were edited.** '
    'A classifier decides FAKE vs REAL, and a segmentation network localises the '
    'inpainted pixels and can overturn the classifier when it misses a partial edit.'
)

st.info(
    '**Scope.** AuthentiLens targets **diffusion inpainting on photoreal images** '
    '(Stable Diffusion 2 / SDXL edits of real photos). It is not trained for GAN '
    'images, fully synthetic images or artwork, and will not be reliable on them. '
    'The pipeline was evaluated on a fake-only test set, so its **false-positive '
    'rate on real photos has not been measured yet** — treat every result as '
    'indicative, not as proof.',
    icon=':material/info:',
)

# =============================================================================
# SIDEBAR
# =============================================================================

if not AVAILABLE_CLASSIFIERS:
    st.error(
        'No classifier weights found. Run `git lfs pull` (see the README) so that '
        'at least one checkpoint under `checkpoints/classification/` is available.'
    )
    st.stop()

with st.sidebar:
    st.header('Pipeline')
    st.caption(
        'Default: the best combination in `results/pipeline/` — EfficientNet-B0 '
        '(balanced, lr 2.5e-5) + DeepLabV3+, pixel threshold 0.7, override 30%. '
        'That pairing reached 96.7% combined TPR on the SD2-FR test set.'
    )

    primary_classifier = st.selectbox(
        'Classifier',
        options=AVAILABLE_CLASSIFIERS,
        index=AVAILABLE_CLASSIFIERS.index(M.DEFAULT_CLASSIFIER)
        if M.DEFAULT_CLASSIFIER in AVAILABLE_CLASSIFIERS else 0,
        help='Decides FAKE vs REAL on the whole image, resized to 224x224.',
    )

    segmenter_options = ['None (classification only)'] + AVAILABLE_SEGMENTERS
    primary_segmenter = st.selectbox(
        'Segmenter',
        options=segmenter_options,
        index=segmenter_options.index(M.DEFAULT_SEGMENTER)
        if M.DEFAULT_SEGMENTER in segmenter_options else 0,
        help='Predicts P(inpainted) per pixel at 512x512 and can override a REAL call.',
    )
    if primary_segmenter == 'None (classification only)':
        primary_segmenter = None

    threshold = st.slider(
        'Pixel threshold',
        min_value=0.10, max_value=0.90, value=M.PIXEL_THRESHOLD, step=0.05,
        help='A pixel counts as inpainted when P(inpainted) is above this value.',
    )

    override_pct = st.slider(
        'Override if flagged area > (%)',
        min_value=0.0, max_value=100.0, value=M.OVERRIDE_PCT, step=5.0,
        help='When the classifier says REAL but more than this share of pixels is '
             'flagged, the pipeline outputs FAKE.',
    )

    st.divider()
    st.subheader('Compare with other models')
    extra_classifiers = st.multiselect(
        'Additional classifiers',
        options=[c for c in AVAILABLE_CLASSIFIERS if c != primary_classifier],
        default=[],
        help='Shown alongside the pipeline decision; they do not change it.',
    )
    extra_segmenters = st.multiselect(
        'Additional segmenters',
        options=[s for s in AVAILABLE_SEGMENTERS if s != primary_segmenter],
        default=[],
        help='Shown alongside the pipeline decision; they do not change it.',
    )

    hidden_classifiers = [c for c in M.CLASSIFIER_MODELS if c not in AVAILABLE_CLASSIFIERS]
    hidden_segmenters = [s for s in M.SEGMENTATION_MODELS if s not in AVAILABLE_SEGMENTERS]
    if hidden_classifiers or hidden_segmenters:
        with st.expander('Models without weights on this machine'):
            for name in hidden_classifiers + hidden_segmenters:
                st.caption(f'- {name}')
            st.caption('Fetch them with `git lfs pull`, then reload this page.')

    st.divider()
    st.subheader('Additional analysis')
    run_noiseprint_model = st.checkbox(
        'Run Noiseprint model',
        value=False,
        disabled=not NOISEPRINT_AVAILABLE,
        help='A separate Noiseprint-like residual analysis. Not part of the '
             'evaluated pipeline and not benchmarked.',
    )
    noiseprint_fake_threshold = st.slider(
        'Noiseprint FAKE threshold',
        min_value=0.05, max_value=0.50, value=0.18, step=0.01,
        disabled=not NOISEPRINT_AVAILABLE,
        help='If the Noiseprint score is above this value, its prediction is FAKE.',
    )
    if not NOISEPRINT_AVAILABLE:
        st.caption(f'Noiseprint unavailable: {NOISEPRINT_IMPORT_ERROR}')

    st.divider()
    st.subheader('Display')
    opacity = st.slider(
        'Heatmap opacity',
        min_value=0.20, max_value=1.00, value=0.6, step=0.05,
        help='Heatmap blend intensity.',
    )
    show_seg_raw = st.checkbox('Show segmentation probability map', value=True)
    show_seg_binary = st.checkbox('Show segmentation binary mask', value=True)
    show_seg_stats = st.checkbox('Show segmentation statistics', value=True)
    show_download_options = st.checkbox('Show download options', value=True)

    st.divider()
    st.markdown(f'**Device:** `{DEVICE}`')
    st.markdown("""
**Heatmap legend**

Blue/green = low inpainting probability
Yellow/red = high inpainting probability
Transparent = below threshold
    """)

selected_classifier_names = [primary_classifier] + extra_classifiers
selected_segmentation_names = ([primary_segmenter] if primary_segmenter else []) + extra_segmenters

# =============================================================================
# MODEL LOADING
# =============================================================================

try:
    seg_models = load_selected_segmentation_models(tuple(selected_segmentation_names))
except Exception as e:
    st.error(f'Failed to load segmentation models: {e}')
    st.stop()

try:
    classifier_models = load_selected_classifier_models(tuple(selected_classifier_names))
    st.session_state.classifier_models = classifier_models
except Exception as e:
    st.error(f'Failed to load classification models: {e}')
    st.stop()

if not classifier_models:
    st.error(f'Could not load the classifier "{primary_classifier}".')
    st.stop()

if primary_segmenter and primary_segmenter not in seg_models:
    st.warning(
        f'Could not load the segmenter "{primary_segmenter}" — running in '
        'classification-only mode, with no localisation and no override.'
    )
    primary_segmenter = None
elif not primary_segmenter:
    st.warning(
        'No segmenter selected: running in classification-only mode, with no '
        'localisation and no override.'
    )

st.sidebar.success(
    f'Loaded {len(classifier_models)} classifier(s) and {len(seg_models)} segmenter(s)'
)
# FILE UPLOADER
# =============================================================================

uploaded_files = st.file_uploader(
    'Upload one or more images',
    type=['png', 'jpg', 'jpeg', 'webp', 'bmp'],
    accept_multiple_files=True,
)

if not uploaded_files:
    st.info('Upload an image to analyze')
    st.stop()

# =============================================================================
# PROCESS EACH IMAGE
# =============================================================================

for uploaded_file in uploaded_files:
    st.divider()
    st.subheader(f'File: {uploaded_file.name}')
    try:
        pil_image = Image.open(uploaded_file).convert('RGB')
        W, H = pil_image.size
        st.caption(f'Resolution: {W}x{H}px')

        # Prepare CPU/RAM baseline
        process = psutil.Process(os.getpid())
        ram_before_mb = process.memory_info().rss / (1024 ** 2)
        psutil.cpu_percent(interval=None)  # Prime CPU percent

        # Run inference
        with st.spinner('Running inference...'):
            overall_start_time = time.time()
            segmentation_results = {}
            for name, model in seg_models.items():
                m_start = time.time()
                prob_map, binary_map, flagged_pct = run_segmentation_inference(model, pil_image, threshold)
                overlay = make_overlay(pil_image, prob_map, opacity, threshold)
                m_time = time.time() - m_start
                segmentation_results[name] = {
                    'prob_map': prob_map,
                    'binary_map': binary_map,
                    'overlay': overlay,
                    'flagged_pct': flagged_pct,
                    'inference_time': m_time
                }

            if classifier_models:
                classification_results = run_classification_inference(classifier_models, pil_image)
            else:
                classification_results = {}

            noiseprint_results = None
            if run_noiseprint_model and NOISEPRINT_AVAILABLE:
                n_start = time.time()
                noiseprint_results = run_noiseprint_inference(pil_image)
                noiseprint_results['inference_time'] = time.time() - n_start
                noiseprint_results['threshold'] = noiseprint_fake_threshold

                fake_score = noiseprint_results['fake_score']
                pred_class = 'FAKE' if fake_score >= noiseprint_fake_threshold else 'REAL'
                if pred_class == 'FAKE':
                    conf = (fake_score - noiseprint_fake_threshold) / max(1.0 - noiseprint_fake_threshold, 1e-8)
                else:
                    conf = (noiseprint_fake_threshold - fake_score) / max(noiseprint_fake_threshold, 1e-8)

                noiseprint_results['prediction'] = pred_class
                noiseprint_results['confidence_pct'] = float(np.clip(conf, 0.0, 1.0) * 100.0)

            overall_time = time.time() - overall_start_time

        # Gather System Info
        ram_after_mb = process.memory_info().rss / (1024 ** 2)
        cpu_used = psutil.cpu_percent(interval=None)
        
        system_stats = {
            "overall_time": overall_time,
            "cpu_percent": cpu_used,
            "ram_used_mb": ram_after_mb,
            "ram_jump_mb": ram_after_mb - ram_before_mb,
            "gpu_vram_used_gb": 0.0,
            "gpu_vram_total_gb": 0.0
        }
        if torch.cuda.is_available():
            try:
                system_stats["gpu_vram_used_gb"] = torch.cuda.memory_allocated(DEVICE) / (1024 ** 3)
                system_stats["gpu_vram_total_gb"] = torch.cuda.get_device_properties(DEVICE).total_memory / (1024 ** 3)
            except:
                pass

        # --- PIPELINE DECISION (evaluation/evaluate_pipeline.py rule) ---
        # The primary classifier decides; the primary segmenter can override a
        # REAL call to FAKE when it flags more than override_pct % of pixels.
        primary_clf_result = classification_results[primary_classifier]
        primary_flagged_pct = (
            segmentation_results[primary_segmenter]['flagged_pct']
            if primary_segmenter in segmentation_results else None
        )

        decision = M.pipeline_decision(
            primary_clf_result['is_fake'], primary_flagged_pct, override_pct
        )
        final_prediction = decision['prediction']
        heuristic_reason = decision['reason']

        mean_flagged_pct = (
            sum(r['flagged_pct'] for r in segmentation_results.values()) / len(segmentation_results)
            if segmentation_results else 0.0
        )

        # Store results in session history
        result_entry = {
            'filename': uploaded_file.name,
            'timestamp': datetime.now().isoformat(),
            'pixel_threshold': threshold,
            'override_pct': override_pct,
            'pipeline_classifier': primary_classifier,
            'pipeline_segmenter': primary_segmenter,
            'selected_classifier_models': list(classifier_models.keys()),
            'selected_segmentation_models': list(seg_models.keys()),
            'noiseprint_enabled': bool(run_noiseprint_model and NOISEPRINT_AVAILABLE),
            'noiseprint_summary': {
                'mean_score': noiseprint_results['mean_score'],
                'std_score': noiseprint_results['std_score'],
                'max_score': noiseprint_results['max_score'],
                'high_activity_ratio': noiseprint_results['high_activity_ratio'],
                'fake_score': noiseprint_results['fake_score'],
                'threshold': noiseprint_results['threshold'],
                'prediction': noiseprint_results['prediction'],
                'confidence_pct': noiseprint_results['confidence_pct'],
                'inference_time': noiseprint_results['inference_time'],
            } if noiseprint_results else None,
            'flagged_pct': primary_flagged_pct if primary_flagged_pct is not None else mean_flagged_pct,
            'classification_results': classification_results,
            'final_prediction': final_prediction,
            'decision_source': decision['decision_source'],
            'heuristic_reason': heuristic_reason,
        }
        st.session_state.results_history.append(result_entry)

        # --- DISPLAY FINAL DECISION ---
        st.markdown('### Pipeline verdict')
        decision_color = "#ff4b4b" if final_prediction == "FAKE" else "#00d084"
        pipeline_desc = primary_classifier + (f' + {primary_segmenter}' if primary_segmenter else ' (classification only)')
        badge = (
            'segmentation override' if decision['decision_source'] == 'segmentation_override'
            else 'classifier'
        )
        st.markdown(f"""
        <div style="background: #f0f2f6; padding: 1.5rem; border-radius: 10px; border-left: 6px solid {decision_color}; margin-bottom: 1rem;">
            <div style="font-size: 2rem; font-weight: bold; color: {decision_color}; margin-bottom: 0.5rem;">
                {final_prediction}
            </div>
            <div style="font-size: 1.1rem; color: #444;">
                {heuristic_reason}
            </div>
            <div style="font-size: 0.85rem; color: #666; margin-top: 0.75rem;">
                {pipeline_desc} &middot; pixel threshold {threshold:.2f} &middot;
                override &gt; {override_pct:g}% &middot; decided by: {badge}
            </div>
        </div>
        """, unsafe_allow_html=True)

        # --- CLASSIFICATION RESULTS ---
        if classification_results:
            st.markdown('### Classification Results')
            cols = st.columns(len(classification_results))
            for idx, (model_name, result) in enumerate(classification_results.items()):
                with cols[idx]:
                    conf_pct = result['confidence_pct']

                    # Color coding by confidence
                    if conf_pct >= 80:
                        conf_class = 'confidence-high'
                    elif conf_pct >= 60:
                        conf_class = 'confidence-med'
                    else:
                        conf_class = 'confidence-low'

                    st.markdown(f"""
                    <div style="background: #f0f2f6; padding: 1.5rem; border-radius: 10px; border-left: 4px solid #667eea;">
                        <div style="font-weight: bold; margin-bottom: 0.5rem; font-size: 1.1rem;">{model_name}</div>
                        <div style="font-size: 1.5rem; font-weight: bold; color: #1f77b4; margin-bottom: 0.5rem;">
                            {result['class']}
                        </div>
                        <div class="{conf_class}" style="font-size: 1.3rem; margin-bottom: 0.5rem;">
                            {conf_pct:.1f}%
                        </div>
                        <div style="font-size: 0.85rem; color: #666; margin-bottom: 0.5rem;">
                            {list(result['probabilities'].keys())[0]}: {result['probabilities'][list(result['probabilities'].keys())[0]]:.1f}%<br/>
                            {list(result['probabilities'].keys())[1]}: {result['probabilities'][list(result['probabilities'].keys())[1]]:.1f}%
                        </div>
                        <div style="font-size: 0.85rem; color: #444; border-top: 1px solid #ddd; padding-top: 0.4rem; font-weight: 500;">
                            Inference Time: {result.get('inference_time', 0.0):.3f}s
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

        # --- SEGMENTATION VISUALIZATION ---
        if segmentation_results:
            st.markdown('### Segmentation Analysis')
            st.image(pil_image, caption='Original Image')
            st.write('')

            for model_name, results in segmentation_results.items():
                st.markdown(f"**{model_name} Results:**")
                
                n_cols = 1 + int(show_seg_raw) + int(show_seg_binary)
                cols = st.columns(n_cols)

                col_idx = 0
                with cols[col_idx]:
                    st.image(results['overlay'], caption=f'{model_name} Heatmap', use_container_width=True)
                col_idx += 1

                if show_seg_raw:
                    with cols[col_idx]:
                        st.image(get_prob_map_image(results['prob_map']), caption=f'{model_name} Probability', use_container_width=True)
                    col_idx += 1

                if show_seg_binary:
                    with cols[col_idx]:
                        st.image((results['binary_map'] * 255).astype(np.uint8), caption=f'{model_name} Mask (t={threshold})', use_container_width=True)

                st.divider()
        else:
            st.info('No segmentation model selected for this run.')

        if noiseprint_results:
            st.markdown('### Noiseprint Analysis')
            np_pred = noiseprint_results['prediction']
            np_conf = noiseprint_results['confidence_pct']
            pred_color = '#ff4b4b' if np_pred == 'FAKE' else '#00d084'
            st.markdown(
                f"""
                <div style="background: #f0f2f6; padding: 1.0rem; border-radius: 10px; border-left: 6px solid {pred_color}; margin-bottom: 1rem;">
                    <div style="font-size: 1.2rem; font-weight: 700; color: {pred_color};">
                        Noiseprint Prediction: {np_pred} ({np_conf:.1f}%)
                    </div>
                    <div style="font-size: 0.92rem; color: #444; margin-top: 0.25rem;">
                        Score: {noiseprint_results['fake_score']:.4f} | Threshold: {noiseprint_results['threshold']:.4f}
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

            n1, n2, n3 = st.columns(3)
            with n1:
                st.image(noiseprint_results['residual_uint8'], caption='Noiseprint Residual', use_container_width=True, clamp=True)
            with n2:
                st.image(noiseprint_results['heatmap_rgb'], caption='Noiseprint Heatmap', use_container_width=True)
            with n3:
                st.image(noiseprint_results['overlay_rgb'], caption='Noiseprint Overlay', use_container_width=True)

            nm1, nm2, nm3, nm4, nm5 = st.columns(5)
            nm1.metric('Residual Mean', f"{noiseprint_results['mean_score']:.4f}")
            nm2.metric('Residual Std', f"{noiseprint_results['std_score']:.4f}")
            nm3.metric('Residual Max', f"{noiseprint_results['max_score']:.4f}")
            nm4.metric('Fake Score', f"{noiseprint_results['fake_score']:.4f}")
            nm5.metric('Inference Time', f"{noiseprint_results['inference_time']:.3f}s")
        elif run_noiseprint_model and not NOISEPRINT_AVAILABLE:
            st.warning(f'Noiseprint model unavailable: {NOISEPRINT_IMPORT_ERROR}')

        # --- DOWNLOAD OPTIONS ---
        if show_download_options:
            st.markdown('#### Download Results')
            if noiseprint_results:
                nd1, nd2, nd3 = st.columns(3)
                with nd1:
                    st.download_button(
                        label='Noiseprint Residual',
                        data=image_to_bytes(Image.fromarray(noiseprint_results['residual_uint8'])),
                        file_name=f'{uploaded_file.name.rsplit(".", 1)[0]}_noiseprint_residual.png',
                        mime='image/png'
                    )
                with nd2:
                    st.download_button(
                        label='Noiseprint Heatmap',
                        data=image_to_bytes(Image.fromarray(noiseprint_results['heatmap_rgb'])),
                        file_name=f'{uploaded_file.name.rsplit(".", 1)[0]}_noiseprint_heatmap.png',
                        mime='image/png'
                    )
                with nd3:
                    st.download_button(
                        label='Noiseprint Overlay',
                        data=image_to_bytes(Image.fromarray(noiseprint_results['overlay_rgb'])),
                        file_name=f'{uploaded_file.name.rsplit(".", 1)[0]}_noiseprint_overlay.png',
                        mime='image/png'
                    )

            for model_name, results in segmentation_results.items():
                st.markdown(f"**{model_name}:**")
                dcols = st.columns(3)
                
                with dcols[0]:
                    overlay_bytes = image_to_bytes(results['overlay'])
                    st.download_button(
                        label=f'{model_name} Heatmap',
                        data=overlay_bytes,
                        file_name=f'{uploaded_file.name.rsplit(".", 1)[0]}_{model_name}_heatmap.png',
                        mime='image/png'
                    )
                
                with dcols[1]:
                    binary_img = Image.fromarray((results['binary_map'] * 255).astype(np.uint8))
                    binary_bytes = image_to_bytes(binary_img)
                    st.download_button(
                        label=f'{model_name} Mask',
                        data=binary_bytes,
                        file_name=f'{uploaded_file.name.rsplit(".", 1)[0]}_{model_name}_mask.png',
                        mime='image/png'
                    )
                
                with dcols[2]:
                    prob_img_bytes = image_to_bytes(get_prob_map_image(results['prob_map']))
                    st.download_button(
                        label=f'{model_name} Prob Map',
                        data=prob_img_bytes,
                        file_name=f'{uploaded_file.name.rsplit(".", 1)[0]}_{model_name}_prob.png',
                        mime='image/png'
                    )

        # --- STATISTICS ---
        if show_seg_stats:
            st.markdown('#### Segmentation Statistics')
            for model_name, results in segmentation_results.items():
                st.markdown(f"**{model_name}**")
                mean_prob = float(results['prob_map'].mean())
                max_prob = float(results['prob_map'].max())
                min_prob = float(results['prob_map'].min())
                high_conf = float((results['prob_map'] > 0.75).mean()) * 100

                c1, c2, c3, c4, c5 = st.columns(5)
                c1.metric('Flagged Pixels', f"{results['flagged_pct']:.1f}%", help='% of pixels classified as inpainted')
                c2.metric('Mean Probability', f'{mean_prob:.3f}', help='Average inpainting probability')
                c3.metric('Peak Probability', f'{max_prob:.3f}', help='Maximum inpainting probability')
                c4.metric('Min Probability', f'{min_prob:.3f}', help='Minimum inpainting probability')
                c5.metric('Inference Time', f"{results.get('inference_time', 0.0):.3f}s", help='Time taken for this model to process')
                st.write("")
        
        # --- SYSTEM STATS ---
        st.markdown('#### System Performance')
        sys_c1, sys_c2, sys_c3, sys_c4 = st.columns(4)
        sys_c1.metric("Total Execution Time", f"{system_stats['overall_time']:.2f} s")
        sys_c2.metric("CPU Util During Inference", f"{system_stats['cpu_percent']}%")
        
        ram_delta_str = f"+{system_stats['ram_jump_mb']:.1f} MB (Inference)" if system_stats['ram_jump_mb'] > 0 else f"{system_stats['ram_jump_mb']:.1f} MB (Inference)"
        sys_c3.metric("App RAM Usage", f"{system_stats['ram_used_mb']:.1f} MB", delta=ram_delta_str, delta_color="inverse")
        
        if torch.cuda.is_available():
            sys_c4.metric("GPU VRAM Used", f"{system_stats['gpu_vram_used_gb']:.1f} / {system_stats['gpu_vram_total_gb']:.1f} GB")
        else:
            sys_c4.metric("GPU VRAM", "N/A (CPU Mode)")

    except Exception as e:
        import traceback
        st.error(f'Error processing image: {e}')
        st.error(traceback.format_exc())
        continue

# =============================================================================
# RESULTS SUMMARY (if multiple images processed)
# =============================================================================

if len(st.session_state.results_history) > 1:
    st.divider()
    st.markdown('### Batch Results Summary')
    summary_data = []
    for result in st.session_state.results_history:
        row = {
            'Filename': result['filename'],
            'Final Prediction': result.get('final_prediction', 'N/A'),
            'Flagged %': f"{result['flagged_pct']:.1f}%",
            'Noiseprint': 'On' if result.get('noiseprint_enabled') else 'Off',
            'Timestamp': result['timestamp'][:19],
        }
        if result.get('noiseprint_summary'):
            row['Noiseprint Pred'] = result['noiseprint_summary']['prediction']
            row['Noiseprint Conf'] = f"{result['noiseprint_summary']['confidence_pct']:.1f}%"
            row['Noise Mean'] = f"{result['noiseprint_summary']['mean_score']:.4f}"
        if result['classification_results']:
            for model_name, clf_result in result['classification_results'].items():
                row[f"{model_name} (%)"] = f"{clf_result['confidence_pct']:.1f}%"
        summary_data.append(row)
    
    st.table(summary_data)
    
    # Export summary as JSON
    json_summary = json.dumps(st.session_state.results_history, indent=2)
    st.download_button(
        label='Export Results as JSON',
        data=json_summary,
        file_name=f'analysis_results_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json',
        mime='application/json'
    )

st.markdown('---')
st.markdown(
    '<div style="text-align: center; color: #999; font-size: 0.85rem;">'
    'AuthentiLens | Powered by Streamlit | Device: ' + str(DEVICE) + '</div>',
    unsafe_allow_html=True
)

