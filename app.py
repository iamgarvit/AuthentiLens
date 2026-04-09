# =============================================================================
# AI Inpainting Detection & Classification — Streamlit Demo (Enhanced)
# =============================================================================
# Combines segmentation (DeepLabV3Plus) and classification (ResNet + EfficientNet)
# - Segmentation: Detects AI-inpainted regions
# - Classification: Classifies the overall image (e.g., FAKE/REAL)
#
# Run:
#   streamlit run new_app.py --server.port 8501
#
# Dependencies:
#   pip install streamlit segmentation-models-pytorch torch torchvision pillow numpy
# =============================================================================

import io
import os
import json
import torch
import torch.nn as nn
import numpy as np
import streamlit as st
import segmentation_models_pytorch as smp
import torchvision.transforms.functional as TF
import torchvision.transforms as T
from PIL import Image
import matplotlib.colors as mcolors
from datetime import datetime

# =============================================================================
# CONFIG
# =============================================================================

# Segmentation
DEEPLAB_CHECKPOINT = './checkpoints_deeplab/best_model.pth'
DEEPLAB_IMAGE_SIZE = 512

# Classification
RESNET_CHECKPOINT = './checkpoints_resnet/best_model.pth'
EFFICIENTNET_CHECKPOINT = './checkpoints_efficientnet/best_model.pth'
CLASSIFIER_IMAGE_SIZE = 224

# Class names for classification
CLASS_NAMES = {0: "Class 0 (e.g., FAKE)", 1: "Class 1 (e.g., REAL)"}

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ImageNet normalization
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]


# =============================================================================
# SESSION STATE INITIALIZATION
# =============================================================================

if 'results_history' not in st.session_state:
    st.session_state.results_history = []


# =============================================================================
# SEGMENTATION MODEL LOADING
# =============================================================================

@st.cache_resource(show_spinner='Loading segmentation model...')
def load_deeplab_model(ckpt_path: str):
    """Load DeepLabV3Plus for inpainting detection."""
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    
    model = smp.DeepLabV3Plus(
        encoder_name='resnet50',
        encoder_weights=None,
        in_channels=3,
        classes=1,
        activation=None,
    )
    model.decoder.block2 = nn.Sequential(
        model.decoder.block2,
        nn.Dropout2d(p=0.3),
    )

    ckpt = torch.load(ckpt_path, map_location=DEVICE)
    state = ckpt.get('model_state_dict', ckpt)

    # Handle buggy checkpoint variant
    is_buggy = any(k.startswith('decoder.block.') for k in state.keys())
    if is_buggy:
        new_state = {}
        for k, v in state.items():
            if k.startswith('decoder.block.1.'):
                suffix = k[len('decoder.block.1.'):]
                new_key = f'decoder.block2.0.{suffix}'
                new_state[new_key] = v
            elif k.startswith('decoder.block.'):
                pass
            elif k.startswith('decoder.block2.'):
                pass
            else:
                new_state[k] = v
        state = new_state

    model.load_state_dict(state, strict=True)
    model.to(DEVICE)
    model.eval()
    return model


# =============================================================================
# CLASSIFICATION MODEL LOADING
# =============================================================================

@st.cache_resource(show_spinner='Loading classification models...')
def load_classifier_models():
    """Load both ResNet-50 and EfficientNet-B0 for classification."""
    from torchvision.models import resnet50, efficientnet_b0

    models = {}

    # ResNet-50
    if os.path.exists(RESNET_CHECKPOINT):
        resnet = resnet50(pretrained=False)
        resnet.fc = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(resnet.fc.in_features, 2)
        )
        resnet.load_state_dict(torch.load(RESNET_CHECKPOINT, map_location=DEVICE))
        resnet.to(DEVICE)
        resnet.eval()
        models['ResNet-50'] = resnet
    else:
        st.warning(f'ResNet checkpoint not found: {RESNET_CHECKPOINT}')

    # EfficientNet-B0
    if os.path.exists(EFFICIENTNET_CHECKPOINT):
        efficientnet = efficientnet_b0(pretrained=False)
        num_ftrs = efficientnet.classifier[1].in_features
        efficientnet.classifier[1] = nn.Linear(num_ftrs, 2)
        efficientnet.load_state_dict(torch.load(EFFICIENTNET_CHECKPOINT, map_location=DEVICE))
        efficientnet.to(DEVICE)
        efficientnet.eval()
        models['EfficientNet-B0'] = efficientnet
    else:
        st.warning(f'EfficientNet checkpoint not found: {EFFICIENTNET_CHECKPOINT}')

    return models


# =============================================================================
# SEGMENTATION INFERENCE
# =============================================================================

def run_segmentation_inference(model, pil_image: Image.Image, threshold: float) -> tuple:
    """
    Returns:
        prob_map   : float32 numpy array [H, W] in [0, 1]
        binary_map : uint8 numpy array [H, W] in {0, 1}
    """
    orig_w, orig_h = pil_image.size

    img_r = pil_image.resize((DEEPLAB_IMAGE_SIZE, DEEPLAB_IMAGE_SIZE), Image.BILINEAR)
    t = TF.to_tensor(img_r)
    t = TF.normalize(t, mean=MEAN, std=STD).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        with torch.cuda.amp.autocast(enabled=(DEVICE.type == 'cuda')):
            logit = model(t)

    prob = torch.sigmoid(logit).squeeze().cpu().float().numpy()

    prob_pil = Image.fromarray((prob * 255).astype(np.uint8)).resize(
        (orig_w, orig_h), Image.BILINEAR
    )
    prob_orig = np.array(prob_pil, dtype=np.float32) / 255.0

    binary = (prob_orig > threshold).astype(np.uint8)
    return prob_orig, binary


# =============================================================================
# CLASSIFICATION INFERENCE
# =============================================================================

def run_classification_inference(models: dict, pil_image: Image.Image) -> dict:
    """
    Run classification on image with all available classifiers.
    
    Returns:
        dict with classification results per model
    """
    transform = T.Compose([
        T.Resize((CLASSIFIER_IMAGE_SIZE, CLASSIFIER_IMAGE_SIZE)),
        T.ToTensor(),
        T.Normalize(mean=MEAN, std=STD)
    ])

    image_tensor = transform(pil_image).unsqueeze(0).to(DEVICE)
    results = {}

    with torch.no_grad():
        for model_name, model in models.items():
            output = model(image_tensor)
            probabilities = torch.softmax(output, dim=1)[0]
            confidence, predicted_idx = torch.max(probabilities, dim=0)

            pred_class = CLASS_NAMES.get(predicted_idx.item(), f"Class {predicted_idx.item()}")
            conf_value = confidence.item()

            results[model_name] = {
                'class': pred_class,
                'confidence': conf_value,
                'confidence_pct': conf_value * 100,
                'probabilities': {
                    CLASS_NAMES[0]: float(probabilities[0].item()) * 100,
                    CLASS_NAMES[1]: float(probabilities[1].item()) * 100,
                }
            }

    return results


# =============================================================================
# VISUALIZATION HELPERS
# =============================================================================

def make_overlay(pil_image: Image.Image, prob_map: np.ndarray, opacity: float, threshold: float) -> Image.Image:
    """Composite probability heatmap over original image."""
    cmap = mcolors.LinearSegmentedColormap.from_list(
        'inpaint_heat',
        [(0.00, '#00c800'), (0.50, '#aaff00'), (1.00, '#ffff00')]
    )

    rgba = cmap(prob_map)

    alpha_mask = np.where(
        prob_map >= threshold,
        opacity * prob_map,
        0.0
    ).astype(np.float32)
    rgba[:, :, 3] = alpha_mask

    heat_rgba = Image.fromarray((rgba * 255).astype(np.uint8), mode='RGBA')
    base = pil_image.convert('RGBA')
    composite = Image.alpha_composite(base, heat_rgba)
    return composite.convert('RGB')


def get_prob_map_image(prob_map: np.ndarray) -> Image.Image:
    """Convert probability map to colored image."""
    cmap = mcolors.LinearSegmentedColormap.from_list(
        'inpaint_heat', ['#00c800', '#aaff00', '#ffff00']
    )
    prob_rgb = (cmap(prob_map)[:, :, :3] * 255).astype(np.uint8)
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
    page_title='Inpainting Detection & Classification',
    page_icon='🤖',
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

st.title('🤖 Inpainting Detection & Classification')
st.markdown(
    '**Analyze images** using segmentation (detect inpainted regions) and classification (overall category prediction).'
)

# =============================================================================
# SIDEBAR
# =============================================================================

with st.sidebar:
    st.header('⚙️ Settings')

    threshold = st.slider(
        'Segmentation threshold',
        min_value=0.10, max_value=0.90, value=0.50, step=0.05,
        help='Pixels above this probability are flagged as inpainted.'
    )

    opacity = st.slider(
        'Heatmap opacity',
        min_value=0.20, max_value=1.00, value=0.65, step=0.05,
        help='Heatmap blend intensity.'
    )

    st.divider()

    show_seg_raw = st.checkbox('Show segmentation probability map', value=True)
    show_seg_binary = st.checkbox('Show segmentation binary mask', value=True)
    show_seg_stats = st.checkbox('Show segmentation statistics', value=True)
    show_download_options = st.checkbox('Show download options', value=True)

    st.divider()
    st.markdown(f'**Device:** `{DEVICE}`')
    st.markdown(f'**Models:** Segmentation + {len(st.session_state.get("classifier_models", {}))} Classifiers')

    st.divider()
    st.markdown("""
**Segmentation color legend**

🟢 Green = Low inpainting probability  
🟡 Yellow = High inpainting probability  
⬜ Transparent = Below threshold
    """)

# =============================================================================
# MODEL LOADING
# =============================================================================

try:
    deeplab_model = load_deeplab_model(DEEPLAB_CHECKPOINT)
    st.sidebar.success('✓ Segmentation model loaded')
except Exception as e:
    st.error(f'❌ Failed to load segmentation model: {e}')
    st.stop()

try:
    classifier_models = load_classifier_models()
    st.session_state.classifier_models = classifier_models
    if classifier_models:
        st.sidebar.success(f'✓ Classification models loaded ({len(classifier_models)})')
    else:
        st.sidebar.warning('⚠️ No classification models available')
except Exception as e:
    st.error(f'❌ Failed to load classification models: {e}')
    st.stop()

# =============================================================================
# FILE UPLOADER
# =============================================================================

uploaded_files = st.file_uploader(
    'Upload one or more images',
    type=['png', 'jpg', 'jpeg', 'webp', 'bmp'],
    accept_multiple_files=True,
)

if not uploaded_files:
    st.info('👆 Upload an image to analyze')
    st.stop()

# =============================================================================
# PROCESS EACH IMAGE
# =============================================================================

for uploaded_file in uploaded_files:
    st.divider()
    st.subheader(f'📄 {uploaded_file.name}')

    try:
        pil_image = Image.open(uploaded_file).convert('RGB')
        W, H = pil_image.size
        st.caption(f'Resolution: {W}×{H}px')

        # Run inference
        with st.spinner('Running inference...'):
            prob_map, binary_map = run_segmentation_inference(deeplab_model, pil_image, threshold)
            overlay = make_overlay(pil_image, prob_map, opacity, threshold)

            if classifier_models:
                classification_results = run_classification_inference(classifier_models, pil_image)
            else:
                classification_results = {}

        # Store results in session history
        result_entry = {
            'filename': uploaded_file.name,
            'timestamp': datetime.now().isoformat(),
            'threshold': threshold,
            'flagged_pct': float((binary_map.mean()) * 100),
            'classification_results': classification_results,
        }
        st.session_state.results_history.append(result_entry)

        # --- CLASSIFICATION RESULTS ---
        if classification_results:
            st.markdown('### 🏷️ Classification Results')

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
                        <div style="font-size: 0.85rem; color: #666;">
                            {list(result['probabilities'].keys())[0]}: {result['probabilities'][list(result['probabilities'].keys())[0]]:.1f}%<br/>
                            {list(result['probabilities'].keys())[1]}: {result['probabilities'][list(result['probabilities'].keys())[1]]:.1f}%
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

        # --- SEGMENTATION VISUALIZATION ---
        st.markdown('### 🔍 Segmentation Analysis')

        n_cols = 2 + int(show_seg_raw) + int(show_seg_binary)
        cols = st.columns(n_cols)

        col_idx = 0
        with cols[col_idx]:
            st.image(pil_image, caption='Original Image', use_column_width=True)
        col_idx += 1

        with cols[col_idx]:
            st.image(overlay, caption='Inpainting Heatmap', use_column_width=True)
        col_idx += 1

        if show_seg_raw:
            prob_img = get_prob_map_image(prob_map)
            with cols[col_idx]:
                st.image(prob_img, caption='Probability Map', use_column_width=True)
            col_idx += 1

        if show_seg_binary:
            binary_rgb = (binary_map * 255).astype(np.uint8)
            with cols[col_idx]:
                st.image(binary_rgb, caption=f'Binary Mask (t={threshold})', use_column_width=True)

        # --- DOWNLOAD OPTIONS ---
        if show_download_options:
            st.markdown('#### 💾 Download Results')
            
            dcols = st.columns(3)
            
            with dcols[0]:
                overlay_bytes = image_to_bytes(overlay)
                st.download_button(
                    label='📥 Heatmap',
                    data=overlay_bytes,
                    file_name=f'{uploaded_file.name.rsplit(".", 1)[0]}_heatmap.png',
                    mime='image/png'
                )
            
            with dcols[1]:
                binary_img = Image.fromarray((binary_map * 255).astype(np.uint8))
                binary_bytes = image_to_bytes(binary_img)
                st.download_button(
                    label='📥 Mask',
                    data=binary_bytes,
                    file_name=f'{uploaded_file.name.rsplit(".", 1)[0]}_mask.png',
                    mime='image/png'
                )
            
            with dcols[2]:
                prob_img_bytes = image_to_bytes(get_prob_map_image(prob_map))
                st.download_button(
                    label='📥 Probability Map',
                    data=prob_img_bytes,
                    file_name=f'{uploaded_file.name.rsplit(".", 1)[0]}_prob.png',
                    mime='image/png'
                )

        # --- STATISTICS ---
        if show_seg_stats:
            st.markdown('#### 📊 Segmentation Statistics')

            flagged_pct = float(binary_map.mean()) * 100
            mean_prob = float(prob_map.mean())
            max_prob = float(prob_map.max())
            min_prob = float(prob_map.min())
            high_conf = float((prob_map > 0.75).mean()) * 100

            c1, c2, c3, c4, c5 = st.columns(5)
            c1.metric('Flagged Pixels', f'{flagged_pct:.1f}%',
                      help='% of pixels classified as inpainted')
            c2.metric('Mean Probability', f'{mean_prob:.3f}',
                      help='Average inpainting probability')
            c3.metric('Peak Probability', f'{max_prob:.3f}',
                      help='Maximum inpainting probability')
            c4.metric('Min Probability', f'{min_prob:.3f}',
                      help='Minimum inpainting probability')
            c5.metric('High Confidence (>75%)', f'{high_conf:.1f}%',
                      help='% of pixels with >75% inpainting confidence')

    except Exception as e:
        st.error(f'❌ Error processing image: {e}')
        continue

# =============================================================================
# RESULTS SUMMARY (if multiple images processed)
# =============================================================================

if len(st.session_state.results_history) > 1:
    st.divider()
    st.markdown('### 📈 Batch Results Summary')
    
    summary_data = []
    for result in st.session_state.results_history:
        row = {
            'Filename': result['filename'],
            'Flagged %': f"{result['flagged_pct']:.1f}%",
            'Timestamp': result['timestamp'][:19],
        }
        if result['classification_results']:
            for model_name, clf_result in result['classification_results'].items():
                row[f"{model_name} (%)"] = f"{clf_result['confidence_pct']:.1f}%"
        summary_data.append(row)
    
    st.table(summary_data)
    
    # Export summary as JSON
    json_summary = json.dumps(st.session_state.results_history, indent=2)
    st.download_button(
        label='📥 Export Results as JSON',
        data=json_summary,
        file_name=f'analysis_results_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json',
        mime='application/json'
    )

st.markdown('---')
st.markdown(
    '<div style="text-align: center; color: #999; font-size: 0.85rem;">'
    'Inpainting Detection & Classification | Powered by Streamlit | Device: ' + str(DEVICE) + '</div>',
    unsafe_allow_html=True
)