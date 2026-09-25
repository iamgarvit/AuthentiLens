"""Model construction, checkpoint loading and inference for AuthentiLens.

This is the single implementation of the pipeline. The Streamlit demo
(``app/streamlit_app.py``) and the Hugging Face Space (``hf_space/``) both
import it, so the deployed demo runs exactly the code in this repository.

The evaluated pipeline is EfficientNet-B0 (balanced patches, lr 2.5e-5) +
DeepLabV3+, with a pixel threshold of 0.7 and a 30% override. See
``results/pipeline/`` for the numbers behind that choice.
"""

import logging
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import segmentation_models_pytorch as smp
import torch
import torch.nn as nn
import torchvision.transforms.functional as TF
from PIL import Image

from authentilens.paths import (
    CHECKPOINTS,
    is_lfs_pointer,
    resolve_segmentation_checkpoint,
)

logger = logging.getLogger("authentilens.models")

# ---------------------------------------------------------------------------
# Constants shared by every entry point
# ---------------------------------------------------------------------------

SEGMENTATION_IMAGE_SIZE = 512
CLASSIFIER_IMAGE_SIZE = 224

# Class order used by the training scripts and evaluation.
CLASS_NAMES = {0: "FAKE", 1: "REAL"}

# ImageNet normalisation, as used in training.
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

# Decision rule from evaluation/evaluate_pipeline.py. A pixel counts as flagged
# when P(inpainted) > PIXEL_THRESHOLD; a REAL prediction is overridden to FAKE
# when more than OVERRIDE_PCT % of pixels are flagged.
PIXEL_THRESHOLD = 0.70
OVERRIDE_PCT = 30.0

# Dropout added to the decoders by the training notebooks. It is inactive in
# eval() mode but changes the checkpoint key names, so it must be reproduced
# for load_state_dict(strict=True) to succeed.
DECODER_DROPOUT = 0.3

# ---------------------------------------------------------------------------
# Model registry
# ---------------------------------------------------------------------------

CLASSIFIER_MODELS: Dict[str, dict] = {
    "EfficientNet-B0 (balanced, lr 2.5e-5)": {
        "architecture": "efficientnet_b0",
        "checkpoint_key": "efficientnet_b0_balanced_lr2.5e-5",
    },
    "EfficientNet-B0 (balanced, lr 1e-4)": {
        "architecture": "efficientnet_b0",
        "checkpoint_key": "efficientnet_b0_balanced_lr1e-4",
    },
    "ResNet-50 (balanced, lr 2.5e-5)": {
        "architecture": "resnet50",
        "checkpoint_key": "resnet50_balanced_lr2.5e-5",
    },
    "ResNet-50 (balanced, lr 1e-4)": {
        "architecture": "resnet50",
        "checkpoint_key": "resnet50_balanced_lr1e-4",
    },
}

SEGMENTATION_MODELS: Dict[str, dict] = {
    "DeepLabV3+": {
        "architecture": "deeplabv3plus",
        "checkpoint_key": "deeplabv3plus",
    },
    "UNet": {
        "architecture": "unet",
        "checkpoint_key": "unet",
    },
}

# Defaults: the best combination in results/pipeline/.
DEFAULT_CLASSIFIER = "EfficientNet-B0 (balanced, lr 2.5e-5)"
DEFAULT_SEGMENTER = "DeepLabV3+"


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------------------------
# Checkpoint availability
# ---------------------------------------------------------------------------


def classifier_checkpoint(label: str) -> Optional[Path]:
    """Path to a classifier's weights, or None if they are absent or unpulled."""
    path = CHECKPOINTS[CLASSIFIER_MODELS[label]["checkpoint_key"]]
    if not path.exists() or is_lfs_pointer(path):
        return None
    return path


def segmenter_checkpoint(label: str) -> Optional[Path]:
    """Path to a segmenter's weights, or None if they are absent or unpulled."""
    path = resolve_segmentation_checkpoint(SEGMENTATION_MODELS[label]["checkpoint_key"])
    if path is None or is_lfs_pointer(path):
        return None
    return path


def available_classifiers() -> list:
    """Classifier labels whose weights are actually on disk, best one first."""
    return [label for label in CLASSIFIER_MODELS if classifier_checkpoint(label)]


def available_segmenters() -> list:
    """Segmenter labels whose weights are actually on disk, best one first."""
    return [label for label in SEGMENTATION_MODELS if segmenter_checkpoint(label)]


# ---------------------------------------------------------------------------
# Architectures
# ---------------------------------------------------------------------------


def build_classifier(architecture: str) -> nn.Module:
    from torchvision.models import efficientnet_b0, resnet50

    if architecture == "resnet50":
        model = resnet50(weights=None)
        model.fc = nn.Sequential(nn.Dropout(0.3), nn.Linear(model.fc.in_features, 2))
        return model
    if architecture == "efficientnet_b0":
        model = efficientnet_b0(weights=None)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, 2)
        return model
    raise ValueError(f"Unknown classifier architecture: {architecture}")


def build_segmenter(architecture: str) -> nn.Module:
    """Rebuild a segmenter exactly as its training notebook did.

    DeepLabV3+: the notebook assigned ``decoder.block = Sequential(Dropout,
    decoder.block2)``. Because ``block2`` is shared by reference, the checkpoint
    stores the same tensors twice, under ``decoder.block2.*`` and
    ``decoder.block.1.*``. Recreating that attribute makes both sets of keys
    resolve, so the checkpoint loads with strict=True.

    UNet: each decoder block's ``conv2`` was wrapped with a Dropout2d whose
    probability scales with depth.
    """
    if architecture == "deeplabv3plus":
        model = smp.DeepLabV3Plus(
            encoder_name="resnet50",
            encoder_weights=None,
            in_channels=3,
            classes=1,
            activation=None,
        )
        model.decoder.block = nn.Sequential(
            nn.Dropout2d(p=DECODER_DROPOUT),
            model.decoder.block2,
        )
        return model

    if architecture == "unet":
        model = smp.Unet(
            encoder_name="resnet50",
            encoder_weights=None,
            in_channels=3,
            classes=1,
            activation=None,
            decoder_channels=(256, 128, 64, 32, 16),
            decoder_use_batchnorm=True,
        )
        blocks = model.decoder.blocks
        for i, block in enumerate(blocks):
            scaled = DECODER_DROPOUT * (0.4 + 0.6 * i / max(len(blocks) - 1, 1))
            block.conv2 = nn.Sequential(block.conv2, nn.Dropout2d(p=scaled))
        return model

    raise ValueError(f"Unknown segmentation architecture: {architecture}")


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def _state_dict(checkpoint) -> dict:
    """The model weights inside a bare state dict or a full training checkpoint."""
    if isinstance(checkpoint, dict):
        for key in ("model_state_dict", "state_dict"):
            if key in checkpoint and isinstance(checkpoint[key], dict):
                return checkpoint[key]
    return checkpoint


def load_classifier(label: str, device: torch.device) -> nn.Module:
    path = classifier_checkpoint(label)
    if path is None:
        raise FileNotFoundError(
            f"Weights for '{label}' not found at "
            f"{CHECKPOINTS[CLASSIFIER_MODELS[label]['checkpoint_key']]}"
        )
    model = build_classifier(CLASSIFIER_MODELS[label]["architecture"])
    state = _state_dict(torch.load(path, map_location="cpu", weights_only=False))
    model.load_state_dict(state, strict=True)
    return model.to(device).eval()


def load_segmenter(label: str, device: torch.device) -> nn.Module:
    path = segmenter_checkpoint(label)
    if path is None:
        raise FileNotFoundError(
            f"Weights for '{label}' not found at "
            f"{CHECKPOINTS[SEGMENTATION_MODELS[label]['checkpoint_key']]}"
        )
    model = build_segmenter(SEGMENTATION_MODELS[label]["architecture"])
    state = _state_dict(torch.load(path, map_location="cpu", weights_only=False))
    model.load_state_dict(state, strict=True)
    return model.to(device).eval()


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------


@torch.no_grad()
def segment(
    model: nn.Module,
    image: Image.Image,
    threshold: float,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """Localise the inpainted region.

    Returns the probability map upsampled to the image's own resolution (for
    display), its binary mask, and the flagged-pixel percentage.

    The percentage is measured on the raw 512x512 probabilities, exactly as
    evaluation/evaluate_pipeline.py does, so the override threshold means the
    same thing here as it did when the pipeline was benchmarked. Deriving it
    from the upsampled map instead would make it depend on the uploaded
    image's resolution and on the 8-bit quantisation used for display.
    """
    orig_w, orig_h = image.size

    resized = image.resize((SEGMENTATION_IMAGE_SIZE, SEGMENTATION_IMAGE_SIZE), Image.BILINEAR)
    tensor = TF.normalize(TF.to_tensor(resized), mean=MEAN, std=STD).unsqueeze(0).to(device)

    with torch.autocast(device_type=device.type, enabled=(device.type == "cuda")):
        logits = model(tensor)

    prob = torch.sigmoid(logits.float())[0, 0].cpu().numpy()
    flagged_pct = float((prob > threshold).mean()) * 100.0

    prob_full = np.asarray(
        Image.fromarray((prob * 255).astype(np.uint8)).resize((orig_w, orig_h), Image.BILINEAR),
        dtype=np.float32,
    ) / 255.0

    return prob_full, (prob_full > threshold).astype(np.uint8), flagged_pct


@torch.no_grad()
def classify(model: nn.Module, image: Image.Image, device: torch.device) -> dict:
    """FAKE/REAL prediction with class probabilities, as in the evaluation scripts."""
    resized = image.resize((CLASSIFIER_IMAGE_SIZE, CLASSIFIER_IMAGE_SIZE), Image.BILINEAR)
    tensor = TF.normalize(TF.to_tensor(resized), mean=MEAN, std=STD).unsqueeze(0).to(device)

    probs = torch.softmax(model(tensor), dim=1)[0].cpu()
    index = int(probs.argmax())

    return {
        "class": CLASS_NAMES[index],
        "is_fake": index == 0,
        "confidence_pct": float(probs[index]) * 100.0,
        "probabilities": {
            "FAKE": float(probs[0]) * 100.0,
            "REAL": float(probs[1]) * 100.0,
        },
    }


# ---------------------------------------------------------------------------
# Pipeline decision
# ---------------------------------------------------------------------------


def pipeline_decision(
    classifier_says_fake: bool,
    flagged_pct: Optional[float],
    override_pct: float = OVERRIDE_PCT,
) -> dict:
    """Combine the classifier and segmenter exactly as evaluate_pipeline.py does.

    The classifier decides, except that a REAL prediction is overridden to FAKE
    when the segmenter flags more than ``override_pct`` % of the pixels.
    """
    override = (
        not classifier_says_fake
        and flagged_pct is not None
        and flagged_pct > override_pct
    )
    is_fake = classifier_says_fake or override

    if classifier_says_fake:
        reason = "The classifier called this image FAKE."
    elif override:
        reason = (
            f"The classifier said REAL, but the segmenter flagged "
            f"{flagged_pct:.1f}% of pixels (more than {override_pct:g}%), so the "
            f"pipeline overrides it to FAKE."
        )
    elif flagged_pct is not None:
        reason = (
            f"The classifier said REAL and the segmenter flagged only "
            f"{flagged_pct:.1f}% of pixels (at most {override_pct:g}%), so the "
            f"prediction stands."
        )
    else:
        reason = "The classifier said REAL; no segmenter was available to override it."

    return {
        "prediction": "FAKE" if is_fake else "REAL",
        "is_fake": is_fake,
        "decision_source": "segmentation_override" if override else "classifier",
        "reason": reason,
    }
