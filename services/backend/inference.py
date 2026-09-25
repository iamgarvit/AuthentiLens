import time
import logging
import os
import sys
from pathlib import Path
import cv2

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torchvision.models import efficientnet_b0
import segmentation_models_pytorch as smp

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repo root (/app in Docker)
from authentilens.paths import (  # noqa: E402
    BEST_CLASSIFIER,
    CHECKPOINTS,
    is_lfs_pointer,
    resolve_segmentation_checkpoint,
)

logger = logging.getLogger("authentilens.inference")
logging.basicConfig(level=logging.INFO)

# ---------------------------------------------------------------------------
# Configuration (one canonical checkpoint per model, overridable by env var)
# ---------------------------------------------------------------------------
# Best pipeline from results/pipeline/: EfficientNet-B0 (balanced, lr 2.5e-5) + DeepLabV3+
CLASSIFIER_CKPT = Path(os.environ.get("AUTHENTILENS_CLASSIFIER_CKPT", CHECKPOINTS[BEST_CLASSIFIER]))
SEGMENTER_CKPT_OVERRIDE = os.environ.get("AUTHENTILENS_SEGMENTER_CKPT")

# Decision rule used in evaluation/evaluate_pipeline.py: if the classifier says REAL,
# override to FAKE when more than OVERRIDE_PCT % of pixels have P(inpainted) > PIXEL_THRESHOLD.
PIXEL_THRESHOLD = float(os.environ.get("AUTHENTILENS_PIXEL_THRESHOLD", "0.7"))
OVERRIDE_PCT = float(os.environ.get("AUTHENTILENS_OVERRIDE_PCT", "30"))


def remap_checkpoint_keys(state_dict):
    """
    Remap checkpoint keys to match current model architecture.
    Handles cases where checkpoint was saved without 'model.' prefix.
    """
    new_state = {}
    for k, v in state_dict.items():
        # If key doesn't start with 'model.', add it
        if not k.startswith('model.'):
            new_key = 'model.' + k
        else:
            new_key = k
        new_state[new_key] = v
    return new_state

def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")

class CustomDeepLabV3(nn.Module):
    def __init__(self, num_classes=1, decoder_dropout=0.3):
        super().__init__()
        self.model = smp.DeepLabV3Plus(
            encoder_name="resnet50",
            encoder_weights=None,
            in_channels=3,
            classes=num_classes,
            activation=None,  # Match notebook: raw logits, sigmoid applied in loss
        )

        # Match notebook: add dropout to decoder
        if decoder_dropout > 0:
            self.model.decoder.block = nn.Sequential(
                nn.Dropout2d(p=decoder_dropout),
                self.model.decoder.block2,
            )

    def forward(self, x):
        return self.model(x)

class AuthenticDetector:
    def __init__(self, device: torch.device):
        self.device = device

        # 1. Load EfficientNet for Classification
        logger.info("Loading EfficientNet-B0 Classifier...")
        self.classifier = efficientnet_b0(pretrained=False)
        num_ftrs = self.classifier.classifier[1].in_features
        self.classifier.classifier[1] = nn.Linear(num_ftrs, 2)

        self.classifier_path = CLASSIFIER_CKPT
        if not self.classifier_path.exists():
            raise FileNotFoundError(
                f"Classifier weights not found at {self.classifier_path}. "
                "See \"Get the weights\" in the README, or set AUTHENTILENS_CLASSIFIER_CKPT."
            )
        if is_lfs_pointer(self.classifier_path):
            raise RuntimeError(
                f"{self.classifier_path} is a Git LFS pointer, not the weights. "
                "See \"Get the weights\" in the README."
            )
        clf_state = torch.load(self.classifier_path, map_location=device)
        if "model_state_dict" in clf_state:
            clf_state = clf_state["model_state_dict"]
        self.classifier.load_state_dict(clf_state)
        logger.info(f"✓ Successfully loaded classifier from {self.classifier_path}")

        # 2. Load DeepLabV3 for Segmentation (match notebook architecture).
        #    Optional: without weights the API runs in classification-only mode.
        self.segmenter = None
        self.segmenter_path = (
            Path(SEGMENTER_CKPT_OVERRIDE) if SEGMENTER_CKPT_OVERRIDE
            else resolve_segmentation_checkpoint("deeplabv3plus")
        )
        if self.segmenter_path is None or not self.segmenter_path.exists():
            logger.error(
                f"Segmentation weights not found (expected {CHECKPOINTS['deeplabv3plus']}). "
                "Segmentation disabled: running classification only."
            )
        elif is_lfs_pointer(self.segmenter_path):
            logger.error(
                f"{self.segmenter_path} is a Git LFS pointer (see \"Get the weights\" in the README). "
                "Segmentation disabled: running classification only."
            )
        else:
            logger.info("Loading DeepLabV3 Segmenter...")
            segmenter = CustomDeepLabV3(num_classes=1, decoder_dropout=0.3)
            try:
                seg_state = torch.load(self.segmenter_path, map_location=device)
                if "model_state_dict" in seg_state:
                    seg_state = seg_state["model_state_dict"]
                elif "state_dict" in seg_state:
                    seg_state = seg_state["state_dict"]

                # Filter out batch norm tracking keys and remove 'model.' prefix if present
                # (checkpoint may have been saved without the wrapper prefix)
                cleaned = {}
                for k, v in seg_state.items():
                    if 'num_batches_tracked' in k:
                        continue
                    # Remove 'model.' prefix if it exists, since we'll load into segmenter.model
                    clean_k = k.replace('model.', '', 1) if k.startswith('model.') else k
                    cleaned[clean_k] = v

                logger.info(f"Cleaned {len(cleaned)} checkpoint keys, loading into inner model...")
                # Load directly into the inner smp model, not the wrapper
                segmenter.model.load_state_dict(cleaned, strict=False)
                self.segmenter = segmenter.to(self.device).eval()
                logger.info(f"✓ Successfully loaded segmenter from {self.segmenter_path}")
            except Exception as e:
                logger.error(f"Failed to load segmenter from {self.segmenter_path}: {e}. "
                             "Segmentation disabled: running classification only.")

        self.classifier.to(self.device)
        self.classifier.eval()

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    @property
    def segmentation_available(self) -> bool:
        return self.segmenter is not None

    @torch.no_grad()
    def analyze(self, image_np: np.ndarray) -> dict:
        # Reset CUDA peak memory tracking if available so we measure just this request
        if torch.cuda.is_available() and self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(self.device)

        start_total = time.time()
        start_clf = time.time()
        # image_np is RGB numpy array
        # Resize for classification (224x224)
        clf_img = cv2.resize(image_np, (224, 224))
        clf_tensor = self.transform(clf_img).unsqueeze(0).to(self.device)

        # Classification (class 0 = FAKE, 1 = REAL); argmax as in evaluation
        outputs = self.classifier(clf_tensor)
        probs = torch.softmax(outputs, dim=1)[0]
        fake_prob = probs[0].item()
        classifier_says_fake = int(probs.argmax().item()) == 0
        clf_time = time.time() - start_clf
        logger.info(f"Classification: fake_prob={fake_prob:.4f}, classifier_says_fake={classifier_says_fake}")

        segmentation_mask = []
        flagged_pct = None
        seg_time = 0.0

        # Segmentation always runs (when available): it localises edits on FAKE images
        # and can override a REAL prediction.
        if self.segmenter is not None:
            start_seg = time.time()
            seg_img = cv2.resize(image_np, (512, 512))
            seg_tensor = self.transform(seg_img).unsqueeze(0).to(self.device)

            with torch.autocast(device_type=self.device.type, enabled=self.device.type == "cuda"):
                mask_out = self.segmenter(seg_tensor)

            # Return probability map (not binary) - matches notebook's predict_single
            # The frontend will apply its own threshold to this probability map
            mask_probs = torch.sigmoid(mask_out.float()).squeeze().cpu().numpy()
            flagged_pct = float((mask_probs > PIXEL_THRESHOLD).mean()) * 100
            logger.info(f"Segmentation mask stats: min={mask_probs.min():.4f}, max={mask_probs.max():.4f}, "
                        f"mean={mask_probs.mean():.4f}, flagged={flagged_pct:.2f}%")
            segmentation_mask = mask_probs.tolist()
            seg_time = time.time() - start_seg

        seg_override = (not classifier_says_fake) and flagged_pct is not None and flagged_pct > OVERRIDE_PCT
        is_fake = classifier_says_fake or seg_override
        decision_source = "segmentation_override" if seg_override else "classifier"

        total_time = time.time() - start_total

        cuda_alloc_mb = 0.0
        cuda_res_mb = 0.0
        if torch.cuda.is_available() and self.device.type == "cuda":
            cuda_alloc_mb = torch.cuda.max_memory_allocated(self.device) / (1024 ** 2)
            cuda_res_mb = torch.cuda.max_memory_reserved(self.device) / (1024 ** 2)

        return {
            "is_fake": is_fake,
            "fake_prob": fake_prob,
            "mask": segmentation_mask,
            "clf_time": clf_time,
            "seg_time": seg_time,
            "total_time": total_time,
            "cuda_alloc_mb": cuda_alloc_mb,
            "cuda_res_mb": cuda_res_mb,
            "flagged_pct": flagged_pct,
            "decision_source": decision_source,
            "segmentation_available": self.segmentation_available,
        }
