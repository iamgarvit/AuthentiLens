import time
import logging
import os
import cv2

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torchvision.models import efficientnet_b0
import segmentation_models_pytorch as smp

logger = logging.getLogger("aperture.inference")
logging.basicConfig(level=logging.INFO)

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
        
        # 2. Load DeepLabV3 for Segmentation (match notebook architecture)
        logger.info("Loading DeepLabV3 Segmenter...")
        self.segmenter = CustomDeepLabV3(num_classes=1, decoder_dropout=0.3)
        
        # Try loading segmenter weights from multiple checkpoint paths in priority order
        # Priority: best_deeplabv3plus(1).pth > best_deeplabv3plus.pth > best_model.pth
        seg_candidates = [
            # Local paths (for non-Docker)
            "../checkpoints_deeplab/best_deeplabv3plus (1).pth",
            "../checkpoints_deeplab/best_deeplabv3plus.pth", 
            "../checkpoints_deeplab/best_model.pth",
            # Docker paths
            "/app/checkpoints_deeplab/best_deeplabv3plus (1).pth",
            "/app/checkpoints_deeplab/best_deeplabv3plus.pth",
            "/app/checkpoints_deeplab/best_model.pth",
        ]
        
        seg_state = None
        seg_loaded_from = None
        for seg_path in seg_candidates:
            try:
                if os.path.exists(seg_path):
                    logger.info(f"Trying segmenter checkpoint: {seg_path}")
                    seg_state = torch.load(seg_path, map_location=device)
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
                        # Remove 'model.' prefix if it exists, since we'll load into self.segmenter.model
                        clean_k = k.replace('model.', '', 1) if k.startswith('model.') else k
                        cleaned[clean_k] = v
                    
                    logger.info(f"Cleaned {len(cleaned)} checkpoint keys, loading into inner model...")
                    # Load directly into the inner smp model, not the wrapper
                    self.segmenter.model.load_state_dict(cleaned, strict=False)
                    seg_loaded_from = seg_path
                    logger.info(f"✓ Successfully loaded segmenter from {seg_path}")
                    break
            except Exception as e:
                logger.warning(f"Failed to load from {seg_path}: {e}")
                continue
        
        if seg_loaded_from is None:
            logger.warning("Could not load segmenter weights from any checkpoint path")
        
        # Load classifier weights
        clf_path = "/app/checkpoints_efficientnet/best_model.pth"
        if os.path.exists("../checkpoints_efficientnet/best_model.pth"):
            clf_path = "../checkpoints_efficientnet/best_model.pth"
        
        try:
            clf_state = torch.load(clf_path, map_location=device)
            if "model_state_dict" in clf_state:
                clf_state = clf_state["model_state_dict"]
            self.classifier.load_state_dict(clf_state)
            logger.info(f"✓ Successfully loaded classifier from {clf_path}")
        except Exception as e:
            logger.warning(f"Could not load classifier weights from {clf_path}: {e}")

        self.classifier.to(self.device)
        self.classifier.eval()
        
        self.segmenter.to(self.device)
        self.segmenter.eval()

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    @torch.no_grad()
    def analyze(self, image_np: np.ndarray):
        # Reset CUDA peak memory tracking if available so we measure just this request
        if torch.cuda.is_available() and self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(self.device)
            
        start_total = time.time()
        start_clf = time.time()
        # image_np is RGB numpy array
        # Resize for classification (224x224)
        clf_img = cv2.resize(image_np, (224, 224))
        clf_tensor = self.transform(clf_img).unsqueeze(0).to(self.device)
        
        # Classification
        outputs = self.classifier(clf_tensor)
        probs = torch.softmax(outputs, dim=1)[0]
        fake_prob = probs[0].item() 
        
        is_fake = fake_prob > 0.7
        clf_time = time.time() - start_clf
        logger.info(f"Classification: fake_prob={fake_prob:.4f}, is_fake={is_fake}")

        segmentation_mask = []
        seg_time = 0.0
        
        # If Fake, run segmentation
        if is_fake:
            start_seg = time.time()
            seg_img = cv2.resize(image_np, (512, 512))
            seg_tensor = self.transform(seg_img).unsqueeze(0).to(self.device)
            
            with torch.cuda.amp.autocast(enabled=True):
                mask_out = self.segmenter(seg_tensor)
            
            # Return probability map (not binary) - matches notebook's predict_single
            # The frontend will apply its own threshold to this probability map
            mask_probs = torch.sigmoid(mask_out).squeeze().cpu().numpy()
            logger.info(f"Segmentation mask stats: min={mask_probs.min():.4f}, max={mask_probs.max():.4f}, mean={mask_probs.mean():.4f}")
            segmentation_mask = mask_probs.tolist()
            seg_time = time.time() - start_seg
        
        total_time = time.time() - start_total
        
        cuda_alloc_mb = 0.0
        cuda_res_mb = 0.0
        if torch.cuda.is_available() and self.device.type == "cuda":
            cuda_alloc_mb = torch.cuda.max_memory_allocated(self.device) / (1024 ** 2)
            cuda_res_mb = torch.cuda.max_memory_reserved(self.device) / (1024 ** 2)
            
        return is_fake, fake_prob, segmentation_mask, clf_time, seg_time, total_time, cuda_alloc_mb, cuda_res_mb

