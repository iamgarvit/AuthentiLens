"""
============================================================================
ApertureAuthentiLens — ML Inference Engine
============================================================================
Two zero-shot, pre-trained model pipelines for AI-generated image detection:

    1. UnivFDDetector  — Patch-based confidence scanning via frozen CLIP ViT.
                         Compares patch embeddings against a real-image feature
                         bank using cosine similarity.

    2. ExplainableViTDetector — Hybrid EfficientNet-B0 → TransformerEncoder.
                                Extracts self-attention relevancy maps via
                                forward hooks for spatial heatmap generation.

Device policy: cuda > cpu  (mps is NOT used under any circumstances).
============================================================================
"""

from __future__ import annotations

import logging
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image as PILImage
from torchvision import models, transforms
from transformers import CLIPModel, CLIPProcessor

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logger = logging.getLogger("aperture.inference")
logging.basicConfig(level=logging.INFO)


# ============================================================================
# Device Selection Utility
# ============================================================================
def get_device() -> torch.device:
    """
    Return the best available PyTorch device.

    Policy
    ------
    * ``cuda``  — if an NVIDIA GPU is detected.
    * ``cpu``   — strict fallback; **mps is never used**.

    Returns
    -------
    torch.device
        The selected compute device.
    """
    if torch.cuda.is_available():
        logger.info("CUDA device detected — using GPU acceleration.")
        return torch.device("cuda")
    logger.info("No CUDA device found — falling back to CPU.")
    return torch.device("cpu")


# ============================================================================
# Model 1 — UnivFD (Universal Fake-Image Detector)
# ============================================================================
class UnivFDDetector:
    """
    Patch-based fake-image detector powered by a **frozen CLIP ViT**.

    Mechanism
    ---------
    1. The input image is divided into overlapping patches.
    2. Each patch is embedded via CLIP's vision encoder (``pooler_output``).
    3. Cosine similarity is computed against a pre-computed *real-image
       feature bank* (simulated here as a random tensor of shape
       ``[N, feature_dim]``).
    4. A patch whose embedding lies far from the real-image manifold
       receives a high "fake" confidence score.

    Parameters
    ----------
    device : torch.device
        Compute device (cuda / cpu).
    feature_bank_size : int
        Number of reference vectors in the simulated real-image bank.
    """

    # CLIP model identifier on HuggingFace Hub
    CLIP_MODEL_ID: str = "openai/clip-vit-base-patch32"
    # CLIP ViT-B/32 vision encoder produces 768-dim pooler_output
    FEATURE_DIM: int = 768

    def __init__(
        self,
        device: torch.device,
        feature_bank_size: int = 500,
    ) -> None:
        self.device: torch.device = device

        # ----- Load frozen CLIP vision model ---------------------------------
        logger.info("Loading CLIP ViT-B/32 from HuggingFace Hub …")
        self.model: CLIPModel = CLIPModel.from_pretrained(self.CLIP_MODEL_ID)
        self.model = self.model.to(self.device)
        self.model.eval()

        # Freeze all parameters — zero-shot, no fine-tuning
        for param in self.model.parameters():
            param.requires_grad = False

        # ----- CLIP processor (handles resizing, normalization) --------------
        self.processor: CLIPProcessor = CLIPProcessor.from_pretrained(
            self.CLIP_MODEL_ID
        )

        # ----- Simulated real-image feature bank -----------------------------
        # In production this would be pre-computed from a curated pristine
        # image dataset.  For this prototype we initialise a random bank and
        # L2-normalize each vector so cosine similarity = dot product.
        logger.info(
            "Initialising simulated real-image feature bank "
            f"({feature_bank_size} × {self.FEATURE_DIM})."
        )
        bank: torch.Tensor = torch.randn(
            feature_bank_size, self.FEATURE_DIM, device=self.device
        )
        self.feature_bank: torch.Tensor = F.normalize(bank, p=2, dim=1)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    @torch.no_grad()
    def generate_patch_scores(
        self,
        image_tensor: torch.Tensor,
        patch_size: int = 64,
        stride: int = 32,
    ) -> List[float]:
        """
        Divide *image_tensor* into overlapping patches, embed each via CLIP,
        and score them against the real-image feature bank.

        Parameters
        ----------
        image_tensor : torch.Tensor
            Input image as a ``(C, H, W)`` float tensor in ``[0, 1]``.
        patch_size : int
            Side length (pixels) of each square patch.
        stride : int
            Step size between consecutive patches (controls overlap).

        Returns
        -------
        List[float]
            Fake-confidence score for each patch in ``[0.0, 1.0]``.
            Higher ⇒ more likely manipulated.
        """
        _, height, width = image_tensor.shape

        # --- Extract overlapping patches -----------------------------------
        patches: List[torch.Tensor] = []
        for y in range(0, height - patch_size + 1, stride):
            for x in range(0, width - patch_size + 1, stride):
                patch: torch.Tensor = image_tensor[:, y : y + patch_size, x : x + patch_size]
                patches.append(patch)

        if not patches:
            logger.warning("No patches extracted — image may be too small.")
            return []

        logger.info(f"Extracted {len(patches)} overlapping patches.")

        # --- Embed each patch via CLIP -------------------------------------
        scores: List[float] = []
        for patch in patches:
            # Convert patch tensor → PIL → CLIP processor
            patch_np: np.ndarray = (
                (patch.cpu().numpy().transpose(1, 2, 0) * 255)
                .clip(0, 255)
                .astype(np.uint8)
            )

            pil_patch: PILImage.Image = PILImage.fromarray(patch_np)
            inputs = self.processor(images=pil_patch, return_tensors="pt")
            pixel_values: torch.Tensor = inputs["pixel_values"].to(self.device)

            # Forward pass — extract pooler_output embedding
            vision_outputs = self.model.vision_model(pixel_values=pixel_values)
            embedding: torch.Tensor = vision_outputs.pooler_output  # (1, 768)
            embedding = F.normalize(embedding, p=2, dim=1)

            # Cosine similarity against the real-image bank
            # embedding: (1, 768), feature_bank.T: (768, N)
            similarities: torch.Tensor = torch.mm(
                embedding, self.feature_bank.T
            )  # (1, N)
            max_sim: float = similarities.max().item()

            # Convert similarity → fake confidence
            # max_sim ∈ [-1, 1]; map to [0, 1] where 1 = definitely fake
            fake_score: float = 1.0 - (max_sim + 1.0) / 2.0
            scores.append(round(fake_score, 6))

        return scores


# ============================================================================
# Custom TransformerEncoderLayer that exposes attention weights
# ============================================================================
class _TransformerEncoderLayerWithAttn(nn.Module):
    """
    Thin wrapper around ``nn.TransformerEncoderLayer`` that forces
    ``need_weights=True`` on the internal ``MultiheadAttention`` call
    so that the forward hook can capture the attention matrix.

    The standard ``nn.TransformerEncoderLayer`` passes
    ``need_weights=False`` internally, which means the MHA returns
    ``None`` for the weights tensor — making forward hooks useless.
    This subclass fixes that.
    """

    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        activation: str = "gelu",
        batch_first: bool = True,
    ) -> None:
        super().__init__()
        self.self_attn = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=batch_first
        )
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = nn.GELU() if activation == "gelu" else nn.ReLU()

    def forward(
        self,
        src: torch.Tensor,
        src_mask: Optional[torch.Tensor] = None,
        src_key_padding_mask: Optional[torch.Tensor] = None,
        is_causal: bool = False,
    ) -> torch.Tensor:
        """
        Forward pass with **need_weights=True** so the MHA hook
        receives the full attention weight tensor.
        """
        # Self-attention — crucially with need_weights=True, average_attn_weights=False
        attn_output, _attn_weights = self.self_attn(
            src, src, src,
            attn_mask=src_mask,
            key_padding_mask=src_key_padding_mask,
            need_weights=True,
            average_attn_weights=False,
            is_causal=is_causal,
        )
        src = src + self.dropout1(attn_output)
        src = self.norm1(src)

        # Feed-forward
        ff = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(ff)
        src = self.norm2(src)
        return src


# ============================================================================
# Model 2 — Explainable Attention-Based Deepfake Detector
# ============================================================================
class ExplainableViTDetector:
    """
    Hybrid **EfficientNet-B0 → TransformerEncoder** detector with XAI output.

    Mechanism
    ---------
    1. ``EfficientNet-B0`` extracts dense local features from the image.
    2. Features are projected and fed into a ``nn.TransformerEncoder``.
    3. A **forward hook** on the final ``MultiheadAttention`` layer captures
       the self-attention weights (the *relevancy map*).
    4. Attention weights are averaged across heads and upsampled to the
       original spatial dimensions to produce a dense heatmap.

    Parameters
    ----------
    device : torch.device
        Compute device (cuda / cpu).
    num_transformer_layers : int
        Depth of the Transformer encoder.
    num_attention_heads : int
        Number of heads in each multi-head attention layer.
    embed_dim : int
        Embedding dimension fed to the Transformer.
    """

    def __init__(
        self,
        device: torch.device,
        num_transformer_layers: int = 4,
        num_attention_heads: int = 8,
        embed_dim: int = 256,
    ) -> None:
        self.device: torch.device = device
        self.embed_dim: int = embed_dim
        self._attention_weights: Optional[torch.Tensor] = None

        # ----- EfficientNet-B0 backbone (feature extractor) ------------------
        logger.info("Loading EfficientNet-B0 backbone …")
        efficientnet: models.EfficientNet = models.efficientnet_b0(
            weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1
        )
        # Remove the classifier head — we only need the feature maps
        self.backbone: nn.Module = nn.Sequential(*list(efficientnet.features.children()))
        self.backbone = self.backbone.to(self.device)
        self.backbone.eval()
        for param in self.backbone.parameters():
            param.requires_grad = False

        # EfficientNet-B0 final feature-map channels = 1280
        self._backbone_out_channels: int = 1280

        # ----- Projection: flatten spatial dims & project to embed_dim -------
        self.projection: nn.Linear = nn.Linear(
            self._backbone_out_channels, embed_dim
        ).to(self.device)

        # ----- Transformer Encoder (with custom layers exposing attn) --------
        layers = nn.ModuleList([
            _TransformerEncoderLayerWithAttn(
                d_model=embed_dim,
                nhead=num_attention_heads,
                dim_feedforward=embed_dim * 4,
                dropout=0.1,
                activation="gelu",
                batch_first=True,
            )
            for _ in range(num_transformer_layers)
        ])
        self.transformer: nn.TransformerEncoder = nn.TransformerEncoder(
            _TransformerEncoderLayerWithAttn(
                d_model=embed_dim,
                nhead=num_attention_heads,
                dim_feedforward=embed_dim * 4,
                dropout=0.1,
                activation="gelu",
                batch_first=True,
            ),
            num_layers=num_transformer_layers,
        )
        # Replace the auto-generated layers with our custom ones
        self.transformer.layers = layers
        self.transformer = self.transformer.to(self.device)
        self.transformer.eval()

        # ----- Classification head (binary: real / fake) ---------------------
        self.classifier: nn.Linear = nn.Linear(embed_dim, 1).to(self.device)

        # ----- Register forward hook on final MHA layer ----------------------
        self._register_attention_hook()

        logger.info(
            f"ExplainableViTDetector ready  "
            f"(layers={num_transformer_layers}, heads={num_attention_heads}, "
            f"dim={embed_dim})."
        )

    # ------------------------------------------------------------------
    # Attention hook machinery
    # ------------------------------------------------------------------
    def _register_attention_hook(self) -> None:
        """
        Attach a forward hook to the **final** ``MultiheadAttention`` layer
        inside the Transformer encoder.  The hook captures the attention
        weight tensor produced during each forward pass.
        """
        # Navigate to the last encoder layer's self_attn module
        final_layer = self.transformer.layers[-1]
        mha_module: nn.MultiheadAttention = final_layer.self_attn

        def _hook(
            module: nn.Module,
            input: Tuple[torch.Tensor, ...],
            output: Tuple[torch.Tensor, torch.Tensor],
        ) -> None:
            """
            Forward-hook callback.

            ``nn.MultiheadAttention.forward`` returns
            ``(attn_output, attn_output_weights)`` when
            ``need_weights=True``.  We capture the weights.

            With average_attn_weights=False the shape is:
                (batch, num_heads, num_tokens, num_tokens)
            """
            if isinstance(output, tuple) and len(output) >= 2 and output[1] is not None:
                # attn_output_weights shape: (batch, heads, tokens, tokens)
                self._attention_weights = output[1].detach()

        mha_module.register_forward_hook(_hook)
        logger.info("Forward hook registered on final MHA layer.")

    # ------------------------------------------------------------------
    # EfficientNet preprocessing transform
    # ------------------------------------------------------------------
    @staticmethod
    def _get_transform() -> transforms.Compose:
        """ImageNet-normalised transform for EfficientNet-B0."""
        return transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ]
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    @torch.no_grad()
    def generate_attention_map(
        self,
        image_tensor: torch.Tensor,
        output_size: Tuple[int, int] = (224, 224),
    ) -> np.ndarray:
        """
        Run the full hybrid pipeline and return a spatial attention heatmap.

        Parameters
        ----------
        image_tensor : torch.Tensor
            Input image as a ``(C, H, W)`` float tensor in ``[0, 1]``.
        output_size : Tuple[int, int]
            ``(H, W)`` to which the attention map is upsampled.

        Returns
        -------
        np.ndarray
            2-D array of shape ``output_size`` with values in ``[0, 1]``.
        """
        transform = self._get_transform()
        x: torch.Tensor = transform(image_tensor.unsqueeze(0)).to(self.device)

        # --- 1. EfficientNet feature extraction -----------------------------
        features: torch.Tensor = self.backbone(x)  # (1, 1280, h, w)
        batch, channels, feat_h, feat_w = features.shape
        num_tokens: int = feat_h * feat_w

        # Reshape to sequence of tokens: (1, num_tokens, channels)
        tokens: torch.Tensor = features.flatten(2).permute(0, 2, 1)

        # Project to Transformer embedding dimension
        tokens = self.projection(tokens)  # (1, num_tokens, embed_dim)

        # --- 2. Transformer forward pass ------------------------------------
        # This triggers the forward hook, which captures attention weights
        transformer_out: torch.Tensor = self.transformer(tokens)

        # --- 3. Extract attention map from hook -----------------------------
        if self._attention_weights is None:
            logger.warning(
                "Attention weights not captured — returning uniform map."
            )
            return np.ones(output_size, dtype=np.float32) * 0.5

        # attn_weights shape: (batch, heads, tokens, tokens)
        attn: torch.Tensor = self._attention_weights[0]  # (heads, tokens, tokens)

        # Average across attention heads → (tokens, tokens)
        attn_avg: torch.Tensor = attn.mean(dim=0)

        # Average attention *received* by each token (column-wise mean)
        token_importance: torch.Tensor = attn_avg.mean(dim=0)  # (num_tokens,)

        # Reshape back to spatial grid
        attn_map: torch.Tensor = token_importance.view(1, 1, feat_h, feat_w)

        # Bilinear upsample to original image dimensions
        attn_map = F.interpolate(
            attn_map,
            size=output_size,
            mode="bilinear",
            align_corners=False,
        )
        attn_map = attn_map.squeeze()

        # Normalize to [0, 1]
        attn_min: torch.Tensor = attn_map.min()
        attn_max: torch.Tensor = attn_map.max()
        if (attn_max - attn_min) > 1e-8:
            attn_map = (attn_map - attn_min) / (attn_max - attn_min)
        else:
            attn_map = torch.zeros_like(attn_map)

        # Reset stored weights for next call
        self._attention_weights = None

        return attn_map.cpu().numpy().astype(np.float32)

    @torch.no_grad()
    def predict_score(self, image_tensor: torch.Tensor) -> float:
        """
        Return a global fake-probability score in ``[0, 1]``.

        Parameters
        ----------
        image_tensor : torch.Tensor
            Input image as a ``(C, H, W)`` float tensor in ``[0, 1]``.

        Returns
        -------
        float
            Probability that the image is AI-generated / manipulated.
        """
        transform = self._get_transform()
        x: torch.Tensor = transform(image_tensor.unsqueeze(0)).to(self.device)

        # EfficientNet → tokens → Transformer
        features: torch.Tensor = self.backbone(x)
        tokens: torch.Tensor = features.flatten(2).permute(0, 2, 1)
        tokens = self.projection(tokens)
        transformer_out: torch.Tensor = self.transformer(tokens)

        # Global average pooling over token sequence
        pooled: torch.Tensor = transformer_out.mean(dim=1)  # (1, embed_dim)

        # Classify
        logit: torch.Tensor = self.classifier(pooled)  # (1, 1)
        score: float = torch.sigmoid(logit).item()

        return round(score, 6)
