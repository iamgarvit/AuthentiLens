"""Canonical filesystem locations for AuthentiLens.

Every script resolves data, checkpoints and results relative to the repository
root (not the current working directory), so they can be run from anywhere.

Environment overrides:
    AUTHENTILENS_DATA_DIR        -> where datasets live (default: <repo>/data)
    AUTHENTILENS_CHECKPOINT_DIR  -> where model weights live (default: <repo>/checkpoints)
"""

import logging
import os
from pathlib import Path
from typing import Optional

logger = logging.getLogger("authentilens.paths")

REPO_ROOT = Path(__file__).resolve().parents[1]


def _env_path(var: str, default: Path) -> Path:
    value = os.environ.get(var)
    return Path(value).expanduser().resolve() if value else default


DATA_DIR = _env_path("AUTHENTILENS_DATA_DIR", REPO_ROOT / "data")
CHECKPOINT_DIR = _env_path("AUTHENTILENS_CHECKPOINT_DIR", REPO_ROOT / "checkpoints")
RESULTS_DIR = REPO_ROOT / "results"

CLASSIFICATION_CKPT_DIR = CHECKPOINT_DIR / "classification"
SEGMENTATION_CKPT_DIR = CHECKPOINT_DIR / "segmentation"

# One canonical weights file per model.
CHECKPOINTS = {
    # Classifiers trained on CIFAKE (fully synthetic images)
    "resnet50_cifake": CLASSIFICATION_CKPT_DIR / "resnet50_cifake" / "best_model.pth",
    "efficientnet_b0_cifake": CLASSIFICATION_CKPT_DIR / "efficientnet_b0_cifake" / "best_model.pth",
    # Classifiers trained on the balanced SD2 patch dataset (data/sd2-classification)
    "resnet50_balanced_lr1e-4": CLASSIFICATION_CKPT_DIR / "resnet50_balanced_lr1e-4" / "best_model.pth",
    "resnet50_balanced_lr2.5e-5": CLASSIFICATION_CKPT_DIR / "resnet50_balanced_lr2.5e-5" / "best_model.pth",
    "efficientnet_b0_balanced_lr1e-4": CLASSIFICATION_CKPT_DIR / "efficientnet_b0_balanced_lr1e-4" / "best_model.pth",
    "efficientnet_b0_balanced_lr2.5e-5": CLASSIFICATION_CKPT_DIR / "efficientnet_b0_balanced_lr2.5e-5" / "best_model.pth",
    # Segmenters trained on SD2-FR / SDXL-FR
    "deeplabv3plus": SEGMENTATION_CKPT_DIR / "deeplabv3plus" / "best_model.pth",
    "unet": SEGMENTATION_CKPT_DIR / "unet" / "best_model.pth",
}

# Filenames written by the original training notebooks, accepted as a fallback.
LEGACY_SEGMENTATION_NAMES = {
    "deeplabv3plus": "best_deeplabv3plus.pth",
    "unet": "best_unet.pth",
}

# Best combination from results/pipeline/ (EfficientNet-B0 balanced lr 2.5e-5 + DeepLabV3+).
BEST_CLASSIFIER = "efficientnet_b0_balanced_lr2.5e-5"
BEST_SEGMENTER = "deeplabv3plus"


def resolve_segmentation_checkpoint(name: str) -> Optional[Path]:
    """Return the weights file for a segmentation model, or None if it is missing.

    Prefers the canonical ``best_model.pth`` and falls back to the filename the
    training notebook originally wrote (logging a warning).
    """
    canonical = CHECKPOINTS[name]
    if canonical.exists():
        return canonical
    legacy = canonical.parent / LEGACY_SEGMENTATION_NAMES[name]
    if legacy.exists():
        logger.warning(
            "Using legacy checkpoint name %s; rename it to %s.", legacy, canonical.name
        )
        return legacy
    return None


def require_segmentation_checkpoint(name: str) -> Path:
    """Like resolve_segmentation_checkpoint, but exit with a clear message if missing."""
    path = resolve_segmentation_checkpoint(name)
    if path is None:
        raise SystemExit(
            f"Segmentation weights for '{name}' not found. Expected {CHECKPOINTS[name]} "
            f"(see {CHECKPOINTS[name].parent / 'README.md'})."
        )
    return require_weights(path)


def require_weights(path: Path) -> Path:
    """Exit with a clear message if ``path`` is missing or an un-pulled Git LFS pointer."""
    path = Path(path)
    if not path.exists():
        raise SystemExit(f"Checkpoint not found: {path}")
    if is_lfs_pointer(path):
        raise SystemExit(f"{path} is a Git LFS pointer, not the weights. Run `git lfs pull` first.")
    return path


def is_lfs_pointer(path: Path) -> bool:
    """True if ``path`` is an un-downloaded Git LFS pointer rather than real weights."""
    path = Path(path)
    if not path.is_file() or path.stat().st_size > 1024:
        return False
    with open(path, "rb") as f:
        return f.read(40).startswith(b"version https://git-lfs.github.com/spec")


def lr_tag(lr: float) -> str:
    """Format a learning rate for folder names: 2.5e-5 -> '2.5e-5', 1e-4 -> '1e-4'."""
    mantissa, exponent = f"{lr:e}".split("e")
    mantissa = mantissa.rstrip("0").rstrip(".")
    return f"{mantissa}e{int(exponent)}"
