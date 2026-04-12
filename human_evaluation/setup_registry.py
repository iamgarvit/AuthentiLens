"""
setup_registry.py — Build image registries for the human evaluation system.

Scans both datasets and creates JSON registry files that map opaque UUIDs
to image file paths (and ground truth labels where available).

These registries are SERVER-SIDE ONLY and must never be served to clients.

Usage:
    python setup_registry.py
"""

import json
import os
import uuid
from pathlib import Path

# ── Dataset paths (relative to cv_project root) ─────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
SD2_FR_DIR = PROJECT_ROOT / "sd2-fr-testing"
CUSTOM_TEST_DIR = PROJECT_ROOT / "sd2-classification" / "test"

# ── Output paths ─────────────────────────────────────────────────────────────
DATA_DIR = Path(__file__).resolve().parent / "data"
SD2_FR_REGISTRY = DATA_DIR / "sd2_fr_registry.json"
CUSTOM_REGISTRY = DATA_DIR / "custom_registry.json"

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}


def scan_sd2_fr():
    """
    Scan sd2-fr-testing directory.
    Structure: sd2-fr-testing/<category>/<image_file>
    All images are FAKE (ground truth), but we store it server-side only.
    """
    registry = {}
    if not SD2_FR_DIR.exists():
        print(f"WARNING: sd2-fr-testing directory not found at {SD2_FR_DIR}")
        return registry

    for category_dir in sorted(SD2_FR_DIR.iterdir()):
        if not category_dir.is_dir():
            continue
        category_name = category_dir.name
        for img_file in sorted(category_dir.iterdir()):
            if img_file.suffix.lower() in IMAGE_EXTENSIONS and img_file.is_file():
                uid = str(uuid.uuid4())
                registry[uid] = {
                    "path": str(img_file.resolve()),
                    "ground_truth": "FAKE",  # All images in sd2-fr-testing are fake
                    "category": category_name,
                    "filename": img_file.name,
                }
    return registry


def scan_custom_test():
    """
    Scan sd2-classification/test directory.
    Structure: sd2-classification/test/{REAL,FAKE}/<image_file>
    Ground truth is the subdirectory name (REAL or FAKE).
    Also extractable from filename suffix (_real.png / _fake.png).
    """
    registry = {}
    if not CUSTOM_TEST_DIR.exists():
        print(f"WARNING: sd2-classification/test directory not found at {CUSTOM_TEST_DIR}")
        return registry

    for label_dir in sorted(CUSTOM_TEST_DIR.iterdir()):
        if not label_dir.is_dir():
            continue
        label = label_dir.name.upper()  # "REAL" or "FAKE"
        if label not in ("REAL", "FAKE"):
            continue

        for img_file in sorted(label_dir.iterdir()):
            if img_file.suffix.lower() in IMAGE_EXTENSIONS and img_file.is_file():
                # Extract category from filename (e.g., "bear_572517_..." → "bear")
                fname = img_file.name
                # Category is the first part before the first underscore+number pattern
                parts = fname.split("_")
                # Handle multi-word categories like "tennis racket", "sports ball"
                # The pattern is: <category>_<coco_id>_mask_...
                # Find the index where a numeric COCO ID starts
                category_parts = []
                for p in parts:
                    if p.isdigit():
                        break
                    category_parts.append(p)
                category = " ".join(category_parts) if category_parts else "unknown"

                uid = str(uuid.uuid4())
                registry[uid] = {
                    "path": str(img_file.resolve()),
                    "ground_truth": label,
                    "category": category,
                    "filename": img_file.name,
                }
    return registry


def main():
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    # Build sd2-fr-testing registry
    print("Scanning sd2-fr-testing...")
    sd2_fr = scan_sd2_fr()
    with open(SD2_FR_REGISTRY, "w") as f:
        json.dump(sd2_fr, f, indent=2)
    print(f"  → {len(sd2_fr)} images registered → {SD2_FR_REGISTRY}")

    # Build custom test registry
    print("Scanning sd2-classification/test...")
    custom = scan_custom_test()
    with open(CUSTOM_REGISTRY, "w") as f:
        json.dump(custom, f, indent=2)
    print(f"  → {len(custom)} images registered → {CUSTOM_REGISTRY}")

    # Summary
    print("\n── Summary ──")
    print(f"sd2-fr-testing:        {len(sd2_fr)} images (all FAKE)")
    if custom:
        real_count = sum(1 for v in custom.values() if v["ground_truth"] == "REAL")
        fake_count = sum(1 for v in custom.values() if v["ground_truth"] == "FAKE")
        print(f"sd2-classification/test: {len(custom)} images ({real_count} REAL, {fake_count} FAKE)")

    # Initialize empty vote files if they don't exist
    results_dir = Path(__file__).resolve().parent / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    for vote_file, registry in [
        (results_dir / "sd2_fr_votes.json", sd2_fr),
        (results_dir / "custom_votes.json", custom),
    ]:
        if not vote_file.exists():
            # Initialize with empty vote records for each image
            votes = {}
            for uid in registry:
                votes[uid] = {
                    "votes": [],
                    "total_real_votes": 0,
                    "total_fake_votes": 0,
                }
            with open(vote_file, "w") as f:
                json.dump(votes, f, indent=2)
            print(f"  Initialized {vote_file}")
        else:
            print(f"  {vote_file} already exists, skipping")


if __name__ == "__main__":
    main()
