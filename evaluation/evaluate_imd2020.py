"""
============================================================================
AuthentiLens — IMD2020 Evaluation Script
============================================================================
Standalone script to evaluate spatial localization accuracy using the
IMD2020 dataset (or appropriately structured custom datasets).

Requires the FastAPI backend (services/backend) to be running.

Metrics:
    1. Mean Intersection over Union (mIoU)
    2. Pixel-wise Average Precision (AP)
============================================================================
"""

import argparse
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import requests
from sklearn.metrics import average_precision_score
from tqdm import tqdm

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repo root
from authentilens.paths import DATA_DIR

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger("evaluator")

# Disable sklearn undefined metric warnings (if ground truth is empty etc.)
import warnings
warnings.filterwarnings("ignore", category=UserWarning)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Evaluate AuthentiLens spatial localization (mIoU, AP)."
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=str(DATA_DIR / "IMD2020"),
        help="Path to the real IMD2020 root dataset directory containing subfolders (default: data/IMD2020).",
    )
    parser.add_argument(
        "--endpoint",
        type=str,
        default="http://localhost:8000/analyze",
        help="Backend API endpoint URL (default: http://localhost:8000/analyze).",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Threshold to binarize predicted attention map for IoU (default: 0.5).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit the number of image pairs evaluated (e.g., 100).",
    )
    return parser.parse_args()


def get_imd2020_pairs(dataset_root: Path) -> List[Tuple[Path, Path]]:
    """
    Traverse the official IMD2020 dataset directory structure.
    Expected structure inside root:
      subdir/
        |- <folder_name>_orig.jpg (or similar)   -> Ignore (Original)
        |- <manip_name>.jpg                      -> Manipulated Image
        |- <manip_name>_mask.png                 -> Ground Truth Mask
    """
    pairs = []
    
    # Iterate through all subdirectories in the dataset root
    for subdir in dataset_root.iterdir():
        if not subdir.is_dir():
            continue
            
        files = list(subdir.iterdir())
        
        # Identify all mask files (typically end with _mask)
        mask_files = [f for f in files if "_mask" in f.stem]
        
        for mask_file in mask_files:
            # The manipulated image has the exact same stem as the mask, minus "_mask"
            manipulated_stem = mask_file.stem.replace("_mask", "")
            
            manipulated_img = next(
                (f for f in files if f.stem == manipulated_stem and f != mask_file), 
                None
            )
            
            if manipulated_img and manipulated_img.exists():
                pairs.append((manipulated_img, mask_file))
            else:
                logger.debug(f"Could not find manipulated image for mask {mask_file.name} in {subdir.name}")
            
    return pairs


def calculate_iou(pred_mask: np.ndarray, gt_mask: np.ndarray) -> float:
    """Calculate Intersection over Union (IoU) for binary masks."""
    intersection = np.logical_and(pred_mask, gt_mask).sum()
    union = np.logical_or(pred_mask, gt_mask).sum()
    
    if union == 0:
        return 1.0 if intersection == 0 else 0.0
        
    return float(intersection) / float(union)


def analyze_image_via_api(img_path: Path, endpoint: str) -> Optional[np.ndarray]:
    """POST image to API and extract the predicted mask probability map."""
    try:
        with open(img_path, "rb") as f:
            files = {"file": (img_path.name, f, "image/jpeg")}
            response = requests.post(endpoint, files=files, timeout=60)
            
        response.raise_for_status()
        data = response.json()
        mask = np.array(data["mask"], dtype=np.float32)
        # An empty mask means segmentation did not run; treat as "nothing flagged".
        return mask if mask.size else np.zeros((1, 1), dtype=np.float32)
        
    except requests.exceptions.RequestException as e:
        logger.error(f"API Request failed for {img_path.name}: {e}")
        return None
    except KeyError:
        logger.error(f"API Response missing 'mask' for {img_path.name}")
        return None


def main() -> None:
    args = parse_args()
    
    dataset_dir = Path(args.dataset)
    
    if not dataset_dir.is_dir():
        logger.error(f"Dataset directory '{dataset_dir}' does not exist.")
        return

    pairs = get_imd2020_pairs(dataset_dir)
    if not pairs:
        logger.error(f"Found no matching image-mask pairs in directories.")
        return
        
    if args.limit:
        pairs = pairs[:args.limit]
        
    logger.info(f"Found {len(pairs)} image-mask pairs. Beginning evaluation...")
    
    iou_scores: List[float] = []
    ap_scores: List[float] = []
    failed_requests = 0
    
    # Progress bar and aggregate accumulators
    pbar = tqdm(pairs, desc="Evaluating", unit="img")
    
    for img_path, mask_path in pbar:
        # Load Ground Truth Mask (Grayscale)
        gt_mask_bgr = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if gt_mask_bgr is None:
            logger.warning(f"Failed to read GT mask {mask_path.name}. Skipping.")
            continue
            
        # Ground truth is typically [0, 255]. Binarize to {0, 1}
        gt_h, gt_w = gt_mask_bgr.shape
        gt_binary = (gt_mask_bgr > 127).astype(np.uint8)
        
        # 1. Prediction via API
        pred_attn_matrix = analyze_image_via_api(img_path, args.endpoint)
        if pred_attn_matrix is None:
            failed_requests += 1
            continue
            
        # 2. Resize prediction to match Ground Truth
        # Note: the backend returns a [512, 512] probability map
        pred_resized = cv2.resize(
            pred_attn_matrix, 
            (gt_w, gt_h), 
            interpolation=cv2.INTER_LINEAR
        )
        
        # 3. Metric: Mean Intersection over Union (mIoU)
        pred_binary = (pred_resized >= args.threshold).astype(np.uint8)
        iou = calculate_iou(pred_binary, gt_binary)
        iou_scores.append(iou)
        
        # 4. Metric: Pixel-wise Average Precision (AP)
        pred_flat = pred_resized.flatten()
        gt_flat = gt_binary.flatten()
        
        # If the GT has no manipulation (all 0s), AP is typically undefined for binary ranking,
        # but in typical benchmark scenarios, IMD2020 has a manipulated region.
        # Fallback handling just in case:
        if gt_flat.sum() == 0:
            ap = 1.0 if pred_flat.max() < args.threshold else 0.0
        else:
            ap = average_precision_score(gt_flat, pred_flat)
            
        ap_scores.append(ap)
        
        # Update running averages on progress bar
        pbar.set_postfix({
            "mIoU": f"{np.mean(iou_scores):.4f}", 
            "AP": f"{np.mean(ap_scores):.4f}"
        })
        
    # Final Reporting
    print("\n" + "="*50)
    print(" AuthentiLens — IMD2020 Evaluation Results")
    print("="*50)
    
    if iou_scores and ap_scores:
        final_miou = np.mean(iou_scores)
        final_ap = np.mean(ap_scores)
        
        print(f" Total Pairs Evaluated : {len(iou_scores)} / {len(pairs)}")
        print(f" Failed API Requests   : {failed_requests}")
        print(f" Threshold Used        : {args.threshold}")
        print("-" * 50)
        print(f" Mean IoU (mIoU)       : {final_miou:.4f}  ({final_miou * 100:.2f}%)")
        print(f" Pixel-wise AP         : {final_ap:.4f}  ({final_ap * 100:.2f}%)")
    else:
        print(" No metrics computed. All requests or predictions failed.")
        
    print("="*50 + "\n")


if __name__ == "__main__":
    main()
