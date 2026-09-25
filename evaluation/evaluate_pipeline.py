"""Evaluate the classifier -> segmentation pipeline (all 4 classifier/segmenter combinations).

Pipeline rule: an image is FAKE if the classifier predicts FAKE, OR if more than
`override_pct` % of its pixels have segmentation probability > `pixel_threshold`
(the segmenter can override a REAL prediction).

Default run (reproduces results/pipeline/pipeline_pixel0.7_override{30,50,70}pct.json):
    python evaluation/evaluate_pipeline.py

data/sd2-fr-testing contains ONLY fake images, so by default only the fake-detection
TPR is measured. Add real images to also measure false positives:
    python evaluation/evaluate_pipeline.py --real-dir data/sd2-classification/test/REAL
This writes *_with_real.json files with real-image accuracy, FPR, precision and F1.

(Merges the former evaluate_combinations.py, which swept pixel thresholds at a fixed
30% override, and combination_results_final.py, which swept the override % at a fixed
pixel threshold of 0.7.)
"""
import os
import sys
import glob
import json
import argparse
from pathlib import Path
import torch
import torch.nn as nn
import numpy as np
import segmentation_models_pytorch as smp
import torchvision.transforms.functional as TF
from PIL import Image
from tqdm import tqdm
from torchvision.models import resnet50, efficientnet_b0

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repo root
from authentilens.paths import CHECKPOINTS, DATA_DIR, RESULTS_DIR, require_segmentation_checkpoint

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]
SEG_SIZE = 512
CLS_SIZE = 224

COMBINATIONS = [
    ('ResNet', 'DeepLabV3Plus'),
    ('ResNet', 'UNet'),
    ('EfficientNet', 'DeepLabV3Plus'),
    ('EfficientNet', 'UNet')
]

def load_deeplab(path):
    deeplab = smp.DeepLabV3Plus(
        encoder_name='resnet50',
        encoder_weights=None,
        in_channels=3,
        classes=1,
        activation=None,
    )
    deeplab.decoder.block2 = nn.Sequential(
        deeplab.decoder.block2,
        nn.Dropout2d(p=0.3),
    )
    ckpt = torch.load(path, map_location=DEVICE)
    state = ckpt.get('model_state_dict', ckpt)

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
    deeplab.load_state_dict(state, strict=True)
    deeplab.to(DEVICE)
    deeplab.eval()
    return deeplab

def load_unet(path):
    unet = smp.Unet(
        encoder_name='resnet50',
        encoder_weights=None,
        in_channels=3,
        classes=1,
        activation=None,
        decoder_channels=(256, 128, 64, 32, 16),
        decoder_use_batchnorm=True,
    )
    dropout_p = 0.3
    for i, block in enumerate(unet.decoder.blocks):
        n = len(unet.decoder.blocks)
        scaled = dropout_p * (0.4 + 0.6 * i / max(n - 1, 1))
        block.conv2 = nn.Sequential(block.conv2, nn.Dropout2d(p=scaled))
    ckpt = torch.load(path, map_location=DEVICE)
    state = ckpt.get('model_state_dict', ckpt)
    unet.load_state_dict(state, strict=True)
    unet.to(DEVICE)
    unet.eval()
    return unet

def load_resnet(path):
    resnet = resnet50(pretrained=False)
    resnet.fc = nn.Sequential(nn.Dropout(0.3), nn.Linear(resnet.fc.in_features, 2))
    state = torch.load(path, map_location=DEVICE)
    if 'model_state_dict' in state: state = state['model_state_dict']
    cache = {k.replace('module.', ''): v for k, v in state.items()}
    resnet.load_state_dict(cache)
    resnet.to(DEVICE)
    resnet.eval()
    return resnet

def load_efficientnet(path):
    effnet = efficientnet_b0(pretrained=False)
    num_ftrs = effnet.classifier[1].in_features
    effnet.classifier[1] = nn.Linear(num_ftrs, 2)
    state = torch.load(path, map_location=DEVICE)
    if 'model_state_dict' in state: state = state['model_state_dict']
    cache = {k.replace('module.', ''): v for k, v in state.items()}
    effnet.load_state_dict(cache)
    effnet.to(DEVICE)
    effnet.eval()
    return effnet

def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--pixel-thresholds', type=float, nargs='+', default=[0.7],
                        help='Per-pixel segmentation probability threshold(s) (default: 0.7).')
    parser.add_argument('--override-pcts', type=float, nargs='+', default=[30, 50, 70],
                        help='%% of pixels above the pixel threshold needed to override REAL->FAKE (default: 30 50 70).')
    parser.add_argument('--fake-dir', type=Path, default=DATA_DIR / 'sd2-fr-testing',
                        help='Fake images, laid out as <category>/<image>.png (default: data/sd2-fr-testing).')
    parser.add_argument('--mask-dir', type=Path, default=DATA_DIR / 'sd2-fr-testing-masks',
                        help='Ground-truth masks for --fake-dir (default: data/sd2-fr-testing-masks).')
    parser.add_argument('--real-dir', type=Path, default=None,
                        help='Optional folder of REAL images (searched recursively), e.g. '
                             'data/sd2-classification/test/REAL. Enables FPR / precision / F1.')
    parser.add_argument('--output-dir', type=Path, default=RESULTS_DIR / 'pipeline',
                        help='Where to write the JSON results (default: results/pipeline).')
    return parser.parse_args()


def load_models():
    classifiers = {
        'ResNet': load_resnet(CHECKPOINTS['resnet50_balanced_lr2.5e-5']),
        'EfficientNet': load_efficientnet(CHECKPOINTS['efficientnet_b0_balanced_lr2.5e-5'])
    }
    segmenters = {
        'DeepLabV3Plus': load_deeplab(require_segmentation_checkpoint('deeplabv3plus')),
        'UNet': load_unet(require_segmentation_checkpoint('unet'))
    }
    return classifiers, segmenters


def predict(pil_image, classifiers, segmenters, pixel_thresholds, gt_mask=None):
    """Run every model once on one image.

    Returns classifier predictions (0 = FAKE, 1 = REAL) and, for each segmenter and
    pixel threshold, the flagged-pixel % plus IoU / pixel accuracy if gt_mask is given.
    """
    # Get Classifier Predictions
    img_c = pil_image.resize((CLS_SIZE, CLS_SIZE), Image.BILINEAR)
    t_c = TF.to_tensor(img_c)
    t_c = TF.normalize(t_c, mean=MEAN, std=STD).unsqueeze(0).to(DEVICE)

    c_preds = {}
    with torch.no_grad():
        for cname, cmodel in classifiers.items():
            out = cmodel(t_c)
            pred = torch.softmax(out, dim=1)[0].argmax().item()
            c_preds[cname] = pred  # 0 is FAKE, 1 is REAL

    # Get Seg Predictions
    img_s = pil_image.resize((SEG_SIZE, SEG_SIZE), Image.BILINEAR)
    t_s = TF.to_tensor(img_s)
    t_s = TF.normalize(t_s, mean=MEAN, std=STD).unsqueeze(0).to(DEVICE)

    s_preds = {}
    with torch.no_grad():
        for sname, smodel in segmenters.items():
            out = smodel(t_s)
            prob = torch.sigmoid(out)[0, 0].cpu().numpy()
            s_preds[sname] = {}
            for pixel_threshold in pixel_thresholds:
                bin_map = (prob > pixel_threshold)
                iou = 0.0
                pixel_acc = 0.0
                if gt_mask is not None:
                    intersection = np.logical_and(bin_map, gt_mask).sum()
                    union = np.logical_or(bin_map, gt_mask).sum()
                    iou = intersection / union if union > 0 else (1.0 if not gt_mask.any() and not bin_map.any() else 0.0)
                    pixel_acc = (bin_map == gt_mask).mean()
                flagged_pct = float(bin_map.mean()) * 100
                s_preds[sname][pixel_threshold] = {"iou": iou, "pixel_acc": pixel_acc, "flagged_pct": flagged_pct}
    return c_preds, s_preds


def run_fake_images(args, classifiers, segmenters):
    img_paths = glob.glob(f'{args.fake_dir}/*/*.png')
    records = []
    for img_path in tqdm(img_paths, desc="Fake images"):
        try:
            pil_image = Image.open(img_path).convert('RGB')
        except Exception:
            continue

        cat = os.path.basename(os.path.dirname(img_path))
        fname = os.path.basename(img_path)
        base = fname.split('_sd2')[0]
        mask_path = os.path.join(args.mask_dir, cat, base.replace('.png', '_512.png'))

        gt_mask = None
        if os.path.exists(mask_path):
            gt_mask = Image.open(mask_path).convert('L').resize((SEG_SIZE, SEG_SIZE), Image.NEAREST)
            gt_mask = np.array(gt_mask) > 127

        c_preds, s_preds = predict(pil_image, classifiers, segmenters, args.pixel_thresholds, gt_mask)
        records.append((c_preds, s_preds, gt_mask is not None))
    return records


def run_real_images(args, classifiers, segmenters):
    img_paths = sorted(p for ext in ('png', 'jpg', 'jpeg') for p in glob.glob(f'{args.real_dir}/**/*.{ext}', recursive=True))
    records = []
    for img_path in tqdm(img_paths, desc="Real images"):
        try:
            pil_image = Image.open(img_path).convert('RGB')
        except Exception:
            continue
        c_preds, s_preds = predict(pil_image, classifiers, segmenters, args.pixel_thresholds)
        records.append((c_preds, s_preds))
    return records


def _precision_f1(tp, fp, fn):
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return precision, f1


def summarize(fake_records, real_records, pixel_threshold, override_pct):
    summary = {}
    for cname, sname in COMBINATIONS:
        classifier_correct = 0
        pipeline_fake_detected = 0
        seg_iou, seg_pixel_acc = [], []
        for c_preds, s_preds, has_mask in fake_records:
            seg = s_preds[sname][pixel_threshold]
            is_cls_fake = (c_preds[cname] == 0)
            if is_cls_fake:
                classifier_correct += 1
            if has_mask:
                seg_iou.append(seg["iou"])
                seg_pixel_acc.append(seg["pixel_acc"])
            # Pipeline logic: FAKE if the classifier says FAKE, or segmentation flags > override_pct % of pixels
            is_seg_fake = seg["flagged_pct"] > override_pct
            if is_cls_fake or is_seg_fake:
                pipeline_fake_detected += 1

        total = len(fake_records)
        res = {
            "total_images": total,
            "classifier_fake_TPR": classifier_correct / total if total > 0 else 0,
            "segmentation_mIoU": float(np.mean(seg_iou)) if len(seg_iou) > 0 else 0.0,
            "segmentation_pixel_accuracy": float(np.mean(seg_pixel_acc)) if len(seg_pixel_acc) > 0 else 0.0,
            "combined_pipeline_TPR": pipeline_fake_detected / total if total > 0 else 0
        }

        if real_records is not None:
            n_real = len(real_records)
            cls_fp = sum(1 for c_preds, _ in real_records if c_preds[cname] == 0)
            comb_fp = sum(1 for c_preds, s_preds in real_records
                          if c_preds[cname] == 0 or s_preds[sname][pixel_threshold]["flagged_pct"] > override_pct)
            for prefix, tp, fp in (("classifier", classifier_correct, cls_fp), ("combined", pipeline_fake_detected, comb_fp)):
                tpr = tp / total if total > 0 else 0.0
                fpr = fp / n_real if n_real > 0 else 0.0
                precision, f1 = _precision_f1(tp, fp, total - tp)
                res[f"{prefix}_real_accuracy"] = 1.0 - fpr
                res[f"{prefix}_FPR"] = fpr
                res[f"{prefix}_precision"] = precision
                res[f"{prefix}_F1"] = f1
                res[f"{prefix}_balanced_accuracy"] = (tpr + 1.0 - fpr) / 2
            res["real_images"] = n_real

        summary[f"{cname}_{sname}"] = res
    return summary


def main():
    args = parse_args()
    if not args.fake_dir.is_dir():
        raise SystemExit(f"Fake image folder not found: {args.fake_dir} (see data/README.md)")
    if args.real_dir is not None and not args.real_dir.is_dir():
        raise SystemExit(f"Real image folder not found: {args.real_dir}")

    classifiers, segmenters = load_models()
    fake_records = run_fake_images(args, classifiers, segmenters)
    real_records = run_real_images(args, classifiers, segmenters) if args.real_dir is not None else None

    args.output_dir.mkdir(parents=True, exist_ok=True)
    suffix = "_with_real" if real_records is not None else ""
    for pixel_threshold in args.pixel_thresholds:
        for override_pct in args.override_pcts:
            summary = summarize(fake_records, real_records, pixel_threshold, override_pct)
            out_path = args.output_dir / f"pipeline_pixel{pixel_threshold:g}_override{override_pct:g}pct{suffix}.json"
            with open(out_path, 'w') as f:
                json.dump(summary, f, indent=4)
            print(f"\nPixel threshold {pixel_threshold}, override > {override_pct:g}% of pixels -> {out_path}")
            print(json.dumps(summary, indent=4))


if __name__ == '__main__':
    main()
