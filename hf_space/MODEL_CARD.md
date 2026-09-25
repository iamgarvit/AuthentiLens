---
license: mit
tags:
  - image-classification
  - image-segmentation
  - image-forensics
  - inpainting-detection
  - deepfake-detection
  - pytorch
library_name: pytorch
pipeline_tag: image-segmentation
---

# AuthentiLens weights

Model weights for [**AuthentiLens**](https://github.com/iamgarvit/AuthentiLens), which
detects **diffusion-inpainted photos** and localises the region that was edited.

A classifier decides FAKE vs REAL on the whole image, and a segmentation network
predicts a per-pixel probability that the pixel was inpainted. When the
classifier says REAL but the segmenter flags more than 30% of the pixels, the
pipeline overrides the call to FAKE — that override is what catches partial
edits an image-level classifier misses.

Try it in the browser: **[aperture-lens.streamlit.app](https://aperture-lens.streamlit.app)**.

## Files

| File | Model | Size | Purpose |
|---|---|---:|---|
| `classification/efficientnet_b0_balanced_lr2.5e-5/best_model.pth` | EfficientNet-B0 | 16 MB | Image-level FAKE/REAL classifier. The default. |
| `segmentation/deeplabv3plus/best_model.pth` | DeepLabV3+ (ResNet-50) | 107 MB | Per-pixel inpainting mask. The default. |
| `segmentation/unet/best_model.pth` | UNet (ResNet-50) | 130 MB | Alternative segmenter. |
| `original/deeplabv3plus_original.pth` | DeepLabV3+ | 321 MB | Backup: the full training checkpoint. |
| `original/unet_original.pth` | UNet | 391 MB | Backup: the full training checkpoint. |

The files under `classification/` and `segmentation/` are **weights-only**: a
bare `state_dict`, with the training checkpoint's `optimizer_state_dict`,
`epoch`, `val_iou` and `config` removed. The tensors are untouched, so they
produce bit-identical outputs to the originals under `original/`, at a third of
the size. The `original/` files are kept only so the training state is not lost.

### Checksums

| File | sha256 |
|---|---|
| `classification/efficientnet_b0_balanced_lr2.5e-5/best_model.pth` | `845be70c7347acce3075fd43f2e74a0e4dd08a4c80c228e464bb7608a5cf8aed` |
| `segmentation/deeplabv3plus/best_model.pth` | `574b34efa947905899f024cf5010a674ce45aa15e3fde63f58e56704ef97a13f` |
| `segmentation/unet/best_model.pth` | `ef9af814c1b85a88e5e3e7b9a6cb97633c650257c81966d2d4de735d1ffe5d5e` |
| `original/deeplabv3plus_original.pth` | `5a73fdd06aa7166a7756fbd197320b53fc68a4ccf8bca18b7d8702e1dcecc60e` |
| `original/unet_original.pth` | `a399c342e4246480567b662ed1831e7b77cd00b1cee3511954ea9c24f0290cb7` |

## The pipeline

1. **Classify.** The image is resized to 224x224 (PIL bilinear), ImageNet-normalised,
   and EfficientNet-B0 predicts FAKE (class 0) or REAL (class 1).
2. **Segment.** The image is resized to 512x512, and DeepLabV3+ predicts a
   sigmoid probability per pixel. A pixel is *flagged* when P(inpainted) > **0.7**.
3. **Override.** If the classifier said REAL but more than **30%** of pixels are
   flagged, the output is FAKE.

## Training

Both segmenters: ResNet-50 encoder (ImageNet init), 512x512 inputs, batch 32,
lr 1e-4, weight decay 5e-3, decoder dropout 0.3, trained on **SD2-FR + SDXL-FR**.
The published checkpoints are from epoch 8 (best validation IoU: 0.7264 for
DeepLabV3+, 0.7249 for UNet).

The classifier was fine-tuned from ImageNet on a **balanced 224x224 patch
dataset** built from SD2-FR: for each inpainted image, one FAKE crop centred on
the inpainting mask and one REAL crop with zero mask overlap from the same
photo. Splits: 11,134 train / 1,596 val / 1,632 test. Learning rate 2.5e-5.

That balanced set exists because fake-only benchmarks hide classifier bias: a
CIFAKE-trained ResNet-50 scores 100% on the fake-only SD2-FR test set but
**1.72%** real-image accuracy on the balanced set — it labels almost everything
FAKE.

## Results

All figures below are from the JSON files in
[`results/`](https://github.com/iamgarvit/AuthentiLens/tree/main/results),
**as of the 11 April 2026 snapshot** of the project.

### Combined pipeline

Fake-detection TPR on the 1,029-image SD2-FR test set, pixel threshold 0.7.

| Classifier + segmenter | Classifier alone | Override > 30% | > 50% | > 70% |
|---|---:|---:|---:|---:|
| **EfficientNet-B0 + DeepLabV3+** | 71.43% | **96.70%** | 76.77% | 71.43% |
| EfficientNet-B0 + UNet | 71.43% | 91.74% | 77.45% | 72.40% |
| ResNet-50 + DeepLabV3+ | 58.99% | 94.85% | 67.15% | 59.18% |
| ResNet-50 + UNet | 58.99% | 86.59% | 67.54% | 59.67% |

### Classifier, on its own

| Test set | Metric | EfficientNet-B0 (lr 2.5e-5) |
|---|---|---:|
| SD2-FR test (1,029, all fake) | Accuracy | 71.43% |
| Balanced patches (1,632) | Overall accuracy | 61.27% |
| Balanced patches | FAKE accuracy | 50.86% |
| Balanced patches | REAL accuracy | 71.69% |

### Segmenters, on their own

On SD2-FR test images with ground-truth masks, pixel threshold 0.7.

| Segmenter | mIoU | Pixel accuracy | Images |
|---|---:|---:|---:|
| DeepLabV3+ | 16.76% | 61.76% | 1,029 |
| UNet | 17.28% | 63.30% | 1,029 |

The low mIoU and the high combined TPR are consistent: the override only needs
the flagged **area** to exceed 30%, which is a much weaker requirement than
matching the mask's shape.

## Limitations

- **The false-positive rate is not measured.** The pipeline was evaluated on a
  fake-only test set, so the 96.70% is a true-positive rate with no
  corresponding false-positive rate. A segmenter that flags large areas on real
  photos would inflate it. Do not read these numbers as accuracy.
- **Narrow domain.** Trained on Stable Diffusion 2 and SDXL inpainting of
  photoreal images. Not trained for GAN images, fully synthetic images, artwork,
  or other inpainting tools.
- **The task is genuinely hard.** The classifier alone is near chance on the
  balanced set (61.27%), and human majority vote scored 44.22% on the fake-only
  SD2-FR images and 60.17% on the balanced patches.
- Not suitable for any consequential decision about whether an image is genuine.

## Usage

```python
import torch, segmentation_models_pytorch as smp, torch.nn as nn
from huggingface_hub import hf_hub_download
from torchvision.models import efficientnet_b0

REPO = "iamgarvit/authentilens-weights"

classifier = efficientnet_b0(weights=None)
classifier.classifier[1] = nn.Linear(classifier.classifier[1].in_features, 2)
classifier.load_state_dict(torch.load(hf_hub_download(
    REPO, "classification/efficientnet_b0_balanced_lr2.5e-5/best_model.pth"),
    map_location="cpu"))

segmenter = smp.DeepLabV3Plus(encoder_name="resnet50", encoder_weights=None,
                              in_channels=3, classes=1, activation=None)
# The training notebook added decoder dropout this way; recreating it makes the
# checkpoint's key names line up, so strict=True succeeds.
segmenter.decoder.block = nn.Sequential(nn.Dropout2d(p=0.3), segmenter.decoder.block2)
segmenter.load_state_dict(torch.load(hf_hub_download(
    REPO, "segmentation/deeplabv3plus/best_model.pth"), map_location="cpu"))
```

`authentilens/models.py` in the repository does all of this, including the
decision rule — prefer importing it over copying the snippet.

## Credit

Course project for **CSE344 Computer Vision, IIIT Delhi** (team Aperture).

- **Divyanshu Yadav** — segmentation datasets, segmentation models, pipeline ablations, web interface
- **Garvit** — classification datasets, classification models, the balanced dataset, pipeline ablations, web interface
- **Rewant Anand** and **Tanish Bachhas** — literature review, SOTA models, NoisePrint analysis, frontend template

Built on the **TGIF / SD2-FR** dataset (H. Mareen et al., WIFS 2024,
[arXiv:2407.11566](https://arxiv.org/abs/2407.11566)), with **ResNet**,
**EfficientNet**, **U-Net** and **DeepLabV3+** architectures.
