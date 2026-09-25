# AuthentiLens Project Report

Last updated: 2026-04-15

## 1. Project Overview

**Project title:** AuthentiLens - Visualizing AI Manipulations  
**Team name:** Aperture  
**Team members:**
- Divyanshu Yadav (2023211)
- Garvit (2023217)
- Rewant Anand (2023429)
- Tanish Bachhas (2023545)

### 1.1 Abstract

AuthentiLens is a computer vision system for image forgery analysis focused on diffusion-era manipulations. Instead of only predicting whether an image is fake, the system combines **global classification** (FAKE vs REAL) with **pixel-level localization** (tampered-region heatmap). The project was built around a key challenge observed during experiments: standard classifiers can appear highly accurate on imbalanced fake-only test sets while failing badly on realistic balanced evaluation. To address this, the team created a balanced SD2-derived classification dataset and built a hybrid decision pipeline that uses segmentation as an error-correction fallback. The final combined system significantly improves fake detection TPR on manipulated images.

## 2. Problem Statement and Scope

Recent generative models (for example diffusion-based systems) make it easy to create realistic edits such as inpainting and outpainting. In practical abuse cases, only parts of an image are manipulated. Many existing detectors are optimized for whole-image real/fake classification and do not localize where manipulation occurred.

**Project goal:**
- Detect whether an input image is likely manipulated.
- Localize suspicious manipulated regions using a segmentation mask.
- Deliver results through an interactive web interface.

**Scope constraints:**
- Focus on photorealistic, diffusion-style manipulations.
- Not designed for traditional copy-move/splice-only forensic settings.
- Not designed for highly stylized or artistic synthetic images.

## 3. Prior Work Context and Design Rationale

The project references modern synthetic image forensics directions (for example reconstruction/frequency-based families such as DIRE/FIRE and artifact localization approaches). A practical design choice in this project was to first establish a strong pixel-domain baseline using conventional CV backbones:
- Classification: ResNet-50, EfficientNet-B0
- Segmentation: DeepLabV3Plus, UNet

This lets us measure how far spatial/semantic learning can go before adding heavier frequency/noise priors.

## 4. Datasets and Data Engineering

## 4.1 Datasets used

1. **SD2-FR subset (from TGIF2 family):**
- Used for manipulation localization and fake-image evaluation.
- Includes image-mask pairs.

2. **Custom SD2 classification dataset (`sd2-classification`):**
- Created in this project to solve class imbalance and label bias issues.
- Balanced FAKE/REAL crops extracted from SD2 source images.

## 4.2 Why custom dataset was necessary

An early issue: the SD2-FR testing split used for some checks contained only fake images. On that data, a classifier can score very high by predicting FAKE almost always. This behavior was confirmed by CIFAKE-trained models during evaluation.

## 4.3 Custom dataset generation method

Implemented in `process_sd2_dataset.py`:
- For each image and mask, extract one **FAKE** 224x224 crop centered on manipulated area.
- Extract one **REAL** 224x224 crop with zero mask overlap.
- Keep only valid pairs and balance classes 1:1.
- Save metadata including source path and crop bounding boxes.

## 4.4 Final split sizes (`README.md`)

| Split | FAKE | REAL | Total |
|---|---:|---:|---:|
| Train | 5567 | 5567 | 11134 |
| Val | 798 | 798 | 1596 |
| Test | 816 | 816 | 1632 |
| **Total** | **7181** | **7181** | **14362** |

## 5. Methodology

## 5.1 Classification training

Training scripts:
- `train_resnet_sd.py`
- `train_efn_sd.py`

Shared characteristics:
- Input size: 224x224
- Optimizer: AdamW
- LR: 2.5e-5
- Weight decay: 1e-4
- Epochs: 15
- Batch size: 256
- Augmentations: resize, horizontal flip, color jitter
- Normalization: ImageNet mean/std
- Mixed precision (AMP): enabled

## 5.2 Segmentation modeling

Inference/evaluation uses:
- DeepLabV3Plus (`segmentation_models_pytorch`)
- UNet (`segmentation_models_pytorch`)

Segmentation checkpoints used by the project:
- `checkpoints_deeplab/best_model.pth`
- `checkpoints_unet/best_model.pth`

## 5.3 Combined heuristic pipeline

In combination evaluation (`evaluate_combinations.py`):
- Classifier predicts FAKE/REAL.
- Segmenter predicts mask probabilities.
- If classifier misses fake but segmented suspicious region is large enough, pipeline flags FAKE.
- For threshold-0.7 experiments, fake decision fallback used flagged pixel percentage criterion (>30%).

## 6. System Architecture and Implementation

## 6.1 Training and evaluation components

- Dataset creation: `process_sd2_dataset.py`
- Classifier training: `train_resnet_sd.py`, `train_efn_sd.py`
- Comprehensive classifier evaluation: `evaluate_all_models.py`
- Combined pipeline evaluation: `evaluate_combinations.py`
- IMD2020 API-based localization evaluator: `evaluate_imd2020.py`

## 6.2 Inference stack (microservices)

- **Backend:** FastAPI (`backend/main.py`) with `/health` and `/analyze`
- **Model runtime:** `backend/inference.py`
- **Frontend:** Streamlit client (`frontend/app.py`)
- **Orchestration:** `docker-compose.yml`

## 6.3 Standalone demo app

`app.py` provides a richer local UI with:
- Multi-image upload
- Model selection menu (classification + segmentation)
- Per-model outputs
- Heuristic final decision
- Heatmap/mask visualization
- Runtime and system metrics

## 7. Experimental Results

Metrics source files:
- `comprehensive_evaluation_results.json`
- `combinations_test_results_0.7.json`

## 7.1 Classification results on balanced SD2 test set (1632 images)

| Model | Overall Acc (%) | Fake Acc (%) | Real Acc (%) |
|---|---:|---:|---:|
| ResNet_CIFake | 50.43 | 99.14 | 1.72 |
| EfficientNet_CIFake | 50.00 | 67.40 | 32.60 |
| ResNet_SD_1e-4 | 61.95 | 57.97 | 65.93 |
| ResNet_SD_2.5e-5 | **63.11** | 61.03 | 65.20 |
| EfficientNet_SD_1e-4 | 60.23 | 55.15 | 65.32 |
| EfficientNet_SD_2.5e-5 | 61.27 | 50.86 | **71.69** |

**Key observation:** CIFAKE-trained models do not generalize to balanced partial-manipulation detection.

## 7.2 Accuracy on SD2-FR all-fake test set (1029 images)

| Model | Accuracy on fake-only set (%) |
|---|---:|
| ResNet_CIFake | 100.00 |
| EfficientNet_CIFake | 68.22 |
| ResNet_SD_1e-4 | 61.22 |
| ResNet_SD_2.5e-5 | 58.99 |
| EfficientNet_SD_1e-4 | 71.14 |
| EfficientNet_SD_2.5e-5 | **71.43** |

**Interpretation:** High scores on fake-only sets can hide severe real-class failure.

## 7.3 Combined pipeline results (threshold = 0.7)

| Combination | Classifier Fake TPR (%) | Seg mIoU (%) | Seg Pixel Acc (%) | Combined Pipeline TPR (%) |
|---|---:|---:|---:|---:|
| ResNet + DeepLabV3Plus | 58.99 | 16.76 | 61.76 | 94.85 |
| ResNet + UNet | 58.99 | 17.28 | 63.30 | 86.59 |
| EfficientNet + DeepLabV3Plus | 71.43 | 16.76 | 61.76 | **96.70** |
| EfficientNet + UNet | 71.43 | 17.28 | 63.30 | 91.74 |

**Main outcome:** Hybrid decision logic substantially improves fake detection TPR versus classifier-only performance.

## 8. Discussion and Analysis

1. The largest methodological contribution is dataset correction (balanced FAKE/REAL extraction) that exposed and reduced classifier bias.
2. Classifier-only performance remains moderate (~60-63% overall), but segmentation-assisted logic strongly improves manipulated-image detection sensitivity.
3. Segmentation quality (mIoU around 0.17) shows localization signal exists but is still coarse; this is a key area for future improvement.

## 9. Limitations

1. Segmentation mIoU remains low for precise localization needs.
2. Some scripts are skeleton/prototype style (for example `train_deeplab.py` requires dataset path population).
3. Domain coverage is concentrated around SD2-style manipulations; broader cross-dataset robustness is still open.
4. Docker compose paths may require alignment with available checkpoint folder names in this repository.

## 10. Future Work

1. End-to-end joint training between classification and segmentation branches.
2. Add frequency/noise residual features inspired by modern forensics literature.
3. Integrate additional SOTA forgery detectors for side-by-side comparison.
4. Improve calibration and threshold tuning using validation-driven operating points.
5. Expand cross-domain testing (different generators, compression levels, real-world post-processing).

## 11. Reproducibility Guide

## 11.1 Environment

- Python 3.10+
- PyTorch, torchvision
- segmentation-models-pytorch
- Streamlit / FastAPI / Uvicorn

## 11.2 Typical workflow

1. Generate balanced dataset:
```bash
python process_sd2_dataset.py
```

2. Train classifiers:
```bash
python train_resnet_sd.py
python train_efn_sd.py
```

3. Evaluate classifiers:
```bash
python evaluate_all_models.py
```

4. Evaluate combined pipeline:
```bash
python evaluate_combinations.py
```

5. Run standalone interactive app:
```bash
streamlit run app.py
```

6. Run microservice stack (optional):
```bash
docker compose up --build
```

## 12. Team Contributions (as documented)

- **Divyanshu:** segmentation datasets, segmentation model tuning/evaluation, web interface.
- **Garvit:** classification dataset research, classifier tuning/evaluation, custom dataset creation, pipeline evaluation, web interface.
- **Rewant:** model and dataset research.
- **Tanish:** model and dataset research.

## 13. Conclusion

AuthentiLens demonstrates a practical dual-purpose forensic pipeline: image-level authenticity prediction plus tamper localization. The project identifies a common evaluation trap (fake-only benchmarks), corrects it with balanced dataset construction, and shows that hybrid classifier-segmenter reasoning can dramatically improve fake detection TPR in diffusion-manipulation settings. The next major step is improving localization precision and generalization through joint learning and broader forensic feature design.
