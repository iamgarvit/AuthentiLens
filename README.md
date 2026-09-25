# AuthentiLens

**Detects AI-inpainted images and highlights the regions that were edited.** A classifier decides REAL vs FAKE, and a segmentation network localises the inpainted pixels and can overturn the classifier when it misses a partial edit.

**Model weights:** [iamgarvit/authentilens-weights](https://huggingface.co/iamgarvit/authentilens-weights) &nbsp;·&nbsp; **Live demo:** https://aperture-lens.streamlit.app

![AuthentiLens demo](docs/images/demo.png)

> Course project for **CSE344 Computer Vision, IIIT Delhi** (team Aperture).

---

## The problem

Diffusion models such as Stable Diffusion 2 and SDXL can **inpaint** a region of a real photo: a text prompt replaces an object and leaves the rest of the photo untouched. Most "AI image detectors" are trained on **fully synthetic** images (e.g. CIFAKE) and make one image-level call. In an inpainted image most pixels are genuine camera pixels, so:

- an image-level classifier sees mostly real content and often says REAL, and
- even when it says FAKE, it can't tell you **where** the manipulation is.

AuthentiLens pairs image-level classification with pixel-level localisation of the inpainted region.

## How it works

```mermaid
flowchart LR
    A[Input image] --> B["Classifier<br/>ResNet-50 / EfficientNet-B0<br/>(224×224)"]
    A --> C["Segmenter<br/>UNet / DeepLabV3+<br/>(512×512)"]
    B -->|FAKE| F[FAKE]
    B -->|REAL| D{"&gt; 30% of pixels with<br/>P(inpainted) &gt; 0.7?"}
    C --> D
    C --> H[Heatmap of edited region]
    D -->|yes| F
    D -->|no| R[REAL]
```

1. **Classifier** (ResNet-50 or EfficientNet-B0, fine-tuned from ImageNet) predicts FAKE/REAL on the resized image.
2. **Segmenter** (UNet or DeepLabV3+ with a ResNet-50 encoder) predicts a per-pixel probability that the pixel was inpainted.
3. **Override:** if the classifier says REAL but more than *X*% of pixels exceed probability 0.7, the pipeline outputs FAKE. We evaluated *X* = 30, 50 and 70; 30% works best.

The FastAPI backend (`services/backend`) implements this rule with **EfficientNet-B0 + DeepLabV3+**, the best combination below. The standalone Streamlit demo (`app/`) defaults to that same pairing at a pixel threshold of 0.7 and a 30% override, and lets you swap in the other models and tune both thresholds. Architectures, checkpoint loading, inference and the decision rule live in `authentilens/models.py`, which the local demo, the hosted demo and this rule's evaluation all share.

## Key finding: fake-only benchmarks hide classifier bias

Our first classifiers were trained on **CIFAKE**. The CIFAKE-trained ResNet-50 scored **100%** on the SD2-FR test set. That set contains **only fake images**, though, so a model that always answers FAKE also scores 100%.

To measure real-image accuracy we built a **balanced patch dataset** from SD2-FR (`scripts/process_sd2_dataset.py`):

- For every inpainted image, cut one **224×224 FAKE crop** centred on the inpainting mask, so it overlaps the edited region.
- Cut one **224×224 REAL crop** from the same image with **zero** mask overlap. It is genuine camera content with the same scene and statistics.
- Balance the two classes 1:1. Splits: **11,134 train / 1,596 val / 1,632 test** (details in [docs/dataset.md](docs/dataset.md)).

On this balanced test set, the CIFAKE ResNet-50 got **1.72% real-image accuracy**: it labels almost everything FAKE. Classifiers retrained on the balanced patches are far less biased (65–72% real accuracy). The task is also plainly hard: overall accuracy is about 60–63%.

## Results

All numbers below come from the JSON files in [`results/`](results/) and [`human_evaluation/results/`](human_evaluation/results/).

### Classification ([results/classification](results/classification/comprehensive_evaluation_results.json))

"SD2-FR test" is 1,029 inpainted images (all fake). "Balanced test" is the 1,632-image patch test set (816 FAKE / 816 REAL).

| Model | Trained on | SD2-FR test (fake-only) acc. | Balanced test: overall | Balanced test: fake acc. | Balanced test: real acc. |
|---|---|---:|---:|---:|---:|
| ResNet-50 | CIFAKE | 100.00% | 50.43% | 99.14% | 1.72% |
| EfficientNet-B0 | CIFAKE | 68.22% | 50.00% | 67.40% | 32.60% |
| ResNet-50 | Balanced SD2 patches, lr 1e-4 | 61.22% | 61.95% | 57.97% | 65.93% |
| ResNet-50 | Balanced SD2 patches, lr 2.5e-5 | 58.99% | 63.11% | 61.03% | 65.20% |
| EfficientNet-B0 | Balanced SD2 patches, lr 1e-4 | 71.14% | 60.23% | 55.15% | 65.32% |
| EfficientNet-B0 | Balanced SD2 patches, lr 2.5e-5 | 71.43% | 61.27% | 50.86% | 71.69% |

### Segmentation ([results/segmentation](results/segmentation/segmentation_only_results.json))

Evaluated on SD2-FR test images with ground-truth masks, using a pixel threshold of 0.7.

| Segmenter | mIoU | Pixel accuracy | Images |
|---|---:|---:|---:|
| DeepLabV3+ | 16.76% | 61.76% | 1,029 |
| UNet | 17.28% | 63.30% | 1,029 |

### Combined pipeline ([results/pipeline](results/pipeline/))

Fake-detection TPR on the 1,029 SD2-FR test images, with the classifiers trained on balanced patches (lr 2.5e-5). Pixel threshold is 0.7. "Override > X%" means REAL is flipped to FAKE when more than X% of pixels exceed that threshold (see [results/pipeline/README.md](results/pipeline/README.md)).

| Classifier + segmenter | Classifier alone | Pipeline, override > 30% | > 50% | > 70% |
|---|---:|---:|---:|---:|
| ResNet-50 + DeepLabV3+ | 58.99% | 94.85% | 67.15% | 59.18% |
| ResNet-50 + UNet | 58.99% | 86.59% | 67.54% | 59.67% |
| **EfficientNet-B0 + DeepLabV3+** | 71.43% | **96.70%** | 76.77% | 71.43% |
| EfficientNet-B0 + UNet | 71.43% | 91.74% | 77.45% | 72.40% |

> **Limitation:** these TPRs were measured on SD2-FR test, which contains **only fake images**, so the pipeline's **false-positive rate is not yet measured**. A segmenter that flags many pixels on real photos would inflate the TPR. The evaluation script can now also run on real images and report real-image accuracy, FPR, precision and F1:
>
> ```bash
> python evaluation/evaluate_pipeline.py --real-dir data/sd2-classification/test/REAL
> ```
>
> The `sd2-classification/test/REAL` crops are 224×224, so they are upsampled to 512 for segmentation. Any folder of full-size real photos also works, e.g. the MS-COCO originals behind SD2-FR.

### Human evaluation ([human_evaluation/](human_evaluation/))

We also asked people to label images through a web voting app. We took a per-image majority vote; ties count as wrong. "Voters" means unique browser sessions.

| Test set | Images | Voters | Majority-vote accuracy | REAL acc. | FAKE acc. | Ties |
|---|---:|---:|---:|---:|---:|---:|
| Balanced patches (`sd2-classification/test`) | 1,632 | 12 | 60.17% | 73.90% | 46.45% | 5 |
| SD2-FR test (all fake, 224×224) | 1,029 | 35 | 44.22% | n/a | 44.22% | 26 |

Humans find these edits hard too: on the fake-only set they are below chance.

## Repository structure

```
AuthentiLens/
├── app/                  # Standalone Streamlit demo (all models, tunable thresholds)
├── services/
│   ├── backend/          # FastAPI inference API (EfficientNet-B0 + DeepLabV3+ pipeline)
│   └── frontend/         # Streamlit client for the API
├── hf_space/             # Hosted demo (Streamlit Community Cloud / HF Space): runs app/ with weights from the Hub
├── authentilens/         # Shared code: paths.py (repo-root-relative paths), models.py (pipeline)
├── training/             # Classifier training scripts + segmentation notebooks
├── evaluation/           # Classifier / segmentation / pipeline / IMD2020 evaluation
├── scripts/              # Dataset preparation and small utilities
├── human_evaluation/     # Human voting app, registries and collected votes
├── checkpoints/          # Model weights (Git LFS, skipped by clones; see step 2): classification/ and segmentation/
├── results/              # Evaluation outputs used in this README
├── data/                 # Datasets (not in git), see data/README.md
├── docs/                 # Project report, presentation, dataset card
├── docker-compose.yml
└── requirements.txt
```

## Quickstart

All commands run from the repo root; scripts also work from any directory.

### 1. Install

```bash
git clone https://github.com/iamgarvit/AuthentiLens.git && cd AuthentiLens
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Get the weights

By default the weights come from the Hugging Face model repo
[iamgarvit/authentilens-weights](https://huggingface.co/iamgarvit/authentilens-weights),
which is also where the live demo downloads them from. No account or token is
needed:

```bash
hf download iamgarvit/authentilens-weights --local-dir checkpoints \
    --include "classification/*" --include "segmentation/*"
```

That fetches the default pipeline (EfficientNet-B0 balanced lr 2.5e-5 +
DeepLabV3+) and the UNet, about 250 MB.

The same checkpoints, plus the ResNet-50 and other EfficientNet-B0 runs, are
also stored in this repository with **Git LFS**. Clones skip them
([`.lfsconfig`](.lfsconfig) excludes `checkpoints/`), so the `.pth` files in a
fresh clone are small pointer files. To opt in to the GitHub copies, override
that exclusion with `--exclude=""`:

```bash
git lfs install
git lfs pull --include="checkpoints/**" --exclude=""     # everything, ~900 MB
# or only the default pipeline:
git lfs pull --include="checkpoints/classification/efficientnet_b0_balanced_lr2.5e-5/best_model.pth,checkpoints/segmentation/deeplabv3plus/best_model.pth" --exclude=""
```

The segmentation checkpoints are the **weights-only** copies (107 MB and 130 MB) of the notebooks' training checkpoints, which also carried optimizer state. Provenance, checksums and the conversion are documented in [checkpoints/segmentation/deeplabv3plus/README.md](checkpoints/segmentation/deeplabv3plus/README.md) and [checkpoints/segmentation/unet/README.md](checkpoints/segmentation/unet/README.md). The Hub files are byte-identical to the Git LFS ones (same sha256).

Any model whose weights are missing is hidden from the demo, and the API falls back to **classification-only mode** and says so on screen.

Paths can be overridden with `AUTHENTILENS_CHECKPOINT_DIR` and `AUTHENTILENS_DATA_DIR`.

### 3. Run the demo

```bash
streamlit run app/streamlit_app.py            # standalone demo on http://localhost:8501
```

Or run the API + client with Docker:

```bash
docker compose up --build                     # UI: http://localhost:8501, API: http://localhost:8000/docs
```

Or run the API + client without Docker:

```bash
uvicorn main:app --app-dir services/backend --port 8000
BACKEND_URL=http://localhost:8000 streamlit run services/frontend/app.py
```

The demo defaults to the pipeline this project evaluated: EfficientNet-B0
(balanced, lr 2.5e-5) + DeepLabV3+, pixel threshold 0.7, override 30%. Models
whose weights are not on disk are hidden rather than offered and broken.

### 3b. Deploy the hosted demo

The live demo runs on **Streamlit Community Cloud** from this repository.
[`hf_space/app.py`](hf_space/app.py) downloads the weights anonymously from
[iamgarvit/authentilens-weights](https://huggingface.co/iamgarvit/authentilens-weights)
and then runs `app/streamlit_app.py` unchanged, so the deployed demo is this
repository's code rather than a copy. Only EfficientNet-B0 and DeepLabV3+ are
loaded at startup; other models load the first time they are selected.

| Streamlit Cloud field | Value |
| --- | --- |
| Repository | `iamgarvit/AuthentiLens` |
| Branch | `main` |
| Main file path | `hf_space/app.py` |
| Python version (Advanced settings) | `3.12` |

Community Cloud installs [`hf_space/requirements.txt`](hf_space/requirements.txt)
(pinned, CPU-only torch) because it sits next to the entrypoint. Streamlit
reads [`.streamlit/config.toml`](.streamlit/config.toml) from the repository
root and also [`hf_space/.streamlit/config.toml`](hf_space/.streamlit/config.toml)
from the entrypoint's directory, which overrides the root one. That is why the
`hf_space/` config must not set a port: Community Cloud would use it instead of
the port it health-checks. The Docker Space passes its port on the command
line. No secrets are needed. To run it the way Community Cloud does:

```bash
python3.12 -m venv .venv && . .venv/bin/activate
pip install -r hf_space/requirements.txt
streamlit run hf_space/app.py                   # from the repository root
```

**Hugging Face Space (needs PRO).** The same entrypoint also runs as a Docker
Space; `sync.py` stages `hf_space/` plus `app/` and `authentilens/`, and
`deploy.py` uploads it. Hugging Face requires a PRO subscription for Docker and
Gradio Spaces on free `cpu-basic` hardware, so `deploy.py space` fails with HTTP
402 without one.

```bash
hf auth login                                   # a write token
python hf_space/deploy.py weights --originals-dir /path/to/originals
python hf_space/deploy.py space
```

### 4. Data, training and evaluation

Put the datasets under `data/` as described in [data/README.md](data/README.md), then:

```bash
# Build the balanced 224×224 patch dataset from SD2-FR
python scripts/process_sd2_dataset.py

# Train classifiers -> checkpoints/classification/<model>/
python training/train_resnet_balanced.py
python training/train_efficientnet_balanced.py      # or: bash training/run_training_balanced.sh
python training/train_resnet_cifake.py              # CIFAKE baselines
python training/train_efficientnet_cifake.py

# Train segmenters -> checkpoints/segmentation/<model>/best_model.pth
jupyter notebook training/notebooks/train_deeplabv3plus.ipynb
jupyter notebook training/notebooks/train_unet.ipynb

# Evaluate -> results/
python evaluation/evaluate_classifiers.py
python evaluation/evaluate_segmentation.py
python evaluation/evaluate_pipeline.py                                              # fake-only TPR
python evaluation/evaluate_pipeline.py --real-dir data/sd2-classification/test/REAL  # + FPR / precision / F1
python evaluation/evaluate_imd2020.py --dataset data/IMD2020                        # needs the API running

# Human evaluation
python human_evaluation/calculate_accuracy.py
```

## Team and contributions

- **Divyanshu Yadav:** segmentation datasets, segmentation models, pipeline ablations, web interface
- **Garvit:** classification datasets, classification models, custom balanced dataset, pipeline ablations, web interface
- **Rewant Anand** and **Tanish Bachhas:** literature review, SOTA models, NoisePrint analysis, frontend template

CSE344 Computer Vision, IIIT Delhi. The full write-up is in [docs/AuthentiLens_Project_Report.md](docs/AuthentiLens_Project_Report.md).

## References

- **DIRE:** Z. Wang et al., "DIRE for Diffusion-Generated Image Detection", ICCV 2023. [arXiv:2303.09295](https://arxiv.org/abs/2303.09295)
- **FIRE:** B. Chu et al., "FIRE: Robust Detection of Diffusion-Generated Images via Frequency-Guided Reconstruction Error", CVPR 2025. [arXiv:2412.07140](https://arxiv.org/abs/2412.07140)
- **Art or Artifact?:** "Art or Artifact? Segmenting AI-Generated Images for Deeper Detection", 4th Workshop on Security Implications of Deepfakes and Cheapfakes, 2025. [doi:10.1145/3709022.3736544](https://dl.acm.org/doi/10.1145/3709022.3736544)
- **CIFAKE:** J. J. Bird and A. Lotfi, "CIFAKE: Image Classification and Explainable Identification of AI-Generated Synthetic Images", IEEE Access, 2024. [arXiv:2303.14126](https://arxiv.org/abs/2303.14126)
- **TGIF / SD2-FR:** H. Mareen et al., "TGIF: Text-Guided Inpainting Forgery Dataset", WIFS 2024. [arXiv:2407.11566](https://arxiv.org/abs/2407.11566), [dataset](https://github.com/IDLabMedia/tgif-dataset)
- **ResNet:** K. He et al., "Deep Residual Learning for Image Recognition", CVPR 2016. [arXiv:1512.03385](https://arxiv.org/abs/1512.03385)
- **EfficientNet:** M. Tan and Q. V. Le, "EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks", ICML 2019. [arXiv:1905.11946](https://arxiv.org/abs/1905.11946)
- **U-Net:** O. Ronneberger et al., "U-Net: Convolutional Networks for Biomedical Image Segmentation", MICCAI 2015. [arXiv:1505.04597](https://arxiv.org/abs/1505.04597)
- **DeepLabV3+:** L.-C. Chen et al., "Encoder-Decoder with Atrous Separable Convolution for Semantic Image Segmentation", ECCV 2018. [arXiv:1802.02611](https://arxiv.org/abs/1802.02611)
