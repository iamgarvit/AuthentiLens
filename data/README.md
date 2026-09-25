# Data

Datasets are **not** stored in git. Everything in `data/` except this file is gitignored. Scripts find this folder through `authentilens/paths.py`. To keep the data somewhere else, set `AUTHENTILENS_DATA_DIR=/path/to/data`.

## Expected layout

```
data/
├── sd2-fr-testing/<category>/<id>_mask_random.png_sd2-512_{0,1}.png   # SD2-FR test images (all inpainted, 512px)
├── sd2-fr-testing-masks/<category>/<id>_mask_512.png                  # their ground-truth masks
├── sd2-fr-training/  + sd2-fr-training-masks/                         # same layout, train split
├── sd2-fr-validation/ + sd2-fr-validation-masks/                      # same layout, val split
│
├── sd2-classification/                  # balanced 224×224 patch dataset (generated, see below)
│   ├── {train,val,test}/{FAKE,REAL,MASKS}/
│   ├── metadata.json
│   └── README.md
│
├── sd2-fr/{training,validation,testing}/<category>/     # segmentation notebooks' layout (SD2, 512px)
├── masks-sd2/{training,validation,testing}/<category>/  #   masks: <id>_mask_512.png
├── sdxl-fr/{training,validation,testing}/<category>/    # SDXL images (1024px)
├── masks-sdxl/{training,validation,testing}/<category>/ #   masks: <id>_mask_1024.png
│
├── cifake/{train,test}/{FAKE,REAL}/     # CIFAKE (the Kaggle download unpacks as archive/)
└── IMD2020/<subdir>/<name>.jpg + <name>_mask.png
```

## Which script uses what

| Folder | Used by |
|---|---|
| `sd2-fr-testing/`, `sd2-fr-testing-masks/` | `evaluation/evaluate_classifiers.py`, `evaluate_segmentation.py`, `evaluate_pipeline.py`, `human_evaluation/setup_registry.py` |
| `sd2-fr-{training,validation,testing}[-masks]/` | `scripts/process_sd2_dataset.py` (input) |
| `sd2-classification/` | output of `scripts/process_sd2_dataset.py`. Used by `training/train_*_balanced.py`, `evaluation/evaluate_classifiers.py`, `evaluate_pipeline.py --real-dir`, `human_evaluation/setup_registry.py` |
| `sd2-fr/`, `masks-sd2/`, `sdxl-fr/`, `masks-sdxl/` | `training/notebooks/train_deeplabv3plus.ipynb`, `train_unet.ipynb` |
| `cifake/` | `training/train_*_cifake.py` |
| `IMD2020/` | `evaluation/evaluate_imd2020.py` |

The flat `sd2-fr-<split>` folders and the notebooks' `sd2-fr/<split>` folders use the same file naming (`<category>/<id>_mask_random.png_sd2-512_N.png`, masks `<id>_mask_512.png`). If you have one layout, symlinks give you the other:

```bash
cd data
ln -s sd2-fr/testing sd2-fr-testing && ln -s masks-sd2/testing sd2-fr-testing-masks
```

## Sources

| Dataset | What it is | Where to get it |
|---|---|---|
| **SD2-FR / SDXL-FR** | The Stable Diffusion 2 / SDXL subsets of **TGIF / TGIF2** (text-guided inpainting forgeries of MS-COCO photos, with masks). We use SD2-FR for classification and evaluation, and SD2-FR + SDXL-FR for segmentation training. | [IDLabMedia/tgif-dataset](https://github.com/IDLabMedia/tgif-dataset) (TGIF2 is also on [Zenodo](https://zenodo.org/records/19735393)) |
| **sd2-classification** | Our balanced 224×224 FAKE/REAL patch dataset, built from SD2-FR (11,134 / 1,596 / 1,632 train / val / test). Dataset card: [docs/dataset.md](../docs/dataset.md). | `python scripts/process_sd2_dataset.py` |
| **CIFAKE** | 60k real (CIFAR-10) + 60k Stable Diffusion 1.4 images, 32×32. | [Kaggle: birdy654/cifake-real-and-ai-generated-synthetic-images](https://www.kaggle.com/datasets/birdy654/cifake-real-and-ai-generated-synthetic-images). Unzip and rename `archive/` to `cifake/`. |
| **IMD2020** | Real-world manipulated images with masks (used for API-based localisation eval). | [staff.utia.cas.cz/novozada/db](https://staff.utia.cas.cz/novozada/db/) |

`scripts/process_sd2_dataset.py` keeps the raw `sd2-fr-*` folders by default. Pass `--delete-sources` only if you no longer need them; evaluation still uses `sd2-fr-testing/`.
