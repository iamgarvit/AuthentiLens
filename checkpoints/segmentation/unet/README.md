# UNet segmentation weights

> **Status: pending.** These weights were never committed to the repo. Until they are added, the Streamlit demo and the API run in **classification-only mode**, and the segmentation and pipeline evaluation scripts exit with a message pointing here.

## Where to put them

```
checkpoints/segmentation/unet/best_model.pth
```

The training notebook originally saved this file as `best_unet.pth`. Loaders accept that name too and log a warning, but please rename it to `best_model.pth`.

`*.pth` files are tracked with **Git LFS** (see `.gitattributes`), so a normal `git add` stores the file in LFS. The notebook's checkpoint also contains the optimizer state, which makes it roughly 3× the size of the bare weights. Keep an eye on your LFS storage quota.

## How to regenerate them

1. Put SD2-FR and SDXL-FR under `data/` in the notebook layout (`data/sd2-fr/`, `data/masks-sd2/`, `data/sdxl-fr/`, `data/masks-sdxl/`, each with `training/`, `validation/` and `testing/` splits). See [data/README.md](../../../data/README.md).
2. Install the dependencies (`pip install -r requirements.txt`, plus `jupyter`). A CUDA GPU is strongly recommended.
3. Run every cell of the notebook from the repo root:
   ```bash
   jupyter notebook training/notebooks/train_unet.ipynb
   ```
4. The notebook writes `best_model.pth` (best validation IoU), `final_model.pth`, `metrics.json` and plots into this folder.
5. Check the file loads, then re-run the evaluations:
   ```bash
   python evaluation/evaluate_segmentation.py
   python evaluation/evaluate_pipeline.py
   ```
