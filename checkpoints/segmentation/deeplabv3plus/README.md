# DeepLabV3+ segmentation weights

The segmenter used by the default pipeline: EfficientNet-B0 (balanced, lr 2.5e-5)
+ DeepLabV3+, which reached 96.70% combined TPR in
[`results/pipeline/`](../../../results/pipeline/).

## The file

```
checkpoints/segmentation/deeplabv3plus/best_model.pth
```

| | |
|---|---|
| Architecture | `smp.DeepLabV3Plus`, ResNet-50 encoder, 1 class, no activation |
| Input | 512x512, ImageNet normalisation |
| Trained on | SD2-FR + SDXL-FR, 512x512, batch 32, lr 1e-4, decoder dropout 0.3 |
| Stored epoch | 8 |
| Validation IoU | 0.7264 |
| Size | 107,067,276 bytes |
| sha256 | `574b34efa947905899f024cf5010a674ce45aa15e3fde63f58e56704ef97a13f` |

The training-time validation IoU (0.726) is measured on the validation split at
the notebook's own threshold. It is not comparable to the 16.76% mIoU in
[`results/segmentation`](../../../results/segmentation/segmentation_only_results.json),
which is measured on the SD2-FR **test** images at a pixel threshold of 0.7.

## Getting the file

A fresh clone has only a small Git LFS pointer here:
[`.lfsconfig`](../../../.lfsconfig) excludes `checkpoints/` from LFS fetches. The
default source is the Hugging Face model repo
[iamgarvit/authentilens-weights](https://huggingface.co/iamgarvit/authentilens-weights),
where this file is `segmentation/deeplabv3plus/best_model.pth`, byte-identical to the
Git LFS copy (same sha256). From the repo root:

```bash
# default pipeline (EfficientNet-B0 balanced lr 2.5e-5 + DeepLabV3+) and the UNet, about 250 MB
hf download iamgarvit/authentilens-weights --local-dir checkpoints \
    --include "classification/*" --include "segmentation/*"
```

To use the Git LFS copies instead, override the exclusion with `--exclude=""`.
Git LFS also holds the ResNet-50 and other EfficientNet-B0 checkpoints, which are
not on the Hub:

```bash
git lfs install
git lfs pull --include="checkpoints/**" --exclude=""     # everything, ~900 MB
# or only the default pipeline:
git lfs pull --include="checkpoints/classification/efficientnet_b0_balanced_lr2.5e-5/best_model.pth,checkpoints/segmentation/deeplabv3plus/best_model.pth" --exclude=""
```

Set `AUTHENTILENS_CHECKPOINT_DIR` to read the weights from somewhere other than
`checkpoints/`.

## Source and how it was made

The weights come from the training notebook's own checkpoint, preserved in an
**11 April 2026 snapshot of this repository** (`AuthentiLensNoisePrint.zip`),
at `AuthentiLens/checkpoints_deeplab/best_model.pth`.

| | |
|---|---|
| Original size | 320,667,020 bytes |
| Original sha256 | `5a73fdd06aa7166a7756fbd197320b53fc68a4ccf8bca18b7d8702e1dcecc60e` |

That original is a full training checkpoint: `model_state_dict`,
`optimizer_state_dict`, `epoch`, `val_iou` and `config`. The optimizer state
makes it about 3x the size of the weights and nothing at inference time reads
it, so the committed file keeps only the weights:

```bash
python scripts/export_weights_only.py \
    /path/to/checkpoints_deeplab/best_model.pth \
    checkpoints/segmentation/deeplabv3plus/best_model.pth
```

`export_weights_only.py` saves `checkpoint["model_state_dict"]` verbatim — no
key is renamed, added or dropped — so both files load identically. Loading each
into the architecture and running the same 512x512 input through them gives
bit-identical logits.

The original is kept as a backup under `original/deeplabv3plus_original.pth` in
the Hugging Face model repo.

## A note on the checkpoint's key names

The state dict contains `decoder.block2.*` **and** `decoder.block.1.*` with the
same shapes. That is not a corrupted file: the notebook added decoder dropout
with

```python
model.decoder.block = nn.Sequential(nn.Dropout2d(p=0.3), model.decoder.block2)
```

Because `block2` is shared by reference, PyTorch serialises the same tensors
under both names (they share storage — verified with `data_ptr()`). The
`block` attribute is never called by `smp`'s decoder `forward`, so the dropout
never affected training or inference; it only changed the key names.

`authentilens.models.build_segmenter` recreates that attribute, so the
checkpoint loads with `strict=True` and no key remapping. Rebuilding the model
the other plausible way — wrapping `block2` in `Sequential(block2, Dropout)` and
remapping the keys — produces bit-identical outputs.

## How to regenerate them

1. Put SD2-FR and SDXL-FR under `data/` in the notebook layout (`data/sd2-fr/`,
   `data/masks-sd2/`, `data/sdxl-fr/`, `data/masks-sdxl/`, each with
   `training/`, `validation/` and `testing/` splits). See
   [data/README.md](../../../data/README.md).
2. Install the dependencies (`pip install -r requirements.txt`, plus `jupyter`).
   A CUDA GPU is strongly recommended.
3. Run every cell of the notebook from the repo root:
   ```bash
   jupyter notebook training/notebooks/train_deeplabv3plus.ipynb
   ```
4. The notebook writes `best_model.pth` (best validation IoU), `final_model.pth`,
   `metrics.json` and plots into this folder.
5. Check the file loads, then re-run the evaluations:
   ```bash
   python scripts/check_deeplab_checkpoint.py
   python evaluation/evaluate_segmentation.py
   python evaluation/evaluate_pipeline.py
   ```
