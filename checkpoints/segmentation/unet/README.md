# UNet segmentation weights

The alternative segmenter. It scores slightly better on segmentation alone
(17.28% vs 16.76% mIoU) but worse in the combined pipeline (91.74% vs 96.70%
TPR), so [`results/pipeline/`](../../../results/pipeline/) picks DeepLabV3+ for
the default pipeline.

## The file

```
checkpoints/segmentation/unet/best_model.pth
```

| | |
|---|---|
| Architecture | `smp.Unet`, ResNet-50 encoder, 1 class, decoder channels (256, 128, 64, 32, 16) |
| Input | 512x512, ImageNet normalisation |
| Trained on | SD2-FR + SDXL-FR, 512x512, batch 32, lr 1e-4, decoder dropout 0.3 |
| Stored epoch | 8 |
| Validation IoU | 0.7249 |
| Size | 130,433,357 bytes |
| sha256 | `ef9af814c1b85a88e5e3e7b9a6cb97633c650257c81966d2d4de735d1ffe5d5e` |

The training-time validation IoU (0.725) is measured on the validation split at
the notebook's own threshold. It is not comparable to the 17.28% mIoU in
[`results/segmentation`](../../../results/segmentation/segmentation_only_results.json),
which is measured on the SD2-FR **test** images at a pixel threshold of 0.7.

Also published at
[iamgarvit/authentilens-weights](https://huggingface.co/iamgarvit/authentilens-weights)
as `segmentation/unet/best_model.pth`.

## Source and how it was made

The weights come from the training notebook's own checkpoint, preserved in an
**11 April 2026 snapshot of this repository** (`AuthentiLensNoisePrint.zip`),
at `AuthentiLens/checkpoints_unet/best_model.pth`.

| | |
|---|---|
| Original size | 390,770,682 bytes |
| Original sha256 | `a399c342e4246480567b662ed1831e7b77cd00b1cee3511954ea9c24f0290cb7` |

That original is a full training checkpoint: `model_state_dict`,
`optimizer_state_dict`, `epoch`, `val_iou` and `config`. The optimizer state
makes it about 3x the size of the weights and nothing at inference time reads
it, so the committed file keeps only the weights:

```bash
python scripts/export_weights_only.py \
    /path/to/checkpoints_unet/best_model.pth \
    checkpoints/segmentation/unet/best_model.pth
```

`export_weights_only.py` saves `checkpoint["model_state_dict"]` verbatim — no
key is renamed, added or dropped — so both files load identically. Loading each
into the architecture and running the same 512x512 input through them gives
bit-identical logits.

The original is kept as a backup under `original/unet_original.pth` in the
Hugging Face model repo.

## A note on the checkpoint's key names

The notebook wrapped each decoder block's second convolution in dropout whose
probability grows with depth:

```python
for i, block in enumerate(model.decoder.blocks):
    scaled = 0.3 * (0.4 + 0.6 * i / (len(model.decoder.blocks) - 1))
    block.conv2 = nn.Sequential(block.conv2, nn.Dropout2d(p=scaled))
```

That nests the convolution one level deeper, so the checkpoint's keys look like
`decoder.blocks.4.conv2.0.1.running_mean`.
`authentilens.models.build_segmenter` reproduces the wrapping, so the
checkpoint loads with `strict=True`. The dropout is inactive in `eval()` mode
and does not change inference.

## How to regenerate them

1. Put SD2-FR and SDXL-FR under `data/` in the notebook layout (`data/sd2-fr/`,
   `data/masks-sd2/`, `data/sdxl-fr/`, `data/masks-sdxl/`, each with
   `training/`, `validation/` and `testing/` splits). See
   [data/README.md](../../../data/README.md).
2. Install the dependencies (`pip install -r requirements.txt`, plus `jupyter`).
   A CUDA GPU is strongly recommended.
3. Run every cell of the notebook from the repo root:
   ```bash
   jupyter notebook training/notebooks/train_unet.ipynb
   ```
4. The notebook writes `best_model.pth` (best validation IoU), `final_model.pth`,
   `metrics.json` and plots into this folder.
5. Check the file loads, then re-run the evaluations:
   ```bash
   python evaluation/evaluate_segmentation.py
   python evaluation/evaluate_pipeline.py
   ```
