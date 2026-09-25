# Pipeline results

Output of `evaluation/evaluate_pipeline.py`: all four classifier + segmenter combinations on `data/sd2-fr-testing` (1,029 inpainted images, **no real images**).

## What the numbers in the filenames mean

`pipeline_pixel<P>_override<X>pct.json`

- **`pixel<P>`**: per-pixel threshold. A pixel counts as "flagged" when the segmenter's sigmoid probability is **> P**. All committed files use **P = 0.7**.
- **`override<X>pct`**: image-level override. When the classifier predicts REAL but **more than X% of the pixels are flagged**, the pipeline outputs FAKE.

| File | Pixel threshold | Override if flagged area > | Produced by (original script) |
|---|---:|---:|---|
| `pipeline_pixel0.7_override30pct.json` | 0.7 | 30% | `evaluate_combinations.py`, pixel sweep with a hard-coded 30% override (committed as `combinations_test_results_0.7.json`, later renamed `..._override_30.json`) |
| `pipeline_pixel0.7_override50pct.json` | 0.7 | 50% | `combination_results_final.py`, override sweep with a fixed 0.7 pixel threshold |
| `pipeline_pixel0.7_override70pct.json` | 0.7 | 70% | `combination_results_final.py` |

Both original scripts are merged into `evaluation/evaluate_pipeline.py`. Running it with no arguments reproduces all three files.

## Fields

| Field | Meaning |
|---|---|
| `total_images` | Fake images evaluated |
| `classifier_fake_TPR` | Share of fakes the classifier alone calls FAKE |
| `segmentation_mIoU`, `segmentation_pixel_accuracy` | Mean over images that have a ground-truth mask (at pixel threshold P) |
| `combined_pipeline_TPR` | Share of fakes the full pipeline (classifier OR override) calls FAKE |

## False positives

Because `sd2-fr-testing` contains only fakes, these files cannot show false positives. Run

```bash
python evaluation/evaluate_pipeline.py --real-dir data/sd2-classification/test/REAL
```

to write `*_with_real.json` files that add `real_images`, `{classifier,combined}_real_accuracy`, `_FPR`, `_precision`, `_F1` and `_balanced_accuracy`.
