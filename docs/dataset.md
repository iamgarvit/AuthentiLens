# Image Forgery Classification Dataset (SD2)

> Lives at `data/sd2-classification/` (not in git). Generate it with `python scripts/process_sd2_dataset.py`; see [data/README.md](../data/README.md).

This dataset contains perfectly class-balanced 224x224 patches extracted from the SD2 forgery dataset mappings. 
For every source image containing a manipulated region:
- `FAKE`: A crop explicitly centered on the bounding box of the generated/manipulated mask.
- `REAL`: A crop from the identical unmanipulated background region from the same image.

This perfectly balances artifacts, scene semantics, and visual properties across the Fake vs. Real classes.

## Dataset Splits

| Split  | FAKE Count | REAL Count | Total Images |
| ------ | ---------- | ---------- | ------------ |
| Train  | 5567       | 5567       | 11134        |
| Val    | 798        | 798        | 1596         |
| Test   | 816        | 816        | 1632         |

- **Total:** 14,362 images
- **Resolution:** 224x224 px

## Directory Structure

```
sd2-classification/
├── train/
│   ├── FAKE/
│   └── REAL/
├── val/
│   ├── FAKE/
│   └── REAL/
├── test/
│   ├── FAKE/
│   └── REAL/
├── metadata.json
└── README.md
```

## Metadata structure

A robust `metadata.json` ties every single patch (like `134886.jpg_fake.png`) back to its original `source` file, its original mask bounding box coordinates, and its label.

