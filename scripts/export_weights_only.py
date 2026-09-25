#!/usr/bin/env python3
"""Strip a training checkpoint down to its model weights.

The segmentation notebooks saved full training checkpoints: ``model_state_dict``
plus ``optimizer_state_dict``, ``epoch``, ``val_iou`` and ``config``. The
optimizer state makes the file roughly 3x larger than the weights, and nothing
at inference time reads it, so the copies we publish keep only the weights.

The output is ``checkpoint["model_state_dict"]`` saved verbatim - no keys are
renamed, added or dropped - so every loader in this repo keeps working exactly
as it does with the original file.

Usage:
    python scripts/export_weights_only.py SOURCE.pth DEST.pth
"""
import argparse
import hashlib
from pathlib import Path

import torch


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def export(src: Path, dest: Path) -> None:
    ckpt = torch.load(src, map_location="cpu", weights_only=False)

    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        state = ckpt["model_state_dict"]
        dropped = sorted(k for k in ckpt if k != "model_state_dict")
    elif isinstance(ckpt, dict) and "state_dict" in ckpt:
        state = ckpt["state_dict"]
        dropped = sorted(k for k in ckpt if k != "state_dict")
    else:
        state = ckpt
        dropped = []

    dest.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, dest)

    src_mb = src.stat().st_size / 1e6
    dest_mb = dest.stat().st_size / 1e6
    print(f"{src.name} -> {dest}")
    print(f"  tensors kept : {len(state)}")
    print(f"  keys dropped : {', '.join(dropped) or '(none)'}")
    print(f"  size         : {src_mb:.1f} MB -> {dest_mb:.1f} MB")
    print(f"  sha256 src   : {sha256(src)}")
    print(f"  sha256 dest  : {sha256(dest)}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("source", type=Path, help="full training checkpoint")
    parser.add_argument("dest", type=Path, help="where to write the weights-only copy")
    args = parser.parse_args()
    export(args.source, args.dest)


if __name__ == "__main__":
    main()
