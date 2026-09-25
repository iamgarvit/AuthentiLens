#!/usr/bin/env python3
"""Publish the AuthentiLens weights and Space to the Hugging Face Hub.

Requires a write token: run `hf auth login` first.

    python hf_space/deploy.py weights   # create/update the model repo
    python hf_space/deploy.py space     # create/update the Space
    python hf_space/deploy.py all       # both

The weights come from the repository's own checkpoints/ directory, except the
two full training checkpoints, which are only kept outside the repo; pass
--originals-dir to include them.
"""

import argparse
import sys
from pathlib import Path

from huggingface_hub import HfApi, whoami

REPO_ROOT = Path(__file__).resolve().parents[1]
SPACE_DIR = REPO_ROOT / "hf_space"

MODEL_REPO = "iamgarvit/authentilens-weights"
SPACE_REPO = "iamgarvit/authentilens"

# local path -> path inside the model repo
WEIGHTS = {
    REPO_ROOT / "checkpoints/classification/efficientnet_b0_balanced_lr2.5e-5/best_model.pth":
        "classification/efficientnet_b0_balanced_lr2.5e-5/best_model.pth",
    REPO_ROOT / "checkpoints/segmentation/deeplabv3plus/best_model.pth":
        "segmentation/deeplabv3plus/best_model.pth",
    REPO_ROOT / "checkpoints/segmentation/unet/best_model.pth":
        "segmentation/unet/best_model.pth",
}

# The full training checkpoints, kept as a backup. Named as they are in
# --originals-dir.
ORIGINALS = {
    "deeplabv3plus_original.pth": "original/deeplabv3plus_original.pth",
    "unet_original.pth": "original/unet_original.pth",
}


def upload_weights(api: HfApi, originals_dir: Path | None) -> None:
    api.create_repo(MODEL_REPO, repo_type="model", private=False, exist_ok=True)
    print(f"model repo ready: https://huggingface.co/{MODEL_REPO}")

    api.upload_file(
        path_or_fileobj=str(SPACE_DIR / "MODEL_CARD.md"),
        path_in_repo="README.md",
        repo_id=MODEL_REPO,
        repo_type="model",
        commit_message="Add model card",
    )
    print("  uploaded README.md (model card)")

    targets = dict(WEIGHTS)
    if originals_dir:
        for name, remote in ORIGINALS.items():
            local = originals_dir / name
            if local.exists():
                targets[local] = remote
            else:
                print(f"  skipping {name}: not found in {originals_dir}")

    for local, remote in targets.items():
        if not local.exists():
            raise SystemExit(f"Missing weights file: {local}")
        size_mb = local.stat().st_size / 1e6
        print(f"  uploading {remote} ({size_mb:.0f} MB)...", flush=True)
        api.upload_file(
            path_or_fileobj=str(local),
            path_in_repo=remote,
            repo_id=MODEL_REPO,
            repo_type="model",
            commit_message=f"Add {remote}",
        )
    print(f"weights published: https://huggingface.co/{MODEL_REPO}")


def upload_space(api: HfApi) -> None:
    sys.path.insert(0, str(SPACE_DIR))
    from sync import sync  # noqa: E402  (same directory as this script)

    build = sync(SPACE_DIR / ".build")

    api.create_repo(SPACE_REPO, repo_type="space", space_sdk="docker",
                    private=False, exist_ok=True)
    print(f"space ready: https://huggingface.co/spaces/{SPACE_REPO}")

    api.upload_folder(
        folder_path=str(build),
        repo_id=SPACE_REPO,
        repo_type="space",
        commit_message="Deploy AuthentiLens demo",
    )
    print(f"space deployed: https://huggingface.co/spaces/{SPACE_REPO}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("target", choices=["weights", "space", "all"])
    parser.add_argument("--originals-dir", type=Path, default=None,
                        help="directory holding deeplabv3plus_original.pth and "
                             "unet_original.pth, uploaded under original/")
    args = parser.parse_args()

    try:
        user = whoami()
    except Exception:
        raise SystemExit("Not logged in. Run `hf auth login` with a write token first.")
    print(f"logged in as: {user['name']}")

    api = HfApi()
    if args.target in ("weights", "all"):
        upload_weights(api, args.originals_dir)
    if args.target in ("space", "all"):
        upload_space(api)


if __name__ == "__main__":
    main()
