"""Entry point for the hosted AuthentiLens demo.

Streamlit Community Cloud runs this file straight from a checkout of the
repository (``streamlit run hf_space/app.py`` from the repo root, dependencies
from ``hf_space/requirements.txt``). The Hugging Face Space runs the same file
from the directory staged by ``hf_space/sync.py``.

Either way it runs the repository's own demo. This module only does the two
things a hosted copy needs on top of it:

1. Download the weights from the model repo (they are too large for the Space
   repo itself) into the directory ``authentilens.paths`` reads.
2. Hand over to ``app/streamlit_app.py`` unchanged, so the deployed demo is the
   same code as ``streamlit run app/streamlit_app.py`` locally.

The weights are fetched once per container and cached by ``st.cache_resource``.
The model repo is public, so no token is needed.
"""

import os
import runpy
from pathlib import Path

import streamlit as st
from huggingface_hub import hf_hub_download

HERE = Path(__file__).resolve().parent

# The demo sits next to this file in the staged Space (hf_space/sync.py copies
# it there) and one level up in a repository checkout (Streamlit Community
# Cloud).
DEMO = next(
    path
    for path in (HERE / "app" / "streamlit_app.py", HERE.parent / "app" / "streamlit_app.py")
    if path.exists()
)

# Which model repo to pull the weights from; overridable so a fork can point at
# its own copy without editing this file.
MODEL_REPO = os.environ.get("AUTHENTILENS_MODEL_REPO", "iamgarvit/authentilens-weights")

# Where the weights land. The paths inside the model repo mirror the layout in
# checkpoints/, so downloading them into this directory is enough for
# authentilens.paths to find everything.
CHECKPOINT_DIR = Path(
    os.environ.get("AUTHENTILENS_CHECKPOINT_DIR", str(HERE / "checkpoints"))
)

# The weights-only copies, not the full training checkpoints: same tensors, a
# third of the size. The originals stay in the model repo under original/.
REQUIRED_FILES = [
    "classification/efficientnet_b0_balanced_lr2.5e-5/best_model.pth",
    "segmentation/deeplabv3plus/best_model.pth",
]

# Fetched when present so the UNet stays selectable; the Space still works
# without it.
OPTIONAL_FILES = [
    "segmentation/unet/best_model.pth",
]


@st.cache_resource(show_spinner="Downloading model weights (first run only)...")
def download_weights() -> str:
    """Fetch the checkpoints into CHECKPOINT_DIR and return the directory."""
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    for filename in REQUIRED_FILES:
        hf_hub_download(
            repo_id=MODEL_REPO,
            filename=filename,
            local_dir=str(CHECKPOINT_DIR),
        )

    for filename in OPTIONAL_FILES:
        try:
            hf_hub_download(
                repo_id=MODEL_REPO,
                filename=filename,
                local_dir=str(CHECKPOINT_DIR),
            )
        except Exception as exc:  # the demo hides models whose weights are absent
            print(f"Optional checkpoint {filename} unavailable: {exc}")

    return str(CHECKPOINT_DIR)


try:
    os.environ["AUTHENTILENS_CHECKPOINT_DIR"] = download_weights()
except Exception as exc:
    st.error(
        f"Could not download the model weights from `{MODEL_REPO}`: {exc}\n\n"
        "The demo cannot start without them."
    )
    st.stop()

# Run the repository's demo. authentilens.paths reads
# AUTHENTILENS_CHECKPOINT_DIR at import time, which is why it is set above.
runpy.run_path(str(DEMO), run_name="__main__")
