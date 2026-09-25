#!/usr/bin/env python3
"""Assemble the Hugging Face Space from this repository.

The Space needs to be self-contained, but the demo must not be forked by hand:
everything under ``app/`` and ``authentilens/`` is copied verbatim from the
repository, so the Space always runs the code that is committed here.

The same directory is both the Docker build context and what gets uploaded, so
a local ``docker build`` exercises exactly what the Space will run.

Usage:
    python hf_space/sync.py                 # stage into hf_space/.build/
    python hf_space/sync.py --dest DIR      # stage somewhere else
"""

import argparse
import shutil
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SPACE_DIR = REPO_ROOT / "hf_space"

# Files that belong to the Space itself.
SPACE_FILES = [
    "Dockerfile",
    "README.md",
    "app.py",
    "requirements.txt",
    ".streamlit/config.toml",
    ".dockerignore",
]

# Repository code the Space runs, copied verbatim. The relative layout is kept
# because app/streamlit_app.py and authentilens/paths.py both resolve paths
# against their own location.
REPO_FILES = [
    "app/streamlit_app.py",
    "app/utils_noiseprint.py",
    "authentilens/__init__.py",
    "authentilens/paths.py",
    "authentilens/models.py",
]


def sync(dest: Path) -> Path:
    if dest.exists():
        shutil.rmtree(dest)
    dest.mkdir(parents=True)

    for rel in SPACE_FILES:
        src = SPACE_DIR / rel
        if not src.exists():
            raise SystemExit(f"Missing Space file: {src}")
        target = dest / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, target)

    for rel in REPO_FILES:
        src = REPO_ROOT / rel
        if not src.exists():
            raise SystemExit(f"Missing repository file: {src}")
        target = dest / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, target)

    files = sorted(p for p in dest.rglob("*") if p.is_file())
    total_kb = sum(p.stat().st_size for p in files) / 1024
    print(f"Staged {len(files)} files ({total_kb:.0f} KB) in {dest}")
    for path in files:
        print(f"  {path.relative_to(dest)}")
    return dest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dest", type=Path, default=SPACE_DIR / ".build",
                        help="where to assemble the Space (default: hf_space/.build)")
    args = parser.parse_args()
    sync(args.dest.resolve())


if __name__ == "__main__":
    main()
