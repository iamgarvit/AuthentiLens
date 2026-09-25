# Deployment and maintenance

Notes for maintainers: how the hosted demo is deployed, how the Hugging Face
weights and Space are published, and how to run the API without Docker. For
using the project, see the [README](../README.md).

## Hosted demo: Streamlit Community Cloud

The live demo, https://aperture-lens.streamlit.app, runs on Streamlit Community
Cloud straight from this repository. Its entrypoint,
[`hf_space/app.py`](../hf_space/app.py), downloads the weights anonymously from
[iamgarvit/authentilens-weights](https://huggingface.co/iamgarvit/authentilens-weights)
and then runs [`app/streamlit_app.py`](../app/streamlit_app.py) unchanged, so the
deployed demo is this repository's code rather than a copy. Only EfficientNet-B0
and DeepLabV3+ are loaded at startup; other models load the first time they are
selected. No secrets are needed.

| Streamlit Cloud field | Value |
| --- | --- |
| Repository | `iamgarvit/AuthentiLens` |
| Branch | `main` |
| Main file path | `hf_space/app.py` |
| Python version (Advanced settings) | `3.12` |

**Dependencies.** Community Cloud uses
[`hf_space/requirements.txt`](../hf_space/requirements.txt) because it sits next
to the entrypoint. Every version is pinned, and torch/torchvision are the
CPU-only `+cpu` builds from the PyTorch index. Community Cloud tries `uv` first,
which rejects this file (its default index strategy only looks at the PyTorch
index for packages that index also hosts, such as `certifi`), then falls back
to pip, which installs it. A uv error followed by a successful pip install in
the build log is expected.

**Config files and the port rule.** Streamlit reads
[`.streamlit/config.toml`](../.streamlit/config.toml) from the repository root
and also [`hf_space/.streamlit/config.toml`](../hf_space/.streamlit/config.toml)
from the entrypoint's directory, and the entrypoint-level file overrides the
root one option by option. `hf_space/.streamlit/config.toml` must therefore
**not set a port**: Community Cloud would use it instead of the port it
health-checks, and the app would never come up. The Docker Space passes its
port on the command line instead.

**Git LFS bandwidth.** [`.lfsconfig`](../.lfsconfig) excludes `checkpoints/`
from Git LFS fetches, so neither clones nor Community Cloud builds download the
~900 MB of checkpoints; the demo gets its weights from the Hub.

**Environment overrides** (optional): `AUTHENTILENS_MODEL_REPO` points
`hf_space/app.py` at another model repo, and `AUTHENTILENS_CHECKPOINT_DIR` sets
where weights are read from (default `checkpoints/`, or `hf_space/checkpoints/`
for the hosted entrypoint). `AUTHENTILENS_DATA_DIR` is described in
[data/README.md](../data/README.md).

To run the demo the way Community Cloud does, from the repository root:

```bash
python3.12 -m venv .venv && . .venv/bin/activate
pip install -r hf_space/requirements.txt
streamlit run hf_space/app.py
```

## Publishing the weights and model card

[`hf_space/deploy.py`](../hf_space/deploy.py) publishes to the Hub and needs a
write token (`hf auth login`, or `HF_TOKEN` in the environment).

```bash
python hf_space/deploy.py weights                                   # model card + weights
python hf_space/deploy.py weights --originals-dir /path/to/originals  # also the full training checkpoints, under original/
```

`weights` always uploads [`hf_space/MODEL_CARD.md`](../hf_space/MODEL_CARD.md) as
the model repo's README, so it is also how the card is refreshed. It compares
each weights file's sha256 with the Hub copy and skips unchanged files. That
makes it safe to run from a clone whose `checkpoints/` are still Git LFS
pointers (a pointer records its file's sha256); a pointer whose file differs
from the Hub copy is refused rather than uploaded.

## Hugging Face Space (needs PRO)

The same entrypoint also runs as a Docker Space
([`hf_space/Dockerfile`](../hf_space/Dockerfile)).
[`hf_space/sync.py`](../hf_space/sync.py) stages `hf_space/` plus `app/` and
`authentilens/` into `hf_space/.build/`, and `deploy.py space` uploads it.
Hugging Face requires a PRO subscription for Docker and Gradio Spaces on free
`cpu-basic` hardware, so `deploy.py space` fails with HTTP 402 without one.

```bash
hf auth login                   # a write token
python hf_space/deploy.py space # or: all (weights + space)
```

## Running the API without Docker

`docker compose up --build` (see the README) runs the FastAPI backend and its
Streamlit client together. To run them directly instead:

```bash
uvicorn main:app --app-dir services/backend --port 8000
BACKEND_URL=http://localhost:8000 streamlit run services/frontend/app.py
```
