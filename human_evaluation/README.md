# Human Evaluation System

A web-based voting system for evaluating image authenticity (REAL vs FAKE).

## Quick Start

### 1. Install dependencies

```bash
cd human_evaluation
pip install -r requirements.txt
```

### 2. Build image registries

```bash
python setup_registry.py
```

This scans both datasets under the repo's `data/` folder (`data/sd2-fr-testing/` and
`data/sd2-classification/test/`, or `$AUTHENTILENS_DATA_DIR`) and creates UUID-based
registries in `human_evaluation/data/`. Image paths are stored relative to the data folder.

> The committed registries and votes are the ones used for the reported results.
> Re-running `setup_registry.py` generates new UUIDs, so only do it for a fresh study.

### 3. Run the server

```bash
EVAL_ADMIN_PASSWORD='<choose a strong password>' python server.py
```

The server starts on `http://localhost:5050`. `EVAL_ADMIN_PASSWORD` is required
for the stats pages; without it only the voting portals run. Optionally set
`FLASK_SECRET` so voter sessions survive a restart.

### 4. Expose to the internet (for remote voters)

Using **ngrok**:
```bash
ngrok http 5050
```

Or using **cloudflared**:
```bash
cloudflared tunnel --url http://localhost:5050
```

Share the generated public URL with your evaluators.

## Portals

| Portal | URL | Dataset | Who evaluates |
|---|---|---|---|
| Portal A | `/sd2-fr/` | `data/sd2-fr-testing` | External evaluators |
| Portal B | `/custom/` | `data/sd2-classification/test` | Developers |

## Admin Stats

Open `/<dataset>/stats` and log in with the `EVAL_ADMIN_PASSWORD` you started
the server with. The password is sent in a form, not the URL, and the login
lasts for the browser session. The stats pages include the ground-truth labels,
so don't share the password with voters.

## Keyboard Shortcuts

While evaluating:
- **R** or **1** → Vote REAL
- **F** or **2** → Vote FAKE

## How It Works

- Each image is served via an opaque UUID endpoint — no file paths or labels are visible
- Images not yet evaluated are prioritized (coverage-first distribution)
- Each voter session sees different images to maximize coverage
- Votes are stored in `results/` as JSON files with per-image granularity
- Stats include per-image, per-category, and per-label breakdowns

## Reproducing the reported accuracy

```bash
python human_evaluation/calculate_accuracy.py
```

Per-image majority vote (ties count as incorrect). Unique voters = unique browser sessions.
