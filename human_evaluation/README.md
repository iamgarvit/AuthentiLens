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

This scans both datasets and creates UUID-based registries in `data/`.

### 3. Run the server

```bash
python server.py
```

The server starts on `http://localhost:5050`.

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
| Portal A | `/sd2-fr/` | sd2-fr-testing | External evaluators |
| Portal B | `/custom/` | sd2-classification/test | Developers |

## Admin Stats

Access stats at `/<dataset>/stats?pwd=<password>`.

Default password: `authentilens2026` (set via `EVAL_ADMIN_PASSWORD` env var).

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
