"""
server.py — Flask backend for the Human Evaluation System.

Serves two independent evaluation portals:
  /sd2-fr/   → sd2-fr-testing dataset (all fake, evaluated by external users)
  /custom/   → sd2-classification/test dataset (50-50, evaluated by developers)

Security: All images are served via opaque UUID endpoints. No file paths,
filenames, or ground truth labels are ever exposed to the client.
"""

import hmac
import json
import os
import sys
import random
import threading
import time
from io import BytesIO
from pathlib import Path

from flask import (
    Flask,
    abort,
    jsonify,
    redirect,
    render_template,
    request,
    send_file,
    session,
    url_for,
)
from PIL import Image

# ── Configuration ────────────────────────────────────────────────────────────
APP_DIR = Path(__file__).resolve().parent
DATA_DIR = APP_DIR / "data"
RESULTS_DIR = APP_DIR / "results"

sys.path.insert(0, str(APP_DIR.parent))  # repo root
from authentilens.paths import DATA_DIR as DATASETS_DIR  # noqa: E402


def resolve_image_path(stored_path):
    """Resolve a registry path. New registries store paths relative to data/;
    absolute paths from another machine are re-anchored at the dataset folder."""
    path = Path(stored_path)
    if not path.is_absolute():
        return DATASETS_DIR / path
    if not path.exists():
        for anchor in ("sd2-fr-testing", "sd2-classification"):
            if anchor in path.parts:
                return DATASETS_DIR.joinpath(*path.parts[path.parts.index(anchor):])
    return path

# Admin password for the stats pages. There is no default: without it the
# stats pages are disabled and only the voting portals run.
ADMIN_PASSWORD = os.environ.get("EVAL_ADMIN_PASSWORD", "")
STATS_DISABLED_MESSAGE = (
    "The stats pages are disabled because EVAL_ADMIN_PASSWORD is not set. "
    "Restart the server with EVAL_ADMIN_PASSWORD=<password> to enable them."
)

# ── Flask setup ──────────────────────────────────────────────────────────────
app = Flask(__name__)
# Signs the session cookie. Without FLASK_SECRET a random key is generated per
# process, so sessions (and stats logins) reset when the server restarts.
app.secret_key = os.environ.get("FLASK_SECRET") or os.urandom(32).hex()

# ── Thread-safe state ───────────────────────────────────────────────────────
lock = threading.Lock()

# Registries: {uuid: {path, ground_truth, category, filename}}
registries = {
    "sd2-fr": {},
    "custom": {},
}

# Vote data: {uuid: {votes: [...], total_real_votes, total_fake_votes}}
vote_data = {
    "sd2-fr": {},
    "custom": {},
}

# Session tracking: {session_id: set(voted_uuids)}
session_votes = {
    "sd2-fr": {},
    "custom": {},
}

# ── Persistence helpers ─────────────────────────────────────────────────────
VOTE_FILES = {
    "sd2-fr": RESULTS_DIR / "sd2_fr_votes.json",
    "custom": RESULTS_DIR / "custom_votes.json",
}

REGISTRY_FILES = {
    "sd2-fr": DATA_DIR / "sd2_fr_registry.json",
    "custom": DATA_DIR / "custom_registry.json",
}

SAVE_INTERVAL = 5  # seconds between auto-saves


def load_data():
    """Load registries and vote data from disk."""
    for ds in ("sd2-fr", "custom"):
        reg_file = REGISTRY_FILES[ds]
        vote_file = VOTE_FILES[ds]

        if reg_file.exists():
            with open(reg_file) as f:
                registries[ds] = json.load(f)
            print(f"Loaded {len(registries[ds])} images for {ds}")
        else:
            print(f"WARNING: Registry file not found: {reg_file}")
            print(f"  Run 'python setup_registry.py' first!")

        if vote_file.exists():
            with open(vote_file) as f:
                vote_data[ds] = json.load(f)
            print(f"Loaded vote data for {ds} ({len(vote_data[ds])} entries)")
        else:
            # Initialize from registry
            vote_data[ds] = {}
            for uid in registries[ds]:
                vote_data[ds][uid] = {
                    "votes": [],
                    "total_real_votes": 0,
                    "total_fake_votes": 0,
                }


def save_votes(dataset=None):
    """Persist vote data to disk."""
    datasets = [dataset] if dataset else ["sd2-fr", "custom"]
    for ds in datasets:
        vote_file = VOTE_FILES[ds]
        vote_file.parent.mkdir(parents=True, exist_ok=True)
        with open(vote_file, "w") as f:
            json.dump(vote_data[ds], f, indent=2)


def auto_save_loop():
    """Background thread that periodically saves vote data."""
    while True:
        time.sleep(SAVE_INTERVAL)
        with lock:
            save_votes()


# ── Helper: get or create session ID ────────────────────────────────────────
def get_session_id():
    if "eval_session_id" not in session:
        session["eval_session_id"] = os.urandom(16).hex()
    return session["eval_session_id"]


def get_session_voted(dataset):
    """Get the set of UUIDs this session has voted on."""
    sid = get_session_id()
    if sid not in session_votes[dataset]:
        session_votes[dataset][sid] = set()
    return session_votes[dataset][sid]


# ── Helper: pick next image ─────────────────────────────────────────────────
def pick_next_image(dataset):
    """
    Pick the next image for the current session.
    
    Priority:
    1. Images not yet voted on by this session AND with lowest global vote count
    2. If this session has voted on everything, pick random (any image)
    """
    registry = registries[dataset]
    votes = vote_data[dataset]
    session_voted = get_session_voted(dataset)

    if not registry:
        return None

    all_uuids = set(registry.keys())
    unseen_by_session = all_uuids - session_voted

    if unseen_by_session:
        # Find the minimum vote count among unseen images
        min_votes = float("inf")
        for uid in unseen_by_session:
            total = 0
            if uid in votes:
                total = votes[uid].get("total_real_votes", 0) + votes[uid].get("total_fake_votes", 0)
            min_votes = min(min_votes, total)

        # Collect all unseen images with the minimum vote count
        candidates = []
        for uid in unseen_by_session:
            total = 0
            if uid in votes:
                total = votes[uid].get("total_real_votes", 0) + votes[uid].get("total_fake_votes", 0)
            if total == min_votes:
                candidates.append(uid)

        return random.choice(candidates)
    else:
        # Session has seen everything — full coverage achieved, pick random
        return random.choice(list(all_uuids))


# ── Dataset validation decorator ────────────────────────────────────────────
VALID_DATASETS = {"sd2-fr", "custom"}


def validate_dataset(dataset):
    if dataset not in VALID_DATASETS:
        abort(404, description="Dataset not found")
    if not registries[dataset]:
        abort(503, description="Dataset not loaded. Run setup_registry.py first.")


# ── Routes ───────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    """Root page — generic landing, no portal links."""
    return render_template("landing.html")


@app.route("/<dataset>/")
def vote_page(dataset):
    """Voting page for a dataset."""
    validate_dataset(dataset)
    sid = get_session_id()
    session_voted = get_session_voted(dataset)
    total_images = len(registries[dataset])
    voted_count = len(session_voted)

    dataset_label = "Image Authenticity Evaluation"

    return render_template(
        "vote.html",
        dataset=dataset,
        dataset_label=dataset_label,
        voted_count=voted_count,
    )


@app.route("/<dataset>/next-image")
def next_image(dataset):
    """Get the next image UUID for the current session."""
    validate_dataset(dataset)

    with lock:
        uid = pick_next_image(dataset)

    if uid is None:
        return jsonify({"error": "No images available"}), 404

    session_voted = get_session_voted(dataset)
    voted_count = len(session_voted)

    return jsonify({
        "uuid": uid,
        "image_url": f"/{dataset}/image/{uid}",
        "voted_count": voted_count,
    })


@app.route("/<dataset>/image/<uid>")
def serve_image(dataset, uid):
    """Serve an image by UUID. Resizes sd2-fr images to 224x224."""
    validate_dataset(dataset)

    if uid not in registries[dataset]:
        abort(404)

    img_path = resolve_image_path(registries[dataset][uid]["path"])

    if not os.path.exists(img_path):
        abort(404, description="Image file not found on disk")

    try:
        img = Image.open(img_path).convert("RGB")

        # Resize sd2-fr images from 512x512 to 224x224
        if dataset == "sd2-fr":
            img = img.resize((224, 224), Image.LANCZOS)

        # Serve as PNG in memory (strip all metadata)
        buf = BytesIO()
        img.save(buf, format="PNG")
        buf.seek(0)

        return send_file(
            buf,
            mimetype="image/png",
            download_name="image.png",  # Generic name, reveals nothing
        )
    except Exception as e:
        print(f"Error serving image {uid}: {e}")
        abort(500)


@app.route("/<dataset>/vote", methods=["POST"])
def record_vote(dataset):
    """Record a vote for an image."""
    validate_dataset(dataset)

    data = request.get_json()
    if not data:
        return jsonify({"error": "Invalid request"}), 400

    uid = data.get("uuid")
    vote = data.get("vote", "").upper()

    if uid not in registries[dataset]:
        return jsonify({"error": "Invalid image ID"}), 400

    if vote not in ("REAL", "FAKE"):
        return jsonify({"error": "Vote must be REAL or FAKE"}), 400

    sid = get_session_id()

    with lock:
        # Initialize vote record if missing
        if uid not in vote_data[dataset]:
            vote_data[dataset][uid] = {
                "votes": [],
                "total_real_votes": 0,
                "total_fake_votes": 0,
            }

        # Record the vote
        vote_data[dataset][uid]["votes"].append({
            "session_id": sid,
            "vote": vote,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        })

        # Update tallies
        if vote == "REAL":
            vote_data[dataset][uid]["total_real_votes"] += 1
        else:
            vote_data[dataset][uid]["total_fake_votes"] += 1

        # Track in session
        session_voted = get_session_voted(dataset)
        session_voted.add(uid)

        # Save immediately on vote
        save_votes(dataset)

    return jsonify({"status": "ok"})


def stats_authorized():
    """True once this session has logged in to the stats pages."""
    return bool(ADMIN_PASSWORD) and session.get("stats_authorized") is True


@app.route("/<dataset>/stats", methods=["GET", "POST"])
def stats_page(dataset):
    """Admin stats page (password-protected).

    The password is POSTed from the login form and the login is kept in the
    signed session, so it never appears in a URL, the browser history or the
    server's request log.
    """
    validate_dataset(dataset)

    if not ADMIN_PASSWORD:
        return STATS_DISABLED_MESSAGE, 503

    if request.method == "POST":
        pwd = request.form.get("pwd", "")
        if hmac.compare_digest(pwd.encode(), ADMIN_PASSWORD.encode()):
            session["stats_authorized"] = True
            return redirect(url_for("stats_page", dataset=dataset))
        return render_template("stats_login.html", dataset=dataset, failed=True), 401

    if not stats_authorized():
        return render_template("stats_login.html", dataset=dataset), 401

    registry = registries[dataset]
    votes = vote_data[dataset]

    # Compute stats
    total_images = len(registry)
    total_votes = 0
    images_with_votes = 0
    images_without_votes = 0

    category_stats = {}  # category → {total, voted, real_votes, fake_votes}
    label_stats = {"REAL": {"total": 0, "correct": 0, "incorrect": 0, "votes": 0},
                   "FAKE": {"total": 0, "correct": 0, "incorrect": 0, "votes": 0}}

    per_image_data = []

    unique_sessions = set()

    for uid, info in registry.items():
        gt = info["ground_truth"]
        cat = info["category"]
        v = votes.get(uid, {"votes": [], "total_real_votes": 0, "total_fake_votes": 0})

        img_total = v["total_real_votes"] + v["total_fake_votes"]
        total_votes += img_total

        if img_total > 0:
            images_with_votes += 1
        else:
            images_without_votes += 1

        # Track unique sessions
        for vote_record in v["votes"]:
            unique_sessions.add(vote_record["session_id"])

        # Category stats
        if cat not in category_stats:
            category_stats[cat] = {
                "total_images": 0,
                "images_voted": 0,
                "total_real_votes": 0,
                "total_fake_votes": 0,
            }
        category_stats[cat]["total_images"] += 1
        if img_total > 0:
            category_stats[cat]["images_voted"] += 1
        category_stats[cat]["total_real_votes"] += v["total_real_votes"]
        category_stats[cat]["total_fake_votes"] += v["total_fake_votes"]

        # Label-level stats (accuracy)
        label_stats[gt]["total"] += 1
        label_stats[gt]["votes"] += img_total
        # Count correct votes for this image
        for vote_record in v["votes"]:
            if vote_record["vote"] == gt:
                label_stats[gt]["correct"] += 1
            else:
                label_stats[gt]["incorrect"] += 1

        per_image_data.append({
            "uuid": uid[:8] + "...",
            "category": cat,
            "ground_truth": gt,
            "real_votes": v["total_real_votes"],
            "fake_votes": v["total_fake_votes"],
            "total_votes": img_total,
        })

    # Sort per-image by total votes ascending (least voted first)
    per_image_data.sort(key=lambda x: x["total_votes"])

    # Overall accuracy
    total_correct = label_stats["REAL"]["correct"] + label_stats["FAKE"]["correct"]
    overall_accuracy = (total_correct / total_votes * 100) if total_votes > 0 else 0

    dataset_label = "sd2-fr-testing" if dataset == "sd2-fr" else "sd2-classification/test"

    return render_template(
        "stats.html",
        dataset=dataset,
        dataset_label=dataset_label,
        total_images=total_images,
        total_votes=total_votes,
        images_with_votes=images_with_votes,
        images_without_votes=images_without_votes,
        unique_sessions=len(unique_sessions),
        category_stats=category_stats,
        label_stats=label_stats,
        overall_accuracy=round(overall_accuracy, 2),
        per_image_data=per_image_data[:100],  # Show top 100 least voted
        per_image_total=len(per_image_data),
    )


@app.route("/<dataset>/stats/download")
def download_stats(dataset):
    """Download full vote data as JSON."""
    validate_dataset(dataset)

    if not ADMIN_PASSWORD:
        return jsonify({"error": STATS_DISABLED_MESSAGE}), 503
    if not stats_authorized():
        return jsonify({"error": "Unauthorized"}), 401

    registry = registries[dataset]
    votes = vote_data[dataset]

    # Build export with ground truth included
    export = {}
    for uid, info in registry.items():
        v = votes.get(uid, {"votes": [], "total_real_votes": 0, "total_fake_votes": 0})
        export[uid] = {
            "category": info["category"],
            "ground_truth": info["ground_truth"],
            "total_real_votes": v["total_real_votes"],
            "total_fake_votes": v["total_fake_votes"],
            "votes": v["votes"],
        }

    buf = BytesIO()
    buf.write(json.dumps(export, indent=2).encode())
    buf.seek(0)

    return send_file(
        buf,
        mimetype="application/json",
        download_name=f"{dataset}_full_results.json",
        as_attachment=True,
    )


# ── Startup ──────────────────────────────────────────────────────────────────
load_data()

# Start auto-save thread
save_thread = threading.Thread(target=auto_save_loop, daemon=True)
save_thread.start()

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5050))
    print(f"\n{'='*60}")
    print(f"  Human Evaluation Server")
    print(f"  Portal A (sd2-fr-testing):        http://localhost:{port}/sd2-fr/")
    print(f"  Portal B (sd2-classification):     http://localhost:{port}/custom/")
    if ADMIN_PASSWORD:
        print(f"  Stats A:  http://localhost:{port}/sd2-fr/stats")
        print(f"  Stats B:  http://localhost:{port}/custom/stats")
    else:
        print("  Stats:    disabled (set EVAL_ADMIN_PASSWORD to enable them)")
    print(f"{'='*60}\n")
    app.run(host="0.0.0.0", port=port, debug=False)
