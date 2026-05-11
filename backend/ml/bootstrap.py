"""
bootstrap.py — run all ML training steps in dependency order.

Usage:
    python -m ml.bootstrap                  # run all, skip if outputs exist
    python -m ml.bootstrap --force          # re-run everything
    python -m ml.bootstrap --only data      # single step

Step order (dependencies must be satisfied in sequence):
    1. data_pipeline    → parquet + scaler_stats.json
    2. train_transformer → models/transformer/model.pt
    3. train_lstm        → models/lstm/model.pt
    4. train_cnn         → models/cnn/model.pt
    5. train_vae         → models/vae/model.pt
    (RAG index and CLIP load at server startup — no offline training needed)
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).parent

STEPS = [
    {
        "name":    "data",
        "label":   "data_pipeline",
        "module":  "ml.data_pipeline",
        "outputs": [
            ROOT / "datasets" / "karachi.parquet",
            ROOT / "datasets" / "scaler_stats.json",
            ROOT / "datasets" / "imputation_log.json",
        ],
        "est_min": "2-4",
    },
    {
        "name":    "transformer",
        "label":   "train_transformer",
        "module":  "ml.train.train_transformer",
        "outputs": [
            ROOT / "models" / "transformer" / "model.pt",
            ROOT / "models" / "transformer" / "attn_sample.npy",
        ],
        "est_min": "8-12",
    },
    {
        "name":    "lstm",
        "label":   "train_lstm",
        "module":  "ml.train.train_lstm",
        "outputs": [
            ROOT / "models" / "lstm" / "model.pt",
        ],
        "est_min": "3-5",
    },
    {
        "name":    "cnn",
        "label":   "train_cnn",
        "module":  "ml.train.train_cnn",
        "outputs": [
            ROOT / "models" / "cnn" / "model.pt",
        ],
        "est_min": "3-5",
    },
    {
        "name":    "vae",
        "label":   "train_vae",
        "module":  "ml.train.train_vae",
        "outputs": [
            ROOT / "models" / "vae" / "model.pt",
            ROOT / "models" / "vae" / "anomaly_threshold.json",
        ],
        "est_min": "2-3",
    },
]


def _all_outputs_exist(step: dict) -> bool:
    return all(p.exists() for p in step["outputs"])


def _run_step(step: dict, force: bool) -> tuple[str, float]:
    """
    Run one step as a subprocess.
    Returns (status, elapsed_seconds).
    status: "ok" | "skipped" | "failed"
    """
    if not force and _all_outputs_exist(step):
        return "skipped", 0.0

    print(f"\n{'='*60}")
    print(f"  [{step['label']}]  estimated: {step['est_min']} min")
    print(f"{'='*60}")

    start = time.time()
    extra = ["--force"] if force else []
    result = subprocess.run(
        [sys.executable, "-m", step["module"]] + extra,
        cwd=ROOT.parent,   # run from backend/
    )
    elapsed = time.time() - start

    if result.returncode != 0:
        print(f"\n  [FAILED] {step['label']} exited with code {result.returncode}")
        return "failed", elapsed

    return "ok", elapsed


def run_bootstrap(force: bool = False, only: str | None = None) -> None:
    steps = STEPS if only is None else [s for s in STEPS if s["name"] == only]

    if not steps:
        names = [s["name"] for s in STEPS]
        print(f"Unknown step '{only}'. Choose from: {names}")
        sys.exit(1)

    print("Weathering With You - ML Bootstrap")
    print(f"Steps to run: {[s['name'] for s in steps]}")
    print(f"Force re-run: {force}\n")

    results = []
    total_start = time.time()

    for step in steps:
        status, elapsed = _run_step(step, force)
        results.append((step["label"], status, elapsed))

        if status == "failed":
            print(f"\n[bootstrap] Stopping - {step['label']} failed.")
            print("Fix the error above, then re-run. Completed steps are cached.")
            break

    # ── summary table ──────────────────────────────────────────────────────────
    total_elapsed = time.time() - total_start
    print(f"\n{'='*60}")
    print(f"  Bootstrap summary  (total: {total_elapsed/60:.1f} min)")
    print(f"{'='*60}")
    print(f"  {'Step':<22} {'Status':<10} {'Time'}")
    print(f"  {'-'*22} {'-'*10} {'-'*8}")
    for label, status, elapsed in results:
        icon = "OK" if status == "ok" else ("--" if status == "skipped" else "XX")
        t    = f"{elapsed/60:.1f} min" if elapsed > 0 else "-"
        print(f"  {icon} {label:<21} {status:<10} {t}")

    all_ok = all(s in ("ok", "skipped") for _, s, _ in results)
    print()
    if all_ok:
        print("  All steps complete. Start the server:")
        print("  cd backend && uvicorn main:app --reload --port 8000")
    else:
        print("  Some steps failed - check output above.")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run all ML training steps.")
    parser.add_argument("--force", action="store_true",
                        help="Re-run even if outputs already exist.")
    parser.add_argument("--only", metavar="STEP",
                        help=f"Run a single step: {[s['name'] for s in STEPS]}")
    args = parser.parse_args()
    run_bootstrap(force=args.force, only=args.only)
