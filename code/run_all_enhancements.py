"""
run_all_enhancements.py
══════════════════════════════════════════════════════════════════════════════
Master Runner — All Enhancement Scripts (07–14)
─────────────────────────────────────────────────────────────────────────────
Runs all enhancement scripts in sequence, tracking time and status.
Estimated total runtime: 15–40 minutes depending on hardware.

Usage (from project root):
    python code/run_all_enhancements.py

Individual scripts can also be run standalone:
    python code/07_spatial_features.py
    python code/08_ablation_study.py
    ... etc.
══════════════════════════════════════════════════════════════════════════════
"""
import subprocess
import sys
import time
from pathlib import Path

BASE     = Path(__file__).parent.parent
CODE_DIR = BASE / "code"

SCRIPTS = [
    ("07_spatial_features.py",  "Spatial Graph Feature Augmentation"),
    ("08_ablation_study.py",    "Feature Ablation Study"),
    ("09_walkforward_cv.py",    "Walk-Forward Cross-Validation"),
    ("10_diwali_analysis.py",   "Diwali Event-Specific Analysis"),
    ("11_vae_anomaly.py",       "Variational Autoencoder Anomaly Detection"),
    ("12_significance_tests.py","Statistical Significance Tests"),
    ("13_ensemble_stacking.py", "Ensemble Stacking"),
    ("14_polish_plots.py",      "Polish Plots (SHAP, t-SNE, Heatmaps)"),
]

print("=" * 70)
print("  DELHI AQ — ENHANCEMENT SUITE RUNNER")
print("  Running all 8 enhancement scripts (07–14)")
print("=" * 70)

total_start = time.time()
results = []

for script_name, description in SCRIPTS:
    script_path = CODE_DIR / script_name
    print(f"\n{'─'*70}")
    print(f"  ▶  {description}")
    print(f"     {script_name}")
    print(f"{'─'*70}")

    if not script_path.exists():
        print(f"  ✗  SKIPPED: script not found at {script_path}")
        results.append((script_name, "SKIPPED", 0))
        continue

    t0 = time.time()
    try:
        proc = subprocess.run(
            [sys.executable, str(script_path)],
            cwd=str(BASE),
            capture_output=False,
            timeout=1800   # 30 min max per script
        )
        elapsed = time.time() - t0
        if proc.returncode == 0:
            print(f"\n  ✓  COMPLETED in {elapsed:.1f}s")
            results.append((script_name, "SUCCESS", elapsed))
        else:
            print(f"\n  ✗  FAILED (exit code {proc.returncode})")
            results.append((script_name, "FAILED", elapsed))
    except subprocess.TimeoutExpired:
        print(f"\n  ✗  TIMEOUT after 30 minutes")
        results.append((script_name, "TIMEOUT", 1800))
    except Exception as e:
        print(f"\n  ✗  ERROR: {e}")
        results.append((script_name, "ERROR", 0))

# ── Final summary ─────────────────────────────────────────────────────────────
total_elapsed = time.time() - total_start
print(f"\n{'=' * 70}")
print(f"  FINAL SUMMARY  (Total time: {total_elapsed/60:.1f} min)")
print(f"{'=' * 70}")
for script, status, elapsed in results:
    symbol = "✓" if status == "SUCCESS" else "✗"
    print(f"  {symbol}  {script:<35}  {status:<10}  {elapsed:.0f}s")

successes = sum(1 for _, s, _ in results if s == "SUCCESS")
print(f"\n  {successes}/{len(SCRIPTS)} scripts completed successfully.")
print(f"\n  New plots saved in:  code/plots/  (07_* through 14_*)")
print(f"  New results saved in: results/   (07_* through 14_*)")
print(f"  New models saved in:  models/    (spatial XGBoost, VAE)")
print("=" * 70)
