"""
15_seasonal_conformal.py
══════════════════════════════════════════════════════════════════════════════
Hierarchical Seasonal Conformal Prediction
─────────────────────────────────────────────────────────────────────────────
Motivation: The global conformal q = 206.74 µg/m³ yields an interval width
of 413 µg/m³ — far too wide in summer when residuals are small.

This script:
  1. Calibrates separate conformal quantiles for Winter (Oct–Feb) and
     Summer (Mar–Sep) using the 2023 validation year.
  2. Evaluates empirical coverage on the 2025 test set for each season.
  3. Compares interval widths: global vs. seasonal calibration.
  4. Saves: seasonal quantiles as PKL, coverage CSV, comparison plot.

Run from project root:
    python code/15_seasonal_conformal.py
══════════════════════════════════════════════════════════════════════════════
"""
import json
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import joblib
import xgboost as xgb
from pathlib import Path
from sklearn.metrics import mean_absolute_error

BASE      = Path(__file__).parent.parent
FEAT_DIR  = BASE / "dataset" / "features"
MODEL_DIR = BASE / "models"
PLOT_DIR  = BASE / "code" / "plots"
RES_DIR   = BASE / "results"
PLOT_DIR.mkdir(exist_ok=True); RES_DIR.mkdir(exist_ok=True)

STATIONS = [
    "Anand_Vihar", "Ashok_Vihar", "Bawana", "Dwarka-Sector_8",
    "Jahangirpuri", "Mundka", "Punjabi_Bagh", "Rohini", "Wazirpur"
]

WINTER_MONTHS = [10, 11, 12, 1, 2]   # Oct–Feb
SUMMER_MONTHS = [3, 4, 5, 6, 7, 8, 9]  # Mar–Sep

print("=" * 70)
print("  15  HIERARCHICAL SEASONAL CONFORMAL PREDICTION")
print("=" * 70)

# ── Load data and model ────────────────────────────────────────────────────────
print("\n[1/4] Loading data and model...")
feat_daily = pd.read_parquet(FEAT_DIR / "features_daily.parquet")
with open(FEAT_DIR / "feature_meta_daily.json") as f:
    meta = json.load(f)

feats  = meta["global_features"]
TARGET = "pm25_target"

model = xgb.XGBRegressor()
model.load_model(str(MODEL_DIR / "xgb_global_daily_4yr.json"))

# Load global conformal q for comparison
global_q = joblib.load(MODEL_DIR / "conformal_q_daily.pkl")
print(f"    Global conformal q (flat): {global_q:.2f} µg/m³")

# ── Step 1: Calibrate seasonal quantiles on 2023 val set ──────────────────────
print("\n[2/4] Calibrating seasonal conformal quantiles on 2023 validation data...")
cal_df = feat_daily[feat_daily.index.year == 2023].copy()
y_cal  = cal_df[TARGET].values
p_cal  = model.predict(cal_df[feats].values)
residuals_cal = np.abs(y_cal - p_cal)

is_winter_cal = cal_df.index.month.isin(WINTER_MONTHS)
is_summer_cal = ~is_winter_cal

# 90th percentile (corrected finite-sample version)
n_winter = is_winter_cal.sum()
n_summer = is_summer_cal.sum()

alpha = 0.10  # 90% coverage
q_winter = np.quantile(residuals_cal[is_winter_cal],
                        np.ceil((1 - alpha) * (n_winter + 1)) / n_winter)
q_summer = np.quantile(residuals_cal[is_summer_cal],
                        np.ceil((1 - alpha) * (n_summer + 1)) / n_summer)

print(f"    Seasonal conformal q (Winter Oct–Feb):  {q_winter:.2f} µg/m³  [width {2*q_winter:.2f}]")
print(f"    Seasonal conformal q (Summer Mar–Sep):  {q_summer:.2f} µg/m³  [width {2*q_summer:.2f}]")
print(f"    Global flat q for comparison:           {global_q:.2f} µg/m³  [width {2*global_q:.2f}]")
print(f"    Summer interval reduction:              {100*(1 - q_summer/global_q):.1f}% narrower than flat cal.")

# Save seasonal quantiles
seasonal_q = {"winter": float(q_winter), "summer": float(q_summer)}
joblib.dump(seasonal_q, MODEL_DIR / "conformal_q_seasonal.pkl")
print(f"    Saved: models/conformal_q_seasonal.pkl")

# ── Step 2: Evaluate coverage on 2025 test set ────────────────────────────────
print("\n[3/4] Evaluating empirical coverage on 2025 test set...")
test_df = feat_daily[feat_daily.index.year == 2025].copy()
y_test  = test_df[TARGET].values
p_test  = model.predict(test_df[feats].values)

is_winter_test = test_df.index.month.isin(WINTER_MONTHS)
is_summer_test = ~is_winter_test

# Assign conformal q per row
conformal_q_test = np.where(is_winter_test, q_winter, q_summer)
lower_seasonal = p_test - conformal_q_test
upper_seasonal = p_test + conformal_q_test
inside_seasonal = (y_test >= lower_seasonal) & (y_test <= upper_seasonal)

# Global flat comparison
lower_global = p_test - global_q
upper_global = p_test + global_q
inside_global = (y_test >= lower_global) & (y_test <= upper_global)

# Per-season breakdown
print(f"\n    {'Season':<12} {'N':<6} {'Seasonal Cov%':<17} {'Global Cov%':<15} {'Seasonal Width':<17} {'Global Width'}")
print(f"    {'-'*12} {'-'*6} {'-'*17} {'-'*15} {'-'*17} {'-'*12}")
for name, mask in [("Winter", is_winter_test), ("Summer", is_summer_test), ("Overall", np.ones(len(y_test), dtype=bool))]:
    if mask.sum() == 0: continue
    cov_s = inside_seasonal[mask].mean() * 100
    cov_g = inside_global[mask].mean()   * 100
    w_s   = 2 * conformal_q_test[mask].mean()
    w_g   = 2 * global_q
    print(f"    {name:<12} {mask.sum():<6} {cov_s:<17.1f} {cov_g:<15.1f} {w_s:<17.1f} {w_g:.1f}")

# Per-station coverage table
print(f"\n    Per-station coverage:")
rows = []
for stn in STATIONS:
    m = test_df["station"] == stn
    if m.sum() == 0: continue
    cov_s = inside_seasonal[m].mean() * 100
    cov_g = inside_global[m].mean()   * 100
    w_s   = 2 * conformal_q_test[m].mean()
    rows.append({
        "Station": stn.replace("_", " "),
        "N": m.sum(),
        "Seasonal Coverage (%)": round(cov_s, 1),
        "Global Coverage (%)":   round(cov_g, 1),
        "Seasonal Width": round(w_s, 1),
        "Global Width": round(2 * global_q, 1),
        "Width Reduction (%)": round(100 * (1 - w_s / (2 * global_q)), 1),
    })
    print(f"    {stn:<22}  Seasonal={cov_s:.1f}%  Global={cov_g:.1f}%  "
          f"Width seasonal={w_s:.1f}  global={2*global_q:.1f}")

cov_df = pd.DataFrame(rows)
cov_df.to_csv(RES_DIR / "15_seasonal_conformal_coverage.csv", index=False)
print(f"\n    Saved: results/15_seasonal_conformal_coverage.csv")

# ── Step 3: Plot comparison ────────────────────────────────────────────────────
print("\n[4/4] Generating comparison plot...")
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle("Hierarchical Seasonal vs. Flat Conformal Prediction",
             fontsize=13, fontweight="bold")

# Coverage comparison bar chart
stn_labels = [r["Station"] for r in rows]
cov_s_vals = [r["Seasonal Coverage (%)"] for r in rows]
cov_g_vals = [r["Global Coverage (%)"]   for r in rows]
x = np.arange(len(stn_labels))
w = 0.35
axes[0].bar(x - w/2, cov_s_vals, w, label="Seasonal cal.", color="#4A9EFF", alpha=0.85)
axes[0].bar(x + w/2, cov_g_vals, w, label="Flat cal.",     color="#C9B8FF", alpha=0.85)
axes[0].axhline(90, ls="--", color="crimson", lw=1.5, label="Claimed 90%")
axes[0].axhline(88, ls=":",  color="orange",  lw=1,   label="Tolerance 88%")
axes[0].set_xticks(x)
axes[0].set_xticklabels([s.replace(" ", "\n") for s in stn_labels], fontsize=7)
axes[0].set_ylabel("Empirical Coverage (%)"); axes[0].set_ylim(80, 102)
axes[0].set_title("Empirical Coverage per Station", fontweight="bold")
axes[0].legend(fontsize=8); axes[0].grid(axis="y", alpha=0.3)

# Interval width comparison
w_s_vals = [r["Seasonal Width"] for r in rows]
w_g_vals = [r["Global Width"]   for r in rows]
axes[1].bar(x - w/2, w_s_vals, w, label="Seasonal cal.", color="#4A9EFF", alpha=0.85)
axes[1].bar(x + w/2, w_g_vals, w, label="Flat cal.",     color="#C9B8FF", alpha=0.85)
axes[1].set_xticks(x)
axes[1].set_xticklabels([s.replace(" ", "\n") for s in stn_labels], fontsize=7)
axes[1].set_ylabel("Interval Width 2q (µg/m³)")
axes[1].set_title("Prediction Interval Width\n(Smaller = More Informative)", fontweight="bold")
axes[1].legend(fontsize=8); axes[1].grid(axis="y", alpha=0.3)

# Width reduction percentage
reduction = [r["Width Reduction (%)"] for r in rows]
bar_colors = ["#43A047" if r > 0 else "#E53935" for r in reduction]
axes[2].barh(range(len(stn_labels)), reduction, color=bar_colors, alpha=0.85)
axes[2].axvline(0, color="black", lw=0.8)
axes[2].set_yticks(range(len(stn_labels)))
axes[2].set_yticklabels(stn_labels, fontsize=9)
axes[2].set_xlabel("Width Reduction vs. Flat Cal. (%)")
axes[2].set_title("% Interval Narrowing\nSeasonal vs. Flat Calibration", fontweight="bold")
axes[2].grid(axis="x", alpha=0.3)

plt.tight_layout()
plt.savefig(PLOT_DIR / "15_seasonal_conformal.png", dpi=150, bbox_inches="tight")
plt.close()
print("    Saved: code/plots/15_seasonal_conformal.png")

print("\n" + "=" * 70)
print("  SEASONAL CONFORMAL SUMMARY")
print(f"  Winter q: {q_winter:.2f}  Summer q: {q_summer:.2f}  (Global flat: {global_q:.2f})")
print(f"  Summer width reduction: {100*(1 - q_summer/global_q):.1f}% narrower")
print(f"  Coverage maintained: {inside_seasonal.mean()*100:.1f}% overall (claimed 90%)")
print("=" * 70)
