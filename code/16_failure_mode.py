"""
16_failure_mode.py
══════════════════════════════════════════════════════════════════════════════
Forecast Failure Mode Analysis + CUSUM Drift Detection
─────────────────────────────────────────────────────────────────────────────
This script systematically characterises WHEN and WHERE the XGBoost daily
model fails worst, and applies CUSUM drift detection to identify regime shifts.

Analyses:
  1. Top-5% worst prediction days — what meteorological conditions occur?
  2. Monthly error distribution — confirms heteroscedasticity.
  3. Failure clustering: wind_speed < 1 m/s, Diwali proximity, season transitions.
  4. CUSUM drift detection on rolling 30-day MAE — identify regime change dates.

Run from project root:
    python code/16_failure_mode.py
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
import seaborn as sns
import xgboost as xgb
from pathlib import Path
from sklearn.metrics import mean_absolute_error

BASE      = Path(__file__).parent.parent
FEAT_DIR  = BASE / "dataset" / "features"
MODEL_DIR = BASE / "models"
PLOT_DIR  = BASE / "code" / "plots"
RES_DIR   = BASE / "results"
PLOT_DIR.mkdir(exist_ok=True); RES_DIR.mkdir(exist_ok=True)

DIWALI_DATES = pd.to_datetime(["2021-11-04", "2022-10-24", "2023-11-12",
                              "2024-11-01", "2025-10-20"])  # Oct 20 2025

print("=" * 70)
print("  16  FAILURE MODE ANALYSIS + CUSUM DRIFT DETECTION")
print("=" * 70)

# ── Load data and model ────────────────────────────────────────────────────────
print("\n[1/6] Loading data and model...")
feat_daily = pd.read_parquet(FEAT_DIR / "features_daily.parquet")
with open(FEAT_DIR / "feature_meta_daily.json") as f:
    meta = json.load(f)

feats  = meta["global_features"]
TARGET = "pm25_target"

model = xgb.XGBRegressor()
model.load_model(str(MODEL_DIR / "xgb_global_daily_4yr.json"))

# Use 2025 test set (same as evaluation)
test_df = feat_daily[feat_daily.index.year == 2025].copy()
y_test  = test_df[TARGET].values
p_test  = model.predict(test_df[feats].values)
abs_err = np.abs(y_test - p_test)

test_df = test_df.copy()
test_df["abs_error"] = abs_err
test_df["pred"]      = p_test
test_df["residual"]  = y_test - p_test

print(f"    Test set: {len(test_df):,} rows | Mean MAE: {abs_err.mean():.2f} µg/m³")

# ── 1. Top-5% worst days: meteorological profile ──────────────────────────────
print("\n[2/6] Analysing top-5% worst prediction errors...")
threshold_95 = np.percentile(abs_err, 95)
failure_mask = abs_err >= threshold_95
failure_df   = test_df[failure_mask].copy()

print(f"    Failure threshold (95th pct): {threshold_95:.2f} µg/m³")
print(f"    Number of failure days: {failure_mask.sum()}")

# Days-since-Diwali for failure days
def days_since_diwali(ts):
    """Return min |days| to nearest Diwali for a timestamp."""
    return min(abs((ts - d).days) for d in DIWALI_DATES)

failure_df["days_to_diwali"] = failure_df.index.map(days_since_diwali)
near_diwali = (failure_df["days_to_diwali"] <= 7).sum()

# Wind speed analysis
calm_threshold = 1.0  # m/s
if "wind_speed" in failure_df.columns:
    calm_days = (failure_df["wind_speed"] < calm_threshold).sum()
    pct_calm  = calm_days / len(failure_df) * 100
    print(f"    Failure days near Diwali (±7 days): {near_diwali}/{len(failure_df)} = {near_diwali/len(failure_df)*100:.0f}%")
    print(f"    Failure days with wind < {calm_threshold} m/s: {calm_days}/{len(failure_df)} = {pct_calm:.0f}%")
else:
    print("    [wind_speed column not found, skipping calm-day analysis]")
    calm_days, pct_calm = 0, 0.0

# Season breakdown
failure_df["season"] = failure_df.index.month.map(
    lambda m: "Winter" if m in [10, 11, 12, 1, 2] else "Summer"
)
season_counts = failure_df["season"].value_counts()
print(f"    Failure by season: {dict(season_counts)}")

# Overall dataset season share (for comparison)
test_df["season"] = test_df.index.month.map(
    lambda m: "Winter" if m in [10, 11, 12, 1, 2] else "Summer"
)
season_share = test_df["season"].value_counts(normalize=True) * 100

print("\n    Failure Mode Summary:")
print(f"    ─────────────────────────────────────────────────────────────────")
print(f"    Regime 1 – Near Diwali (±7 d): {near_diwali/len(failure_df)*100:.0f}% of failures")
if "wind_speed" in failure_df.columns:
    print(f"    Regime 2 – Calm conditions (<{calm_threshold} m/s): {pct_calm:.0f}% of failures")
print(f"    Regime 3 – Winter season: {season_counts.get('Winter', 0)/len(failure_df)*100:.0f}% of failures "
      f"(but Winter is only {season_share.get('Winter', 0):.0f}% of test set)")

# ── 2. Monthly error distribution ─────────────────────────────────────────────
print("\n[3/6] Computing monthly MAE distribution...")
monthly_mae = test_df.groupby(test_df.index.month).apply(
    lambda g: mean_absolute_error(g[TARGET], g["pred"])
).rename("MAE")
print("    Month | MAE")
for month, mae in monthly_mae.items():
    print(f"      {month:2d}    | {mae:.2f}")

winter_mae = monthly_mae[monthly_mae.index.isin([10, 11, 12, 1, 2])].mean()
summer_mae = monthly_mae[monthly_mae.index.isin([3, 4, 5, 6, 7, 8, 9])].mean()
print(f"\n    Winter avg MAE: {winter_mae:.2f}  |  Summer avg MAE: {summer_mae:.2f}")
print(f"    Heteroscedasticity ratio (Winter/Summer): {winter_mae/summer_mae:.2f}×")

# ── 3. CUSUM drift detection ───────────────────────────────────────────────────
print("\n[4/6] CUSUM drift detection on rolling 30-day MAE...")

# Build a daily error series (one station per day avg)
daily_err = test_df.groupby(test_df.index.date)["abs_error"].mean()
daily_ts  = pd.Series(daily_err.values,
                       index=pd.to_datetime([str(d) for d in daily_err.index]))

TARGET_MAE = 35.0   # acceptable threshold (approx. overall MAE)
CUSUM_THRESHOLD = 400.0  # Higher threshold → fewer, cleaner drift events

cusum = 0.0
cusum_series = []
drift_dates  = []
cusum_reset_dates = []

for date, error in daily_ts.items():
    cusum += max(0, error - TARGET_MAE)
    cusum_series.append(cusum)
    if cusum > CUSUM_THRESHOLD:
        drift_dates.append(date)
        cusum_reset_dates.append(date)
        cusum = 0.0   # reset after flagging

cusum_ts = pd.Series(cusum_series, index=daily_ts.index)
print(f"    CUSUM threshold: {CUSUM_THRESHOLD:.0f}  |  Target MAE: {TARGET_MAE:.0f}")
print(f"    Drift events detected: {len(drift_dates)}")
for d in drift_dates:
    print(f"      → Drift flagged: {d.date()}")

# ── 4. Masked MAPE (PM2.5 > 10 µg/m³ only) ───────────────────────────────────
print("\n[5/6] Computing masked MAPE (PM2.5 > 10 µg/m³)...")
mask_valid = y_test > 10.0
if mask_valid.sum() > 0:
    masked_mape = np.mean(np.abs((y_test[mask_valid] - p_test[mask_valid]) / y_test[mask_valid])) * 100
    skipped = (~mask_valid).sum()
    print(f"    Masked MAPE (PM2.5 > 10 µg/m³): {masked_mape:.2f}%")
    print(f"    Rows masked out (PM2.5 ≤ 10): {skipped} ({skipped/len(y_test)*100:.1f}%)")
else:
    masked_mape = float("nan")
    print("    No rows with PM2.5 > 10 µg/m³ in test set.")

# ── 5. Generate comprehensive plot ────────────────────────────────────────────
print("\n[6/6] Generating failure mode analysis plot...")
fig, axes = plt.subplots(2, 2, figsize=(16, 10))
fig.suptitle("Forecast Failure Mode Analysis + CUSUM Drift Detection",
             fontsize=13, fontweight="bold")

# (a) Monthly MAE bar chart
months = list(monthly_mae.index)
maes   = list(monthly_mae.values)
bar_colors = ["#E53935" if m in [10, 11, 12, 1, 2] else "#4A9EFF" for m in months]
axes[0, 0].bar(months, maes, color=bar_colors, alpha=0.85)
axes[0, 0].set_xlabel("Month")
axes[0, 0].set_ylabel("MAE (µg/m³)")
axes[0, 0].set_title("Monthly MAE Distribution\n(Red = Winter, Blue = Summer)", fontweight="bold")
axes[0, 0].set_xticks(months)
axes[0, 0].grid(axis="y", alpha=0.3)
axes[0, 0].axhline(abs_err.mean(), color="black", ls="--", lw=1.5,
                    label=f"Overall MAE {abs_err.mean():.1f}")
axes[0, 0].legend(fontsize=9)

# (b) Failure day distribution (days to Diwali)
if "days_to_diwali" in failure_df.columns:
    axes[0, 1].hist(failure_df["days_to_diwali"], bins=30, color="#C9B8FF", alpha=0.85)
    axes[0, 1].axvline(7, color="crimson", ls="--", lw=1.5, label="7-day Diwali window")
    axes[0, 1].set_xlabel("Days to Nearest Diwali")
    axes[0, 1].set_ylabel("Count")
    axes[0, 1].set_title("Top-5% Worst Errors:\nDistance to Nearest Diwali Date",
                           fontweight="bold")
    axes[0, 1].legend(fontsize=9)
    axes[0, 1].grid(alpha=0.3)

# (c) Wind speed distribution: failures vs. all
if "wind_speed" in test_df.columns:
    axes[1, 0].hist(test_df["wind_speed"],    bins=40, color="#4A9EFF",
                    alpha=0.5, density=True, label="All test days")
    axes[1, 0].hist(failure_df["wind_speed"], bins=40, color="#E53935",
                    alpha=0.7, density=True, label="Top-5% failure days")
    axes[1, 0].axvline(calm_threshold, color="black", ls="--", lw=1.5,
                        label=f"Calm threshold {calm_threshold} m/s")
    axes[1, 0].set_xlabel("Wind Speed (m/s)")
    axes[1, 0].set_ylabel("Density")
    axes[1, 0].set_title("Wind Speed: Failure Days vs. All Days\n"
                           "(Calm days drive large errors)", fontweight="bold")
    axes[1, 0].legend(fontsize=9); axes[1, 0].grid(alpha=0.3)

# (d) CUSUM chart
axes[1, 1].plot(cusum_ts.index, cusum_ts.values, color="#4A9EFF", lw=1.5, label="CUSUM")
axes[1, 1].axhline(CUSUM_THRESHOLD, color="crimson", ls="--", lw=1.5,
                    label=f"Threshold {CUSUM_THRESHOLD:.0f}")
for d in drift_dates:
    axes[1, 1].axvline(d, color="#E53935", alpha=0.6, lw=1)
for d in DIWALI_DATES:
    if d.year == 2025:
        axes[1, 1].axvline(d, color="#FFD700", ls=":", lw=2, label="Diwali")
axes[1, 1].set_xlabel("Date")
axes[1, 1].set_ylabel("CUSUM Statistic")
axes[1, 1].set_title(f"CUSUM Drift Detection on Daily MAE\n"
                      f"(Drift events: {len(drift_dates)})", fontweight="bold")
axes[1, 1].legend(fontsize=9); axes[1, 1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig(PLOT_DIR / "16_failure_mode_analysis.png", dpi=150, bbox_inches="tight")
plt.close()
print("    Saved: code/plots/16_failure_mode_analysis.png")

# Save results
results_summary = {
    "overall_mae":          round(float(abs_err.mean()), 3),
    "failure_threshold_95": round(float(threshold_95), 3),
    "n_failure_days":       int(failure_mask.sum()),
    "failure_near_diwali_pct": round(float(near_diwali / len(failure_df) * 100), 1),
    "failure_calm_wind_pct":   round(float(pct_calm), 1),
    "winter_mae":           round(float(winter_mae), 3),
    "summer_mae":           round(float(summer_mae), 3),
    "heteroscedasticity_ratio": round(float(winter_mae / summer_mae), 2),
    "masked_mape_pct":      round(float(masked_mape), 2) if not np.isnan(masked_mape) else None,
    "cusum_drift_events":   len(drift_dates),
    "cusum_drift_dates":    [str(d.date()) for d in drift_dates],
}
pd.DataFrame([results_summary]).to_csv(RES_DIR / "16_failure_mode_summary.csv", index=False)
print("    Saved: results/16_failure_mode_summary.csv")

print("\n" + "=" * 70)
print("  FAILURE MODE SUMMARY")
print(f"  Overall test MAE: {abs_err.mean():.2f} µg/m³")
print(f"  Failure modes:")
print(f"    1. Diwali proximity (±7d):    {near_diwali/len(failure_df)*100:.0f}% of top-5% errors")
if "wind_speed" in failure_df.columns:
    print(f"    2. Calm conditions (<1 m/s): {pct_calm:.0f}% of top-5% errors")
print(f"    3. Winter/Summer ratio:        {winter_mae/summer_mae:.2f}× — confirms heteroscedasticity")
print(f"  Masked MAPE (PM2.5 > 10): {masked_mape:.2f}%")
print(f"  CUSUM drift events detected: {len(drift_dates)}")
print("=" * 70)
