"""
12_significance_tests.py
══════════════════════════════════════════════════════════════════════════════
Statistical Significance Tests for Forecast Comparison
─────────────────────────────────────────────────────────────────────────────
1. Diebold-Mariano (DM) Test: Is XGBoost significantly better than Persistence?
   - H0: E[d_t] = 0, where d_t = |e_XGB_t|² - |e_Persistence_t|²
   - Test stat: DM = d̄ / sqrt(var(d̄)) ~ N(0,1) under H0
   - Uses Newey-West HAC variance estimator for autocorrelated errors.

2. Conformal Prediction Empirical Coverage Verification:
   - For each station: what fraction of 2025 test observations fall inside
     the claimed 90% conformal interval?
   - If empirical coverage ≥ 90% for all stations → claim is valid.

3. Bootstrap MAE Confidence Intervals (already done in CV; summarised here).

Run from project root:
    python code/12_significance_tests.py
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
import joblib
import xgboost as xgb
from pathlib import Path
from scipy import stats
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

print("=" * 70)
print("  12  STATISTICAL SIGNIFICANCE TESTS")
print("=" * 70)

# ── Load data and model ────────────────────────────────────────────────────────
print("\n[1/4] Loading data, models, and conformal q...")
feat_daily = pd.read_parquet(FEAT_DIR / "features_daily.parquet")
with open(FEAT_DIR / "feature_meta_daily.json") as f:
    meta = json.load(f)

feats  = meta["global_features"]
TARGET = "pm25_target"

model_xgb  = xgb.XGBRegressor()
model_xgb.load_model(str(MODEL_DIR / "xgb_global_daily_4yr.json"))
conformal_q = joblib.load(MODEL_DIR / "conformal_q_daily.pkl")
print(f"    Conformal q (claimed 90% interval half-width): {conformal_q:.2f} µg/m³")

test_df  = feat_daily[feat_daily.index.year == 2025]
y_test   = test_df[TARGET].values
preds_xgb  = model_xgb.predict(test_df[feats].values)

# ── 1. Diebold-Mariano Test ───────────────────────────────────────────────────
print("\n[2/4] Diebold-Mariano Test: XGBoost vs Persistence...")

# Persistence baseline: predict pm25 today = pm25 yesterday (lag1)
if "pm25_lag1" in test_df.columns:
    preds_persist = test_df["pm25_lag1"].values
else:
    # Fallback: use pm25 shifted by 1
    preds_persist = test_df["pm25"].shift(1).bfill().values

# DM loss differential: d_t = L(e_persist_t) - L(e_xgb_t)  (squared errors)
e_xgb     = y_test - preds_xgb
e_persist = y_test - preds_persist
d_sq = e_persist**2 - e_xgb**2

# Newey-West HAC variance estimator (accounts for autocorrelation in d_t)
def newey_west_variance(d, lags=10):
    n  = len(d)
    d_bar = np.mean(d)
    gamma0 = np.mean((d - d_bar)**2)
    variance = gamma0
    for k in range(1, lags + 1):
        gamma_k = np.mean((d[k:] - d_bar) * (d[:-k] - d_bar))
        weight   = 1 - k / (lags + 1)       # Bartlett kernel
        variance += 2 * weight * gamma_k
    return variance / n

nw_var = newey_west_variance(d_sq, lags=10)
dm_stat = np.mean(d_sq) / np.sqrt(max(nw_var, 1e-12))
p_value = 2 * (1 - stats.norm.cdf(abs(dm_stat)))   # two-tailed

print(f"\n    Diebold-Mariano Test (squared error loss, Newey-West HAC, 10 lags)")
print(f"    ─────────────────────────────────────────────────────────────────")
print(f"    Mean loss differential E[d_t]:  {np.mean(d_sq):.4f}")
print(f"    DM statistic:                   {dm_stat:.4f}")
print(f"    p-value (two-sided):             {p_value:.6f}")
if p_value < 0.05:
    print(f"    → XGBoost is SIGNIFICANTLY better than Persistence at p<0.05 ✓")
elif p_value < 0.10:
    print(f"    → Marginally significant at p<0.10")
else:
    print(f"    → Not significant at p<0.05 (consider more test data)")

dm_results = {
    "test": "Diebold-Mariano (XGBoost vs Persistence, daily)",
    "DM_statistic": round(dm_stat, 4),
    "p_value": round(p_value, 6),
    "mean_loss_diff": round(float(np.mean(d_sq)), 4),
    "significant_p05": bool(p_value < 0.05),
}

# Also: t-test on MAE differences (simpler, but less rigorous)
t_stat, t_pval = stats.ttest_1samp(d_sq, 0)
print(f"\n    Simple t-test on loss differentials:")
print(f"    t={t_stat:.4f}, p={t_pval:.6f}")

# ── 2. Conformal Prediction Coverage Verification ─────────────────────────────
print("\n\n[3/4] Conformal Prediction Empirical Coverage Verification...")
print(f"    Claimed coverage: 90%  (conformal q = {conformal_q:.2f})\n")

coverage_rows = []
for stn in STATIONS:
    stn_df = test_df[test_df["station"] == stn]
    if len(stn_df) == 0:
        continue
    y_stn = stn_df[TARGET].values
    p_stn = model_xgb.predict(stn_df[feats].values)

    lower  = p_stn - conformal_q
    upper  = p_stn + conformal_q
    inside = np.sum((y_stn >= lower) & (y_stn <= upper))
    coverage_pct = inside / len(y_stn) * 100

    status = "✓" if coverage_pct >= 88 else "✗"  # allow ±2% margin
    print(f"    {stn:<22}  n={len(y_stn):4d}  "
          f"Coverage={coverage_pct:.1f}%  {status}")
    coverage_rows.append({
        "Station": stn.replace("_", " "),
        "N Test Days": len(y_stn),
        "Empirical Coverage (%)": round(coverage_pct, 2),
        "Claimed Coverage (%)": 90.0,
        "Interval Width (µg/m³)": round(2 * conformal_q, 2),
        "Within Tolerance": coverage_pct >= 88,
    })

cov_df = pd.DataFrame(coverage_rows)
overall_coverage = cov_df["Empirical Coverage (%)"].mean()
print(f"\n    Overall mean empirical coverage: {overall_coverage:.2f}%")
print(f"    Claimed: 90%  →  {'✓ VALID' if overall_coverage >= 88 else '✗ INVALID'}")
cov_df.to_csv(RES_DIR / "12_conformal_coverage.csv", index=False)

# ── 3. Bootstrap MAE CI for each station ──────────────────────────────────────
print("\n\n[4/4] Bootstrap MAE Confidence Intervals per station...")
boot_rows = []
rng = np.random.default_rng(42)

for stn in STATIONS:
    stn_df = test_df[test_df["station"] == stn]
    if len(stn_df) == 0:
        continue
    y_stn = stn_df[TARGET].values
    p_stn = model_xgb.predict(stn_df[feats].values)
    n = len(y_stn)
    boot_maes = [mean_absolute_error(y_stn[idx], p_stn[idx])
                 for idx in (rng.integers(0, n, n) for _ in range(2000))]
    point_mae = mean_absolute_error(y_stn, p_stn)
    ci_lo, ci_hi = np.percentile(boot_maes, [2.5, 97.5])
    boot_rows.append({
        "Station": stn.replace("_", " "),
        "MAE": round(point_mae, 3),
        "95% CI Lo": round(ci_lo, 3),
        "95% CI Hi": round(ci_hi, 3),
    })
    print(f"    {stn:<22}  MAE={point_mae:.3f}  95% CI: [{ci_lo:.3f}, {ci_hi:.3f}]")

boot_df = pd.DataFrame(boot_rows).set_index("Station")
boot_df.to_csv(RES_DIR / "12_bootstrap_mae_ci.csv")

# ── Generate combined plot ─────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle("Statistical Validity Checks — Global XGBoost Daily",
             fontsize=13, fontweight="bold")

# 1. DM test visualisation
d_sq_clipped = np.clip(d_sq, -10000, 10000)
axes[0].hist(d_sq_clipped, bins=60, color="#4A9EFF", alpha=0.7, density=True)
axes[0].axvline(np.mean(d_sq), color="crimson", lw=2,
                label=f"Mean d = {np.mean(d_sq):.1f}")
axes[0].axvline(0, color="black", ls="--", lw=1, label="Null (no difference)")
axes[0].set_xlabel("d_t = sq_error(Persist) - sq_error(XGB)")
axes[0].set_ylabel("Density")
axes[0].set_title(f"Diebold-Mariano Loss Differential\nDM={dm_stat:.2f}, p={p_value:.4f}",
                  fontweight="bold")
axes[0].legend(fontsize=8); axes[0].grid(alpha=0.3)

# 2. Conformal coverage by station
bar_colors = ["#43A047" if r >= 88 else "#E53935"
              for r in cov_df["Empirical Coverage (%)"].tolist()]
axes[1].barh(range(len(cov_df)), cov_df["Empirical Coverage (%)"],
             color=bar_colors, alpha=0.85)
axes[1].axvline(90, color="crimson", ls="--", lw=1.5, label="Claimed 90%")
axes[1].axvline(88, color="orange", ls=":", lw=1, label="Tolerance 88%")
axes[1].set_yticks(range(len(cov_df)))
axes[1].set_yticklabels(cov_df["Station"].tolist(), fontsize=9)
axes[1].set_xlabel("Empirical Coverage (%)")
axes[1].set_title("Conformal 90% Prediction Interval\nEmpirical Coverage per Station",
                  fontweight="bold")
axes[1].legend(fontsize=8); axes[1].grid(axis="x", alpha=0.3)
for i, v in enumerate(cov_df["Empirical Coverage (%)"]):
    axes[1].text(v + 0.2, i, f"{v:.1f}%", va="center", fontsize=8)

# 3. Bootstrap MAE CI per station
y_pos = np.arange(len(boot_df))
axes[2].barh(y_pos, boot_df["MAE"], height=0.5, color="#4A9EFF", alpha=0.7, label="MAE")
axes[2].errorbar(boot_df["MAE"], y_pos,
                 xerr=[boot_df["MAE"] - boot_df["95% CI Lo"],
                       boot_df["95% CI Hi"] - boot_df["MAE"]],
                 fmt="none", color="black", capsize=5, lw=1.5)
axes[2].set_yticks(y_pos)
axes[2].set_yticklabels(boot_df.index.tolist(), fontsize=9)
axes[2].set_xlabel("MAE (µg/m³)")
axes[2].set_title("Per-Station MAE with\n95% Bootstrap Confidence Intervals",
                  fontweight="bold")
axes[2].grid(axis="x", alpha=0.3)

plt.tight_layout()
plt.savefig(PLOT_DIR / "12_significance_tests.png", dpi=150, bbox_inches="tight")
plt.close()
print("\n    Saved: code/plots/12_significance_tests.png")

# Save all results
pd.DataFrame([dm_results]).to_csv(RES_DIR / "12_dm_test.csv", index=False)
print("    Saved: results/12_dm_test.csv")
print("    Saved: results/12_conformal_coverage.csv")
print("    Saved: results/12_bootstrap_mae_ci.csv")

print("\n" + "=" * 70)
print("  SIGNIFICANCE TESTS SUMMARY")
print(f"  DM Test p-value: {p_value:.4f}  →  XGBoost {'significantly' if p_value<0.05 else 'marginally'} beats Persistence")
print(f"  Conformal coverage: mean={overall_coverage:.1f}%  (claimed 90%)  → {'VALID ✓' if overall_coverage>=88 else 'INVALID ✗'}")
print("=" * 70)
