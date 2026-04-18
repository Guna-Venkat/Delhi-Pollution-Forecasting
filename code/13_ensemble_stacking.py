"""
13_ensemble_stacking.py
══════════════════════════════════════════════════════════════════════════════
Ensemble Stacking: Global XGBoost + Quantile XGBoost Meta-Learner
─────────────────────────────────────────────────────────────────────────────
Trains a linear meta-learner that combines:
  - Global XGBoost point prediction
  - Per-station XGBoost point prediction
  - Q05 quantile XGBoost lower bound
  - Q95 quantile XGBoost upper bound
  - (midpoint and range of quantile interval as derived features)

This demonstrates whether model predictions are complementary.
The meta-learner is trained on 2023 validation year and tested on 2025.

Also produces a per-station × model MAE heatmap showing where each model
excels and where it struggles (Tier 3 "polish" deliverable).

Run from project root:
    python code/13_ensemble_stacking.py
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
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.linear_model import Ridge

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
print("  13  ENSEMBLE STACKING")
print("=" * 70)

# ── Load models ────────────────────────────────────────────────────────────────
print("\n[1/4] Loading all models...")
feat_daily = pd.read_parquet(FEAT_DIR / "features_daily.parquet")
with open(FEAT_DIR / "feature_meta_daily.json") as f:
    meta = json.load(f)
feats  = meta["global_features"]
TARGET = "pm25_target"

global_xgb = xgb.XGBRegressor()
global_xgb.load_model(str(MODEL_DIR / "xgb_global_daily_4yr.json"))

q05_xgb = joblib.load(MODEL_DIR / "xgb_daily_q05.pkl")
q95_xgb = joblib.load(MODEL_DIR / "xgb_daily_q95.pkl")

per_station_models = {}
for stn in STATIONS:
    path = MODEL_DIR / f"xgb_{stn}_daily_4yr.json"
    if path.exists():
        m = xgb.XGBRegressor(); m.load_model(str(path))
        per_station_models[stn] = m
        print(f"    Loaded per-station: {stn}")

# ── Generate predictions ───────────────────────────────────────────────────────
print("\n[2/4] Generating base model predictions for val (2023) and test (2025)...")

def gen_predictions(df):
    """Generate all base model predictions for a dataframe."""
    X = df[feats].values
    preds_global = global_xgb.predict(X)
    preds_q05    = q05_xgb.predict(X)
    preds_q95    = q95_xgb.predict(X)
    preds_mid    = (preds_q05 + preds_q95) / 2.0
    preds_range  = preds_q95 - preds_q05  # Interval width (uncertainty proxy)

    # Per-station prediction (using the correct model for each row)
    preds_pst = np.zeros(len(df))
    for stn, m in per_station_models.items():
        stn_mask = df["station"] == stn
        if stn_mask.any():
            feats_pst = meta.get("per_station_features", feats)
            available = [f for f in feats_pst if f in df.columns]
            preds_pst[stn_mask.values] = m.predict(df.loc[stn_mask, available].values)

    return np.column_stack([preds_global, preds_pst, preds_q05, preds_q95,
                             preds_mid, preds_range])

val_df  = feat_daily[feat_daily.index.year == 2023]
test_df = feat_daily[feat_daily.index.year == 2025]

X_val_meta  = gen_predictions(val_df)
y_val       = val_df[TARGET].values
X_test_meta = gen_predictions(test_df)
y_test      = test_df[TARGET].values

print(f"    Meta-feature matrix shape: val={X_val_meta.shape}, test={X_test_meta.shape}")

# ── Train stacking meta-learner ────────────────────────────────────────────────
print("\n[3/4] Training Ridge meta-learner on val set, evaluating on test...")

meta_learner = Ridge(alpha=1.0)
meta_learner.fit(X_val_meta, y_val)
meta_preds  = meta_learner.predict(X_test_meta)
stack_mae   = mean_absolute_error(y_test, meta_preds)
stack_r2    = r2_score(y_test, meta_preds)

base_preds  = X_test_meta[:, 0]   # global XGBoost
base_mae    = mean_absolute_error(y_test, base_preds)
base_r2     = r2_score(y_test, base_preds)

print(f"\n    Global XGBoost (base)   →  MAE: {base_mae:.3f}  R²: {base_r2:.3f}")
print(f"    Stacking Ensemble       →  MAE: {stack_mae:.3f}  R²: {stack_r2:.3f}")
print(f"    Improvement             →  ΔMAE: {base_mae - stack_mae:+.3f}  ΔR²: {stack_r2 - base_r2:+.3f}")
print(f"\n    Meta-learner weights (Ridge):")
col_names = ["Global XGB", "Per-Station XGB", "Q05 XGB", "Q95 XGB",
             "Quantile Midpoint", "Interval Width"]
for name, w in zip(col_names, meta_learner.coef_):
    print(f"      {name:<22}  weight = {w:.4f}")

# Save stacking results
stack_results = pd.DataFrame([{
    "Model": "Global XGBoost", "MAE": base_mae, "R2": base_r2
}, {
    "Model": "Stacking Ensemble", "MAE": stack_mae, "R2": stack_r2
}])
stack_results.to_csv(RES_DIR / "13_stacking_results.csv", index=False)

# ── Per-station × model heatmap ───────────────────────────────────────────────
print("\n[4/4] Generating per-station × model error heatmap...")

# Build results from the existing CSV
res_csv_path = MODEL_DIR / "results_all_models.csv"
results_all = pd.read_csv(res_csv_path)

# Filter to 4yr, daily, global models we care about
models_to_show = ["SARIMA", "XGBoost", "LSTM", "PatchTST", "Informer"]
hm_data = {}

# Compute per-station MAE for global XGBoost (for which we have the model)
xgb_per_stn_mae = {}
for stn in STATIONS:
    sdf = test_df[test_df["station"] == stn]
    if len(sdf) == 0: continue
    xgb_per_stn_mae[stn] = mean_absolute_error(
        sdf[TARGET], global_xgb.predict(sdf[feats].values)
    )

# Stacking per station
stack_per_stn_mae = {}
for stn in STATIONS:
    sdf     = test_df[test_df["station"] == stn]
    if len(sdf) == 0: continue
    mask    = test_df["station"] == stn
    preds_s = meta_preds[mask.values]
    y_s     = test_df.loc[mask, TARGET].values
    stack_per_stn_mae[stn] = mean_absolute_error(y_s, preds_s)

# Build heatmap matrix
hm_df = pd.DataFrame({
    "XGBoost\n(Global)": xgb_per_stn_mae,
    "XGBoost\n+Stacking": stack_per_stn_mae,
}).T

# ── All plots ──────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle("Ensemble Stacking Results — Daily Global Models",
             fontsize=13, fontweight="bold")

# Stacking comparison bar chart
labels   = ["Global XGBoost", "Stacking\nEnsemble"]
maes     = [base_mae, stack_mae]
r2s      = [base_r2, stack_r2]
colors_b = ["#4A9EFF", "#C9B8FF"]
x_pos    = np.arange(len(labels))

axes[0].bar(x_pos, maes, color=colors_b, alpha=0.85, edgecolor="none", width=0.5)
for i, v in enumerate(maes):
    axes[0].text(i, v + 0.3, f"{v:.2f}", ha="center", fontsize=10, fontweight="bold")
axes[0].set_xticks(x_pos); axes[0].set_xticklabels(labels)
axes[0].set_ylabel("MAE (µg/m³)"); axes[0].set_title("Stacking vs Global XGBoost", fontweight="bold")
axes[0].set_ylim(0, max(maes) * 1.2); axes[0].grid(axis="y", alpha=0.3)

# Meta-learner weights
weights = meta_learner.coef_
w_colors = ["#43A047" if w > 0 else "#E53935" for w in weights]
axes[1].barh(range(len(col_names)), weights, color=w_colors, alpha=0.85)
axes[1].axvline(0, color="k", lw=0.8)
axes[1].set_yticks(range(len(col_names)))
axes[1].set_yticklabels(col_names, fontsize=9)
axes[1].set_xlabel("Ridge Coefficient")
axes[1].set_title("Meta-Learner Feature Weights\n(Model Complementarity)", fontweight="bold")
axes[1].grid(axis="x", alpha=0.3)

# Per-station heatmap
stn_labels = [s.replace("_", " ").replace("Sector ", "S") for s in STATIONS]
plot_df = hm_df.copy()
plot_df.columns = [s.replace("_", " ").replace("Sector ", "S") for s in STATIONS]
sns.heatmap(plot_df, ax=axes[2], cmap="YlOrRd", annot=True, fmt=".1f",
            linewidths=0.5, cbar_kws={"label": "MAE (µg/m³)"})
axes[2].set_title("Per-Station MAE Heatmap:\nGlobal vs Stacking", fontweight="bold")
axes[2].set_xlabel(""); axes[2].set_ylabel("")

plt.tight_layout()
plt.savefig(PLOT_DIR / "13_ensemble_stacking.png", dpi=150, bbox_inches="tight")
plt.close()
print("    Saved: code/plots/13_ensemble_stacking.png")

print("\n" + "=" * 70)
print("  STACKING SUMMARY")
print(f"  Base (Global XGBoost)  MAE={base_mae:.3f}  R²={base_r2:.3f}")
print(f"  Stacking Ensemble      MAE={stack_mae:.3f}  R²={stack_r2:.3f}")
print(f"  Improvement            ΔMAE={base_mae-stack_mae:+.3f}")
print("=" * 70)
