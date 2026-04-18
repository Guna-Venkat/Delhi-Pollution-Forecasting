"""
08_ablation_study.py
══════════════════════════════════════════════════════════════════════════════
Systematic Feature Ablation Study for Global XGBoost (Daily, 4yr)
─────────────────────────────────────────────────────────────────────────────
Removes each feature group one at a time, retrains, and measures MAE delta.
Feature groups tested:
  1. Event features       : days_since_diwali, is_diwali, is_stubble_season
  2. Wind decomposition   : wind_u, wind_v  (keep raw wind_speed, wind_dir)
  3. Cyclical encoding    : *_sin, *_cos    (keep raw month, year, dow_sin etc.)
  4. Rolling windows      : pm25_roll_*     (all rolling features)
  5. Interaction feature  : humid_temp_interaction
  6. AQI lags             : aqi_lag*        (keep pm25 lags only)

Run from project root:
    python code/08_ablation_study.py
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
import xgboost as xgb
from pathlib import Path
from sklearn.metrics import mean_absolute_error, r2_score

BASE      = Path(__file__).parent.parent
FEAT_DIR  = BASE / "dataset" / "features"
PLOT_DIR  = BASE / "code" / "plots"
RES_DIR   = BASE / "results"
PLOT_DIR.mkdir(exist_ok=True); RES_DIR.mkdir(exist_ok=True)

print("=" * 70)
print("  08  FEATURE ABLATION STUDY")
print("=" * 70)

# ── Load data ─────────────────────────────────────────────────────────────────
print("\n[1/3] Loading features and metadata...")
feat_daily = pd.read_parquet(FEAT_DIR / "features_daily.parquet")
with open(FEAT_DIR / "feature_meta_daily.json") as f:
    meta = json.load(f)

all_global = meta["global_features"]
TARGET = "pm25_target"

train_df = feat_daily[feat_daily.index.year.isin([2021, 2022, 2023, 2024])]
test_df  = feat_daily[feat_daily.index.year == 2025]
y_train  = train_df[TARGET].values
y_test   = test_df[TARGET].values

# ── Define ablation groups ────────────────────────────────────────────────────
ABLATION_GROUPS = {
    "Full Model (baseline)": [],   # no features removed
    "- Event Features\n  (Diwali flags, stubble season)": [
        "days_since_diwali", "is_diwali", "is_stubble_season"
    ],
    "- Wind Decomposition\n  (U/V vectors → raw speed only)": [
        "wind_u", "wind_v"
    ],
    "- Cyclical Encoding\n  (*_sin/*_cos time features)": [
        c for c in all_global if c.endswith("_sin") or c.endswith("_cos")
    ],
    "- Rolling Windows\n  (pm25_roll_mean/std/max)": [
        c for c in all_global if "roll" in c
    ],
    "- Interaction Feature\n  (humidity × temperature)": [
        "humid_temp_interaction"
    ],
    "- AQI Lag Features\n  (aqi_lag1 through aqi_lag30)": [
        c for c in all_global if c.startswith("aqi_lag")
    ],
}

# ── Train function ────────────────────────────────────────────────────────────
def fast_xgb(X_tr, y_tr, X_te):
    m = xgb.XGBRegressor(
        n_estimators=400, learning_rate=0.07, max_depth=6,
        subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0,
        random_state=42, n_jobs=-1
    )
    m.fit(X_tr, y_tr, verbose=False)
    return m.predict(X_te)

# ── Run ablations ─────────────────────────────────────────────────────────────
print("\n[2/3] Running ablations (retraining XGBoost for each condition)...\n")
results = []
baseline_mae = None

for name, drop_cols in ABLATION_GROUPS.items():
    feats = [f for f in all_global if f not in drop_cols]
    n_dropped = len(drop_cols)

    X_tr = train_df[feats].values
    X_te = test_df[feats].values
    preds = fast_xgb(X_tr, y_tr=y_train, X_te=X_te)
    mae  = mean_absolute_error(y_test, preds)
    r2   = r2_score(y_test, preds)

    if baseline_mae is None:
        baseline_mae = mae

    delta = mae - baseline_mae
    short = name.split("\n")[0].strip()
    print(f"  {short:<42}  features={len(feats):2d}  "
          f"MAE={mae:.3f}  R²={r2:.3f}  ΔMAE={delta:+.3f}")

    results.append({
        "Condition": name.split("\n")[0].strip().replace("- ", ""),
        "Features Removed": ", ".join(drop_cols) if drop_cols else "None",
        "N Features": len(feats),
        "MAE": round(mae, 4),
        "R2": round(r2, 4),
        "Delta MAE": round(delta, 4),
    })

results_df = pd.DataFrame(results)
results_df.to_csv(RES_DIR / "08_ablation_results.csv", index=False)
print(f"\n    Saved: results/08_ablation_results.csv")

# ── Plot ──────────────────────────────────────────────────────────────────────
print("\n[3/3] Generating ablation plots...")
fig, axes = plt.subplots(1, 2, figsize=(16, 5))
fig.suptitle("Feature Ablation Study — Global XGBoost Daily (4yr Training)",
             fontsize=13, fontweight="bold")

labels  = results_df["Condition"].tolist()
maes    = results_df["MAE"].tolist()
deltas  = results_df["Delta MAE"].tolist()
colors  = ["#4A9EFF" if i == 0 else "#E53935" if d > 1 else "#FB8C00" if d > 0 else "#43A047"
           for i, d in enumerate(deltas)]

y_pos = np.arange(len(labels))

# Left: absolute MAE
axes[0].barh(y_pos, maes, color=colors, alpha=0.85, edgecolor="white", linewidth=0.3)
for i, (mae, delta) in enumerate(zip(maes, deltas)):
    axes[0].text(mae + 0.2, i, f"{mae:.2f}", va="center", fontsize=9)
axes[0].set_yticks(y_pos)
axes[0].set_yticklabels(labels, fontsize=9)
axes[0].set_xlabel("MAE (µg/m³)")
axes[0].set_title("Absolute MAE per Condition")
axes[0].axvline(maes[0], color="black", lw=0.8, ls="--", alpha=0.5)
axes[0].grid(axis="x", alpha=0.3)
axes[0].invert_yaxis()

# Right: ΔMAE (positive = degradation = feature was important)
delta_colors = ["#4A9EFF" if i == 0 else "#E53935" if d > 0 else "#43A047"
                for i, d in enumerate(deltas)]
axes[1].barh(y_pos[1:], deltas[1:], color=delta_colors[1:], alpha=0.85,
             edgecolor="white", linewidth=0.3)
for i, d in enumerate(deltas[1:]):
    axes[1].text(d + (0.1 if d >= 0 else -0.1), i,
                 f"{d:+.2f}", va="center", fontsize=9,
                 ha="left" if d >= 0 else "right")
axes[1].set_yticks(y_pos[1:])
axes[1].set_yticklabels(labels[1:], fontsize=9)
axes[1].set_xlabel("ΔMAE vs Full Model (µg/m³)  [+ = feature was helpful]")
axes[1].set_title("Marginal Contribution of Each Feature Group")
axes[1].axvline(0, color="black", lw=0.8)
axes[1].grid(axis="x", alpha=0.3)
axes[1].invert_yaxis()

plt.tight_layout()
plt.savefig(PLOT_DIR / "08_ablation_study.png", dpi=150, bbox_inches="tight")
plt.close()
print("    Saved: code/plots/08_ablation_study.png")

print("\n" + "=" * 70)
print("  ABLATION SUMMARY")
print("  " + "-" * 66)
print(f"  Most important group (highest ΔMAE when removed):")
most_imp = results_df.iloc[1:].sort_values("Delta MAE", ascending=False).iloc[0]
print(f"    → {most_imp['Condition']}  (ΔMAE = {most_imp['Delta MAE']:+.3f})")
print("=" * 70)
