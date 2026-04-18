"""
09_walkforward_cv.py
══════════════════════════════════════════════════════════════════════════════
Walk-Forward (Expanding Window) Cross-Validation for Global XGBoost
─────────────────────────────────────────────────────────────────────────────
Replaces single train/test split with 4-fold temporal expanding CV:
  Fold 1: Train 2021       → Test 2022
  Fold 2: Train 2021-2022  → Test 2023
  Fold 3: Train 2021-2023  → Test 2024
  Fold 4: Train 2021-2024  → Test 2025  (main reported result)

Reports mean ± std of MAE/RMSE/R² plus bootstrap 95% CI on test MAE.
Saves: fold performance plot, bootstrap CI chart, summary CSV.

Run from project root:
    python code/09_walkforward_cv.py
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
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

BASE     = Path(__file__).parent.parent
FEAT_DIR = BASE / "dataset" / "features"
PLOT_DIR = BASE / "code" / "plots"
RES_DIR  = BASE / "results"
PLOT_DIR.mkdir(exist_ok=True); RES_DIR.mkdir(exist_ok=True)

print("=" * 70)
print("  09  WALK-FORWARD CROSS-VALIDATION")
print("=" * 70)

# ── Load ──────────────────────────────────────────────────────────────────────
print("\n[1/4] Loading daily features...")
feat_daily = pd.read_parquet(FEAT_DIR / "features_daily.parquet")
with open(FEAT_DIR / "feature_meta_daily.json") as f:
    meta = json.load(f)

feats  = meta["global_features"]
TARGET = "pm25_target"
YEARS  = sorted(feat_daily.index.year.unique().tolist())
print(f"    Years in data: {YEARS}")

# ── CV configuration ──────────────────────────────────────────────────────────
FOLDS = [
    {"train": [2021],               "test": [2022], "label": "Train 2021\n→ Test 2022"},
    {"train": [2021, 2022],         "test": [2023], "label": "Train 2021-22\n→ Test 2023"},
    {"train": [2021, 2022, 2023],   "test": [2024], "label": "Train 2021-23\n→ Test 2024"},
    {"train": [2021, 2022, 2023, 2024], "test": [2025], "label": "Train 2021-24\n→ Test 2025"},
]

# ── Bootstrap CI ──────────────────────────────────────────────────────────────
def bootstrap_mae_ci(y_true, y_pred, n_boot=2000, ci=0.95):
    rng = np.random.default_rng(42)
    n   = len(y_true)
    boot_maes = [
        mean_absolute_error(y_true[idx], y_pred[idx])
        for idx in (rng.integers(0, n, n) for _ in range(n_boot))
    ]
    lo = np.percentile(boot_maes, (1 - ci) / 2 * 100)
    hi = np.percentile(boot_maes, (1 + ci) / 2 * 100)
    return np.mean(boot_maes), lo, hi

# ── Run folds ─────────────────────────────────────────────────────────────────
print("\n[2/4] Running walk-forward CV folds...\n")
fold_results = []

for fold_idx, fold in enumerate(FOLDS, 1):
    tr = feat_daily[feat_daily.index.year.isin(fold["train"])]
    te = feat_daily[feat_daily.index.year.isin(fold["test"])]

    X_tr = tr[feats].values; y_tr = tr[TARGET].values
    X_te = te[feats].values; y_te = te[TARGET].values

    model = xgb.XGBRegressor(
        n_estimators=500, learning_rate=0.05, max_depth=6,
        subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0,
        random_state=42, n_jobs=-1
    )
    model.fit(X_tr, y_tr, verbose=False)
    preds = model.predict(X_te)

    mae  = mean_absolute_error(y_te, preds)
    rmse = np.sqrt(mean_squared_error(y_te, preds))
    r2   = r2_score(y_te, preds)

    boot_mean, ci_lo, ci_hi = bootstrap_mae_ci(y_te, preds)

    fold_results.append({
        "Fold": fold_idx,
        "Label": fold["label"].replace("\n", " "),
        "Train Years": str(fold["train"]),
        "Test Year": str(fold["test"]),
        "MAE": round(mae, 4),
        "RMSE": round(rmse, 4),
        "R2": round(r2, 4),
        "Boot MAE Mean": round(boot_mean, 4),
        "Boot CI Lo": round(ci_lo, 4),
        "Boot CI Hi": round(ci_hi, 4),
    })

    print(f"  Fold {fold_idx}: {fold['label'].replace(chr(10), ' ')}")
    print(f"    MAE={mae:.3f}  RMSE={rmse:.3f}  R²={r2:.3f}")
    print(f"    Bootstrap 95% CI for MAE: [{ci_lo:.3f}, {ci_hi:.3f}]\n")

cv_df = pd.DataFrame(fold_results)
cv_df.to_csv(RES_DIR / "09_walkforward_cv_results.csv", index=False)
print(f"  Saved: results/09_walkforward_cv_results.csv")

mean_mae  = cv_df["MAE"].mean()
std_mae   = cv_df["MAE"].std()
mean_r2   = cv_df["R2"].mean()
std_r2    = cv_df["R2"].std()
print(f"\n  Aggregate across all folds:")
print(f"    MAE  = {mean_mae:.3f} ± {std_mae:.3f}")
print(f"    R²   = {mean_r2:.3f} ± {std_r2:.3f}")

# ── Plots ─────────────────────────────────────────────────────────────────────
print("\n[3/4] Generating CV performance plots...")
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle("Walk-Forward Cross-Validation — Global XGBoost Daily",
             fontsize=13, fontweight="bold")

fold_labels = [f"Fold {r['Fold']}\n{r['Label'].split('Test ')[1]}" for _, r in cv_df.iterrows()]
x_pos = np.arange(len(cv_df))
colors = plt.cm.Blues(np.linspace(0.4, 0.85, len(cv_df)))

# MAE per fold
axes[0].bar(x_pos, cv_df["MAE"], color=colors, edgecolor="none")
axes[0].errorbar(x_pos, cv_df["Boot MAE Mean"],
                 yerr=[cv_df["Boot MAE Mean"] - cv_df["Boot CI Lo"],
                       cv_df["Boot CI Hi"] - cv_df["Boot MAE Mean"]],
                 fmt="none", color="black", capsize=5, linewidth=1.5,
                 label="95% Bootstrap CI")
axes[0].axhline(mean_mae, color="crimson", ls="--", lw=1.2,
                label=f"Mean MAE = {mean_mae:.2f} ± {std_mae:.2f}")
axes[0].set_xticks(x_pos); axes[0].set_xticklabels(fold_labels, fontsize=8)
axes[0].set_ylabel("MAE (µg/m³)"); axes[0].set_title("MAE with Bootstrap 95% CI")
axes[0].legend(fontsize=7); axes[0].grid(axis="y", alpha=0.3)
for i, v in enumerate(cv_df["MAE"]):
    axes[0].text(i, v + 0.3, f"{v:.2f}", ha="center", fontsize=9)

# R² per fold
axes[1].bar(x_pos, cv_df["R2"], color=colors, edgecolor="none")
axes[1].axhline(mean_r2, color="crimson", ls="--", lw=1.2,
                label=f"Mean R² = {mean_r2:.3f} ± {std_r2:.3f}")
axes[1].set_xticks(x_pos); axes[1].set_xticklabels(fold_labels, fontsize=8)
axes[1].set_ylabel("R² Score"); axes[1].set_title("R² per Fold")
axes[1].legend(fontsize=7); axes[1].grid(axis="y", alpha=0.3)
for i, v in enumerate(cv_df["R2"]):
    axes[1].text(i, v + 0.005, f"{v:.3f}", ha="center", fontsize=9)

# Bootstrap CI violin-style for final fold
final_mae  = cv_df.iloc[-1]["MAE"]
final_lo   = cv_df.iloc[-1]["Boot CI Lo"]
final_hi   = cv_df.iloc[-1]["Boot CI Hi"]
ci_x  = [0]; ci_y = [final_mae]
ci_lo = [[final_mae - final_lo]]; ci_hi = [[final_hi - final_mae]]
axes[2].errorbar([0], [final_mae], yerr=[[final_mae - final_lo], [final_hi - final_mae]],
                 fmt="o", ms=10, color="#4A9EFF", capsize=10, linewidth=2,
                 capthick=2, ecolor="#4A9EFF")
axes[2].text(0, final_mae + 0.5, f"MAE = {final_mae:.3f}", ha="center", fontsize=11)
axes[2].text(0, final_hi + 0.5, f"95% CI: [{final_lo:.2f}, {final_hi:.2f}]",
             ha="center", fontsize=9, color="gray")
axes[2].set_xlim(-1, 1); axes[2].set_xticks([]); axes[2].set_ylabel("MAE (µg/m³)")
axes[2].set_title("Final Fold (2025 Test Set)\nWith Bootstrap 95% CI")
axes[2].grid(axis="y", alpha=0.3)

plt.tight_layout()
plt.savefig(PLOT_DIR / "09_walkforward_cv.png", dpi=150, bbox_inches="tight")
plt.close()
print("    Saved: code/plots/09_walkforward_cv.png")

# Learning curve (RMSE vs training years)
print("\n[4/4] Learning curve from CV data...")
fig, ax = plt.subplots(figsize=(7, 4))
train_sizes = [1, 2, 3, 4]
ax.plot(train_sizes, cv_df["MAE"].tolist(), "o-", color="#4A9EFF",
        lw=2, ms=8, label="MAE (lower = better)")
ax.fill_between(train_sizes,
                cv_df["Boot CI Lo"].tolist(),
                cv_df["Boot CI Hi"].tolist(),
                alpha=0.15, color="#4A9EFF", label="95% Bootstrap CI")
ax.set_xlabel("Training Set Size (years)"); ax.set_ylabel("MAE (µg/m³)")
ax.set_title("Learning Curve: XGBoost Daily — MAE vs Training Years", fontweight="bold")
ax.set_xticks(train_sizes)
ax.set_xticklabels(["1yr\n(2021)", "2yr\n(2021-22)", "3yr\n(2021-23)", "4yr\n(2021-24)"])
ax.legend(); ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(PLOT_DIR / "09_learning_curve_xgb.png", dpi=150, bbox_inches="tight")
plt.close()
print("    Saved: code/plots/09_learning_curve_xgb.png")

print("\n" + "=" * 70)
print("  CV SUMMARY")
print(f"    MAE across 4 folds : {mean_mae:.3f} ± {std_mae:.3f} µg/m³")
print(f"    R²  across 4 folds : {mean_r2:.3f} ± {std_r2:.3f}")
print(f"    This quantifies evaluation uncertainty beyond single test split.")
print("=" * 70)
