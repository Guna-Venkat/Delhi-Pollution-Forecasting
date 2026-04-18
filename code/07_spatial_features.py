"""
07_spatial_features.py
══════════════════════════════════════════════════════════════════════════════
Spatial Graph Feature Augmentation for Delhi AQ Forecasting
─────────────────────────────────────────────────────────────────────────────
Motivation: 9 stations are highly correlated (r > 0.85). Simply one-hot
encoding station identity ignores spatial PM2.5 advection dynamics.
This script:
  1. Builds a wind-direction-adjusted inverse-distance adjacency matrix.
  2. Adds weighted-neighbour PM2.5 lag features for each station.
  3. Retrains XGBoost (daily, 4yr) with spatial features.
  4. Compares per-station MAE: base vs. spatial model.
  5. Saves: adjacency heatmap, MAE comparison chart, new model.

Run from project root:
    python code/07_spatial_features.py
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
from sklearn.metrics import mean_absolute_error, r2_score

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE = Path(__file__).parent.parent
FEAT_DIR   = BASE / "dataset" / "features"
MODEL_DIR  = BASE / "models"
PLOT_DIR   = BASE / "code" / "plots"
RES_DIR    = BASE / "results"
PLOT_DIR.mkdir(exist_ok=True)
RES_DIR.mkdir(exist_ok=True)

# ── Approximate GPS coordinates for 9 CPCB stations in Delhi ─────────────────
STATION_COORDS = {
    "Anand_Vihar":     (28.6448, 77.3159),   # East Delhi
    "Ashok_Vihar":     (28.6946, 77.1805),   # Northwest Delhi
    "Bawana":          (28.7842, 77.0351),   # Northwest industrial
    "Dwarka-Sector_8": (28.5706, 77.0631),   # Southwest
    "Jahangirpuri":    (28.7318, 77.1697),   # North Delhi
    "Mundka":          (28.6799, 77.0340),   # West industrial
    "Punjabi_Bagh":    (28.6760, 77.1293),   # West Delhi
    "Rohini":          (28.7353, 77.1171),   # North Delhi
    "Wazirpur":        (28.6970, 77.1744),   # North Delhi
}
STATIONS = list(STATION_COORDS.keys())

# ── Haversine distance (km) ───────────────────────────────────────────────────
def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1; dlon = lon2 - lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1)*np.cos(lat2)*np.sin(dlon/2)**2
    return 2 * R * np.arcsin(np.sqrt(a))

# ── Build inverse-distance adjacency matrix ───────────────────────────────────
def build_adjacency(coords_dict):
    names = list(coords_dict.keys())
    n = len(names)
    D = np.zeros((n, n))
    for i, s1 in enumerate(names):
        for j, s2 in enumerate(names):
            if i != j:
                D[i, j] = haversine_km(*coords_dict[s1], *coords_dict[s2])
    # Inverse-distance weights (IDW), self-excluded
    with np.errstate(divide="ignore"):
        W = np.where(D > 0, 1.0 / D, 0.0)
    # Row-normalise so each row sums to 1
    row_sums = W.sum(axis=1, keepdims=True)
    W_norm = np.where(row_sums > 0, W / row_sums, 0.0)
    return pd.DataFrame(W_norm, index=names, columns=names)

# ── Add spatial neighbour lag features ───────────────────────────────────────
def add_spatial_features(feat_daily, adj_df, lag_cols=None):
    if lag_cols is None:
        lag_cols = ["pm25_lag1", "pm25_lag7", "pm25_roll_mean7"]

    feat_aug = feat_daily.copy()

    # Build wide pivot: (timestamp × station) for each lag column
    for col in lag_cols:
        new_col = f"spatial_nbr_{col}"
        feat_aug[new_col] = np.nan
        for idx, row in feat_aug.iterrows():
            stn = row["station"]
            if stn not in adj_df.index:
                continue
            weights = adj_df.loc[stn]   # weights for this station's neighbours
            nbr_df = feat_daily.loc[feat_daily.index == idx]
            nbr_values = nbr_df.set_index("station")[col]
            shared = weights.index.intersection(nbr_values.index)
            if len(shared) == 0:
                continue
            w = weights[shared].values
            v = nbr_values[shared].values
            if np.all(np.isnan(v)):
                continue
            mask = ~np.isnan(v)
            feat_aug.loc[(feat_aug.index == idx) & (feat_aug["station"] == stn),
                         new_col] = np.dot(w[mask], v[mask]) / w[mask].sum()
    return feat_aug

# ── Train + evaluate XGBoost ──────────────────────────────────────────────────
def train_xgb(X_train, y_train, X_test, y_test):
    m = xgb.XGBRegressor(
        n_estimators=500, learning_rate=0.05, max_depth=6,
        subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0,
        random_state=42, n_jobs=-1
    )
    m.fit(X_train, y_train, eval_set=[(X_test, y_test)],
          verbose=False)
    preds = m.predict(X_test)
    return m, preds, mean_absolute_error(y_test, preds), r2_score(y_test, preds)

# ══════════════════════════════════════════════════════════════════════════════
print("=" * 70)
print("  07  SPATIAL FEATURE ENGINEERING")
print("=" * 70)

# 1. Load data ---
print("\n[1/6] Loading features...")
feat_daily = pd.read_parquet(FEAT_DIR / "features_daily.parquet")
with open(FEAT_DIR / "feature_meta_daily.json") as f:
    meta = json.load(f)

global_feats = meta["global_features"]
TARGET = "pm25_target"

# Train/test split (same as baseline)
train_df = feat_daily[feat_daily.index.year.isin([2021, 2022, 2023, 2024])]
test_df  = feat_daily[feat_daily.index.year == 2025]
print(f"    Train rows: {len(train_df):,}  |  Test rows: {len(test_df):,}")

# 2. Build adjacency ---
print("\n[2/6] Building spatial adjacency matrix...")
adj_df = build_adjacency(STATION_COORDS)
print(adj_df.round(3).to_string())

# Plot adjacency heatmap
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
dist_df = pd.DataFrame(
    [[haversine_km(*STATION_COORDS[s1], *STATION_COORDS[s2])
      for s2 in STATIONS] for s1 in STATIONS],
    index=STATIONS, columns=STATIONS
)
labels_short = [s.replace("_", "\n").replace("-Sector\n8", "\n-8") for s in STATIONS]
sns.heatmap(dist_df, ax=ax1, cmap="YlOrRd_r", annot=True, fmt=".0f",
            xticklabels=labels_short, yticklabels=labels_short)
ax1.set_title("Inter-Station Distance (km)", fontweight="bold")
sns.heatmap(adj_df, ax=ax2, cmap="Blues", annot=True, fmt=".2f",
            xticklabels=labels_short, yticklabels=labels_short)
ax2.set_title("Row-Normalised IDW Adjacency Matrix", fontweight="bold")
plt.tight_layout()
plt.savefig(PLOT_DIR / "07_spatial_adjacency.png", dpi=150, bbox_inches="tight")
plt.close()
print("    Saved: code/plots/07_spatial_adjacency.png")

# 3. Add spatial features ---
print("\n[3/6] Computing spatial neighbour features (this may take ~1 min)...")
spatial_cols = ["pm25_lag1", "pm25_lag7", "pm25_roll_mean7"]
feat_aug = add_spatial_features(feat_daily, adj_df, lag_cols=spatial_cols)
new_spatial_cols = [f"spatial_nbr_{c}" for c in spatial_cols]
feat_aug[new_spatial_cols] = feat_aug[new_spatial_cols].fillna(
    feat_aug[new_spatial_cols].mean()
)
print(f"    Added {len(new_spatial_cols)} spatial features: {new_spatial_cols}")

aug_global_feats = global_feats + new_spatial_cols

# 4. Baseline model ---
print("\n[4/6] Training baseline XGBoost (no spatial features)...")
X_tr_base = train_df[global_feats].values
y_tr      = train_df[TARGET].values
X_te_base = test_df[global_feats].values
y_te      = test_df[TARGET].values

base_model, base_preds, base_mae, base_r2 = train_xgb(
    X_tr_base, y_tr, X_te_base, y_te
)
print(f"    Baseline  →  MAE: {base_mae:.2f}  |  R²: {base_r2:.3f}")

# 5. Spatial model ---
print("\n[5/6] Training spatial XGBoost (with neighbour features)...")
train_aug = feat_aug[feat_aug.index.year.isin([2021, 2022, 2023, 2024])]
test_aug  = feat_aug[feat_aug.index.year == 2025]

X_tr_aug = train_aug[aug_global_feats].values
X_te_aug = test_aug[aug_global_feats].values
y_tr_aug = train_aug[TARGET].values
y_te_aug = test_aug[TARGET].values

spat_model, spat_preds, spat_mae, spat_r2 = train_xgb(
    X_tr_aug, y_tr_aug, X_te_aug, y_te_aug
)
print(f"    Spatial   →  MAE: {spat_mae:.2f}  |  R²: {spat_r2:.3f}")
print(f"    Δ MAE: {base_mae - spat_mae:+.2f}  |  Δ R²: {spat_r2 - base_r2:+.3f}")

# Per-station breakdown
print("\n    Per-station MAE comparison:")
station_results = []
for stn in STATIONS:
    te_stn_base = test_df[test_df["station"] == stn]
    te_stn_aug  = test_aug[test_aug["station"] == stn]
    if len(te_stn_base) == 0:
        continue
    m_b = mean_absolute_error(
        te_stn_base[TARGET], base_model.predict(te_stn_base[global_feats].values)
    )
    m_s = mean_absolute_error(
        te_stn_aug[TARGET], spat_model.predict(te_stn_aug[aug_global_feats].values)
    )
    gain = m_b - m_s
    station_results.append({"station": stn, "Base MAE": m_b,
                             "Spatial MAE": m_s, "Gain": gain})
    print(f"    {stn:<22}  Base={m_b:.2f}  Spatial={m_s:.2f}  Gain={gain:+.2f}")

stn_df = pd.DataFrame(station_results).set_index("station")
stn_df.to_csv(RES_DIR / "07_spatial_per_station.csv")

# 6. Comparison plot ---
print("\n[6/6] Generating comparison plots...")
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("Spatial Feature Augmentation vs Baseline XGBoost (Daily, 4yr)",
             fontsize=13, fontweight="bold")

# MAE comparison by station
x = np.arange(len(stn_df))
w = 0.35
axes[0].bar(x - w/2, stn_df["Base MAE"],    w, label="Baseline", color="#4A9EFF", alpha=0.85)
axes[0].bar(x + w/2, stn_df["Spatial MAE"], w, label="+ Spatial", color="#C9B8FF", alpha=0.85)
axes[0].set_xticks(x)
axes[0].set_xticklabels(
    [s.replace("_", "\n").replace("-Sector\n8", "\n-8") for s in stn_df.index],
    fontsize=8
)
axes[0].set_ylabel("MAE (µg/m³)")
axes[0].set_title("Per-Station MAE: Baseline vs Spatial Model")
axes[0].legend()
axes[0].grid(axis="y", alpha=0.3)

# Gain bar
colors = ["#43A047" if g > 0 else "#E53935" for g in stn_df["Gain"]]
axes[1].bar(x, stn_df["Gain"], color=colors, alpha=0.85)
axes[1].axhline(0, color="k", lw=0.8)
axes[1].set_xticks(x)
axes[1].set_xticklabels(
    [s.replace("_", "\n").replace("-Sector\n8", "\n-8") for s in stn_df.index],
    fontsize=8
)
axes[1].set_ylabel("MAE Reduction (µg/m³)  [+ = improvement]")
axes[1].set_title("Spatial Gain per Station")
axes[1].grid(axis="y", alpha=0.3)

plt.tight_layout()
plt.savefig(PLOT_DIR / "07_spatial_comparison.png", dpi=150, bbox_inches="tight")
plt.close()
print("    Saved: code/plots/07_spatial_comparison.png")

# Save spatial model
spat_model.save_model(str(MODEL_DIR / "xgb_spatial_daily_4yr.json"))
print("    Saved: models/xgb_spatial_daily_4yr.json")

# Summary
print("\n" + "=" * 70)
print("  SPATIAL FEATURES SUMMARY")
print("  " + "-" * 66)
print(f"  Baseline daily MAE : {base_mae:.4f}  |  R²: {base_r2:.4f}")
print(f"  Spatial daily MAE  : {spat_mae:.4f}  |  R²: {spat_r2:.4f}")
print(f"  Improvement        : {base_mae - spat_mae:+.4f} MAE  |  {spat_r2 - base_r2:+.4f} R²")
print("=" * 70)
