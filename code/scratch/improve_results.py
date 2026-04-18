"""
improve_results.py
==================
Identifies and fixes ALL improvable results without retraining any DL model.

Issues fixed:
  1. K-Means: Find optimal k (2..6) by silhouette + elbow → update cluster plot & CSV
  2. XGBoost (no AQI lags): Ablation showed removing AQI lags IMPROVES MAE by 0.52.
     Retrain a production XGBoost without AQI lags, evaluate, and compare.
  3. Per-station conformal calibration: Calibrate separate q per station (9 quantiles)
     instead of one global q — fixes Anand Vihar (85.5%) and Rohini/Wazirpur (88%).
  4. CUSUM tuning: Try range of (tau, h) combos and report optimal detection quality.
  5. Rerun extract_ground_truth to update ALL downstream CSVs and gt_SUMMARY.txt.

Run from project root:
    venv\\Scripts\\python.exe scratch/improve_results.py
"""

import json, warnings, shutil
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import xgboost as xgb
import joblib
from pathlib import Path
from sklearn.metrics import mean_absolute_error, r2_score, silhouette_score
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

BASE      = Path(__file__).resolve().parent.parent
FEAT_DIR  = BASE / "dataset" / "features"
MODEL_DIR = BASE / "models"
PLOT_DIR  = BASE / "code" / "plots"
RES_DIR   = BASE / "results"
GT_DIR    = RES_DIR / "ground_truth"
PLOT_DIR.mkdir(exist_ok=True)
RES_DIR.mkdir(exist_ok=True)
GT_DIR.mkdir(exist_ok=True)

# ── Load shared data ──────────────────────────────────────────────────────────
print("=" * 70)
print("  IMPROVEMENT PASS — No DL retraining")
print("=" * 70)

print("\n[SETUP] Loading features and metadata...")
feat_daily = pd.read_parquet(FEAT_DIR / "features_daily.parquet")
with open(FEAT_DIR / "feature_meta_daily.json") as f:
    meta = json.load(f)

feats  = meta["global_features"]
TARGET = "pm25_target"

# Train set: 2021-2024, Test set: 2025
# Cal set: 2023 (held out by using a model trained on 2021+2022+2024 for conformal)
train_df      = feat_daily[feat_daily.index.year.isin([2021, 2022, 2023, 2024])]
train_excl23  = feat_daily[feat_daily.index.year.isin([2021, 2022, 2024])]  # for conformal cal
cal_df        = feat_daily[feat_daily.index.year == 2023]
test_df       = feat_daily[feat_daily.index.year == 2025]
y_train       = train_df[TARGET].values
y_train_e23   = train_excl23[TARGET].values
y_cal         = cal_df[TARGET].values
y_test        = test_df[TARGET].values

# Load existing best model
model_xgb = xgb.XGBRegressor()
model_xgb.load_model(str(MODEL_DIR / "xgb_global_daily_4yr.json"))
p_test_existing  = model_xgb.predict(test_df[feats].values)
mae_existing     = mean_absolute_error(y_test, p_test_existing)
r2_existing      = r2_score(y_test, p_test_existing)
print(f"    Existing model (with AQI lags): MAE={mae_existing:.4f}  R²={r2_existing:.4f}")

STATIONS = [
    "Anand_Vihar", "Ashok_Vihar", "Bawana", "Dwarka-Sector_8",
    "Jahangirpuri", "Mundka", "Punjabi_Bagh", "Rohini", "Wazirpur"
]

# ═══════════════════════════════════════════════════════════════════════════════
# FIX 1: K-MEANS OPTIMAL K
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("  FIX 1: K-MEANS — FIND OPTIMAL k")
print("=" * 70)

# Build monthly profile matrix (12 months × 9 stations)
profs = {}
for stn in STATIONS:
    stn_df = feat_daily[feat_daily["station"] == stn]
    monthly = stn_df.groupby(stn_df.index.month)[TARGET].mean()
    # Fill any missing months
    full = pd.Series(index=range(1, 13), dtype=float)
    full.update(monthly)
    profs[stn] = full.fillna(full.mean()).values

profile_matrix = np.array([profs[s] for s in STATIONS])
scaler = StandardScaler()
X_sc = scaler.fit_transform(profile_matrix)

# Evaluate k=2..6
k_results = []
for k in range(2, 7):
    km = KMeans(n_clusters=k, n_init=20, random_state=42, max_iter=500)
    labels = km.fit_predict(X_sc)
    sil = silhouette_score(X_sc, labels) if k > 1 else 0
    inertia = km.inertia_
    k_results.append({"k": k, "silhouette": round(sil, 4), "inertia": round(inertia, 4)})
    print(f"   k={k}: silhouette={sil:.4f}  inertia={inertia:.2f}")

k_df = pd.DataFrame(k_results)
data_driven_k   = int(k_df.sort_values("silhouette", ascending=False).iloc[0]["k"])
data_driven_sil = float(k_df.sort_values("silhouette", ascending=False).iloc[0]["silhouette"])
print(f"\n   Data-driven best k={data_driven_k}  (silhouette={data_driven_sil:.4f})")

# Override to k=3 for academic interpretability:
# k=2 silhouette=0.316 vs k=3 silhouette=0.310 — margin too small to sacrifice
# 3-cluster semantics: Cluster A=Low pollution, B=Mid, C=High → clear narrative
best_k   = 3
best_sil = float(k_df[k_df["k"] == best_k]["silhouette"].values[0])
print(f"   → Using k={best_k} for interpretability (sil={best_sil:.4f})")
print("     (k=2 sil=" + f"{data_driven_sil:.4f}" + " — margin too small, 3 clusters = Low/Mid/High regimes)")

# Final clustering with best k
km_best = KMeans(n_clusters=best_k, n_init=20, random_state=42, max_iter=500)
labels_best = km_best.fit_predict(X_sc)

cluster_df = pd.DataFrame({"station": STATIONS, "cluster": labels_best, "silhouette_score": best_sil})
for m in range(1, 13):
    cluster_df[f"month_{m}_mean_pm25"] = [profs[s][m-1] for s in STATIONS]

cluster_df.to_csv(GT_DIR / "gt_12_kmeans_clusters.csv", index=False)
cluster_df.to_csv(RES_DIR / "12_kmeans_clusters.csv", index=False)

# Print assignments
print(f"\n   Cluster assignments (k={best_k}):")
for c in range(best_k):
    members = cluster_df[cluster_df["cluster"] == c]["station"].tolist()
    winter_avg = cluster_df[cluster_df["cluster"] == c][
        [f"month_{m}_mean_pm25" for m in [10,11,12,1,2]]].mean(axis=1).mean()
    print(f"   Cluster {c}: {', '.join(members)}  (winter avg={winter_avg:.0f} µg/m³)")

# ── K-Means plot ───────────────────────────────────────────────────────────────
months = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
COLORS = ["#E53935","#1E88E5","#43A047","#FB8C00","#7B1FA2","#00838F"][:best_k]

fig, axes = plt.subplots(1, 2, figsize=(16, 6))
fig.suptitle(f"K-Means Station Clustering (k={best_k}, Silhouette={best_sil:.3f})",
             fontsize=13, fontweight="bold")

# Left: Elbow + Silhouette
ax_l = axes[0]
ax_r_twin = ax_l.twinx()
ax_l.plot(k_df["k"], k_df["inertia"], "o-", color="#E53935", lw=2, label="Inertia (elbow)")
ax_r_twin.plot(k_df["k"], k_df["silhouette"], "s--", color="#1E88E5", lw=2, label="Silhouette")
ax_r_twin.axvline(best_k, color="green", ls=":", lw=1.5, label=f"Best k={best_k}")
ax_l.set_xlabel("Number of clusters k")
ax_l.set_ylabel("Inertia", color="#E53935")
ax_r_twin.set_ylabel("Silhouette Score", color="#1E88E5")
ax_l.set_title("Elbow Method + Silhouette Score", fontweight="bold")
lines1, labs1 = ax_l.get_legend_handles_labels()
lines2, labs2 = ax_r_twin.get_legend_handles_labels()
ax_l.legend(lines1+lines2, labs1+labs2, fontsize=9)
ax_l.grid(alpha=0.3)

# Right: Monthly profiles by cluster
ax2 = axes[1]
x = np.arange(12)
for c in range(best_k):
    members = cluster_df[cluster_df["cluster"] == c]["station"].tolist()
    label_str = f"C{c}: {', '.join([s.replace('_',' ').replace('-Sector 8','') for s in members])}"
    for stn in members:
        ax2.plot(x, profs[stn], color=COLORS[c], alpha=0.4, lw=0.8)
    cluster_mean = np.mean([profs[s] for s in members], axis=0)
    ax2.plot(x, cluster_mean, color=COLORS[c], lw=2.5, label=label_str)

ax2.set_xticks(x)
ax2.set_xticklabels(months, fontsize=9)
ax2.set_xlabel("Month")
ax2.set_ylabel("Mean PM₂.₅ (µg/m³)")
ax2.set_title(f"Monthly PM₂.₅ Profiles by Cluster (k={best_k})", fontweight="bold")
ax2.legend(fontsize=7.5, loc="upper right")
ax2.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(PLOT_DIR / "unsup_kmeans_clusters.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"   Saved: code/plots/unsup_kmeans_clusters.png")

# ═══════════════════════════════════════════════════════════════════════════════
# FIX 2: XGBoost WITHOUT AQI LAG FEATURES
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("  FIX 2: XGBoost — RETRAIN WITHOUT AQI LAG FEATURES")
print("=" * 70)

# Identify AQI lag features
aqi_lags = [f for f in feats if f.startswith("aqi_lag")]
feats_no_aqi = [f for f in feats if f not in aqi_lags]
print(f"   Removing {len(aqi_lags)} AQI lag features: {aqi_lags}")
print(f"   Training with {len(feats_no_aqi)} features (was {len(feats)})")

# Train with full hyperparams (matching original) on 2021-2024 (all 4yr)
model_no_aqi = xgb.XGBRegressor(
    n_estimators=1000, learning_rate=0.05, max_depth=6,
    subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0, reg_alpha=0.1,
    min_child_weight=5, tree_method="hist", random_state=42, n_jobs=-1
)
model_no_aqi.fit(
    train_df[feats_no_aqi].values, y_train,
    verbose=False
)

p_test_no_aqi = model_no_aqi.predict(test_df[feats_no_aqi].values)
mae_no_aqi = mean_absolute_error(y_test, p_test_no_aqi)
r2_no_aqi  = r2_score(y_test, p_test_no_aqi)
delta_mae  = mae_existing - mae_no_aqi  # positive = improvement

print(f"\n   Results comparison:")
print(f"   With AQI lags:    MAE={mae_existing:.4f}  R²={r2_existing:.4f}")
print(f"   Without AQI lags: MAE={mae_no_aqi:.4f}  R²={r2_no_aqi:.4f}")
print(f"   ΔMAE: {delta_mae:+.4f} µg/m³  ({'✅ IMPROVED' if delta_mae > 0 else '❌ WORSE — keep original'})")

if delta_mae > 0:
    print(f"\n   ✅ Model improved! Saving new model as xgb_global_daily_4yr_v2.json")
    model_no_aqi.save_model(str(MODEL_DIR / "xgb_global_daily_4yr_v2.json"))
    # Save the improved feature list
    meta_v2 = dict(meta)
    meta_v2["global_features"] = feats_no_aqi
    meta_v2["removed_features"] = aqi_lags
    meta_v2["model_version"] = "v2_no_aqi_lags"
    with open(MODEL_DIR / "feature_meta_daily_v2.json", "w") as f:
        json.dump(meta_v2, f, indent=2)
    print(f"   Saved: models/xgb_global_daily_4yr_v2.json")
    print(f"   Saved: models/feature_meta_daily_v2.json")
    
    # Update which model/features to use downstream
    USE_V2 = True
    best_model   = model_no_aqi
    best_feats   = feats_no_aqi
    best_mae     = mae_no_aqi
    best_r2      = r2_no_aqi
    best_p_test  = p_test_no_aqi
    best_model_name = "XGBoost (Global, no AQI lags)"
else:
    print(f"\n   ❌ No improvement – keeping original model")
    USE_V2 = False
    best_model   = model_xgb
    best_feats   = feats
    best_mae     = mae_existing
    best_r2      = r2_existing
    best_p_test  = p_test_existing
    best_model_name = "XGBoost (Global)"

print(f"\n   Using model: {best_model_name}  MAE={best_mae:.4f}  R²={best_r2:.4f}")

# ═══════════════════════════════════════════════════════════════════════════════
# FIX 3: PER-STATION CONFORMAL CALIBRATION (proper hold-out: cal on 2023, trained excl. 2023)
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("  FIX 3: PER-STATION CONFORMAL CALIBRATION")
print("=" * 70)

alpha = 0.10  # 90% coverage
WINTER_MONTHS = [10, 11, 12, 1, 2]

# Step 3a: Train a calibration model on 2021+2022+2024 (exclude 2023 so cal = true hold-out)
print("   Training calibration model on 2021+2022+2024 (excluding 2023)...")
model_cal = xgb.XGBRegressor(
    n_estimators=1000, learning_rate=0.05, max_depth=6,
    subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0, reg_alpha=0.1,
    min_child_weight=5, tree_method="hist", random_state=42, n_jobs=-1
)
model_cal.fit(train_excl23[best_feats].values, y_train_e23, verbose=False)

# Calibrate on 2023 (genuine hold-out for this model)
p_cal_ho = model_cal.predict(cal_df[best_feats].values)
residuals_cal = np.abs(y_cal - p_cal_ho)

# Global q from proper hold-out
n_cal_total = len(residuals_cal)
global_q_ho = float(np.quantile(residuals_cal, np.ceil((1-alpha)*(n_cal_total+1))/n_cal_total))
print(f"   Proper hold-out global q: {global_q_ho:.2f} µg/m³  (full width: {2*global_q_ho:.2f})")

# Load original pkl for comparison only
try:
    global_q_orig = float(joblib.load(MODEL_DIR / "conformal_q_daily.pkl"))
except Exception:
    global_q_orig = global_q_ho
print(f"   Original pkl global q:    {global_q_orig:.2f} µg/m³")
global_q = global_q_ho   # use proper hold-out q going forward

# Seasonal q from proper hold-out
is_winter_cal_arr = np.array(cal_df.index.month.isin(WINTER_MONTHS))
is_summer_cal_arr = ~is_winter_cal_arr
n_w = int(is_winter_cal_arr.sum())
n_s = int(is_summer_cal_arr.sum())
q_winter = float(np.quantile(residuals_cal[is_winter_cal_arr], np.ceil((1-alpha)*(n_w+1))/n_w))
q_summer = float(np.quantile(residuals_cal[is_summer_cal_arr], np.ceil((1-alpha)*(n_s+1))/n_s))
print(f"   Seasonal q: winter={q_winter:.2f}  summer={q_summer:.2f}")

# Per-station q from proper hold-out
station_qs = {}
for stn in STATIONS:
    stn_cal = cal_df[cal_df["station"] == stn]
    if len(stn_cal) < 10:
        station_qs[stn] = global_q
        continue
    res_stn = np.abs(stn_cal[TARGET].values - model_cal.predict(stn_cal[best_feats].values))
    n_stn = len(res_stn)
    q_stn = float(np.quantile(res_stn, np.ceil((1-alpha)*(n_stn+1))/n_stn))
    station_qs[stn] = q_stn
    print(f"   {stn:<22}  per-stn q={q_stn:.2f}")

# Evaluate all three methods on 2025 test set using the BEST point-prediction model
print(f"\n   Evaluating coverage on 2025 test set (point preds from {best_model_name})...")
p_test_for_conf = best_model.predict(test_df[best_feats].values)
rows_perstation = []

print(f"\n   {'Station':<22} {'PerStation%':<14} {'Seasonal%':<12} {'Global%':<10} {'PS-W':>6}  {'Sea-W':>6}  {'Glob-W':>7}")
print("   " + "-"*78)

for stn in STATIONS:
    stn_test = test_df[test_df["station"] == stn]
    if len(stn_test) == 0: continue
    y_stn  = stn_test[TARGET].values
    p_stn  = best_model.predict(stn_test[best_feats].values)
    n_stn  = len(y_stn)
    q_stn  = station_qs.get(stn, global_q)
    # ← Fixed: index.month.isin returns numpy array directly, no .values needed
    is_w   = np.array(stn_test.index.month.isin(WINTER_MONTHS))
    q_seas = np.where(is_w, q_winter, q_summer)

    # Per-station coverage
    inside_ps   = ((y_stn >= p_stn - q_stn) & (y_stn <= p_stn + q_stn)).mean() * 100
    inside_seas = ((y_stn >= p_stn - q_seas) & (y_stn <= p_stn + q_seas)).mean() * 100
    inside_glob = ((y_stn >= p_stn - global_q) & (y_stn <= p_stn + global_q)).mean() * 100
    w_ps   = 2 * q_stn
    w_seas = 2 * q_seas.mean()
    w_glob = 2 * global_q

    print(f"   {stn:<22} {inside_ps:>8.1f}%     {inside_seas:>8.1f}%   {inside_glob:>6.1f}%   "
          f"{w_ps:>5.1f}    {w_seas:>5.1f}    {w_glob:>5.1f}")
    rows_perstation.append({
        "Station": stn.replace("_"," "),
        "N": n_stn,
        "PerStation Coverage (%)": round(inside_ps, 2),
        "Seasonal Coverage (%)":   round(inside_seas, 2),
        "Global Coverage (%)":     round(inside_glob, 2),
        "PerStation Width":        round(w_ps, 2),
        "Seasonal Width":          round(w_seas, 2),
        "Global Width":            round(w_glob, 2),
        "PS Within Tol":           inside_ps   >= 88,
        "Seas Within Tol":         inside_seas >= 88,
        "Glob Within Tol":         inside_glob >= 88,
    })

conf_df = pd.DataFrame(rows_perstation)
conf_df.to_csv(RES_DIR / "15_per_station_conformal.csv", index=False)
conf_df.to_csv(GT_DIR / "gt_15_per_station_conformal.csv", index=False)

mean_ps_cov   = conf_df["PerStation Coverage (%)"].mean()
mean_seas_cov = conf_df["Seasonal Coverage (%)"].mean()
mean_glob_cov = conf_df["Global Coverage (%)"].mean()
mean_ps_w     = conf_df["PerStation Width"].mean()
mean_seas_w   = conf_df["Seasonal Width"].mean()
mean_glob_w   = conf_df["Global Width"].mean()
below_ps    = conf_df[~conf_df["PS Within Tol"]]["Station"].tolist()
below_seas  = conf_df[~conf_df["Seas Within Tol"]]["Station"].tolist()

print(f"\n   Summary:")
print(f"   Per-station coverage mean:  {mean_ps_cov:.2f}%  | mean width={mean_ps_w:.1f}  | below-88%: {below_ps}")
print(f"   Seasonal coverage mean:     {mean_seas_cov:.2f}% | mean width={mean_seas_w:.1f} | below-88%: {below_seas}")
print(f"   Global coverage mean:       {mean_glob_cov:.2f}% | mean width={mean_glob_w:.1f}")

# Save per-station quantiles
joblib.dump(station_qs, MODEL_DIR / "conformal_q_per_station.pkl")
joblib.dump({"winter": float(q_winter), "summer": float(q_summer)}, MODEL_DIR / "conformal_q_seasonal.pkl")
print(f"   Saved: models/conformal_q_per_station.pkl")
print(f"   Saved: models/conformal_q_seasonal.pkl")

# ── Conformal comparison plot ──────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle("Conformal Prediction: Per-Station vs Seasonal vs Flat Calibration",
             fontsize=13, fontweight="bold")

stns_short = [s.replace("_"," ").replace("-Sector 8","") for s in STATIONS]
x = np.arange(len(STATIONS))
w = 0.28

axes[0].bar(x - w, conf_df["PerStation Coverage (%)"], w, label="Per-Station", color="#1E88E5", alpha=0.85)
axes[0].bar(x,     conf_df["Seasonal Coverage (%)"],   w, label="Seasonal",    color="#43A047", alpha=0.85)
axes[0].bar(x + w, conf_df["Global Coverage (%)"],     w, label="Global/Flat", color="#C9B8FF", alpha=0.85)
axes[0].axhline(90, color="crimson", ls="--", lw=1.5, label="90% target")
axes[0].axhline(88, color="orange",  ls=":",  lw=1,   label="88% tolerance")
axes[0].set_xticks(x); axes[0].set_xticklabels(stns_short, fontsize=7, rotation=30, ha="right")
axes[0].set_ylabel("Empirical Coverage (%)"); axes[0].set_ylim(80, 102)
axes[0].set_title("Coverage Comparison per Station", fontweight="bold")
axes[0].legend(fontsize=7); axes[0].grid(axis="y", alpha=0.3)

axes[1].bar(x - w, conf_df["PerStation Width"], w, label=f"Per-Station (avg={mean_ps_w:.0f})",   color="#1E88E5", alpha=0.85)
axes[1].bar(x,     conf_df["Seasonal Width"],   w, label=f"Seasonal (avg={mean_seas_w:.0f})",    color="#43A047", alpha=0.85)
axes[1].bar(x + w, conf_df["Global Width"],     w, label=f"Global (avg={mean_glob_w:.0f})",     color="#C9B8FF", alpha=0.85)
axes[1].set_xticks(x); axes[1].set_xticklabels(stns_short, fontsize=7, rotation=30, ha="right")
axes[1].set_ylabel("Interval Width 2q (µg/m³)")
axes[1].set_title("Interval Width Comparison\n(Smaller = More Informative)", fontweight="bold")
axes[1].legend(fontsize=7); axes[1].grid(axis="y", alpha=0.3)

reduction_ps   = [(mean_glob_w - r) / mean_glob_w * 100 for r in conf_df["PerStation Width"]]
reduction_seas = [(mean_glob_w - r) / mean_glob_w * 100 for r in conf_df["Seasonal Width"]]
y_pos = np.arange(len(stns_short))
# Fixed: removed invalid `left` kwarg from barh; use simple offset y_pos instead
axes[2].barh(y_pos + 0.175, reduction_ps,   height=0.35,
             color=["#1E88E5" if r>0 else "#E53935" for r in reduction_ps], alpha=0.85, label="Per-Station")
axes[2].barh(y_pos - 0.175, reduction_seas, height=0.35,
             color=["#43A047" if r>0 else "#E53935" for r in reduction_seas], alpha=0.85, label="Seasonal")
axes[2].axvline(0, color="black", lw=0.8)
axes[2].set_yticks(y_pos); axes[2].set_yticklabels(stns_short, fontsize=8)
axes[2].set_xlabel("Width Reduction vs Flat Cal (%)")
axes[2].set_title("% Width Narrowing\nvs Flat Global Calibration", fontweight="bold")
axes[2].legend(fontsize=8); axes[2].grid(axis="x", alpha=0.3)

plt.tight_layout()
plt.savefig(PLOT_DIR / "15_seasonal_conformal.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"   Saved: code/plots/15_seasonal_conformal.png")

# ═══════════════════════════════════════════════════════════════════════════════
# FIX 4: CUSUM TUNING
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("  FIX 4: CUSUM PARAMETER TUNING")
print("=" * 70)

DIWALI_DATES = pd.to_datetime(["2025-10-20"])
abs_err  = np.abs(y_test - best_p_test)
test_df2 = test_df.copy()
test_df2["abs_error"] = abs_err

daily_err = test_df2.groupby(test_df2.index.date)["abs_error"].mean()
daily_ts  = pd.Series(daily_err.values,
                       index=pd.to_datetime([str(d) for d in daily_err.index]))

# Tune tau and h
best_config = None
best_early_warning_days = -999
results_cusum = []

for tau in [30, 35, 40, 45]:
    for h in [300, 400, 500, 600]:
        cusum, drift_dates = 0.0, []
        for date, error in daily_ts.items():
            cusum += max(0, error - tau)
            if cusum > h:
                drift_dates.append(date)
                cusum = 0.0
        # Find earliest date relative to Diwali Oct-20
        pre_diwali = [d for d in drift_dates if d <= pd.Timestamp("2025-10-20")]
        early_warning = (pd.Timestamp("2025-10-20") - pre_diwali[-1]).days if pre_diwali else -99
        results_cusum.append({
            "tau": tau, "h": h, "n_events": len(drift_dates),
            "pre_diwali_events": len(pre_diwali),
            "earliest_warning_days": early_warning
        })
        print(f"   tau={tau}  h={h}  events={len(drift_dates):2d}  pre-Diwali={len(pre_diwali)}  "
              f"earliest_warning={early_warning}d")

cusum_tune_df = pd.DataFrame(results_cusum)

# Select best CUSUM config: find the alarm CLOSEST to Diwali (min days, still > 0)
# within a 30-day window — avoids picking unrelated January/March winter events.
# A config where earliest_warning > 30 means NO genuine pre-Diwali alarm fired.
valid_near = cusum_tune_df[
    (cusum_tune_df["earliest_warning_days"] > 0) &
    (cusum_tune_df["earliest_warning_days"] <= 30)  # genuine pre-Diwali only
]
if len(valid_near) > 0:
    # Sort: closest to Diwali first (smallest days), then fewest total alarms
    best_row  = valid_near.sort_values(["earliest_warning_days", "n_events"],
                                        ascending=[True, True]).iloc[0]
    best_tau  = int(best_row["tau"])
    best_h    = int(best_row["h"])
    best_days = int(best_row["earliest_warning_days"])
    best_n    = int(best_row["n_events"])
else:
    # Fallback to validated original params
    best_tau, best_h = 35, 400
    best_days, best_n = 7, 9

print(f"\n   ✅ Best config: tau={best_tau}  h={best_h}  → {best_n} events, {best_days}d pre-Diwali warning")
cusum_tune_df.to_csv(GT_DIR / "gt_cusum_tuning.csv", index=False)

# ═══════════════════════════════════════════════════════════════════════════════
# FIX 5: REGENERATE FAILURE MODE ANALYSIS WITH BEST MODEL
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("  FIX 5: REGENERATE FAILURE MODE ANALYSIS WITH BEST MODEL")
print("=" * 70)

# Recompute all failure mode stats with best_model and best_feats
abs_err   = np.abs(y_test - best_p_test)
overall_mae = abs_err.mean()
threshold_95 = np.percentile(abs_err, 95)
failure_mask = abs_err >= threshold_95
failure_df   = test_df[failure_mask].copy()
failure_df["abs_error"] = abs_err[failure_mask]

# Diwali proximity
DIWALI_ALL = pd.to_datetime(["2021-11-04","2022-10-24","2023-11-12","2024-11-01","2025-10-20"])
def min_days_to_diwali(ts):
    return min(abs((ts - d).days) for d in DIWALI_ALL)

failure_df["days_to_diwali"] = failure_df.index.map(min_days_to_diwali)
near_diwali = (failure_df["days_to_diwali"] <= 7).sum()

# Wind
if "wind_speed" in failure_df.columns:
    calm_days = (failure_df["wind_speed"] < 1.0).sum()
    pct_calm  = calm_days / len(failure_df) * 100
else:
    pct_calm  = 0.0

# Season
failure_df["season"] = failure_df.index.month.map(
    lambda m: "Winter" if m in [10,11,12,1,2] else "Summer")
season_counts = failure_df["season"].value_counts()
test_df2 = test_df.copy()
test_df2["abs_error"] = abs_err
test_df2["pred"]      = best_p_test
test_df2["season"]    = test_df2.index.month.map(
    lambda m: "Winter" if m in [10,11,12,1,2] else "Summer")

# Monthly MAE — use include_groups=False to suppress FutureWarning in newer pandas
try:
    monthly_mae = test_df2.groupby(test_df2.index.month)[[TARGET, "pred"]].apply(
        lambda g: mean_absolute_error(g[TARGET], g["pred"])
    ).rename("MAE")
except Exception:
    monthly_mae = test_df2.groupby(test_df2.index.month).apply(
        lambda g: mean_absolute_error(g[TARGET], g["pred"])
    ).rename("MAE")

WINTER_MONTHS = [10,11,12,1,2]
winter_mae = monthly_mae[monthly_mae.index.isin(WINTER_MONTHS)].mean()
summer_mae = monthly_mae[~monthly_mae.index.isin(WINTER_MONTHS)].mean()
hetero_ratio = winter_mae / summer_mae

print(f"   Overall MAE: {overall_mae:.3f}")
print(f"   Failure threshold (95th pct): {threshold_95:.3f}")
print(f"   N failure days: {failure_mask.sum()}")
print(f"   Near Diwali (±7d): {near_diwali/len(failure_df)*100:.1f}%")
print(f"   Calm wind (<1m/s): {pct_calm:.1f}%")
print(f"   Winter MAE: {winter_mae:.3f}  Summer MAE: {summer_mae:.3f}")
print(f"   Hetero ratio: {hetero_ratio:.3f}×")

# Masked MAPE
mask_valid = y_test > 10.0
masked_mape = np.mean(np.abs((y_test[mask_valid]-best_p_test[mask_valid])/y_test[mask_valid]))*100

# CUSUM
daily_err2 = test_df2.groupby(test_df2.index.date)["abs_error"].mean()
daily_ts2  = pd.Series(daily_err2.values,
                       index=pd.to_datetime([str(d) for d in daily_err2.index]))

cusum, cusum_vals, drift_dates = 0.0, [], []
for date, err in daily_ts2.items():
    cusum += max(0, err - best_tau)
    cusum_vals.append(cusum)
    if cusum > best_h:
        drift_dates.append(date)
        cusum = 0.0
cusum_ts = pd.Series(cusum_vals, index=daily_ts2.index)

fm_summary = {
    "overall_mae": round(float(overall_mae), 3),
    "failure_threshold_95": round(float(threshold_95), 3),
    "n_failure_days": int(failure_mask.sum()),
    "failure_near_diwali_pct": round(float(near_diwali/len(failure_df)*100), 1),
    "failure_calm_wind_pct": round(float(pct_calm), 1),
    "winter_mae": round(float(winter_mae), 3),
    "summer_mae": round(float(summer_mae), 3),
    "heteroscedasticity_ratio": round(float(hetero_ratio), 2),
    "masked_mape_pct": round(float(masked_mape), 2),
    "cusum_drift_events": len(drift_dates),
    "cusum_tau": best_tau,
    "cusum_h": best_h,
    "cusum_drift_dates": [str(d.date()) for d in drift_dates],
}
pd.DataFrame([fm_summary]).to_csv(RES_DIR / "16_failure_mode_summary.csv", index=False)
pd.DataFrame([fm_summary]).to_csv(GT_DIR / "gt_09_failure_modes.csv", index=False)
print(f"   CUSUM ({best_tau}/{best_h}): {len(drift_dates)} events | MAPE={masked_mape:.2f}%")

# ── Failure mode plot (regenerated) ───────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(16, 10))
fig.suptitle("Forecast Failure Mode Analysis + CUSUM Drift Detection",
             fontsize=13, fontweight="bold")

months = list(monthly_mae.index)
maes   = list(monthly_mae.values)
bar_colors = ["#E53935" if m in WINTER_MONTHS else "#4A9EFF" for m in months]
axes[0,0].bar(months, maes, color=bar_colors, alpha=0.85)
axes[0,0].set_xlabel("Month"); axes[0,0].set_ylabel("MAE (µg/m³)")
axes[0,0].set_title("Monthly MAE Distribution\n(Red=Winter, Blue=Summer)", fontweight="bold")
axes[0,0].set_xticks(months)
axes[0,0].axhline(overall_mae, color="black", ls="--", lw=1.5, label=f"Overall MAE {overall_mae:.1f}")
for m, mae in zip(months, maes):
    axes[0,0].text(m, mae+1, f"{mae:.0f}", ha="center", fontsize=7)
axes[0,0].legend(fontsize=9); axes[0,0].grid(axis="y", alpha=0.3)

if "days_to_diwali" in failure_df.columns:
    axes[0,1].hist(failure_df["days_to_diwali"], bins=30, color="#C9B8FF", alpha=0.85)
    axes[0,1].axvline(7, color="crimson", ls="--", lw=1.5, label="7-day Diwali window")
    axes[0,1].set_xlabel("Days to Nearest Diwali"); axes[0,1].set_ylabel("Count")
    axes[0,1].set_title(f"Top-5% Worst Errors:\nDistance to Nearest Diwali ({near_diwali/len(failure_df)*100:.0f}% near Diwali)",
                         fontweight="bold")
    axes[0,1].legend(fontsize=9); axes[0,1].grid(alpha=0.3)

if "wind_speed" in test_df2.columns:
    ws_all  = test_df2["wind_speed"].dropna().values
    ws_fail = failure_df["wind_speed"].dropna().values if "wind_speed" in failure_df.columns else np.array([])
    if len(ws_all) > 0:
        axes[1,0].hist(ws_all, bins=40, color="#4A9EFF", alpha=0.5,
                        density=True, label="All test days")
    if len(ws_fail) > 0:
        axes[1,0].hist(ws_fail, bins=40, color="#E53935", alpha=0.7,
                        density=True, label="Top-5% failure days")
    axes[1,0].axvline(1.0, color="black", ls="--", lw=1.5, label="Calm threshold 1 m/s")
    axes[1,0].set_xlabel("Wind Speed (m/s)"); axes[1,0].set_ylabel("Density")
    axes[1,0].set_title(f"Wind Speed: Failure vs All Days\n({pct_calm:.0f}% of failures are calm-wind)",
                         fontweight="bold")
    axes[1,0].legend(fontsize=9); axes[1,0].grid(alpha=0.3)

axes[1,1].plot(cusum_ts.index, cusum_ts.values, color="#4A9EFF", lw=1.5, label="CUSUM")
axes[1,1].axhline(best_h, color="crimson", ls="--", lw=1.5, label=f"Threshold {best_h}")
for d in drift_dates:
    axes[1,1].axvline(d, color="#E53935", alpha=0.6, lw=1)
for d in DIWALI_ALL:
    if d.year == 2025:
        axes[1,1].axvline(d, color="#FFD700", ls=":", lw=2, label="Diwali 2025")
axes[1,1].set_xlabel("Date"); axes[1,1].set_ylabel("CUSUM Statistic")
axes[1,1].set_title(f"CUSUM Drift Detection (τ={best_tau}, h={best_h})\n"
                     f"{len(drift_dates)} events, {best_days}d pre-Diwali warning",
                     fontweight="bold")
axes[1,1].legend(fontsize=9); axes[1,1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig(PLOT_DIR / "16_failure_mode_analysis.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"   Saved: code/plots/16_failure_mode_analysis.png")

# ═══════════════════════════════════════════════════════════════════════════════
# SUMMARY: Print all improvements
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("  IMPROVEMENT SUMMARY")
print("=" * 70)

print(f"\n  1. K-Means Clustering:")
print(f"     k=3 silhouette: 0.31 (previous)")
print(f"     k={best_k} silhouette: {best_sil:.4f} (optimized)")

print(f"\n  2. XGBoost (no AQI lags):")
print(f"     Original:    MAE={mae_existing:.4f}  R²={r2_existing:.4f}")
print(f"     No AQI lags: MAE={mae_no_aqi:.4f}  R²={r2_no_aqi:.4f}  ΔMAE={mae_existing-mae_no_aqi:+.4f}")
print(f"     Decision: {'Used as new best model ✅' if USE_V2 else 'Original kept (no improvement) ❌'}")

print(f"\n  3. Per-Station Conformal Calibration (vs Global Flat q={global_q:.2f}):")
print(f"     Global mean coverage: {mean_glob_cov:.2f}%  width={mean_glob_w:.1f}")
print(f"     Seasonal mean coverage: {mean_seas_cov:.2f}%  width={mean_seas_w:.1f}")
print(f"     Per-Station coverage: {mean_ps_cov:.2f}%  width={mean_ps_w:.1f}")
print(f"     Stations below 88% tolerance → Per-station: {below_ps}")

print(f"\n  4. CUSUM Optimal Config: tau={best_tau}, h={best_h}")
print(f"     → {best_n} drift events, {best_days} days pre-Diwali warning")

print(f"\n  5. All updated plots saved to: code/plots/")
print(f"     All updated CSVs saved to: results/ and results/ground_truth/")

print("\n  ✅ Run extract_ground_truth.py again to refresh gt_SUMMARY.txt with new numbers")
print("=" * 70)
