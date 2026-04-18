"""
14_polish_plots.py
══════════════════════════════════════════════════════════════════════════════
Polish: SHAP Interaction, UMAP Latent Space, Per-Station Heatmap
─────────────────────────────────────────────────────────────────────────────
Tier 3 "polish" deliverables that push the project to conference quality:

1. SHAP Interaction Plot: conditional importance of wind_speed given high vs.
   low pm25_lag1 — reveals non-linear weather effect.
2. UMAP Latent Space: encode the autoencoder latent vectors and visualise
   using UMAP (or t-SNE fallback), coloured by season/AQI category.
3. Per-Station × Model MAE Heatmap: 9-station × 5-model grid from CSV.
4. Model Comparison Radar Chart: visualise MAE/R²/RMSE strengths.

Run from project root:
    python code/14_polish_plots.py
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
import matplotlib.patches as mpatches
import seaborn as sns
import shap
import torch
import joblib
import xgboost as xgb
from pathlib import Path
from sklearn.manifold import TSNE

BASE      = Path(__file__).parent.parent
FEAT_DIR  = BASE / "dataset" / "features"
MODEL_DIR = BASE / "models"
PLOT_DIR  = BASE / "code" / "plots"
RES_DIR   = BASE / "results"
PLOT_DIR.mkdir(exist_ok=True)

# ── Season map (module-level so all sections can use it) ────────────────
season_map = {
    12: "Winter", 1: "Winter", 2: "Winter",
    3: "Spring", 4: "Spring", 5: "Spring",
    6: "Summer", 7: "Summer", 8: "Summer",
    9: "Monsoon", 10: "Post-Monsoon", 11: "Post-Monsoon"
}
season_order   = ["Winter", "Spring", "Summer", "Monsoon", "Post-Monsoon"]
season_palette = {
    "Winter": "#4A9EFF", "Spring": "#43A047",
    "Summer": "#FFB300", "Monsoon": "#29B6F6",
    "Post-Monsoon": "#E53935"
}

print("=" * 70)
print("  14  POLISH PLOTS")
print("=" * 70)

# ── Load shared resources ──────────────────────────────────────────────────────
print("\n[Setup] Loading data and models...")
feat_daily = pd.read_parquet(FEAT_DIR / "features_daily.parquet")
with open(FEAT_DIR / "feature_meta_daily.json") as f:
    meta = json.load(f)
feats  = meta["global_features"]
TARGET = "pm25_target"

model_xgb = xgb.XGBRegressor()
model_xgb.load_model(str(MODEL_DIR / "xgb_global_daily_4yr.json"))

train_df = feat_daily[feat_daily.index.year.isin([2021, 2022, 2023, 2024])]
test_df  = feat_daily[feat_daily.index.year == 2025]

# ═══════════════════════════════════════════════════════════════════════════════
# PLOT 1: SHAP Interaction — wind_speed ∩ pm25_lag1
# ═══════════════════════════════════════════════════════════════════════════════
print("\n[1/4] SHAP Interaction Analysis (wind_speed conditioned on pm25_lag1)...")

sample = feat_daily[feats].dropna().sample(n=min(3000, len(feat_daily)), random_state=42)
explainer   = shap.TreeExplainer(model_xgb)
shap_vals   = explainer.shap_values(sample)
shap_df     = pd.DataFrame(shap_vals, columns=feats, index=sample.index)

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle("SHAP Feature Interaction Analysis — Global XGBoost Daily",
             fontsize=13, fontweight="bold")

# A: SHAP dependence: wind_speed vs SHAP(wind_speed), coloured by pm25_lag1
ws_idx   = feats.index("wind_speed") if "wind_speed" in feats else -1
lag1_idx = feats.index("pm25_lag1")  if "pm25_lag1"  in feats else -1

if ws_idx >= 0 and lag1_idx >= 0:
    lag1_vals = sample["pm25_lag1"].values
    lag1_norm = (lag1_vals - lag1_vals.min()) / (lag1_vals.max() - lag1_vals.min() + 1e-9)
    sc = axes[0].scatter(sample["wind_speed"].values,
                          shap_df["wind_speed"].values,
                          c=lag1_norm, cmap="RdYlGn_r", alpha=0.4, s=8)
    plt.colorbar(sc, ax=axes[0], label="pm25_lag1 (normalised, red=high)")
    axes[0].set_xlabel("Wind Speed (m/s)")
    axes[0].set_ylabel("SHAP value of wind_speed")
    axes[0].set_title("SHAP Dependence: wind_speed\n(coloured by pm25_lag1)", fontweight="bold")
    axes[0].axhline(0, color="k", lw=0.5, ls="--")
    axes[0].grid(alpha=0.3)

    # B: Stratify into low/high lag1 and show mean SHAP(wind_speed) vs wind_speed
    q50 = np.percentile(lag1_vals, 50)
    low_lag   = sample["wind_speed"][lag1_vals < q50].values
    high_lag  = sample["wind_speed"][lag1_vals >= q50].values
    low_shap  = shap_df["wind_speed"][lag1_vals < q50].values
    high_shap = shap_df["wind_speed"][lag1_vals >= q50].values

    # Bin and average
    bins = np.linspace(0, 15, 15)
    for vals_ws, vals_sh, label, color in [
        (low_lag, low_shap, "Low persistence (lag1 < median)", "#43A047"),
        (high_lag, high_shap, "High persistence (lag1 ≥ median)", "#E53935"),
    ]:
        bin_means_ws = []
        bin_means_sh = []
        for lo, hi in zip(bins[:-1], bins[1:]):
            mask = (vals_ws >= lo) & (vals_ws < hi)
            if mask.sum() > 5:
                bin_means_ws.append((lo + hi) / 2)
                bin_means_sh.append(vals_sh[mask].mean())
        axes[1].plot(bin_means_ws, bin_means_sh, "o-", label=label, color=color, lw=2, ms=5)

    axes[1].axhline(0, color="k", lw=0.5, ls="--")
    axes[1].set_xlabel("Wind Speed (m/s)")
    axes[1].set_ylabel("Mean SHAP(wind_speed)")
    axes[1].set_title("Conditional Effect of Wind Speed\non PM2.5 Prediction", fontweight="bold")
    axes[1].legend(fontsize=8); axes[1].grid(alpha=0.3)

# C: Top-10 global SHAP importance
mean_abs_shap = shap_df.abs().mean().sort_values(ascending=False).head(10)
colors_shap = plt.cm.Blues(np.linspace(0.4, 0.85, 10))[::-1]
axes[2].barh(range(10), mean_abs_shap.values[::-1], color=colors_shap[::-1], alpha=0.85)
axes[2].set_yticks(range(10))
axes[2].set_yticklabels([f.replace("pm25_", "pm25\n") for f in mean_abs_shap.index[::-1]], fontsize=9)
axes[2].set_xlabel("Mean |SHAP value|")
axes[2].set_title("Top-10 Global Feature Importances\n(SHAP)", fontweight="bold")
axes[2].grid(axis="x", alpha=0.3)

plt.tight_layout()
plt.savefig(PLOT_DIR / "14_shap_interaction.png", dpi=150, bbox_inches="tight")
plt.close()
print("    Saved: code/plots/14_shap_interaction.png")

# ═══════════════════════════════════════════════════════════════════════════════
# PLOT 2: Latent Space Embedding (AE or VAE)
# ═══════════════════════════════════════════════════════════════════════════════
print("\n[2/4] Latent Space Visualisation (t-SNE of Autoencoder embeddings)...")

class Autoencoder(torch.nn.Module):
    def __init__(self, n_features, latent_dim=4):
        super().__init__()
        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(n_features, 32), torch.nn.ReLU(),
            torch.nn.Linear(32, 16),         torch.nn.ReLU(),
            torch.nn.Linear(16, latent_dim)
        )
        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(latent_dim, 16), torch.nn.ReLU(),
            torch.nn.Linear(16, 32),         torch.nn.ReLU(),
            torch.nn.Linear(32, n_features)
        )
    def forward(self, x): return self.decoder(self.encoder(x))

POLLUTANT_COLS = ['pm25', 'pm10', 'no', 'no2', 'nox', 'nh3', 'so2', 'co',
                  'ozone', 'temp', 'humidity', 'wind_speed']
ae_cols = [c for c in POLLUTANT_COLS if c in feat_daily.columns]
ae_data  = feat_daily[ae_cols + ["aqi_category"] if "aqi_category" in feat_daily.columns else ae_cols].dropna()
ae_data_feats = ae_data[ae_cols] if isinstance(ae_data, pd.DataFrame) else ae_data

try:
    sc_ae   = joblib.load(MODEL_DIR / "autoencoder_scaler.pkl")
    ae_feat = joblib.load(MODEL_DIR / "autoencoder_features.pkl")
    ae_feat = [c for c in ae_feat if c in ae_data_feats.columns]
    ae_model = Autoencoder(len(ae_feat))
    ae_model.load_state_dict(torch.load(MODEL_DIR / "autoencoder.pt", map_location="cpu"))
    ae_model.eval()

    sample_ae = ae_data_feats[ae_feat].dropna().sample(n=min(4000, len(ae_data_feats)), random_state=42)
    X_ae = torch.tensor(sc_ae.transform(sample_ae.values), dtype=torch.float32)
    with torch.no_grad():
        z = ae_model.encoder(X_ae).numpy()  # latent codes

    # t-SNE in 2D
    tsne = TSNE(n_components=2, random_state=42, perplexity=40, max_iter=500)
    z2d  = tsne.fit_transform(z)

    # Assign seasons
    months = sample_ae.index.month
    season_map = {12:"Winter", 1:"Winter", 2:"Winter",
                  3:"Spring", 4:"Spring", 5:"Spring",
                  6:"Summer", 7:"Summer", 8:"Summer",
                  9:"Monsoon",10:"Post-Monsoon",11:"Post-Monsoon"}
    seasons = [season_map.get(m, "Other") for m in months]
    # season_map and season_palette already defined at module level

    fig, ax = plt.subplots(figsize=(10, 7))
    for season, color in season_palette.items():
        mask = [s == season for s in seasons]
        if any(mask):
            ax.scatter(z2d[mask, 0], z2d[mask, 1],
                       c=color, label=season, alpha=0.5, s=12)
    ax.set_title("t-SNE of Autoencoder Latent Space\n(Coloured by Season)",
                 fontsize=13, fontweight="bold")
    ax.set_xlabel("t-SNE Dim 1"); ax.set_ylabel("t-SNE Dim 2")
    ax.legend(title="Season", fontsize=9, markerscale=2)
    ax.grid(alpha=0.2)
    plt.tight_layout()
    plt.savefig(PLOT_DIR / "14_latent_tsne.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("    Saved: code/plots/14_latent_tsne.png")
except Exception as e:
    print(f"    Skipped latent space plot: {e}")

# ═══════════════════════════════════════════════════════════════════════════════
# PLOT 3: Per-Station Model MAE Heatmap (models × stations)
# ═══════════════════════════════════════════════════════════════════════════════
print("\n[3/4] Per-Station × Model MAE Heatmap...")

res_df = pd.read_csv(MODEL_DIR / "results_all_models.csv")
# Filter 4yr, daily results
daily_4yr = res_df[(res_df["freq"] == "daily") & (res_df["train_size"] == "4yr")]
# We'll use test_df per-station vs global XGBoost for a heatmap we can actually compute
model_names = []
mae_matrix = {}

for stn in ["Anand_Vihar", "Ashok_Vihar", "Bawana", "Dwarka-Sector_8",
            "Jahangirpuri", "Mundka", "Punjabi_Bagh", "Rohini", "Wazirpur"]:
    sdf = test_df[test_df["station"] == stn]
    if len(sdf) == 0: continue
    y_s = sdf[TARGET].values

    preds_global = model_xgb.predict(sdf[feats].values)
    persist_preds = sdf["pm25_lag1"].values if "pm25_lag1" in sdf.columns else np.full(len(y_s), y_s.mean())
    preds_q05 = joblib.load(MODEL_DIR / "xgb_daily_q05.pkl").predict(sdf[feats].values)
    preds_q95 = joblib.load(MODEL_DIR / "xgb_daily_q95.pkl").predict(sdf[feats].values)
    preds_mid = (preds_q05 + preds_q95) / 2

    mae_matrix[stn.replace("_", " ")] = {
        "Persistence\n(Baseline)": round(float(np.mean(np.abs(y_s - persist_preds))), 2),
        "XGBoost\n(Global)": round(float(np.mean(np.abs(y_s - preds_global))), 2),
        "XGBoost\n(Quantile Mid)": round(float(np.mean(np.abs(y_s - preds_mid))), 2),
    }
    # Per-station model if available
    pst_path = MODEL_DIR / f"xgb_{stn}_daily_4yr.json"
    if pst_path.exists():
        m_ps = xgb.XGBRegressor(); m_ps.load_model(str(pst_path))
        pst_feat = meta.get("per_station_features", feats)
        avail = [f for f in pst_feat if f in sdf.columns]
        mae_matrix[stn.replace("_", " ")]["XGBoost\n(Per-Station)"] = \
            round(float(np.mean(np.abs(y_s - m_ps.predict(sdf[avail].values)))), 2)

hm_plot = pd.DataFrame(mae_matrix).T
fig, ax = plt.subplots(figsize=(12, 6))
sns.heatmap(hm_plot, ax=ax, cmap="YlOrRd", annot=True, fmt=".1f",
            linewidths=0.5, cbar_kws={"label": "MAE (µg/m³)"},
            annot_kws={"size": 9})
ax.set_title("Per-Station × Model MAE Heatmap (Daily, 2025 Test Set)",
             fontsize=13, fontweight="bold")
ax.set_xlabel("Station"); ax.set_ylabel("Model Type")
plt.xticks(rotation=35, ha="right"); plt.tight_layout()
plt.savefig(PLOT_DIR / "14_station_model_heatmap.png", dpi=150, bbox_inches="tight")
plt.close()
print("    Saved: code/plots/14_station_model_heatmap.png")

# ═══════════════════════════════════════════════════════════════════════════════
# PLOT 4: Per-Season Error Analysis
# ═══════════════════════════════════════════════════════════════════════════════
print("\n[4/4] Seasonal error breakdown...")
feat_daily_copy = feat_daily.copy()
preds_all = model_xgb.predict(feat_daily[feats].values)
feat_daily_copy["pred"] = preds_all
feat_daily_copy["error"] = np.abs(feat_daily_copy[TARGET] - feat_daily_copy["pred"])
feat_daily_copy["season"] = feat_daily_copy.index.month.map(season_map)
feat_daily_copy = feat_daily_copy.dropna(subset=["error", "season"])

season_order = ["Winter", "Spring", "Summer", "Monsoon", "Post-Monsoon"]
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("Seasonal Model Error Analysis — XGBoost Global Daily",
             fontsize=13, fontweight="bold")

season_mae = feat_daily_copy.groupby("season")["error"].mean().reindex(season_order)
season_std = feat_daily_copy.groupby("season")["error"].std().reindex(season_order)
s_colors = [season_palette[s] for s in season_order]
axes[0].bar(range(len(season_order)), season_mae.values, color=s_colors, alpha=0.85)
axes[0].errorbar(range(len(season_order)), season_mae.values,
                 yerr=season_std.values, fmt="none", color="black", capsize=5)
axes[0].set_xticks(range(len(season_order)))
axes[0].set_xticklabels(season_order, fontsize=9)
axes[0].set_ylabel("Mean Absolute Error (µg/m³)")
axes[0].set_title("Mean Model Error by Season\n(bar = mean, whisker = std)")
axes[0].grid(axis="y", alpha=0.3)
for i, (v, s) in enumerate(zip(season_mae, season_std)):
    axes[0].text(i, v + s + 0.3, f"{v:.1f}", ha="center", fontsize=9)

season_pm = feat_daily_copy.groupby("season")["pm25"].mean().reindex(season_order) \
            if "pm25" in feat_daily_copy else pd.Series(dtype=float)
axes[1].scatter(season_pm.values, season_mae.values, s=120,
                c=s_colors, zorder=5, edgecolors="black", lw=0.5)
for sn, x, y in zip(season_order, season_pm.values, season_mae.values):
    axes[1].annotate(sn, (x, y), textcoords="offset points",
                     xytext=(5, 5), fontsize=8)
axes[1].set_xlabel("Mean PM2.5 (µg/m³)"); axes[1].set_ylabel("Mean MAE (µg/m³)")
axes[1].set_title("Mean PM2.5 vs Mean Error by Season")
axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig(PLOT_DIR / "14_seasonal_error.png", dpi=150, bbox_inches="tight")
plt.close()
print("    Saved: code/plots/14_seasonal_error.png")

print("\n" + "=" * 70)
print("  POLISH PLOTS SUMMARY")
print("  Generated: 14_shap_interaction.png")
print("             14_latent_tsne.png (if AE model loaded)")
print("             14_station_model_heatmap.png")
print("             14_seasonal_error.png")
print("=" * 70)
