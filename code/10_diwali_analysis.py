"""
10_diwali_analysis.py
══════════════════════════════════════════════════════════════════════════════
Diwali Event-Specific Error Analysis (Out-of-Fold — Correct Methodology)
─────────────────────────────────────────────────────────────────────────────
Uses walk-forward OOF (out-of-fold) predictions so Diwali events are always
evaluated on HELD-OUT test years, never on training data.

Fold assignments (matching script 09):
  Fold 2: train 2021-22 → test 2023 → Diwali 2023 (Nov 12)
  Fold 3: train 2021-23 → test 2024 → Diwali 2024 (Nov 1)

Analyses:
  1. Diwali window (±7 days) MAE vs non-event days — full vs ablated model.
  2. Per-station Diwali MAE breakdown.
  3. Residual decay curve post-Diwali (how fast error normalises).
  4. The `days_since_diwali` feature's direct contribution.

Run from project root:
    python code/10_diwali_analysis.py
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

# Diwali dates in TEST years (2023, 2024) only — never in training data
DIWALI_IN_TEST = {
    2023: pd.Timestamp("2023-11-12"),
    2024: pd.Timestamp("2024-11-01"),
}
STATIONS = [
    "Anand_Vihar", "Ashok_Vihar", "Bawana", "Dwarka-Sector_8",
    "Jahangirpuri", "Mundka", "Punjabi_Bagh", "Rohini", "Wazirpur"
]
EVENT_WINDOW = 7  # ±7 days

print("=" * 70)
print("  10  DIWALI EVENT-SPECIFIC ERROR ANALYSIS (OOF — Correct)")
print("=" * 70)

# ── Load data ──────────────────────────────────────────────────────────────────
print("\n[1/5] Loading features...")
feat_daily = pd.read_parquet(FEAT_DIR / "features_daily.parquet")
with open(FEAT_DIR / "feature_meta_daily.json") as f:
    meta = json.load(f)
feats  = meta["global_features"]
TARGET = "pm25_target"
event_cols = ["days_since_diwali", "is_diwali"]
feats_noevent = [f for f in feats if f not in event_cols]

# ── OOF folds: train on prior years, test on target year ─────────────────────
print("\n[2/5] Generating out-of-fold predictions for 2023 and 2024...")

OOF_FOLDS = [
    {"train": [2021, 2022],          "test": 2023},
    {"train": [2021, 2022, 2023],    "test": 2024},
]

XGB_PARAMS = dict(n_estimators=500, learning_rate=0.05, max_depth=6,
                  subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0,
                  random_state=42, n_jobs=-1)

oof_records = []   # list of dicts with columns: date, station, y_true, pred_full, pred_noevent

for fold in OOF_FOLDS:
    tr_df = feat_daily[feat_daily.index.year.isin(fold["train"])]
    te_df = feat_daily[feat_daily.index.year == fold["test"]]
    print(f"  Fold: train {fold['train']} → test {fold['test']}")

    # Full model (with Diwali features)
    m_full = xgb.XGBRegressor(**XGB_PARAMS)
    m_full.fit(tr_df[feats].values, tr_df[TARGET].values, verbose=False)
    preds_full = m_full.predict(te_df[feats].values)

    # Ablated model (without Diwali features)
    m_ablate = xgb.XGBRegressor(**XGB_PARAMS)
    m_ablate.fit(tr_df[feats_noevent].values, tr_df[TARGET].values, verbose=False)
    preds_ablate = m_ablate.predict(te_df[feats_noevent].values)

    y_te = te_df[TARGET].values
    mae_full   = mean_absolute_error(y_te, preds_full)
    mae_ablate = mean_absolute_error(y_te, preds_ablate)
    print(f"    Full model MAE:    {mae_full:.3f}")
    print(f"    Ablated model MAE: {mae_ablate:.3f}")

    for i, (idx, row) in enumerate(te_df.iterrows()):
        oof_records.append({
            "date":         idx,
            "station":      row["station"],
            "y_true":       y_te[i],
            "pred_full":    preds_full[i],
            "pred_noevent": preds_ablate[i],
        })

oof_df = pd.DataFrame(oof_records)
oof_df["date"] = pd.to_datetime(oof_df["date"])
oof_df["resid_full"]    = oof_df["y_true"] - oof_df["pred_full"]
oof_df["resid_noevent"] = oof_df["y_true"] - oof_df["pred_noevent"]
oof_df["abs_err_full"]  = oof_df["resid_full"].abs()
oof_df["abs_err_noevent"] = oof_df["resid_noevent"].abs()

# ── Create Diwali window mask ─────────────────────────────────────────────────
print("\n[3/5] Computing event masks and MAE comparisons...")
all_diwali = list(DIWALI_IN_TEST.values())

def in_diwali_window(ts, window=EVENT_WINDOW):
    return any(abs((ts - d).days) <= window for d in all_diwali)

oof_df["is_diwali_window"] = oof_df["date"].apply(in_diwali_window)

ev  = oof_df[oof_df["is_diwali_window"]]
nev = oof_df[~oof_df["is_diwali_window"]]

rows_comp = []
for period, subset in [("Diwali Window (±7 days)", ev), ("Non-Event Days", nev)]:
    for model_label, err_col in [
        ("Full Model (with Diwali features)", "abs_err_full"),
        ("Ablated (no Diwali features)",      "abs_err_noevent"),
    ]:
        mae = subset[err_col].mean()
        rows_comp.append({"Period": period, "Model": model_label, "MAE": round(mae, 3)})
        print(f"    [{period}]  {model_label}  →  MAE = {mae:.3f}")

comp_df = pd.DataFrame(rows_comp)
comp_df.to_csv(RES_DIR / "10_diwali_event_comparison.csv", index=False)

# ── Per-station Diwali breakdown ───────────────────────────────────────────────
print("\n[4/5] Per-station Diwali breakdown...")
stn_rows = []
for stn in STATIONS:
    ev_stn  = ev[ev["station"] == stn]
    nev_stn = nev[nev["station"] == stn]
    if len(ev_stn) == 0:
        continue
    stn_rows.append({
        "Station": stn.replace("_", " "),
        "Diwali Window MAE": round(ev_stn["abs_err_full"].mean(), 2),
        "Non-Event MAE":     round(nev_stn["abs_err_full"].mean(), 2),
        "Diwali Uplift (×)": round(ev_stn["abs_err_full"].mean() /
                                    max(nev_stn["abs_err_full"].mean(), 1), 2),
    })
stn_df = pd.DataFrame(stn_rows).set_index("Station")
print(stn_df.to_string())

# ── Residual decay curves post-Diwali ─────────────────────────────────────────
print("\n[5/5] Generating all plots...")
fig = plt.figure(figsize=(18, 12))
fig.suptitle("Diwali Event-Specific Analysis — OOF Evaluation\n"
             "(Diwali 2023 & 2024 evaluated on out-of-sample data)",
             fontsize=14, fontweight="bold")

yr_colors = {"2023": "#4A9EFF", "2024": "#E53935"}

# ── Subplot 1: Event vs non-event MAE ─────────────────────────────────────────
ax1 = fig.add_subplot(2, 3, 1)
ev_full  = rows_comp[0]["MAE"]
ev_no    = rows_comp[1]["MAE"]
nev_full = rows_comp[2]["MAE"]
nev_no   = rows_comp[3]["MAE"]
bar_w = 0.35
bars1 = ax1.bar([0 - bar_w/2, 1 - bar_w/2], [ev_full, nev_full],
                bar_w, label="Full Model (with Diwali feat.)", color="#4A9EFF", alpha=0.85)
bars2 = ax1.bar([0 + bar_w/2, 1 + bar_w/2], [ev_no,   nev_no],
                bar_w, label="Ablated (no Diwali feat.)",  color="#FF7043", alpha=0.85)
ax1.set_xticks([0, 1])
ax1.set_xticklabels(["Diwali Window\n(±7 days)", "Non-Event Days"])
ax1.set_ylabel("MAE (µg/m³)")
ax1.set_title("Event vs Non-Event MAE\n(OOF, 2023 + 2024)", fontweight="bold")
ax1.legend(fontsize=7); ax1.grid(axis="y", alpha=0.3)
for b in list(bars1) + list(bars2):
    ax1.text(b.get_x() + b.get_width()/2, b.get_height() + 0.5,
             f"{b.get_height():.1f}", ha="center", fontsize=8)

# ── Subplot 2: Per-station Diwali heatmap ─────────────────────────────────────
ax2 = fig.add_subplot(2, 3, 2)
hm_data = stn_df[["Diwali Window MAE", "Non-Event MAE"]]
sns.heatmap(hm_data, ax=ax2, cmap="YlOrRd", annot=True, fmt=".1f",
            linewidths=0.5, cbar_kws={"label": "MAE (µg/m³)"})
ax2.set_title("Per-Station MAE\n(Diwali Window vs Non-Event)", fontweight="bold")

# ── Subplot 3: Diwali uplift by station ───────────────────────────────────────
ax3 = fig.add_subplot(2, 3, 3)
stn_df_sorted = stn_df.sort_values("Diwali Uplift (×)", ascending=True)
colors_uplift = ["#E53935" if u > 3 else "#FB8C00" if u > 2 else "#43A047"
                 for u in stn_df_sorted["Diwali Uplift (×)"]]
ax3.barh(range(len(stn_df_sorted)), stn_df_sorted["Diwali Uplift (×)"],
         color=colors_uplift, alpha=0.85)
ax3.axvline(1, color="black", lw=0.8, ls="--")
ax3.set_yticks(range(len(stn_df_sorted)))
ax3.set_yticklabels(stn_df_sorted.index.tolist(), fontsize=9)
ax3.set_xlabel("Diwali MAE / Non-Event MAE  (× uplift)")
ax3.set_title("Prediction Difficulty Uplift\nduring Diwali Events", fontweight="bold")
ax3.grid(axis="x", alpha=0.3)
for i, v in enumerate(stn_df_sorted["Diwali Uplift (×)"]):
    ax3.text(v + 0.05, i, f"{v:.1f}×", va="center", fontsize=9)

# ── Subplot 4+5: Residual decay curves (one per test Diwali year) ─────────────
for ax_idx, (yr, d_date) in enumerate(DIWALI_IN_TEST.items(), 4):
    ax = fig.add_subplot(2, 3, ax_idx)
    color = yr_colors[str(yr)]
    window_data = oof_df[
        (oof_df["date"] >= d_date - pd.Timedelta(f"{EVENT_WINDOW}d")) &
        (oof_df["date"] <= d_date + pd.Timedelta("12d")) &
        (oof_df["date"].dt.year == yr)
    ]
    if len(window_data) == 0:
        continue

    daily_err_full = window_data.groupby(window_data["date"].dt.date)["abs_err_full"].mean()
    daily_err_noe  = window_data.groupby(window_data["date"].dt.date)["abs_err_noevent"].mean()
    offsets = [(pd.Timestamp(d) - d_date).days for d in daily_err_full.index]

    ax.plot(offsets, daily_err_full.values, "o-", color=color,
            label="Full model", lw=2, ms=6)
    ax.plot(offsets, daily_err_noe.values, "s--", color="#888888",
            label="No Diwali feat.", lw=1.5, ms=5, alpha=0.8)
    ax.axvline(0, color="gold", ls="--", lw=1.5, alpha=0.9, label="Diwali Day 0")
    ax.axvspan(-EVENT_WINDOW, 0, alpha=0.04, color="gold")
    ax.axvspan(0, 5, alpha=0.06, color="red")
    ax.set_xlabel("Days Relative to Diwali")
    ax.set_ylabel("Mean |Residual| (µg/m³)")
    ax.set_title(f"Residual Decay Curve — Diwali {yr}\n({d_date.strftime('%b %d')})",
                 fontweight="bold")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

# ── Subplot 6: Feature gain from Diwali features ──────────────────────────────
ax6 = fig.add_subplot(2, 3, 6)
feature_gain = ev_no - ev_full  # positive means full model is better
nonevent_gain = nev_no - nev_full

categories  = ["Diwali Window\n(±7 days)", "Non-Event\nDays"]
gains       = [feature_gain, nonevent_gain]
bar_colors  = ["#43A047" if g > 0 else "#E53935" for g in gains]
bars6 = ax6.bar(range(len(categories)), gains, color=bar_colors, alpha=0.85, width=0.5)
ax6.axhline(0, color="black", lw=1)
ax6.set_xticks(range(len(categories)))
ax6.set_xticklabels(categories)
ax6.set_ylabel("MAE reduction from Diwali features (µg/m³)")
ax6.set_title("Marginal Value of Diwali Features\n(+ means helpful)", fontweight="bold")
ax6.grid(axis="y", alpha=0.3)
for b, g in zip(bars6, gains):
    ax6.text(b.get_x() + b.get_width()/2, g + (0.3 if g >= 0 else -1.5),
             f"{g:+.1f}", ha="center", fontsize=10, fontweight="bold")

plt.tight_layout()
plt.savefig(PLOT_DIR / "10_diwali_event_analysis.png", dpi=150, bbox_inches="tight")
plt.close()
print("    Saved: code/plots/10_diwali_event_analysis.png")

# Save per-station results
stn_df.to_csv(RES_DIR / "10_diwali_per_station.csv")

print("\n" + "=" * 70)
print("  DIWALI ANALYSIS SUMMARY (OOF)")
print(f"  Diwali window:  Full model MAE = {ev_full:.2f}")
print(f"                  No Diwali feat = {ev_no:.2f}")
print(f"  Diwali feature gain:  {ev_no - ev_full:+.2f} µg/m³")
print(f"  Non-event MAE (full): {nev_full:.2f}")
print(f"  Diwali window is {ev_full/max(nev_full,1):.2f}× harder to predict than non-event days")
print("=" * 70)
