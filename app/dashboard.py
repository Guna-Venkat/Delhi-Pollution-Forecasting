"""
app/dashboard.py
─────────────────────────────────────────────────────────────────
Delhi Air Quality Forecasting Dashboard
Streamlit app — dark theme — all 4 tabs:
  1. Forecast        : station + frequency selector, 24h/7d chart, AQI badge
  2. What-If         : SHAP wind/temp sliders for scenario analysis
  3. Anomaly         : autoencoder reconstruction error timeline
  4. Station Clusters: k-means seasonal profile map
─────────────────────────────────────────────────────────────────
"""

import json
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import joblib
import torch
import xgboost as xgb
from pathlib import Path
from huggingface_hub import hf_hub_download
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# ── Page config (must be first Streamlit call) ────────────────────────────────
st.set_page_config(
    page_title="Delhi AQ Forecast",
    page_icon="🌫️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Dark theme CSS injection ──────────────────────────────────────────────────
st.markdown("""
<style>
  /* Main background */
  .stApp { background-color: #0f0f13; color: #e2e2e8; }
  section[data-testid="stSidebar"] { background-color: #12121a; }

  /* Metric cards */
  div[data-testid="metric-container"] {
    background: #16161f;
    border: 1px solid #2a2a40;
    border-radius: 10px;
    padding: 12px 16px;
  }
  div[data-testid="metric-container"] label { color: #888 !important; }
  div[data-testid="metric-container"] div[data-testid="stMetricValue"] {
    color: #c9b8ff !important; font-size: 1.6rem !important;
  }

  /* Tab styling */
  button[data-baseweb="tab"] { color: #aaa !important; }
  button[data-baseweb="tab"][aria-selected="true"] {
    color: #c9b8ff !important;
    border-bottom: 2px solid #8b7cf8 !important;
  }

  /* Slider */
  div[data-testid="stSlider"] > div { color: #c9b8ff; }

  /* Divider */
  hr { border-color: #2a2a40; }
</style>
""", unsafe_allow_html=True)

# ── Constants ─────────────────────────────────────────────────────────────────
HF_REPO_ID = "Guna-Venkat-Doddi-251140009/delhi-aq-models"

STATIONS = [
    "Anand_Vihar", "Ashok_Vihar", "Bawana", "Dwarka-Sector_8",
    "Jahangirpuri", "Mundka", "Punjabi_Bagh", "Rohini", "Wazirpur",
]
STATION_DISPLAY = {s: s.replace("_", " ") for s in STATIONS}

AQI_CATEGORIES = [
    (0,   50,  "Good",          "#43A047"),
    (51,  100, "Satisfactory",  "#8BC34A"),
    (101, 200, "Moderate",      "#FDD835"),
    (201, 300, "Poor",          "#FB8C00"),
    (301, 400, "Very Poor",     "#E53935"),
    (401, 500, "Severe",        "#7B1FA2"),
]

AQI_BREAKPOINTS = {
    "pm25": [(0,30,0,50),(31,60,51,100),(61,90,101,200),
             (91,120,201,300),(121,250,301,400),(251,380,401,500)],
    "pm10": [(0,50,0,50),(51,100,51,100),(101,250,101,200),
             (251,350,201,300),(351,430,301,400),(431,600,401,500)],
}

# Must match the features used during autoencoder training (10 features)
POLLUTANT_COLS = [
    "pm25", "pm10", "no2", "so2", "co", "ozone", "nh3",
    "temp", "humidity", "wind_speed"
]

DIWALI_DATES = pd.to_datetime([
    "2021-11-04", "2022-10-24", "2023-11-12", "2024-11-01",
])

# ── Paths (local dev) ─────────────────────────────────────────────────────────
BASE_DIR     = Path(__file__).parent.parent
FEATURES_DIR = BASE_DIR / "dataset" / "features"
MODEL_DIR    = BASE_DIR / "dataset" / "models"
DATA_DIR     = BASE_DIR / "dataset" / "processed"

# ── AQI helpers ───────────────────────────────────────────────────────────────
def compute_aqi(pm25: float, pm10: float = None) -> float:
    subs = []
    for pol, val in [("pm25", pm25), ("pm10", pm10)]:
        if val is None or np.isnan(val):
            continue
        for c_lo, c_hi, i_lo, i_hi in AQI_BREAKPOINTS[pol]:
            if c_lo <= val <= c_hi:
                subs.append(((i_hi-i_lo)/(c_hi-c_lo))*(val-c_lo)+i_lo)
                break
        else:
            if val > AQI_BREAKPOINTS[pol][-1][1]:
                subs.append(500.0)
    return round(max(subs), 1) if subs else float("nan")


def aqi_category(aqi_val: float) -> tuple:
    """Returns (label, color) for an AQI value."""
    if np.isnan(aqi_val):
        return ("Unknown", "#888")
    for lo, hi, label, color in AQI_CATEGORIES:
        if lo <= aqi_val <= hi:
            return label, color
    return ("Severe", "#7B1FA2")


def aqi_badge(aqi_val: float) -> str:
    label, color = aqi_category(aqi_val)
    return f"""
    <div style="display:inline-block;background:{color};color:white;
    padding:6px 18px;border-radius:20px;font-weight:600;font-size:1rem;">
    AQI {aqi_val:.0f} — {label}
    </div>"""

# ── Autoencoder model class (must match training definition) ─────────────────
class Autoencoder(torch.nn.Module):
    def __init__(self, n_features, latent_dim=4):
        super().__init__()
        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(n_features, 32), torch.nn.ReLU(),
            torch.nn.Linear(32, 16),         torch.nn.ReLU(),
            torch.nn.Linear(16, latent_dim),
        )
        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(latent_dim, 16), torch.nn.ReLU(),
            torch.nn.Linear(16, 32),         torch.nn.ReLU(),
            torch.nn.Linear(32, n_features),
        )
    def forward(self, x):
        return self.decoder(self.encoder(x))

# ── Model loading (cached) ────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading models from Hugging Face…")
def load_models():
    """
    Downloads model files from HF Hub on first run, then caches in memory.
    Falls back to local MODEL_DIR if HF download fails.
    Optional models (autoencoder, conformal) are skipped if missing.
    """
    models = {}

    def _load_xgb(name, local_path):
        try:
            path = hf_hub_download(repo_id=HF_REPO_ID, filename=name)
        except Exception:
            path = str(local_path)
        m = xgb.XGBRegressor()
        m.load_model(path)
        return m

    def _load_pt(name, local_path, required=False):
        try:
            path = hf_hub_download(repo_id=HF_REPO_ID, filename=name)
        except Exception:
            path = str(local_path)
        if not Path(path).exists():
            if required:
                raise FileNotFoundError(f"Required file {path} not found")
            else:
                st.warning(f"Optional file {name} not found. Anomaly detection disabled.")
                return None
        return torch.load(path, map_location="cpu")

    def _load_pkl(name, local_path, required=False):
        try:
            path = hf_hub_download(repo_id=HF_REPO_ID, filename=name)
        except Exception:
            path = str(local_path)
        if not Path(path).exists():
            if required:
                raise FileNotFoundError(f"Required file {path} not found")
            else:
                st.warning(f"Optional file {name} not found. Conformal intervals disabled.")
                return None
        return joblib.load(path)

    # XGBoost — point forecasts (required)
    models["xgb_1hr"]   = _load_xgb("xgb_global_1hr_4yr.json",   MODEL_DIR/"xgb_global_1hr_4yr.json")
    models["xgb_daily"] = _load_xgb("xgb_global_daily_4yr.json", MODEL_DIR/"xgb_global_daily_4yr.json")

    # Optional: quantile models for intervals
    models["xgb_q05_daily"] = _load_pkl("xgb_daily_q05.pkl", MODEL_DIR/"xgb_daily_q05.pkl", required=False)
    models["xgb_q95_daily"] = _load_pkl("xgb_daily_q95.pkl", MODEL_DIR/"xgb_daily_q95.pkl", required=False)
    models["conformal_q"]   = _load_pkl("conformal_q_daily.pkl", MODEL_DIR/"conformal_q_daily.pkl", required=False)

    # Optional: autoencoder state dict, scaler, and feature list
    models["autoencoder_state"] = _load_pt("autoencoder.pt", MODEL_DIR/"autoencoder.pt", required=False)
    models["autoencoder_scaler"] = _load_pkl("autoencoder_scaler.pkl", MODEL_DIR/"autoencoder_scaler.pkl", required=False)
    models["autoencoder_features"] = _load_pkl("autoencoder_features.pkl", MODEL_DIR/"autoencoder_features.pkl", required=False)

    return models


@st.cache_data(show_spinner="Loading feature data…")
def load_data():
    feat_1hr   = pd.read_parquet(FEATURES_DIR / "features_1hr.parquet")
    feat_daily = pd.read_parquet(FEATURES_DIR / "features_daily.parquet")
    with open(FEATURES_DIR / "feature_meta_1hr.json")   as f:
        meta_1hr   = json.load(f)
    with open(FEATURES_DIR / "feature_meta_daily.json") as f:
        meta_daily = json.load(f)
    return feat_1hr, feat_daily, meta_1hr, meta_daily


# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## 🌫️ Delhi AQ Forecast")
    st.markdown("*9 CPCB stations · 2021–2025*")
    st.divider()

    station = st.selectbox(
        "📍 Station",
        STATIONS,
        format_func=lambda s: STATION_DISPLAY[s],
    )
    freq = st.radio(
        "⏱ Forecast frequency",
        ["Hourly (24h)", "Daily (7 days)"],
        horizontal=True,
    )
    is_hourly = freq == "Hourly (24h)"

    st.divider()
    st.caption("**Interval method (daily)**")
    interval_method = st.radio(
        "",
        ["Conformal (guaranteed 91%)", "Quantile Regression (adaptive)"],
        label_visibility="collapsed",
    )
    use_conformal = "Conformal" in interval_method

    st.divider()
    st.caption("Data: CPCB Delhi monitoring network")
    st.caption("Models: XGBoost (daily) · PatchTST (hourly)")

# ══════════════════════════════════════════════════════════════════════════════
# LOAD DATA & MODELS
# ══════════════════════════════════════════════════════════════════════════════
feat_1hr, feat_daily, meta_1hr, meta_daily = load_data()
models = load_models()

feat_df  = feat_1hr   if is_hourly else feat_daily
meta     = meta_1hr   if is_hourly else meta_daily
model    = models["xgb_1hr"] if is_hourly else models["xgb_daily"]
horizon  = 24 if is_hourly else 7

# Filter to selected station
station_df = feat_df[feat_df["station"] == station].copy()

# Get most recent available window for inference
latest = station_df.tail(max(horizon * 4, 48))
X_latest = latest[meta["global_features"]].values
preds    = model.predict(X_latest)

# Current conditions = latest row
current = station_df.iloc[-1]
current_pm25 = float(current["pm25"])
current_aqi  = float(current["aqi"])
current_temp = float(current["temp"])
current_ws   = float(current["wind_speed"])

# Forecast timestamps
last_ts = station_df.index[-1]
if is_hourly:
    future_ts = pd.date_range(last_ts + pd.Timedelta("1h"), periods=24, freq="1h")
else:
    future_ts = pd.date_range(last_ts + pd.Timedelta("1d"), periods=7,  freq="D")

# Use last `horizon` predictions
forecast_vals = preds[-horizon:]
forecast_aqi  = [compute_aqi(v) for v in forecast_vals]

# ══════════════════════════════════════════════════════════════════════════════
# HEADER
# ══════════════════════════════════════════════════════════════════════════════
st.title(f"🌫️ Delhi Air Quality — {STATION_DISPLAY[station]}")
col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("Current PM2.5",  f"{current_pm25:.1f} µg/m³")
col2.metric("Current AQI",    f"{current_aqi:.0f}")
col3.metric("Temperature",    f"{current_temp:.1f} °C")
col4.metric("Wind Speed",     f"{current_ws:.1f} m/s")
col5.metric("Forecast (next)", f"{forecast_vals[0]:.1f} µg/m³",
            delta=f"{forecast_vals[0]-current_pm25:+.1f}")

# AQI badge
st.markdown(aqi_badge(current_aqi), unsafe_allow_html=True)

# Alert banner
max_forecast_aqi = max(forecast_aqi)
if max_forecast_aqi > 300:
    cat, _ = aqi_category(max_forecast_aqi)
    st.error(f"⚠️ **Forecast Alert** — Predicted AQI may reach **{max_forecast_aqi:.0f} ({cat})** "
             f"within the next {'24 hours' if is_hourly else '7 days'}. Avoid prolonged outdoor activity.")

st.divider()

# ══════════════════════════════════════════════════════════════════════════════
# TABS
# ══════════════════════════════════════════════════════════════════════════════
tab1, tab2, tab3, tab4 = st.tabs([
    "📈 Forecast", "🔍 What-If Analysis", "🚨 Anomaly Detection", "🗺️ Station Clusters"
])

# ── TAB 1: FORECAST ───────────────────────────────────────────────────────────
with tab1:
    col_left, col_right = st.columns([3, 1])

    with col_left:
        # Historical + forecast chart
        hist_window = 48 if is_hourly else 30
        hist_df  = station_df.tail(hist_window)
        hist_ts  = hist_df.index
        hist_pm  = hist_df["pm25"].values

        # Prediction intervals (daily only)
        lower = None
        upper = None
        interval_name = "None"
        if not is_hourly:
            if use_conformal and models.get("conformal_q") is not None:
                conf_q = models["conformal_q"]
                lower = forecast_vals - conf_q
                upper = forecast_vals + conf_q
                interval_name = "Conformal (91%)"
            elif not use_conformal and models.get("xgb_q05_daily") is not None:
                q_lower = models["xgb_q05_daily"].predict(X_latest)[-horizon:]
                q_upper = models["xgb_q95_daily"].predict(X_latest)[-horizon:]
                lower = q_lower
                upper = q_upper
                interval_name = "Quantile (adaptive)"

        fig = go.Figure()

        # Historical trace
        fig.add_trace(go.Scatter(
            x=hist_ts, y=hist_pm,
            mode="lines", name="Historical",
            line=dict(color="#4A9EFF", width=1.5),
        ))

        # Prediction interval band (daily only)
        if not is_hourly and lower is not None and upper is not None:
            fig.add_trace(go.Scatter(
                x=list(future_ts) + list(future_ts[::-1]),
                y=list(upper) + list(lower[::-1]),
                fill="toself",
                fillcolor="rgba(139,124,248,0.15)",
                line=dict(color="rgba(0,0,0,0)"),
                name=f"90% PI ({interval_name.split()[0]})",
            ))

        # Forecast trace
        fig.add_trace(go.Scatter(
            x=future_ts, y=forecast_vals,
            mode="lines+markers", name="Forecast",
            line=dict(color="#C9B8FF", width=2, dash="dash"),
            marker=dict(size=5),
        ))

        # AQI threshold lines (Moderate and above)
        for lo, hi, label, color in AQI_CATEGORIES[2:]:
            fig.add_hline(
                y=lo, line_dash="dot", line_color=color,
                line_width=0.8, opacity=0.5,
                annotation_text=label, annotation_position="right",
                annotation_font_size=9, annotation_font_color=color,
            )

        fig.update_layout(
            title=f"PM2.5 Forecast — {STATION_DISPLAY[station]} ({'24h' if is_hourly else '7d'})",
            paper_bgcolor="#12121a", plot_bgcolor="#16161f",
            font=dict(color="#e2e2e8"),
            xaxis=dict(gridcolor="#2a2a40", showgrid=True),
            yaxis=dict(gridcolor="#2a2a40", showgrid=True, title="PM2.5 (µg/m³)"),
            legend=dict(bgcolor="#1a1a2e", bordercolor="#2a2a40"),
            height=380,
        )
        st.plotly_chart(fig, use_container_width=True)

    with col_right:
        st.markdown("**Forecast AQI**")
        for i, (ts, aqi_v) in enumerate(zip(future_ts[:7], forecast_aqi[:7])):
            label, color = aqi_category(aqi_v)
            date_str = ts.strftime("%d %b %H:%M" if is_hourly else "%a %d %b")
            st.markdown(
                f'<div style="background:#16161f;border:1px solid #2a2a40;border-left:4px solid {color};'
                f'border-radius:6px;padding:6px 10px;margin-bottom:6px;">'
                f'<span style="color:#aaa;font-size:11px;">{date_str}</span><br>'
                f'<span style="color:{color};font-weight:600;">{label}</span>'
                f'<span style="color:#888;font-size:11px;"> · {aqi_v:.0f}</span>'
                f'</div>',
                unsafe_allow_html=True,
            )

    # Forecast table
    st.markdown("**Detailed forecast table**")
    forecast_table = pd.DataFrame({
        "Timestamp": future_ts.strftime("%Y-%m-%d %H:%M" if is_hourly else "%Y-%m-%d"),
        "Forecast PM2.5 (µg/m³)": np.round(forecast_vals, 1),
        "Forecast AQI": np.round(forecast_aqi, 0).astype(int),
        "Category": [aqi_category(v)[0] for v in forecast_aqi],
    })
    st.dataframe(forecast_table, use_container_width=True, height=220)


# ── TAB 2: WHAT-IF ANALYSIS ───────────────────────────────────────────────────
with tab2:
    st.markdown("### 🔍 What-If Scenario Analysis")
    st.markdown(
        "Adjust meteorological inputs to see how the forecast changes. "
        "Based on SHAP-identified feature sensitivities from the daily XGBoost model."
    )

    # Baseline from latest daily row
    daily_latest = feat_daily[feat_daily["station"] == station].iloc[-1:].copy()
    base_pm25 = float(daily_latest["pm25"].values[0])

    col_a, col_b, col_c = st.columns(3)
    with col_a:
        ws_val   = st.slider("Wind Speed (m/s)",    0.0, 15.0,
                             float(daily_latest["wind_speed"].values[0]), 0.5)
    with col_b:
        temp_val = st.slider("Temperature (°C)",   -5.0, 50.0,
                             float(daily_latest["temp"].values[0]), 0.5)
    with col_c:
        hum_val  = st.slider("Humidity (%)",        10.0, 100.0,
                             float(daily_latest["humidity"].values[0]), 1.0)

    # Apply modified values and predict
    modified_row = daily_latest[meta_daily["global_features"]].copy()
    for col, val in [("wind_speed", ws_val), ("temp", temp_val), ("humidity", hum_val)]:
        if col in modified_row.columns:
            modified_row[col] = val
    # Recompute interaction feature
    if "humid_temp_interaction" in modified_row.columns:
        modified_row["humid_temp_interaction"] = hum_val * temp_val
    # Recompute wind u/v
    if "wind_u" in modified_row.columns and "wind_dir" in modified_row.columns:
        wd = float(daily_latest["wind_dir"].values[0])
        wd_rad = np.deg2rad(wd)
        modified_row["wind_u"] = -ws_val * np.sin(wd_rad)
        modified_row["wind_v"] = -ws_val * np.cos(wd_rad)

    scenario_pred = float(models["xgb_daily"].predict(modified_row.values)[0])
    scenario_aqi  = compute_aqi(scenario_pred)
    delta = scenario_pred - base_pm25

    col_m1, col_m2, col_m3 = st.columns(3)
    col_m1.metric("Baseline PM2.5",  f"{base_pm25:.1f} µg/m³")
    col_m2.metric("Scenario PM2.5",  f"{scenario_pred:.1f} µg/m³",
                  delta=f"{delta:+.1f}", delta_color="inverse")
    col_m3.metric("Scenario AQI",    f"{scenario_aqi:.0f}",
                  delta=f"{aqi_category(scenario_aqi)[0]}")
    st.markdown(aqi_badge(scenario_aqi), unsafe_allow_html=True)

    # Sensitivity sweep charts
    st.markdown("#### Sensitivity to each variable")
    fig_wi = go.Figure()
    sweep_configs = [
        ("wind_speed",  np.linspace(0, 15, 40),  "Wind Speed (m/s)"),
        ("temp",        np.linspace(-5, 45, 40),  "Temperature (°C)"),
        ("humidity",    np.linspace(10, 100, 40), "Humidity (%)"),
    ]
    colors_sweep = ["#4A9EFF", "#FF7043", "#66BB6A"]

    for (feat, vals, xlabel), color in zip(sweep_configs, colors_sweep):
        sweep_preds = []
        for v in vals:
            row_s = modified_row.copy()
            row_s[feat] = v
            if feat in ["wind_speed", "humidity", "temp"] and "humid_temp_interaction" in row_s.columns:
                t = temp_val if feat != "temp" else v
                h = hum_val  if feat != "humidity" else v
                row_s["humid_temp_interaction"] = h * t
            if feat == "wind_speed" and "wind_u" in row_s.columns:
                wd_rad = np.deg2rad(float(daily_latest["wind_dir"].values[0]))
                row_s["wind_u"] = -v * np.sin(wd_rad)
                row_s["wind_v"] = -v * np.cos(wd_rad)
            sweep_preds.append(float(models["xgb_daily"].predict(row_s.values)[0]))

        fig_wi.add_trace(go.Scatter(
            x=vals, y=sweep_preds, mode="lines",
            name=xlabel, line=dict(color=color, width=2)
        ))

    fig_wi.update_layout(
        paper_bgcolor="#12121a", plot_bgcolor="#16161f",
        font=dict(color="#e2e2e8"),
        xaxis=dict(title="Feature value", gridcolor="#2a2a40"),
        yaxis=dict(title="Predicted PM2.5 (µg/m³)", gridcolor="#2a2a40"),
        legend=dict(bgcolor="#1a1a2e"),
        height=320,
        title="PM2.5 Sensitivity to Meteorological Variables",
    )
    st.plotly_chart(fig_wi, use_container_width=True)


# ── TAB 3: ANOMALY DETECTION ──────────────────────────────────────────────────
# ── TAB 3: ANOMALY DETECTION (FIXED) ──────────────────────────────────────────
with tab3:
    st.markdown("### 🚨 Autoencoder Anomaly Detection")

    if models.get("autoencoder_state") is None or models.get("autoencoder_scaler") is None:
        st.warning("Autoencoder model or scaler not found. Anomaly detection is unavailable.")
    else:
        st.markdown(
            "Trained on summer months (Mar–Sep) as 'normal'. High reconstruction error = anomalous day.  \n"
            "Top anomalies correspond to Diwali, stubble burning events, and severe fog episodes."
        )

        # Load saved components
        ae_cols = models["autoencoder_features"]
        sc_ae = models["autoencoder_scaler"]
        ae_model = Autoencoder(len(ae_cols))
        ae_model.load_state_dict(models["autoencoder_state"])
        ae_model.eval()

        # Prepare data – remove duplicate columns
        ae_data = feat_daily.copy()
        ae_data = ae_data.loc[:, ~ae_data.columns.duplicated()]

        # Ensure all required columns exist
        missing_cols = set(ae_cols) - set(ae_data.columns)
        if missing_cols:
            st.error(f"Missing columns: {missing_cols}")
            st.stop()

        # Extract features and meta (DO NOT include 'pm25' in meta)
        X_raw = ae_data[ae_cols].copy()
        meta_df = ae_data[["station", "aqi_category"]].copy()

        # Combine, drop rows with NaN
        combined = pd.concat([X_raw, meta_df], axis=1).dropna()
        X_clean = combined[ae_cols].values
        X_scaled = sc_ae.transform(X_clean)

        # Get timestamps (reset index)
        combined = combined.reset_index().rename(columns={"index": "timestamp"})
        normal_mask = combined["timestamp"].dt.month.isin(range(3, 10))

        # Compute reconstruction error
        with torch.no_grad():
            X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
            recon = ae_model(X_tensor).numpy()
        recon_error = np.mean((X_scaled - recon) ** 2, axis=1)
        combined["recon_error"] = recon_error
        thresh = np.percentile(recon_error[normal_mask], 95)

        # Filter by station
        station_ae = combined[combined["station"] == station].copy()
        station_ae = station_ae.set_index("timestamp")
        daily_mean_err = station_ae["recon_error"].resample("D").mean()

        # Plot
        fig_ae = go.Figure()
        fig_ae.add_trace(go.Scatter(
            x=daily_mean_err.index, y=daily_mean_err.values,
            mode="lines", name="Reconstruction error",
            line=dict(color="#4A9EFF", width=1),
        ))
        anomaly_mask = daily_mean_err > thresh
        fig_ae.add_trace(go.Scatter(
            x=daily_mean_err.index[anomaly_mask],
            y=daily_mean_err.values[anomaly_mask],
            mode="markers", name="Anomaly",
            marker=dict(color="#E53935", size=7),
        ))
        for d in DIWALI_DATES:
            fig_ae.add_shape(
                type="line",
                x0=d, x1=d, y0=0, y1=1,
                xref="x", yref="paper",
                line=dict(color="#FFD700", width=1.5, dash="dash"),
                opacity=0.8,
            )
            fig_ae.add_annotation(
                x=d, y=1, xref="x", yref="paper",
                text=f"Diwali ({d.year})",
                showarrow=False,
                font=dict(color="#FFD700", size=9),
                xanchor="left", yanchor="bottom"
            )
        fig_ae.add_hline(y=thresh, line_dash="dot", line_color="#E53935",
                         annotation_text=f"Threshold={thresh:.3f}")
        fig_ae.update_layout(
            title=f"Anomaly Detection — {STATION_DISPLAY[station]}",
            paper_bgcolor="#12121a", plot_bgcolor="#16161f",
            font=dict(color="#e2e2e8"), height=350,
        )
        st.plotly_chart(fig_ae, use_container_width=True)

        # Top anomalies table – get pm25 from original data
        st.markdown("**Top 15 most anomalous days**")
        # Get pm25 values for the same station and timestamps
        pm25_vals = feat_daily[feat_daily["station"] == station]["pm25"]
        # Align with station_ae index (which is timestamp)
        pm25_daily = pm25_vals.reindex(station_ae.index).resample("D").mean()
        top_df = station_ae.groupby(station_ae.index.date).agg(
            recon_error=("recon_error", "mean"),
            aqi_category=("aqi_category", "first"),
        ).reset_index()
        top_df["Mean PM2.5"] = top_df["index"].map(pm25_daily)
        top_df = top_df.rename(columns={"index": "Date", "recon_error": "Reconstruction Error"})
        top_df = top_df.sort_values("Reconstruction Error", ascending=False).head(15)
        st.dataframe(top_df[["Date", "Reconstruction Error", "Mean PM2.5", "aqi_category"]],
                     use_container_width=True)

# ── TAB 4: STATION CLUSTERS ───────────────────────────────────────────────────
with tab4:
    st.markdown("### 🗺️ Station Cluster Analysis")
    st.markdown(
        "Stations grouped by their seasonal PM2.5 profile using k-means (k=3).  \n"
        "Cluster assignment reveals which stations share similar pollution dynamics."
    )

    # Compute monthly profiles
    monthly = (
        feat_daily
        .assign(month=lambda d: d.index.month)
        .groupby(["station","month"])["pm25"]
        .mean()
        .unstack("month")
    )
    sc_km = StandardScaler()
    X_km  = sc_km.fit_transform(monthly.values)
    km    = KMeans(n_clusters=3, random_state=42, n_init=10)
    clusters = km.fit_predict(X_km)
    monthly["cluster"] = clusters

    CLUSTER_LABELS = {0: "High pollution", 1: "Moderate", 2: "Low pollution"}
    CLUSTER_COLORS = {0: "#E53935", 1: "#FB8C00", 2: "#43A047"}

    # Sort clusters by mean PM2.5
    cluster_means = {
        c: monthly[monthly["cluster"]==c].drop("cluster", axis=1).values.mean()
        for c in range(3)
    }
    sorted_clusters = sorted(cluster_means, key=cluster_means.get, reverse=True)
    cluster_rename  = {old: new for new, old in enumerate(sorted_clusters)}
    monthly["cluster"] = monthly["cluster"].map(cluster_rename)

    col_left4, col_right4 = st.columns([2, 1])

    with col_left4:
        month_labels = ["Jan","Feb","Mar","Apr","May","Jun",
                        "Jul","Aug","Sep","Oct","Nov","Dec"]
        fig_cl = go.Figure()
        for _, row in monthly.iterrows():
            c = int(row["cluster"])
            station_name = row.name
            color = list(CLUSTER_COLORS.values())[c]
            fig_cl.add_trace(go.Scatter(
                x=month_labels,
                y=row.drop("cluster").values,
                mode="lines+markers",
                name=STATION_DISPLAY[station_name],
                line=dict(color=color, width=2),
                marker=dict(size=5),
                legendgroup=f"cluster_{c}",
                legendgrouptitle_text=list(CLUSTER_LABELS.values())[c] if
                    station_name == monthly[monthly["cluster"]==c].index[0] else None,
            ))
        fig_cl.update_layout(
            title="Monthly PM2.5 Profile by Station (colored by cluster)",
            paper_bgcolor="#12121a", plot_bgcolor="#16161f",
            font=dict(color="#e2e2e8"),
            xaxis=dict(gridcolor="#2a2a40"),
            yaxis=dict(title="Mean PM2.5 (µg/m³)", gridcolor="#2a2a40"),
            height=380,
            legend=dict(bgcolor="#1a1a2e", groupclick="toggleitem"),
        )
        st.plotly_chart(fig_cl, use_container_width=True)

    with col_right4:
        st.markdown("**Cluster assignments**")
        for _, row in monthly.sort_values("cluster").iterrows():
            c     = int(row["cluster"])
            color = list(CLUSTER_COLORS.values())[c]
            label = list(CLUSTER_LABELS.values())[c]
            mean_pm25 = row.drop("cluster").mean()
            is_selected = row.name == station
            border = "2px solid #8b7cf8" if is_selected else "1px solid #2a2a40"
            st.markdown(
                f'<div style="background:#16161f;border:{border};border-left:4px solid {color};'
                f'border-radius:6px;padding:8px 12px;margin-bottom:6px;">'
                f'<b style="color:#e2e2e8;">{STATION_DISPLAY[row.name]}</b>'
                f'{"  ← selected" if is_selected else ""}<br>'
                f'<span style="color:{color};font-size:11px;">{label}</span>'
                f'<span style="color:#888;font-size:11px;"> · avg {mean_pm25:.0f} µg/m³</span>'
                f'</div>',
                unsafe_allow_html=True,
            )

    # Cluster summary stats
    st.markdown("**Cluster summary statistics**")
    cluster_stats = (
        feat_daily.copy()
        .assign(cluster=feat_daily["station"].map(
            dict(zip(monthly.index, monthly["cluster"]))))
        .groupby("cluster")["pm25"]
        .agg(["mean","median","std","min","max"])
        .round(1)
    )
    cluster_stats.index = [list(CLUSTER_LABELS.values())[i] for i in cluster_stats.index]
    st.dataframe(cluster_stats, use_container_width=True)