"""
app/dashboard.py
─────────────────────────────────────────────────────────────────
Delhi Air Quality Forecasting Dashboard
Streamlit app — dark theme — Dual Mode:
  1. 🔬 Lab Results: Research dashboard with 4 tabs (Forecast, What-If, Anomaly, Clusters)
  2. 🔮 Try yourself: Consumer weather-style interactive explorer
─────────────────────────────────────────────────────────────────
"""

import json
import datetime
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

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Delhi AQ Forecast",
    page_icon="🌫️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Dark theme CSS ────────────────────────────────────────────────────────────
st.markdown("""
<style>
  .stApp { background-color: #0f0f13; color: #e2e2e8; }
  section[data-testid="stSidebar"] { background-color: #12121a; }
  div[data-testid="metric-container"] {
    background: #16161f; border: 1px solid #2a2a40; border-radius: 10px; padding: 12px 16px;
  }
  div[data-testid="metric-container"] label { color: #888 !important; }
  div[data-testid="metric-container"] div[data-testid="stMetricValue"] {
    color: #c9b8ff !important; font-size: 1.6rem !important;
  }
  button[data-baseweb="tab"] { color: #aaa !important; }
  button[data-baseweb="tab"][aria-selected="true"] {
    color: #c9b8ff !important; border-bottom: 2px solid #8b7cf8 !important;
  }
  hr { border-color: #2a2a40; }
</style>
""", unsafe_allow_html=True)

# ── Constants ─────────────────────────────────────────────────────────────────
HF_REPO_ID = "Guna-Venkat-Doddi-251140009/delhi-aq-models"
STATIONS = [
    "Ashok_Vihar", "Anand_Vihar", "Bawana", "Dwarka-Sector_8",
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
}
DIWALI_DATES = pd.to_datetime(["2021-11-04", "2022-10-24", "2023-11-12", "2024-11-01"])

# ── Helpers ───────────────────────────────────────────────────────────────────
def compute_aqi(pm25: float) -> float:
    if np.isnan(pm25): return float("nan")
    for c_lo, c_hi, i_lo, i_hi in AQI_BREAKPOINTS["pm25"]:
        if c_lo <= pm25 <= c_hi:
            return round(((i_hi-i_lo)/(c_hi-c_lo))*(pm25-c_lo)+i_lo, 1)
    return 500.0 if pm25 > AQI_BREAKPOINTS["pm25"][-1][1] else float("nan")

def aqi_category(aqi_val: float) -> tuple:
    if np.isnan(aqi_val): return ("Unknown", "#888")
    for lo, hi, label, color in AQI_CATEGORIES:
        if lo <= aqi_val <= hi: return label, color
    return ("Severe", "#7B1FA2")

def aqi_badge(aqi_val: float) -> str:
    label, color = aqi_category(aqi_val)
    return f'<div style="display:inline-block;background:{color};color:white;padding:6px 18px;border-radius:20px;font-weight:600;font-size:1rem;">AQI {aqi_val:.0f} — {label}</div>'

class Autoencoder(torch.nn.Module):
    def __init__(self, n_features, latent_dim=4):
        super().__init__()
        self.encoder = torch.nn.Sequential(torch.nn.Linear(n_features, 32), torch.nn.ReLU(), torch.nn.Linear(32, 16), torch.nn.ReLU(), torch.nn.Linear(16, latent_dim))
        self.decoder = torch.nn.Sequential(torch.nn.Linear(latent_dim, 16), torch.nn.ReLU(), torch.nn.Linear(16, 32), torch.nn.ReLU(), torch.nn.Linear(32, n_features))
    def forward(self, x): return self.decoder(self.encoder(x))

# ── Data & Model Loading ──────────────────────────────────────────────────────
@st.cache_resource
def load_models():
    models = {}
    def _load(name, hub_name, type="xgb"):
        try: path = hf_hub_download(repo_id=HF_REPO_ID, filename=hub_name)
        except: path = str(BASE_DIR / "dataset" / "models" / hub_name)
        if type == "xgb": 
            m = xgb.XGBRegressor(); m.load_model(path); return m
        elif type == "pt": return torch.load(path, map_location="cpu")
        else: return joblib.load(path)
    
    models["xgb_1hr"] = _load("xgb_1hr", "xgb_global_1hr_4yr.json")
    models["xgb_daily"] = _load("xgb_daily", "xgb_global_daily_4yr.json")
    models["xgb_q05_daily"] = _load("q05", "xgb_daily_q05.pkl", type="pkl")
    models["xgb_q95_daily"] = _load("q95", "xgb_daily_q95.pkl", type="pkl")
    models["conformal_q"] = _load("conf", "conformal_q_daily.pkl", type="pkl")
    models["autoencoder_state"] = _load("ae", "autoencoder.pt", type="pt")
    models["autoencoder_scaler"] = _load("ae_sc", "autoencoder_scaler.pkl", type="pkl")
    models["autoencoder_features"] = _load("ae_ft", "autoencoder_features.pkl", type="pkl")
    return models

@st.cache_data
def load_data():
    BASE_DIR = Path(__file__).parent.parent
    feat_1hr = pd.read_parquet(BASE_DIR / "dataset" / "features" / "features_1hr.parquet")
    feat_daily = pd.read_parquet(BASE_DIR / "dataset" / "features" / "features_daily.parquet")
    with open(BASE_DIR / "dataset" / "features" / "feature_meta_1hr.json") as f: m1 = json.load(f)
    with open(BASE_DIR / "dataset" / "features" / "feature_meta_daily.json") as f: md = json.load(f)
    return feat_1hr, feat_daily, m1, md

BASE_DIR = Path(__file__).parent.parent
feat_1hr, feat_daily, meta_1hr, meta_daily = load_data()
models = load_models()
data_min, data_max = feat_daily.index.min().date(), feat_daily.index.max().date()

@st.cache_data
def get_climatology(df, features):
    df_clim = df.copy(); df_clim["doy"] = df_clim.index.dayofyear
    return df_clim.groupby("doy")[features].median()

climatology_map = get_climatology(feat_daily, meta_daily["global_features"])

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🌫️ Delhi AQ Control")
    mode = st.radio("⚡ View Mode", ["🔬 Lab Results", "🔮 Try yourself"])
    station = st.selectbox("📍 Station", STATIONS, format_func=lambda s: STATION_DISPLAY[s])
    freq = st.radio("⏱ Frequency", ["Hourly (24h)", "Daily (7 days)"], horizontal=True)
    is_hourly = "Hourly" in freq
    
    if mode == "🔮 Try yourself":
        selected_date = st.date_input("📅 Target Date", value=data_max, min_value=data_min, max_value=datetime.date(2026, 12, 31))
        horizon_val = st.slider("Forecast Horizon", 1, 14, 7)
    else:
        selected_date, horizon_val = data_max, 7
    
    if mode == "🔬 Lab Results":
        use_conformal = "Conformal" in st.radio("Intervals", ["Conformal", "Quantile"], horizontal=True)
    else:
        use_conformal = True # Default for Try Yourself

# ── Core Logic: Recursive Simulation ──────────────────────────────────────────
feat_df, meta = (feat_1hr, meta_1hr) if is_hourly else (feat_daily, meta_daily)
model = models["xgb_1hr"] if is_hourly else models["xgb_daily"]
horizon = 24 if is_hourly else horizon_val
station_df = feat_df[feat_df["station"] == station].copy()

is_future = mode == "🔮 Try yourself" and selected_date > data_max
if is_future:
    with st.status("🔮 Simulating future conditions...", expanded=False) as status:
        curr_ts = pd.Timestamp(data_max)
        sim_row = station_df.loc[curr_ts].copy()
        sim_history = []
        while curr_ts.date() < selected_date:
            curr_ts += pd.Timedelta("1d"); doy = curr_ts.dayofyear
            clim = climatology_map.loc[doy]
            for f in meta_daily["global_features"]:
                if f in clim: sim_row[f] = clim[f]
            sim_row["month"], sim_row["year"] = curr_ts.month, curr_ts.year
            sim_row["doy_sin"] = np.sin(2 * np.pi * doy / 366)
            sim_row["doy_cos"] = np.cos(2 * np.pi * doy / 366)
            X = sim_row[meta_daily["global_features"]].values.reshape(1, -1)
            next_pm = float(models["xgb_daily"].predict(X)[0])
            sim_row["pm25_lag7"] = sim_row["pm25_lag1"]
            sim_row["pm25_lag1"] = next_pm
            sim_row["pm25"], sim_row["aqi"] = next_pm, compute_aqi(next_pm)
            sim_history.append(sim_row.copy())
        
        sim_df = pd.DataFrame(sim_history, index=pd.date_range(pd.Timestamp(data_max)+pd.Timedelta("1d"), periods=len(sim_history), freq="D"))
        df_context = pd.concat([station_df.tail(60), sim_df]) # Ensure at least 60 days of context
        status.update(label="🔮 Simulation Complete!", state="complete")
else:
    target_ts = pd.Timestamp(selected_date); 
    if is_hourly: target_ts = target_ts.replace(hour=23)
    df_context = station_df[station_df.index <= target_ts] if mode == "🔮 Try yourself" else station_df.copy()

current = df_context.iloc[-1]
current_pm25, current_aqi = current["pm25"], current["aqi"]
X_inf = df_context.tail(48)[meta["global_features"]].values
forecast_vals = model.predict(X_inf)[-horizon:]
forecast_aqi = [compute_aqi(v) for v in forecast_vals]
future_ts = pd.date_range(df_context.index[-1] + pd.Timedelta("1h" if is_hourly else "1d"), periods=horizon, freq="H" if is_hourly else "D")

# ── Main UI ───────────────────────────────────────────────────────────────────
st.title(f"🌫️ {STATION_DISPLAY[station]}")
st.markdown(f"**Context Date:** {df_context.index[-1].strftime('%A, %d %b %Y')}")

if mode == "🔬 Lab Results":
    tab1, tab2, tab3, tab4 = st.tabs(["📈 Forecast", "🔍 What-If Analysis", "🚨 Anomaly Detection", "🗺️ Station Clusters"])
    
    with tab1:
        st.markdown("### 📈 Comprehensive Forecast")
        col_left, col_right = st.columns([3, 1])

        with col_left:
            # Historical + forecast chart
            hist_window = 48 if is_hourly else 30
            hist_df  = df_context.tail(hist_window)
            hist_ts  = hist_df.index
            hist_pm  = hist_df["pm25"].values

            # Prediction intervals (daily only)
            lower, upper = None, None
            interval_name = "None"
            if not is_hourly:
                if use_conformal and models.get("conformal_q") is not None:
                    conf_q = models["conformal_q"]
                    lower = forecast_vals - conf_q
                    upper = forecast_vals + conf_q
                    interval_name = "Conformal (91%)"
                elif not use_conformal and models.get("xgb_q05_daily") is not None:
                    # To calculate intervals correctly, we need the raw X for the horizon
                    X_latest = df_context.tail(max(horizon * 4, 48))[meta["global_features"]].values
                    q_lower = models["xgb_q05_daily"].predict(X_latest)[-horizon:]
                    q_upper = models["xgb_q95_daily"].predict(X_latest)[-horizon:]
                    lower = q_lower
                    upper = q_upper
                    interval_name = "Quantile (adaptive)"

            fig = go.Figure()
            # Historical trace
            fig.add_trace(go.Scatter(x=hist_ts, y=hist_pm, mode="lines", name="Historical", line=dict(color="#4A9EFF", width=1.5)))

            # Prediction interval band (daily only)
            if not is_hourly and lower is not None and upper is not None:
                fig.add_trace(go.Scatter(
                    x=list(future_ts) + list(future_ts[::-1]),
                    y=list(upper) + list(lower[::-1]),
                    fill="toself", fillcolor="rgba(139,124,248,0.15)",
                    line=dict(color="rgba(0,0,0,0)"), name=f"90% PI ({interval_name.split()[0]})")
                )

            # Forecast trace
            fig.add_trace(go.Scatter(x=future_ts, y=forecast_vals, mode="lines+markers", name="Forecast", line=dict(color="#C9B8FF", width=2, dash="dash"), marker=dict(size=5)))

            # AQI threshold lines
            for lo, hi, label, color in AQI_CATEGORIES[2:]:
                fig.add_hline(y=lo, line_dash="dot", line_color=color, line_width=0.8, opacity=0.5,
                               annotation_text=label, annotation_position="right", annotation_font_size=9)

            fig.update_layout(
                title=f"PM2.5 Forecast — {STATION_DISPLAY[station]} ({'24h' if is_hourly else '7d'})",
                paper_bgcolor="#12121a", plot_bgcolor="#16161f", font=dict(color="#e2e2e8"),
                xaxis=dict(gridcolor="#2a2a40", showgrid=True),
                yaxis=dict(gridcolor="#2a2a40", showgrid=True, title="PM2.5 (µg/m³)"),
                height=380, legend=dict(bgcolor="#1a1a2e", bordercolor="#2a2a40")
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

    # ── TAB 2: WHAT-IF ANALYSIS ───────────────────────────────────────────────
    with tab2:
        st.markdown("### 🔍 What-If Scenario Analysis")
        st.markdown(
            "Adjust meteorological inputs to see how the forecast changes. "
            "Based on SHAP-identified feature sensitivities from the daily XGBoost model."
        )

        daily_latest = feat_daily[feat_daily["station"] == station].iloc[-1:].copy()
        base_pm25 = float(daily_latest["pm25"].values[0])

        col_a, col_b, col_c = st.columns(3)
        with col_a:
            ws_val   = st.slider("Wind Speed (m/s)",    0.0, 15.0, float(daily_latest["wind_speed"].values[0]), 0.5)
        with col_b:
            temp_val = st.slider("Temperature (°C)",   -5.0, 50.0, float(daily_latest["temp"].values[0]), 0.5)
        with col_c:
            hum_val  = st.slider("Humidity (%)",        10.0, 100.0, float(daily_latest["humidity"].values[0]), 1.0)

        modified_row = daily_latest[meta_daily["global_features"]].copy()
        for col, val in [("wind_speed", ws_val), ("temp", temp_val), ("humidity", hum_val)]:
            if col in modified_row.columns: modified_row[col] = val
        if "humid_temp_interaction" in modified_row.columns:
            modified_row["humid_temp_interaction"] = hum_val * temp_val
        if "wind_u" in modified_row.columns and "wind_dir" in modified_row.columns:
            wd_rad = np.deg2rad(float(daily_latest["wind_dir"].values[0]))
            modified_row["wind_u"] = -ws_val * np.sin(wd_rad)
            modified_row["wind_v"] = -ws_val * np.cos(wd_rad)

        scenario_pred = float(models["xgb_daily"].predict(modified_row.values)[0])
        scenario_aqi  = compute_aqi(scenario_pred)
        delta = scenario_pred - base_pm25

        col_m1, col_m2, col_m3 = st.columns(3)
        col_m1.metric("Baseline PM2.5",  f"{base_pm25:.1f} µg/m³")
        col_m2.metric("Scenario PM2.5",  f"{scenario_pred:.1f} µg/m³", delta=f"{delta:+.1f}", delta_color="inverse")
        col_m3.metric("Scenario AQI",    f"{scenario_aqi:.0f}", delta=f"{aqi_category(scenario_aqi)[0]}")
        st.markdown(aqi_badge(scenario_aqi), unsafe_allow_html=True)

        st.markdown("#### Sensitivity to each variable")
        fig_wi = go.Figure()
        sweep_configs = [("wind_speed", np.linspace(0, 15, 40), "Wind Speed (m/s)"), 
                        ("temp", np.linspace(-5, 45, 40), "Temperature (°C)"), 
                        ("humidity", np.linspace(10, 100, 40), "Humidity (%)")]
        colors_sweep = ["#4A9EFF", "#FF7043", "#66BB6A"]

        for (feat, vals, xlabel), color in zip(sweep_configs, colors_sweep):
            s_preds = []
            for v in vals:
                rs = modified_row.copy()
                rs[feat] = v
                if feat in ["wind_speed", "humidity", "temp"] and "humid_temp_interaction" in rs.columns:
                    rs["humid_temp_interaction"] = (hum_val if feat != "humidity" else v) * (temp_val if feat != "temp" else v)
                s_preds.append(float(models["xgb_daily"].predict(rs.values)[0]))
            fig_wi.add_trace(go.Scatter(x=vals, y=s_preds, mode="lines", name=xlabel, line=dict(color=color)))

        fig_wi.update_layout(paper_bgcolor="#12121a", plot_bgcolor="#16161f", font=dict(color="#e2e2e8"), height=320)
        st.plotly_chart(fig_wi, use_container_width=True)

    # ── TAB 3: ANOMALY DETECTION ──────────────────────────────────────────────
    with tab3:
        st.markdown("### 🚨 Autoencoder Anomaly Detection")

        if models.get("autoencoder_state") is None or models.get("autoencoder_scaler") is None:
            st.warning("Anomaly detection model unavailable.")
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

            # Prepare data
            ae_data = feat_daily.copy()
            ae_data = ae_data.loc[:, ~ae_data.columns.duplicated()]

            # Extract features and meta
            X_raw = ae_data[ae_cols].copy()
            meta_df = ae_data[["station", "aqi_category"]].copy()

            # Combine, drop rows with NaN
            combined = pd.concat([X_raw, meta_df], axis=1).dropna()
            X_clean = combined[ae_cols].values
            X_scaled = sc_ae.transform(X_clean)

            # Get timestamps
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
                fig_ae.add_vline(x=d, line_dash="dash", line_color="#FFD700", opacity=0.8)
                fig_ae.add_annotation(x=d, y=1, xref="x", yref="paper", text=f"Diwali {d.year}", showarrow=False, font=dict(color="#FFD700", size=9))

            fig_ae.add_hline(y=thresh, line_dash="dot", line_color="#E53935", annotation_text="Threshold")
            fig_ae.update_layout(paper_bgcolor="#12121a", plot_bgcolor="#16161f", font=dict(color="#e2e2e8"), height=350, title=f"Anomaly Timeline — {STATION_DISPLAY[station]}")
            st.plotly_chart(fig_ae, use_container_width=True)

            st.markdown("**Top 15 most anomalous days**")
            pm25_vals = feat_daily[feat_daily["station"] == station]["pm25"]
            pm25_daily = pm25_vals.reindex(station_ae.index).resample("D").mean()
            top_df = station_ae.groupby(station_ae.index.date).agg(recon_error=("recon_error", "mean"), aqi_category=("aqi_category", "first")).reset_index()
            top_df["Mean PM2.5"] = top_df["index"].map(pm25_daily)
            top_df = top_df.sort_values("recon_error", ascending=False).head(15).rename(columns={"index": "Date"})
            st.dataframe(top_df[["Date", "recon_error", "Mean PM2.5", "aqi_category"]], use_container_width=True)

    # ── TAB 4: STATION CLUSTERS ───────────────────────────────────────────────
    with tab4:
        st.markdown("### 🗺️ Station Cluster Analysis")
        st.markdown("Stations grouped by their seasonal PM2.5 profile using k-means (k=3).")

        monthly = feat_daily.assign(month=lambda d: d.index.month).groupby(["station","month"])["pm25"].mean().unstack("month")
        sc_km = StandardScaler(); X_km = sc_km.fit_transform(monthly.values)
        km = KMeans(n_clusters=3, random_state=42, n_init=10); clusters = km.fit_predict(X_km)
        monthly["cluster"] = clusters
        
        cluster_means = {c: monthly[monthly["cluster"]==c].drop("cluster", axis=1).values.mean() for c in range(3)}
        sorted_cl = sorted(cluster_means, key=cluster_means.get, reverse=True)
        rename_map = {old: new for new, old in enumerate(sorted_cl)}
        monthly["cluster"] = monthly["cluster"].map(rename_map)

        CLUSTER_LABELS = {0: "High pollution", 1: "Moderate", 2: "Low pollution"}
        CLUSTER_COLORS = {0: "#E53935", 1: "#FB8C00", 2: "#43A047"}

        c_left4, c_right4 = st.columns([2, 1])
        with c_left4:
            fig_cl = go.Figure()
            month_names = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
            for name, row in monthly.iterrows():
                cid = int(row["cluster"])
                fig_cl.add_trace(go.Scatter(x=month_names, y=row.drop("cluster").values, mode="lines+markers", 
                                            name=STATION_DISPLAY[name], line=dict(color=CLUSTER_COLORS[cid], width=2),
                                            marker=dict(size=5), legendgroup=f"c{cid}",
                                            legendgrouptitle_text=CLUSTER_LABELS[cid] if name == monthly[monthly["cluster"]==cid].index[0] else None))
            fig_cl.update_layout(paper_bgcolor="#12121a", plot_bgcolor="#16161f", font=dict(color="#e2e2e8"), height=380, title="Monthly PM2.5 Profile by Cluster")
            st.plotly_chart(fig_cl, use_container_width=True)

        with c_right4:
            st.markdown("**Cluster Assignments**")
            for name, row in monthly.sort_values("cluster").iterrows():
                cid = int(row["cluster"])
                is_sel = name == station
                border = "2px solid #8b7cf8" if is_sel else "1px solid #2a2a40"
                st.markdown(f'<div style="background:#16161f;border:{border};border-left:4px solid {CLUSTER_COLORS[cid]};border-radius:6px;padding:8px 12px;margin-bottom:6px;">'
                            f'<b style="color:#e2e2e8;">{STATION_DISPLAY[name]}</b><br>'
                            f'<span style="color:{CLUSTER_COLORS[cid]};font-size:11px;">{CLUSTER_LABELS[cid]}</span></div>', unsafe_allow_html=True)
        
        st.markdown("**Cluster Statistics**")
        stats = feat_daily.assign(cluster=feat_daily["station"].map(dict(zip(monthly.index, monthly["cluster"])))).groupby("cluster")["pm25"].agg(["mean","median","std"]).round(1)
        stats.index = [CLUSTER_LABELS[i] for i in stats.index]
        st.dataframe(stats, use_container_width=True)

else:
    # 🔮 Try yourself Mode
    st.markdown("### 🔮 Predicted Profile")
    if is_future: st.info(f"Simulating target date {selected_date}...")
    
    # Hero metric
    h1, h2 = st.columns([1, 2])
    h1.metric("Predicted AQI", f"{forecast_aqi[0]:.0f}", delta=f"{forecast_aqi[0]-current_aqi:+.1f}", delta_color="inverse")
    h2.markdown(aqi_badge(forecast_aqi[0]), unsafe_allow_html=True)
    
    # Weather cards
    st.markdown("---")
    n_display = min(len(forecast_aqi), 12)
    cols = st.columns(n_display)
    for i in range(n_display):
        ts, v = future_ts[i], forecast_aqi[i]; lbl, clr = aqi_category(v)
        cols[i].markdown(f'<div style="background:#16161f;border-top:4px solid {clr};padding:8px;text-align:center;border-radius:8px;"><small>{ts.strftime("%H:%M" if is_hourly else "%a %d")}</small><br><b style="color:{clr};font-size:1.2rem;">{v:.0f}</b></div>', unsafe_allow_html=True)
    
    st.markdown("---")
    fig_try = go.Figure()
    for lo, hi, lbl, clr in AQI_CATEGORIES:
        fig_try.add_hrect(y0=lo, y1=hi if hi < 500 else max(forecast_vals)+50, fillcolor=clr, opacity=0.05, line_width=0)
    fig_try.add_trace(go.Scatter(x=df_context.tail(30).index, y=df_context.tail(30)["pm25"], name="History", line=dict(color="#4A9EFF", width=2)))
    fig_try.add_trace(go.Scatter(x=future_ts, y=forecast_vals, name="Forecast", line=dict(color="#C9B8FF", width=3, dash="dot")))
    fig_try.update_layout(paper_bgcolor="#0f0f13", plot_bgcolor="#0f0f13", font=dict(color="#e2e2e8"), height=500, margin=dict(t=30))
    st.plotly_chart(fig_try, use_container_width=True)
    st.stop()