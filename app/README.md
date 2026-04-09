---
title: Delhi Air Quality Forecasting
emoji: 🌫️
colorFrom: purple
colorTo: blue
sdk: docker
pinned: true
license: mit
short_description: PM2.5 forecasting using PatchTST & XGBoost
---

# Delhi Air Quality Forecasting Dashboard

**IITK AML Course Project** — End-to-end air quality forecasting system for 9 CPCB monitoring stations in Delhi (2021–2025).

## Features

| Tab | Description |
|-----|-------------|
| 📈 Forecast | 24h hourly or 7-day daily PM2.5 forecast with AQI badge and alert system |
| 🔍 What-If | Interactive SHAP-based scenario analysis — adjust wind, temperature, humidity |
| 🚨 Anomaly | Autoencoder reconstruction error timeline with Diwali markers |
| 🗺️ Clusters | k-means station grouping by seasonal pollution profile |

## Models

- **Hourly:** PatchTST Transformer (MAE 30.4 µg/m³, R² 0.821)
- **Daily:** XGBoost global model (MAE 33.6 µg/m³, R² 0.570)
- **Intervals:** Conformal prediction (91% coverage) + Quantile regression (adaptive)

## Stations Covered

Anand Vihar · Ashok Vihar · Bawana · Dwarka-Sector 8 · Jahangirpuri · Mundka · Punjabi Bagh · Rohini · Wazirpur

## Data Source

[CPCB Delhi Monitoring Network](https://cpcb.nic.in) — 25 parameters including PM2.5, PM10, NOx, SO2, CO, Ozone, and meteorological variables (2021–2025).

## Project Structure

```
app/
  dashboard.py          ← Streamlit app (this file runs in the container)
dataset/
  features/             ← Preprocessed parquet files + metadata JSON
  models/               ← Loaded from HF model repo at runtime
Dockerfile              ← HF Spaces Docker SDK config
requirements.txt
```

## Local Development

```bash
# Clone and install
git clone https://huggingface.co/spaces/your-username/delhi-aq-dashboard
pip install -r requirements.txt

# Run locally
streamlit run app/dashboard.py
```

## Citation

If you use this project, please cite:

```
Delhi Air Quality Forecasting (2025)
IITK MTech AML Course Project
Data: CPCB National Ambient Air Quality Monitoring Programme
```
