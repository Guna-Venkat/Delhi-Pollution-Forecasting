"""
save_dashboard_artifacts.py
────────────────────────────────────────────────────────────────
Run this ONCE after 06_evaluation.ipynb completes.
It saves the two additional artifacts the dashboard needs:
  1. conformal_q_daily.pkl  — the conformal prediction score
  2. autoencoder.pt         — autoencoder state dict

Then upload ALL files in dataset/models/ to your HF model repo.
────────────────────────────────────────────────────────────────
Usage:
    python save_dashboard_artifacts.py
"""

import torch, joblib
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler
import xgboost as xgb

BASE_DIR     = Path(r'C:\Users\gunav\Downloads\Mtech_2025_Admission\IITK\MTech\Sem2\AML\Project\dataset')
FEATURES_DIR = BASE_DIR / 'features'
MODEL_DIR    = BASE_DIR / 'models'
MODEL_DIR.mkdir(exist_ok=True)

# ── 1. Save conformal q score ─────────────────────────────────────────────────
# Recompute from the calibration set (2023 validation year)
feat_daily = pd.read_parquet(FEATURES_DIR / 'features_daily.parquet')
import json
with open(FEATURES_DIR / 'feature_meta_daily.json') as f:
    meta_daily = json.load(f)

val_daily  = feat_daily[feat_daily.index.year == 2023]
xgb_daily  = xgb.XGBRegressor()
xgb_daily.load_model(str(MODEL_DIR / 'xgb_global_daily_4yr.json'))

y_vl   = val_daily['pm25_target'].values
y_pred = xgb_daily.predict(val_daily[meta_daily['global_features']].values)
cal_resid = np.abs(y_vl - y_pred)

alpha   = 0.10
n       = len(cal_resid)
q_level = np.ceil((1-alpha)*(n+1)) / n
q_hat   = float(np.quantile(cal_resid, min(q_level, 1.0)))

joblib.dump(q_hat, MODEL_DIR / 'conformal_q_daily.pkl')
print(f'Saved conformal_q_daily.pkl  (q_hat = {q_hat:.2f})')

# ── 2. Retrain and save autoencoder state dict ────────────────────────────────
POLLUTANT_COLS = ['pm25','pm10','no','no2','nox','nh3','so2','co','ozone',
                  'temp','humidity','wind_speed','wind_dir','solar_rad','baro_pressure']
ae_cols = [c for c in POLLUTANT_COLS if c in feat_daily.columns]
ae_data = feat_daily[ae_cols].dropna()
normal  = ae_data[ae_data.index.month.isin(range(3,10))].values

sc  = StandardScaler().fit(normal)
X_t = torch.tensor(sc.transform(normal), dtype=torch.float32)

import torch.nn as nn, torch.optim as optim

class Autoencoder(nn.Module):
    def __init__(self, n_features, latent_dim=4):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(n_features, 32), nn.ReLU(),
            nn.Linear(32, 16),         nn.ReLU(),
            nn.Linear(16, latent_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 16), nn.ReLU(),
            nn.Linear(16, 32),         nn.ReLU(),
            nn.Linear(32, n_features),
        )
    def forward(self, x): return self.decoder(self.encoder(x))

ae  = Autoencoder(len(ae_cols))
opt = optim.Adam(ae.parameters(), lr=1e-3)
crit= nn.MSELoss()
for epoch in range(1, 101):
    ae.train(); opt.zero_grad()
    loss = crit(ae(X_t), X_t)
    loss.backward(); opt.step()
    if epoch % 25 == 0:
        print(f'  AE epoch {epoch}  loss={loss.item():.5f}')

torch.save(ae.state_dict(), MODEL_DIR / 'autoencoder.pt')
print('Saved autoencoder.pt')

# ── 3. List all files to upload to HF ─────────────────────────────────────────
print('\nFiles to upload to HF model repo:')
for f in sorted(MODEL_DIR.glob('*.json')) + \
         sorted(MODEL_DIR.glob('*.pkl'))  + \
         sorted(MODEL_DIR.glob('*.pt')):
    size_mb = f.stat().st_size / 1e6
    print(f'  {f.name:<45}  {size_mb:.1f} MB')

print('\nUpload command:')
print(f'  huggingface-cli upload your-hf-username/delhi-aq-models {MODEL_DIR}/ .')
