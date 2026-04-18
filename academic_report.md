# Spatiotemporal Air Quality Forecasting in Delhi: A Multi-Model Approach with Uncertainty Estimation and Explainability

**Course**: DMS673: Applied Machine Learning  
**Institution**: Indian Institute of Technology (IIT), Kanpur  
**Author**: Guna Venkat Doddi (Roll No. 251140009)

---

## Abstract
Delhi periodically experiences some of the highest levels of particulate matter (PM2.5) globally, driven by a complex interplay of meteorological conditions, local emissions, and seasonal biomass burning. This project presents an end-to-end machine learning pipeline to forecast Delhi's AQI with high precision and reliable uncertainty bounds. We systematically compare traditional statistical baselines (ARIMA/SARIMA) with modern Gradient Boosted Trees (XGBoost) and state-of-the-art Deep Learning architectures (LSTM, Informer, PatchTST Transformers). On the held-out 2024–2025 test set, PatchTST achieves the best hourly performance (MAE 30.42 µg/m³, R²=0.821), while Global XGBoost leads on daily aggregates (MAE 33.62 µg/m³, R²=0.570). **Hierarchical seasonal conformal prediction** yields an empirical overall coverage of 90.6%, correcting the flat calibration's 14.5 ppt winter under-coverage deficit. A systematic **failure mode analysis** identifies three dominant error regimes: Diwali proximity (±7 days), calm-wind conditions (<1 m/s), and winter season — with quantified heteroscedasticity ratio of **3.28×** between Winter and Summer MAE. SHAP is used for model explainability. Finally, we explore unsupervised manifolds using Autoencoders for anomaly detection and K-Means for station-wise behavioural clustering.

---

## 1. Introduction
Air pollution in the National Capital Territory (NCT) of Delhi is a public health crisis. The primary pollutant of concern is PM2.5 (Particulate Matter < 2.5 microns), which penetrates deep into the respiratory system.

### 1.1 Problem Statement
The objective is to build a system capable of:
1. **Forecasting**: Predicting PM2.5 and AQI levels for specified horizons (1 hour to 1 week).
2. **Simulation**: Providing a "What-If" engine to understand the impact of meteorological shifts (Wind, Temp, Humidity).
3. **Governance**: Identifying anomalies (e.g., specific firecracker spikes) and clustering monitoring stations based on seasonal profiles.

### 1.2 Dataset
The primary dataset consists of multi-station air quality data sourced from CPCB (Central Pollution Control Board). The raw data spans from 2021 to early 2025, covering 9 major monitoring stations: Anand Vihar, Ashok Vihar, Bawana, Dwarka-Sector 8, Jahangirpuri, Mundka, Punjabi Bagh, Rohini, and Wazirpur. Raw data is recorded at 15-minute intervals for 15 pollutant and meteorological variables per station.

---

## 2. Data Acquisition & Harmonization
Data from 9 distinct sources were combined into a unified temporal framework, yielding ~393,735 hourly and ~16,096 daily records across all stations.

### 2.1 Preprocessing Pipeline ([01_combine_data.ipynb](file:///c:/Users/gunav/Downloads/Mtech_2025_Admission/IITK/MTech/Sem2/AML/Project/code/01_combine_data.ipynb), [02_preprocess.ipynb](file:///c:/Users/gunav/Downloads/Mtech_2025_Admission/IITK/MTech/Sem2/AML/Project/code/02_preprocess.ipynb))
-   **Frequency Alignment**: CPCB data provided at 15-minute intervals was resampled to **Hourly** and **Daily** frequencies using mean aggregation.
-   **Timestamp Fixes**: Corrected a known issue where 2023-2024 data had inconsistent datetime formatting, ensuring a continuous time-series.
-   **KNN Imputation**: Missing values (caused by sensor downtime) were imputed using K-Nearest Neighbors with inverse-distance weighting.
    -   **Mathematics**: For a missing value $x_i$, let $N_k(x_i)$ be the set of $k$ nearest neighbors in the feature space. The imputed value is:
        $$\hat{x}_i = \frac{\sum_{j \in N_k(x_i)} w_j x_j}{\sum_{j \in N_k(x_i)} w_j}$$
        where $w_j = \frac{1}{dist(x_i, x_j)}$.
-   **Outlier Treatment**: Applied a 99th-percentile clip *per station* to preserve signal while preventing model explosion during training on extreme spikes (e.g., peak Diwali nights exceeding 800 µg/m³).

---

## 3. Exploratory Data Analysis (EDA)
Comprehensive EDA was performed to validate the "physics" of the data. ([03_EDA.ipynb](file:///c:/Users/gunav/Downloads/Mtech_2025_Admission/IITK/MTech/Sem2/AML/Project/code/03_EDA.ipynb))

### 3.1 Seasonal Patterns
Analysis revealed a massive 3x ratio between Winter (Nov-Jan) and Summer (May-Jul) PM2.5 levels.

![Monthly Boxplots](/c:/Users/gunav/Downloads/Mtech_2025_Admission/IITK/MTech/Sem2/AML/Project/code/plots/03_monthly_boxplots_all_stations.png)

### 3.2 Diurnal Cycles
The diurnal cycle shows two distinct peaks (morning and late evening), corresponding to vehicular traffic and atmospheric boundary layer collapse (inversion).

![Diurnal Cycle](/c:/Users/gunav/Downloads/Mtech_2025_Admission/IITK/MTech/Sem2/AML/Project/code/plots/05_diurnal_cycle.png)

### 3.3 Event Spikes: Diwali and Stubble Burning
We quantified the impact of Diwali by comparing a 10-day window around the event across four years. The residual analysis shows that even after accounting for seasonality, Diwali causes a massive local variance spike 2–5× the winter seasonal baseline.

![Diwali Event Windows](/c:/Users/gunav/Downloads/Mtech_2025_Admission/IITK/MTech/Sem2/AML/Project/code/plots/06_diwali_event_windows.png)

### 3.4 Spatial Consistency
The stations are highly correlated ($r > 0.85$), suggesting that while local point sources exist, the pollution crisis is city-wide. This justifies the use of a global model pooling all stations.

![Correlation Heatmap](/c:/Users/gunav/Downloads/Mtech_2025_Admission/IITK/MTech/Sem2/AML/Project/code/plots/07_correlation_heatmap.png)

---

## 4. Feature Engineering ([04_feature_engineering.ipynb](file:///c:/Users/gunav/Downloads/Mtech_2025_Admission/IITK/MTech/Sem2/AML/Project/code/04_feature_engineering.ipynb))
To transform tabular data into a sequence-aware format, we engineered the following:

-   **Lagged Features**: Daily model uses $PM2.5_{t-1}, PM2.5_{t-2}, PM2.5_{t-3}, PM2.5_{t-7}, PM2.5_{t-14}, PM2.5_{t-30}$. Hourly model uses lags at 1, 2, 3, 6, 12, 24, 48 hours. Lags were selected based on PACF drops.
-   **Rolling Windows**: Mean, Standard Deviation, and Maximum over 3, 7, 14, and 30-day windows to capture multi-scale volatility.
-   **Cyclical Encoding**: To prevent the model from seeing December (12) and January (1) as distant, we used trigonometric transformations:
    $$Month_{sin} = \sin\left(\frac{2\pi \cdot m}{12}\right), Month_{cos} = \cos\left(\frac{2\pi \cdot m}{12}\right)$$
-   **Meteorological Decomposition**: Wind speed and direction were decomposed into $U$ (zonal) and $V$ (meridional) vectors to capture smoke transport from the NW (Punjab) during stubble season:
    $$U = -ws \cdot \sin(\theta), \quad V = -ws \cdot \cos(\theta)$$
-   **Event Flags**: A `days_since_diwali` countdown feature (capped at 10 days) was added to help the model learn the post-event decay. A binary `is_stubble_season` flag covers Oct 1–Nov 30.
-   **Interaction Feature**: `humid_temp_interaction` = humidity × temperature, capturing hygroscopic PM2.5 growth under cold, humid winter inversions.

---

## 5. Methodology ([05-modeling.ipynb](file:///c:/Users/gunav/Downloads/Mtech_2025_Admission/IITK/MTech/Sem2/AML/Project/code/05-modeling.ipynb))

### 5.1 Baseline: Statistical Models
-   **ARIMA/SARIMA**: Trained per-station. While capturing seasonality, these models failed catastrophically on non-linear events (R² < 0 on daily data). SARIMA's assumption of Gaussian, stationary residuals is violated by Diwali structural breaks.

### 5.2 Global Machine Learning: XGBoost
We implemented a **Global XGBoost** model where station identity is a one-hot encoded feature. This allows the model to learn city-wide dynamics while adjusting the intercept for specific stations. A per-station XGBoost was also trained for comparison — the global model consistently outperformed it.

#### **Mathematics of XGBoost**
XGBoost minimizes a regularized objective function:
$$Obj(\Theta) = \sum_i l(y_i, \hat{y}_i) + \sum_k \Omega(f_k)$$
Using a second-order Taylor expansion:
$$L^{(t)} \approx \sum_{i=1}^n [l(y_i, \hat{y}_i^{(t-1)}) + g_i f_t(x_i) + \frac{1}{2} h_i f_t^2(x_i)] + \Omega(f_t)$$
where $g_i = \partial_{\hat{y}^{(t-1)}} l(y_i, \hat{y}_i^{(t-1)})$ and $h_i = \partial^2_{\hat{y}^{(t-1)}} l(y_i, \hat{y}_i^{(t-1)})$.

### 5.3 LSTM (Recurrent Baseline)
A single-layer LSTM (hidden dim 64) was trained as a recurrent neural network baseline. LSTM performed reasonably on hourly data (4yr MAE=38.19, R²=0.725) but **failed severely on daily aggregates (R²=−0.884)**. This is an important negative result: with only ~1,460 daily training steps over 4 years, LSTM cannot learn long seasonal dependencies through gradient flow. XGBoost overcomes this through explicit lag features and calendar encodings — demonstrating that domain-aware feature engineering can substitute for recurrent inductive bias when sequence length is constrained.

### 5.4 Deep Learning: PatchTST (State of the Art)
For multi-step forecasting, we utilized **PatchTST** [Nie et al., 2023, ICLR]. Unlike standard LSTMs that process point-wise, PatchTST segments the time series into patches of length $P$.
-   **Benefits**: Reduces memory complexity from $O(L^2)$ to $O((L/P)^2)$ and preserves local semantic context.
-   **Channel Independence**: Each station is treated as an independent channel, allowing the shared backbone to learn common temporal filters.

### 5.5 Informer
Informer [Zhou et al., 2021] uses ProbSparse attention to reduce Transformer complexity. Competitive on hourly data (4yr MAE=32.48, R²=0.777) but unstable on daily data, supporting the data-volume threshold finding.

---

## 6. Performance Evaluation ([06_evaluation.ipynb](file:///c:/Users/gunav/Downloads/Mtech_2025_Admission/IITK/MTech/Sem2/AML/Project/code/06_evaluation.ipynb))

### 6.1 Benchmarking Results
The data-scaling experiment (1yr → 4yr training) tested on 2024–2025 data yielded the following (4yr results shown):

| Model | Resolution | MAE (µg/m³) | RMSE | R² | Notes |
|---|---|---|---|---|---|
| Persistence | Daily | 68.2 | — | 0.42 | Naive Baseline |
| SARIMA | Daily | 111.22 | 124.64 | −0.717 | Stationarity assumption violated |
| **XGBoost (Global)** | **Daily** | **33.62** | **63.06** | **0.570** | **Best daily model** |
| XGBoost (Per-stn) | Daily | 34.03 | 64.68 | 0.548 | Global > per-station |
| LSTM | Daily | 89.67 | 130.35 | −0.884 | Insufficient sequence length |
| Informer | Daily | 52.15 | 90.42 | 0.093 | Unstable on daily aggs |
| PatchTST | Daily | 44.16 | 73.22 | 0.406 | Better than LSTM/Informer |
| XGBoost (Global) | Hourly | 39.47 | 65.42 | 0.652 | Strong short-term baseline |
| LSTM | Hourly | 38.19 | 60.17 | 0.725 | Recovers well on hourly |
| Informer | Hourly | 32.48 | 54.11 | 0.777 | ProbSparse helpful here |
| **PatchTST** | **Hourly** | **30.42** | **48.47** | **0.821** | **Best overall model** |

![Learning Curve](/c:/Users/gunav/Downloads/Mtech_2025_Admission/IITK/MTech/Sem2/AML/Project/code/plots/learning_curve_all_models.png)

### 6.2 Key Observations
- **PatchTST wins on hourly data**: Patch-level tokenisation captures the 24-hour diurnal cycle structure naturally.
- **XGBoost wins on daily data**: Explicit lag features and event flags compensate for shorter sequence length.
- **LSTM daily failure (R²=−0.884)** is an architectural mismatch, not a bug: with ~1,460 daily training steps, recurrent gradient flow cannot learn seasonal structure that XGBoost captures through designed features.
- **Global XGBoost consistently beats per-station** by 1–2 MAE units, confirming positive cross-station transfer learning.

### 6.3 Residual Analysis
Residuals are heteroscedastic: errors are larger during peak winter episodes. This justifies the use of uncertainty estimation rather than single-point predictions.

![Residual Analysis](/c:/Users/gunav/Downloads/Mtech_2025_Admission/IITK/MTech/Sem2/AML/Project/code/plots/eval_residuals.png)

---

## 7. Explainability & Uncertainty

### 7.1 SHAP Explainability
Global SHAP importance highlights:
- `pm25_lag1` (persistence) is the dominant driver — high lagged PM2.5 strongly predicts continued high levels.
- `month_cos` and `is_winter` drive seasonal variation.
- `wind_speed` and `wind_u` (zonal) are the most important meteorological variables — north-westerly winds transport stubble smoke from Punjab.
- `days_since_diwali` has non-zero SHAP values, confirming the event feature adds signal beyond seasonality.

For the simulation engine, SHAP sensitivity sweeps allow us to ask: *"What if the wind speed doubles tomorrow?"*

![SHAP Summary](/c:/Users/gunav/Downloads/Mtech_2025_Admission/IITK/MTech/Sem2/AML/Project/code/plots/shap_daily_summary.png)

### 7.2 Conformal Prediction
To ensure safety in public health warnings, we implemented **Split Conformal Prediction**.
1.  **Calibration**: Calculate non-conformity scores $s_i = |y_i - \hat{y}_i|$ on the 2023 hold-out set ($n=3209$ records).
2.  **Quantile**: Find the corrected quantile $\hat{q} = \text{Quantile}(s_1,...,s_n; \frac{\lceil(1-\alpha)(n+1)\rceil}{n})$.
3.  **Interval**: Prediction Interval = $[\hat{y}_{test} - \hat{q},\ \hat{y}_{test} + \hat{q}]$.

This provides a **distribution-free theoretical guarantee** that $\geq 90\%$ of future observations will fall within this band, regardless of error distribution — critical when Gaussian assumptions fail on heavy-tailed pollution data.

![Prediction Intervals](/c:/Users/gunav/Downloads/Mtech_2025_Admission/IITK/MTech/Sem2/AML/Project/code/plots/prediction_intervals.png)

---

## 8. Unsupervised Domain Insights

### 8.1 Station Clustering
K-Means (*k*=3, random_state=42) was applied to the 12-dimensional monthly-mean PM2.5 profiles of all stations (standardized). The choice of *k*=3 was validated using both the **elbow method** (inertia plateaus at *k*=3) and a **silhouette score of 0.61** (scale: −1 to +1), confirming well-separated, meaningful clusters. Three physically meaningful clusters emerged:
-   **Cluster 1 (High Pollution — Transport Hubs)**: Anand Vihar / Mundka. High winter baseline (>250 µg/m³), slow clearance. Proximity to major arterials and industrial estates.
-   **Cluster 2 (Moderate — Mixed Urban)**: Ashok Vihar, Bawana, Jahangirpuri, Rohini, Wazirpur, Punjabi Bagh. Mid-range winter levels.
-   **Cluster 3 (Low Pollution — Residential/Windward)**: Dwarka-Sector 8. Lowest baseline, fastest clearance under westerly winds.

![K-Means Clusters](/c:/Users/gunav/Downloads/Mtech_2025_Admission/IITK/MTech/Sem2/AML/Project/code/plots/unsup_kmeans_clusters.png)

### 8.2 Anomaly Detection with Autoencoders
We trained a symmetric Autoencoder (15→32→16→4→16→32→15) on **clean summer data** (Mar–Sep) as the "normal" regime. The Reconstruction Error $E$ is used as an anomaly score:
$$E = \|x - f_{dec}(f_{enc}(x))\|^2$$
Spikes in $E$ perfectly align with all four Diwali dates (2021: Nov 4, 2022: Oct 24, 2023: Nov 12, 2024: Nov 1) and October stubble-burning windows — without any event labelling — confirming that these episodes represent a genuine shift in the underlying pollution physics (human intervention vs. natural atmospheric cycles).

![Autoencoder Anomalies](/c:/Users/gunav/Downloads/Mtech_2025_Admission/IITK/MTech/Sem2/AML/Project/code/plots/unsup_autoencoder_anomalies.png)

---

## 9. Productization: Delhi AQ Dashboard
The final pipeline was deployed via a Streamlit Dashboard with dual modes:
- **Lab Results** (research mode): 4-tab interface — Forecast, What-If Analysis, Anomaly Detection, Station Clusters.
- **Try Yourself** (consumer mode): Weather-app-style AQI health-risk cards for any future date via recursive climatology simulation, including animated AQI gauge and 7-day outlook.

The recursive simulation engine correctly shifts all lag features (pm25_lag1 through pm25_lag30, aqi lags, and all rolling windows) at each simulated time step. Models are hosted on Hugging Face Hub for lightweight deployment. The application is Docker-containerised (Dockerfile SDK, port 7860) for portability.

**🚀 Live Demo:** [https://huggingface.co/spaces/Guna-Venkat-Doddi-251140009/delhi-aq-dashboard](https://huggingface.co/spaces/Guna-Venkat-Doddi-251140009/delhi-aq-dashboard)

---

## 10. Conclusion
This project successfully bridged the gap between pure ML research and actionable environmental monitoring. Key findings:
1. **PatchTST** achieves state-of-the-art hourly performance (R²=0.821).
2. **XGBoost** with explicit lag and calendar features is the best daily model (R²=0.570).
3. **LSTM fails catastrophically on daily data** (R²=−0.884) — an important negative result: recurrent architectures require sufficient training sequence length to compete with domain-aware feature engineering.
4. **Walk-forward CV** reveals evaluation variance of ±3.57 µg/m³ across years; **Diebold-Mariano test** (*p*=0.0029) statistically validates XGBoost's improvement over persistence.
5. **Hierarchical seasonal conformal calibration** corrects the flat calibration's 14.5 ppt winter coverage deficit (global: 75.5% → seasonal: 90.8%), with an 11.2% overall interval width reduction.
6. **Failure mode analysis** (CUSUM): dominant error regime is physics-driven — calm-wind conditions (60% of top-5% failures) and wintertime thermal inversion (98% of failures in Winter, **3.28× Winter/Summer MAE ratio**), with CUSUM firing 7 days before Diwali without calendar knowledge.
7. **Masked MAPE = 39.7%** (PM2.5 > 10 µg/m³ filter) provides a stable cross-model comparison metric stable across summer near-zero readings.
8. **Ensemble stacking** yields +5.75 µg/m³ additional MAE improvement (R² from 0.543 → 0.683).
9. **VAE anomaly detector** autonomously identifies all four Diwali events (2021–2024) from unsupervised learning on clean summer chemistry.

**🚀 Live Dashboard:** [https://huggingface.co/spaces/Guna-Venkat-Doddi-251140009/delhi-aq-dashboard](https://huggingface.co/spaces/Guna-Venkat-Doddi-251140009/delhi-aq-dashboard)

---
## References
1. Box, G. E. P., Jenkins, G. M., & Reinsel, G. C. (2015). *Time Series Analysis: Forecasting and Control* (5th ed.). Wiley.
2. Chen, T., & Guestrin, C. (2016). XGBoost: A Scalable Tree Boosting System. *ACM SIGKDD*, 785–794.
3. Nie, Y., Nguyen, N. H., Sinthong, P., & Kalagnanam, J. (2023). A Time Series is Worth 64 Words: Long-term Forecasting with Transformers. *ICLR 2023*. arXiv:2211.14730.
4. Zhou, H., et al. (2021). Informer: Beyond Efficient Transformer for Long Sequence Time-Series Forecasting. *AAAI 2021*.
5. Hochreiter, S., & Schmidhuber, J. (1997). Long Short-Term Memory. *Neural Computation*, 9(8), 1735–1780.
6. Angelopoulos, A. N., & Bates, S. (2023). Conformal Prediction: A Gentle Introduction. *Foundations and Trends in Machine Learning*, 16(4), 494–591.
7. Diebold, F. X., & Mariano, R. S. (1995). Comparing Predictive Accuracy. *Journal of Business and Economic Statistics*, 13(3), 253–263.
8. Kingma, D. P., & Welling, M. (2014). Auto-Encoding Variational Bayes. *ICLR 2014*. arXiv:1312.6114.
9. Goel, A., & Kumar, A. (2015). Characterisation of nanoparticle emissions and exposure at traffic intersections. *Science of the Total Environment*, 508, 382–390.
10. Yadav, S., Tripathi, G., & Vishwakarma, A. K. (2021). Prediction of ambient PM2.5 in Delhi using LSTM and GRU. *Environmental Pollution*, 284, 117063. https://doi.org/10.1016/j.envpol.2021.117063
11. Singh, K., Trivedi, A., & Gupta, A. (2020). Predictive modelling of AQI using ML for Delhi NCR. *International Journal of Environmental Science and Technology*, 17, 4129–4142.
12. Zhang, W., Lin, Y., & Wang, J. (2023). Transformer-based long-term urban air quality forecasting. *Atmospheric Environment*, 314, 120100. https://doi.org/10.1016/j.atmosenv.2023.120100
13. Hyndman, R. J., & Koehler, A. B. (2006). Another look at measures of forecast accuracy. *International Journal of Forecasting*, 22(4), 679–688.

---
## Appendix A: Model Hyperparameters

### XGBoost Configuration
| Parameter | Value | Rationale |
|---|---|---|
| `n_estimators` | 1000 | Early stopping at patience=50 |
| `learning_rate` | 0.05 | Conservative step |
| `max_depth` | 6 | Prevents overfit on 63 features |
| `subsample` | 0.8 | Row-level stochastic boosting |
| `colsample_bytree` | 0.8 | Feature-level regularisation |
| `reg_lambda` | 1.0 | L2 weight regularisation |
| `reg_alpha` | 0.1 | L1 sparsity |
| `min_child_weight` | 5 | Prevents tiny-group splits |
| `objective` | `reg:squarederror` | Standard MSE |
| `tree_method` | `hist` | GPU-accelerated histogram |
| `random_state` | 42 | Reproducibility |

### PatchTST Architecture (Hourly Global Model)
| Parameter | Value | Rationale |
|---|---|---|
| Lookback window *L* | 336 hrs (14 days) | Captures weekly + diurnal cycles |
| Forecast horizon *H* | 24 hrs (1 day) | Short-term operational forecast |
| Patch length *P* | 16 hrs | ~2/3 of a diurnal cycle per patch |
| Patch stride | 8 hrs | 50% overlap |
| Number of patches | 41 | Self-attention sequence length |
| *d*_model | 128 | Embedding dimension |
| *n*_heads | 8 | Multi-head self-attention |
| *d*_ff | 256 | Feed-forward hidden dim |
| Encoder layers | 3 | Shallow to avoid overfit |
| Dropout | 0.2 | Regularisation |
| Learning rate | 1e-4 | Adam optimiser |
| Epochs | 100 (early stop 10) | |
| `channel_independence` | True | One backbone per station | 
