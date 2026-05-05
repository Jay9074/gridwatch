# ⚡ GridWatch — AI-Powered Power Grid Outage Risk Intelligence

[![Live Dashboard](https://img.shields.io/badge/Dashboard-Live-brightgreen.svg)](https://gridwatch-dashboard.streamlit.app/)
[![Python](https://img.shields.io/badge/Python-3.11-blue.svg)](https://python.org)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Data](https://img.shields.io/badge/Data-EAGLE--I%20ORNL-orange.svg)](https://figshare.com/articles/dataset/24237376)

> The open-source AI platform integrating EAGLE-I, NOAA Storm Events, and EIA-861 data into a unified power grid outage risk intelligence tool for the Northeast United States.

**🌐 Live Dashboard:** [gridwatch-dashboard.streamlit.app](https://gridwatch-dashboard.streamlit.app/)
**📄 Research Paper:** In submission — April 2026
**👤 Author:** Jaykumar Patel

---

## 🎯 What Problem Does This Solve?

US power outages cost **$121–150 billion annually** (DOE/ORNL 2024). The Northeast US is disproportionately affected by aging infrastructure, coastal storm exposure, and accelerating climate change.

**The gap:** No publicly available AI tool existed to predict, visualize, and explain regional outage risk using open federal data - until GridWatch.

GridWatch provides:
- **County-level risk intelligence** across 9 Northeast states using 11 years of federal data
- **Seasonal early warning** — LSTM forecasts 1, 3, and 6 months ahead
- **Economic impact estimation** using DOE Value of Lost Load methodology
- **2026–2030 projections** with climate-adjusted scenarios
- **Real-time weather integration** from NOAA API

---

## 📊 Key Research Findings

| Finding | Value |
|---|---|
| Dataset | 89,945 county-days, 2014–2025, 9 Northeast states |
| Best ML model | Random Forest — ROC-AUC 0.712 |
| LSTM 1-month forecast | r = 0.968 |
| LSTM 6-month forecast | r = 1.000 |
| Highest risk state | New Jersey (15.0% outage rate) |
| Highest risk county | Philadelphia, PA (180 outage days) |
| Peak single event | 106,447 customers (New Jersey) |
| Top SHAP feature | Prior outage history (22.5% importance) |
| Trend | All 9 states improving (-0.284%/year) |
| Climate risk | NOAA projects +9–18% extreme events by 2030 |

---

## 🗂️ Repository Structure

```
gridwatch/
│
├── 📊 Dashboard
│   └── app.py                        # Streamlit dashboard (14 sections)
│
├── 📓 notebooks/
│   ├── 01_data_exploration.ipynb     # EDA - 8 publication figures
│   ├── 02_feature_engineering.ipynb  # 24 features, 6 figures
│   ├── 03_ml_models_shap.ipynb       # ML + SHAP, 6 figures
│   ├── 04_lstm_forecasting.ipynb     # Deep learning, 4 figures
│   └── 05_nlp_analysis.ipynb         # NLP text mining, 5 figures
│
├── 🔧 src/
│   ├── load_data_simple.py           # Load EAGLE-I + NOAA data
│   ├── load_all_years.py             # Load all available years
│   ├── model.py                      # ML training pipeline (leakage-free)
│   ├── data_processing.py            # Full data processing pipeline
│   ├── generate_summary.py           # Generate dashboard summary CSVs
│   ├── generate_yearly.py            # Year-over-year state breakdown
│   ├── generate_noaa_correlation.py  # NOAA weather correlation analysis
│   ├── generate_eia_summary.py       # EIA-861 SAIDI/SAIFI extraction
│   ├── generate_county_summary.py    # County-level risk scores
│   ├── generate_projections.py       # 2026–2030 outage projections
│   ├── get_importances.py            # Extract SHAP feature importances
│   └── print_metrics.py              # Print trained model metrics
│
├── 📈 reports/
│   └── figures/                      # 29 publication-quality figures
│       ├── fig1_outage_distribution.png
│       ├── fig2_state_comparison.png
│       ├── fig3_yearly_trend.png
│       ├── fig4_seasonal_analysis.png
│       ├── fig5_state_month_heatmap.png
│       ├── fig6_top_counties.png
│       ├── fig7_correlation_matrix.png
│       ├── fig8_yearly_state_trends.png
│       ├── fig_fe[1-6]_*.png         # Feature engineering figures
│       ├── fig_ml[1-6]_*.png         # ML model figures
│       ├── fig_lstm[1-4]_*.png       # LSTM forecasting figures
│       └── fig_nlp[1-5]_*.png        # NLP analysis figures
│
├── 📋 data/
│   └── summary/                      # Pre-computed summaries (GitHub-hosted)
│       ├── state_risk_summary.csv    # State risk scores - feeds dashboard
│       ├── monthly_trend.csv         # Monthly outage trend 2014–2025
│       └── seasonal_summary.csv      # Seasonal outage breakdown
│
├── requirements.txt                  # Python dependencies
└── README.md
```

> **Note:** Raw data files (`data/raw/`) and trained model files (`models/`) are not included in the repository due to file size. See Data Sources section below to download the raw data.

---

## 🚀 Quick Start

### 1. Clone the repository
```bash
git clone https://github.com/Jay9074/gridwatch.git
cd gridwatch
```

### 2. Create environment and install dependencies
```bash
conda create -n gridwatch python=3.11 -y
conda activate gridwatch
pip install -r requirements.txt
```

### 3. Download data
Download from the sources listed below and place in `data/raw/`:
- EAGLE-I: `eaglei_outages_YEAR.csv` (2014–2025)
- NOAA: `noaa_storms_YEAR.csv.gz` (2019–2025)
- Supplementary: `MCC.csv`, `DQI.csv`, `coverage_history.csv`

### 4. Process data
```bash
python src/load_data_simple.py
python src/generate_summary.py
```

### 5. Train models
```bash
python src/model.py
```

### 6. Run dashboard locally
```bash
streamlit run Dashboard/app.py
```

---

## 📦 Data Sources

| Dataset | Source | Coverage | Description |
|---|---|---|---|
| **EAGLE-I** | [ORNL/DOE via Figshare](https://figshare.com/articles/dataset/24237376) | 2014–2025 | County-level outages at 15-min intervals |
| **NOAA Storm Events** | [NCEI](https://www.ncei.noaa.gov/pub/data/swdi/stormevents/csvfiles/) | 2019–2025 | Weather events by type and location |
| **EIA Form 861** | [EIA](https://www.eia.gov/electricity/data/eia861/) | 2019–2023 | Utility reliability metrics (SAIDI/SAIFI) |
| **MCC / DQI** | Included with EAGLE-I download | 2014–2025 | Customer counts and data quality index |

---

## 🤖 Model Architecture

### Classification Models (Outage Risk Prediction)
- **Features:** 24 leakage-free features across 5 categories (time, geographic, historical, weather, interaction)
- **Target:** Major outage (≥1,000 customers affected)
- **Best model:** Random Forest - ROC-AUC 0.712, Recall 0.603
- **Class imbalance:** SMOTE oversampling (7.9% positive rate)
- **Validation:** 5-fold stratified cross-validation

### LSTM (Time-Series Forecasting)
- **Architecture:** 2-layer stacked LSTM (32→16 units) + Dropout(0.2) + BatchNorm
- **Loss:** Huber (robust to outlier months)
- **Input:** 6-month sequence of outage rate + lag features
- **Horizons:** 1-month (r=0.968), 3-month (r=1.000), 6-month (r=1.000)

### NLP (Text Analysis)
- **Method:** TF-IDF + LDA Topic Modeling on NOAA storm event descriptions
- **Topics discovered:** 5 coherent failure categories
- **Key finding:** High-severity events dominated by snow/precipitation language; lower-severity by wind/tree damage

---

## 📈 Solution Framework

GridWatch translates research findings into three actionable tiers:

**Tier 1 - County Infrastructure Prioritization**
48 HIGH-risk counties identified for immediate infrastructure investment priority. Philadelphia PA, Cumberland ME, Nassau NY lead the list.

**Tier 2 - Seasonal Early Warning**
LSTM forecasts enable 3–6 month advance risk bulletins to state emergency managers - enough lead time for crew pre-positioning and equipment procurement.

**Tier 3 - Policy & Regulatory Applications**
Data-driven SAIDI/SAIFI target setting, climate resilience investment justification, and open data advocacy for utility infrastructure records.

**Estimated impact:** 10% prevention of major outage events in HIGH-risk counties → $12–15B annual national economic benefit (DOE VoLL methodology).

---

## 📊 Dashboard Sections

The live dashboard at [gridwatch-dashboard.streamlit.app](https://gridwatch-dashboard.streamlit.app/) includes:

1. KPI Summary - real EAGLE-I data
2. State Risk Map - interactive, county-colored
3. State Risk Rankings - color-coded table
4. Monthly Trend - 2014–2025
5. Seasonal Analysis
6. Year-over-Year by State
7. County Drill-Down - 176 counties, 9 states
8. EIA SAIDI/SAIFI Panel
9. NOAA Weather Correlation
10. ML Model Performance
11. SHAP Feature Importance
12. Future Projections 2026–2030
13. Live Weather + Real-Time Risk
14. Economic Impact Calculator
15. Outage Risk Calculator

---

## 🗺️ Roadmap

- [x] EAGLE-I data pipeline (2014–2025)
- [x] NOAA storm event integration
- [x] EIA-861 reliability metrics
- [x] Random Forest + XGBoost classification
- [x] SHAP explainability
- [x] LSTM multi-horizon forecasting
- [x] NLP text analysis
- [x] Live Streamlit dashboard (14 sections)
- [x] Real-time NOAA weather API
- [x] 2026–2030 climate-adjusted projections
- [x] White paper (in submission)
- [ ] **EconoGrid** - county-level economic cost modeling (Project 2)
- [ ] **StormSight** - 72-hour deep learning outage warning (Project 3)
- [ ] arXiv publication
- [ ] Extension to full US coverage

---

## 📝 Citation

If you use GridWatch in your research, please cite:

```
Patel, J. (2026). GridWatch: AI-Powered Power Grid Outage Risk Intelligence
for the Northeast United States. 
Dashboard: gridwatch-dashboard.streamlit.app
GitHub: github.com/Jay9074/gridwatch
```

---

## 📬 Contact

**Jaykumar Patel**
MS Data Science — Stevens Institute of Technology
MS IT Project Management (in progress) — New England College
🌐 [gridwatch-dashboard.streamlit.app](https://gridwatch-dashboard.streamlit.app/)

---

*Data sources: EAGLE-I (ORNL/DOE), NOAA Storm Events, EIA Form 861. All analysis is independent research.*
