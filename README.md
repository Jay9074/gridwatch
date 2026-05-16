# ⚡ GridWatch - AI-Powered Power Grid Outage Risk Intelligence

[![Live Dashboard](https://img.shields.io/badge/Dashboard-Live-brightgreen.svg)](https://gridwatch-dashboard.streamlit.app/)
[![Python](https://img.shields.io/badge/Python-3.11-blue.svg)](https://python.org)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Data](https://img.shields.io/badge/Data-EAGLE - I%20ORNL-orange.svg)](https://figshare.com/articles/dataset/24237376)

> Open-source AI platform integrating EAGLE-I, NOAA Storm Events, and EIA-861 data into a unified power grid outage risk intelligence tool for the Northeast United States.

**🌐 Live Dashboard:** [gridwatch-dashboard.streamlit.app](https://gridwatch-dashboard.streamlit.app/)
**📄 Research Paper:** Available in this repository (white_paper.docx)
**👤 Author:** Jaykumar Patel [GitHub](https://github.com/Jay9074)
## 🎯 What Problem Does This Solve?

US power outages cost between **$121 billion and $150 billion every year** (DOE/ORNL, 2024). The Northeast region carries a disproportionate share of that burden because its infrastructure is aging, weather is intensifying, and no public tool exists that lets researchers, planners, or the general public examine where outage risk is concentrated and why.

GridWatch fills that gap. It uses only public federal data - no proprietary utility records - so anyone can replicate or build on this work.

GridWatch provides:
- **County-level risk intelligence** across 9 Northeast states using 11 years of federal data
- **State-monthly forecasting** with R² of 0.84
- **Economic impact estimation** using DOE Value of Lost Load methodology
- **2026-2030 projections** with climate-adjusted scenarios
- **Real-time NOAA weather integration**
- **SHAP explainability** for every prediction
## 📊 Key Findings

| Finding | Value
| - -| - -|
| Total dataset | **767,855 county-days** (2014-2025, 9 states)
| Major outage rate | **10.17%** of all county-days
| Critical outage rate | 0.41% (events with 10K+ customers)
| Peak single event | **599,357 customers** (New York)
| **Highest-risk state** | **New Jersey - 19.4%** outage rate
| **Highest-risk county** | **Philadelphia, PA - 37.0%** of days
| **Highest-risk season** | **Summer - 12.4%** (NOT Winter)
| Lowest-risk state | Vermont - 4.4%
| Best ML model | Random Forest regression - **R² 0.84**
| Day-level classifier | AUC 0.69 - useful for ranking
| Top SHAP feature | Historical patterns (78% combined importance)
| Climate outlook | NOAA projects +9-18% extreme events by 2030
### The Surprises

1. **Summer is the worst season for outages, not Winter.** This challenges the conventional emphasis on winter storm preparedness in Northeast utility planning.
2. **New Jersey has the highest outage rate at 19.4%.** Nearly 1 in 5 county-days experience a major outage.
3. **Philadelphia is the single highest-risk county.** Major outages occur on 37% of all observed days.
4. **Grid vulnerability is structural, not weather-driven.** Historical outage patterns dominate predictions over weather features.
## 🗂️ Repository Structure

```
gridwatch/
│
├── 📊 Dashboard/
│ └── app.py # Streamlit dashboard (14 sections)
│
├── ⚙️ .streamlit/
│ └── config.toml # Forces light theme
│
├── 📓 notebooks/
│ ├── 01_data_exploration.ipynb # EDA - 8 figures
│ ├── 02_feature_engineering.ipynb # 24 features, 6 figures
│ ├── 03_ml_models_shap.ipynb # Classification + SHAP, 6 figures
│ ├── 04_lstm_forecasting.ipynb # Deep learning analysis, 4 figures
│ └── 05_nlp_analysis.ipynb # NLP text mining, 5 figures
│
├── 🔧 src/
│ ├── load_data_fixed.py # Load all 12 months from raw EAGLE-I
│ ├── build_monthly_dataset.py # Build state-monthly dataset
│ ├── train_monthly_models.py # Train regression models
│ ├── tune_random_forest.py # Hyperparameter tuning
│ ├── train_lstm_v2.py # LSTM with full data
│ ├── regenerate_summaries.py # Build dashboard CSVs
│ ├── model.py # Classification training
│ ├── data_processing.py # Full data pipeline
│ ├── generate_*.py # Various summary generators
│ ├── diagnose_lstm.py # LSTM diagnostic
│ └── check_raw_data.py # Raw data validator
│
├── 📈 reports/
│ └── figures/ # 29 publication-quality figures
│
├── 📋 data/
│ └── summary/ # Pre-computed CSVs (GitHub-hosted)
│ ├── state_risk_summary.csv
│ ├── monthly_trend.csv
│ ├── seasonal_summary.csv
│ ├── yearly_trend.csv
│ ├── state_month_heatmap.csv
│ └── county_risk_summary.csv
│
├── requirements.txt
├── white_paper.docx # Research paper (APA format)
└── README.md
```

> **Note:** Raw EAGLE-I CSVs (228M+ rows total) and trained model `.pkl` files are not in the repository due to file size. See Data Sources section to download raw data.
## 🚀 Quick Start

### 1. Clone the repository
```bash
git clone https://github.com/Jay9074/gridwatch.git
cd gridwatch
```

### 2. Set up environment
```bash
conda create -n gridwatch python=3.11 -y
conda activate gridwatch
pip install -r requirements.txt
```

### 3. Download raw data
Download from sources listed below and place in `data/raw/`:
- EAGLE-I: `eaglei_outages_YEAR.csv` (2014-2025)
- NOAA: `noaa_storms_YEAR.csv.gz` (2019-2025)

### 4. Process data and train models
```bash
python src/load_data_fixed.py # Load all 12 months from EAGLE-I (15-30 min)
python src/regenerate_summaries.py # Build dashboard summaries
python src/build_monthly_dataset.py # Build state-monthly dataset
python src/train_monthly_models.py # Train regression models
python src/model.py # Train classification models
```

### 5. Run dashboard locally
```bash
streamlit run Dashboard/app.py
```
## 📦 Data Sources

| Dataset | Source | Coverage | Description
| - -| - -| - -| - -|
| **EAGLE-I** | [ORNL/DOE via Figshare](https://figshare.com/articles/dataset/24237376) | 2014-2025 | County-level outages at 15-min intervals
| **NOAA Storm Events** | [NOAA NCEI](https://www.ncei.noaa.gov/pub/data/swdi/stormevents/csvfiles/) | 2019-2025 | Weather events by type and location
| **EIA Form 861** | [EIA](https://www.eia.gov/electricity/data/eia861/) | 2019-2023 | Utility reliability metrics (SAIDI/SAIFI)
## 🤖 Model Architecture

### State-Monthly Regression (HEADLINE - R² 0.84)
- **Target:** `major_outage_days` per state per month
- **Dataset:** 1,205 state-months × 38 features
- **Best model:** Random Forest (tuned)
- **R²:** 0.8386 on held-out 2024-2025 test set
- **Correlation:** 0.92
- **RMSE:** 27.0 outage days

### County-Day Classification
- **Target:** Major outage (≥1,000 customers affected)
- **Dataset:** 767,855 county-days × 24 leakage-free features
- **Best model:** XGBoost / Random Forest (tied)
- **AUC:** 0.69
- **F1:** 0.29
- **Honest framing:** Useful for risk ranking; limited by absent daily weather and infrastructure data

### LSTM Forecasting (Negative Finding)
LSTM was tested across 1, 3, and 6-month horizons. All horizons returned negative R² (worse than predicting the mean). **Tree-based ensembles outperform LSTM at this aggregation level** because outage patterns have more interaction structure than sequential structure. This is documented in the paper as a research contribution.

### NLP Analysis
- **Method:** TF-IDF + LDA Topic Modeling on 23,605 NOAA storm event descriptions
- **Topics discovered:** 5 coherent failure categories
- **Key finding:** High-severity events dominated by snow/precipitation language; lower-severity by wind/tree damage
## 📈 Solution Framework

GridWatch translates research findings into three actionable tiers:

**Tier 1 - County Infrastructure Prioritization**
Counties with outage rates above 25% receive Priority 1 designation for immediate vegetation audit, substation inspection, and crew pre-positioning.

**Tier 2 - State-Monthly Forecasting**
The R² 0.84 regression model enables monthly outage volume forecasts with 1-month lead time for state emergency management agencies.

**Tier 3 - Policy & Regulatory Applications**
Data-driven SAIDI/SAIFI target setting, climate resilience investment justification, and open data advocacy for utility infrastructure records.

**Estimated impact:** 10% prevention of major outage events in identified high-risk counties → $140-$180 million in direct avoided costs over 5 years → $12-15 billion annual at national scale (DOE VoLL methodology).
## 📊 Dashboard Sections

The live dashboard at [gridwatch-dashboard.streamlit.app](https://gridwatch-dashboard.streamlit.app/) includes:

1. **KPI Summary** - top-line numbers from full dataset
2. **State Risk Map** - interactive county-colored map
3. **State Risk Rankings** - sortable composite risk table
4. **Monthly Trend** - outage events 2014-2025 with rolling average
5. **Seasonal Analysis** - Summer is highest at 12.4%
6. **Year-over-Year by State** - 9-state historical comparison
7. **County Drill-Down** - 226 counties with risk scores
8. **EIA SAIDI/SAIFI Panel** - industry-standard reliability metrics
9. **NOAA Weather Correlation** - storm data integration
10. **ML Model Performance** - classification model results
11. **SHAP Feature Importance** - explainable AI
12. **Future Projections (2026-2030)** - climate-adjusted forecasts
13. **Live Weather + Real-Time Risk** - current NOAA conditions
14. **Economic Impact Calculator** - DOE Value of Lost Load methodology
15. **Outage Risk Calculator** - interactive scenario builder

Each section includes an inline description explaining what it shows and what the key finding is.
## 🗺️ Roadmap

### Completed
- [x] EAGLE-I full-year data pipeline (767K county-days)
- [x] NOAA storm event integration
- [x] EIA-861 reliability metrics
- [x] State-monthly Random Forest regression (R² 0.84)
- [x] County-day classification with SHAP explainability
- [x] LSTM evaluation (documented as negative finding)
- [x] NLP analysis on storm event text
- [x] Live Streamlit dashboard (15 sections)
- [x] Real-time NOAA weather API integration
- [x] 2026-2030 climate-adjusted projections
- [x] White paper (APA format)

### Planned
- [ ] **EconoGrid** - county-level economic cost modeling (Project 2)
- [ ] **StormSight** - 72-hour deep learning outage warning (Project 3)
- [ ] Extension to additional US regions
- [ ] Integration of satellite vegetation imagery as proxy for tree management risk
## 📝 Citation

If you use GridWatch in your research, please cite:

```
Patel, J. (2026). GridWatch: AI-Powered Power Grid Outage Risk Intelligence
for the Northeast United States. Independent Research.
Dashboard: gridwatch-dashboard.streamlit.app
GitHub: github.com/Jay9074/gridwatch
```
## 📬 Contact

**Jaykumar Patel**
- 🌐 [gridwatch-dashboard.streamlit.app](https://gridwatch-dashboard.streamlit.app/)
- 💻 [github.com/Jay9074/gridwatch](https://github.com/Jay9074/gridwatch)
## 📄 License

MIT License - feel free to use, modify, and distribute. See LICENSE file for details.
*Data sources: EAGLE-I (Brelsford et al., 2024), NOAA Storm Events Database, EIA Form 861. All analysis is independent research conducted using only publicly available federal data.*
