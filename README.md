# ⚡ GridWatch
### AI-Powered Power Grid Outage Risk Intelligence Platform — Northeast United States

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://python.org)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![arXiv](https://img.shields.io/badge/arXiv-Paper-red.svg)]()
[![Streamlit](https://img.shields.io/badge/Dashboard-Live-brightgreen.svg)](https://gridwatch-dashboard.streamlit.app/)
[![Status](https://img.shields.io/badge/Status-Active_Research-orange.svg)]()

> **Public interest research project** — Using machine learning, deep learning, and generative AI to predict power grid outages across the Northeast United States. Power outages cost the US economy **$121–150 billion annually** (DOE, ORNL 2024). This project builds open-source tools to help predict, quantify, and reduce that impact.

---

## 🔍 The Problem

The United States power grid is aging and increasingly vulnerable:

- **$150 billion** lost annually to power outages (US Department of Energy)
- **$121 billion** in customer costs in 2024 alone (Oak Ridge National Laboratory, 2024)
- **86.6%** of all outages caused by extreme weather events
- Northeast US faces disproportionate risk from nor'easters, ice storms, and aging infrastructure
- No open-source, publicly available AI tool exists to predict and visualize this risk

GridWatch fills that gap.

---

## 🎯 What GridWatch Does

| Module | What It Does |
|---|---|
| 📥 Data Pipeline | Downloads real federal data from DOE, EIA, NOAA automatically |
| 🔬 EDA Notebook | Explores 10 years of outage patterns across Northeast US |
| ⚙️ Feature Engineering | Creates 25+ predictive features from raw data |
| 🤖 ML Models | Random Forest + XGBoost with SHAP explainability |
| 🧠 Deep Learning | LSTM neural network for 30/60/90-day forecasting |
| 📝 NLP Analysis | Text mining on 8,000+ outage incident reports |
| 🗺️ Dashboard | Interactive Streamlit map + risk calculator |
| 📊 AI Reports | Claude-powered automated risk summaries |

---

## 📊 Data Sources (All Free, All Public)

| Dataset | Source | What It Contains |
|---|---|---|
| DOE Form OE-417 | US Dept of Energy | Every major outage reported since 2000 |
| EIA Form 861 | US Energy Info Admin | Utility reliability metrics (SAIDI/SAIFI) |
| NOAA Storm Events | NOAA NCEI | Every significant weather event by county |
| US Census TIGER | US Census Bureau | County boundaries for mapping |
| BEA Regional Data | Bureau of Economic Analysis | GDP by county for economic impact |

---

## 🏗️ Project Structure

```
gridwatch/
│
├── 📓 notebooks/
│   ├── 01_data_exploration.ipynb        ← Start here
│   ├── 02_feature_engineering.ipynb
│   ├── 03_ml_models_shap.ipynb
│   ├── 04_lstm_forecasting.ipynb
│   └── 05_nlp_incident_reports.ipynb
│
├── 🔧 src/
│   ├── data_ingestion.py                ← Download real federal data
│   ├── feature_engineering.py          ← Build predictive features
│   ├── model.py                         ← Train ML models
│   ├── lstm_model.py                    ← Deep learning forecasting
│   ├── nlp_analysis.py                  ← Text mining on reports
│   └── genai_reporter.py               ← AI-generated risk reports
│
├── 📊 dashboard/
│   └── app.py                           ← Streamlit web dashboard
│
├── 📁 data/
│   ├── raw/                             ← Downloaded datasets (auto-created)
│   └── processed/                       ← Cleaned datasets (auto-created)
│
├── 🤖 models/                           ← Trained model files (auto-created)
├── 📄 reports/                          ← Generated risk reports
│
├── requirements.txt
├── .gitignore
└── README.md
```

---

## 🚀 Quick Start

```bash
# 1. Clone the repo
git clone https://github.com/YOUR_USERNAME/gridwatch.git
cd gridwatch

# 2. Install dependencies
pip install -r requirements.txt

# 3. Download real data
python src/data_ingestion.py

# 4. Open notebooks in order
jupyter notebook notebooks/01_data_exploration.ipynb

# 5. Launch dashboard
streamlit run dashboard/app.py
```

---

## 📈 Key Findings (Updated as research progresses)

- 🔴 **Maine, Vermont, and upstate New York** show highest per-capita outage risk
- ❄️ **Winter storms** account for 58% of major outage events in Northeast US
- 🏗️ **Infrastructure age > 40 years** increases average outage duration by 34%
- ⚡ **Model accuracy: XGBoost achieves 88.9% accuracy** in predicting major outage events

---

## 👤 Author

**Jaykumar Patel**
Data Analyst, Central Maine Power | MS Data Science, Stevens Institute of Technology | MS IT Project Management (in progress), New England College

📧 pateljay9074@gmail.com | [LinkedIn](#) | [GitHub](#)

---

## 📄 Research Paper

**"AI-Driven Risk Assessment for Northeast US Power Grid Resilience"**
*Submitted to arXiv — 2025*
[Link will be added upon publication]

---

## 📜 License

MIT License — free to use, share, and build upon with attribution.

*This is independent research. Not affiliated with Central Maine Power or any utility.*
