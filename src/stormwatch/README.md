# GridWatch Storm Watch

Real-time storm detection and outage prediction system. Built on top of the existing GridWatch infrastructure.

## What It Does

1. **Pulls NOAA forecasts** for 30 Northeast counties every 6 hours
2. **Detects storms** at 3 severity tiers (SEVERE / MODERATE / MINOR)
3. **Predicts outages** per storm using historical baselines + storm severity
4. **Validates accuracy** by comparing past predictions to actual EAGLE-I data
5. **Publishes scorecard** with rolling accuracy metrics

## Why It Matters

Every prediction is logged publicly and accuracy is tracked in real time against actual EAGLE-I outcomes. This open validation is the GridWatch approach: "verify our predictions" instead of "trust our predictions."

## Setup (One-Time)

### 1. Install dependencies (already in your environment)
```bash
pip install requests pandas numpy
```

### 2. Make sure the existing GridWatch data exists
You need these files (already created by the main project):
- `data/processed/eaglei_daily_northeast.csv`
- `data/summary/county_risk_summary.csv`

### 3. Test the pipeline once manually
```bash
conda activate gridwatch
cd D:\projects\gridwatch-main
python src/stormwatch/run_pipeline.py
```

This will:
- Create `data/stormwatch/forecasts/latest.csv`
- Create `data/stormwatch/storms/active_storms.csv`
- Create `data/stormwatch/predictions/active_predictions.csv`
- Create `data/stormwatch/validation/accuracy_scorecard.json`

## Run On Schedule (Windows)

To run every 6 hours automatically:

1. Open Task Scheduler
2. Create Basic Task: "GridWatch Storm Watch"
3. Trigger: Daily, repeat every 6 hours
4. Action: Start a program
   - Program: `C:\Users\JAY PATEL\anaconda3\envs\gridwatch\python.exe`
   - Arguments: `src/stormwatch/run_pipeline.py`
   - Start in: `D:\projects\gridwatch-main`

## File Outputs

```
data/stormwatch/
├── forecasts/
│   ├── latest.csv                    # Current forecast
│   └── forecast_YYYYMMDD_HHMM.csv    # Historical snapshots
├── storms/
│   ├── active_storms.csv             # Currently detected storms
│   └── storms_YYYYMMDD_HHMM.csv      # Snapshots
├── predictions/
│   ├── active_predictions.csv        # Current predictions
│   ├── prediction_log.csv            # Cumulative log (used for validation)
│   └── predictions_YYYYMMDD_HHMM.csv # Snapshots
└── validation/
    ├── validation_results.csv        # Per-prediction accuracy
    └── accuracy_scorecard.json       # Aggregate accuracy metrics
```

## Backtesting (Validate on Historical Data First)

Before running live, you should backtest on past storms. We'll build this next.

For now, the live system can run starting today, and validation will kick in after 60 days as predictions age past the EAGLE-I lag window.

## Dashboard Integration

The next step is adding a "Storm Watch" section to the main dashboard that displays:
- Current active storms (from active_storms.csv)
- Outage predictions (from active_predictions.csv)
- Public accuracy scorecard (from accuracy_scorecard.json)

This will be the headline new feature that differentiates GridWatch from any other tool.
