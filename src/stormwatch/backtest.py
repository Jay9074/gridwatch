"""
GridWatch Storm Watch - Historical Backtesting
================================================
Runs the prediction model on PAST storms and compares to actual EAGLE-I outcomes.
Generates a real accuracy scorecard you can publish TODAY (no 60-day wait).

This is what proves GridWatch works without needing to collect new data:
- Pick storms from 2020-2024 in NOAA Storm Events database
- For each, identify affected counties and storm tier
- Run the prediction model as if it were forecasting in advance
- Compare predicted vs actual customers affected (from EAGLE-I)
- Compute per-tier accuracy metrics

Run: python src/stormwatch/backtest.py

Output: data/stormwatch/backtest/backtest_results.csv
        data/stormwatch/backtest/backtest_scorecard.json
"""
import pandas as pd
import numpy as np
import json
from pathlib import Path
from datetime import datetime, timedelta
import sys

OUT_DIR = Path("data/stormwatch/backtest")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Same county set as the live prediction system (so backtest is comparable)
TARGET_COUNTIES = [
    ("Cumberland", "Maine"), ("Penobscot", "Maine"), ("Kennebec", "Maine"),
    ("York", "Maine"), ("Androscoggin", "Maine"),
    ("Hillsborough", "New Hampshire"), ("Rockingham", "New Hampshire"),
    ("Chittenden", "Vermont"),
    ("Middlesex", "Massachusetts"), ("Worcester", "Massachusetts"),
    ("Essex", "Massachusetts"), ("Suffolk", "Massachusetts"),
    ("Providence", "Rhode Island"),
    ("Hartford", "Connecticut"), ("New Haven", "Connecticut"),
    ("Fairfield", "Connecticut"),
    ("Suffolk", "New York"), ("Nassau", "New York"),
    ("Westchester", "New York"), ("Erie", "New York"),
    ("Essex", "New Jersey"), ("Bergen", "New Jersey"),
    ("Middlesex", "New Jersey"), ("Monmouth", "New Jersey"),
    ("Ocean", "New Jersey"),
    ("Philadelphia", "Pennsylvania"), ("Allegheny", "Pennsylvania"),
    ("Montgomery", "Pennsylvania"), ("Bucks", "Pennsylvania"),
    ("Chester", "Pennsylvania"),
]


def classify_storm_tier(storm_row):
    """Map NOAA storm event to our prediction tier (SEVERE/MODERATE/MINOR).
    
    Uses storm type + magnitude when available.
    Mirrors the logic in detect_storms.py for consistency.
    """
    event_type = str(storm_row.get("event_type", "")).lower()
    mag = storm_row.get("magnitude", 0) or 0
    
    # SEVERE: storms that historically cause widespread outages
    severe_types = [
        "ice storm", "blizzard", "hurricane", "tropical storm",
        "tornado", "freezing rain"
    ]
    if any(t in event_type for t in severe_types):
        return "SEVERE"
    if "thunderstorm" in event_type and mag >= 60:
        return "SEVERE"
    
    # MODERATE: significant storms
    moderate_types = [
        "winter storm", "heavy snow", "high wind",
        "thunderstorm wind"
    ]
    if any(t in event_type for t in moderate_types):
        return "MODERATE"
    if mag >= 35:
        return "MODERATE"
    
    # MINOR: smaller events
    return "MINOR"


def predict_customers(tier, county_baseline, magnitude=0):
    """Predict customers affected — mirrors predict_outages.py logic.
    
    Same calibrated approach: median major outage as baseline,
    tier and wind multipliers.
    """
    if tier == "MINOR":
        baseline = county_baseline["typical_major_outage"] * 0.45
    elif tier == "MODERATE":
        baseline = county_baseline["typical_major_outage"] * 0.85
    elif tier == "SEVERE":
        baseline = county_baseline["high_outage"] * 1.1
    else:
        baseline = county_baseline["typical_major_outage"] * 0.4
    
    # Wind multiplier
    if magnitude >= 60:
        wind_mult = 1.6
    elif magnitude >= 45:
        wind_mult = 1.3
    elif magnitude >= 30:
        wind_mult = 1.1
    else:
        wind_mult = 1.0
    
    predicted = baseline * wind_mult
    
    # Caps (same as live system)
    if tier == "MINOR":
        cap = county_baseline["typical_major_outage"] * 2
    elif tier == "MODERATE":
        cap = county_baseline["high_outage"] * 1.5
    elif tier == "SEVERE":
        cap = county_baseline["extreme_outage"]
    else:
        cap = county_baseline["typical_major_outage"] * 1.5
    
    predicted = min(predicted, cap)
    predicted = max(predicted, 200)
    return round(predicted)


def load_county_baselines(eaglei_df):
    """Compute baselines from EAGLE-I, excluding the test period.
    
    This prevents data leakage - we calculate baselines using 2014-2019 data,
    then test on 2020-2024 storms.
    """
    # Use pre-2020 data to compute baselines
    train = eaglei_df[eaglei_df["date"] < "2020-01-01"]
    train_major = train[train["max_customers_out"] >= 1000]
    
    baselines = {}
    for (county, state) in TARGET_COUNTIES:
        sub = train_major[(train_major["county"] == county) &
                          (train_major["state"] == state)]
        if len(sub) >= 5:
            baselines[f"{county}, {state}"] = {
                "typical_major_outage": float(sub["max_customers_out"].median()),
                "high_outage":          float(sub["max_customers_out"].quantile(0.75)),
                "extreme_outage":       float(sub["max_customers_out"].quantile(0.95)),
                "n_training_events":    len(sub),
            }
        else:
            baselines[f"{county}, {state}"] = {
                "typical_major_outage": 1500.0,
                "high_outage":          3000.0,
                "extreme_outage":       8000.0,
                "n_training_events":    len(sub),
            }
    return baselines


def main():
    print("=" * 70)
    print("GridWatch Storm Watch - Historical Backtesting")
    print("Train: 2014-2019  |  Test: 2020-2024")
    print("=" * 70)
    
    # Load EAGLE-I data
    eaglei_path = Path("data/processed/eaglei_daily_northeast.csv")
    if not eaglei_path.exists():
        print(f"ERROR: {eaglei_path} not found")
        print("Run load_data_fixed.py first to generate the EAGLE-I dataset.")
        return 1
    
    print("Loading EAGLE-I data...")
    eagle = pd.read_csv(eaglei_path, parse_dates=["date"])
    print(f"  Loaded {len(eagle):,} county-days")
    
    # Load NOAA storm events
    noaa_path = Path("data/processed/noaa_storms_northeast.csv")
    if not noaa_path.exists():
        print(f"ERROR: {noaa_path} not found")
        return 1
    
    print("Loading NOAA storm events...")
    noaa = pd.read_csv(noaa_path, low_memory=False)
    
    # Normalize column names to lowercase (NOAA file uses uppercase)
    noaa.columns = [c.lower() for c in noaa.columns]
    
    # Identify date column - NOAA uses BEGIN_DATE_TIME
    date_col = next((c for c in ["begin_date_time", "begin_date", "event_date", "date"]
                     if c in noaa.columns), None)
    if not date_col:
        print(f"ERROR: no date column found in NOAA file. Columns: {list(noaa.columns)}")
        return 1
    
    noaa["event_date"] = pd.to_datetime(noaa[date_col], errors="coerce")
    noaa = noaa.dropna(subset=["event_date"])
    print(f"  Loaded {len(noaa):,} storm events from column '{date_col}'")
    
    # Map NOAA columns to what the rest of the script expects
    # NOAA uses STATE, CZ_NAME (county/zone name), EVENT_TYPE, MAGNITUDE
    if "cz_name" in noaa.columns and "county" not in noaa.columns:
        noaa["county"] = noaa["cz_name"].astype(str).str.title()
    if "state" in noaa.columns:
        # NOAA state names are sometimes uppercase
        noaa["state"] = noaa["state"].astype(str).str.title()
    if "event_type" not in noaa.columns and "event_type" in noaa.columns:
        pass  # already lowercase
    if "magnitude" not in noaa.columns:
        noaa["magnitude"] = 0
    
    # Filter to test period and significant storms only
    test_start = pd.Timestamp("2020-01-01")
    test_end   = pd.Timestamp("2024-12-31")
    
    noaa_test = noaa[(noaa["event_date"] >= test_start) &
                     (noaa["event_date"] <= test_end)].copy()
    
    # Filter to outage-causing storm types
    outage_types = [
        "thunderstorm wind", "high wind", "ice storm", "blizzard",
        "winter storm", "heavy snow", "tornado", "tropical storm",
        "hurricane", "freezing rain", "strong wind"
    ]
    pattern = "|".join(outage_types)
    noaa_test = noaa_test[
        noaa_test["event_type"].astype(str).str.lower().str.contains(pattern, na=False, regex=True)
    ]
    print(f"  Filtered to {len(noaa_test):,} outage-relevant storms in test period")
    
    # Compute baselines from training period
    print("\nComputing county baselines (using pre-2020 data only)...")
    baselines = load_county_baselines(eagle)
    print(f"  {len(baselines)} county baselines computed")
    
    # Backtest loop
    print("\nBacktesting predictions vs actual outcomes...")
    results = []
    target_county_names = set([f"{c}, {s}" for c,s in TARGET_COUNTIES])
    
    for _, storm in noaa_test.iterrows():
        # Build county key
        county_name = str(storm.get("county", "") or storm.get("cz_name", ""))
        state_name  = str(storm.get("state", "") or "")
        # NOAA often uses uppercase state names
        state_name = state_name.title()
        county_key = f"{county_name}, {state_name}"
        
        if county_key not in target_county_names:
            continue
        if county_key not in baselines:
            continue
        
        # Predict
        tier = classify_storm_tier(storm)
        mag = storm.get("magnitude", 0) or 0
        predicted = predict_customers(tier, baselines[county_key], mag)
        
        # Find actual outcome - max customers out in the 3-day window after storm
        storm_date = storm["event_date"]
        window_end = storm_date + timedelta(days=3)
        
        actual_rows = eagle[
            (eagle["county"] == county_name) &
            (eagle["state"] == state_name) &
            (eagle["date"] >= storm_date) &
            (eagle["date"] <= window_end)
        ]
        
        if len(actual_rows) == 0:
            continue
        
        actual_max = float(actual_rows["max_customers_out"].max())
        actual_was_major    = actual_max >= 1000
        actual_was_critical = actual_max >= 10000
        predicted_was_major    = predicted >= 1000
        predicted_was_critical = predicted >= 10000
        
        # Compute confidence bounds (same as live system)
        ci_pct = 0.75 if tier == "SEVERE" else 0.65 if tier == "MODERATE" else 0.55
        ci_low  = predicted * (1 - ci_pct)
        ci_high = predicted * (1 + ci_pct)
        in_ci = ci_low <= actual_max <= ci_high
        
        # Error metrics
        if actual_max > 0:
            pct_err = abs(predicted - actual_max) / actual_max * 100
        else:
            pct_err = 0 if predicted == 0 else 100
        
        results.append({
            "storm_date":             storm_date,
            "county":                 county_name,
            "state":                  state_name,
            "event_type":             storm.get("event_type"),
            "storm_tier":             tier,
            "magnitude":              mag,
            "predicted_customers":    predicted,
            "actual_customers":       actual_max,
            "ci_low":                 round(ci_low),
            "ci_high":                round(ci_high),
            "in_ci":                  in_ci,
            "predicted_major":        predicted_was_major,
            "actual_major":           actual_was_major,
            "major_correct":          predicted_was_major == actual_was_major,
            "predicted_critical":     predicted_was_critical,
            "actual_critical":        actual_was_critical,
            "critical_correct":       predicted_was_critical == actual_was_critical,
            "pct_error":              round(pct_err, 1),
        })
    
    if not results:
        print("ERROR: No backtest results generated")
        print("Possible causes: NOAA county names don't match TARGET_COUNTIES,")
        print("or no storms in 2020-2024 match outage-causing types.")
        return 1
    
    df = pd.DataFrame(results)
    df.to_csv(OUT_DIR / "backtest_results.csv", index=False)
    
    print(f"\n  Backtested {len(df):,} storm-county pairs")
    
    # Compute aggregate scorecard
    scorecard = {
        "test_period":               "2020-01-01 to 2024-12-31",
        "training_period":           "2014-01-01 to 2019-12-31",
        "total_storms_tested":       len(df),
        "major_outage_accuracy_pct": round(df["major_correct"].mean() * 100, 1),
        "critical_outage_accuracy_pct": round(df["critical_correct"].mean() * 100, 1),
        "within_ci_pct":             round(df["in_ci"].mean() * 100, 1),
        "median_pct_error":          round(df["pct_error"].median(), 1),
        "mean_pct_error":            round(df["pct_error"].mean(), 1),
        "by_tier": {},
        "by_state": {},
        "generated_at":              datetime.utcnow().isoformat() + "Z",
    }
    
    for tier in ["SEVERE", "MODERATE", "MINOR"]:
        sub = df[df["storm_tier"] == tier]
        if len(sub) > 0:
            scorecard["by_tier"][tier] = {
                "n":                     len(sub),
                "major_accuracy_pct":    round(sub["major_correct"].mean() * 100, 1),
                "critical_accuracy_pct": round(sub["critical_correct"].mean() * 100, 1),
                "median_pct_error":      round(sub["pct_error"].median(), 1),
                "within_ci_pct":         round(sub["in_ci"].mean() * 100, 1),
            }
    
    for state in df["state"].unique():
        sub = df[df["state"] == state]
        scorecard["by_state"][state] = {
            "n":                  len(sub),
            "major_accuracy_pct": round(sub["major_correct"].mean() * 100, 1),
        }
    
    with open(OUT_DIR / "backtest_scorecard.json", "w") as f:
        json.dump(scorecard, f, indent=2, default=str)
    
    # Print report
    print(f"\n{'=' * 70}")
    print("HISTORICAL BACKTEST SCORECARD")
    print(f"{'=' * 70}")
    print(f"Test period:         {scorecard['test_period']}")
    print(f"Storms tested:       {scorecard['total_storms_tested']:,}")
    print(f"Major outage accuracy:    {scorecard['major_outage_accuracy_pct']}%")
    print(f"Critical outage accuracy: {scorecard['critical_outage_accuracy_pct']}%")
    print(f"Within confidence:        {scorecard['within_ci_pct']}%")
    print(f"Median % error:           {scorecard['median_pct_error']}%")
    
    print(f"\nBy storm tier:")
    for tier, stats in scorecard["by_tier"].items():
        print(f"  {tier:9s} n={stats['n']:4d}  "
              f"major acc={stats['major_accuracy_pct']:5.1f}%  "
              f"median err={stats['median_pct_error']:5.1f}%")
    
    print(f"\nSaved:")
    print(f"  {OUT_DIR / 'backtest_results.csv'}")
    print(f"  {OUT_DIR / 'backtest_scorecard.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
