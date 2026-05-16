"""
GridWatch Storm Watch - Outage Prediction Engine (CALIBRATED v2)
For each detected storm event, predict customers affected per county.

CALIBRATION CHANGES (v2):
- Uses MEDIAN historical outage (not peak) as baseline
- Storm severity multipliers tuned against real EAGLE-I data
- Caps predictions at realistic upper bounds per tier
- Adds "is_outlier_event" flag for predictions that exceed typical range

Run: python src/stormwatch/predict_outages.py
"""
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime

STORM_DIR    = Path("data/stormwatch/storms")
PREDICT_DIR  = Path("data/stormwatch/predictions")
PREDICT_DIR.mkdir(parents=True, exist_ok=True)
PROC_DIR     = Path("data/processed")


def load_county_baselines():
    """Load county baselines using MEDIAN customer-affected per outage day.
    
    Previously used peak (worst-ever event) which led to massive overprediction.
    Median represents what a TYPICAL major outage looks like.
    """
    # First try to load county_risk_summary for outage rate / peak
    try:
        summary = pd.read_csv("data/summary/county_risk_summary.csv")
    except Exception as e:
        print(f"Could not load county_risk_summary: {e}")
        return {}
    
    # Now load raw county-day data to compute realistic median
    try:
        raw = pd.read_csv(PROC_DIR / "eaglei_daily_northeast.csv")
        # Filter to only days that HAD a major outage
        major_days = raw[raw["max_customers_out"] >= 1000]
        # Get median customers-out per county on major outage days
        county_median = major_days.groupby(["county","state"]).agg(
            median_customers_outage=("max_customers_out", "median"),
            p75_customers_outage=("max_customers_out", lambda x: x.quantile(0.75)),
            p95_customers_outage=("max_customers_out", lambda x: x.quantile(0.95)),
        ).reset_index()
    except Exception as e:
        print(f"Could not load raw data: {e}")
        county_median = pd.DataFrame()
    
    baseline = {}
    for _, row in summary.iterrows():
        key = f"{row['county']}, {row['state']}"
        
        # Look up realistic baselines from raw data
        match = county_median[
            (county_median["county"] == row["county"]) &
            (county_median["state"] == row["state"])
        ]
        
        if len(match) > 0:
            typical_outage = match.iloc[0]["median_customers_outage"]
            high_outage    = match.iloc[0]["p75_customers_outage"]
            extreme_outage = match.iloc[0]["p95_customers_outage"]
        else:
            # Fallback for counties without major outages
            typical_outage = 1500
            high_outage    = 3000
            extreme_outage = 8000
        
        baseline[key] = {
            "typical_major_outage":  typical_outage,
            "high_outage":           high_outage,
            "extreme_outage":        extreme_outage,
            "all_time_peak":         row["peak_customers_out"],
            "outage_rate":           row["outage_rate"],
            "composite_risk":        row.get("composite_risk_score", 0.5)
        }
    return baseline


def predict_storm_outage(storm_row, baseline):
    """Predict outage impact for a storm event.
    
    CALIBRATED LOGIC:
    - MINOR storm: baseline = typical major outage (county median)
    - MODERATE storm: baseline = 75th percentile (high but not extreme)
    - SEVERE storm: baseline = 95th percentile (worst case for normal storms)
    - Caps at historical peak for true outliers
    """
    county_key = f"{storm_row['county']}, {storm_row['state']}"
    if county_key not in baseline:
        return None
    
    base = baseline[county_key]
    
    # Pick baseline based on storm tier
    tier = storm_row["storm_tier"]
    if tier == "MINOR":
        baseline_customers = base["typical_major_outage"] * 0.6   # below median
    elif tier == "MODERATE":
        baseline_customers = base["typical_major_outage"]         # median major outage
    elif tier == "SEVERE":
        baseline_customers = base["high_outage"]                  # 75th percentile
    else:
        baseline_customers = base["typical_major_outage"] * 0.5
    
    # Duration multiplier (small effect - capped tightly)
    duration = storm_row.get("duration_hrs", 6)
    duration_mult = 1.0 + (duration - 6) * 0.03   # +3% per hour over 6
    duration_mult = max(0.7, min(duration_mult, 2.0))   # cap between 0.7x and 2x
    
    # Wind multiplier (small effect, only matters for severe wind)
    wind = storm_row.get("max_wind_mph", 0)
    if wind >= 60:
        wind_mult = 1.6
    elif wind >= 45:
        wind_mult = 1.3
    elif wind >= 30:
        wind_mult = 1.1
    else:
        wind_mult = 1.0
    
    # Combine
    predicted_customers = baseline_customers * duration_mult * wind_mult
    
    # Cap at realistic upper bound (not hurricane-level for routine storms)
    if tier == "MINOR":
        upper_cap = base["typical_major_outage"] * 2
    elif tier == "MODERATE":
        upper_cap = base["high_outage"] * 1.5
    elif tier == "SEVERE":
        upper_cap = base["extreme_outage"]
    else:
        upper_cap = base["typical_major_outage"] * 1.5
    
    predicted_customers = min(predicted_customers, upper_cap)
    
    # Floor (don't predict zero for a real storm)
    predicted_customers = max(predicted_customers, 200)
    
    # Confidence intervals
    if tier == "SEVERE":
        ci_pct = 0.45
    elif tier == "MODERATE":
        ci_pct = 0.35
    else:
        ci_pct = 0.30
    
    # Flag if prediction is much higher than typical (real outlier event possible)
    is_outlier = predicted_customers > base["typical_major_outage"] * 3
    
    return {
        "predicted_customers":      round(predicted_customers),
        "ci_low":                   round(predicted_customers * (1 - ci_pct)),
        "ci_high":                  round(predicted_customers * (1 + ci_pct)),
        "confidence_level":         "LOW" if ci_pct >= 0.45 else "MEDIUM" if ci_pct >= 0.35 else "HIGH",
        "baseline_used":            round(baseline_customers),
        "duration_multiplier":      round(duration_mult, 2),
        "wind_multiplier":          round(wind_mult, 2),
        "is_major_outage_likely":   predicted_customers >= 1000,
        "is_critical_outage_likely":predicted_customers >= 10000,
        "is_outlier_prediction":    is_outlier,
        "typical_county_outage":    round(base["typical_major_outage"]),
        "county_all_time_peak":     round(base["all_time_peak"]),
    }


def main():
    print("=" * 60)
    print("GridWatch Storm Watch - Outage Prediction (CALIBRATED v2)")
    print("=" * 60)
    
    storms_file = STORM_DIR / "active_storms.csv"
    if not storms_file.exists():
        print("Run detect_storms.py first")
        return
    
    storms = pd.read_csv(storms_file, parse_dates=["start_time","end_time"])
    if len(storms) == 0:
        print("No storms to predict")
        pd.DataFrame().to_csv(PREDICT_DIR / "active_predictions.csv", index=False)
        return
    print(f"Loaded {len(storms)} storm events")
    
    baseline = load_county_baselines()
    if not baseline:
        print("Could not load baselines")
        return
    print(f"Loaded baselines for {len(baseline)} counties")
    
    # Print calibration stats
    typical_outages = [b["typical_major_outage"] for b in baseline.values()]
    print(f"\nCalibration check:")
    print(f"  Median typical outage across counties: {np.median(typical_outages):,.0f} customers")
    print(f"  Range: {min(typical_outages):,.0f} - {max(typical_outages):,.0f}")
    
    # Predict
    predictions = []
    for _, storm in storms.iterrows():
        pred = predict_storm_outage(storm, baseline)
        if pred is None:
            continue
        predictions.append({
            **storm.to_dict(),
            **pred,
            "predicted_at": datetime.utcnow().isoformat(),
            "prediction_id": f"{storm['county']}_{storm['state']}_{storm['start_time'].strftime('%Y%m%d%H')}".replace(" ","_").replace(",","")
        })
    
    if not predictions:
        print("No predictions generated")
        return
    
    pred_df = pd.DataFrame(predictions)
    pred_df = pred_df.sort_values(["peak_severity","predicted_customers"], ascending=[False, False])
    
    pred_df.to_csv(PREDICT_DIR / "active_predictions.csv", index=False)
    ts = datetime.utcnow().strftime("%Y%m%d_%H%M")
    pred_df.to_csv(PREDICT_DIR / f"predictions_{ts}.csv", index=False)
    
    # Cumulative log
    log_file = PREDICT_DIR / "prediction_log.csv"
    if log_file.exists():
        existing = pd.read_csv(log_file)
        combined = pd.concat([existing, pred_df], ignore_index=True)
        combined = combined.drop_duplicates(subset=["prediction_id"], keep="last")
    else:
        combined = pred_df
    combined.to_csv(log_file, index=False)
    
    print(f"\n{'=' * 60}")
    print("CALIBRATED PREDICTIONS")
    print(f"{'=' * 60}")
    print(f"Storms predicted: {len(pred_df)}")
    print(f"Total customers at risk: {pred_df['predicted_customers'].sum():,.0f}")
    print(f"Major outages predicted: {pred_df['is_major_outage_likely'].sum()}")
    print(f"Critical outages predicted: {pred_df['is_critical_outage_likely'].sum()}")
    print(f"Outlier predictions flagged: {pred_df['is_outlier_prediction'].sum()}")
    
    print(f"\nPrediction range:")
    print(f"  Min:    {pred_df['predicted_customers'].min():,}")
    print(f"  Median: {pred_df['predicted_customers'].median():,}")
    print(f"  Max:    {pred_df['predicted_customers'].max():,}")
    
    print(f"\nTop 5 events by predicted customers:")
    top = pred_df.head(5)[
        ["county","state","storm_tier","predicted_customers",
         "ci_low","ci_high","typical_county_outage","primary_trigger"]
    ]
    print(top.to_string(index=False))
    
    print(f"\nSaved: {PREDICT_DIR / 'active_predictions.csv'}")


if __name__ == "__main__":
    main()
