"""
GridWatch Storm Watch - Validation Tracker
Compares historical predictions against actual EAGLE-I outage data.
Builds a public accuracy scorecard.

Run weekly: python src/stormwatch/validate_predictions.py
"""
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta

PREDICT_DIR = Path("data/stormwatch/predictions")
VALID_DIR   = Path("data/stormwatch/validation")
VALID_DIR.mkdir(parents=True, exist_ok=True)
PROC_DIR    = Path("data/processed")

# EAGLE-I data has ~60 day lag. Validate predictions older than that.
VALIDATION_LAG_DAYS = 60


def main():
    print("=" * 60)
    print("GridWatch Storm Watch - Prediction Validation")
    print("=" * 60)
    
    # Load prediction log
    log_file = PREDICT_DIR / "prediction_log.csv"
    if not log_file.exists():
        print("No prediction log found. Run predict_outages.py first.")
        return
    
    preds = pd.read_csv(log_file, parse_dates=["start_time","end_time","predicted_at"])
    print(f"Loaded {len(preds)} predictions from log")
    
    # Filter to predictions that are old enough to validate
    cutoff = datetime.utcnow() - timedelta(days=VALIDATION_LAG_DAYS)
    cutoff_aware = pd.Timestamp(cutoff, tz="UTC")
    
    validatable = preds[preds["start_time"] < cutoff_aware].copy()
    print(f"Validatable (>{VALIDATION_LAG_DAYS} days old): {len(validatable)}")
    
    if len(validatable) == 0:
        print("No predictions old enough to validate yet.")
        print(f"Wait until predictions from before {cutoff.date()} accumulate.")
        return
    
    # Load actual outage data
    actual_file = PROC_DIR / "eaglei_daily_northeast.csv"
    if not actual_file.exists():
        print(f"Actual data not found at {actual_file}")
        print("Re-run load_data_fixed.py to refresh EAGLE-I data")
        return
    
    actuals = pd.read_csv(actual_file)
    actuals["date"] = pd.to_datetime(actuals["date"])
    print(f"Loaded actual outage data: {len(actuals):,} county-days")
    
    # For each prediction, find matching actual outage
    results = []
    for _, pred in validatable.iterrows():
        # Find actual outage for this county on this date
        date_start = pd.Timestamp(pred["start_time"]).tz_localize(None).normalize()
        date_end   = pd.Timestamp(pred["end_time"]).tz_localize(None).normalize()
        
        county_actuals = actuals[
            (actuals["county"] == pred["county"]) &
            (actuals["state"] == pred["state"]) &
            (actuals["date"] >= date_start) &
            (actuals["date"] <= date_end)
        ]
        
        if len(county_actuals) == 0:
            actual_peak = 0
            actual_was_major = False
            actual_was_critical = False
        else:
            actual_peak = county_actuals["max_customers_out"].max()
            actual_was_major    = bool(actual_peak >= 1000)
            actual_was_critical = bool(actual_peak >= 10000)
        
        # Compute accuracy metrics
        pred_count = pred["predicted_customers"]
        in_ci = bool(pred["ci_low"] <= actual_peak <= pred["ci_high"])
        
        # Major outage classification accuracy
        pred_major = bool(pred["is_major_outage_likely"])
        major_correct = pred_major == actual_was_major
        
        # Critical outage classification
        pred_critical = bool(pred["is_critical_outage_likely"])
        critical_correct = pred_critical == actual_was_critical
        
        # Percent error on customer count
        if actual_peak > 0:
            pct_error = abs(pred_count - actual_peak) / actual_peak * 100
        else:
            pct_error = 0 if pred_count == 0 else 100
        
        results.append({
            "prediction_id":     pred["prediction_id"],
            "county":            pred["county"],
            "state":             pred["state"],
            "storm_tier":        pred["storm_tier"],
            "start_time":        pred["start_time"],
            "predicted_customers": pred_count,
            "actual_customers":  actual_peak,
            "in_confidence_interval": in_ci,
            "predicted_major":   pred_major,
            "actual_major":      actual_was_major,
            "major_correct":     major_correct,
            "predicted_critical":pred_critical,
            "actual_critical":   actual_was_critical,
            "critical_correct":  critical_correct,
            "pct_error":         round(pct_error, 1),
        })
    
    val_df = pd.DataFrame(results)
    val_df.to_csv(VALID_DIR / "validation_results.csv", index=False)
    
    # Compute aggregate accuracy metrics
    n = len(val_df)
    
    summary = {
        "total_predictions_validated": n,
        "ci_hit_rate":          round(val_df["in_confidence_interval"].mean() * 100, 1),
        "major_accuracy":       round(val_df["major_correct"].mean() * 100, 1),
        "critical_accuracy":    round(val_df["critical_correct"].mean() * 100, 1),
        "median_pct_error":     round(val_df["pct_error"].median(), 1),
        "mean_pct_error":       round(val_df["pct_error"].mean(), 1),
        "by_tier": {
            tier: {
                "n":              len(val_df[val_df["storm_tier"] == tier]),
                "major_accuracy": round(val_df[val_df["storm_tier"] == tier]["major_correct"].mean() * 100, 1) if len(val_df[val_df["storm_tier"] == tier]) > 0 else None,
                "median_error":   round(val_df[val_df["storm_tier"] == tier]["pct_error"].median(), 1) if len(val_df[val_df["storm_tier"] == tier]) > 0 else None,
            }
            for tier in ["SEVERE", "MODERATE", "MINOR"]
        },
        "validated_through":    cutoff.date().isoformat(),
        "last_updated":         datetime.utcnow().isoformat(),
    }
    
    import json
    with open(VALID_DIR / "accuracy_scorecard.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)
    
    print(f"\n{'=' * 60}")
    print("PUBLIC ACCURACY SCORECARD")
    print(f"{'=' * 60}")
    print(f"Total predictions validated:    {n}")
    print(f"Within confidence interval:     {summary['ci_hit_rate']}%")
    print(f"Major outage classification:    {summary['major_accuracy']}% correct")
    print(f"Critical outage classification: {summary['critical_accuracy']}% correct")
    print(f"Median % error in count:        {summary['median_pct_error']}%")
    print(f"\nBy storm tier:")
    for tier, stats in summary["by_tier"].items():
        if stats["n"] > 0:
            print(f"  {tier}: n={stats['n']}, "
                  f"major accuracy={stats['major_accuracy']}%, "
                  f"median error={stats['median_error']}%")
    
    print(f"\nSaved: {VALID_DIR / 'validation_results.csv'}")
    print(f"Score: {VALID_DIR / 'accuracy_scorecard.json'}")


if __name__ == "__main__":
    main()
