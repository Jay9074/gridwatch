"""
GridWatch Storm Watch - Restoration Time and Crew Estimator
For each predicted storm event, estimate:
- Restoration time (hours)
- Number of crews needed
- Total restoration cost (rough)

This provides industry-standard "impact statistics, damage and 
restoration effort estimates."

Run: python src/stormwatch/restoration_estimator.py
"""
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime

PREDICT_DIR = Path("data/stormwatch/predictions")
PROC_DIR    = Path("data/processed")

# Industry-standard parameters from utility operations research:
# Average crew can restore ~200 customers/hour during a storm response
# Average crew cost: $250-400/hour including overtime + equipment
# Critical events (>10K customers) require mutual aid from neighboring utilities

CREW_RESTORATION_RATE_PER_HOUR = 200      # customers per crew per hour
COST_PER_CREW_HOUR             = 325      # USD, mid-range estimate
TARGET_RESTORATION_HOURS       = 24       # most utilities target 24h restoration

# Storm tier multipliers for restoration difficulty
TIER_DIFFICULTY = {
    "SEVERE":   1.8,   # storms slow down crews (ice/wind/blocked roads)
    "MODERATE": 1.3,
    "MINOR":    1.0,
}


def load_historical_restoration():
    """Compute average restoration hours per state from EAGLE-I.
    
    Uses outage_duration_hrs from your processed dataset.
    """
    try:
        df = pd.read_csv(PROC_DIR / "eaglei_daily_northeast.csv")
        df = df[df["max_customers_out"] >= 1000]  # major outages only
        baseline = df.groupby("state")["outage_duration_hrs"].agg(
            ["mean", "median", "max"]
        ).round(2)
        return baseline.to_dict("index")
    except Exception as e:
        print(f"Could not load restoration history: {e}")
        return {}


def estimate_restoration(pred_row, state_baseline):
    """Estimate restoration time, crews, and cost for one prediction."""
    customers = pred_row["predicted_customers"]
    tier = pred_row["storm_tier"]
    state = pred_row["state"]
    
    # Apply tier difficulty multiplier
    difficulty = TIER_DIFFICULTY.get(tier, 1.0)
    
    # Base restoration time (hours) — if we wanted to hit the 24h target
    crews_for_24h = max(1, np.ceil(customers / (TARGET_RESTORATION_HOURS * CREW_RESTORATION_RATE_PER_HOUR)))
    crews_for_24h = int(crews_for_24h * difficulty)
    
    # Realistic restoration estimate using state historical median
    state_data = state_baseline.get(state, {})
    historical_median = state_data.get("median", 4.0)  # hours
    
    # Scale historical median by storm severity
    realistic_restoration_hours = historical_median * difficulty
    
    # If single severe event, use historical max as upper bound
    if tier == "SEVERE" and customers > 10000:
        realistic_restoration_hours = max(realistic_restoration_hours, 12)
        if customers > 50000:
            realistic_restoration_hours = max(realistic_restoration_hours, 24)
    
    # Cost estimate
    estimated_cost = round(crews_for_24h * realistic_restoration_hours * COST_PER_CREW_HOUR)
    
    # Mutual aid needed?
    mutual_aid_needed = customers > 10000 or crews_for_24h > 50
    
    return {
        "estimated_restoration_hrs":     round(realistic_restoration_hours, 1),
        "crews_recommended":             crews_for_24h,
        "estimated_cost_usd":            estimated_cost,
        "mutual_aid_recommended":        mutual_aid_needed,
        "restoration_target_hours":      TARGET_RESTORATION_HOURS,
    }


def main():
    print("=" * 60)
    print("GridWatch - Restoration & Crew Estimator")
    print("=" * 60)
    
    pred_file = PREDICT_DIR / "active_predictions.csv"
    if not pred_file.exists():
        print(f"No predictions found at {pred_file}")
        print("Run predict_outages.py first")
        return
    
    preds = pd.read_csv(pred_file)
    if len(preds) == 0:
        print("No active predictions to estimate.")
        return
    print(f"Loaded {len(preds)} active predictions")
    
    state_baseline = load_historical_restoration()
    if state_baseline:
        print(f"Loaded historical restoration data for {len(state_baseline)} states")
    
    # Estimate restoration for each prediction
    estimates = []
    for _, pred in preds.iterrows():
        est = estimate_restoration(pred, state_baseline)
        estimates.append({**pred.to_dict(), **est})
    
    est_df = pd.DataFrame(estimates)
    
    # Save
    est_df.to_csv(PREDICT_DIR / "restoration_estimates.csv", index=False)
    
    print(f"\n{'=' * 60}")
    print("RESTORATION ESTIMATES")
    print(f"{'=' * 60}")
    
    # Summary stats
    total_customers     = est_df["predicted_customers"].sum()
    total_crews         = est_df["crews_recommended"].sum()
    total_cost          = est_df["estimated_cost_usd"].sum()
    mutual_aid_events   = est_df["mutual_aid_recommended"].sum()
    
    print(f"Total predicted customers affected: {total_customers:,}")
    print(f"Total crews recommended:            {total_crews:,}")
    print(f"Total estimated cost:               ${total_cost:,}")
    print(f"Events requiring mutual aid:        {mutual_aid_events}")
    
    # Top 5 most impactful events
    print(f"\nTop 5 events by estimated impact:")
    top = est_df.nlargest(5, "estimated_cost_usd")[
        ["county", "state", "storm_tier", "predicted_customers",
         "estimated_restoration_hrs", "crews_recommended", "estimated_cost_usd",
         "mutual_aid_recommended"]
    ]
    print(top.to_string(index=False))
    
    print(f"\nSaved: {PREDICT_DIR / 'restoration_estimates.csv'}")


if __name__ == "__main__":
    main()
