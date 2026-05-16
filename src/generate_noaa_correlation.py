"""
GridWatch - Generate NOAA Weather Correlation Data
Run: python src/generate_noaa_correlation.py
"""
import pandas as pd
import numpy as np
from pathlib import Path

PROC_DIR = Path("data/processed")

print("Loading EAGLE-I data...")
eaglei = pd.read_csv(PROC_DIR / "eaglei_daily_northeast.csv")
eaglei["is_major_outage"] = (eaglei["max_customers_out"] >= 1000).astype(int)

print("Loading NOAA storm data...")
noaa = pd.read_csv(PROC_DIR / "noaa_storms_northeast.csv", low_memory=False)
noaa.columns = noaa.columns.str.lower().str.strip()

# Parse NOAA date
date_col = next((c for c in noaa.columns if "begin" in c and "date" in c), None)
if date_col:
    noaa["event_date"] = pd.to_datetime(noaa[date_col], errors="coerce")
    noaa["year"]  = noaa["event_date"].dt.year
    noaa["month"] = noaa["event_date"].dt.month

# Find state column
state_col = next((c for c in noaa.columns if c == "state"), None)
evt_col   = next((c for c in noaa.columns if "event_type" in c.lower()), None)

print(f"NOAA columns: {list(noaa.columns[:10])}")
print(f"State col: {state_col}, Event col: {evt_col}")

# Storm severity weights
severity = {
    "Ice Storm":5,"Blizzard":5,"Winter Storm":4,
    "Extreme Cold/Wind Chill":4,"Tornado":5,
    "Hurricane (Typhoon)":5,"Tropical Storm":4,
    "High Wind":3,"Thunderstorm Wind":3,
    "Heavy Snow":3,"Flood":3,"Flash Flood":4,
    "Lightning":2,"Heavy Rain":2
}

if evt_col:
    noaa["severity"] = noaa[evt_col].map(severity).fillna(2)

# State name mapping
state_map = {
    "MAINE":"Maine","NEW HAMPSHIRE":"New Hampshire",
    "VERMONT":"Vermont","MASSACHUSETTS":"Massachusetts",
    "RHODE ISLAND":"Rhode Island","CONNECTICUT":"Connecticut",
    "NEW YORK":"New York","NEW JERSEY":"New Jersey",
    "PENNSYLVANIA":"Pennsylvania"
}
if state_col:
    noaa["state_clean"] = noaa[state_col].str.upper().map(state_map).fillna(noaa[state_col])
else:
    noaa["state_clean"] = "Unknown"

# Aggregate NOAA to state-month
noaa_agg = noaa.groupby(["state_clean","year","month"]).agg(
    storm_count   = (evt_col if evt_col else "severity", "count"),
    max_severity  = ("severity", "max"),
    mean_severity = ("severity", "mean"),
).reset_index().rename(columns={"state_clean":"state"})

# Aggregate EAGLE-I to state-month
eaglei_agg = eaglei.groupby(["state","year","month"]).agg(
    outage_days   = ("is_major_outage","sum"),
    total_days    = ("is_major_outage","count"),
    avg_customers = ("max_customers_out","mean"),
).reset_index()
eaglei_agg["outage_rate"] = eaglei_agg["outage_days"] / eaglei_agg["total_days"]

# Merge
merged = eaglei_agg.merge(noaa_agg, on=["state","year","month"], how="left")
merged["storm_count"]   = merged["storm_count"].fillna(0)
merged["max_severity"]  = merged["max_severity"].fillna(0)
merged["mean_severity"] = merged["mean_severity"].fillna(0)

# Save
merged.to_csv(PROC_DIR / "noaa_correlation.csv", index=False)
print(f"\nSaved: data/processed/noaa_correlation.csv ({len(merged):,} rows)")

# Print correlation stats
corr_storm   = merged["storm_count"].corr(merged["outage_rate"])
corr_severity= merged["max_severity"].corr(merged["outage_rate"])
print(f"\nCorrelation — storm count vs outage rate:    {corr_storm:.3f}")
print(f"Correlation — max severity vs outage rate:   {corr_severity:.3f}")

# Storm type breakdown
if evt_col:
    storm_types = noaa.groupby(evt_col).agg(
        count=("severity","count"),
        avg_severity=("severity","mean")
    ).reset_index().sort_values("count", ascending=False).head(10)
    print("\nTop storm types in Northeast:")
    print(storm_types.to_string(index=False))

# State x storm summary
print("\nAvg storms per month by state:")
state_storms = merged.groupby("state")["storm_count"].mean().sort_values(ascending=False)
print(state_storms.round(2).to_string())
