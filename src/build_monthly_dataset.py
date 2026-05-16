"""
GridWatch v2 - Build Monthly State-Level Dataset
Creates a 132 months x 9 states = 1,188 row dataset for regression forecasting.
Run: python src/build_monthly_dataset.py
"""
import pandas as pd
import numpy as np
from pathlib import Path

PROC_DIR = Path("data/processed")
PROC_DIR.mkdir(parents=True, exist_ok=True)

print("="*60)
print("GridWatch v2 - Building Monthly State Dataset")
print("="*60)

# Load full county-day data
df = pd.read_csv(PROC_DIR / "eaglei_daily_northeast.csv")
df["is_major_outage"]    = (df["max_customers_out"] >= 1000).astype(int)
df["is_critical_outage"] = (df["max_customers_out"] >= 10000).astype(int)
print(f"\nLoaded: {len(df):,} county-days")

# Aggregate to state-month level
print("\nAggregating to state-month level...")
state_month = df.groupby(["state","year","month"]).agg(
    n_counties              = ("county", "nunique"),
    n_county_days           = ("is_major_outage", "count"),
    major_outage_days       = ("is_major_outage", "sum"),
    critical_outage_days    = ("is_critical_outage", "sum"),
    total_customer_hours    = ("total_customer_hours", "sum"),
    max_customers_out_peak  = ("max_customers_out", "max"),
    avg_customers_out       = ("max_customers_out", "mean"),
).reset_index()

# TARGET VARIABLES (what we'll predict)
state_month["outage_rate"]    = state_month["major_outage_days"] / state_month["n_county_days"]
state_month["log_outage_days"] = np.log1p(state_month["major_outage_days"])
state_month["log_cust_hours"]  = np.log1p(state_month["total_customer_hours"])

# Date features
state_month["date"]      = pd.to_datetime(
    state_month[["year","month"]].assign(day=1)
)
state_month = state_month.sort_values(["state","date"]).reset_index(drop=True)

# Cyclical month
state_month["month_sin"] = np.sin(2 * np.pi * state_month["month"] / 12)
state_month["month_cos"] = np.cos(2 * np.pi * state_month["month"] / 12)

# Season indicators
state_month["is_winter"] = state_month["month"].isin([12,1,2]).astype(int)
state_month["is_spring"] = state_month["month"].isin([3,4,5]).astype(int)
state_month["is_summer"] = state_month["month"].isin([6,7,8]).astype(int)
state_month["is_fall"]   = state_month["month"].isin([9,10,11]).astype(int)

# Year trend
state_month["year_trend"] = state_month["year"] - state_month["year"].min()

# State encoding
state_to_id = {s:i for i, s in enumerate(sorted(state_month["state"].unique()))}
state_month["state_id"] = state_month["state"].map(state_to_id)

# State vulnerability score
STATE_RISK = {
    "Maine":0.87,"Vermont":0.78,"New Hampshire":0.75,
    "New York":0.72,"Pennsylvania":0.68,"Massachusetts":0.65,
    "Connecticut":0.61,"New Jersey":0.60,"Rhode Island":0.58
}
state_month["state_risk"] = state_month["state"].map(STATE_RISK)

# LAG FEATURES (per state, time-shifted to prevent leakage)
print("Building lag features...")
for lag in [1, 2, 3, 6, 12]:
    state_month[f"lag_{lag}_outage_days"] = (
        state_month.groupby("state")["major_outage_days"].shift(lag)
    )
    state_month[f"lag_{lag}_outage_rate"] = (
        state_month.groupby("state")["outage_rate"].shift(lag)
    )
    state_month[f"lag_{lag}_cust_hours"] = (
        state_month.groupby("state")["total_customer_hours"].shift(lag)
    )

# Rolling features (3, 6, 12 months — all using prior periods only)
for window in [3, 6, 12]:
    state_month[f"roll_{window}m_outage_days"] = (
        state_month.groupby("state")["major_outage_days"]
        .shift(1).rolling(window, min_periods=1).mean()
    )
    state_month[f"roll_{window}m_outage_rate"] = (
        state_month.groupby("state")["outage_rate"]
        .shift(1).rolling(window, min_periods=1).mean()
    )

# Same-month-last-year (seasonal pattern)
state_month["same_month_last_year"] = (
    state_month.groupby(["state","month"])["major_outage_days"].shift(1)
)

# State-month historical average (excluding current row)
state_month["state_month_avg"] = (
    state_month.groupby(["state","month"])["major_outage_days"]
    .transform(lambda x: x.expanding().mean().shift(1))
)

# NOAA storm features at state-month level
noaa_path = PROC_DIR / "noaa_storms_northeast.csv"
if noaa_path.exists():
    print("\nMerging NOAA storm features...")
    noaa = pd.read_csv(noaa_path, low_memory=False)
    noaa.columns = noaa.columns.str.lower().str.strip()

    SEVERITY = {
        "Ice Storm":5,"Blizzard":5,"Winter Storm":4,"Extreme Cold/Wind Chill":4,
        "Tornado":5,"Hurricane (Typhoon)":5,"Tropical Storm":4,"High Wind":3,
        "Thunderstorm Wind":3,"Heavy Snow":3,"Flood":3,"Flash Flood":4,
        "Lightning":2,"Heavy Rain":2
    }
    STATE_MAP = {
        "MAINE":"Maine","NEW HAMPSHIRE":"New Hampshire","VERMONT":"Vermont",
        "MASSACHUSETTS":"Massachusetts","RHODE ISLAND":"Rhode Island",
        "CONNECTICUT":"Connecticut","NEW YORK":"New York",
        "NEW JERSEY":"New Jersey","PENNSYLVANIA":"Pennsylvania"
    }

    evt_col = next((c for c in noaa.columns if "event_type" in c.lower()), None)
    if evt_col:
        noaa["severity"] = noaa[evt_col].map(SEVERITY).fillna(2)
        noaa["is_ice"] = noaa[evt_col].isin(["Ice Storm","Blizzard"]).astype(int)
        noaa["is_wind"] = noaa[evt_col].isin(
            ["High Wind","Thunderstorm Wind","Tornado"]).astype(int)
        noaa["is_winter_storm"] = noaa[evt_col].isin(
            ["Winter Storm","Heavy Snow","Blizzard","Ice Storm"]).astype(int)

    date_col = next((c for c in noaa.columns if "begin" in c and "date" in c), None)
    if date_col:
        noaa["event_date"] = pd.to_datetime(noaa[date_col], errors="coerce")
        noaa["year"]  = noaa["event_date"].dt.year
        noaa["month"] = noaa["event_date"].dt.month

    sc = next((c for c in noaa.columns if c == "state"), None)
    if sc:
        noaa["state"] = noaa[sc].str.upper().map(STATE_MAP).fillna(noaa[sc])

    storm_agg = noaa.groupby(["state","year","month"]).agg(
        storm_count   = ("severity", "count"),
        max_severity  = ("severity", "max"),
        mean_severity = ("severity", "mean"),
        ice_events    = ("is_ice", "sum"),
        wind_events   = ("is_wind", "sum"),
        winter_storms = ("is_winter_storm", "sum"),
    ).reset_index()
    state_month = state_month.merge(storm_agg, on=["state","year","month"], how="left")
    for col in ["storm_count","max_severity","mean_severity",
                "ice_events","wind_events","winter_storms"]:
        state_month[col] = state_month[col].fillna(0)
    print("  NOAA features merged")

# Save
state_month.to_csv(PROC_DIR / "state_monthly_dataset.csv", index=False)

print(f"\n{'='*60}")
print(f"DONE")
print(f"{'='*60}")
print(f"Total rows: {len(state_month):,} (state-months)")
print(f"Date range: {state_month['date'].min()} -> {state_month['date'].max()}")
print(f"States: {state_month['state'].nunique()}")
print(f"Months per state:")
for s in sorted(state_month["state"].unique()):
    n = (state_month["state"] == s).sum()
    print(f"  {s}: {n} months")
print(f"\nFeature columns built: {len(state_month.columns)}")
print(f"\nTarget statistics (major_outage_days):")
print(state_month["major_outage_days"].describe().round(2))

print(f"\nSaved: data/processed/state_monthly_dataset.csv")
