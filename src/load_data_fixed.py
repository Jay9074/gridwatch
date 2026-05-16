"""
GridWatch - FIXED Data Loader
Loads ALL 12 months from each year (2014-2025).
Aggregates to county-day level for the 9 Northeast states.

Run: python src/load_data_fixed.py
"""
import pandas as pd
import numpy as np
import os
import gc
from pathlib import Path
from datetime import datetime

RAW_DIR  = Path("data/raw")
PROC_DIR = Path("data/processed")
PROC_DIR.mkdir(parents=True, exist_ok=True)

NORTHEAST_STATES = [
    "Maine", "New Hampshire", "Vermont", "Massachusetts",
    "Rhode Island", "Connecticut", "New York", "New Jersey", "Pennsylvania"
]

YEARS = list(range(2014, 2026))

def get_season(month):
    if month in [12, 1, 2]:    return "Winter"
    if month in [3, 4, 5]:     return "Spring"
    if month in [6, 7, 8]:     return "Summer"
    if month in [9, 10, 11]:   return "Fall"
    return "Unknown"

print("="*60)
print("GridWatch - Loading FULL EAGLE-I Data (all 12 months)")
print("="*60)

all_county_days = []

for year in YEARS:
    fpath = RAW_DIR / f"eaglei_outages_{year}.csv"
    if not fpath.exists():
        print(f"\n{year}: file not found, skipping")
        continue

    size_mb = os.path.getsize(fpath) / (1024 * 1024)
    print(f"\n{year}: loading {fpath.name} ({size_mb:.0f} MB)...")
    
    chunks = []
    chunk_iter = pd.read_csv(fpath, chunksize=2_000_000, low_memory=False)
    
    for i, chunk in enumerate(chunk_iter):
        # Handle different column names across years
        if "customers_out" not in chunk.columns and "sum" in chunk.columns:
            chunk = chunk.rename(columns={"sum": "customers_out"})
        
        # Filter to Northeast states
        chunk = chunk[chunk["state"].isin(NORTHEAST_STATES)]
        if len(chunk) == 0:
            continue
        
        # Parse timestamp
        chunk["run_start_time"] = pd.to_datetime(
            chunk["run_start_time"], errors="coerce"
        )
        chunk = chunk.dropna(subset=["run_start_time"])
        
        # Extract date components
        chunk["date"]  = chunk["run_start_time"].dt.date
        chunk["year"]  = chunk["run_start_time"].dt.year
        chunk["month"] = chunk["run_start_time"].dt.month
        
        # Ensure customers_out is numeric
        chunk["customers_out"] = pd.to_numeric(
            chunk["customers_out"], errors="coerce"
        ).fillna(0)
        
        chunks.append(chunk[["fips_code", "county", "state", "date",
                             "year", "month", "customers_out"]])
        
        if (i+1) % 5 == 0:
            print(f"  processed {(i+1)*2_000_000:,} rows so far...")
    
    if not chunks:
        print(f"  no Northeast data found in {year}")
        continue
    
    year_df = pd.concat(chunks, ignore_index=True)
    del chunks
    gc.collect()
    
    print(f"  {year}: {len(year_df):,} Northeast rows")
    print(f"  Months in data: {sorted(year_df['month'].unique().tolist())}")
    
    # Aggregate to county-day
    county_day = year_df.groupby(
        ["fips_code", "county", "state", "date", "year", "month"],
        observed=True
    ).agg(
        max_customers_out=("customers_out", "max"),
        mean_customers_out=("customers_out", "mean"),
        outage_intervals=("customers_out", lambda x: (x > 0).sum()),
        total_customer_hours=("customers_out", lambda x: x.sum() * 0.25),
    ).reset_index()
    
    county_day["season"] = county_day["month"].apply(get_season)
    
    print(f"  {year}: {len(county_day):,} county-days produced")
    all_county_days.append(county_day)
    
    del year_df, county_day
    gc.collect()

print("\n" + "="*60)
print("Combining all years...")
print("="*60)

combined = pd.concat(all_county_days, ignore_index=True)
del all_county_days
gc.collect()

# Add derived flags
combined["is_major_outage"]    = (combined["max_customers_out"] >= 1000).astype(int)
combined["is_critical_outage"] = (combined["max_customers_out"] >= 10000).astype(int)
combined["log_customers_out"]  = np.log1p(combined["max_customers_out"])
combined["outage_duration_hrs"] = combined["outage_intervals"] * 0.25

# Save
out_path = PROC_DIR / "eaglei_daily_northeast.csv"
combined.to_csv(out_path, index=False)
size_mb = os.path.getsize(out_path) / (1024*1024)

print(f"\nSaved: {out_path} ({size_mb:.1f} MB)")
print(f"Total county-days: {len(combined):,}")
print(f"\nMonths per year:")
for yr in sorted(combined["year"].unique().astype(int)):
    months = sorted(combined[combined["year"]==yr]["month"].unique().astype(int).tolist())
    print(f"  {yr}: {len(months)} months {months}")

print(f"\nMajor outage rate: {combined['is_major_outage'].mean():.2%}")
print(f"Critical outage rate: {combined['is_critical_outage'].mean():.2%}")
print(f"Peak event: {combined['max_customers_out'].max():,.0f} customers")
print(f"\nSeasonal breakdown:")
print(combined.groupby("season")["is_major_outage"].agg(["mean","sum","count"]).round(4))
print(f"\nState breakdown:")
print(combined.groupby("state")["is_major_outage"].agg(["mean","sum","count"]).round(4).sort_values("mean", ascending=False))

print("\n" + "="*60)
print("DONE - Full data ready for model training")
print("="*60)
