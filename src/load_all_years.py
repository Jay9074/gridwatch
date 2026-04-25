"""
GridWatch - Load All Years 2014-2025
======================================
Loads EAGLE-I data for all available years.
Memory safe - processes one year at a time.

Run: python src/load_all_years.py
Author: Jaykumar Patel
"""

import pandas as pd
import numpy as np
from pathlib import Path

RAW_DIR  = Path("data/raw")
PROC_DIR = Path("data/processed")
PROC_DIR.mkdir(parents=True, exist_ok=True)

NORTHEAST = [
    "Maine", "New Hampshire", "Vermont", "Massachusetts",
    "Rhode Island", "Connecticut", "New York", "New Jersey", "Pennsylvania"
]

# All possible years
YEARS = list(range(2014, 2026))

# Max rows per year — increase if computer handles it
ROWS_PER_YEAR = 2_000_000

print("=" * 55)
print("GridWatch — Loading EAGLE-I 2014-2025")
print("=" * 55)

# ── Check which files exist ───────────────────────────────────────
print("\nChecking available files...")
available = []
for yr in YEARS:
    p = RAW_DIR / f"eaglei_outages_{yr}.csv"
    if p.exists():
        size_mb = p.stat().st_size / 1_000_000
        print(f"  {yr}: Found ({size_mb:.0f} MB)")
        available.append(yr)
    else:
        print(f"  {yr}: Not found — skipping")

print(f"\nFound {len(available)} years: {available}")

# ── Load one year at a time ───────────────────────────────────────
print("\nLoading data year by year...")
frames = []
total_rows = 0

for yr in available:
    p = RAW_DIR / f"eaglei_outages_{yr}.csv"
    print(f"  Loading {yr}...", end=" ", flush=True)

    try:
        df = pd.read_csv(p, low_memory=False, nrows=ROWS_PER_YEAR)
        df.columns = df.columns.str.lower().str.strip()

        # Find and filter state column
        state_col = next((c for c in df.columns if "state" in c), None)
        if state_col:
            df = df[df[state_col].isin(NORTHEAST)].copy()
            if state_col != "state":
                df = df.rename(columns={state_col: "state"})

        # Standardize column names
        rename = {
            "fips_code":          "fips",
            "run_start_time":     "timestamp",
            "recorded_date":      "timestamp",
            "customers_out":      "customers_out",
            "sum(customers_out)": "customers_out",
        }
        df = df.rename(columns={k: v for k, v in rename.items() if k in df.columns})
        df["source_year"] = yr
        frames.append(df)
        total_rows += len(df)
        print(f"{len(df):,} rows (total so far: {total_rows:,})")

    except MemoryError:
        print(f"MEMORY ERROR on {yr} — reducing rows")
        try:
            df = pd.read_csv(p, low_memory=False, nrows=500_000)
            df.columns = df.columns.str.lower().str.strip()
            state_col = next((c for c in df.columns if "state" in c), None)
            if state_col:
                df = df[df[state_col].isin(NORTHEAST)].copy()
            df["source_year"] = yr
            frames.append(df)
            print(f"  Loaded {len(df):,} rows (reduced)")
        except Exception as e:
            print(f"  Skipping {yr}: {e}")

    except Exception as e:
        print(f"ERROR: {e}")

if not frames:
    print("No data loaded!")
    exit()

# ── Combine all years ─────────────────────────────────────────────
print(f"\nCombining {len(frames)} years...")
eaglei = pd.concat(frames, ignore_index=True)
print(f"Total rows: {len(eaglei):,}")

# Free memory
del frames

# ── Parse timestamps ──────────────────────────────────────────────
print("Parsing timestamps...")
if "timestamp" in eaglei.columns:
    eaglei["timestamp"] = pd.to_datetime(eaglei["timestamp"], errors="coerce")
    eaglei["date"]   = eaglei["timestamp"].dt.date
    eaglei["year"]   = eaglei["timestamp"].dt.year
    eaglei["month"]  = eaglei["timestamp"].dt.month
    eaglei["season"] = eaglei["month"].map({
        12:"Winter", 1:"Winter",  2:"Winter",
        3:"Spring",  4:"Spring",  5:"Spring",
        6:"Summer",  7:"Summer",  8:"Summer",
        9:"Fall",    10:"Fall",   11:"Fall"
    })
    print(f"Date range: {eaglei['timestamp'].min()} → {eaglei['timestamp'].max()}")

# Clean customers_out
if "customers_out" in eaglei.columns:
    eaglei["customers_out"] = pd.to_numeric(
        eaglei["customers_out"], errors="coerce"
    ).fillna(0)

# ── Aggregate to daily ────────────────────────────────────────────
print("Aggregating to daily county level...")
group_cols = [c for c in ["fips","county","state","date","year","month","season"]
              if c in eaglei.columns]

daily = eaglei.groupby(group_cols).agg(
    max_customers_out    = ("customers_out", "max"),
    mean_customers_out   = ("customers_out", "mean"),
    outage_intervals     = ("customers_out", lambda x: (x > 0).sum()),
    total_customer_hours = ("customers_out", lambda x: x.sum() * 0.25),
).reset_index()

# Free memory
del eaglei

daily["is_major_outage"]    = (daily["max_customers_out"] >= 1_000).astype(int)
daily["is_critical_outage"] = (daily["max_customers_out"] >= 10_000).astype(int)
daily["log_customers_out"]  = np.log1p(daily["max_customers_out"])
daily["outage_duration_hrs"]= daily["outage_intervals"] * 0.25

print(f"County-days: {len(daily):,}")

# ── Save ──────────────────────────────────────────────────────────
print("Saving processed file...")
daily.to_csv(PROC_DIR / "eaglei_daily_northeast.csv", index=False)
print("Saved: data/processed/eaglei_daily_northeast.csv")

# ── Summary ───────────────────────────────────────────────────────
print("\n" + "=" * 55)
print("SUMMARY")
print("=" * 55)
print(f"Years loaded       : {sorted(daily['year'].dropna().unique().tolist())}")
print(f"County-days        : {len(daily):,}")
print(f"States             : {daily['state'].nunique()}")
print(f"Major outages      : {daily['is_major_outage'].sum():,} ({daily['is_major_outage'].mean():.2%})")
print(f"Critical outages   : {daily['is_critical_outage'].sum():,} ({daily['is_critical_outage'].mean():.2%})")
print(f"Peak customers out : {daily['max_customers_out'].max():,.0f}")
print(f"\nOutages by season:")
if "season" in daily.columns:
    for season, grp in daily.groupby("season"):
        print(f"  {season:<8}: {grp['is_major_outage'].sum():,} days ({grp['is_major_outage'].mean():.2%})")

print("\nNext steps:")
print("  1. python src/generate_summary.py")
print("  2. python src/model.py")
