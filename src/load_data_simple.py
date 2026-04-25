"""
GridWatch - Simple Data Loader
================================
Loads 2020-2023 EAGLE-I + NOAA data for Northeast US.
Memory-safe version.

Run: python src/load_data_simple.py
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

YEARS = [2020, 2021, 2022, 2023, 2024]
ROWS_PER_YEAR = 2_000_000  # Increased — your computer handled it fine

print("=" * 55)
print("GridWatch — Data Processing Pipeline")
print("=" * 55)

# ── Step 1: Show all files in raw folder ─────────────
print("\n Files found in data/raw/:")
all_files = sorted(RAW_DIR.glob("*"))
for f in all_files:
    size_mb = f.stat().st_size / 1_000_000
    print(f"  {f.name:<50} {size_mb:.1f} MB")

# ── Step 2: Load EAGLE-I ─────────────────────────────
print("\n[1/3] Loading EAGLE-I outage data...")
frames = []

for yr in YEARS:
    path = RAW_DIR / f"eaglei_outages_{yr}.csv"
    if not path.exists():
        print(f"  eaglei_outages_{yr}.csv — not found, skipping")
        continue

    print(f"  Loading {yr}...", end=" ", flush=True)
    try:
        df = pd.read_csv(path, low_memory=False, nrows=ROWS_PER_YEAR)
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
        print(f"{len(df):,} rows")

    except Exception as e:
        print(f"ERROR: {e}")

if not frames:
    print("\nNo EAGLE-I files found in data/raw/")
    print("Expected: eaglei_outages_2022.csv etc.")
else:
    eaglei = pd.concat(frames, ignore_index=True)
    print(f"\n  Total rows: {len(eaglei):,}")

    # Parse timestamps
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

    # Aggregate to daily
    if "customers_out" in eaglei.columns:
        eaglei["customers_out"] = pd.to_numeric(
            eaglei["customers_out"], errors="coerce"
        ).fillna(0)

    group_cols = [c for c in ["fips","county","state","date","year","month","season"]
                  if c in eaglei.columns]

    if group_cols and "customers_out" in eaglei.columns:
        print("  Aggregating to daily county level...")
        daily = eaglei.groupby(group_cols).agg(
            max_customers_out    = ("customers_out", "max"),
            mean_customers_out   = ("customers_out", "mean"),
            outage_intervals     = ("customers_out", lambda x: (x > 0).sum()),
            total_customer_hours = ("customers_out", lambda x: x.sum() * 0.25),
        ).reset_index()

        daily["is_major_outage"]    = (daily["max_customers_out"] >= 10_000).astype(int)
        daily["is_critical_outage"] = (daily["max_customers_out"] >= 50_000).astype(int)
        daily["log_customers_out"]  = np.log1p(daily["max_customers_out"])
        daily["outage_duration_hrs"]= daily["outage_intervals"] * 0.25

        daily.to_csv(PROC_DIR / "eaglei_daily_northeast.csv", index=False)
        print(f"  Saved: eaglei_daily_northeast.csv  ({len(daily):,} county-days)")


# ── Step 3: Load NOAA ─────────────────────────────────
print("\n[2/3] Loading NOAA storm data...")

STORM_TYPES = [
    "Winter Storm", "Ice Storm", "Blizzard", "Heavy Snow",
    "High Wind", "Thunderstorm Wind", "Tornado",
    "Flood", "Flash Flood", "Lightning",
    "Extreme Cold/Wind Chill", "Heavy Rain"
]

# Try multiple naming patterns
noaa_frames = []
for yr in YEARS:
    found = False
    candidates = [
        RAW_DIR / f"noaa_storms_{yr}.csv.gz",
        RAW_DIR / f"noaa_storms_{yr}.csv",
        RAW_DIR / f"StormEvents_details_{yr}.csv.gz",
        RAW_DIR / f"storm_events_{yr}.csv.gz",
    ]
    # Also check for any file containing the year
    candidates += list(RAW_DIR.glob(f"*{yr}*.csv.gz"))
    candidates += list(RAW_DIR.glob(f"*{yr}*.csv"))

    for path in candidates:
        if not path.exists():
            continue
        try:
            compression = "gzip" if str(path).endswith(".gz") else None
            df = pd.read_csv(path, compression=compression,
                           low_memory=False)

            # Filter state
            state_col = next((c for c in df.columns
                            if c.upper() == "STATE"), None)
            if state_col:
                df = df[df[state_col].str.title().isin(NORTHEAST)]

            # Filter event types
            evt_col = next((c for c in df.columns
                          if "EVENT_TYPE" in c.upper()), None)
            if evt_col:
                df = df[df[evt_col].isin(STORM_TYPES)]

            df["source_year"] = yr
            noaa_frames.append(df)
            print(f"  NOAA {yr}: {len(df):,} events  ({path.name})")
            found = True
            break
        except Exception as e:
            continue

    if not found:
        print(f"  NOAA {yr}: not found")

if noaa_frames:
    noaa = pd.concat(noaa_frames, ignore_index=True)
    noaa.to_csv(PROC_DIR / "noaa_storms_northeast.csv", index=False)
    print(f"  Saved: noaa_storms_northeast.csv  ({len(noaa):,} total events)")


# ── Step 4: Copy supplementary files ─────────────────
print("\n[3/3] Copying supplementary files...")
for fname in ["MCC.csv", "DQI.csv", "coverage_history.csv"]:
    src = RAW_DIR / fname
    if src.exists():
        pd.read_csv(src).to_csv(PROC_DIR / fname, index=False)
        print(f"  Saved: {fname}")


# ── Final Summary ─────────────────────────────────────
print("\n" + "=" * 55)
print("SUMMARY")
print("=" * 55)

proc_files = sorted(PROC_DIR.glob("*.csv"))
print(f"\nProcessed files saved ({len(proc_files)} total):")
for f in proc_files:
    size_mb = f.stat().st_size / 1_000_000
    print(f"  {f.name:<45} {size_mb:.2f} MB")

if frames:
    print(f"\nEAGLE-I:")
    print(f"  Raw rows loaded    : {len(eaglei):,}")
    print(f"  County-days        : {len(daily):,}")
    print(f"  States             : {daily['state'].nunique() if 'state' in daily.columns else 'N/A'}")
    print(f"  Years              : {sorted(daily['year'].dropna().unique().tolist()) if 'year' in daily.columns else 'N/A'}")
    print(f"  Major outages      : {daily['is_major_outage'].sum():,} ({daily['is_major_outage'].mean():.1%})")
    print(f"  Peak customers out : {daily['max_customers_out'].max():,.0f}")

print("\nNext steps:")
print("  1. python src/model.py      (train ML on real data)")
print("  2. python src/lstm_model.py (train deep learning)")
print("  3. python src/nlp_analysis.py (text mining)")
