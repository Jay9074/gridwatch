"""
GridWatch - Fetch NOAA Storm Events 2014-2019 and merge with existing data
============================================================================
Expands the NOAA storm dataset from 2020-2024 back to 2014-2024 (the full
range where EAGLE-I outage data also exists - earlier years can't be paired
with outages so are not useful).

WHY 2014-2019 (not 1950): EAGLE-I outage data starts 2014. Storms before
that have no outage to pair with. Also, NOAA's modern, consistent recording
(post-1996 standards) and the current-era grid make recent data the right
training signal. More data helps ONLY when it's the right data.

WHAT THIS HELPS: a more robust CLASSIFIER + proper decade-long temporal
validation. It does NOT fix the ~69% count-error ceiling (that needs
proprietary utility asset data, proven stable across 5 splits).

Downloads from NCEI, filters to 9 Northeast states + outage event types,
matches existing CSV column format, appends to noaa_storms_northeast.csv.

Runs on YOUR laptop (NCEI not in sandbox allowlist). No API key needed.

Run: python src/stormwatch/fetch_noaa_history.py
"""
import pandas as pd
import numpy as np
import requests
import gzip
import io
import re
import sys
from pathlib import Path

PROC_DIR = Path("data/processed")
EXISTING = PROC_DIR / "noaa_storms_northeast.csv"
BACKUP   = PROC_DIR / "noaa_storms_northeast_backup_pre2014.csv"

# NCEI bulk CSV directory
NCEI_DIR = "https://www.ncei.noaa.gov/pub/data/swdi/stormevents/csvfiles/"

# Years to fetch (EAGLE-I starts 2014; 2020-2024 you already have)
YEARS = [2014, 2015, 2016, 2017, 2018, 2019]

# 9 Northeast states (uppercase as NOAA stores them)
NE_STATES = {
    "MAINE","NEW HAMPSHIRE","VERMONT","MASSACHUSETTS","RHODE ISLAND",
    "CONNECTICUT","NEW YORK","NEW JERSEY","PENNSYLVANIA"
}

# Outage-relevant event types (matches the pipeline filter)
OUTAGE_TYPES = ["thunderstorm wind","high wind","ice storm","blizzard",
                "winter storm","heavy snow","tornado","tropical storm",
                "hurricane","freezing rain","strong wind"]


def find_file_url(year):
    """Scrape the NCEI directory listing to find the details file for a year."""
    try:
        r = requests.get(NCEI_DIR, timeout=30)
        r.raise_for_status()
    except Exception as e:
        print(f"  ERROR listing NCEI directory: {e}")
        return None
    # Filenames look like: StormEvents_details-ftp_v1.0_d2014_c20220425.csv.gz
    pattern = rf'StormEvents_details-ftp_v1\.0_d{year}_c\d+\.csv\.gz'
    matches = re.findall(pattern, r.text)
    if not matches:
        print(f"  WARNING: no details file found for {year}")
        return None
    # If multiple creation dates, take the latest (last alphabetically)
    fname = sorted(set(matches))[-1]
    return NCEI_DIR + fname


def fetch_year(year):
    """Download + filter one year's storm details."""
    url = find_file_url(year)
    if not url:
        return None
    print(f"  {year}: downloading {url.split('/')[-1]}")
    try:
        r = requests.get(url, timeout=120)
        r.raise_for_status()
        with gzip.open(io.BytesIO(r.content), "rt", encoding="latin-1") as f:
            df = pd.read_csv(f, low_memory=False)
    except Exception as e:
        print(f"    ERROR: {e}")
        return None
    
    # NOAA columns are UPPERCASE: STATE, CZ_NAME, EVENT_TYPE, BEGIN_DATE_TIME, etc.
    df.columns = [c.upper() for c in df.columns]
    
    # Filter to Northeast states
    if "STATE" in df.columns:
        df = df[df["STATE"].astype(str).str.upper().isin(NE_STATES)]
    
    # Filter to outage-relevant event types
    pattern = "|".join(OUTAGE_TYPES)
    df = df[df["EVENT_TYPE"].astype(str).str.lower().str.contains(pattern, na=False, regex=True)]
    
    print(f"    -> {len(df):,} outage-relevant NE storms in {year}")
    return df


def main():
    print("="*68)
    print("GridWatch - Fetch NOAA Storm Events 2014-2019")
    print("="*68)
    
    if not EXISTING.exists():
        print(f"ERROR: {EXISTING} not found. Run from project root.")
        return 1
    
    # Load existing data to learn its exact columns
    print(f"Loading existing data: {EXISTING}")
    existing = pd.read_csv(EXISTING, low_memory=False)
    existing_cols = list(existing.columns)
    print(f"  Existing: {len(existing):,} rows, {len(existing_cols)} columns")
    
    # Detect existing date range
    date_col = next((c for c in existing.columns if c.upper()=="BEGIN_DATE_TIME"), None)
    if date_col:
        ed = pd.to_datetime(existing[date_col], errors="coerce")
        print(f"  Existing date range: {ed.min()} to {ed.max()}")
    
    # Fetch each historical year
    print(f"\nFetching {YEARS[0]}-{YEARS[-1]} from NCEI...")
    new_frames = []
    for y in YEARS:
        d = fetch_year(y)
        if d is not None and len(d) > 0:
            new_frames.append(d)
    
    if not new_frames:
        print("\nNo new data fetched. Check network / NCEI availability.")
        return 1
    
    new_data = pd.concat(new_frames, ignore_index=True)
    print(f"\nTotal new historical storms: {len(new_data):,}")
    
    # Align columns to existing file format
    # NOAA raw uses uppercase; match whatever case the existing file uses
    existing_upper = {c.upper(): c for c in existing_cols}
    aligned = pd.DataFrame()
    for up_col, orig_col in existing_upper.items():
        if up_col in new_data.columns:
            aligned[orig_col] = new_data[up_col]
        else:
            aligned[orig_col] = np.nan  # column not present in raw NOAA, fill blank
    
    # Report any columns we couldn't fill
    filled = [c for c in existing_cols if aligned[c].notna().any()]
    empty  = [c for c in existing_cols if not aligned[c].notna().any()]
    print(f"  Matched {len(filled)} columns; {len(empty)} left blank: {empty[:8]}{'...' if len(empty)>8 else ''}")
    
    # Backup before modifying
    print(f"\nBacking up existing file -> {BACKUP.name}")
    existing.to_csv(BACKUP, index=False)
    
    # Merge: historical first, then existing (chronological-ish)
    combined = pd.concat([aligned, existing], ignore_index=True)
    
    # De-duplicate on key fields if present
    key_cols = [c for c in existing_cols if c.upper() in
                ("EVENT_ID","BEGIN_DATE_TIME","CZ_NAME","STATE","EVENT_TYPE")]
    if "EVENT_ID" in existing_upper:
        before = len(combined)
        combined = combined.drop_duplicates(subset=[existing_upper["EVENT_ID"]])
        print(f"  De-duplicated on EVENT_ID: {before:,} -> {len(combined):,}")
    
    combined.to_csv(EXISTING, index=False)
    
    # Verify new range
    if date_col:
        nd = pd.to_datetime(combined[date_col], errors="coerce")
        print(f"\nNew dataset: {len(combined):,} rows")
        print(f"New date range: {nd.min()} to {nd.max()}")
    
    print(f"\n{'='*68}")
    print("DONE. Next steps:")
    print("  1. Re-run weather fetch for the new 2014-2019 storm-dates:")
    print("       python src\\stormwatch\\fetch_storm_weather.py")
    print("     (it's resumable - only fetches the new dates)")
    print("  2. Re-run the v5 backtest to rebuild features on 10 years:")
    print("       python src\\stormwatch\\backtest_ml_v5.py")
    print("  3. Re-run multi-split validation on the fuller dataset:")
    print("       python src\\stormwatch\\verify_ceiling_multisplit.py")
    print()
    print("  Expected: classifier gets MORE robust (more data, better validation).")
    print("  Count error stays ~69% (ceiling is real - proven across 5 splits).")
    print("  Backup saved as", BACKUP.name, "if you need to revert.")
    print(f"{'='*68}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
