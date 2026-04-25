"""
GridWatch — src/data_processing.py
=====================================
Processes EAGLE-I, NOAA, and EIA-861 data from local files.

This replaces data_ingestion.py now that we have real data downloaded.

Data sources:
  - EAGLE-I: eaglei_outages_YEAR.csv (county-level, 15-min intervals)
  - NOAA:    noaa_storms_YEAR.csv.gz (weather events)
  - EIA-861: eia861_YEAR.zip (utility reliability metrics)
  - MCC:     MCC.csv (customers per county)
  - DQI:     DQI.csv (data quality index)
  - coverage_history.csv (state coverage rates)

"""

import logging
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
warnings.filterwarnings("ignore")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
log = logging.getLogger(__name__)

BASE_DIR  = Path(__file__).parent.parent
RAW_DIR   = BASE_DIR / "data" / "raw"
PROC_DIR  = BASE_DIR / "data" / "processed"
PROC_DIR.mkdir(parents=True, exist_ok=True)

# Northeast states 
NORTHEAST_STATES = [
    "Maine", "New Hampshire", "Vermont", "Massachusetts",
    "Rhode Island", "Connecticut", "New York", "New Jersey", "Pennsylvania"
]
NORTHEAST_ABBR = ["ME", "NH", "VT", "MA", "RI", "CT", "NY", "NJ", "PA"]


# ── 1. Load EAGLE-I Outage Data ──────────────────────────────────
def load_eaglei(years: list = None, northeast_only: bool = True) -> pd.DataFrame:
    """
    Loads EAGLE-I outage CSV files.

    Columns in EAGLE-I:
      fips_code, county, state, customers_out, run_start_time, total_customers (in newer files)

    This is 15-minute interval data — we aggregate to hourly/daily
    for ML modeling.
    """
    if years is None:
        years = list(range(2018, 2025))

    frames = []
    for yr in tqdm(years, desc="Loading EAGLE-I"):
        path = RAW_DIR / f"eaglei_outages_{yr}.csv"
        if not path.exists():
            log.warning(f"Not found: {path.name}")
            continue
        try:
            df = pd.read_csv(path, low_memory=False)
            df["source_year"] = yr
            frames.append(df)
            log.info(f"EAGLE-I {yr}: {len(df):,} records loaded")
        except Exception as e:
            log.warning(f"Error loading EAGLE-I {yr}: {e}")

    if not frames:
        log.error("No EAGLE-I files found in data/raw/")
        return pd.DataFrame()

    df = pd.concat(frames, ignore_index=True)
    log.info(f"Total EAGLE-I records: {len(df):,}")

    # Standardize column names
    df.columns = df.columns.str.lower().str.strip()

    # Common EAGLE-I column name variants
    rename_map = {
        "fips_code":       "fips",
        "county":          "county",
        "state":           "state",
        "customers_out":   "customers_out",
        "run_start_time":  "timestamp",
        "recorded_date":   "timestamp",
        "sum(customers_out)": "customers_out",
    }
    df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})

    # Parse timestamp
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
        df["date"]   = df["timestamp"].dt.date
        df["year"]   = df["timestamp"].dt.year
        df["month"]  = df["timestamp"].dt.month
        df["hour"]   = df["timestamp"].dt.hour
        df["season"] = df["month"].map({
            12:"Winter", 1:"Winter",  2:"Winter",
            3:"Spring",  4:"Spring",  5:"Spring",
            6:"Summer",  7:"Summer",  8:"Summer",
            9:"Fall",    10:"Fall",   11:"Fall"
        })

    # Filter Northeast
    if northeast_only and "state" in df.columns:
        mask = (
            df["state"].isin(NORTHEAST_STATES) |
            df["state"].isin(NORTHEAST_ABBR)
        )
        df = df[mask].copy()
        log.info(f"After Northeast filter: {len(df):,} records")

    # Clean customers_out
    if "customers_out" in df.columns:
        df["customers_out"] = pd.to_numeric(df["customers_out"], errors="coerce").fillna(0)

    return df


# ── 2. Aggregate EAGLE-I to Daily ────────────────────────────────
def aggregate_eaglei_daily(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregates 15-minute EAGLE-I data to daily county-level summary.

    For each county-day we calculate:
      - max_customers_out: peak outage in that day
      - total_customer_hours: sum of customers_out * 0.25 hrs (15-min intervals)
      - outage_events: number of 15-min intervals with customers_out > 0
      - is_major_outage: 1 if max_customers_out >= 10,000
      - is_critical_outage: 1 if max_customers_out >= 50,000
    """
    if df.empty:
        return df

    group_cols = [c for c in ["fips", "county", "state", "date", "year",
                               "month", "season"] if c in df.columns]

    agg = df.groupby(group_cols).agg(
        max_customers_out    = ("customers_out", "max"),
        mean_customers_out   = ("customers_out", "mean"),
        total_customer_hours = ("customers_out", lambda x: x.sum() * 0.25),
        outage_intervals     = ("customers_out", lambda x: (x > 0).sum()),
    ).reset_index()

    agg["is_major_outage"]    = (agg["max_customers_out"] >= 10_000).astype(int)
    agg["is_critical_outage"] = (agg["max_customers_out"] >= 50_000).astype(int)
    agg["log_customers_out"]  = np.log1p(agg["max_customers_out"])
    agg["outage_duration_hrs"]= agg["outage_intervals"] * 0.25

    log.info(f"Daily aggregation: {len(agg):,} county-days")
    log.info(f"Major outage rate: {agg['is_major_outage'].mean():.1%}")
    log.info(f"Critical outage rate: {agg['is_critical_outage'].mean():.1%}")

    return agg


# ── 3. Load MCC (Customers per County) ───────────────────────────
def load_mcc() -> pd.DataFrame:
    """
    Loads MCC.csv — modeled number of electric customers per county.
    Used to normalize outage counts (% of customers affected).
    """
    path = RAW_DIR / "MCC.csv"
    if not path.exists():
        log.warning("MCC.csv not found")
        return pd.DataFrame()

    df = pd.read_csv(path)
    df.columns = df.columns.str.lower().str.strip()
    log.info(f"MCC loaded: {len(df):,} counties")
    return df


# ── 4. Load DQI (Data Quality Index) ─────────────────────────────
def load_dqi() -> pd.DataFrame:
    """
    Loads DQI.csv — data quality index by FEMA region and year.
    Using this in your paper shows methodological sophistication —
    very few researchers account for data quality in EAGLE-I analysis.
    """
    path = RAW_DIR / "DQI.csv"
    if not path.exists():
        log.warning("DQI.csv not found")
        return pd.DataFrame()

    df = pd.read_csv(path)
    df.columns = df.columns.str.lower().str.strip()
    log.info(f"DQI loaded: {len(df):,} records")
    return df


# ── 5. Load Coverage History ──────────────────────────────────────
def load_coverage() -> pd.DataFrame:
    """
    Loads coverage_history.csv — what % of customers are covered
    by EAGLE-I monitoring in each state each year.
    Important for understanding data completeness.
    """
    path = RAW_DIR / "coverage_history.csv"
    if not path.exists():
        log.warning("coverage_history.csv not found")
        return pd.DataFrame()

    df = pd.read_csv(path)
    df.columns = df.columns.str.lower().str.strip()
    log.info(f"Coverage history loaded: {len(df):,} records")
    return df


# ── 6. Load NOAA Storm Events ─────────────────────────────────────
def load_noaa(years: list = None) -> pd.DataFrame:
    """Loads NOAA storm events CSV.gz files."""
    if years is None:
        years = list(range(2019, 2025))

    OUTAGE_EVENTS = [
        "Winter Storm", "Ice Storm", "Heavy Snow", "Blizzard",
        "High Wind", "Thunderstorm Wind", "Tornado",
        "Hurricane (Typhoon)", "Tropical Storm", "Flood",
        "Flash Flood", "Lightning", "Extreme Cold/Wind Chill", "Heavy Rain"
    ]

    frames = []
    for yr in years:
        path = RAW_DIR / f"noaa_storms_{yr}.csv.gz"
        if not path.exists():
            log.warning(f"Not found: {path.name}")
            continue
        try:
            df = pd.read_csv(path, compression="gzip", low_memory=False)
            if "STATE" in df.columns:
                df = df[df["STATE"].str.title().isin(NORTHEAST_STATES)]
            if "EVENT_TYPE" in df.columns:
                df = df[df["EVENT_TYPE"].isin(OUTAGE_EVENTS)]
            df["source_year"] = yr
            frames.append(df)
            log.info(f"NOAA {yr}: {len(df):,} relevant storm events")
        except Exception as e:
            log.warning(f"Error loading NOAA {yr}: {e}")

    if not frames:
        return pd.DataFrame()

    return pd.concat(frames, ignore_index=True)


# ── 7. Load EIA-861 ───────────────────────────────────────────────
def load_eia861(years: list = None) -> pd.DataFrame:
    """Loads EIA-861 reliability metrics from ZIP files."""
    import zipfile

    if years is None:
        years = list(range(2019, 2024))

    frames = []
    for yr in years:
        zip_path    = RAW_DIR / f"eia861_{yr}.zip"
        extract_dir = RAW_DIR / f"eia861_{yr}"

        if not zip_path.exists():
            log.warning(f"Not found: {zip_path.name}")
            continue

        if not extract_dir.exists():
            try:
                with zipfile.ZipFile(zip_path, "r") as z:
                    z.extractall(extract_dir)
            except Exception as e:
                log.warning(f"Could not unzip EIA-861 {yr}: {e}")
                continue

        rel_files = (
            list(extract_dir.glob("*eliability*")) +
            list(extract_dir.glob("*Reliability*")) +
            list(extract_dir.glob("*reliability*"))
        )

        if rel_files:
            try:
                df = pd.read_excel(rel_files[0], skiprows=1)
                df["source_year"] = yr
                frames.append(df)
                log.info(f"EIA-861 {yr}: {len(df):,} utility records")
            except Exception as e:
                log.warning(f"Error reading EIA-861 {yr}: {e}")

    if not frames:
        return pd.DataFrame()

    return pd.concat(frames, ignore_index=True)


# ── 8. Build Master Dataset ───────────────────────────────────────
def build_master_dataset(years: list = None) -> dict:
    """
    Builds the complete analysis-ready dataset.
    Saves processed files to data/processed/
    """
    log.info("=" * 60)
    log.info("GridWatch — Data Processing Pipeline")
    log.info("=" * 60)

    results = {}

    # EAGLE-I
    log.info("\n[1/5] Loading EAGLE-I outage data...")
    eaglei_raw   = load_eaglei(years)
    eaglei_daily = aggregate_eaglei_daily(eaglei_raw)
    if not eaglei_daily.empty:
        eaglei_daily.to_csv(PROC_DIR / "eaglei_daily_northeast.csv", index=False)
        log.info(f"Saved → data/processed/eaglei_daily_northeast.csv")
        results["eaglei"] = eaglei_daily

    # MCC
    log.info("\n[2/5] Loading MCC (customers per county)...")
    mcc = load_mcc()
    if not mcc.empty:
        mcc.to_csv(PROC_DIR / "mcc_customers.csv", index=False)
        results["mcc"] = mcc

    # DQI
    log.info("\n[3/5] Loading DQI (data quality index)...")
    dqi = load_dqi()
    if not dqi.empty:
        dqi.to_csv(PROC_DIR / "dqi.csv", index=False)
        results["dqi"] = dqi

    # NOAA
    log.info("\n[4/5] Loading NOAA storm events...")
    noaa = load_noaa(years)
    if not noaa.empty:
        noaa.to_csv(PROC_DIR / "noaa_storms_northeast.csv", index=False)
        log.info(f"Saved → data/processed/noaa_storms_northeast.csv")
        results["noaa"] = noaa

    # EIA-861
    log.info("\n[5/5] Loading EIA-861 reliability metrics...")
    eia = load_eia861()
    if not eia.empty:
        eia.to_csv(PROC_DIR / "eia861_reliability.csv", index=False)
        log.info(f"Saved → data/processed/eia861_reliability.csv")
        results["eia"] = eia

    # Summary
    log.info("\n" + "=" * 60)
    log.info("✅ Data processing complete!")
    log.info("=" * 60)

    if "eaglei" in results:
        df = results["eaglei"]
        log.info(f"\nEAGLE-I Summary:")
        log.info(f"  County-days:        {len(df):,}")
        log.info(f"  Date range:         {df['date'].min()} → {df['date'].max()}")
        log.info(f"  States covered:     {df['state'].nunique()}")
        log.info(f"  Major outages:      {df['is_major_outage'].sum():,} ({df['is_major_outage'].mean():.1%})")
        log.info(f"  Critical outages:   {df['is_critical_outage'].sum():,} ({df['is_critical_outage'].mean():.1%})")
        log.info(f"  Peak customers out: {df['max_customers_out'].max():,.0f}")
        if "season" in df.columns:
            log.info(f"\n  Outages by season:")
            season_counts = df.groupby("season")["is_major_outage"].sum().sort_values(ascending=False)
            for season, count in season_counts.items():
                log.info(f"    {season}: {count:,}")

    return results


if __name__ == "__main__":
    results = build_master_dataset()
    if results:
        print("\n✅ All data processed and saved to data/processed/")
        print("Next step: run src/model.py to train on real data")
