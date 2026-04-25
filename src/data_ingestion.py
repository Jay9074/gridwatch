"""
GridWatch — src/data_ingestion.py
===================================
Downloads real US power grid data from federal sources:
  - DOE Form OE-417 (major outage events)
  - EIA Form 861 (utility reliability — SAIDI/SAIFI)
  - NOAA Storm Events (weather data)

"""

import os
import zipfile
import logging
import requests
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
log = logging.getLogger(__name__)

BASE_DIR   = Path(__file__).parent.parent
RAW_DIR    = BASE_DIR / "data" / "raw"
PROC_DIR   = BASE_DIR / "data" / "processed"
RAW_DIR.mkdir(parents=True, exist_ok=True)
PROC_DIR.mkdir(parents=True, exist_ok=True)

# Northeast states 
NORTHEAST = {
    "ME": "Maine", "NH": "New Hampshire", "VT": "Vermont",
    "MA": "Massachusetts", "RI": "Rhode Island", "CT": "Connecticut",
    "NY": "New York", "NJ": "New Jersey", "PA": "Pennsylvania"
}

NORTHEAST_FULL = list(NORTHEAST.values()) + list(NORTHEAST.keys())


# ── Helpers ──────────────────────────────────────────────────────
def _download(url: str, dest: Path, desc: str = "") -> bool:
    """Downloads a file with a progress bar. Returns True if successful."""
    if dest.exists():
        log.info(f"Already downloaded: {dest.name}")
        return True
    try:
        r = requests.get(url, stream=True, timeout=60)
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        with open(dest, "wb") as f, tqdm(
            desc=desc or dest.name, total=total,
            unit="B", unit_scale=True, unit_divisor=1024
        ) as bar:
            for chunk in r.iter_content(1024):
                f.write(chunk)
                bar.update(len(chunk))
        return True
    except Exception as e:
        log.warning(f"Download failed for {url}: {e}")
        if dest.exists():
            dest.unlink()
        return False


def _parse_number(val) -> float:
    """Converts strings like '1,234' or '2.5K' to float."""
    if pd.isna(val):
        return np.nan
    s = str(val).strip().upper().replace(",", "")
    try:
        if s.endswith("K"):
            return float(s[:-1]) * 1_000
        if s.endswith("M"):
            return float(s[:-1]) * 1_000_000
        return float(s)
    except ValueError:
        return np.nan


# ── 1. DOE OE-417 ────────────────────────────────────────────────
def fetch_doe_oe417(years: list = None) -> pd.DataFrame:
    """
    Downloads DOE Form OE-417 — every major power disturbance
    reported to the US Dept of Energy.
    Source: https://www.oe.netl.doe.gov/OE417_annual_summary.aspx
    """
    if years is None:
        years = list(range(2015, datetime.now().year))

    frames = []
    for yr in years:
        url  = f"https://www.oe.netl.doe.gov/fileuploads/OE417_Annual_Summary_{yr}.xlsx"
        dest = RAW_DIR / f"doe_oe417_{yr}.xlsx"
        if _download(url, dest, f"DOE OE-417 {yr}"):
            try:
                df = pd.read_excel(dest, skiprows=1)
                df["source_year"] = yr
                frames.append(df)
            except Exception as e:
                log.warning(f"Parse error OE-417 {yr}: {e}")

    if not frames:
        log.error("No OE-417 data downloaded.")
        return pd.DataFrame()

    df = pd.concat(frames, ignore_index=True)
    log.info(f"OE-417 loaded: {len(df):,} rows")
    return df


def clean_doe_oe417(df: pd.DataFrame) -> pd.DataFrame:
    """Standardises column names and filters to Northeast."""
    if df.empty:
        return df

    df.columns = (
        df.columns
        .str.strip()
        .str.lower()
        .str.replace(r"[\s/\(\)]+", "_", regex=True)
        .str.strip("_")
    )

    rename = {
        "date_event_began":              "event_date",
        "nerc_region":                   "nerc_region",
        "area_affected":                 "area",
        "event_type":                    "event_type",
        "demand_loss_mw":                "demand_loss_mw",
        "number_of_customers_affected":  "customers_affected",
        "respondent":                    "utility",
    }
    df = df.rename(columns={k: v for k, v in rename.items() if k in df.columns})

    if "event_date" in df.columns:
        df["event_date"] = pd.to_datetime(df["event_date"], errors="coerce")
        df["year"]   = df["event_date"].dt.year
        df["month"]  = df["event_date"].dt.month
        df["season"] = df["month"].map({
            12:"Winter", 1:"Winter",  2:"Winter",
            3:"Spring",  4:"Spring",  5:"Spring",
            6:"Summer",  7:"Summer",  8:"Summer",
            9:"Fall",    10:"Fall",   11:"Fall"
        })

    for col in ["demand_loss_mw", "customers_affected"]:
        if col in df.columns:
            df[col] = df[col].apply(_parse_number)

    # Filter Northeast
    if "area" in df.columns:
        mask = df["area"].astype(str).str.contains(
            "|".join(NORTHEAST_FULL), case=False, na=False
        )
        df = df[mask].copy()

    df = df.dropna(subset=["event_date"])
    log.info(f"OE-417 cleaned: {len(df):,} Northeast records")
    return df


# ── 2. EIA-861 ───────────────────────────────────────────────────
def fetch_eia_861(years: list = None) -> pd.DataFrame:
    """
    Downloads EIA Form 861 — annual utility reliability metrics.
    SAIDI = System Average Interruption Duration Index (minutes lost per customer)
    SAIFI = System Average Interruption Frequency Index (outages per customer)
    """
    if years is None:
        years = list(range(2019, datetime.now().year))

    frames = []
    for yr in years:
        suffix = str(yr)[-2:]
        url  = f"https://www.eia.gov/electricity/data/eia861/archive/zip/f8612{suffix}.zip"
        dest = RAW_DIR / f"eia861_{yr}.zip"
        extract = RAW_DIR / f"eia861_{yr}"

        if not extract.exists():
            if _download(url, dest, f"EIA-861 {yr}"):
                try:
                    with zipfile.ZipFile(dest, "r") as z:
                        z.extractall(extract)
                except Exception as e:
                    log.warning(f"Unzip error EIA-861 {yr}: {e}")
                    continue

        rel_files = list(extract.glob("*eliability*")) + list(extract.glob("*Reliability*"))
        if rel_files:
            try:
                df = pd.read_excel(rel_files[0], skiprows=1)
                df["source_year"] = yr
                frames.append(df)
                log.info(f"EIA-861 reliability {yr}: {len(df):,} rows")
            except Exception as e:
                log.warning(f"Parse error EIA-861 {yr}: {e}")

    if not frames:
        log.warning("No EIA-861 data loaded.")
        return pd.DataFrame()

    return pd.concat(frames, ignore_index=True)


# ── 3. NOAA Storm Events ─────────────────────────────────────────
def fetch_noaa_storms(years: list = None) -> pd.DataFrame:
    """
    Downloads NOAA Storm Events — every significant weather event
    by county. We filter for events that commonly cause power outages.
    Source: https://www.ncei.noaa.gov/pub/data/swdi/stormevents/csvfiles/
    """
    if years is None:
        years = list(range(2018, datetime.now().year))

    OUTAGE_EVENTS = [
        "Winter Storm", "Ice Storm", "Heavy Snow", "Blizzard",
        "High Wind", "Thunderstorm Wind", "Tornado", "Hurricane (Typhoon)",
        "Tropical Storm", "Flood", "Flash Flood", "Lightning",
        "Extreme Cold/Wind Chill", "Heavy Rain"
    ]

    frames = []
    base = "https://www.ncei.noaa.gov/pub/data/swdi/stormevents/csvfiles"

    for yr in years:
        dest = RAW_DIR / f"noaa_storms_{yr}.csv.gz"
        # NOAA filenames change — try a couple of patterns
        urls = [
            f"{base}/StormEvents_details-ftp_v1.0_d{yr}_c20240716.csv.gz",
            f"{base}/StormEvents_details-ftp_v1.0_d{yr}_c20231017.csv.gz",
            f"{base}/StormEvents_details-ftp_v1.0_d{yr}_c20230901.csv.gz",
        ]
        downloaded = False
        for url in urls:
            if _download(url, dest, f"NOAA Storms {yr}"):
                downloaded = True
                break

        if not downloaded:
            continue

        try:
            df = pd.read_csv(dest, compression="gzip", low_memory=False)
            # Filter Northeast states
            if "STATE" in df.columns:
                df = df[df["STATE"].str.title().isin(NORTHEAST.values())]
            # Filter outage-relevant events
            if "EVENT_TYPE" in df.columns:
                df = df[df["EVENT_TYPE"].isin(OUTAGE_EVENTS)]
            df["source_year"] = yr
            frames.append(df)
            log.info(f"NOAA {yr}: {len(df):,} relevant events in Northeast")
        except Exception as e:
            log.warning(f"Parse error NOAA {yr}: {e}")

    if not frames:
        log.warning("No NOAA data loaded.")
        return pd.DataFrame()

    return pd.concat(frames, ignore_index=True)


def clean_noaa(df: pd.DataFrame) -> pd.DataFrame:
    """Standardises NOAA data."""
    if df.empty:
        return df

    df.columns = df.columns.str.lower()

    if "begin_date_time" in df.columns:
        df["event_date"] = pd.to_datetime(df["begin_date_time"], errors="coerce")
        df["month"] = df["event_date"].dt.month
        df["year"]  = df["event_date"].dt.year

    def _dmg(v):
        if pd.isna(v): return 0.0
        s = str(v).upper().strip()
        try:
            if s.endswith("K"): return float(s[:-1]) * 1e3
            if s.endswith("M"): return float(s[:-1]) * 1e6
            if s.endswith("B"): return float(s[:-1]) * 1e9
            return float(s)
        except ValueError:
            return 0.0

    for col in ["damage_property", "damage_crops"]:
        if col in df.columns:
            df[f"{col}_usd"] = df[col].apply(_dmg)

    return df


# ── 4. Build Master Dataset ──────────────────────────────────────
def build_master_dataset() -> dict:
    """
    Runs the full pipeline and saves processed files.
    Returns a dict of DataFrames.
    """
    log.info("=" * 55)
    log.info("GridWatch — Data Ingestion Pipeline")
    log.info("=" * 55)

    # DOE
    log.info("\n[1/3] DOE OE-417 Outage Events...")
    doe_raw   = fetch_doe_oe417()
    doe_clean = clean_doe_oe417(doe_raw)
    if not doe_clean.empty:
        doe_clean.to_csv(PROC_DIR / "doe_outages_northeast.csv", index=False)
        log.info(f"Saved → data/processed/doe_outages_northeast.csv  ({len(doe_clean):,} rows)")

    # EIA
    log.info("\n[2/3] EIA-861 Reliability Metrics...")
    eia = fetch_eia_861()
    if not eia.empty:
        eia.to_csv(PROC_DIR / "eia861_reliability.csv", index=False)
        log.info(f"Saved → data/processed/eia861_reliability.csv  ({len(eia):,} rows)")

    # NOAA
    log.info("\n[3/3] NOAA Storm Events...")
    noaa_raw   = fetch_noaa_storms()
    noaa_clean = clean_noaa(noaa_raw)
    if not noaa_clean.empty:
        noaa_clean.to_csv(PROC_DIR / "noaa_storms_northeast.csv", index=False)
        log.info(f"Saved → data/processed/noaa_storms_northeast.csv  ({len(noaa_clean):,} rows)")

    # Summary
    log.info("\n" + "=" * 55)
    log.info("✅ Data ingestion complete!")
    if not doe_clean.empty:
        log.info(f"   Total outage events : {len(doe_clean):,}")
        if "customers_affected" in doe_clean.columns:
            total = doe_clean["customers_affected"].sum()
            log.info(f"   Total customers hit : {total:,.0f}")
        if "season" in doe_clean.columns:
            log.info(f"   Outages by season:\n{doe_clean['season'].value_counts().to_string()}")
    log.info("=" * 55)

    return {"doe": doe_clean, "eia": eia, "noaa": noaa_clean}


if __name__ == "__main__":
    build_master_dataset()
