"""
GridWatch v5 - Fetch granular daily weather for historical storms
==================================================================
Pulls actual daily wind speed + gusts from Open-Meteo's free historical
archive for each storm-county-date in the backtest dataset.

LEAKAGE-SAFE: only fetches weather for the storm date itself (not after).

Must run on YOUR laptop (Open-Meteo not reachable from sandbox).
No API key needed. Free. Rate-limited politely.

Output: data/processed/storm_weather_v5.csv  (cached - run once)

Run: python src/stormwatch/fetch_storm_weather.py
"""
import pandas as pd
import numpy as np
import requests
import time
import sys
from pathlib import Path
from datetime import timedelta
import re

sys.path.insert(0, str(Path(__file__).parent))
from backtest import TARGET_COUNTIES, classify_storm_tier

PROC_DIR = Path("data/processed")
OUT_PATH = PROC_DIR / "storm_weather_v5.csv"

# County centroids (lat, lon) for the 30 target counties - for weather lookup
COUNTY_COORDS = {
    ("Cumberland","Maine"): (43.66, -70.26),    ("Penobscot","Maine"): (45.20, -68.70),
    ("Kennebec","Maine"): (44.40, -69.78),      ("York","Maine"): (43.45, -70.70),
    ("Androscoggin","Maine"): (44.16, -70.21),
    ("Hillsborough","New Hampshire"): (42.92, -71.66),
    ("Rockingham","New Hampshire"): (42.99, -71.08),
    ("Chittenden","Vermont"): (44.51, -73.06),
    ("Middlesex","Massachusetts"): (42.48, -71.39),
    ("Worcester","Massachusetts"): (42.36, -71.90),
    ("Essex","Massachusetts"): (42.64, -70.86),
    ("Suffolk","Massachusetts"): (42.36, -71.06),
    ("Providence","Rhode Island"): (41.87, -71.42),
    ("Hartford","Connecticut"): (41.80, -72.69),
    ("New Haven","Connecticut"): (41.35, -72.90),
    ("Fairfield","Connecticut"): (41.23, -73.37),
    ("Suffolk","New York"): (40.92, -72.62),
    ("Nassau","New York"): (40.73, -73.59),
    ("Westchester","New York"): (41.12, -73.73),
    ("Erie","New York"): (42.75, -78.77),
    ("Essex","New Jersey"): (40.79, -74.25),
    ("Bergen","New Jersey"): (40.96, -74.07),
    ("Middlesex","New Jersey"): (40.44, -74.41),
    ("Monmouth","New Jersey"): (40.29, -74.15),
    ("Ocean","New Jersey"): (39.87, -74.27),
    ("Philadelphia","Pennsylvania"): (40.00, -75.13),
    ("Allegheny","Pennsylvania"): (40.47, -79.98),
    ("Montgomery","Pennsylvania"): (40.21, -75.37),
    ("Bucks","Pennsylvania"): (40.34, -75.11),
    ("Chester","Pennsylvania"): (39.97, -75.75),
}


def normalize_county_name(name):
    if not isinstance(name, str):
        return ""
    name = re.sub(r'\s*\(zone\)\s*', '', name, flags=re.IGNORECASE)
    name = re.sub(r'\s*metro\s*$', '', name, flags=re.IGNORECASE)
    name = re.sub(r'\s*county\s*$', '', name, flags=re.IGNORECASE)
    return name.strip().title()


def load_storm_dates():
    """Get unique (county, state, date) combos from NOAA outage storms."""
    df = pd.read_csv("data/processed/noaa_storms_northeast.csv", low_memory=False)
    df.columns = [c.lower() for c in df.columns]
    df["event_date"] = pd.to_datetime(df["begin_date_time"], errors="coerce")
    df = df.dropna(subset=["event_date"])
    if "cz_name" in df.columns:
        df["county"] = df["cz_name"].apply(normalize_county_name)
    if "state" in df.columns:
        df["state"] = df["state"].astype(str).str.title()
    
    outage_types = ["thunderstorm wind","high wind","ice storm","blizzard",
                    "winter storm","heavy snow","tornado","tropical storm",
                    "hurricane","freezing rain","strong wind"]
    pattern = "|".join(outage_types)
    df = df[df["event_type"].astype(str).str.lower().str.contains(pattern, na=False, regex=True)]
    
    target_set = set(TARGET_COUNTIES)
    df = df[df.apply(lambda r: (r["county"], r["state"]) in target_set, axis=1)]
    
    # Unique county-date combos
    df["date_only"] = df["event_date"].dt.date
    combos = df[["county","state","date_only"]].drop_duplicates()
    return combos


def fetch_weather(lat, lon, date):
    """Fetch daily wind for one location-date from Open-Meteo archive."""
    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": lat, "longitude": lon,
        "start_date": str(date), "end_date": str(date),
        "daily": ",".join([
            "wind_speed_10m_max", "wind_gusts_10m_max", "wind_direction_10m_dominant",
            "precipitation_sum", "rain_sum", "snowfall_sum",
            "temperature_2m_mean", "temperature_2m_max", "temperature_2m_min",
        ]),
        "hourly": "soil_moisture_0_to_7cm,soil_moisture_7_to_28cm",
        "timezone": "America/New_York",
        "wind_speed_unit": "mph",
    }
    try:
        r = requests.get(url, params=params, timeout=20)
        if r.status_code != 200:
            return None
        j = r.json()
        d = j.get("daily", {})
        h = j.get("hourly", {})
        
        # Aggregate hourly soil moisture to a daily mean (if present)
        def _mean(vals):
            v = [x for x in (vals or []) if x is not None]
            return round(sum(v)/len(v), 4) if v else None
        soil_shallow = _mean(h.get("soil_moisture_0_to_7cm"))
        soil_deep    = _mean(h.get("soil_moisture_7_to_28cm"))
        
        return {
            "wind_speed_max":  (d.get("wind_speed_10m_max") or [None])[0],
            "wind_gust_max":   (d.get("wind_gusts_10m_max") or [None])[0],
            "wind_dir":        (d.get("wind_direction_10m_dominant") or [None])[0],
            "precip_sum":      (d.get("precipitation_sum") or [None])[0],
            "rain_sum":        (d.get("rain_sum") or [None])[0],
            "snowfall_sum":    (d.get("snowfall_sum") or [None])[0],
            "temp_mean":       (d.get("temperature_2m_mean") or [None])[0],
            "temp_max":        (d.get("temperature_2m_max") or [None])[0],
            "temp_min":        (d.get("temperature_2m_min") or [None])[0],
            "soil_moisture_shallow": soil_shallow,
            "soil_moisture_deep":    soil_deep,
        }
    except Exception:
        return None


def main():
    print("=" * 60)
    print("GridWatch v5 - Fetch Storm Weather (Open-Meteo)")
    print("=" * 60)
    
    combos = load_storm_dates()
    print(f"Unique storm county-dates to fetch: {len(combos):,}")
    
    # Resume support - skip already-fetched
    done = set()
    if OUT_PATH.exists():
        existing = pd.read_csv(OUT_PATH)
        done = set(zip(existing["county"], existing["state"], existing["date_only"].astype(str)))
        print(f"Already fetched: {len(done):,} (resuming)")
        rows = existing.to_dict("records")
    else:
        rows = []
    
    fetched = 0
    skipped = 0
    for i, (_, c) in enumerate(combos.iterrows(), 1):
        key = (c["county"], c["state"], str(c["date_only"]))
        if key in done:
            skipped += 1
            continue
        
        coords = COUNTY_COORDS.get((c["county"], c["state"]))
        if not coords:
            continue
        
        w = fetch_weather(coords[0], coords[1], c["date_only"])
        if w:
            rows.append({
                "county": c["county"], "state": c["state"],
                "date_only": str(c["date_only"]), **w
            })
            fetched += 1
        
        # Polite rate limiting + periodic save
        if i % 50 == 0:
            print(f"  {i}/{len(combos)}  (fetched {fetched}, skipped {skipped})")
            pd.DataFrame(rows).to_csv(OUT_PATH, index=False)
        time.sleep(0.3)  # be nice to free API
    
    pd.DataFrame(rows).to_csv(OUT_PATH, index=False)
    print(f"\nDone. Fetched {fetched} new, total {len(rows)} records")
    print(f"Saved: {OUT_PATH}")


if __name__ == "__main__":
    sys.exit(main())
