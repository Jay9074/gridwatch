"""
GridWatch Storm Watch - NOAA Forecast Ingestion
Pulls live weather forecasts for all 9 Northeast states at county level.
Runs every 6 hours via scheduler.
Uses api.weather.gov (no API key needed - just requires User-Agent header).

Run manually: python src/stormwatch/fetch_forecasts.py
Run on schedule: add to Windows Task Scheduler or cron
"""
import requests
import pandas as pd
import json
from pathlib import Path
from datetime import datetime, timedelta
import time
import os

# Output location
OUT_DIR = Path("data/stormwatch/forecasts")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# API config
NWS_BASE = "https://api.weather.gov"
HEADERS = {
    "User-Agent": "GridWatch/1.0 (research, contact via github.com/Jay9074/gridwatch)",
    "Accept": "application/geo+json"
}

# County coordinates for 9 Northeast states (county seat lat/lon)
# This is a subset - expand to all 226 counties for full coverage
NORTHEAST_COUNTIES = {
    # Maine - 16 counties
    "Cumberland, Maine":       (43.66, -70.26),
    "Penobscot, Maine":        (44.81, -68.78),
    "Kennebec, Maine":         (44.31, -69.78),
    "York, Maine":             (43.36, -70.75),
    "Androscoggin, Maine":     (44.10, -70.21),
    # New Hampshire
    "Hillsborough, NH":        (42.99, -71.46),
    "Rockingham, NH":          (42.93, -71.06),
    # Vermont
    "Chittenden, Vermont":     (44.48, -73.21),
    # Massachusetts
    "Middlesex, Massachusetts":(42.49, -71.39),
    "Worcester, Massachusetts":(42.27, -71.81),
    "Essex, Massachusetts":    (42.61, -70.93),
    "Suffolk, Massachusetts":  (42.36, -71.06),
    # Rhode Island
    "Providence, Rhode Island":(41.82, -71.41),
    # Connecticut
    "Hartford, Connecticut":   (41.76, -72.67),
    "New Haven, Connecticut":  (41.31, -72.92),
    "Fairfield, Connecticut":  (41.15, -73.39),
    # New York
    "Suffolk, New York":       (40.92, -72.66),
    "Nassau, New York":        (40.72, -73.59),
    "Westchester, New York":   (41.12, -73.79),
    "Erie, New York":          (42.89, -78.87),
    # New Jersey
    "Essex, New Jersey":       (40.74, -74.24),
    "Bergen, New Jersey":      (40.96, -74.07),
    "Middlesex, New Jersey":   (40.46, -74.40),
    "Monmouth, New Jersey":    (40.27, -74.20),
    "Ocean, New Jersey":       (39.94, -74.21),
    # Pennsylvania
    "Philadelphia, PA":        (39.95, -75.16),
    "Allegheny, PA":           (40.44, -79.99),
    "Montgomery, PA":          (40.21, -75.34),
    "Bucks, PA":               (40.34, -75.13),
    "Chester, PA":             (40.00, -75.61),
}


def get_forecast_url(lat, lon):
    """Step 1: Get the forecast endpoint for a lat/lon."""
    try:
        r = requests.get(f"{NWS_BASE}/points/{lat},{lon}",
                         headers=HEADERS, timeout=10)
        if r.status_code == 200:
            data = r.json()
            return data["properties"]["forecastHourly"]
    except Exception as e:
        print(f"  Error getting endpoint for {lat},{lon}: {e}")
    return None


def fetch_hourly_forecast(forecast_url):
    """Step 2: Get hourly forecast (up to 156 hours / ~6.5 days ahead)."""
    try:
        r = requests.get(forecast_url, headers=HEADERS, timeout=10)
        if r.status_code == 200:
            return r.json()
    except Exception as e:
        print(f"  Forecast fetch error: {e}")
    return None


def parse_forecast(forecast_data, county_name, state):
    """Extract relevant fields from NOAA forecast response."""
    if not forecast_data or "properties" not in forecast_data:
        return []
    
    periods = forecast_data["properties"].get("periods", [])
    rows = []
    
    for period in periods:
        rows.append({
            "county":               county_name,
            "state":                state,
            "forecast_time":        period.get("startTime"),
            "forecast_end":         period.get("endTime"),
            "temperature_f":        period.get("temperature"),
            "wind_speed_str":       period.get("windSpeed", ""),
            "wind_direction":       period.get("windDirection"),
            "short_forecast":       period.get("shortForecast", ""),
            "detailed_forecast":    period.get("detailedForecast", ""),
            "precipitation_pct":    (period.get("probabilityOfPrecipitation", {}) or {}).get("value"),
            "fetched_at":           datetime.utcnow().isoformat()
        })
    return rows


def parse_wind_speed(wind_str):
    """Convert '10 to 15 mph' or '15 mph' to numeric (max value)."""
    if not wind_str:
        return 0
    parts = wind_str.replace("mph", "").strip().split("to")
    try:
        return float(parts[-1].strip())
    except Exception:
        return 0


def main():
    print("=" * 60)
    print("GridWatch Storm Watch - Forecast Ingestion")
    print(f"Started: {datetime.utcnow().isoformat()}")
    print("=" * 60)
    
    all_forecasts = []
    failed = []
    
    for i, (county_name, (lat, lon)) in enumerate(NORTHEAST_COUNTIES.items(), 1):
        county_parts = county_name.split(", ")
        county = county_parts[0]
        state  = county_parts[1] if len(county_parts) > 1 else ""
        
        print(f"[{i}/{len(NORTHEAST_COUNTIES)}] {county_name}...", end=" ")
        
        forecast_url = get_forecast_url(lat, lon)
        if not forecast_url:
            print("FAILED (endpoint)")
            failed.append(county_name)
            continue
        
        time.sleep(0.5)  # be polite to NOAA API
        
        forecast = fetch_hourly_forecast(forecast_url)
        if not forecast:
            print("FAILED (forecast)")
            failed.append(county_name)
            continue
        
        rows = parse_forecast(forecast, county, state)
        all_forecasts.extend(rows)
        print(f"OK ({len(rows)} hours)")
        
        time.sleep(0.5)
    
    if not all_forecasts:
        print("\nNo forecasts collected. Exiting.")
        return
    
    # Build dataframe
    df = pd.DataFrame(all_forecasts)
    df["wind_mph_max"] = df["wind_speed_str"].apply(parse_wind_speed)
    df["forecast_time"] = pd.to_datetime(df["forecast_time"], utc=True)
    
    # Save with timestamp
    ts = datetime.utcnow().strftime("%Y%m%d_%H%M")
    out_file = OUT_DIR / f"forecast_{ts}.csv"
    df.to_csv(out_file, index=False)
    
    # Also save a "latest" snapshot for the dashboard
    df.to_csv(OUT_DIR / "latest.csv", index=False)
    
    print(f"\n{'=' * 60}")
    print(f"DONE")
    print(f"{'=' * 60}")
    print(f"Counties processed:  {len(NORTHEAST_COUNTIES) - len(failed)}")
    print(f"Counties failed:     {len(failed)}")
    print(f"Total forecast rows: {len(df):,}")
    print(f"Forecast horizon:    up to {df['forecast_time'].max()}")
    print(f"Saved to:            {out_file}")
    
    if failed:
        print(f"\nFailed counties: {failed}")


if __name__ == "__main__":
    main()
