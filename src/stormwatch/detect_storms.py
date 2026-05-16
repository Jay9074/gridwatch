"""
GridWatch Storm Watch - Storm Detection
Reads latest NOAA forecasts and flags upcoming storms by severity tier.
Outputs a county-storm risk table for the next 7 days.

Run: python src/stormwatch/detect_storms.py
"""
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta

FORECAST_DIR = Path("data/stormwatch/forecasts")
STORM_DIR    = Path("data/stormwatch/storms")
STORM_DIR.mkdir(parents=True, exist_ok=True)

# Storm severity thresholds based on outage risk
# Tuned from NOAA Storm Events severity scale + EAGLE-I historical correlation
STORM_THRESHOLDS = {
    "SEVERE": {
        "wind_mph":          50,    # high wind warning level
        "precip_pct":        80,
        "keywords":          ["ice storm", "blizzard", "hurricane", "tornado",
                              "tropical storm", "freezing rain"],
        "severity_score":    5,
        "expected_outages":  "HIGH"
    },
    "MODERATE": {
        "wind_mph":          35,
        "precip_pct":        70,
        "keywords":          ["thunderstorm", "heavy snow", "winter storm",
                              "heavy rain", "high wind"],
        "severity_score":    3,
        "expected_outages":  "MEDIUM"
    },
    "MINOR": {
        "wind_mph":          25,
        "precip_pct":        60,
        "keywords":          ["snow", "rain", "wind", "showers"],
        "severity_score":    2,
        "expected_outages":  "LOW"
    }
}


def classify_storm(row):
    """Determine storm tier for a single forecast hour."""
    wind = row.get("wind_mph_max", 0) or 0
    pct  = row.get("precipitation_pct", 0) or 0
    text = (str(row.get("short_forecast", "")) + " " +
            str(row.get("detailed_forecast", ""))).lower()
    
    # Check severe tier first
    severe = STORM_THRESHOLDS["SEVERE"]
    if wind >= severe["wind_mph"]:
        return "SEVERE", severe["severity_score"], f"Wind {wind}mph"
    for kw in severe["keywords"]:
        if kw in text:
            return "SEVERE", severe["severity_score"], f"Keyword: {kw}"
    
    # Check moderate
    moderate = STORM_THRESHOLDS["MODERATE"]
    if wind >= moderate["wind_mph"]:
        return "MODERATE", moderate["severity_score"], f"Wind {wind}mph"
    if pct >= moderate["precip_pct"] and wind >= 20:
        return "MODERATE", moderate["severity_score"], f"Heavy precip {pct}% + wind"
    for kw in moderate["keywords"]:
        if kw in text:
            return "MODERATE", moderate["severity_score"], f"Keyword: {kw}"
    
    # Check minor
    minor = STORM_THRESHOLDS["MINOR"]
    if wind >= minor["wind_mph"]:
        return "MINOR", minor["severity_score"], f"Wind {wind}mph"
    if pct >= minor["precip_pct"]:
        return "MINOR", minor["severity_score"], f"Precip {pct}%"
    
    return None, 0, None


def main():
    print("=" * 60)
    print("GridWatch Storm Watch - Storm Detection")
    print("=" * 60)
    
    # Load latest forecast
    latest_file = FORECAST_DIR / "latest.csv"
    if not latest_file.exists():
        print(f"No forecast file found at {latest_file}")
        print("Run fetch_forecasts.py first")
        return
    
    df = pd.read_csv(latest_file, parse_dates=["forecast_time"])
    print(f"Loaded forecast: {len(df):,} hourly records")
    print(f"Date range: {df['forecast_time'].min()} -> {df['forecast_time'].max()}")
    
    # Classify each hour
    classifications = df.apply(classify_storm, axis=1)
    df["storm_tier"]      = [c[0] for c in classifications]
    df["severity_score"]  = [c[1] for c in classifications]
    df["trigger_reason"]  = [c[2] for c in classifications]
    
    # Filter to only storm hours
    storms_hourly = df[df["storm_tier"].notna()].copy()
    
    print(f"\nStorm-flagged hours: {len(storms_hourly):,}")
    
    if len(storms_hourly) == 0:
        print("No storms forecast in next 7 days. Calm skies ahead.")
        # Still save an empty file so the dashboard knows
        empty = pd.DataFrame(columns=["county","state","storm_tier","start_time",
                                       "end_time","hours","peak_severity",
                                       "max_wind","max_precip","trigger"])
        empty.to_csv(STORM_DIR / "active_storms.csv", index=False)
        return
    
    # Group consecutive hours into "storm events" per county
    storms_hourly = storms_hourly.sort_values(["county", "forecast_time"])
    
    events = []
    for (county, state), group in storms_hourly.groupby(["county","state"]):
        group = group.sort_values("forecast_time").reset_index(drop=True)
        # Find time gaps > 3 hours = new storm event
        group["time_diff"] = group["forecast_time"].diff().dt.total_seconds() / 3600
        group["event_id"]  = (group["time_diff"] > 3).cumsum()
        
        for eid, ev in group.groupby("event_id"):
            events.append({
                "county":         county,
                "state":          state,
                "storm_tier":     ev["storm_tier"].mode().iloc[0] if not ev["storm_tier"].mode().empty else "MINOR",
                "peak_severity":  ev["severity_score"].max(),
                "start_time":     ev["forecast_time"].min(),
                "end_time":       ev["forecast_time"].max(),
                "duration_hrs":   round((ev["forecast_time"].max() - ev["forecast_time"].min()).total_seconds() / 3600, 1),
                "max_wind_mph":   ev["wind_mph_max"].max(),
                "max_precip_pct": ev["precipitation_pct"].max(),
                "primary_trigger":ev.iloc[0]["trigger_reason"],
                "hours_ahead":    round((ev["forecast_time"].min() - datetime.utcnow().replace(tzinfo=ev["forecast_time"].min().tz)).total_seconds() / 3600, 1)
            })
    
    storms_df = pd.DataFrame(events)
    storms_df = storms_df.sort_values(["peak_severity","start_time"], ascending=[False, True])
    
    # Save
    storms_df.to_csv(STORM_DIR / "active_storms.csv", index=False)
    
    # Also save timestamped snapshot
    ts = datetime.utcnow().strftime("%Y%m%d_%H%M")
    storms_df.to_csv(STORM_DIR / f"storms_{ts}.csv", index=False)
    
    print(f"\n{'=' * 60}")
    print("STORM EVENTS DETECTED")
    print(f"{'=' * 60}")
    print(f"Total events: {len(storms_df)}")
    print(f"\nBy tier:")
    print(storms_df["storm_tier"].value_counts().to_string())
    
    severe = storms_df[storms_df["storm_tier"] == "SEVERE"]
    if len(severe) > 0:
        print(f"\n⚠️  SEVERE STORM ALERTS ({len(severe)}):")
        for _, s in severe.head(10).iterrows():
            print(f"  {s['county']}, {s['state']} | "
                  f"{s['start_time'].strftime('%Y-%m-%d %H:%M')} | "
                  f"{s['duration_hrs']:.1f}h | "
                  f"Wind: {s['max_wind_mph']:.0f}mph | "
                  f"{s['primary_trigger']}")
    
    print(f"\nSaved to: {STORM_DIR / 'active_storms.csv'}")


if __name__ == "__main__":
    main()
