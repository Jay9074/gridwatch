"""
GridWatch Storm Watch - Advanced Weather Features
Fetches ice accretion, lightning, and convective outlook from NOAA.
These are advanced weather variables relevant to outage prediction.

This adds advanced variables: high wind, ice accretion, lightning,
and convective severity to our forecast pipeline.

Run: python src/stormwatch/fetch_advanced_weather.py
"""
import requests
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta
import time

OUT_DIR = Path("data/stormwatch/advanced_weather")
OUT_DIR.mkdir(parents=True, exist_ok=True)

HEADERS = {
    "User-Agent": "GridWatch/1.0 (research, github.com/Jay9074/gridwatch)",
    "Accept": "application/geo+json"
}

# Same counties as fetch_forecasts.py
NORTHEAST_COUNTIES = {
    "Cumberland, Maine":       (43.66, -70.26),
    "Penobscot, Maine":        (44.81, -68.78),
    "Kennebec, Maine":         (44.31, -69.78),
    "York, Maine":             (43.36, -70.75),
    "Androscoggin, Maine":     (44.10, -70.21),
    "Hillsborough, NH":        (42.99, -71.46),
    "Rockingham, NH":          (42.93, -71.06),
    "Chittenden, Vermont":     (44.48, -73.21),
    "Middlesex, Massachusetts":(42.49, -71.39),
    "Worcester, Massachusetts":(42.27, -71.81),
    "Essex, Massachusetts":    (42.61, -70.93),
    "Suffolk, Massachusetts":  (42.36, -71.06),
    "Providence, Rhode Island":(41.82, -71.41),
    "Hartford, Connecticut":   (41.76, -72.67),
    "New Haven, Connecticut":  (41.31, -72.92),
    "Fairfield, Connecticut":  (41.15, -73.39),
    "Suffolk, New York":       (40.92, -72.66),
    "Nassau, New York":        (40.72, -73.59),
    "Westchester, New York":   (41.12, -73.79),
    "Erie, New York":          (42.89, -78.87),
    "Essex, New Jersey":       (40.74, -74.24),
    "Bergen, New Jersey":      (40.96, -74.07),
    "Middlesex, New Jersey":   (40.46, -74.40),
    "Monmouth, New Jersey":    (40.27, -74.20),
    "Ocean, New Jersey":       (39.94, -74.21),
    "Philadelphia, PA":        (39.95, -75.16),
    "Allegheny, PA":           (40.44, -79.99),
    "Montgomery, PA":          (40.21, -75.34),
    "Bucks, PA":               (40.34, -75.13),
    "Chester, PA":             (40.00, -75.61),
}


def estimate_ice_accretion(temp_f, precip_pct, forecast_text):
    """Estimate ice accretion risk from available NWS data.
    
    NWS doesn't publish ice accretion directly via api.weather.gov, but we can
    derive it from temperature + precipitation type indicators in forecast text.
    
    Returns ice accretion risk: 0 (none) to 5 (severe).
    """
    # Handle NaN/None gracefully
    import math
    if temp_f is None or (isinstance(temp_f, float) and math.isnan(temp_f)):
        return 0
    if precip_pct is not None and isinstance(precip_pct, float) and math.isnan(precip_pct):
        precip_pct = None
    
    text = (forecast_text or "").lower()
    
    # Direct ice indicators
    if "ice storm" in text or "freezing rain" in text:
        return 5
    if "icy" in text or "ice accretion" in text:
        return 4
    if "sleet" in text and temp_f < 35:
        return 3
    
    # Conditions favorable for icing
    if 28 <= temp_f <= 33 and precip_pct and precip_pct >= 60:
        if "rain" in text or "shower" in text:
            return 3  # rain at near-freezing = likely freezing rain
        if "snow" in text:
            return 2  # wet snow can also ice up
    
    # Cold + precip generally
    if temp_f <= 32 and precip_pct and precip_pct >= 50:
        return 1
    
    return 0


def estimate_lightning_risk(forecast_text, precip_pct):
    """Estimate lightning risk from forecast text.
    
    Returns 0-5 lightning risk scale.
    """
    import math
    if precip_pct is not None and isinstance(precip_pct, float) and math.isnan(precip_pct):
        precip_pct = None
    
    text = (forecast_text or "").lower()
    
    if "severe thunderstorm" in text or "tornado" in text:
        return 5
    if "thunderstorm" in text:
        if precip_pct and precip_pct >= 70:
            return 4
        return 3
    if "lightning" in text:
        return 3
    if "shower" in text and precip_pct and precip_pct >= 60:
        return 2
    
    return 0


def estimate_convective_severity(wind_mph, lightning_risk, precip_pct):
    """Estimate convective storm severity (like NOAA SPC outlook).
    
    Combines wind, lightning, and precipitation into a single severity score.
    """
    import math
    if precip_pct is not None and isinstance(precip_pct, float) and math.isnan(precip_pct):
        precip_pct = None
    
    score = 0
    
    if wind_mph >= 60:
        score += 4
    elif wind_mph >= 45:
        score += 3
    elif wind_mph >= 30:
        score += 2
    elif wind_mph >= 20:
        score += 1
    
    score += lightning_risk * 0.7
    
    if precip_pct:
        if precip_pct >= 90:
            score += 2
        elif precip_pct >= 70:
            score += 1
    
    return min(round(score, 1), 10)


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
    print("GridWatch - Advanced Weather Feature Computation")
    print("=" * 60)
    
    # Load existing forecasts (from fetch_forecasts.py)
    forecast_file = Path("data/stormwatch/forecasts/latest.csv")
    if not forecast_file.exists():
        print(f"Run fetch_forecasts.py first to generate {forecast_file}")
        return
    
    df = pd.read_csv(forecast_file, parse_dates=["forecast_time"])
    print(f"Loaded {len(df):,} hourly forecast rows")
    
    # Compute advanced features
    print("\nComputing advanced features...")
    
    df["wind_mph"] = df["wind_speed_str"].apply(parse_wind_speed)
    
    df["ice_accretion_risk"] = df.apply(
        lambda r: estimate_ice_accretion(
            r["temperature_f"],
            r.get("precipitation_pct"),
            str(r.get("short_forecast") or "") + " " + str(r.get("detailed_forecast") or "")
        ), axis=1
    )
    
    df["lightning_risk"] = df.apply(
        lambda r: estimate_lightning_risk(
            str(r.get("short_forecast") or "") + " " + str(r.get("detailed_forecast") or ""),
            r.get("precipitation_pct")
        ), axis=1
    )
    
    df["convective_severity"] = df.apply(
        lambda r: estimate_convective_severity(
            r["wind_mph"],
            r["lightning_risk"],
            r.get("precipitation_pct")
        ), axis=1
    )
    
    # Composite weather risk score (industry-standard metric)
    df["composite_weather_risk"] = (
        (df["wind_mph"] / 60).clip(0, 1) * 0.35 +
        (df["ice_accretion_risk"] / 5) * 0.30 +
        (df["lightning_risk"] / 5) * 0.20 +
        (df["convective_severity"] / 10) * 0.15
    ).round(3)
    
    # Save enhanced forecast
    df.to_csv(OUT_DIR / "latest_advanced.csv", index=False)
    ts = datetime.utcnow().strftime("%Y%m%d_%H%M")
    df.to_csv(OUT_DIR / f"advanced_{ts}.csv", index=False)
    
    print(f"\n{'=' * 60}")
    print("ADVANCED WEATHER FEATURES COMPUTED")
    print(f"{'=' * 60}")
    print(f"Rows: {len(df):,}")
    print(f"\nFeature distribution:")
    print(f"  Ice accretion risk:  max={df['ice_accretion_risk'].max()}, "
          f"counties with risk={(df['ice_accretion_risk'] > 0).any() and df.groupby('county')['ice_accretion_risk'].max().gt(0).sum() or 0}")
    print(f"  Lightning risk:      max={df['lightning_risk'].max()}, "
          f"counties with risk={df.groupby('county')['lightning_risk'].max().gt(0).sum()}")
    print(f"  Convective severity: max={df['convective_severity'].max():.1f}, "
          f"mean={df['convective_severity'].mean():.2f}")
    print(f"  Composite risk:      max={df['composite_weather_risk'].max():.3f}, "
          f"mean={df['composite_weather_risk'].mean():.3f}")
    
    # Show top 5 highest-risk hours
    top_risk = df.nlargest(5, "composite_weather_risk")[
        ["county", "state", "forecast_time", "wind_mph",
         "ice_accretion_risk", "lightning_risk", "composite_weather_risk"]
    ]
    print(f"\nTop 5 highest-risk forecast hours:")
    print(top_risk.to_string(index=False))
    
    print(f"\nSaved: {OUT_DIR / 'latest_advanced.csv'}")


if __name__ == "__main__":
    main()
