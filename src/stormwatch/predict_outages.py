"""
GridWatch Storm Watch - Outage Prediction Engine (v4 ML-powered)
=================================================================
Uses the trained v4 ensemble model (XGBoost + LightGBM) with:
- Vegetation features (tree canopy, impervious surface)
- Population features (density, total)
- Storm history lag features
- Storm type one-hot encoding
- County baselines
- Tier and weather features

VALIDATED PERFORMANCE (5-fold CV on 3,074 historical storms):
- Major outage accuracy:    88.5%
- Critical outage accuracy: 90.5%
- Median prediction error:  31.8%
- Within confidence:        63.2%

Falls back to rule-based prediction if ML model not available.

Run: python src/stormwatch/predict_outages.py
"""
import pandas as pd
import numpy as np
import pickle
from pathlib import Path
from datetime import datetime, timedelta
import sys

STORM_DIR    = Path("data/stormwatch/storms")
PREDICT_DIR  = Path("data/stormwatch/predictions")
PREDICT_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR   = Path("models")
PROC_DIR     = Path("data/processed")

# Impervious surface % per county (USGS NLCD 2021)
IMPERVIOUS_PCT = {
    ("Cumberland", "Maine"): 6,    ("Penobscot", "Maine"): 3,
    ("Kennebec", "Maine"): 4,      ("York", "Maine"): 7,
    ("Androscoggin", "Maine"): 8,
    ("Hillsborough", "New Hampshire"): 10, ("Rockingham", "New Hampshire"): 15,
    ("Chittenden", "Vermont"): 9,
    ("Middlesex", "Massachusetts"): 25, ("Worcester", "Massachusetts"): 12,
    ("Essex", "Massachusetts"): 28,     ("Suffolk", "Massachusetts"): 65,
    ("Providence", "Rhode Island"): 22,
    ("Hartford", "Connecticut"): 20,    ("New Haven", "Connecticut"): 24,
    ("Fairfield", "Connecticut"): 22,
    ("Suffolk", "New York"): 30,        ("Nassau", "New York"): 45,
    ("Westchester", "New York"): 25,    ("Erie", "New York"): 18,
    ("Essex", "New Jersey"): 50,        ("Bergen", "New Jersey"): 45,
    ("Middlesex", "New Jersey"): 38,    ("Monmouth", "New Jersey"): 28,
    ("Ocean", "New Jersey"): 22,
    ("Philadelphia", "Pennsylvania"): 60, ("Allegheny", "Pennsylvania"): 22,
    ("Montgomery", "Pennsylvania"): 30,   ("Bucks", "Pennsylvania"): 25,
    ("Chester", "Pennsylvania"): 18,
}


def classify_storm_type(storm_tier, trigger_text=""):
    """Map storm context to ML feature categories."""
    t = (trigger_text or "").lower()
    if "ice" in t or "freezing" in t:    return "ice"
    if "blizzard" in t or "heavy snow" in t: return "snow"
    if "winter storm" in t:              return "winter_storm"
    if "hurricane" in t or "tropical" in t: return "hurricane"
    if "tornado" in t:                   return "tornado"
    if "thunderstorm" in t:              return "thunderstorm"
    if "wind" in t:                      return "wind"
    return "other"


def load_county_features():
    path = PROC_DIR / "county_features.csv"
    if not path.exists():
        print(f"WARN: {path} not found - run fetch_county_features.py")
        return {}
    df = pd.read_csv(path)
    return {(r["county"], r["state"]): r.to_dict() for _, r in df.iterrows()}


def load_storm_history():
    """Load historical NOAA storms for computing lag features."""
    path = PROC_DIR / "noaa_storms_northeast.csv"
    if not path.exists():
        return pd.DataFrame()
    
    df = pd.read_csv(path, low_memory=False)
    df.columns = [c.lower() for c in df.columns]
    df["event_date"] = pd.to_datetime(df["begin_date_time"], errors="coerce")
    df = df.dropna(subset=["event_date"])
    if "cz_name" in df.columns:
        df["county"] = df["cz_name"].astype(str).str.strip().str.title()
    if "state" in df.columns:
        df["state"] = df["state"].astype(str).str.title()
    return df[["event_date", "county", "state"]]


def compute_lag_features(storm_history, county, state, target_date):
    """Compute storms_30d/90d/365d_prior + days_since_last."""
    sub = storm_history[
        (storm_history["county"] == county) &
        (storm_history["state"] == state) &
        (storm_history["event_date"] < target_date)
    ]
    if len(sub) == 0:
        return 0, 0, 0, 9999
    
    d30 = (sub["event_date"] >= target_date - timedelta(days=30)).sum()
    d90 = (sub["event_date"] >= target_date - timedelta(days=90)).sum()
    d365 = (sub["event_date"] >= target_date - timedelta(days=365)).sum()
    days_since = (target_date - sub["event_date"].max()).days
    return int(d30), int(d90), int(d365), int(days_since)


def build_features_for_storm(storm_row, county_features_map, storm_history, baselines):
    """Build the 35-feature vector for a single storm prediction."""
    county = storm_row["county"]
    state = storm_row["state"]
    tier = storm_row["storm_tier"]
    
    storm_date = pd.to_datetime(storm_row["start_time"])
    if storm_date.tz is not None:
        storm_date = storm_date.tz_localize(None)
    
    duration = float(storm_row.get("duration_hrs", 1) or 1)
    wind = float(storm_row.get("max_wind_mph", 0) or 0)
    
    cf = county_features_map.get((county, state), {})
    storm_type = classify_storm_type(tier, storm_row.get("primary_trigger", ""))
    
    d30, d90, d365, days_since = compute_lag_features(
        storm_history, county, state, storm_date
    )
    
    month = storm_date.month
    base = baselines.get(f"{county}, {state}", {
        "typical_major_outage": 1500.0,
        "high_outage": 3000.0,
        "extreme_outage": 8000.0,
    })
    
    features = {
        "tier_severe":        1 if tier == "SEVERE" else 0,
        "tier_moderate":      1 if tier == "MODERATE" else 0,
        "magnitude":          wind,
        "storm_duration_hrs": duration,
        "log_duration":       np.log1p(duration),
        "month":              month,
        "month_sin":          np.sin(2 * np.pi * month / 12),
        "month_cos":          np.cos(2 * np.pi * month / 12),
        "is_winter":          1 if month in [12,1,2] else 0,
        "is_summer":          1 if month in [6,7,8] else 0,
        "is_hurricane_season":1 if month in [8,9,10] else 0,
        "type_ice":           1 if storm_type == "ice" else 0,
        "type_snow":          1 if storm_type == "snow" else 0,
        "type_winter_storm":  1 if storm_type == "winter_storm" else 0,
        "type_hurricane":     1 if storm_type == "hurricane" else 0,
        "type_tornado":       1 if storm_type == "tornado" else 0,
        "type_thunderstorm":  1 if storm_type == "thunderstorm" else 0,
        "type_wind":          1 if storm_type == "wind" else 0,
        "storms_30d_prior":   d30,
        "storms_90d_prior":   d90,
        "storms_365d_prior":  d365,
        "days_since_last_storm": days_since,
        "log_days_since":     np.log1p(days_since),
        "tree_canopy_pct":    cf.get("tree_canopy_pct", 50),
        "population_density": cf.get("population_density", 500),
        "log_pop_density":    np.log1p(cf.get("population_density", 500)),
        "infrastructure_vulnerability": cf.get("infrastructure_vulnerability", 0.5),
        "land_area_sqmi":     cf.get("land_area_sqmi", 500),
        "log_pop":            np.log1p(cf.get("population_2023", 100000)),
        "impervious_pct":     IMPERVIOUS_PCT.get((county, state), 20),
        "tier_x_canopy":      (1 if tier == "SEVERE" else 0.5 if tier == "MODERATE" else 0) * cf.get("tree_canopy_pct", 50) / 100,
        "tier_x_density":     (1 if tier == "SEVERE" else 0.5 if tier == "MODERATE" else 0) * np.log1p(cf.get("population_density", 500)),
        "baseline_typical":   base["typical_major_outage"],
        "baseline_high":      base["high_outage"],
        "baseline_extreme":   base["extreme_outage"],
    }
    return features


def predict_with_ml_model(storm_row, model_payload, county_features_map, storm_history):
    """Run the v4 ensemble model on a single storm."""
    features = build_features_for_storm(
        storm_row, county_features_map, storm_history,
        model_payload["baselines"]
    )
    
    feature_cols = model_payload["feature_cols"]
    X = np.array([[features[c] for c in feature_cols]])
    
    # Ensemble prediction
    pred_xgb = np.expm1(model_payload["xgb"].predict(X))[0]
    pred_lgb = np.expm1(model_payload["lgb"].predict(X))[0]
    predicted = 0.6 * pred_xgb + 0.4 * pred_lgb
    predicted = max(200, predicted)
    
    # Confidence intervals (calibrated from v4 validation)
    tier = storm_row["storm_tier"]
    if tier == "SEVERE":     ci_pct = 0.55
    elif tier == "MODERATE": ci_pct = 0.50
    else:                    ci_pct = 0.45
    
    return {
        "predicted_customers":   round(predicted),
        "ci_low":                round(predicted * (1 - ci_pct)),
        "ci_high":               round(predicted * (1 + ci_pct)),
        "confidence_level":      "HIGH" if ci_pct < 0.5 else "MEDIUM",
        "model_version":         model_payload.get("version", "v4"),
        "is_major_outage_likely":    predicted >= 1000,
        "is_critical_outage_likely": predicted >= 10000,
    }


def predict_with_rule_based(storm_row, baselines):
    """Fallback: rule-based prediction if ML model unavailable."""
    county_key = f"{storm_row['county']}, {storm_row['state']}"
    if county_key not in baselines:
        return None
    
    base = baselines[county_key]
    tier = storm_row["storm_tier"]
    
    if tier == "MINOR":
        baseline_customers = base["typical_major_outage"] * 0.45
        ci_pct = 0.55
    elif tier == "MODERATE":
        baseline_customers = base["typical_major_outage"] * 0.85
        ci_pct = 0.65
    elif tier == "SEVERE":
        baseline_customers = base["high_outage"] * 1.1
        ci_pct = 0.75
    else:
        baseline_customers = base["typical_major_outage"] * 0.4
        ci_pct = 0.6
    
    duration = storm_row.get("duration_hrs", 6)
    duration_mult = max(0.7, min(1.0 + (duration - 6) * 0.03, 2.0))
    
    wind = storm_row.get("max_wind_mph", 0)
    wind_mult = 1.5 if wind >= 60 else 1.25 if wind >= 45 else 1.05 if wind >= 30 else 1.0
    
    predicted = baseline_customers * duration_mult * wind_mult
    predicted = max(200, predicted)
    
    return {
        "predicted_customers":   round(predicted),
        "ci_low":                round(predicted * (1 - ci_pct)),
        "ci_high":                round(predicted * (1 + ci_pct)),
        "confidence_level":      "LOW",
        "model_version":         "rule_based_v2",
        "is_major_outage_likely":    predicted >= 1000,
        "is_critical_outage_likely": predicted >= 10000,
    }


def main():
    print("=" * 60)
    print("GridWatch Storm Watch - Outage Prediction (v4 ML)")
    print("=" * 60)
    
    storms_file = STORM_DIR / "active_storms.csv"
    if not storms_file.exists():
        print("Run detect_storms.py first")
        return 1
    
    storms = pd.read_csv(storms_file, parse_dates=["start_time","end_time"])
    if len(storms) == 0:
        print("No storms to predict")
        pd.DataFrame().to_csv(PREDICT_DIR / "active_predictions.csv", index=False)
        return 0
    print(f"Loaded {len(storms)} storm events")
    
    # Try to load v4 ML model
    model_path = MODELS_DIR / "outage_ml_model_v4_final.pkl"
    use_ml = False
    if model_path.exists():
        try:
            with open(model_path, "rb") as f:
                model_payload = pickle.load(f)
            use_ml = True
            print(f"Loaded v4 ML model (validation metrics):")
            for k, v in model_payload.get("validation_metrics", {}).items():
                print(f"  {k}: {v}")
        except Exception as e:
            print(f"WARN: could not load v4 model: {e}")
            print("Falling back to rule-based predictions")
    else:
        print(f"v4 model not found at {model_path}")
        print("Run: python src/stormwatch/save_v4_model.py")
        print("Falling back to rule-based predictions")
    
    if use_ml:
        cf_map = load_county_features()
        history = load_storm_history()
        print(f"Loaded county features for {len(cf_map)} counties")
        print(f"Loaded {len(history):,} historical storm records for lag features")
        baselines = model_payload["baselines"]
    else:
        # Fall back to rule-based
        from build_monthly_dataset import compute_baselines  # placeholder fallback
        county_summary = pd.read_csv("data/summary/county_risk_summary.csv")
        baselines = {}
        for _, r in county_summary.iterrows():
            baselines[f"{r['county']}, {r['state']}"] = {
                "typical_major_outage": r.get("peak_customers_out", 1500) * 0.3,
                "high_outage": r.get("peak_customers_out", 3000) * 0.5,
                "extreme_outage": r.get("peak_customers_out", 8000) * 0.8,
            }
    
    # Predict each storm
    predictions = []
    for _, storm in storms.iterrows():
        if use_ml:
            pred = predict_with_ml_model(storm, model_payload, cf_map, history)
        else:
            pred = predict_with_rule_based(storm, baselines)
        
        if pred is None:
            continue
        
        predictions.append({
            **storm.to_dict(),
            **pred,
            "predicted_at": datetime.utcnow().isoformat(),
            "prediction_id": f"{storm['county']}_{storm['state']}_{storm['start_time'].strftime('%Y%m%d%H')}".replace(" ","_").replace(",",""),
        })
    
    if not predictions:
        print("No predictions generated")
        return 0
    
    pred_df = pd.DataFrame(predictions)
    pred_df = pred_df.sort_values(["peak_severity", "predicted_customers"], ascending=[False, False])
    
    pred_df.to_csv(PREDICT_DIR / "active_predictions.csv", index=False)
    ts = datetime.utcnow().strftime("%Y%m%d_%H%M")
    pred_df.to_csv(PREDICT_DIR / f"predictions_{ts}.csv", index=False)
    
    # Cumulative log
    log_file = PREDICT_DIR / "prediction_log.csv"
    if log_file.exists():
        existing = pd.read_csv(log_file)
        combined = pd.concat([existing, pred_df], ignore_index=True)
        combined = combined.drop_duplicates(subset=["prediction_id"], keep="last")
    else:
        combined = pred_df
    combined.to_csv(log_file, index=False)
    
    print(f"\n{'=' * 60}")
    print(f"PREDICTIONS GENERATED ({pred_df['model_version'].iloc[0]})")
    print(f"{'=' * 60}")
    print(f"Storms predicted: {len(pred_df)}")
    print(f"Total customers at risk: {pred_df['predicted_customers'].sum():,.0f}")
    print(f"Major outages predicted: {pred_df['is_major_outage_likely'].sum()}")
    print(f"Critical outages predicted: {pred_df['is_critical_outage_likely'].sum()}")
    print(f"\nMax single event: {pred_df['predicted_customers'].max():,.0f}")
    print(f"Median per event: {pred_df['predicted_customers'].median():,.0f}")
    
    print(f"\nTop 5 events:")
    cols = ["county", "state", "storm_tier", "predicted_customers", "ci_low", "ci_high"]
    print(pred_df.head(5)[cols].to_string(index=False))
    
    print(f"\nSaved: {PREDICT_DIR / 'active_predictions.csv'}")


if __name__ == "__main__":
    sys.exit(main())
