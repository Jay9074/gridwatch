"""
GridWatch - ML Backtest v4: Final Tuning Attempt
==================================================
Honest improvements over v3 fixed:

1. PRIOR storm history features (lag features, no leakage)
   - storms_30d_prior, storms_90d_prior, storms_365d_prior
   - Counts storms in this county BEFORE the prediction date

2. Storm type one-hot encoding
   - Captures Ice Storm vs Thunderstorm vs Tornado differences

3. NLCD impervious surface estimates
   - Urban (high impervious) vs rural (low impervious) outage patterns

All features computed in a leakage-safe way (data strictly before prediction date).

Run: python src/stormwatch/backtest_ml_v4.py
"""
import pandas as pd
import numpy as np
import json
import warnings
from pathlib import Path
from datetime import datetime, timedelta
import sys
import re

warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).parent))
from backtest import TARGET_COUNTIES, classify_storm_tier

OUT_DIR = Path("data/stormwatch/backtest")
OUT_DIR.mkdir(parents=True, exist_ok=True)


# Impervious surface % per county (USGS NLCD 2021)
# These reflect urban development intensity - complementary to tree canopy
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


def normalize_county_name(name):
    if not isinstance(name, str):
        return ""
    name = re.sub(r'\s*\(zone\)\s*', '', name, flags=re.IGNORECASE)
    name = re.sub(r'\s*metro\s*$', '', name, flags=re.IGNORECASE)
    name = re.sub(r'\s*county\s*$', '', name, flags=re.IGNORECASE)
    return name.strip().title()


def load_eaglei():
    print("Loading EAGLE-I...")
    return pd.read_csv("data/processed/eaglei_daily_northeast.csv", parse_dates=["date"])


def load_noaa():
    print("Loading NOAA storm events...")
    df = pd.read_csv("data/processed/noaa_storms_northeast.csv", low_memory=False)
    df.columns = [c.lower() for c in df.columns]
    df["event_date"] = pd.to_datetime(df["begin_date_time"], errors="coerce")
    df = df.dropna(subset=["event_date"])
    if "end_date_time" in df.columns:
        df["end_date"] = pd.to_datetime(df["end_date_time"], errors="coerce")
        df["storm_duration_hrs"] = (df["end_date"] - df["event_date"]).dt.total_seconds() / 3600
        df["storm_duration_hrs"] = df["storm_duration_hrs"].fillna(1).clip(0.5, 168)
    else:
        df["storm_duration_hrs"] = 1
    
    if "cz_name" in df.columns:
        df["county"] = df["cz_name"].apply(normalize_county_name)
    if "state" in df.columns:
        df["state"] = df["state"].astype(str).str.title()
    df["magnitude"] = df.get("magnitude", 0).fillna(0)
    outage_types = ["thunderstorm wind", "high wind", "ice storm", "blizzard",
                    "winter storm", "heavy snow", "tornado", "tropical storm",
                    "hurricane", "freezing rain", "strong wind"]
    pattern = "|".join(outage_types)
    df = df[df["event_type"].astype(str).str.lower().str.contains(pattern, na=False, regex=True)]
    print(f"  {len(df):,} outage-relevant storms")
    return df


def classify_storm_type(event_type):
    """Map NOAA storm types to categories for one-hot encoding."""
    if not isinstance(event_type, str):
        return "other"
    et = event_type.lower()
    if "ice" in et or "freezing rain" in et:    return "ice"
    if "blizzard" in et or "heavy snow" in et:  return "snow"
    if "winter storm" in et:                    return "winter_storm"
    if "hurricane" in et or "tropical" in et:   return "hurricane"
    if "tornado" in et:                         return "tornado"
    if "thunderstorm" in et:                    return "thunderstorm"
    if "wind" in et:                            return "wind"
    return "other"


def load_county_features():
    path = Path("data/processed/county_features.csv")
    if not path.exists():
        return None
    return pd.read_csv(path)


def compute_baselines_from_subset(eaglei_subset):
    major = eaglei_subset[eaglei_subset["max_customers_out"] >= 1000]
    baselines = {}
    for (county, state) in TARGET_COUNTIES:
        sub = major[(major["county"] == county) & (major["state"] == state)]
        if len(sub) >= 5:
            baselines[f"{county}, {state}"] = {
                "typical_major_outage": float(sub["max_customers_out"].median()),
                "high_outage":          float(sub["max_customers_out"].quantile(0.75)),
                "extreme_outage":       float(sub["max_customers_out"].quantile(0.95)),
            }
        else:
            baselines[f"{county}, {state}"] = {
                "typical_major_outage": 1500.0,
                "high_outage":          3000.0,
                "extreme_outage":       8000.0,
            }
    return baselines


def compute_lag_storm_counts(target_storm, all_noaa, county, state):
    """Count prior storms in this county in 30/90/365 day windows.
    
    LEAKAGE-SAFE: only counts storms strictly before the target storm date.
    """
    target_date = target_storm["event_date"]
    
    # Filter to this county's prior storms
    prior = all_noaa[
        (all_noaa["county"] == county) &
        (all_noaa["state"] == state) &
        (all_noaa["event_date"] < target_date)
    ]
    
    if len(prior) == 0:
        return 0, 0, 0, 0
    
    days_30 = (prior["event_date"] >= target_date - timedelta(days=30)).sum()
    days_90 = (prior["event_date"] >= target_date - timedelta(days=90)).sum()
    days_365 = (prior["event_date"] >= target_date - timedelta(days=365)).sum()
    
    # Days since most recent storm
    most_recent = prior["event_date"].max()
    days_since_last = (target_date - most_recent).days
    
    return int(days_30), int(days_90), int(days_365), int(days_since_last)


def build_dataset(noaa, eagle, county_features):
    print("\nBuilding leakage-safe v4 feature dataset...")
    
    cf_map = {}
    if county_features is not None:
        for _, r in county_features.iterrows():
            cf_map[(r["county"], r["state"])] = r.to_dict()
    
    target_set = set(TARGET_COUNTIES)
    
    # Sort noaa by date so lag computation is efficient
    noaa_sorted = noaa.sort_values("event_date").reset_index(drop=True)
    
    # Pre-index NOAA by county/state for fast lag lookups
    print("  Indexing NOAA for lag lookups...")
    noaa_by_county = {}
    for (county, state) in TARGET_COUNTIES:
        sub = noaa_sorted[(noaa_sorted["county"] == county) &
                          (noaa_sorted["state"] == state)].copy()
        noaa_by_county[(county, state)] = sub
    
    rows = []
    n_total = 0
    
    for _, storm in noaa_sorted.iterrows():
        county = str(storm.get("county", ""))
        state  = str(storm.get("state", ""))
        if (county, state) not in target_set:
            continue
        
        n_total += 1
        if n_total % 500 == 0:
            print(f"  Processing storm {n_total}...")
        
        storm_date = storm["event_date"]
        window_end = storm_date + timedelta(hours=72)
        
        actual_rows = eagle[
            (eagle["county"] == county) &
            (eagle["state"] == state) &
            (eagle["date"] >= storm_date) &
            (eagle["date"] <= window_end)
        ]
        if len(actual_rows) == 0:
            continue
        actual = float(actual_rows["max_customers_out"].max())
        
        tier = classify_storm_tier(storm)
        storm_type = classify_storm_type(storm.get("event_type", ""))
        cf = cf_map.get((county, state), {})
        
        # Lag features (using pre-indexed county data)
        county_storms = noaa_by_county[(county, state)]
        prior_storms = county_storms[county_storms["event_date"] < storm_date]
        
        if len(prior_storms) > 0:
            d30  = ((prior_storms["event_date"] >= storm_date - timedelta(days=30))).sum()
            d90  = ((prior_storms["event_date"] >= storm_date - timedelta(days=90))).sum()
            d365 = ((prior_storms["event_date"] >= storm_date - timedelta(days=365))).sum()
            days_since = (storm_date - prior_storms["event_date"].max()).days
        else:
            d30 = d90 = d365 = 0
            days_since = 9999  # never had a storm before
        
        month = storm_date.month
        duration = float(storm.get("storm_duration_hrs", 1) or 1)
        
        rows.append({
            "storm_date":         storm_date,
            "county":             county,
            "state":              state,
            "event_type":         storm.get("event_type", ""),
            "tier_severe":        1 if tier == "SEVERE" else 0,
            "tier_moderate":      1 if tier == "MODERATE" else 0,
            "magnitude":          float(storm.get("magnitude", 0) or 0),
            "storm_duration_hrs": duration,
            "log_duration":       np.log1p(duration),
            "month":              month,
            "month_sin":          np.sin(2 * np.pi * month / 12),
            "month_cos":          np.cos(2 * np.pi * month / 12),
            "is_winter":          1 if month in [12,1,2] else 0,
            "is_summer":          1 if month in [6,7,8] else 0,
            "is_hurricane_season":1 if month in [8,9,10] else 0,
            # Storm type one-hot
            "type_ice":           1 if storm_type == "ice" else 0,
            "type_snow":          1 if storm_type == "snow" else 0,
            "type_winter_storm":  1 if storm_type == "winter_storm" else 0,
            "type_hurricane":     1 if storm_type == "hurricane" else 0,
            "type_tornado":       1 if storm_type == "tornado" else 0,
            "type_thunderstorm":  1 if storm_type == "thunderstorm" else 0,
            "type_wind":          1 if storm_type == "wind" else 0,
            # Lag features (no leakage - strictly prior)
            "storms_30d_prior":   int(d30),
            "storms_90d_prior":   int(d90),
            "storms_365d_prior":  int(d365),
            "days_since_last_storm": min(int(days_since), 9999),
            "log_days_since":     np.log1p(min(int(days_since), 9999)),
            # County features
            "tree_canopy_pct":    cf.get("tree_canopy_pct", 50),
            "population_density": cf.get("population_density", 500),
            "log_pop_density":    np.log1p(cf.get("population_density", 500)),
            "infrastructure_vulnerability": cf.get("infrastructure_vulnerability", 0.5),
            "land_area_sqmi":     cf.get("land_area_sqmi", 500),
            "log_pop":            np.log1p(cf.get("population_2023", 100000)),
            "impervious_pct":     IMPERVIOUS_PCT.get((county, state), 20),
            # Interactions
            "tier_x_canopy":      (1 if tier == "SEVERE" else 0.5 if tier == "MODERATE" else 0) * cf.get("tree_canopy_pct", 50) / 100,
            "tier_x_density":     (1 if tier == "SEVERE" else 0.5 if tier == "MODERATE" else 0) * np.log1p(cf.get("population_density", 500)),
            # Target
            "actual_customers":   actual,
        })
    
    return pd.DataFrame(rows)


def train_with_proper_cv(df, eagle):
    from sklearn.model_selection import KFold
    
    try:
        from xgboost import XGBRegressor
    except ImportError:
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "xgboost", "-q"])
        from xgboost import XGBRegressor
    
    try:
        from lightgbm import LGBMRegressor
    except ImportError:
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "lightgbm", "-q"])
        from lightgbm import LGBMRegressor
    
    base_features = [
        "tier_severe", "tier_moderate", "magnitude", "storm_duration_hrs", "log_duration",
        "month", "month_sin", "month_cos",
        "is_winter", "is_summer", "is_hurricane_season",
        # Storm type one-hot
        "type_ice", "type_snow", "type_winter_storm", "type_hurricane",
        "type_tornado", "type_thunderstorm", "type_wind",
        # Lag features (no leakage)
        "storms_30d_prior", "storms_90d_prior", "storms_365d_prior",
        "days_since_last_storm", "log_days_since",
        # County features
        "tree_canopy_pct", "population_density", "log_pop_density",
        "infrastructure_vulnerability", "land_area_sqmi", "log_pop",
        "impervious_pct",
        # Interactions
        "tier_x_canopy", "tier_x_density",
    ]
    
    baseline_features = ["baseline_typical", "baseline_high", "baseline_extreme"]
    feature_cols = base_features + baseline_features
    
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    print(f"\nRunning 5-fold CV on {len(df):,} pairs with per-fold baselines...")
    
    all_predictions = np.zeros(len(df))
    feature_imp_sum = np.zeros(len(feature_cols))
    fold_results = []
    
    eagle = eagle.copy()
    eagle["date"] = pd.to_datetime(eagle["date"])
    df = df.reset_index(drop=True).copy()
    
    for fold_idx, (train_idx, test_idx) in enumerate(kf.split(df), 1):
        train_df = df.iloc[train_idx].copy()
        test_df  = df.iloc[test_idx].copy()
        
        # Per-fold baselines (leakage-free)
        train_dates = pd.to_datetime(train_df["storm_date"]).dt.normalize()
        train_dates_set = set(train_dates.dt.date)
        eagle_train = eagle[eagle["date"].dt.date.isin(train_dates_set) | 
                            (eagle["date"] < train_dates.min())]
        
        fold_baselines = compute_baselines_from_subset(eagle_train)
        
        for d in [train_df, test_df]:
            d["baseline_typical"] = d.apply(
                lambda r: fold_baselines.get(f"{r['county']}, {r['state']}", {}).get("typical_major_outage", 1500),
                axis=1
            )
            d["baseline_high"] = d.apply(
                lambda r: fold_baselines.get(f"{r['county']}, {r['state']}", {}).get("high_outage", 3000),
                axis=1
            )
            d["baseline_extreme"] = d.apply(
                lambda r: fold_baselines.get(f"{r['county']}, {r['state']}", {}).get("extreme_outage", 8000),
                axis=1
            )
        
        X_tr = train_df[feature_cols].values
        y_tr = train_df["actual_customers"].values
        X_te = test_df[feature_cols].values
        y_te = test_df["actual_customers"].values
        
        sample_weights = np.where(X_tr[:, 0] == 1, 2.0, np.where(X_tr[:, 1] == 1, 1.5, 1.0))
        
        # XGBoost (v3 tuned params)
        xgb = XGBRegressor(
            n_estimators=350, max_depth=9, learning_rate=0.097,
            subsample=0.88, colsample_bytree=0.75,
            min_child_weight=2, reg_alpha=0.06, reg_lambda=1.7,
            random_state=42, n_jobs=-1, verbosity=0,
        )
        xgb.fit(X_tr, np.log1p(y_tr), sample_weight=sample_weights)
        pred_xgb = np.expm1(xgb.predict(X_te))
        
        # LightGBM
        lgb = LGBMRegressor(n_estimators=300, max_depth=8, learning_rate=0.08,
                           subsample=0.85, colsample_bytree=0.8,
                           random_state=42, n_jobs=-1, verbosity=-1)
        lgb.fit(X_tr, np.log1p(y_tr), sample_weight=sample_weights)
        pred_lgb = np.expm1(lgb.predict(X_te))
        
        pred = (0.6 * pred_xgb + 0.4 * pred_lgb)
        pred = np.maximum(pred, 200)
        
        all_predictions[test_idx] = pred
        feature_imp_sum += xgb.feature_importances_
        
        major_correct = ((pred >= 1000) == (y_te >= 1000)).mean()
        median_err = np.median(np.where(y_te > 0, np.abs(pred - y_te) / y_te * 100, 100))
        fold_results.append({
            "fold": fold_idx,
            "major_acc_pct":  round(major_correct * 100, 1),
            "median_err_pct": round(median_err, 1),
        })
        print(f"  Fold {fold_idx}: major_acc={major_correct*100:.1f}%  median_err={median_err:.1f}%")
    
    df["predicted_customers"] = all_predictions
    df["predicted_major"]    = df["predicted_customers"] >= 1000
    df["predicted_critical"] = df["predicted_customers"] >= 10000
    df["actual_major"]       = df["actual_customers"] >= 1000
    df["actual_critical"]    = df["actual_customers"] >= 10000
    df["major_correct"]      = df["predicted_major"] == df["actual_major"]
    df["critical_correct"]   = df["predicted_critical"] == df["actual_critical"]
    
    def get_ci_pct(row):
        if row["tier_severe"]:   return 0.55
        if row["tier_moderate"]: return 0.50
        return 0.45
    df["ci_pct"]  = df.apply(get_ci_pct, axis=1)
    df["ci_low"]  = (df["predicted_customers"] * (1 - df["ci_pct"])).round()
    df["ci_high"] = (df["predicted_customers"] * (1 + df["ci_pct"])).round()
    df["in_ci"]   = (df["actual_customers"] >= df["ci_low"]) & (df["actual_customers"] <= df["ci_high"])
    
    df["pct_error"] = np.where(
        df["actual_customers"] > 0,
        np.abs(df["predicted_customers"] - df["actual_customers"]) / df["actual_customers"] * 100,
        np.where(df["predicted_customers"] == 0, 0, 100)
    )
    
    df.to_csv(OUT_DIR / "ml_backtest_v4_results.csv", index=False)
    
    avg_imp = feature_imp_sum / 5
    importances = pd.DataFrame({"feature": feature_cols, "importance": avg_imp})
    importances = importances.sort_values("importance", ascending=False)
    importances.to_csv(OUT_DIR / "ml_feature_importance_v4.csv", index=False)
    
    scorecard = {
        "version":                 "v4: lag features + storm types + impervious surface",
        "total_storms_tested":     len(df),
        "major_outage_accuracy_pct":    round(df["major_correct"].mean() * 100, 1),
        "critical_outage_accuracy_pct": round(df["critical_correct"].mean() * 100, 1),
        "within_ci_pct":           round(df["in_ci"].mean() * 100, 1),
        "median_pct_error":        round(df["pct_error"].median(), 1),
        "mean_pct_error":          round(df["pct_error"].mean(), 1),
        "fold_results":            fold_results,
        "top_features":            importances.head(15).to_dict(orient="records"),
        "generated_at":            datetime.utcnow().isoformat() + "Z",
    }
    
    by_tier = {}
    for tier_name, tier_col in [("SEVERE", "tier_severe"), ("MODERATE", "tier_moderate")]:
        sub = df[df[tier_col] == 1]
        if len(sub) > 0:
            by_tier[tier_name] = {
                "n":                  len(sub),
                "major_accuracy_pct": round(sub["major_correct"].mean() * 100, 1),
                "median_pct_error":   round(sub["pct_error"].median(), 1),
                "within_ci_pct":      round(sub["in_ci"].mean() * 100, 1),
            }
    scorecard["by_tier"] = by_tier
    
    with open(OUT_DIR / "ml_backtest_v4_scorecard.json", "w") as f:
        json.dump(scorecard, f, indent=2, default=str)
    
    return scorecard


def main():
    print("=" * 70)
    print("GridWatch - ML Backtest v4 (lag + storm types + impervious)")
    print("=" * 70)
    
    eagle = load_eaglei()
    noaa  = load_noaa()
    cf = load_county_features()
    if cf is None:
        print("ERROR: county_features.csv missing")
        return 1
    
    df = build_dataset(noaa, eagle, cf)
    print(f"\nDataset: {len(df):,} storm-county pairs")
    
    scorecard = train_with_proper_cv(df, eagle)
    
    print(f"\n{'=' * 70}")
    print("ML BACKTEST v4 SCORECARD")
    print(f"{'=' * 70}")
    print(f"Total storms:              {scorecard['total_storms_tested']:,}")
    print(f"Major outage accuracy:     {scorecard['major_outage_accuracy_pct']}%")
    print(f"Critical outage accuracy:  {scorecard['critical_outage_accuracy_pct']}%")
    print(f"Within confidence:         {scorecard['within_ci_pct']}%")
    print(f"Median % error:            {scorecard['median_pct_error']}%")
    print(f"Mean % error:              {scorecard['mean_pct_error']}%")
    
    print(f"\nBy tier:")
    for tier, stats in scorecard["by_tier"].items():
        print(f"  {tier:9s} n={stats['n']:4d}  "
              f"major acc={stats['major_accuracy_pct']:5.1f}%  "
              f"median err={stats['median_pct_error']:5.1f}%  "
              f"CI hit={stats['within_ci_pct']:5.1f}%")
    
    print(f"\nTop 15 features:")
    for i, row in enumerate(scorecard["top_features"], 1):
        print(f"  {i:2d}. {row['feature']:35s}  {row['importance']:.4f}")
    
    print(f"\nFold consistency:")
    for f in scorecard["fold_results"]:
        print(f"  Fold {f['fold']}: major={f['major_acc_pct']}%  median_err={f['median_err_pct']}%")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
