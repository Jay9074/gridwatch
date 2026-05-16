"""
GridWatch - ML Backtest v2: Tighter Time Window + Better Matching
==================================================================
Improvements over v1:
- 24-hour outcome window (was 72 hours)
  Tighter window = cleaner storm-to-outage causation
- Better NOAA county name normalization
  Captures more matches between NOAA storm zones and EAGLE-I counties
- Storm intensity features (END_TIME duration)

Validation: 5-fold cross-validation.

Run: python src/stormwatch/backtest_ml_v2.py
"""
import pandas as pd
import numpy as np
import json
from pathlib import Path
from datetime import datetime, timedelta
import sys
import re

sys.path.insert(0, str(Path(__file__).parent))
from backtest import TARGET_COUNTIES, classify_storm_tier

OUT_DIR = Path("data/stormwatch/backtest")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def load_eaglei():
    print("Loading EAGLE-I...")
    df = pd.read_csv("data/processed/eaglei_daily_northeast.csv", parse_dates=["date"])
    print(f"  {len(df):,} county-days")
    return df


def normalize_county_name(name):
    """Normalize NOAA county names to match EAGLE-I."""
    if not isinstance(name, str):
        return ""
    name = name.strip()
    # NOAA sometimes uses "MIDDLESEX (ZONE)" or "BOSTON METRO"
    name = re.sub(r'\s*\(zone\)\s*', '', name, flags=re.IGNORECASE)
    name = re.sub(r'\s*metro\s*$', '', name, flags=re.IGNORECASE)
    name = re.sub(r'\s*county\s*$', '', name, flags=re.IGNORECASE)
    return name.strip().title()


def load_noaa():
    print("Loading NOAA storm events...")
    df = pd.read_csv("data/processed/noaa_storms_northeast.csv", low_memory=False)
    df.columns = [c.lower() for c in df.columns]
    
    # Date parsing - try multiple formats for speed
    df["event_date"] = pd.to_datetime(df["begin_date_time"], errors="coerce")
    df = df.dropna(subset=["event_date"])
    
    # Also parse end time if available for storm duration
    if "end_date_time" in df.columns:
        df["end_date"] = pd.to_datetime(df["end_date_time"], errors="coerce")
        df["storm_duration_hrs"] = (df["end_date"] - df["event_date"]).dt.total_seconds() / 3600
        df["storm_duration_hrs"] = df["storm_duration_hrs"].fillna(1).clip(0.5, 168)
    else:
        df["storm_duration_hrs"] = 1
    
    # Better county name handling
    if "cz_name" in df.columns:
        df["county"] = df["cz_name"].apply(normalize_county_name)
    if "state" in df.columns:
        df["state"] = df["state"].astype(str).str.title()
    if "magnitude" not in df.columns:
        df["magnitude"] = 0
    df["magnitude"] = df["magnitude"].fillna(0)
    
    outage_types = [
        "thunderstorm wind", "high wind", "ice storm", "blizzard",
        "winter storm", "heavy snow", "tornado", "tropical storm",
        "hurricane", "freezing rain", "strong wind", "thunderstorm"
    ]
    pattern = "|".join(outage_types)
    df = df[df["event_type"].astype(str).str.lower().str.contains(pattern, na=False, regex=True)]
    print(f"  {len(df):,} outage-relevant storms")
    return df


def load_county_baselines_all_data(eaglei_df):
    major = eaglei_df[eaglei_df["max_customers_out"] >= 1000]
    baselines = {}
    for (county, state) in TARGET_COUNTIES:
        sub = major[(major["county"] == county) & (major["state"] == state)]
        if len(sub) >= 5:
            baselines[f"{county}, {state}"] = {
                "typical_major_outage": float(sub["max_customers_out"].median()),
                "high_outage":          float(sub["max_customers_out"].quantile(0.75)),
                "extreme_outage":       float(sub["max_customers_out"].quantile(0.95)),
                "p25_outage":           float(sub["max_customers_out"].quantile(0.25)),
                "n_historical":         len(sub),
            }
        else:
            baselines[f"{county}, {state}"] = {
                "typical_major_outage": 1500.0,
                "high_outage":          3000.0,
                "extreme_outage":       8000.0,
                "p25_outage":           800.0,
                "n_historical":         0,
            }
    return baselines


def load_county_features():
    path = Path("data/processed/county_features.csv")
    if not path.exists():
        print("Run: python src/stormwatch/fetch_county_features.py")
        return None
    return pd.read_csv(path)


def build_dataset(noaa, eagle, baselines, county_features):
    """Build (X, y) with TIGHTER 24-hour outcome window."""
    print("\nBuilding feature dataset (24-hour outcome window)...")
    
    cf_map = {}
    if county_features is not None:
        for _, r in county_features.iterrows():
            cf_map[(r["county"], r["state"])] = {
                "tree_canopy_pct":              r["tree_canopy_pct"],
                "population_density":           r["population_density"],
                "infrastructure_vulnerability": r["infrastructure_vulnerability"],
                "land_area_sqmi":               r["land_area_sqmi"],
                "population_2023":              r.get("population_2023", 100000),
            }
    
    target_set = set(TARGET_COUNTIES)
    rows = []
    
    matched_targets = 0
    unmatched_storms = 0
    
    for _, storm in noaa.iterrows():
        county = str(storm.get("county", ""))
        state  = str(storm.get("state", ""))
        if (county, state) not in target_set:
            unmatched_storms += 1
            continue
        key = f"{county}, {state}"
        if key not in baselines:
            continue
        matched_targets += 1
        
        storm_date = storm["event_date"]
        # KEY CHANGE: 24-hour window instead of 72-hour
        window_end = storm_date + timedelta(hours=24)
        
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
        base = baselines[key]
        cf = cf_map.get((county, state), {
            "tree_canopy_pct":              50,
            "population_density":           500,
            "infrastructure_vulnerability": 0.5,
            "land_area_sqmi":               500,
            "population_2023":              100000,
        })
        
        # Storm duration as feature
        duration = storm.get("storm_duration_hrs", 1) or 1
        
        rows.append({
            "storm_date":         storm_date,
            "county":             county,
            "state":              state,
            "event_type":         storm.get("event_type", ""),
            "tier_severe":        1 if tier == "SEVERE" else 0,
            "tier_moderate":      1 if tier == "MODERATE" else 0,
            "magnitude":          float(storm.get("magnitude", 0) or 0),
            "storm_duration_hrs": float(duration),
            "month":              storm_date.month,
            "is_winter":          1 if storm_date.month in [12,1,2] else 0,
            "is_summer":          1 if storm_date.month in [6,7,8] else 0,
            "is_hurricane_season":1 if storm_date.month in [8,9,10] else 0,
            "baseline_typical":   base["typical_major_outage"],
            "baseline_high":      base["high_outage"],
            "baseline_extreme":   base["extreme_outage"],
            "baseline_p25":       base["p25_outage"],
            "baseline_n_history": base["n_historical"],
            "tree_canopy_pct":    cf["tree_canopy_pct"],
            "population_density": cf["population_density"],
            "infrastructure_vulnerability": cf["infrastructure_vulnerability"],
            "land_area_sqmi":     cf["land_area_sqmi"],
            "population_2023":    cf["population_2023"],
            "actual_customers":   actual,
        })
    
    print(f"  Storms in target counties:  {matched_targets:,}")
    print(f"  Storms with valid outcomes: {len(rows):,}")
    return pd.DataFrame(rows)


def train_with_kfold(df):
    from sklearn.model_selection import KFold
    
    try:
        from xgboost import XGBRegressor
    except ImportError:
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "xgboost", "-q"])
        from xgboost import XGBRegressor
    
    feature_cols = [
        "tier_severe", "tier_moderate", "magnitude", "storm_duration_hrs",
        "month", "is_winter", "is_summer", "is_hurricane_season",
        "baseline_typical", "baseline_high", "baseline_extreme",
        "baseline_p25", "baseline_n_history",
        "tree_canopy_pct", "population_density",
        "infrastructure_vulnerability", "land_area_sqmi", "population_2023",
    ]
    
    X = df[feature_cols].values
    y = df["actual_customers"].values
    
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    print(f"\nRunning 5-fold CV on {len(df):,} pairs...")
    
    all_predictions = np.zeros(len(df))
    feature_imp_sum = np.zeros(len(feature_cols))
    fold_results    = []
    
    for fold_idx, (train_idx, test_idx) in enumerate(kf.split(X), 1):
        X_tr, X_te = X[train_idx], X[test_idx]
        y_tr, y_te = y[train_idx], y[test_idx]
        y_tr_log = np.log1p(y_tr)
        
        model = XGBRegressor(
            n_estimators=400, max_depth=6, learning_rate=0.07,
            subsample=0.85, colsample_bytree=0.85,
            min_child_weight=3,
            random_state=42, n_jobs=-1, verbosity=0,
        )
        model.fit(X_tr, y_tr_log)
        pred = np.maximum(np.expm1(model.predict(X_te)), 200)
        all_predictions[test_idx] = pred
        feature_imp_sum += model.feature_importances_
        
        major_correct = ((pred >= 1000) == (y_te >= 1000)).mean()
        median_err = np.median(np.where(y_te > 0, np.abs(pred - y_te) / y_te * 100, 100))
        fold_results.append({
            "fold":           fold_idx,
            "major_acc_pct":  round(major_correct * 100, 1),
            "median_err_pct": round(median_err, 1),
        })
        print(f"  Fold {fold_idx}: major_acc={major_correct*100:.1f}%  median_err={median_err:.1f}%")
    
    df = df.copy()
    df["predicted_customers"] = all_predictions
    df["predicted_major"]    = df["predicted_customers"] >= 1000
    df["predicted_critical"] = df["predicted_customers"] >= 10000
    df["actual_major"]       = df["actual_customers"] >= 1000
    df["actual_critical"]    = df["actual_customers"] >= 10000
    df["major_correct"]      = df["predicted_major"] == df["actual_major"]
    df["critical_correct"]   = df["predicted_critical"] == df["actual_critical"]
    
    def get_ci_pct(row):
        if row["tier_severe"]:   return 0.65
        if row["tier_moderate"]: return 0.55
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
    
    df.to_csv(OUT_DIR / "ml_backtest_v2_results.csv", index=False)
    
    avg_imp = feature_imp_sum / 5
    importances = pd.DataFrame({"feature": feature_cols, "importance": avg_imp})
    importances = importances.sort_values("importance", ascending=False)
    importances.to_csv(OUT_DIR / "ml_feature_importance_v2.csv", index=False)
    
    # Train final model on all data
    print("\nTraining final model on all data...")
    final_model = XGBRegressor(
        n_estimators=400, max_depth=6, learning_rate=0.07,
        subsample=0.85, colsample_bytree=0.85, min_child_weight=3,
        random_state=42, n_jobs=-1, verbosity=0,
    )
    final_model.fit(X, np.log1p(y))
    
    import pickle
    models_dir = OUT_DIR.parent.parent.parent / "models"
    models_dir.mkdir(exist_ok=True)
    with open(models_dir / "outage_ml_model_v2.pkl", "wb") as f:
        pickle.dump({"model": final_model, "feature_cols": feature_cols}, f)
    
    scorecard = {
        "version":                 "v2: 24-hour window + better matching + storm duration",
        "total_storms_tested":     len(df),
        "major_outage_accuracy_pct":    round(df["major_correct"].mean() * 100, 1),
        "critical_outage_accuracy_pct": round(df["critical_correct"].mean() * 100, 1),
        "within_ci_pct":           round(df["in_ci"].mean() * 100, 1),
        "median_pct_error":        round(df["pct_error"].median(), 1),
        "mean_pct_error":          round(df["pct_error"].mean(), 1),
        "fold_results":            fold_results,
        "top_features":            importances.head(10).to_dict(orient="records"),
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
    
    with open(OUT_DIR / "ml_backtest_v2_scorecard.json", "w") as f:
        json.dump(scorecard, f, indent=2, default=str)
    
    print(f"\n{'=' * 70}")
    print("ML BACKTEST v2 SCORECARD")
    print(f"{'=' * 70}")
    print(f"Total storms:              {len(df):,}")
    print(f"Major outage accuracy:     {scorecard['major_outage_accuracy_pct']}%")
    print(f"Critical outage accuracy:  {scorecard['critical_outage_accuracy_pct']}%")
    print(f"Within confidence:         {scorecard['within_ci_pct']}%")
    print(f"Median % error:            {scorecard['median_pct_error']}%")
    print(f"Mean % error:              {scorecard['mean_pct_error']}%")
    
    print(f"\nBy tier:")
    for tier, stats in by_tier.items():
        print(f"  {tier:9s} n={stats['n']:4d}  "
              f"major acc={stats['major_accuracy_pct']:5.1f}%  "
              f"median err={stats['median_pct_error']:5.1f}%  "
              f"CI hit={stats['within_ci_pct']:5.1f}%")
    
    print(f"\nTop 10 features:")
    for i, row in enumerate(scorecard["top_features"], 1):
        print(f"  {i}. {row['feature']:35s}  {row['importance']:.4f}")
    
    return scorecard


def main():
    print("=" * 70)
    print("GridWatch - ML Backtest v2 (24-hr window + storm duration)")
    print("=" * 70)
    
    eagle = load_eaglei()
    noaa  = load_noaa()
    
    print("\nComputing county baselines...")
    baselines = load_county_baselines_all_data(eagle)
    print(f"  {len(baselines)} baselines")
    
    print("\nLoading county features...")
    cf = load_county_features()
    if cf is None: return 1
    print(f"  {len(cf)} counties")
    
    df = build_dataset(noaa, eagle, baselines, cf)
    
    if len(df) < 100:
        print(f"ERROR: only {len(df)} pairs")
        return 1
    
    print(f"\nDataset: {len(df):,} storm-county pairs")
    
    train_with_kfold(df)
    return 0


if __name__ == "__main__":
    sys.exit(main())
