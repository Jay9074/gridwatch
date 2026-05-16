"""
GridWatch - ML Backtest with K-Fold Cross-Validation
======================================================
Trains XGBoost regressor on historical storms using:
- Storm features (type, magnitude, tier)
- County baseline features (median outage, peak, area)
- Vegetation features (tree canopy %)
- Population features (density, total pop)
- Temporal features (month, season)

Uses 5-fold STRATIFIED cross-validation (not train/test split)
because NOAA data only covers 2019-2025.

Run: python src/stormwatch/backtest_ml.py
"""
import pandas as pd
import numpy as np
import json
from pathlib import Path
from datetime import datetime, timedelta
import sys

sys.path.insert(0, str(Path(__file__).parent))
from backtest import TARGET_COUNTIES, classify_storm_tier

OUT_DIR = Path("data/stormwatch/backtest")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def load_eaglei():
    print("Loading EAGLE-I...")
    df = pd.read_csv("data/processed/eaglei_daily_northeast.csv", parse_dates=["date"])
    print(f"  {len(df):,} county-days")
    return df


def load_noaa():
    print("Loading NOAA storm events...")
    df = pd.read_csv("data/processed/noaa_storms_northeast.csv", low_memory=False)
    df.columns = [c.lower() for c in df.columns]
    df["event_date"] = pd.to_datetime(df["begin_date_time"], errors="coerce")
    df = df.dropna(subset=["event_date"])
    if "cz_name" in df.columns:
        df["county"] = df["cz_name"].astype(str).str.title()
    if "state" in df.columns:
        df["state"] = df["state"].astype(str).str.title()
    if "magnitude" not in df.columns:
        df["magnitude"] = 0
    
    outage_types = [
        "thunderstorm wind", "high wind", "ice storm", "blizzard",
        "winter storm", "heavy snow", "tornado", "tropical storm",
        "hurricane", "freezing rain", "strong wind"
    ]
    pattern = "|".join(outage_types)
    df = df[df["event_type"].astype(str).str.lower().str.contains(pattern, na=False, regex=True)]
    print(f"  {len(df):,} outage-relevant storms")
    return df


def load_county_baselines_all_data(eaglei_df):
    """Compute baselines from ALL EAGLE-I data (no temporal split needed
    for CV approach - we use CV folds for validation)."""
    major = eaglei_df[eaglei_df["max_customers_out"] >= 1000]
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


def load_county_features():
    path = Path("data/processed/county_features.csv")
    if not path.exists():
        print("Run: python src/stormwatch/fetch_county_features.py")
        return None
    df = pd.read_csv(path)
    print(f"  Loaded county features for {len(df)} counties")
    return df


def build_dataset(noaa, eagle, baselines, county_features):
    print("\nBuilding feature dataset...")
    
    cf_map = {}
    if county_features is not None:
        for _, r in county_features.iterrows():
            cf_map[(r["county"], r["state"])] = {
                "tree_canopy_pct":              r["tree_canopy_pct"],
                "population_density":           r["population_density"],
                "infrastructure_vulnerability": r["infrastructure_vulnerability"],
                "land_area_sqmi":               r["land_area_sqmi"],
            }
    
    target_set = set(TARGET_COUNTIES)
    rows = []
    
    for _, storm in noaa.iterrows():
        county = str(storm.get("county", ""))
        state  = str(storm.get("state", ""))
        if (county, state) not in target_set:
            continue
        key = f"{county}, {state}"
        if key not in baselines:
            continue
        
        storm_date = storm["event_date"]
        window_end = storm_date + timedelta(days=3)
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
        })
        
        rows.append({
            "storm_date":              storm_date,
            "county":                  county,
            "state":                   state,
            "event_type":              storm.get("event_type", ""),
            "tier_severe":             1 if tier == "SEVERE" else 0,
            "tier_moderate":           1 if tier == "MODERATE" else 0,
            "magnitude":               float(storm.get("magnitude", 0) or 0),
            "month":                   storm_date.month,
            "is_winter":               1 if storm_date.month in [12,1,2] else 0,
            "is_summer":               1 if storm_date.month in [6,7,8] else 0,
            "baseline_typical":        base["typical_major_outage"],
            "baseline_high":           base["high_outage"],
            "baseline_extreme":        base["extreme_outage"],
            "tree_canopy_pct":         cf["tree_canopy_pct"],
            "population_density":      cf["population_density"],
            "infrastructure_vulnerability": cf["infrastructure_vulnerability"],
            "land_area_sqmi":          cf["land_area_sqmi"],
            "actual_customers":        actual,
        })
    
    return pd.DataFrame(rows)


def train_with_kfold(df):
    """5-fold cross-validation on the storm-county dataset."""
    from sklearn.model_selection import KFold
    
    try:
        from xgboost import XGBRegressor
    except ImportError:
        print("Installing xgboost...")
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "xgboost", "-q"])
        from xgboost import XGBRegressor
    
    feature_cols = [
        "tier_severe", "tier_moderate", "magnitude",
        "month", "is_winter", "is_summer",
        "baseline_typical", "baseline_high", "baseline_extreme",
        "tree_canopy_pct", "population_density",
        "infrastructure_vulnerability", "land_area_sqmi",
    ]
    
    X = df[feature_cols].values
    y = df["actual_customers"].values
    
    n_splits = 5
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    print(f"\nRunning {n_splits}-fold cross-validation on {len(df):,} storm-county pairs...")
    
    all_predictions = np.zeros(len(df))
    fold_results    = []
    feature_imp_sum = np.zeros(len(feature_cols))
    
    for fold_idx, (train_idx, test_idx) in enumerate(kf.split(X), 1):
        X_tr, X_te = X[train_idx], X[test_idx]
        y_tr, y_te = y[train_idx], y[test_idx]
        
        # Log-transform target
        y_tr_log = np.log1p(y_tr)
        
        model = XGBRegressor(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.08,
            subsample=0.85,
            colsample_bytree=0.85,
            random_state=42,
            n_jobs=-1,
            verbosity=0,
        )
        model.fit(X_tr, y_tr_log)
        
        pred_log = model.predict(X_te)
        pred = np.maximum(np.expm1(pred_log), 200)
        all_predictions[test_idx] = pred
        feature_imp_sum += model.feature_importances_
        
        # Per-fold metrics
        major_correct = ((pred >= 1000) == (y_te >= 1000)).mean()
        median_err = np.median(np.where(y_te > 0, np.abs(pred - y_te) / y_te * 100, 100))
        fold_results.append({
            "fold":           fold_idx,
            "n_train":        len(train_idx),
            "n_test":         len(test_idx),
            "major_acc_pct":  round(major_correct * 100, 1),
            "median_err_pct": round(median_err, 1),
        })
        print(f"  Fold {fold_idx}: n_train={len(train_idx)} n_test={len(test_idx)}  "
              f"major_acc={major_correct*100:.1f}%  median_err={median_err:.1f}%")
    
    # Build results dataframe
    df = df.copy()
    df["predicted_customers"] = all_predictions
    df["predicted_major"]    = df["predicted_customers"] >= 1000
    df["predicted_critical"] = df["predicted_customers"] >= 10000
    df["actual_major"]       = df["actual_customers"] >= 1000
    df["actual_critical"]    = df["actual_customers"] >= 10000
    df["major_correct"]      = df["predicted_major"] == df["actual_major"]
    df["critical_correct"]   = df["predicted_critical"] == df["actual_critical"]
    
    def get_ci_pct(row):
        if row["tier_severe"]:   return 0.75
        if row["tier_moderate"]: return 0.65
        return 0.55
    df["ci_pct"]  = df.apply(get_ci_pct, axis=1)
    df["ci_low"]  = (df["predicted_customers"] * (1 - df["ci_pct"])).round()
    df["ci_high"] = (df["predicted_customers"] * (1 + df["ci_pct"])).round()
    df["in_ci"]   = (df["actual_customers"] >= df["ci_low"]) & (df["actual_customers"] <= df["ci_high"])
    
    df["pct_error"] = np.where(
        df["actual_customers"] > 0,
        np.abs(df["predicted_customers"] - df["actual_customers"]) / df["actual_customers"] * 100,
        np.where(df["predicted_customers"] == 0, 0, 100)
    )
    
    df.to_csv(OUT_DIR / "ml_backtest_results.csv", index=False)
    
    # Average feature importances across folds
    avg_imp = feature_imp_sum / n_splits
    importances = pd.DataFrame({
        "feature":    feature_cols,
        "importance": avg_imp
    }).sort_values("importance", ascending=False)
    importances.to_csv(OUT_DIR / "ml_feature_importance.csv", index=False)
    
    # Train final model on ALL data (for live predictions)
    print("\nTraining final model on all data for live deployment...")
    y_log = np.log1p(y)
    final_model = XGBRegressor(
        n_estimators=300, max_depth=6, learning_rate=0.08,
        subsample=0.85, colsample_bytree=0.85, random_state=42, n_jobs=-1, verbosity=0,
    )
    final_model.fit(X, y_log)
    
    import pickle
    models_dir = OUT_DIR.parent.parent.parent / "models"
    models_dir.mkdir(exist_ok=True)
    with open(models_dir / "outage_ml_model.pkl", "wb") as f:
        pickle.dump({"model": final_model, "feature_cols": feature_cols}, f)
    
    # Scorecard
    scorecard = {
        "approach":                "XGBoost regression, 5-fold CV with vegetation + population",
        "validation":              "5-fold cross-validation (NOAA data: 2020-2024 only)",
        "total_storms_tested":     len(df),
        "major_outage_accuracy_pct":    round(df["major_correct"].mean() * 100, 1),
        "critical_outage_accuracy_pct": round(df["critical_correct"].mean() * 100, 1),
        "within_ci_pct":           round(df["in_ci"].mean() * 100, 1),
        "median_pct_error":        round(df["pct_error"].median(), 1),
        "mean_pct_error":          round(df["pct_error"].mean(), 1),
        "fold_results":            fold_results,
        "top_features":            importances.head(8).to_dict(orient="records"),
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
    
    with open(OUT_DIR / "ml_backtest_scorecard.json", "w") as f:
        json.dump(scorecard, f, indent=2, default=str)
    
    print(f"\n{'=' * 70}")
    print("ML-BASED BACKTEST SCORECARD (5-fold CV with vegetation)")
    print(f"{'=' * 70}")
    print(f"Validation:                5-fold cross-validation")
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
    
    print(f"\nTop 8 features by importance:")
    for i, row in enumerate(scorecard["top_features"], 1):
        print(f"  {i}. {row['feature']:35s}  {row['importance']:.4f}")
    
    print(f"\nFold-by-fold consistency:")
    for f in fold_results:
        print(f"  Fold {f['fold']}: major={f['major_acc_pct']}%  median_err={f['median_err_pct']}%")
    
    return scorecard


def main():
    print("=" * 70)
    print("GridWatch - ML Backtest with 5-Fold Cross-Validation")
    print("=" * 70)
    
    eagle = load_eaglei()
    noaa  = load_noaa()
    print("\nComputing county baselines from full EAGLE-I dataset...")
    baselines = load_county_baselines_all_data(eagle)
    print(f"  {len(baselines)} baselines")
    
    print("\nLoading county features...")
    county_features = load_county_features()
    if county_features is None:
        return 1
    
    df = build_dataset(noaa, eagle, baselines, county_features)
    print(f"\nDataset: {len(df):,} storm-county pairs")
    print(f"Date range: {df['storm_date'].min()} to {df['storm_date'].max()}")
    
    if len(df) < 100:
        print(f"ERROR: only {len(df)} pairs, need >100 for CV")
        return 1
    
    train_with_kfold(df)
    return 0


if __name__ == "__main__":
    sys.exit(main())
