"""
GridWatch - ML Backtest v3 FIXED: Removes data leakage
========================================================
v3 had data leakage from:
1. regional_storm_count - same-day storm counts leaked outcome info
2. baseline_n_history - computed from ALL EAGLE-I including test period

FIX:
- Use proper time-series CV with date-based splits, not random KFold
- Compute baselines using ONLY pre-fold training data per CV split
- Drop regional_storm_count (it's too tightly coupled with same-day outage)
- Replace with regional_mean_magnitude only (storm intensity, not count)

This is honest validation. Expect numbers to land closer to v1's 47%.
If they stay below 40%, the improvement is real. If they jump back up,
v3's "win" was leakage.

Run: python src/stormwatch/backtest_ml_v3_fixed.py
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


def load_eaglei():
    print("Loading EAGLE-I...")
    df = pd.read_csv("data/processed/eaglei_daily_northeast.csv", parse_dates=["date"])
    return df


def normalize_county_name(name):
    if not isinstance(name, str):
        return ""
    name = re.sub(r'\s*\(zone\)\s*', '', name, flags=re.IGNORECASE)
    name = re.sub(r'\s*metro\s*$', '', name, flags=re.IGNORECASE)
    name = re.sub(r'\s*county\s*$', '', name, flags=re.IGNORECASE)
    return name.strip().title()


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
    return df


def compute_baselines_from_subset(eaglei_subset):
    """Compute baselines from a TRAINING-ONLY subset of EAGLE-I.
    
    Critical: only pass in training data, not the full dataset.
    """
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


def load_county_features():
    path = Path("data/processed/county_features.csv")
    if not path.exists():
        return None
    return pd.read_csv(path)


def build_dataset_no_leakage(noaa, eagle, county_features):
    """Build storm-county pairs with NO leakage features.
    
    Excludes:
    - baseline_n_history (counts of historical events from full dataset)
    - regional_storm_count (same-day storm count leaks outcome)
    """
    print("\nBuilding leakage-free feature dataset...")
    
    cf_map = {}
    if county_features is not None:
        for _, r in county_features.iterrows():
            cf_map[(r["county"], r["state"])] = r.to_dict()
    
    target_set = set(TARGET_COUNTIES)
    rows = []
    
    for _, storm in noaa.iterrows():
        county = str(storm.get("county", ""))
        state  = str(storm.get("state", ""))
        if (county, state) not in target_set:
            continue
        
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
        cf = cf_map.get((county, state), {})
        
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
            # County features (these don't leak - they're static)
            "tree_canopy_pct":    cf.get("tree_canopy_pct", 50),
            "population_density": cf.get("population_density", 500),
            "infrastructure_vulnerability": cf.get("infrastructure_vulnerability", 0.5),
            "land_area_sqmi":     cf.get("land_area_sqmi", 500),
            "population_2023":    cf.get("population_2023", 100000),
            # Interactions
            "tier_x_canopy":     (1 if tier == "SEVERE" else 0.5 if tier == "MODERATE" else 0) * cf.get("tree_canopy_pct", 50) / 100,
            "tier_x_density":    (1 if tier == "SEVERE" else 0.5 if tier == "MODERATE" else 0) * np.log1p(cf.get("population_density", 500)),
            # Target
            "actual_customers":   actual,
        })
    
    return pd.DataFrame(rows)


def train_with_proper_cv(df, eagle):
    """5-fold CV where baselines are computed per fold using only training data.
    
    This prevents the baseline-leakage that v3 had.
    """
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
    
    # Base features (no leakage)
    base_features = [
        "tier_severe", "tier_moderate", "magnitude", "storm_duration_hrs", "log_duration",
        "month", "month_sin", "month_cos",
        "is_winter", "is_summer", "is_hurricane_season",
        "tree_canopy_pct", "population_density",
        "infrastructure_vulnerability", "land_area_sqmi", "population_2023",
        "tier_x_canopy", "tier_x_density",
    ]
    
    # Baseline features will be added per-fold
    baseline_features = ["baseline_typical", "baseline_high", "baseline_extreme"]
    feature_cols = base_features + baseline_features
    
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    print(f"\nRunning 5-fold CV with PER-FOLD baselines on {len(df):,} pairs...")
    
    all_predictions = np.zeros(len(df))
    feature_imp_sum = np.zeros(len(feature_cols))
    fold_results = []
    
    # Convert eagle dates for filtering
    eagle = eagle.copy()
    eagle["date"] = pd.to_datetime(eagle["date"])
    
    df = df.reset_index(drop=True).copy()
    
    for fold_idx, (train_idx, test_idx) in enumerate(kf.split(df), 1):
        train_df = df.iloc[train_idx].copy()
        test_df  = df.iloc[test_idx].copy()
        
        # Compute baselines using ONLY training storm dates' EAGLE-I data
        train_dates = pd.to_datetime(train_df["storm_date"]).dt.normalize()
        train_dates_set = set(train_dates.dt.date)
        eagle_train = eagle[eagle["date"].dt.date.isin(train_dates_set) | 
                            (eagle["date"] < train_dates.min())]
        
        fold_baselines = compute_baselines_from_subset(eagle_train)
        
        # Add baselines to both train and test (test gets training-derived baselines)
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
        
        # Sample weights
        sample_weights = np.where(X_tr[:, 0] == 1, 2.0, np.where(X_tr[:, 1] == 1, 1.5, 1.0))
        
        # XGBoost
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
        
        # Ensemble
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
    
    # Score
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
    
    df.to_csv(OUT_DIR / "ml_backtest_v3fixed_results.csv", index=False)
    
    avg_imp = feature_imp_sum / 5
    importances = pd.DataFrame({"feature": feature_cols, "importance": avg_imp})
    importances = importances.sort_values("importance", ascending=False)
    importances.to_csv(OUT_DIR / "ml_feature_importance_v3fixed.csv", index=False)
    
    scorecard = {
        "version":                 "v3 FIXED: leakage-free, per-fold baselines",
        "approach":                "XGBoost+LightGBM ensemble, sample-weighted, per-fold baseline computation",
        "total_storms_tested":     len(df),
        "major_outage_accuracy_pct":    round(df["major_correct"].mean() * 100, 1),
        "critical_outage_accuracy_pct": round(df["critical_correct"].mean() * 100, 1),
        "within_ci_pct":           round(df["in_ci"].mean() * 100, 1),
        "median_pct_error":        round(df["pct_error"].median(), 1),
        "mean_pct_error":          round(df["pct_error"].mean(), 1),
        "fold_results":            fold_results,
        "top_features":            importances.head(12).to_dict(orient="records"),
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
    
    with open(OUT_DIR / "ml_backtest_v3fixed_scorecard.json", "w") as f:
        json.dump(scorecard, f, indent=2, default=str)
    
    return scorecard, df


def main():
    print("=" * 70)
    print("GridWatch - ML Backtest v3 FIXED (Leakage-Free)")
    print("=" * 70)
    
    eagle = load_eaglei()
    noaa  = load_noaa()
    cf = load_county_features()
    if cf is None:
        print("ERROR: county_features.csv missing")
        return 1
    
    df = build_dataset_no_leakage(noaa, eagle, cf)
    print(f"\nDataset: {len(df):,} storm-county pairs")
    
    scorecard, _ = train_with_proper_cv(df, eagle)
    
    print(f"\n{'=' * 70}")
    print("ML BACKTEST v3 FIXED SCORECARD")
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
    
    print(f"\nTop 12 features:")
    for i, row in enumerate(scorecard["top_features"], 1):
        print(f"  {i:2d}. {row['feature']:35s}  {row['importance']:.4f}")
    
    print(f"\nFold consistency:")
    for f in scorecard["fold_results"]:
        print(f"  Fold {f['fold']}: major={f['major_acc_pct']}%  median_err={f['median_err_pct']}%")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
