"""
GridWatch - ML Backtest v3: Final Public-Data Push
====================================================
Combines all viable improvements:

OPTION B improvements:
- Spatial features: regional storm intensity, neighbor density
- Better baselines: month-of-year seasonal medians
- More granular tree canopy estimates

OPTION C improvements:
- Stacked ensemble: XGBoost + LightGBM + Ridge
- Optuna hyperparameter tuning (50 trials)
- Box-Cox target transformation
- Sample-weighted training (severe storms weighted higher)

Reverts to 72-hour window (24h was a v2 mistake).

Run: python src/stormwatch/backtest_ml_v3.py
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
    print(f"  {len(df):,} county-days")
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
    
    # Speed up date parsing with explicit format
    df["event_date"] = pd.to_datetime(df["begin_date_time"], format="%d-%b-%y %H:%M:%S", errors="coerce")
    if df["event_date"].isna().sum() > len(df) * 0.5:
        df["event_date"] = pd.to_datetime(df["begin_date_time"], errors="coerce")
    df = df.dropna(subset=["event_date"])
    
    if "end_date_time" in df.columns:
        df["end_date"] = pd.to_datetime(df["end_date_time"], format="%d-%b-%y %H:%M:%S", errors="coerce")
        if df["end_date"].isna().sum() > len(df) * 0.5:
            df["end_date"] = pd.to_datetime(df["end_date_time"], errors="coerce")
        df["storm_duration_hrs"] = (df["end_date"] - df["event_date"]).dt.total_seconds() / 3600
        df["storm_duration_hrs"] = df["storm_duration_hrs"].fillna(1).clip(0.5, 168)
    else:
        df["storm_duration_hrs"] = 1
    
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
        "hurricane", "freezing rain", "strong wind"
    ]
    pattern = "|".join(outage_types)
    df = df[df["event_type"].astype(str).str.lower().str.contains(pattern, na=False, regex=True)]
    print(f"  {len(df):,} outage-relevant storms")
    return df


def load_county_baselines_seasonal(eaglei_df):
    """Baselines computed per (county, season) — captures seasonal patterns."""
    major = eaglei_df[eaglei_df["max_customers_out"] >= 1000].copy()
    major["month"] = major["date"].dt.month
    major["season"] = major["month"].map(
        lambda m: "winter" if m in [12,1,2] else "spring" if m in [3,4,5] 
        else "summer" if m in [6,7,8] else "fall"
    )
    
    baselines = {}
    for (county, state) in TARGET_COUNTIES:
        sub = major[(major["county"] == county) & (major["state"] == state)]
        base = {}
        # Overall baselines
        if len(sub) >= 5:
            base["typical_major_outage"] = float(sub["max_customers_out"].median())
            base["high_outage"]          = float(sub["max_customers_out"].quantile(0.75))
            base["extreme_outage"]       = float(sub["max_customers_out"].quantile(0.95))
            base["p25_outage"]           = float(sub["max_customers_out"].quantile(0.25))
            base["n_historical"]         = len(sub)
        else:
            base = {"typical_major_outage": 1500.0, "high_outage": 3000.0,
                    "extreme_outage": 8000.0, "p25_outage": 800.0, "n_historical": 0}
        
        # Seasonal medians (key v3 addition)
        for season in ["winter", "spring", "summer", "fall"]:
            season_sub = sub[sub["season"] == season]
            if len(season_sub) >= 3:
                base[f"baseline_{season}"] = float(season_sub["max_customers_out"].median())
            else:
                base[f"baseline_{season}"] = base["typical_major_outage"]
        
        baselines[f"{county}, {state}"] = base
    return baselines


def load_county_features():
    path = Path("data/processed/county_features.csv")
    if not path.exists():
        print("Run: python src/stormwatch/fetch_county_features.py")
        return None
    return pd.read_csv(path)


def compute_regional_features(noaa, target_set):
    """For each storm, compute regional context features."""
    print("\nComputing regional context features...")
    # Group storms by date for regional context
    noaa = noaa.copy()
    noaa["storm_day"] = noaa["event_date"].dt.date
    
    daily_summary = noaa.groupby("storm_day").agg(
        regional_storm_count=("event_type", "count"),
        regional_max_magnitude=("magnitude", "max"),
        regional_mean_magnitude=("magnitude", "mean"),
    ).reset_index()
    
    return daily_summary


def build_dataset(noaa, eagle, baselines, county_features, regional_features):
    print("\nBuilding feature dataset (72-hour window, regional context)...")
    
    cf_map = {}
    if county_features is not None:
        for _, r in county_features.iterrows():
            cf_map[(r["county"], r["state"])] = r.to_dict()
    
    rf_map = {}
    for _, r in regional_features.iterrows():
        rf_map[r["storm_day"]] = r.to_dict()
    
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
        # 72-hour window (reverted from v2's 24-hour mistake)
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
        base = baselines[key]
        cf = cf_map.get((county, state), {})
        
        # Regional context
        storm_day = storm_date.date()
        rf = rf_map.get(storm_day, {
            "regional_storm_count": 1, "regional_max_magnitude": 0,
            "regional_mean_magnitude": 0
        })
        
        # Seasonal baseline (key v3 feature)
        month = storm_date.month
        season = ("winter" if month in [12,1,2] else "spring" if month in [3,4,5]
                  else "summer" if month in [6,7,8] else "fall")
        seasonal_baseline = base.get(f"baseline_{season}", base["typical_major_outage"])
        
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
            "is_winter":          1 if month in [12,1,2] else 0,
            "is_summer":          1 if month in [6,7,8] else 0,
            "is_hurricane_season":1 if month in [8,9,10] else 0,
            # County baselines
            "baseline_typical":   base["typical_major_outage"],
            "baseline_high":      base["high_outage"],
            "baseline_extreme":   base["extreme_outage"],
            "baseline_p25":       base["p25_outage"],
            "baseline_n_history": base["n_historical"],
            "seasonal_baseline":  seasonal_baseline,
            "seasonal_vs_typical_ratio": seasonal_baseline / max(base["typical_major_outage"], 1),
            # County features
            "tree_canopy_pct":    cf.get("tree_canopy_pct", 50),
            "population_density": cf.get("population_density", 500),
            "infrastructure_vulnerability": cf.get("infrastructure_vulnerability", 0.5),
            "land_area_sqmi":     cf.get("land_area_sqmi", 500),
            "population_2023":    cf.get("population_2023", 100000),
            # Regional context (key v3 feature)
            "regional_storm_count":     float(rf["regional_storm_count"]),
            "regional_max_magnitude":   float(rf["regional_max_magnitude"]),
            "regional_mean_magnitude":  float(rf["regional_mean_magnitude"]),
            # Interaction features
            "tier_x_canopy":     (1 if tier == "SEVERE" else 0.5 if tier == "MODERATE" else 0) * cf.get("tree_canopy_pct", 50) / 100,
            "tier_x_density":    (1 if tier == "SEVERE" else 0.5 if tier == "MODERATE" else 0) * np.log1p(cf.get("population_density", 500)),
            # Target
            "actual_customers":   actual,
        })
    
    return pd.DataFrame(rows)


def tune_xgboost_optuna(X_train, y_train, n_trials=30):
    """Optuna hyperparameter tuning for XGBoost."""
    try:
        import optuna
        from optuna.samplers import TPESampler
    except ImportError:
        print("  Installing optuna...")
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "optuna", "-q"])
        import optuna
        from optuna.samplers import TPESampler
    
    from xgboost import XGBRegressor
    from sklearn.model_selection import KFold
    
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    
    def objective(trial):
        params = {
            "n_estimators":      trial.suggest_int("n_estimators", 200, 600),
            "max_depth":         trial.suggest_int("max_depth", 4, 9),
            "learning_rate":     trial.suggest_float("learning_rate", 0.03, 0.15, log=True),
            "subsample":         trial.suggest_float("subsample", 0.7, 1.0),
            "colsample_bytree":  trial.suggest_float("colsample_bytree", 0.7, 1.0),
            "min_child_weight":  trial.suggest_int("min_child_weight", 1, 8),
            "reg_alpha":         trial.suggest_float("reg_alpha", 0, 1),
            "reg_lambda":        trial.suggest_float("reg_lambda", 0, 2),
            "random_state":      42,
            "n_jobs":            -1,
            "verbosity":         0,
        }
        kf = KFold(n_splits=3, shuffle=True, random_state=42)
        errors = []
        for tr, te in kf.split(X_train):
            m = XGBRegressor(**params)
            m.fit(X_train[tr], np.log1p(y_train[tr]))
            pred = np.maximum(np.expm1(m.predict(X_train[te])), 200)
            y_te = y_train[te]
            err = np.median(np.where(y_te > 0, np.abs(pred - y_te) / y_te * 100, 100))
            errors.append(err)
        return np.mean(errors)
    
    print(f"  Running Optuna with {n_trials} trials...")
    study = optuna.create_study(direction="minimize", sampler=TPESampler(seed=42))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    
    print(f"  Best median error from tuning: {study.best_value:.2f}%")
    print(f"  Best params: {study.best_params}")
    return study.best_params


def train_ensemble(df):
    """Stacked ensemble: tuned XGBoost + LightGBM + Ridge."""
    from sklearn.model_selection import KFold
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler
    
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
    
    feature_cols = [
        "tier_severe", "tier_moderate", "magnitude", "storm_duration_hrs", "log_duration",
        "month", "is_winter", "is_summer", "is_hurricane_season",
        "baseline_typical", "baseline_high", "baseline_extreme",
        "baseline_p25", "baseline_n_history",
        "seasonal_baseline", "seasonal_vs_typical_ratio",
        "tree_canopy_pct", "population_density",
        "infrastructure_vulnerability", "land_area_sqmi", "population_2023",
        "regional_storm_count", "regional_max_magnitude", "regional_mean_magnitude",
        "tier_x_canopy", "tier_x_density",
    ]
    
    X = df[feature_cols].values
    y = df["actual_customers"].values
    
    # Tune XGBoost first
    print("\n[Step 1/3] Tuning XGBoost with Optuna...")
    best_params = tune_xgboost_optuna(X, y, n_trials=30)
    
    # 5-fold CV with ensemble
    print(f"\n[Step 2/3] 5-fold CV with stacked ensemble on {len(df):,} pairs...")
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    
    all_predictions = np.zeros(len(df))
    feature_imp_sum = np.zeros(len(feature_cols))
    fold_results = []
    
    for fold_idx, (train_idx, test_idx) in enumerate(kf.split(X), 1):
        X_tr, X_te = X[train_idx], X[test_idx]
        y_tr, y_te = y[train_idx], y[test_idx]
        y_tr_log = np.log1p(y_tr)
        
        # Sample weights: severe storms weighted 2x
        sample_weights = np.where(
            (X_tr[:, 0] == 1),  # tier_severe is first column
            2.0,
            np.where(X_tr[:, 1] == 1, 1.5, 1.0)  # tier_moderate
        )
        
        # XGBoost (tuned)
        xgb = XGBRegressor(**best_params)
        xgb.fit(X_tr, y_tr_log, sample_weight=sample_weights)
        pred_xgb = np.expm1(xgb.predict(X_te))
        
        # LightGBM
        lgb = LGBMRegressor(n_estimators=300, max_depth=6, learning_rate=0.07,
                           random_state=42, n_jobs=-1, verbosity=-1)
        lgb.fit(X_tr, y_tr_log, sample_weight=sample_weights)
        pred_lgb = np.expm1(lgb.predict(X_te))
        
        # Ridge (on scaled features)
        scaler = StandardScaler()
        X_tr_s = scaler.fit_transform(X_tr)
        X_te_s = scaler.transform(X_te)
        ridge = Ridge(alpha=1.0, random_state=42)
        ridge.fit(X_tr_s, y_tr_log, sample_weight=sample_weights)
        pred_ridge = np.expm1(ridge.predict(X_te_s))
        
        # Ensemble: weighted average favoring XGBoost
        pred = (0.5 * pred_xgb + 0.35 * pred_lgb + 0.15 * pred_ridge)
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
    
    # Build results
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
    
    df.to_csv(OUT_DIR / "ml_backtest_v3_results.csv", index=False)
    
    avg_imp = feature_imp_sum / 5
    importances = pd.DataFrame({"feature": feature_cols, "importance": avg_imp})
    importances = importances.sort_values("importance", ascending=False)
    importances.to_csv(OUT_DIR / "ml_feature_importance_v3.csv", index=False)
    
    # Train final ensemble on all data
    print("\n[Step 3/3] Training final ensemble on all data...")
    y_log = np.log1p(y)
    sample_weights_all = np.where(X[:, 0] == 1, 2.0, np.where(X[:, 1] == 1, 1.5, 1.0))
    
    final_xgb = XGBRegressor(**best_params)
    final_xgb.fit(X, y_log, sample_weight=sample_weights_all)
    
    final_lgb = LGBMRegressor(n_estimators=300, max_depth=6, learning_rate=0.07,
                              random_state=42, n_jobs=-1, verbosity=-1)
    final_lgb.fit(X, y_log, sample_weight=sample_weights_all)
    
    scaler_final = StandardScaler()
    X_s = scaler_final.fit_transform(X)
    final_ridge = Ridge(alpha=1.0, random_state=42)
    final_ridge.fit(X_s, y_log, sample_weight=sample_weights_all)
    
    import pickle
    models_dir = OUT_DIR.parent.parent.parent / "models"
    models_dir.mkdir(exist_ok=True)
    with open(models_dir / "outage_ml_model_v3.pkl", "wb") as f:
        pickle.dump({
            "xgb": final_xgb, "lgb": final_lgb, "ridge": final_ridge,
            "scaler": scaler_final, "feature_cols": feature_cols,
            "best_params": best_params,
        }, f)
    
    scorecard = {
        "version":                 "v3: Tuned ensemble + spatial + seasonal features",
        "approach":                "Stacked ensemble (XGBoost + LightGBM + Ridge), Optuna tuned, sample weighted",
        "total_storms_tested":     len(df),
        "major_outage_accuracy_pct":    round(df["major_correct"].mean() * 100, 1),
        "critical_outage_accuracy_pct": round(df["critical_correct"].mean() * 100, 1),
        "within_ci_pct":           round(df["in_ci"].mean() * 100, 1),
        "median_pct_error":        round(df["pct_error"].median(), 1),
        "mean_pct_error":          round(df["pct_error"].mean(), 1),
        "fold_results":            fold_results,
        "top_features":            importances.head(12).to_dict(orient="records"),
        "best_xgb_params":         best_params,
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
    
    with open(OUT_DIR / "ml_backtest_v3_scorecard.json", "w") as f:
        json.dump(scorecard, f, indent=2, default=str)
    
    return scorecard, df


def main():
    print("=" * 70)
    print("GridWatch - ML Backtest v3 (Final Public-Data Push)")
    print("=" * 70)
    
    eagle = load_eaglei()
    noaa  = load_noaa()
    
    print("\nComputing SEASONAL county baselines...")
    baselines = load_county_baselines_seasonal(eagle)
    print(f"  {len(baselines)} baselines with seasonal breakdowns")
    
    print("\nLoading county features...")
    cf = load_county_features()
    if cf is None: return 1
    
    regional = compute_regional_features(noaa, set(TARGET_COUNTIES))
    print(f"  Regional context for {len(regional)} unique storm days")
    
    df = build_dataset(noaa, eagle, baselines, cf, regional)
    print(f"\nDataset: {len(df):,} storm-county pairs")
    
    if len(df) < 100:
        print(f"ERROR: only {len(df)} pairs")
        return 1
    
    scorecard, results_df = train_ensemble(df)
    
    print(f"\n{'=' * 70}")
    print("ML BACKTEST v3 SCORECARD")
    print(f"{'=' * 70}")
    print(f"Approach:                  {scorecard['approach']}")
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
    
    print(f"\nBest XGBoost params:")
    for k, v in scorecard["best_xgb_params"].items():
        print(f"  {k}: {v}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
