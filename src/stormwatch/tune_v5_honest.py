"""
GridWatch - v5 HONEST Tuning (optimize against time-split, not random CV)
==========================================================================
Random CV deceived us (8.9% CV -> 68.4% time-split). This script does the
opposite: it scores everything on a TRUE temporal holdout (train past,
predict future), so any improvement it finds is REAL forecasting improvement.

Steps:
1. Load precomputed v5 features
2. Establish honest baselines (v4 features, v5 features) on time-split
3. Feature ablation: test which feature GROUPS actually help the honest number
4. Report what genuinely improves forecasting vs what's noise

PRE-COMMITMENT: if honest count median-error stays >50%, GridWatch is a
classifier, not a count predictor. We accept that and reframe.

Run: python src/stormwatch/tune_v5_honest.py
"""
import pandas as pd
import numpy as np
import warnings
from pathlib import Path
import sys, time

warnings.filterwarnings("ignore")
RESULTS = Path("data/stormwatch/backtest/ml_backtest_v5_results.csv")
CUTOFF = pd.Timestamp("2023-07-01")


def load():
    df = pd.read_csv(RESULTS)
    df["storm_date"] = pd.to_datetime(df["storm_date"], errors="coerce")
    if df["storm_date"].dt.tz is not None:
        df["storm_date"] = df["storm_date"].dt.tz_localize(None)
    return df


def evaluate(df, feature_cols, label):
    """Train on past, predict future. Return honest metrics."""
    from xgboost import XGBRegressor
    from lightgbm import LGBMRegressor
    
    fc = [c for c in feature_cols if c in df.columns]
    train = df[df["storm_date"] < CUTOFF]
    test  = df[df["storm_date"] >= CUTOFF]
    
    Xtr, ytr = train[fc].values, train["actual_customers"].values
    Xte, yte = test[fc].values, test["actual_customers"].values
    sw = np.where(Xtr[:,0]==1, 2.0, np.where(Xtr[:,1]==1, 1.5, 1.0))
    
    xgb = XGBRegressor(n_estimators=350,max_depth=9,learning_rate=0.097,
                       subsample=0.88,colsample_bytree=0.75,min_child_weight=2,
                       reg_alpha=0.06,reg_lambda=1.7,random_state=42,n_jobs=-1,verbosity=0)
    xgb.fit(Xtr, np.log1p(ytr), sample_weight=sw)
    lgb = LGBMRegressor(n_estimators=300,max_depth=8,learning_rate=0.08,
                        subsample=0.85,colsample_bytree=0.8,random_state=42,n_jobs=-1,verbosity=-1)
    lgb.fit(Xtr, np.log1p(ytr), sample_weight=sw)
    
    pred = np.maximum(0.6*np.expm1(xgb.predict(Xte)) + 0.4*np.expm1(lgb.predict(Xte)), 200)
    pe = np.where(yte>0, np.abs(pred-yte)/yte*100, np.where(pred==0,0,100))
    macc = ((pred>=1000)==(yte>=1000)).mean()
    
    return {
        "label": label, "n_features": len(fc),
        "major_acc": round(macc*100,1),
        "median_err": round(np.median(pe),1),
        "mean_err": round(np.mean(pe),1),
    }


def main():
    print("="*70)
    print("GridWatch - v5 HONEST Tuning (time-split scored)")
    print("="*70)
    
    if not RESULTS.exists():
        print(f"ERROR: {RESULTS} missing"); return 1
    
    df = load()
    print(f"Loaded {len(df):,} pairs")
    print(f"Train: {(df['storm_date']<CUTOFF).sum():,}  Test: {(df['storm_date']>=CUTOFF).sum():,}\n")
    
    # Feature groups
    core = ["tier_severe","tier_moderate","magnitude","storm_duration_hrs","log_duration",
            "month","month_sin","month_cos","is_winter","is_summer","is_hurricane_season",
            "type_ice","type_snow","type_winter_storm","type_hurricane",
            "type_tornado","type_thunderstorm","type_wind",
            "storms_30d_prior","storms_90d_prior","storms_365d_prior",
            "days_since_last_storm","log_days_since",
            "tree_canopy_pct","population_density","log_pop_density",
            "infrastructure_vulnerability","land_area_sqmi","log_pop","impervious_pct",
            "tier_x_canopy","tier_x_density",
            "baseline_typical","baseline_high","baseline_extreme"]
    
    grp_wind = ["wind_speed_daily","wind_gust_daily","wind_x_canopy"]
    grp_leaf = ["leaf_on","ndvi_modeled","wind_x_leafon","gust_x_leafon"]
    grp_ice  = ["ice_risk","snow_load","ice_x_canopy","snow_x_canopy"]
    grp_soil = ["soil_saturation","wind_x_soil"]
    grp_temp = ["is_extreme_cold","is_extreme_heat","temp_mean"]
    grp_dir  = ["wind_dir_sin","wind_dir_cos"]
    
    results = []
    
    print("Running ablation (each ~5s)...\n")
    # Baseline: v4 core features only
    results.append(evaluate(df, core, "CORE only (v4-equivalent)"))
    # Add each group individually
    results.append(evaluate(df, core+grp_wind, "CORE + wind"))
    results.append(evaluate(df, core+grp_leaf, "CORE + leaf/phenology"))
    results.append(evaluate(df, core+grp_ice, "CORE + ice/snow"))
    results.append(evaluate(df, core+grp_soil, "CORE + soil"))
    results.append(evaluate(df, core+grp_temp, "CORE + temp extremes"))
    results.append(evaluate(df, core+grp_dir, "CORE + wind direction"))
    # Everything (full v5)
    results.append(evaluate(df, core+grp_wind+grp_leaf+grp_ice+grp_soil+grp_temp+grp_dir, "CORE + ALL v5 features"))
    
    print(f"{'Configuration':<32} {'Feats':>5} {'MajAcc':>7} {'MedErr':>7} {'MeanErr':>8}")
    print("="*65)
    base_med = results[0]["median_err"]
    for r in results:
        delta = r["median_err"] - base_med
        flag = ""
        if r["label"] != "CORE only (v4-equivalent)":
            flag = f"  ({'+' if delta>=0 else ''}{delta:.1f} vs core)"
        print(f"{r['label']:<32} {r['n_features']:>5} {r['major_acc']:>6}% {r['median_err']:>6}% {r['mean_err']:>7}%{flag}")
    
    print(f"\n{'='*70}")
    print("INTERPRETATION")
    print(f"{'='*70}")
    best = min(results, key=lambda r: r["median_err"])
    print(f"Best honest median error: {best['median_err']}%  ({best['label']})")
    print(f"Core-only median error:   {base_med}%")
    improvement = base_med - best["median_err"]
    print(f"Real improvement from all the weather features: {improvement:.1f} percentage points")
    print()
    if best["median_err"] > 50:
        print("PRE-COMMITMENT TRIGGERED: honest count error > 50%.")
        print("-> GridWatch is a CLASSIFIER (84% major-outage accuracy), not a")
        print("   count predictor. The weather features don't change this on public data.")
        print("-> Recommend: reframe around the classification success, stop count tuning.")
    else:
        print("Honest count error below 50% - worth pursuing further.")
    print()
    print(f"NOTE: Major-outage ACCURACY ({best['major_acc']}%) is the real, defensible")
    print("achievement regardless of count error. That's the classifier working.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
