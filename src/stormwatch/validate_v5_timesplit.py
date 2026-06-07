"""
GridWatch - v5 Time-Split Validation (the definitive test)
===========================================================
Random 5-fold CV can hide optimism if similar storms leak across folds.
The real test for a FORECASTING model: train on past, predict future.

Train: storms before 2023-07-01
Test:  storms 2023-07-01 onward (never-seen future storms)

If v5 holds (<~20% median error) on a true time split, the improvement is
real and deployable. If it collapses toward 30%, the random-CV number was
optimistic and v4 stays official.

Run: python src/stormwatch/validate_v5_timesplit.py
"""
import pandas as pd
import numpy as np
import warnings
from pathlib import Path
import sys

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).parent))

# Reuse v5's dataset builder
from backtest_ml_v5 import (
    load_eaglei, load_noaa, load_county_features, load_weather, build_dataset
)
from backtest_ml_v4 import compute_baselines_from_subset

def main():
    print("="*70)
    print("GridWatch - v5 TIME-SPLIT Validation (train past -> predict future)")
    print("="*70)
    
    eagle = load_eaglei()
    noaa  = load_noaa()
    cf = load_county_features()
    weather = load_weather()
    
    df = build_dataset(noaa, eagle, cf, weather)
    df["storm_date"] = pd.to_datetime(df["storm_date"])
    if df["storm_date"].dt.tz is not None:
        df["storm_date"] = df["storm_date"].dt.tz_localize(None)
    
    cutoff = pd.Timestamp("2023-07-01")
    train = df[df["storm_date"] < cutoff].copy()
    test  = df[df["storm_date"] >= cutoff].copy()
    print(f"\nTrain (before {cutoff.date()}): {len(train):,} storms")
    print(f"Test  ({cutoff.date()} onward): {len(test):,} storms")
    
    if len(test) < 100:
        print("ERROR: too few test storms for a meaningful split")
        return 1
    
    # Compute baselines from TRAINING dates only (leakage-safe)
    train_dates = pd.to_datetime(train["storm_date"]).dt.normalize()
    eagle["date"] = pd.to_datetime(eagle["date"])
    eagle_train = eagle[eagle["date"] < cutoff]
    fb = compute_baselines_from_subset(eagle_train)
    
    for d in [train, test]:
        d["baseline_typical"] = d.apply(lambda r: fb.get(f"{r['county']}, {r['state']}",{}).get("typical_major_outage",1500), axis=1)
        d["baseline_high"]    = d.apply(lambda r: fb.get(f"{r['county']}, {r['state']}",{}).get("high_outage",3000), axis=1)
        d["baseline_extreme"] = d.apply(lambda r: fb.get(f"{r['county']}, {r['state']}",{}).get("extreme_outage",8000), axis=1)
    
    feature_cols = [
        "tier_severe","tier_moderate","magnitude",
        "wind_speed_daily","wind_gust_daily",
        "storm_duration_hrs","log_duration",
        "month","month_sin","month_cos","is_winter","is_summer","is_hurricane_season",
        "leaf_on","ndvi_modeled",
        "type_ice","type_snow","type_winter_storm","type_hurricane",
        "type_tornado","type_thunderstorm","type_wind",
        "storms_30d_prior","storms_90d_prior","storms_365d_prior",
        "days_since_last_storm","log_days_since",
        "tree_canopy_pct","population_density","log_pop_density",
        "infrastructure_vulnerability","land_area_sqmi","log_pop","impervious_pct",
        "tier_x_canopy","tier_x_density",
        "wind_x_leafon","gust_x_leafon","wind_x_canopy",
        "ice_risk","snow_load","soil_saturation",
        "is_extreme_cold","is_extreme_heat","temp_mean",
        "wind_dir_sin","wind_dir_cos",
        "wind_x_soil","ice_x_canopy","snow_x_canopy",
        "baseline_typical","baseline_high","baseline_extreme",
    ]
    
    from xgboost import XGBRegressor
    from lightgbm import LGBMRegressor
    
    Xtr, ytr = train[feature_cols].values, train["actual_customers"].values
    Xte, yte = test[feature_cols].values, test["actual_customers"].values
    sw = np.where(Xtr[:,0]==1, 2.0, np.where(Xtr[:,1]==1, 1.5, 1.0))
    
    print("\nTraining on PAST, predicting FUTURE...")
    xgb = XGBRegressor(n_estimators=350,max_depth=9,learning_rate=0.097,
                       subsample=0.88,colsample_bytree=0.75,min_child_weight=2,
                       reg_alpha=0.06,reg_lambda=1.7,random_state=42,n_jobs=-1,verbosity=0)
    xgb.fit(Xtr, np.log1p(ytr), sample_weight=sw)
    lgb = LGBMRegressor(n_estimators=300,max_depth=8,learning_rate=0.08,
                        subsample=0.85,colsample_bytree=0.8,random_state=42,n_jobs=-1,verbosity=-1)
    lgb.fit(Xtr, np.log1p(ytr), sample_weight=sw)
    
    pred = np.maximum(0.6*np.expm1(xgb.predict(Xte)) + 0.4*np.expm1(lgb.predict(Xte)), 200)
    
    pct_err = np.where(yte>0, np.abs(pred-yte)/yte*100, np.where(pred==0,0,100))
    major_acc = ((pred>=1000)==(yte>=1000)).mean()
    crit_acc = ((pred>=10000)==(yte>=10000)).mean()
    
    print(f"\n{'='*70}")
    print("TIME-SPLIT RESULTS (the honest forecasting test)")
    print(f"{'='*70}")
    print(f"Test storms (future, unseen): {len(test):,}")
    print(f"Major outage accuracy:  {major_acc*100:.1f}%")
    print(f"Critical accuracy:      {crit_acc*100:.1f}%")
    print(f"Median % error:         {np.median(pct_err):.1f}%")
    print(f"Mean % error:           {np.mean(pct_err):.1f}%")
    print(f"\nError percentiles:")
    for p in [10,25,50,75,90,95]:
        print(f"  {p}th: {np.percentile(pct_err, p):.1f}%")
    
    print(f"\n{'='*70}")
    print("VERDICT")
    print(f"{'='*70}")
    print(f"  Random 5-fold CV said:  8.9% median error")
    print(f"  Time-split (honest):    {np.median(pct_err):.1f}% median error")
    print(f"  v4 baseline:            31.8% median error")
    print()
    med = np.median(pct_err)
    if med < 20:
        print(f"  -> v5 holds up on unseen future storms. Improvement is REAL.")
    elif med < 28:
        print(f"  -> v5 partially holds. Real but less dramatic than CV suggested.")
    else:
        print(f"  -> v5 collapses on time split. Random CV was optimistic. Keep v4.")
    return 0

if __name__ == "__main__":
    sys.exit(main())
