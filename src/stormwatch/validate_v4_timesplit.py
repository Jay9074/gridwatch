"""
GridWatch - v4 Time-Split Validation (integrity check)
=======================================================
v4's official 31.8% came from RANDOM 5-fold CV. Since random CV fooled us on
v5 (8.9% CV -> 68.4% time-split), we must check v4 honestly too.

Reuses precomputed ml_backtest_v4_results.csv. Trains on past, predicts future.

Run: python src/stormwatch/validate_v4_timesplit.py
"""
import pandas as pd
import numpy as np
import warnings
from pathlib import Path
import sys, time

warnings.filterwarnings("ignore")
RESULTS = Path("data/stormwatch/backtest/ml_backtest_v4_results.csv")

def main():
    print("="*70)
    print("GridWatch - v4 TIME-SPLIT Validation (integrity check)")
    print("="*70)
    if not RESULTS.exists():
        print(f"ERROR: {RESULTS} not found.")
        return 1
    
    df = pd.read_csv(RESULTS)
    df["storm_date"] = pd.to_datetime(df["storm_date"], errors="coerce")
    if df["storm_date"].dt.tz is not None:
        df["storm_date"] = df["storm_date"].dt.tz_localize(None)
    print(f"Loaded {len(df):,} pairs")
    
    # v4 feature set
    feature_cols = [
        "tier_severe","tier_moderate","magnitude","storm_duration_hrs","log_duration",
        "month","month_sin","month_cos","is_winter","is_summer","is_hurricane_season",
        "type_ice","type_snow","type_winter_storm","type_hurricane",
        "type_tornado","type_thunderstorm","type_wind",
        "storms_30d_prior","storms_90d_prior","storms_365d_prior",
        "days_since_last_storm","log_days_since",
        "tree_canopy_pct","population_density","log_pop_density",
        "infrastructure_vulnerability","land_area_sqmi","log_pop","impervious_pct",
        "tier_x_canopy","tier_x_density",
        "baseline_typical","baseline_high","baseline_extreme",
    ]
    feature_cols = [c for c in feature_cols if c in df.columns]
    print(f"Using {len(feature_cols)} features")
    
    cutoff = pd.Timestamp("2023-07-01")
    train = df[df["storm_date"] < cutoff].copy()
    test  = df[df["storm_date"] >= cutoff].copy()
    print(f"Train: {len(train):,}  Test: {len(test):,}")
    
    Xtr, ytr = train[feature_cols].values, train["actual_customers"].values
    Xte, yte = test[feature_cols].values, test["actual_customers"].values
    sw = np.where(Xtr[:,0]==1, 2.0, np.where(Xtr[:,1]==1, 1.5, 1.0))
    
    from xgboost import XGBRegressor
    from lightgbm import LGBMRegressor
    
    print("Training XGBoost...")
    t=time.time()
    xgb = XGBRegressor(n_estimators=350,max_depth=9,learning_rate=0.097,
                       subsample=0.88,colsample_bytree=0.75,min_child_weight=2,
                       reg_alpha=0.06,reg_lambda=1.7,random_state=42,n_jobs=-1,verbosity=0)
    xgb.fit(Xtr, np.log1p(ytr), sample_weight=sw)
    print(f"  done {time.time()-t:.1f}s")
    lgb = LGBMRegressor(n_estimators=300,max_depth=8,learning_rate=0.08,
                        subsample=0.85,colsample_bytree=0.8,random_state=42,n_jobs=-1,verbosity=-1)
    lgb.fit(Xtr, np.log1p(ytr), sample_weight=sw)
    
    pred = np.maximum(0.6*np.expm1(xgb.predict(Xte)) + 0.4*np.expm1(lgb.predict(Xte)), 200)
    pct_err = np.where(yte>0, np.abs(pred-yte)/yte*100, np.where(pred==0,0,100))
    major_acc = ((pred>=1000)==(yte>=1000)).mean()
    
    print(f"\n{'='*70}")
    print("v4 TIME-SPLIT RESULTS")
    print(f"{'='*70}")
    print(f"Test storms (unseen future): {len(test):,}")
    print(f"Major outage accuracy:  {major_acc*100:.1f}%   (random-CV claimed 88.5%)")
    print(f"Median % error:         {np.median(pct_err):.1f}%   (random-CV claimed 31.8%)")
    print(f"Mean % error:           {np.mean(pct_err):.1f}%")
    print(f"\nPercentiles:")
    for p in [10,25,50,75,90,95]:
        print(f"  {p}th: {np.percentile(pct_err,p):.1f}%")
    print(f"\n{'='*70}")
    print("This is v4's HONEST forecasting accuracy. If much worse than 31.8%,")
    print("the dashboard/paper numbers need updating to the time-split figures.")
    print(f"{'='*70}")
    return 0

if __name__ == "__main__":
    sys.exit(main())
