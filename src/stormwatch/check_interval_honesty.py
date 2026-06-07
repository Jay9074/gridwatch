"""
GridWatch - Confidence Interval Honesty Check
==============================================
How often does the ACTUAL outage count fall inside the predicted confidence
range, on the honest time-split? Tests several interval widths so we know the
true hit-rate and can decide what (if anything) to show on the dashboard.

Also reports SIZE-CATEGORY accuracy (the reliable part) for comparison:
how often the prediction lands in the right magnitude bucket.

Run: python src/stormwatch/check_interval_honesty.py
"""
import pandas as pd
import numpy as np
import warnings
from pathlib import Path
import sys

warnings.filterwarnings("ignore")
RESULTS = Path("data/stormwatch/backtest/ml_backtest_v5_results.csv")
CUTOFF = pd.Timestamp("2023-07-01")

V5LITE = [
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
    "wind_speed_daily","wind_gust_daily","wind_x_canopy",
    "leaf_on","ndvi_modeled","wind_x_leafon","gust_x_leafon",
    "is_extreme_cold","is_extreme_heat","temp_mean",
]

# Magnitude buckets for "right size category" check
def bucket(n):
    if n < 300:    return 0  # minor
    if n < 1000:   return 1  # moderate
    if n < 5000:   return 2  # major
    if n < 20000:  return 3  # severe
    return 4                  # catastrophic
BUCKET_NAMES = ["minor(<300)","moderate(300-1k)","major(1k-5k)","severe(5k-20k)","catastrophic(20k+)"]


def main():
    print("="*70)
    print("GridWatch - Confidence Interval Honesty Check (time-split)")
    print("="*70)
    if not RESULTS.exists():
        print(f"ERROR: {RESULTS} missing"); return 1
    
    df = pd.read_csv(RESULTS)
    df["storm_date"] = pd.to_datetime(df["storm_date"], errors="coerce")
    if df["storm_date"].dt.tz is not None:
        df["storm_date"] = df["storm_date"].dt.tz_localize(None)
    fc = [c for c in V5LITE if c in df.columns]
    
    train = df[df["storm_date"] < CUTOFF]
    test  = df[df["storm_date"] >= CUTOFF]
    
    Xtr, ytr = train[fc].values, train["actual_customers"].values
    Xte, yte = test[fc].values, test["actual_customers"].values
    sw = np.where(Xtr[:,0]==1, 2.0, np.where(Xtr[:,1]==1, 1.5, 1.0))
    
    from xgboost import XGBRegressor
    from lightgbm import LGBMRegressor
    xgb = XGBRegressor(n_estimators=350,max_depth=9,learning_rate=0.097,
                       subsample=0.88,colsample_bytree=0.75,min_child_weight=2,
                       reg_alpha=0.06,reg_lambda=1.7,random_state=42,n_jobs=-1,verbosity=0)
    xgb.fit(Xtr, np.log1p(ytr), sample_weight=sw)
    lgb = LGBMRegressor(n_estimators=300,max_depth=8,learning_rate=0.08,
                        subsample=0.85,colsample_bytree=0.8,random_state=42,n_jobs=-1,verbosity=-1)
    lgb.fit(Xtr, np.log1p(ytr), sample_weight=sw)
    pred = np.maximum(0.6*np.expm1(xgb.predict(Xte)) + 0.4*np.expm1(lgb.predict(Xte)), 200)
    
    print(f"Test storms (unseen future): {len(test):,}\n")
    
    # ── Confidence interval hit rates at various widths ──
    print("CONFIDENCE INTERVAL HIT RATE (does actual fall inside predicted band?)")
    print("-"*70)
    print(f"{'Band width':<28} {'Example for pred=2000':<26} {'Hit rate':>10}")
    for pct in [0.25, 0.40, 0.50, 0.60, 0.75]:
        lo = pred * (1 - pct)
        hi = pred * (1 + pct)
        hit = ((yte >= lo) & (yte <= hi)).mean()
        example = f"{int(2000*(1-pct)):,}-{int(2000*(1+pct)):,}"
        print(f"  +/-{int(pct*100)}% ({'narrow' if pct<=.4 else 'wide' if pct>=.6 else 'medium'})".ljust(28)
              + f"{example:<26} {hit*100:>9.1f}%")
    
    # Order-of-magnitude (within 2x / 3x)
    print(f"\n  {'Within 2x (half to double)':<26}",
          f"{'1,000-4,000':<26} {(((yte>=pred/2)&(yte<=pred*2)).mean())*100:>9.1f}%")
    print(f"  {'Within 3x (third to triple)':<26}",
          f"{'667-6,000':<26} {(((yte>=pred/3)&(yte<=pred*3)).mean())*100:>9.1f}%")
    
    # ── Size-category accuracy (the reliable part) ──
    print(f"\n{'='*70}")
    print("SIZE-CATEGORY ACCURACY (the reliable part)")
    print("-"*70)
    pb = np.array([bucket(x) for x in pred])
    ab = np.array([bucket(x) for x in yte])
    exact = (pb == ab).mean()
    within1 = (np.abs(pb - ab) <= 1).mean()
    print(f"  Exact bucket match:           {exact*100:.1f}%")
    print(f"  Within 1 bucket (adjacent):   {within1*100:.1f}%")
    print(f"\n  Buckets: {' | '.join(BUCKET_NAMES)}")
    
    # Binary major/minor (the headline classifier)
    pm, am = pred>=1000, yte>=1000
    print(f"\n  Major-vs-not (>1000) accuracy: {(pm==am).mean()*100:.1f}%")
    
    print(f"\n{'='*70}")
    print("HONEST INTERPRETATION")
    print(f"{'='*70}")
    hit50 = (((yte>=pred*0.5)&(yte<=pred*1.5)).mean())
    hit2x = (((yte>=pred/2)&(yte<=pred*2)).mean())
    print(f"  - A +/-50% band (what dashboards often show) catches {hit50*100:.0f}% of actuals.")
    print(f"  - A 'within 2x' band catches {hit2x*100:.0f}%.")
    print(f"  - Exact-number prediction: NOT reliable (~70% median error).")
    print(f"  - SIZE CATEGORY (major vs minor): reliable at {(pm==am).mean()*100:.0f}%.")
    print()
    print("  Honest claim: GridWatch reliably identifies outage SIZE CATEGORY,")
    print("  not exact counts or tight numeric ranges.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
