"""
GridWatch - Count-Error Ceiling Verification (multi-split)
===========================================================
Before accepting that count prediction tops out ~68% on public data, verify
the ceiling is stable across DIFFERENT temporal train/test boundaries - not
an artifact of one specific 2023-07-01 split.

Tests v5-lite (best feature set) across several time cutoffs. If median count
error clusters ~65-70% regardless of cutoff, the ceiling is real.

Also reports classification metrics at each split (the real product) to
confirm THAT is stable too.

Run: python src/stormwatch/verify_ceiling_multisplit.py
"""
import pandas as pd
import numpy as np
import warnings
from pathlib import Path
import sys, time

warnings.filterwarnings("ignore")
RESULTS = Path("data/stormwatch/backtest/ml_backtest_v5_results.csv")

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


def run_split(df, fc, cutoff):
    from xgboost import XGBRegressor
    from lightgbm import LGBMRegressor
    
    train = df[df["storm_date"] < cutoff]
    test  = df[df["storm_date"] >= cutoff]
    if len(test) < 80 or len(train) < 300:
        return None
    
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
    
    pm, am = pred>=1000, yte>=1000
    tp=(pm&am).sum(); fp=(pm&~am).sum(); fn=(~pm&am).sum()
    prec = tp/(tp+fp) if (tp+fp)>0 else 0
    rec = tp/(tp+fn) if (tp+fn)>0 else 0
    f1 = 2*prec*rec/(prec+rec) if (prec+rec)>0 else 0
    
    return {
        "cutoff": str(cutoff.date()),
        "n_train": len(train), "n_test": len(test),
        "major_acc": round((pm==am).mean()*100,1),
        "precision": round(prec*100,1),
        "recall": round(rec*100,1),
        "f1": round(f1*100,1),
        "median_count_err": round(np.median(pe),1),
        "mean_count_err": round(np.mean(pe),1),
    }


def main():
    print("="*72)
    print("GridWatch - Count-Error Ceiling Verification (multi-split)")
    print("="*72)
    if not RESULTS.exists():
        print(f"ERROR: {RESULTS} missing"); return 1
    
    df = pd.read_csv(RESULTS)
    df["storm_date"] = pd.to_datetime(df["storm_date"], errors="coerce")
    if df["storm_date"].dt.tz is not None:
        df["storm_date"] = df["storm_date"].dt.tz_localize(None)
    fc = [c for c in V5LITE if c in df.columns]
    
    dmin, dmax = df["storm_date"].min(), df["storm_date"].max()
    print(f"Data spans {dmin.date()} to {dmax.date()}, {len(df):,} storms")
    print(f"v5-lite: {len(fc)} features\n")
    
    # Multiple temporal cutoffs across the data range
    cutoffs = [
        pd.Timestamp("2022-07-01"),
        pd.Timestamp("2023-01-01"),
        pd.Timestamp("2023-07-01"),
        pd.Timestamp("2024-01-01"),
        pd.Timestamp("2024-07-01"),
    ]
    
    print("Running splits (each ~3s)...\n")
    rows = []
    for c in cutoffs:
        r = run_split(df, fc, c)
        if r: rows.append(r)
    
    print(f"{'Cutoff':<12} {'Train':>6} {'Test':>5} {'MajAcc':>7} {'Prec':>6} {'Recall':>7} {'F1':>6} {'CountErr':>9}")
    print("="*72)
    for r in rows:
        print(f"{r['cutoff']:<12} {r['n_train']:>6} {r['n_test']:>5} "
              f"{r['major_acc']:>6}% {r['precision']:>5}% {r['recall']:>6}% "
              f"{r['f1']:>5}% {r['median_count_err']:>8}%")
    
    # Stability analysis
    count_errs = [r["median_count_err"] for r in rows]
    f1s = [r["f1"] for r in rows]
    accs = [r["major_acc"] for r in rows]
    
    print(f"\n{'='*72}")
    print("STABILITY ANALYSIS")
    print(f"{'='*72}")
    print(f"Count error across splits:  min={min(count_errs)}%  max={max(count_errs)}%  "
          f"mean={np.mean(count_errs):.1f}%  std={np.std(count_errs):.1f}")
    print(f"Major accuracy across splits: min={min(accs)}%  max={max(accs)}%  "
          f"mean={np.mean(accs):.1f}%  std={np.std(accs):.1f}")
    print(f"F1 across splits:            min={min(f1s)}%  max={max(f1s)}%  "
          f"mean={np.mean(f1s):.1f}%  std={np.std(f1s):.1f}")
    
    print(f"\n{'='*72}")
    print("VERDICT")
    print(f"{'='*72}")
    if np.mean(count_errs) > 50 and np.std(count_errs) < 15:
        print(f"  COUNT CEILING CONFIRMED: error stable at ~{np.mean(count_errs):.0f}% across")
        print(f"  all temporal splits. This is a real data ceiling, not an artifact.")
        print(f"  -> Count prediction is NOT achievable on public data. Accept classifier framing.")
    elif np.mean(count_errs) <= 50:
        print(f"  Count error averages {np.mean(count_errs):.0f}% - below 50%. Worth more investigation.")
    else:
        print(f"  Count error varies widely (std={np.std(count_errs):.1f}) - split-dependent, inconclusive.")
    print()
    if np.mean(f1s) > 85 and np.std(f1s) < 10:
        print(f"  CLASSIFIER CONFIRMED STABLE: F1 ~{np.mean(f1s):.0f}% across all splits.")
        print(f"  This is your real, robust, defensible product.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
