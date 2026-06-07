"""
GridWatch - ML Backtest v5: Leaf-On Phenology + Granular Wind
==============================================================
EXPERIMENTAL - does NOT touch v4 production model.

New features over v4 (based on CMP ops feedback):
- leaf_on: modeled vegetation state (leaf-on vs leaf-off) from latitude +
  day-of-year + growing-degree-day phenology model. Captures the "trees have
  leaves or not" effect ops highlighted.
- ndvi_modeled: continuous greenness estimate (0-1)
- wind_speed_daily / wind_gust_daily: actual daily wind from Open-Meteo
  (if storm_weather_v5.csv exists), else falls back to NOAA magnitude
- wind_x_leafon: the key interaction - high wind + full leaf canopy = worst

LEAKAGE-SAFE: phenology is a function of date+location only (deterministic,
cannot encode outcome). Wind is from the storm date only. Per-fold baselines
as in v4.

HARD STOP: must beat v4 (get below ~29% median error) to be considered better.

Run: python src/stormwatch/backtest_ml_v5.py
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
from backtest_ml_v4 import (
    IMPERVIOUS_PCT, normalize_county_name, classify_storm_type,
    compute_baselines_from_subset, load_county_features
)

OUT_DIR = Path("data/stormwatch/backtest")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# County centroids for latitude (phenology depends on latitude)
COUNTY_LAT = {
    ("Cumberland","Maine"): 43.66, ("Penobscot","Maine"): 45.20,
    ("Kennebec","Maine"): 44.40, ("York","Maine"): 43.45,
    ("Androscoggin","Maine"): 44.16,
    ("Hillsborough","New Hampshire"): 42.92, ("Rockingham","New Hampshire"): 42.99,
    ("Chittenden","Vermont"): 44.51,
    ("Middlesex","Massachusetts"): 42.48, ("Worcester","Massachusetts"): 42.36,
    ("Essex","Massachusetts"): 42.64, ("Suffolk","Massachusetts"): 42.36,
    ("Providence","Rhode Island"): 41.87,
    ("Hartford","Connecticut"): 41.80, ("New Haven","Connecticut"): 41.35,
    ("Fairfield","Connecticut"): 41.23,
    ("Suffolk","New York"): 40.92, ("Nassau","New York"): 40.73,
    ("Westchester","New York"): 41.12, ("Erie","New York"): 42.75,
    ("Essex","New Jersey"): 40.79, ("Bergen","New Jersey"): 40.96,
    ("Middlesex","New Jersey"): 40.44, ("Monmouth","New Jersey"): 40.29,
    ("Ocean","New Jersey"): 39.87,
    ("Philadelphia","Pennsylvania"): 40.00, ("Allegheny","Pennsylvania"): 40.47,
    ("Montgomery","Pennsylvania"): 40.21, ("Bucks","Pennsylvania"): 40.34,
    ("Chester","Pennsylvania"): 39.97,
}


def modeled_leaf_state(lat, day_of_year):
    """Model leaf-on fraction (0=bare, 1=full canopy) from latitude + day of year.
    
    Deciduous phenology in the Northeast US:
    - Leaf-out: ~day 110-135 (mid-Apr to mid-May), later at higher latitude
    - Full canopy: ~day 150-270 (late May - late Sep)
    - Leaf-off: ~day 285-315 (mid-Oct to mid-Nov), earlier at higher latitude
    
    This is a deterministic function of date+location. It CANNOT leak outcome
    information because it knows nothing about outages.
    """
    # Latitude shifts the season: higher lat = later spring, earlier fall
    # Reference latitude ~42 (central New England)
    lat_shift = (lat - 42.0) * 2.5  # days of shift per degree
    
    leaf_out_start = 105 + lat_shift   # begins leafing
    leaf_out_full  = 145 + lat_shift   # full canopy reached
    leaf_fall_start = 280 - lat_shift  # begins dropping
    leaf_fall_done  = 315 - lat_shift  # bare
    
    d = day_of_year
    if d < leaf_out_start or d > leaf_fall_done:
        return 0.05  # winter - essentially bare (some conifers)
    elif d < leaf_out_full:
        # spring ramp-up
        return 0.05 + 0.95 * (d - leaf_out_start) / (leaf_out_full - leaf_out_start)
    elif d < leaf_fall_start:
        return 1.0  # full canopy
    else:
        # fall ramp-down
        return 1.0 - 0.95 * (d - leaf_fall_start) / (leaf_fall_done - leaf_fall_start)


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
    outage_types = ["thunderstorm wind","high wind","ice storm","blizzard",
                    "winter storm","heavy snow","tornado","tropical storm",
                    "hurricane","freezing rain","strong wind"]
    pattern = "|".join(outage_types)
    df = df[df["event_type"].astype(str).str.lower().str.contains(pattern, na=False, regex=True)]
    print(f"  {len(df):,} outage-relevant storms")
    return df


def load_weather():
    """Load Open-Meteo wind data if available."""
    path = Path("data/processed/storm_weather_v5.csv")
    if not path.exists():
        print("  NOTE: storm_weather_v5.csv not found - wind will fall back to NOAA magnitude")
        print("  Run fetch_storm_weather.py on your laptop first for granular wind.")
        return {}
    df = pd.read_csv(path)
    wmap = {}
    for _, r in df.iterrows():
        wmap[(r["county"], r["state"], str(r["date_only"]))] = {
            "wind_speed_max": r.get("wind_speed_max"),
            "wind_gust_max":  r.get("wind_gust_max"),
            "wind_dir":       r.get("wind_dir"),
            "precip_sum":     r.get("precip_sum"),
            "rain_sum":       r.get("rain_sum"),
            "snowfall_sum":   r.get("snowfall_sum"),
            "temp_mean":      r.get("temp_mean"),
            "temp_max":       r.get("temp_max"),
            "temp_min":       r.get("temp_min"),
            "soil_moisture_shallow": r.get("soil_moisture_shallow"),
            "soil_moisture_deep":    r.get("soil_moisture_deep"),
        }
    print(f"  Loaded granular weather for {len(wmap):,} county-dates")
    return wmap


def build_dataset(noaa, eagle, county_features, weather_map):
    print("\nBuilding v5 dataset (leaf-on phenology + granular wind)...")
    
    cf_map = {}
    if county_features is not None:
        for _, r in county_features.iterrows():
            cf_map[(r["county"], r["state"])] = r.to_dict()
    
    target_set = set(TARGET_COUNTIES)
    noaa_sorted = noaa.sort_values("event_date").reset_index(drop=True)
    
    noaa_by_county = {}
    for (county, state) in TARGET_COUNTIES:
        sub = noaa_sorted[(noaa_sorted["county"] == county) &
                          (noaa_sorted["state"] == state)].copy()
        noaa_by_county[(county, state)] = sub
    
    rows = []
    n_with_real_wind = 0
    
    for _, storm in noaa_sorted.iterrows():
        county = str(storm.get("county",""))
        state  = str(storm.get("state",""))
        if (county, state) not in target_set:
            continue
        
        storm_date = storm["event_date"]
        window_end = storm_date + timedelta(hours=72)
        actual_rows = eagle[
            (eagle["county"] == county) & (eagle["state"] == state) &
            (eagle["date"] >= storm_date) & (eagle["date"] <= window_end)
        ]
        if len(actual_rows) == 0:
            continue
        actual = float(actual_rows["max_customers_out"].max())
        
        tier = classify_storm_tier(storm)
        storm_type = classify_storm_type(storm.get("event_type",""))
        cf = cf_map.get((county, state), {})
        
        # Lag features
        cs = noaa_by_county[(county, state)]
        prior = cs[cs["event_date"] < storm_date]
        if len(prior) > 0:
            d30  = (prior["event_date"] >= storm_date - timedelta(days=30)).sum()
            d90  = (prior["event_date"] >= storm_date - timedelta(days=90)).sum()
            d365 = (prior["event_date"] >= storm_date - timedelta(days=365)).sum()
            days_since = (storm_date - prior["event_date"].max()).days
        else:
            d30 = d90 = d365 = 0
            days_since = 9999
        
        month = storm_date.month
        doy = storm_date.timetuple().tm_yday
        duration = float(storm.get("storm_duration_hrs",1) or 1)
        
        # NEW v5: modeled leaf state
        lat = COUNTY_LAT.get((county, state), 42.0)
        leaf_on = modeled_leaf_state(lat, doy)
        
        # NEW v5: granular wind + weather (or fall back to NOAA magnitude)
        wkey = (county, state, str(storm_date.date()))
        noaa_mag = float(storm.get("magnitude",0) or 0)
        w = weather_map.get(wkey, {})
        
        def _f(v, default):
            try:
                return float(v) if v is not None else float(default)
            except (TypeError, ValueError):
                return float(default)
        
        if w and w.get("wind_speed_max") is not None:
            wind_speed = _f(w.get("wind_speed_max"), noaa_mag)
            wind_gust  = _f(w.get("wind_gust_max"), wind_speed)
            n_with_real_wind += 1
        else:
            wind_speed = noaa_mag
            wind_gust  = noaa_mag
        
        wind_dir   = _f(w.get("wind_dir"), 0)         # 0-360, low expectations
        precip     = _f(w.get("precip_sum"), 0)
        snowfall   = _f(w.get("snowfall_sum"), 0)
        temp_mean  = _f(w.get("temp_mean"), 50)        # F
        temp_min   = _f(w.get("temp_min"), temp_mean)
        temp_max   = _f(w.get("temp_max"), temp_mean)
        soil_shallow = _f(w.get("soil_moisture_shallow"), 0.30)  # m3/m3, ~0.3 typical
        soil_deep    = _f(w.get("soil_moisture_deep"), 0.30)
        
        canopy = cf.get("tree_canopy_pct", 50)
        density = cf.get("population_density", 500)
        
        # ── Derived ops-driven features ──
        # Ice accretion risk: precip falling in the freezing-rain temperature zone (28-34F)
        # Strong ice risk when there's precip and temp hovers near/just below freezing
        if 28 <= temp_mean <= 34 and precip > 0:
            ice_risk = precip * (1.0 - abs(temp_mean - 31) / 3.0)  # peaks at 31F
            ice_risk = max(ice_risk, 0)
        else:
            ice_risk = 0.0
        
        snow_load = snowfall  # cm of snow - weight on lines/branches
        soil_saturation = (soil_shallow + soil_deep) / 2.0  # higher = wetter = easier uproot
        is_extreme_cold = 1 if temp_min <= 10 else 0   # severe cold (F)
        is_extreme_heat = 1 if temp_max >= 95 else 0   # severe heat (F)
        # Wind direction as cyclical (so model treats 359 and 1 as close)
        wind_dir_sin = np.sin(2 * np.pi * wind_dir / 360.0)
        wind_dir_cos = np.cos(2 * np.pi * wind_dir / 360.0)
        
        rows.append({
            "storm_date": storm_date, "county": county, "state": state,
            "event_type": storm.get("event_type",""),
            "tier_severe": 1 if tier=="SEVERE" else 0,
            "tier_moderate": 1 if tier=="MODERATE" else 0,
            "magnitude": noaa_mag,
            # NEW wind features
            "wind_speed_daily": wind_speed,
            "wind_gust_daily":  wind_gust,
            "storm_duration_hrs": duration, "log_duration": np.log1p(duration),
            "month": month, "month_sin": np.sin(2*np.pi*month/12),
            "month_cos": np.cos(2*np.pi*month/12),
            "is_winter": 1 if month in [12,1,2] else 0,
            "is_summer": 1 if month in [6,7,8] else 0,
            "is_hurricane_season": 1 if month in [8,9,10] else 0,
            # NEW phenology features
            "leaf_on": leaf_on,
            "ndvi_modeled": leaf_on * (canopy/100),  # greenness scaled by canopy density
            # storm type one-hot
            "type_ice": 1 if storm_type=="ice" else 0,
            "type_snow": 1 if storm_type=="snow" else 0,
            "type_winter_storm": 1 if storm_type=="winter_storm" else 0,
            "type_hurricane": 1 if storm_type=="hurricane" else 0,
            "type_tornado": 1 if storm_type=="tornado" else 0,
            "type_thunderstorm": 1 if storm_type=="thunderstorm" else 0,
            "type_wind": 1 if storm_type=="wind" else 0,
            # lag
            "storms_30d_prior": d30, "storms_90d_prior": d90,
            "storms_365d_prior": d365, "days_since_last_storm": min(days_since,9999),
            "log_days_since": np.log1p(min(days_since,9999)),
            # county
            "tree_canopy_pct": canopy, "population_density": density,
            "log_pop_density": np.log1p(density),
            "infrastructure_vulnerability": cf.get("infrastructure_vulnerability",0.5),
            "land_area_sqmi": cf.get("land_area_sqmi",500),
            "log_pop": np.log1p(cf.get("population_2023",100000)),
            "impervious_pct": IMPERVIOUS_PCT.get((county,state),20),
            # interactions (v4 + NEW leaf interactions)
            "tier_x_canopy": (1 if tier=="SEVERE" else 0.5 if tier=="MODERATE" else 0)*canopy/100,
            "tier_x_density": (1 if tier=="SEVERE" else 0.5 if tier=="MODERATE" else 0)*np.log1p(density),
            # NEW: the key ops-driven interaction - wind x leaf state
            "wind_x_leafon": wind_speed * leaf_on,
            "gust_x_leafon": wind_gust * leaf_on,
            "wind_x_canopy": wind_speed * (canopy/100),
            # ── v5.1 ops-driven weather features ──
            "ice_risk": ice_risk,
            "snow_load": snow_load,
            "soil_saturation": soil_saturation,
            "is_extreme_cold": is_extreme_cold,
            "is_extreme_heat": is_extreme_heat,
            "temp_mean": temp_mean,
            "wind_dir_sin": wind_dir_sin,
            "wind_dir_cos": wind_dir_cos,
            # v5.1 interactions (ops physics)
            "wind_x_soil": wind_speed * soil_saturation,   # wet soil + wind = uproot
            "ice_x_canopy": ice_risk * (canopy/100),       # ice load on more branches
            "snow_x_canopy": snow_load * (canopy/100),
            "actual_customers": actual,
        })
    
    print(f"  Built {len(rows):,} pairs ({n_with_real_wind:,} with granular wind, rest used NOAA magnitude)")
    return pd.DataFrame(rows)


def train_cv(df, eagle):
    from sklearn.model_selection import KFold
    try:
        from xgboost import XGBRegressor
        from lightgbm import LGBMRegressor
    except ImportError:
        import subprocess
        subprocess.check_call([sys.executable,"-m","pip","install","xgboost","lightgbm","-q"])
        from xgboost import XGBRegressor
        from lightgbm import LGBMRegressor
    
    base_features = [
        "tier_severe","tier_moderate","magnitude",
        "wind_speed_daily","wind_gust_daily",  # NEW
        "storm_duration_hrs","log_duration",
        "month","month_sin","month_cos","is_winter","is_summer","is_hurricane_season",
        "leaf_on","ndvi_modeled",  # NEW phenology
        "type_ice","type_snow","type_winter_storm","type_hurricane",
        "type_tornado","type_thunderstorm","type_wind",
        "storms_30d_prior","storms_90d_prior","storms_365d_prior",
        "days_since_last_storm","log_days_since",
        "tree_canopy_pct","population_density","log_pop_density",
        "infrastructure_vulnerability","land_area_sqmi","log_pop","impervious_pct",
        "tier_x_canopy","tier_x_density",
        "wind_x_leafon","gust_x_leafon","wind_x_canopy",  # NEW interactions
        # v5.1 ops-driven weather
        "ice_risk","snow_load","soil_saturation",
        "is_extreme_cold","is_extreme_heat","temp_mean",
        "wind_dir_sin","wind_dir_cos",
        "wind_x_soil","ice_x_canopy","snow_x_canopy",
    ]
    baseline_features = ["baseline_typical","baseline_high","baseline_extreme"]
    feature_cols = base_features + baseline_features
    
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    print(f"\nRunning 5-fold CV on {len(df):,} pairs with per-fold baselines...")
    
    all_pred = np.zeros(len(df))
    imp_sum = np.zeros(len(feature_cols))
    fold_results = []
    
    eagle = eagle.copy()
    eagle["date"] = pd.to_datetime(eagle["date"])
    df = df.reset_index(drop=True).copy()
    
    for fi, (tr_idx, te_idx) in enumerate(kf.split(df), 1):
        tr, te = df.iloc[tr_idx].copy(), df.iloc[te_idx].copy()
        tr_dates = pd.to_datetime(tr["storm_date"]).dt.normalize()
        tr_set = set(tr_dates.dt.date)
        eagle_tr = eagle[eagle["date"].dt.date.isin(tr_set) | (eagle["date"] < tr_dates.min())]
        fb = compute_baselines_from_subset(eagle_tr)
        for d in [tr, te]:
            d["baseline_typical"] = d.apply(lambda r: fb.get(f"{r['county']}, {r['state']}",{}).get("typical_major_outage",1500), axis=1)
            d["baseline_high"]    = d.apply(lambda r: fb.get(f"{r['county']}, {r['state']}",{}).get("high_outage",3000), axis=1)
            d["baseline_extreme"] = d.apply(lambda r: fb.get(f"{r['county']}, {r['state']}",{}).get("extreme_outage",8000), axis=1)
        
        Xtr, ytr = tr[feature_cols].values, tr["actual_customers"].values
        Xte, yte = te[feature_cols].values, te["actual_customers"].values
        sw = np.where(Xtr[:,0]==1, 2.0, np.where(Xtr[:,1]==1, 1.5, 1.0))
        
        xgb = XGBRegressor(n_estimators=350,max_depth=9,learning_rate=0.097,
                           subsample=0.88,colsample_bytree=0.75,min_child_weight=2,
                           reg_alpha=0.06,reg_lambda=1.7,random_state=42,n_jobs=-1,verbosity=0)
        xgb.fit(Xtr, np.log1p(ytr), sample_weight=sw)
        pxgb = np.expm1(xgb.predict(Xte))
        
        lgb = LGBMRegressor(n_estimators=300,max_depth=8,learning_rate=0.08,
                            subsample=0.85,colsample_bytree=0.8,random_state=42,n_jobs=-1,verbosity=-1)
        lgb.fit(Xtr, np.log1p(ytr), sample_weight=sw)
        plgb = np.expm1(lgb.predict(Xte))
        
        pred = np.maximum(0.6*pxgb + 0.4*plgb, 200)
        all_pred[te_idx] = pred
        imp_sum += xgb.feature_importances_
        
        macc = ((pred>=1000)==(yte>=1000)).mean()
        merr = np.median(np.where(yte>0, np.abs(pred-yte)/yte*100, 100))
        fold_results.append({"fold":fi,"major_acc_pct":round(macc*100,1),"median_err_pct":round(merr,1)})
        print(f"  Fold {fi}: major_acc={macc*100:.1f}%  median_err={merr:.1f}%")
    
    df["predicted_customers"] = all_pred
    df["major_correct"] = (df["predicted_customers"]>=1000)==(df["actual_customers"]>=1000)
    df["critical_correct"] = (df["predicted_customers"]>=10000)==(df["actual_customers"]>=10000)
    df["pct_error"] = np.where(df["actual_customers"]>0,
                               np.abs(df["predicted_customers"]-df["actual_customers"])/df["actual_customers"]*100,
                               np.where(df["predicted_customers"]==0,0,100))
    def cip(r): return 0.55 if r["tier_severe"] else 0.50 if r["tier_moderate"] else 0.45
    df["ci_pct"]=df.apply(cip,axis=1)
    df["in_ci"]=(df["actual_customers"]>=df["predicted_customers"]*(1-df["ci_pct"]))&(df["actual_customers"]<=df["predicted_customers"]*(1+df["ci_pct"]))
    
    df.to_csv(OUT_DIR/"ml_backtest_v5_results.csv", index=False)
    imp = pd.DataFrame({"feature":feature_cols,"importance":imp_sum/5}).sort_values("importance",ascending=False)
    imp.to_csv(OUT_DIR/"ml_feature_importance_v5.csv", index=False)
    
    sc = {
        "version":"v5: leaf-on phenology + granular wind + ops-driven interactions",
        "total_storms_tested": len(df),
        "major_outage_accuracy_pct": round(df["major_correct"].mean()*100,1),
        "critical_outage_accuracy_pct": round(df["critical_correct"].mean()*100,1),
        "within_ci_pct": round(df["in_ci"].mean()*100,1),
        "median_pct_error": round(df["pct_error"].median(),1),
        "mean_pct_error": round(df["pct_error"].mean(),1),
        "fold_results": fold_results,
        "top_features": imp.head(15).to_dict(orient="records"),
        "generated_at": datetime.utcnow().isoformat()+"Z",
    }
    by_tier={}
    for tn,tc in [("SEVERE","tier_severe"),("MODERATE","tier_moderate")]:
        sub=df[df[tc]==1]
        if len(sub)>0:
            by_tier[tn]={"n":len(sub),"major_accuracy_pct":round(sub["major_correct"].mean()*100,1),
                         "median_pct_error":round(sub["pct_error"].median(),1),
                         "within_ci_pct":round(sub["in_ci"].mean()*100,1)}
    sc["by_tier"]=by_tier
    json.dump(sc, open(OUT_DIR/"ml_backtest_v5_scorecard.json","w"), indent=2, default=str)
    return sc


def main():
    print("="*70)
    print("GridWatch - ML Backtest v5 (EXPERIMENTAL - v4 stays official)")
    print("="*70)
    eagle = load_eaglei()
    noaa  = load_noaa()
    cf = load_county_features()
    if cf is None: return 1
    weather = load_weather()
    
    df = build_dataset(noaa, eagle, cf, weather)
    print(f"\nDataset: {len(df):,} pairs")
    
    sc = train_cv(df, eagle)
    
    print(f"\n{'='*70}")
    print("ML BACKTEST v5 SCORECARD")
    print(f"{'='*70}")
    print(f"Major outage accuracy:     {sc['major_outage_accuracy_pct']}%   (v4: 88.5%)")
    print(f"Critical outage accuracy:  {sc['critical_outage_accuracy_pct']}%   (v4: 90.5%)")
    print(f"Within confidence:         {sc['within_ci_pct']}%   (v4: 63.2%)")
    print(f"Median % error:            {sc['median_pct_error']}%   (v4: 31.8%)  <-- HARD STOP: must be <29%")
    print(f"Mean % error:              {sc['mean_pct_error']}%")
    print(f"\nBy tier:")
    for t,s in sc["by_tier"].items():
        print(f"  {t:9s} n={s['n']:4d}  major={s['major_accuracy_pct']}%  median_err={s['median_pct_error']}%  CI={s['within_ci_pct']}%")
    print(f"\nTop 15 features (watch for leaf_on, wind_x_leafon, wind_speed_daily):")
    for i,r in enumerate(sc["top_features"],1):
        flag = "  <-- NEW v5" if r["feature"] in ["leaf_on","ndvi_modeled","wind_speed_daily","wind_gust_daily","wind_x_leafon","gust_x_leafon","wind_x_canopy","ice_risk","snow_load","soil_saturation","is_extreme_cold","is_extreme_heat","wind_dir_sin","wind_dir_cos","wind_x_soil","ice_x_canopy","snow_x_canopy","temp_mean"] else ""
        print(f"  {i:2d}. {r['feature']:30s} {r['importance']:.4f}{flag}")
    print(f"\nFold consistency:")
    for f in sc["fold_results"]:
        print(f"  Fold {f['fold']}: major={f['major_acc_pct']}%  median_err={f['median_err_pct']}%")
    
    print(f"\n{'='*70}")
    v5_err = sc["median_pct_error"]
    if v5_err < 29:
        print(f"  v5 median error {v5_err}% BEATS the 29% bar. Candidate for promotion - AUDIT FIRST.")
    else:
        print(f"  v5 median error {v5_err}% does NOT beat 29% bar. Keep v4 official.")
    print(f"{'='*70}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
