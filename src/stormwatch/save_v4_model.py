"""
Save the v4 model trained on ALL data for live deployment.
The v4 backtest used 5-fold CV; this trains a final model on the full dataset
using the same architecture, then saves it for predict_outages.py to use.

Run: python src/stormwatch/save_v4_model.py
"""
import pandas as pd
import numpy as np
import pickle
import sys
import warnings
from pathlib import Path
warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).parent))
from backtest_ml_v4 import (
    load_eaglei, load_noaa, load_county_features,
    build_dataset, compute_baselines_from_subset
)

MODELS_DIR = Path("models")
MODELS_DIR.mkdir(exist_ok=True)


def main():
    print("=" * 60)
    print("Training final v4 model on all data for live deployment")
    print("=" * 60)
    
    eagle = load_eaglei()
    noaa  = load_noaa()
    cf = load_county_features()
    if cf is None:
        print("ERROR: county_features.csv missing")
        return 1
    
    df = build_dataset(noaa, eagle, cf)
    print(f"\nDataset: {len(df):,} pairs")
    
    # Use baselines computed from FULL data (for live deployment only,
    # since at inference time we have access to full historical data)
    baselines = compute_baselines_from_subset(eagle)
    
    df = df.copy()
    df["baseline_typical"] = df.apply(
        lambda r: baselines.get(f"{r['county']}, {r['state']}", {}).get("typical_major_outage", 1500),
        axis=1
    )
    df["baseline_high"] = df.apply(
        lambda r: baselines.get(f"{r['county']}, {r['state']}", {}).get("high_outage", 3000),
        axis=1
    )
    df["baseline_extreme"] = df.apply(
        lambda r: baselines.get(f"{r['county']}, {r['state']}", {}).get("extreme_outage", 8000),
        axis=1
    )
    
    feature_cols = [
        "tier_severe", "tier_moderate", "magnitude", "storm_duration_hrs", "log_duration",
        "month", "month_sin", "month_cos",
        "is_winter", "is_summer", "is_hurricane_season",
        "type_ice", "type_snow", "type_winter_storm", "type_hurricane",
        "type_tornado", "type_thunderstorm", "type_wind",
        "storms_30d_prior", "storms_90d_prior", "storms_365d_prior",
        "days_since_last_storm", "log_days_since",
        "tree_canopy_pct", "population_density", "log_pop_density",
        "infrastructure_vulnerability", "land_area_sqmi", "log_pop",
        "impervious_pct",
        "tier_x_canopy", "tier_x_density",
        "baseline_typical", "baseline_high", "baseline_extreme",
    ]
    
    X = df[feature_cols].values
    y = df["actual_customers"].values
    
    sample_weights = np.where(X[:, 0] == 1, 2.0, np.where(X[:, 1] == 1, 1.5, 1.0))
    
    print("\nTraining XGBoost on full data...")
    from xgboost import XGBRegressor
    xgb = XGBRegressor(
        n_estimators=350, max_depth=9, learning_rate=0.097,
        subsample=0.88, colsample_bytree=0.75,
        min_child_weight=2, reg_alpha=0.06, reg_lambda=1.7,
        random_state=42, n_jobs=-1, verbosity=0,
    )
    xgb.fit(X, np.log1p(y), sample_weight=sample_weights)
    
    print("Training LightGBM on full data...")
    from lightgbm import LGBMRegressor
    lgb = LGBMRegressor(
        n_estimators=300, max_depth=8, learning_rate=0.08,
        subsample=0.85, colsample_bytree=0.8,
        random_state=42, n_jobs=-1, verbosity=-1,
    )
    lgb.fit(X, np.log1p(y), sample_weight=sample_weights)
    
    payload = {
        "xgb": xgb,
        "lgb": lgb,
        "feature_cols": feature_cols,
        "baselines": baselines,
        "version": "v4",
        "validation_metrics": {
            "major_outage_accuracy": 88.5,
            "critical_outage_accuracy": 90.5,
            "median_pct_error": 31.8,
            "within_ci_pct": 63.2,
            "validated_on_storms": 3074,
            "validation_method": "5-fold cross-validation",
        },
    }
    
    out_path = MODELS_DIR / "outage_ml_model_v4_final.pkl"
    with open(out_path, "wb") as f:
        pickle.dump(payload, f)
    
    print(f"\nSaved: {out_path}")
    print(f"\nValidation metrics included in payload:")
    for k, v in payload["validation_metrics"].items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
