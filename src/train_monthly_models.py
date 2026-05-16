"""
GridWatch v2 - Train Monthly Regression Models
Predicts: major_outage_days per state per month
Run: python src/train_monthly_models.py
"""
import pandas as pd
import numpy as np
import json
import pickle
from pathlib import Path
from sklearn.model_selection import train_test_split, TimeSeriesSplit, cross_val_score
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score
)
import warnings
warnings.filterwarnings('ignore')

PROC_DIR  = Path("data/processed")
MODEL_DIR = Path("models")
MODEL_DIR.mkdir(exist_ok=True)

print("="*60)
print("GridWatch v2 - Monthly Regression Training")
print("="*60)

df = pd.read_csv(PROC_DIR / "state_monthly_dataset.csv")
df["date"] = pd.to_datetime(df["date"])
df = df.sort_values(["state","date"]).reset_index(drop=True)
print(f"Loaded: {len(df)} state-months")

TARGET = "major_outage_days"

EXCLUDE = [
    TARGET, "outage_rate", "log_outage_days", "log_cust_hours",
    "total_customer_hours", "max_customers_out_peak", "avg_customers_out",
    "critical_outage_days", "n_counties", "n_county_days",
    "state", "date", "year", "month"
]
FEATURES = [c for c in df.columns if c not in EXCLUDE]
print(f"\nUsing {len(FEATURES)} features")
print(f"Target: {TARGET}")
print(f"Target range: {df[TARGET].min():.0f} - {df[TARGET].max():.0f}")
print(f"Target mean: {df[TARGET].mean():.1f}")

# Drop rows with any NaN in features (from lag features at start)
df_clean = df.dropna(subset=FEATURES).reset_index(drop=True)
print(f"\nAfter dropping rows with missing lag features: {len(df_clean)} state-months")

X = df_clean[FEATURES]
y = df_clean[TARGET]

# Time-based train/test split (NOT random - critical for time series)
print("\n" + "="*60)
print("TIME-BASED SPLIT (last 20% of months for testing)")
print("="*60)

split_date = df_clean["date"].quantile(0.80)
train_mask = df_clean["date"] <= split_date
test_mask  = df_clean["date"] >  split_date

X_train = X[train_mask]
y_train = y[train_mask]
X_test  = X[test_mask]
y_test  = y[test_mask]

print(f"Train: {len(X_train)} samples ({df_clean[train_mask]['date'].min().date()} -> {df_clean[train_mask]['date'].max().date()})")
print(f"Test:  {len(X_test)} samples ({df_clean[test_mask]['date'].min().date()} -> {df_clean[test_mask]['date'].max().date()})")

# Train multiple models
models = {
    "Ridge Regression": Ridge(alpha=1.0, random_state=42),
    "Random Forest": RandomForestRegressor(
        n_estimators=200, max_depth=10, min_samples_split=5,
        n_jobs=-1, random_state=42
    ),
    "Gradient Boosting": GradientBoostingRegressor(
        n_estimators=200, max_depth=5, learning_rate=0.05,
        random_state=42
    ),
}

# Try XGBoost if available
try:
    from xgboost import XGBRegressor
    models["XGBoost"] = XGBRegressor(
        n_estimators=200, max_depth=5, learning_rate=0.05,
        random_state=42, n_jobs=-1
    )
except ImportError:
    print("XGBoost not available — skipping")

# Try LightGBM if available
try:
    from lightgbm import LGBMRegressor
    models["LightGBM"] = LGBMRegressor(
        n_estimators=200, max_depth=5, learning_rate=0.05,
        random_state=42, n_jobs=-1, verbose=-1
    )
except ImportError:
    print("LightGBM not available — skipping (run: pip install lightgbm)")

results = {}

for name, model in models.items():
    print(f"\n{'='*60}")
    print(f"Training {name}...")
    print(f"{'='*60}")
    
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae  = mean_absolute_error(y_test, y_pred)
    r2   = r2_score(y_test, y_pred)
    corr = np.corrcoef(y_test, y_pred)[0,1]
    
    # Time series cross-validation
    tscv = TimeSeriesSplit(n_splits=5)
    cv_scores = cross_val_score(model, X_train, y_train,
                                  cv=tscv, scoring="r2", n_jobs=-1)
    
    print(f"  RMSE         : {rmse:.2f}")
    print(f"  MAE          : {mae:.2f}")
    print(f"  R²           : {r2:.4f}")
    print(f"  Correlation  : {corr:.4f}")
    print(f"  CV R² (5-fold time series): {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
    
    results[name] = {
        "rmse": round(float(rmse), 4),
        "mae": round(float(mae), 4),
        "r2": round(float(r2), 4),
        "correlation": round(float(corr), 4),
        "cv_r2_mean": round(float(cv_scores.mean()), 4),
        "cv_r2_std": round(float(cv_scores.std()), 4),
    }

# Find best
best_name = max(results, key=lambda k: results[k]["r2"])
print(f"\n{'='*60}")
print(f"BEST MODEL: {best_name} (R² = {results[best_name]['r2']:.4f})")
print(f"{'='*60}")

# Save best model
best_model = models[best_name]
bundle = {
    "model": best_model,
    "model_name": best_name,
    "features": FEATURES,
    "target": TARGET,
    "metrics": results[best_name],
}
with open(MODEL_DIR / "monthly_outage_model.pkl", "wb") as f:
    pickle.dump(bundle, f)

# Save all metrics
all_metrics = {**results, "best_model": best_name}
with open(MODEL_DIR / "monthly_model_metrics.json", "w") as f:
    json.dump(all_metrics, f, indent=2)

# Save feature importances if available
if hasattr(best_model, "feature_importances_"):
    fi = pd.DataFrame({
        "feature": FEATURES,
        "importance": best_model.feature_importances_
    }).sort_values("importance", ascending=False)
    fi.to_csv(MODEL_DIR / "monthly_feature_importance.csv", index=False)
    print(f"\nTop 10 most important features:")
    print(fi.head(10).to_string(index=False))

# Save predictions for plotting
preds_df = df_clean[test_mask].copy()
preds_df["predicted"] = best_model.predict(X_test)
preds_df["actual"] = y_test.values
preds_df[["state","date","year","month","actual","predicted"]].to_csv(
    PROC_DIR / "monthly_predictions.csv", index=False
)

print(f"\nSaved:")
print(f"  models/monthly_outage_model.pkl")
print(f"  models/monthly_model_metrics.json")
print(f"  models/monthly_feature_importance.csv")
print(f"  data/processed/monthly_predictions.csv")
