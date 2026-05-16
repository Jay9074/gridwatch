"""
GridWatch v2 - Random Forest Hyperparameter Tuning
Systematic grid search to push R² from 0.84 to higher.
Run: python src/tune_random_forest.py
"""
import pandas as pd
import numpy as np
import json
import pickle
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import warnings
warnings.filterwarnings("ignore")

PROC_DIR  = Path("data/processed")
MODEL_DIR = Path("models")

print("="*60)
print("GridWatch v2 - Random Forest Hyperparameter Tuning")
print("="*60)

df = pd.read_csv(PROC_DIR / "state_monthly_dataset.csv")
df["date"] = pd.to_datetime(df["date"])
df = df.sort_values(["state","date"]).reset_index(drop=True)

TARGET = "major_outage_days"
EXCLUDE = [
    TARGET, "outage_rate", "log_outage_days", "log_cust_hours",
    "total_customer_hours", "max_customers_out_peak", "avg_customers_out",
    "critical_outage_days", "n_counties", "n_county_days",
    "state", "date", "year", "month"
]
FEATURES = [c for c in df.columns if c not in EXCLUDE]

df_clean = df.dropna(subset=FEATURES).reset_index(drop=True)
print(f"Loaded: {len(df_clean)} state-months, {len(FEATURES)} features")

X = df_clean[FEATURES]
y = df_clean[TARGET]

# Time-based split
split_date = df_clean["date"].quantile(0.80)
train_mask = df_clean["date"] <= split_date
test_mask  = df_clean["date"] >  split_date

X_train, X_test = X[train_mask], X[test_mask]
y_train, y_test = y[train_mask], y[test_mask]

print(f"Train: {len(X_train)} | Test: {len(X_test)}")

# Hyperparameter grid (focused on what matters most)
param_grid = {
    "n_estimators":      [200, 400, 600],
    "max_depth":         [8, 12, 16, None],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf":  [1, 2, 4],
    "max_features":      ["sqrt", 0.5, 0.7],
}

# Time series CV
tscv = TimeSeriesSplit(n_splits=5)

print(f"\nGrid search: {3*4*3*3*3} = 324 combinations × 5 folds = 1,620 fits")
print("This will take 15-25 minutes...")
print()

grid_search = GridSearchCV(
    RandomForestRegressor(random_state=42, n_jobs=-1),
    param_grid=param_grid,
    cv=tscv,
    scoring="r2",
    n_jobs=-1,
    verbose=2
)
grid_search.fit(X_train, y_train)

print(f"\n{'='*60}")
print("BEST HYPERPARAMETERS")
print(f"{'='*60}")
for k, v in grid_search.best_params_.items():
    print(f"  {k}: {v}")
print(f"\n  Best CV R²: {grid_search.best_score_:.4f}")

# Evaluate on test set
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)

rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mae  = mean_absolute_error(y_test, y_pred)
r2   = r2_score(y_test, y_pred)
corr = np.corrcoef(y_test, y_pred)[0,1]

# Compare to baseline
print(f"\n{'='*60}")
print("TEST SET PERFORMANCE COMPARISON")
print(f"{'='*60}")
print(f"\nBaseline (default RF):")
print(f"  R²: 0.8371 | RMSE: 27.13 | MAE: 16.72")
print(f"\nTuned RF:")
print(f"  R²:    {r2:.4f}")
print(f"  RMSE:  {rmse:.2f}")
print(f"  MAE:   {mae:.2f}")
print(f"  Corr:  {corr:.4f}")

improvement = r2 - 0.8371
print(f"\nR² improvement: {improvement:+.4f}")

if improvement > 0:
    print("Tuned model is BETTER")
else:
    print("Tuned model is NOT better - keeping baseline")

# Save best model
bundle = {
    "model": best_model,
    "model_name": "Random Forest (Tuned)",
    "features": FEATURES,
    "target": TARGET,
    "best_params": grid_search.best_params_,
    "metrics": {
        "rmse": round(float(rmse), 4),
        "mae": round(float(mae), 4),
        "r2": round(float(r2), 4),
        "correlation": round(float(corr), 4),
        "best_cv_r2": round(float(grid_search.best_score_), 4),
    }
}
with open(MODEL_DIR / "monthly_outage_model_tuned.pkl", "wb") as f:
    pickle.dump(bundle, f)

with open(MODEL_DIR / "tuning_results.json", "w") as f:
    json.dump({
        "best_params": grid_search.best_params_,
        "metrics": bundle["metrics"]
    }, f, indent=2)

# Top features
fi = pd.DataFrame({
    "feature": FEATURES,
    "importance": best_model.feature_importances_
}).sort_values("importance", ascending=False)
print(f"\nTop 10 features (tuned model):")
print(fi.head(10).to_string(index=False))

print(f"\nSaved: models/monthly_outage_model_tuned.pkl")
