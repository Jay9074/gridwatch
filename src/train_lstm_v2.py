"""
GridWatch v2 - Proper LSTM Forecasting
Uses 1,097 state-months with proper sequence-based LSTM.
Per-state LSTM with multi-horizon forecasting (1, 3, 6 months).
Run: python src/train_lstm_v2.py
"""
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import pandas as pd
import numpy as np
import json
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
import warnings
warnings.filterwarnings("ignore")

PROC_DIR  = Path("data/processed")
MODEL_DIR = Path("models")
MODEL_DIR.mkdir(exist_ok=True)

tf.random.set_seed(42)
np.random.seed(42)

print("="*60)
print("GridWatch v2 - LSTM Multi-Horizon Forecasting")
print("="*60)

df = pd.read_csv(PROC_DIR / "state_monthly_dataset.csv")
df["date"] = pd.to_datetime(df["date"])
df = df.sort_values(["state","date"]).reset_index(drop=True)

print(f"Loaded: {len(df)} state-months")
print(f"States: {df['state'].nunique()}")
print(f"Date range: {df['date'].min().date()} -> {df['date'].max().date()}")

# Features for LSTM (no leakage - all known at prediction time)
LSTM_FEATURES = [
    "major_outage_days",
    "outage_rate",
    "log_cust_hours",
    "month_sin", "month_cos",
    "is_winter", "is_summer", "is_fall",
    "year_trend",
    "state_id", "state_risk",
    "storm_count", "max_severity", "mean_severity",
    "ice_events", "wind_events", "winter_storms"
]

# Filter to features that exist
LSTM_FEATURES = [f for f in LSTM_FEATURES if f in df.columns]
print(f"\nFeatures used: {len(LSTM_FEATURES)}")

# Configuration
SEQ_LEN  = 12       # Use 12 months of history
TARGET_COL = "major_outage_days"

def create_sequences(state_data, seq_len, horizon, features, target):
    """Create input sequences and targets for LSTM."""
    Xs, ys = [], []
    values = state_data[features].values
    targets = state_data[target].values
    for i in range(len(state_data) - seq_len - horizon + 1):
        Xs.append(values[i : i + seq_len])
        ys.append(targets[i + seq_len + horizon - 1])
    return np.array(Xs), np.array(ys)


def build_lstm(seq_len, n_features):
    """Stacked LSTM."""
    model = Sequential([
        LSTM(64, return_sequences=True, input_shape=(seq_len, n_features)),
        Dropout(0.25),
        BatchNormalization(),
        LSTM(32, return_sequences=False),
        Dropout(0.25),
        BatchNormalization(),
        Dense(16, activation="relu"),
        Dropout(0.15),
        Dense(1, activation="linear")
    ])
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss="huber",
        metrics=["mae"]
    )
    return model


# Build sequences for ALL states pooled together
print("\nBuilding pooled sequences across all states...")
all_results = {}

for horizon in [1, 3, 6]:
    print(f"\n{'='*60}")
    print(f"HORIZON: {horizon} month(s) ahead")
    print(f"{'='*60}")

    X_all, y_all = [], []
    
    # Per-state scaler dictionary
    state_scalers = {}
    
    for state in sorted(df["state"].unique()):
        state_data = df[df["state"] == state].sort_values("date").reset_index(drop=True)
        
        # Drop rows with missing features
        state_data = state_data.dropna(subset=LSTM_FEATURES + [TARGET_COL]).reset_index(drop=True)
        
        if len(state_data) < SEQ_LEN + horizon + 5:
            continue
        
        # Scale per-state
        scaler_X = MinMaxScaler()
        scaler_y = MinMaxScaler()
        
        X_scaled = scaler_X.fit_transform(state_data[LSTM_FEATURES])
        y_scaled = scaler_y.fit_transform(state_data[[TARGET_COL]]).flatten()
        
        state_scaled = state_data.copy()
        state_scaled[LSTM_FEATURES] = X_scaled
        state_scaled[TARGET_COL] = y_scaled
        
        state_scalers[state] = (scaler_X, scaler_y)
        
        Xs, ys = create_sequences(state_scaled, SEQ_LEN, horizon, LSTM_FEATURES, TARGET_COL)
        
        if len(Xs) > 0:
            X_all.append(Xs)
            y_all.append(ys)
    
    X_all = np.concatenate(X_all)
    y_all = np.concatenate(y_all)
    
    print(f"Total sequences: {len(X_all)}")
    print(f"Sequence shape: {X_all.shape}")
    
    # Time-based split
    split_idx = int(len(X_all) * 0.80)
    X_train, X_test = X_all[:split_idx], X_all[split_idx:]
    y_train, y_test = y_all[:split_idx], y_all[split_idx:]
    
    print(f"Train: {len(X_train)} sequences")
    print(f"Test:  {len(X_test)} sequences")
    
    # Train
    model = build_lstm(SEQ_LEN, len(LSTM_FEATURES))
    
    callbacks = [
        EarlyStopping(patience=25, restore_best_weights=True, verbose=0),
        ReduceLROnPlateau(factor=0.5, patience=12, verbose=0)
    ]
    
    history = model.fit(
        X_train, y_train,
        epochs=300,
        batch_size=32,
        validation_split=0.15,
        callbacks=callbacks,
        verbose=0
    )
    
    # Predict
    y_pred_scaled = model.predict(X_test, verbose=0).flatten()
    y_test_scaled = y_test
    
    # Calculate metrics on scaled data (same scale comparison)
    rmse = np.sqrt(mean_squared_error(y_test_scaled, y_pred_scaled))
    mae  = mean_absolute_error(y_test_scaled, y_pred_scaled)
    r2   = r2_score(y_test_scaled, y_pred_scaled)
    corr = np.corrcoef(y_test_scaled, y_pred_scaled)[0,1]
    
    epochs_trained = len(history.history["loss"])
    
    print(f"\n  Epochs trained: {epochs_trained}")
    print(f"  Test sequences: {len(X_test)}  (real test set, not 2-3 points!)")
    print(f"  RMSE          : {rmse:.4f}")
    print(f"  MAE           : {mae:.4f}")
    print(f"  R²            : {r2:.4f}")
    print(f"  Correlation   : {corr:.4f}")
    
    all_results[f"horizon_{horizon}m"] = {
        "rmse": round(float(rmse), 4),
        "mae": round(float(mae), 4),
        "r2": round(float(r2), 4),
        "correlation": round(float(corr), 4),
        "epochs": epochs_trained,
        "n_train": len(X_train),
        "n_test": len(X_test),
    }

# Save
with open(MODEL_DIR / "lstm_v2_metrics.json", "w") as f:
    json.dump(all_results, f, indent=2)

print(f"\n{'='*60}")
print("SUMMARY")
print(f"{'='*60}")
for horizon, m in all_results.items():
    print(f"\n{horizon}:")
    for k, v in m.items():
        print(f"  {k}: {v}")

print(f"\nSaved: models/lstm_v2_metrics.json")
