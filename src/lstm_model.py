"""
GridWatch — src/lstm_model.py
================================
Deep learning time-series forecasting for power outage risk.
Uses LSTM (Long Short-Term Memory) neural network to predict
outage risk 30 / 60 / 90 days in advance.

Why LSTM?
- Power outages have temporal dependencies (yesterday's storm
  affects today's grid stress)
- Traditional ML treats each event independently
- LSTM remembers sequences — critical for forecasting
"""

import logging
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")

from pathlib import Path
from datetime import datetime, timedelta

import tensorflow as tf
from tensorflow.keras.models    import Sequential, load_model
from tensorflow.keras.layers    import LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing      import MinMaxScaler
from sklearn.metrics            import mean_squared_error, mean_absolute_error

log = logging.getLogger(__name__)

BASE_DIR  = Path(__file__).parent.parent
PROC_DIR  = BASE_DIR / "data" / "processed"
MODEL_DIR = BASE_DIR / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)


# ── Build time-series dataset ────────────────────────────────────
def build_time_series(doe_path: Path = None) -> pd.DataFrame:
    """
    Aggregates outage data to a monthly time series per state.
    This is the input format LSTM needs — sequences over time.
    """
    if doe_path is None:
        doe_path = PROC_DIR / "doe_outages_northeast.csv"

    if doe_path.exists():
        df = pd.read_csv(doe_path, parse_dates=["event_date"])
        log.info(f"Loaded real data: {len(df):,} records")
    else:
        log.info("Generating synthetic time series for development...")
        df = _synthetic_time_series()

    # Aggregate to monthly state-level
    df["year_month"] = df["event_date"].dt.to_period("M")

    agg = df.groupby("year_month").agg(
        total_events       = ("event_date",        "count"),
        total_customers    = ("customers_affected", "sum"),
        max_customers      = ("customers_affected", "max"),
        total_demand_loss  = ("demand_loss_mw",     "sum"),
        weather_events_pct = ("is_weather_caused",  "mean"),
    ).reset_index()

    agg["year_month_dt"] = agg["year_month"].dt.to_timestamp()
    agg = agg.sort_values("year_month_dt").reset_index(drop=True)

    # Add lag features
    for lag in [1, 2, 3, 6, 12]:
        agg[f"events_lag_{lag}"]     = agg["total_events"].shift(lag)
        agg[f"customers_lag_{lag}"]  = agg["total_customers"].shift(lag)

    # Rolling statistics
    agg["events_rolling_3m"]  = agg["total_events"].rolling(3,  min_periods=1).mean()
    agg["events_rolling_12m"] = agg["total_events"].rolling(12, min_periods=1).mean()
    agg["events_trend"]       = agg["total_events"] - agg["events_rolling_12m"]

    # Month seasonality
    agg["month"]     = agg["year_month_dt"].dt.month
    agg["month_sin"] = np.sin(2 * np.pi * agg["month"] / 12)
    agg["month_cos"] = np.cos(2 * np.pi * agg["month"] / 12)

    agg = agg.dropna()
    log.info(f"Time series built: {len(agg)} months")
    return agg


def _synthetic_time_series(n_months: int = 120) -> pd.DataFrame:
    """Synthetic monthly outage data with realistic seasonal patterns."""
    rng   = np.random.default_rng(42)
    dates = pd.date_range("2015-01-01", periods=n_months, freq="MS")

    # Seasonal pattern — winter peaks
    month_risk = {1:3.2, 2:3.0, 3:2.5, 4:1.5, 5:1.2, 6:1.8,
                  7:2.0, 8:2.2, 9:1.8, 10:2.0, 11:2.5, 12:3.3}
    base = np.array([month_risk[d.month] for d in dates])
    trend = np.linspace(1.0, 1.4, n_months)  # worsening over time (real pattern)
    noise = rng.normal(0, 0.4, n_months)

    customers = (base * trend * 15_000 + rng.exponential(5_000, n_months)).clip(0)

    return pd.DataFrame({
        "event_date":         dates.repeat(
            np.maximum(1, (base * trend + noise).round().astype(int))
        )[:n_months * 3],
        "customers_affected": rng.exponential(20_000, n_months * 3),
        "demand_loss_mw":     rng.exponential(100,    n_months * 3),
        "is_weather_caused":  rng.binomial(1, 0.58,   n_months * 3),
    })


# ── Sequence builder ─────────────────────────────────────────────
def create_sequences(data: np.ndarray, seq_len: int = 12,
                     horizon: int = 1) -> tuple:
    """
    Converts time series to (X, y) sequences for LSTM.

    seq_len = 12 means: use 12 months of history to predict
    horizon  = 1  means: predict 1 month ahead
                 = 3  means: predict 3 months ahead

    X shape: (samples, seq_len, features)
    y shape: (samples,)
    """
    X, y = [], []
    for i in range(len(data) - seq_len - horizon + 1):
        X.append(data[i : i + seq_len])
        y.append(data[i + seq_len + horizon - 1, 0])  # col 0 = total_events
    return np.array(X), np.array(y)


# ── Build LSTM model ─────────────────────────────────────────────
def build_lstm(seq_len: int, n_features: int,
               units: int = 64) -> tf.keras.Model:
    """
    Builds a stacked LSTM architecture.

    Architecture:
      LSTM(64) → Dropout(0.2) → BatchNorm →
      LSTM(32) → Dropout(0.2) → BatchNorm →
      Dense(16, relu) → Dense(1)
    """
    model = Sequential([
        LSTM(units, return_sequences=True,
             input_shape=(seq_len, n_features),
             name="lstm_1"),
        Dropout(0.2),
        BatchNormalization(),

        LSTM(units // 2, return_sequences=False, name="lstm_2"),
        Dropout(0.2),
        BatchNormalization(),

        Dense(16, activation="relu", name="dense_hidden"),
        Dense(1,  activation="linear", name="output")
    ])

    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss="huber",         # robust to outliers vs MSE
        metrics=["mae"]
    )
    return model


# ── Train ────────────────────────────────────────────────────────
def train_lstm(ts_df: pd.DataFrame,
               seq_len:  int = 12,
               horizon:  int = 1,
               epochs:   int = 100) -> dict:
    """
    Full LSTM training pipeline.
    Trains models for 1-month, 3-month, and 6-month horizons.
    """
    feature_cols = [
        "total_events", "total_customers", "max_customers",
        "total_demand_loss", "weather_events_pct",
        "events_rolling_3m", "events_rolling_12m", "events_trend",
        "month_sin", "month_cos"
    ]
    feature_cols = [c for c in feature_cols if c in ts_df.columns]

    data_raw = ts_df[feature_cols].values.astype(float)
    scaler   = MinMaxScaler()
    data_sc  = scaler.fit_transform(data_raw)

    results = {}

    for h in [1, 3, 6]:   # 1-month, 3-month, 6-month forecast
        log.info(f"\nTraining LSTM — {h}-month horizon...")

        X, y = create_sequences(data_sc, seq_len=seq_len, horizon=h)
        split = int(len(X) * 0.80)
        X_tr, X_te = X[:split], X[split:]
        y_tr, y_te = y[:split], y[split:]

        model = build_lstm(seq_len, len(feature_cols))

        callbacks = [
            EarlyStopping(patience=15, restore_best_weights=True, verbose=0),
            ReduceLROnPlateau(factor=0.5, patience=8, verbose=0),
            ModelCheckpoint(
                str(MODEL_DIR / f"lstm_{h}mo.keras"),
                save_best_only=True, verbose=0
            )
        ]

        history = model.fit(
            X_tr, y_tr,
            epochs=epochs, batch_size=16,
            validation_split=0.15,
            callbacks=callbacks, verbose=0
        )

        y_pred = model.predict(X_te, verbose=0).flatten()
        rmse = np.sqrt(mean_squared_error(y_te, y_pred))
        mae  = mean_absolute_error(y_te, y_pred)

        # Inverse transform for real-scale metrics
        dummy     = np.zeros((len(y_te), len(feature_cols)))
        dummy[:,0] = y_te
        y_te_real  = scaler.inverse_transform(dummy)[:,0]
        dummy[:,0] = y_pred
        y_pr_real  = scaler.inverse_transform(dummy)[:,0]

        rmse_real = np.sqrt(mean_squared_error(y_te_real, y_pr_real))
        mae_real  = mean_absolute_error(y_te_real, y_pr_real)

        results[f"{h}mo"] = {
            "model": model, "history": history.history,
            "y_test": y_te_real, "y_pred": y_pr_real,
            "rmse": round(rmse_real, 2), "mae": round(mae_real, 2),
            "epochs_trained": len(history.history["loss"])
        }

        log.info(f"  Epochs trained : {len(history.history['loss'])}")
        log.info(f"  RMSE (events)  : {rmse_real:.2f}")
        log.info(f"  MAE  (events)  : {mae_real:.2f}")

    # Save scaler
    import joblib
    joblib.dump(scaler, MODEL_DIR / "lstm_scaler.pkl")
    results["scaler"]       = scaler
    results["feature_cols"] = feature_cols
    return results


# ── Forecast Plot ────────────────────────────────────────────────
def plot_forecast(results: dict, ts_df: pd.DataFrame):
    """Plots actual vs predicted for all three horizons."""
    horizons = [k for k in results if k.endswith("mo")]
    fig, axes = plt.subplots(len(horizons), 1,
                             figsize=(12, 4 * len(horizons)),
                             sharex=False)

    if len(horizons) == 1:
        axes = [axes]

    colors = {"1mo": "#e63946", "3mo": "#457b9d", "6mo": "#2a9d8f"}

    for ax, h in zip(axes, horizons):
        r = results[h]
        n = len(r["y_test"])
        x = range(n)
        ax.plot(x, r["y_test"], label="Actual",    color="#333", lw=2)
        ax.plot(x, r["y_pred"], label="Predicted", color=colors[h],
                lw=2, linestyle="--")
        ax.fill_between(x,
                        np.array(r["y_pred"]) * 0.85,
                        np.array(r["y_pred"]) * 1.15,
                        alpha=0.15, color=colors[h], label="±15% band")
        ax.set_title(f"LSTM {h.replace('mo','-month')} Ahead Forecast  "
                     f"|  RMSE={r['rmse']:.1f}  MAE={r['mae']:.1f}",
                     fontsize=12, fontweight="bold")
        ax.set_ylabel("Monthly Outage Events")
        ax.legend(fontsize=9); ax.grid(alpha=0.3)
        ax.spines[["top","right"]].set_visible(False)

    plt.suptitle("GridWatch — LSTM Power Outage Forecasting\nNortheast US",
                 fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()
    plt.savefig(MODEL_DIR / "lstm_forecast.png", dpi=150, bbox_inches="tight")
    plt.close()
    log.info("Forecast plot saved → models/lstm_forecast.png")


# ── Training loss plot ───────────────────────────────────────────
def plot_training_history(results: dict):
    """Plots training vs validation loss curves."""
    horizons = [k for k in results if k.endswith("mo")]
    fig, axes = plt.subplots(1, len(horizons),
                              figsize=(5 * len(horizons), 4))
    if len(horizons) == 1:
        axes = [axes]

    for ax, h in zip(axes, horizons):
        hist = results[h]["history"]
        epochs = range(1, len(hist["loss"]) + 1)
        ax.plot(epochs, hist["loss"],     label="Train loss", color="#457b9d")
        ax.plot(epochs, hist["val_loss"], label="Val loss",   color="#e63946")
        ax.set_title(f"{h} Horizon", fontweight="bold")
        ax.set_xlabel("Epoch"); ax.set_ylabel("Huber Loss")
        ax.legend(fontsize=9); ax.grid(alpha=0.3)
        ax.spines[["top","right"]].set_visible(False)

    plt.suptitle("LSTM Training History", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(MODEL_DIR / "lstm_training_history.png", dpi=150, bbox_inches="tight")
    plt.close()
    log.info("Training history saved → models/lstm_training_history.png")


# ── Full pipeline ────────────────────────────────────────────────
def run_pipeline():
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s")

    log.info("=" * 55)
    log.info("GridWatch — LSTM Deep Learning Pipeline")
    log.info("=" * 55)

    ts_df   = build_time_series()
    results = train_lstm(ts_df, seq_len=12, epochs=100)

    plot_forecast(results, ts_df)
    plot_training_history(results)

    log.info("\n✅ LSTM training complete!")
    for h in ["1mo", "3mo", "6mo"]:
        if h in results:
            log.info(f"  {h}: RMSE={results[h]['rmse']}  MAE={results[h]['mae']}")


if __name__ == "__main__":
    run_pipeline()
