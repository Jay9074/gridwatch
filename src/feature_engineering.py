"""
GridWatch — src/feature_engineering.py
========================================
Transforms raw outage + weather data into ML-ready features.
This is where Data Science expertise shows — thoughtful feature
creation from domain knowledge is more valuable than raw model tuning.

"""

import numpy as np
import pandas as pd
import logging
from pathlib import Path

log = logging.getLogger(__name__)

BASE_DIR = Path(__file__).parent.parent
PROC_DIR = BASE_DIR / "data" / "processed"


def load_processed_data() -> tuple:
    """Loads cleaned datasets from disk."""
    doe_path  = PROC_DIR / "doe_outages_northeast.csv"
    noaa_path = PROC_DIR / "noaa_storms_northeast.csv"

    doe  = pd.read_csv(doe_path,  parse_dates=["event_date"]) if doe_path.exists()  else pd.DataFrame()
    noaa = pd.read_csv(noaa_path) if noaa_path.exists() else pd.DataFrame()

    if doe.empty and noaa.empty:
        log.warning("No processed data found. Run data_ingestion.py first.")
    return doe, noaa


def create_outage_features(doe: pd.DataFrame) -> pd.DataFrame:
    """
    Builds outage-level features from DOE data.

    Feature categories:
    1. Time-based      — month, season, weekday, holiday proximity
    2. Historical      — rolling outage counts, EWMA trends
    3. Severity        — customer impact tiers, demand loss buckets
    4. Geographic      — state-level risk encoding
    5. Event type      — weather vs equipment vs other
    """
    if doe.empty:
        log.warning("DOE data empty — returning empty feature set")
        return pd.DataFrame()

    df = doe.copy()

    # ── 1. Time features ─────────────────────────────────────────
    if "event_date" in df.columns:
        df["year"]        = df["event_date"].dt.year
        df["month"]       = df["event_date"].dt.month
        df["day_of_week"] = df["event_date"].dt.dayofweek   # 0=Mon
        df["quarter"]     = df["event_date"].dt.quarter
        df["is_weekend"]  = (df["day_of_week"] >= 5).astype(int)

        # Sine/cosine encoding of month — captures circular nature of seasons
        df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
        df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)

        # Winter risk flag (Dec, Jan, Feb, Mar = high risk for Northeast)
        df["is_high_risk_month"] = df["month"].isin([12, 1, 2, 3]).astype(int)

    # ── 2. Season encoding ────────────────────────────────────────
    if "season" in df.columns:
        season_risk = {"Winter": 3, "Fall": 2, "Spring": 1, "Summer": 2}
        df["season_risk_score"] = df["season"].map(season_risk).fillna(1)
        # One-hot encode seasons
        season_dummies = pd.get_dummies(df["season"], prefix="season")
        df = pd.concat([df, season_dummies], axis=1)

    # ── 3. Severity tiers ─────────────────────────────────────────
    if "customers_affected" in df.columns:
        df["customers_affected"] = pd.to_numeric(df["customers_affected"], errors="coerce")
        df["is_major_outage"]    = (df["customers_affected"] >= 50_000).astype(int)
        df["is_critical_outage"] = (df["customers_affected"] >= 200_000).astype(int)
        df["log_customers"]      = np.log1p(df["customers_affected"].fillna(0))

        df["severity_tier"] = pd.cut(
            df["customers_affected"].fillna(0),
            bins=[0, 10_000, 50_000, 200_000, np.inf],
            labels=["Minor", "Moderate", "Major", "Critical"]
        )

    if "demand_loss_mw" in df.columns:
        df["demand_loss_mw"] = pd.to_numeric(df["demand_loss_mw"], errors="coerce")
        df["log_demand_loss"] = np.log1p(df["demand_loss_mw"].fillna(0))
        df["is_high_mw_loss"] = (df["demand_loss_mw"] >= 300).astype(int)

    # ── 4. Event type encoding ────────────────────────────────────
    if "event_type" in df.columns:
        df["event_type_lower"] = df["event_type"].astype(str).str.lower()
        df["is_weather_caused"] = df["event_type_lower"].str.contains(
            "weather|storm|wind|snow|ice|flood|hurricane|tornado|lightning",
            na=False
        ).astype(int)
        df["is_equipment_failure"] = df["event_type_lower"].str.contains(
            "equipment|failure|fault|fire|physical", na=False
        ).astype(int)
        df["is_cyber"] = df["event_type_lower"].str.contains(
            "cyber|attack|vandal|sabotage", na=False
        ).astype(int)

    # ── 5. State risk encoding ────────────────────────────────────
    if "area" in df.columns:
        # Derived from historical outage frequency in NERC data
        state_risk_map = {
            "Maine": 0.82, "Vermont": 0.78, "New Hampshire": 0.75,
            "New York": 0.72, "Massachusetts": 0.65, "Connecticut": 0.61,
            "Rhode Island": 0.58, "New Jersey": 0.60, "Pennsylvania": 0.68
        }
        df["state_risk_score"] = 0.65  # default
        for state, score in state_risk_map.items():
            mask = df["area"].astype(str).str.contains(state, case=False, na=False)
            df.loc[mask, "state_risk_score"] = score

    # ── 6. Rolling historical features ───────────────────────────
    # Sort by date for rolling calculations
    if "event_date" in df.columns:
        df = df.sort_values("event_date")

        # 12-month rolling outage count (proxy for chronic vulnerability)
        df["rolling_12mo_events"] = (
            df.groupby(df.get("nerc_region", "area") if "nerc_region" in df.columns else "area")
            ["customers_affected"]
            .transform(lambda x: x.rolling(12, min_periods=1).count())
        )

        # Exponentially weighted trend
        df["ewma_customers"] = (
            df["customers_affected"].fillna(0)
            .ewm(span=6, min_periods=1)
            .mean()
        )

    log.info(f"Feature engineering complete: {df.shape[1]} features, {len(df):,} records")
    return df


def create_weather_features(noaa: pd.DataFrame) -> pd.DataFrame:
    """
    Builds weather-level features from NOAA storm events.
    Aggregated to state-month level for joining with outage data.
    """
    if noaa.empty:
        return pd.DataFrame()

    df = noaa.copy()
    df.columns = df.columns.str.lower()

    if "event_date" not in df.columns and "begin_date_time" in df.columns:
        df["event_date"] = pd.to_datetime(df["begin_date_time"], errors="coerce")

    if "event_date" in df.columns:
        df["year"]  = df["event_date"].dt.year
        df["month"] = df["event_date"].dt.month

    # Severity score for each storm type
    storm_severity = {
        "Ice Storm": 5, "Winter Storm": 4, "Blizzard": 5,
        "Hurricane (Typhoon)": 5, "Tornado": 5,
        "High Wind": 3, "Thunderstorm Wind": 3,
        "Extreme Cold/Wind Chill": 4, "Heavy Snow": 3,
        "Flood": 3, "Flash Flood": 4,
        "Lightning": 2, "Heavy Rain": 2, "Tropical Storm": 4
    }

    if "event_type" in df.columns:
        df["storm_severity_score"] = df["event_type"].map(storm_severity).fillna(2)
        df["is_ice_event"]    = df["event_type"].isin(["Ice Storm", "Blizzard"]).astype(int)
        df["is_wind_event"]   = df["event_type"].isin(["High Wind", "Thunderstorm Wind"]).astype(int)
        df["is_winter_event"] = df["event_type"].isin(
            ["Ice Storm", "Winter Storm", "Blizzard", "Heavy Snow",
             "Extreme Cold/Wind Chill"]
        ).astype(int)

    # Damage features
    if "damage_property_usd" in df.columns:
        df["log_property_damage"] = np.log1p(df["damage_property_usd"].fillna(0))
        df["is_high_damage"]      = (df["damage_property_usd"] > 1_000_000).astype(int)

    log.info(f"Weather features: {df.shape[1]} features, {len(df):,} records")
    return df


def build_ml_dataset(doe_features: pd.DataFrame,
                     weather_features: pd.DataFrame = None,
                     target_col: str = "is_major_outage") -> tuple:
    """
    Assembles the final ML-ready dataset.

    Returns: X (features), y (target), feature_names
    """
    df = doe_features.copy()

    # Define feature columns — only numeric
    feature_cols = [c for c in [
        # Time
        "month", "quarter", "day_of_week", "is_weekend",
        "month_sin", "month_cos", "is_high_risk_month",
        # Season
        "season_risk_score", "season_Winter", "season_Summer",
        "season_Spring", "season_Fall",
        # Severity signals (from prior events)
        "log_demand_loss", "is_high_mw_loss",
        "log_customers", "state_risk_score",
        # Event type
        "is_weather_caused", "is_equipment_failure", "is_cyber",
        # Historical
        "rolling_12mo_events", "ewma_customers",
        # Year trend
        "year",
    ] if c in df.columns]

    # Drop rows where target is missing
    df = df.dropna(subset=[target_col])

    X = df[feature_cols].fillna(0)
    y = df[target_col].astype(int)

    log.info(f"ML dataset: {X.shape[0]:,} samples × {X.shape[1]} features")
    log.info(f"Target distribution: {y.value_counts().to_dict()}")

    return X, y, feature_cols


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s")

    doe, noaa = load_processed_data()

    doe_feat     = create_outage_features(doe)
    weather_feat = create_weather_features(noaa)
    X, y, names  = build_ml_dataset(doe_feat)

    print(f"\n✅ Features ready: {X.shape}")
    print(f"Feature names: {names}")
