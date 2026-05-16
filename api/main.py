"""
GridWatch Public API v1
========================
Free, open API for power grid outage risk intelligence in the Northeast US.

Built on FastAPI. Deployed to Render.com free tier.
Live URL: https://gridwatch-api.onrender.com

This is the GridWatch differentiator vs commercial providers like DTN:
- Their API costs $50K+/year per customer
- Ours is free and publicly auditable

Endpoints:
    GET /                                 - API metadata
    GET /api/v1/health                    - Service health check
    GET /api/v1/storms/active             - All active storm events
    GET /api/v1/predictions/active        - All current outage predictions
    GET /api/v1/predictions/state/{state} - Predictions for one state
    GET /api/v1/predictions/county/{state}/{county} - One county
    GET /api/v1/accuracy                  - Public accuracy scorecard
    GET /api/v1/counties                  - All monitored counties
    GET /api/v1/states                    - State-level risk summary
    GET /docs                             - Interactive API docs (Swagger UI)

Run locally:
    pip install fastapi uvicorn[standard] pandas
    uvicorn api.main:app --reload --port 8000
    Open http://localhost:8000/docs
"""
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path
from typing import Optional, List
from datetime import datetime
import pandas as pd
import json
import os

# ── App Setup ──────────────────────────────────────────────────────
app = FastAPI(
    title="GridWatch Public API",
    description=(
        "Open API for Northeast US power grid outage risk intelligence. "
        "Free, publicly auditable, built on EAGLE-I (DOE/ORNL) and NOAA data. "
        "Source: https://github.com/Jay9074/gridwatch"
    ),
    version="1.0.0",
    contact={
        "name":  "Jaykumar Patel",
        "url":   "https://github.com/Jay9074/gridwatch",
    },
    license_info={
        "name": "MIT License",
        "url":  "https://opensource.org/licenses/MIT",
    },
)

# CORS - allow anyone to call this from anywhere
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["GET"],
    allow_headers=["*"],
)

# ── Data Sources ───────────────────────────────────────────────────
DATA_DIR     = Path(os.getenv("GRIDWATCH_DATA_DIR", "data"))
STORM_DIR    = DATA_DIR / "stormwatch"
SUMMARY_DIR  = DATA_DIR / "summary"

# Github fallback when running on Render without local data
GITHUB_BASE = "https://raw.githubusercontent.com/Jay9074/gridwatch/main/data"


def _load_csv(local_path: Path, github_path: str) -> pd.DataFrame:
    """Try local file first, fall back to GitHub if not present."""
    try:
        if local_path.exists():
            return pd.read_csv(local_path)
    except Exception:
        pass
    
    try:
        return pd.read_csv(f"{GITHUB_BASE}/{github_path}")
    except Exception:
        return pd.DataFrame()


def _load_json(local_path: Path, github_path: str) -> dict:
    """Try local file first, fall back to GitHub."""
    try:
        if local_path.exists():
            with open(local_path) as f:
                return json.load(f)
    except Exception:
        pass
    
    try:
        import urllib.request
        with urllib.request.urlopen(f"{GITHUB_BASE}/{github_path}") as r:
            return json.loads(r.read())
    except Exception:
        return {}


# ── Root ───────────────────────────────────────────────────────────
@app.get("/", tags=["meta"])
def root():
    """API metadata and documentation links."""
    return {
        "service":     "GridWatch Public API",
        "version":     "1.0.0",
        "description": "Open power grid outage risk intelligence",
        "docs":        "/docs",
        "endpoints": {
            "health":            "/api/v1/health",
            "active_storms":     "/api/v1/storms/active",
            "active_predictions":"/api/v1/predictions/active",
            "by_state":          "/api/v1/predictions/state/{state}",
            "by_county":         "/api/v1/predictions/county/{state}/{county}",
            "accuracy":          "/api/v1/accuracy",
            "counties":          "/api/v1/counties",
            "states":            "/api/v1/states",
        },
        "data_sources": [
            "EAGLE-I (DOE/ORNL) - county outage data",
            "NOAA Storm Events - weather events",
            "NOAA api.weather.gov - live forecasts",
            "EIA Form 861 - utility reliability metrics",
        ],
        "github": "https://github.com/Jay9074/gridwatch",
        "dashboard": "https://gridwatch-dashboard.streamlit.app",
    }


# ── Health ─────────────────────────────────────────────────────────
@app.get("/api/v1/health", tags=["meta"])
def health():
    """Service health check with data freshness."""
    storms      = _load_csv(STORM_DIR / "storms" / "active_storms.csv",
                            "stormwatch/storms/active_storms.csv")
    predictions = _load_csv(STORM_DIR / "predictions" / "active_predictions.csv",
                            "stormwatch/predictions/active_predictions.csv")
    
    return {
        "status":              "ok",
        "timestamp_utc":       datetime.utcnow().isoformat() + "Z",
        "storms_loaded":       len(storms),
        "predictions_loaded":  len(predictions),
        "data_source":         "local" if (STORM_DIR / "predictions" / "active_predictions.csv").exists() else "github_fallback",
    }


# ── Active Storms ──────────────────────────────────────────────────
@app.get("/api/v1/storms/active", tags=["storms"])
def get_active_storms(
    tier: Optional[str] = Query(None, description="Filter by tier: SEVERE, MODERATE, MINOR"),
    state: Optional[str] = Query(None, description="Filter by state name"),
):
    """All active storm events detected in the next 7-day forecast window."""
    df = _load_csv(STORM_DIR / "storms" / "active_storms.csv",
                   "stormwatch/storms/active_storms.csv")
    
    if df.empty:
        return {"count": 0, "storms": []}
    
    if tier:
        df = df[df["storm_tier"].str.upper() == tier.upper()]
    if state:
        df = df[df["state"].str.lower() == state.lower()]
    
    return {
        "count":  len(df),
        "filters": {"tier": tier, "state": state},
        "storms": df.to_dict(orient="records"),
    }


# ── Active Predictions ─────────────────────────────────────────────
@app.get("/api/v1/predictions/active", tags=["predictions"])
def get_active_predictions(
    tier: Optional[str] = Query(None),
    state: Optional[str] = Query(None),
    min_customers: Optional[int] = Query(None, description="Minimum predicted customers"),
    limit: int = Query(100, ge=1, le=500),
):
    """All current outage predictions across all counties."""
    df = _load_csv(STORM_DIR / "predictions" / "active_predictions.csv",
                   "stormwatch/predictions/active_predictions.csv")
    
    if df.empty:
        return {"count": 0, "total_customers_at_risk": 0, "predictions": []}
    
    if tier:
        df = df[df["storm_tier"].str.upper() == tier.upper()]
    if state:
        df = df[df["state"].str.lower() == state.lower()]
    if min_customers:
        df = df[df["predicted_customers"] >= min_customers]
    
    df = df.head(limit)
    
    return {
        "count":                    len(df),
        "total_customers_at_risk":  int(df["predicted_customers"].sum()) if len(df) > 0 else 0,
        "max_single_event":         int(df["predicted_customers"].max()) if len(df) > 0 else 0,
        "median_event":             float(df["predicted_customers"].median()) if len(df) > 0 else 0,
        "filters": {"tier": tier, "state": state, "min_customers": min_customers},
        "predictions": df.to_dict(orient="records"),
    }


# ── Predictions By State ───────────────────────────────────────────
@app.get("/api/v1/predictions/state/{state}", tags=["predictions"])
def get_predictions_by_state(state: str):
    """Outage predictions for one state."""
    df = _load_csv(STORM_DIR / "predictions" / "active_predictions.csv",
                   "stormwatch/predictions/active_predictions.csv")
    
    if df.empty:
        return {"state": state, "count": 0, "predictions": []}
    
    df = df[df["state"].str.lower() == state.lower()]
    
    if df.empty:
        raise HTTPException(404, f"No active predictions for state '{state}'. "
                                 f"Valid states: Maine, New Hampshire, Vermont, "
                                 f"Massachusetts, Rhode Island, Connecticut, "
                                 f"New York, New Jersey, Pennsylvania")
    
    return {
        "state":                    state,
        "count":                    len(df),
        "total_customers_at_risk":  int(df["predicted_customers"].sum()),
        "predictions":              df.to_dict(orient="records"),
    }


# ── Predictions By County ──────────────────────────────────────────
@app.get("/api/v1/predictions/county/{state}/{county}", tags=["predictions"])
def get_predictions_by_county(state: str, county: str):
    """Outage predictions for a specific county."""
    df = _load_csv(STORM_DIR / "predictions" / "active_predictions.csv",
                   "stormwatch/predictions/active_predictions.csv")
    
    if df.empty:
        return {"state": state, "county": county, "count": 0, "predictions": []}
    
    df = df[
        (df["state"].str.lower()  == state.lower()) &
        (df["county"].str.lower() == county.lower())
    ]
    
    if df.empty:
        raise HTTPException(404, f"No active predictions for {county}, {state}")
    
    return {
        "state":         state,
        "county":        county,
        "count":         len(df),
        "predictions":   df.to_dict(orient="records"),
    }


# ── Accuracy Scorecard ─────────────────────────────────────────────
@app.get("/api/v1/accuracy", tags=["meta"])
def get_accuracy():
    """Public accuracy scorecard - how well past predictions matched real outages.
    
    Updated every 60 days as EAGLE-I data becomes available for validation.
    """
    scorecard = _load_json(STORM_DIR / "validation" / "accuracy_scorecard.json",
                           "stormwatch/validation/accuracy_scorecard.json")
    
    if not scorecard:
        return {
            "status":   "accumulating",
            "message":  ("Accuracy data is being collected. EAGLE-I outage data "
                         "is published with a ~60 day lag. Initial accuracy "
                         "metrics will be available after first batch of predictions "
                         "ages past the lag window."),
            "validation_lag_days": 60,
        }
    
    return scorecard


# ── Counties ───────────────────────────────────────────────────────
@app.get("/api/v1/counties", tags=["reference"])
def get_counties(
    state: Optional[str] = None,
    min_risk: Optional[float] = Query(None, ge=0, le=1),
):
    """All monitored counties with historical risk metrics from EAGLE-I."""
    df = _load_csv(SUMMARY_DIR / "county_risk_summary.csv",
                   "summary/county_risk_summary.csv")
    
    if df.empty:
        return {"count": 0, "counties": []}
    
    if state:
        df = df[df["state"].str.lower() == state.lower()]
    if min_risk is not None:
        col = "composite_risk_score" if "composite_risk_score" in df.columns else "risk_score"
        if col in df.columns:
            df = df[df[col] >= min_risk]
    
    return {
        "count":     len(df),
        "filters":   {"state": state, "min_risk": min_risk},
        "counties":  df.to_dict(orient="records"),
    }


# ── States ─────────────────────────────────────────────────────────
@app.get("/api/v1/states", tags=["reference"])
def get_states():
    """State-level risk summary from full EAGLE-I dataset."""
    df = _load_csv(SUMMARY_DIR / "state_risk_summary.csv",
                   "summary/state_risk_summary.csv")
    
    if df.empty:
        return {"count": 0, "states": []}
    
    return {
        "count":  len(df),
        "states": df.to_dict(orient="records"),
    }


# ── Entry Point ────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", 8000)))
