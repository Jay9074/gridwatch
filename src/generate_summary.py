"""
GridWatch - Generate Summary CSV for Dashboard
================================================
Run this once to create a small summary file that
the Streamlit dashboard can read from GitHub.

Run: python src/generate_summary.py
"""

import pandas as pd
import numpy as np
from pathlib import Path

PROC_DIR = Path("data/processed")

STATE_RISK = {
    "Maine":0.87,"Vermont":0.78,"New Hampshire":0.75,
    "New York":0.72,"Pennsylvania":0.68,"Massachusetts":0.65,
    "Connecticut":0.61,"New Jersey":0.60,"Rhode Island":0.58
}
STATE_COORDS = {
    "Maine":         {"lat":44.69,"lon":-69.38},
    "New Hampshire": {"lat":43.97,"lon":-71.57},
    "Vermont":       {"lat":44.56,"lon":-72.58},
    "Massachusetts": {"lat":42.23,"lon":-71.53},
    "Rhode Island":  {"lat":41.68,"lon":-71.51},
    "Connecticut":   {"lat":41.60,"lon":-72.69},
    "New York":      {"lat":42.97,"lon":-75.15},
    "New Jersey":    {"lat":40.06,"lon":-74.41},
    "Pennsylvania":  {"lat":40.99,"lon":-77.60},
}

print("Loading EAGLE-I data...")
df = pd.read_csv(PROC_DIR / "eaglei_daily_northeast.csv")
df["is_major_outage"] = (df["max_customers_out"] >= 1000).astype(int)
print(f"Loaded: {len(df):,} county-days")

# ── State summary ─────────────────────────────────────────────────
print("Building state summary...")
state_summary = df.groupby("state").agg(
    total_outage_days    = ("is_major_outage",      "sum"),
    total_days           = ("is_major_outage",      "count"),
    max_customers_out    = ("max_customers_out",    "max"),
    avg_customers_out    = ("max_customers_out",    "mean"),
    total_customer_hours = ("total_customer_hours", "sum"),
).reset_index()

state_summary["outage_rate"] = (
    state_summary["total_outage_days"] / state_summary["total_days"]
)
state_summary["state_risk"] = state_summary["state"].map(STATE_RISK).fillna(0.65)
state_summary["risk_score"] = (
    state_summary["outage_rate"] * 0.6 +
    state_summary["state_risk"]  * 0.4
).round(4)
state_summary["lat"] = state_summary["state"].map(
    lambda s: STATE_COORDS.get(s, {}).get("lat", 43.0)
)
state_summary["lon"] = state_summary["state"].map(
    lambda s: STATE_COORDS.get(s, {}).get("lon", -72.0)
)

def risk_level(score):
    if score >= 0.15:   return "HIGH"
    elif score >= 0.10: return "MEDIUM-HIGH"
    elif score >= 0.06: return "MEDIUM"
    else:               return "LOW-MEDIUM"

state_summary["risk_level"] = state_summary["risk_score"].apply(risk_level)
state_summary = state_summary.sort_values("risk_score", ascending=False)
state_summary.to_csv(PROC_DIR / "state_risk_summary.csv", index=False)
print("Saved: data/processed/state_risk_summary.csv")

# ── Monthly trend ─────────────────────────────────────────────────
print("Building monthly trend...")
trend = df.groupby(["year","month"]).agg(
    outage_events     = ("is_major_outage",   "sum"),
    avg_customers_out = ("max_customers_out", "mean"),
    total_cust_hours  = ("total_customer_hours","sum"),
).reset_index()
trend.to_csv(PROC_DIR / "monthly_trend.csv", index=False)
print("Saved: data/processed/monthly_trend.csv")

# ── Seasonal breakdown ────────────────────────────────────────────
print("Building seasonal breakdown...")
seasonal = df.groupby("season").agg(
    outage_days   = ("is_major_outage",   "sum"),
    total_days    = ("is_major_outage",   "count"),
    avg_customers = ("max_customers_out", "mean"),
).reset_index()
seasonal["outage_rate"] = seasonal["outage_days"] / seasonal["total_days"]
seasonal.to_csv(PROC_DIR / "seasonal_summary.csv", index=False)
print("Saved: data/processed/seasonal_summary.csv")

# ── Print results ─────────────────────────────────────────────────
print("\n" + "="*55)
print("STATE RISK SUMMARY")
print("="*55)
print(state_summary[[
    "state","risk_score","risk_level",
    "outage_rate","total_outage_days","max_customers_out"
]].to_string(index=False))

print("\n" + "="*55)
print("SEASONAL BREAKDOWN")
print("="*55)
print(seasonal.to_string(index=False))

print("\nDone! Three summary files saved to data/processed/")
print("Next: upload them to GitHub under a new folder called 'data/summary/'")
