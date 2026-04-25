"""
GridWatch - Generate County Level Summary
Run: python src/generate_county_summary.py
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

print("Loading EAGLE-I data...")
df = pd.read_csv(PROC_DIR / "eaglei_daily_northeast.csv")
df["is_major_outage"] = (df["max_customers_out"] >= 1000).astype(int)
print(f"Loaded: {len(df):,} county-days")

# Check county column
if "county" not in df.columns:
    print("No county column found")
    print("Available columns:", list(df.columns))
    exit()

print(f"Unique counties: {df['county'].nunique()}")

# Aggregate by county
print("Building county summary...")
county = df.groupby(["state","county"]).agg(
    outage_days      = ("is_major_outage","sum"),
    total_days       = ("is_major_outage","count"),
    max_customers    = ("max_customers_out","max"),
    avg_customers    = ("max_customers_out","mean"),
    total_cust_hours = ("total_customer_hours","sum"),
).reset_index()

county["outage_rate"]  = county["outage_days"] / county["total_days"]
county["state_risk"]   = county["state"].map(STATE_RISK).fillna(0.65)
county["risk_score"]   = (county["outage_rate"]*0.6 + county["state_risk"]*0.4).round(4)

def risk_level(score):
    if score >= 0.35:   return "HIGH"
    elif score >= 0.25: return "MEDIUM-HIGH"
    elif score >= 0.15: return "MEDIUM"
    else:               return "LOW-MEDIUM"

county["risk_level"] = county["risk_score"].apply(risk_level)
county = county.sort_values("risk_score", ascending=False)

county.to_csv(PROC_DIR / "county_summary.csv", index=False)
print(f"Saved: data/processed/county_summary.csv ({len(county):,} counties)")

print("\n" + "="*55)
print("TOP 20 HIGHEST RISK COUNTIES")
print("="*55)
print(county[["county","state","risk_score","risk_level",
              "outage_days","max_customers"]].head(20).to_string(index=False))

print("\nCounties by state:")
print(county.groupby("state")["county"].count().sort_values(ascending=False).to_string())

print("\nRisk level distribution:")
print(county["risk_level"].value_counts().to_string())
