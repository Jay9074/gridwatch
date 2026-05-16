"""
GridWatch v2 - Regenerate Dashboard Summary CSVs
Run: python src/regenerate_summaries.py
"""
import pandas as pd
import numpy as np
from pathlib import Path

PROC_DIR = Path("data/processed")
SUMMARY_DIR = Path("data/summary")
SUMMARY_DIR.mkdir(parents=True, exist_ok=True)

print("="*60)
print("GridWatch v2 - Regenerating Dashboard Summaries")
print("="*60)

df = pd.read_csv(PROC_DIR / "eaglei_daily_northeast.csv")
df["date_parsed"] = pd.to_datetime(df["date"])
df["yr"]  = df["date_parsed"].dt.year
df["mo"]  = df["date_parsed"].dt.month
df["is_major_outage"]    = (df["max_customers_out"] >= 1000).astype(int)
df["is_critical_outage"] = (df["max_customers_out"] >= 10000).astype(int)
print(f"Loaded: {len(df):,} county-days")

STATE_RISK = {
    "Maine":0.87,"Vermont":0.78,"New Hampshire":0.75,
    "New York":0.72,"Pennsylvania":0.68,"Massachusetts":0.65,
    "Connecticut":0.61,"New Jersey":0.60,"Rhode Island":0.58
}

# 1. STATE RISK SUMMARY
print("\n[1/6] State risk summary...")
state_sum = df.groupby("state").agg(
    total_county_days     = ("is_major_outage", "count"),
    major_outage_days     = ("is_major_outage", "sum"),
    critical_outage_days  = ("is_critical_outage", "sum"),
    peak_customers_out    = ("max_customers_out", "max"),
    total_customer_hours  = ("total_customer_hours", "sum"),
).reset_index()
state_sum["outage_rate"] = state_sum["major_outage_days"] / state_sum["total_county_days"]
state_sum["state_vulnerability"] = state_sum["state"].map(STATE_RISK)
state_sum["composite_risk_score"] = (
    0.6 * (state_sum["outage_rate"] / state_sum["outage_rate"].max()) +
    0.4 * state_sum["state_vulnerability"]
).round(4)
state_sum = state_sum.sort_values("composite_risk_score", ascending=False)
state_sum.to_csv(SUMMARY_DIR / "state_risk_summary.csv", index=False)
print(f"  Saved: {len(state_sum)} states")
print(f"  Highest: {state_sum.iloc[0]['state']} ({state_sum.iloc[0]['outage_rate']:.1%})")

# 2. MONTHLY TREND
print("\n[2/6] Monthly trend...")
monthly = df.groupby(["yr","mo"]).agg(
    total_county_days   = ("is_major_outage", "count"),
    major_outage_days   = ("is_major_outage", "sum"),
    critical_outage_days= ("is_critical_outage", "sum"),
    peak_customers_out  = ("max_customers_out", "max"),
).reset_index()
monthly = monthly.rename(columns={"yr":"year","mo":"month"})
monthly["outage_rate"] = monthly["major_outage_days"] / monthly["total_county_days"]
monthly["month_date"] = pd.to_datetime(monthly[["year","month"]].assign(day=1))
monthly = monthly.sort_values("month_date").reset_index(drop=True)
monthly.to_csv(SUMMARY_DIR / "monthly_trend.csv", index=False)
print(f"  Saved: {len(monthly)} months")

# 3. SEASONAL SUMMARY
print("\n[3/6] Seasonal summary...")
def get_season(m):
    if m in [12,1,2]: return "Winter"
    if m in [3,4,5]:  return "Spring"
    if m in [6,7,8]:  return "Summer"
    return "Fall"
df["season"] = df["mo"].apply(get_season)

seasonal = df.groupby("season").agg(
    total_county_days  = ("is_major_outage", "count"),
    major_outage_days  = ("is_major_outage", "sum"),
    avg_customers      = ("max_customers_out", "mean"),
).reset_index()
seasonal["outage_rate"] = seasonal["major_outage_days"] / seasonal["total_county_days"]
season_order = {"Winter":1,"Spring":2,"Summer":3,"Fall":4}
seasonal["sort_key"] = seasonal["season"].map(season_order)
seasonal = seasonal.sort_values("sort_key").drop(columns=["sort_key"]).reset_index(drop=True)
seasonal.to_csv(SUMMARY_DIR / "seasonal_summary.csv", index=False)
print(f"  Saved: {len(seasonal)} seasons")
print(f"  Highest: {seasonal.loc[seasonal['outage_rate'].idxmax(),'season']} ({seasonal['outage_rate'].max():.1%})")

# 4. YEARLY TREND
print("\n[4/6] Yearly trend...")
yearly = df.groupby("yr").agg(
    total_county_days  = ("is_major_outage", "count"),
    major_outage_days  = ("is_major_outage", "sum"),
    critical_outages   = ("is_critical_outage", "sum"),
    peak_event         = ("max_customers_out", "max"),
).reset_index().rename(columns={"yr":"year"})
yearly["outage_rate"] = yearly["major_outage_days"] / yearly["total_county_days"]
yearly.to_csv(SUMMARY_DIR / "yearly_trend.csv", index=False)
print(f"  Saved: {len(yearly)} years")

# 5. STATE-MONTH HEATMAP
print("\n[5/6] State-month heatmap...")
state_month = df.groupby(["state","mo"]).agg(
    total_county_days = ("is_major_outage", "count"),
    major_outage_days = ("is_major_outage", "sum"),
).reset_index().rename(columns={"mo":"month"})
state_month["outage_rate"] = state_month["major_outage_days"] / state_month["total_county_days"]
state_month.to_csv(SUMMARY_DIR / "state_month_heatmap.csv", index=False)
print(f"  Saved: {len(state_month)} state-month cells")

# 6. COUNTY RISK SUMMARY
print("\n[6/6] County risk summary...")
county = df.groupby(["state","county","fips_code"]).agg(
    total_county_days  = ("is_major_outage", "count"),
    major_outage_days  = ("is_major_outage", "sum"),
    peak_customers_out = ("max_customers_out", "max"),
).reset_index()
county["outage_rate"] = county["major_outage_days"] / county["total_county_days"]
county["pct_rate"]     = county["outage_rate"] / county["outage_rate"].max()
county["pct_peak"]     = county["peak_customers_out"] / county["peak_customers_out"].max()
county["pct_exposure"] = county["major_outage_days"] / county["major_outage_days"].max()
county["composite_risk_score"] = (
    0.5 * county["pct_rate"] +
    0.3 * county["pct_peak"] +
    0.2 * county["pct_exposure"]
).round(4)
county = county.sort_values("composite_risk_score", ascending=False).drop(
    columns=["pct_rate","pct_peak","pct_exposure"]
)
county.to_csv(SUMMARY_DIR / "county_risk_summary.csv", index=False)
print(f"  Saved: {len(county)} counties")
print(f"  Highest risk: {county.iloc[0]['county']}, {county.iloc[0]['state']}")

print(f"\n{'='*60}")
print("ALL SUMMARIES REGENERATED")
print(f"{'='*60}")
for f in sorted(SUMMARY_DIR.glob("*.csv")):
    size_kb = f.stat().st_size / 1024
    print(f"  {f.name} ({size_kb:.1f} KB)")

print(f"\n{'='*60}")
print("KEY FINDINGS WITH FULL DATA")
print(f"{'='*60}")
print(f"\nTotal county-days: {len(df):,}")
print(f"Major outage rate: {df['is_major_outage'].mean():.2%}")
print(f"Critical outage rate: {df['is_critical_outage'].mean():.2%}")
print(f"Peak event: {df['max_customers_out'].max():,.0f} customers")
print(f"\nTop 5 states by outage rate:")
print(state_sum.head(5)[["state","outage_rate"]].to_string(index=False))
print(f"\nSeasonal ranking:")
print(seasonal[["season","outage_rate"]].to_string(index=False))
