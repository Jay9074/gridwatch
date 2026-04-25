"""
GridWatch - Generate Year-over-Year Summary
Run: python src/generate_yearly.py
"""
import pandas as pd
import numpy as np
from pathlib import Path

PROC_DIR = Path("data/processed")

print("Loading EAGLE-I data...")
df = pd.read_csv(PROC_DIR / "eaglei_daily_northeast.csv")
df["is_major_outage"] = (df["max_customers_out"] >= 1000).astype(int)
print(f"Loaded: {len(df):,} county-days")

# ── Year x State summary ──────────────────────────────────────────
print("Building year-by-state summary...")
yearly = df.groupby(["year", "state"]).agg(
    outage_days      = ("is_major_outage", "sum"),
    total_days       = ("is_major_outage", "count"),
    max_customers    = ("max_customers_out", "max"),
    avg_customers    = ("max_customers_out", "mean"),
    total_cust_hours = ("total_customer_hours", "sum"),
).reset_index()

yearly["outage_rate"]   = yearly["outage_days"] / yearly["total_days"]
yearly["log_avg"]       = np.log1p(yearly["avg_customers"])

yearly.to_csv(PROC_DIR / "yearly_state_summary.csv", index=False)
print(f"Saved: data/processed/yearly_state_summary.csv")

# ── Print summary ─────────────────────────────────────────────────
print("\n" + "=" * 55)
print("YEAR-OVER-YEAR OUTAGE RATE BY STATE")
print("=" * 55)
pivot = yearly.pivot(index="state", columns="year", values="outage_rate")
pivot = pivot.round(3)
print(pivot.to_string())

print("\nStates getting WORSE over time:")
for state in yearly["state"].unique():
    s = yearly[yearly["state"] == state].sort_values("year")
    if len(s) >= 4:
        first_half = s.iloc[:len(s)//2]["outage_rate"].mean()
        second_half = s.iloc[len(s)//2:]["outage_rate"].mean()
        change = (second_half - first_half) / max(first_half, 0.001)
        if change > 0.10:
            print(f"  {state}: +{change:.0%} increase")

print("\nStates getting BETTER over time:")
for state in yearly["state"].unique():
    s = yearly[yearly["state"] == state].sort_values("year")
    if len(s) >= 4:
        first_half = s.iloc[:len(s)//2]["outage_rate"].mean()
        second_half = s.iloc[len(s)//2:]["outage_rate"].mean()
        change = (second_half - first_half) / max(first_half, 0.001)
        if change < -0.10:
            print(f"  {state}: {change:.0%} decrease")
