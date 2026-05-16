"""
GridWatch - LSTM Diagnostic v2
Find out why we only have 17 months
Run: python src/diagnose_lstm_v2.py
"""
import pandas as pd
from pathlib import Path

PROC_DIR = Path("data/processed")

print("="*60)
print("WHY ONLY 17 MONTHS?")
print("="*60)

df = pd.read_csv(PROC_DIR / "eaglei_daily_northeast.csv")
print(f"\nTotal county-days in dataset: {len(df):,}")
print(f"Years available: {sorted(df['year'].dropna().unique().astype(int).tolist())}")
print(f"Months available per year:")
for yr in sorted(df['year'].dropna().unique().astype(int)):
    months = sorted(df[df['year']==yr]['month'].dropna().unique().astype(int).tolist())
    print(f"  {yr}: {months} ({len(months)} months)")

# Now reproduce the aggregation
df["is_major_outage"] = (df["max_customers_out"] >= 1000).astype(int)
df = df[df["year"] != 2023]
print(f"\nAfter excluding 2023: {len(df):,} county-days")

monthly = df.groupby(["year", "month"]).agg(
    outage_rate=("is_major_outage", "mean")
).reset_index()
print(f"\nUnique year-month combinations: {len(monthly)}")
print("\nAll year-months:")
print(monthly[["year","month"]].to_string(index=False))
