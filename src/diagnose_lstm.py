"""
GridWatch — LSTM Diagnostic
Investigates why correlation hit 1.000 at 3 and 6 month horizons.
Run: python src/diagnose_lstm.py
"""
import pandas as pd
import numpy as np
from pathlib import Path

PROC_DIR = Path("data/processed")

print("="*60)
print("LSTM DIAGNOSTIC — Why is r=1.000?")
print("="*60)

# Load monthly time series the same way notebook 4 does
df = pd.read_csv(PROC_DIR / "eaglei_daily_northeast.csv")
df["is_major_outage"] = (df["max_customers_out"] >= 1000).astype(int)
df = df[df["year"] != 2023]

monthly = df.groupby(["year", "month"]).agg(
    outage_rate=("is_major_outage", "mean"),
    outage_events=("is_major_outage", "sum")
).reset_index()
monthly["year_month"] = pd.to_datetime(monthly[["year", "month"]].assign(day=1))
monthly = monthly.sort_values("year_month").reset_index(drop=True)

# Add lag features (same as notebook 4)
for lag in [1, 2, 3, 6]:
    monthly[f"outage_lag_{lag}"] = monthly["outage_rate"].shift(lag)
monthly["rolling_3m"] = monthly["outage_rate"].shift(1).rolling(3, min_periods=1).mean()
monthly["rolling_6m"] = monthly["outage_rate"].shift(1).rolling(6, min_periods=1).mean()
monthly = monthly.dropna().reset_index(drop=True)

print(f"\nTotal months in time series: {len(monthly)}")
print(f"Date range: {monthly['year_month'].min()} -> {monthly['year_month'].max()}")

SEQ_LEN = 6
SPLIT = 0.80

for horizon in [1, 3, 6]:
    n_total = len(monthly) - SEQ_LEN - horizon + 1
    n_train = int(n_total * SPLIT)
    n_test  = n_total - n_train
    print(f"\n{horizon}-month horizon:")
    print(f"  Total sequences: {n_total}")
    print(f"  Train sequences: {n_train}")
    print(f"  Test sequences:  {n_test}")
    if n_test < 5:
        print(f"  *** WARNING: Only {n_test} test points — correlation is unreliable!")
        print(f"  *** With 1-2 test points, correlation can artificially equal 1.000")
