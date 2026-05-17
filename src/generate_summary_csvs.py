"""
GridWatch - Generate data/summary/ CSVs from processed EAGLE-I data.
====================================================================
The dashboard loaders look for these files first and fall back to
hardcoded values. Creating them is the proper long-term approach.

Generates:
- state_risk_summary.csv     (per-state outage rates and risk scores)
- county_risk_summary.csv    (per-county aggregates)
- monthly_trend.csv          (month-by-month time series 2014-2025)
- seasonal_summary.csv       (4 seasons aggregated)
- yearly_state_summary.csv   (per-state per-year outage rates)

Run: python src/generate_summary_csvs.py
"""
import pandas as pd
from pathlib import Path
import sys

PROC_DIR = Path("data/processed")
OUT_DIR  = Path("data/summary")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Major outage threshold (matches v4 model definition)
MAJOR_THRESHOLD = 1000


def load_eaglei():
    path = PROC_DIR / "eaglei_daily_northeast.csv"
    if not path.exists():
        print(f"ERROR: {path} not found")
        sys.exit(1)
    print(f"Loading {path}...")
    df = pd.read_csv(path, parse_dates=["date"])
    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month
    df["season"] = df["month"].map(
        lambda m: "Winter" if m in [12,1,2] else "Spring" if m in [3,4,5]
        else "Summer" if m in [6,7,8] else "Fall"
    )
    df["is_major_outage"] = (df["max_customers_out"] >= MAJOR_THRESHOLD).astype(int)
    return df


def build_state_summary(df):
    print("Building state_risk_summary.csv...")
    grouped = df.groupby("state").agg(
        outage_rate=("is_major_outage", "mean"),
        major_outage_days=("is_major_outage", "sum"),
        peak_customers_out=("max_customers_out", "max"),
        avg_customers_out=("max_customers_out", "mean"),
        total_county_days=("date", "count"),
    ).reset_index()
    
    # Vulnerability proxy: peak/avg ratio (more peaked = more vulnerable)
    grouped["vulnerability_score"] = (
        grouped["peak_customers_out"] / grouped["avg_customers_out"].replace(0, 1)
    )
    # Normalize to 0-1
    vmin, vmax = grouped["vulnerability_score"].min(), grouped["vulnerability_score"].max()
    grouped["vulnerability_normalized"] = (
        (grouped["vulnerability_score"] - vmin) / (vmax - vmin)
    ).clip(0, 1)
    
    # Composite risk: 60% outage rate + 40% vulnerability
    or_min, or_max = grouped["outage_rate"].min(), grouped["outage_rate"].max()
    grouped["outage_rate_normalized"] = (
        (grouped["outage_rate"] - or_min) / (or_max - or_min)
    ).clip(0, 1)
    
    grouped["composite_risk_score"] = (
        0.6 * grouped["outage_rate_normalized"] +
        0.4 * grouped["vulnerability_normalized"]
    ).round(3)
    
    # Risk levels
    grouped["risk_level"] = pd.cut(
        grouped["composite_risk_score"],
        bins=[-0.01, 0.40, 0.65, 1.01],
        labels=["MEDIUM", "MEDIUM-HIGH", "HIGH"]
    ).astype(str)
    
    grouped = grouped.sort_values("composite_risk_score", ascending=False)
    grouped.to_csv(OUT_DIR / "state_risk_summary.csv", index=False)
    print(f"  Saved {len(grouped)} states")


def build_county_summary(df):
    print("Building county_risk_summary.csv...")
    grouped = df.groupby(["county", "state"]).agg(
        outage_rate=("is_major_outage", "mean"),
        major_outage_days=("is_major_outage", "sum"),
        peak_customers_out=("max_customers_out", "max"),
        avg_customers_out=("max_customers_out", "mean"),
        total_days=("date", "count"),
    ).reset_index()
    grouped = grouped.sort_values("outage_rate", ascending=False)
    grouped.to_csv(OUT_DIR / "county_risk_summary.csv", index=False)
    print(f"  Saved {len(grouped)} counties")


def build_monthly_trend(df):
    print("Building monthly_trend.csv...")
    grouped = df.groupby(["year", "month"]).agg(
        outage_events=("is_major_outage", "sum"),
        avg_customers_out=("max_customers_out", "mean"),
        max_customers_out=("max_customers_out", "max"),
        county_days_total=("date", "count"),
    ).reset_index()
    grouped["outage_rate"] = grouped["outage_events"] / grouped["county_days_total"]
    grouped = grouped.sort_values(["year", "month"])
    grouped.to_csv(OUT_DIR / "monthly_trend.csv", index=False)
    print(f"  Saved {len(grouped)} month-years")


def build_seasonal(df):
    print("Building seasonal_summary.csv...")
    grouped = df.groupby("season").agg(
        outage_rate=("is_major_outage", "mean"),
        outage_days=("is_major_outage", "sum"),
        avg_customers=("max_customers_out", "mean"),
        peak_customers=("max_customers_out", "max"),
    ).reset_index()
    # Sort by calendar order
    order = {"Winter": 0, "Spring": 1, "Summer": 2, "Fall": 3}
    grouped["sort_key"] = grouped["season"].map(order)
    grouped = grouped.sort_values("sort_key").drop(columns="sort_key")
    grouped.to_csv(OUT_DIR / "seasonal_summary.csv", index=False)
    print(f"  Saved {len(grouped)} seasons")


def build_yearly_state(df):
    print("Building yearly_state_summary.csv...")
    grouped = df.groupby(["year", "state"]).agg(
        outage_rate=("is_major_outage", "mean"),
        outage_events=("is_major_outage", "sum"),
        peak_customers=("max_customers_out", "max"),
    ).reset_index()
    grouped = grouped.sort_values(["state", "year"])
    grouped.to_csv(OUT_DIR / "yearly_state_summary.csv", index=False)
    print(f"  Saved {len(grouped)} state-years")


def main():
    print("=" * 60)
    print("GridWatch - Generate Dashboard Summary CSVs")
    print("=" * 60)
    
    df = load_eaglei()
    print(f"Loaded {len(df):,} county-days from {df['year'].min()} to {df['year'].max()}")
    
    build_state_summary(df)
    build_county_summary(df)
    build_monthly_trend(df)
    build_seasonal(df)
    build_yearly_state(df)
    
    print(f"\n{'=' * 60}")
    print(f"Done. Files saved to: {OUT_DIR.resolve()}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
