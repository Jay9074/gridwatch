"""
GridWatch - Check what's actually in raw EAGLE-I files
Run: python src/check_raw_data.py
"""
import pandas as pd
import os
from pathlib import Path

RAW_DIR = Path("data/raw")

print("="*60)
print("CHECKING RAW EAGLE-I FILES")
print("="*60)

if not RAW_DIR.exists():
    print(f"ERROR: {RAW_DIR} does not exist")
    exit()

files = sorted([f for f in os.listdir(RAW_DIR) if "eaglei" in f.lower() and f.endswith(".csv")])
print(f"\nFound {len(files)} EAGLE-I CSV files:")

for fname in files:
    path = RAW_DIR / fname
    size_mb = os.path.getsize(path) / (1024*1024)
    print(f"\n{fname} ({size_mb:.1f} MB)")
    
    try:
        df = pd.read_csv(path, nrows=10000, low_memory=False)
        print(f"  Sample columns: {list(df.columns)[:6]}")
        
        # Find time column
        time_col = None
        for c in df.columns:
            if "run_start_time" in c.lower() or "timestamp" in c.lower() or "date" in c.lower():
                time_col = c
                break
        
        if time_col:
            df[time_col] = pd.to_datetime(df[time_col], errors="coerce")
            months = sorted(df[time_col].dt.month.dropna().unique().astype(int).tolist())
            print(f"  Months in first 10K rows: {months}")
        
        # Full count
        full = pd.read_csv(path, low_memory=False)
        if time_col:
            full[time_col] = pd.to_datetime(full[time_col], errors="coerce")
            all_months = sorted(full[time_col].dt.month.dropna().unique().astype(int).tolist())
            print(f"  ALL months in file: {all_months}")
            print(f"  Total rows: {len(full):,}")
    except Exception as e:
        print(f"  Error reading: {e}")
