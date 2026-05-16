"""
GridWatch Storm Watch - Heartbeat Logger
Appends a log entry every time the pipeline runs.
Use this to verify Windows Task Scheduler is firing on schedule.

This script is called automatically by run_pipeline.py.
View the log: type data\stormwatch\heartbeat.log
"""
import os
from pathlib import Path
from datetime import datetime
import pandas as pd

LOG_DIR  = Path("data/stormwatch")
LOG_FILE = LOG_DIR / "heartbeat.log"
LOG_DIR.mkdir(parents=True, exist_ok=True)


def log_heartbeat(status="OK", note=""):
    """Append one line to the heartbeat log."""
    now = datetime.now()
    
    # Try to grab quick metrics from the last pipeline output
    forecast_rows = 0
    storm_events  = 0
    predictions   = 0
    
    try:
        f = pd.read_csv(LOG_DIR / "forecasts" / "latest.csv")
        forecast_rows = len(f)
    except Exception:
        pass
    
    try:
        s = pd.read_csv(LOG_DIR / "storms" / "active_storms.csv")
        storm_events = len(s)
    except Exception:
        pass
    
    try:
        p = pd.read_csv(LOG_DIR / "predictions" / "active_predictions.csv")
        predictions = len(p)
    except Exception:
        pass
    
    line = (f"{now.isoformat()} | {status} | "
            f"forecasts={forecast_rows} storms={storm_events} predictions={predictions} | "
            f"{note}\n")
    
    with open(LOG_FILE, "a") as f:
        f.write(line)
    
    print(f"\n[HEARTBEAT] {line.strip()}")


def show_recent(n=10):
    """Show the last N heartbeats. Useful for verification."""
    if not LOG_FILE.exists():
        print("No heartbeats logged yet")
        return
    
    with open(LOG_FILE) as f:
        lines = f.readlines()
    
    print(f"\nLast {min(n, len(lines))} heartbeats:")
    print("=" * 80)
    for line in lines[-n:]:
        print(line.rstrip())
    
    # Compute interval between recent runs
    if len(lines) >= 2:
        timestamps = []
        for line in lines[-10:]:
            try:
                ts = datetime.fromisoformat(line.split(" | ")[0])
                timestamps.append(ts)
            except Exception:
                pass
        
        if len(timestamps) >= 2:
            intervals = [(timestamps[i] - timestamps[i-1]).total_seconds() / 3600
                         for i in range(1, len(timestamps))]
            avg_hrs = sum(intervals) / len(intervals)
            print(f"\nAverage interval between runs: {avg_hrs:.1f} hours")
            print(f"Expected: 6.0 hours (if scheduler set correctly)")
            
            if 5.5 <= avg_hrs <= 6.5:
                print("Scheduler is running on time")
            elif 23 <= avg_hrs <= 25:
                print("Running DAILY, not every 6 hours - check scheduler config")
            else:
                print(f"Unusual interval - check Task Scheduler trigger settings")


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "show":
        show_recent(int(sys.argv[2]) if len(sys.argv) > 2 else 20)
    else:
        log_heartbeat("OK", "manual run")
