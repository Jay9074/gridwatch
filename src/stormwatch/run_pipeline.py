"""
GridWatch Storm Watch - Full Pipeline Runner
Runs all 6 steps in order, then logs a heartbeat.

Run: python src/stormwatch/run_pipeline.py
Schedule: Windows Task Scheduler every 6 hours
Verify schedule: python src/stormwatch/heartbeat.py show
"""
import subprocess
import sys
from datetime import datetime
from pathlib import Path

SCRIPTS = [
    ("fetch_forecasts.py",          "Fetching NOAA forecasts"),
    ("fetch_advanced_weather.py",   "Computing advanced weather features"),
    ("detect_storms.py",            "Detecting storm events"),
    ("predict_outages.py",          "Predicting outage impacts"),
    ("restoration_estimator.py",    "Estimating restoration time and crews"),
    ("validate_predictions.py",     "Validating past predictions"),
]


def main():
    print("=" * 70)
    print(f"GridWatch Storm Watch Pipeline")
    print(f"Started: {datetime.now().isoformat()}")
    print("=" * 70)
    
    script_dir = Path(__file__).parent
    failed = []
    
    for i, (script, label) in enumerate(SCRIPTS, 1):
        print(f"\n[{i}/{len(SCRIPTS)}] {label}")
        print("-" * 70)
        result = subprocess.run(
            [sys.executable, str(script_dir / script)],
            capture_output=False
        )
        if result.returncode != 0:
            print(f"WARNING: {script} exited with code {result.returncode}")
            failed.append(script)
    
    # Log heartbeat
    print(f"\n{'=' * 70}")
    status = "OK" if not failed else "PARTIAL"
    note   = "all_steps_ok" if not failed else f"failed: {','.join(failed)}"
    
    try:
        sys.path.insert(0, str(script_dir))
        from heartbeat import log_heartbeat
        log_heartbeat(status, note)
    except Exception as e:
        print(f"Heartbeat log failed: {e}")
    
    print(f"Pipeline complete: {datetime.now().isoformat()}")
    if failed:
        print(f"Failed steps: {failed}")
    else:
        print("All steps succeeded.")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
