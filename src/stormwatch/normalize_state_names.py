"""
Quick utility to fix state name inconsistencies in predictions output.
Run after pipeline if abbreviated state names creep in.
"""
import pandas as pd
from pathlib import Path

STATE_MAP = {
    "PA": "Pennsylvania", "NH": "New Hampshire", "NJ": "New Jersey",
    "NY": "New York", "MA": "Massachusetts", "CT": "Connecticut",
    "RI": "Rhode Island", "ME": "Maine", "VT": "Vermont",
}

def main():
    files = [
        "data/stormwatch/predictions/active_predictions.csv",
        "data/stormwatch/predictions/prediction_log.csv",
        "data/stormwatch/storms/active_storms.csv",
        "data/stormwatch/predictions/restoration_estimates.csv",
    ]
    for fp in files:
        p = Path(fp)
        if not p.exists():
            continue
        df = pd.read_csv(p)
        if "state" not in df.columns:
            continue
        before = df["state"].value_counts().to_dict()
        df["state"] = df["state"].replace(STATE_MAP)
        after = df["state"].value_counts().to_dict()
        if before != after:
            df.to_csv(p, index=False)
            print(f"Normalized {p}")
        else:
            print(f"OK (no changes): {p}")

if __name__ == "__main__":
    main()
