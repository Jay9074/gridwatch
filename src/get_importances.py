"""
GridWatch - Get Feature Importances
Run: python src/get_importances.py
"""
import pickle
import json
from pathlib import Path

path = Path("models") / "outage_risk_model.pkl"

if not path.exists():
    print("Model not found at models/outage_risk_model.pkl")
else:
    with open(path, "rb") as f:
        bundle = pickle.load(f)

    model = bundle["model"]
    feats = bundle["feature_names"]

    if hasattr(model, "feature_importances_"):
        imp   = model.feature_importances_
        pairs = sorted(zip(feats, imp), key=lambda x: x[1], reverse=True)
        print("FEATURE IMPORTANCES:")
        for feat, val in pairs:
            print(f"  {feat}: {val:.4f}")

        # Save as JSON for dashboard
        result = {f: round(float(v), 4) for f, v in pairs}
        out = Path("models") / "feature_importances.json"
        with open(out, "w") as f:
            json.dump(result, f, indent=2)
        print(f"\nSaved to models/feature_importances.json")
    else:
        print("Model does not have feature_importances_")
        print("Model type:", type(model).__name__)
