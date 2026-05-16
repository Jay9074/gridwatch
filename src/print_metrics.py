"""
GridWatch - Print Model Metrics
Run: python src/print_metrics.py
"""
import json
from pathlib import Path

path = Path("models") / "model_metrics.json"

if not path.exists():
    print("model_metrics.json not found in models/ folder")
else:
    m = json.loads(path.read_text())
    print(json.dumps(m, indent=2))
