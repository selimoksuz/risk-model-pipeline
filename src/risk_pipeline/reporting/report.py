from pathlib import Path
import json

def save_metrics(metrics: dict, out_dir: str):
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    with open(Path(out_dir) / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
