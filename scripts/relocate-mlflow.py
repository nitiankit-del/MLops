"""Run after moving this submission to repair MLflow file-store artifact paths."""

import yaml

from heart.data import ROOT

store = ROOT / "mlruns"
for meta in store.glob("*/meta.yaml"):
    data = yaml.safe_load(meta.read_text())
    data["artifact_location"] = meta.parent.as_uri()
    meta.write_text(yaml.safe_dump(data))
for meta in store.glob("*/*/meta.yaml"):
    data = yaml.safe_load(meta.read_text())
    data["artifact_uri"] = (meta.parent / "artifacts").as_uri()
    meta.write_text(yaml.safe_dump(data))
print(f"MLflow file-store paths updated to {store}")
