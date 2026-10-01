"""Export the selected release's complete MLflow run tree for portable review."""

import json

import mlflow
import pandas as pd

from heart.data import ROOT

mlflow.set_tracking_uri((ROOT / "mlruns").as_uri())
client = mlflow.MlflowClient()
root_id = json.loads((ROOT / "models/metadata.json").read_text())["mlflow_run_id"]
parent = client.get_run(root_id)
all_runs = client.search_runs([parent.info.experiment_id], max_results=10000)
selected = {root_id}
while True:
    children = {
        r.info.run_id for r in all_runs if r.data.tags.get("mlflow.parentRunId") in selected
    }
    if children <= selected:
        break
    selected |= children
rows = []
for r in all_runs:
    if r.info.run_id in selected:
        rows.append(
            {
                "run_id": r.info.run_id,
                "name": r.data.tags.get("mlflow.runName"),
                "parent": r.data.tags.get("mlflow.parentRunId"),
                "status": r.info.status,
                "params": r.data.params,
                "metrics": r.data.metrics,
            }
        )
(ROOT / "reports/experiments.json").write_text(json.dumps(rows, indent=2))
pd.json_normalize(rows).to_csv(ROOT / "reports/experiments.csv", index=False)
assert len(rows) == 17
assert all(r["status"] == "FINISHED" for r in rows)
print(f"Exported {len(rows)} finished runs for release {root_id}")
