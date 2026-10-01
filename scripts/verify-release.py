"""Verify release provenance and holdout predictions independently of training."""

import hashlib
import json

import joblib
import numpy as np
import pandas as pd

from heart.data import FEATURES, ROOT
from heart.model import metrics

meta = json.loads((ROOT / "models/metadata.json").read_text())
split = json.loads((ROOT / "reports/split.json").read_text())
df = pd.read_csv(ROOT / "data/processed/heart.csv")
assert set(split["train"]).isdisjoint(split["test"])
assert set(split["train"]) | set(split["test"]) == set(range(len(df)))
assert len(split["train"]) == meta["train_rows"]
assert len(split["test"]) == meta["test_rows"]
assert (
    hashlib.sha256((ROOT / "models/heart_pipeline.joblib").read_bytes()).hexdigest()
    == meta["model_sha256"]
)
model = joblib.load(ROOT / "models/heart_pipeline.joblib")
prob = model.predict_proba(df.iloc[split["test"]][FEATURES])[:, 1]
stored = pd.read_csv(ROOT / "reports/holdout_predictions.csv")
assert np.allclose(prob, stored.probability)
actual = metrics(df.iloc[split["test"]].target, prob)
assert all(np.isclose(actual[k], v) for k, v in meta["test_metrics"].items())
print(
    json.dumps(
        {
            "verification": "passed",
            "holdout_rows": len(prob),
            "model_sha256": meta["model_sha256"],
            "recomputed_metrics": actual,
        },
        indent=2,
    )
)
