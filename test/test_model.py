import joblib
import numpy as np
import pandas as pd
import pytest

from heart.data import FEATURES, ROOT
from heart.model import pipeline


@pytest.mark.parametrize("family", ["logistic", "forest"])
def test_pipeline_imputes_and_roundtrips(family, tmp_path):
    df = pd.read_csv(ROOT / "data/processed/heart.csv")
    X = df[FEATURES].iloc[:100].copy()
    X.loc[X.index[:3], "ca"] = np.nan
    model = pipeline(family).fit(X, df.target.iloc[:100])
    before = model.predict_proba(X)
    assert before.shape == (100, 2)
    assert np.isfinite(before).all()
    assert np.allclose(before.sum(axis=1), 1)
    path = tmp_path / "model.joblib"
    joblib.dump(model, path)
    assert np.allclose(before, joblib.load(path).predict_proba(X))


def test_imputer_learns_only_fit_rows():
    df = pd.read_csv(ROOT / "data/processed/heart.csv")
    X = df[FEATURES].iloc[:100].copy()
    model = pipeline("logistic").fit(X, df.target.iloc[:100])
    imputer = (
        model.named_steps["preprocess"].named_transformers_["continuous"].named_steps["impute"]
    )
    assert imputer.statistics_[0] == X.age.median()
    extreme = X.iloc[:1].copy()
    extreme["age"] = 10000
    model.predict_proba(extreme)
    assert imputer.statistics_[0] == X.age.median()
