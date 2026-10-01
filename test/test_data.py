import numpy as np
import pandas as pd
import pytest

from heart.data import FEATURES, clean


def sample():
    return pd.DataFrame(
        [[63, 1, 1, 145, 233, 1, 2, 150, 0, 2.3, 3, "?", 6, 2]], columns=FEATURES + ["num"]
    )


def test_target_mapping_missing_and_duplicates():
    frame = sample()
    frame = pd.concat([frame, frame], ignore_index=True)
    result = clean(frame)
    assert len(result) == 1
    assert result.target.tolist() == [1]
    assert np.isnan(result.ca.iloc[0])
    assert "num" not in result


@pytest.mark.parametrize("value,expected", [(0, 0), (1, 1), (2, 1), (3, 1), (4, 1)])
def test_target_classes(value, expected):
    frame = sample()
    frame["num"] = value
    assert clean(frame).target.iloc[0] == expected


@pytest.mark.parametrize(
    "col,value", [("num", 5), ("num", None), ("cp", 0), ("thal", 2), ("age", -1), ("chol", np.inf)]
)
def test_invalid_data_rejected(col, value):
    frame = sample()
    frame[col] = value
    with pytest.raises(ValueError):
        clean(frame)


def test_schema_drift_rejected():
    with pytest.raises(ValueError):
        clean(sample().drop(columns="age"))
