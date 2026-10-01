"""All learned preprocessing is fitted inside each cross-validation fold."""

import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from heart.data import CATEGORICAL, CATEGORIES, NUMERIC

SEED = 523


def pipeline(family):
    continuous = Pipeline(
        [
            ("impute", SimpleImputer(strategy="median", add_indicator=True)),
            ("scale", StandardScaler()),
        ]
    )
    categorical = Pipeline(
        [
            ("impute", SimpleImputer(strategy="most_frequent")),
            (
                "encode",
                OneHotEncoder(
                    categories=[CATEGORIES[c] for c in CATEGORICAL],
                    handle_unknown="error",
                    sparse_output=False,
                ),
            ),
        ]
    )
    preprocessing = ColumnTransformer(
        [("continuous", continuous, NUMERIC), ("categorical", categorical, CATEGORICAL)]
    )
    if family == "logistic":
        clf = LogisticRegression(max_iter=2000, random_state=SEED)
    elif family == "forest":
        clf = RandomForestClassifier(n_estimators=200, random_state=SEED, n_jobs=1)
    else:
        raise ValueError(f"Unknown model: {family}")
    return Pipeline([("preprocess", preprocessing), ("classifier", clf)])


def metrics(y, probability):
    pred = (np.asarray(probability) >= 0.5).astype(int)
    return {
        "accuracy": accuracy_score(y, pred),
        "precision": precision_score(y, pred, zero_division=0),
        "recall": recall_score(y, pred, zero_division=0),
        "f1": f1_score(y, pred, zero_division=0),
        "roc_auc": roc_auc_score(y, probability),
    }
