"""Nested CV family comparison, candidate logging, then one held-out evaluation."""

import hashlib
import json
import os
import platform

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from mlflow.models import infer_signature
from sklearn.metrics import ConfusionMatrixDisplay, RocCurveDisplay
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_validate, train_test_split

from heart.data import FEATURES, ROOT
from heart.model import SEED, metrics, pipeline

GRIDS = {
    "logistic": {"classifier__C": [0.1, 1.0, 10.0], "classifier__class_weight": [None, "balanced"]},
    "forest": {
        "classifier__max_depth": [3, None],
        "classifier__min_samples_leaf": [2, 5],
        "classifier__class_weight": [None, "balanced"],
    },
}
SCORING = {
    "accuracy": "accuracy",
    "precision": "precision",
    "recall": "recall",
    "roc_auc": "roc_auc",
}


def run():
    df = pd.read_csv(ROOT / "data/processed/heart.csv")
    train, test = train_test_split(
        np.arange(len(df)), test_size=0.2, stratify=df.target, random_state=SEED
    )
    X, y = df[FEATURES], df.target
    Xtr, ytr = X.iloc[train], y.iloc[train]
    out = ROOT / "reports"
    out.mkdir(exist_ok=True)
    (ROOT / "models").mkdir(exist_ok=True)
    tracking = os.environ.get("MLFLOW_TRACKING_URI", (ROOT / "mlruns").as_uri())
    mlflow.set_tracking_uri(tracking)
    mlflow.set_experiment("heart-disease-cleveland")
    inner = StratifiedKFold(4, shuffle=True, random_state=SEED)
    outer = StratifiedKFold(5, shuffle=True, random_state=SEED + 1)
    summary, fitted = {}, {}
    with mlflow.start_run(run_name="reproducible-comparison") as parent:
        mlflow.log_params(
            {
                "seed": SEED,
                "train_rows": len(train),
                "test_rows": len(test),
                "inner_folds": 4,
                "outer_folds": 5,
                "threshold": 0.5,
                "data_sha256": hashlib.sha256(
                    (ROOT / "data/processed/heart.csv").read_bytes()
                ).hexdigest(),
            }
        )
        for family, grid in GRIDS.items():
            with mlflow.start_run(run_name=family, nested=True):
                search = GridSearchCV(
                    pipeline(family),
                    grid,
                    cv=inner,
                    scoring=SCORING,
                    refit="roc_auc",
                    n_jobs=1,
                    error_score="raise",
                )
                nested = cross_validate(
                    search, Xtr, ytr, cv=outer, scoring=SCORING, n_jobs=1, error_score="raise"
                )
                search.fit(Xtr, ytr)
                fitted[family] = search.best_estimator_
                summary[family] = {
                    "best_params": search.best_params_,
                    "nested_cv": {
                        m: {
                            "mean": float(nested["test_" + m].mean()),
                            "std": float(nested["test_" + m].std(ddof=1)),
                            "folds": nested["test_" + m].tolist(),
                        }
                        for m in SCORING
                    },
                }
                mlflow.log_params(search.best_params_)
                mlflow.log_metrics(
                    {
                        f"nested_{m}_{s}": v[s]
                        for m, v in summary[family]["nested_cv"].items()
                        for s in ["mean", "std"]
                    }
                )
                candidates = pd.DataFrame(search.cv_results_)
                candidate_path = out / f"{family}_cv_results.csv"
                candidates.to_csv(candidate_path, index=False)
                mlflow.log_artifact(str(candidate_path))
                fig, ax = plt.subplots(figsize=(8, 3.5))
                ax.errorbar(
                    np.arange(len(candidates)),
                    candidates.mean_test_roc_auc,
                    yerr=candidates.std_test_roc_auc,
                    fmt="o",
                    color="#167d9a",
                    capsize=3,
                )
                ax.set(
                    xlabel="Candidate index",
                    ylabel="Inner CV ROC-AUC",
                    title=f"{family}: training-only tuning",
                )
                fig.tight_layout()
                plot_path = out / "figures" / f"{family}_tuning.png"
                fig.savefig(plot_path, dpi=160)
                plt.close(fig)
                mlflow.log_artifact(str(plot_path))
                for i, row in candidates.iterrows():
                    with mlflow.start_run(run_name=f"{family}-candidate-{i:02d}", nested=True):
                        mlflow.log_params(row["params"])
                        mlflow.log_metrics(
                            {
                                f"cv_{m}_{s}": float(row[f"{s}_test_{m}"])
                                for m in SCORING
                                for s in ["mean", "std"]
                            }
                        )
                        mlflow.log_dict(
                            {
                                "candidate": int(i),
                                "params": row["params"],
                                "fold_auc": [
                                    float(row[f"split{k}_test_roc_auc"]) for k in range(4)
                                ],
                            },
                            "candidate.json",
                        )
                        mlflow.log_artifact(str(plot_path))
        winner = max(summary, key=lambda f: summary[f]["nested_cv"]["roc_auc"]["mean"])
        model = fitted[winner]
        probability = model.predict_proba(X.iloc[test])[:, 1]
        test_metrics = metrics(y.iloc[test], probability)
        rng = np.random.default_rng(SEED)
        boot = []
        for _ in range(2000):
            ix = rng.integers(0, len(test), len(test))
            if len(np.unique(y.iloc[test].to_numpy()[ix])) == 2:
                boot.append(metrics(y.iloc[test].to_numpy()[ix], probability[ix])["roc_auc"])
        ci = np.quantile(boot, [0.025, 0.975]).tolist()
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        RocCurveDisplay.from_predictions(y.iloc[test], probability, ax=axes[0], name=winner)
        axes[0].plot([0, 1], [0, 1], "--", color="gray")
        ConfusionMatrixDisplay.from_predictions(
            y.iloc[test], probability >= 0.5, ax=axes[1], colorbar=False, cmap="Blues"
        )
        fig.tight_layout()
        fig.savefig(out / "figures/evaluation.png", dpi=180)
        plt.close(fig)
        joblib.dump(model, ROOT / "models/heart_pipeline.joblib")
        metadata = {
            "selected_family": winner,
            "selection": "Highest mean outer nested-CV ROC-AUC on training partition",
            "threshold": 0.5,
            "features": FEATURES,
            "seed": SEED,
            "train_rows": len(train),
            "test_rows": len(test),
            "test_metrics": test_metrics,
            "test_auc_bootstrap_95_ci": ci,
            "models": summary,
            "model_sha256": hashlib.sha256(
                (ROOT / "models/heart_pipeline.joblib").read_bytes()
            ).hexdigest(),
            "data_sha256": hashlib.sha256(
                (ROOT / "data/processed/heart.csv").read_bytes()
            ).hexdigest(),
            "mlflow_run_id": parent.info.run_id,
            "python": platform.python_version(),
        }
        (ROOT / "models/metadata.json").write_text(json.dumps(metadata, indent=2))
        (out / "results.json").write_text(json.dumps(metadata, indent=2))
        (out / "split.json").write_text(
            json.dumps({"train": train.tolist(), "test": test.tolist()}, indent=2)
        )
        pd.DataFrame(
            {
                "row": test,
                "actual": y.iloc[test].to_numpy(),
                "probability": probability,
                "prediction": (probability >= 0.5).astype(int),
            }
        ).to_csv(out / "holdout_predictions.csv", index=False)
        mlflow.log_params({"winner": winner})
        mlflow.log_metrics({"holdout_" + k: v for k, v in test_metrics.items()})
        mlflow.log_artifacts(str(out), artifact_path="report")
        mlflow.sklearn.log_model(
            model,
            "model",
            input_example=Xtr.head(2),
            signature=infer_signature(Xtr, model.predict(Xtr)),
            pip_requirements=str(ROOT / "requirements-serve.txt"),
        )
        mlflow.log_artifact(str(ROOT / "models/metadata.json"))
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    run()
