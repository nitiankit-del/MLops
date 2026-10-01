import json

import pytest
from fastapi.testclient import TestClient

from heart.api import app
from heart.data import ROOT


@pytest.fixture
def client():
    with TestClient(app) as c:
        yield c


@pytest.fixture
def patient():
    return json.loads((ROOT / "scripts/sample.json").read_text())


def test_prediction_contract_and_metrics(client, patient):
    r = client.post("/predict", json=patient)
    assert r.status_code == 200
    data = r.json()
    assert data["prediction"] == int(data["disease_probability"] >= data["threshold"])
    assert data["confidence"] == pytest.approx(
        max(data["disease_probability"], 1 - data["disease_probability"])
    )
    assert r.headers["X-Request-ID"]
    assert client.get("/health").status_code == 200
    assert client.get("/ready").json()["model_version"] == data["model_version"]
    metrics = client.get("/metrics").text
    assert 'heart_requests_total{route="/predict",status="200"}' in metrics
    assert "heart_predictions_total" in metrics
    assert client.get("/monitor").status_code == 200


@pytest.mark.parametrize(
    "field,value", [("cp", 0), ("cp", 99), ("thal", 2), ("age", -2), ("chol", 900), ("extra", 1)]
)
def test_invalid_input(client, patient, field, value):
    patient[field] = value
    assert client.post("/predict", json=patient).status_code == 422


def test_missing_field(client, patient):
    del patient["age"]
    assert client.post("/predict", json=patient).status_code == 422


def test_nullable_values(client, patient):
    patient["ca"] = None
    patient["thal"] = None
    patient["chol"] = None
    assert client.post("/predict", json=patient).status_code == 200


def test_startup_rejects_tampered_model(monkeypatch, tmp_path):
    (tmp_path / "metadata.json").write_text(json.dumps({"model_sha256": "wrong"}))
    (tmp_path / "heart_pipeline.joblib").write_bytes(b"tampered")
    monkeypatch.setenv("MODEL_DIR", str(tmp_path))
    with pytest.raises(RuntimeError, match="checksum"):
        with TestClient(app):
            pass
