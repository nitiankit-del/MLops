"""Validated JSON API, structured request logs and low-cardinality metrics."""

import hashlib
import json
import logging
import os
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

import joblib
import pandas as pd
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, Response
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest
from pydantic import BaseModel, ConfigDict, Field

from heart.data import FEATURES, ROOT

log = logging.getLogger("heart.requests")
logging.basicConfig(level=logging.INFO, format="%(message)s")
REQUESTS = Counter("heart_requests_total", "HTTP requests", ["route", "status"])
LATENCY = Histogram(
    "heart_request_seconds",
    "HTTP latency",
    ["route"],
    buckets=[0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2],
)
PREDICTIONS = Counter("heart_predictions_total", "Predicted classes", ["label"])


class Patient(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    age: float = Field(ge=18, le=120)
    sex: Literal[0, 1]
    cp: Literal[1, 2, 3, 4]
    trestbps: float | None = Field(..., ge=50, le=300)
    chol: float | None = Field(..., ge=50, le=700)
    fbs: Literal[0, 1]
    restecg: Literal[0, 1, 2]
    thalach: float | None = Field(..., ge=40, le=250)
    exang: Literal[0, 1]
    oldpeak: float | None = Field(..., ge=0, le=10)
    slope: Literal[1, 2, 3]
    ca: Literal[0, 1, 2, 3] | None
    thal: Literal[3, 6, 7] | None


class Prediction(BaseModel):
    prediction: int
    disease_probability: float
    confidence: float
    model_version: str
    threshold: float


@asynccontextmanager
async def lifespan(app):
    model_dir = Path(os.environ.get("MODEL_DIR", ROOT / "models"))
    metadata = json.loads((model_dir / "metadata.json").read_text())
    model_path = model_dir / "heart_pipeline.joblib"
    digest = hashlib.sha256(model_path.read_bytes()).hexdigest()
    if digest != metadata["model_sha256"]:
        raise RuntimeError("Model checksum mismatch")
    app.state.model = joblib.load(model_path)
    app.state.metadata = metadata
    app.state.version = digest[:12]
    yield


app = FastAPI(
    title="Cleveland Heart Disease API",
    version="1.0.0",
    lifespan=lifespan,
    description="Educational classifier; uncalibrated model scores, not a clinical risk estimate.",
)


@app.middleware("http")
async def observe(request: Request, call_next):
    started = time.perf_counter()
    request_id = str(uuid.uuid4())
    status = 500
    try:
        response = await call_next(request)
        status = response.status_code
        response.headers["X-Request-ID"] = request_id
        return response
    finally:
        route = request.scope.get("route")
        route = route.path if route else "unmatched"
        duration = time.perf_counter() - started
        REQUESTS.labels(route, str(status)).inc()
        LATENCY.labels(route).observe(duration)
        log.info(
            json.dumps(
                {
                    "event": "http_request",
                    "request_id": request_id,
                    "method": request.method,
                    "route": route,
                    "status": status,
                    "duration_ms": round(duration * 1000, 3),
                }
            )
        )


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/ready")
def ready(request: Request):
    return {"status": "ready", "model_version": request.app.state.version}


@app.post("/predict", response_model=Prediction)
def predict(patient: Patient, request: Request):
    frame = pd.DataFrame([patient.model_dump()], columns=FEATURES).astype(float)
    probability = float(request.app.state.model.predict_proba(frame)[0, 1])
    threshold = request.app.state.metadata["threshold"]
    label = int(probability >= threshold)
    PREDICTIONS.labels(str(label)).inc()
    return Prediction(
        prediction=label,
        disease_probability=probability,
        confidence=probability if label else 1 - probability,
        model_version=request.app.state.version,
        threshold=threshold,
    )


@app.get("/metrics", include_in_schema=False)
def prometheus_metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.get("/monitor", response_class=HTMLResponse, include_in_schema=False)
def monitor():
    return (ROOT / "monitoring/dashboard.html").read_text()
