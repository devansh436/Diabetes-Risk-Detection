"""FastAPI service: prediction API + tiny HTML UI + Prometheus metrics."""
import json
import logging
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Gauge, Histogram, generate_latest
from pydantic import BaseModel, Field

from src.data import FEATURES

log = logging.getLogger("api")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
STATIC_DIR = Path(__file__).parent / "static"

# ---- Prometheus metrics ----------------------------------------------------
REQUESTS = Counter("http_requests_total", "HTTP requests", ["method", "path", "status"])
LATENCY = Histogram("http_request_duration_seconds", "Request latency", ["path"])
PREDICTIONS = Counter("predictions_total", "Predictions served", ["risk_level"])
PROBABILITY = Histogram(
    "prediction_probability", "Predicted diabetes probability",
    buckets=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
)
# input histograms = cheap data-drift signal (compare against training distribution)
INPUT_GLUCOSE = Histogram("input_glucose", "Glucose inputs", buckets=[70, 90, 110, 130, 150, 180, 220, 300])
INPUT_BMI = Histogram("input_bmi", "BMI inputs", buckets=[18, 22, 26, 30, 35, 40, 50, 70])
MODEL_INFO_G = Gauge("model_loaded_info", "Loaded model version (value is always 1)", ["version"])


class PatientData(BaseModel):
    Pregnancies: int = Field(ge=0, le=20, examples=[2])
    Glucose: float = Field(gt=0, le=400, examples=[120])
    BloodPressure: float = Field(ge=0, le=250, examples=[70])
    SkinThickness: float = Field(ge=0, le=100, examples=[20])
    Insulin: float = Field(ge=0, le=900, examples=[80])
    BMI: float = Field(gt=0, le=80, examples=[28.5])
    DiabetesPedigreeFunction: float = Field(ge=0, le=3, examples=[0.4])
    Age: int = Field(ge=1, le=120, examples=[35])


def risk_level(p: float) -> str:
    return "low" if p < 0.33 else "moderate" if p < 0.66 else "high"


@asynccontextmanager
async def lifespan(app: FastAPI):
    path = Path(os.getenv("MODEL_PATH", "models/model.joblib"))
    app.state.model, app.state.meta = None, {}
    if path.exists():
        app.state.model = joblib.load(path)
        meta_file = path.with_name("metrics.json")
        app.state.meta = json.loads(meta_file.read_text()) if meta_file.exists() else {}
        version = app.state.meta.get("model_version", "unknown")
        MODEL_INFO_G.labels(version).set(1)
        log.info("model loaded: v%s", version)
    else:
        log.error("model file %s not found", path)
    yield


app = FastAPI(title="Diabetes Risk API", version="1.0.0", lifespan=lifespan)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.middleware("http")
async def metrics_middleware(request: Request, call_next):
    start, status = time.perf_counter(), 500
    try:
        response = await call_next(request)
        status = response.status_code
        return response
    finally:
        route = request.scope.get("route")
        path = route.path if route else "unmatched"
        if path != "/metrics":
            REQUESTS.labels(request.method, path, str(status)).inc()
            LATENCY.labels(path).observe(time.perf_counter() - start)


@app.get("/", include_in_schema=False)
def index():
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/health")
def health(request: Request, response: Response):
    loaded = request.app.state.model is not None
    if not loaded:
        response.status_code = 503
    return {
        "status": "ok" if loaded else "unhealthy",
        "model_loaded": loaded,
        "model_version": request.app.state.meta.get("model_version"),
    }


@app.get("/model-info")
def model_info(request: Request):
    if request.app.state.model is None:
        raise HTTPException(503, "Model not loaded")
    return request.app.state.meta


@app.post("/predict")
def predict(patient: PatientData, request: Request):
    model = request.app.state.model
    if model is None:
        raise HTTPException(503, "Model not loaded")
    row = pd.DataFrame([patient.model_dump()])[FEATURES]
    prob = float(model.predict_proba(row)[0, 1])
    level = risk_level(prob)
    PREDICTIONS.labels(level).inc()
    PROBABILITY.observe(prob)
    INPUT_GLUCOSE.observe(patient.Glucose)
    INPUT_BMI.observe(patient.BMI)
    log.info("prediction prob=%.3f risk=%s", prob, level)
    return {
        "prediction": int(prob >= 0.5),
        "probability": round(prob, 4),
        "risk_level": level,
        "model_version": request.app.state.meta.get("model_version"),
        "disclaimer": "Educational project - not a medical diagnosis.",
    }


@app.get("/metrics", include_in_schema=False)
def metrics(request: Request, raw: bool = False):
    """Prometheus text for scrapers; a visual dashboard for browsers (Accept: text/html)."""
    if not raw and "text/html" in request.headers.get("accept", ""):
        return FileResponse(STATIC_DIR / "metrics.html")
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
