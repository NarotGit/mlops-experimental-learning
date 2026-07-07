"""
src/api/app.py
FastAPI application exposing /predict, /health and Prometheus /metrics endpoints.

Run locally:
    uvicorn src.api.app:app --host 0.0.0.0 --port 8000 --reload

Swagger UI available at:  http://localhost:8000/docs
"""

import logging
import time
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from prometheus_fastapi_instrumentator import Instrumentator

# ---- Logging setup ----
logging.basicConfig(
    level=logging.INFO,
    format='{"time": "%(asctime)s", "level": "%(levelname)s", "message": "%(message)s"}',
    datefmt="%Y-%m-%dT%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ---- Import inference module ----
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from src.pipeline.inference import predict, predict_batch, get_feature_order


# ---- Lifespan (startup / shutdown) ----
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting Heart Disease Prediction API")
    # Pre-warm: load model at startup so first request isn't slow
    try:
        get_feature_order()
        logger.info("Model loaded successfully")
    except Exception as e:
        logger.error(f"Model load failed: {e}")
    yield
    logger.info("Shutting down Heart Disease Prediction API")


# ---- App definition ----
app = FastAPI(
    title="Heart Disease Prediction API",
    description=(
        "MLOps Assignment 01 — AIMLCZG523\n\n"
        "Predicts the presence of heart disease from patient health data "
        "using a trained sklearn Pipeline."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

# ---- Prometheus metrics (adds /metrics endpoint automatically) ----
Instrumentator().instrument(app).expose(app)


# ---- Request / Response schemas ----
class PredictRequest(BaseModel):
    age: float = Field(..., ge=1, le=120, description="Age in years")
    sex: int = Field(..., ge=0, le=1, description="Sex (1=male, 0=female)")
    cp: int = Field(..., ge=0, le=3, description="Chest pain type (0-3)")
    trestbps: float = Field(..., ge=50, le=250, description="Resting blood pressure (mmHg)")
    chol: float = Field(..., ge=100, le=600, description="Serum cholesterol (mg/dl)")
    fbs: int = Field(..., ge=0, le=1, description="Fasting blood sugar >120 mg/dl (1=true)")
    restecg: int = Field(..., ge=0, le=2, description="Resting ECG results (0-2)")
    thalach: float = Field(..., ge=60, le=250, description="Max heart rate achieved")
    exang: int = Field(..., ge=0, le=1, description="Exercise induced angina (1=yes)")
    oldpeak: float = Field(..., ge=0.0, le=10.0, description="ST depression induced by exercise")
    slope: int = Field(..., ge=0, le=2, description="Slope of peak exercise ST segment (0-2)")
    ca: int = Field(..., ge=0, le=4, description="Number of major vessels (0-4)")
    thal: int = Field(..., ge=0, le=3, description="Thalassemia (0=normal, 1=fixed, 2=reversible, 3=other)")

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "age": 63, "sex": 1, "cp": 3, "trestbps": 145,
                    "chol": 233, "fbs": 1, "restecg": 0, "thalach": 150,
                    "exang": 0, "oldpeak": 2.3, "slope": 0, "ca": 0, "thal": 1
                }
            ]
        }
    }


class PredictResponse(BaseModel):
    prediction: int = Field(..., description="0 = No heart disease, 1 = Heart disease")
    label: str = Field(..., description="Human-readable prediction label")
    confidence: float = Field(..., description="Confidence of predicted class (0-1)")
    probability_no_disease: float
    probability_disease: float


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    version: str


# ---- Middleware: log every request ----
@app.middleware("http")
async def log_requests(request: Request, call_next):
    start = time.time()
    response = await call_next(request)
    duration_ms = round((time.time() - start) * 1000, 2)
    logger.info(
        f"method={request.method} path={request.url.path} "
        f"status={response.status_code} duration_ms={duration_ms}"
    )
    return response


# ---- Endpoints ----
@app.get("/health", response_model=HealthResponse, tags=["Health"])
def health_check():
    """Liveness / readiness probe for Kubernetes."""
    try:
        get_feature_order()
        model_loaded = True
    except Exception:
        model_loaded = False
    return {
        "status": "healthy" if model_loaded else "degraded",
        "model_loaded": model_loaded,
        "version": "1.0.0",
    }


@app.post("/predict", response_model=PredictResponse, tags=["Prediction"])
def predict_endpoint(request: PredictRequest):
    """
    Predict heart disease risk from patient health features.

    Returns prediction (0/1), human-readable label, and probability scores.
    """
    try:
        input_dict = request.model_dump()
        pred, confidence = predict(input_dict)

        # Get both class probabilities
        from src.pipeline.inference import predict_batch
        result = predict_batch([input_dict])[0]

        label = "Heart Disease" if pred == 1 else "No Heart Disease"
        logger.info(f"Prediction: {label} | Confidence: {confidence:.4f}")

        return PredictResponse(
            prediction=pred,
            label=label,
            confidence=confidence,
            probability_no_disease=result["probability_no_disease"],
            probability_disease=result["probability_disease"],
        )
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@app.get("/", tags=["Info"])
def root():
    """API root — returns basic info."""
    return {
        "name": "Heart Disease Prediction API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health",
        "predict": "/predict",
        "metrics": "/metrics",
    }