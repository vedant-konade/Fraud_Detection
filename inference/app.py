from fastapi import FastAPI, Response
from pydantic import BaseModel
import mlflow
from pathlib import Path
from joblib import load
import time

from prometheus_client import (
    Counter,
    Histogram,
    generate_latest,
    CONTENT_TYPE_LATEST
)


# MLflow setup (same as training)

mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_registry_uri("sqlite:///mlflow.db")


# FastAPI app

app = FastAPI(title="Fraud Detection API")


# Prometheus metrics

REQUEST_COUNT = Counter(
    "http_requests_total",
    "Total HTTP requests",
    ["method", "endpoint", "status"]
)

REQUEST_LATENCY = Histogram(
    "http_request_latency_seconds",
    "Request latency",
    ["endpoint"]
)

PREDICTION_COUNT = Counter(
    "fraud_predictions_total",
    "Fraud predictions",
    ["prediction"]
)


# Load model

MODEL_PATH = Path(__file__).resolve().parents[1] / "models" / "production" / "model.joblib"
model = load(MODEL_PATH)

THRESHOLD = 0.9


# Input schema (TOP LEVEL)

class Transaction(BaseModel):
    Time: float
    V1: float
    V2: float
    V3: float
    V4: float
    V5: float
    V6: float
    V7: float
    V8: float
    V9: float
    V10: float
    V11: float
    V12: float
    V13: float
    V14: float
    V15: float
    V16: float
    V17: float
    V18: float
    V19: float
    V20: float
    V21: float
    V22: float
    V23: float
    V24: float
    V25: float
    V26: float
    V27: float
    V28: float
    Amount: float

#  REQUIRED for Pydantic v2 + FastAPI
Transaction.model_rebuild()


# Health check

@app.get("/health")
def health():
    REQUEST_COUNT.labels("GET", "/health", "200").inc()
    return {"status": "ok"}


# Prediction endpoint

@app.post("/predict")
def predict(data: Transaction):
    start_time = time.time()
    endpoint = "/predict"

    try:
        X = [[
            data.Time, data.V1, data.V2, data.V3, data.V4, data.V5,
            data.V6, data.V7, data.V8, data.V9, data.V10, data.V11,
            data.V12, data.V13, data.V14, data.V15, data.V16, data.V17,
            data.V18, data.V19, data.V20, data.V21, data.V22, data.V23,
            data.V24, data.V25, data.V26, data.V27, data.V28, data.Amount
        ]]

        prob = model.predict_proba(X)[0][1]
        prediction = int(prob >= THRESHOLD)

        PREDICTION_COUNT.labels(str(prediction)).inc()
        REQUEST_COUNT.labels("POST", endpoint, "200").inc()

        return {
            "fraud_probability": round(prob, 4),
            "fraud_prediction": prediction,
            "threshold_used": THRESHOLD
        }

    except Exception:
        REQUEST_COUNT.labels("POST", endpoint, "500").inc()
        raise

    finally:
        REQUEST_LATENCY.labels(endpoint).observe(
            time.time() - start_time
        )


# Prometheus metrics endpoint

@app.get("/metrics")
def metrics():
    return Response(
        generate_latest(),
        media_type=CONTENT_TYPE_LATEST
    )
