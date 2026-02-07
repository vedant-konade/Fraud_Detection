from fastapi import FastAPI
from pydantic import BaseModel
import mlflow
import pandas as pd
import numpy as np
from pathlib import Path
from joblib import load


# Use the SAME backend as training
mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_registry_uri("sqlite:///mlflow.db")
# App setup

app = FastAPI(title="Fraud Detection API")


# Load model from MLflow registry

MODEL_PATH = Path(__file__).resolve().parents[1] / "models" / "production" / "model.joblib"
model = load(MODEL_PATH)

# Chosen production threshold
THRESHOLD = 0.9



# Input schema

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

# Health check

@app.get("/health")
def health():
    return {"status": "ok"}



# Prediction endpoint

@app.post("/predict")
def predict(transaction: Transaction):
    data = pd.DataFrame([transaction.dict()])
    prob = model.predict_proba(data)[0][1]
    prediction = int(prob >= THRESHOLD)

    return {
        "fraud_probability": round(float(prob), 4),
        "fraud_prediction": prediction,
        "threshold_used": THRESHOLD
    }
