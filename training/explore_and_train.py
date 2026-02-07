"""
Phase 1: Fraud Detection Training with MLflow (Container-Safe)

- Loads credit card fraud dataset
- Performs stratified train/validation split
- Trains Logistic Regression with scaling
- Evaluates using Precision, Recall, F1, AUPRC
- Performs threshold tuning
- Logs model, metrics, and artifacts to MLflow
- Registers model with container-safe artifact paths
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from joblib import dump
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    average_precision_score,
    precision_recall_curve
)

import mlflow
import mlflow.sklearn



# MLflow setup (CRITICAL: container-safe)

mlflow.set_tracking_uri("sqlite:///mlflow.db")

EXPERIMENT_NAME = "fraud_detection_phase_1"

experiment = mlflow.get_experiment_by_name(EXPERIMENT_NAME)
if experiment is None:
    mlflow.create_experiment(
        name=EXPERIMENT_NAME,
        artifact_location="mlruns"  # RELATIVE path (Docker-safe)
    )

mlflow.set_experiment(EXPERIMENT_NAME)



# Resolve project paths

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "raw" / "creditcard.csv"



# Load data

df = pd.read_csv(DATA_PATH)

print(f"Loaded data from: {DATA_PATH}")
print("Shape:", df.shape)

print("\nClass distribution:")
print(df["Class"].value_counts())

print("\nClass ratio:")
print(df["Class"].value_counts(normalize=True))

print("\nMax missing values in any column:")
print(df.isnull().sum().max())



# Separate features and target

X = df.drop(columns=["Class"])
y = df["Class"]

print("\nFeature matrix shape:", X.shape)
print("Target vector shape:", y.shape)



# Stratified split

X_train, X_val, y_train, y_val = train_test_split(
    X,
    y,
    test_size=0.2,
    stratify=y,
    random_state=42
)

print("\nFraud ratio in training set:", y_train.mean())
print("Fraud ratio in validation set:", y_val.mean())



# Build baseline pipeline

model = Pipeline(
    steps=[
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            random_state=42
        ))
    ]
)



# MLflow run

with mlflow.start_run(run_name="baseline_logreg_threshold_0.9"):

    # Train model
    model.fit(X_train, y_train)

    # Predict probabilities
    y_val_probs = model.predict_proba(X_val)[:, 1]

    # Baseline evaluation (threshold = 0.5)

    default_threshold = 0.5
    y_pred_default = (y_val_probs >= default_threshold).astype(int)

    default_precision = precision_score(y_val, y_pred_default)
    default_recall = recall_score(y_val, y_pred_default)
    default_f1 = f1_score(y_val, y_pred_default)
    auprc = average_precision_score(y_val, y_val_probs)

    print("\nBaseline Logistic Regression (threshold = 0.5)")
    print(f"Precision: {default_precision:.4f}")
    print(f"Recall:    {default_recall:.4f}")
    print(f"F1-score:  {default_f1:.4f}")
    print(f"AUPRC:     {auprc:.4f}")

    # Threshold tuning

    thresholds_to_test = [0.9, 0.7, 0.5, 0.3, 0.1]
    selected_threshold = 0.9

    print("\nThreshold Tuning Results:")
    for t in thresholds_to_test:
        y_pred_t = (y_val_probs >= t).astype(int)
        p = precision_score(y_val, y_pred_t)
        r = recall_score(y_val, y_pred_t)
        f = f1_score(y_val, y_pred_t)

        print(
            f"Threshold={t:.1f} | "
            f"Precision={p:.4f} | "
            f"Recall={r:.4f} | "
            f"F1={f:.4f}"
        )

        if t == selected_threshold:
            final_precision = p
            final_recall = r
            final_f1 = f

    # Precision–Recall Curve

    precisions, recalls, _ = precision_recall_curve(y_val, y_val_probs)

    plt.figure(figsize=(6, 4))
    plt.plot(recalls, precisions)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision–Recall Curve (Baseline Logistic Regression)")
    plt.grid(True)

    pr_curve_path = PROJECT_ROOT / "pr_curve.png"
    plt.savefig(pr_curve_path)
    plt.close()

    # MLflow logging

    mlflow.log_param("model", "LogisticRegression")
    mlflow.log_param("scaler", "StandardScaler")
    mlflow.log_param("class_weight", "balanced")
    mlflow.log_param("selected_threshold", selected_threshold)

    mlflow.log_metric("precision", final_precision)
    mlflow.log_metric("recall", final_recall)
    mlflow.log_metric("f1_score", final_f1)
    mlflow.log_metric("auprc", auprc)

    mlflow.log_artifact(pr_curve_path)

    mlflow.sklearn.log_model(
        sk_model=model,
        artifact_path="model",
        registered_model_name="fraud_detection_logreg"
    )

    print("\nModel and metrics logged to MLflow successfully.")


    MODEL_DIR = PROJECT_ROOT / "models" / "production"
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    dump(model, MODEL_DIR / "model.joblib")

    print(f"Production model exported to {MODEL_DIR}")
