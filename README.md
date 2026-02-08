🚀 Production-Grade Fraud Detection System (MLOps + Cloud)

End-to-end, production-ready ML system for credit card fraud detection with experiment tracking, cloud deployment, observability, and data drift monitoring — built following 2026 MLOps standards.

🔍 Why This Project Matters

Most ML projects stop at training a model.

This project goes all the way to production:

Model training with severe class imbalance

Experiment tracking & model registry

Containerized inference API

Cloud deployment on AWS

Observability with Prometheus metrics

Data drift detection using Evidently AI

This mirrors how real-world ML systems are built, deployed, and maintained.

This project was built end-to-end by a single engineer to simulate real production ML workflows.

📊 Dataset Used

Kaggle Credit Card Fraud Dataset
https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud

~284,000 transactions

~0.2% fraud cases

Highly imbalanced, real-world distribution

❗ Problem Statement

Credit card fraud detection is a highly imbalanced classification problem:

~99.8% legitimate transactions

~0.2% fraudulent transactions

Using accuracy alone is misleading.

This system focuses on:

Precision–Recall tradeoffs

AUPRC (Area Under Precision-Recall Curve)

Threshold tuning for cost-sensitive decisions

🏗️ System Architecture
Raw Data (CSV)
   ↓
Exploration & Training
   ↓
MLflow (Experiments + Model Registry)
   ↓
Production Model (Joblib)
   ↓
FastAPI Inference Service
   ↓
Docker Container
   ↓
AWS ECR → AWS App Runner
   ↓
Monitoring & Drift Detection

🧰 Tech Stack
Machine Learning

Scikit-learn (Logistic Regression)

Class imbalance handling (class_weight="balanced")

Precision / Recall / F1 / AUPRC optimization

MLOps

MLflow (experiment tracking + model registry)

Docker (containerization)

Evidently AI (data drift detection)

Backend & Infrastructure

FastAPI (inference API)

AWS ECR + AWS App Runner

Prometheus-compatible metrics

Testing

Pytest (API-level tests)

🧠 Model Training Highlights

Stratified train/validation split

Explicit handling of class imbalance

Threshold tuning to control false positives

Primary metric: AUPRC (not accuracy)

Baseline results:

Recall: ~0.91

AUPRC: ~0.71

Production threshold: 0.9

All experiments and metrics are tracked using MLflow.

🧪 Experiment Tracking (MLflow)

MLflow is used to:

Track hyperparameters and metrics

Version trained models

Maintain a model registry for production

Artifacts logged:

Precision, Recall, F1-score, AUPRC

Model binaries

Threshold configuration

🔮 Inference API
🔹 Prediction Endpoint

POST /predict

Request Body

{
  "Time": 0.0,
  "V1": 0.1,
  "V2": 0.1,
  ...
  "V28": 0.1,
  "Amount": 100.0
}


Response

{
  "fraud_probability": 0.0297,
  "fraud_prediction": 0,
  "threshold_used": 0.9
}

🔹 Health Check
GET /health

📈 Observability (Production Monitoring)

The API exposes Prometheus-compatible metrics:

GET /metrics


Tracked metrics:

Request count (by endpoint & status)

Request latency

Fraud prediction counts

Error rates (4xx / 5xx)

This enables real-time monitoring via Grafana or CloudWatch.

📉 Data Drift Detection

To detect changing data distributions:

Reference dataset sampled from training data

Current dataset simulated from new incoming data

Drift analysis generated using Evidently AI

Outputs:

Interactive HTML drift report

Feature-level drift statistics

Dataset-level drift summary

This reflects real Day-2 ML operations.

☁️ Cloud Deployment (AWS)

Docker image pushed to Amazon ECR

Deployed using AWS App Runner

Public inference endpoint exposed

Cost controlled within free-tier limits

Demonstrates:

Cloud-native ML deployment

IAM-based access control

Production container workflows

🧪 Testing

Basic API tests implemented using Pytest:

Health endpoint validation

Prediction endpoint validation

📁 Project Structure
fraud-detection-mlops/
│
├── training/               # Data exploration & model training
├── inference/              # FastAPI inference service
├── monitoring/             # Drift detection scripts
├── models/                 # Production model artifacts
├── data/
│   ├── raw/
│   ├── reference/
│   └── current/
├── reports/                # Drift reports
├── tests/                  # API tests
├── Dockerfile
├── requirements.txt
└── README.md

✅ What This Project Demonstrates

Building ML systems beyond notebooks

Production thinking (monitoring, drift, retraining readiness)

Cloud deployment with cost awareness

Handling real-world ML challenges

🔮 Planned Improvements

Automated retraining pipeline triggered by drift

RAG-based transaction explanation agent

CI/CD integration for model promotion

Cost-aware threshold optimization

👤 Author

Vedant Konade
Final-year IT student | AI / MLOps Engineer

Built to production standards, not academic demos.
