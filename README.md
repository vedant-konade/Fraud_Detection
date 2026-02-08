
#  Production-Grade Fraud Detection System (MLOps + Cloud)

> **End-to-end, production-ready ML system** for credit card fraud detection with experiment tracking, cloud deployment, observability, and data drift monitoring — built following 2026 MLOps standards.

---

##  Why This Project Matters

Most ML projects stop at training a model.

This one goes **all the way to production**:
- Model training with imbalanced data handling
- Experiment tracking & model registry
- Containerized inference API
- Cloud deployment (AWS)
- Observability with Prometheus
- Data drift detection with Evidently AI

This mirrors how **real ML systems are built and maintained in production**.

---

##  Dataset Used

https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud


##  Problem Statement

Credit card fraud detection is a **highly imbalanced classification problem**:
- ~99.8% legitimate transactions
- ~0.2% fraudulent transactions

Using accuracy alone is misleading.

This system focuses on **Precision-Recall tradeoffs**, **AUPRC**, and **threshold tuning**, which are critical in real-world fraud systems.

---

##  System Architecture

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


---

##  Tech Stack

**Machine Learning**
- Scikit-learn (Logistic Regression with class weighting)
- Precision / Recall / F1 / AUPRC optimization

**MLOps**
- MLflow (experiment tracking + model registry)
- Docker (containerization)
- Evidently AI (data drift detection)

**Backend & Infra**
- FastAPI (inference API)
- AWS ECR + AWS App Runner
- Prometheus metrics

**Testing**
- Pytest (API-level tests)

---

##  Model Training Highlights

- **Stratified train/validation split**
- **Class imbalance handled using `class_weight="balanced"`**
- **Threshold tuning** to control false positives
- **Primary metric:** AUPRC (not accuracy)

Example baseline results:
- **Recall:** ~0.91
- **AUPRC:** ~0.71
- **Production threshold:** 0.9 (cost-sensitive decision)

All experiments are tracked in **MLflow**.

---

## 🧪 Experiment Tracking (MLflow)

This project uses MLflow to:
- Track parameters & metrics
- Version trained models
- Register production-ready models

Artifacts logged:
- Metrics (Precision, Recall, F1, AUPRC)
- Model binaries
- Threshold decisions

---

##  Inference API

### Prediction
POST /predict
Content-Type: application/json

### Request Body
{
  "Time": 0.0,
  "V1": 0.1,
  "V2": 0.1,
  ...
  "V28": 0.1,
  "Amount": 100.0
}

### Response
{
  "fraud_probability": 0.0297,
  "fraud_prediction": 0,
  "threshold_used": 0.9
}

### Health Check
```http
GET /health



 Observability (Production Monitoring)

The API exposes Prometheus-compatible metrics:

GET /metrics

Tracked metrics:
Request count (by endpoint & status)
Request latency
Fraud prediction counts
Error rates (4xx / 5xx)
This enables real-time monitoring in Grafana or CloudWatch.

 Data Drift Detection

To handle changing data distributions:
Reference dataset sampled from training data
Current dataset simulated from new incoming data
Drift report generated using Evidently AI

Output:

Interactive HTML drift report
Feature-level drift statistics
Dataset-level drift summary
This reflects real “Day-2” ML operations.

 Cloud Deployment (AWS)

Docker image pushed to Amazon ECR
Deployed using AWS App Runner
Public inference endpoint exposed
Cost controlled under free-tier limits

Demonstrates:

Cloud-native ML deployment
IAM-based access control
Production container workflows

 Testing

Basic API tests included using Pytest:
Health endpoint test
Prediction endpoint validation

📂 Project Structure
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

 What This Demonstrates

This project shows ability to:
Build ML systems beyond notebooks
Think in terms of production, monitoring, and maintenance
Use cloud infrastructure responsibly
Handle real-world ML challenges (imbalance, drift, observability)

 Next Improvements (Planned)

Automated retraining pipeline triggered by drift
RAG-based transaction explanation agent
CI/CD integration for model promotion
Cost-aware threshold optimization

 Author

Vedant Konade
Final-year IT student | AI / MLOps Engineer
Built to production standards, not academic demos.