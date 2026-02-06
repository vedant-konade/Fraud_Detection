from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    average_precision_score
)
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve
import pandas as pd
import numpy as np
# Resolve project root
PROJECT_ROOT = Path(__file__).resolve().parents[1]

data_path = PROJECT_ROOT / "data" / "raw" / "creditcard.csv"

df = pd.read_csv(data_path)

# print("Loaded data from:", data_path)
# print("Shape:", df.shape)

# # Target distribution
# class_counts = df["Class"].value_counts()
# class_ratio = df["Class"].value_counts(normalize=True)

# print("\nClass counts:")
# print(class_counts)

# print("\nClass ratio:")
# print(class_ratio)

# print("\nMax missing values in any column:")
# print(df.isnull().sum().max())

# print("\nData types:")
# print(df.dtypes)

# Separate features and target
X = df.drop(columns=["Class"])
y = df["Class"]

print("\nFeature matrix shape:", X.shape)
print("Target vector shape:", y.shape)

X_train, X_val, y_train, y_val = train_test_split(
    X,
    y,
    test_size=0.2,
    stratify=y,
    random_state=42
)

print("\nFraud ratio in training set:", y_train.mean())
print("Fraud ratio in validation set:", y_val.mean())


baseline_model = Pipeline(
    steps=[
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            random_state=42
        ))
    ]
)

baseline_model.fit(X_train, y_train)

# Get fraud probabilities on validation set
y_val_probs = baseline_model.predict_proba(X_val)[:, 1]

# Default threshold
threshold = 0.5
y_val_pred = (y_val_probs >= threshold).astype(int)


precision = precision_score(y_val, y_val_pred)
recall = recall_score(y_val, y_val_pred)
f1 = f1_score(y_val, y_val_pred)
auprc = average_precision_score(y_val, y_val_probs)

print("\nBaseline Logistic Regression (threshold = 0.5)")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1-score:  {f1:.4f}")
print(f"AUPRC:     {auprc:.4f}")

precisions, recalls, thresholds = precision_recall_curve(y_val, y_val_probs)

plt.figure(figsize=(6, 4))
plt.plot(recalls, precisions)
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision–Recall Curve (Baseline Model)")
plt.grid(True)
plt.show()

thresholds_to_test = [0.9, 0.7, 0.5, 0.3, 0.1]

print("\nThreshold Tuning Results:")
for t in thresholds_to_test:
    y_pred_t = (y_val_probs >= t).astype(int)

    p = precision_score(y_val, y_pred_t)
    r = recall_score(y_val, y_pred_t)
    f = f1_score(y_val, y_pred_t)

    print(f"Threshold={t:.1f} | Precision={p:.4f} | Recall={r:.4f} | F1={f:.4f}")