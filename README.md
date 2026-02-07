//DATASET USED
https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
This dataset is highly imbalanced (~0.17% fraud). Therefore, accuracy is misleading and evaluation must focus on precision, recall, and PR-AUC.

Why this dataset is PERFECT for MLOps
Because:
Clean data → less time cleaning
Severe imbalance → realistic ML challenge
Stable schema → easier deployment
Large size → meaningful monitoring later

//LOGISTIC REGRESSION ALGORITHM USED
Strong baseline
Fast
Interpretable

I selected a threshold of 0.9 because it provides the best balance between high fraud recall (~89%) and significantly improved precision (~25%), making it suitable for a review-based fraud detection workflow.