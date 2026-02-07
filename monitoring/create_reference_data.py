import pandas as pd

df = pd.read_csv("data/raw/creditcard.csv")

df.sample(10000, random_state=42).to_csv(
    "data/reference/reference.csv",
    index=False
)

print("Reference dataset created")
