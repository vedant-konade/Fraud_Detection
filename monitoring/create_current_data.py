import pandas as pd
import numpy as np

df = pd.read_csv("data/raw/creditcard.csv")

current = df.sample(3000, random_state=7)

# Simulate drift
current["Amount"] = current["Amount"] * np.random.uniform(1.3, 1.6)

current.to_csv("data/current/current.csv", index=False)

print("Current dataset created (with drift)")
