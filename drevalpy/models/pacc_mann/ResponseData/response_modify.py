import pandas as pd
import numpy as np

response = pd.read_csv("response_GDSC2.csv")

np.random.seed(42) 
response_shuffled = response.sample(frac=1).reset_index(drop=True)

size_75 = int(0.75 * len(response_shuffled))

response_75 = response_shuffled.iloc[:size_75]

response_75.to_csv("response_75.csv", index=False)

print(f"Größe des 75% Teils: {len(response_75)}")
