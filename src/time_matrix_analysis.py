import pandas as pd

data = pd.read_csv("../data/processed/time_matrix.csv", index_col=0)
print(data.head())
data.mean().mean()
data.mean().describe()
