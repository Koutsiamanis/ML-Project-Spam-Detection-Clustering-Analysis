import pandas as pd

df = pd.read_csv("StudentsPerformance_categorical.csv")

df_encoded = pd.get_dummies(df)

df_encoded.to_csv("StudentsPerformance_encoded.csv", index=False)

print(df_encoded.head())
