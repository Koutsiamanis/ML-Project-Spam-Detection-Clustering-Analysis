import pandas as pd

df = pd.read_csv("/StudentsPerformance.csv")

score_bins = [0, 60, 80, 100]
score_labels = ['χαμηλό', 'μέτριο', 'υψηλό']

df['math_cat'] = pd.cut(df['math score'], bins=score_bins, labels=score_labels, include_lowest=True)
df['reading_cat'] = pd.cut(df['reading score'], bins=score_bins, labels=score_labels, include_lowest=True)
df['writing_cat'] = pd.cut(df['writing score'], bins=score_bins, labels=score_labels, include_lowest=True)

df_apriori = df.drop(columns=['math score', 'reading score', 'writing score'])

df_apriori.to_csv("/StudentsPerformance_categorical.csv", index=False)

df_loaded = pd.read_csv("/StudentsPerformance_categorical.csv")
print(df_loaded.head())
