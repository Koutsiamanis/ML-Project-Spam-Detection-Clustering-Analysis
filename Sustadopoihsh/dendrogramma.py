import pandas as pd
from sklearn.preprocessing import StandardScaler
import scipy.cluster.hierarchy as sch
import matplotlib.pyplot as plt

df = pd.read_csv("/StudentsPerformance.csv")

X = df[["math score", "reading score", "writing score"]]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

plt.figure(figsize=(12, 6))
dendrogram = sch.dendrogram(sch.linkage(X_scaled, method="ward"))
plt.title("Δενδρόγραμμα")
plt.xlabel("Δείγματα Μαθητών")
plt.ylabel("Ευκλείδεια απόσταση")
plt.grid(True)
plt.show()
