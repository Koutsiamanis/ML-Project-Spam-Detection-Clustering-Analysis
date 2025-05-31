import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt

df = pd.read_csv("/StudentsPerformance.csv")
X = df[["math score", "reading score", "writing score"]]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

dbscan = DBSCAN(eps=0.4, min_samples=5)
labels = dbscan.fit_predict(X_scaled)

df["dbscan_cluster"] = labels

cluster_counts = df["dbscan_cluster"].value_counts().sort_index()
print("Πλήθος Μαθητών ανά Cluster (DBSCAN):")
for cluster, count in cluster_counts.items():
    if cluster == -1:
        print(f"Cluster -1 (Θόρυβος): {count} μαθητές")
    else:
        print(f"Cluster {cluster}: {count} μαθητές")

plt.figure(figsize=(10, 6))
plt.scatter(
    df["math score"],
    df["reading score"],
    c=df["dbscan_cluster"],
    cmap="coolwarm",
    alpha=0.7
)
plt.title("DBSCAN")
plt.xlabel("Math Score")
plt.ylabel("Reading Score")
plt.grid(True)
plt.colorbar(label="Cluster")
plt.tight_layout()
plt.show()
