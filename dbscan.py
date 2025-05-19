import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt

# 1. Φόρτωση του αρχείου
df = pd.read_csv("C:/Users/patatakis/Desktop/odep/StudentsPerformance.csv")
X = df[["math score", "reading score", "writing score"]]

# 2. Κανονικοποίηση
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 3. Εφαρμογή DBSCAN
dbscan = DBSCAN(eps=0.4, min_samples=5)
labels = dbscan.fit_predict(X_scaled)

# 4. Προσθήκη των ετικετών στον πίνακα
df["dbscan_cluster"] = labels

# 5. Εκτύπωση των αποτελεσμάτων
cluster_counts = df["dbscan_cluster"].value_counts().sort_index()
print("Πλήθος Μαθητών ανά Cluster (DBSCAN):")
for cluster, count in cluster_counts.items():
    if cluster == -1:
        print(f"Cluster -1 (Θόρυβος): {count} μαθητές")
    else:
        print(f"Cluster {cluster}: {count} μαθητές")

# 6. 2D γράφημα (math vs reading)
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
