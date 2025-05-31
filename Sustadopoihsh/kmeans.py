import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

df = pd.read_csv("/StudentsPerformance.csv")

X = df[["math score", "reading score", "writing score"]]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
kmeans.fit(X_scaled)

df["kmeans_cluster"] = kmeans.labels_

print("Κεντρικές τιμές για κάθε συστάδα:")
print(kmeans.cluster_centers_)


plt.figure(figsize=(10, 6))
plt.scatter(X["math score"], X["reading score"], c=df["kmeans_cluster"], cmap="viridis", alpha=0.6)
plt.title("Συσταδοποίηση με K-Means (k=3)")
plt.xlabel("Math Score")
plt.ylabel("Reading Score")
plt.grid(True)
plt.colorbar(label="Cluster")
plt.show()
