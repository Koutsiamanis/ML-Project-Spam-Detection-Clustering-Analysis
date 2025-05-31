import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AgglomerativeClustering
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  

df = pd.read_csv("/StudentsPerformance.csv")
X = df[["math score", "reading score", "writing score"]]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

agg_model = AgglomerativeClustering(n_clusters=3, metric='euclidean', linkage='ward')
labels = agg_model.fit_predict(X_scaled)

df["agg_cluster"] = labels

fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')

scatter = ax.scatter(
    df["math score"],
    df["reading score"],
    df["writing score"],
    c=df["agg_cluster"],
    cmap="plasma",
    alpha=0.8
)

ax.set_title("Agglomerative Clustering 3D Visualization")
ax.set_xlabel("Math Score")
ax.set_ylabel("Reading Score")
ax.set_zlabel("Writing Score")

plt.colorbar(scatter, label="Cluster")
plt.tight_layout()
plt.show()

print("\nΠλήθος Μαθητών ανά Cluster:")
cluster_counts = df["agg_cluster"].value_counts().sort_index()
for cluster_id, count in cluster_counts.items():
    print(f"Cluster {cluster_id}: {count} μαθητές")
