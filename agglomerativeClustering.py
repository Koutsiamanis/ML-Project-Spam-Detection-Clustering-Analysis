import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AgglomerativeClustering
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # αναγκαίο για 3D γραφήματα

# 1. Φόρτωση αρχείου
df = pd.read_csv("C:/Users/patatakis/Desktop/odep/StudentsPerformance.csv")
X = df[["math score", "reading score", "writing score"]]

# 2. Κανονικοποίηση των δεδομένων
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 3. Agglomerative Clustering
agg_model = AgglomerativeClustering(n_clusters=3, metric='euclidean', linkage='ward')
labels = agg_model.fit_predict(X_scaled)

# 4. Αποθήκευση labels στο dataframe
df["agg_cluster"] = labels

# 5. 3D Γράφημα
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')

# Σχεδίαση σημείων με χρωματισμό βάσει cluster
scatter = ax.scatter(
    df["math score"],
    df["reading score"],
    df["writing score"],
    c=df["agg_cluster"],
    cmap="plasma",
    alpha=0.8
)

# Προσθήκη τίτλων και ετικετών αξόνων
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
