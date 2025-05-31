import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv("/StudentsPerformance.csv")
X = df[["math score", "reading score", "writing score"]]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

k = 4 
neighbors = NearestNeighbors(n_neighbors=k)
neighbors_fit = neighbors.fit(X_scaled)
distances, indices = neighbors_fit.kneighbors(X_scaled)


distances_sorted = np.sort(distances[:, k - 1])

plt.figure(figsize=(8, 5))
plt.plot(distances_sorted, color="orange")
plt.title("k-Distance Graph (k=4) για επιλογή eps στο DBSCAN")
plt.xlabel("Δείγματα")
plt.ylabel("Απόσταση στον 4ο γείτονα")
plt.grid(True)
plt.tight_layout()
plt.show()
