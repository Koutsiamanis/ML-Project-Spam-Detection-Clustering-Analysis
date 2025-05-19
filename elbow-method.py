import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# 1. Φόρτωση αρχείου και επιλογή αριθμητικών χαρακτηριστικών
df = pd.read_csv("C:/Users/patatakis/Desktop/odep/StudentsPerformance.csv")
X = df[["math score", "reading score", "writing score"]]

# 2. Κανονικοποίηση (ώστε όλα τα γνωρίσματα να είναι στην ίδια κλίμακα)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 3. Υπολογισμός inertia για διάφορα k (π.χ. από 1 έως 10)
inertias = []
k_values = range(1, 11)

for k in k_values:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X_scaled)
    inertias.append(kmeans.inertia_)

# 4. Γράφημα Elbow
plt.figure(figsize=(8, 5))
plt.plot(k_values, inertias, marker='o')
plt.title("Elbow Method για εύρεση του k")
plt.xlabel("Number of clusters (k)")
plt.ylabel("Inertia (σφάλμα)")
plt.xticks(k_values)
plt.grid(True)
plt.show()
