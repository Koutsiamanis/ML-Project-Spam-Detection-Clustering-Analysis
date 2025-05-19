import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# 1. Φόρτωση δεδομένων
df = pd.read_csv("C:/Users/patatakis/Desktop/odep/StudentsPerformance.csv")

# 2. Επιλογή αριθμητικών γνωρισμάτων
X = df[["math score", "reading score", "writing score"]]

# 3. Κανονικοποίηση (ώστε όλα τα χαρακτηριστικά να έχουν ίδια "ζυγαριά")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 4. KMeans clustering με k=3
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
kmeans.fit(X_scaled)

# 5. Προσθήκη ετικέτας cluster στον αρχικό πίνακα
df["kmeans_cluster"] = kmeans.labels_


# 6. Εμφάνιση κέντρων των συστάδων
print("Κεντρικές τιμές για κάθε συστάδα:")
print(kmeans.cluster_centers_)

# 7. 2D Γράφημα με χρώματα ανά συστάδα
plt.figure(figsize=(10, 6))
plt.scatter(X["math score"], X["reading score"], c=df["kmeans_cluster"], cmap="viridis", alpha=0.6)
plt.title("Συσταδοποίηση με K-Means (k=3)")
plt.xlabel("Math Score")
plt.ylabel("Reading Score")
plt.grid(True)
plt.colorbar(label="Cluster")
plt.show()
