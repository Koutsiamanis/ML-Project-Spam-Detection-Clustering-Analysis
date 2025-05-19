import pandas as pd
from sklearn.preprocessing import StandardScaler
import scipy.cluster.hierarchy as sch
import matplotlib.pyplot as plt

# 1. Φόρτωση του dataset
df = pd.read_csv("C:/Users/patatakis/Desktop/odep/StudentsPerformance.csv")

# 2. Επιλογή των αριθμητικών γνωρισμάτων
X = df[["math score", "reading score", "writing score"]]

# 3. Κανονικοποίηση των γνωρισμάτων
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 4. Δημιουργία του δενδρογράμματος
plt.figure(figsize=(12, 6))
dendrogram = sch.dendrogram(sch.linkage(X_scaled, method="ward"))
plt.title("Δενδρόγραμμα")
plt.xlabel("Δείγματα Μαθητών")
plt.ylabel("Ευκλείδεια απόσταση")
plt.grid(True)
plt.show()
