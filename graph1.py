import pandas as pd
import matplotlib.pyplot as plt

# Δημιουργία συγκεντρωτικού πίνακα
comparison_df = pd.DataFrame({
    "Model": ["Decision Tree", "kNN (k=5)", "Naive Bayes", "Random Forest"],
    "Accuracy": [0.8806, 0.9012, 0.8773, 0.8936],
    "Precision (Spam)": [0.86, 0.91, 0.88, 0.88],
    "Recall (Spam)": [0.83, 0.83, 0.80, 0.85],
    "F1-score (Spam)": [0.85, 0.87, 0.84, 0.87]
})

# Ρύθμιση μεγέθους γραφήματος
plt.figure(figsize=(10, 6))

# Ανάδειξη κάθε μετρικής ως ξεχωριστή μπάρα
x = comparison_df["Model"]
bar_width = 0.2
r1 = range(len(x))
r2 = [i + bar_width for i in r1]
r3 = [i + bar_width for i in r2]
r4 = [i + bar_width for i in r3]

plt.bar(r1, comparison_df["Accuracy"], width=bar_width, label="Accuracy")
plt.bar(r2, comparison_df["Precision (Spam)"], width=bar_width, label="Precision (Spam)")
plt.bar(r3, comparison_df["Recall (Spam)"], width=bar_width, label="Recall (Spam)")
plt.bar(r4, comparison_df["F1-score (Spam)"], width=bar_width, label="F1-score (Spam)")

# Προσαρμογή του άξονα x
plt.xticks([r + bar_width*1.5 for r in range(len(x))], x, rotation=15)
plt.ylim(0.75, 1.0)
plt.ylabel("Απόδοση")
plt.title("Σύγκριση Αλγορίθμων Ταξινόμησης")
plt.legend()
plt.grid(axis="y", linestyle="--", alpha=0.7)
plt.tight_layout()

# Εμφάνιση γραφήματος
plt.show()
