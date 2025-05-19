import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules

# 1. Φόρτωση δεδομένων με One-Hot Encoding
df = pd.read_csv("StudentsPerformance_encoded.csv")

# 2. Apriori: Εξαγωγή frequent itemsets
frequent_itemsets = apriori(df, min_support=0.05, use_colnames=True)

# 3. Εξαγωγή κανόνων συσχέτισης με Lift ≥ 1.0
rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1.0)

# 4. Ταξινόμηση κανόνων κατά Lift (προαιρετικά)
rules_sorted = rules.sort_values(by="lift", ascending=False)

# 5. Αποθήκευση σε CSV
rules_sorted.to_csv("association_rules_apriori.csv", index=False)

print("Οι κανόνες συσχέτισης αποθηκεύτηκαν στο αρχείο 'association_rules_apriori.csv'")
