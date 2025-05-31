import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules

df = pd.read_csv("/StudentsPerformance_encoded.csv")

frequent_itemsets = apriori(df, min_support=0.05, use_colnames=True)

rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1.0)

rules_sorted = rules.sort_values(by="lift", ascending=False)

rules_sorted.to_csv("/association_rules_apriori.csv", index=False)

print("Οι κανόνες συσχέτισης αποθηκεύτηκαν στο αρχείο 'association_rules_apriori.csv'")
