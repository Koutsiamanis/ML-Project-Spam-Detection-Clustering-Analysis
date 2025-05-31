import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.naive_bayes import GaussianNB

df = pd.read_csv("/spam.csv")

print("Μοναδικές τιμές της class:", df["class"].unique())

df["class"] = df["class"].replace("emai", "email")

df.rename(columns={
    "cap_ave numeric": "cap_ave",
    "ooo": "000"
}, inplace=True)

df["class"] = df["class"].map({"email": 0, "spam": 1})
df.to_csv("/cleaned_spam.csv", index=False)

X = df.drop("class", axis=1)
y = df["class"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print("Training set:", X_train.shape)
print("Test set:", X_test.shape)
print(y.value_counts(normalize=True))

#Decision Tree
tree_model = DecisionTreeClassifier(random_state=42)
tree_model.fit(X_train, y_train)

y_pred = tree_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
report = classification_report(y_test, y_pred)

print("Accuracy: {:.2f}%".format(accuracy * 100))
print("\nConfusion Matrix:")
print(conf_matrix)
print("\nClassification Report:")
print(report)

#kNN με k=5
knn_model = KNeighborsClassifier(n_neighbors=5)
knn_model.fit(X_train, y_train)

y_pred_knn = knn_model.predict(X_test)

accuracy_knn = accuracy_score(y_test, y_pred_knn)
conf_matrix_knn = confusion_matrix(y_test, y_pred_knn)
report_knn = classification_report(y_test, y_pred_knn)

print("Accuracy: {:.2f}%".format(accuracy_knn * 100))
print("\nConfusion Matrix:")
print(conf_matrix_knn)
print("\nClassification Report:")
print(report_knn)

#Naive Bayes
nb_model = GaussianNB()
nb_model.fit(X_train, y_train)

y_pred_nb = nb_model.predict(X_test)
accuracy_nb = accuracy_score(y_test, y_pred_nb)
conf_matrix_nb = confusion_matrix(y_test, y_pred_nb)
report_nb = classification_report(y_test, y_pred_nb)

print("Accuracy: {:.2f}%".format(accuracy_nb * 100))
print("\nConfusion Matrix:")
print(conf_matrix_nb)
print("\nClassification Report:")
print(report_nb)

#Random Forest
rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train, y_train)

y_pred_rf = rf_model.predict(X_test)
accuracy_rf = accuracy_score(y_test, y_pred_rf)
conf_matrix_rf = confusion_matrix(y_test, y_pred_rf)
report_rf = classification_report(y_test, y_pred_rf)

print("Accuracy: {:.2f}%".format(accuracy_rf * 100))
print("\nConfusion Matrix:")
print(conf_matrix_rf)
print("\nClassification Report:")
print(report_rf)