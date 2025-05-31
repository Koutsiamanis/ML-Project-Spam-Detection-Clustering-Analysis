## Πίνακας Περιεχομένων

# Πίνακας Περιεχομένων

1. [Εισαγωγή](#εισαγωγή)  
2. [Μέρος Α - Κατηγοριοποίηση](#μέρος-α---κατηγοριοποίηση)  
   - [Επεξεργασία Δεδομένων](#επεξεργασία-δεδομένων)  
   - [Decision Tree](#decision-tree)  
   - [kNN](#knn)  
   - [Naive Bayes](#naive-bayes)  
   - [Random Forest](#random-forest)  
   - [Συμπέρασμα](#συμπέρασμα)  
3. [Μέρος Β - Συσταδοποίηση](#μέρος-β---συσταδοποίηση)  
   - [K-Means & Elbow Method](#k-means--elbow-method)  
   - [Agglomerative Clustering και Δενδρόγραμμα](#agglomerative-clustering-και-δενδρόγραμμα)  
   - [DBSCAN και k-Distance Graph](#dbscan-και-k-distance-graph)  
   - [Συμπέρασμα](#συμπέρασμα-συσταδοποίησης)  
4. [Μέρος Γ - Εξαγωγή Κανόνων Συσχέτισης](#μέρος-γ---εξαγωγή-κανόνων-συσχέτισης)  
   - [Διακριτοποίηση & One-Hot Encoding](#διακριτοποίηση--one-hot-encoding)  
   - [Αλγόριθμος Apriori](#αλγόριθμος-apriori)  
   - [Ανάλυση Κανόνων](#ανάλυση-κανόνων)  


### Εισαγωγή
Η παρούσα εργασία εστιάζει στην **ανάλυση και επεξεργασία δεδομένων μέσω αλγορίθμων μηχανικής μάθησης**, με σκοπό την κατηγοριοποίηση, συσταδοποίηση και εξαγωγή κανόνων συσχέτισης. Πραγματοποιούνται εφαρμογές σε δύο διαφορετικά σύνολα δεδομένων: ένα που αφορά την ταξινόμηση email σε spam ή μη, και ένα που περιλαμβάνει τις επιδόσεις μαθητών σε τρία βασικά μαθήματα. Μέσω της χρήσης δημοφιλών αλγορίθμων, όπως Decision Tree, k-NN, Naive Bayes, Random Forest, K-Means, Agglomerative Clustering και Apriori, επιχειρείται η ανάδειξη κρυφών μοτίβων και η εξαγωγή χρήσιμων συμπερασμάτων.

### Μέρος Α - Κατηγοριοποίηση

#### Επεξεργασία Δεδομένων
Αρχικά ξεκινάμε φορτώνοντας τα δεδομένα από το αρχείο spam.csv έτσι ώστε να τα προετοιμάσουμε.
```python
df = pd.read_csv("spam.csv")
```

Έπειτα κοιτάμε τις μοναδικές τιμές που έχει η στήλη "class" για να δούμε αν υπάρχουν πιθανά λάθη και να τα διορθώσουμε.
```python
print("Μοναδικές τιμές της class:", df["class"].unique())
```
`Μοναδικές τιμές της class: ['spam' 'emai']`
Από το αποτέλεσμα βλέπουμε ότι έχουμε μόνο 2 τιμες ωστόσο παρατηρούμε ενα ορθογραφικό λάθος διότι αντί για να λέει "email" λέει "emai" οπότε πρέπει να το διορθώσουμε.
```python
df["class"] = df["class"].replace("emai", "email")
```

Συνεχίζουμε μετονομαζοντας 2 στήλες για να ταιριάζουν με τα στοιχεία που μας έχουν δοθεί στην εκφώνηση της εργασίας
```python
df.rename(columns={"cap_ave numeric": "cap_ave", "ooo": "000"}, inplace=True)
```
και μετατρέπουμε την ετικέτα σε αριθμητική
```python
df["class"] = df["class"].map({"email": 0, "spam": 1})
```

Χωρίζουμε σε χαρακτηριστικά (X) και στόχο (y) και μετά σε training/ test set (80%-20%)
```python
X = df.drop("class", axis=1)
y = df["class"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print("Training set:", X_train.shape)
print("Test set:", X_test.shape)
```
`Training set: (3680, 8)`
`Test set: (921, 8)`

και ελέγχουμε αν τα δεδομένα μας είναι ανισόρροπα
```python
print(y.value_counts(normalize=True))
```
`0    0.605955`
`1    0.394045`
αυτό σημαίνει ότι 60.6% των emails είναι κανονικά και 39.4% είναι spam. Αυτό δεν είναι σοβαρά ανισόρροπο. Η διαφορά είναι αποδεκτή και δεν χρειάζεται oversampling.

#### Decision tree 
```python
tree_model = DecisionTreeClassifier(random_state=42)
tree_model.fit(X_train, y_train)

y_pred = tree_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
report = classification_report(y_test, y_pred)

print("Accuracy: {:.2f}%", accuracy)

print("\nConfusion Matrix:")
print(conf_matrix)

print("\nClassification Report:")
print(report)
```
Πέτυχε ακρίβεια **88.06%**, με αρκετά ικανοποιητικά αποτελέσματα σε precision (0.86) και recall (0.83) για την κατηγορία "spam". Ο πίνακας σύγχυσης έδειξε ότι **301 από τα 363 spam** αναγνωρίστηκαν σωστά, ενώ **510 από τα 558 κανονικά email** ταξινομήθηκαν σωστά.

|                          | Προβλέφθηκε Email (0)      | Προβλέφθηκε Spam (1)       |
| ------------------------ | -------------------------- | -------------------------- |
| **Πραγματικό Email (0)** | 510 σωστά (True Negatives) | 48 λάθος (False Positives) |
| **Πραγματικό Spam (1)**  | 62 λάθος (False Negatives) | 301 σωστά (True Positives) |

| Κατηγορία      | Precision | Recall   | F1-score | Υποδείγματα |
| -------------- | --------- | -------- | -------- | ----------- |
| Email (0)      | 0.89      | 0.91     | 0.90     | 558         |
| Spam (1)       | 0.86      | 0.83     | 0.85     | 363         |
| **Μέσος Όρος** | **0.88**  | **0.88** | **0.88** | **921**     |

#### kNN
```python
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
```
Το μοντέλο πέτυχε ακρίβεια **90.12%**. Από τα 921 δείγματα στο test set, ταξινομήθηκαν σωστά τα 527 από τα 558 email και τα 303 από τα 363 spam. Οι επιμέρους μετρικές (F1-score, precision, recall) ήταν ισορροπημένες και δείχνουν ότι ο αλγόριθμος αποδίδει σταθερά και στις δύο κατηγορίες, με ιδιαίτερη ικανότητα στην αναγνώριση κανονικών email (recall = 0.94).

|                          | Προβλέφθηκε Email (0)      | Προβλέφθηκε Spam (1)       |
| ------------------------ | -------------------------- | -------------------------- |
| **Πραγματικό Email (0)** | 527 σωστά (True Negatives) | 31 λάθος (False Positives) |
| **Πραγματικό Spam (1)**  | 60 λάθος (False Negatives) | 303 σωστά (True Positives) |

| Κατηγορία      | Precision | Recall   | F1-score | Υποδείγματα |
| -------------- | --------- | -------- | -------- | ----------- |
| Email (0)      | 0.90      | 0.94     | 0.92     | 558         |
| Spam (1)       | 0.91      | 0.83     | 0.87     | 363         |
| **Μέσος Όρος** | **0.90**  | **0.90** | **0.90** | **921**     |

#### Naive Bayes
```python
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
```
Πέτυχε ακρίβεια **87.73%**, με υψηλό precision (0.88) και recall (0.93) για την κατηγορία "email". Από τα 921 δείγματα στο test set, ταξινομήθηκαν σωστά τα 517 από τα 558 email και τα 291 από τα 363 spam.

|                          | Προβλέφθηκε Email (0)      | Προβλέφθηκε Spam (1)       |
| ------------------------ | -------------------------- | -------------------------- |
| **Πραγματικό Email (0)** | 517 σωστά (True Negatives) | 41 λάθος (False Positives) |
| **Πραγματικό Spam (1)**  | 72 λάθος (False Negatives) | 291 σωστά (True Positives) |

| Κατηγορία      | Precision | Recall   | F1-score | Υποδείγματα |
| -------------- | --------- | -------- | -------- | ----------- |
| Email (0)      | 0.88      | 0.93     | 0.90     | 558         |
| Spam (1)       | 0.88      | 0.80     | 0.84     | 363         |
| **Μέσος Όρος** | **0.88**  | **0.88** | **0.88** | **921**     |

#### Random forest
```python
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
```
Πέτυχε ακρίβεια **89.36%** στο test set. Ταξινόμησε σωστά 515 από τα 558 email και 308 από τα 363 spam.

|                          | Προβλέφθηκε Email (0)      | Προβλέφθηκε Spam (1)       |
| ------------------------ | -------------------------- | -------------------------- |
| **Πραγματικό Email (0)** | 515 σωστά (True Negatives) | 43 λάθος (False Positives) |
| **Πραγματικό Spam (1)**  | 55 λάθος (False Negatives) | 308 σωστά (True Positives) |

| Κατηγορία      | Precision | Recall   | F1-score | Υποδείγματα |
| -------------- | --------- | -------- | -------- | ----------- |
| Email (0)      | 0.90      | 0.92     | 0.91     | 558         |
| Spam (1)       | 0.88      | 0.85     | 0.87     | 363         |
| **Μέσος Όρος** | **0.89**  | **0.89** | **0.89** | **921**     |
#### Συμπέρασμα
Στο πλαίσιο της  ταξινόμησης μηνυμάτων σε κανονικά email και spam εφαρμόστηκαν οι αλγόριθμοι: Decision Tree, kNN (k=5), Naive Bayes και Random Forest. Τα αποτελέσματα έδειξαν ότι ο kNN είχε τη μεγαλύτερη ακρίβεια (90.12%), με εξαιρετικό precision στην αναγνώριση spam. Ο Random Forest παρουσίασε σταθερότητα και αξιοπιστία με το μεγαλύτερο recall για spam και πολύ καλό F1-score (0.87), υποδεικνύοντας ότι γενικεύει αποτελεσματικά χωρίς υπερεκπαίδευση. Ο Naive Bayes αν και ταχύτερος και πιο απλός, είχε ελαφρώς χαμηλότερες επιδόσεις. Ο Decision Tree παρουσίασε τη χαμηλότερη σχετική απόδοση.

| Model         | Accuracy | Precision (Spam) | Recall (Spam) | F1-score (Spam) |
| ------------- | -------- | ---------------- | ------------- | --------------- |
| Decision Tree | 0.8806   | 0.86             | 0.83          | 0.85            |
| kNN (k=5)     | 0.9012   | 0.91             | 0.83          | 0.87            |
| Naive Bayes   | 0.8773   | 0.88             | 0.8           | 0.84            |
| Random Forest | 0.8936   | 0.88             | 0.85          | 0.87            |

### Μέρος Β - Συσταδοποίηση

#### K-Means + Elbow Method
Ξεκινάμε με την μέθοδο Elbow έτσι ώστε να βρούμε τον σωστό αριθμό συστάδων που θα χρησιμοποιήσουμε έπειτα στον k-means.
```python
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

df = pd.read_csv("StudentsPerformance.csv")
X = df[["math score", "reading score", "writing score"]]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

inertias = []
k_values = range(1, 11)

for k in k_values:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X_scaled)
    inertias.append(kmeans.inertia_)

plt.figure(figsize=(8, 5))
plt.plot(k_values, inertias, marker='o')
plt.title("Elbow Method για εύρεση του k")
plt.xlabel("Number of clusters (k)")
plt.ylabel("Inertia (σφάλμα)")
plt.xticks(k_values)
plt.grid(True)
plt.show()
```
Χρησιμοποιώντας τον παραπάνω κώδικα με την βιβλιοθήκη Matplotlib δημιουργούμε το παρακάτω διάγραμμα.
![[elbow-method.png]]
Όπως παρατηρούμε το "elbow" δημιουργείται στο k=3 οπότε συνεχίζουμε με το K-means.
```python
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

df = pd.read_csv("/StudentsPerformance.csv")

X = df[["math score", "reading score", "writing score"]]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
kmeans.fit(X_scaled)

df["kmeans_cluster"] = kmeans.labels_


print("Κεντρικές τιμές για κάθε συστάδα:")
print(kmeans.cluster_centers_)

#Γράφημα
plt.figure(figsize=(10, 6))
plt.scatter(X["math score"], X["reading score"], c=df["kmeans_cluster"], cmap="viridis", alpha=0.6)
plt.title("Συσταδοποίηση με K-Means (k=3)")
plt.xlabel("Math Score")
plt.ylabel("Reading Score")
plt.grid(True)
plt.colorbar(label="Cluster")
plt.show()
```
Για την εφαρμογή του αλγορίθμου K-Means, έγινε αρχικά κανονικοποίηση των αριθμητικών γνωρισμάτων `math score`, `reading score` και `writing score` με χρήση του `StandardScaler.` Ο αλγόριθμος εκπαιδεύτηκε με:
- `n_clusters = 3` (σύμφωνα με το elbow)

| Cluster | Math Score | Reading Score | Writing Score |
|---------|-------------|----------------|----------------|
| 0       | –0.05       | –0.05          | –0.01          |
| 1       | –1.18       | –1.27          | –1.29          |
| 2       | +1.03       | +1.09          | +1.06          |

Ο αλγόριθμος κατέταξε τους μαθητές σε **τρεις συστάδες** με βάση τα κανονικοποιημένα σκορ τους:
- **Cluster 0**: Μαθητές με **μέσες επιδόσεις** (τιμές κοντά στον μέσο όρο). Αυτό το cluster περιλαμβάνει τους μαθητές που δεν ξεχωρίζουν ούτε θετικά ούτε αρνητικά στις επιδόσεις τους.
- **Cluster 1**: Μαθητές με **χαμηλές επιδόσεις**, καθώς και οι τρεις βαθμοί βρίσκονται περίπου 1.2-1.3 τυπικές αποκλίσεις κάτω από το μέσο όρο. Η ομάδα αυτή αντιπροσωπεύει τους αδύναμους μαθητές.
- **Cluster 2**: Μαθητές με **υψηλές επιδόσεις**, με σκορ περίπου 1 τυπική απόκλιση πάνω από το μέσο όρο σε όλα τα μαθήματα. Πρόκειται για τους πιο ικανούς μαθητές του δείγματος.![[k-means.png]]
Το παραπάνω γράφημα δείχνει την κατανομή των μαθητών στο επίπεδο `math score` και `reading score` μετά την εφαρμογή του K-Means clustering με **k=3**. Οι τρεις ομάδες διακρίνονται καθαρά οπτικά:
- Η **κίτρινη συστάδα (Cluster 2)** αντιπροσωπεύει τους **μαθητές υψηλών επιδόσεων**, οι οποίοι συγκεντρώνονται στο πάνω δεξί τμήμα του γραφήματος.
- Η **μοβ συστάδα (Cluster 0)** περιλαμβάνει **μαθητές με μέσες επιδόσεις**, κοντά στο κέντρο του διαγράμματος.
- Η **πράσινη συστάδα (Cluster 1)** συγκεντρώνει **μαθητές χαμηλής απόδοσης**, εμφανιζόμενους στο κάτω αριστερό τμήμα.

#### Αgglomerative clustering και δενδρόγραμμα
```python
import pandas as pd
from sklearn.preprocessing import StandardScaler
import scipy.cluster.hierarchy as sch
import matplotlib.pyplot as plt

df = pd.read_csv("./StudentsPerformance.csv")

X = df[["math score", "reading score", "writing score"]]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


plt.figure(figsize=(12, 6))
dendrogram = sch.dendrogram(sch.linkage(X_scaled, method="ward"))
plt.title("Δενδρόγραμμα")
plt.xlabel("Δείγματα Μαθητών")
plt.ylabel("Ευκλείδεια απόσταση")
plt.grid(True)
plt.show()
```
Mε την χρήση του παραπάνω κώδικα δημιουργούμε το δενδρόγραμμα έτσι ώστε να επιλέξουμε τον κατάλληλο αριθμό συστάδων. Η μέθοδος σύνδεσης που χρησιμοποιήθηκε ήταν η **Ward linkage** με **Ευκλείδεια απόσταση**.![[dendrogramma 1.png]]
Από την ανάλυση του διαγράμματος προτείνεται η διαίρεση του συνόλου σε 3 συστάδες.
```python
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AgglomerativeClustering
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  

df = pd.read_csv("./StudentsPerformance.csv")
X = df[["math score", "reading score", "writing score"]]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

agg_model = AgglomerativeClustering(n_clusters=3, metric='euclidean', linkage='ward')
labels = agg_model.fit_predict(X_scaled)

df["agg_cluster"] = labels

# 3D Γράφημα
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
```
- **Cluster 0**: Περιλαμβάνει μαθητές με **υψηλές επιδόσεις** και στα τρία μαθήματα, συγκεντρωμένους στις υψηλότερες περιοχές του τρισδιάστατου χώρου. Αποτελεί τη μεγαλύτερη συστάδα, με **481 μαθητές**.
- **Cluster 1**: Αντιπροσωπεύει μαθητές με **χαμηλές επιδόσεις**, με σκορ σημαντικά χαμηλότερα του μέσου όρου. Περιλαμβάνει **166 μαθητές**.
- **Cluster 2**: Αντιστοιχεί σε μαθητές **μέσης επίδοσης**, με σκορ κοντά στον μέσο όρο. Περιλαμβάνει **353 μαθητές**.

Η κατανομή των μαθητών στον χώρο επιβεβαιώθηκε μέσω **3D απεικόνισης**, με άξονες `math score`, `reading score`, και `writing score`. Το γράφημα έδειξε ξεκάθαρα τρεις ομαδοποιημένες περιοχές, οι οποίες αντιστοιχούν σε διαφορετικά επίπεδα επίδοσης.
![[Agglomerative Clustering3d.png]]
#### DBSCAN και k-distance graph
Ξεκινάμε δημιουργώντας το γράφημα k-distance.
```python
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
```
![[k-distanceGraph.png]]Παρατηρώντας το γράφημα, η απότομη αύξηση γίνεται περίπου στο ύψος eps ≈ 0.4.
```python
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
  

df = pd.read_csv("/StudentsPerformance.csv")
X = df[["math score", "reading score", "writing score"]]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

dbscan = DBSCAN(eps=0.4)
labels = dbscan.fit_predict(X_scaled)

df["dbscan_cluster"] = labels

cluster_counts = df["dbscan_cluster"].value_counts().sort_index()
print("Πλήθος Μαθητών ανά Cluster (DBSCAN):")
for cluster, count in cluster_counts.items():
    if cluster == -1:
        print(f"Cluster -1 (Θόρυβος): {count} μαθητές")
    else:
        print(f"Cluster {cluster}: {count} μαθητές")


plt.figure(figsize=(10, 6))
plt.scatter(
    df["math score"],
    df["reading score"],
    c=df["dbscan_cluster"],
    cmap="coolwarm",
    alpha=0.7
)

plt.title("DBSCAN")
plt.xlabel("Math Score")
plt.ylabel("Reading Score")
plt.grid(True)
plt.colorbar(label="Cluster")
plt.tight_layout()
plt.show()
```

| Cluster      | Πλήθος Μαθητών  |
| ------------ | --------------- |
| 0            | **976** μαθητές |
| –1 (Θόρυβος) | **24** μαθητές  |
Ο DBSCAN δεν εντόπισε πολλαπλές συστάδες, αλλά σχημάτισε μόνο μία κύρια ομάδα και 24 σημεία ως θόρυβο. Από αυτό καταλαβαίνουμε ότι οι επιδόσεις των μαθητών δεν παρουσιάζουν φυσικούς διαχωρισμούς βάσει πυκνότητας. Συνεπώς, ο DBSCAN δεν είναι η κατάλληλη μέθοδος για την ανάδειξη νοηματικών συστάδων στην παρούσα περίπτωση, σε αντίθεση με το K-Means και το Agglomerative Clustering, τα οποία παρήγαγαν πιο ερμηνεύσιμα αποτελέσματα.![[dbscan.png]]

#### Συμπέρασμα Συσταδοποίησης
Οι αλγόριθμοι KMeans και Agglomerative Clustering ήταν οι  κατάλληλοι για την ανάλυση του συγκεκριμένου συνόλου δεδομένων καθώς παρήγαγαν ξεκάθαρες και ερμηνεύσιμες συστάδες επιδόσεων μαθητών. Αντίθετα, ο DBSCAN δεν εντόπισε περισσότερες από μία ομάδες, φανερώνοντας ότι οι βαθμολογίες δεν παρουσιάζουν φυσικούς διαχωρισμούς σε πυκνότητα. Η σύγκριση των αποτελεσμάτων προσέφερε μια σφαιρική εικόνα της δομής των δεδομένων και υπογράμμισε την σημασία της επιλογής του κατάλληλου αλγορίθμου ανάλογα με τα χαρακτηριστικά του dataset.

### Μέρος Γ - Εξαγωγή Κανόνων Συσχέτισης

#### Διακριτοποίηση & One-Hot Encoding
Δεδομένου ότι ο αλγόριθμος Apriori απαιτεί κατηγορικά γνωρίσματα, έγινε διακριτοποίηση των αριθμητικών μεταβλητών `math score`, `reading score`, και `writing score`με το παρακάτω κώδικα:
```python
import pandas as pd

df = pd.read_csv("StudentsPerformance.csv")

score_bins = [0, 60, 80, 100]
score_labels = ['χαμηλό', 'μέτριο', 'υψηλό']

df['math_cat'] = pd.cut(df['math score'], bins=score_bins, labels=score_labels, include_lowest=True)
df['reading_cat'] = pd.cut(df['reading score'], bins=score_bins, labels=score_labels, include_lowest=True)
df['writing_cat'] = pd.cut(df['writing score'], bins=score_bins, labels=score_labels, include_lowest=True)

df_apriori = df.drop(columns=['math score', 'reading score', 'writing score'])

#Αποθήκευση του νέου dataset σε αρχείο
df_apriori.to_csv("StudentsPerformance_categorical.csv", index=False)
```
Οι βαθμολογίες κατανεμήθηκαν σε τρεις κατηγορίες:
- `χαμηλό`: 0–60
- `μέτριο`: 61–80
- `υψηλό`: 81–100

Μετά τη διακριτοποίηση των αριθμητικών γνωρισμάτων εφαρμόστηκε **One-Hot Encoding** ώστε όλα τα γνωρίσματα να μετατραπούν σε δυαδική μορφή. Η μετατροπή αυτή είναι απαραίτητη για την εφαρμογή του αλγορίθμου Apriori, καθώς ορίζεται ότι κάθε γραμμή του συνόλου δεδομένων πρέπει να αναπαρίσταται ως σύνολο κατηγορικών χαρακτηριστικών. 
```python
import pandas as pd

df = pd.read_csv("StudentsPerformance_categorical.csv")

df_encoded = pd.get_dummies(df)

df_encoded.to_csv("StudentsPerformance_encoded.csv", index=False)

print(df_encoded.head())
```

#### Αλγόριθμος Apriori
Κώδικας για τον αλγόριθμο Apriori που εξάγει σε csv όλους τους κανόνες με lift ≥ 1.0
```python
import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules

df = pd.read_csv("StudentsPerformance_encoded.csv")

frequent_itemsets = apriori(df, min_support=0.05, use_colnames=True)

rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1.0)

rules_sorted = rules.sort_values(by="lift", ascending=False)

rules_sorted.to_csv("association_rules_apriori.csv", index=False)

print("Οι κανόνες συσχέτισης αποθηκεύτηκαν στο αρχείο 'association_rules_apriori.csv'")
```

#### Ανάλυση Κανόνων
Μπορείτε να βρείτε όλους τους κανόνες στο αρχείο csv που δημιουργήθηκε. Ακολουθεί σχολιασμός των 10 κανόνων που βρήκα πιο ενδιαφέρον.
- **Κανόνας 1**  
    _IF_ `math_cat_υψηλό` & `gender_female` → _THEN_ `reading_cat_υψηλό`, `writing_cat_υψηλό`, `lunch_standard`  
    **Lift**: 6.82 | **Confidence**: 0.87
    Τα κορίτσια με υψηλή επίδοση στα μαθηματικά είναι πολύ πιθανό να έχουν επίσης υψηλές επιδόσεις στην ανάγνωση και το γράψιμο και να λαμβάνουν κανονικό γεύμα.
- **Κανόνας 2**  
    _IF_ `reading_cat_υψηλό`, `writing_cat_υψηλό`, `lunch_standard` → _THEN_ `math_cat_υψηλό`, `gender_female`  
    **Lift**: 5.54 | **Confidence**: 0.89
    Οι μαθήτριες που έχουν καλές επιδόσεις σε ανάγνωση και γραφή και τρώνε κανονικό γεύμα τείνουν να διαπρέπουν και στα μαθηματικά.
- **Κανόνας 3**  
    _IF_ `writing_cat_υψηλό`, `lunch_standard` → _THEN_ `math_cat_υψηλό`, `reading_cat_υψηλό`, `gender_female`  
    **Lift**: 5.51 | **Confidence**: 0.91
    Η επίδοση στο γράψιμο σε συνδυασμό με καλές διατροφικές συνθήκες συνδέεται με επιτυχία και στους άλλους δύο τομείς.
- **Κανόνας 4**  
    _IF_ `math_cat_υψηλό`, `gender_female` → _THEN_ `writing_cat_υψηλό`, `lunch_standard`  
    **Lift**: 5.12 | **Confidence**: 0.87
    Οι μαθήτριες που αριστεύουν στα μαθηματικά τείνουν να έχουν υψηλή γραπτή ικανότητα και πρόσβαση σε καλύτερη διατροφή.
- **Κανόνας 5**  
    _IF_ `math_cat_υψηλό` → _THEN_ `reading_cat_υψηλό`  
    **Lift**: 3.25 | **Confidence**: 0.82
    Η καλή μαθηματική επίδοση σχετίζεται σημαντικά με την καλή αναγνωστική ικανότητα.
- **Κανόνας 6**  
    _IF_ `reading_cat_υψηλό`, `writing_cat_υψηλό` → _THEN_ `math_cat_υψηλό`  
    **Lift**: 3.07 | **Confidence**: 0.84
    Αν κάποιος είναι καλός σε ανάγνωση και γραφή πιθανότατα είναι καλός και στα μαθηματικά.
- **Κανόνας 7**  
    _IF_ `test preparation course_completed`, `reading_cat_υψηλό` → _THEN_ `writing_cat_υψηλό`  
    **Lift**: 2.78 | **Confidence**: 0.84
    Η ολοκλήρωση του μαθήματος προετοιμασίας ενισχύει την απόδοση στο γράψιμο σε μαθητές με ήδη καλή αναγνωστική ικανότητα.
- **Κανόνας 8**  
    _IF_ `gender_female`, `writing_cat_υψηλό` → _THEN_ `reading_cat_υψηλό`  
    **Lift**: 1.91 | **Confidence**: 0.98
    Οι καλές γραπτές επιδόσεις των κοριτσιών συνοδεύονται από εξίσου καλές ικανότητες στην ανάγνωση.
- **Κανόνας 9**  
    _IF_ `reading_cat_υψηλό`, `gender_female`, `lunch_standard` → _THEN_ `math_cat_υψηλό`, `writing_cat_υψηλό`  
    **Lift**: 4.59 | **Confidence**: 0.89
    Συνδυασμός ανάγνωσης, φύλου και υποστήριξης προβλέπει επιτυχία σε όλα τα μαθήματα.
- **Κανόνας 10**  
    _IF_ `test preparation course_none`, `writing_cat_υψηλό` → _THEN_ `math_cat_υψηλό`, `gender_female`, `lunch_standard`  
    **Lift**: 4.31 | **Confidence**: 0.82
    Παρότι δεν έγινε προετοιμασία, η καλή επίδοση στο γράψιμο συσχετίζεται με υψηλές αποδόσεις και υποστηρικτικό περιβάλλον, ίσως υποδεικνύει μαθητές με φυσική ικανότητα ή εξωτερική βοήθεια.

Οι περισσότεροι κανόνες επιβεβαιώνουν ότι η υψηλή επίδοση σε ένα γνωστικό τομέα συνδέεται θετικά με την επίδοση και στους υπόλοιπους, ιδιαίτερα για τις μαθήτριες. Επιπλέον, παρατηρείται ότι η διατροφική υποστήριξη και η συμμετοχή σε προγράμματα προετοιμασίας παίζουν ρόλο στην επιτυχία τους. 