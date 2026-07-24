# Implement K-Nearest Neighbors (KNN) from Scratch

K-Nearest Neighbors (KNN) is one of the simplest and most intuitive machine learning algorithms. It is a **non-parametric, lazy learning algorithm** that makes predictions by looking at the **K most similar data points**.

KNN is used for:

* Classification
* Regression
* Recommendation Systems
* Image Recognition
* Anomaly Detection
* Semantic Search (conceptually similar to vector search in LLMs)

---

# Intuition

Suppose we have customers with two features:

* Age
* Salary

We want to predict whether a new customer will buy a product.

Training Data

| Age | Salary | Buy |
| --: | -----: | :-: |
|  25 |     40 | Yes |
|  30 |     50 | Yes |
|  45 |     80 |  No |
|  50 |     90 |  No |

New Customer

```text
Age = 28

Salary = 45
```

Instead of training a model,

KNN simply asks:

> Which existing customers are most similar?

---

# Algorithm

```text
New Sample
      │
      ▼
Compute Distance
      │
      ▼
Sort Distances
      │
      ▼
Choose K Nearest
      │
      ▼
Majority Vote
      │
      ▼
Prediction
```

---

# Step 1 — Calculate Distance

The most common distance is **Euclidean Distance**.

Formula

[
d(x,y)=\sqrt{\sum_{i=1}^{n}(x_i-y_i)^2}
]

Example

```text
Point A

(2,3)

Point B

(5,7)
```

Distance

```text
√((5−2)²+(7−3)²)

= √(9+16)

= 5
```

---

# Step 2 — Find Nearest Neighbors

Training Data

| Point | Label | Distance |
| ----- | ----- | -------- |
| A     | Yes   | 1.2      |
| B     | Yes   | 2.1      |
| C     | No    | 4.5      |
| D     | No    | 5.2      |

If

```text
K = 3
```

Nearest neighbors

```text
A

B

C
```

---

# Step 3 — Majority Vote

Votes

```text
Yes

Yes

No
```

Prediction

```text
Yes
```

---

# Implement Euclidean Distance

```python
import math

def euclidean_distance(point1, point2):
    if len(point1) != len(point2):
        raise ValueError("Points must have the same dimensions")

    distance = 0.0

    for x, y in zip(point1, point2):
        distance += (x - y) ** 2

    return math.sqrt(distance)


a = [2, 3]
b = [5, 7]

print(euclidean_distance(a, b))
```

Output

```text
5.0
```

---

# Implement KNN Classifier from Scratch

```python
from collections import Counter
import math

class KNN:

    def __init__(self, k=3):
        self.k = k

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    def _distance(self, p1, p2):
        return math.sqrt(
            sum((x - y) ** 2 for x, y in zip(p1, p2))
        )

    def predict(self, sample):

        distances = []

        for x, label in zip(self.X_train, self.y_train):

            d = self._distance(sample, x)

            distances.append((d, label))

        distances.sort(key=lambda x: x[0])

        neighbors = distances[:self.k]

        labels = [label for _, label in neighbors]

        prediction = Counter(labels).most_common(1)[0][0]

        return prediction
```

---

# Example

```python
X = [

    [25,40],
    [30,50],
    [45,80],
    [50,90]

]

y = [

    "Yes",
    "Yes",
    "No",
    "No"

]

knn = KNN(k=3)

knn.fit(X, y)

prediction = knn.predict([28,45])

print(prediction)
```

Output

```text
Yes
```

---

# Predict Multiple Samples

```python
class KNN:

    def __init__(self, k=3):
        self.k = k

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    def _distance(self, p1, p2):
        return math.sqrt(sum(
            (a-b)**2
            for a,b in zip(p1,p2)
        ))

    def predict(self, X):

        predictions = []

        for sample in X:

            distances = []

            for point,label in zip(
                self.X_train,
                self.y_train
            ):

                d = self._distance(sample, point)

                distances.append((d,label))

            distances.sort()

            labels = [
                label
                for _,label in distances[:self.k]
            ]

            predictions.append(
                Counter(labels).most_common(1)[0][0]
            )

        return predictions
```

---

# Example

```python
test = [

    [28,45],
    [48,85]

]

print(knn.predict(test))
```

Output

```text
['Yes', 'No']
```

---

# KNN Regression

Instead of majority voting,

take the **average**.

Training

| House Size | Price (₹ Lakhs) |
| ---------- | --------------: |
| 1000       |              50 |
| 1200       |              60 |
| 1500       |              75 |

Suppose nearest neighbors are

```text
₹50

₹60

₹75
```

Prediction

```text
Average

=

(50+60+75)/3

=

₹61.67 Lakhs
```

Implementation

```python
prediction = sum(values)/len(values)
```

---

# Using scikit-learn

```python
from sklearn.neighbors import KNeighborsClassifier

X = [

    [25,40],
    [30,50],
    [45,80],
    [50,90]

]

y = [

    "Yes",
    "Yes",
    "No",
    "No"

]

model = KNeighborsClassifier(n_neighbors=3)

model.fit(X,y)

print(model.predict([[28,45]]))
```

Output

```text
['Yes']
```

---

# Time Complexity

Let

* **N** = number of training samples
* **D** = number of features

Prediction requires computing the distance to every training point:

```text
Distance Computation

O(N × D)
```

Sorting all distances:

```text
O(N log N)
```

Overall (naive implementation):

```text
O(N × D + N log N)
```

Using a heap to keep only the top **K** neighbors reduces the selection cost to **O(N log K)**.

---

# Improving Search with KD-Tree

Instead of checking every point:

```text
Dataset
    │
    ▼
KD Tree
    │
    ▼
Nearest Candidates
    │
    ▼
Prediction
```

Complexity (average case)

```text
O(log N)
```

Works well for **low-dimensional** data (typically fewer than 20–30 features).

---

# Ball Tree

For moderately higher-dimensional data, Ball Trees often outperform KD-Trees.

```text
Dataset
     │
     ▼
Ball Tree
     │
     ▼
Nearby Regions
     │
     ▼
Nearest Neighbors
```

---

# KNN for Image Classification

Suppose we have image embeddings.

```text
Cat Image

↓

Embedding

↓

[0.21,0.44,0.81,...]
```

Query Image

↓

Embedding

↓

Compare against all stored embeddings

↓

Nearest Images

↓

Prediction

This is conceptually similar to how many vector search systems retrieve the most similar items.

---

# KNN in LLM Applications

Modern Retrieval-Augmented Generation (RAG) systems use the same nearest-neighbor idea, but with high-dimensional embedding vectors and efficient Approximate Nearest Neighbor (ANN) indexes.

```text
User Question
      │
      ▼
Embedding Model
      │
      ▼
Query Vector
      │
      ▼
Vector Database
(HNSW / IVF / PQ)
      │
      ▼
Top-K Nearest Embeddings
      │
      ▼
Retrieved Documents
      │
      ▼
LLM
```

Instead of comparing against every vector (exact KNN), production systems use ANN algorithms to return very similar neighbors much faster.

---

# Interview Questions

### Why is KNN called a lazy learner?

Because it **does not build a model during training**. It simply stores the training data and performs computation only when a prediction is requested.

---

### Why do we normalize features before KNN?

Distance metrics are sensitive to feature scale.

Example:

```text
Age = 25

Salary = 800000
```

Without normalization, the salary feature dominates the Euclidean distance. Standardization or Min-Max scaling ensures each feature contributes appropriately.

---

### What happens if K is too small?

* Sensitive to noise
* High variance
* May overfit

---

### What happens if K is too large?

* Smooths decision boundaries
* High bias
* May underfit

---

### Why is KNN slow on large datasets?

Every prediction requires comparing the query point with many (or all) training samples. For millions of vectors, exact search becomes too expensive, so production systems use KD-Trees, Ball Trees, or ANN indexes such as HNSW, IVF, or PQ for fast nearest-neighbor search.
