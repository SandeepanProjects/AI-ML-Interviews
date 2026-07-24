# Implementing Cosine Similarity from Scratch

Cosine similarity is one of the most important algorithms in AI and LLM applications. It measures **how similar two vectors are by comparing their angle rather than their magnitude**.

It is widely used in:

* Semantic Search
* RAG Retrieval
* Vector Databases (FAISS, Pinecone, Qdrant)
* Recommendation Systems
* Document Search
* Embedding Similarity
* Duplicate Detection

---

# Why Not Euclidean Distance?

Suppose we have two embeddings:

```text
A = [1,2,3]

B = [2,4,6]
```

B is just a scaled version of A.

Euclidean distance:

```text
Distance = 3.74
```

Looks far apart.

But semantically they are identical.

Cosine similarity:

```text
1.0
```

Exactly the same direction.

This is why embedding search uses cosine similarity.

---

# Mathematical Formula

The cosine similarity between vectors **A** and **B** is:

[
\text{Cosine Similarity} =
\frac{A \cdot B}
{|A||B|}
]

where:

* (A \cdot B) = dot product
* (|A|) = magnitude of A
* (|B|) = magnitude of B

---

## Geometric Interpretation

genui{"trigonometry_vectors_learning_block":{"type_id":"VECTOR_DOT_PRODUCT"}}

The dot product is:

[
A \cdot B = |A| |B| \cos(\theta)
]

Rearranging gives the cosine similarity formula above. When the angle (\theta) is small, the vectors point in nearly the same direction and similarity is high.

---

# Example

```
A = [1,2]

B = [2,4]
```

Dot Product

```
1×2 + 2×4

= 10
```

Magnitude of A

```
sqrt(1²+2²)

= sqrt(5)

= 2.236
```

Magnitude of B

```
sqrt(2²+4²)

= sqrt(20)

= 4.472
```

Cosine Similarity

```
10 / (2.236 × 4.472)

= 1.0
```

Perfect similarity.

---

# Example 2

```
A = [1,0]

B = [0,1]
```

Dot Product

```
1×0 + 0×1 = 0
```

Cosine

```
0
```

Vectors are orthogonal.

No similarity.

---

# Example 3

```
A = [1,0]

B = [-1,0]
```

Dot Product

```
-1
```

Cosine

```
-1
```

Vectors point in opposite directions.

---

# Cosine Similarity Range

| Value | Meaning                |
| ----- | ---------------------- |
| 1     | Exactly similar        |
| 0.9   | Very similar           |
| 0.7   | Similar                |
| 0.5   | Somewhat similar       |
| 0     | Unrelated (orthogonal) |
| -1    | Opposite direction     |

For many embedding models, values below 0 are uncommon because embeddings are often optimized to cluster semantically similar items.

---

# Implement from Scratch

```python
import math

def cosine_similarity(a, b):
    if len(a) != len(b):
        raise ValueError("Vectors must have same length")

    dot = 0.0
    norm_a = 0.0
    norm_b = 0.0

    for x, y in zip(a, b):
        dot += x * y
        norm_a += x * x
        norm_b += y * y

    norm_a = math.sqrt(norm_a)
    norm_b = math.sqrt(norm_b)

    if norm_a == 0 or norm_b == 0:
        raise ValueError("Zero vector not allowed")

    return dot / (norm_a * norm_b)


A = [1, 2, 3]
B = [2, 4, 6]

print(cosine_similarity(A, B))
```

Output

```
1.0
```

---

# Using NumPy

```python
import numpy as np

A = np.array([1,2,3])
B = np.array([2,4,6])

similarity = np.dot(A, B) / (
    np.linalg.norm(A) *
    np.linalg.norm(B)
)

print(similarity)
```

Output

```
1.0
```

---

# Batch Cosine Similarity

Suppose we have one query embedding and many document embeddings.

```python
import numpy as np

query = np.array([0.1, 0.3, 0.8])

documents = np.array([
    [0.2, 0.1, 0.7],
    [0.8, 0.2, 0.1],
    [0.1, 0.2, 0.9]
])

query_norm = np.linalg.norm(query)

doc_norms = np.linalg.norm(documents, axis=1)

scores = documents @ query / (doc_norms * query_norm)

print(scores)
```

Output

```
[0.965
 0.314
 0.997]
```

The third document is most similar.

---

# Production-Scale Implementation

Imagine a vector database storing document embeddings.

```python
documents = [
    ("doc1", [0.1,0.2,0.7]),
    ("doc2", [0.8,0.1,0.1]),
    ("doc3", [0.2,0.3,0.9]),
    ("doc4", [0.9,0.7,0.3])
]
```

Search function:

```python
import math

def cosine(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    return dot / (norm_a * norm_b)


def search(query, documents, top_k=3):
    results = []

    for doc_id, embedding in documents:
        score = cosine(query, embedding)
        results.append((doc_id, score))

    results.sort(key=lambda x: x[1], reverse=True)

    return results[:top_k]


query = [0.2, 0.2, 0.8]

print(search(query, documents))
```

Example output

```
[
 ('doc3', 0.995),
 ('doc1', 0.981),
 ('doc4', 0.641)
]
```

This is the basic idea behind semantic retrieval in a vector database.

---

# Complexity

For a query vector of dimension **d** and **N** documents:

* Dot product: **O(d)**
* One similarity calculation: **O(d)**
* Comparing against all documents: **O(N × d)**

For millions of vectors, a linear scan is too slow.

---

# How FAISS and Qdrant Make It Fast

Instead of comparing with every vector:

```
Query
   │
   ▼
Index (HNSW / IVF / PQ)
   │
   ▼
Candidate Vectors
   │
   ▼
Cosine Similarity
   │
   ▼
Top-K Results
```

Approximate Nearest Neighbor (ANN) indexes reduce search time dramatically while returning results that are usually very close to the exact nearest neighbors.

---

# Interview Questions

### Why use cosine similarity for embeddings?

Because embeddings encode semantic meaning in their **direction**. Cosine similarity ignores vector magnitude and measures directional alignment.

### Why normalize vectors?

If vectors are normalized to unit length, cosine similarity becomes simply:

[
\text{cosine}(A,B) = A \cdot B
]

This avoids repeated norm calculations and makes retrieval faster.

### Can cosine similarity be greater than 1?

No. It always lies in the range **[-1, 1]**.

### Why do vector databases often normalize embeddings?

After normalization:

* Magnitude = 1
* Cosine similarity = dot product
* Faster computation
* Lower memory overhead during search

This is why many production embedding pipelines normalize vectors before indexing them.
