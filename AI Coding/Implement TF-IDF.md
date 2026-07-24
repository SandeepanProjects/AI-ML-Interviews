# Implement TF-IDF from Scratch

TF-IDF (**Term Frequency – Inverse Document Frequency**) is a statistical technique used to measure how important a word is in a document relative to an entire collection (corpus).

It is one of the foundations of:

* Search Engines
* Information Retrieval
* RAG (classical retrieval)
* Keyword Search
* Document Ranking
* Text Similarity
* Recommendation Systems

Although modern LLM applications often use embeddings, TF-IDF is still widely used as:

* A baseline retrieval model
* A feature for machine learning
* Part of hybrid search (BM25 + embeddings)

---

# Why Do We Need TF-IDF?

Suppose we have three documents:

```text
Doc1:
I love machine learning

Doc2:
Machine learning is amazing

Doc3:
I love football
```

Query:

```text
machine learning
```

Both words appear only in documents about machine learning.

Now consider the word:

```text
I
```

It appears almost everywhere.

Should it have the same importance?

No.

Common words should contribute less.

Rare words should contribute more.

TF-IDF achieves exactly this.

---

# Components of TF-IDF

```text
TF-IDF

=

Term Frequency (TF)

×

Inverse Document Frequency (IDF)
```

---

# Step 1 — Term Frequency (TF)

TF measures how often a word appears in a document.

Formula:

[
TF(t,d)=\frac{\text{Count of term }t\text{ in document }d}
{\text{Total words in document}}
]

Example

Document

```text
cat cat dog fish
```

Word counts

| Word | Count |
| ---- | ----: |
| cat  |     2 |
| dog  |     1 |
| fish |     1 |

Total words

```text
4
```

Therefore

```text
TF(cat)

=

2 / 4

=

0.5
```

---

# Step 2 — Inverse Document Frequency (IDF)

IDF measures how rare a word is across all documents.

Formula

[
IDF(t)=\log\left(\frac{N}{df(t)}\right)
]

Where

* **N** = total documents
* **df(t)** = number of documents containing the term

---

Example

Suppose

```text
Total documents = 1000
```

Word

```text
machine
```

appears in

```text
20 documents
```

Then

```text
IDF

=

log(1000/20)

=

log(50)

≈3.91
```

A large IDF means the term is rare and therefore informative.

---

Suppose

```text
the
```

appears in every document.

```text
IDF

=

log(1000/1000)

=

0
```

So "the" contributes almost nothing.

---

# Final TF-IDF Score

[
TFIDF = TF \times IDF
]

Example

```text
TF = 0.5

IDF = 3.9
```

```text
TF-IDF

=

1.95
```

High score means:

* Frequent in this document
* Rare across the corpus

---

# Implement from Scratch

```python
import math
from collections import Counter


def compute_tf(document):
    words = document.lower().split()
    total = len(words)
    counts = Counter(words)

    tf = {}

    for word, count in counts.items():
        tf[word] = count / total

    return tf


def compute_idf(documents):
    N = len(documents)

    df = Counter()

    for doc in documents:
        words = set(doc.lower().split())

        for word in words:
            df[word] += 1

    idf = {}

    for word, freq in df.items():
        idf[word] = math.log(N / freq)

    return idf


def compute_tfidf(document, idf):

    tf = compute_tf(document)

    tfidf = {}

    for word, tf_value in tf.items():
        tfidf[word] = tf_value * idf[word]

    return tfidf
```

---

# Example

```python
documents = [

    "I love machine learning",

    "Machine learning is amazing",

    "I love football"

]

idf = compute_idf(documents)

for doc in documents:

    print(compute_tfidf(doc, idf))
```

Example output (approximate)

```text
Doc1

{
'I':0.10,

'love':0.10,

'machine':0.10,

'learning':0.10
}

Doc3

{
'football':0.27
}
```

Notice that **football** receives a higher score because it appears in only one document.

---

# Build a TF-IDF Matrix

Each row represents a document.

Each column represents a word.

```text
            cat   dog   fish

Doc1        0.3   0.1   0

Doc2        0     0.5   0.2

Doc3        0.1   0     0.6
```

Implementation:

```python
def tfidf_matrix(documents):

    idf = compute_idf(documents)

    vocabulary = sorted(idf.keys())

    matrix = []

    for doc in documents:

        tfidf = compute_tfidf(doc, idf)

        row = []

        for word in vocabulary:
            row.append(tfidf.get(word, 0))

        matrix.append(row)

    return vocabulary, matrix
```

---

# Search Using TF-IDF

Suppose

```text
Doc1

Machine Learning Guide

Doc2

Football News

Doc3

Deep Learning
```

Query

```text
learning
```

Algorithm

```text
Query

↓

TF-IDF Vector

↓

Cosine Similarity

↓

Top Documents
```

Implementation

```python
def cosine_similarity(a, b):
    dot = sum(x * y for x, y in zip(a, b))

    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))

    if norm_a == 0 or norm_b == 0:
        return 0.0

    return dot / (norm_a * norm_b)
```

---

# Scikit-learn Implementation

In production, you usually use `TfidfVectorizer` instead of writing it from scratch.

```python
from sklearn.feature_extraction.text import TfidfVectorizer

documents = [

    "I love machine learning",

    "Machine learning is amazing",

    "I love football"

]

vectorizer = TfidfVectorizer()

matrix = vectorizer.fit_transform(documents)

print(vectorizer.get_feature_names_out())

print(matrix.toarray())
```

Example output

```text
Features

['amazing',
 'football',
 'learning',
 'love',
 'machine']
```

Matrix

```text
[[0.00 0.00 0.57 0.57 0.57]

 [0.65 0.00 0.39 0.00 0.65]

 [0.00 0.79 0.00 0.61 0.00]]
```

---

# Production Search Pipeline

```text
User Query
      │
      ▼
Text Preprocessing
      │
      ▼
TF-IDF Vectorizer
      │
      ▼
Sparse Query Vector
      │
      ▼
Cosine Similarity
      │
      ▼
Rank Documents
      │
      ▼
Top-K Results
```

This is the basis of many classical search systems.

---

# Time Complexity

Let:

* **N** = number of documents
* **V** = vocabulary size
* **L** = average document length

Building IDF:

```text
O(N × L)
```

Building TF:

```text
O(L)
```

TF-IDF Matrix:

```text
O(N × V)
```

Cosine Similarity:

```text
O(V)
```

Because most TF-IDF vectors are sparse, practical implementations store them in sparse matrix formats (such as CSR) to reduce memory usage and speed up computations.

---

# TF-IDF vs Embeddings

| Feature                                 | TF-IDF                          | Embeddings                     |
| --------------------------------------- | ------------------------------- | ------------------------------ |
| Matches exact words                     | ✅                               | ❌ (not required)               |
| Captures semantic meaning               | ❌                               | ✅                              |
| Handles synonyms                        | ❌                               | ✅                              |
| Fast on small datasets                  | ✅                               | ✅                              |
| Dense vectors                           | ❌ (sparse)                      | ✅                              |
| Memory efficient for large vocabularies | Often yes (with sparse storage) | Depends on embedding dimension |
| Best use case                           | Keyword search                  | Semantic search                |

Example:

```text
Query:
car
```

Document:

```text
automobile
```

TF-IDF similarity:

```text
0
```

Embedding similarity:

```text
0.93
```

because the model has learned that *car* and *automobile* are semantically related.

---

# TF-IDF in Modern RAG Systems

A common production architecture uses both lexical and semantic retrieval:

```text
User Query
      │
      ├──────────────┐
      ▼              ▼
 TF-IDF/BM25     Embeddings
      │              │
      ▼              ▼
 Keyword Hits   Semantic Hits
      └──────┬───────┘
             ▼
      Hybrid Ranking
             ▼
        Top-K Documents
             ▼
            LLM
```

This hybrid approach combines the precision of exact keyword matching with the semantic understanding of embeddings, making it a popular choice for enterprise search and RAG applications.
