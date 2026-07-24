# Implement Hybrid Search (BM25 + Embedding Search)

Hybrid search combines **lexical search** (BM25/TF-IDF) with **semantic search** (embeddings).

It is the retrieval technique used by many production RAG systems because it combines:

* Exact keyword matching
* Semantic understanding
* Better recall
* Better precision

Companies like Microsoft, OpenAI, Elastic, and many enterprise search platforms use hybrid retrieval.

---

# Why Hybrid Search?

Suppose the user asks:

```text
How do I reset my password?
```

Documents:

```text
Doc1:
Password reset instructions

Doc2:
How to change your password

Doc3:
Vacation policy
```

### BM25 Search

Matches keywords.

```text
reset password

↓

Doc1
```

Problem:

If a document says

```text
change password
```

BM25 may rank it lower.

---

### Embedding Search

Embedding model understands meaning.

```text
reset password

≈

change password

≈

recover account
```

Embedding search retrieves both.

---

# Hybrid Search

```text
                    Query
                      │
        ┌─────────────┴─────────────┐
        ▼                           ▼
     BM25 Search             Embedding Search
        │                           │
        ▼                           ▼
  Keyword Results          Semantic Results
        └─────────────┬─────────────┘
                      ▼
               Score Fusion
                      ▼
               Top-K Documents
```

This provides both lexical precision and semantic recall.

---

# Production Architecture

```text
User Query
      │
      ▼
Preprocessing
      │
      ▼
Query
 ┌───────────────┐
 ▼               ▼
BM25         Embedding
 ▼               ▼
Top 50       Top 50
 └──────┬────────┘
        ▼
 Score Fusion
        ▼
 Cross Encoder
        ▼
 Top 5
        ▼
   LLM
```

---

# Step 1: Install

```bash
pip install sentence-transformers rank-bm25 numpy
```

---

# Step 2: Documents

```python
documents = [
    "Python supports decorators and generators.",
    "Transformers use self attention.",
    "Neural networks learn hierarchical features.",
    "Password reset instructions for enterprise users.",
    "How to change your account password."
]
```

---

# Step 3: Build BM25 Index

```python
from rank_bm25 import BM25Okapi

tokenized_docs = [
    doc.lower().split()
    for doc in documents
]

bm25 = BM25Okapi(tokenized_docs)
```

Search:

```python
query = "reset password"

bm25_scores = bm25.get_scores(
    query.lower().split()
)

print(bm25_scores)
```

Example output

```text
[0.0, 0.0, 0.0, 5.82, 2.94]
```

---

# Step 4: Build Embedding Index

```python
from sentence_transformers import SentenceTransformer

import numpy as np

model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)

embeddings = model.encode(
    documents,
    normalize_embeddings=True
)
```

---

# Step 5: Semantic Search

```python
query_embedding = model.encode(
    "reset password",
    normalize_embeddings=True
)

embedding_scores = (
    embeddings @ query_embedding
)

print(embedding_scores)
```

Example

```text
[0.31
 0.28
 0.18
 0.91
 0.88]
```

---

# Step 6: Normalize Scores

BM25 and cosine similarity use different ranges, so normalize them before combining.

```python
import numpy as np

def normalize(scores):

    scores = np.array(scores)

    return (
        scores - scores.min()
    ) / (
        scores.max() - scores.min() + 1e-8
    )
```

---

# Step 7: Score Fusion

A simple weighted sum works well.

```python
bm25_norm = normalize(
    bm25_scores
)

embedding_norm = normalize(
    embedding_scores
)

alpha = 0.4

hybrid_scores = (
    alpha * bm25_norm
    +
    (1 - alpha) * embedding_norm
)
```

Sort results.

```python
top = np.argsort(
    hybrid_scores
)[::-1]

for i in top:

    print(
        hybrid_scores[i],
        documents[i]
    )
```

Example output

```text
0.99 Password reset instructions...
0.96 How to change password...
0.12 Python decorators...
```

---

# Complete Hybrid Retriever

```python
import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

class HybridRetriever:

    def __init__(self, documents):

        self.documents = documents

        self.model = SentenceTransformer(
            "all-MiniLM-L6-v2"
        )

        self.tokenized = [
            d.lower().split()
            for d in documents
        ]

        self.bm25 = BM25Okapi(
            self.tokenized
        )

        self.embeddings = self.model.encode(
            documents,
            normalize_embeddings=True
        )

    def _normalize(self, scores):

        scores = np.array(scores)

        return (
            scores - scores.min()
        ) / (
            scores.max() - scores.min() + 1e-8
        )

    def search(
        self,
        query,
        top_k=5,
        alpha=0.4
    ):

        bm25 = self.bm25.get_scores(
            query.lower().split()
        )

        query_embedding = self.model.encode(
            query,
            normalize_embeddings=True
        )

        semantic = (
            self.embeddings @ query_embedding
        )

        bm25 = self._normalize(bm25)
        semantic = self._normalize(semantic)

        scores = (
            alpha * bm25
            +
            (1 - alpha) * semantic
        )

        top = np.argsort(scores)[::-1][:top_k]

        return [
            (scores[i], self.documents[i])
            for i in top
        ]
```

Example

```python
retriever = HybridRetriever(documents)

results = retriever.search(
    "reset my password"
)

for score, doc in results:
    print(score, doc)
```

---

# Reciprocal Rank Fusion (RRF)

Instead of combining raw scores, many production systems combine **ranks**.

Formula:

[
\text{RRF}(d)=\sum_i \frac{1}{k+r_i(d)}
]

Where:

* (r_i(d)) = rank of document *d* in retrieval method *i*
* (k) = constant (commonly 60)

Implementation:

```python
def reciprocal_rank_fusion(
    rankings,
    k=60
):
    scores = {}

    for ranking in rankings:

        for rank, doc in enumerate(ranking):

            scores.setdefault(doc, 0)

            scores[doc] += (
                1 / (k + rank + 1)
            )

    return sorted(
        scores.items(),
        key=lambda x: x[1],
        reverse=True
    )
```

Why RRF?

* Doesn't depend on score scales.
* Robust across different retrievers.
* Widely used in production search systems.

---

# Add Metadata Filtering

```python
results = [

    doc

    for doc in results

    if doc.metadata["department"] == "HR"

]
```

---

# Add a Cross-Encoder Reranker

Hybrid search returns candidates.

The reranker improves ranking.

```text
Query
   │
   ▼
Hybrid Search
   │
   ▼
Top 50 Documents
   │
   ▼
Cross Encoder
   │
   ▼
Top 5 Documents
   │
   ▼
LLM
```

Example:

```python
reranked = reranker.rank(
    query,
    retrieved_docs
)
```

---

# Production Architecture

```text
                  User Query
                       │
          ┌────────────┴────────────┐
          ▼                         ▼
     BM25 Retriever         Embedding Retriever
          │                         │
          ▼                         ▼
      Top 100                  Top 100
          └────────────┬────────────┘
                       ▼
            Reciprocal Rank Fusion
                       ▼
              Metadata Filtering
                       ▼
            Cross-Encoder Reranker
                       ▼
                  Top 5 Chunks
                       ▼
                 Prompt Builder
                       ▼
                      LLM
```

---

# Production Optimizations

### 1. Cache Query Embeddings

```python
from functools import lru_cache

@lru_cache(maxsize=1000)
def embed(query):
    return model.encode(
        query,
        normalize_embeddings=True
    )
```

---

### 2. Batch Embedding

```python
embeddings = model.encode(
    documents,
    batch_size=128,
    normalize_embeddings=True
)
```

---

### 3. Use ANN Search

Instead of comparing against every vector:

```text
Query
   │
   ▼
HNSW Index
   │
   ▼
Top 100
```

Common ANN algorithms:

* HNSW
* IVF
* IVF-PQ
* ScaNN

---

### 4. Parallel Retrieval

Run BM25 and embedding search concurrently.

```python
from concurrent.futures import ThreadPoolExecutor

with ThreadPoolExecutor() as executor:
    bm25_future = executor.submit(
        bm25.get_scores,
        query.lower().split()
    )

    embedding_future = executor.submit(
        semantic_search,
        query
    )

    bm25_scores = bm25_future.result()
    embedding_scores = embedding_future.result()
```

This reduces end-to-end latency.

---

# Folder Structure

```text
retrieval/
│
├── hybrid_retriever.py
├── bm25_retriever.py
├── semantic_retriever.py
├── reranker.py
├── embedding_model.py
├── vector_store.py
├── metadata_filter.py
├── cache.py
├── metrics.py
└── tests/
```

---

# Interview Questions

### Why is hybrid search better than embedding search alone?

Embedding search captures semantic meaning but can miss exact identifiers such as product IDs, error codes, or rare keywords. BM25 excels at exact lexical matching. Hybrid search combines both strengths.

---

### Why normalize scores?

BM25 scores and cosine similarity are on different numeric scales. Normalization or rank-based fusion ensures one retriever doesn't dominate simply because of its score range.

---

### Why is Reciprocal Rank Fusion preferred in production?

RRF combines **ranks** instead of raw scores, making it robust across heterogeneous retrievers without requiring score calibration. It's simple, effective, and widely adopted.

---

### How would you build an enterprise-grade hybrid retriever?

A typical production pipeline includes:

* BM25 (e.g., Elasticsearch/OpenSearch)
* Embedding retrieval (FAISS, Qdrant, Milvus, Pinecone)
* Parallel execution of both retrievers
* Reciprocal Rank Fusion (RRF)
* Metadata and access-control filtering (tenant, department, RBAC)
* Cross-encoder reranking of the top candidates
* Retrieval observability (latency, Recall@K, Precision@K, NDCG)
* Embedding caching and ANN indexes for scalability

This architecture provides high recall, strong precision, and low latency for large-scale RAG systems.
