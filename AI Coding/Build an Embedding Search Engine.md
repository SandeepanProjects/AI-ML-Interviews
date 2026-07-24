# Build an Embedding Search Engine from Scratch (Production-Style)

Embedding search is the foundation of modern AI applications like:

* ChatGPT Retrieval (RAG)
* GitHub Copilot
* Enterprise Search
* Customer Support Bots
* Semantic Search
* Recommendation Systems

Instead of matching **keywords**, embedding search matches **meaning**.

---

# Traditional Search vs Embedding Search

Suppose the query is:

```text
"How do I reset my password?"
```

Documents:

```text
Doc1:
How to change your password

Doc2:
Vacation policy

Doc3:
Password reset instructions
```

### Keyword Search

Matches exact words.

```text
Query
     │
     ▼
"reset"

↓

Doc3
```

If the document says **change password** instead of **reset password**, keyword search may miss it.

---

### Embedding Search

```text
Query
      │
      ▼
Embedding Model
      │
      ▼
Dense Vector
      │
      ▼
Nearest Neighbor Search
      │
      ▼
Most Similar Documents
```

Even if the wording differs:

```text
reset password

≈

change password

≈

recover account
```

Embedding search understands semantic similarity.

---

# End-to-End Architecture

```text
Documents
     │
     ▼
Chunking
     │
     ▼
Embedding Model
     │
     ▼
Embedding Vectors
     │
     ▼
Vector Index
     │
     ▼
Query
     │
     ▼
Query Embedding
     │
     ▼
Cosine Similarity
     │
     ▼
Top-K Documents
```

---

# Step 1 — Install

```bash
pip install sentence-transformers numpy
```

---

# Step 2 — Sample Documents

```python
documents = [
    "Python is a programming language.",
    "Machine learning is a subset of artificial intelligence.",
    "Transformers power modern large language models.",
    "Paris is the capital of France.",
    "Neural networks learn from data."
]
```

---

# Step 3 — Generate Embeddings

```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")

embeddings = model.encode(
    documents,
    normalize_embeddings=True
)
```

Shape:

```python
print(embeddings.shape)
```

Output

```text
(5, 384)
```

Each document becomes a **384-dimensional vector**.

---

# Step 4 — Cosine Similarity

Implement from scratch.

```python
import numpy as np

def cosine_similarity(a, b):
    return np.dot(a, b) / (
        np.linalg.norm(a) *
        np.linalg.norm(b)
    )
```

---

# Step 5 — Build Search

```python
def search(query, top_k=3):

    query_embedding = model.encode(
        query,
        normalize_embeddings=True
    )

    scores = []

    for i, doc_embedding in enumerate(embeddings):

        score = cosine_similarity(
            query_embedding,
            doc_embedding
        )

        scores.append((score, documents[i]))

    scores.sort(reverse=True)

    return scores[:top_k]
```

---

# Step 6 — Search

```python
results = search(
    "How do neural networks work?"
)

for score, doc in results:
    print(score, doc)
```

Example output

```text
0.93  Neural networks learn from data.
0.79  Machine learning is a subset...
0.62  Transformers power modern LLMs.
```

Notice:

The query never used the exact sentence.

Semantic similarity found the correct document.

---

# Build Your Own Embedding Index

Instead of a vector database,

store vectors yourself.

```python
class VectorIndex:

    def __init__(self):
        self.documents = []
        self.embeddings = []

    def add(self, doc, embedding):
        self.documents.append(doc)
        self.embeddings.append(embedding)
```

Insert documents.

```python
index = VectorIndex()

for doc in documents:
    emb = model.encode(
        doc,
        normalize_embeddings=True
    )
    index.add(doc, emb)
```

---

# Search the Index

```python
class VectorIndex:

    def __init__(self):
        self.documents = []
        self.embeddings = []

    def add(self, doc, embedding):
        self.documents.append(doc)
        self.embeddings.append(embedding)

    def search(self, query, model, top_k=3):

        query_embedding = model.encode(
            query,
            normalize_embeddings=True
        )

        similarities = []

        for doc, emb in zip(
            self.documents,
            self.embeddings
        ):

            score = np.dot(
                query_embedding,
                emb
            )

            similarities.append((score, doc))

        similarities.sort(reverse=True)

        return similarities[:top_k]
```

Because the embeddings are normalized, the dot product equals cosine similarity.

---

# Batch Search (Vectorized)

Avoid Python loops.

```python
query_embedding = model.encode(
    "deep learning",
    normalize_embeddings=True
)

scores = embeddings @ query_embedding

top = np.argsort(scores)[::-1][:5]

for idx in top:
    print(scores[idx], documents[idx])
```

This uses optimized linear algebra and is much faster.

---

# Scaling to Millions of Documents

The naive implementation compares the query with **every document**.

```
Documents

1,000,000

↓

Cosine Similarity

↓

1,000,000 comparisons
```

Complexity

```text
O(N × D)
```

where

* **N** = documents
* **D** = embedding dimension

This becomes too slow.

---

# Production Architecture

```text
PDFs
      │
      ▼
Chunking
      │
      ▼
Embedding Model
      │
      ▼
Vector Database
(Qdrant / Pinecone / FAISS)
      │
      ▼
ANN Index
(HNSW / IVF / PQ)
      │
      ▼
Search API
      │
      ▼
Top-K Chunks
      │
      ▼
LLM
```

Instead of scanning every vector, Approximate Nearest Neighbor (ANN) indexes quickly narrow down the search space.

---

# Example Using FAISS

```python
import faiss
import numpy as np

# embeddings: shape (N, D), dtype float32
vectors = embeddings.astype("float32")

dimension = vectors.shape[1]

index = faiss.IndexFlatIP(dimension)

# Embeddings are normalized, so inner product = cosine similarity
index.add(vectors)

query = model.encode(
    ["deep learning"],
    normalize_embeddings=True
).astype("float32")

scores, indices = index.search(query, k=3)

for score, idx in zip(scores[0], indices[0]):
    print(score, documents[idx])
```

For larger datasets, replace `IndexFlatIP` with indexes such as `IndexHNSWFlat` or `IndexIVFPQ` to achieve much faster retrieval with a small trade-off in accuracy.

---

# Integrating into a RAG Pipeline

```python
def rag_search(query, top_k=3):
    results = index.search(query, model, top_k)

    context = "\n".join(
        doc for _, doc in results
    )

    prompt = f"""
Answer the question using only the context.

Context:
{context}

Question:
{query}
"""

    return prompt
```

The retrieved context is then sent to the LLM.

---

# Production Folder Structure

```text
embedding_search/
│
├── app.py
├── config.py
├── embedding_model.py
├── vector_index.py
├── search_service.py
├── retriever.py
├── reranker.py
├── chunker.py
├── document_loader.py
├── api.py
├── tests/
└── requirements.txt
```

---

# Optimizations Used by Senior AI Engineers

## 1. Normalize Embeddings Once

```python
embeddings = model.encode(
    documents,
    normalize_embeddings=True
)
```

Then cosine similarity becomes a simple dot product.

---

## 2. Batch Encoding

Instead of:

```python
for doc in documents:
    model.encode(doc)
```

Use:

```python
embeddings = model.encode(
    documents,
    batch_size=128
)
```

This significantly improves throughput.

---

## 3. Cache Query Embeddings

Frequently repeated queries should reuse cached embeddings.

```python
from functools import lru_cache

@lru_cache(maxsize=1000)
def get_query_embedding(query):
    return model.encode(
        query,
        normalize_embeddings=True
    )
```

---

## 4. Metadata Filtering

Store metadata alongside vectors.

```python
{
    "document": "...",
    "department": "HR",
    "language": "en",
    "created_at": "2026-01-10"
}
```

Filter before similarity search to reduce the candidate set.

---

## 5. Reranking

A common production pipeline is:

```text
User Query
      │
      ▼
Embedding Search
      │
      ▼
Top 50 Documents
      │
      ▼
Cross-Encoder Reranker
      │
      ▼
Top 5 Documents
      │
      ▼
LLM
```

The embedding model provides high recall, while the reranker improves precision.

---

# Complexity

| Stage                 | Complexity                               |
| --------------------- | ---------------------------------------- |
| Embedding generation  | O(N × D)                                 |
| Exact cosine search   | O(N × D)                                 |
| ANN search (HNSW/IVF) | Approximately sublinear in N             |
| Reranking (top K)     | O(K) forward passes through the reranker |

---

# Senior AI Engineer Interview Questions

### Why normalize embeddings?

Normalization makes every vector have unit length, so:

[
\text{cosine}(a,b)=a \cdot b
]

This eliminates repeated norm calculations and speeds up search.

---

### Why use FAISS or Qdrant instead of NumPy?

NumPy performs an **exact linear scan** over all vectors. FAISS and Qdrant use Approximate Nearest Neighbor indexes (such as HNSW or IVF) to search millions of vectors efficiently with much lower latency.

---

### Why rerank after embedding search?

Embedding models are optimized for **recall** (finding relevant candidates). Cross-encoder rerankers jointly score the query and each candidate, producing more accurate rankings before passing the final context to the LLM.

---

### How would you scale to 100 million documents?

A production design typically includes:

* Chunking and preprocessing pipeline
* Batch embedding generation
* Distributed vector database (e.g., FAISS shards or Qdrant cluster)
* ANN indexes (HNSW, IVF-PQ)
* Metadata filtering
* Redis caching for frequent queries
* Cross-encoder reranking
* Observability for latency, recall, token usage, and cost
* Horizontal scaling behind an API layer

This architecture is representative of production semantic search and Retrieval-Augmented Generation systems.
