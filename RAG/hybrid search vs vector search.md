In modern RAG systems, there are **three major retrieval methods**:

1. **Keyword Search (Lexical Search)**
2. **Vector Search (Semantic Search)**
3. **Hybrid Search (Keyword + Vector)** ← Used by most production AI systems.

Large production systems like Microsoft Copilot, OpenAI Assistants, Notion AI, Perplexity, and enterprise RAG platforms rarely rely on vector search alone. They typically combine lexical and semantic retrieval, often followed by reranking.

---

# Why Search Matters

Suppose your vector database contains these chunks.

```text
Chunk 1:
Python supports decorators.

Chunk 2:
FastAPI dependency injection tutorial.

Chunk 3:
Neural networks are inspired by the human brain.

Chunk 4:
JWT Authentication using FastAPI.

Chunk 5:
OAuth2 Authorization Flow.
```

User asks:

> Explain FastAPI authentication

How should we retrieve the best chunk?

Different search methods produce different results.

---

# 1. Keyword Search (Lexical Search)

Traditional search engines work like this.

```
Query

↓

Tokenize

↓

Count matching words

↓

Rank documents
```

For the query

```
FastAPI authentication
```

it searches for documents containing:

```
FastAPI
authentication
```

Example:

```python
documents = [
    "Python supports decorators.",
    "FastAPI dependency injection.",
    "JWT Authentication using FastAPI.",
    "Neural networks."
]

query = "FastAPI authentication"

query_words = set(query.lower().split())

scores = []

for doc in documents:
    words = set(doc.lower().split())
    score = len(query_words & words)
    scores.append(score)

print(scores)
```

Output

```text
[0, 1, 2, 0]
```

The highest score is selected.

---

## Problems with Keyword Search

Suppose the user asks

```
How do I secure APIs?
```

Document

```
JWT Authentication using FastAPI
```

No common words.

Keyword search says

```
Similarity = 0
```

But both discuss API security.

Keyword search cannot understand meaning.

---

# Vector Search (Semantic Search)

Instead of matching words,

we compare meanings.

Pipeline

```text
Query

↓

Embedding Model

↓

Vector

↓

Compare against vectors in database

↓

Most similar vectors
```

Example

```text
Document

JWT Authentication

↓

Embedding

[0.13, -0.54, 0.72, ...]

↓

Store
```

Later

```
How to secure APIs?

↓

Embedding

↓

Compare vectors
```

Even though the words differ,

their vectors are close.

---

# Example

Document

```
JWT Authentication using FastAPI
```

User Query

```
How do I secure REST APIs?
```

Keyword search

```
No word overlap
```

Vector search

```
API Security

↓

Authentication

↓

JWT

↓

High similarity
```

This is semantic retrieval.

---

# How Embeddings Look

Suppose our embedding model generates

```python
JWT Authentication

↓

[0.18, 0.92, -0.31, 0.77]
```

Query

```
Secure APIs
```

↓

```python
[0.20, 0.88, -0.29, 0.75]
```

These vectors are very close.

---

# Computing Similarity

Most vector databases use **cosine similarity**.

```python
import numpy as np

doc = np.array([0.18,0.92,-0.31,0.77])

query = np.array([0.20,0.88,-0.29,0.75])

similarity = np.dot(doc, query) / (
    np.linalg.norm(doc) * np.linalg.norm(query)
)

print(similarity)
```

Output

```text
0.996
```

Almost identical meaning.

---

# Vector Search Example

Let's build a tiny semantic search engine.

```python
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

model = SentenceTransformer("all-MiniLM-L6-v2")

documents = [
    "Python decorators",
    "JWT Authentication using FastAPI",
    "Neural Networks",
    "Docker Compose tutorial"
]

doc_embeddings = model.encode(documents)

query = "How to secure APIs?"

query_embedding = model.encode([query])

scores = cosine_similarity(
    query_embedding,
    doc_embeddings
)

print(scores)
```

Output

```text
JWT Authentication using FastAPI

Score = 0.91
```

Notice

```
Secure APIs

≠

JWT Authentication
```

Different words.

Same meaning.

---

# How Vector Databases Work

```text
               Documents

                     │

                     ▼

            Embedding Model

                     │

                     ▼

     [0.23,0.71,-0.18,...]

                     │

                     ▼

      Qdrant / Pinecone / FAISS

                     │

          Approximate Nearest Neighbor

                     │

                     ▼

         Top K Most Similar Chunks
```

---

# Problems with Vector Search

Vector search is amazing,

but it has weaknesses.

Example

Document

```
GPT-4.1 costs $2 per million tokens
```

Query

```
GPT-4.1
```

Embedding model may focus on

```
LLM
OpenAI
Language Model
```

instead of

```
GPT-4.1
```

Sometimes another LLM document is returned.

---

Another example

Query

```
Error 500
```

Embedding may retrieve

```
Error handling
```

instead of

```
HTTP Error 500
```

Exact numbers are difficult.

---

# Where Vector Search Fails

It struggles with

```
Invoice #89321

API Version v2.3.7

Product Code

UUID

Customer IDs

Email addresses

Dates

Serial numbers
```

These require exact matching.

---

# Hybrid Search

Production systems solve this problem.

Instead of choosing

```
Keyword

OR

Vector
```

they use both.

Pipeline

```text
                Query

                   │

        ┌──────────┴──────────┐

        ▼                     ▼

 Keyword Search         Vector Search

        ▼                     ▼

  Candidate Docs      Candidate Docs

        └──────────┬──────────┘

                   ▼

              Merge Results

                   ▼

             Optional Reranker

                   ▼

               Final Top K
```

This is Hybrid Search.

---

# Example

Documents

```
1. GPT-4.1 API pricing

2. FastAPI Authentication

3. Docker tutorial

4. Neural Networks
```

Query

```
GPT-4.1 pricing
```

Keyword search

```
Matches GPT-4.1
```

Vector search

```
Pricing
Language model
API cost
```

Hybrid combines both.

---

# Example Score

Suppose

```
Keyword Score

0.82
```

Vector Score

```
0.91
```

Hybrid

```
0.4 × Keyword

+

0.6 × Vector

=

0.874
```

This document ranks highest.

---

# Hybrid Search Code (Simple)

```python
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

model = SentenceTransformer("all-MiniLM-L6-v2")

documents = [
    "FastAPI JWT Authentication",
    "Docker Compose",
    "Neural Networks"
]

query = "How to secure APIs?"

# Vector scores
doc_vectors = model.encode(documents)
query_vector = model.encode([query])

vector_scores = cosine_similarity(
    query_vector,
    doc_vectors
)[0]

# Keyword scores
query_words = set(query.lower().split())

keyword_scores = []

for doc in documents:
    words = set(doc.lower().split())
    keyword_scores.append(
        len(query_words & words)
    )

# Normalize keyword scores
max_score = max(keyword_scores) or 1
keyword_scores = [s / max_score for s in keyword_scores]

# Hybrid
hybrid_scores = (
    0.7 * vector_scores +
    0.3 * keyword_scores
)

for doc, score in zip(documents, hybrid_scores):
    print(doc, score)
```

Output (example)

```text
FastAPI JWT Authentication   0.92
Docker Compose               0.18
Neural Networks              0.12
```

---

# Real Production Hybrid Search

Most vector databases support hybrid retrieval.

Example pipeline

```text
PDF

↓

Chunking

↓

Embeddings

↓

Qdrant

↓

Query

↓

Dense Search

+

BM25 Search

↓

Fusion

↓

Cross Encoder Reranker

↓

LLM
```

---

# What is BM25?

BM25 is the most widely used keyword-ranking algorithm in search engines.

Instead of simply counting matching words, it considers:

* How many query terms appear in the document.
* How rare those terms are across the entire collection (IDF).
* Whether the document is very long (length normalization).

Example:

```
Query:
FastAPI authentication
```

Document A:

```
FastAPI JWT Authentication Guide
```

Document B:

```
Authentication Basics
```

BM25 ranks Document A higher because it matches more important query terms and more closely aligns with the query.

---

# Production Architecture

```text
                    User Query
                         │
          ┌──────────────┴──────────────┐
          ▼                             ▼
     Dense Embedding              BM25 Search
          ▼                             ▼
      Vector DB                  Inverted Index
          ▼                             ▼
          └──────────────┬──────────────┘
                         ▼
                  Reciprocal Rank Fusion
                         ▼
               Cross Encoder Reranker
                         ▼
                    Top 5 Chunks
                         ▼
                        LLM
```

---

# Difference Between Vector Search and Hybrid Search

| Feature                             | Vector Search   | Hybrid Search                        |
| ----------------------------------- | --------------- | ------------------------------------ |
| Uses embeddings                     | ✅               | ✅                                    |
| Uses keyword matching               | ❌               | ✅                                    |
| Understands semantic meaning        | ✅               | ✅                                    |
| Finds synonyms                      | ✅               | ✅                                    |
| Finds exact product names           | ❌ Sometimes     | ✅                                    |
| Handles IDs, versions, codes        | ❌ Weak          | ✅ Strong                             |
| Handles natural-language questions  | ✅ Excellent     | ✅ Excellent                          |
| Works well for enterprise documents | Good            | Excellent                            |
| Typical retrieval pipeline          | Embedding → ANN | BM25 + Embedding → Fusion → Reranker |

---

# When to Use Which

| Scenario                                                         | Recommended Search |
| ---------------------------------------------------------------- | ------------------ |
| Semantic Q&A over documents                                      | Vector Search      |
| Research papers                                                  | Vector Search      |
| Chat with company knowledge base                                 | Hybrid Search      |
| Legal contracts                                                  | Hybrid Search      |
| Medical documents                                                | Hybrid Search      |
| Source code search                                               | Hybrid Search      |
| Product catalogs                                                 | Hybrid Search      |
| Documents containing IDs, version numbers, SKUs, invoice numbers | Hybrid Search      |

---

# What Senior AI Engineers Build

A production-grade RAG system typically follows this retrieval pipeline:

```text
Documents
      │
      ▼
Chunking
      │
      ▼
Embeddings
      │
      ▼
Vector Database (Dense Index)
      │
      ├──────────────┐
      ▼              ▼
Dense Retrieval   BM25 Retrieval
      │              │
      └──────┬───────┘
             ▼
 Reciprocal Rank Fusion (RRF)
             ▼
 Cross-Encoder Reranker
             ▼
      Top 5 Relevant Chunks
             ▼
            LLM
```

This architecture combines the strengths of semantic understanding (vector search) with exact keyword matching (BM25), then uses a reranker to select the most relevant chunks before they are sent to the language model. It provides significantly better retrieval quality than using vector search alone and is the standard approach in many enterprise AI applications.
