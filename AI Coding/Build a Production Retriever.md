# Build a Production Retriever for a RAG System

A **retriever** is responsible for finding the most relevant documents for a user's query. It is one of the most important components of a Retrieval-Augmented Generation (RAG) pipeline.

```
User Question
      │
      ▼
Retriever
      │
      ▼
Top-K Relevant Documents
      │
      ▼
Prompt Builder
      │
      ▼
LLM
      │
      ▼
Answer
```

---

# Responsibilities of a Retriever

A production retriever should:

* Generate query embeddings
* Search a vector index
* Apply metadata filters
* Remove duplicates
* Return the top-K documents
* Support hybrid search (optional)
* Return similarity scores
* Be observable (latency, recall, errors)

---

# Project Structure

```
rag/
│
├── retriever.py
├── embedding_model.py
├── vector_store.py
├── reranker.py
├── prompt_builder.py
├── models.py
└── app.py
```

---

# Step 1: Document Model

```python
from dataclasses import dataclass
from typing import Dict

@dataclass
class Document:
    id: str
    text: str
    metadata: Dict
    score: float = 0.0
```

Example:

```python
Document(
    id="101",
    text="Python supports async programming.",
    metadata={
        "source": "python.pdf",
        "page": 15
    }
)
```

---

# Step 2: Embedding Model

```python
from sentence_transformers import SentenceTransformer

class EmbeddingModel:

    def __init__(self):
        self.model = SentenceTransformer(
            "all-MiniLM-L6-v2"
        )

    def embed(self, text):
        return self.model.encode(
            text,
            normalize_embeddings=True
        )
```

---

# Step 3: Simple Vector Store

```python
import numpy as np

class VectorStore:

    def __init__(self):
        self.documents = []
        self.embeddings = []

    def add(self, document, embedding):
        self.documents.append(document)
        self.embeddings.append(embedding)

    def search(self, query_embedding, top_k=5):

        scores = []

        for doc, emb in zip(
                self.documents,
                self.embeddings):

            score = np.dot(query_embedding, emb)

            scores.append((score, doc))

        scores.sort(
            key=lambda x: x[0],
            reverse=True
        )

        return scores[:top_k]
```

Since embeddings are normalized, the dot product equals cosine similarity.

---

# Step 4: Build the Retriever

```python
class Retriever:

    def __init__(
        self,
        embedding_model,
        vector_store
    ):

        self.embedding_model = embedding_model
        self.vector_store = vector_store

    def retrieve(
        self,
        query,
        top_k=5
    ):

        query_embedding = (
            self.embedding_model.embed(query)
        )

        results = self.vector_store.search(
            query_embedding,
            top_k
        )

        documents = []

        for score, doc in results:
            doc.score = float(score)
            documents.append(doc)

        return documents
```

---

# Example

```python
embedding_model = EmbeddingModel()

store = VectorStore()

docs = [
    Document(
        id="1",
        text="Python supports decorators.",
        metadata={}
    ),
    Document(
        id="2",
        text="Transformers use self-attention.",
        metadata={}
    ),
    Document(
        id="3",
        text="Neural networks learn features.",
        metadata={}
    )
]

for d in docs:
    emb = embedding_model.embed(d.text)
    store.add(d, emb)

retriever = Retriever(
    embedding_model,
    store
)

results = retriever.retrieve(
    "Explain transformer attention"
)

for doc in results:
    print(doc.score, doc.text)
```

Example output:

```
0.91 Transformers use self-attention.
0.62 Neural networks learn features.
0.28 Python supports decorators.
```

---

# Add Metadata Filtering

Enterprise systems often retrieve only matching documents.

```python
def search(
    self,
    query_embedding,
    top_k=5,
    department=None
):

    scores = []

    for doc, emb in zip(
            self.documents,
            self.embeddings):

        if (
            department and
            doc.metadata.get("department")
            != department
        ):
            continue

        score = np.dot(query_embedding, emb)

        scores.append((score, doc))

    scores.sort(
        key=lambda x: x[0],
        reverse=True
    )

    return scores[:top_k]
```

Example:

```python
results = store.search(
    embedding,
    department="HR"
)
```

---

# Add Similarity Threshold

Don't return unrelated documents.

```python
threshold = 0.75

results = [
    (score, doc)
    for score, doc in results
    if score >= threshold
]
```

---

# Batch Retrieval

```python
queries = [
    "What is Python?",
    "Explain transformers",
    "Neural networks"
]

for q in queries:

    docs = retriever.retrieve(q)

    print(q)

    for d in docs:
        print(d.text)
```

---

# Async Retriever

Useful when the embedding API or vector database is remote.

```python
class Retriever:

    async def retrieve(
        self,
        query,
        top_k=5
    ):

        query_embedding = (
            await self.embedding_model.embed(query)
        )

        docs = await self.vector_store.search(
            query_embedding,
            top_k
        )

        return docs
```

---

# Production Retriever with FAISS

```python
import faiss
import numpy as np

vectors = np.array(
    embeddings,
    dtype=np.float32
)

index = faiss.IndexFlatIP(
    vectors.shape[1]
)

index.add(vectors)

def retrieve(query):

    query_embedding = embedding_model.embed(query)
    query_embedding = np.array(
        [query_embedding],
        dtype=np.float32
    )

    scores, ids = index.search(
        query_embedding,
        k=5
    )

    return [
        documents[i]
        for i in ids[0]
    ]
```

For millions of documents, replace `IndexFlatIP` with `IndexHNSWFlat` or `IndexIVFPQ`.

---

# Hybrid Retriever

Many enterprise systems combine keyword search and embeddings.

```
                 Query
                   │
        ┌──────────┴──────────┐
        ▼                     ▼
      BM25             Embedding Search
        │                     │
        └──────────┬──────────┘
                   ▼
            Score Fusion
                   ▼
              Top Documents
```

Example:

```python
final_score = (
    0.4 * bm25_score +
    0.6 * embedding_score
)
```

---

# Add a Reranker

The retriever prioritizes recall. A reranker improves precision.

```python
candidate_docs = retriever.retrieve(
    query,
    top_k=20
)

reranked = reranker.rank(
    query,
    candidate_docs
)

final_docs = reranked[:5]
```

---

# Add Observability

Track retrieval quality and performance.

```python
import time

start = time.time()

docs = retriever.retrieve(query)

latency = time.time() - start

print({
    "query": query,
    "latency": latency,
    "documents": len(docs)
})
```

Typical production metrics include:

* Retrieval latency
* Similarity scores
* Empty result rate
* Average top-K score
* Recall@K
* Precision@K
* Query embedding time

---

# Complete RAG Flow

```
User Question
      │
      ▼
Embedding Model
      │
      ▼
Query Vector
      │
      ▼
Retriever
      │
      ▼
Vector Database
      │
      ▼
Top 20 Chunks
      │
      ▼
Reranker
      │
      ▼
Top 5 Chunks
      │
      ▼
Prompt Builder
      │
      ▼
LLM
      │
      ▼
Final Answer
```

---

# How Senior AI Engineers Build Retrievers

A production-grade retriever typically includes:

* **Embedding model abstraction** (easy to swap models)
* **Vector database** (FAISS, Qdrant, Milvus, Pinecone, Weaviate)
* **Hybrid retrieval** (BM25 + semantic search)
* **Metadata filtering** (department, tenant, language, permissions)
* **Similarity thresholding** to avoid irrelevant results
* **Cross-encoder reranking** for better ranking quality
* **Async I/O** for remote embedding and vector services
* **Caching** of frequent query embeddings
* **Observability** (latency, recall, cost, errors)
* **Multi-tenancy and RBAC** for enterprise document isolation

This architecture scales from thousands to millions of documents while maintaining low latency and high retrieval quality.
