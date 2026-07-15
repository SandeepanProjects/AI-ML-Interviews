# Build a Retrieval Pipeline (Senior AI Engineer Interview)

A **Retrieval Pipeline** is the backbone of **Retrieval-Augmented Generation (RAG)** systems. It retrieves the most relevant information from a knowledge base before an LLM generates an answer.

This is one of the most common system design and coding topics for **Senior AI Engineer**, **LLM Engineer**, and **GenAI Engineer** interviews.

---

# What is a Retrieval Pipeline?

Without retrieval:

```text
User Question
      │
      ▼
     LLM
      │
      ▼
 Answer
```

The LLM relies only on its training data, which may be outdated or incomplete.

With retrieval:

```text
User Question
      │
      ▼
Embed Query
      │
      ▼
Vector Search
      │
      ▼
Top-K Documents
      │
      ▼
Context + Question
      │
      ▼
LLM
      │
      ▼
Answer
```

The LLM answers using retrieved, relevant documents.

---

# High-Level Architecture

```text
                Documents
                    │
                    ▼
             Document Loader
                    │
                    ▼
             Text Chunking
                    │
                    ▼
              Embedding Model
                    │
                    ▼
               Vector Database
                    │
────────────────────────────────────
                    ▲
                    │
               User Query
                    │
                    ▼
             Query Embedding
                    │
                    ▼
              Similarity Search
                    │
                    ▼
            Top-K Retrieved Chunks
                    │
                    ▼
                  LLM
                    │
                    ▼
                 Response
```

---

# Step 1 — Document Loader

```python
from pathlib import Path

def load_documents(folder: str):
    documents = []

    for file in Path(folder).glob("*.txt"):
        documents.append(file.read_text())

    return documents
```

Example:

```text
docs/
   policy.txt
   handbook.txt
   faq.txt
```

---

# Step 2 — Text Chunking

LLMs have context limits, so documents are split into smaller chunks.

```python
def chunk_text(text, chunk_size=500, overlap=100):
    chunks = []

    start = 0

    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap

    return chunks
```

Example:

```text
10000 characters

↓

Chunk 1
Chunk 2
Chunk 3
...
```

Overlap helps preserve context across chunk boundaries.

---

# Step 3 — Generate Embeddings

```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)

embeddings = model.encode(chunks)
```

Each chunk becomes a dense vector.

Example:

```text
"This is AI"

↓

[0.12,
 0.88,
 0.45,
 ...]
```

---

# Step 4 — Store in a Vector Database

Example using FAISS:

```python
import faiss
import numpy as np

dimension = embeddings.shape[1]

index = faiss.IndexFlatL2(dimension)

index.add(
    np.array(embeddings).astype("float32")
)
```

Metadata should also be stored so retrieved vectors can be mapped back to their source documents.

---

# Step 5 — Query Embedding

User asks:

```text
"What is the leave policy?"
```

Convert the query to an embedding:

```python
query_vector = model.encode(
    ["What is the leave policy?"]
)
```

---

# Step 6 — Similarity Search

```python
distances, indices = index.search(
    query_vector.astype("float32"),
    k=3
)
```

Output:

```text
Top 3 document chunks

Policy Chunk

Leave Rules

HR FAQ
```

---

# Step 7 — Build the Prompt

```python
context = "\n".join(
    chunks[i]
    for i in indices[0]
)

prompt = f"""
Context:
{context}

Question:
What is the leave policy?

Answer using only the context.
"""
```

---

# Step 8 — Generate Response

```python
response = llm.generate(prompt)
```

The LLM answers using the retrieved context.

---

# End-to-End Pipeline

```python
def retrieve(query):

    query_embedding = model.encode([query])

    _, idx = index.search(
        query_embedding.astype("float32"),
        k=5
    )

    return [chunks[i] for i in idx[0]]
```

---

# Complete Flow

```text
User Question
      │
      ▼
Embedding Model
      │
      ▼
Vector Search
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
Answer
```

---

# Retrieval Strategies

| Strategy              | Description                      | Use Case              |
| --------------------- | -------------------------------- | --------------------- |
| Dense Retrieval       | Embedding similarity             | Semantic search       |
| Sparse Retrieval      | BM25 / keyword search            | Exact keyword matches |
| Hybrid Retrieval      | Dense + sparse combined          | Enterprise search     |
| Metadata Filtering    | Filter by tags, user, date       | Multi-tenant systems  |
| Multi-Query Retrieval | Generate multiple query variants | Improve recall        |

---

# Add a Reranker

Initial retrieval may return relevant but poorly ordered documents.

```text
Retriever
     │
Top 20 Documents
     │
     ▼
Cross-Encoder Reranker
     │
Top 5 Documents
     │
     ▼
LLM
```

Rerankers often improve answer quality by reordering documents using a more expensive model.

---

# Production Retrieval Pipeline

```text
              S3 / PDFs / Docs
                      │
                      ▼
              Document Loader
                      │
                      ▼
               Text Chunker
                      │
                      ▼
              Embedding Model
                      │
                      ▼
      Vector DB (FAISS/Qdrant/Milvus)
                      │
────────────────────────────────────────
                      ▲
                      │
                User Query
                      │
                      ▼
             Query Embedding
                      │
                      ▼
             Metadata Filters
                      │
                      ▼
             Vector Search (Top 50)
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
                  Response
```

---

# Production Enhancements

A senior-level retrieval pipeline typically includes:

| Component            | Purpose                                                          |
| -------------------- | ---------------------------------------------------------------- |
| Chunking Strategy    | Semantic or sentence-aware chunking instead of fixed-size splits |
| Embedding Cache      | Avoid recomputing embeddings for unchanged documents             |
| Metadata Filters     | Tenant ID, language, document type, permissions                  |
| Hybrid Search        | Combine BM25 with dense retrieval                                |
| Reranking            | Improve precision before sending context to the LLM              |
| Context Compression  | Remove redundant passages to fit context limits                  |
| Observability        | Track retrieval latency, hit rate, recall, and token usage       |
| Incremental Indexing | Update only changed documents                                    |
| Security             | Enforce access control before retrieval                          |
| Evaluation           | Measure Recall@K, MRR, NDCG, and answer quality                  |

---

# Common Interview Follow-ups

**Q: Why chunk documents?**

To fit model context limits and improve retrieval granularity.

**Q: Why overlap chunks?**

To preserve information that spans chunk boundaries.

**Q: Why embeddings instead of keyword search?**

Embeddings capture semantic similarity, allowing retrieval of related concepts even when exact words differ.

**Q: Why use a reranker?**

Vector search is optimized for recall. A reranker improves precision by reordering the retrieved candidates.

**Q: What if the knowledge base contains millions of documents?**

Use an approximate nearest neighbor (ANN) index (e.g., HNSW or IVF), shard the vector database, cache popular queries, and perform distributed retrieval.

---

# Senior AI Engineer Interview Answer

A strong summary is:

> "A production retrieval pipeline ingests documents, chunks them into semantically meaningful passages, generates embeddings, and stores them in a vector database with metadata. At query time, the user's question is embedded, relevant chunks are retrieved using vector similarity (optionally combined with keyword search and metadata filters), reranked, and assembled into a prompt for the LLM. I optimize for both retrieval quality and latency using hybrid search, reranking, embedding caching, incremental indexing, and comprehensive observability."
