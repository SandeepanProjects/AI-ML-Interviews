# What Does a Vector Database Do? (End-to-End with Code)

A **Vector Database** is a specialized database designed to **store, index, and search embeddings (vectors)** efficiently.

It is one of the core components of **RAG (Retrieval-Augmented Generation)** systems.

Without a vector database, an LLM has no efficient way to search millions of document embeddings.

---

# Why Do We Need a Vector Database?

Suppose your company has:

```text
10 Million Documents
```

After chunking:

```text
10 Million Documents

↓

500 Million Chunks
```

After embedding:

```text
Chunk 1

↓

[0.12, -0.43, 0.98, ...]

Chunk 2

↓

[-0.56, 0.77, 0.23, ...]

...

500 Million Vectors
```

Where do we store these vectors?

A relational database like PostgreSQL is not optimized for nearest-neighbor vector search over hundreds of millions of embeddings. This is where vector databases come in.

---

# What Does a Vector Database Actually Store?

Each record typically contains:

```text
Vector

+

Original Text

+

Metadata
```

Example:

```text
ID: 101

Text:
Employees receive 180 days of maternity leave.

Vector:
[0.124, -0.882, 0.456, ...]

Metadata:
{
    source: "leave_policy.pdf",
    page: 7,
    department: "HR",
    tenant: "EY"
}
```

So the database stores much more than just vectors.

---

# High-Level Architecture

```text
                  PDF Documents
                        │
                        ▼
                  Text Chunking
                        │
                        ▼
                Embedding Model
                        │
                        ▼
      ┌─────────────────────────────────┐
      │        Vector Database          │
      │                                 │
      │  Vector + Text + Metadata       │
      └─────────────────────────────────┘
                        │
                        ▼
               Similarity Search
                        │
                        ▼
               Relevant Documents
                        │
                        ▼
                       LLM
```

---

# Without a Vector Database

Imagine storing vectors in a Python list.

```python
vectors = [
    [0.12, 0.53],
    [0.44, 0.89],
    [0.91, 0.12],
    ...
]
```

Searching means comparing the query against **every vector**.

```text
Query

↓

Compare with Vector 1

↓

Compare with Vector 2

↓

Compare with Vector 3

...

↓

Compare with Vector 500 Million
```

Time Complexity:

```text
O(N)
```

For millions of vectors, this is too slow.

---

# With a Vector Database

Instead:

```text
Query Vector

↓

Vector Index (HNSW / IVF / PQ)

↓

Only Compare Nearby Candidates

↓

Top K Results
```

Approximate nearest-neighbor (ANN) indexes reduce search time dramatically while maintaining high recall.

---

# Example

Suppose we have vectors:

```text
Document A

↓

[0.11, 0.21]

Document B

↓

[0.12, 0.20]

Document C

↓

[-0.82, 0.74]
```

Question:

```text
Car
```

Embedding:

```text
[0.10, 0.19]
```

Vector DB returns:

```text
Document B

Document A
```

Not because of keywords, but because they are close in vector space.

---

# Building a Vector Database with LangChain

## Step 1: Load Documents

```python
from langchain_core.documents import Document

docs = [
    Document(page_content="Python is a programming language."),
    Document(page_content="Machine learning is part of AI."),
    Document(page_content="Cats are mammals.")
]
```

---

## Step 2: Create Embeddings

```python
from langchain_openai import OpenAIEmbeddings

embedding = OpenAIEmbeddings(
    model="text-embedding-3-small"
)
```

---

## Step 3: Store in Qdrant

```python
from langchain_qdrant import QdrantVectorStore

vector_store = QdrantVectorStore.from_documents(
    documents=docs,
    embedding=embedding,
    url="http://localhost:6333",
    collection_name="knowledge_base"
)
```

What happens internally?

```text
Document

↓

Embedding

↓

Vector

↓

Stored in Qdrant
```

---

# Searching the Database

Create a retriever.

```python
retriever = vector_store.as_retriever(
    search_kwargs={"k": 2}
)
```

Search.

```python
results = retriever.invoke(
    "Explain artificial intelligence."
)

for doc in results:
    print(doc.page_content)
```

Output:

```text
Machine learning is part of AI.
Python is a programming language.
```

The retriever returns semantically relevant chunks.

---

# Metadata Filtering

Each vector can include metadata.

```python
Document(
    page_content="Leave policy",
    metadata={
        "department": "HR",
        "tenant": "EY",
        "page": 5
    }
)
```

Retrieve only HR documents.

```python
results = vector_store.similarity_search(
    "leave policy",
    k=3,
    filter={
        "department": "HR"
    }
)
```

This is essential for multi-tenant and RBAC-aware systems.

---

# Similarity Search

The vector database receives:

```text
Question

↓

Embedding

↓

[0.21, 0.82, ...]
```

Then computes similarity against indexed vectors.

```text
Question Vector

↓

Cosine Similarity

↓

Document A → 0.95

Document B → 0.82

Document C → 0.13
```

Top matches are returned.

---

# What Happens Internally?

```text
             Query

               │

               ▼

       Embedding Model

               │

               ▼

      Query Vector

               │

               ▼

      Vector Database

               │

    ANN Index (HNSW)

               │

               ▼

Nearest Neighbors

               │

               ▼

Document Chunks
```

The database does **not** generate answers. It retrieves relevant content.

---

# Popular Vector Databases

| Database | Notes                                    |
| -------- | ---------------------------------------- |
| Qdrant   | Open source, payload filtering, HNSW     |
| Pinecone | Managed cloud service                    |
| Milvus   | Large-scale distributed deployments      |
| Weaviate | Built-in hybrid search and modules       |
| Chroma   | Lightweight, great for local development |
| pgvector | PostgreSQL extension with vector support |

---

# Why Not Use PostgreSQL Alone?

A standard relational query:

```sql
SELECT *
FROM documents
WHERE text LIKE '%leave%';
```

Problems:

* Exact keyword matching
* No semantic understanding
* Poor performance for vector similarity unless using extensions such as pgvector

---

# ANN (Approximate Nearest Neighbor)

Instead of comparing every vector:

```text
500 Million Vectors

↓

Compare All

↓

Too Slow
```

ANN index:

```text
500 Million Vectors

↓

Index

↓

Few Hundred Candidates

↓

Best Matches
```

Common ANN algorithms:

* HNSW (Hierarchical Navigable Small World)
* IVF (Inverted File Index)
* Product Quantization (PQ)
* DiskANN (for large-scale disk-based search)

---

# Hybrid Search

Production systems rarely rely only on vectors.

```text
            User Query
                │
       ┌────────┴────────┐
       ▼                 ▼
 Vector Search      BM25 Search
       │                 │
       └────────┬────────┘
                ▼
          Merge Results
                ▼
            Reranker
                ▼
               LLM
```

This handles both semantic queries and exact identifiers like invoice numbers or API names.

---

# Complete RAG Flow

```text
User Question

↓

Embedding Model

↓

Vector Database

↓

Top 5 Chunks

↓

Prompt

↓

LLM

↓

Answer
```

The vector database's responsibility ends after retrieving the most relevant chunks.

---

# Common Mistakes

### Embedding Whole Documents

❌

```text
Entire 300-page PDF

↓

One Vector
```

Poor retrieval quality.

---

### Correct

```text
PDF

↓

Chunk

↓

Embedding

↓

Vector DB
```

---

### Ignoring Metadata

Without metadata:

```text
Question

↓

All Documents
```

With metadata:

```text
Question

↓

Tenant Filter

↓

Department Filter

↓

Similarity Search
```

Much more secure and efficient.

---

### Using Different Embedding Models

Indexing with one embedding model and querying with another produces vectors in different embedding spaces, leading to poor retrieval quality. Always use the same embedding model (or a compatible family explicitly designed for interoperability) for both indexing and querying.

---

# Common Interview Questions

### Does a vector database generate answers?

No. A vector database stores embeddings and retrieves the most similar vectors. The LLM generates the answer using the retrieved documents.

---

### Why can't we use a normal SQL database?

Traditional SQL databases are optimized for structured queries and exact matches. Vector databases are optimized for efficient nearest-neighbor search over high-dimensional embeddings. PostgreSQL can support vector search with extensions like **pgvector**, but dedicated vector databases often provide more advanced ANN indexing and scaling features.

---

### What is the role of metadata?

Metadata enables filtering by tenant, department, document type, language, access level, or other attributes before or during vector search, improving both security and retrieval quality.

---

### Why use ANN instead of exact search?

Exact search compares the query with every vector, which is expensive at scale. ANN algorithms retrieve highly relevant neighbors much faster, making interactive search practical for millions or billions of vectors.

---

# Traditional Database vs Vector Database

| Feature                    | Relational Database          | Vector Database                      |
| -------------------------- | ---------------------------- | ------------------------------------ |
| Stores structured rows     | ✅                            | Can store metadata alongside vectors |
| Stores embeddings          | Limited (without extensions) | ✅                                    |
| Semantic similarity search | ❌                            | ✅                                    |
| ANN indexing               | ❌                            | ✅                                    |
| Metadata filtering         | ✅                            | ✅                                    |
| Used in RAG                | Usually for metadata         | Core retrieval component             |

---

# Senior AI Engineer Interview Answer

> **A vector database is a specialized database that stores embeddings together with the original content and metadata, and performs efficient nearest-neighbor search over those embeddings. During indexing, documents are chunked, embedded, and stored with metadata such as source, tenant, and access level. At query time, the user's question is embedded using the same embedding model, and the vector database retrieves the most semantically similar chunks using ANN algorithms like HNSW. Those retrieved chunks are then passed to the LLM to generate a grounded response. In production, vector databases are typically combined with metadata filtering, hybrid search, reranking, and caching to deliver fast, accurate, and secure retrieval at scale.
