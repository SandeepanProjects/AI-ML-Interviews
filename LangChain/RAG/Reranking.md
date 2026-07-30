# Reranking Explained End-to-End (with Production Code)

**Reranking** is one of the **most important components of production RAG systems**.

Many people think:

> **Vector Search → LLM**

This works for demos, but **production RAG systems almost always rerank retrieved documents before sending them to the LLM.**

A typical enterprise RAG pipeline looks like:

```text
User Query
      │
      ▼
Query Rewriter
      │
      ▼
Hybrid Search (Vector + BM25)
      │
      ▼
Top 50 Documents
      │
      ▼
Reranker
      │
      ▼
Top 5 Documents
      │
      ▼
Context Compression
      │
      ▼
LLM
      │
      ▼
Answer
```

The reranker is responsible for improving the quality of the retrieved context.

---

# Why Do We Need a Reranker?

Suppose the user asks:

> **How do I reset my corporate VPN password?**

Your vector database returns:

```
1. VPN Installation Guide
2. Password Policy
3. VPN Password Reset Guide
4. IT Helpdesk Contacts
5. VPN Troubleshooting
```

The vector database found documents that are **semantically similar**, but they are **not perfectly ordered**.

The best document is actually:

```
VPN Password Reset Guide
```

which is ranked **third**.

If your prompt only includes the **top 2** documents:

```
VPN Installation Guide
Password Policy
```

the LLM may answer incorrectly because the most relevant document was excluded.

---

# Without Reranking

```
User Question

↓

Vector Search

↓

Doc A
Doc B
Doc C
Doc D
Doc E

↓

Top 3

↓

LLM
```

The best document may not be included.

---

# With Reranking

```
User Question

↓

Vector Search

↓

Top 50

↓

Cross Encoder

↓

Top 5

↓

LLM
```

The most relevant documents are promoted before generation.

---

# Why Doesn't the Vector Database Return the Perfect Ranking?

A vector database uses **embeddings**.

Example:

```
Question

↓

Embedding

↓

1536 Numbers

↓

Cosine Similarity
```

It compares vectors.

Cosine similarity measures **semantic closeness**, not detailed relevance.

For example:

```
Question

↓

VPN password reset

↓

Document

VPN installation

Similarity = 0.90
```

```
Question

↓

VPN password reset

↓

Document

VPN password reset

Similarity = 0.89
```

Even though the second document is clearly better, approximate nearest-neighbor search and embedding similarity may rank the first document higher.

---

# What Does a Reranker Do?

Instead of comparing vectors, a reranker compares:

```
Question

+

Document
```

together.

It asks:

> "How relevant is this document for this specific question?"

This is a much richer comparison than cosine similarity alone.

---

# Retrieval Pipeline

```
Question

↓

Embedding

↓

Vector Search

↓

50 Documents

↓

Cross Encoder

↓

5 Documents

↓

LLM
```

---

# Bi-Encoder vs Cross-Encoder

This distinction is crucial.

## Vector Search (Bi-Encoder)

Documents are embedded **beforehand**.

```
Document

↓

Embedding

↓

Stored
```

Query:

```
Question

↓

Embedding

↓

Cosine Similarity
```

Fast because document embeddings are precomputed.

---

## Cross Encoder (Reranker)

A cross encoder processes the question and document **together**.

```
Question

+

Document

↓

Transformer

↓

Relevance Score
```

This is slower but much more accurate.

---

# Example

Question

```
How do I reset my VPN password?
```

Document A

```
VPN installation guide
```

Score:

```
0.72
```

Document B

```
Reset VPN password
```

Score:

```
0.99
```

The reranker moves Document B above Document A.

---

# Code Using LangChain + Cohere Reranker

Install:

```bash
pip install langchain-cohere
```

---

Create retriever:

```python
from langchain_qdrant import QdrantVectorStore
from langchain_openai import OpenAIEmbeddings

embedding = OpenAIEmbeddings()

retriever = vector_store.as_retriever(
    search_kwargs={"k": 20}
)
```

Retrieve:

```python
docs = retriever.invoke(
    "How do I reset my VPN password?"
)
```

Now rerank.

```python
from langchain_cohere import CohereRerank

reranker = CohereRerank(
    model="rerank-v3.5"
)

reranked = reranker.compress_documents(
    documents=docs,
    query="How do I reset my VPN password?"
)
```

Now the returned list is ordered by estimated relevance rather than vector similarity alone.

---

# Pipeline

```
Question

↓

Retriever

↓

20 Documents

↓

Reranker

↓

Top 5

↓

Prompt
```

---

# LangChain Contextual Compression Retriever

LangChain lets you wrap a retriever with a reranker.

```python
from langchain.retrievers import ContextualCompressionRetriever
from langchain_cohere import CohereRerank

base_retriever = vector_store.as_retriever(
    search_kwargs={"k": 20}
)

compressor = CohereRerank(
    model="rerank-v3.5"
)

retriever = ContextualCompressionRetriever(
    base_retriever=base_retriever,
    base_compressor=compressor,
)

docs = retriever.invoke(
    "Reset VPN password"
)
```

Your application still calls `retriever.invoke()`, but the results have been reranked.

---

# Using Hugging Face Cross Encoder

Open-source rerankers are also common.

```bash
pip install sentence-transformers
```

```python
from sentence_transformers import CrossEncoder

model = CrossEncoder(
    "cross-encoder/ms-marco-MiniLM-L-6-v2"
)

pairs = [
    (
        "How do I reset my VPN password?",
        "VPN installation guide."
    ),
    (
        "How do I reset my VPN password?",
        "VPN password reset guide."
    )
]

scores = model.predict(pairs)

print(scores)
```

Example output:

```
[
    0.31,
    0.98
]
```

The second document is clearly more relevant.

---

# Complete Production Pipeline

```python
query = "Reset VPN password"

# Step 1: Retrieve candidates
docs = retriever.invoke(query)

# Step 2: Rerank
reranked_docs = reranker.compress_documents(
    documents=docs,
    query=query
)

# Step 3: Build context
context = "\n\n".join(
    doc.page_content for doc in reranked_docs[:5]
)

# Step 4: Generate answer
response = llm.invoke(
    f"""
Question:
{query}

Context:
{context}
"""
)
```

---

# How Many Documents Should Be Retrieved?

Typical production values:

```
Retrieve 50

↓

Rerank

↓

Keep 5–10

↓

LLM
```

Retrieving only 5 documents initially may miss the best document.

Retrieving hundreds increases latency and reranker cost.

---

# Cost vs Accuracy

```
Vector Search

10 ms
```

```
Cross Encoder

150 ms
```

So reranking is slower.

However:

```
Better Retrieval

↓

Better Context

↓

Better Answer
```

The accuracy gain often outweighs the extra latency.

---

# Common Reranker Models

Commercial:

* Cohere Rerank
* Voyage AI Rerank

Open source:

* `cross-encoder/ms-marco-MiniLM-L-6-v2`
* `BAAI/bge-reranker-base`
* `BAAI/bge-reranker-large`

These models are trained specifically to score the relevance of query-document pairs.

---

# Enterprise Architecture

```
                    User

                      │

                      ▼

              Query Rewriter

                      ▼

      Hybrid Search (BM25 + Vector)

                      ▼

             Top 50 Documents

                      ▼

         Cross Encoder Reranker

                      ▼

              Top 5 Documents

                      ▼

         Context Compression

                      ▼

                    LLM

                      ▼

                  Response
```

---

# Common Interview Questions

### Why not let the vector database do the ranking?

Vector databases rank by embedding similarity, which is fast but approximate. A reranker performs a more detailed relevance evaluation using both the query and each document together.

---

### Why is a reranker slower?

Unlike vector search, which compares precomputed embeddings, a cross encoder runs a transformer model for every query-document pair.

For 50 retrieved documents:

```
Question + Doc1

↓

Transformer

↓

Score
```

repeated 50 times.

---

### Should every RAG system use a reranker?

Not necessarily.

For:

* 100 documents
* Internal prototypes
* Low-latency applications

vector search alone may be sufficient.

For enterprise search over thousands or millions of documents, reranking usually improves retrieval quality significantly.

---

# Vector Search vs Reranker

| Feature                              | Vector Search       | Reranker                                                                |
| ------------------------------------ | ------------------- | ----------------------------------------------------------------------- |
| Uses embeddings                      | ✅                   | Usually no precomputed embeddings; scores query-document pairs directly |
| Fast                                 | ✅                   | ❌                                                                       |
| ANN index                            | ✅                   | ❌                                                                       |
| Semantic retrieval                   | ✅                   | ✅                                                                       |
| Detailed relevance scoring           | Limited             | ✅                                                                       |
| Precomputed document representations | ✅                   | ❌                                                                       |
| Typical role                         | Candidate retrieval | Final ranking                                                           |

---

# Senior AI Engineer Interview Answer

> **A reranker is a second-stage retrieval model that improves the ordering of documents returned by vector or hybrid search. The retriever first retrieves a relatively large candidate set—typically 20 to 100 documents—using embeddings or keyword search. The reranker then evaluates each query-document pair with a cross-encoder model to produce a more accurate relevance score. The highest-ranked documents are passed to the LLM, improving answer quality and reducing hallucinations. In production RAG systems, I typically retrieve a larger candidate set, rerank it with a cross-encoder such as Cohere Rerank or BGE Reranker, keep the top 5–10 documents, and then send those to the LLM. This two-stage retrieval pipeline provides a much better balance of speed and accuracy than relying on vector search alone.
