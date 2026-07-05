This is one of the **most common Senior AI Engineer system design interviews**.

The interviewer is not looking for a simple RAG demo. They expect you to design a **production-grade system** that can handle:

* Millions of documents
* Thousands of concurrent users
* High availability
* Low latency
* Security
* Monitoring
* Cost optimization
* Fault tolerance

---

# Interview Question

> **Design a Production RAG System**

A good answer should explain:

1. Architecture
2. Components
3. Request Flow
4. Ingestion Pipeline
5. Retrieval Pipeline
6. Caching
7. Scaling
8. Deployment
9. Monitoring
10. Code Structure

---

# Step 1 — High-Level Architecture

```text
                        Internet
                            │
                    AWS Application Load Balancer
                            │
              ┌─────────────┴─────────────┐
              │                           │
         FastAPI Pod                 FastAPI Pod
              │                           │
              └─────────────┬─────────────┘
                            │
                     Authentication
                            │
                     Rate Limiting
                            │
                       AI Service
                            │
        ┌──────────────┬───────────────┬──────────────┐
        │              │               │              │
        ▼              ▼               ▼              ▼
     Redis        PostgreSQL      Embedding      Prompt Builder
      │                               │
      │                               ▼
      │                          Qdrant Vector DB
      │                               │
      └──────────────► Retrieved Documents
                                      │
                                      ▼
                                 LLM Gateway
                                      │
                                      ▼
                                 GPT / Claude
                                      │
                                      ▼
                                  JSON Response
```

---

# Why Each Component Exists

| Component         | Responsibility                          |
| ----------------- | --------------------------------------- |
| FastAPI           | HTTP API                                |
| Redis             | Response cache, sessions, rate limiting |
| PostgreSQL        | Users, chats, metadata                  |
| Qdrant            | Vector similarity search                |
| Embedding Service | Convert text → vectors                  |
| Prompt Builder    | Construct LLM prompt                    |
| LLM Gateway       | OpenAI/Anthropic abstraction            |
| Monitoring        | Metrics, logs, tracing                  |

---

# Folder Structure

```text
app/
│
├── api/
│      chat.py
│      upload.py
│
├── services/
│      rag_service.py
│      embedding_service.py
│      retrieval_service.py
│      llm_service.py
│      cache_service.py
│
├── repositories/
│      postgres_repo.py
│      qdrant_repo.py
│
├── models/
│
├── schemas/
│
├── workers/
│      ingestion_worker.py
│
├── config.py
│
└── main.py
```

Notice that:

* API contains no business logic.
* Services contain business logic.
* Repositories talk to databases.

---

# Document Ingestion Pipeline

When a PDF is uploaded, we **don't** send it directly to the LLM.

Instead:

```text
Upload PDF
     │
     ▼
Extract Text
     │
     ▼
Chunk Document
     │
     ▼
Generate Embeddings
     │
     ▼
Store Chunks in Qdrant
     │
     ▼
Store Metadata in PostgreSQL
```

---

## Document Chunking

```python
class ChunkService:

    def split(self, text, size=500, overlap=100):

        chunks = []

        start = 0

        while start < len(text):

            end = start + size

            chunks.append(text[start:end])

            start += size - overlap

        return chunks
```

Example:

```text
100-page PDF

↓

400 chunks

↓

400 embeddings

↓

Qdrant
```

---

# Embedding Service

```python
from sentence_transformers import SentenceTransformer

class EmbeddingService:

    def __init__(self):

        self.model = SentenceTransformer(
            "all-MiniLM-L6-v2"
        )

    def embed(self, text):

        return self.model.encode(text).tolist()
```

---

# Store in Qdrant

```python
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct

class VectorRepository:

    def __init__(self):

        self.client = QdrantClient("localhost", port=6333)

    def insert(self, chunk_id, vector, text):

        self.client.upsert(

            collection_name="documents",

            points=[

                PointStruct(

                    id=chunk_id,

                    vector=vector,

                    payload={"text": text}

                )

            ]
        )
```

---

# Chat Request Pipeline

User asks:

```text
Explain Kubernetes deployments.
```

Flow:

```text
FastAPI
    │
    ▼
Redis Cache
    │
Cache Miss
    │
    ▼
Embedding Service
    │
    ▼
Qdrant Search
    │
Top-5 Chunks
    │
    ▼
Prompt Builder
    │
    ▼
LLM
    │
    ▼
Redis Cache
    │
    ▼
Save Chat
    │
    ▼
Return Response
```

---

# Retrieval Service

```python
class RetrievalService:

    def __init__(self):

        self.qdrant = VectorRepository()

        self.embedding = EmbeddingService()

    def search(self, question):

        vector = self.embedding.embed(question)

        results = self.qdrant.search(vector)

        return [
            r.payload["text"]
            for r in results
        ]
```

---

# Prompt Builder

```python
class PromptBuilder:

    def build(self, question, docs):

        context = "\n\n".join(docs)

        return f"""
You are an AI assistant.

Context:

{context}

Question:

{question}

Answer only from context.
"""
```

---

# LLM Service

```python
from openai import OpenAI

class LLMService:

    def __init__(self):

        self.client = OpenAI()

    def generate(self, prompt):

        response = self.client.chat.completions.create(

            model="gpt-4.1",

            messages=[

                {
                    "role": "user",
                    "content": prompt
                }

            ]
        )

        return response.choices[0].message.content
```

---

# Redis Cache

```python
class CacheService:

    def get(self, key):

        return redis.get(key)

    def set(self, key, value):

        redis.setex(

            key,

            3600,

            value
        )
```

---

# Main RAG Service

```python
class RAGService:

    def __init__(self):

        self.cache = CacheService()

        self.retriever = RetrievalService()

        self.llm = LLMService()

    def answer(self, question):

        cached = self.cache.get(question)

        if cached:

            return cached

        docs = self.retriever.search(question)

        prompt = PromptBuilder().build(question, docs)

        answer = self.llm.generate(prompt)

        self.cache.set(question, answer)

        return answer
```

Notice how each class has a single responsibility, making the system easier to test and maintain.

---

# FastAPI Endpoint

```python
from fastapi import APIRouter

router = APIRouter()

service = RAGService()

@router.post("/chat")

def chat(request):

    return {

        "answer":

        service.answer(

            request.question
        )
    }
```

---

# Why PostgreSQL?

Qdrant stores vectors, not application data.

Use PostgreSQL for:

```text
Users

Chats

Files

Permissions

Billing

Audit Logs

Feedback
```

Example:

```text
Conversation

↓

User ID

↓

Question

↓

Answer

↓

Timestamp
```

---

# Background Ingestion

Uploading a large PDF can take time.

Instead of blocking the API:

```text
Upload PDF
      │
      ▼
S3
      │
      ▼
Queue (SQS / Kafka / RabbitMQ)
      │
      ▼
Worker
      │
      ▼
Chunk
      │
      ▼
Embedding
      │
      ▼
Qdrant
```

This keeps the API responsive.

---

# Kubernetes Deployment

```text
                  Kubernetes
                       │
        ┌──────────────┼──────────────┐
        ▼              ▼              ▼
   FastAPI Pod    FastAPI Pod    FastAPI Pod
        │              │              │
        └──────────────┼──────────────┘
                       │
                   Redis Cluster
                       │
                PostgreSQL Primary
                       │
               PostgreSQL Replica
                       │
                 Qdrant Cluster
                       │
                  OpenAI API
```

Scale the stateless FastAPI pods independently of the stateful data stores.

---

# Monitoring

Collect metrics such as:

* Request latency
* Retrieval latency
* Embedding latency
* LLM latency
* Cache hit rate
* Token usage
* Cost per request
* Qdrant search latency
* Error rate

A common stack is **Prometheus** for metrics, **Grafana** for dashboards, and **OpenTelemetry** for distributed tracing.

---

# Complete Production Request Flow

```text
User
 │
 ▼
Load Balancer
 │
 ▼
FastAPI
 │
 ├── Authentication
 ├── Rate Limiting
 ├── Logging
 └── Request Validation
 │
 ▼
RAG Service
 │
 ├── Redis Cache
 │      │
 │      ├── Hit → Return
 │      └── Miss
 │
 ├── Embedding Service
 │
 ├── Qdrant Search (Top-K)
 │
 ├── Prompt Builder
 │
 ├── LLM Gateway
 │
 ├── Save Conversation (PostgreSQL)
 │
 ├── Cache Response (Redis)
 │
 └── Return JSON
 │
 ▼
Client
```

---

# How a Senior AI Engineer Improves This Design

A production system typically adds:

* **Hybrid retrieval** (BM25 + vector search) to improve recall.
* **Metadata filtering** (tenant, department, document type, language) before vector search.
* **Reranking** with a cross-encoder or reranker model to improve the quality of retrieved chunks.
* **Streaming responses** so users see tokens as they are generated.
* **Asynchronous ingestion** using queues and workers for large document uploads.
* **Observability** with request IDs, traces, and detailed latency metrics for each stage.
* **Security** through authentication, authorization, document-level access control, and encryption.
* **Versioning** of embeddings so documents can be re-embedded after changing embedding models.
* **Fallback strategies**, such as returning "I don't know" when retrieval confidence is low instead of encouraging hallucinations.

These are the kinds of production considerations that distinguish a simple RAG prototype from a system that can reliably support enterprise workloads.

If I were interviewing a **Senior/Staff AI Engineer (Meta, Amazon, Microsoft, OpenAI, Anthropic, Nvidia, etc.)**, I would expect **much more** than the previous design.

The previous architecture is a **basic production RAG**. Real enterprise AI systems have many additional layers for reliability, scalability, observability, security, and cost control.

Let's redesign it like a production system handling **10 million documents**, **100,000 concurrent users**, and **multiple AI models**.

---

# Production Enterprise RAG Architecture

```text
                                        Internet
                                           │
                              AWS Application Load Balancer
                                           │
                                  API Gateway / WAF
                                           │
                    ┌──────────────────────┴──────────────────────┐
                    │                                             │
              FastAPI Pod 1                                 FastAPI Pod N
                    │                                             │
                    └──────────────────────┬──────────────────────┘
                                           │
                              Authentication (JWT/OAuth)
                                           │
                             Authorization (RBAC / ABAC)
                                           │
                               Rate Limiter (Redis)
                                           │
                                   Request ID
                                           │
                                    AI Orchestrator
                                           │
      ┌──────────────┬──────────────┬──────────────┬───────────────┐
      │              │              │              │               │
      ▼              ▼              ▼              ▼               ▼
 Conversation    Prompt       Retrieval      Tool Calling      Model Router
 Memory          Builder         Engine         (ReAct)             │
      │              │              │              │                │
      │              │              ▼              │                ▼
      │              │      Hybrid Search          │       GPT / Claude / Local
      │              │       BM25 + Vector         │
      │              │              │              │
      │              │        Metadata Filter      │
      │              │              │              │
      │              │          Cross Encoder      │
      │              │            Reranker         │
      │              │              │              │
      │              │        Context Compression  │
      │              │              │              │
      └──────────────┴──────────────┴──────────────┘
                              │
                         Final Prompt
                              │
                              ▼
                           LLM API
                              │
                              ▼
                     Streaming Response
                              │
                              ▼
                          FastAPI
```

Notice something important:

The LLM is **only one small box**.

Everything else exists to make the system reliable.

---

# Layer 1 — API Layer

Responsibilities:

* Authentication
* Authorization
* Validation
* Streaming
* Logging
* Metrics
* Rate limiting
* Tenant isolation

FastAPI should contain almost no business logic.

```text
POST /chat

↓

Router

↓

AIService.answer()
```

---

# Layer 2 — AI Orchestrator

This becomes the brain of the application.

Instead of calling the LLM directly:

```text
User

↓

AI Orchestrator

↓

Decides:

Need Retrieval?

Need Calculator?

Need SQL?

Need Web Search?

Need Memory?

Need Agent?
```

Example:

```python
class AIOrchestrator:

    async def answer(self, question):

        intent = self.intent_classifier.predict(question)

        if intent == "RAG":
            return await self.rag.answer(question)

        if intent == "SQL":
            return await self.sql_agent.run(question)

        if intent == "Calculator":
            return await self.math_agent.run(question)

        if intent == "Search":
            return await self.web_agent.run(question)
```

Instead of one AI pipeline, you now have multiple specialized pipelines.

---

# Layer 3 — Hybrid Retrieval

Most beginners use only vector search.

Production systems rarely do.

Instead:

```text
Question

↓

Vector Search

+

BM25 Keyword Search

↓

Merge Results

↓

Rerank

↓

Top Documents
```

Why?

Consider:

```text
HTTP 500
```

Embedding models often don't represent error codes well.

Keyword search does.

Hybrid retrieval combines semantic and exact matching.

---

# Layer 4 — Metadata Filtering

Suppose the company has:

```text
Engineering Docs

HR Docs

Finance Docs

Legal Docs
```

A Finance user should not retrieve HR policies.

Instead of searching all vectors:

```python
qdrant.search(

    vector=query,

    filter={

        "department":"finance"

    }
)
```

Metadata filtering dramatically improves both security and retrieval quality.

---

# Layer 5 — Cross-Encoder Reranker

Vector search returns approximate matches.

Example:

```text
Chunk A

Similarity 0.82

Chunk B

Similarity 0.80

Chunk C

Similarity 0.79
```

These scores are based on embedding distance, not deep understanding.

A reranker evaluates the **query and document together**:

```text
Question

+

Candidate Document

↓

Cross Encoder

↓

Relevance Score
```

The reranker often changes the order, improving answer quality.

---

# Layer 6 — Context Compression

Suppose retrieval returns:

```text
20 documents

×

1000 tokens

=

20,000 tokens
```

The LLM cannot efficiently process that.

Instead:

```text
Retrieve

↓

Summarize

↓

Remove duplicates

↓

Keep only relevant paragraphs

↓

Prompt
```

Benefits:

* Lower cost
* Lower latency
* Better answers

---

# Layer 7 — Conversation Memory

Users ask follow-up questions:

```text
User

What is Kubernetes?

↓

Assistant

...

↓

User

How do I scale it?
```

"What" refers to Kubernetes.

Maintain conversation memory:

```text
Conversation

↓

Summary

↓

Current Question

↓

Prompt
```

Rather than sending the full chat history every time, summarize older turns to stay within the model's context window.

---

# Layer 8 — Model Router

Different questions require different models.

```text
Small Question

↓

Small Model

Large Reasoning

↓

GPT-4.1

Code

↓

Code-specialized model

Vision

↓

Vision model
```

Example:

```python
if tokens < 300:

    model = "gpt-small"

else:

    model = "gpt-4.1"
```

This reduces inference costs.

---

# Layer 9 — Streaming

Don't wait for the complete answer.

```text
LLM

↓

Token

↓

Token

↓

Token

↓

Browser
```

Users see the response immediately, improving perceived latency.

---

# Layer 10 — Background Workers

Uploading PDFs should never block the API.

Instead:

```text
Upload

↓

S3

↓

SQS

↓

Worker

↓

Chunk

↓

Embed

↓

Qdrant
```

Workers can be scaled independently from the API.

---

# Layer 11 — Multi-Level Caching

Instead of only caching final answers:

```text
Redis

├── Embedding Cache

├── Retrieval Cache

├── Prompt Cache

├── LLM Response Cache

└── Session Cache
```

This reduces latency and API costs.

---

# Layer 12 — Observability

Every request gets a unique Request ID.

```text
Request

↓

Request ID

↓

Redis

↓

Qdrant

↓

OpenAI

↓

Postgres

↓

Logs
```

Collect metrics such as:

* API latency
* Retrieval latency
* Embedding latency
* LLM latency
* Cache hit ratio
* Prompt token count
* Completion token count
* Cost per request
* Error rate

Tracing allows you to pinpoint slow or failing stages.

---

# Layer 13 — Security

Enterprise systems require:

```text
JWT

↓

RBAC

↓

Document Permissions

↓

Tenant Isolation

↓

Audit Logging

↓

Encryption
```

Never allow retrieval across tenants.

---

# Layer 14 — Fault Tolerance

What if Qdrant is unavailable?

```text
Retrieve

↓

Failure

↓

Retry

↓

Fallback

↓

Cached Result

↓

"I don't know"
```

Similarly:

* Retry transient LLM failures.
* Circuit-break repeated failures.
* Use graceful degradation instead of crashing.

---

# Improved End-to-End Flow

```text
User
 │
 ▼
FastAPI
 │
 ├── JWT Authentication
 ├── Authorization
 ├── Validation
 ├── Rate Limiting
 ├── Logging / Tracing
 └── Request ID
 │
 ▼
AI Orchestrator
 │
 ├── Intent Classification
 ├── Conversation Memory
 ├── Model Selection
 ├── Tool Selection
 └── RAG Pipeline
 │
 ▼
Hybrid Retrieval
 │
 ├── BM25 Search
 ├── Vector Search
 ├── Metadata Filter
 ├── Merge Results
 ├── Cross-Encoder Rerank
 └── Context Compression
 │
 ▼
Prompt Builder
 │
 ▼
LLM Gateway
 │
 ├── Streaming
 ├── Retry
 ├── Timeout
 └── Cost Tracking
 │
 ▼
Redis Cache
 │
 ▼
PostgreSQL (Conversation, Audit, Feedback)
 │
 ▼
Streaming Response to Client
```

# What would I add for an interview?

If you're interviewing for a senior AI engineer role, I'd also discuss concerns that often distinguish production systems:

* **Document ingestion pipeline:** OCR for scanned PDFs, language detection, PII redaction, chunk quality validation, embedding versioning, and idempotent reprocessing.
* **Retrieval quality:** query rewriting, hypothetical document generation (HyDE), multi-query retrieval, and evaluation using metrics like Recall@K and MRR.
* **LLM safety:** prompt injection detection, output moderation, citation generation, and confidence-based refusal when retrieved evidence is weak.
* **Deployment:** Kubernetes with autoscaling, GPU inference pools, blue/green or canary deployments, and separate scaling policies for API servers, workers, and vector databases.
* **Evaluation:** an offline benchmark dataset plus online A/B testing to compare prompt versions, retrievers, rerankers, and models before rolling changes into production.

These additions move the design from a "working RAG application" to an **enterprise-grade AI platform** capable of serving large organizations reliably.
