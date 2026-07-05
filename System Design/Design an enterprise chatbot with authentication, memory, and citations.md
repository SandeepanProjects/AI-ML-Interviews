This is one of the most common **Senior AI Engineer / Staff AI Engineer system design interviews**.

The interviewer is evaluating whether you understand how to build a **secure, scalable enterprise AI assistant**—not just a chatbot that calls an LLM.

A production enterprise chatbot must support:

* Authentication & authorization
* Multi-tenancy
* Conversation memory
* RAG over enterprise documents
* Citations
* Security guardrails
* Tool calling
* Streaming responses
* Monitoring & auditing

---

# Functional Requirements

### User Features

* Login using SSO (OAuth/OIDC/SAML)
* Chat with AI
* Upload PDFs, DOCX, PPTX
* Search company knowledge
* Follow-up questions
* View citations
* Conversation history
* Streaming responses

---

### Enterprise Features

* Multi-tenant
* Role-based access control (RBAC)
* Department-based document permissions
* Audit logs
* Cost tracking
* Feedback collection
* Admin dashboard

---

# High-Level Architecture

```text
                           Employee
                               │
                               ▼
                     Browser / Mobile App
                               │
                               ▼
                     API Gateway / Load Balancer
                               │
                               ▼
                           FastAPI
                               │
        ┌──────────────────────┼──────────────────────┐
        ▼                      ▼                      ▼
 Authentication          Authorization         Rate Limiting
        │                      │                      │
        └──────────────────────┴──────────────────────┘
                               │
                         AI Orchestrator
                               │
      ┌──────────────┬───────────────┬──────────────┐
      ▼              ▼               ▼              ▼
 Conversation     RAG Engine     Tool Engine   Model Router
 Memory
      │              │
      │              ▼
      │      Hybrid Retrieval
      │      ├── Vector Search
      │      ├── BM25
      │      ├── Metadata Filter
      │      └── Reranker
      │
      └──────────────┬───────────────┐
                     ▼               ▼
               Prompt Builder    Citation Builder
                     │
                     ▼
                LLM Gateway
                     │
                     ▼
               Streaming Response
```

---

# Storage Architecture

Different storage systems have different responsibilities.

```text
                PostgreSQL
        -------------------------
        Users
        Roles
        Conversations
        Messages
        Documents
        Audit Logs
        Feedback
        Citations Metadata

                Redis
        -------------------------
        Session Cache
        Prompt Cache
        Conversation Summary
        Rate Limiting

                Qdrant
        -------------------------
        Document Embeddings
        Chunk Metadata

                Object Storage
        -------------------------
        PDFs
        Word Files
        Images
```

---

# Folder Structure

```text
enterprise-chatbot/

app/
├── api/
│   ├── auth.py
│   ├── chat.py
│   ├── upload.py
│   └── admin.py
│
├── orchestrator/
│   └── ai_orchestrator.py
│
├── services/
│   ├── auth_service.py
│   ├── memory_service.py
│   ├── rag_service.py
│   ├── citation_service.py
│   ├── llm_service.py
│   └── audit_service.py
│
├── repositories/
│   ├── postgres_repo.py
│   ├── redis_repo.py
│   └── qdrant_repo.py
│
├── workers/
│   └── ingestion_worker.py
│
└── main.py
```

Notice that the API layer is intentionally thin. All business logic lives in services.

---

# Authentication

Use enterprise SSO (OIDC/OAuth).

```python
class AuthService:

    async def authenticate(self, token):

        user = verify_jwt(token)

        return user
```

Every request passes through authentication.

```text
Request

↓

JWT Validation

↓

User Object
```

---

# Authorization

Suppose:

```text
Alice → Finance

Bob → Engineering
```

When Alice searches:

```text
Quarterly Revenue
```

She must **never** retrieve engineering documents.

Instead:

```python
filter = {

    "department": user.department,

    "tenant": user.tenant
}
```

Every vector search includes this metadata filter.

---

# Conversation Memory

Instead of sending the entire chat history:

```text
100 Messages

↓

Summary

↓

Relevant History

↓

Prompt
```

Example service:

```python
class MemoryService:

    async def build_context(self, conversation_id):

        summary = load_summary(conversation_id)

        recent = load_recent_messages(conversation_id)

        return summary + recent
```

This reduces prompt size and cost.

---

# RAG Pipeline

```text
Question

↓

Embedding

↓

Hybrid Retrieval

↓

Metadata Filter

↓

Top Documents

↓

Reranker

↓

Context Builder
```

Only documents the user is authorized to access are considered.

---

# Citation Generation

Every retrieved chunk carries metadata.

```python
chunk = {

    "text": "...",

    "source": "Employee_Handbook.pdf",

    "page": 18,

    "section": "Vacation Policy"
}
```

Prompt:

```text
Context

Question

For every claim, reference the provided source IDs.
```

After generation:

```json
{
  "answer": "Employees receive 20 days of annual leave.",
  "citations": [
    {
      "document": "Employee_Handbook.pdf",
      "page": 18
    }
  ]
}
```

The frontend can render clickable references.

---

# AI Orchestrator

Instead of directly calling the LLM:

```python
class AIOrchestrator:

    async def answer(self, request):

        memory = await memory_service.build_context(
            request.conversation_id
        )

        docs = await rag.retrieve(
            request.question,
            request.user
        )

        prompt = prompt_builder.build(
            memory,
            docs,
            request.question
        )

        response = await llm.generate(prompt)

        return citation_service.attach(
            response,
            docs
        )
```

The orchestrator coordinates the workflow while each service has a single responsibility.

---

# Document Upload

Uploading a PDF:

```text
Browser

↓

FastAPI

↓

Object Storage

↓

Queue

↓

Worker

↓

Extract Text

↓

Chunk

↓

Embed

↓

Store in Qdrant

↓

Store Metadata in PostgreSQL
```

The upload API responds quickly because processing happens asynchronously.

---

# Streaming Responses

Users shouldn't wait for the full answer.

```python
async def stream():

    async for token in llm.stream(prompt):
        yield token
```

The browser receives tokens as they are generated.

---

# Audit Logging

Every enterprise request should be recorded.

```python
audit.log({

    "user": user.id,

    "question": question,

    "documents": retrieved_ids,

    "model": model_name,

    "tokens": usage.total_tokens
})
```

Store:

* User
* Time
* Retrieved documents
* Prompt version
* Model
* Token usage
* Cost
* Latency

---

# Security

Before the LLM:

```text
Authentication

↓

Authorization

↓

Rate Limiting

↓

Prompt Injection Detection

↓

PII Masking

↓

LLM
```

After the LLM:

```text
Output Validation

↓

Citation Validation

↓

Return
```

---

# End-to-End Flow

```text
Employee
 │
 ▼
Browser
 │
 ▼
API Gateway
 │
 ▼
FastAPI
 │
 ├── Authentication
 ├── Authorization
 ├── Rate Limiting
 ├── Logging
 └── Validation
 │
 ▼
AI Orchestrator
 │
 ├── Conversation Memory
 ├── Hybrid RAG
 ├── Metadata Filtering
 ├── Prompt Builder
 ├── LLM Gateway
 └── Citation Builder
 │
 ▼
Streaming Response
 │
 ▼
Save Conversation
 │
 ├── PostgreSQL
 ├── Redis (summary/cache)
 └── Audit Log
```

---

# Enterprise Deployment

```text
                 Internet
                     │
               WAF / CDN
                     │
              Load Balancer
                     │
          Kubernetes (EKS/AKS/GKE)
                     │
      ┌──────────────┼──────────────┐
      ▼              ▼              ▼
  FastAPI Pods   Worker Pods   Inference Pods
      │              │              │
      ├──────────────┼──────────────┤
      ▼              ▼              ▼
    Redis      PostgreSQL      Qdrant Cluster
      │
      ▼
Object Storage (PDFs, Images)
```

Each component scales independently based on workload.

---

# Production Improvements

A production enterprise chatbot typically goes beyond this baseline:

* **Conversation memory:** summarize older conversations, retrieve only relevant history, and expire inactive sessions.
* **Retrieval quality:** use hybrid search (BM25 + vectors), reranking, and context compression before prompting the LLM.
* **Citations:** attach document ID, page number, section, and confidence score so users can verify answers.
* **Security:** enforce tenant isolation, document-level permissions, prompt injection detection, and comprehensive audit logging.
* **Reliability:** implement retries, circuit breakers, caching, and graceful degradation if the vector database or LLM is unavailable.
* **Observability:** collect metrics for latency, retrieval quality, token usage, cost, and user feedback to continuously improve the system.

---

# How I would answer this in a senior AI interview

> I would design the chatbot as a collection of loosely coupled services with FastAPI as the API layer and an AI orchestrator coordinating authentication, memory, retrieval, prompt construction, model invocation, and citation generation. Authentication and RBAC ensure users only access authorized documents. The RAG pipeline performs hybrid retrieval with metadata filtering and reranking, while the memory service summarizes long conversations to reduce token costs. Every generated answer includes citations derived from retrieved document metadata. Redis provides caching and session storage, PostgreSQL stores conversations and audit logs, Qdrant stores embeddings, and document ingestion runs asynchronously through worker services. The entire platform is deployed on Kubernetes with monitoring, streaming responses, and comprehensive security controls suitable for enterprise environments.
