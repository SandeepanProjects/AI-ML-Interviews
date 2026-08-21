For a **production RAG API**, I would avoid putting everything inside the FastAPI route. I would build it as a layered system where **PostgreSQL owns transactional metadata, Qdrant owns vector retrieval, Redis handles caching/distributed coordination, and FastAPI exposes the API**.

Since you're preparing for senior AI-engineer interviews, this is the architecture I'd be comfortable explaining on a whiteboard.

## 1. High-level architecture

```text
                         ┌──────────────────┐
                         │      Client      │
                         └────────┬─────────┘
                                  │
                                  ▼
                         ┌──────────────────┐
                         │     FastAPI      │
                         │ Auth / RBAC      │
                         │ Validation       │
                         │ Rate limiting   │
                         └────────┬─────────┘
                                  │
                     ┌────────────┴────────────┐
                     │                         │
                     ▼                         ▼
             ┌──────────────┐          ┌──────────────┐
             │ RAG Service  │          │ Ingestion    │
             │              │          │ Service      │
             └──────┬───────┘          └──────┬───────┘
                    │                         │
          ┌─────────┼─────────┐       ┌───────┼────────┐
          ▼         ▼         ▼       ▼       ▼        ▼
      PostgreSQL  Redis    Qdrant   Storage  Embedder  Queue
          │         │         │
          │         │         │
          ▼         ▼         ▼
       metadata   cache    vectors
                              │
                              ▼
                         ┌─────────┐
                         │   LLM   │
                         └─────────┘
```

The key responsibility split is:

| Component           | Responsibility                                                 |
| ------------------- | -------------------------------------------------------------- |
| **FastAPI**         | API, validation, authentication, orchestration                 |
| **PostgreSQL**      | Users, tenants, documents, chunks metadata, permissions, audit |
| **Qdrant**          | Embeddings + vector/hybrid retrieval                           |
| **Redis**           | Cache, rate limiting, locks, short-lived state                 |
| **Object storage**  | Original PDFs/docs                                             |
| **Embedding model** | Text → vectors                                                 |
| **LLM**             | Final answer generation                                        |
| **Worker/queue**    | Async document ingestion                                       |

---

# 2. Project structure

I would use something like:

```text
app/
├── main.py
│
├── api/
│   ├── deps.py
│   ├── auth.py
│   ├── chat.py
│   └── documents.py
│
├── core/
│   ├── config.py
│   ├── security.py
│   ├── logging.py
│   └── exceptions.py
│
├── db/
│   ├── postgres.py
│   ├── models.py
│   └── repositories/
│
├── cache/
│   └── redis.py
│
├── vector/
│   └── qdrant.py
│
├── ingestion/
│   ├── loader.py
│   ├── chunker.py
│   └── pipeline.py
│
├── rag/
│   ├── retriever.py
│   ├── reranker.py
│   ├── prompt.py
│   └── service.py
│
├── llm/
│   ├── gateway.py
│   ├── streaming.py
│   └── retry.py
│
└── schemas/
    ├── chat.py
    └── documents.py
```

This gives you:

```text
Router
   ↓
Service
   ↓
Repository / Infrastructure
```

instead of:

```text
Router
   ↓
Everything
```

---

# 3. PostgreSQL data model

PostgreSQL shouldn't store the actual vector embeddings if Qdrant is your vector database.

I'd keep metadata in PostgreSQL.

For a multi-tenant system:

```text
tenants
   │
   ├── users
   │
   └── documents
          │
          └── chunks
```

Example:

```python
class Document(Base):
    __tablename__ = "documents"

    id = mapped_column(UUID, primary_key=True)
    tenant_id = mapped_column(UUID, index=True)
    filename = mapped_column(String)
    storage_path = mapped_column(String)
    status = mapped_column(String)
    created_at = mapped_column(DateTime)
```

And:

```python
class DocumentChunk(Base):
    __tablename__ = "document_chunks"

    id = mapped_column(UUID, primary_key=True)
    document_id = mapped_column(UUID, index=True)
    tenant_id = mapped_column(UUID, index=True)
    chunk_index = mapped_column(Integer)
    text = mapped_column(Text)
```

The important thing is:

**Every tenant-owned entity carries tenant identity or is reachable through it.**

---

# 4. Document ingestion

I would make ingestion **asynchronous**.

The API should not hold an HTTP request open while processing a 500-page PDF.

```text
POST /documents
       │
       ▼
Store document
       │
       ▼
PostgreSQL
status = PROCESSING
       │
       ▼
Queue
       │
       ▼
Worker
       │
       ├── Load
       ├── Parse
       ├── Clean
       ├── Chunk
       ├── Embed
       └── Index
       │
       ▼
Qdrant
       │
       ▼
PostgreSQL
status = COMPLETED
```

The API immediately returns something like:

```json
{
    "document_id": "123",
    "status": "processing"
}
```

---

# 5. Chunking

The ingestion worker loads the document and creates chunks.

```python
chunks = chunker.split(
    text,
    chunk_size=800,
    overlap=100
)
```

But in production, I wouldn't blindly use fixed-size chunks.

I'd consider:

* headings
* paragraphs
* tables
* semantic boundaries
* document type
* overlap
* metadata

Each chunk should carry metadata:

```python
{
    "tenant_id": "...",
    "document_id": "...",
    "chunk_id": "...",
    "page": 12,
    "section": "Authentication",
}
```

That metadata becomes extremely important during retrieval and authorization.

---

# 6. Embedding + Qdrant

The worker generates embeddings:

```python
vectors = embedding_model.embed(chunks)
```

Then stores them in Qdrant.

Conceptually:

```python
qdrant.upsert(
    collection_name="documents",
    points=[
        {
            "id": chunk_id,
            "vector": vector,
            "payload": {
                "tenant_id": tenant_id,
                "document_id": document_id,
                "chunk_id": chunk_id,
                "page": page,
            }
        }
    ]
)
```

I would use the Qdrant payload for retrieval filters.

For example:

```text
tenant_id = tenant_123
```

This is critical for multi-tenancy.

---

# 7. Retrieval

The query flow is:

```text
User question
      │
      ▼
Query embedding
      │
      ▼
Qdrant
      │
      ▼
Top K candidates
      │
      ▼
Metadata/security filtering
      │
      ▼
Reranker
      │
      ▼
Top N context
      │
      ▼
LLM
```

Example:

```python
async def retrieve(
    question: str,
    tenant_id: str,
):

    query_vector = await embedder.embed(question)

    results = await qdrant.search(
        vector=query_vector,
        limit=20,
        query_filter={
            "tenant_id": tenant_id
        }
    )

    return results
```

I would typically retrieve more candidates than I finally send to the LLM.

For example:

```text
Qdrant → top 20
          ↓
Reranker → top 5
          ↓
LLM
```

This improves context quality.

---

# 8. Hybrid search

For enterprise RAG, I wouldn't automatically rely only on vector similarity.

Consider a query:

> "What is policy ID SEC-2026-17?"

Keyword matching can be very useful.

So I might implement:

```text
             User Query
                 │
        ┌────────┴────────┐
        ▼                 ▼
   Vector Search      Keyword Search
        │                 │
        └────────┬────────┘
                 ▼
            Fusion
                 │
                 ▼
             Reranker
                 │
                 ▼
              Top K
```

Qdrant can support vector and richer retrieval patterns, while PostgreSQL can also be useful for metadata/keyword-oriented filtering depending on the design.

---

# 9. Redis usage

I would **not use Redis as the source of truth**.

I'd use it for things that are temporary or expensive to recompute.

### Query caching

```text
question
   ↓
normalize
   ↓
hash
   ↓
Redis
```

Example:

```python
cache_key = f"rag:{tenant_id}:{hash(question)}"

cached = await redis.get(cache_key)

if cached:
    return cached
```

Then:

```python
await redis.setex(
    cache_key,
    300,
    serialized_result
)
```

But I'd be careful caching final answers when permissions or document versions can change.

---

# 10. Redis for rate limiting

For example:

```text
tenant A
   ↓
100 requests/minute

tenant B
   ↓
500 requests/minute
```

Redis gives all FastAPI instances shared rate-limit state.

This is much better than an in-memory Python dictionary when running multiple replicas.

---

# 11. Redis for distributed locks

Suppose the same document is uploaded twice.

Without coordination:

```text
Worker 1 → process document
Worker 2 → process same document
Worker 3 → process same document
```

You may duplicate work.

A distributed lock can help:

```text
lock:document:123
```

Only one worker proceeds.

---

# 12. RAG service

I'd put orchestration here.

```python
class RAGService:

    async def answer(
        self,
        question: str,
        tenant_id: str,
    ):

        cached = await self.cache.get(
            tenant_id,
            question
        )

        if cached:
            return cached

        documents = await self.retriever.retrieve(
            question=question,
            tenant_id=tenant_id
        )

        reranked = await self.reranker.rank(
            question,
            documents
        )

        context = build_context(reranked)

        answer = await self.llm.generate(
            question=question,
            context=context
        )

        await self.cache.set(
            tenant_id,
            question,
            answer
        )

        return answer
```

The FastAPI route becomes very thin:

```python
@router.post("/chat")
async def chat(
    request: ChatRequest,
    service: RAGService = Depends(get_rag_service),
    user=Depends(get_current_user),
):

    return await service.answer(
        question=request.question,
        tenant_id=user.tenant_id,
    )
```

That's the architecture I'd want.

---

# 13. Security is critical

One of the biggest RAG production mistakes is:

> Retrieve first, check permissions later.

I would avoid that.

Suppose:

```text
Tenant A
 ├── Public document
 └── Confidential document

Tenant B
 └── Public document
```

Tenant B must never retrieve Tenant A's confidential chunks.

Therefore:

```text
User
 ↓
JWT
 ↓
tenant_id
 ↓
authorization
 ↓
Qdrant filter
 ↓
retrieval
```

For example:

```python
query_filter = {
    "must": [
        {
            "key": "tenant_id",
            "match": {
                "value": tenant_id
            }
        }
    ]
}
```

For document-level RBAC, I would add ACL metadata as well.

---

# 14. Prompt construction

After retrieval:

```python
prompt = f"""
You are an enterprise assistant.

Answer using only the provided context.

If the answer cannot be found in the
context, say that you don't have enough
information.

Context:
{context}

Question:
{question}
"""
```

I'd also instruct the model to return citations.

For example:

```json
{
    "answer": "...",
    "citations": [
        {
            "document_id": "...",
            "page": 12
        }
    ]
}
```

This makes the system much easier to evaluate.

---

# 15. Streaming

For a chat endpoint, I'd usually stream the LLM response:

```text
FastAPI
   │
   ▼
RAG retrieval
   │
   ▼
Prompt
   │
   ▼
LLM streaming
   │
   ├── token
   ├── token
   ├── token
   └── done
   │
   ▼
SSE
```

Example:

```python
@router.post("/chat/stream")
async def chat_stream(request: ChatRequest):

    return StreamingResponse(
        rag_service.stream(
            request.question
        ),
        media_type="text/event-stream"
    )
```

---

# 16. Failure handling

I would build resilience into the infrastructure layer.

```text
Qdrant timeout
     ↓
retry + backoff
     ↓
still failing
     ↓
return retrieval service unavailable
```

For LLM:

```text
LLM timeout
     ↓
retry
     ↓
429?
     ↓
respect Retry-After
     ↓
still failing
     ↓
fallback model
```

For Redis:

```text
Redis unavailable
     ↓
don't fail entire RAG request
     ↓
skip cache
     ↓
continue
```

That's an important distinction.

**Redis is usually an optimization, not the source of truth.**

---

# 17. Observability

For production RAG, I'd measure the pipeline separately.

```text
Request
  │
  ├── embedding latency
  ├── Qdrant latency
  ├── reranking latency
  ├── LLM TTFT
  ├── LLM total latency
  ├── token usage
  └── total cost
```

And RAG quality:

```text
retrieval precision
retrieval recall
context precision
context recall
faithfulness
answer relevance
citation accuracy
```

I'd use structured logs with:

```text
request_id
tenant_id
user_id
document_ids
query
model
latency
tokens
cost
retrieval scores
```

while being careful not to log sensitive document content.

---

# 18. Evaluation architecture

I would also create an offline evaluation pipeline:

```text
Golden Dataset
      │
      ▼
RAG Pipeline
      │
      ├── Retrieval metrics
      ├── Context metrics
      ├── Answer metrics
      └── Latency/cost
      │
      ▼
Evaluation Store
```

For example:

```text
Question
Expected documents
Expected answer
Retrieved documents
Generated answer
Faithfulness
Context recall
```

This allows us to compare:

```text
RAG version 1
        vs
RAG version 2
```

before deploying.

---

# 19. Database transactions

I'd make document ingestion stateful:

```text
UPLOADED
   ↓
PROCESSING
   ↓
EMBEDDING
   ↓
INDEXING
   ↓
COMPLETED
```

Failures:

```text
PROCESSING
     ↓
FAILED
```

Store:

```text
status
error_message
retry_count
updated_at
```

This makes worker failures recoverable.

---

# 20. Deployment

For production:

```text
                    Load Balancer
                         │
              ┌──────────┼──────────┐
              ▼          ▼          ▼
          FastAPI     FastAPI    FastAPI
              │
              ▼
          Redis Cluster
              │
        ┌─────┴─────┐
        ▼           ▼
 PostgreSQL       Qdrant
        │           │
        │           │
        └─────┬─────┘
              ▼
          RAG Service
              │
              ▼
          LLM Provider
```

And ingestion separately:

```text
FastAPI
   │
   ▼
Queue
   │
   ├── Worker 1
   ├── Worker 2
   ├── Worker 3
   └── Worker N
```

This allows ingestion to scale independently from query traffic.

---

# 21. The complete request flow

If the interviewer asks **"Walk me through one RAG request"**, I would say:

```text
POST /chat
     │
     ▼
JWT authentication
     │
     ▼
Extract tenant_id + permissions
     │
     ▼
Validate request
     │
     ▼
Check Redis cache
     │
     ├── HIT ────────► return cached answer
     │
     ▼ MISS
Generate query embedding
     │
     ▼
Qdrant vector/hybrid search
     │
     ▼
Tenant + ACL filtering
     │
     ▼
Retrieve top 20
     │
     ▼
Rerank
     │
     ▼
Select top 5
     │
     ▼
Build grounded prompt
     │
     ▼
LLM
     │
     ▼
Stream answer + citations
     │
     ▼
Cache appropriate result
     │
     ▼
Record metrics/traces
```

---

# 22. What I would say in the interview

> **"I would build the RAG API using a layered architecture with FastAPI as the API layer, PostgreSQL as the transactional metadata and authorization store, Qdrant as the vector retrieval layer, and Redis for caching, rate limiting and distributed coordination. Document ingestion would be asynchronous: the API stores metadata and queues a job, and workers load, parse, chunk and embed documents before indexing them in Qdrant. For a query, I authenticate the user, derive the tenant and ACL filters, check the Redis cache, generate the query embedding, retrieve candidates from Qdrant, apply security filters, rerank the results, construct a grounded prompt and stream the LLM response. I would keep the router thin and put orchestration in a RAG service with repository/infrastructure abstractions underneath. In production I'd add retries, timeouts, circuit breakers, rate limiting, observability, evaluation, idempotent ingestion and tenant isolation. The most important security requirement is that tenant and document-level authorization is enforced during retrieval, not after the LLM has already seen the data."**

### The architecture to remember

```text
FastAPI
   ↓
Auth / RBAC
   ↓
RAG Service
   ↓
Redis ──────── cache / rate limit
   ↓
Retriever
   ↓
Qdrant ─────── vector search
   ↓
Reranker
   ↓
Prompt
   ↓
LLM Gateway ── retry / timeout / fallback
   ↓
Streaming response

PostgreSQL ─── metadata / tenants / users /
               documents / permissions / audit

Workers ────── asynchronous ingestion
```

That's a **production-oriented RAG architecture**, rather than just "FastAPI → Qdrant → LLM."
