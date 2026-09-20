Absolutely. For a **production AI project**, Redis and background tasks solve different problems:

* **Redis** → fast shared state/cache, rate limiting, job broker, locks, short-lived data.
* **Background worker** → executes slow work outside the HTTP request.
* **FastAPI `BackgroundTasks`** → useful for small post-response tasks, but **not** the right choice for durable long-running AI jobs.
* **Celery + Redis** (or an equivalent durable queue) → better for document ingestion, embedding generation, batch processing, evaluations, etc.

For your enterprise RAG / multi-agent project, I'd use:

```text
                    Client
                       │
                       ▼
                    FastAPI
                       │
          ┌────────────┼─────────────┐
          │            │             │
          ▼            ▼             ▼
       PostgreSQL    Redis       Celery
          │            │             │
          │            │             ▼
          │            │          Worker
          │            │             │
          │            │             ▼
          │            │      Embedding / RAG
          │            │             │
          │            │             ▼
          │            │        PostgreSQL
          │            │        + pgvector
          │            │
          │            └── Cache / Rate Limit
          │
          └──── Source of truth
```

---

# 1. What problem are we solving?

Imagine this API:

```text
POST /documents
```

User uploads a 200-page PDF.

A naive implementation does:

```text
HTTP Request
     │
     ▼
Upload PDF
     │
     ▼
Extract text
     │
     ▼
Chunk
     │
     ▼
Generate embeddings
     │
     ▼
Insert 10,000 chunks
     │
     ▼
Return response
```

The user could wait minutes.

Instead:

```text
POST /documents
       │
       ▼
Store job
       │
       ▼
Queue job
       │
       ▼
Return 202 Accepted
       │
       ▼
             Worker
                │
                ├── Extract
                ├── Chunk
                ├── Embed
                └── Store
```

This is the key production pattern.

---

# 2. Production Folder Structure

I would structure it like this:

```text
enterprise_ai/
│
├── app/
│   │
│   ├── main.py
│   │
│   ├── api/
│   │   └── routes/
│   │       ├── documents.py
│   │       ├── jobs.py
│   │       └── chat.py
│   │
│   ├── core/
│   │   ├── config.py
│   │   ├── database.py
│   │   └── redis.py
│   │
│   ├── models/
│   │   ├── base.py
│   │   ├── document.py
│   │   └── job.py
│   │
│   ├── schemas/
│   │   ├── document.py
│   │   └── job.py
│   │
│   ├── repositories/
│   │   ├── document_repository.py
│   │   └── job_repository.py
│   │
│   ├── services/
│   │   ├── document_service.py
│   │   ├── embedding_service.py
│   │   ├── cache_service.py
│   │   └── rate_limit_service.py
│   │
│   ├── workers/
│   │   ├── celery_app.py
│   │   └── document_tasks.py
│   │
│   └── background/
│       └── lightweight_tasks.py
│
├── tests/
│   ├── unit/
│   └── integration/
│
├── docker-compose.yml
├── Dockerfile
├── requirements.txt
└── .env
```

---

# 3. Redis Architecture

Redis will have several responsibilities.

```text
Redis
│
├── Cache
│
├── Rate limiting
│
├── Celery broker
│
├── Distributed locks
│
└── Temporary job state
```

Don't treat all Redis data the same.

For example:

```text
Cache
TTL → 5 minutes

Rate limit
TTL → 1 minute

Distributed lock
TTL → 30 seconds

Celery queue
→ managed by Celery
```

---

# 4. Docker Compose

We'll run:

```text
PostgreSQL
Redis
FastAPI
Celery Worker
```

## `docker-compose.yml`

```yaml
services:

  postgres:
    image: postgres:16

    environment:
      POSTGRES_DB: ai_db
      POSTGRES_USER: postgres
      POSTGRES_PASSWORD: postgres

    ports:
      - "5432:5432"

    volumes:
      - postgres_data:/var/lib/postgresql/data


  redis:
    image: redis:7-alpine

    ports:
      - "6379:6379"

    command:
      - redis-server
      - --appendonly
      - yes

    volumes:
      - redis_data:/data


  api:
    build: .

    command:
      - uvicorn
      - app.main:app
      - --host
      - 0.0.0.0
      - --port
      - "8000"

    ports:
      - "8000:8000"

    env_file:
      - .env

    depends_on:
      - postgres
      - redis


  worker:
    build: .

    command:
      - celery
      - -A
      - app.workers.celery_app
      - worker
      - --loglevel=INFO

    env_file:
      - .env

    depends_on:
      - postgres
      - redis


volumes:

  postgres_data:

  redis_data:
```

---

# 5. Configuration

## `app/core/config.py`

```python
from pydantic_settings import BaseSettings


class Settings(BaseSettings):

    database_url: str

    redis_url: str = "redis://redis:6379/0"

    celery_broker_url: str = "redis://redis:6379/1"

    celery_result_backend: str = "redis://redis:6379/2"

    cache_ttl: int = 300

    rate_limit_requests: int = 100

    rate_limit_window: int = 60

    class Config:
        env_file = ".env"


settings = Settings()
```

---

# 6. `.env`

```text
DATABASE_URL=postgresql+asyncpg://postgres:postgres@postgres:5432/ai_db

REDIS_URL=redis://redis:6379/0

CELERY_BROKER_URL=redis://redis:6379/1

CELERY_RESULT_BACKEND=redis://redis:6379/2
```

Notice that we use separate Redis logical databases:

```text
Redis DB 0 → application cache

Redis DB 1 → Celery broker

Redis DB 2 → Celery results
```

In a larger production deployment, separate Redis instances/clusters may be preferable for stronger workload isolation.

---

# 7. Redis Client

## `app/core/redis.py`

```python
import redis.asyncio as redis

from app.core.config import settings


redis_client = redis.from_url(
    settings.redis_url,
    encoding="utf-8",
    decode_responses=True,
)


async def get_redis():

    return redis_client
```

---

# 8. Cache Service

Now let's create a reusable cache abstraction.

## `app/services/cache_service.py`

```python
import json

from app.core.redis import redis_client
from app.core.config import settings


class CacheService:

    async def get(
        self,
        key: str
    ):

        value = await redis_client.get(key)

        if value is None:
            return None

        return json.loads(value)

    async def set(
        self,
        key: str,
        value,
        ttl: int | None = None
    ):

        if ttl is None:
            ttl = settings.cache_ttl

        await redis_client.set(
            key,
            json.dumps(value),
            ex=ttl
        )

    async def delete(
        self,
        key: str
    ):

        await redis_client.delete(key)
```

---

# 9. Why Cache AI Responses?

Consider:

```text
User:
"What is the refund policy?"
```

Embedding:

```text
[0.12, -0.42, ...]
```

Vector retrieval:

```text
PostgreSQL + pgvector
```

LLM:

```text
GPT/Claude/etc.
```

This costs time and potentially money.

If the same request comes repeatedly:

```text
User
 ↓
Redis
 ↓
CACHE HIT
 ↓
Return
```

instead of:

```text
User
 ↓
Embedding API
 ↓
PostgreSQL
 ↓
Reranker
 ↓
LLM
 ↓
Response
```

---

# 10. Cache-aside Pattern

The standard pattern is:

```text
              Request
                 │
                 ▼
              Redis?
              /    \
           HIT      MISS
            │         │
            ▼         ▼
         Return    Database
                      │
                      ▼
                    LLM
                      │
                      ▼
                 Redis SET
                      │
                      ▼
                   Return
```

Example:

```python
from app.services.cache_service import CacheService


cache = CacheService()


async def get_answer(
    question: str
):

    cache_key = f"answer:{question}"

    cached = await cache.get(
        cache_key
    )

    if cached:

        return cached

    answer = await expensive_llm_call(
        question
    )

    await cache.set(
        cache_key,
        answer,
        ttl=300
    )

    return answer
```

---

# 11. Production Cache Key

Don't simply use:

```python
f"answer:{question}"
```

In a multi-tenant enterprise system, use something like:

```python
cache_key = (
    f"tenant:{tenant_id}:"
    f"user:{user_id}:"
    f"model:{model_version}:"
    f"query:{query_hash}"
)
```

Why?

Because:

```text
Tenant A
```

must not accidentally receive cached information generated for:

```text
Tenant B
```

---

# 12. Hash the Query

## `app/services/cache_service.py`

Add:

```python
import hashlib


def create_query_hash(
    query: str
) -> str:

    normalized = (
        query.strip()
        .lower()
    )

    return hashlib.sha256(
        normalized.encode("utf-8")
    ).hexdigest()
```

Then:

```python
query_hash = create_query_hash(
    query
)

cache_key = (
    f"tenant:{tenant_id}:"
    f"query:{query_hash}"
)
```

---

# 13. Redis Rate Limiting

Redis is excellent for distributed rate limiting.

Imagine:

```text
10 FastAPI instances
```

You don't want each instance to have its own counter.

Bad:

```text
Instance 1 → counter 10
Instance 2 → counter 10
Instance 3 → counter 10
```

Redis gives you:

```text
             Redis
               │
      ┌────────┼────────┐
      │        │        │
     API1     API2     API3
      │        │        │
      └────────┴────────┘
```

One shared counter.

---

# 14. Rate Limit Service

## `app/services/rate_limit_service.py`

```python
import time

from app.core.redis import redis_client


class RateLimitService:

    async def check(
        self,
        identifier: str,
        limit: int,
        window: int
    ) -> bool:

        key = (
            f"rate_limit:{identifier}:"
            f"{int(time.time()) // window}"
        )

        count = await redis_client.incr(key)

        if count == 1:

            await redis_client.expire(
                key,
                window
            )

        return count <= limit
```

This provides a simple fixed-window limiter.

For production, you may use a Lua-script-based atomic limiter or a more sophisticated sliding-window/token-bucket design.

---

# 15. FastAPI Rate Limiting

Example:

```python
from fastapi import HTTPException

from app.services.rate_limit_service import (
    RateLimitService
)

rate_limiter = RateLimitService()


async def enforce_rate_limit(
    user_id: str
):

    allowed = await rate_limiter.check(
        identifier=user_id,
        limit=100,
        window=60
    )

    if not allowed:

        raise HTTPException(
            status_code=429,
            detail="Rate limit exceeded"
        )
```

Then:

```python
@router.post("/chat")
async def chat(
    request: ChatRequest,
    user=Depends(get_current_user)
):

    await enforce_rate_limit(
        user.id
    )

    ...
```

---

# 16. Now Background Tasks

There are **two different things** you should know for interviews.

## FastAPI BackgroundTasks

```python
from fastapi import BackgroundTasks
```

Good for:

```text
send audit event
write lightweight log
cleanup
small notification
```

Not ideal for:

```text
10-minute PDF processing
100,000 embeddings
large RAG evaluation
GPU inference
long-running agent
```

---

# 17. FastAPI BackgroundTasks Example

## `app/background/lightweight_tasks.py`

```python
import logging


logger = logging.getLogger(__name__)


def write_audit_log(
    user_id: str,
    action: str
):

    logger.info(
        "user=%s action=%s",
        user_id,
        action
    )
```

API:

```python
from fastapi import (
    APIRouter,
    BackgroundTasks
)

from app.background.lightweight_tasks import (
    write_audit_log
)


router = APIRouter()


@router.post("/documents")
async def create_document(
    background_tasks: BackgroundTasks
):

    document_id = "doc-123"

    background_tasks.add_task(
        write_audit_log,
        "user-123",
        "document_created"
    )

    return {
        "document_id": document_id
    }
```

The response can return while the lightweight task is scheduled.

---

# 18. Why FastAPI BackgroundTasks Is Not Enough

Suppose:

```text
Upload PDF
      ↓
50,000 chunks
      ↓
50,000 embeddings
```

Don't do:

```python
background_tasks.add_task(
    process_huge_document
)
```

Why?

Because FastAPI's background task is tied to the application process.

Problems include:

```text
worker restart
deployment
process crash
long-running CPU work
no durable queue
limited retry semantics
no distributed worker pool
```

For serious AI workloads:

```text
FastAPI
   ↓
Celery
   ↓
Redis
   ↓
Worker
```

---

# 19. Celery

Install:

```text
celery
redis
```

Celery gives us:

```text
Producer
   ↓
Redis Broker
   ↓
Queue
   ↓
Worker
   ↓
Task
```

---

# 20. Celery Application

## `app/workers/celery_app.py`

```python
from celery import Celery

from app.core.config import settings


celery_app = Celery(
    "enterprise_ai",
    broker=settings.celery_broker_url,
    backend=settings.celery_result_backend,
)


celery_app.conf.update(

    task_serializer="json",

    result_serializer="json",

    accept_content=[
        "json"
    ],

    timezone="UTC",

    enable_utc=True,

    task_track_started=True,

    task_acks_late=True,

    worker_prefetch_multiplier=1,

    task_reject_on_worker_lost=True,
)
```

These settings matter in production.

---

# 21. What Does `acks_late` Do?

Suppose:

```text
Worker
  ↓
processing document
```

Worker crashes.

With appropriate acknowledgement configuration, the broker can avoid treating the task as successfully completed before the worker finishes it.

Conceptually:

```text
Task
 ↓
Worker receives
 ↓
Processing
 ↓
SUCCESS
 ↓
ACK
```

instead of:

```text
Task
 ↓
ACK immediately
 ↓
Worker crashes
 ↓
Task potentially lost
```

Exactly-once execution is not automatically guaranteed, though. Your tasks should be **idempotent**.

---

# 22. Document Task

## `app/workers/document_tasks.py`

```python
import logging

from app.workers.celery_app import celery_app


logger = logging.getLogger(__name__)


@celery_app.task(
    bind=True,
    autoretry_for=(
        Exception,
    ),
    retry_backoff=True,
    retry_kwargs={
        "max_retries": 3
    }
)
def process_document(
    self,
    document_id: str
):

    logger.info(
        "Processing document %s",
        document_id
    )

    # 1. Download document

    # 2. Extract text

    # 3. Chunk text

    # 4. Generate embeddings

    # 5. Store chunks + vectors

    logger.info(
        "Completed document %s",
        document_id
    )

    return {
        "document_id": document_id,
        "status": "completed"
    }
```

---

# 23. Retry Handling

AI applications depend on external services:

```text
OpenAI
Anthropic
embedding API
object storage
PostgreSQL
Redis
```

Failures happen.

For example:

```text
Embedding API
      │
      ▼
429 Too Many Requests
```

You don't necessarily want:

```text
FAILED
```

immediately.

Use:

```text
Retry
 ↓
backoff
 ↓
Retry
 ↓
backoff
 ↓
Retry
 ↓
Dead letter / failure state
```

Celery supports retry configuration, but production systems should distinguish transient failures from permanent failures rather than retry every exception blindly.

---

# 24. Better Retry Example

```python
from celery import Task


@celery_app.task(
    bind=True,
    max_retries=3
)
def generate_embeddings(
    self,
    document_id: str
):

    try:

        return process_embeddings(
            document_id
        )

    except TemporaryEmbeddingError as exc:

        raise self.retry(
            exc=exc,
            countdown=2 ** self.request.retries
        )
```

This gives approximately:

```text
retry 1 → 1/2 sec
retry 2 → 2/4 sec
retry 3 → 4/8 sec
```

depending on the chosen formula/configuration.

---

# 25. API Creates the Job

## `app/api/routes/documents.py`

```python
from fastapi import APIRouter

from app.workers.document_tasks import (
    process_document
)


router = APIRouter(
    prefix="/documents",
    tags=["documents"]
)


@router.post(
    "/{document_id}/process",
    status_code=202
)
async def process(
    document_id: str
):

    task = process_document.delay(
        document_id
    )

    return {
        "job_id": task.id,
        "document_id": document_id,
        "status": "queued"
    }
```

This is the important production pattern:

```text
POST /documents/doc-123/process

             │
             ▼

         Celery .delay()

             │
             ▼

          Redis Queue

             │
             ▼

           Worker
```

The API doesn't wait.

---

# 26. Why HTTP 202?

Because the operation hasn't completed.

```text
200 OK
```

usually means the requested operation has completed successfully.

For an asynchronous operation:

```text
202 Accepted
```

communicates:

```text
Request accepted
but processing is happening asynchronously.
```

---

# 27. Job Status API

Users need to know:

```text
Is my document ready?
```

Create:

## `app/api/routes/jobs.py`

```python
from fastapi import APIRouter

from celery.result import AsyncResult

from app.workers.celery_app import celery_app


router = APIRouter(
    prefix="/jobs",
    tags=["jobs"]
)


@router.get("/{job_id}")
async def get_job(
    job_id: str
):

    result = AsyncResult(
        job_id,
        app=celery_app
    )

    response = {
        "job_id": job_id,
        "status": result.status
    }

    if result.successful():

        response["result"] = (
            result.result
        )

    if result.failed():

        response["error"] = str(
            result.result
        )

    return response
```

---

# 28. Job Lifecycle

Now:

```text
POST /documents/doc-123/process
             │
             ▼
          202 Accepted
             │
             ▼
         job_id = abc
             │
             ▼
       Redis / Celery
             │
             ▼
           Worker
             │
             ▼
          STARTED
             │
             ▼
         PROCESSING
             │
             ▼
          SUCCESS
```

Client can call:

```text
GET /jobs/abc
```

and receive:

```json
{
  "job_id": "abc",
  "status": "SUCCESS",
  "result": {
    "document_id": "doc-123",
    "status": "completed"
  }
}
```

---

# 29. Better Production Design: Persist Job State in PostgreSQL

Don't make Redis/Celery result storage your only durable job history.

Create:

## `app/models/job.py`

```python
import uuid

from datetime import datetime

from sqlalchemy import (
    String,
    DateTime,
    Text
)

from sqlalchemy.orm import (
    Mapped,
    mapped_column
)

from app.models.base import Base


class Job(Base):

    __tablename__ = "jobs"

    id: Mapped[str] = mapped_column(
        String(36),
        primary_key=True,
        default=lambda: str(uuid.uuid4())
    )

    tenant_id: Mapped[str] = mapped_column(
        String(100),
        index=True
    )

    celery_task_id: Mapped[str] = mapped_column(
        String(100),
        unique=True,
        index=True
    )

    job_type: Mapped[str] = mapped_column(
        String(100)
    )

    status: Mapped[str] = mapped_column(
        String(50),
        default="queued"
    )

    error: Mapped[str | None] = mapped_column(
        Text,
        nullable=True
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime,
        default=datetime.utcnow
    )

    completed_at: Mapped[
        datetime | None
    ] = mapped_column(
        DateTime,
        nullable=True
    )
```

Now you have:

```text
PostgreSQL
     │
     └── jobs
           │
           ├── queued
           ├── started
           ├── completed
           └── failed
```

This gives you durable auditability.

---

# 30. Production Document Processing

The actual task should look more like:

```python
@celery_app.task(
    bind=True,
    max_retries=3
)
def process_document(
    self,
    document_id: str
):

    try:

        update_job_status(
            self.request.id,
            "started"
        )

        document = load_document(
            document_id
        )

        text = extract_text(
            document
        )

        chunks = chunk_document(
            text
        )

        embeddings = generate_embeddings(
            chunks
        )

        save_chunks(
            document_id,
            chunks,
            embeddings
        )

        update_job_status(
            self.request.id,
            "completed"
        )

        return {
            "document_id": document_id,
            "chunks": len(chunks)
        }

    except TemporaryError as exc:

        update_job_status(
            self.request.id,
            "retrying"
        )

        raise self.retry(
            exc=exc,
            countdown=10
        )

    except Exception as exc:

        update_job_status(
            self.request.id,
            "failed",
            error=str(exc)
        )

        raise
```

---

# 31. Don't Put Everything Into One Celery Task

For a large AI project, you can split the pipeline:

```text
Document
   │
   ▼
Extract
   │
   ▼
Chunk
   │
   ▼
Embed
   │
   ▼
Persist
   │
   ▼
Evaluate
```

For example:

```python
extract_task
    |
    v
chunk_task
    |
    v
embedding_task
    |
    v
persist_task
```

Celery primitives can coordinate workflows such as chains, groups and chords.

---

# 32. Parallel Embedding Jobs

Suppose:

```text
100,000 chunks
```

You could split:

```text
100,000
   │
   ├── batch 1 → worker 1
   ├── batch 2 → worker 2
   ├── batch 3 → worker 3
   ├── batch 4 → worker 4
   └── ...
```

Conceptually:

```text
                 Embedding Job
                       │
              ┌────────┼────────┐
              │        │        │
              ▼        ▼        ▼
           Batch 1  Batch 2  Batch 3
              │        │        │
              ▼        ▼        ▼
           Worker   Worker   Worker
              │        │        │
              └────────┼────────┘
                       ▼
                   PostgreSQL
```

But you must respect embedding API rate limits and database capacity.

---

# 33. Redis Distributed Lock

Another important production Redis use case is preventing duplicate work.

Imagine:

```text
Worker 1 → process document X
Worker 2 → process document X
```

You may accidentally generate embeddings twice.

Use a distributed lock.

Conceptually:

```text
Redis

lock:document:123
       │
       ▼
      SET NX
```

Only one worker gets the lock.

Example:

```python
from app.core.redis import redis_client


async def acquire_document_lock(
    document_id: str
):

    key = f"lock:document:{document_id}"

    acquired = await redis_client.set(
        key,
        "1",
        nx=True,
        ex=300
    )

    return acquired
```

Then:

```python
locked = await acquire_document_lock(
    document_id
)

if not locked:

    return {
        "status": "already_processing"
    }
```

Production distributed locks require careful ownership/release semantics; a simple `SET NX EX` pattern should not be treated as a complete lock implementation for every failure scenario.

---

# 34. Idempotency

This is extremely important.

Imagine:

```text
Task
 ↓
Insert 1,000 chunks
 ↓
Worker crashes
 ↓
Celery retries
```

Without idempotency:

```text
chunks
chunks
chunks
chunks
```

You get duplicates.

Instead use:

```text
document_id
+
chunk_index
```

as a uniqueness constraint.

For example:

```sql
UNIQUE(document_id, chunk_index)
```

Then retries don't corrupt your data.

---

# 35. Redis Cache + PostgreSQL + pgvector

Now combine this with your previous PostgreSQL/pgvector architecture:

```text
                         FastAPI
                            │
                            ▼
                         Redis
                      Cache lookup
                       /        \
                    HIT          MISS
                    │              │
                    ▼              ▼
                 Return       PostgreSQL
                                  │
                                  ▼
                              pgvector
                                  │
                                  ▼
                               Reranker
                                  │
                                  ▼
                                  LLM
                                  │
                                  ▼
                               Redis
                               SET
```

---

# 36. Complete AI Request Architecture

For your enterprise AI project:

```text
                           CLIENT
                              │
                              ▼
                         ┌─────────┐
                         │ FastAPI │
                         └────┬────┘
                              │
                 ┌────────────┼────────────┐
                 │            │            │
                 ▼            ▼            ▼
             Auth/RBAC     Rate Limit    Cache
                              │          Redis
                              │            │
                              └─────┬──────┘
                                    │
                                    ▼
                               LangGraph
                                    │
                                    ▼
                                Retrieval
                                    │
                          ┌─────────┴─────────┐
                          │                   │
                          ▼                   ▼
                     PostgreSQL           pgvector
                          │                   │
                          └─────────┬─────────┘
                                    │
                                    ▼
                                 Reranker
                                    │
                                    ▼
                                   LLM
                                    │
                                    ▼
                              Guardrails
                                    │
                                    ▼
                                 Response
```

For long-running work:

```text
                 FastAPI
                    │
                    ▼
                 Celery
                    │
                    ▼
                  Redis
                    │
                    ▼
             ┌──────┴──────┐
             │             │
          Worker 1       Worker 2
             │             │
             └──────┬──────┘
                    │
                    ▼
              PostgreSQL
```

---

# 37. What Goes to Background Workers?

Good candidates:

```text
PDF ingestion
Document parsing
OCR
Chunking
Embedding generation
Bulk indexing
RAG evaluation
Dataset processing
LLM batch jobs
Report generation
Large exports
Long-running agent workflows
Scheduled cleanup
```

Don't put:

```text
simple validation
small DB query
normal API response
tiny calculation
```

into a background queue unnecessarily.

---

# 38. Redis vs PostgreSQL

A senior engineer should clearly distinguish them.

| Requirement           | Redis     | PostgreSQL        |
| --------------------- | --------- | ----------------- |
| Cache                 | ✅         | Possible          |
| Rate limiting         | ✅         | Possible          |
| Distributed locks     | ✅         | Possible          |
| Job broker            | ✅         | Usually not ideal |
| Durable business data | ❌         | ✅                 |
| Transactions          | Limited   | ✅                 |
| Users                 | ❌         | ✅                 |
| Documents             | ❌         | ✅                 |
| Permissions           | ❌         | ✅                 |
| Vector search         | ❌         | pgvector          |
| Temporary state       | ✅         | Possible          |
| Audit history         | Not ideal | ✅                 |

---

# 39. Redis Should Not Become Your Database

Bad architecture:

```text
Everything
   ↓
Redis
```

For example, don't make Redis your permanent source of truth for:

```text
users
financial transactions
documents
audit history
permissions
```

Instead:

```text
PostgreSQL
    ↓
source of truth

Redis
    ↓
cache / transient state / queue support
```

---

# 40. Production Failure Scenario

Imagine Redis goes down.

Your application should not blindly crash every endpoint.

For example:

```python
async def get_cached_answer(
    key: str
):

    try:

        return await cache.get(key)

    except Exception:

        # Cache failure should not
        # necessarily break the request.

        return None
```

Then:

```text
Redis unavailable
      │
      ▼
Cache miss/fallback
      │
      ▼
PostgreSQL / LLM
```

Whether Redis failure should fail closed or open depends on the feature. For rate limiting/security controls, fail-open vs fail-closed must be an explicit risk decision.

---

# 41. Production Observability

Track:

```text
Redis
├── hit rate
├── memory
├── evictions
├── latency
└── connection count

Celery
├── queue depth
├── task latency
├── success rate
├── failure rate
├── retry count
└── worker utilization

AI
├── LLM latency
├── token usage
├── embedding latency
├── cost
└── RAG retrieval latency
```

For example:

```text
Request latency

FastAPI
   │
   ├── Redis = 3ms
   ├── PostgreSQL = 15ms
   ├── Embedding = 80ms
   ├── Reranker = 120ms
   └── LLM = 1.5s
```

Now you know where the bottleneck is.

---

# 42. Health Checks

## `app/main.py`

```python
from fastapi import FastAPI

from app.core.redis import redis_client


app = FastAPI(
    title="Enterprise AI API"
)


@app.get("/health")
async def health():

    return {
        "status": "healthy"
    }


@app.get("/health/redis")
async def redis_health():

    try:

        await redis_client.ping()

        return {
            "status": "healthy",
            "redis": "up"
        }

    except Exception:

        return {
            "status": "unhealthy",
            "redis": "down"
        }
```

For Kubernetes, separate liveness/readiness semantics are preferable, and health endpoints should reflect dependencies according to whether the service can actually accept traffic.

---

# 43. Testing Redis

## `tests/unit/test_cache.py`

```python
import pytest


@pytest.mark.asyncio
async def test_cache_set_get(
    redis_client
):

    await redis_client.set(
        "test:key",
        "hello"
    )

    value = await redis_client.get(
        "test:key"
    )

    assert value == "hello"
```

In real tests, use an isolated Redis instance/container or a controlled test double rather than sharing production-like data.

---

# 44. Testing Celery

Don't test only:

```text
API returns 202
```

Also test:

```text
Task submitted
Task succeeds
Task retries
Task fails
Idempotency
Job status
```

Example:

```python
def test_document_task():

    result = process_document.apply(
        args=["doc-123"]
    )

    assert result.successful()
```

Celery's eager execution mode can be useful for unit tests, while integration tests should exercise the actual broker/worker path.

---

# 45. Security Considerations

Never do:

```python
redis_client.set(
    user_input,
    sensitive_data
)
```

without thinking about:

```text
TTL
PII
tenant isolation
encryption
access controls
memory limits
logging
```

For your financial AI system, particularly avoid putting highly sensitive raw financial information into general-purpose cache keys or logs.

Instead:

```text
tenant_id
+
hash
+
short TTL
```

and minimize sensitive cached payloads.

---

# 46. Kubernetes Production Deployment

Eventually:

```text
                    Load Balancer
                          │
                          ▼
                     FastAPI Pods
                   /      |      \
                  /       |       \
                 ▼        ▼        ▼
              API-1     API-2    API-3
                 │        │        │
                 └────────┼────────┘
                          │
                    Redis Cluster
                          │
                    Celery Queue
                     /         \
                    ▼           ▼
              Worker Pods   Worker Pods
                    │           │
                    └─────┬─────┘
                          ▼
                     PostgreSQL
```

Scale API independently:

```text
FastAPI: 10 replicas
```

and workers independently:

```text
Celery: 20 workers
```

This is one of the major benefits of decoupling HTTP traffic from asynchronous processing.

---

# 47. Autoscaling

Suppose queue depth increases:

```text
Normal:

Queue = 10
Workers = 5
```

Traffic spike:

```text
Queue = 10,000
```

You can scale workers:

```text
Workers = 5
      ↓
Workers = 30
```

The API layer doesn't necessarily need to scale at the same rate.

This is especially useful for:

```text
document ingestion
embedding
batch evaluation
```

---

# 48. The Most Important Distinction

Remember this table for interviews:

| Feature                  | FastAPI BackgroundTasks | Celery + Redis |
| ------------------------ | ----------------------- | -------------- |
| Small task               | ✅                       | ✅              |
| Post-response work       | ✅                       | ✅              |
| Long-running AI job      | ❌                       | ✅              |
| Distributed workers      | ❌                       | ✅              |
| Durable queue semantics  | Limited                 | ✅              |
| Retries                  | Manual/basic            | ✅              |
| Worker scaling           | ❌                       | ✅              |
| Large document ingestion | ❌                       | ✅              |
| Embedding batches        | Not ideal               | ✅              |
| RAG evaluation           | Not ideal               | ✅              |

---

# 49. What I Would Actually Use in Your Project

For your **Enterprise Multi-Agent Financial Advisor Copilot**, I'd use:

```text
                    FastAPI
                       │
                       ▼
                Authentication
                       │
                       ▼
                 Rate Limiting
                       │
                       ▼
                    Redis
                 Cache lookup
                       │
                 ┌─────┴─────┐
                 │           │
               HIT          MISS
                 │           │
                 │           ▼
                 │       LangGraph
                 │           │
                 │           ▼
                 │      PostgreSQL
                 │       + pgvector
                 │           │
                 │           ▼
                 │        Reranker
                 │           │
                 │           ▼
                 │          LLM
                 │           │
                 └─────┬─────┘
                       ▼
                    Response


Long-running work:

FastAPI
   │
   ▼
Celery
   │
   ▼
Redis
   │
   ├──────────┐
   ▼          ▼
Worker 1   Worker 2
   │          │
   └────┬─────┘
        ▼
PostgreSQL + pgvector
```

---

# 50. Interview-Level Answer

If the interviewer asks:

> **"How do you use Redis and background tasks in a production AI application?"**

A strong answer is:

> "I use Redis for low-latency shared state such as response and retrieval caching, distributed rate limiting, short-lived locks and as the broker for asynchronous workloads. For lightweight post-response operations I can use FastAPI BackgroundTasks, but I don't use them for long-running or failure-sensitive AI workloads because they are tied to the application process. For document ingestion, embedding generation, batch evaluation and other long-running workloads, I use a distributed task queue such as Celery with Redis. The API submits the job and immediately returns a 202 with a job ID. Workers consume the task asynchronously, with retries and exponential backoff for transient failures. I persist durable job state in PostgreSQL, make processing idempotent, isolate tenants in cache keys and authorization checks, and monitor queue depth, task latency, failures, retries and Redis health."

That is the key **senior/Staff-level architecture**:

```text
Redis ≠ Database

Redis
 ├── Cache
 ├── Rate limit
 ├── Locks
 └── Queue/Broker

PostgreSQL
 ├── Durable data
 ├── Jobs
 ├── Documents
 ├── ACLs
 └── pgvector

Celery
 ├── Long-running jobs
 ├── Retries
 ├── Workers
 └── Horizontal scaling

FastAPI BackgroundTasks
 └── Small, non-critical post-response work
```

And for your broader project, the pieces you've asked about now fit together as **FastAPI + Redis + Celery + PostgreSQL/pgvector + LangGraph + Guardrails + HITL**, with each component having a clear responsibility rather than putting everything into one application process.
