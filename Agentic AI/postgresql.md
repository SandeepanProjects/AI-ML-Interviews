Yes. For a **production AI/RAG project**, PostgreSQL + pgvector is a very strong architecture because PostgreSQL can hold your **application data, document metadata, chunks, permissions, tenants, conversations, and embeddings**, while pgvector provides vector similarity search.

For the kind of **Enterprise RAG / Financial Advisor Copilot** you've been building, I would structure it like this:

```text
                         FastAPI
                            │
              ┌─────────────┴─────────────┐
              │                           │
         Application DB              Vector Search
              │                           │
              └─────────────┬─────────────┘
                            │
                        PostgreSQL
                            │
                     ┌──────┴──────┐
                     │             │
                  Normal        pgvector
                   SQL          embeddings
                     │             │
                     ▼             ▼
                users/docs      similarity
                tenants         search
                ACLs            top-k chunks
                conversations
```

The important point is that **pgvector is an extension inside PostgreSQL**, not a separate database.

---

# 1. What We Are Building

We'll build this production-style RAG backend:

```text
enterprise_rag/
│
├── app/
│   ├── main.py
│   │
│   ├── api/
│   │   └── routes/
│   │       ├── documents.py
│   │       └── search.py
│   │
│   ├── core/
│   │   ├── config.py
│   │   ├── database.py
│   │   └── exceptions.py
│   │
│   ├── models/
│   │   ├── base.py
│   │   ├── document.py
│   │   └── chunk.py
│   │
│   ├── schemas/
│   │   ├── document.py
│   │   └── search.py
│   │
│   ├── repositories/
│   │   ├── document_repository.py
│   │   └── vector_repository.py
│   │
│   ├── services/
│   │   ├── document_service.py
│   │   ├── embedding_service.py
│   │   └── retrieval_service.py
│   │
│   └── db/
│       └── init.sql
│
├── migrations/
│   └── ...
│
├── docker-compose.yml
├── Dockerfile
├── requirements.txt
└── .env
```

We'll use:

```text
FastAPI
SQLAlchemy Async
PostgreSQL
pgvector
Pydantic
OpenAI embeddings
Alembic
```

---

# 2. Why PostgreSQL + pgvector?

Traditional PostgreSQL:

```sql
SELECT *
FROM documents
WHERE tenant_id = 'tenant-123';
```

Excellent for:

```text
users
documents
permissions
tenants
transactions
metadata
```

But semantic search needs:

```text
"What is the refund policy?"
```

to find:

```text
"Customers can request reimbursement within 30 days..."
```

even though the words don't exactly match.

That's where embeddings come in:

```text
Text
 ↓
Embedding model
 ↓
[0.021, -0.182, 0.734, ...]
 ↓
pgvector
 ↓
Similarity search
```

---

# 3. PostgreSQL Setup

## `docker-compose.yml`

```yaml
services:

  postgres:
    image: pgvector/pgvector:pg16

    container_name: enterprise-postgres

    environment:
      POSTGRES_DB: ai_db
      POSTGRES_USER: postgres
      POSTGRES_PASSWORD: postgres

    ports:
      - "5432:5432"

    volumes:
      - postgres_data:/var/lib/postgresql/data

    healthcheck:
      test:
        [
          "CMD-SHELL",
          "pg_isready -U postgres -d ai_db"
        ]

      interval: 5s
      timeout: 5s
      retries: 10


volumes:

  postgres_data:
```

Start:

```bash
docker compose up -d
```

Then PostgreSQL is available at:

```text
localhost:5432
```

---

# 4. Enable pgvector

Inside PostgreSQL:

```sql
CREATE EXTENSION IF NOT EXISTS vector;
```

You can verify:

```sql
SELECT *
FROM pg_extension
WHERE extname = 'vector';
```

You should see:

```text
vector
```

---

# 5. Requirements

## `requirements.txt`

```text
fastapi
uvicorn[standard]

sqlalchemy
asyncpg
psycopg[binary]

pgvector

pydantic
pydantic-settings

alembic

openai
```

---

# 6. Configuration

## `app/core/config.py`

```python
from pydantic_settings import BaseSettings


class Settings(BaseSettings):

    database_url: str

    openai_api_key: str

    embedding_model: str = (
        "text-embedding-3-small"
    )

    embedding_dimensions: int = 1536

    top_k: int = 5

    class Config:
        env_file = ".env"


settings = Settings()
```

---

# 7. Environment

## `.env`

```text
DATABASE_URL=postgresql+asyncpg://postgres:postgres@localhost:5432/ai_db

OPENAI_API_KEY=your-api-key
```

Don't commit `.env` to Git.

---

# 8. SQLAlchemy Base

## `app/models/base.py`

```python
from sqlalchemy.orm import DeclarativeBase


class Base(DeclarativeBase):

    pass
```

Every model inherits from:

```python
Base
```

---

# 9. Document Model

We need to separate:

```text
Document
```

from:

```text
Chunk
```

A document could be:

```text
employee_handbook.pdf
```

and contain:

```text
chunk 1
chunk 2
chunk 3
...
```

## `app/models/document.py`

```python
import uuid

from datetime import datetime

from sqlalchemy import (
    String,
    Text,
    DateTime
)

from sqlalchemy.orm import (
    Mapped,
    mapped_column,
    relationship
)

from app.models.base import Base


class Document(Base):

    __tablename__ = "documents"

    id: Mapped[str] = mapped_column(
        String(36),
        primary_key=True,
        default=lambda: str(uuid.uuid4())
    )

    tenant_id: Mapped[str] = mapped_column(
        String(100),
        index=True
    )

    title: Mapped[str] = mapped_column(
        String(500)
    )

    source: Mapped[str] = mapped_column(
        String(1000)
    )

    content_hash: Mapped[str] = mapped_column(
        String(64),
        unique=True,
        index=True
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime,
        default=datetime.utcnow
    )

    chunks = relationship(
        "Chunk",
        back_populates="document",
        cascade="all, delete-orphan"
    )
```

---

# 10. Chunk + Vector

This is the most important model.

## `app/models/chunk.py`

```python
import uuid

from sqlalchemy import (
    String,
    Text,
    ForeignKey,
    Integer
)

from sqlalchemy.orm import (
    Mapped,
    mapped_column,
    relationship
)

from pgvector.sqlalchemy import Vector

from app.models.base import Base


class Chunk(Base):

    __tablename__ = "document_chunks"

    id: Mapped[str] = mapped_column(
        String(36),
        primary_key=True,
        default=lambda: str(uuid.uuid4())
    )

    document_id: Mapped[str] = mapped_column(
        ForeignKey(
            "documents.id",
            ondelete="CASCADE"
        ),
        index=True
    )

    tenant_id: Mapped[str] = mapped_column(
        String(100),
        index=True
    )

    chunk_index: Mapped[int] = mapped_column(
        Integer
    )

    content: Mapped[str] = mapped_column(
        Text
    )

    embedding = mapped_column(
        Vector(1536)
    )

    document = relationship(
        "Document",
        back_populates="chunks"
    )
```

The important line is:

```python
Vector(1536)
```

That means:

```text
embedding
    ↓
1536-dimensional vector
```

because `text-embedding-3-small` is configured here with 1536 dimensions.

**Your vector dimension must match the embedding model/output dimension.**

---

# 11. What the Database Looks Like

Conceptually:

```text
documents
─────────────────────────────────────
id
tenant_id
title
source
content_hash
created_at
```

and:

```text
document_chunks
─────────────────────────────────────
id
document_id
tenant_id
chunk_index
content
embedding
```

Example:

```text
id:          chunk-123
document_id: doc-456
tenant_id:   bank-001
chunk_index: 3

content:
"Customers can request a refund within 30 days..."

embedding:
[0.023, -0.187, 0.552, ...]
```

---

# 12. Database Connection

## `app/core/database.py`

```python
from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
    create_async_engine
)

from app.core.config import settings


engine = create_async_engine(

    settings.database_url,

    pool_size=20,

    max_overflow=30,

    pool_pre_ping=True,

    pool_recycle=1800
)


SessionLocal = async_sessionmaker(

    engine,

    class_=AsyncSession,

    expire_on_commit=False
)


async def get_db():

    async with SessionLocal() as session:

        yield session
```

---

# 13. Why Connection Pooling?

Imagine:

```text
10,000 requests
```

You don't want:

```text
10,000 PostgreSQL connections
```

Instead:

```text
FastAPI
   │
   ├── Request 1 ─┐
   ├── Request 2  │
   ├── Request 3  ├── Connection Pool
   ├── Request 4  │
   └── Request N ─┘
                     │
                     ▼
                PostgreSQL
```

For production, pool size should be sized based on your DB capacity and application concurrency rather than blindly making it large.

---

# 14. Embedding Service

## `app/services/embedding_service.py`

```python
from openai import AsyncOpenAI

from app.core.config import settings


class EmbeddingService:

    def __init__(self):

        self.client = AsyncOpenAI(
            api_key=settings.openai_api_key
        )

    async def embed(
        self,
        text: str
    ) -> list[float]:

        response = await (
            self.client.embeddings.create(
                model=settings.embedding_model,
                input=text
            )
        )

        return response.data[0].embedding

    async def embed_many(
        self,
        texts: list[str]
    ) -> list[list[float]]:

        response = await (
            self.client.embeddings.create(
                model=settings.embedding_model,
                input=texts
            )
        )

        return [
            item.embedding
            for item in response.data
        ]
```

---

# 15. Why Batch Embeddings?

Bad:

```python
for chunk in chunks:

    await embed(chunk)
```

If you have:

```text
10,000 chunks
```

that's potentially:

```text
10,000 API calls
```

Better:

```python
await embed_many(chunks)
```

and batch appropriately.

Production ingestion:

```text
PDF
 ↓
Extract
 ↓
Chunk
 ↓
Batch embeddings
 ↓
Bulk insert
 ↓
PostgreSQL + pgvector
```

---

# 16. Document Repository

## `app/repositories/document_repository.py`

```python
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.document import Document


class DocumentRepository:

    def __init__(
        self,
        session: AsyncSession
    ):

        self.session = session

    async def create(
        self,
        document: Document
    ):

        self.session.add(
            document
        )

        await self.session.flush()

        return document
```

Notice:

```python
flush()
```

rather than immediately committing.

The service can then insert:

```text
Document
+
Chunks
```

in one transaction.

---

# 17. Vector Repository

This is where pgvector is actually used.

## `app/repositories/vector_repository.py`

```python
from sqlalchemy import select

from sqlalchemy.ext.asyncio import AsyncSession

from app.models.chunk import Chunk


class VectorRepository:

    def __init__(
        self,
        session: AsyncSession
    ):

        self.session = session

    async def similarity_search(
        self,
        query_embedding: list[float],
        tenant_id: str,
        limit: int = 5
    ):

        distance = Chunk.embedding.cosine_distance(
            query_embedding
        )

        statement = (

            select(
                Chunk,
                distance.label("distance")
            )

            .where(
                Chunk.tenant_id == tenant_id
            )

            .order_by(
                distance
            )

            .limit(limit)
        )

        result = await self.session.execute(
            statement
        )

        return result.all()
```

This:

```python
Chunk.embedding.cosine_distance(
    query_embedding
)
```

is the key pgvector operation.

---

# 18. What Is Cosine Distance?

Suppose:

```text
User question embedding:

A = [0.2, 0.7, 0.3]

Document embedding:

B = [0.2, 0.6, 0.4]
```

pgvector calculates how close they are.

Conceptually:

```text
distance → small
    ↓
semantically similar

distance → large
    ↓
semantically different
```

For cosine similarity:

```text
similarity ≈ 1
```

means very similar vectors.

---

# 19. Raw SQL Version

It's useful to understand what SQLAlchemy is doing.

Conceptually:

```sql
SELECT
    id,
    content,
    embedding <=> :query_embedding AS distance
FROM document_chunks
WHERE tenant_id = :tenant_id
ORDER BY embedding <=> :query_embedding
LIMIT 5;
```

The operator:

```text
<=>
```

is pgvector's cosine-distance operator.

Other common operators include:

```text
<->    Euclidean/L2 distance
<#>    negative inner product
<=>    cosine distance
```

---

# 20. Tenant Isolation

This is **extremely important** for enterprise AI.

Never do:

```sql
SELECT *
FROM document_chunks
ORDER BY embedding <=> query_vector
LIMIT 5;
```

if multiple customers share the database.

Otherwise:

```text
Tenant A
   │
   ▼
could retrieve
   │
   ▼
Tenant B documents
```

Instead:

```sql
WHERE tenant_id = :tenant_id
```

before similarity search.

In our repository:

```python
.where(
    Chunk.tenant_id == tenant_id
)
```

This is a fundamental enterprise RAG security control.

---

# 21. Document Service

## `app/services/document_service.py`

```python
import hashlib
import uuid

from sqlalchemy.ext.asyncio import AsyncSession

from app.models.document import Document
from app.models.chunk import Chunk

from app.services.embedding_service import (
    EmbeddingService
)


class DocumentService:

    def __init__(
        self,
        session: AsyncSession
    ):

        self.session = session

        self.embedding_service = (
            EmbeddingService()
        )

    async def ingest(
        self,
        tenant_id: str,
        title: str,
        source: str,
        chunks: list[str]
    ):

        content_hash = hashlib.sha256(
            "".join(chunks).encode()
        ).hexdigest()

        document = Document(

            id=str(uuid.uuid4()),

            tenant_id=tenant_id,

            title=title,

            source=source,

            content_hash=content_hash
        )

        self.session.add(
            document
        )

        await self.session.flush()

        embeddings = (
            await self.embedding_service
            .embed_many(chunks)
        )

        for index, (
            content,
            embedding
        ) in enumerate(
            zip(chunks, embeddings)
        ):

            chunk = Chunk(

                id=str(uuid.uuid4()),

                document_id=document.id,

                tenant_id=tenant_id,

                chunk_index=index,

                content=content,

                embedding=embedding
            )

            self.session.add(
                chunk
            )

        await self.session.commit()

        return document
```

---

# 22. Ingestion Flow

This code produces:

```text
                PDF
                 │
                 ▼
          Text extraction
                 │
                 ▼
              Chunking
                 │
                 ▼
       ┌───────────────────┐
       │ "Refund policy..." │
       │ "Eligibility..."  │
       │ "Exceptions..."   │
       └─────────┬─────────┘
                 │
                 ▼
          Embedding Model
                 │
                 ▼
       ┌────────────────────┐
       │ [0.21, -0.3, ...] │
       │ [0.72,  0.1, ...] │
       │ [0.11, -0.8, ...] │
       └──────────┬─────────┘
                  │
                  ▼
             PostgreSQL
                  │
                  ▼
              pgvector
```

---

# 23. Retrieval Service

## `app/services/retrieval_service.py`

```python
from sqlalchemy.ext.asyncio import AsyncSession

from app.repositories.vector_repository import (
    VectorRepository
)

from app.services.embedding_service import (
    EmbeddingService
)


class RetrievalService:

    def __init__(
        self,
        session: AsyncSession
    ):

        self.vector_repository = (
            VectorRepository(session)
        )

        self.embedding_service = (
            EmbeddingService()
        )

    async def search(
        self,
        query: str,
        tenant_id: str,
        top_k: int = 5
    ):

        query_embedding = (
            await self.embedding_service
            .embed(query)
        )

        rows = await (
            self.vector_repository
            .similarity_search(
                query_embedding=query_embedding,
                tenant_id=tenant_id,
                limit=top_k
            )
        )

        return [
            {
                "chunk_id": chunk.id,
                "document_id": chunk.document_id,
                "content": chunk.content,
                "distance": float(distance)
            }
            for chunk, distance in rows
        ]
```

---

# 24. Search API

## `app/schemas/search.py`

```python
from pydantic import BaseModel


class SearchRequest(BaseModel):

    query: str

    top_k: int = 5
```

---

## `app/api/routes/search.py`

```python
from fastapi import (
    APIRouter,
    Depends
)

from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_db

from app.schemas.search import SearchRequest

from app.services.retrieval_service import (
    RetrievalService
)


router = APIRouter(
    prefix="/search",
    tags=["search"]
)


@router.post("")
async def search(
    request: SearchRequest,
    db: AsyncSession = Depends(get_db)
):

    service = RetrievalService(db)

    results = await service.search(

        query=request.query,

        tenant_id="tenant-123",

        top_k=request.top_k
    )

    return {
        "results": results
    }
```

In real production code, `tenant_id` should come from authenticated identity/claims, not a hardcoded value.

---

# 25. Document API

## `app/schemas/document.py`

```python
from pydantic import BaseModel


class DocumentIngestRequest(BaseModel):

    tenant_id: str

    title: str

    source: str

    chunks: list[str]
```

---

## `app/api/routes/documents.py`

```python
from fastapi import (
    APIRouter,
    Depends
)

from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_db

from app.schemas.document import (
    DocumentIngestRequest
)

from app.services.document_service import (
    DocumentService
)


router = APIRouter(
    prefix="/documents",
    tags=["documents"]
)


@router.post("")
async def ingest_document(

    request: DocumentIngestRequest,

    db: AsyncSession = Depends(get_db)
):

    service = DocumentService(db)

    document = await service.ingest(

        tenant_id=request.tenant_id,

        title=request.title,

        source=request.source,

        chunks=request.chunks
    )

    return {
        "document_id": document.id
    }
```

---

# 26. Main Application

## `app/main.py`

```python
from fastapi import FastAPI

from app.api.routes.search import (
    router as search_router
)

from app.api.routes.documents import (
    router as documents_router
)


app = FastAPI(
    title="Enterprise PostgreSQL RAG API"
)


app.include_router(
    documents_router
)

app.include_router(
    search_router
)


@app.get("/health")
async def health():

    return {
        "status": "healthy"
    }
```

Run:

```bash
uvicorn app.main:app --reload
```

---

# 27. Inserting a Document

Call:

```http
POST /documents
```

with:

```json
{
  "tenant_id": "bank-001",
  "title": "Refund Policy",
  "source": "refund-policy.pdf",
  "chunks": [
    "Customers can request a refund within 30 days.",
    "Refunds are available only for eligible transactions.",
    "Some transaction categories are excluded."
  ]
}
```

The system does:

```text
API
 ↓
DocumentService
 ↓
EmbeddingService
 ↓
Embedding API
 ↓
1536-dimensional vectors
 ↓
PostgreSQL
 ↓
pgvector
```

---

# 28. Search

Request:

```http
POST /search
```

```json
{
  "query": "How long do I have to request a refund?",
  "top_k": 5
}
```

Flow:

```text
Question
   │
   ▼
Embedding
   │
   ▼
Query Vector
   │
   ▼
pgvector
   │
   ▼
Cosine Distance
   │
   ▼
Top K chunks
```

Example response:

```json
{
  "results": [
    {
      "chunk_id": "abc",
      "document_id": "doc-123",
      "content": "Customers can request a refund within 30 days.",
      "distance": 0.12
    }
  ]
}
```

---

# 29. Add a Vector Index

This is important for production scale.

Without an index:

```text
1 million vectors
       ↓
compare against everything
       ↓
slow
```

With an ANN index:

```text
1 million vectors
       ↓
vector index
       ↓
candidate search
       ↓
top K
```

pgvector supports approximate nearest-neighbor indexes such as:

```text
HNSW
IVFFlat
```

For many modern workloads, HNSW is a strong default to evaluate.

---

# 30. HNSW Index

SQL:

```sql
CREATE INDEX idx_chunks_embedding_hnsw
ON document_chunks
USING hnsw (embedding vector_cosine_ops);
```

This tells PostgreSQL/pgvector:

```text
Use HNSW
+
cosine distance
```

For production, tune the index parameters and benchmark against your actual corpus and workload.

---

# 31. Add Tenant Index

We also want:

```sql
CREATE INDEX idx_chunks_tenant
ON document_chunks(tenant_id);
```

And often a composite/filtering strategy should be evaluated based on your query patterns.

Why?

Because our query is:

```sql
WHERE tenant_id = ?
ORDER BY embedding <=> ?
LIMIT 5
```

Filtering and vector search need to work efficiently together.

---

# 32. Alembic

Don't manually run SQL migrations in production.

Use:

```text
Alembic
```

Structure:

```text
migrations/
├── env.py
├── script.py.mako
└── versions/
    ├── 001_initial.py
    └── 002_add_vector_index.py
```

Initialize:

```bash
alembic init migrations
```

Then:

```bash
alembic revision --autogenerate -m "create documents and chunks"
```

Apply:

```bash
alembic upgrade head
```

---

# 33. Vector Index Migration

Example migration:

```python
from alembic import op


def upgrade():

    op.execute(
        """
        CREATE INDEX IF NOT EXISTS
        idx_chunks_embedding_hnsw
        ON document_chunks
        USING hnsw (
            embedding vector_cosine_ops
        );
        """
    )


def downgrade():

    op.execute(
        """
        DROP INDEX IF EXISTS
        idx_chunks_embedding_hnsw;
        """
    )
```

---

# 34. A Better Production Data Model

For your enterprise project, I'd actually go beyond the simple model.

```text
tenants
│
├── users
│
├── documents
│     │
│     └── document_chunks
│
├── conversations
│
├── messages
│
└── permissions
```

For example:

```text
tenants
────────────────────
id
name


documents
────────────────────
id
tenant_id
title
source
content_hash
version
created_at


document_chunks
────────────────────
id
document_id
tenant_id
chunk_index
content
embedding
metadata


users
────────────────────
id
tenant_id
role


document_permissions
────────────────────
document_id
user_id
access_level
```

Then retrieval becomes:

```text
User
 ↓
Tenant
 ↓
Authorization
 ↓
Allowed documents
 ↓
Vector search
 ↓
Top K
```

This is much more appropriate for enterprise RAG.

---

# 35. Metadata

Don't store only:

```text
content
embedding
```

Store useful metadata:

```json
{
  "department": "finance",
  "document_type": "policy",
  "classification": "internal",
  "language": "en",
  "version": "3",
  "page": 17
}
```

You can have a PostgreSQL JSONB column:

```python
from sqlalchemy.dialects.postgresql import JSONB

metadata = mapped_column(
    JSONB,
    default=dict
)
```

Then:

```text
embedding
+
metadata
```

becomes powerful.

---

# 36. Metadata Filtering

Imagine the user is only allowed:

```text
department = finance
classification = internal
tenant = bank-001
```

Your retrieval query can apply filters before/alongside vector ranking.

Conceptually:

```sql
SELECT *
FROM document_chunks
WHERE tenant_id = :tenant
  AND metadata->>'department' = 'finance'
ORDER BY embedding <=> :query_embedding
LIMIT 5;
```

This is much safer than retrieving everything and filtering afterward.

---

# 37. Hybrid Search

Production RAG often shouldn't rely only on vectors.

Suppose the user searches:

```text
"Policy ID FIN-2026-001"
```

Keyword search can be better for exact identifiers.

So:

```text
                 Query
                   │
           ┌───────┴────────┐
           │                │
           ▼                ▼
      Vector Search     Keyword Search
       pgvector          PostgreSQL FTS
           │                │
           └───────┬────────┘
                   ▼
                 Fusion
                   │
                   ▼
                Reranker
                   │
                   ▼
                 Top K
```

PostgreSQL is particularly attractive here because you can combine:

```text
relational queries
+
full-text search
+
vector search
```

in one platform.

---

# 38. Production RAG Retrieval

For your enterprise project, I'd make retrieval:

```text
User Query
    │
    ▼
Authentication
    │
    ▼
Tenant Resolution
    │
    ▼
ACL Filter
    │
    ▼
Query Embedding
    │
    ├───────────────┐
    ▼               ▼
pgvector          PostgreSQL FTS
    │               │
    └───────┬───────┘
            ▼
        Candidate Set
            │
            ▼
         Reranker
            │
            ▼
       Top 5 Documents
            │
            ▼
        Context Builder
            │
            ▼
            LLM
```

---

# 39. Reranking

Vector search might produce:

```text
1. score 0.82
2. score 0.80
3. score 0.79
4. score 0.77
5. score 0.75
```

But semantic similarity doesn't always mean:

```text
best answer relevance
```

So production RAG can do:

```text
pgvector
   ↓
top 20
   ↓
reranker
   ↓
top 5
```

This reduces the amount of context sent to the LLM.

---

# 40. PostgreSQL Transactions

This is another important production concept.

Suppose ingestion does:

```text
Create document
       ↓
Generate embeddings
       ↓
Insert chunks
```

You don't want:

```text
Document created
       ↓
embedding generation fails
       ↓
half-created database state
```

Use a transaction.

Conceptually:

```python
try:

    document = ...

    session.add(document)

    await session.flush()

    # insert chunks

    await session.commit()

except Exception:

    await session.rollback()

    raise
```

The service can be written more explicitly:

```python
async def ingest(...):

    try:

        ...

        await self.session.commit()

    except Exception:

        await self.session.rollback()

        raise
```

---

# 41. Don't Hold DB Transactions During Slow LLM Calls

This is a subtle but important production issue.

Avoid:

```text
BEGIN TRANSACTION
      ↓
call embedding API
      ↓
wait 3 seconds
      ↓
call another API
      ↓
INSERT
      ↓
COMMIT
```

because the database transaction stays open while waiting on external services.

Better:

```text
Extract
 ↓
Chunk
 ↓
Embedding API
 ↓
Validate
 ↓
Short DB transaction
 ↓
Bulk insert
 ↓
Commit
```

This reduces database lock/connection pressure.

---

# 42. Bulk Insert

If you have:

```text
100,000 chunks
```

don't necessarily do:

```python
session.add(chunk)
```

100,000 times with frequent commits.

Use bulk insertion strategies and tune batch sizes.

Conceptually:

```text
100,000 chunks
      ↓
batch 1 → 1,000
batch 2 → 1,000
...
batch 100
      ↓
PostgreSQL
```

The optimal batch size depends on row size, network, database capacity and workload.

---

# 43. Duplicate Document Detection

We already added:

```python
content_hash
```

Why?

Suppose the same PDF is uploaded twice.

```text
policy.pdf
```

has:

```text
SHA256 = abc123
```

Upload again:

```text
SHA256 = abc123
```

Then:

```text
already exists
```

You can avoid:

```text
duplicate embeddings
duplicate chunks
duplicate storage
duplicate cost
```

---

# 44. Versioning

Enterprise documents change.

Example:

```text
Refund Policy v1
Refund Policy v2
Refund Policy v3
```

Don't blindly overwrite historical data.

Use:

```text
document_id
version
is_active
```

or an immutable version model.

Then retrieval can select:

```text
latest active version
```

while preserving audit history.

---

# 45. Soft Delete

Don't necessarily:

```sql
DELETE FROM documents;
```

immediately.

Use:

```text
deleted_at
```

or:

```text
is_active
```

Then retrieval:

```sql
WHERE is_active = true
```

This helps with:

```text
audit
rollback
compliance
debugging
document lifecycle
```

---

# 46. PostgreSQL + pgvector vs Qdrant

Since your existing project has used Qdrant, this is an important architectural decision.

### PostgreSQL + pgvector

```text
Application DB
       +
Vector DB
       ↓
same PostgreSQL
```

Advantages:

```text
simple architecture
ACID transactions
SQL
joins
metadata
permissions
FTS
vectors
fewer services
```

### Qdrant

```text
PostgreSQL
     +
Qdrant
```

Advantages can include:

```text
specialized vector retrieval
vector-native capabilities
independent scaling
dedicated vector workloads
```

The decision depends on workload, scale, operational requirements, and team expertise.

---

# 47. When I'd Use PostgreSQL + pgvector

For an enterprise RAG application where:

```text
document count isn't enormous
+
metadata relationships are important
+
ACLs are complex
+
transactions matter
+
you already use PostgreSQL
```

PostgreSQL + pgvector can be an excellent architecture.

Example:

```text
PostgreSQL
├── users
├── tenants
├── permissions
├── documents
├── conversations
├── messages
├── chunks
└── embeddings
```

Very clean.

---

# 48. When I'd Consider a Dedicated Vector Database

If vector retrieval becomes a dominant workload with requirements such as:

```text
very large vector corpus
high vector-QPS
specialized ANN tuning
independent vector scaling
```

then:

```text
PostgreSQL
       +
Qdrant
```

can make sense.

Architecture:

```text
              FastAPI
                 │
       ┌─────────┴─────────┐
       │                   │
       ▼                   ▼
 PostgreSQL             Qdrant
       │                   │
 metadata               vectors
 ACL                    embeddings
 users                  similarity
 tenants
```

---

# 49. Production Architecture I'd Recommend for Your Project

For your **Enterprise Multi-Agent Financial Advisor Copilot**, a mature architecture could be:

```text
                         ┌───────────────┐
                         │    Client     │
                         └───────┬───────┘
                                 │
                                 ▼
                         ┌───────────────┐
                         │    FastAPI    │
                         └───────┬───────┘
                                 │
                         Authentication
                                 │
                         Tenant Resolution
                                 │
                         RBAC / ABAC
                                 │
                                 ▼
                         ┌───────────────┐
                         │   LangGraph   │
                         └───────┬───────┘
                                 │
                            Retrieval
                                 │
                  ┌──────────────┴──────────────┐
                  │                             │
                  ▼                             ▼
             PostgreSQL                    pgvector
                  │                             │
             Documents                      Embeddings
             Users                          Similarity
             Tenants                        Search
             ACLs
                  │                             │
                  └──────────────┬──────────────┘
                                 │
                                 ▼
                              Reranker
                                 │
                                 ▼
                               Context
                                 │
                                 ▼
                                LLM
                                 │
                                 ▼
                         Output Guardrails
                                 │
                                 ▼
                              Response
```

---

# 50. Add Redis

PostgreSQL + pgvector doesn't mean Redis isn't useful.

Use Redis for:

```text
conversation cache
rate limiting
embedding cache
retrieval cache
session data
distributed locks
```

For example:

```text
Query
 ↓
Normalize
 ↓
Redis cache?
 ├── HIT → return
 │
 └── MISS
       ↓
   pgvector
       ↓
   Redis SET
       ↓
   return
```

But don't use Redis as the durable source of truth for your document corpus.

---

# 51. Add Guardrails

Connecting this with your previous question:

```text
User
 ↓
Input Guardrail
 ↓
Authentication
 ↓
Tenant / RBAC
 ↓
Query
 ↓
Embedding
 ↓
PostgreSQL + pgvector
 ↓
Document Guardrail
 ↓
Reranker
 ↓
LLM
 ↓
Output Guardrail
 ↓
Response
```

This gives you:

```text
Security
+
RAG
+
Vector Search
+
Governance
```

---

# 52. Add HITL for High-Risk Actions

And connecting it to your previous HITL design:

```text
User
 ↓
FastAPI
 ↓
Guardrails
 ↓
LangGraph
 ↓
PostgreSQL + pgvector
 ↓
LLM
 ↓
Tool decision
 ↓
Risk Engine
       │
       ├── low risk → execute
       │
       └── high risk
              ↓
             HITL
              ↓
           PostgreSQL
              ↓
            Human
           /      \
       Reject     Approve
                    │
                    ▼
                 Execute
```

Now PostgreSQL is doing multiple jobs:

```text
Application database
+
Workflow persistence
+
Approval records
+
Document metadata
+
Vector storage
```

---

# 53. Production Folder Structure — Final Version

For your project, I would ultimately evolve the structure into:

```text
enterprise_ai/
│
├── app/
│
│   ├── main.py
│
│   ├── api/
│   │   └── routes/
│   │       ├── chat.py
│   │       ├── documents.py
│   │       ├── search.py
│   │       └── approvals.py
│
│   ├── core/
│   │   ├── config.py
│   │   ├── database.py
│   │   ├── security.py
│   │   └── logging.py
│
│   ├── models/
│   │   ├── base.py
│   │   ├── user.py
│   │   ├── tenant.py
│   │   ├── document.py
│   │   ├── chunk.py
│   │   ├── conversation.py
│   │   └── approval.py
│
│   ├── schemas/
│   │   ├── chat.py
│   │   ├── document.py
│   │   ├── search.py
│   │   └── approval.py
│
│   ├── repositories/
│   │   ├── document_repository.py
│   │   ├── vector_repository.py
│   │   ├── user_repository.py
│   │   └── approval_repository.py
│
│   ├── services/
│   │   ├── document_service.py
│   │   ├── embedding_service.py
│   │   ├── retrieval_service.py
│   │   ├── reranking_service.py
│   │   └── chat_service.py
│
│   ├── guardrails/
│   │   ├── input/
│   │   ├── retrieval/
│   │   ├── tools/
│   │   └── output/
│
│   ├── workflows/
│   │   ├── state.py
│   │   ├── graph.py
│   │   └── nodes/
│
│   └── tools/
│       ├── account_tool.py
│       └── transaction_tool.py
│
├── migrations/
│   └── versions/
│
├── tests/
│   ├── unit/
│   ├── integration/
│   └── e2e/
│
├── docker-compose.yml
├── Dockerfile
├── requirements.txt
├── alembic.ini
└── .env
```

---

# 54. The Complete Production Flow

This is the flow I want you to remember for interviews:

```text
                        USER QUERY
                            │
                            ▼
                     ┌─────────────┐
                     │   FastAPI   │
                     └──────┬──────┘
                            │
                            ▼
                      Authentication
                            │
                            ▼
                        Tenant ID
                            │
                            ▼
                         RBAC/ACL
                            │
                            ▼
                       Input Guard
                            │
                            ▼
                     Query Embedding
                            │
                            ▼
                 ┌────────────────────┐
                 │     PostgreSQL     │
                 │                    │
                 │   ┌────────────┐   │
                 │   │  pgvector  │   │
                 │   └─────┬──────┘   │
                 │         │          │
                 │      similarity    │
                 │       search       │
                 └─────────┬──────────┘
                           │
                           ▼
                      Top 20 chunks
                           │
                           ▼
                        Reranker
                           │
                           ▼
                       Top 5 chunks
                           │
                           ▼
                   Document Guardrail
                           │
                           ▼
                       Context
                           │
                           ▼
                          LLM
                           │
                           ▼
                   Output Guardrails
                           │
                           ▼
                        Response
```

---

# 55. The Interview Answer

If an interviewer asks:

> **"How have you implemented PostgreSQL and pgvector in a production RAG system?"**

You can answer:

> "I use PostgreSQL as the system of record for tenants, users, documents, document metadata, ACLs, conversations and document chunks, and use pgvector to store the embedding associated with each chunk. During ingestion, documents are parsed and chunked, embeddings are generated in batches, and the document metadata and chunks are persisted transactionally. For retrieval, I generate an embedding for the query and perform cosine-similarity search using pgvector, while applying tenant and authorization filters as part of the database query rather than filtering results afterward. For scale, I use connection pooling, HNSW or another appropriate ANN index, batching, query monitoring and carefully tuned database resources. For enterprise workloads I also combine vector search with PostgreSQL full-text search and reranking when appropriate. PostgreSQL remains the source of truth, while Redis can be used for caching and rate limiting."

Then explain:

```text
PostgreSQL
     │
     ├── users
     ├── tenants
     ├── permissions
     ├── documents
     ├── chunks
     └── conversations
             │
             ▼
          pgvector
             │
             ▼
       ANN similarity
             │
             ▼
          Reranker
             │
             ▼
             LLM
```

That is the architecture I'd use as the baseline for your **production RAG / multi-agent financial copilot**.
