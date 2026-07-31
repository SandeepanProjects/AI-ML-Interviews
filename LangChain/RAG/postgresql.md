# PostgreSQL in Production AI Systems (Complete Guide with Production-Level Code)

Most beginners think PostgreSQL is only used for storing users.

In **production AI systems**, PostgreSQL is the **system of record**. It stores structured business data, while vector databases like Qdrant store embeddings.

A typical enterprise AI architecture looks like:

```text
                 Enterprise AI Platform

                +-----------------------+
                |      FastAPI API      |
                +-----------------------+
                           │
        ┌──────────────────┴─────────────────┐
        ▼                                    ▼
+-------------------+               +------------------+
|   PostgreSQL      |               |     Qdrant       |
|-------------------|               |------------------|
| Users             |               | Embeddings       |
| Chat Sessions     |               | Vector Search    |
| Chat Messages     |               | Similarity Index |
| Documents         |               | Chunk Metadata   |
| Audit Logs        |               +------------------+
| Permissions       |
| Feedback          |
+-------------------+
```

**Rule of thumb**

* PostgreSQL → structured data
* Qdrant → semantic search

---

# What Does PostgreSQL Store?

Example for an enterprise RAG application:

```text
PostgreSQL

├── Users
├── Organizations
├── Documents
├── Chat Sessions
├── Messages
├── Feedback
├── API Keys
├── Roles
├── Permissions
├── Audit Logs
├── Model Usage
├── Billing
└── Prompt Templates
```

Notice:

**Documents are stored in PostgreSQL.**

Embeddings are stored in Qdrant.

---

# Why Not Store Everything in Qdrant?

Suppose HR uploads a policy.

You need:

```text
Document

Name

Owner

Department

Upload Date

Version

Approval Status
```

These require relational queries.

Qdrant is not designed for this.

---

# Production Folder Structure

```text
app/

    db/

        database.py

        models.py

        repository.py

        session.py

    services/

    api/

    auth/

    monitoring/
```

Separate database logic from business logic.

---

# Install

```bash
pip install sqlalchemy
pip install psycopg[binary]
pip install alembic
```

---

# Database Connection

Use SQLAlchemy 2.x.

```python
# db/database.py

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

DATABASE_URL = (
    "postgresql+psycopg://user:password@localhost/ragdb"
)

engine = create_engine(

    DATABASE_URL,

    pool_size=20,

    max_overflow=10,

    pool_pre_ping=True,

    pool_recycle=1800

)

SessionLocal = sessionmaker(

    autoflush=False,

    autocommit=False,

    bind=engine

)
```

### Why Connection Pooling?

Without pooling:

```text
Every Request

↓

Open Connection

↓

Close Connection
```

Very slow.

With pooling:

```text
Pool

↓

Reuse Existing Connections
```

Much faster.

---

# ORM Models

## User

```python
from sqlalchemy.orm import DeclarativeBase
from sqlalchemy.orm import Mapped
from sqlalchemy.orm import mapped_column

class Base(DeclarativeBase):
    pass


class User(Base):

    __tablename__ = "users"

    id: Mapped[int] = mapped_column(primary_key=True)

    email: Mapped[str]

    organization_id: Mapped[int]
```

---

## Document

```python
class Document(Base):

    __tablename__ = "documents"

    id = mapped_column(primary_key=True)

    filename = mapped_column()

    owner_id = mapped_column()

    department = mapped_column()

    version = mapped_column()

    uploaded_at = mapped_column()

    qdrant_collection = mapped_column()
```

Notice:

Only metadata lives here.

The vectors live in Qdrant.

---

## Chat Session

```python
class ChatSession(Base):

    __tablename__ = "chat_sessions"

    id = mapped_column(primary_key=True)

    user_id = mapped_column()

    created_at = mapped_column()

    title = mapped_column()
```

---

## Chat Message

```python
class ChatMessage(Base):

    __tablename__ = "chat_messages"

    id = mapped_column(primary_key=True)

    session_id = mapped_column()

    role = mapped_column()

    content = mapped_column()

    tokens = mapped_column()

    created_at = mapped_column()
```

This stores conversation history.

---

# Dependency Injection

```python
from fastapi import Depends

def get_db():

    db = SessionLocal()

    try:

        yield db

    finally:

        db.close()
```

API

```python
@app.get("/users")

def get_users(db=Depends(get_db)):

    ...
```

Every request gets one database session.

---

# Repository Pattern

Don't write SQL inside API routes.

Bad

```python
@app.get("/user")

def route():

    db.query(User).all()
```

Good

```python
class UserRepository:

    def __init__(self, db):

        self.db = db

    def get_user(self, user_id):

        return self.db.get(User, user_id)
```

Service layer:

```python
repo = UserRepository(db)

user = repo.get_user(1)
```

---

# Transactions

Suppose uploading a PDF.

Steps:

```text
Insert Document

↓

Insert Chunks

↓

Store Metadata

↓

Commit
```

If one step fails:

Rollback.

```python
try:

    db.add(document)

    db.commit()

except:

    db.rollback()

    raise
```

Never leave partial data.

---

# Alembic Migrations

Never manually change production tables.

Initialize:

```bash
alembic init migrations
```

Create migration:

```bash
alembic revision --autogenerate -m "create documents"
```

Apply:

```bash
alembic upgrade head
```

Production teams use migrations for every schema change.

---

# Relationships

```python
class User(Base):

    __tablename__ = "users"

    id = mapped_column(primary_key=True)

    sessions = relationship(
        "ChatSession"
    )
```

```python
class ChatSession(Base):

    __tablename__ = "chat_sessions"

    user_id = mapped_column(
        ForeignKey("users.id")
    )
```

Now:

```python
user.sessions
```

works automatically.

---

# Indexes

Never scan millions of rows.

```python
Index(
    "idx_department",
    Document.department
)
```

Useful indexes:

* email
* organization_id
* session_id
* uploaded_at
* created_at

---

# Multi-Tenant Design

Enterprise SaaS:

```text
Organizations

↓

Users

↓

Documents

↓

Chat Sessions
```

Every table contains:

```python
organization_id
```

Every query filters:

```python
WHERE organization_id=?
```

Never allow tenants to access each other's data.

---

# Audit Logging

Store every request.

```python
class AuditLog(Base):

    __tablename__ = "audit_logs"

    id = mapped_column(primary_key=True)

    user_id = mapped_column()

    action = mapped_column()

    timestamp = mapped_column()
```

Example:

```text
User uploaded LeavePolicy.pdf

↓

Stored forever
```

Useful for compliance.

---

# Storing Model Usage

```python
class ModelUsage(Base):

    __tablename__ = "model_usage"

    id = mapped_column(primary_key=True)

    model = mapped_column()

    prompt_tokens = mapped_column()

    completion_tokens = mapped_column()

    cost = mapped_column()

    latency = mapped_column()
```

Later:

```sql
SELECT SUM(cost)
FROM model_usage;
```

to compute spending.

---

# Combining PostgreSQL and Qdrant

When uploading a document:

```text
PDF

↓

Store Metadata → PostgreSQL

↓

Chunk

↓

Embedding

↓

Vectors → Qdrant
```

Example:

```python
document = Document(

    filename="Leave.pdf",

    department="HR"

)

db.add(document)

db.commit()

qdrant.upsert(
    ...
)
```

Store the PostgreSQL document ID in the Qdrant payload:

```python
payload = {

    "document_id": document.id,

    "department":"HR"
}
```

Now search results can link back to the relational record.

---

# Production Architecture

```text
               FastAPI

                  │

       ┌──────────┴──────────┐

       ▼                     ▼

 PostgreSQL             Redis Cache

       │

 Document Metadata

 Users

 Sessions

 Billing

 Audit Logs

       │

       ▼

    Qdrant

       │

 Embeddings

 Similarity Search
```

---

# Best Practices

### Use UUIDs

Instead of sequential IDs:

```python
import uuid

id = uuid.uuid4()
```

Better for distributed systems.

---

### Soft Deletes

Instead of deleting:

```sql
DELETE
```

Use:

```python
deleted_at
```

This preserves history.

---

### Optimistic Locking

Prevent concurrent overwrites by adding a `version` column and checking it during updates.

---

### Batch Inserts

Instead of:

```python
for doc in docs:
    db.add(doc)
```

Use:

```python
db.add_all(docs)
db.commit()
```

Much faster.

---

### Environment Variables

Never hard-code credentials.

```python
DATABASE_URL = os.getenv("DATABASE_URL")
```

---

### Health Check

```python
@app.get("/health")

def health(db=Depends(get_db)):

    db.execute(text("SELECT 1"))

    return {
        "status":"healthy"
    }
```

Useful for Kubernetes readiness probes.

---

# Monitoring

Track:

* Active connections
* Slow queries
* Lock waits
* Deadlocks
* CPU
* Storage
* Cache hit ratio
* Replication lag
* Transaction time

These metrics are commonly exported to Prometheus and visualized in Grafana.

---

# Common Interview Questions

### Why PostgreSQL and Qdrant together?

PostgreSQL stores structured business data with ACID guarantees and rich relational queries. Qdrant stores embeddings optimized for semantic similarity search. They complement each other.

---

### Why use SQLAlchemy?

It provides ORM mapping, connection pooling, transactions, migrations, and portability while still allowing raw SQL for performance-critical paths.

---

### Why use Alembic?

To version and safely apply database schema changes across development, staging, and production environments.

---

### Why use connection pooling?

Creating a new database connection for every request is expensive. A connection pool reuses established connections, reducing latency and increasing throughput.

---

# Production Database Schema

```text
Organizations
      │
      ▼
Users
      │
      ▼
Chat Sessions
      │
      ▼
Chat Messages

Documents
      │
      ▼
Document Versions
      │
      ▼
Feedback

Audit Logs

Model Usage

API Keys

Roles

Permissions
```

---

# Senior AI Engineer Interview Answer

> **In production AI systems, PostgreSQL serves as the system of record for structured business data such as users, organizations, documents, chat sessions, permissions, audit logs, and model usage. I use SQLAlchemy 2.x with connection pooling, dependency injection, repository and service layers, and Alembic migrations for schema management. Document metadata is stored in PostgreSQL, while embeddings are stored in Qdrant, with each vector payload referencing the PostgreSQL document ID. All multi-step operations run inside database transactions, every query is scoped by tenant for multi-tenant isolation, and indexes are added for frequently queried columns. I monitor connection pools, slow queries, locks, replication, and transaction latency, and I expose health checks for orchestration platforms such as Kubernetes. This architecture provides strong consistency for business data while allowing Qdrant to handle high-performance semantic retrieval.
