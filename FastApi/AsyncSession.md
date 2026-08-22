These three questions are closely related. For a **senior FastAPI interview**, explain them as one lifecycle:

```text
FastAPI Request
      ↓
Depends(get_db)
      ↓
AsyncSession
      ↓
SQLAlchemy Engine
      ↓
Connection Pool
      ↓
PostgreSQL
```

---

# 1. What is `AsyncSession`?

`AsyncSession` is SQLAlchemy's **asynchronous ORM session**.

It allows an async FastAPI application to perform database operations without blocking the event loop while waiting for the database.

Import:

```python
from sqlalchemy.ext.asyncio import AsyncSession
```

Example:

```python
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

async def get_user(
    db: AsyncSession,
    user_id: int,
):
    result = await db.execute(
        select(User).where(
            User.id == user_id
        )
    )

    return result.scalar_one_or_none()
```

Notice:

```python
await db.execute(...)
```

The database operation is asynchronous.

---

## `Session` vs `AsyncSession`

### Synchronous SQLAlchemy

```python
from sqlalchemy.orm import Session

def get_user(
    db: Session,
    user_id: int,
):
    result = db.execute(
        select(User).where(
            User.id == user_id
        )
    )

    return result.scalar_one_or_none()
```

### Asynchronous SQLAlchemy

```python
from sqlalchemy.ext.asyncio import AsyncSession

async def get_user(
    db: AsyncSession,
    user_id: int,
):
    result = await db.execute(
        select(User).where(
            User.id == user_id
        )
    )

    return result.scalar_one_or_none()
```

The important difference is:

```text
Session
   ↓
Synchronous DB operations

AsyncSession
   ↓
Asynchronous DB operations
```

For an async FastAPI application, `AsyncSession` is commonly used with an async database driver such as `asyncpg`.

---

# 2. How do you create an `AsyncSession`?

You generally don't create it manually inside every endpoint.

Instead, configure SQLAlchemy once.

## Step 1 — Create the async engine

```python
# app/db/session.py

from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

DATABASE_URL = (
    "postgresql+asyncpg://"
    "postgres:password@localhost:5432/mydb"
)

engine = create_async_engine(
    DATABASE_URL,
    pool_size=10,
    max_overflow=20,
    pool_pre_ping=True,
    echo=False,
)
```

The engine manages the connection pool.

```text
                SQLAlchemy Engine
                       │
                Connection Pool
                       │
          ┌────────────┼────────────┐
          ↓            ↓            ↓
      Connection    Connection   Connection
          │            │            │
          └────────────┼────────────┘
                       ↓
                   PostgreSQL
```

---

## Step 2 — Create the session factory

```python
AsyncSessionLocal = async_sessionmaker(
    bind=engine,
    class_=AsyncSession,
    expire_on_commit=False,
)
```

Now `AsyncSessionLocal` is a factory that creates sessions.

---

# 3. How do you inject a database session?

Create a FastAPI dependency.

```python
# app/db/dependencies.py

from collections.abc import AsyncGenerator

from sqlalchemy.ext.asyncio import AsyncSession

from app.db.session import AsyncSessionLocal


async def get_db() -> AsyncGenerator[
    AsyncSession,
    None,
]:

    async with AsyncSessionLocal() as session:
        yield session
```

Then inject it:

```python
from fastapi import Depends
from sqlalchemy.ext.asyncio import AsyncSession


@router.get("/users/{user_id}")
async def get_user(
    user_id: int,
    db: AsyncSession = Depends(get_db),
):
    ...
```

FastAPI automatically executes:

```text
Request
   ↓
get_db()
   ↓
Create AsyncSession
   ↓
yield session
   ↓
Endpoint
   ↓
Database queries
   ↓
Endpoint finishes
   ↓
Session context closes
```

---

# 4. Complete example

Let's make this concrete.

## Model

```python
from sqlalchemy import String
from sqlalchemy.orm import (
    DeclarativeBase,
    Mapped,
    mapped_column,
)


class Base(DeclarativeBase):
    pass


class User(Base):

    __tablename__ = "users"

    id: Mapped[int] = mapped_column(
        primary_key=True
    )

    email: Mapped[str] = mapped_column(
        String(255),
        unique=True,
        nullable=False,
    )

    name: Mapped[str] = mapped_column(
        String(100),
        nullable=False,
    )
```

---

## Dependency

```python
async def get_db():
    async with AsyncSessionLocal() as session:
        yield session
```

---

## Endpoint

```python
from fastapi import Depends
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession


@router.get("/users/{user_id}")
async def get_user(
    user_id: int,
    db: AsyncSession = Depends(get_db),
):

    result = await db.execute(
        select(User).where(
            User.id == user_id
        )
    )

    user = result.scalar_one_or_none()

    if user is None:
        raise HTTPException(
            status_code=404,
            detail="User not found",
        )

    return {
        "id": user.id,
        "email": user.email,
        "name": user.name,
    }
```

The endpoint doesn't know how the database connection is created.

It only knows:

```python
db: AsyncSession
```

That's dependency injection.

---

# 5. Why shouldn't you create a database connection inside every endpoint?

This is a **very important production question**.

You might be tempted to do something like:

```python
@router.get("/users")
async def get_users():

    engine = create_async_engine(
        DATABASE_URL
    )

    async with AsyncSession(engine) as db:

        result = await db.execute(
            select(User)
        )

        return result.scalars().all()
```

Technically, you can write code like this.

But it is a **bad architecture**.

---

# 6. Problem 1 — Connection creation is expensive

If every request creates its own engine/connection infrastructure:

```text
Request 1
 ↓
Create engine
 ↓
Create connection
 ↓
Query
 ↓
Close

Request 2
 ↓
Create engine
 ↓
Create connection
 ↓
Query
 ↓
Close
```

You repeatedly pay the setup cost.

Instead:

```text
Application startup
       ↓
Create Engine
       ↓
Create Connection Pool
       ↓
Requests reuse pool
```

Then:

```text
Request 1 ──┐
Request 2 ──┤
Request 3 ──┼──→ Connection Pool → PostgreSQL
Request 4 ──┤
Request 5 ──┘
```

Much more efficient.

---

# 7. Problem 2 — Connection exhaustion

Imagine your API receives:

```text
1,000 concurrent requests
```

If every endpoint independently creates database connections, you can overwhelm PostgreSQL.

For example:

```text
FastAPI
  │
  ├── Request 1 → DB connection
  ├── Request 2 → DB connection
  ├── Request 3 → DB connection
  ├── ...
  └── Request 1000 → DB connection
                     ↓
                  PostgreSQL
                     ↓
              CONNECTION LIMIT
                     ↓
                  FAILURE
```

PostgreSQL has finite connection capacity.

A connection pool allows you to control concurrency.

For example:

```python
engine = create_async_engine(
    DATABASE_URL,
    pool_size=10,
    max_overflow=20,
)
```

Conceptually:

```text
10 persistent pool connections
+
up to 20 overflow connections
```

So you don't blindly create a new database connection for every request.

---

# 8. Problem 3 — Connection leaks

Suppose someone writes:

```python
@router.get("/users")
async def users():

    db = create_connection()

    result = db.execute(...)

    return result
```

What happens if an exception occurs?

```text
Create connection
      ↓
Execute query
      ↓
Exception
      ↓
return/cleanup never happens
      ↓
Connection remains open
```

After enough requests:

```text
Connection 1 → leaked
Connection 2 → leaked
Connection 3 → leaked
...
Connection N → leaked
```

Eventually:

```text
too many connections
```

Using:

```python
async with AsyncSessionLocal() as session:
    ...
```

ensures the session lifecycle is managed correctly.

---

# 9. Problem 4 — Database configuration becomes duplicated

Bad:

```python
@router.get("/users")
async def users():

    engine = create_async_engine(
        DATABASE_URL,
        pool_size=10,
    )
```

Another endpoint:

```python
@router.get("/orders")
async def orders():

    engine = create_async_engine(
        DATABASE_URL,
        pool_size=10,
    )
```

Another:

```python
@router.get("/payments")
async def payments():

    engine = create_async_engine(
        DATABASE_URL,
        pool_size=10,
    )
```

Now database configuration is scattered everywhere.

Instead:

```text
                Database Configuration
                         │
                         ↓
                      Engine
                         │
                  Connection Pool
                         │
                  AsyncSession Factory
                         │
              ┌──────────┼──────────┐
              ↓          ↓          ↓
           Users       Orders     Payments
```

Centralized and reusable.

---

# 10. Problem 5 — Difficult testing

If the endpoint creates its own connection:

```python
@router.get("/users")
async def users():

    engine = create_async_engine(
        PRODUCTION_DATABASE
    )
```

How do you test it?

You now have to modify the endpoint or somehow intercept its internal database creation.

With dependency injection:

```python
async def get_db():
    ...
```

you can override it.

Production:

```python
app.dependency_overrides[get_db] = override_get_db
```

Test:

```python
async def override_get_db():

    async with TestSessionLocal() as session:
        yield session
```

Now:

```text
Production
get_db
  ↓
Production PostgreSQL


Tests
get_db
  ↓
Test PostgreSQL
```

The endpoint itself doesn't change.

This is a **major benefit of FastAPI DI**.

---

# 11. Why use `yield` instead of `return`?

Compare:

### Bad for lifecycle management

```python
async def get_db():

    session = AsyncSessionLocal()

    return session
```

Now you need some other mechanism to make sure it closes.

### Better

```python
async def get_db():

    async with AsyncSessionLocal() as session:
        yield session
```

The `async with` block controls the lifetime.

Conceptually:

```text
get_db()
   │
   ├── create session
   │
   ├── yield session ───────→ endpoint
   │                              │
   │                              ↓
   │                         DB operations
   │
   └── cleanup
```

---

# 12. What about transactions?

`AsyncSession` also participates in transactions.

For example:

```python
async def create_user(
    db: AsyncSession,
    email: str,
):

    try:

        user = User(
            email=email,
            name="Sandeep",
        )

        db.add(user)

        await db.commit()

        await db.refresh(user)

        return user

    except Exception:

        await db.rollback()

        raise
```

Flow:

```text
AsyncSession
     ↓
BEGIN
     ↓
INSERT
     ↓
COMMIT
```

If something fails:

```text
AsyncSession
     ↓
BEGIN
     ↓
INSERT
     ↓
ERROR
     ↓
ROLLBACK
```

---

# 13. Important distinction: Engine vs Session

This is frequently asked in interviews.

### Engine

Responsible for:

```text
Database connectivity
Connection pool
Database dialect/driver
```

Example:

```python
engine = create_async_engine(...)
```

### AsyncSession

Responsible for:

```text
ORM interaction
Queries
Transactions
Identity map
Unit of work
```

Example:

```python
async with AsyncSessionLocal() as session:
    result = await session.execute(...)
```

Think:

```text
Engine
  ↓
"How do I connect to PostgreSQL?"

AsyncSession
  ↓
"How do I work with PostgreSQL for this unit of work?"
```

---

# 14. Don't create one global `AsyncSession`

Another common mistake:

```python
# ❌ Don't do this

db = AsyncSessionLocal()


@app.get("/users")
async def users():
    result = await db.execute(...)
```

Why?

Because multiple concurrent requests could share the same session.

You can end up with:

```text
Request A ──┐
            ├── Same AsyncSession
Request B ──┤
            │
Request C ──┘
```

That's not the intended concurrency model.

Instead:

```text
Request A → Session A
Request B → Session B
Request C → Session C
```

while those sessions can reuse the underlying connection pool.

---

# 15. Production pattern

For a real FastAPI application, I would use:

```python
# session.py

engine = create_async_engine(
    settings.database_url,
    pool_size=settings.db_pool_size,
    max_overflow=settings.db_max_overflow,
    pool_pre_ping=True,
)

SessionLocal = async_sessionmaker(
    engine,
    class_=AsyncSession,
    expire_on_commit=False,
)
```

Dependency:

```python
# dependencies.py

async def get_db():
    async with SessionLocal() as session:
        yield session
```

Endpoint:

```python
@router.get("/users/{user_id}")
async def get_user(
    user_id: int,
    db: AsyncSession = Depends(get_db),
):

    result = await db.execute(
        select(User).where(
            User.id == user_id
        )
    )

    return result.scalar_one_or_none()
```

Architecture:

```text
                    FastAPI
                       │
                       ↓
                 Depends(get_db)
                       │
                       ↓
                  AsyncSession
                       │
                       ↓
                 SQLAlchemy
                       │
                       ↓
               Connection Pool
                       │
                       ↓
                  PostgreSQL
```

---

# 16. Senior-level interview answer

If the interviewer asks:

### "What is AsyncSession?"

Say:

> **"`AsyncSession` is SQLAlchemy's asynchronous ORM session. It provides the unit-of-work interface for executing database operations, managing ORM objects and transactions using awaitable database calls. In an async FastAPI application, I typically use it with an async driver such as asyncpg."**

### "How do you inject a database session?"

Say:

> **"I create a single application-level async engine and session factory, then expose an `AsyncSession` through a FastAPI dependency using `async with` and `yield`. The endpoint receives the session through `Depends(get_db)`, which gives each request its own session lifecycle."**

Example:

```python
async def get_db():
    async with SessionLocal() as session:
        yield session
```

and:

```python
async def get_users(
    db: AsyncSession = Depends(get_db),
):
    ...
```

### "Why shouldn't you create a database connection inside every endpoint?"

Say:

> **"Because it causes unnecessary connection setup overhead, can exhaust PostgreSQL's connection limit under concurrency, makes connection cleanup and transaction management harder, duplicates configuration, and makes testing difficult. Instead, I create one SQLAlchemy engine and connection pool at application scope and create short-lived request-scoped `AsyncSession` objects through dependency injection."**

---

## The one diagram to remember

```text
             APPLICATION STARTUP
                     │
                     ↓
             create_async_engine()
                     │
                     ↓
              Connection Pool
                     │
        ┌────────────┼────────────┐
        ↓            ↓            ↓
      Conn 1       Conn 2       Conn 3
        │            │            │
        └────────────┼────────────┘
                     ↓
                 PostgreSQL


             EACH HTTP REQUEST
                     │
                     ↓
                Depends(get_db)
                     │
                     ↓
                AsyncSession
                     │
                     ↓
              Service/Repository
                     │
                     ↓
              SQLAlchemy Query
                     │
                     ↓
              Connection Pool
                     │
                     ↓
                PostgreSQL
                     │
                     ↓
              Request finishes
                     │
                     ↓
              Session closes
                     │
                     ↓
           Connection → Pool
```

**Key interview point:** you don't create a new *engine/database connection infrastructure* per endpoint. You create the **engine and pool once**, then use a **short-lived `AsyncSession` per request/unit of work**.
