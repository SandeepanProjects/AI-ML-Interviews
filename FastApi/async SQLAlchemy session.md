# How do you inject an async SQLAlchemy session in FastAPI?

This is a **very common senior FastAPI interview question**, especially when the interviewer wants to see whether you understand **FastAPI DI + SQLAlchemy async + connection pooling + session lifecycle**.

The production pattern is:

```text
FastAPI Request
      ↓
Depends(get_db)
      ↓
AsyncSession
      ↓
Service
      ↓
Repository
      ↓
PostgreSQL
      ↓
Session cleanup
```

---

# 1. Start with the async SQLAlchemy engine

For PostgreSQL, you'd typically use an async driver such as `asyncpg`.

```bash
pip install fastapi sqlalchemy asyncpg uvicorn
```

Create the engine:

```python
# db/session.py

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
    pool_pre_ping=True,
)

AsyncSessionLocal = async_sessionmaker(
    bind=engine,
    class_=AsyncSession,
    expire_on_commit=False,
)
```

There are three important objects here:

```text
engine
   ↓
async_sessionmaker
   ↓
AsyncSession
```

---

# 2. What is the `engine`?

The engine manages communication with PostgreSQL and, importantly, the **connection pool**.

```python
engine = create_async_engine(
    DATABASE_URL,
    pool_pre_ping=True,
)
```

Conceptually:

```text
                SQLAlchemy Engine
                       │
                 Connection Pool
              ┌───────┼───────┐
              ▼       ▼       ▼
           Conn 1   Conn 2   Conn 3
              │       │       │
              └───────┼───────┘
                      ▼
                  PostgreSQL
```

You generally **do not create a new engine for every request**.

The engine is application-level infrastructure.

---

# 3. What is `AsyncSession`?

`AsyncSession` represents your database session/unit of work.

For example:

```python
async with AsyncSessionLocal() as session:
    result = await session.execute(...)
```

The session is what your application uses to:

* execute queries
* add objects
* update objects
* delete objects
* commit transactions
* rollback transactions

---

# 4. Create the FastAPI dependency

Now create:

```python
# db/dependencies.py

from collections.abc import AsyncGenerator

from sqlalchemy.ext.asyncio import AsyncSession

from .session import AsyncSessionLocal


async def get_db() -> AsyncGenerator[AsyncSession, None]:

    async with AsyncSessionLocal() as session:
        yield session
```

This is the key piece.

---

# 5. Inject it using `Depends()`

Now your FastAPI endpoint:

```python
from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.dependencies import get_db

router = APIRouter()


@router.get("/users")
async def get_users(
    db: AsyncSession = Depends(get_db),
):
    result = await db.execute(
        select(User)
    )

    users = result.scalars().all()

    return users
```

The important line is:

```python
db: AsyncSession = Depends(get_db)
```

This means:

> FastAPI, resolve `get_db()` and inject the resulting `AsyncSession` into this endpoint.

---

# 6. What actually happens during a request?

Suppose the request is:

```text
GET /users
```

FastAPI sees:

```python
db = Depends(get_db)
```

and performs the dependency resolution.

Conceptually:

```text
HTTP Request
     │
     ▼
FastAPI
     │
     ▼
Depends(get_db)
     │
     ▼
get_db()
     │
     ▼
AsyncSessionLocal()
     │
     ▼
AsyncSession
     │
     ▼
endpoint(db)
     │
     ▼
SQL query
     │
     ▼
Response
     │
     ▼
session cleanup
```

---

# 7. Why do we use `yield`?

This is an important interview point.

Consider:

```python
async def get_db():

    async with AsyncSessionLocal() as session:
        yield session
```

The `yield` separates:

```text
Before yield
    ↓
Create resource

yield
    ↓
Give resource to endpoint

After endpoint finishes
    ↓
Cleanup resource
```

So FastAPI can manage the lifecycle of the session.

You don't have to do:

```python
session = AsyncSessionLocal()

try:
    ...
finally:
    await session.close()
```

inside every endpoint.

---

# 8. What does `async with` do?

This:

```python
async with AsyncSessionLocal() as session:
```

ensures the session is properly closed when the context exits.

Conceptually:

```text
Request
  ↓
Open session
  ↓
yield session
  ↓
Endpoint
  ↓
Query
  ↓
Endpoint finishes
  ↓
Exit async context
  ↓
Close session
```

This is why the pattern is very safe for request-scoped sessions.

---

# 9. Real production architecture

For a larger application, I wouldn't put all database logic in the endpoint.

Instead:

```text
FastAPI Router
      │
      ▼
Service Layer
      │
      ▼
Repository
      │
      ▼
AsyncSession
      │
      ▼
PostgreSQL
```

For example:

### Router

```python
@router.get("/users/{user_id}")
async def get_user(
    user_id: int,
    service: UserService = Depends(get_user_service),
):
    return await service.get_user(user_id)
```

### Service

```python
class UserService:

    def __init__(self, repository):
        self.repository = repository

    async def get_user(self, user_id: int):
        return await self.repository.get_by_id(user_id)
```

### Repository

```python
class UserRepository:

    def __init__(self, db: AsyncSession):
        self.db = db

    async def get_by_id(self, user_id: int):

        result = await self.db.execute(
            select(User).where(User.id == user_id)
        )

        return result.scalar_one_or_none()
```

### Dependency

```python
def get_user_repository(
    db: AsyncSession = Depends(get_db),
):
    return UserRepository(db)
```

Then:

```python
def get_user_service(
    repository: UserRepository = Depends(
        get_user_repository
    ),
):
    return UserService(repository)
```

The dependency graph becomes:

```text
                 HTTP Request
                      │
                      ▼
                   Router
                      │
                      ▼
               UserService
                      │
                      ▼
             UserRepository
                      │
                      ▼
                AsyncSession
                      │
                      ▼
                  PostgreSQL
```

This is a very good production pattern.

---

# 10. What does `expire_on_commit=False` mean?

You'll often see:

```python
AsyncSessionLocal = async_sessionmaker(
    engine,
    class_=AsyncSession,
    expire_on_commit=False,
)
```

Why?

Normally, SQLAlchemy can expire ORM object attributes after:

```python
await session.commit()
```

With async applications, this can sometimes lead to unexpected lazy-loading behavior because accessing an expired attribute may require another database operation.

Using:

```python
expire_on_commit=False
```

allows objects to retain their loaded state after commit.

For example:

```python
user = User(name="Sandeep")

session.add(user)

await session.commit()

print(user.name)
```

With async SQLAlchemy, `expire_on_commit=False` is commonly used to make post-commit object access more predictable.

---

# 11. Should you commit inside `get_db()`?

Generally, **no**.

Avoid:

```python
async def get_db():

    async with AsyncSessionLocal() as session:
        try:
            yield session
            await session.commit()
        except:
            await session.rollback()
            raise
```

as a universal pattern unless you've intentionally designed your entire application around that transaction model.

A cleaner approach is usually to let the **service/unit-of-work layer control transaction boundaries**.

For example:

```python
async def create_user(
    self,
    user_data,
):

    user = User(**user_data)

    self.db.add(user)

    await self.db.commit()

    await self.db.refresh(user)

    return user
```

Or use an explicit transaction:

```python
async with self.db.begin():

    self.db.add(user)
```

The important architectural principle is:

> **The layer that owns the business transaction should generally own commit/rollback decisions.**

---

# 12. Handling rollback correctly

Suppose:

```python
async def create_user(self, data):

    user = User(**data)

    self.db.add(user)

    try:
        await self.db.commit()

    except Exception:
        await self.db.rollback()
        raise

    await self.db.refresh(user)

    return user
```

Why rollback?

Suppose PostgreSQL returns:

```text
UNIQUE constraint violation
```

The transaction may be in a failed state.

You need:

```python
await self.db.rollback()
```

before continuing to use the session for another transaction.

---

# 13. Better transaction pattern

For more complex business operations:

```python
async def create_order(self, order_data):

    async with self.db.begin():

        order = Order(**order_data)

        self.db.add(order)

        payment = Payment(...)

        self.db.add(payment)
```

If everything succeeds:

```text
COMMIT
```

If something fails:

```text
ROLLBACK
```

Conceptually:

```text
        Transaction
             │
      ┌──────┴──────┐
      ▼             ▼
 Create Order   Create Payment
      │             │
      └──────┬──────┘
             ▼
          Success?
         /        \
       Yes         No
        │           │
     COMMIT      ROLLBACK
```

---

# 14. Async session does NOT mean one connection per request

This is another senior-level interview trap.

You might think:

```text
100 requests
    ↓
100 PostgreSQL connections
```

That's not necessarily what happens.

The engine has a connection pool.

Conceptually:

```text
100 HTTP Requests
       │
       ▼
   Session layer
       │
       ▼
 Connection Pool
   ┌───┼───┐
   ▼   ▼   ▼
 Conn1 Conn2 Conn3
   │   │   │
   └───┼───┘
       ▼
   PostgreSQL
```

The pool limits the number of active database connections.

This is extremely important when scaling FastAPI.

---

# 15. What happens if connection pool is too small?

Suppose:

```text
FastAPI
  ↓
100 concurrent requests
  ↓
PostgreSQL pool = 5
```

Only a limited number of requests can hold database connections simultaneously.

Others wait for a connection to become available.

So you need to tune:

```text
pool size
max overflow
connection timeout
database max connections
number of FastAPI workers
```

You cannot independently maximize all of these.

---

# 16. Connection pooling and FastAPI workers

Suppose you deploy:

```text
4 FastAPI workers
```

Each worker is a separate process.

Each process can have its own SQLAlchemy engine/pool.

So if each pool allows:

```text
10 connections
```

you could potentially have roughly:

```text
4 workers × 10 connections
=
40 database connections
```

plus overflow depending on configuration.

This is why database pool sizing must consider the **number of application processes**, not just the pool size in one process.

---

# 17. Production configuration

You might configure:

```python
engine = create_async_engine(
    DATABASE_URL,
    pool_pre_ping=True,
    pool_size=10,
    max_overflow=20,
    pool_timeout=30,
)
```

Meaning, conceptually:

```text
10 persistent pooled connections
+
up to 20 overflow connections
```

But don't blindly use these numbers.

They depend on:

* PostgreSQL capacity
* application traffic
* number of workers
* query latency
* concurrency
* infrastructure limits

---

# 18. What is `pool_pre_ping=True`?

This:

```python
pool_pre_ping=True
```

helps detect stale/dead database connections before using them.

For example:

```text
Connection Pool
     │
     ▼
Connection became stale
     │
     ▼
Application requests connection
     │
     ▼
pre_ping detects problem
     │
     ▼
Connection replaced/reconnected
```

It's commonly useful in production environments where connections can become invalid because of network issues, database restarts, load balancers, etc.

---

# 19. Don't create the engine inside `get_db()`

Avoid:

```python
async def get_db():

    engine = create_async_engine(
        DATABASE_URL
    )

    async with AsyncSession(engine) as session:
        yield session
```

Why?

Because you'd potentially create a new engine/pool repeatedly.

That's terrible for performance and connection management.

Instead:

```text
Application startup
       ↓
Create engine
       ↓
Create pool
       ↓
Requests reuse it
```

---

# 20. Complete production-style example

A clean setup could look like:

```text
app/
│
├── db/
│   ├── session.py
│   └── dependencies.py
│
├── repositories/
│   └── user_repository.py
│
├── services/
│   └── user_service.py
│
└── api/
    └── users.py
```

### `session.py`

```python
from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

DATABASE_URL = (
    "postgresql+asyncpg://"
    "postgres:password@localhost:5432/app"
)

engine = create_async_engine(
    DATABASE_URL,
    pool_pre_ping=True,
)

SessionLocal = async_sessionmaker(
    engine,
    class_=AsyncSession,
    expire_on_commit=False,
)
```

### `dependencies.py`

```python
from collections.abc import AsyncGenerator

from sqlalchemy.ext.asyncio import AsyncSession

from .session import SessionLocal


async def get_db() -> AsyncGenerator[
    AsyncSession,
    None,
]:

    async with SessionLocal() as session:
        yield session
```

### Repository

```python
class UserRepository:

    def __init__(self, db: AsyncSession):
        self.db = db

    async def get_user(self, user_id: int):

        result = await self.db.execute(
            select(User).where(
                User.id == user_id
            )
        )

        return result.scalar_one_or_none()
```

### Dependency for repository

```python
def get_user_repository(
    db: AsyncSession = Depends(get_db),
):
    return UserRepository(db)
```

### Router

```python
@router.get("/users/{user_id}")
async def get_user(
    user_id: int,
    repo: UserRepository = Depends(
        get_user_repository
    ),
):

    return await repo.get_user(user_id)
```

---

# 21. The complete dependency chain

When the request arrives:

```text
GET /users/123
        │
        ▼
FastAPI Router
        │
        ▼
Depends(get_user_repository)
        │
        ▼
Depends(get_db)
        │
        ▼
SessionLocal()
        │
        ▼
AsyncSession
        │
        ▼
UserRepository
        │
        ▼
PostgreSQL
        │
        ▼
Response
        │
        ▼
Session cleanup
```

This is the architecture you should be able to explain in an interview.

---

# 22. Common interview mistakes

### ❌ Mistake 1

> "I create a database connection for every request."

Better:

> "I create an application-level async engine with a connection pool and create request-scoped `AsyncSession` objects from an `async_sessionmaker`."

---

### ❌ Mistake 2

> "AsyncSession makes database queries automatically non-blocking."

Not exactly.

The entire stack needs to be async-compatible:

```text
FastAPI
   ↓
async endpoint
   ↓
SQLAlchemy AsyncSession
   ↓
async driver
   ↓
asyncpg
   ↓
PostgreSQL
```

---

### ❌ Mistake 3

Using a synchronous driver:

```text
postgresql://...
```

while expecting async behavior.

For async PostgreSQL SQLAlchemy, use an async dialect such as:

```text
postgresql+asyncpg://...
```

---

### ❌ Mistake 4

Creating the engine inside the dependency.

Don't:

```python
async def get_db():
    engine = create_async_engine(...)
```

Create it once at application level.

---

### ❌ Mistake 5

Forgetting rollback.

If you manually control transactions:

```python
try:
    await db.commit()
except:
    await db.rollback()
    raise
```

---

# 23. Interview answer

If they ask:

> **"How do you inject an async SQLAlchemy session in FastAPI?"**

A strong senior-level answer is:

> "I create a single application-level SQLAlchemy `AsyncEngine` using an async PostgreSQL driver such as `asyncpg`, and then create an `async_sessionmaker` from that engine. I define a FastAPI dependency called `get_db()` that creates an `AsyncSession` using an `async with` context manager and yields it. The endpoint or service receives the session through `AsyncSession = Depends(get_db)`.
>
> The session is request-scoped, while the engine and connection pool are application-scoped. I keep transaction boundaries in the service or unit-of-work layer rather than blindly committing inside the dependency. Using `yield` ensures the session lifecycle is properly managed and the session is cleaned up after the request.
>
> In production, I also configure connection pooling, `pool_pre_ping`, appropriate pool limits, and make sure pool sizing accounts for the number of application workers."

### Memorize this architecture:

```text
Application startup
       │
       ▼
AsyncEngine
       │
       ▼
Connection Pool
       │
       ▼
async_sessionmaker
       │
       │ per request
       ▼
AsyncSession
       │
       ▼
Depends(get_db)
       │
       ▼
Service / Repository
       │
       ▼
PostgreSQL
       │
       ▼
Cleanup
```

**The key distinction to remember:**

> **Engine/pool = application-level. `AsyncSession` = typically request/unit-of-work-level. `Depends()` = mechanism that injects the session into FastAPI's dependency graph.**
