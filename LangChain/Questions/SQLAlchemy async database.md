Yes. If you're building **production FastAPI applications**, understanding **SQLAlchemy async** properly is important because it sits between your async API layer and PostgreSQL.

The architecture is:

```text
Client
   ↓
FastAPI
   ↓
Service Layer
   ↓
Repository
   ↓
AsyncSession
   ↓
SQLAlchemy Async Engine
   ↓
asyncpg
   ↓
PostgreSQL
```

The key idea is:

> **SQLAlchemy Async allows your FastAPI application to perform database I/O without blocking the event loop.**

---

# 1. What is SQLAlchemy?

SQLAlchemy is a Python database toolkit and ORM.

It provides two major concepts:

### SQL Expression / Core

You can write:

```python
stmt = select(User).where(User.email == email)
```

### ORM

You can represent a database table as a Python class:

```python
class User(Base):
    __tablename__ = "users"

    id = mapped_column(...)
    email = mapped_column(...)
```

Then:

```python
user = User(
    email="john@example.com"
)
```

instead of manually writing:

```sql
INSERT INTO users ...
```

SQLAlchemy converts your Python operations into SQL.

---

# 2. Why async SQLAlchemy?

Traditional synchronous SQLAlchemy:

```text
FastAPI
   ↓
sync endpoint
   ↓
SQLAlchemy
   ↓
PostgreSQL
```

The thread waits while PostgreSQL processes the query.

With async:

```text
FastAPI event loop
       |
       v
AsyncSession
       |
       v
PostgreSQL
```

When the database is waiting on I/O, your async application can yield control and handle other work.

Conceptually:

```text
Request A
   |
   | DB query ────────────────┐
   |                          |
   |                          | waiting
   |                          |
Request B                     |
   |                          |
   | DB query                 |
   |                          |
   +--------------------------+
```

This is particularly useful for I/O-heavy APIs with many concurrent requests.

---

# 3. The async stack

A common PostgreSQL stack is:

```text
FastAPI
    ↓
SQLAlchemy Async
    ↓
asyncpg
    ↓
PostgreSQL
```

Install:

```bash
pip install fastapi sqlalchemy asyncpg
```

For migrations, commonly:

```bash
pip install alembic
```

---

# 4. Project structure

A production project might look like:

```text
app/
│
├── main.py
│
├── api/
│   └── v1/
│       └── users.py
│
├── services/
│   └── user_service.py
│
├── repositories/
│   └── user_repository.py
│
├── models/
│   ├── base.py
│   └── user.py
│
├── schemas/
│   └── user.py
│
└── db/
    ├── engine.py
    └── session.py
```

This fits directly with the **Router → Service → Repository** architecture we discussed.

---

# 5. Create the async engine

The first important piece is the engine.

```python
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    create_async_engine,
)

DATABASE_URL = (
    "postgresql+asyncpg://"
    "postgres:password@localhost:5432/mydb"
)

engine: AsyncEngine = create_async_engine(
    DATABASE_URL,
    echo=False,
    pool_pre_ping=True,
)
```

Notice:

```text
postgresql+asyncpg://
```

The important part is:

```text
asyncpg
```

This tells SQLAlchemy to use the async PostgreSQL driver.

---

# 6. What is the Engine?

Think of the engine as the **database connectivity and connection-pooling layer**.

```text
SQLAlchemy Engine
       |
       +--- connection pool
       |
       +--- DB driver
       |
       +--- PostgreSQL
```

You generally create **one engine per process**, not one engine per request.

Bad:

```python
async def endpoint():

    engine = create_async_engine(...)
```

This repeatedly creates database infrastructure.

Instead:

```text
Application startup
       ↓
Create engine
       ↓
Create connection pool
       ↓
Reuse it
```

---

# 7. Connection pooling

The engine maintains a pool of database connections.

Conceptually:

```text
                  Connection Pool

       +---------------------------+
       |                           |
       | Connection 1              |
       | Connection 2              |
       | Connection 3              |
       | Connection 4              |
       | Connection 5              |
       |                           |
       +---------------------------+
```

Requests borrow connections:

```text
Request
   ↓
AsyncSession
   ↓
Pool
   ↓
Connection
   ↓
PostgreSQL
```

When finished:

```text
Connection
   ↓
returned to pool
```

This is much cheaper than creating a brand-new PostgreSQL connection for every request.

---

# 8. Configure the pool

For production, you may configure:

```python
engine = create_async_engine(
    DATABASE_URL,

    pool_size=10,

    max_overflow=20,

    pool_timeout=30,

    pool_recycle=1800,

    pool_pre_ping=True,
)
```

Meaning roughly:

```text
pool_size
    ↓
Normal persistent connections

max_overflow
    ↓
Temporary additional connections

pool_timeout
    ↓
How long to wait for a connection

pool_pre_ping
    ↓
Check whether pooled connection is still alive
```

Exact values depend on:

* database capacity
* number of application workers
* traffic
* query latency
* PostgreSQL connection limits

Don't blindly set huge pool sizes.

---

# 9. AsyncSession

Next:

```python
from sqlalchemy.ext.asyncio import (
    async_sessionmaker,
)

SessionLocal = async_sessionmaker(
    bind=engine,
    expire_on_commit=False,
)
```

Now:

```text
Engine
  ↓
Session Factory
  ↓
AsyncSession
```

The `AsyncSession` is what your application uses to interact with the database.

---

# 10. What is AsyncSession?

Think of it as a **unit-of-work / database interaction context**.

For example:

```python
async with SessionLocal() as session:

    result = await session.execute(...)
```

It manages:

* database interaction
* ORM identity map
* transactions
* flushing
* commits
* rollbacks

But remember:

> **An `AsyncSession` should generally not be shared concurrently between unrelated requests/tasks.**

A request-scoped session is the common pattern.

---

# 11. FastAPI database dependency

This is where FastAPI dependency injection becomes useful.

```python
from collections.abc import AsyncGenerator

async def get_db() -> AsyncGenerator[AsyncSession, None]:

    async with SessionLocal() as session:

        yield session
```

Then:

```python
@router.get("/users")
async def get_users(
    db: AsyncSession = Depends(get_db),
):
    ...
```

Flow:

```text
HTTP Request
      ↓
FastAPI
      ↓
get_db()
      ↓
AsyncSession
      ↓
Endpoint
      ↓
request complete
      ↓
Session closes
```

---

# 12. Why use `yield`?

This:

```python
async def get_db():

    async with SessionLocal() as session:
        yield session
```

allows FastAPI to manage the lifecycle.

Before `yield`:

```text
create session
```

During:

```text
endpoint executes
```

After:

```text
close session
```

That's dependency lifecycle management.

---

# 13. SQLAlchemy Model

Now create your model.

```python
from sqlalchemy.orm import (
    DeclarativeBase,
    Mapped,
    mapped_column,
)

from sqlalchemy import String


class Base(DeclarativeBase):
    pass


class User(Base):

    __tablename__ = "users"

    id: Mapped[int] = mapped_column(
        primary_key=True
    )

    name: Mapped[str] = mapped_column(
        String(100)
    )

    email: Mapped[str] = mapped_column(
        String(255),
        unique=True,
        index=True,
    )
```

This represents:

```text
users
--------------------------------
id
name
email
```

---

# 14. Create tables

For learning:

```python
async with engine.begin() as conn:

    await conn.run_sync(
        Base.metadata.create_all
    )
```

But **don't use `create_all()` as your production schema migration strategy**.

Production applications generally use **Alembic**.

We'll come to that.

---

# 15. Insert data

Using SQLAlchemy 2.x style:

```python
async def create_user(
    db: AsyncSession,
):

    user = User(
        name="John",
        email="john@example.com",
    )

    db.add(user)

    await db.commit()

    await db.refresh(user)

    return user
```

Flow:

```text
User object
    ↓
db.add()
    ↓
flush/commit
    ↓
PostgreSQL
```

---

# 16. `flush()` vs `commit()`

This is extremely important.

### `flush()`

Sends pending SQL to the database within the current transaction.

```python
db.add(user)

await db.flush()
```

The database can now generate things like an auto-increment ID.

But the transaction isn't committed yet.

### `commit()`

Makes the transaction durable:

```python
await db.commit()
```

Think:

```text
add()
 ↓
Python session

flush()
 ↓
SQL sent to DB

commit()
 ↓
Transaction finalized
```

---

# 17. Why use flush in services?

Suppose:

```text
Create User
   ↓
Create Profile
```

You need the user ID.

```python
user = User(...)

db.add(user)

await db.flush()

profile = Profile(
    user_id=user.id
)

db.add(profile)

await db.commit()
```

You can use the generated `user.id` after `flush()` without committing the whole transaction.

This is very useful for multi-step operations.

---

# 18. Querying data

Use SQLAlchemy 2.x:

```python
from sqlalchemy import select

stmt = select(User)

result = await db.execute(stmt)

users = result.scalars().all()
```

For one user:

```python
stmt = (
    select(User)
    .where(User.id == user_id)
)

result = await db.execute(stmt)

user = result.scalar_one_or_none()
```

---

# 19. Filtering

```python
stmt = (
    select(User)
    .where(
        User.email == email
    )
)

result = await db.execute(stmt)

user = result.scalar_one_or_none()
```

Multiple conditions:

```python
stmt = (
    select(User)
    .where(
        User.tenant_id == tenant_id,
        User.is_active.is_(True),
    )
)
```

SQLAlchemy translates this into SQL roughly like:

```sql
SELECT *
FROM users
WHERE tenant_id = $1
AND is_active = true;
```

---

# 20. Repository pattern

This is where your previous question about repositories connects.

Instead of putting:

```python
select(User)
```

inside your router, create:

```python
class UserRepository:

    def __init__(
        self,
        db: AsyncSession,
    ):
        self.db = db
```

Then:

```python
async def get_by_id(
    self,
    user_id: int,
):

    stmt = (
        select(User)
        .where(User.id == user_id)
    )

    result = await self.db.execute(stmt)

    return result.scalar_one_or_none()
```

Now your architecture becomes:

```text
FastAPI Router
       ↓
UserService
       ↓
UserRepository
       ↓
AsyncSession
       ↓
PostgreSQL
```

---

# 21. Complete Repository example

```python
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.user import User


class UserRepository:

    def __init__(
        self,
        db: AsyncSession,
    ):
        self.db = db

    async def get_by_id(
        self,
        user_id: int,
    ) -> User | None:

        stmt = (
            select(User)
            .where(User.id == user_id)
        )

        result = await self.db.execute(stmt)

        return result.scalar_one_or_none()

    async def get_by_email(
        self,
        email: str,
    ) -> User | None:

        stmt = (
            select(User)
            .where(User.email == email)
        )

        result = await self.db.execute(stmt)

        return result.scalar_one_or_none()

    async def create(
        self,
        user: User,
    ) -> User:

        self.db.add(user)

        await self.db.flush()

        await self.db.refresh(user)

        return user
```

Notice:

**Repository doesn't call `commit()` here.**

That's a deliberate design choice in many architectures.

---

# 22. Why not commit inside repository?

Suppose your service does:

```text
Create order
 ↓
Create payment
 ↓
Update inventory
 ↓
Create audit
```

If every repository commits independently:

```text
OrderRepository.commit()
        ↓
PaymentRepository.commit()
        ↓
InventoryRepository.commit()
```

you can end up with partial updates.

Instead, the service can control the transaction boundary.

```text
Service
   |
   | transaction
   |
   +---- Repository
   |
   +---- Repository
   |
   +---- Repository
   |
   ↓
commit
```

This is one of the most important reasons to keep transaction ownership explicit.

---

# 23. Service layer + AsyncSession

For example:

```python
class UserService:

    def __init__(
        self,
        repository: UserRepository,
        db: AsyncSession,
    ):
        self.repository = repository
        self.db = db

    async def create_user(
        self,
        name: str,
        email: str,
    ):

        existing = (
            await self.repository
            .get_by_email(email)
        )

        if existing:
            raise ValueError(
                "Email already exists"
            )

        user = User(
            name=name,
            email=email,
        )

        await self.repository.create(user)

        await self.db.commit()

        return user
```

The responsibility is clear:

```text
Repository
    ↓
Persistence operations

Service
    ↓
Business rules + transaction orchestration
```

---

# 24. Better transaction handling

For larger applications, you can explicitly define:

```python
async with db.begin():

    user = await repository.create(...)

    profile = await profile_repository.create(...)

    audit = await audit_repository.create(...)
```

If everything succeeds:

```text
COMMIT
```

If an exception occurs:

```text
ROLLBACK
```

Conceptually:

```text
BEGIN
  |
  +-- INSERT user
  |
  +-- INSERT profile
  |
  +-- INSERT audit
  |
COMMIT
```

Failure:

```text
BEGIN
  |
  +-- INSERT user
  |
  +-- INSERT profile
  |
  X-- INSERT audit fails
  |
ROLLBACK
```

This is much safer for multi-step business operations.

---

# 25. Relationships

SQLAlchemy also manages relationships.

For example:

```python
class User(Base):

    __tablename__ = "users"

    id: Mapped[int] = mapped_column(
        primary_key=True
    )

    email: Mapped[str]

    orders: Mapped[list["Order"]] = relationship(
        back_populates="user"
    )
```

Order:

```python
class Order(Base):

    __tablename__ = "orders"

    id: Mapped[int] = mapped_column(
        primary_key=True
    )

    user_id: Mapped[int] = mapped_column(
        ForeignKey("users.id")
    )

    user: Mapped["User"] = relationship(
        back_populates="orders"
    )
```

Database:

```text
users
   |
   | 1
   |
   | *
orders
```

---

# 26. Async relationship loading

This is an area where async SQLAlchemy requires care.

In synchronous code, developers often rely on lazy loading:

```python
user.orders
```

But implicit database I/O can be problematic in async code.

You generally want explicit loading strategies such as:

```python
selectinload()
```

Example:

```python
from sqlalchemy.orm import selectinload

stmt = (
    select(User)
    .options(
        selectinload(User.orders)
    )
    .where(User.id == user_id)
)

result = await db.execute(stmt)

user = result.scalar_one_or_none()
```

Now the relationship is explicitly loaded.

This helps avoid unexpected async I/O.

---

# 27. The N+1 problem

Consider:

```python
users = await get_users()

for user in users:
    print(user.orders)
```

You could accidentally generate:

```text
1 query → users

+
N queries → orders for each user
```

For 1,000 users:

```text
1001 database queries
```

That's the classic **N+1 query problem**.

Instead:

```python
select(User).options(
    selectinload(User.orders)
)
```

can load related data efficiently.

---

# 28. Pagination

Never do this for a production API:

```python
users = result.scalars().all()
```

if you have millions of users.

Use pagination.

Offset pagination:

```python
stmt = (
    select(User)
    .order_by(User.id)
    .offset(offset)
    .limit(limit)
)
```

For large datasets, **keyset/cursor pagination** is often better:

```python
stmt = (
    select(User)
    .where(User.id > last_id)
    .order_by(User.id)
    .limit(limit)
)
```

This avoids increasingly expensive offsets on large tables.

---

# 29. Async SQLAlchemy doesn't make every query faster

This is an important misconception.

Async doesn't mean:

```text
SQL query executes faster
```

Instead, it primarily helps with:

```text
concurrency
resource utilization
non-blocking I/O
```

If your SQL query takes:

```text
2 seconds
```

async doesn't magically make it:

```text
10 milliseconds
```

You still need:

* indexes
* efficient SQL
* query optimization
* appropriate schema design
* connection pool tuning

---

# 30. Indexes

For example:

```python
email: Mapped[str] = mapped_column(
    String(255),
    unique=True,
    index=True,
)
```

This helps:

```sql
WHERE email = ...
```

But don't add indexes everywhere.

Indexes:

```text
Improve reads
        ↓
Cost storage
        ↓
Can slow writes
```

Index based on actual query patterns.

---

# 31. PostgreSQL + SQLAlchemy transaction architecture

A good production flow is:

```text
HTTP Request
      ↓
FastAPI
      ↓
get_db()
      ↓
AsyncSession
      ↓
Service
      ↓
Repository
      ↓
SQLAlchemy
      ↓
asyncpg
      ↓
PostgreSQL
```

The session lifecycle is typically request-scoped:

```text
Request begins
    ↓
Acquire/create AsyncSession
    ↓
Service operations
    ↓
Commit or rollback
    ↓
Session closes
    ↓
Connection returned to pool
```

---

# 32. Handling rollback

If something fails:

```python
try:

    async with db.begin():

        await repository.create(...)

        await repository.update(...)

except Exception:

    # transaction is rolled back
    raise
```

Or when manually controlling a transaction:

```python
try:

    ...

    await db.commit()

except Exception:

    await db.rollback()

    raise
```

Don't swallow database exceptions.

Bad:

```python
except Exception:
    return None
```

This hides production failures.

---

# 33. Database dependency

Your database dependency can be:

```python
async def get_db():
    async with SessionLocal() as session:
        yield session
```

Then repository dependency:

```python
def get_user_repository(
    db: AsyncSession = Depends(get_db),
):

    return UserRepository(db)
```

Service dependency:

```python
def get_user_service(
    repository: UserRepository = Depends(
        get_user_repository
    ),
):

    return UserService(repository)
```

Router:

```python
@router.get("/{user_id}")
async def get_user(
    user_id: int,
    service: UserService = Depends(
        get_user_service
    ),
):

    return await service.get_user(
        user_id
    )
```

This is the complete dependency chain we've been discussing.

---

# 34. SQLAlchemy + Pydantic

Keep ORM models and API schemas separate.

SQLAlchemy:

```python
class User(Base):

    id: Mapped[int]

    email: Mapped[str]

    name: Mapped[str]
```

Pydantic:

```python
from pydantic import BaseModel


class UserResponse(BaseModel):

    id: int
    email: str
    name: str

    model_config = {
        "from_attributes": True
    }
```

This is important.

Don't expose your ORM model directly as your public API contract.

---

# 35. Alembic migrations

For production databases, don't rely on:

```python
Base.metadata.create_all()
```

Instead:

```text
SQLAlchemy Models
       ↓
Alembic
       ↓
Migration
       ↓
PostgreSQL
```

For example:

```bash
alembic revision --autogenerate -m "add users"

alembic upgrade head
```

Migration:

```python
def upgrade():

    op.add_column(
        "users",
        sa.Column(
            "is_active",
            sa.Boolean(),
            nullable=False,
            server_default="true",
        ),
    )
```

This gives you controlled database schema evolution.

---

# 36. Production configuration

Don't hardcode:

```python
DATABASE_URL = "postgresql+asyncpg://postgres:password..."
```

Use environment configuration:

```python
from pydantic_settings import BaseSettings


class Settings(BaseSettings):

    database_url: str

    model_config = {
        "env_file": ".env"
    }
```

Then:

```text
DATABASE_URL=postgresql+asyncpg://...
```

In production, use your secret-management system rather than committing credentials to source control.

---

# 37. SQLAlchemy async in your enterprise RAG project

Suppose your database contains:

```text
tenants
users
documents
conversations
messages
audit_logs
```

Your architecture might be:

```text
                    FastAPI
                       |
                       v
                    Router
                       |
                       v
                  Service Layer
                       |
          +------------+-------------+
          |            |             |
          v            v             v
      Repository   QdrantService   Redis
          |
          v
    AsyncSession
          |
          v
       SQLAlchemy
          |
          v
     asyncpg
          |
          v
     PostgreSQL
```

For example:

```python
class DocumentService:

    def __init__(
        self,
        document_repo: DocumentRepository,
        vector_store: VectorStore,
        db: AsyncSession,
    ):
        self.document_repo = document_repo
        self.vector_store = vector_store
        self.db = db

    async def delete_document(
        self,
        document_id: UUID,
        tenant_id: UUID,
    ):

        document = await (
            self.document_repo
            .get_for_tenant(
                document_id,
                tenant_id,
            )
        )

        if not document:
            raise DocumentNotFound()

        await self.vector_store.delete(
            document.vector_id
        )

        document.is_deleted = True

        await self.db.commit()
```

Now PostgreSQL handles metadata:

```text
documents
   |
   +-- tenant_id
   +-- filename
   +-- status
   +-- vector_id
   +-- created_at
```

while Qdrant handles embeddings.

That's a very common division of responsibilities.

---

# 38. Important production rule: don't use one AsyncSession across concurrent tasks

Avoid:

```python
await asyncio.gather(
    repository.operation_1(db),
    repository.operation_2(db),
    repository.operation_3(db),
)
```

when all operations share the same `AsyncSession` unless you've designed the transaction/concurrency semantics carefully.

`AsyncSession` is not intended to be an unrestricted concurrent shared state object.

Better architecture:

```text
Task A → Session A
Task B → Session B
Task C → Session C
```

or sequence operations through one session when they belong to the same transaction.

---

# 39. Read/write separation

At larger scale you may have:

```text
                    Application
                         |
             +-----------+-----------+
             |                       |
             v                       v
         Primary DB             Read Replica
             |                       |
          writes                    reads
```

Your application can route:

```text
INSERT / UPDATE / DELETE
       ↓
Primary

SELECT
       ↓
Replica
```

But this introduces consistency considerations.

For example, immediately after a write:

```text
write primary
   ↓
read replica
   ↓
replication lag
```

The newly written data might not yet be visible.

So don't introduce read replicas without understanding these tradeoffs.

---

# 40. Common mistakes

### Mistake 1 — Creating engine per request

Bad:

```python
async def endpoint():

    engine = create_async_engine(...)
```

Use one engine per process.

---

### Mistake 2 — Sharing session globally

Bad:

```python
db = AsyncSession(...)
```

and using it across requests.

Use request-scoped sessions.

---

### Mistake 3 — Calling synchronous DB APIs inside async code

Bad:

```python
result = db.execute(...)
```

with an async session.

Use:

```python
result = await db.execute(...)
```

---

### Mistake 4 — Lazy-loading relationships unexpectedly

Prefer explicit loading:

```python
selectinload(...)
```

---

### Mistake 5 — Returning millions of rows

Use:

```text
pagination
```

---

### Mistake 6 — Commit inside every repository method

This can destroy transaction boundaries.

---

### Mistake 7 — Using `create_all()` for production migrations

Use:

```text
Alembic
```

---

# 41. A complete minimal implementation

### `db/engine.py`

```python
from sqlalchemy.ext.asyncio import (
    create_async_engine,
    async_sessionmaker,
    AsyncSession,
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

### `db/session.py`

```python
from collections.abc import AsyncGenerator

from sqlalchemy.ext.asyncio import AsyncSession

from .engine import SessionLocal


async def get_db() -> AsyncGenerator[
    AsyncSession,
    None,
]:

    async with SessionLocal() as session:

        yield session
```

### `models/base.py`

```python
from sqlalchemy.orm import DeclarativeBase


class Base(DeclarativeBase):
    pass
```

### `models/user.py`

```python
from sqlalchemy import String
from sqlalchemy.orm import Mapped, mapped_column

from .base import Base


class User(Base):

    __tablename__ = "users"

    id: Mapped[int] = mapped_column(
        primary_key=True
    )

    name: Mapped[str] = mapped_column(
        String(100)
    )

    email: Mapped[str] = mapped_column(
        String(255),
        unique=True,
        index=True,
    )
```

### `repositories/user_repository.py`

```python
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.user import User


class UserRepository:

    def __init__(
        self,
        db: AsyncSession,
    ):
        self.db = db

    async def get_by_id(
        self,
        user_id: int,
    ):

        result = await self.db.execute(
            select(User)
            .where(User.id == user_id)
        )

        return result.scalar_one_or_none()

    async def create(
        self,
        user: User,
    ):

        self.db.add(user)

        await self.db.flush()

        await self.db.refresh(user)

        return user
```

### `services/user_service.py`

```python
class UserService:

    def __init__(
        self,
        repository: UserRepository,
        db: AsyncSession,
    ):
        self.repository = repository
        self.db = db

    async def create_user(
        self,
        name: str,
        email: str,
    ):

        user = User(
            name=name,
            email=email,
        )

        await self.repository.create(user)

        await self.db.commit()

        return user
```

### Router

```python
@router.post("/users")
async def create_user(
    request: CreateUserRequest,
    db: AsyncSession = Depends(get_db),
):

    repository = UserRepository(db)

    service = UserService(
        repository,
        db,
    )

    return await service.create_user(
        request.name,
        request.email,
    )
```

For a small application this works, but in a larger application I'd move service/repository construction into dedicated dependencies rather than manually constructing them in every endpoint.

---

# 42. Production version of the dependency chain

A cleaner implementation is:

```python
def get_user_repository(
    db: AsyncSession = Depends(get_db),
):
    return UserRepository(db)
```

Then:

```python
def get_user_service(
    db: AsyncSession = Depends(get_db),
    repository: UserRepository = Depends(
        get_user_repository
    ),
):
    return UserService(
        repository=repository,
        db=db,
    )
```

Router:

```python
@router.post("/users")
async def create_user(
    request: CreateUserRequest,
    service: UserService = Depends(
        get_user_service
    ),
):

    return await service.create_user(
        request.name,
        request.email,
    )
```

Now your router knows almost nothing about database infrastructure.

That's the architecture I'd prefer for a larger application.

---

# 43. The complete mental model

Remember these relationships:

```text
                 FastAPI
                    |
                    v
                 Router
                    |
                    v
                 Service
                    |
                    v
              Repository
                    |
                    v
              AsyncSession
                    |
                    v
            SQLAlchemy Async
                    |
                    v
                 asyncpg
                    |
                    v
               PostgreSQL
```

And:

```text
Engine
  ↓
Connection Pool
  ↓
AsyncSession
  ↓
Transaction
  ↓
SQL Query
  ↓
PostgreSQL
```

The most important concepts to remember are:

| Concept                 | Purpose                                  |
| ----------------------- | ---------------------------------------- |
| `create_async_engine()` | Creates DB engine/pool                   |
| `async_sessionmaker()`  | Creates session factory                  |
| `AsyncSession`          | Performs DB operations                   |
| `get_db()`              | Request-scoped DB dependency             |
| `select()`              | SQLAlchemy 2.x query construction        |
| `execute()`             | Executes SQL                             |
| `scalars()`             | Extracts ORM objects                     |
| `flush()`               | Sends pending changes without committing |
| `commit()`              | Commits transaction                      |
| `rollback()`            | Reverts transaction                      |
| `selectinload()`        | Efficient relationship loading           |
| Alembic                 | Production schema migrations             |
| `asyncpg`               | Async PostgreSQL driver                  |

### The senior-level takeaway

For a production FastAPI system, I would generally use **one async SQLAlchemy engine per application process, a bounded connection pool, request-scoped `AsyncSession`s through FastAPI dependency injection, repositories for persistence operations, services for business logic and transaction orchestration, explicit relationship loading to avoid async lazy-loading issues, Alembic for schema migrations, and PostgreSQL indexes/pagination/query optimization for performance**.

The important point is that **async SQLAlchemy is not just `await db.execute()`**. The production design is really about **engine lifecycle + connection pooling + session lifecycle + transaction boundaries + repository/service separation + query performance + concurrency safety**.
