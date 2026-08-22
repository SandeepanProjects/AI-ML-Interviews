These three are fundamental FastAPI + database interview questions. For a **senior AI/FastAPI engineer**, you should explain not just the definition, but the complete flow:

```text
HTTP Request
     ↓
FastAPI Router
     ↓
Service Layer
     ↓
Repository Layer
     ↓
SQLAlchemy AsyncSession
     ↓
PostgreSQL
```

I'll use **SQLAlchemy 2.x + async PostgreSQL**, which is the style I'd recommend for a modern production FastAPI application.

---

# 1. How do you integrate SQLAlchemy with FastAPI?

There are five main pieces:

```text
1. Database URL
2. SQLAlchemy Engine
3. Session Factory
4. FastAPI DB Dependency
5. Models / Repository / Service
```

Let's build it properly.

---

## Step 1 — Install dependencies

For PostgreSQL:

```bash
pip install fastapi uvicorn sqlalchemy asyncpg pydantic-settings
```

We use:

```text
FastAPI       → API framework
SQLAlchemy    → ORM/database toolkit
asyncpg       → Async PostgreSQL driver
```

---

# Step 2 — Configure the database

Create:

```python
# app/core/config.py

from pydantic_settings import BaseSettings


class Settings(BaseSettings):

    database_url: str

    class Config:
        env_file = ".env"


settings = Settings()
```

`.env`:

```env
DATABASE_URL=postgresql+asyncpg://postgres:password@localhost:5432/mydb
```

The important part is:

```text
postgresql+asyncpg://
```

because we're using the asynchronous PostgreSQL driver.

---

# Step 3 — Create SQLAlchemy engine

```python
# app/db/session.py

from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from app.core.config import settings


engine = create_async_engine(
    settings.database_url,

    # Connection pool
    pool_size=10,
    max_overflow=20,

    # Useful for detecting stale connections
    pool_pre_ping=True,

    # SQL logging; normally False in production
    echo=False,
)
```

The engine is responsible for managing connections to PostgreSQL.

Conceptually:

```text
FastAPI
   │
   ↓
SQLAlchemy Engine
   │
   ↓
Connection Pool
   │
   ├── Connection 1
   ├── Connection 2
   ├── Connection 3
   └── ...
          │
          ↓
      PostgreSQL
```

---

# Step 4 — Create the session factory

```python
# app/db/session.py

AsyncSessionLocal = async_sessionmaker(
    bind=engine,
    class_=AsyncSession,
    expire_on_commit=False,
)
```

This doesn't create one database connection forever.

Instead, it creates sessions when required.

---

# 2. What is a database session?

A **database session** represents a unit of interaction between your application and the database.

In SQLAlchemy:

```python
AsyncSession
```

is commonly used to:

* execute queries
* track ORM objects
* start/commit transactions
* rollback transactions
* flush changes
* manage database interaction

Think of it approximately as:

```text
Request
   │
   ↓
AsyncSession
   │
   ├── SELECT
   ├── INSERT
   ├── UPDATE
   ├── DELETE
   │
   ↓
commit / rollback
   │
   ↓
Request ends
   │
   ↓
Session closes
```

---

# Step 5 — Create the FastAPI database dependency

This is extremely important.

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

Then use it in FastAPI:

```python
from fastapi import Depends
from sqlalchemy.ext.asyncio import AsyncSession


@router.get("/users")
async def get_users(
    db: AsyncSession = Depends(get_db),
):
    ...
```

FastAPI effectively does:

```text
HTTP Request
     ↓
get_db()
     ↓
Create AsyncSession
     ↓
Endpoint
     ↓
Database queries
     ↓
Session closes
```

---

# 3. Why use `yield` in `get_db()`?

This:

```python
async def get_db():

    async with AsyncSessionLocal() as session:
        yield session
```

allows FastAPI to manage the lifecycle.

Conceptually:

```text
Before yield
     ↓
Create session
     ↓
yield session
     ↓
Endpoint executes
     ↓
Request finishes
     ↓
async with exits
     ↓
Session closes
```

This prevents database sessions from being left open.

---

# Step 6 — Create SQLAlchemy ORM model

Now we need a database model.

```python
# app/models/user.py

from sqlalchemy import String, Boolean
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
        index=True,
        nullable=False,
    )

    name: Mapped[str] = mapped_column(
        String(100),
        nullable=False,
    )

    is_active: Mapped[bool] = mapped_column(
        Boolean,
        default=True,
    )
```

This is where SQLAlchemy ORM comes in.

---

# 4. What is SQLAlchemy ORM?

ORM means:

> **Object Relational Mapping**

It maps:

```text
Python objects
      ↕
Database tables
```

For example:

```python
class User(Base):

    __tablename__ = "users"

    id: Mapped[int]

    email: Mapped[str]

    name: Mapped[str]
```

maps approximately to:

```sql
CREATE TABLE users (
    id INTEGER PRIMARY KEY,
    email VARCHAR(255) UNIQUE NOT NULL,
    name VARCHAR(100) NOT NULL
);
```

So instead of thinking only in terms of SQL tables, your Python application can work with Python objects.

---

# 5. ORM example

Without ORM, you might write:

```sql
SELECT id, email, name
FROM users
WHERE id = 10;
```

With SQLAlchemy ORM:

```python
from sqlalchemy import select

result = await db.execute(
    select(User).where(
        User.id == 10
    )
)

user = result.scalar_one_or_none()
```

Then:

```python
print(user.id)
print(user.email)
print(user.name)
```

SQLAlchemy translates the ORM expression into SQL.

Conceptually:

```text
Python
   ↓
select(User)
   ↓
SQLAlchemy
   ↓
SQL
   ↓
PostgreSQL
```

---

# 6. Create a GET endpoint

Let's create:

```http
GET /users/{user_id}
```

```python
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.dependencies import get_db
from app.models.user import User


router = APIRouter()


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

Request:

```http
GET /users/10
```

Flow:

```text
GET /users/10
       ↓
FastAPI
       ↓
get_db()
       ↓
AsyncSession
       ↓
select(User)
       ↓
PostgreSQL
       ↓
User object
       ↓
JSON response
```

---

# 7. Create a user

For example:

```python
from pydantic import BaseModel, EmailStr


class UserCreate(BaseModel):

    email: EmailStr

    name: str
```

Endpoint:

```python
@router.post("/users")
async def create_user(
    request: UserCreate,
    db: AsyncSession = Depends(get_db),
):

    user = User(
        email=request.email,
        name=request.name,
    )

    db.add(user)

    await db.commit()

    await db.refresh(user)

    return {
        "id": user.id,
        "email": user.email,
        "name": user.name,
    }
```

Important operations:

```python
db.add(user)
```

puts the object into the session.

Then:

```python
await db.commit()
```

commits the transaction.

Then:

```python
await db.refresh(user)
```

reloads the object from the database, which is useful for values generated by the DB such as an auto-incremented ID.

---

# 8. What happens during `commit()`?

Suppose:

```python
user = User(
    email="sandeep@example.com",
    name="Sandeep",
)

db.add(user)

await db.commit()
```

Conceptually:

```text
Python User object
       ↓
AsyncSession
       ↓
Transaction
       ↓
INSERT INTO users (...)
       ↓
PostgreSQL
       ↓
COMMIT
```

---

# 9. What is a transaction?

A transaction groups database operations into one logical unit.

For example, suppose you're transferring money:

```text
Account A
   ↓
- ₹1000

Account B
   ↓
+ ₹1000
```

You don't want:

```text
A → -₹1000
B → failure
```

Instead:

```text
BEGIN
   ↓
Debit A
   ↓
Credit B
   ↓
COMMIT
```

If something fails:

```text
BEGIN
   ↓
Debit A
   ↓
Credit B → ERROR
   ↓
ROLLBACK
```

---

# 10. Handling rollback

A robust service should handle failures.

```python
async def create_user(
    db: AsyncSession,
    email: str,
    name: str,
):

    try:

        user = User(
            email=email,
            name=name,
        )

        db.add(user)

        await db.commit()

        await db.refresh(user)

        return user

    except Exception:

        await db.rollback()

        raise
```

Why rollback?

Because after a failed transaction, the SQLAlchemy session can be in a failed transactional state.

---

# 11. SQLAlchemy ORM CRUD

## CREATE

```python
user = User(
    email="a@example.com",
    name="Alice",
)

db.add(user)

await db.commit()
```

---

## READ

```python
result = await db.execute(
    select(User).where(
        User.id == user_id
    )
)

user = result.scalar_one_or_none()
```

---

## UPDATE

```python
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

user.name = "Updated Name"

await db.commit()
```

---

## DELETE

```python
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

await db.delete(user)

await db.commit()
```

---

# 12. Why not put all database logic in the router?

You could do:

```python
@router.get("/users/{id}")
async def get_user(
    id: int,
    db: AsyncSession = Depends(get_db),
):

    result = await db.execute(
        select(User).where(User.id == id)
    )

    ...
```

But for a small application that's fine.

For a production application, I prefer:

```text
Router
   ↓
Service
   ↓
Repository
   ↓
SQLAlchemy
   ↓
PostgreSQL
```

---

# 13. Repository layer

The repository handles database operations.

```python
# app/repositories/user_repository.py

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

        result = await self.db.execute(
            select(User).where(
                User.id == user_id
            )
        )

        return result.scalar_one_or_none()

    async def get_by_email(
        self,
        email: str,
    ) -> User | None:

        result = await self.db.execute(
            select(User).where(
                User.email == email
            )
        )

        return result.scalar_one_or_none()

    async def create(
        self,
        user: User,
    ) -> User:

        self.db.add(user)

        await self.db.commit()

        await self.db.refresh(user)

        return user
```

---

# 14. Service layer

The service contains business logic.

```python
# app/services/user_service.py

from app.models.user import User
from app.repositories.user_repository import (
    UserRepository,
)


class UserService:

    def __init__(
        self,
        repository: UserRepository,
    ):
        self.repository = repository

    async def create_user(
        self,
        email: str,
        name: str,
    ):

        existing = await self.repository.get_by_email(
            email
        )

        if existing:
            raise ValueError(
                "User already exists"
            )

        user = User(
            email=email,
            name=name,
        )

        return await self.repository.create(
            user
        )
```

Now:

```text
Router
  ↓
Service
  ↓
Repository
  ↓
SQLAlchemy
  ↓
PostgreSQL
```

This separation becomes very useful as the application grows.

---

# 15. Dependency injection for the service

You can also inject the repository/service.

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

Endpoint:

```python
@router.post("/users")
async def create_user(
    request: UserCreate,
    service: UserService = Depends(
        get_user_service
    ),
):

    user = await service.create_user(
        email=request.email,
        name=request.name,
    )

    return user
```

Now FastAPI resolves:

```text
create_user()
      │
      ↓
get_user_service()
      │
      ↓
get_user_repository()
      │
      ↓
get_db()
      │
      ↓
AsyncSession
```

That's a very common production architecture.

---

# 16. SQLAlchemy ORM vs raw SQL

You should understand both.

### ORM

```python
result = await db.execute(
    select(User).where(
        User.email == email
    )
)
```

Advantages:

* Python objects
* relationships
* type safety
* composable queries
* database abstraction
* easier application development

### Raw SQL

```python
from sqlalchemy import text

result = await db.execute(
    text(
        """
        SELECT id, email, name
        FROM users
        WHERE email = :email
        """
    ),
    {
        "email": email
    },
)
```

Raw SQL is useful when:

* query is highly specialized
* you need database-specific functionality
* complex analytical queries are easier in SQL
* you need precise control over the generated SQL

SQLAlchemy actually supports both approaches.

---

# 17. SQLAlchemy ORM relationships

Suppose a user has many documents.

```python
from sqlalchemy import ForeignKey
from sqlalchemy.orm import relationship


class Document(Base):

    __tablename__ = "documents"

    id: Mapped[int] = mapped_column(
        primary_key=True
    )

    title: Mapped[str]

    owner_id: Mapped[int] = mapped_column(
        ForeignKey("users.id")
    )

    owner: Mapped["User"] = relationship(
        back_populates="documents"
    )
```

User:

```python
class User(Base):

    __tablename__ = "users"

    id: Mapped[int] = mapped_column(
        primary_key=True
    )

    email: Mapped[str]

    documents: Mapped[list["Document"]] = (
        relationship(
            back_populates="owner"
        )
    )
```

Now:

```python
user.documents
```

can represent the user's related documents.

---

# 18. Async SQLAlchemy: an important interview point

Don't confuse:

```python
async def
```

with:

```python
await
```

For async database calls:

```python
result = await db.execute(...)
```

The database operation is asynchronous.

For example:

```python
@router.get("/users")
async def get_users(
    db: AsyncSession = Depends(get_db),
):

    result = await db.execute(
        select(User)
    )

    return result.scalars().all()
```

You should **not** use a synchronous SQLAlchemy session inside an async endpoint and assume that makes the DB call asynchronous.

Bad architecture:

```python
async def endpoint():

    # synchronous DB operation
    result = sync_session.execute(...)
```

That can block the event-loop worker handling the request.

---

# 19. Connection pool

SQLAlchemy maintains a connection pool.

Instead of:

```text
Request 1 → create DB connection → destroy
Request 2 → create DB connection → destroy
Request 3 → create DB connection → destroy
```

you typically have:

```text
                SQLAlchemy
                    │
              Connection Pool
          ┌─────────┼─────────┐
          ↓         ↓         ↓
       Conn 1     Conn 2    Conn 3
          │         │         │
          └─────────┼─────────┘
                    ↓
                PostgreSQL
```

This is much more efficient.

Typical configuration:

```python
engine = create_async_engine(
    DATABASE_URL,
    pool_size=10,
    max_overflow=20,
    pool_pre_ping=True,
)
```

Don't blindly choose these numbers in production; tune them based on application concurrency and PostgreSQL's connection limits.

---

# 20. What happens to the connection after the request?

Suppose:

```python
@router.get("/users")
async def users(
    db: AsyncSession = Depends(get_db)
):
    ...
```

Flow:

```text
Request
   ↓
get_db()
   ↓
AsyncSession
   ↓
Acquire connection from pool
   ↓
Execute SQL
   ↓
Return connection to pool
   ↓
Close session
```

The connection is generally **returned to the pool**, not necessarily physically destroyed.

---

# 21. FastAPI + SQLAlchemy production architecture

For the kind of AI/RAG backend you've been preparing for, I'd structure it approximately like:

```text
                         FastAPI
                            │
                         Router
                            │
                            ↓
                        Service
                            │
              ┌─────────────┴─────────────┐
              ↓                           ↓
        Repository                  Other Services
              │
              ↓
        AsyncSession
              │
              ↓
       SQLAlchemy ORM
              │
              ↓
      PostgreSQL Connection Pool
              │
              ↓
         PostgreSQL
```

For example, your RAG application might have:

```text
POST /chat
     │
     ↓
ChatRouter
     │
     ↓
ChatService
     │
     ├── ConversationRepository
     │       ↓
     │   PostgreSQL
     │
     ├── RetrievalService
     │       ↓
     │   Qdrant
     │
     ├── CacheService
     │       ↓
     │   Redis
     │
     └── LLMService
             ↓
          LLM API
```

SQLAlchemy would typically handle the **transactional application data**:

```text
PostgreSQL
├── users
├── tenants
├── conversations
├── messages
├── documents
├── permissions
├── refresh_tokens
└── audit_logs
```

while Qdrant handles vector retrieval and Redis handles caching/session-like ephemeral state.

---

# 22. The three interview answers

## 1. How do you integrate SQLAlchemy with FastAPI?

A strong answer:

> **"I create an async SQLAlchemy engine using an async PostgreSQL driver such as asyncpg, configure an `async_sessionmaker`, and expose an `AsyncSession` through a FastAPI `Depends()` dependency using `yield`. The router receives the session through dependency injection. In a larger application I keep database operations in a repository and business logic in a service layer."**

Example:

```python
engine = create_async_engine(
    DATABASE_URL,
    pool_size=10,
    max_overflow=20,
)

SessionLocal = async_sessionmaker(
    engine,
    class_=AsyncSession,
    expire_on_commit=False,
)


async def get_db():
    async with SessionLocal() as session:
        yield session
```

Then:

```python
@router.get("/users")
async def users(
    db: AsyncSession = Depends(get_db),
):
    ...
```

---

# 23. What is a database session?

A strong answer:

> **"A SQLAlchemy `Session` or `AsyncSession` represents a unit of interaction with the database. It manages ORM objects, executes queries, and participates in transactions such as commit and rollback. In FastAPI I normally create one session per request or logical unit of work using a dependency and ensure it's closed after the request."**

Important distinction:

```text
Engine
 ↓
Connection Pool
 ↓
Connections

Session
 ↓
Uses those connections
 ↓
Executes queries / manages transaction
```

Don't say:

> "A session is a database connection."

That's not quite correct.

A session is an ORM/unit-of-work abstraction that **uses database connections**.

---

# 24. What is SQLAlchemy ORM?

A strong answer:

> **"SQLAlchemy ORM is SQLAlchemy's object-relational mapping layer. It maps Python classes to relational database tables and Python objects to database rows. It lets me work with typed Python models and construct queries using SQLAlchemy expressions while SQLAlchemy generates the corresponding SQL."**

Example:

```python
class User(Base):
    __tablename__ = "users"

    id: Mapped[int] = mapped_column(
        primary_key=True
    )

    email: Mapped[str]
```

Then:

```python
result = await db.execute(
    select(User).where(
        User.id == 10
    )
)
```

instead of manually writing:

```sql
SELECT *
FROM users
WHERE id = 10;
```

---

# 25. The key distinctions to memorize

```text
SQLAlchemy
    ↓
Database toolkit / ORM

Engine
    ↓
Manages database connectivity + pool

Session / AsyncSession
    ↓
Unit of work / ORM interaction / transactions

ORM Model
    ↓
Python class ↔ database table

Repository
    ↓
Database access abstraction

Service
    ↓
Business logic

FastAPI Dependency
    ↓
Injects DB session into request handling
```

And the production request flow:

```text
HTTP Request
     ↓
FastAPI Router
     ↓
Depends(get_db)
     ↓
AsyncSession
     ↓
Service
     ↓
Repository
     ↓
SQLAlchemy
     ↓
Connection Pool
     ↓
PostgreSQL
     ↓
Response
```

**That's the architecture I would describe in a senior FastAPI interview.**
