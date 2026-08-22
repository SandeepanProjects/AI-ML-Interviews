These are **very important FastAPI interview questions**, especially for a senior role. The key idea is that FastAPI's Dependency Injection (DI) system lets you keep **routers thin** and move reusable infrastructure concerns—database sessions, authentication, configuration, services, etc.—outside your endpoint logic.

---

# 1. What is Dependency Injection?

**Dependency Injection means that an object/function receives the things it depends on from the outside instead of creating those dependencies itself.**

Consider this bad design:

```python
@app.get("/users")
async def get_users():

    db = create_database_connection()

    users = db.query_users()

    return users
```

The endpoint is responsible for:

```text
Create DB connection
       ↓
Use DB
       ↓
Query users
```

This creates tight coupling.

With dependency injection:

```python
@app.get("/users")
async def get_users(
    db: AsyncSession = Depends(get_db)
):
    users = await db.execute(...)
    return users
```

Now:

```text
FastAPI
   ↓
get_db()
   ↓
AsyncSession
   ↓
Endpoint
```

The endpoint doesn't need to know **how the session is created**.

---

# 2. Why does FastAPI use Dependency Injection?

FastAPI's DI system is useful for several reasons.

## A. Reusability

You can create one dependency:

```python
async def get_db():
    ...
```

and use it in many endpoints:

```python
@app.get("/users")
async def users(db = Depends(get_db)):
    ...


@app.get("/orders")
async def orders(db = Depends(get_db)):
    ...


@app.get("/payments")
async def payments(db = Depends(get_db)):
    ...
```

Instead of repeating database setup everywhere.

---

## B. Separation of concerns

Your router should primarily handle HTTP concerns:

```text
HTTP Request
     ↓
Router
     ↓
Service
     ↓
Repository
     ↓
Database
```

The router shouldn't contain:

```python
create DB connection
validate JWT
create Redis client
load configuration
```

Instead, dependencies provide these things.

---

## C. Testability

This is a **very important senior-level benefit**.

Suppose:

```python
@app.get("/users")
async def get_users(
    db: AsyncSession = Depends(get_db)
):
    ...
```

During testing, you can replace `get_db` with a test database/session.

Conceptually:

```text
Production

get_db()
   ↓
PostgreSQL


Test

get_db()
   ↓
Test DB / mock
```

This is called **dependency overriding**.

---

## D. Authentication and authorization

For example:

```python
async def get_current_user():
    ...
```

Then:

```python
@app.get("/profile")
async def profile(
    user = Depends(get_current_user)
):
    return user
```

You can reuse the same authentication dependency across many endpoints.

---

## E. Resource lifecycle management

Dependencies can create and clean up resources.

For example:

```python
async def get_db():
    db = create_session()

    try:
        yield db
    finally:
        await db.close()
```

The `finally` block ensures cleanup.

This is particularly useful for:

* database sessions
* files
* transactions
* connections
* other resources

---

# 3. What is `Depends()`?

`Depends()` tells FastAPI:

> **"This parameter should be provided by FastAPI's dependency injection system."**

Example:

```python
from fastapi import Depends


def get_config():
    return {
        "environment": "production"
    }


@app.get("/config")
async def config(
    settings = Depends(get_config)
):
    return settings
```

FastAPI sees:

```python
Depends(get_config)
```

and calls:

```python
get_config()
```

Then injects the result into:

```python
settings
```

Conceptually:

```text
Endpoint
   │
   │ needs settings
   ↓
Depends(get_config)
   │
   ↓
FastAPI calls get_config()
   │
   ↓
returns settings
   │
   ↓
endpoint receives settings
```

---

# 4. Simple dependency example

```python
def get_user():
    return {
        "id": 123,
        "name": "Sandeep"
    }


@app.get("/profile")
async def profile(
    user = Depends(get_user)
):
    return user
```

FastAPI effectively does:

```text
get_user()
    ↓
user
    ↓
profile(user)
```

You don't manually call:

```python
user = get_user()
```

FastAPI handles it.

---

# 5. Dependency can itself have dependencies

This is where FastAPI DI becomes powerful.

Suppose:

```python
async def get_db():
    ...


async def get_current_user(
    db = Depends(get_db)
):
    ...
```

Then:

```python
@app.get("/profile")
async def profile(
    user = Depends(get_current_user)
):
    return user
```

FastAPI builds the dependency graph:

```text
profile()
   │
   ↓
get_current_user()
   │
   ↓
get_db()
```

So the dependency system can form a **dependency graph**.

---

# 6. How do you inject a database session?

For a production FastAPI application using **SQLAlchemy async + PostgreSQL**, a common pattern is:

```text
FastAPI
   ↓
get_db()
   ↓
AsyncSession
   ↓
Router
   ↓
Service
   ↓
Repository
   ↓
PostgreSQL
```

---

# 7. Create async SQLAlchemy engine

```python
from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)


DATABASE_URL = (
    "postgresql+asyncpg://user:password@localhost/mydb"
)

engine = create_async_engine(
    DATABASE_URL,
    pool_pre_ping=True,
)

SessionLocal = async_sessionmaker(
    bind=engine,
    class_=AsyncSession,
    expire_on_commit=False,
)
```

---

# 8. Create the database dependency

```python
from collections.abc import AsyncGenerator


async def get_db() -> AsyncGenerator[AsyncSession, None]:

    async with SessionLocal() as session:
        yield session
```

The important part is:

```python
yield session
```

FastAPI injects that session into the endpoint.

When the request finishes, the context manager handles cleanup.

---

# 9. Inject it into an endpoint

```python
from fastapi import Depends
from sqlalchemy import select


@app.get("/users")
async def get_users(
    db: AsyncSession = Depends(get_db)
):

    result = await db.execute(
        select(User)
    )

    users = result.scalars().all()

    return users
```

The flow is:

```text
GET /users
    ↓
FastAPI
    ↓
Depends(get_db)
    ↓
SessionLocal()
    ↓
AsyncSession
    ↓
get_users(db)
    ↓
await db.execute(...)
    ↓
PostgreSQL
```

---

# 10. Why use `yield` in a DB dependency?

Compare:

```python
async def get_db():
    return SessionLocal()
```

with:

```python
async def get_db():
    async with SessionLocal() as session:
        yield session
```

The second approach gives you lifecycle management.

Conceptually:

```text
Request starts
     ↓
Create DB session
     ↓
yield session
     ↓
Endpoint executes
     ↓
Request finishes
     ↓
Close session
```

This is much safer for connection management.

---

# 11. Better production architecture

For the kind of AI/RAG backend you're preparing for, I'd structure it approximately like this:

```text
app/
│
├── api/
│   └── routes/
│       └── chat.py
│
├── services/
│   └── chat_service.py
│
├── repositories/
│   └── document_repository.py
│
├── db/
│   ├── engine.py
│   └── dependencies.py
│
├── models/
│   ├── database.py
│   └── schemas.py
│
└── core/
    └── config.py
```

Then:

### `db/dependencies.py`

```python
async def get_db() -> AsyncGenerator[AsyncSession, None]:
    async with SessionLocal() as session:
        yield session
```

### Router

```python
@router.get("/documents")
async def get_documents(
    db: AsyncSession = Depends(get_db),
):
    return await document_service.get_documents(db)
```

### Service

```python
class DocumentService:

    async def get_documents(
        self,
        db: AsyncSession,
    ):
        return await self.repository.get_documents(db)
```

### Repository

```python
class DocumentRepository:

    async def get_documents(
        self,
        db: AsyncSession,
    ):
        result = await db.execute(
            select(Document)
        )

        return result.scalars().all()
```

Now responsibilities are separated:

```text
Router
  ↓
HTTP concerns

Service
  ↓
Business logic

Repository
  ↓
Database access

Dependency
  ↓
Database lifecycle
```

---

# 12. Dependency Injection for services

You don't have to inject only databases.

For example:

```python
def get_chat_service() -> ChatService:
    return ChatService()
```

Then:

```python
@router.post("/chat")
async def chat(
    request: ChatRequest,
    service: ChatService = Depends(get_chat_service),
):
    return await service.generate(request)
```

Now the router doesn't construct the service.

---

# 13. Dependency Injection for authentication

A common pattern:

```python
async def get_current_user(
    token: str = Depends(oauth2_scheme),
):
    payload = decode_token(token)

    return payload
```

Then:

```python
@router.get("/profile")
async def profile(
    current_user = Depends(get_current_user),
):
    return current_user
```

You can then build RBAC on top:

```python
def require_admin(
    current_user = Depends(get_current_user),
):
    if current_user["role"] != "admin":
        raise HTTPException(
            status_code=403,
            detail="Admin access required",
        )

    return current_user
```

And:

```python
@router.delete("/users/{user_id}")
async def delete_user(
    user_id: int,
    admin = Depends(require_admin),
):
    ...
```

Dependency graph:

```text
delete_user()
     │
     ↓
require_admin()
     │
     ↓
get_current_user()
     │
     ↓
OAuth/JWT
```

---

# 14. Dependency overrides in testing

This is one of the strongest reasons to use DI.

Production:

```python
app.dependency_overrides
```

can replace:

```python
get_db
```

with a test dependency.

For example:

```python
async def override_get_db():
    async with TestSessionLocal() as session:
        yield session


app.dependency_overrides[get_db] = override_get_db
```

Now your endpoint:

```python
db: AsyncSession = Depends(get_db)
```

receives the test session instead.

This means your test doesn't need to modify production code.

---

# 15. Interview traps

### ❌ Don't say:

> "`Depends()` creates the dependency."

More accurately:

> **`Depends()` declares that a parameter should be resolved by FastAPI's dependency injection system.**

---

### ❌ Don't say:

> "Dependency injection is only for database connections."

It can be used for:

```text
Database
Authentication
Authorization
Services
Repositories
Configuration
Redis
HTTP clients
Feature flags
Current user
Tenant context
```

---

### ❌ Don't create DB connections inside every endpoint

Avoid:

```python
@app.get("/users")
async def users():

    db = create_connection()

    ...
```

Prefer:

```python
@app.get("/users")
async def users(
    db: AsyncSession = Depends(get_db)
):
    ...
```

---

# Senior interview answer

If asked:

### **"What is Dependency Injection?"**

> **"Dependency Injection is a design pattern where a component receives its dependencies from an external provider rather than constructing them itself. It reduces coupling, improves testability, and makes resource lifecycle management easier."**

### **"Why does FastAPI use Dependency Injection?"**

> **"FastAPI uses DI to provide reusable and composable dependencies such as database sessions, authentication, authorization, configuration, Redis clients, and services. It also makes testing easier through dependency overrides."**

### **"What is `Depends()`?"**

> **"`Depends()` declares a dependency in a FastAPI endpoint or another dependency. FastAPI resolves the dependency, executes it, and injects its result into the parameter."**

### **"How do you inject a database session?"**

> **"I create an async SQLAlchemy session factory, expose an async generator dependency using `yield`, and inject it with `Depends(get_db)`. The dependency manages the session lifecycle, while the endpoint or service uses `await` for database operations."**

The mental model to remember is:

```text
                    FastAPI
                       │
                 Depends(...)
                       │
          ┌────────────┼────────────┐
          ↓            ↓            ↓
        get_db     get_user     get_config
          │            │
          ↓            ↓
    AsyncSession    JWT/User
          │            │
          └──────┬─────┘
                 ↓
              Endpoint
```

**DI = "I need this dependency; FastAPI figures out how to provide it."**
