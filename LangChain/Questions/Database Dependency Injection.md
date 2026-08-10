Absolutely. **Database Dependency Injection** is the combination of two things:

1. **Database session management** — create, provide, and close a DB session correctly.
2. **FastAPI Dependency Injection** — inject that session into the endpoint/service that needs it.

For a production FastAPI + PostgreSQL + SQLAlchemy application, this is one of the most important patterns to understand.

genui{"data_networks_databases_learning_block":{"type_id":"SQL_JOIN"}}

# 1. The basic idea

Without dependency injection, you might write:

```python
@router.get("/users")
async def get_users():

    db = create_database_connection()

    users = await db.execute(...)

    db.close()

    return users
```

Every endpoint now needs to know:

* how to create a database connection
* how to configure it
* how to close it
* what happens when an exception occurs

Instead:

```python
@router.get("/users")
async def get_users(
    db: AsyncSession = Depends(get_db)
):
    ...
```

The endpoint simply says:

> **"I need a database session."**

FastAPI takes care of obtaining the dependency.

---

# 2. The architecture

A production application typically looks like:

```text
                    HTTP Request
                         |
                         v
                    FastAPI Router
                         |
                         |
                   Depends(get_db)
                         |
                         v
                  Database Session
                         |
                         v
                    Repository
                         |
                         v
                     SQLAlchemy
                         |
                         v
                    PostgreSQL
```

And after the request:

```text
Database Session
      |
      v
transaction cleanup
      |
      v
session.close()
      |
      v
connection returned to pool
```

That last part is very important.

---

# 3. First create the SQLAlchemy engine

For an async FastAPI application using PostgreSQL:

```python
from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

DATABASE_URL = (
    "postgresql+asyncpg://"
    "postgres:password@localhost:5432/app_db"
)

engine = create_async_engine(
    DATABASE_URL,
    pool_pre_ping=True,
)
```

The architecture is:

```text
FastAPI
   |
   v
SQLAlchemy Engine
   |
   v
Connection Pool
   |
   v
PostgreSQL
```

The engine is normally created **once for the application**, not once per HTTP request.

---

# 4. Create the session factory

```python
SessionLocal = async_sessionmaker(
    bind=engine,
    class_=AsyncSession,
    expire_on_commit=False,
)
```

Now `SessionLocal` can create sessions:

```python
async with SessionLocal() as session:
    ...
```

Think:

```text
Engine
  |
  v
Session Factory
  |
  +---- Session 1
  +---- Session 2
  +---- Session 3
  +---- Session 4
```

---

# 5. What is a database session?

This is often misunderstood.

A SQLAlchemy `AsyncSession` is **not simply a raw database connection**.

It manages things such as:

* database operations
* ORM identity state
* pending changes
* transactions
* communication with the engine

For example:

```python
db.add(user)

await db.commit()
```

The session is managing that unit of work.

---

# 6. Create `get_db()`

This is the heart of database dependency injection.

```python
from collections.abc import AsyncGenerator

from sqlalchemy.ext.asyncio import AsyncSession


async def get_db() -> AsyncGenerator[
    AsyncSession,
    None,
]:

    async with SessionLocal() as session:

        try:
            yield session

        except Exception:
            await session.rollback()
            raise
```

You can think of it as:

```text
Request starts
     |
     v
Create session
     |
     v
yield session
     |
     v
Endpoint/service uses session
     |
     v
Exception?
   /     \
 yes      no
  |        |
rollback  continue
  |        |
  +----+---+
       |
       v
session context closes
```

---

# 7. Why `yield`?

This is one of the most important FastAPI concepts.

When you write:

```python
yield session
```

you're effectively saying:

> "Give this resource to the endpoint, and when the request is finished, resume this dependency so cleanup can happen."

Conceptually:

```python
async def get_db():

    session = create_session()

    try:

        yield session

    finally:

        await session.close()
```

The `yield` divides the dependency into:

```text
Before yield
    ↓
Setup
    ↓
yield
    ↓
Application uses dependency
    ↓
After yield
    ↓
Cleanup
```

This pattern is useful for request-scoped resources.

---

# 8. Inject it into a FastAPI endpoint

Now:

```python
from fastapi import Depends


@router.get("/users")
async def get_users(
    db: AsyncSession = Depends(get_db),
):

    result = await db.execute(
        select(User)
    )

    return result.scalars().all()
```

FastAPI sees:

```python
Depends(get_db)
```

and resolves:

```text
get_db()
   |
   v
AsyncSession
   |
   v
get_users(db=session)
```

You don't manually call:

```python
get_db()
```

---

# 9. Why this is better

Without DI:

```text
Endpoint
   |
   +--> create DB
   +--> configure DB
   +--> query DB
   +--> handle errors
   +--> close DB
```

With DI:

```text
Endpoint
   |
   +--> use DB
```

The infrastructure concern is separated.

---

# 10. Real project structure

For a production project:

```text
app/
│
├── main.py
│
├── api/
│   └── v1/
│       ├── users.py
│       ├── documents.py
│       └── chat.py
│
├── db/
│   ├── base.py
│   └── session.py
│
├── models/
│   ├── user.py
│   ├── document.py
│   └── conversation.py
│
├── repositories/
│   ├── user_repository.py
│   ├── document_repository.py
│   └── conversation_repository.py
│
├── services/
│   ├── user_service.py
│   ├── document_service.py
│   └── chat_service.py
│
└── schemas/
    ├── user.py
    └── document.py
```

Put database infrastructure in:

```text
app/db/session.py
```

For example:

```python
from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from app.core.config import settings


engine = create_async_engine(
    settings.DATABASE_URL,
    pool_pre_ping=True,
)

SessionLocal = async_sessionmaker(
    engine,
    class_=AsyncSession,
    expire_on_commit=False,
)


async def get_db():
    async with SessionLocal() as session:

        try:
            yield session

        except Exception:
            await session.rollback()
            raise
```

---

# 11. Database dependency + Repository

Now combine this with the Repository Pattern you just learned.

Repository:

```python
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
```

Then dependency:

```python
def get_user_repository(
    db: AsyncSession = Depends(get_db),
):
    return UserRepository(db)
```

Then service:

```python
class UserService:

    def __init__(
        self,
        repository: UserRepository,
    ):
        self.repository = repository

    async def get_user(
        self,
        user_id: int,
    ):

        user = await self.repository.get_by_id(
            user_id
        )

        if not user:
            raise UserNotFoundError()

        return user
```

Dependency:

```python
def get_user_service(
    repository: UserRepository = Depends(
        get_user_repository
    ),
):
    return UserService(repository)
```

Finally:

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

Now look at the entire chain:

```text
HTTP Request
     |
     v
FastAPI Router
     |
     v
get_user_service()
     |
     v
get_user_repository()
     |
     v
get_db()
     |
     v
AsyncSession
     |
     v
UserRepository
     |
     v
PostgreSQL
```

This is **Dependency Injection + Repository Pattern + Service Pattern** working together.

---

# 12. Request-scoped database sessions

Usually, you want one appropriate session associated with a request's unit of work.

For example:

```text
Request A
   |
   +--> Session A
   |
   +--> Query
   +--> Query
   +--> Update
   |
   +--> Commit/Rollback
   |
   +--> Session cleanup


Request B
   |
   +--> Session B
   |
   +--> Query
   |
   +--> Session cleanup
```

Don't create one global `AsyncSession` and share it across all requests.

This is a common mistake.

Bad:

```python
db = AsyncSession(engine)
```

and then:

```python
@app.get(...)
async def endpoint():
    await db.execute(...)
```

A session is not intended to be a global shared object across concurrent requests.

---

# 13. Engine vs Session

This distinction is critical.

### Engine

Application-level infrastructure:

```text
Engine
   |
   v
Connection Pool
   |
   v
Database
```

Normally long-lived.

### Session

Unit-of-work/request-level object:

```text
Request
   |
   v
Session
   |
   v
Queries
   |
   v
Commit/Rollback
```

Normally short-lived.

So:

```text
Engine → long-lived

Session → short-lived
```

---

# 14. Connection pooling

Suppose your FastAPI application receives:

```text
100 concurrent requests
```

You don't want:

```text
100 requests
   |
   +--> 100 brand-new DB connections
```

Instead SQLAlchemy's engine manages a pool:

```text
                 SQLAlchemy Engine
                        |
                  Connection Pool
                 /   /   |   \   \
                /   /    |    \   \
              C1   C2    C3    C4   C5
                       |
                       v
                   PostgreSQL
```

When a session is finished, the underlying connection can be returned to the pool rather than permanently discarded.

This is why proper session cleanup matters.

---

# 15. `close()` vs connection pool

When you do:

```python
await session.close()
```

you should think:

> "I'm done with this session."

It doesn't necessarily mean:

> "Destroy the PostgreSQL server connection."

The connection can be returned to SQLAlchemy's pool for reuse.

That's one reason connection pooling improves performance.

---

# 16. Transaction handling

Here's an important production issue.

Suppose:

```text
Create Order
   |
   +--> Create order
   +--> Update inventory
   +--> Create payment record
```

You want:

```text
BEGIN
 |
 +--> INSERT order
 |
 +--> UPDATE inventory
 |
 +--> INSERT payment
 |
COMMIT
```

If payment fails:

```text
BEGIN
 |
 +--> INSERT order
 |
 +--> UPDATE inventory
 |
 +--> ERROR
 |
ROLLBACK
```

You can use:

```python
async with db.begin():

    order = ...
    db.add(order)

    inventory.quantity -= 1

    payment = ...
    db.add(payment)
```

This creates a transaction boundary.

---

# 17. Where should `commit()` happen?

This is an important architectural decision.

You generally don't want every repository method blindly doing:

```python
await self.db.commit()
```

For example:

```python
class UserRepository:

    async def create_user(self, user):
        self.db.add(user)
        await self.db.commit()
```

Then:

```python
class OrderRepository:

    async def create_order(self, order):
        self.db.add(order)
        await self.db.commit()
```

Now a service doing:

```text
create user
create order
create audit log
```

could have multiple independent commits.

That makes atomic operations harder.

A better approach for complex workflows is often:

```text
Service
   |
   v
Transaction boundary
   |
   +--> Repository 1
   |
   +--> Repository 2
   |
   +--> Repository 3
   |
   v
Commit
```

For example:

```python
async def create_order(
    db: AsyncSession,
    ...
):

    async with db.begin():

        order = await order_repo.create(...)

        await inventory_repo.reserve(...)

        await payment_repo.create(...)

        await audit_repo.create(...)
```

Now one business operation can have one transaction.

---

# 18. Database dependency in authentication

You can also use the database dependency inside authentication.

For example:

```python
async def get_current_user(
    token: str = Depends(oauth2_scheme),
    db: AsyncSession = Depends(get_db),
):
    payload = decode_token(token)

    user_id = payload["sub"]

    user = await db.get(
        User,
        user_id,
    )

    if not user:
        raise HTTPException(
            status_code=401,
            detail="User not found",
        )

    return user
```

Then:

```python
@router.get("/profile")
async def profile(
    user = Depends(get_current_user),
):
    return user
```

The dependency chain becomes:

```text
profile()
   |
   v
get_current_user()
   |
   +---- oauth2_scheme
   |
   +---- get_db()
            |
            v
        AsyncSession
```

This is a powerful FastAPI feature.

---

# 19. Multi-tenant database dependency

For an enterprise SaaS/RAG system, you might have:

```text
User
 |
 v
Authentication
 |
 v
Tenant Context
 |
 v
Database Session
 |
 v
Repository
```

For example:

```python
async def get_tenant_id(
    user = Depends(get_current_user),
) -> UUID:

    return user.tenant_id
```

Then:

```python
async def get_document_repository(
    db: AsyncSession = Depends(get_db),
    tenant_id: UUID = Depends(get_tenant_id),
):

    return DocumentRepository(
        db=db,
        tenant_id=tenant_id,
    )
```

Now repository methods can enforce tenant scoping:

```python
stmt = (
    select(Document)
    .where(
        Document.id == document_id,
        Document.tenant_id == self.tenant_id,
    )
)
```

So:

```text
Request
  |
  v
Authentication
  |
  v
Current User
  |
  v
Tenant ID
  |
  v
DB Session
  |
  v
Tenant-aware Repository
  |
  v
PostgreSQL
```

This is highly relevant to enterprise AI platforms.

---

# 20. Testing database dependencies

One of the biggest advantages of DI is testing.

Production:

```python
async def get_db():
    async with SessionLocal() as session:
        yield session
```

Test:

```python
async def get_test_db():
    async with TestSessionLocal() as session:
        yield session
```

Then:

```python
app.dependency_overrides[
    get_db
] = get_test_db
```

Your endpoint continues using:

```python
db: AsyncSession = Depends(get_db)
```

but during the test FastAPI gives it the test database.

Architecture:

```text
Production:

FastAPI
   |
 get_db
   |
PostgreSQL


Testing:

FastAPI
   |
 get_test_db
   |
Test PostgreSQL
```

This is dependency injection's major benefit.

---

# 21. Even better: testing without a database

If you're using the Repository Pattern:

```text
Router
   |
Service
   |
Repository interface
```

You can test the service with:

```text
Fake Repository
```

instead of PostgreSQL.

For example:

```python
class FakeUserRepository:

    async def get_by_id(self, user_id):

        return User(
            id=user_id,
            name="Test User",
            email="test@example.com",
        )
```

Then:

```python
service = UserService(
    repository=FakeUserRepository()
)
```

Now you're testing business logic without a real DB.

---

# 22. Database dependency in your RAG architecture

For the type of enterprise RAG platform you've been working through, I'd structure it approximately like:

```text
                    FastAPI
                       |
                       v
                    Router
                       |
              Dependency Injection
                       |
          +------------+-------------+
          |            |             |
          v            v             v
       Current       DB           Tenant
        User        Session        Context
          |            |             |
          +------------+-------------+
                       |
                       v
                  ChatService
                 /     |      \
                /      |       \
               v       v        v
          PostgreSQL Qdrant    LLM
              |        |
              v        v
        conversation  vectors
        metadata      chunks
        audit logs
        users
        tenants
```

A chat request might therefore use:

```python
@router.post("/chat")
async def chat(
    request: ChatRequest,
    service: ChatService = Depends(
        get_chat_service
    ),
):
    return await service.chat(request)
```

And:

```python
def get_chat_service(
    db: AsyncSession = Depends(get_db),
    user = Depends(get_current_user),
    qdrant = Depends(get_qdrant),
    llm = Depends(get_llm),
):

    return ChatService(
        db=db,
        user=user,
        qdrant=qdrant,
        llm=llm,
    )
```

The router remains extremely thin.

---

# 23. A production `db/session.py`

A reasonable starting point is:

```python
from collections.abc import AsyncGenerator

from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from app.core.config import settings


engine = create_async_engine(
    settings.DATABASE_URL,
    pool_pre_ping=True,
    echo=False,
)

SessionLocal = async_sessionmaker(
    bind=engine,
    class_=AsyncSession,
    expire_on_commit=False,
)


async def get_db() -> AsyncGenerator[
    AsyncSession,
    None,
]:

    async with SessionLocal() as session:

        try:
            yield session

        except Exception:

            await session.rollback()

            raise
```

Then your application can use:

```python
db: AsyncSession = Depends(get_db)
```

everywhere a request-scoped database session is needed.

---

# 24. Don't put everything in `get_db()`

A common mistake is making the database dependency huge:

```python
async def get_db():

    # authenticate
    # check tenant
    # create repository
    # execute business rules
    # query users
    # call Redis
    # call Qdrant
    # call LLM
```

Don't do this.

`get_db()` should primarily be responsible for:

```text
Create session
     ↓
Provide session
     ↓
Cleanup/rollback
```

Keep business logic elsewhere.

---

# 25. Dependency Injection hierarchy

A clean production hierarchy might look like:

```text
get_db()
   |
   v
AsyncSession
   |
   v
get_user_repository()
   |
   v
UserRepository
   |
   v
get_user_service()
   |
   v
UserService
   |
   v
Router
```

And for RAG:

```text
get_db()
get_current_user()
get_qdrant()
get_llm()
       |
       v
get_chat_service()
       |
       v
ChatService
       |
       v
Router
```

FastAPI resolves this dependency graph automatically.

---

# 26. Important production rules

### Rule 1 — Don't create sessions globally

Avoid:

```python
db = AsyncSession(...)
```

as a shared global.

Use a dependency/session factory.

---

### Rule 2 — Don't create the engine per request

Avoid:

```python
async def get_db():

    engine = create_async_engine(...)
```

This defeats connection pooling and wastes resources.

Create the engine once.

---

### Rule 3 — Close sessions

Use:

```python
async with SessionLocal() as session:
```

or an equivalent lifecycle-managed approach.

---

### Rule 4 — Handle rollback correctly

If a transaction fails:

```python
await session.rollback()
```

before trying to continue using that session.

---

### Rule 5 — Don't commit blindly inside every repository method

For complex business operations, coordinate the transaction at the appropriate service/unit-of-work boundary.

---

### Rule 6 — Don't share `AsyncSession` between concurrent tasks

A session is not a globally shareable concurrent object.

Each request/task should have appropriate session ownership.

---

# 27. Interview answer

If an interviewer asks:

> **"How do you implement database dependency injection in FastAPI?"**

A strong answer is:

> "I create a long-lived SQLAlchemy async engine and session factory at application initialization. Then I expose an async generator such as `get_db()` that creates an `AsyncSession`, yields it to the request, rolls back on exceptions, and allows the context manager to clean it up afterward. FastAPI's `Depends(get_db)` injects that session into routers or service dependencies. In a larger application, I typically inject the session into repositories, while services coordinate business logic and transaction boundaries. This keeps database lifecycle management separate from HTTP and business logic and also makes the database dependency easy to override in tests."

---

# 28. The complete mental model

Remember this:

```text
             APPLICATION STARTUP
                    |
                    v
             SQLAlchemy Engine
                    |
                    v
              Connection Pool
                    |
                    v
             Session Factory
                    |
                    |
             HTTP REQUEST
                    |
                    v
             FastAPI Router
                    |
                    v
             Depends(get_db)
                    |
                    v
             AsyncSession
                    |
                    v
              Service Layer
                    |
                    v
             Repository Layer
                    |
                    v
               PostgreSQL
                    |
                    v
              Request ends
                    |
                    v
          rollback if necessary
                    |
                    v
            session cleanup
                    |
                    v
       connection returned to pool
```

The key distinction is:

**Engine = application-level database infrastructure**

**Session = short-lived unit of work**

**`get_db()` = dependency that provides/manages the session**

**`Depends(get_db)` = asks FastAPI to inject that session**

**Repository = uses the session for persistence**

**Service = coordinates business logic and, where appropriate, transactions**

This pattern is the foundation for a clean **FastAPI + SQLAlchemy + PostgreSQL** architecture, and it fits directly with the **Router → Service → Repository → Database** architecture we've been discussing.
