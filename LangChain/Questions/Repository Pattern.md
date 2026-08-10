Absolutely. The **Repository Pattern** is one of the most useful patterns to understand when building a production FastAPI application with **PostgreSQL + SQLAlchemy**.

The simplest definition is:

> **A Repository is a layer that hides database-access details from the business/service layer.**

Instead of your service knowing how to write SQLAlchemy queries, it asks the repository for data.

---

# 1. The problem Repository Pattern solves

Imagine your FastAPI endpoint directly talks to SQLAlchemy:

```python
@router.get("/users/{user_id}")
async def get_user(
    user_id: int,
    db: AsyncSession = Depends(get_db),
):
    result = await db.execute(
        select(User).where(User.id == user_id)
    )

    user = result.scalar_one_or_none()

    if not user:
        raise HTTPException(
            status_code=404,
            detail="User not found",
        )

    return user
```

This works.

But now imagine you have:

```text
50 endpoints
20 services
30 database queries
```

You'll end up with SQLAlchemy code everywhere.

```text
Router
  |
  +--> SQLAlchemy

Service
  |
  +--> SQLAlchemy

Another Service
  |
  +--> SQLAlchemy

Another Router
  |
  +--> SQLAlchemy
```

That's difficult to maintain.

---

# 2. Repository Pattern changes the architecture

Instead:

```text
Router
   |
   v
Service
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

Now:

### Router

Handles HTTP.

### Service

Handles business logic.

### Repository

Handles data access.

### SQLAlchemy

Handles database communication.

---

# 3. Simple example

Suppose you need:

```python
get_user_by_id()
```

Create:

```text
repositories/
    user_repository.py
```

Then:

```python
class UserRepository:

    def __init__(
        self,
        session: AsyncSession,
    ):
        self.session = session

    async def get_by_id(
        self,
        user_id: int,
    ) -> User | None:

        result = await self.session.execute(
            select(User).where(
                User.id == user_id
            )
        )

        return result.scalar_one_or_none()
```

Now the service doesn't need to know about:

```python
select(...)
where(...)
scalar_one_or_none()
```

It simply does:

```python
user = await repository.get_by_id(user_id)
```

---

# 4. Think of Repository as a database gateway

Imagine your database is:

```text
PostgreSQL
```

The service shouldn't need to know:

```text
How PostgreSQL works
How SQLAlchemy works
How joins are written
How queries are constructed
How rows are mapped
```

Instead:

```text
             Service
                |
                | "Give me user 123"
                v
        UserRepository
                |
                | SQLAlchemy query
                v
           PostgreSQL
```

The repository acts as a **gateway to persistence**.

---

# 5. Repository responsibilities

A repository typically handles:

### Reading

```python
get_by_id()
get_by_email()
list_users()
search_users()
```

### Creating

```python
create()
```

### Updating

```python
update()
```

### Deleting

```python
delete()
```

### Database-specific queries

```python
find_active_users()
find_documents_by_tenant()
get_conversation_messages()
```

The repository is primarily concerned with:

> **How do I get/store data?**

It should generally not decide:

> **What does the business want to do?**

That's the service's responsibility.

---

# 6. Repository vs Service

This distinction is extremely important for interviews.

Suppose you're processing an order.

### Repository

```python
async def get_product(
    product_id: int
):
    ...
```

It retrieves a product.

### Service

```python
async def place_order(
    user_id: int,
    product_id: int,
):

    product = await repository.get_product(
        product_id
    )

    if product.stock <= 0:
        raise OutOfStockError()

    ...

```

The service decides:

```text
Is the product available?
Can this user buy it?
How much should they pay?
Should the order be created?
```

The repository decides:

```text
How do I retrieve the product?
How do I save the order?
```

---

# 7. The most important rule

A good rule is:

> **Repository = persistence logic.**

> **Service = business logic.**

For example:

### Bad repository

```python
class OrderRepository:

    async def place_order(
        self,
        user_id,
        product_id,
    ):

        if product.stock <= 0:
            raise Exception("Out of stock")

        if user.balance < price:
            raise Exception("Insufficient balance")

        ...
```

This repository contains business logic.

Not ideal.

Instead:

```text
Service
  |
  +--> Check stock
  +--> Check balance
  +--> Calculate price
  +--> Apply business rules
  |
  v
Repository
  |
  +--> Save order
```

---

# 8. Basic production structure

For FastAPI:

```text
app/
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
│   └── user.py
│
├── schemas/
│   └── user.py
│
└── db/
    └── session.py
```

The flow:

```text
HTTP
 |
 v
users.py
 |
 v
UserService
 |
 v
UserRepository
 |
 v
SQLAlchemy
 |
 v
PostgreSQL
```

---

# 9. Let's build a real example

Suppose we have:

```python
class User(Base):
    __tablename__ = "users"

    id: Mapped[int] = mapped_column(
        primary_key=True
    )

    name: Mapped[str]

    email: Mapped[str]
```

Now create the repository.

## `user_repository.py`

```python
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.user import User


class UserRepository:

    def __init__(
        self,
        session: AsyncSession,
    ):
        self.session = session

    async def get_by_id(
        self,
        user_id: int,
    ) -> User | None:

        stmt = select(User).where(
            User.id == user_id
        )

        result = await self.session.execute(
            stmt
        )

        return result.scalar_one_or_none()
```

This repository now owns the query.

---

# 10. Add more repository operations

```python
class UserRepository:

    def __init__(
        self,
        session: AsyncSession,
    ):
        self.session = session

    async def get_by_id(
        self,
        user_id: int,
    ) -> User | None:

        stmt = select(User).where(
            User.id == user_id
        )

        result = await self.session.execute(stmt)

        return result.scalar_one_or_none()

    async def get_by_email(
        self,
        email: str,
    ) -> User | None:

        stmt = select(User).where(
            User.email == email
        )

        result = await self.session.execute(stmt)

        return result.scalar_one_or_none()

    async def list_users(
        self,
        limit: int = 100,
        offset: int = 0,
    ) -> list[User]:

        stmt = (
            select(User)
            .offset(offset)
            .limit(limit)
        )

        result = await self.session.execute(stmt)

        return list(result.scalars().all())

    async def create(
        self,
        user: User,
    ) -> User:

        self.session.add(user)

        await self.session.flush()

        await self.session.refresh(user)

        return user

    async def delete(
        self,
        user: User,
    ) -> None:

        await self.session.delete(user)
```

Now all user persistence logic is in one place.

---

# 11. Why `flush()` instead of `commit()`?

This is an important production concept.

Suppose:

```python
user = User(...)
```

Then:

```python
session.add(user)
```

The database may not have received the INSERT yet.

Calling:

```python
await session.flush()
```

sends pending changes to the database **within the current transaction**, without committing the transaction.

That allows the service to do:

```text
BEGIN
 |
 +--> create user
 |
 +--> create profile
 |
 +--> create audit record
 |
COMMIT
```

If the repository commits independently:

```python
await session.commit()
```

you can accidentally create transaction boundaries inside individual repository methods.

That's often undesirable for multi-step business operations.

---

# 12. Transaction ownership

This is a subtle but important production design decision.

A common pattern is:

```text
Service
  |
  +---- Repository A
  |
  +---- Repository B
  |
  +---- Repository C
  |
  v
One transaction
```

For example:

```python
async def create_order(...):

    async with session.begin():

        order = await order_repo.create(...)

        await inventory_repo.reduce_stock(...)

        await payment_repo.create(...)

```

If payment fails:

```text
ROLLBACK
```

Everything is rolled back.

This is usually preferable to having each repository independently call `commit()`.

---

# 13. Service layer

Now:

```text
services/user_service.py
```

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
    ) -> User:

        user = await self.repository.get_by_id(
            user_id
        )

        if not user:
            raise UserNotFoundError(
                user_id
            )

        return user
```

The service doesn't know:

```text
SQLAlchemy
select()
where()
PostgreSQL
```

It just knows:

```python
repository.get_by_id()
```

---

# 14. FastAPI dependency injection

Now connect everything.

```python
def get_user_repository(
    db: AsyncSession = Depends(get_db),
) -> UserRepository:

    return UserRepository(db)
```

Then:

```python
def get_user_service(
    repository: UserRepository = Depends(
        get_user_repository
    ),
) -> UserService:

    return UserService(repository)
```

Then your router:

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

This gives:

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
PostgreSQL
```

This is where **Dependency Injection + Repository Pattern + Service Pattern** work together.

---

# 15. Why not put Repository directly in Router?

You could:

```python
@router.get("/{user_id}")
async def get_user(
    user_id: int,
    repository: UserRepository = Depends(
        get_user_repository
    ),
):
    return await repository.get_by_id(user_id)
```

Technically this works.

But you're bypassing the service layer.

For simple CRUD applications, that can be perfectly reasonable.

For complex business systems, I'd use:

```text
Router
 ↓
Service
 ↓
Repository
```

because business rules tend to grow.

---

# 16. Repository Pattern in a RAG application

This becomes much more interesting for your enterprise AI project.

Suppose PostgreSQL contains:

```text
tenants
users
documents
document_versions
conversations
messages
agent_runs
audit_logs
```

You could have:

```text
repositories/
│
├── user_repository.py
├── tenant_repository.py
├── document_repository.py
├── conversation_repository.py
├── message_repository.py
└── agent_run_repository.py
```

And services:

```text
services/
│
├── auth_service.py
├── document_service.py
├── chat_service.py
└── agent_service.py
```

---

# 17. Document repository

For example:

```python
class DocumentRepository:

    def __init__(
        self,
        session: AsyncSession,
    ):
        self.session = session

    async def get_by_id(
        self,
        document_id: UUID,
        tenant_id: UUID,
    ) -> Document | None:

        stmt = (
            select(Document)
            .where(
                Document.id == document_id,
                Document.tenant_id == tenant_id,
            )
        )

        result = await self.session.execute(stmt)

        return result.scalar_one_or_none()
```

Notice something important:

```python
Document.tenant_id == tenant_id
```

This helps enforce tenant isolation at the persistence boundary.

But don't rely on this alone for authorization. Your service/security layer should still validate access.

---

# 18. Document service

Then:

```python
class DocumentService:

    def __init__(
        self,
        repository: DocumentRepository,
        qdrant: QdrantClient,
        embedding_service: EmbeddingService,
    ):
        self.repository = repository
        self.qdrant = qdrant
        self.embedding_service = embedding_service
```

Then:

```python
async def delete_document(
    self,
    document_id: UUID,
    tenant_id: UUID,
):

    document = await self.repository.get_by_id(
        document_id,
        tenant_id,
    )

    if not document:
        raise DocumentNotFoundError()

    # Business operation

    await self.qdrant.delete(
        document_id
    )

    await self.repository.delete(
        document
    )
```

Now the responsibilities are clear.

```text
DocumentService
       |
       +---- Business rules
       |
       +---- Qdrant
       |
       +---- DocumentRepository
                    |
                    v
                PostgreSQL
```

---

# 19. PostgreSQL vs Qdrant

This is particularly important for RAG.

You should **not** think of your repository as necessarily meaning "everything related to data."

You can have different abstractions:

```text
DocumentRepository
       |
       v
PostgreSQL


VectorRepository
       |
       v
Qdrant


CacheRepository / CacheClient
       |
       v
Redis
```

For example:

```python
class VectorRepository:

    async def search(
        self,
        embedding: list[float],
        tenant_id: str,
        limit: int,
    ):
        ...
```

Then your service doesn't need to know the details of the Qdrant API.

---

# 20. Repository Pattern and abstraction

This is where the pattern becomes more powerful.

You can define an interface:

```python
from typing import Protocol


class UserRepositoryProtocol(Protocol):

    async def get_by_id(
        self,
        user_id: int,
    ) -> User | None:
        ...
```

Then:

```python
class SQLAlchemyUserRepository:

    async def get_by_id(
        self,
        user_id: int,
    ) -> User | None:
        ...
```

Your service can depend on the abstraction:

```python
class UserService:

    def __init__(
        self,
        repository: UserRepositoryProtocol,
    ):
        self.repository = repository
```

Now you can have:

```text
UserService
     |
     v
UserRepositoryProtocol
     |
     +---- SQLAlchemyUserRepository
     |
     +---- MockUserRepository
     |
     +---- InMemoryUserRepository
```

This is useful for testing and for isolating infrastructure.

---

# 21. Testing becomes easier

Without repository abstraction:

```text
Test
 |
 v
Service
 |
 v
SQLAlchemy
 |
 v
PostgreSQL
```

Your unit test now needs a database.

With repository abstraction:

```text
Test
 |
 v
Service
 |
 v
FakeRepository
```

Example:

```python
class FakeUserRepository:

    async def get_by_id(
        self,
        user_id: int,
    ):

        return User(
            id=user_id,
            name="Test User",
            email="test@example.com",
        )
```

Then:

```python
service = UserService(
    FakeUserRepository()
)
```

Now you're testing business logic without PostgreSQL.

---

# 22. Repository Pattern is not mandatory

This is important.

Don't blindly create repositories for every application.

For a tiny CRUD API:

```text
Router
  |
  v
SQLAlchemy
```

may be enough.

For a medium application:

```text
Router
  |
  v
Service
  |
  v
SQLAlchemy
```

may be enough.

For a large enterprise application:

```text
Router
  |
  v
Service
  |
  v
Repository
  |
  v
SQLAlchemy
```

is often useful.

The pattern has a cost: more files, abstractions, and indirection.

Use it when the complexity justifies it.

---

# 23. Common mistake: Generic Repository

You might see people create:

```python
class GenericRepository:

    async def create(...):
        ...

    async def get(...):
        ...

    async def update(...):
        ...

    async def delete(...):
        ...
```

and then:

```python
UserRepository(GenericRepository)
OrderRepository(GenericRepository)
DocumentRepository(GenericRepository)
```

This can look elegant.

But be careful.

Real applications have domain-specific queries:

```python
find_active_subscriptions()
get_documents_for_tenant()
get_latest_document_version()
find_unprocessed_agent_runs()
get_conversation_history()
```

A giant generic repository often becomes awkward.

I generally prefer:

```text
UserRepository
DocumentRepository
ConversationRepository
AgentRunRepository
```

with domain-specific methods.

---

# 24. Repository should not become a "God class"

Bad:

```python
class UserRepository:

    async def get_user():
        ...

    async def create_order():
        ...

    async def search_documents():
        ...

    async def get_agent_runs():
        ...

    async def update_payment():
        ...
```

That's not a user repository anymore.

Keep repositories aligned with their domain.

---

# 25. Repository Pattern vs DAO

You may hear **DAO — Data Access Object**.

They're closely related.

A DAO generally focuses on:

> Data access operations.

Repository generally has a slightly broader domain-oriented abstraction:

> Collection/gateway through which the application accesses domain persistence.

In many Python/FastAPI projects, you'll see the terms used somewhat interchangeably.

Don't get too hung up on the naming. The important thing is the separation of persistence logic from business logic.

---

# 26. The complete enterprise architecture

For the kind of systems you're preparing for:

```text
                         Client
                           |
                           v
                        FastAPI
                           |
                           v
                         Router
                           |
                           v
                    Dependency Injection
                           |
                           v
                        Service
                    /       |       \
                   /        |        \
                  v         v         v
             Repository   Qdrant    LLM
                  |
                  v
             SQLAlchemy
                  |
                  v
              PostgreSQL
```

For a RAG request:

```text
POST /api/v1/chat
       |
       v
ChatRouter
       |
       v
ChatService
       |
       +----------------------+
       |                      |
       v                      v
ConversationRepository    VectorRepository
       |                      |
       v                      v
   PostgreSQL               Qdrant
       |
       v
 conversation history

       ChatService
            |
            v
       LLM Provider
            |
            v
         Response
```

This is a very realistic architecture.

---

# 27. The three layers you should remember

### Router

**Question:**

> "How do I handle this HTTP request?"

```python
@router.get("/{id}")
```

---

### Service

**Question:**

> "What should the application do?"

```python
await service.delete_document(...)
```

Business rules live here.

---

### Repository

**Question:**

> "How do I persist/retrieve the data?"

```python
await repository.get_by_id(...)
```

Database access lives here.

---

# 28. Interview answer

If an interviewer asks:

> **"Explain the Repository Pattern."**

A strong senior-level answer:

> "The Repository Pattern abstracts persistence and data-access logic from the business layer. In a FastAPI application, I typically structure the flow as Router → Service → Repository → SQLAlchemy → PostgreSQL. The router handles HTTP concerns, the service contains business rules, and the repository encapsulates database queries and persistence operations. This improves separation of concerns, testability, and maintainability. For example, a `DocumentService` can call `DocumentRepository.get_by_id()` without knowing whether the implementation uses SQLAlchemy, another database, or a test double. I also prefer keeping transaction ownership at the service or unit-of-work level rather than having every repository method independently commit."

That last sentence is particularly useful in a **senior/staff interview**.

---

## The mental model

Remember:

```text
Router
   ↓
"What HTTP request is this?"

Service
   ↓
"What business operation should happen?"

Repository
   ↓
"How do I get/save the data?"

SQLAlchemy
   ↓
"How do I translate that into database operations?"

PostgreSQL
   ↓
"Store the data."
```

And the complete flow:

```text
                HTTP
                 ↓
              FastAPI
                 ↓
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

**DI wires these layers together; the Repository isolates persistence; the Service owns business logic; and the Router remains thin.** This combination is one of the core architectural patterns for production FastAPI applications.
