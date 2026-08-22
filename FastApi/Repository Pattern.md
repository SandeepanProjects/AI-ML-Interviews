# Repository Pattern in FastAPI — and Why Not Put SQLAlchemy Queries in Endpoints?

This is an important **backend architecture interview question**, especially for a Senior AI Engineer working with **FastAPI + PostgreSQL + SQLAlchemy**.

The short answer is:

> **A FastAPI endpoint should handle HTTP concerns, not database concerns. The Repository Pattern isolates database access from the API layer, while the Service Layer contains business logic.**

The architecture is:

```text
Client
  ↓
FastAPI Router / Endpoint
  ↓
Service Layer
  ↓
Repository Layer
  ↓
SQLAlchemy AsyncSession
  ↓
PostgreSQL
```

Let's build it properly.

---

# 1. What is the Repository Pattern?

The Repository Pattern creates a dedicated class responsible for **reading and writing data**.

For example:

```python
class UserRepository:

    async def get_by_id(self, user_id: int):
        ...

    async def get_by_email(self, email: str):
        ...

    async def create(self, user: User):
        ...

    async def delete(self, user: User):
        ...
```

The rest of the application doesn't need to know how the data is stored.

The service can simply say:

```python
user = await user_repository.get_by_id(user_id)
```

It doesn't need to know whether the repository uses:

```text
SQLAlchemy
PostgreSQL
raw SQL
another database
```

---

# 2. First, let's see the bad approach

Suppose you don't use a repository.

You might write:

```python
from fastapi import APIRouter, Depends
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

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

    return user
```

This works.

For a small application, it's perfectly acceptable.

But as the application grows, this becomes problematic.

---

# 3. Why is this a problem?

Imagine your endpoint becomes:

```python
@router.post("/users")
async def create_user(...):
    ...
```

Then:

```python
@router.put("/users/{user_id}")
async def update_user(...):
    ...
```

Then:

```python
@router.delete("/users/{user_id}")
async def delete_user(...):
    ...
```

And each endpoint contains:

```text
SQLAlchemy queries
validation
business logic
transactions
error handling
authorization
logging
```

Your router starts becoming huge.

For example:

```text
users.py

GET /users
    ↓
SQLAlchemy

GET /users/{id}
    ↓
SQLAlchemy

POST /users
    ↓
SQLAlchemy

PUT /users/{id}
    ↓
SQLAlchemy

DELETE /users/{id}
    ↓
SQLAlchemy
```

Now your API layer is tightly coupled to your database implementation.

---

# 4. Better architecture

Instead:

```text
                  FastAPI
                     │
                     ↓
                API Router
                     │
                     ↓
               UserService
                     │
                     ↓
             UserRepository
                     │
                     ↓
               AsyncSession
                     │
                     ↓
                PostgreSQL
```

Each layer has one primary responsibility.

---

# 5. Project structure

A good starting structure:

```text
app/
│
├── main.py
│
├── api/
│   ├── dependencies.py
│   └── routes/
│       └── users.py
│
├── models/
│   └── user.py
│
├── schemas/
│   └── user.py
│
├── repositories/
│   └── user_repository.py
│
├── services/
│   └── user_service.py
│
└── db/
    └── session.py
```

---

# 6. SQLAlchemy model

Let's create a User model.

```python
# models/user.py

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
        index=True,
        nullable=False,
    )

    name: Mapped[str] = mapped_column(
        String(100),
        nullable=False,
    )
```

---

# 7. Pydantic request schema

Don't use SQLAlchemy models directly as your API input models.

Use Pydantic.

```python
# schemas/user.py

from pydantic import BaseModel, EmailStr


class UserCreate(BaseModel):

    email: EmailStr

    name: str
```

Response model:

```python
class UserResponse(BaseModel):

    id: int
    email: EmailStr
    name: str

    model_config = {
        "from_attributes": True
    }
```

So:

```text
HTTP
 ↓
Pydantic
 ↓
Service
 ↓
SQLAlchemy
```

---

# 8. Database session

For production FastAPI with PostgreSQL, use `AsyncSession`.

```python
# db/session.py

from sqlalchemy.ext.asyncio import (
    create_async_engine,
    async_sessionmaker,
    AsyncSession,
)


DATABASE_URL = (
    "postgresql+asyncpg://"
    "postgres:password@localhost/mydb"
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

Dependency:

```python
from collections.abc import AsyncGenerator


async def get_db()
    -> AsyncGenerator[
        AsyncSession,
        None
    ]:

    async with AsyncSessionLocal() as session:
        yield session
```

---

# 9. Now create the Repository

This is the important part.

```python
# repositories/user_repository.py

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.user import User


class UserRepository:

    def __init__(
        self,
        db: AsyncSession,
    ):
        self.db = db
```

The repository receives the session.

It doesn't create a new database connection.

---

# 10. `get_by_id()`

```python
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

The SQLAlchemy query is now isolated here.

The endpoint doesn't know anything about:

```python
select(User)
```

or:

```python
scalar_one_or_none()
```

---

# 11. `get_by_email()`

```python
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
```

---

# 12. `create()`

```python
async def create(
    self,
    user: User,
) -> User:

    self.db.add(user)

    await self.db.flush()

    return user
```

Notice:

```python
await self.db.flush()
```

instead of immediately doing:

```python
await self.db.commit()
```

This allows the service/application layer to control the transaction.

---

# 13. Complete Repository

```python
# repositories/user_repository.py

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

        await self.db.flush()

        return user

    async def delete(
        self,
        user: User,
    ) -> None:

        await self.db.delete(user)

        await self.db.flush()
```

Now **all persistence logic is inside the repository**.

---

# 14. Create the Service Layer

The Service Layer handles **business logic**.

```python
# services/user_service.py

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

    async def get_user(
        self,
        user_id: int,
    ) -> User | None:

        return await self.repository.get_by_id(
            user_id
        )
```

Now suppose we create a user.

```python
async def create_user(
    self,
    email: str,
    name: str,
) -> User:

    existing = (
        await self.repository.get_by_email(
            email
        )
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

Notice the difference.

The repository handles:

```text
How to query the database
```

The service handles:

```text
What the application should do
```

---

# 15. Dependency Injection

Now connect everything through FastAPI.

```python
# api/dependencies.py

from fastapi import Depends
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.session import get_db
from app.repositories.user_repository import (
    UserRepository,
)
from app.services.user_service import UserService


def get_user_repository(
    db: AsyncSession = Depends(get_db),
) -> UserRepository:

    return UserRepository(db)


def get_user_service(
    repository: UserRepository = Depends(
        get_user_repository
    ),
) -> UserService:

    return UserService(repository)
```

FastAPI resolves this dependency graph:

```text
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

---

# 16. FastAPI endpoint

Now look how clean the endpoint becomes.

```python
# api/routes/users.py

from fastapi import (
    APIRouter,
    Depends,
    HTTPException,
)

from app.api.dependencies import (
    get_user_service,
)
from app.schemas.user import UserResponse
from app.services.user_service import (
    UserService,
)


router = APIRouter(
    prefix="/users",
    tags=["Users"],
)


@router.get(
    "/{user_id}",
    response_model=UserResponse,
)
async def get_user(
    user_id: int,
    service: UserService = Depends(
        get_user_service
    ),
):

    user = await service.get_user(
        user_id
    )

    if user is None:
        raise HTTPException(
            status_code=404,
            detail="User not found",
        )

    return user
```

Notice what is **not** here:

```text
❌ select(User)

❌ SQLAlchemy query

❌ AsyncSession.execute()

❌ scalar_one_or_none()

❌ database connection

❌ SQL
```

The endpoint only handles HTTP concerns.

---

# 17. POST endpoint

```python
@router.post(
    "",
    response_model=UserResponse,
)
async def create_user(
    request: UserCreate,
    service: UserService = Depends(
        get_user_service
    ),
):

    try:

        user = await service.create_user(
            email=request.email,
            name=request.name,
        )

        return user

    except ValueError as exc:

        raise HTTPException(
            status_code=409,
            detail=str(exc),
        )
```

The flow is:

```text
POST /users
     ↓
Pydantic validation
     ↓
Router
     ↓
UserService
     ↓
UserRepository
     ↓
SQLAlchemy
     ↓
PostgreSQL
```

---

# 18. Why is this better?

## Reason 1 — Separation of concerns

Without Repository:

```text
Endpoint
 ├── HTTP
 ├── validation
 ├── business logic
 ├── SQLAlchemy
 ├── transactions
 └── database access
```

With Repository:

```text
Router
 └── HTTP

Service
 └── Business logic

Repository
 └── Database access
```

Much easier to reason about.

---

# 19. Reason 2 — Reusability

Suppose you need to get a user from:

```text
GET /users/{id}
```

and:

```text
GET /orders/{id}/customer
```

Both may need:

```python
get_by_id(user_id)
```

Without a repository, you may duplicate:

```python
select(User).where(...)
```

in multiple endpoints.

With repository:

```python
user = await user_repository.get_by_id(
    user_id
)
```

Reuse it.

---

# 20. Reason 3 — Testing

This is one of the biggest benefits.

Imagine the service:

```python
class UserService:

    async def create_user(...):
        ...
```

You can mock the repository:

```python
repository = AsyncMock()

repository.get_by_email.return_value = None

repository.create.return_value = user
```

Then:

```python
service = UserService(repository)
```

You can test business logic **without PostgreSQL**.

That's much harder when your endpoint directly contains:

```python
await db.execute(
    select(User)
)
```

---

# 21. Reason 4 — Database changes

Suppose today you use:

```text
PostgreSQL
```

and your application later needs:

```text
another persistence mechanism
```

If database access is spread across 50 endpoints, changing it becomes painful.

If persistence is concentrated in repositories:

```text
UserService
     ↓
UserRepository
     ↓
Database implementation
```

the impact is much more contained.

You don't necessarily change databases often, but **loose coupling is still valuable**.

---

# 22. Reason 5 — Complex queries

Imagine this query:

```python
select(User)
.where(
    User.is_active == True
)
.options(
    selectinload(User.orders)
)
.join(Order)
.where(
    Order.created_at >= start_date
)
```

If this is inside an endpoint:

```text
users.py
```

your router becomes tightly coupled to persistence details.

Instead:

```python
users = await repository.get_active_users_with_recent_orders(
    start_date
)
```

The repository contains the database complexity.

---

# 23. Repository vs Service

This distinction is **extremely important in interviews**.

### Repository

Answers:

> **"How do I get/store this data?"**

Example:

```python
async def get_by_email(
    self,
    email: str,
):
    ...
```

### Service

Answers:

> **"What should the application do?"**

Example:

```python
existing = await repo.get_by_email(email)

if existing:
    raise UserAlreadyExists()

user = User(...)

return await repo.create(user)
```

---

# 24. Don't put business logic into repositories

Avoid:

```python
class UserRepository:

    async def create_user(self, user):

        if user.age < 18:
            raise ValueError(
                "User must be 18"
            )

        ...
```

Why?

Because:

```text
age < 18
```

is a **business rule**, not a database operation.

Better:

```python
class UserService:

    async def create_user(self, user_data):

        if user_data.age < 18:
            raise UserMustBeAdult()

        return await self.repository.create(...)
```

Repository:

```python
class UserRepository:

    async def create(self, user):

        self.db.add(user)

        await self.db.flush()

        return user
```

---

# 25. What about transactions?

This is where senior-level architecture comes in.

Suppose creating a user involves:

```text
User
 +
Profile
 +
Subscription
```

You want:

```text
BEGIN
   ↓
Create User
   ↓
Create Profile
   ↓
Create Subscription
   ↓
COMMIT
```

If subscription creation fails:

```text
ROLLBACK
```

The service can coordinate this workflow.

For example:

```python
async def register_user(...):

    async with self.db.begin():

        user = await self.user_repo.create(
            user
        )

        await self.profile_repo.create(
            profile
        )

        await self.subscription_repo.create(
            subscription
        )
```

This is why I generally avoid unconditional:

```python
await db.commit()
```

inside every repository method.

The repository performs persistence operations.

The service/application layer can control the transaction boundary.

---

# 26. Don't over-engineer the Repository Pattern

A senior engineer should also know when **not** to use it.

For a tiny application:

```python
@app.get("/health")
async def health(db: AsyncSession = Depends(get_db)):
    ...
```

You don't need:

```text
HealthRepository
HealthService
HealthManager
HealthFactory
HealthProvider
```

That is unnecessary abstraction.

For a simple CRUD application, direct SQLAlchemy access may be perfectly reasonable.

Repository patterns become valuable when you have:

```text
Multiple endpoints
Complex queries
Business logic
Multiple data sources
Testing requirements
Large codebase
Multiple developers
Long-term maintenance
```

---

# 27. Repository Pattern in an AI/RAG application

For the kind of production AI system you're likely to discuss in interviews, I would structure the persistence side roughly like:

```text
FastAPI
   │
   ├── Auth Router
   ├── User Router
   ├── Document Router
   └── Chat Router
          │
          ↓
       Services
          │
    ┌─────┼─────────────┐
    ↓     ↓             ↓
 User   Document    Conversation
 Repo     Repo          Repo
    │      │             │
    └──────┼─────────────┘
           ↓
       PostgreSQL
```

And separately:

```text
RAG Service
    │
    ├── Embedding Service
    ├── Retriever
    ├── Reranker
    └── Qdrant Client
```

and:

```text
LLM Service
    ↓
OpenAI / Azure OpenAI / Anthropic / vLLM
```

You generally don't want to force Qdrant or an LLM API into a "SQL repository" abstraction. Those are different external-system clients/services.

---

# 28. A complete mental model

When designing a FastAPI application, think:

```text
                    HTTP
                     │
                     ↓
              ┌─────────────┐
              │   Router    │
              └──────┬──────┘
                     │
                     ↓
              ┌─────────────┐
              │   Service   │
              │             │
              │  Business   │
              │   Logic     │
              └──────┬──────┘
                     │
                     ↓
              ┌─────────────┐
              │ Repository  │
              │             │
              │ Data Access │
              └──────┬──────┘
                     │
                     ↓
              ┌─────────────┐
              │ SQLAlchemy  │
              │ AsyncSession│
              └──────┬──────┘
                     │
                     ↓
                PostgreSQL
```

---

# 29. The interview answer you should give

If they ask:

### **"Why don't you directly write SQLAlchemy queries inside FastAPI endpoints?"**

I'd answer:

> **"I can do that for a small application, but in a production application I prefer separating HTTP, business, and persistence concerns. If I put SQLAlchemy queries directly in endpoints, the router becomes tightly coupled to the database and becomes difficult to test and maintain. I use a repository to encapsulate database operations and a service layer for business logic. FastAPI's dependency injection provides the `AsyncSession` to the repository. This gives me reusable data-access methods, cleaner endpoints, easier unit testing, and centralized transaction handling."**

Then show:

```python
# Router

@router.get("/users/{user_id}")
async def get_user(
    user_id: int,
    service: UserService = Depends(
        get_user_service
    ),
):
    user = await service.get_user(user_id)

    if not user:
        raise HTTPException(404)

    return user
```

```python
# Service

async def get_user(
    self,
    user_id: int,
):
    return await self.repository.get_by_id(
        user_id
    )
```

```python
# Repository

async def get_by_id(
    self,
    user_id: int,
):
    result = await self.db.execute(
        select(User).where(
            User.id == user_id
        )
    )

    return result.scalar_one_or_none()
```

### The one-line answer to memorize

> **"The router handles HTTP, the service handles business logic, and the repository handles persistence."**

That's the clean architectural principle interviewers are looking for.
