In a **FastAPI + SQLAlchemy application**, a repository is the layer responsible for **database access**. It keeps SQLAlchemy queries out of your routers and usually out of your business logic.

A good production architecture is:

```text
HTTP Request
     ↓
FastAPI Router
     ↓
Service Layer       ← business logic
     ↓
Repository Layer    ← database access
     ↓
AsyncSession
     ↓
SQLAlchemy
     ↓
PostgreSQL
```

---

# 1. What is the Repository Pattern?

A repository provides methods such as:

```python
get_by_id()
get_by_email()
list_users()
create()
update()
delete()
```

Instead of doing this directly in your endpoint:

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

    return user
```

you move the database logic into:

```python
class UserRepository:

    async def get_by_id(
        self,
        user_id: int,
    ):
        ...
```

Then your service calls the repository.

---

# 2. Why use repositories?

Without repositories:

```text
Router
 ├── SQL queries
 ├── business logic
 ├── transactions
 ├── validation
 └── database operations
```

This becomes difficult to maintain.

With repositories:

```text
Router
   ↓
Service
   ↓
Repository
   ↓
Database
```

Each layer has a responsibility.

### Router

Handles:

* HTTP
* request/response
* status codes
* dependency injection

### Service

Handles:

* business rules
* workflows
* orchestration
* transaction decisions

### Repository

Handles:

* SQLAlchemy
* SQL queries
* CRUD
* persistence

---

# 3. Project structure

A production-style structure could be:

```text
app/
├── main.py
│
├── api/
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
    ├── session.py
    └── dependencies.py
```

For a larger application:

```text
repositories/
├── base.py
├── user.py
├── tenant.py
├── document.py
└── conversation.py
```

---

# 4. SQLAlchemy model

Let's use a simple `User`.

```python
# models/user.py

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

---

# 5. Create the repository

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

The repository receives the `AsyncSession`.

We don't create a database connection inside the repository.

The session is injected from FastAPI.

---

# 6. Implement `get_by_id`

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

Now the service doesn't need to know SQLAlchemy query syntax.

It can simply do:

```python
user = await repository.get_by_id(user_id)
```

---

# 7. Implement `get_by_email`

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

# 8. Implement `list_users`

```python
async def list_users(
    self,
    offset: int = 0,
    limit: int = 100,
) -> list[User]:

    result = await self.db.execute(
        select(User)
        .offset(offset)
        .limit(limit)
    )

    return list(result.scalars().all())
```

This gives us pagination.

Request:

```text
GET /users?offset=0&limit=20
```

could eventually call:

```python
await repository.list_users(
    offset=0,
    limit=20,
)
```

---

# 9. Implement `create`

There are two common approaches.

### Repository only stages the object

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

rather than:

```python
await self.db.commit()
```

This is important.

The **service can own the transaction boundary**.

For example:

```python
async with db.begin():

    user = await repository.create(user)
```

Then the transaction commits after the entire business operation succeeds.

This is generally preferable when one business operation involves multiple repositories.

---

# 10. Why shouldn't every repository call `commit()`?

This is a very good senior interview question.

Imagine:

```python
user_repository.create()
```

commits immediately.

Then:

```python
profile_repository.create()
```

commits immediately.

Suppose:

```text
Create User       ✓ COMMIT
Create Profile    ✓ COMMIT
Create Membership ✗ ERROR
```

Now you have partially completed business state.

Instead:

```text
BEGIN
  ↓
UserRepository
  ↓
ProfileRepository
  ↓
MembershipRepository
  ↓
Everything succeeds
  ↓
COMMIT
```

or:

```text
BEGIN
  ↓
UserRepository ✓
ProfileRepository ✓
MembershipRepository ✗
  ↓
ROLLBACK
```

Therefore, I usually prefer:

```text
Repository → database operations
Service → transaction/business workflow
```

rather than putting unconditional `commit()` inside every repository method.

---

# 11. Implement update

```python
async def update(
    self,
    user: User,
) -> User:

    await self.db.flush()

    return user
```

If the user object was already loaded:

```python
user.name = "New Name"
```

SQLAlchemy tracks the change.

Then:

```python
await db.flush()
```

generates the appropriate SQL.

The service can later commit.

---

# 12. Implement delete

```python
async def delete(
    self,
    user: User,
) -> None:

    await self.db.delete(user)

    await self.db.flush()
```

Again, the repository doesn't necessarily need to commit.

---

# 13. Complete repository

Putting it together:

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

    async def list_users(
        self,
        offset: int = 0,
        limit: int = 100,
    ) -> list[User]:

        result = await self.db.execute(
            select(User)
            .offset(offset)
            .limit(limit)
        )

        return list(result.scalars().all())

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

---

# 14. Add the service layer

Now business logic goes into the service.

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

        return await self.repository.create(user)
```

Notice the important separation.

The repository knows:

```python
select(User)
```

The service knows:

```python
if existing:
    raise ValueError(...)
```

That's the distinction.

---

# 15. Inject repository into service

Create dependencies:

```python
# api/dependencies.py

from fastapi import Depends
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.dependencies import get_db
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

FastAPI resolves:

```text
get_user_service()
       ↓
get_user_repository()
       ↓
get_db()
       ↓
AsyncSession
```

---

# 16. Router

Now the router becomes very clean:

```python
# api/routes/users.py

from fastapi import (
    APIRouter,
    Depends,
    HTTPException,
)

from app.services.user_service import UserService
from app.api.dependencies import get_user_service


router = APIRouter(
    prefix="/users",
    tags=["Users"],
)


@router.get("/{user_id}")
async def get_user(
    user_id: int,
    service: UserService = Depends(
        get_user_service
    ),
):

    user = await service.repository.get_by_id(
        user_id
    )

    if user is None:
        raise HTTPException(
            status_code=404,
            detail="User not found",
        )

    return user
```

But I'd improve this further.

The router shouldn't know that the service has a repository.

Instead, expose the operation through the service.

---

# 17. Better service

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
    ) -> User | None:

        return await self.repository.get_by_id(
            user_id
        )

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

        return await self.repository.create(user)
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

    user = await service.get_user(user_id)

    if user is None:
        raise HTTPException(
            status_code=404,
            detail="User not found",
        )

    return user
```

Now:

```text
Router
  ↓
Service
  ↓
Repository
  ↓
Database
```

is clean.

---

# 18. Where does transaction management belong?

This is one of the most important architectural decisions.

Suppose you have:

```text
Create order
 ├── Create Order
 ├── Create OrderItems
 ├── Update Inventory
 └── Create PaymentRecord
```

You want:

```text
BEGIN
  ↓
OrderRepository
  ↓
OrderItemRepository
  ↓
InventoryRepository
  ↓
PaymentRepository
  ↓
COMMIT
```

If any operation fails:

```text
ROLLBACK
```

The **service/application layer** is a natural place to coordinate this transaction.

For example:

```python
class OrderService:

    def __init__(
        self,
        db: AsyncSession,
        order_repo: OrderRepository,
        inventory_repo: InventoryRepository,
    ):
        self.db = db
        self.order_repo = order_repo
        self.inventory_repo = inventory_repo

    async def create_order(
        self,
        request,
    ):

        async with self.db.begin():

            order = await self.order_repo.create(
                request
            )

            await self.inventory_repo.reserve(
                request.items
            )

            return order
```

Now the whole workflow is atomic.

---

# 19. Generic Base Repository — should you use one?

You might see:

```python
class BaseRepository:
    async def get_by_id(...)
    async def create(...)
    async def update(...)
    async def delete(...)
```

Then:

```python
class UserRepository(BaseRepository):
    ...
```

This can reduce repetitive CRUD code.

For example:

```python
from typing import Generic, TypeVar

ModelType = TypeVar(
    "ModelType",
    bound=Base,
)


class BaseRepository(Generic[ModelType]):

    def __init__(
        self,
        db: AsyncSession,
        model: type[ModelType],
    ):
        self.db = db
        self.model = model

    async def get_by_id(
        self,
        object_id: int,
    ) -> ModelType | None:

        result = await self.db.execute(
            select(self.model).where(
                self.model.id == object_id
            )
        )

        return result.scalar_one_or_none()
```

Then:

```python
class UserRepository(
    BaseRepository[User]
):

    def __init__(
        self,
        db: AsyncSession,
    ):
        super().__init__(
            db,
            User,
        )
```

But **don't force everything into a generic repository**.

A senior engineer should recognize that generic CRUD abstractions can become awkward for complex queries.

For example:

```python
get_users_with_active_subscription()
```

is domain-specific.

A specific repository method can be much clearer:

```python
async def get_users_with_active_subscription(
    self,
    tenant_id: int,
):
    ...
```

---

# 20. Repository interfaces / Protocol

For larger systems, you can define an abstraction.

```python
from typing import Protocol


class UserRepositoryProtocol(Protocol):

    async def get_by_id(
        self,
        user_id: int,
    ) -> User | None:
        ...

    async def get_by_email(
        self,
        email: str,
    ) -> User | None:
        ...

    async def create(
        self,
        user: User,
    ) -> User:
        ...
```

Then:

```python
class UserRepository:

    async def get_by_id(...):
        ...

    async def get_by_email(...):
        ...

    async def create(...):
        ...
```

implements the expected behavior structurally.

This can make testing and dependency substitution easier.

---

# 21. Testing repositories

Repositories are very easy to test because database access is isolated.

Example:

```python
@pytest.mark.asyncio
async def test_get_user(
    session,
):

    repo = UserRepository(session)

    user = await repo.get_by_id(1)

    assert user is not None
    assert user.id == 1
```

For service tests, you can mock the repository:

```python
@pytest.mark.asyncio
async def test_create_user():

    repository = AsyncMock()

    repository.get_by_email.return_value = None

    repository.create.return_value = User(
        id=1,
        email="test@example.com",
        name="Test",
    )

    service = UserService(repository)

    user = await service.create_user(
        email="test@example.com",
        name="Test",
    )

    assert user.id == 1
```

Now the service test doesn't need PostgreSQL.

That's one of the biggest advantages of separating:

```text
Service
   ↓
Repository
```

---

# 22. Repository vs Service — memorize this

This is a very common interview question.

| Layer      | Responsibility              |
| ---------- | --------------------------- |
| Router     | HTTP/API                    |
| Schema     | Request/response validation |
| Service    | Business logic              |
| Repository | Database access             |
| ORM Model  | Database representation     |
| Database   | Persistence                 |

Example:

### Repository

```python
async def get_by_email(
    self,
    email: str,
):
    result = await self.db.execute(
        select(User).where(
            User.email == email
        )
    )

    return result.scalar_one_or_none()
```

This is database logic.

### Service

```python
existing = await repo.get_by_email(email)

if existing:
    raise UserAlreadyExists()

if not validate_business_rule(...):
    raise InvalidOperation()
```

This is business logic.

---

# 23. What NOT to do

### ❌ SQL directly in router

```python
@router.get("/users")
async def users(db: AsyncSession = Depends(get_db)):

    result = await db.execute(
        select(User)
    )

    ...
```

For a tiny application, this is acceptable.

For a larger application, move it to a repository.

---

### ❌ Business logic in repository

Don't do:

```python
class UserRepository:

    async def create_user(self, user):

        if user.age < 18:
            raise ValueError(
                "User must be 18"
            )

        ...
```

The repository should generally focus on persistence.

The service should own that business rule.

---

### ❌ Commit every repository method

Avoid:

```python
async def create(...):

    db.add(user)

    await db.commit()
```

when multiple repository operations need to form one atomic business transaction.

Prefer:

```python
async def create(...):

    db.add(user)

    await db.flush()
```

and let the service/application transaction boundary decide when to commit.

---

# 24. Production architecture

For the type of **FastAPI + PostgreSQL + Redis + Qdrant + LLM/RAG application** you're preparing for, I'd typically use:

```text
                         FastAPI
                            │
                     ┌──────┴──────┐
                     │   Router    │
                     └──────┬──────┘
                            ↓
                       Service Layer
                            │
            ┌───────────────┼────────────────┐
            ↓               ↓                ↓
      UserRepository   DocumentRepo    ConversationRepo
            │               │                │
            └───────────────┼────────────────┘
                            ↓
                       AsyncSession
                            ↓
                     SQLAlchemy ORM
                            ↓
                    PostgreSQL Pool
                            ↓
                       PostgreSQL
```

Meanwhile:

```text
RAG retrieval
     ↓
Qdrant

Caching
     ↓
Redis

LLM calls
     ↓
LLM Provider
```

So PostgreSQL repositories should handle things like:

```text
UserRepository
TenantRepository
ConversationRepository
MessageRepository
DocumentRepository
AuditLogRepository
```

while Qdrant/Redis/LLM integrations should generally have their **own service/client abstractions**, rather than pretending everything is a SQLAlchemy repository.

---

# 25. Senior interview answer

If they ask:

> **"How do you implement repositories in FastAPI?"**

A strong answer is:

> **"I use the Repository Pattern to isolate persistence logic from business and HTTP logic. Each repository receives an `AsyncSession` through dependency injection and exposes domain-specific data-access methods such as `get_by_id`, `get_by_email`, `list`, and `create`. The service layer uses the repository and owns business rules and, when necessary, the transaction boundary. The router only handles HTTP concerns."**

Then show:

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

    async def create(
        self,
        user: User,
    ) -> User:

        self.db.add(user)

        await self.db.flush()

        return user
```

Then:

```text
Router
   ↓
Service
   ↓
Repository
   ↓
AsyncSession
   ↓
PostgreSQL
```

### The key principle

> **Repository = "How do I store/retrieve data?"**
> **Service = "What should the application do?"**
> **Router = "How does the client communicate with the application?"**

That's the clean separation interviewers are usually looking for.
