# Router → Service → Repository Architecture in FastAPI

This is a **very important senior-level FastAPI interview topic** because it tests whether you understand **separation of concerns, clean architecture, dependency injection, testing, and maintainability**.

The basic architecture is:

```text
Client
  │
  ▼
FastAPI Router
  │
  ▼
Service Layer
  │
  ▼
Repository Layer
  │
  ▼
Database
```

The simplest rule is:

> **Router handles HTTP, Service handles business logic, Repository handles data access.**

---

# 1. Why do we need these layers?

Imagine putting everything into one FastAPI endpoint:

```python
@router.post("/users")
async def create_user(request: UserRequest):

    # Validate authentication
    # Check permissions
    # Check if user exists
    # Hash password
    # Create SQL query
    # Execute query
    # Commit transaction
    # Send email
    # Return response

    ...
```

This becomes a **fat controller/router**.

As the application grows:

```text
100+ endpoints
      ↓
huge route files
      ↓
business logic duplicated
      ↓
database queries duplicated
      ↓
hard to test
      ↓
hard to maintain
```

Instead, separate responsibilities.

---

# 2. The three layers

## Router

Responsible for:

* HTTP request/response
* URL/path
* query parameters
* request schemas
* response schemas
* authentication/authorization dependencies
* calling the service

## Service

Responsible for:

* business logic
* workflows
* business rules
* orchestration
* transactions/business decisions

## Repository

Responsible for:

* database queries
* persistence
* CRUD operations
* translating database results into application objects

So:

```text
Router
  ↓
"How does the API work?"

Service
  ↓
"How does the business work?"

Repository
  ↓
"How does the data get stored/retrieved?"
```

---

# 3. Simple example

Suppose we have:

```text
POST /users
```

The flow is:

```text
HTTP Request
     │
     ▼
   Router
     │
     ▼
  Service
     │
     ▼
 Repository
     │
     ▼
 PostgreSQL
```

---

# 4. Router layer

Your router should be thin.

```python
from fastapi import APIRouter, Depends

router = APIRouter()


@router.post("/users")
async def create_user(
    request: UserCreate,
    service: UserService = Depends(get_user_service),
):
    return await service.create_user(request)
```

Notice what the router **doesn't** contain:

```text
❌ SQL query
❌ password hashing
❌ transaction logic
❌ duplicate-user business rule
❌ complex workflow
```

It simply:

```text
HTTP
 ↓
Service
 ↓
HTTP response
```

---

# 5. Service layer

The service contains business logic.

```python
class UserService:

    def __init__(
        self,
        repository: UserRepository,
    ):
        self.repository = repository

    async def create_user(
        self,
        request: UserCreate,
    ):

        existing_user = await self.repository.get_by_email(
            request.email
        )

        if existing_user:
            raise UserAlreadyExistsError()

        hashed_password = hash_password(
            request.password
        )

        user = User(
            email=request.email,
            password_hash=hashed_password,
        )

        return await self.repository.create(user)
```

The service is answering:

> "What does it mean to create a user?"

For example:

```text
Check duplicate
     ↓
Hash password
     ↓
Create user
     ↓
Persist user
```

That's business logic.

---

# 6. Repository layer

The repository handles database interaction.

```python
from sqlalchemy import select


class UserRepository:

    def __init__(self, db: AsyncSession):
        self.db = db

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

    async def create(
        self,
        user: User,
    ):

        self.db.add(user)

        await self.db.flush()

        await self.db.refresh(user)

        return user
```

The repository knows:

```text
SQLAlchemy
SQL queries
tables
database session
```

The service doesn't need to know how the SQL query is implemented.

---

# 7. Dependency Injection connects the layers

Now we use FastAPI's `Depends()`.

### Database

```python
async def get_db():
    async with SessionLocal() as db:
        yield db
```

### Repository

```python
def get_user_repository(
    db: AsyncSession = Depends(get_db),
):
    return UserRepository(db)
```

### Service

```python
def get_user_service(
    repository: UserRepository = Depends(
        get_user_repository
    ),
):
    return UserService(repository)
```

### Router

```python
@router.post("/users")
async def create_user(
    request: UserCreate,
    service: UserService = Depends(
        get_user_service
    ),
):
    return await service.create_user(request)
```

Now FastAPI constructs the dependency graph.

```text
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

---

# 8. Full architecture

A production project might look like:

```text
app/
│
├── main.py
│
├── api/
│   ├── dependencies.py
│   └── routes/
│       ├── users.py
│       ├── auth.py
│       └── orders.py
│
├── services/
│   ├── user_service.py
│   ├── auth_service.py
│   └── order_service.py
│
├── repositories/
│   ├── user_repository.py
│   ├── order_repository.py
│   └── payment_repository.py
│
├── models/
│   ├── user.py
│   └── order.py
│
├── schemas/
│   ├── user.py
│   └── order.py
│
└── db/
    └── session.py
```

---

# 9. What should each layer know?

This is a very good interview question.

### Router knows

```text
HTTP
FastAPI
Pydantic
Authentication dependencies
Service
```

### Service knows

```text
Business rules
Repositories
Domain models
Business workflows
```

### Repository knows

```text
Database
SQLAlchemy
Queries
Persistence
```

The dependency direction should generally be:

```text
Router
   ↓
Service
   ↓
Repository
   ↓
Database
```

Not:

```text
Repository
   ↓
FastAPI Router
```

The lower layers shouldn't depend on FastAPI-specific HTTP concepts unnecessarily.

---

# 10. Example: Order creation

This architecture becomes clearer with a realistic workflow.

Suppose:

```text
POST /orders
```

The business requirement is:

> Create an order only if the product exists, sufficient inventory exists, and the user has enough credit.

### Router

```python
@router.post("/orders")
async def create_order(
    request: OrderCreate,
    service: OrderService = Depends(
        get_order_service
    ),
):
    return await service.create_order(request)
```

Very thin.

---

# 11. Service handles the workflow

```python
class OrderService:

    def __init__(
        self,
        order_repo: OrderRepository,
        product_repo: ProductRepository,
    ):
        self.order_repo = order_repo
        self.product_repo = product_repo

    async def create_order(
        self,
        request: OrderCreate,
    ):

        product = await self.product_repo.get_by_id(
            request.product_id
        )

        if not product:
            raise ProductNotFoundError()

        if product.stock < request.quantity:
            raise InsufficientStockError()

        order = Order(
            user_id=request.user_id,
            product_id=product.id,
            quantity=request.quantity,
        )

        return await self.order_repo.create(order)
```

The service decides:

```text
Does product exist?
        ↓
Is stock available?
        ↓
Can order be created?
```

That's business logic.

---

# 12. Repository handles persistence

```python
class OrderRepository:

    def __init__(self, db: AsyncSession):
        self.db = db

    async def create(
        self,
        order: Order,
    ):

        self.db.add(order)

        await self.db.flush()

        await self.db.refresh(order)

        return order
```

The repository doesn't decide:

> "Should this order be allowed?"

That's the service's responsibility.

---

# 13. Why not put business logic in the Repository?

Bad:

```python
class UserRepository:

    async def create_user(self, data):

        if data.email.endswith("@company.com"):
            ...

        if user_is_premium:
            ...

        if subscription_expired:
            ...

        ...
```

Now the repository is doing business logic.

That's problematic because repositories should primarily answer:

> "How do I retrieve/store this data?"

Not:

> "What should the business do?"

---

# 14. Why not put SQL in the Service?

Bad:

```python
class UserService:

    async def create_user(self):

        result = await self.db.execute(
            select(User)
            .where(User.email == email)
        )

        ...
```

Now the service knows too much about persistence.

Better:

```python
existing = await self.repository.get_by_email(
    email
)
```

The service doesn't care whether the repository uses:

```text
PostgreSQL
MySQL
MongoDB
Redis
mock
```

---

# 15. The biggest benefit: testability

Suppose your service is:

```python
class UserService:

    def __init__(self, repository):
        self.repository = repository
```

During testing:

```python
mock_repository = MockUserRepository()

service = UserService(
    repository=mock_repository
)
```

You can test business logic without PostgreSQL.

```text
Test
 │
 ▼
UserService
 │
 ▼
MockRepository
```

Instead of:

```text
Test
 │
 ▼
UserService
 │
 ▼
Real Repository
 │
 ▼
PostgreSQL
```

This makes unit tests:

* faster
* deterministic
* isolated

---

# 16. Router testing

You can also override dependencies.

```python
app.dependency_overrides[
    get_user_service
] = get_mock_user_service
```

Then:

```text
HTTP Test
    ↓
FastAPI Router
    ↓
Mock Service
```

No real database is required.

---

# 17. Where does Pydantic belong?

Usually:

```text
Router
   ↓
Pydantic Request Schema
   ↓
Service
   ↓
Domain/ORM model
   ↓
Repository
```

For example:

```python
class UserCreate(BaseModel):
    email: EmailStr
    password: str
```

The router receives:

```python
request: UserCreate
```

FastAPI/Pydantic validates it before the service is called.

---

# 18. Pydantic model vs SQLAlchemy model

Don't confuse these.

### Pydantic

Used primarily for API input/output:

```python
class UserCreate(BaseModel):
    email: EmailStr
    password: str
```

### SQLAlchemy

Used for persistence:

```python
class User(Base):
    __tablename__ = "users"

    id = mapped_column(Integer, primary_key=True)
    email = mapped_column(String)
```

Flow:

```text
HTTP JSON
   ↓
Pydantic
   ↓
Service
   ↓
SQLAlchemy ORM
   ↓
PostgreSQL
```

---

# 19. Where should transactions live?

This is a **senior-level question**.

Don't blindly put:

```python
await db.commit()
```

inside every repository method.

For example:

```python
async def create_order(...):
    await order_repo.create(...)
    await inventory_repo.decrease(...)
    await payment_repo.create(...)
```

These operations may need to be one transaction:

```text
BEGIN
  │
  ├── Create Order
  ├── Decrease Inventory
  └── Create Payment
  │
  ▼
COMMIT
```

If something fails:

```text
ROLLBACK
```

So transaction ownership usually belongs at the **service/unit-of-work boundary**, depending on your architecture.

---

# 20. More advanced architecture: Unit of Work

For complex applications:

```text
Router
   ↓
Service
   ↓
Unit of Work
   ├── UserRepository
   ├── OrderRepository
   ├── InventoryRepository
   └── PaymentRepository
            │
            ▼
       AsyncSession
```

Example:

```python
class OrderService:

    def __init__(self, uow):
        self.uow = uow

    async def create_order(self, data):

        async with self.uow:

            product = await (
                self.uow.products
                .get_by_id(data.product_id)
            )

            order = await (
                self.uow.orders
                .create(data)
            )

            await self.uow.inventory.decrease(
                product.id,
                data.quantity,
            )

            await self.uow.commit()

            return order
```

This becomes valuable when multiple repositories participate in one business transaction.

---

# 21. AI/RAG Application Example

For an enterprise AI application, the same architecture works extremely well.

Imagine:

```text
POST /chat
```

Architecture:

```text
Router
   ↓
ChatService
   ↓
┌───────────────┬────────────────┐
│               │                │
▼               ▼                ▼
Conversation   Retriever       LLM
Repository       │
│                ▼
PostgreSQL     Qdrant
```

### Router

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

### Service

```python
class ChatService:

    def __init__(
        self,
        conversation_repo,
        retriever,
        llm,
    ):
        self.conversation_repo = conversation_repo
        self.retriever = retriever
        self.llm = llm

    async def chat(self, request):

        history = await (
            self.conversation_repo
            .get_history(request.conversation_id)
        )

        documents = await self.retriever.search(
            request.question
        )

        answer = await self.llm.generate(
            question=request.question,
            history=history,
            documents=documents,
        )

        await self.conversation_repo.save_message(
            request.conversation_id,
            answer,
        )

        return answer
```

Notice something important:

The service orchestrates:

```text
Conversation
     ↓
Retrieval
     ↓
LLM
     ↓
Persistence
```

The router doesn't contain this workflow.

---

# 22. Where does Qdrant belong?

This depends on your architecture.

You could have:

```text
ChatService
   ↓
Retriever interface
   ↓
QdrantRetriever
   ↓
Qdrant
```

You don't necessarily have to call a vector database a traditional "repository."

For example:

```python
class Retriever:

    async def search(
        self,
        query: str,
    ):
        ...
```

Then:

```python
class QdrantRetriever(Retriever):

    async def search(self, query: str):
        ...
```

The service depends on the abstraction:

```text
ChatService
     ↓
Retriever
     ↓
QdrantRetriever
     ↓
Qdrant
```

This makes replacing Qdrant with another vector store easier.

---

# 23. Repository vs Service — easiest way to remember

Imagine an e-commerce application.

### Repository

> "Give me order #123."

```python
order = await repo.get_by_id(123)
```

### Service

> "Can I cancel order #123?"

```python
order = await repo.get_by_id(123)

if order.status != "PENDING":
    raise CannotCancelOrder()

await repo.cancel(order)
```

So:

```text
Repository:
"How do I access the data?"

Service:
"What should I do with the data?"
```

---

# 24. Router vs Service — easiest way to remember

### Router

```text
HTTP concerns
```

Examples:

```text
GET /users/123
POST /orders
HTTP status codes
headers
request body
authentication
```

### Service

```text
Business concerns
```

Examples:

```text
Can user place order?
Can user cancel order?
Should inventory be decreased?
Should payment be processed?
```

---

# 25. Common mistake: fat routers

Bad:

```python
@router.post("/orders")
async def create_order(
    request: OrderCreate,
    db: AsyncSession = Depends(get_db),
):

    product = await db.execute(...)

    if product.stock < request.quantity:
        raise HTTPException(...)

    payment = await stripe.charge(...)

    order = Order(...)

    db.add(order)

    await db.commit()

    return order
```

Everything is mixed together:

```text
HTTP
SQL
Business logic
Payment
Transaction
```

---

# 26. Better

```python
@router.post("/orders")
async def create_order(
    request: OrderCreate,
    service: OrderService = Depends(
        get_order_service
    ),
):

    return await service.create_order(request)
```

Then:

```text
Router
  ↓
OrderService
  ├── ProductRepository
  ├── OrderRepository
  └── PaymentService
       ↓
     Database
```

Much cleaner.

---

# 27. Do you always need all three layers?

**No.**

For a tiny application:

```text
Router
  ↓
Database
```

might be perfectly reasonable.

For a medium/large application:

```text
Router
  ↓
Service
  ↓
Repository
```

becomes much more valuable.

Don't introduce abstractions just for the sake of having more files.

A good senior engineer should be able to say:

> "I introduce the service/repository layers when complexity, reuse, testing requirements, or domain logic justify them."

---

# 28. Interview question: "Can the service directly access the database?"

Technically:

**Yes.**

Architecturally:

**It depends.**

For a simple project:

```text
Router
 ↓
Service
 ↓
AsyncSession
```

may be enough.

For a larger application:

```text
Router
 ↓
Service
 ↓
Repository
 ↓
AsyncSession
```

provides better separation.

The important thing is not blindly following a pattern.

---

# 29. Interview question: "Why not call the repository directly from the router?"

You can, but then the router becomes responsible for business orchestration.

For example:

```python
@router.post("/orders")
async def create_order(...):

    product = await product_repo.get(...)
    inventory = await inventory_repo.get(...)
    payment = await payment_repo.process(...)
    order = await order_repo.create(...)
```

Now your API layer knows too much.

Better:

```python
@router.post("/orders")
async def create_order(
    service: OrderService = Depends(...)
):
    return await service.create_order(...)
```

The router remains thin.

---

# 30. Interview question: "What are the benefits?"

You should mention:

### Separation of concerns

```text
Router → HTTP
Service → Business
Repository → Data
```

### Testability

Mock repositories/services easily.

### Loose coupling

Replace implementations without rewriting routers.

### Reusability

Services can be called from:

```text
REST API
background jobs
CLI
scheduled tasks
```

### Maintainability

Large applications remain manageable.

### Scalability

Teams can work on different layers independently.

---

# 31. Strong senior-level answer

If the interviewer asks:

> **"Explain Router → Service → Repository architecture."**

You can answer:

> "I use the Router-Service-Repository pattern to separate HTTP concerns, business logic, and persistence concerns.
>
> The Router is responsible for the API layer: request validation, response schemas, HTTP status codes, authentication dependencies, and delegating the operation to a service. I try to keep routers thin.
>
> The Service layer contains business logic and orchestrates workflows. For example, when creating an order it might validate inventory, check business rules, invoke payment processing, and coordinate multiple repositories.
>
> The Repository layer abstracts persistence. It owns SQLAlchemy queries and CRUD operations and receives an `AsyncSession` through dependency injection. The service shouldn't need to know how the database query is implemented.
>
> FastAPI's `Depends()` connects these layers. For example, the router depends on a service, the service can depend on repositories, and repositories depend on the database session.
>
> This gives us separation of concerns, loose coupling, easier unit testing, and better maintainability. For complex transactions involving multiple repositories, I'd consider a Unit of Work so the service can control the transaction boundary."

---

# 32. The architecture to memorize

```text
                         HTTP Request
                              │
                              ▼
                    ┌─────────────────┐
                    │     Router      │
                    │                 │
                    │ HTTP concerns   │
                    │ Pydantic        │
                    │ Auth/DI         │
                    └────────┬────────┘
                             │
                             ▼
                    ┌─────────────────┐
                    │    Service      │
                    │                 │
                    │ Business logic  │
                    │ Workflows       │
                    │ Transactions    │
                    └────────┬────────┘
                             │
                   ┌─────────┴─────────┐
                   ▼                   ▼
          ┌────────────────┐   ┌────────────────┐
          │   Repository   │   │   Repository   │
          │                │   │                │
          │ SQLAlchemy     │   │ SQLAlchemy     │
          │ Queries        │   │ Queries        │
          └───────┬────────┘   └───────┬────────┘
                  │                    │
                  └─────────┬──────────┘
                            ▼
                     AsyncSession
                            │
                            ▼
                       PostgreSQL
```

### One sentence to remember

> **Router decides how to expose the operation, Service decides what the business should do, and Repository decides how the data is persisted.**
