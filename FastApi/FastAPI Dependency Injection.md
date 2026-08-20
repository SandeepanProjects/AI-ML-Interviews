# FastAPI Dependency Injection with `Depends()`

This is one of the **most important FastAPI interview topics**, especially for senior/backend and AI-engineering roles.

The simplest definition is:

> **Dependency Injection (DI) means that a function receives the objects/services it needs from outside instead of creating those objects itself. `Depends()` is FastAPI's mechanism for declaring and resolving those dependencies.**

---

# 1. Why do we need Dependency Injection?

Imagine you have an API endpoint:

```python
@app.get("/users")
async def get_users():
    db = create_database_connection()

    users = await db.get_users()

    return users
```

The endpoint is doing too many things:

```text
Endpoint
 ├── create DB connection
 ├── query DB
 ├── business logic
 └── return response
```

This creates problems:

* difficult to test
* tightly coupled code
* difficult to replace implementations
* database lifecycle becomes harder to manage
* authentication logic gets duplicated
* services become difficult to reuse

Instead, we can inject the database dependency:

```python
@app.get("/users")
async def get_users(
    db = Depends(get_db)
):
    users = await db.get_users()

    return users
```

Now:

```text
FastAPI
   │
   ├── resolve get_db()
   │
   ▼
db
   │
   ▼
get_users(db)
```

The endpoint doesn't need to know **how the database was created**.

---

# 2. What is `Depends()`?

`Depends()` tells FastAPI:

> "Before calling this endpoint, resolve this dependency and give me its result."

Example:

```python
from fastapi import FastAPI, Depends

app = FastAPI()


def get_current_user():
    return {
        "id": 123,
        "name": "Sandeep"
    }


@app.get("/profile")
async def profile(
    user = Depends(get_current_user)
):
    return user
```

When a request arrives:

```text
GET /profile
```

FastAPI sees:

```python
Depends(get_current_user)
```

and effectively does:

```text
1. Call get_current_user()
2. Get its result
3. Pass result to profile()
```

Conceptually:

```python
user = get_current_user()

profile(user)
```

FastAPI handles this dependency resolution for you.

---

# 3. Basic Dependency Example

Let's start with a simple example.

```python
from fastapi import FastAPI, Depends

app = FastAPI()


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

Request:

```text
GET /config
```

Flow:

```text
Request
   ↓
FastAPI
   ↓
Depends(get_config)
   ↓
get_config()
   ↓
settings
   ↓
config(settings)
   ↓
Response
```

---

# 4. Why is this called Dependency Injection?

Because:

```python
async def config(settings):
```

needs:

```text
settings
```

Instead of creating it itself:

```python
settings = get_config()
```

we **inject** it:

```python
settings = Depends(get_config)
```

So:

```text
Without DI:

Function
   ↓
creates dependency
   ↓
uses dependency


With DI:

Dependency
   ↓
injected into
   ↓
Function
```

---

# 5. Real-world example: Database Dependency

This is probably the most common real-world FastAPI use case.

Suppose we have:

```python
async def get_db():
    db = create_db_session()

    try:
        yield db
    finally:
        await db.close()
```

Then:

```python
@app.get("/users")
async def get_users(
    db = Depends(get_db)
):
    return await db.get_users()
```

FastAPI manages the dependency lifecycle.

Conceptually:

```text
Request
   ↓
get_db()
   ↓
Create DB session
   ↓
Inject db
   ↓
Endpoint executes
   ↓
Endpoint completes
   ↓
Close DB session
```

This is extremely useful because you don't want every endpoint doing:

```python
db = create_db_session()

try:
    ...
finally:
    db.close()
```

---

# 6. Why use `yield` in dependencies?

A dependency can use `yield` when you need setup and cleanup.

Example:

```python
async def get_db():

    db = create_db_session()

    try:
        yield db

    finally:
        await db.close()
```

Think of it as:

```text
Before yield
     ↓
Setup
     ↓
yield dependency
     ↓
Endpoint executes
     ↓
finally
     ↓
Cleanup
```

This pattern is very useful for:

* database sessions
* temporary resources
* transactions
* connections
* cleanup operations

---

# 7. Dependency Injection with SQLAlchemy

For a production application, you might have:

```python
from sqlalchemy.ext.asyncio import AsyncSession


async def get_db() -> AsyncSession:

    async with SessionLocal() as session:
        yield session
```

Then:

```python
@app.get("/users")
async def get_users(
    db: AsyncSession = Depends(get_db)
):

    result = await db.execute(
        select(User)
    )

    return result.scalars().all()
```

Architecture:

```text
FastAPI
   │
   ▼
Depends(get_db)
   │
   ▼
AsyncSession
   │
   ▼
Endpoint
   │
   ▼
PostgreSQL
```

---

# 8. Dependency Injection for Authentication

This is another major use case.

Suppose you have:

```python
def get_current_user():
    # decode JWT
    # validate token
    # retrieve user
    return user
```

Then:

```python
@app.get("/profile")
async def profile(
    user = Depends(get_current_user)
):
    return user
```

Now every protected endpoint can reuse it:

```python
@app.get("/profile")
async def profile(
    user = Depends(get_current_user)
):
    ...


@app.get("/orders")
async def orders(
    user = Depends(get_current_user)
):
    ...


@app.get("/payments")
async def payments(
    user = Depends(get_current_user)
):
    ...
```

Instead of repeating:

```text
Extract JWT
 ↓
Validate JWT
 ↓
Decode JWT
 ↓
Find user
 ↓
Check user
```

in every endpoint.

---

# 9. Dependency Chain

This is where FastAPI DI becomes powerful.

A dependency can itself have dependencies.

For example:

```python
def get_db():
    ...


def get_current_user(
    db = Depends(get_db)
):
    ...


def get_admin_user(
    user = Depends(get_current_user)
):
    ...


@app.get("/admin")
async def admin(
    user = Depends(get_admin_user)
):
    ...
```

The dependency graph is:

```text
                    /admin
                       │
                       ▼
                get_admin_user
                       │
                       ▼
                get_current_user
                       │
                       ▼
                    get_db
                       │
                       ▼
                  PostgreSQL
```

FastAPI resolves the dependency tree.

---

# 10. This is extremely useful for Authentication

Imagine:

```text
Request
   │
   ▼
get_admin_user()
   │
   ▼
get_current_user()
   │
   ▼
get_db()
   │
   ▼
PostgreSQL
```

Each layer has one responsibility.

For example:

### `get_db()`

Responsible for:

```text
Database session
```

### `get_current_user()`

Responsible for:

```text
JWT
 ↓
User
```

### `get_admin_user()`

Responsible for:

```text
User
 ↓
Role check
 ↓
Admin
```

---

# 11. Dependency Injection with Service Layer

This becomes important for senior architecture.

Suppose we have:

```python
class UserRepository:

    def __init__(self, db):
        self.db = db
```

Then:

```python
class UserService:

    def __init__(self, repository):
        self.repository = repository
```

Now create dependencies:

```python
def get_user_repository(
    db = Depends(get_db)
):
    return UserRepository(db)


def get_user_service(
    repository = Depends(get_user_repository)
):
    return UserService(repository)
```

Then:

```python
@app.get("/users/{user_id}")
async def get_user(
    user_id: int,
    service = Depends(get_user_service)
):

    return await service.get_user(user_id)
```

Architecture:

```text
FastAPI
   │
   ▼
get_user_service()
   │
   ▼
get_user_repository()
   │
   ▼
get_db()
   │
   ▼
UserService
   │
   ▼
UserRepository
   │
   ▼
PostgreSQL
```

This is a very clean architecture.

---

# 12. Why is this better?

Without DI:

```python
@app.get("/users")
async def get_users():

    db = create_db()
    repo = UserRepository(db)
    service = UserService(repo)

    return await service.get_users()
```

The endpoint knows everything.

With DI:

```python
@app.get("/users")
async def get_users(
    service = Depends(get_user_service)
):
    return await service.get_users()
```

The endpoint only knows:

```text
"I need UserService."
```

That's **loose coupling**.

---

# 13. Dependency Injection and Testing

This is one of the biggest advantages.

Suppose your real dependency is:

```python
def get_payment_service():
    return StripePaymentService()
```

Your endpoint:

```python
@app.post("/payment")
async def payment(
    service = Depends(get_payment_service)
):
    return await service.charge()
```

During testing, you don't want to call the real payment provider.

You can replace the dependency with a fake/mock implementation.

FastAPI provides:

```python
app.dependency_overrides
```

Example:

```python
def mock_payment_service():
    return MockPaymentService()


app.dependency_overrides[
    get_payment_service
] = mock_payment_service
```

Now:

```text
Production:

Endpoint
   ↓
Real Payment Service
   ↓
Stripe


Test:

Endpoint
   ↓
Mock Payment Service
   ↓
Fake result
```

This makes testing much easier.

---

# 14. Dependency Injection vs Manual Instantiation

### Without DI

```python
def endpoint():

    service = UserService(
        UserRepository(
            Database()
        )
    )

    return service.get_user()
```

The endpoint is tightly coupled.

### With DI

```python
def endpoint(
    service = Depends(get_user_service)
):
    return service.get_user()
```

Now the endpoint doesn't care how the service is constructed.

---

# 15. Dependency Injection is NOT just for databases

You can inject almost anything.

Examples:

### Authentication

```python
user = Depends(get_current_user)
```

### Database

```python
db = Depends(get_db)
```

### Configuration

```python
settings = Depends(get_settings)
```

### Service

```python
service = Depends(get_user_service)
```

### Repository

```python
repo = Depends(get_user_repository)
```

### Tenant

```python
tenant = Depends(get_current_tenant)
```

### Authorization

```python
user = Depends(require_admin)
```

---

# 16. AI/RAG Example

This is particularly relevant for your AI Engineer interviews.

Imagine:

```text
POST /chat
```

Your architecture is:

```text
Client
  ↓
FastAPI
  ↓
Authentication
  ↓
Tenant
  ↓
Chat Service
  ↓
Retriever
  ↓
Qdrant
  ↓
LLM
```

You can implement dependencies like:

```python
def get_current_user():
    ...


def get_current_tenant(
    user = Depends(get_current_user)
):
    ...


def get_retriever():
    ...


def get_llm():
    ...


def get_chat_service(
    retriever = Depends(get_retriever),
    llm = Depends(get_llm)
):
    return ChatService(
        retriever=retriever,
        llm=llm
    )
```

Then:

```python
@app.post("/chat")
async def chat(
    request: ChatRequest,
    tenant = Depends(get_current_tenant),
    service = Depends(get_chat_service)
):

    return await service.answer(
        tenant_id=tenant.id,
        question=request.question
    )
```

Now the dependency graph is:

```text
                    /chat
                       │
             ┌─────────┴──────────┐
             │                    │
             ▼                    ▼
     get_current_tenant     get_chat_service
             │                    │
             ▼              ┌─────┴─────┐
     get_current_user       │           │
             │              ▼           ▼
             ▼          Retriever      LLM
           JWT             │             │
                           ▼             ▼
                         Qdrant        Model
```

That's a very good production architecture.

---

# 17. Multi-Tenant AI Application

For an enterprise AI application, you can use dependencies to establish tenant context.

```python
async def get_current_tenant(
    user = Depends(get_current_user)
):
    tenant = await tenant_service.get_tenant(
        user.tenant_id
    )

    return tenant
```

Then:

```python
@app.post("/chat")
async def chat(
    request: ChatRequest,
    tenant = Depends(get_current_tenant)
):
    ...
```

Now your service can enforce:

```text
tenant_id
    ↓
PostgreSQL filtering
    ↓
Qdrant filtering
    ↓
Redis namespace
```

This is extremely useful for SaaS AI platforms.

---

# 18. Dependency Caching

FastAPI can cache dependency results within a request.

Suppose:

```python
def get_current_user():
    ...
```

is required by multiple dependencies.

FastAPI can avoid unnecessarily calling the same dependency repeatedly within the same request.

Conceptually:

```text
Request
   │
   ├── Dependency A
   │       ↓
   │   get_current_user()
   │
   └── Dependency B
           ↓
       get_current_user()
```

FastAPI can reuse the dependency result for that request.

You can control this with:

```python
Depends(
    get_current_user,
    use_cache=False
)
```

This is an advanced detail worth knowing.

---

# 19. `Depends()` Doesn't Immediately Execute the Function

This is a subtle point.

When you write:

```python
user = Depends(get_current_user)
```

you're not manually calling:

```python
get_current_user()
```

Instead, you're declaring:

> "FastAPI, this parameter depends on `get_current_user`."

FastAPI resolves it when processing the request.

Conceptually:

```text
Depends()
     ↓
Dependency declaration
     ↓
FastAPI dependency resolver
     ↓
Execute dependency
     ↓
Inject result
```

---

# 20. Dependency Injection vs Middleware

Interviewers sometimes ask:

> "Why not just use middleware instead of dependencies?"

They serve different purposes.

### Middleware

Runs around requests globally.

```text
Request
  ↓
Middleware
  ↓
Endpoint
  ↓
Middleware
  ↓
Response
```

Good for:

* logging
* metrics
* request IDs
* CORS
* global processing

### Dependency

Used when a particular endpoint/router needs a reusable object or validation.

```text
Endpoint
   ↓
Dependency
   ↓
Service
```

Good for:

* database sessions
* authentication
* authorization
* services
* repositories
* tenant context

---

# 21. Dependency Injection vs Service Locator

Another architectural concept.

Good DI:

```python
async def endpoint(
    service = Depends(get_service)
):
    ...
```

The dependency is explicit.

A service locator approach might hide dependencies:

```python
service = global_container.get("user_service")
```

That makes dependencies less obvious and can make testing harder.

FastAPI's DI approach encourages dependencies to be visible in the function signature.

---

# 22. Production Architecture

A senior-level FastAPI project might look like:

```text
app/
│
├── main.py
│
├── api/
│   ├── routes/
│   │   ├── auth.py
│   │   ├── users.py
│   │   └── chat.py
│   │
│   └── dependencies.py
│
├── services/
│   ├── user_service.py
│   ├── auth_service.py
│   └── rag_service.py
│
├── repositories/
│   ├── user_repository.py
│   └── document_repository.py
│
├── db/
│   └── session.py
│
├── models/
│
├── schemas/
│
└── core/
    ├── security.py
    └── config.py
```

Your dependencies might connect these layers:

```text
Router
  │
  │ Depends
  ▼
Service
  │
  │ Depends
  ▼
Repository
  │
  │ Depends
  ▼
Database
```

---

# 23. A Complete Example

Let's put everything together.

### Database dependency

```python
async def get_db():
    async with SessionLocal() as db:
        yield db
```

### Repository dependency

```python
def get_user_repository(
    db: AsyncSession = Depends(get_db)
):
    return UserRepository(db)
```

### Service dependency

```python
def get_user_service(
    repository: UserRepository = Depends(
        get_user_repository
    )
):
    return UserService(repository)
```

### Authentication dependency

```python
async def get_current_user(
    token: str = Depends(oauth2_scheme)
):
    return decode_token(token)
```

### Endpoint

```python
@app.get("/users/{user_id}")
async def get_user(
    user_id: int,
    current_user = Depends(get_current_user),
    service: UserService = Depends(
        get_user_service
    )
):

    return await service.get_user(
        user_id
    )
```

The dependency graph:

```text
                    HTTP Request
                         │
                         ▼
                     /users/{id}
                         │
              ┌──────────┴──────────┐
              ▼                     ▼
       get_current_user      get_user_service
              │                     │
              ▼                     ▼
             JWT            get_user_repository
                                    │
                                    ▼
                                 get_db
                                    │
                                    ▼
                               PostgreSQL
```

This is **dependency injection in action**.

---

# 24. Why senior engineers care about DI

Dependency Injection gives you:

### 1. Loose coupling

```text
Endpoint
   ↓
Interface/service
```

instead of:

```text
Endpoint
   ↓
specific implementation
```

### 2. Testability

Replace:

```text
Real DB
```

with:

```text
Mock DB
```

### 3. Reusability

One dependency can be used across many endpoints.

### 4. Lifecycle management

Database sessions and resources can be created/cleaned up consistently.

### 5. Separation of concerns

Authentication doesn't have to live inside every endpoint.

### 6. Maintainability

Changing implementation doesn't require changing every endpoint.

---

# 25. Interview Answer

If the interviewer asks:

> **"Explain FastAPI Dependency Injection with `Depends()`."**

A strong senior-level answer is:

> "`Depends()` is FastAPI's dependency injection mechanism. Instead of an endpoint creating the objects or services it needs, it declares those dependencies in its function signature, and FastAPI resolves and injects them when processing the request.
>
> For example, I can define `get_db()` to create an async SQLAlchemy session and use `db: AsyncSession = Depends(get_db)` in an endpoint. FastAPI creates the dependency, injects it into the endpoint, and handles the dependency lifecycle, including cleanup when using `yield`.
>
> Dependencies can also depend on other dependencies. For example, an authenticated endpoint might depend on `get_current_user()`, which depends on a JWT token, while a service might depend on a repository, which depends on a database session. FastAPI resolves this dependency graph.
>
> I primarily use DI for database sessions, authentication, authorization, repositories, services, configuration and tenant context. It improves loose coupling, testability and separation of concerns. In tests, I can use `app.dependency_overrides` to replace production dependencies with mocks or test implementations."

---

## The mental model to remember

```text
              Depends()
                  │
                  ▼
        "FastAPI, give me this"
                  │
                  ▼
          Dependency Resolver
                  │
        ┌─────────┴──────────┐
        ▼                    ▼
   Dependency A        Dependency B
        │                    │
        └─────────┬──────────┘
                  ▼
               Endpoint
```

And for a production AI application:

```text
                    FastAPI
                       │
                       ▼
                    Router
                       │
          ┌────────────┼────────────┐
          ▼            ▼            ▼
       Auth DI      Tenant DI    Service DI
          │            │            │
          ▼            ▼       ┌────┴────┐
         JWT        tenant   Retriever   LLM
                                  │         │
                                Qdrant    Model
```

**The key phrase to remember for interviews:**

> **"`Depends()` declares what an endpoint needs; FastAPI's dependency resolver constructs and injects those dependencies, allowing us to keep authentication, database sessions, services, repositories and other infrastructure concerns outside the endpoint itself."**
