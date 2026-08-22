Absolutely. These four questions are really about **how FastAPI Dependency Injection works in a production application**. For a senior interview, you should be able to explain not only the syntax but also **why the pattern is used, how dependencies compose, and how testing works**.

---

# 1. How do you inject the current authenticated user?

The standard pattern is:

```text
HTTP Request
     ↓
Authorization: Bearer <JWT>
     ↓
Authentication dependency
     ↓
Decode + validate JWT
     ↓
Load current user
     ↓
Endpoint receives current_user
```

FastAPI uses `Depends()` to inject the authenticated user into the endpoint.

---

## Step 1: Create an authentication dependency

For example, using OAuth2 Bearer tokens:

```python
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer

oauth2_scheme = OAuth2PasswordBearer(
    tokenUrl="/auth/login"
)
```

Now create:

```python
async def get_current_user(
    token: str = Depends(oauth2_scheme),
):
    # 1. Decode JWT
    # 2. Validate signature
    # 3. Check expiration
    # 4. Extract user ID
    # 5. Load user
    # 6. Return user

    user = ...

    if user is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication credentials",
        )

    return user
```

Then inject it:

```python
@app.get("/profile")
async def profile(
    current_user = Depends(get_current_user),
):
    return current_user
```

FastAPI automatically calls:

```text
get_current_user()
       ↓
current_user
       ↓
profile(current_user)
```

---

# 2. A more realistic JWT example

In a production application, your dependency might look like:

```python
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from jose import JWTError, jwt

oauth2_scheme = OAuth2PasswordBearer(
    tokenUrl="/auth/login"
)

SECRET_KEY = "your-secret-key"
ALGORITHM = "HS256"


async def get_current_user(
    token: str = Depends(oauth2_scheme),
):
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={
            "WWW-Authenticate": "Bearer"
        },
    )

    try:
        payload = jwt.decode(
            token,
            SECRET_KEY,
            algorithms=[ALGORITHM],
        )

        user_id = payload.get("sub")

        if user_id is None:
            raise credentials_exception

    except JWTError:
        raise credentials_exception

    user = await user_repository.get_by_id(
        int(user_id)
    )

    if user is None:
        raise credentials_exception

    return user
```

Then:

```python
@app.get("/profile")
async def profile(
    current_user = Depends(get_current_user),
):
    return {
        "id": current_user.id,
        "email": current_user.email,
    }
```

The endpoint doesn't need to know:

* where the token came from
* how JWT is decoded
* how the user ID is extracted
* how the user is retrieved

That's the benefit of DI.

---

# 3. What happens internally?

Suppose the request is:

```http
GET /profile
Authorization: Bearer eyJhbGci...
```

FastAPI sees:

```python
current_user = Depends(get_current_user)
```

It resolves the dependency:

```text
profile()
   │
   ↓
get_current_user()
   │
   ├── Depends(oauth2_scheme)
   │        ↓
   │     extract token
   │
   ├── decode JWT
   │
   ├── validate JWT
   │
   └── query user
          ↓
       User object
          ↓
   current_user
          ↓
      profile()
```

This is a **dependency graph**.

---

# 4. Can dependencies depend on other dependencies?

**Yes. Absolutely.**

This is one of FastAPI's most powerful features.

For example:

```python
async def get_db():
    ...


async def get_current_user(
    db = Depends(get_db),
):
    ...


@app.get("/profile")
async def profile(
    current_user = Depends(get_current_user),
):
    ...
```

The dependency graph becomes:

```text
profile()
   │
   ↓
get_current_user()
   │
   ↓
get_db()
```

FastAPI resolves them recursively.

---

# 5. Real-world authentication example

You might have:

```python
async def get_db():
    ...


async def get_current_user(
    token: str = Depends(oauth2_scheme),
    db: AsyncSession = Depends(get_db),
):
    ...


async def require_admin(
    current_user = Depends(get_current_user),
):
    if current_user.role != "admin":
        raise HTTPException(
            status_code=403,
            detail="Admin access required",
        )

    return current_user
```

Then:

```python
@app.delete("/users/{user_id}")
async def delete_user(
    user_id: int,
    admin = Depends(require_admin),
):
    ...
```

FastAPI resolves:

```text
delete_user
     │
     ↓
require_admin
     │
     ↓
get_current_user
     │
     ├──────────────┐
     ↓              ↓
OAuth2          get_db
     │              │
     └──────┬───────┘
            ↓
        User object
            ↓
      require_admin
            ↓
      Admin verified
            ↓
       delete_user
```

This is very common in production FastAPI applications.

---

# 6. Why is nested dependency injection useful?

Because it allows you to compose security and infrastructure concerns.

For example:

```text
get_db
   ↓
get_current_user
   ↓
require_admin
   ↓
endpoint
```

Each dependency has one responsibility.

### `get_db`

Responsible for:

```text
Database session
```

### `get_current_user`

Responsible for:

```text
Authentication
```

### `require_admin`

Responsible for:

```text
Authorization
```

### Endpoint

Responsible for:

```text
Business operation
```

This gives you clean separation of concerns.

---

# 7. How do you create reusable dependencies?

There are several ways.

The simplest is to define a normal function:

```python
def get_settings():
    return settings
```

Then:

```python
@app.get("/config")
async def config(
    settings = Depends(get_settings),
):
    return settings
```

You can reuse it:

```python
@app.get("/users")
async def users(
    settings = Depends(get_settings),
):
    ...


@app.get("/orders")
async def orders(
    settings = Depends(get_settings),
):
    ...
```

---

# 8. Reusable database dependency

A very common production dependency:

```python
from collections.abc import AsyncGenerator
from sqlalchemy.ext.asyncio import AsyncSession


async def get_db() -> AsyncGenerator[AsyncSession, None]:

    async with SessionLocal() as session:
        yield session
```

Use it anywhere:

```python
@app.get("/users")
async def users(
    db: AsyncSession = Depends(get_db),
):
    ...
```

and:

```python
@app.get("/orders")
async def orders(
    db: AsyncSession = Depends(get_db),
):
    ...
```

---

# 9. Reusable authentication dependency

```python
async def get_current_user(
    token: str = Depends(oauth2_scheme),
):
    ...
```

Then:

```python
@app.get("/profile")
async def profile(
    user = Depends(get_current_user),
):
    ...


@app.get("/orders")
async def orders(
    user = Depends(get_current_user),
):
    ...


@app.get("/documents")
async def documents(
    user = Depends(get_current_user),
):
    ...
```

You don't duplicate authentication code.

---

# 10. Reusable authorization dependency

You can build a reusable RBAC dependency.

For example:

```python
def require_role(required_role: str):

    async def checker(
        current_user = Depends(get_current_user),
    ):
        if current_user.role != required_role:
            raise HTTPException(
                status_code=403,
                detail="Insufficient permissions",
            )

        return current_user

    return checker
```

Then:

```python
@app.delete("/users/{user_id}")
async def delete_user(
    user = Depends(require_role("admin")),
):
    ...
```

And:

```python
@app.post("/documents")
async def create_document(
    user = Depends(require_role("editor")),
):
    ...
```

This creates reusable authorization rules.

---

# 11. Class-based dependencies

FastAPI dependencies don't have to be functions.

You can use classes:

```python
class Pagination:

    def __init__(
        self,
        skip: int = 0,
        limit: int = 20,
    ):
        self.skip = skip
        self.limit = limit
```

Then:

```python
@app.get("/users")
async def users(
    pagination: Pagination = Depends(Pagination),
):
    return {
        "skip": pagination.skip,
        "limit": pagination.limit,
    }
```

FastAPI creates the class and injects it.

This can be useful when the dependency has state or configuration.

---

# 12. Reusable dependencies in a real AI/RAG application

For your RAG/LLM backend, you might have:

```text
                FastAPI
                   │
          ┌────────┼─────────┐
          ↓        ↓         ↓
        get_db  get_user  get_settings
          │        │
          └────┬───┘
               ↓
         get_rag_service
               │
               ↓
         Chat Endpoint
```

For example:

```python
async def get_rag_service(
    db: AsyncSession = Depends(get_db),
    user = Depends(get_current_user),
    settings = Depends(get_settings),
):
    return RAGService(
        db=db,
        user=user,
        settings=settings,
    )
```

Then:

```python
@router.post("/chat")
async def chat(
    request: ChatRequest,
    rag_service: RAGService = Depends(
        get_rag_service
    ),
):
    return await rag_service.answer(
        request
    )
```

The router stays extremely thin.

---

# 13. How do you override dependencies in tests?

This is one of the most important FastAPI testing features.

Suppose production uses:

```python
async def get_current_user():
    ...
```

Your endpoint:

```python
@app.get("/profile")
async def profile(
    user = Depends(get_current_user),
):
    return {
        "id": user.id,
        "email": user.email,
    }
```

In a test, you don't necessarily want to generate and validate a real JWT every time.

You can replace the dependency.

---

# 14. Create a fake dependency

```python
async def override_get_current_user():

    return User(
        id=1,
        email="test@example.com",
        role="admin",
    )
```

Then:

```python
app.dependency_overrides[
    get_current_user
] = override_get_current_user
```

Now when the test calls:

```python
client.get("/profile")
```

FastAPI uses:

```text
get_current_user
        ↓
OVERRIDE
        ↓
override_get_current_user
```

instead of the production dependency.

---

# 15. Complete pytest example

Suppose your application:

```python
@app.get("/profile")
async def profile(
    user = Depends(get_current_user),
):
    return {
        "id": user.id,
        "email": user.email,
    }
```

Test:

```python
from fastapi.testclient import TestClient


def override_get_current_user():
    return User(
        id=1,
        email="test@example.com",
        role="admin",
    )


app.dependency_overrides[
    get_current_user
] = override_get_current_user


def test_profile():

    client = TestClient(app)

    response = client.get("/profile")

    assert response.status_code == 200

    assert response.json() == {
        "id": 1,
        "email": "test@example.com",
    }
```

---

# 16. Clear overrides after tests

You shouldn't leave overrides installed globally.

Use:

```python
app.dependency_overrides.clear()
```

For example:

```python
def test_profile():

    app.dependency_overrides[
        get_current_user
    ] = override_get_current_user

    try:
        response = client.get("/profile")

        assert response.status_code == 200

    finally:
        app.dependency_overrides.clear()
```

Or use a pytest fixture:

```python
import pytest


@pytest.fixture
def authenticated_client():

    app.dependency_overrides[
        get_current_user
    ] = override_get_current_user

    client = TestClient(app)

    yield client

    app.dependency_overrides.clear()
```

Then:

```python
def test_profile(authenticated_client):

    response = authenticated_client.get(
        "/profile"
    )

    assert response.status_code == 200
```

---

# 17. Overriding database dependencies

This is even more common.

Production:

```python
async def get_db():
    async with SessionLocal() as session:
        yield session
```

Test:

```python
async def override_get_db():

    async with TestSessionLocal() as session:
        yield session
```

Override:

```python
app.dependency_overrides[
    get_db
] = override_get_db
```

Now:

```text
Production:

get_db
 ↓
Production PostgreSQL


Tests:

get_db
 ↓
override_get_db
 ↓
Test PostgreSQL
```

Your application code doesn't change.

---

# 18. Mocking external services

Suppose:

```python
async def get_llm_client():
    return OpenAIClient(...)
```

Your service uses:

```python
llm = Depends(get_llm_client)
```

In tests:

```python
class FakeLLM:

    async def generate(self, prompt):
        return "mock response"
```

Override:

```python
def override_get_llm_client():
    return FakeLLM()
```

Then:

```python
app.dependency_overrides[
    get_llm_client
] = override_get_llm_client
```

Now your tests don't call the real LLM API.

This is **extremely valuable in AI applications** because it makes tests:

* faster
* deterministic
* cheaper
* independent of external APIs

---

# 19. Complete production-style example

Imagine a RAG endpoint:

```python
@router.post("/chat")
async def chat(
    request: ChatRequest,
    current_user = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
    rag_service: RAGService = Depends(get_rag_service),
):
    return await rag_service.answer(
        user=current_user,
        request=request,
        db=db,
    )
```

You might have:

```text
chat()
 │
 ├── get_current_user()
 │      └── oauth2_scheme
 │
 ├── get_db()
 │
 └── get_rag_service()
        ├── get_settings()
        ├── get_qdrant_client()
        └── get_llm_client()
```

FastAPI builds and resolves this dependency graph automatically.

---

# 20. Important interview distinction: authentication vs authorization

When discussing:

```python
get_current_user()
```

remember:

### Authentication

> **Who are you?**

```text
JWT
 ↓
Validate token
 ↓
Find user
 ↓
current_user
```

### Authorization

> **Are you allowed to perform this action?**

```text
current_user
      ↓
role / permission check
      ↓
allowed / forbidden
```

For example:

```python
async def require_admin(
    user = Depends(get_current_user),
):
    if user.role != "admin":
        raise HTTPException(
            status_code=403,
            detail="Forbidden",
        )

    return user
```

Then:

```python
@app.delete("/users/{user_id}")
async def delete_user(
    user = Depends(require_admin),
):
    ...
```

That's a clean authentication → authorization dependency chain.

---

# 21. Senior-level architecture

For a production FastAPI + RAG application, I would think about dependencies like this:

```text
                         FastAPI
                            │
                     Dependency Graph
                            │
        ┌───────────────────┼────────────────────┐
        ↓                   ↓                    ↓
     get_db          get_current_user       get_settings
        │                   │                    │
 PostgreSQL              JWT/Auth             Config
                            │
                            ↓
                       require_role
                            │
                            ↓
                     get_rag_service
                            │
              ┌─────────────┼─────────────┐
              ↓             ↓             ↓
           Qdrant         Redis          LLM
              │             │             │
              └─────────────┼─────────────┘
                            ↓
                         Endpoint
```

This gives you:

* **low coupling**
* **reusability**
* **testability**
* **clear lifecycle management**
* **centralized authentication**
* **centralized authorization**
* **easy mocking**
* **clean routers**

---

# Interview-ready answers

### 1. How do you inject the current authenticated user?

> **"I create a `get_current_user` dependency that extracts the bearer token, validates and decodes the JWT, retrieves the user, and returns the user object. Endpoints declare `current_user = Depends(get_current_user)`. FastAPI resolves the dependency before executing the endpoint."**

---

### 2. Can dependencies depend on other dependencies?

> **"Yes. FastAPI supports nested dependencies. For example, `get_current_user` can depend on `get_db` and an OAuth2 token dependency, while `require_admin` can depend on `get_current_user`. FastAPI builds and resolves this dependency graph recursively."**

Example:

```python
async def require_admin(
    user = Depends(get_current_user),
):
    ...
```

---

### 3. How do you create reusable dependencies?

> **"I define functions or classes representing reusable concerns such as database sessions, authentication, authorization, configuration, Redis clients, or external service clients, and expose them through `Depends()`. This keeps endpoint logic focused on HTTP and business operations."**

---

### 4. How do you override dependencies in tests?

> **"FastAPI provides `app.dependency_overrides`. I map the production dependency to a fake or test implementation, run the test, and then clear the overrides. This lets me replace real databases, authentication, LLM clients, or external APIs without changing application code."**

Example:

```python
app.dependency_overrides[
    get_current_user
] = override_get_current_user

response = client.get("/profile")

app.dependency_overrides.clear()
```

---

## The 4 things to remember

```text
1. Current user
   Depends(get_current_user)

2. Nested dependencies
   Depends can depend on Depends

3. Reusable dependencies
   DB / Auth / RBAC / Services / Config / Clients

4. Testing
   app.dependency_overrides
```

The **senior-level takeaway** is:

> **FastAPI Dependency Injection isn't just a convenience for passing a database session. It is a mechanism for composing application infrastructure—authentication, authorization, database sessions, services, clients, configuration, and resources—while keeping endpoints loosely coupled and easy to test.**
