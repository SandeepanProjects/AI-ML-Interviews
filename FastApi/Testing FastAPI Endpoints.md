# Testing FastAPI Endpoints Properly

Testing is extremely important in production FastAPI applications. Typically, you test:

```text
                    FastAPI Application
                            │
          ┌─────────────────┼─────────────────┐
          │                 │                 │
       Unit Tests     Integration Tests    E2E Tests
          │                 │                 │
     Service/Repo       API + Database    Full System
```

For the three questions:

1. **How do you test FastAPI endpoints?**
2. **What is `TestClient`?**
3. **How do you test async endpoints?**

Let's build a proper example.

---

# 1. How do you test FastAPI endpoints?

The standard Python testing stack is:

```text
pytest
   +
FastAPI TestClient / HTTPX
   +
pytest-asyncio or AnyIO
```

Install:

```bash
pip install pytest httpx pytest-asyncio
```

A typical project:

```text
project/
│
├── app/
│   ├── main.py
│   ├── schemas/
│   │   └── user.py
│   └── api/
│       └── users.py
│
└── tests/
    ├── test_users.py
    └── conftest.py
```

---

# 2. Simple FastAPI application

```python
# app/main.py

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel


app = FastAPI()


class UserCreate(BaseModel):
    name: str
    email: str


class UserResponse(BaseModel):
    id: int
    name: str
    email: str


fake_db = {}
next_id = 1


@app.post(
    "/users",
    response_model=UserResponse,
    status_code=201,
)
async def create_user(
    user: UserCreate,
):
    global next_id

    new_user = {
        "id": next_id,
        "name": user.name,
        "email": user.email,
    }

    fake_db[next_id] = new_user

    next_id += 1

    return new_user


@app.get(
    "/users/{user_id}",
    response_model=UserResponse,
)
async def get_user(
    user_id: int,
):

    user = fake_db.get(user_id)

    if not user:
        raise HTTPException(
            status_code=404,
            detail="User not found",
        )

    return user
```

Now we want to test these endpoints.

---

# 3. What is `TestClient`?

FastAPI provides:

```python
from fastapi.testclient import TestClient
```

`TestClient` allows you to send HTTP requests to your FastAPI application **without starting a real server**.

Normally:

```text
Browser/Postman
       │
       ▼
     Network
       │
       ▼
Uvicorn
       │
       ▼
    FastAPI
```

With `TestClient`:

```text
Test
 │
 ▼
TestClient
 │
 ▼
FastAPI Application
```

No actual network server is required.

---

# 4. Your first endpoint test

```python
# tests/test_users.py

from fastapi.testclient import TestClient

from app.main import app


client = TestClient(app)


def test_create_user():

    response = client.post(
        "/users",
        json={
            "name": "John",
            "email": "john@example.com",
        },
    )

    assert response.status_code == 201

    data = response.json()

    assert data["name"] == "John"
    assert data["email"] == "john@example.com"
    assert "id" in data
```

Run:

```bash
pytest
```

Flow:

```text
pytest
  │
  ▼
test_create_user()
  │
  ▼
client.post()
  │
  ▼
POST /users
  │
  ▼
FastAPI endpoint
  │
  ▼
Response
  │
  ▼
Assertions
```

---

# 5. Testing a GET endpoint

```python
def test_get_user():

    response = client.post(
        "/users",
        json={
            "name": "Alice",
            "email": "alice@example.com",
        },
    )

    user_id = response.json()["id"]

    response = client.get(
        f"/users/{user_id}"
    )

    assert response.status_code == 200

    data = response.json()

    assert data["id"] == user_id
    assert data["name"] == "Alice"
```

---

# 6. Testing error responses

You should always test error cases.

```python
def test_user_not_found():

    response = client.get(
        "/users/999999"
    )

    assert response.status_code == 404

    assert response.json() == {
        "detail": "User not found"
    }
```

Test:

```text
GET /users/999999
        │
        ▼
User doesn't exist
        │
        ▼
HTTPException
        │
        ▼
404
```

---

# 7. Testing request validation

FastAPI uses Pydantic validation automatically.

Endpoint:

```python
class UserCreate(BaseModel):
    name: str
    email: str
```

Test invalid data:

```python
def test_invalid_user_request():

    response = client.post(
        "/users",
        json={
            "name": "John"
            # email missing
        },
    )

    assert response.status_code == 422
```

You can also verify the validation response:

```python
def test_invalid_user_request():

    response = client.post(
        "/users",
        json={
            "name": "John",
        },
    )

    assert response.status_code == 422

    errors = response.json()["detail"]

    assert len(errors) > 0
```

---

# 8. Important: `TestClient` can test `async def` endpoints

This confuses many people.

Your endpoint can be:

```python
@app.get("/users")
async def get_users():
    return {"users": []}
```

But your test can still be:

```python
def test_get_users():

    response = client.get(
        "/users"
    )

    assert response.status_code == 200
```

You do **not** need:

```python
async def test_get_users():
```

Why?

Because `TestClient` handles the ASGI application and async execution internally.

```text
Synchronous Test
      │
      ▼
TestClient
      │
      ▼
ASGI
      │
      ▼
async FastAPI Endpoint
      │
      ▼
Response
```

So this is perfectly valid:

```python
@app.get("/health")
async def health():
    return {"status": "ok"}
```

Test:

```python
def test_health():

    response = client.get(
        "/health"
    )

    assert response.status_code == 200
```

---

# 9. So when do you need to test asynchronously?

Use an async HTTP client when:

* Your tests themselves need to call async code.
* You are testing async database operations.
* You want a fully async test setup.
* You are testing async dependencies.
* You use async fixtures.
* You want to test multiple async operations.

For this, use:

```text
pytest
+
HTTPX AsyncClient
```

---

# 10. Testing async FastAPI endpoints with `AsyncClient`

Install:

```bash
pip install httpx pytest pytest-asyncio
```

Example:

```python
# tests/test_async.py

import pytest

from httpx import (
    ASGITransport,
    AsyncClient,
)

from app.main import app


@pytest.mark.asyncio
async def test_get_user_async():

    transport = ASGITransport(
        app=app,
    )

    async with AsyncClient(
        transport=transport,
        base_url="http://test",
    ) as client:

        response = await client.get(
            "/users/1"
        )

    assert response.status_code in [
        200,
        404,
    ]
```

Notice:

```python
response = await client.get(...)
```

Because this is truly asynchronous.

---

# 11. A better async test example

Let's create an endpoint:

```python
@app.get("/health")
async def health():

    return {
        "status": "healthy"
    }
```

Async test:

```python
import pytest

from httpx import (
    ASGITransport,
    AsyncClient,
)

from app.main import app


@pytest.mark.asyncio
async def test_health():

    transport = ASGITransport(
        app=app,
    )

    async with AsyncClient(
        transport=transport,
        base_url="http://test",
    ) as client:

        response = await client.get(
            "/health"
        )

    assert response.status_code == 200

    assert response.json() == {
        "status": "healthy"
    }
```

Architecture:

```text
async pytest test
        │
        ▼
await client.get()
        │
        ▼
HTTPX AsyncClient
        │
        ▼
ASGITransport
        │
        ▼
FastAPI
        │
        ▼
async endpoint
```

---

# 12. `TestClient` vs `AsyncClient`

| Feature                            | TestClient     | AsyncClient          |
| ---------------------------------- | -------------- | -------------------- |
| Test function                      | `def`          | `async def`          |
| Request                            | `client.get()` | `await client.get()` |
| Endpoint can be async              | Yes            | Yes                  |
| Easy to use                        | Very easy      | More setup           |
| Async test code                    | No             | Yes                  |
| Async DB testing                   | Less suitable  | Better               |
| Production async integration tests | Good           | Excellent            |

Example:

### TestClient

```python
def test_health():

    response = client.get(
        "/health"
    )

    assert response.status_code == 200
```

### AsyncClient

```python
@pytest.mark.asyncio
async def test_health():

    response = await client.get(
        "/health"
    )

    assert response.status_code == 200
```

---

# 13. Real-world project: testing dependencies

Suppose your application has a database dependency.

```python
# app/database.py

from sqlalchemy.ext.asyncio import (
    AsyncSession,
)


async def get_db():

    async with SessionLocal() as session:

        try:

            yield session

        finally:

            await session.close()
```

Endpoint:

```python
@app.get("/users/{user_id}")
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

    if not user:

        raise HTTPException(
            status_code=404,
            detail="User not found",
        )

    return user
```

During testing, you usually don't want to use the production database.

---

# 14. Dependency override

FastAPI provides:

```python
app.dependency_overrides
```

Example:

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

Now:

```text
Production

Endpoint
   │
   ▼
get_db()
   │
   ▼
Production PostgreSQL


Testing

Endpoint
   │
   ▼
get_test_db()
   │
   ▼
Test Database
```

Test:

```python
from fastapi.testclient import TestClient

from app.main import app
from app.database import get_db


async def get_test_db():
    async with TestSessionLocal() as session:
        yield session


app.dependency_overrides[
    get_db
] = get_test_db


client = TestClient(app)


def test_get_user():

    response = client.get(
        "/users/1"
    )

    assert response.status_code == 200
```

Important: always clean up overrides.

```python
app.dependency_overrides.clear()
```

Otherwise one test can affect another.

---

# 15. Proper pytest fixture

Instead of global setup:

```python
client = TestClient(app)
```

use fixtures.

```python
# tests/conftest.py

import pytest

from fastapi.testclient import TestClient

from app.main import app


@pytest.fixture
def client():

    with TestClient(app) as test_client:

        yield test_client
```

Test:

```python
def test_health(client):

    response = client.get(
        "/health"
    )

    assert response.status_code == 200
```

Why fixtures?

```text
Fixture
   │
   ├── Setup
   │
   ▼
Test
   │
   ▼
Assertions
   │
   ▼
Cleanup
```

---

# 16. Async fixture

For async resources, use async fixtures.

Example concept:

```python
import pytest_asyncio


@pytest_asyncio.fixture
async def async_client():

    transport = ASGITransport(
        app=app,
    )

    async with AsyncClient(
        transport=transport,
        base_url="http://test",
    ) as client:

        yield client
```

Test:

```python
@pytest.mark.asyncio
async def test_health(
    async_client,
):

    response = await async_client.get(
        "/health"
    )

    assert response.status_code == 200
```

---

# 17. Testing POST endpoints properly

Let's test:

```text
POST /users
```

Success:

```python
def test_create_user_success(client):

    payload = {
        "name": "Sandeep",
        "email": "sandeep@example.com",
    }

    response = client.post(
        "/users",
        json=payload,
    )

    assert response.status_code == 201

    data = response.json()

    assert data["name"] == payload["name"]
    assert data["email"] == payload["email"]
    assert "id" in data
```

Invalid input:

```python
def test_create_user_missing_email(
    client,
):

    response = client.post(
        "/users",
        json={
            "name": "Sandeep",
        },
    )

    assert response.status_code == 422
```

---

# 18. Testing PUT endpoint

Endpoint:

```python
@app.put("/users/{user_id}")
async def update_user(
    user_id: int,
    user: UserCreate,
):

    existing = fake_db.get(
        user_id
    )

    if not existing:

        raise HTTPException(
            status_code=404,
            detail="User not found",
        )

    existing.update(
        user.model_dump()
    )

    return existing
```

Test:

```python
def test_update_user(client):

    create_response = client.post(
        "/users",
        json={
            "name": "John",
            "email": "john@example.com",
        },
    )

    user_id = create_response.json()["id"]

    response = client.put(
        f"/users/{user_id}",
        json={
            "name": "John Updated",
            "email": "new@example.com",
        },
    )

    assert response.status_code == 200

    data = response.json()

    assert data["name"] == (
        "John Updated"
    )
```

---

# 19. Testing DELETE endpoint

Endpoint:

```python
@app.delete(
    "/users/{user_id}",
    status_code=204,
)
async def delete_user(
    user_id: int,
):

    if user_id not in fake_db:

        raise HTTPException(
            status_code=404,
            detail="User not found",
        )

    del fake_db[user_id]
```

Test:

```python
def test_delete_user(client):

    create_response = client.post(
        "/users",
        json={
            "name": "John",
            "email": "john@example.com",
        },
    )

    user_id = create_response.json()["id"]

    response = client.delete(
        f"/users/{user_id}"
    )

    assert response.status_code == 204

    response = client.get(
        f"/users/{user_id}"
    )

    assert response.status_code == 404
```

---

# 20. How I test authentication

Suppose:

```python
@app.get("/profile")
async def get_profile(
    user=Depends(get_current_user),
):
    return user
```

In unit/API tests, you can override authentication.

```python
async def override_get_current_user():

    return {
        "id": 1,
        "name": "Test User",
        "role": "admin",
    }
```

Override:

```python
app.dependency_overrides[
    get_current_user
] = override_get_current_user
```

Test:

```python
def test_profile(client):

    response = client.get(
        "/profile"
    )

    assert response.status_code == 200

    assert response.json()["name"] == (
        "Test User"
    )
```

This is better than generating real JWT tokens for every service-level test.

For dedicated authentication integration tests, however, you should test the real JWT flow separately.

---

# 21. Unit test vs API test

This distinction is important.

## Unit test

Tests a small piece of code.

```python
def test_calculate_total():

    result = calculate_total(
        100,
        0.1,
    )

    assert result == 110
```

No:

```text
HTTP
Database
Redis
Network
```

---

## API/integration test

```text
Test
 │
 ▼
HTTP Request
 │
 ▼
FastAPI Router
 │
 ▼
Dependencies
 │
 ▼
Service
 │
 ▼
Repository
 │
 ▼
Database
```

Example:

```python
def test_create_user_api(client):

    response = client.post(
        "/users",
        json={
            "name": "John",
            "email": "john@example.com",
        },
    )

    assert response.status_code == 201
```

---

# 22. A production test structure

```text
project/
│
├── app/
│   ├── main.py
│   ├── api/
│   │   └── users.py
│   ├── services/
│   │   └── user_service.py
│   └── repositories/
│       └── user_repository.py
│
└── tests/
    │
    ├── unit/
    │   ├── test_user_service.py
    │   └── test_user_repository.py
    │
    ├── integration/
    │   ├── test_users_api.py
    │   └── test_auth.py
    │
    └── conftest.py
```

---

# 23. Example complete API test

Application:

```python
# app/main.py

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel


app = FastAPI()

users = {}


class UserCreate(BaseModel):

    name: str
    email: str


@app.post(
    "/users",
    status_code=201,
)
async def create_user(
    user: UserCreate,
):

    user_id = len(users) + 1

    users[user_id] = {
        "id": user_id,
        **user.model_dump(),
    }

    return users[user_id]


@app.get("/users/{user_id}")
async def get_user(
    user_id: int,
):

    user = users.get(user_id)

    if not user:

        raise HTTPException(
            status_code=404,
            detail="User not found",
        )

    return user
```

Test:

```python
# tests/test_users.py

from fastapi.testclient import TestClient

from app.main import app


client = TestClient(app)


def test_create_and_get_user():

    # CREATE
    response = client.post(
        "/users",
        json={
            "name": "Sandeep",
            "email": "sandeep@example.com",
        },
    )

    assert response.status_code == 201

    created_user = response.json()

    user_id = created_user["id"]

    # GET
    response = client.get(
        f"/users/{user_id}"
    )

    assert response.status_code == 200

    user = response.json()

    assert user["name"] == "Sandeep"
    assert (
        user["email"]
        == "sandeep@example.com"
    )
```

---

# 24. Best interview answer

### Question: How do you test FastAPI endpoints?

> **"I typically use pytest with FastAPI's TestClient or HTTPX. TestClient allows me to send requests directly to the ASGI application without starting a real Uvicorn server. I test successful responses, validation errors, authentication and authorization, exception handling, and dependency behavior. For database tests, I override dependencies to inject a test database. I use pytest fixtures for setup and cleanup."**

### Question: What is TestClient?

> **"TestClient is a testing client provided by FastAPI through Starlette. It lets tests send HTTP requests directly to the ASGI application without running an external server. Even if the FastAPI endpoint is async, I can write normal synchronous tests because TestClient handles the async application execution internally."**

### Question: How do you test async endpoints?

> **"A synchronous TestClient can test async endpoints. However, when the test itself needs asynchronous behavior—such as async database setup, async fixtures, or concurrent calls—I use pytest with HTTPX AsyncClient and ASGITransport. The test becomes async and HTTP requests are awaited."**

## Remember this

```text
Simple API Test
      ↓
pytest + TestClient

Async endpoint only
      ↓
pytest + TestClient is enough

Async test setup / async DB / concurrency
      ↓
pytest + AsyncClient + ASGITransport
```

This is the key distinction interviewers usually want to hear.
