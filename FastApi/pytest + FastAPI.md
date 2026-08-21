In a real-world FastAPI project, I would test the API using **pytest + FastAPI `TestClient`/`AsyncClient`**, and use **dependency overrides** to replace production dependencies such as the database, authentication, external APIs, or repositories with mocks/fakes.

### 1. Application structure

For example:

```text
app/
├── main.py
├── api/
│   └── routes.py
├── dependencies.py
├── services/
│   └── user_service.py
└── db/
    └── database.py

tests/
├── conftest.py
└── test_users.py
```

Suppose our endpoint has a database dependency:

```python
# dependencies.py

from app.db.database import get_db


async def get_current_user():
    # Normally validates JWT
    return {"id": 1, "role": "admin"}
```

And:

```python
# routes.py

from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from app.dependencies import get_db, get_current_user

router = APIRouter()


@router.get("/users/me")
async def get_me(
    db: AsyncSession = Depends(get_db),
    user=Depends(get_current_user),
):
    return {
        "id": user["id"],
        "role": user["role"],
    }
```

---

# 2. Override dependencies in tests

The important FastAPI feature is:

```python
app.dependency_overrides[original_dependency] = test_dependency
```

For example:

```python
# conftest.py

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.dependencies import get_current_user


async def override_current_user():
    return {
        "id": 123,
        "role": "admin",
    }


@pytest.fixture
def client():

    app.dependency_overrides[get_current_user] = override_current_user

    with TestClient(app) as client:
        yield client

    app.dependency_overrides.clear()
```

Now the test doesn't need to generate a real JWT.

```python
# test_users.py

def test_get_current_user(client):

    response = client.get("/users/me")

    assert response.status_code == 200

    data = response.json()

    assert data["id"] == 123
    assert data["role"] == "admin"
```

This is extremely useful because we're testing the **API behavior**, not the authentication implementation.

---

# 3. Override the database

In production:

```python
async def get_db():
    async with SessionLocal() as session:
        yield session
```

For testing, we can provide a test database/session:

```python
async def override_get_db():
    async with TestSessionLocal() as session:
        yield session
```

Then:

```python
app.dependency_overrides[get_db] = override_get_db
```

Now every endpoint using:

```python
Depends(get_db)
```

receives the test database.

---

# 4. Mock external services

Suppose our service calls an external payment API:

```python
async def get_payment_client():
    return StripeClient()
```

Instead of calling Stripe during tests:

```python
class FakePaymentClient:

    async def charge(self, amount):
        return {
            "status": "success",
            "transaction_id": "test-123"
        }


async def override_payment_client():
    return FakePaymentClient()
```

Then:

```python
app.dependency_overrides[get_payment_client] = override_payment_client
```

The test becomes deterministic and doesn't depend on an external system.

---

# 5. Testing authentication/RBAC

I would create separate overrides for different users.

### Admin

```python
async def admin_user():
    return {
        "id": 1,
        "role": "admin"
    }
```

### Normal user

```python
async def normal_user():
    return {
        "id": 2,
        "role": "user"
    }
```

Then:

```python
app.dependency_overrides[get_current_user] = normal_user

response = client.delete("/users/123")

assert response.status_code == 403
```

And:

```python
app.dependency_overrides[get_current_user] = admin_user

response = client.delete("/users/123")

assert response.status_code == 200
```

This makes RBAC tests very easy.

---

# 6. Async tests

For production applications using **async SQLAlchemy**, I generally prefer `httpx.AsyncClient`.

```python
import pytest
from httpx import ASGITransport, AsyncClient

from app.main import app


@pytest.mark.asyncio
async def test_get_me():

    async def override_user():
        return {
            "id": 123,
            "role": "admin"
        }

    app.dependency_overrides[get_current_user] = override_user

    transport = ASGITransport(app=app)

    async with AsyncClient(
        transport=transport,
        base_url="http://test"
    ) as client:

        response = await client.get("/users/me")

    assert response.status_code == 200
    assert response.json()["id"] == 123

    app.dependency_overrides.clear()
```

For an async FastAPI application, this is particularly useful when testing async DB operations, async repositories, and async external clients.

---

# 7. What I would test

For a production API, I wouldn't only test the happy path.

I'd cover:

```text
API Tests
│
├── 200 success
├── 201 creation
├── 400 validation/business error
├── 401 unauthenticated
├── 403 unauthorized
├── 404 resource not found
├── 409 conflict
├── 422 request validation
├── 500 unexpected exception
│
├── Authentication
├── RBAC
├── Database behavior
├── Repository behavior
├── External API failures
├── Timeouts
├── Retries
└── Dependency failures
```

For example:

```python
def test_user_not_found(client):
    response = client.get("/users/99999")

    assert response.status_code == 404
```

And validation:

```python
def test_invalid_user_id(client):
    response = client.get("/users/not-an-id")

    assert response.status_code == 422
```

---

# 8. Testing service layer separately

I would **not put all logic into API tests**.

For example:

```text
Router
   ↓
Service
   ↓
Repository
   ↓
Database
```

I would test each layer independently.

### Service test

```python
@pytest.mark.asyncio
async def test_create_user():

    repository = AsyncMock()

    repository.create.return_value = {
        "id": 1,
        "email": "test@example.com"
    }

    service = UserService(repository)

    result = await service.create_user(
        email="test@example.com"
    )

    assert result["id"] == 1

    repository.create.assert_awaited_once()
```

This is much faster than going through the entire HTTP stack.

---

# 9. Important cleanup

One common mistake is forgetting to clear overrides.

Always do:

```python
app.dependency_overrides.clear()
```

Otherwise one test can accidentally affect another test.

A fixture is cleaner:

```python
@pytest.fixture(autouse=True)
def clear_overrides():
    yield
    app.dependency_overrides.clear()
```

---

# Interview answer

If the interviewer asks **"How would you test FastAPI using pytest and dependency overrides?"**, I'd answer:

> "I use pytest with FastAPI's TestClient for synchronous tests or HTTPX AsyncClient for async applications. I use `app.dependency_overrides` to replace production dependencies such as database sessions, authentication, repositories, and external API clients with test implementations or mocks. This allows me to test endpoints deterministically without calling real databases or third-party services. I test both happy paths and failure scenarios such as 401, 403, 404, 422, timeouts and dependency failures. I also test the service and repository layers independently, so API tests focus mainly on HTTP behavior and integration between components. Finally, I make sure dependency overrides are cleaned up after every test to prevent test pollution."

**Key point to remember:** `Depends()` makes FastAPI highly testable because you can replace the dependency graph without changing the production code.
