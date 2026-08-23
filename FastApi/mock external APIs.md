# 1. How do you mock external APIs in FastAPI?

In a real FastAPI application, your service may call:

* Payment APIs
* LLM APIs
* Email services
* Weather APIs
* Other microservices

Example:

```text
FastAPI Endpoint
      │
      ▼
UserService
      │
      ▼
External Payment API
```

During tests, you usually **do not want to call the real API** because:

* Tests become slow
* Tests can fail because the external service is down
* Tests may cost money (especially LLM APIs)
* Tests depend on network availability
* Results may be unpredictable
* You could accidentally modify real production data

Instead:

```text
Production

FastAPI → Service → Real External API


Test

FastAPI → Service → Mock External API
```

There are several approaches.

---

# Project example

```text
app/
├── main.py
├── api/
│   └── weather.py
├── services/
│   └── weather_service.py
└── dependencies.py

tests/
├── conftest.py
└── test_weather.py
```

---

# 2. Example: Service calling an external API

Suppose we call a weather service using `httpx`.

```python
# app/services/weather_service.py

import httpx


class WeatherService:

    async def get_weather(
        self,
        city: str,
    ) -> dict:

        async with httpx.AsyncClient(
            timeout=5.0,
        ) as client:

            response = await client.get(
                "https://api.weather.example/weather",
                params={
                    "city": city,
                },
            )

            response.raise_for_status()

            return response.json()
```

FastAPI endpoint:

```python
# app/main.py

from fastapi import FastAPI, HTTPException

from app.services.weather_service import (
    WeatherService,
)


app = FastAPI()


@app.get("/weather/{city}")
async def get_weather(
    city: str,
):

    service = WeatherService()

    try:

        weather = await service.get_weather(
            city
        )

        return weather

    except Exception:

        raise HTTPException(
            status_code=503,
            detail="Weather service unavailable",
        )
```

Now let's test this.

---

# 3. Mock using `unittest.mock.AsyncMock`

Since the external method is async:

```python
async def get_weather(...)
```

we should use:

```python
AsyncMock
```

Test:

```python
# tests/test_weather.py

from unittest.mock import AsyncMock, patch

from fastapi.testclient import TestClient

from app.main import app


client = TestClient(app)


def test_get_weather():

    fake_response = {
        "city": "Bangalore",
        "temperature": 25,
        "condition": "Sunny",
    }

    with patch(
        "app.main.WeatherService.get_weather",
        new_callable=AsyncMock,
    ) as mock_get_weather:

        mock_get_weather.return_value = (
            fake_response
        )

        response = client.get(
            "/weather/Bangalore"
        )

    assert response.status_code == 200

    assert response.json() == fake_response

    mock_get_weather.assert_awaited_once_with(
        "Bangalore"
    )
```

## What happened?

```text
Test
 │
 ▼
GET /weather/Bangalore
 │
 ▼
FastAPI endpoint
 │
 ▼
WeatherService.get_weather()
 │
 ├── Production → Real API ❌
 │
 └── Test → AsyncMock returns fake data ✅
```

The real external API is never called.

---

# 4. Important: Patch where the object is USED

This is a very common interview question.

Suppose:

```python
# app/main.py

from app.services.weather_service import WeatherService
```

Then inside `main.py`:

```python
service = WeatherService()
```

You should generally patch the reference where it is **looked up**:

```python
patch(
    "app.main.WeatherService.get_weather"
)
```

Not necessarily:

```python
patch(
    "app.services.weather_service.WeatherService.get_weather"
)
```

Why?

Because Python imports create references.

```text
weather_service.py
       │
       │ import
       ▼
main.py
       │
       ▼
WeatherService reference used here
```

A good rule:

> **Patch where the code under test looks up the dependency.**

---

# 5. Mock different scenarios

A good test suite tests more than success.

## Success

```python
mock_get_weather.return_value = {
    "temperature": 25
}
```

## External API timeout

```python
import httpx


mock_get_weather.side_effect = (
    httpx.TimeoutException(
        "Connection timeout"
    )
)
```

Test:

```python
from unittest.mock import (
    AsyncMock,
    patch,
)

import httpx


def test_weather_timeout():

    with patch(
        "app.main.WeatherService.get_weather",
        new_callable=AsyncMock,
    ) as mock_get_weather:

        mock_get_weather.side_effect = (
            httpx.TimeoutException(
                "Timeout"
            )
        )

        response = client.get(
            "/weather/Bangalore"
        )

    assert response.status_code == 503
```

You should also test:

```text
200 → Success
400 → External validation error
401 → Invalid external credentials
429 → External rate limit
500 → External service failure
Timeout → Timeout
```

---

# 6. Mock HTTP requests directly using `respx`

For `httpx`, a very useful approach is to mock at the HTTP layer.

Install:

```bash
pip install respx
```

Suppose:

```python
# app/services/weather_service.py

import httpx


class WeatherService:

    async def get_weather(
        self,
        city: str,
    ):

        async with httpx.AsyncClient() as client:

            response = await client.get(
                "https://weather-api.com/weather",
                params={
                    "city": city,
                },
            )

            response.raise_for_status()

            return response.json()
```

Test:

```python
import httpx
import pytest
import respx

from app.services.weather_service import (
    WeatherService,
)


@pytest.mark.asyncio
@respx.mock
async def test_weather_service():

    respx.get(
        "https://weather-api.com/weather",
        params={
            "city": "Bangalore",
        },
    ).mock(
        return_value=httpx.Response(
            200,
            json={
                "temperature": 25,
                "condition": "Sunny",
            },
        )
    )

    service = WeatherService()

    result = await service.get_weather(
        "Bangalore"
    )

    assert result["temperature"] == 25
```

Architecture:

```text
WeatherService
       │
       ▼
httpx.AsyncClient
       │
       ▼
respx intercepts request
       │
       ▼
Fake HTTP Response
```

This is useful because you're testing your actual HTTP client code without calling the real internet.

---

# 7. Mock an LLM API

For an AI project, suppose:

```python
class LLMService:

    async def generate(
        self,
        prompt: str,
    ) -> str:

        response = await client.generate(
            model="some-model",
            prompt=prompt,
        )

        return response.text
```

Test:

```python
from unittest.mock import (
    AsyncMock,
    patch,
)


@pytest.mark.asyncio
async def test_llm_service():

    with patch(
        "app.services.llm_service.client.generate",
        new_callable=AsyncMock,
    ) as mock_generate:

        mock_generate.return_value.text = (
            "This is a fake response"
        )

        service = LLMService()

        result = await service.generate(
            "Hello"
        )

    assert result == (
        "This is a fake response"
    )

    mock_generate.assert_awaited_once()
```

The important idea:

```text
Production:
Prompt → LLM API → Cost 💰


Testing:
Prompt → Mock → Fake Response
```

---

# 8. How do you override dependencies in FastAPI?

FastAPI provides:

```python
app.dependency_overrides
```

This is one of the most powerful testing features in FastAPI.

Suppose we have:

```python
@app.get("/users")
async def get_users(
    db=Depends(get_db),
):
    ...
```

Production:

```text
Endpoint
   │
   ▼
Depends(get_db)
   │
   ▼
Production PostgreSQL
```

During tests:

```text
Endpoint
   │
   ▼
Depends(get_db)
   │
   ▼
Test Database
```

We override:

```python
app.dependency_overrides[
    get_db
] = get_test_db
```

---

# 9. Example: Database dependency override

Production dependency:

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
# app/api/users.py

from fastapi import (
    APIRouter,
    Depends,
)

router = APIRouter()


@router.get("/users")
async def get_users(
    db: AsyncSession = Depends(get_db),
):

    result = await db.execute(
        select(User)
    )

    users = result.scalars().all()

    return users
```

---

# 10. Test database dependency

```python
# tests/conftest.py

import pytest_asyncio

from app.database import (
    TestSessionLocal,
)


async def get_test_db():

    async with TestSessionLocal() as session:

        try:

            yield session

        finally:

            await session.close()
```

Override:

```python
from app.main import app
from app.database import get_db


app.dependency_overrides[
    get_db
] = get_test_db
```

Now every endpoint using:

```python
Depends(get_db)
```

receives:

```text
TestSessionLocal
```

instead of:

```text
Production SessionLocal
```

---

# 11. Proper dependency override fixture

You should avoid globally changing dependencies without cleanup.

Use pytest fixtures:

```python
# tests/conftest.py

import pytest

from fastapi.testclient import TestClient

from app.main import app
from app.database import get_db


async def get_test_db():
    async with TestSessionLocal() as session:
        yield session


@pytest.fixture
def client():

    app.dependency_overrides[
        get_db
    ] = get_test_db

    with TestClient(app) as test_client:

        yield test_client

    app.dependency_overrides.clear()
```

Test:

```python
def test_get_users(client):

    response = client.get(
        "/users"
    )

    assert response.status_code == 200
```

The lifecycle:

```text
Test starts
     │
     ▼
Override dependency
     │
     ▼
Run test
     │
     ▼
Remove override
```

This prevents one test from affecting another.

---

# 12. Override authentication dependency

Suppose:

```python
async def get_current_user():

    token = ...

    user = validate_token(token)

    return user
```

Endpoint:

```python
@app.get("/profile")
async def get_profile(
    user=Depends(
        get_current_user
    ),
):

    return user
```

For most endpoint tests, you don't want to generate a real JWT.

Override it:

```python
async def override_current_user():

    return {
        "id": 1,
        "name": "Test User",
        "role": "admin",
    }
```

Test setup:

```python
app.dependency_overrides[
    get_current_user
] = override_current_user
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

Flow:

```text
Production:

Request
  │
  ▼
JWT Validation
  │
  ▼
Database
  │
  ▼
Current User


Test:

Request
  │
  ▼
Mock Dependency
  │
  ▼
Fake User
```

---

# 13. Override dependency for external services

This is often cleaner than patching.

Production:

```python
class PaymentService:

    async def charge(
        self,
        amount: float,
    ):
        ...
```

Dependency:

```python
def get_payment_service():

    return PaymentService()
```

Endpoint:

```python
@app.post("/payments")
async def create_payment(
    amount: float,
    payment_service=Depends(
        get_payment_service
    ),
):

    result = await payment_service.charge(
        amount
    )

    return result
```

Test:

```python
from unittest.mock import AsyncMock


mock_payment_service = AsyncMock()

mock_payment_service.charge.return_value = {
    "status": "success",
    "transaction_id": "test-123",
}


def override_payment_service():

    return mock_payment_service


app.dependency_overrides[
    get_payment_service
] = override_payment_service
```

Test:

```python
def test_payment(client):

    response = client.post(
        "/payments",
        params={
            "amount": 100,
        },
    )

    assert response.status_code == 200

    assert response.json()["status"] == (
        "success"
    )

    mock_payment_service.charge.assert_awaited_once_with(
        100
    )
```

This is a very clean design.

---

# 14. Patch vs Dependency Override

This is extremely important.

## `patch()`

```python
patch(
    "app.main.PaymentService"
)
```

You are replacing a Python object.

```text
Code
 │
 ▼
Mock Python Object
```

Use it when:

* Legacy code directly creates dependencies
* You cannot change application design
* You want to mock a specific method/function

---

## Dependency Override

```python
app.dependency_overrides[
    get_payment_service
] = mock_service
```

You are replacing FastAPI's dependency.

```text
FastAPI
 │
 ▼
Depends()
 │
 ▼
Production dependency
       │
       └── Test override
```

Use it when:

* Testing FastAPI applications
* Your code uses `Depends()`
* You want clean dependency injection
* You want to replace databases/services/authentication

### Comparison

|                        | `patch()`     | Dependency Override |
| ---------------------- | ------------- | ------------------- |
| Replaces               | Python object | FastAPI dependency  |
| Best for               | Unit tests    | FastAPI API tests   |
| Requires `Depends()`   | No            | Yes                 |
| Works with legacy code | Excellent     | Requires DI         |
| Cleanup needed         | Yes           | Yes                 |
| Test readability       | Medium        | High                |

---

# 15. Production architecture for testable code

A good architecture:

```text
Router
   │
   ▼
Service Dependency
   │
   ▼
Service
   │
   ├── Repository
   │
   └── External API Client
```

Example:

```python
# dependencies.py

def get_weather_service():

    return WeatherService()
```

Router:

```python
@router.get("/weather/{city}")
async def get_weather(
    city: str,
    service=Depends(
        get_weather_service
    ),
):

    return await service.get_weather(
        city
    )
```

Production:

```text
Depends(get_weather_service)
             │
             ▼
       WeatherService
             │
             ▼
       Real API Client
```

Test:

```text
Depends(get_weather_service)
             │
             ▼
         Mock Service
```

Test:

```python
from unittest.mock import AsyncMock


@pytest.fixture
def mock_weather_service():

    service = AsyncMock()

    service.get_weather.return_value = {
        "temperature": 25,
    }

    return service


def test_weather(
    client,
    mock_weather_service,
):

    app.dependency_overrides[
        get_weather_service
    ] = lambda: mock_weather_service

    response = client.get(
        "/weather/Bangalore"
    )

    assert response.status_code == 200

    assert response.json() == {
        "temperature": 25
    }

    mock_weather_service.get_weather.assert_awaited_once_with(
        "Bangalore"
    )

    app.dependency_overrides.clear()
```

---

# 16. Using `MagicMock` vs `AsyncMock`

This is important.

## Synchronous function

```python
def get_user():
    ...
```

Use:

```python
MagicMock
```

Example:

```python
from unittest.mock import MagicMock


mock_repo = MagicMock()

mock_repo.get_user.return_value = {
    "id": 1
}
```

---

## Async function

```python
async def get_user():
    ...
```

Use:

```python
AsyncMock
```

Example:

```python
from unittest.mock import AsyncMock


mock_repo = AsyncMock()

mock_repo.get_user.return_value = {
    "id": 1
}
```

Then:

```python
await repo.get_user()
```

Test:

```python
mock_repo.get_user.assert_awaited_once()
```

Do not use a normal `MagicMock` for an awaited async method unless you intentionally configure it as an async mock.

---

# 17. A complete realistic example

Let's build a clean endpoint.

## External API client

```python
# app/clients/payment_client.py

class PaymentClient:

    async def charge(
        self,
        amount: float,
    ):

        # Real external API call
        ...
```

## Service

```python
# app/services/payment_service.py

class PaymentService:

    def __init__(
        self,
        client: PaymentClient,
    ):

        self.client = client

    async def create_payment(
        self,
        amount: float,
    ):

        result = await self.client.charge(
            amount
        )

        return result
```

## Dependency

```python
# app/dependencies.py

from app.clients.payment_client import (
    PaymentClient,
)

from app.services.payment_service import (
    PaymentService,
)


def get_payment_service():

    client = PaymentClient()

    return PaymentService(
        client
    )
```

## Router

```python
# app/api/payments.py

from fastapi import (
    APIRouter,
    Depends,
)

router = APIRouter()


@router.post("/payments")
async def create_payment(
    amount: float,
    service=Depends(
        get_payment_service
    ),
):

    return await service.create_payment(
        amount
    )
```

---

# 18. Test using dependency override

```python
# tests/test_payments.py

from unittest.mock import AsyncMock

from fastapi.testclient import TestClient

from app.main import app
from app.dependencies import (
    get_payment_service,
)


def test_create_payment():

    mock_service = AsyncMock()

    mock_service.create_payment.return_value = {
        "status": "success",
        "transaction_id": "txn_test_123",
    }

    app.dependency_overrides[
        get_payment_service
    ] = lambda: mock_service

    try:

        with TestClient(app) as client:

            response = client.post(
                "/payments",
                params={
                    "amount": 100,
                },
            )

        assert response.status_code == 200

        assert response.json() == {
            "status": "success",
            "transaction_id": "txn_test_123",
        }

        mock_service.create_payment.assert_awaited_once_with(
            100,
        )

    finally:

        app.dependency_overrides.clear()
```

This is a clean production testing pattern.

---

# 19. Recommended strategy

For your FastAPI projects, I recommend:

```text
Router
  │
  │ Depends()
  ▼
Service
  │
  ├── Repository
  │       │
  │       ▼
  │      Database
  │
  └── External Client
          │
          ▼
       LLM / API
```

### Unit test the service

Mock:

```text
Repository
External Client
```

### API test the router

Override:

```text
Authentication
Database
Service
```

### Integration test

Use:

```text
Real Test Database
Real Redis
Mocked External APIs
```

### End-to-end test

Use:

```text
Real deployed services
Sandbox external APIs
```

---

# 20. Best interview answer

### How do you mock external APIs?

> **"I avoid calling real external APIs in unit and most integration tests because they make tests slow, flaky, expensive, and dependent on external availability. For a service abstraction, I use `unittest.mock`, specifically `AsyncMock` for async methods and `MagicMock` for synchronous methods. I configure return values and side effects to test success, timeouts, rate limits, and failures. For testing the actual HTTP client layer, I can intercept HTTPX requests using tools such as `respx`."**

### How do you override dependencies in FastAPI?

> **"FastAPI provides `app.dependency_overrides`, which lets me replace production dependencies during tests. For example, I can replace a production database session with a test database, JWT authentication with a fake current user, or an external service with an `AsyncMock`. I normally apply overrides in pytest fixtures and clear them after the test to prevent test contamination."**

## The most important distinction

```text
Mock external API:
Python object / HTTP call is replaced
        ↓
AsyncMock / MagicMock / respx


Override dependency:
FastAPI dependency is replaced
        ↓
app.dependency_overrides
```

For a **production FastAPI + AI application**, the cleanest approach is usually to design external clients, repositories, and services behind dependencies so they can be easily overridden and tested.
