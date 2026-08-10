Below is a **real-world FastAPI project**, structured the way you would commonly build a backend for an **AI/LLM/RAG application**. I’ll use an **Order API** as the business domain because it demonstrates the production patterns clearly.

The important goal is not just to write:

```python
@app.get("/users")
```

but to show:

* Project structure
* Configuration management
* Dependency Injection
* Pydantic validation
* SQLAlchemy async database
* Repository pattern
* Service layer
* API routers
* Centralized exception handling
* Logging
* Middleware
* Authentication pattern
* Health checks
* Docker
* Testing

---

# 1. Real-world architecture

```text
Client
   |
   v
FastAPI Router
   |
   v
Dependencies
(Auth, DB Session, etc.)
   |
   v
Service Layer
(Business Logic)
   |
   v
Repository Layer
(Database Queries)
   |
   v
PostgreSQL
```

For an AI application, the architecture might become:

```text
Client
   |
   v
FastAPI API
   |
   +--> Auth / RBAC
   |
   +--> AI Service
   |      |
   |      +--> LLM
   |      +--> RAG Retriever
   |      +--> Vector DB
   |
   +--> PostgreSQL
   |
   +--> Redis
   |
   +--> Observability
```

---

# 2. Project structure

```text
fastapi-production/
│
├── app/
│   ├── main.py
│   │
│   ├── api/
│   │   ├── deps.py
│   │   └── v1/
│   │       ├── router.py
│   │       └── orders.py
│   │
│   ├── core/
│   │   ├── config.py
│   │   ├── logging.py
│   │   ├── exceptions.py
│   │   └── middleware.py
│   │
│   ├── db/
│   │   ├── session.py
│   │   └── base.py
│   │
│   ├── models/
│   │   └── order.py
│   │
│   ├── schemas/
│   │   └── order.py
│   │
│   ├── repositories/
│   │   └── order_repository.py
│   │
│   ├── services/
│   │   └── order_service.py
│   │
│   └── tests/
│       └── test_orders.py
│
├── .env
├── requirements.txt
├── Dockerfile
└── docker-compose.yml
```

---

# 3. Install dependencies

`requirements.txt`

```txt
fastapi
uvicorn[standard]

sqlalchemy
asyncpg

pydantic
pydantic-settings

python-json-logger

pytest
pytest-asyncio
httpx
```

Install:

```bash
pip install -r requirements.txt
```

---

# 4. Configuration

## `app/core/config.py`

Never hardcode database passwords, API keys, or secrets.

```python
from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    APP_NAME: str = "Production FastAPI"
    ENVIRONMENT: str = "development"

    API_V1_PREFIX: str = "/api/v1"

    DATABASE_URL: str

    DEBUG: bool = False

    model_config = SettingsConfigDict(
        env_file=".env",
        case_sensitive=True,
        extra="ignore",
    )


@lru_cache
def get_settings() -> Settings:
    return Settings()
```

## `.env`

```env
APP_NAME=Production FastAPI
ENVIRONMENT=development

DATABASE_URL=postgresql+asyncpg://postgres:postgres@localhost:5432/orders_db

DEBUG=true
```

### Why `@lru_cache`?

Without caching:

```python
get_settings()
```

creates the settings object repeatedly.

With caching:

```python
@lru_cache
def get_settings():
```

we create it once and reuse it.

---

# 5. Database connection

## `app/db/session.py`

For real-world APIs, we commonly use SQLAlchemy async sessions.

```python
from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from app.core.config import get_settings


settings = get_settings()

engine = create_async_engine(
    settings.DATABASE_URL,
    echo=settings.DEBUG,
    pool_pre_ping=True,
    pool_size=10,
    max_overflow=20,
)

AsyncSessionLocal = async_sessionmaker(
    bind=engine,
    class_=AsyncSession,
    expire_on_commit=False,
)
```

### Why connection pooling?

Suppose 1,000 requests arrive.

Bad approach:

```text
Request 1 --> Create DB Connection
Request 2 --> Create DB Connection
Request 3 --> Create DB Connection
...
```

This can exhaust database connections.

Instead:

```text
FastAPI
   |
   v
Connection Pool
   |
   +--> Connection 1
   +--> Connection 2
   +--> Connection 3
```

Connections are reused.

---

# 6. Database dependency injection

## `app/api/deps.py`

```python
from collections.abc import AsyncGenerator

from sqlalchemy.ext.asyncio import AsyncSession

from app.db.session import AsyncSessionLocal


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    async with AsyncSessionLocal() as session:
        try:
            yield session
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()
```

FastAPI usage:

```python
@router.get("/")
async def get_data(
    db: AsyncSession = Depends(get_db)
):
    ...
```

The flow is:

```text
Request
   |
   v
get_db()
   |
   v
Create/Get Session
   |
   v
API executes
   |
   v
Close Session
```

This is one of the most important real-world FastAPI patterns.

---

# 7. SQLAlchemy model

## `app/models/order.py`

```python
from datetime import datetime

from sqlalchemy import DateTime, Float, Integer, String, func
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    pass


class Order(Base):
    __tablename__ = "orders"

    id: Mapped[int] = mapped_column(
        Integer,
        primary_key=True,
        index=True,
    )

    customer_name: Mapped[str] = mapped_column(
        String(100),
        nullable=False,
        index=True,
    )

    product_name: Mapped[str] = mapped_column(
        String(200),
        nullable=False,
    )

    amount: Mapped[float] = mapped_column(
        Float,
        nullable=False,
    )

    status: Mapped[str] = mapped_column(
        String(50),
        default="PENDING",
        nullable=False,
        index=True,
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )
```

The database table is roughly:

```text
orders
------------------------------------------------
id
customer_name
product_name
amount
status
created_at
```

---

# 8. Pydantic schemas

This is very important.

Do **not** directly expose your database models everywhere.

## `app/schemas/order.py`

```python
from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field


class OrderCreate(BaseModel):
    customer_name: str = Field(
        min_length=2,
        max_length=100,
        examples=["Sandeep"],
    )

    product_name: str = Field(
        min_length=2,
        max_length=200,
        examples=["Laptop"],
    )

    amount: float = Field(
        gt=0,
        examples=[50000],
    )


class OrderUpdate(BaseModel):
    product_name: str | None = Field(
        default=None,
        min_length=2,
        max_length=200,
    )

    amount: float | None = Field(
        default=None,
        gt=0,
    )

    status: str | None = None


class OrderResponse(BaseModel):
    id: int
    customer_name: str
    product_name: str
    amount: float
    status: str
    created_at: datetime

    model_config = ConfigDict(
        from_attributes=True,
    )
```

### Why separate schemas?

Bad:

```python
class Order:
    password
    internal_notes
    amount
    customer_name
```

If you return the ORM model directly, you may accidentally expose:

```json
{
  "password": "secret",
  "internal_notes": "VIP customer"
}
```

A response schema explicitly controls what the API exposes.

---

# 9. Repository layer

The repository should handle **database operations only**.

## `app/repositories/order_repository.py`

```python
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.order import Order


class OrderRepository:

    def __init__(self, db: AsyncSession):
        self.db = db

    async def create(
        self,
        order: Order,
    ) -> Order:

        self.db.add(order)

        await self.db.flush()
        await self.db.refresh(order)

        return order

    async def get_by_id(
        self,
        order_id: int,
    ) -> Order | None:

        result = await self.db.execute(
            select(Order).where(
                Order.id == order_id
            )
        )

        return result.scalar_one_or_none()

    async def get_all(
        self,
        offset: int = 0,
        limit: int = 20,
    ) -> list[Order]:

        result = await self.db.execute(
            select(Order)
            .offset(offset)
            .limit(limit)
            .order_by(Order.created_at.desc())
        )

        return list(result.scalars().all())

    async def update(
        self,
        order: Order,
    ) -> Order:

        await self.db.flush()

        await self.db.refresh(order)

        return order

    async def delete(
        self,
        order: Order,
    ) -> None:

        await self.db.delete(order)
```

### Repository responsibility

Good:

```text
Repository
    |
    +--> SELECT
    +--> INSERT
    +--> UPDATE
    +--> DELETE
```

Bad:

```text
Repository
    |
    +--> Database
    +--> Email
    +--> Business rules
    +--> Payment
    +--> Authentication
```

Keep responsibilities separate.

---

# 10. Service layer

The service contains business logic.

## `app/core/exceptions.py`

```python
class OrderNotFoundError(Exception):

    def __init__(self, order_id: int):
        self.order_id = order_id
        super().__init__(
            f"Order {order_id} was not found"
        )


class InvalidOrderStatusError(Exception):
    pass
```

## `app/services/order_service.py`

```python
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.exceptions import (
    InvalidOrderStatusError,
    OrderNotFoundError,
)

from app.models.order import Order
from app.repositories.order_repository import OrderRepository
from app.schemas.order import (
    OrderCreate,
    OrderUpdate,
)


class OrderService:

    VALID_STATUSES = {
        "PENDING",
        "PROCESSING",
        "COMPLETED",
        "CANCELLED",
    }

    def __init__(
        self,
        db: AsyncSession,
    ):
        self.db = db
        self.repository = OrderRepository(db)

    async def create_order(
        self,
        data: OrderCreate,
    ) -> Order:

        order = Order(
            customer_name=data.customer_name,
            product_name=data.product_name,
            amount=data.amount,
            status="PENDING",
        )

        created_order = await self.repository.create(
            order
        )

        await self.db.commit()

        return created_order

    async def get_order(
        self,
        order_id: int,
    ) -> Order:

        order = await self.repository.get_by_id(
            order_id
        )

        if not order:
            raise OrderNotFoundError(order_id)

        return order

    async def list_orders(
        self,
        offset: int,
        limit: int,
    ) -> list[Order]:

        return await self.repository.get_all(
            offset=offset,
            limit=limit,
        )

    async def update_order(
        self,
        order_id: int,
        data: OrderUpdate,
    ) -> Order:

        order = await self.get_order(order_id)

        update_data = data.model_dump(
            exclude_unset=True
        )

        if "status" in update_data:

            status = update_data["status"].upper()

            if status not in self.VALID_STATUSES:
                raise InvalidOrderStatusError(
                    f"Invalid status: {status}"
                )

            update_data["status"] = status

        for field, value in update_data.items():
            setattr(order, field, value)

        updated_order = await self.repository.update(
            order
        )

        await self.db.commit()

        return updated_order

    async def delete_order(
        self,
        order_id: int,
    ) -> None:

        order = await self.get_order(order_id)

        await self.repository.delete(order)

        await self.db.commit()
```

### This separation is important

```text
Router
  |
  | HTTP concerns
  v
Service
  |
  | Business logic
  v
Repository
  |
  | Database queries
  v
Database
```

---

# 11. Dependency injection for the service

Add this to `app/api/deps.py`:

```python
from fastapi import Depends
from sqlalchemy.ext.asyncio import AsyncSession

from app.services.order_service import OrderService


def get_order_service(
    db: AsyncSession = Depends(get_db),
) -> OrderService:

    return OrderService(db)
```

Now FastAPI automatically builds:

```text
Request
   |
   v
get_db()
   |
   v
AsyncSession
   |
   v
get_order_service()
   |
   v
OrderService(db)
   |
   v
Router
```

---

# 12. API router

## `app/api/v1/orders.py`

```python
from fastapi import (
    APIRouter,
    Depends,
    HTTPException,
    Query,
    status,
)

from app.api.deps import get_order_service

from app.core.exceptions import (
    InvalidOrderStatusError,
    OrderNotFoundError,
)

from app.schemas.order import (
    OrderCreate,
    OrderResponse,
    OrderUpdate,
)

from app.services.order_service import OrderService


router = APIRouter(
    prefix="/orders",
    tags=["Orders"],
)


@router.post(
    "",
    response_model=OrderResponse,
    status_code=status.HTTP_201_CREATED,
)
async def create_order(
    payload: OrderCreate,
    service: OrderService = Depends(
        get_order_service
    ),
):

    return await service.create_order(payload)


@router.get(
    "/{order_id}",
    response_model=OrderResponse,
)
async def get_order(
    order_id: int,
    service: OrderService = Depends(
        get_order_service
    ),
):

    try:
        return await service.get_order(order_id)

    except OrderNotFoundError as error:
        raise HTTPException(
            status_code=404,
            detail=str(error),
        )


@router.get(
    "",
    response_model=list[OrderResponse],
)
async def list_orders(
    offset: int = Query(
        default=0,
        ge=0,
    ),

    limit: int = Query(
        default=20,
        ge=1,
        le=100,
    ),

    service: OrderService = Depends(
        get_order_service
    ),
):

    return await service.list_orders(
        offset=offset,
        limit=limit,
    )


@router.patch(
    "/{order_id}",
    response_model=OrderResponse,
)
async def update_order(
    order_id: int,
    payload: OrderUpdate,
    service: OrderService = Depends(
        get_order_service
    ),
):

    try:

        return await service.update_order(
            order_id,
            payload,
        )

    except OrderNotFoundError as error:

        raise HTTPException(
            status_code=404,
            detail=str(error),
        )

    except InvalidOrderStatusError as error:

        raise HTTPException(
            status_code=400,
            detail=str(error),
        )


@router.delete(
    "/{order_id}",
    status_code=status.HTTP_204_NO_CONTENT,
)
async def delete_order(
    order_id: int,
    service: OrderService = Depends(
        get_order_service
    ),
):

    try:

        await service.delete_order(
            order_id
        )

    except OrderNotFoundError as error:

        raise HTTPException(
            status_code=404,
            detail=str(error),
        )
```

---

# 13. Better approach: centralized exception handling

Instead of writing this repeatedly:

```python
try:
    ...
except OrderNotFoundError:
    ...
```

handle it globally.

## `app/core/exception_handlers.py`

```python
from fastapi import Request
from fastapi.responses import JSONResponse

from app.core.exceptions import (
    InvalidOrderStatusError,
    OrderNotFoundError,
)


async def order_not_found_handler(
    request: Request,
    exc: OrderNotFoundError,
):

    return JSONResponse(
        status_code=404,
        content={
            "error": "ORDER_NOT_FOUND",
            "message": str(exc),
        },
    )


async def invalid_status_handler(
    request: Request,
    exc: InvalidOrderStatusError,
):

    return JSONResponse(
        status_code=400,
        content={
            "error": "INVALID_ORDER_STATUS",
            "message": str(exc),
        },
    )
```

Now the router becomes much cleaner:

```python
@router.get(
    "/{order_id}",
    response_model=OrderResponse,
)
async def get_order(
    order_id: int,
    service: OrderService = Depends(
        get_order_service
    ),
):
    return await service.get_order(order_id)
```

This is much closer to how larger production applications are structured.

---

# 14. Main application

## `app/main.py`

```python
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.api.v1.orders import router as orders_router

from app.core.config import get_settings

from app.core.exception_handlers import (
    invalid_status_handler,
    order_not_found_handler,
)

from app.core.exceptions import (
    InvalidOrderStatusError,
    OrderNotFoundError,
)


settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI):

    # Startup
    print("Application starting")

    yield

    # Shutdown
    print("Application shutting down")


app = FastAPI(
    title=settings.APP_NAME,
    version="1.0.0",
    lifespan=lifespan,
)


# Exception handlers

app.add_exception_handler(
    OrderNotFoundError,
    order_not_found_handler,
)

app.add_exception_handler(
    InvalidOrderStatusError,
    invalid_status_handler,
)


# Routes

app.include_router(
    orders_router,
    prefix=settings.API_V1_PREFIX,
)


@app.get("/health")
async def health_check():

    return {
        "status": "healthy",
    }
```

---

# 15. Add versioned routers

## `app/api/v1/router.py`

```python
from fastapi import APIRouter

from app.api.v1.orders import router as orders_router


api_router = APIRouter()

api_router.include_router(
    orders_router
)
```

Then `main.py`:

```python
from app.api.v1.router import api_router


app.include_router(
    api_router,
    prefix="/api/v1",
)
```

Your APIs become:

```text
POST   /api/v1/orders
GET    /api/v1/orders
GET    /api/v1/orders/1
PATCH  /api/v1/orders/1
DELETE /api/v1/orders/1
```

Later, if you introduce breaking changes:

```text
/api/v1/orders
/api/v2/orders
```

This is a good production practice.

---

# 16. Logging middleware

## `app/core/middleware.py`

```python
import logging
import time
import uuid

from fastapi import Request


logger = logging.getLogger(__name__)


async def request_logging_middleware(
    request: Request,
    call_next,
):

    request_id = str(uuid.uuid4())

    start_time = time.perf_counter()

    response = await call_next(request)

    duration = (
        time.perf_counter()
        - start_time
    )

    response.headers["X-Request-ID"] = (
        request_id
    )

    logger.info(
        "request_completed",
        extra={
            "request_id": request_id,
            "method": request.method,
            "path": request.url.path,
            "status_code": response.status_code,
            "duration_ms": round(
                duration * 1000,
                2,
            ),
        },
    )

    return response
```

Register:

```python
from app.core.middleware import (
    request_logging_middleware
)

app.middleware("http")(
    request_logging_middleware
)
```

Now your logs can look like:

```text
request_completed

request_id=abc-123
method=POST
path=/api/v1/orders
status_code=201
duration_ms=42.15
```

This request ID becomes very useful when debugging production issues.

---

# 17. Structured JSON logging

## `app/core/logging.py`

```python
import logging

from pythonjsonlogger.json import JsonFormatter


def configure_logging():

    handler = logging.StreamHandler()

    formatter = JsonFormatter(
        "%(asctime)s %(levelname)s %(name)s %(message)s"
    )

    handler.setFormatter(formatter)

    root_logger = logging.getLogger()

    root_logger.setLevel(logging.INFO)

    root_logger.handlers.clear()

    root_logger.addHandler(handler)
```

In `main.py`:

```python
from app.core.logging import configure_logging


configure_logging()
```

Production logs:

```json
{
  "asctime": "2026-08-10T10:00:00",
  "levelname": "INFO",
  "message": "request_completed",
  "request_id": "abc123",
  "status_code": 200
}
```

This is easier to search in systems such as:

* [Grafana](https://grafana.com/?utm_source=chatgpt.com)
* [Datadog](https://www.datadoghq.com/?utm_source=chatgpt.com)
* [Elastic](https://www.elastic.co/?utm_source=chatgpt.com)

---

# 18. Authentication pattern

For production, authentication should also use dependency injection.

## `app/api/deps.py`

```python
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials
from fastapi.security import HTTPBearer


security = HTTPBearer()


async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(
        security
    ),
):

    token = credentials.credentials

    # Example:
    # Decode JWT
    # Verify signature
    # Verify expiry
    # Load user

    if token != "demo-token":

        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication credentials",
        )

    return {
        "user_id": "123",
        "role": "admin",
    }
```

Usage:

```python
@router.get("/secure")
async def secure_endpoint(
    current_user: dict = Depends(
        get_current_user
    ),
):

    return {
        "message": "Authenticated",
        "user": current_user,
    }
```

The request flow:

```text
Request
  |
  v
Authentication Dependency
  |
  +--> Invalid Token --> 401
  |
  v
Authenticated User
  |
  v
API Endpoint
```

In a real project, I would use a proper JWT implementation rather than comparing the token directly.

---

# 19. Pagination response

For larger datasets, don't just return a list.

Instead:

```python
class PaginatedOrders(BaseModel):

    items: list[OrderResponse]

    offset: int

    limit: int

    total: int
```

Response:

```json
{
  "items": [
    {
      "id": 1,
      "customer_name": "Sandeep",
      "product_name": "Laptop",
      "amount": 50000,
      "status": "PENDING"
    }
  ],
  "offset": 0,
  "limit": 20,
  "total": 1250
}
```

For very large data, prefer **cursor-based pagination** over offset pagination.

---

# 20. Health checks

Real production systems usually need more than:

```python
{"status": "healthy"}
```

Example:

```python
from sqlalchemy import text

from app.db.session import engine


@app.get("/health/live")
async def liveness():

    return {
        "status": "alive"
    }


@app.get("/health/ready")
async def readiness():

    try:

        async with engine.connect() as connection:

            await connection.execute(
                text("SELECT 1")
            )

        return {
            "status": "ready"
        }

    except Exception:

        return JSONResponse(
            status_code=503,
            content={
                "status": "not_ready"
            },
        )
```

Difference:

```text
Liveness
   |
   --> Is the application process alive?

Readiness
   |
   --> Is the application ready to serve traffic?
       |
       +--> Database available?
       +--> Redis available?
       +--> Required dependencies available?
```

This is especially important when deploying with Kubernetes.

---

# 21. Testing

## `app/tests/test_orders.py`

A service-layer unit test:

```python
import pytest
from unittest.mock import AsyncMock

from app.services.order_service import OrderService
from app.schemas.order import OrderCreate


@pytest.mark.asyncio
async def test_create_order():

    mock_db = AsyncMock()

    service = OrderService(
        db=mock_db
    )

    service.repository.create = AsyncMock()

    payload = OrderCreate(
        customer_name="Sandeep",
        product_name="Laptop",
        amount=50000,
    )

    await service.create_order(
        payload
    )

    service.repository.create.assert_called_once()

    mock_db.commit.assert_called_once()
```

### Why test services separately?

```text
Unit Test
   |
   +--> Test business logic
   +--> Mock database
   +--> Fast
```

Then:

```text
Integration Test
   |
   +--> Real FastAPI
   +--> Test database
   +--> Actual HTTP request
```

And finally:

```text
End-to-End Test
   |
   +--> Entire deployed system
```

---

# 22. Dockerfile

```dockerfile
FROM python:3.12-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

COPY requirements.txt .

RUN pip install --no-cache-dir \
    -r requirements.txt

COPY . .

EXPOSE 8000

CMD [
    "uvicorn",
    "app.main:app",
    "--host",
    "0.0.0.0",
    "--port",
    "8000",
    "--workers",
    "4"
]
```

---

# 23. Docker Compose

```yaml
services:

  api:

    build: .

    ports:
      - "8000:8000"

    environment:
      DATABASE_URL: postgresql+asyncpg://postgres:postgres@db:5432/orders_db

    depends_on:
      db:
        condition: service_healthy


  db:

    image: postgres:16

    environment:
      POSTGRES_DB: orders_db
      POSTGRES_USER: postgres
      POSTGRES_PASSWORD: postgres

    ports:
      - "5432:5432"

    healthcheck:
      test:
        - CMD-SHELL
        - pg_isready -U postgres

      interval: 10s
      timeout: 5s
      retries: 5
```

Run:

```bash
docker compose up --build
```

---

# 24. How the complete request flows

Suppose the client sends:

```http
POST /api/v1/orders
```

Body:

```json
{
  "customer_name": "Sandeep",
  "product_name": "Laptop",
  "amount": 50000
}
```

The complete flow is:

```text
                     HTTP Request
                          |
                          v
                  Logging Middleware
                          |
                          v
                    API Router
                          |
                          v
                  Pydantic Validation
                          |
                          v
                  Dependency Injection
                          |
              +-----------+-----------+
              |                       |
              v                       v
        Database Session        Order Service
                                      |
                                      v
                              Business Validation
                                      |
                                      v
                                Repository
                                      |
                                      v
                                 PostgreSQL
                                      |
                                      v
                                 Commit Data
                                      |
                                      v
                              Response Schema
                                      |
                                      v
                                JSON Response
```

---

# 25. What I would use for your AI/RAG project

Since you're building enterprise AI/RAG systems, I would structure FastAPI more like this:

```text
app/
│
├── main.py
│
├── api/
│   ├── deps.py
│   └── v1/
│       ├── chat.py
│       ├── documents.py
│       ├── agents.py
│       ├── auth.py
│       └── health.py
│
├── core/
│   ├── config.py
│   ├── logging.py
│   ├── security.py
│   └── exceptions.py
│
├── db/
│   ├── postgres.py
│   ├── redis.py
│   └── qdrant.py
│
├── schemas/
│   ├── chat.py
│   ├── document.py
│   └── agent.py
│
├── repositories/
│   ├── conversation_repository.py
│   └── document_repository.py
│
├── services/
│   ├── chat_service.py
│   ├── rag_service.py
│   ├── ingestion_service.py
│   └── agent_service.py
│
├── agents/
│   ├── supervisor.py
│   ├── retrieval_agent.py
│   ├── finance_agent.py
│   └── compliance_agent.py
│
├── llm/
│   ├── factory.py
│   ├── openai_client.py
│   └── fallback.py
│
├── middleware/
│   ├── tenant.py
│   ├── request_id.py
│   └── rate_limit.py
│
└── observability/
    ├── metrics.py
    └── tracing.py
```

For example:

```text
POST /api/v1/chat
       |
       v
FastAPI Router
       |
       v
Tenant Middleware
       |
       v
Authentication
       |
       v
Chat Service
       |
       +--> Redis Cache
       |
       +--> RAG Service
       |       |
       |       +--> Qdrant
       |       +--> Reranker
       |
       +--> LLM Factory
               |
               +--> GPT
               +--> Claude
               +--> Local LLM
       |
       v
PostgreSQL Conversation Storage
       |
       v
Streaming Response
```

## Key takeaway

A production FastAPI application should generally follow:

```text
Router
  ↓
Dependencies
  ↓
Service
  ↓
Repository
  ↓
Database / External Services
```

And around it:

```text
Middleware
Authentication
Validation
Exception Handling
Logging
Metrics
Tracing
Health Checks
Tests
Docker
CI/CD
```

This is the structure I would recommend for your **enterprise multi-agent RAG platform**, because it keeps the API layer thin while allowing your RAG, agent, LLM, PostgreSQL, Qdrant, and Redis components to scale independently.


------------------
------------------
------------------
------------------

Below is a **real-world FastAPI project structure**—not just a single `main.py`. This is the kind of structure you can use for a backend, AI service, RAG API, SaaS application, or microservice.

I'll build a **production-style User API** with:

* FastAPI
* Async SQLAlchemy
* PostgreSQL
* Pydantic validation
* Dependency Injection
* Repository pattern
* Service layer
* JWT-ready security structure
* Environment configuration
* Centralized exception handling
* Logging
* Health checks
* Docker
* Tests

---

# 1. Project Structure

```text
fastapi-production/
│
├── app/
│   ├── main.py
│   │
│   ├── api/
│   │   ├── deps.py
│   │   └── v1/
│   │       ├── router.py
│   │       └── users.py
│   │
│   ├── core/
│   │   ├── config.py
│   │   ├── exceptions.py
│   │   ├── logging.py
│   │   └── security.py
│   │
│   ├── db/
│   │   ├── base.py
│   │   └── session.py
│   │
│   ├── models/
│   │   └── user.py
│   │
│   ├── schemas/
│   │   └── user.py
│   │
│   ├── repositories/
│   │   └── user_repository.py
│   │
│   ├── services/
│   │   └── user_service.py
│   │
│   └── middleware/
│       └── request_id.py
│
├── tests/
│   └── test_users.py
│
├── .env
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── README.md
```

---

# 2. Install Dependencies

```txt
fastapi
uvicorn[standard]
sqlalchemy
asyncpg
pydantic-settings
python-dotenv
email-validator
passlib[bcrypt]
python-jose[cryptography]
httpx
pytest
pytest-asyncio
```

Install:

```bash
pip install -r requirements.txt
```

---

# 3. Configuration

## `app/core/config.py`

Never hardcode database URLs, API keys, JWT secrets, or environment-specific settings.

```python
from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    APP_NAME: str = "Production FastAPI"
    ENVIRONMENT: str = "development"

    DATABASE_URL: str

    JWT_SECRET_KEY: str = "change-me"
    JWT_ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
```

---

## `.env`

```env
APP_NAME=Production FastAPI
ENVIRONMENT=development

DATABASE_URL=postgresql+asyncpg://postgres:postgres@localhost:5432/fastapi_db

JWT_SECRET_KEY=super-secret-key
JWT_ALGORITHM=HS256
ACCESS_TOKEN_EXPIRE_MINUTES=30
```

In production, secrets should normally come from:

* AWS Secrets Manager
* Azure Key Vault
* HashiCorp Vault
* Kubernetes Secrets
* GitHub Actions Secrets

Do not commit `.env` files containing real credentials.

---

# 4. Database Configuration

## `app/db/base.py`

```python
from sqlalchemy.orm import DeclarativeBase


class Base(DeclarativeBase):
    pass
```

---

## `app/db/session.py`

```python
from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from app.core.config import settings


engine = create_async_engine(
    settings.DATABASE_URL,
    echo=False,
    pool_pre_ping=True,
    pool_size=10,
    max_overflow=20,
)


AsyncSessionLocal = async_sessionmaker(
    bind=engine,
    class_=AsyncSession,
    expire_on_commit=False,
)
```

Why this matters:

```text
Client
   |
   v
FastAPI
   |
   v
Dependency
   |
   v
AsyncSession
   |
   v
SQLAlchemy Pool
   |
   v
PostgreSQL
```

`pool_pre_ping=True` checks whether database connections are still valid before reusing them.

---

# 5. Database Dependency

## `app/api/deps.py`

```python
from collections.abc import AsyncGenerator

from app.db.session import AsyncSessionLocal


async def get_db() -> AsyncGenerator:
    async with AsyncSessionLocal() as session:
        try:
            yield session
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()
```

Usage:

```python
from fastapi import Depends
from sqlalchemy.ext.asyncio import AsyncSession


async def endpoint(
    db: AsyncSession = Depends(get_db),
):
    ...
```

FastAPI automatically handles the lifecycle:

```text
Request starts
     ↓
Create DB session
     ↓
Inject session
     ↓
Execute endpoint
     ↓
Commit/Rollback logic
     ↓
Close session
```

---

# 6. Database Model

## `app/models/user.py`

```python
from datetime import datetime

from sqlalchemy import Boolean, DateTime, String
from sqlalchemy.orm import Mapped, mapped_column
from sqlalchemy.sql import func

from app.db.base import Base


class User(Base):
    __tablename__ = "users"

    id: Mapped[int] = mapped_column(
        primary_key=True,
        index=True,
    )

    email: Mapped[str] = mapped_column(
        String(255),
        unique=True,
        index=True,
        nullable=False,
    )

    full_name: Mapped[str] = mapped_column(
        String(255),
        nullable=False,
    )

    hashed_password: Mapped[str] = mapped_column(
        String(255),
        nullable=False,
    )

    is_active: Mapped[bool] = mapped_column(
        Boolean,
        default=True,
        nullable=False,
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )
```

Important: `Mapped[...]` and `mapped_column()` are modern SQLAlchemy patterns.

---

# 7. Pydantic Request and Response Schemas

## `app/schemas/user.py`

```python
from datetime import datetime

from pydantic import BaseModel, ConfigDict, EmailStr, Field


class UserCreate(BaseModel):
    email: EmailStr

    full_name: str = Field(
        min_length=2,
        max_length=255,
    )

    password: str = Field(
        min_length=8,
        max_length=128,
    )


class UserResponse(BaseModel):
    id: int
    email: EmailStr
    full_name: str
    is_active: bool
    created_at: datetime

    model_config = ConfigDict(
        from_attributes=True,
    )
```

Request:

```json
{
  "email": "user@example.com",
  "full_name": "John Doe",
  "password": "StrongPassword123"
}
```

Response:

```json
{
  "id": 1,
  "email": "user@example.com",
  "full_name": "John Doe",
  "is_active": true,
  "created_at": "2026-08-10T10:00:00Z"
}
```

Notice that `hashed_password` is never returned.

---

# 8. Repository Layer

The repository should only handle database operations.

## `app/repositories/user_repository.py`

```python
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.user import User


class UserRepository:

    def __init__(
        self,
        db: AsyncSession,
    ):
        self.db = db

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
        *,
        email: str,
        full_name: str,
        hashed_password: str,
    ) -> User:

        user = User(
            email=email,
            full_name=full_name,
            hashed_password=hashed_password,
        )

        self.db.add(user)

        await self.db.flush()

        return user
```

Repository responsibility:

```text
Repository
    |
    ├── SELECT
    ├── INSERT
    ├── UPDATE
    └── DELETE
```

Business logic should not be here.

Bad:

```python
if user.email.endswith("@company.com"):
    give_bonus()
```

That belongs in the service layer.

---

# 9. Security

## `app/core/security.py`

```python
from passlib.context import CryptContext


password_context = CryptContext(
    schemes=["bcrypt"],
    deprecated="auto",
)


def hash_password(
    password: str,
) -> str:

    return password_context.hash(password)


def verify_password(
    plain_password: str,
    hashed_password: str,
) -> bool:

    return password_context.verify(
        plain_password,
        hashed_password,
    )
```

Never do this:

```python
hashed_password = password
```

Never do this:

```python
hash(password)
```

Python's built-in `hash()` is not suitable for password storage.

---

# 10. Custom Exceptions

## `app/core/exceptions.py`

```python
class UserAlreadyExistsError(Exception):
    pass


class UserNotFoundError(Exception):
    pass
```

These are domain/business exceptions.

They should not know anything about HTTP.

This is important because your business layer should be reusable outside FastAPI.

For example:

```text
FastAPI
CLI
Background Worker
Kafka Consumer
Celery Worker
```

All can use the same service layer.

---

# 11. Service Layer

## `app/services/user_service.py`

```python
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.exceptions import (
    UserAlreadyExistsError,
    UserNotFoundError,
)
from app.core.security import hash_password
from app.repositories.user_repository import UserRepository
from app.schemas.user import UserCreate


class UserService:

    def __init__(
        self,
        db: AsyncSession,
    ):
        self.db = db
        self.repository = UserRepository(db)

    async def create_user(
        self,
        user_data: UserCreate,
    ):

        existing_user = (
            await self.repository.get_by_email(
                user_data.email
            )
        )

        if existing_user:
            raise UserAlreadyExistsError(
                "User already exists"
            )

        hashed_password = hash_password(
            user_data.password
        )

        user = await self.repository.create(
            email=user_data.email,
            full_name=user_data.full_name,
            hashed_password=hashed_password,
        )

        await self.db.commit()

        await self.db.refresh(user)

        return user

    async def get_user(
        self,
        user_id: int,
    ):

        user = await self.repository.get_by_id(
            user_id
        )

        if not user:
            raise UserNotFoundError(
                "User not found"
            )

        return user
```

The flow is:

```text
API Router
    ↓
Service
    ↓
Repository
    ↓
Database
```

Example:

```text
POST /users
       ↓
UserRouter
       ↓
UserService
       ↓
Validate business rules
       ↓
Hash password
       ↓
UserRepository
       ↓
PostgreSQL
```

---

# 12. Dependency Injection for Service

Instead of manually creating services everywhere, use FastAPI dependencies.

## `app/api/deps.py`

Add:

```python
from fastapi import Depends
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.session import AsyncSessionLocal
from app.services.user_service import UserService


async def get_db():
    async with AsyncSessionLocal() as session:
        try:
            yield session
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()


def get_user_service(
    db: AsyncSession = Depends(get_db),
) -> UserService:

    return UserService(db)
```

Now FastAPI constructs the dependency chain:

```text
Endpoint
    ↓
UserService
    ↓
Database Session
```

---

# 13. API Router

## `app/api/v1/users.py`

```python
from fastapi import (
    APIRouter,
    Depends,
    HTTPException,
    status,
)

from app.api.deps import get_user_service
from app.core.exceptions import (
    UserAlreadyExistsError,
    UserNotFoundError,
)
from app.schemas.user import (
    UserCreate,
    UserResponse,
)
from app.services.user_service import UserService


router = APIRouter(
    prefix="/users",
    tags=["Users"],
)


@router.post(
    "",
    response_model=UserResponse,
    status_code=status.HTTP_201_CREATED,
)
async def create_user(
    payload: UserCreate,
    service: UserService = Depends(
        get_user_service
    ),
):

    try:
        return await service.create_user(
            payload
        )

    except UserAlreadyExistsError as exc:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=str(exc),
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

    try:
        return await service.get_user(
            user_id
        )

    except UserNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(exc),
        )
```

A router should be thin.

Avoid:

```python
@router.post("/users")
async def create_user():
    # 300 lines
    # business logic
    # database logic
    # API calls
    # password hashing
```

Instead:

```text
Router
  ↓
Service
  ↓
Repository
```

---

# 14. API Version Router

## `app/api/v1/router.py`

```python
from fastapi import APIRouter

from app.api.v1.users import router as users_router


api_router = APIRouter()

api_router.include_router(
    users_router
)
```

This allows API versioning:

```text
/api/v1/users
/api/v2/users
```

Later:

```python
app.include_router(
    api_v1_router,
    prefix="/api/v1",
)

app.include_router(
    api_v2_router,
    prefix="/api/v2",
)
```

This is much safer than breaking existing clients.

---

# 15. Centralized Exception Handling

Instead of handling exceptions in every endpoint:

```python
try:
    ...
except UserNotFoundError:
    ...
```

Create global handlers.

## `app/main.py`

```python
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from app.api.v1.router import api_router
from app.core.config import settings
from app.core.exceptions import (
    UserAlreadyExistsError,
    UserNotFoundError,
)
from app.db.session import engine


@asynccontextmanager
async def lifespan(app: FastAPI):

    print("Application starting")

    yield

    await engine.dispose()

    print("Application shutting down")


app = FastAPI(
    title=settings.APP_NAME,
    version="1.0.0",
    lifespan=lifespan,
)


@app.exception_handler(
    UserAlreadyExistsError
)
async def user_exists_handler(
    request: Request,
    exc: UserAlreadyExistsError,
):

    return JSONResponse(
        status_code=409,
        content={
            "detail": str(exc),
        },
    )


@app.exception_handler(
    UserNotFoundError
)
async def user_not_found_handler(
    request: Request,
    exc: UserNotFoundError,
):

    return JSONResponse(
        status_code=404,
        content={
            "detail": str(exc),
        },
    )


app.include_router(
    api_router,
    prefix="/api/v1",
)


@app.get(
    "/health",
    tags=["Health"],
)
async def health_check():

    return {
        "status": "healthy"
    }
```

Now routers become much cleaner:

```python
@router.get("/{user_id}")
async def get_user(
    user_id: int,
    service: UserService = Depends(
        get_user_service
    ),
):
    return await service.get_user(user_id)
```

---

# 16. Request ID Middleware

In production, every request should ideally have a correlation ID.

## `app/middleware/request_id.py`

```python
import uuid

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request


class RequestIDMiddleware(
    BaseHTTPMiddleware,
):

    async def dispatch(
        self,
        request: Request,
        call_next,
    ):

        request_id = request.headers.get(
            "X-Request-ID",
            str(uuid.uuid4()),
        )

        response = await call_next(
            request
        )

        response.headers[
            "X-Request-ID"
        ] = request_id

        return response
```

Add it:

```python
from app.middleware.request_id import (
    RequestIDMiddleware
)

app.add_middleware(
    RequestIDMiddleware
)
```

This is extremely useful when debugging distributed systems:

```text
Client
X-Request-ID: abc123
      ↓
API
      ↓
PostgreSQL
      ↓
Redis
      ↓
LLM
      ↓
Qdrant
```

Logs can all contain:

```text
request_id=abc123
```

---

# 17. Logging

## `app/core/logging.py`

```python
import logging


def configure_logging():

    logging.basicConfig(
        level=logging.INFO,
        format=(
            "%(asctime)s "
            "%(levelname)s "
            "%(name)s "
            "%(message)s"
        ),
    )
```

In `main.py`:

```python
from app.core.logging import configure_logging


configure_logging()
```

Use:

```python
import logging

logger = logging.getLogger(__name__)


logger.info(
    "Creating user"
)
```

Production logs should eventually go to:

```text
Application
     ↓
Structured Logs
     ↓
CloudWatch / ELK / Loki
     ↓
Grafana
```

---

# 18. Dockerfile

```dockerfile
FROM python:3.12-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

COPY requirements.txt .

RUN pip install --no-cache-dir \
    -r requirements.txt

COPY . .

EXPOSE 8000

CMD [
    "uvicorn",
    "app.main:app",
    "--host",
    "0.0.0.0",
    "--port",
    "8000"
]
```

For production, you may tune workers depending on deployment model. When running in Kubernetes, it is often better to run a smaller number of processes per container and scale horizontally.

---

# 19. Docker Compose

## `docker-compose.yml`

```yaml
services:

  api:
    build: .
    ports:
      - "8000:8000"

    env_file:
      - .env

    depends_on:
      postgres:
        condition: service_healthy


  postgres:
    image: postgres:16

    environment:
      POSTGRES_USER: postgres
      POSTGRES_PASSWORD: postgres
      POSTGRES_DB: fastapi_db

    ports:
      - "5432:5432"

    healthcheck:
      test:
        [
          "CMD-SHELL",
          "pg_isready -U postgres"
        ]

      interval: 5s
      timeout: 5s
      retries: 5
```

Run:

```bash
docker compose up --build
```

API:

```text
http://localhost:8000
```

Swagger:

```text
http://localhost:8000/docs
```

---

# 20. Testing

## `tests/test_users.py`

For real applications, separate unit tests and integration tests.

Example endpoint test:

```python
from fastapi.testclient import TestClient

from app.main import app


client = TestClient(app)


def test_health():

    response = client.get(
        "/health"
    )

    assert response.status_code == 200

    assert response.json() == {
        "status": "healthy"
    }
```

For service tests, mock the repository/database boundary rather than requiring PostgreSQL for every unit test.

Example:

```python
import pytest
from unittest.mock import AsyncMock

from app.core.exceptions import (
    UserAlreadyExistsError
)
from app.schemas.user import UserCreate
from app.services.user_service import (
    UserService
)


@pytest.mark.asyncio
async def test_duplicate_user():

    db = AsyncMock()

    service = UserService(db)

    service.repository.get_by_email = (
        AsyncMock(
            return_value=object()
        )
    )

    payload = UserCreate(
        email="test@example.com",
        full_name="Test User",
        password="Password123",
    )

    with pytest.raises(
        UserAlreadyExistsError
    ):
        await service.create_user(
            payload
        )
```

---

# 21. How a Real Request Flows

Suppose the client calls:

```http
POST /api/v1/users
```

With:

```json
{
  "email": "john@example.com",
  "full_name": "John Doe",
  "password": "Password123"
}
```

The execution flow is:

```text
Client
   |
   v
FastAPI Middleware
   |
   v
Request ID Middleware
   |
   v
Router
   |
   v
Pydantic Validation
   |
   v
Dependency Injection
   |
   v
UserService
   |
   v
UserRepository
   |
   v
SQLAlchemy AsyncSession
   |
   v
PostgreSQL
```

Then:

```text
PostgreSQL
   |
   v
Repository returns User
   |
   v
Service commits transaction
   |
   v
Router returns User
   |
   v
Pydantic filters response
   |
   v
JSON Response
```

---

# 22. The Most Important Architecture Principle

For a real project:

```text
                 HTTP
                  │
                  ▼
              API Router
                  │
                  ▼
              Service Layer
             Business Logic
                  │
                  ▼
             Repository Layer
              Data Access
                  │
          ┌───────┴───────┐
          ▼               ▼
      PostgreSQL        Redis
```

For an AI application, it might become:

```text
                    FastAPI
                       │
                       ▼
                    Router
                       │
                       ▼
                  Agent Service
                       │
          ┌────────────┼────────────┐
          ▼            ▼            ▼
       LLM Client   Retriever      Memory
          │            │            │
          ▼            ▼            ▼
       OpenAI       Qdrant        Redis
```

---

# 23. What I Would Add for a Senior/Staff-Level Production API

For a serious enterprise project, I would also add:

```text
FastAPI
│
├── Authentication
│   ├── JWT
│   ├── OAuth2
│   └── RBAC
│
├── Security
│   ├── Rate limiting
│   ├── CORS
│   ├── Input validation
│   └── Secret management
│
├── Reliability
│   ├── Retry
│   ├── Timeout
│   ├── Circuit breaker
│   └── Idempotency
│
├── Observability
│   ├── Structured logging
│   ├── Prometheus
│   ├── OpenTelemetry
│   └── Grafana
│
├── Data
│   ├── PostgreSQL
│   ├── Alembic migrations
│   └── Redis caching
│
├── Quality
│   ├── Pytest
│   ├── Integration tests
│   ├── Ruff
│   ├── mypy
│   └── Pre-commit
│
└── Deployment
    ├── Docker
    ├── Kubernetes
    ├── CI/CD
    ├── Health checks
    └── Autoscaling
```

## My recommendation

For your **Enterprise Multi-Agent / RAG projects**, I would use this FastAPI architecture:

```text
app/
├── main.py
├── api/
│   ├── deps.py
│   └── v1/
│       ├── chat.py
│       ├── documents.py
│       ├── auth.py
│       └── admin.py
│
├── core/
│   ├── config.py
│   ├── security.py
│   ├── exceptions.py
│   └── logging.py
│
├── db/
│   ├── session.py
│   └── models/
│
├── services/
│   ├── chat_service.py
│   ├── ingestion_service.py
│   └── evaluation_service.py
│
├── agents/
│   ├── supervisor.py
│   ├── retrieval_agent.py
│   ├── finance_agent.py
│   └── research_agent.py
│
├── rag/
│   ├── retriever.py
│   ├── reranker.py
│   ├── chunker.py
│   └── embedder.py
│
├── repositories/
│   ├── user_repository.py
│   ├── conversation_repository.py
│   └── document_repository.py
│
├── integrations/
│   ├── llm/
│   ├── qdrant/
│   ├── redis/
│   └── postgres/
│
└── tests/
```

The next logical step is to build a **complete production FastAPI backend from scratch**, including **JWT authentication, PostgreSQL, Alembic migrations, Redis caching, async patterns, Docker, pytest, logging, middleware, rate limiting, and a clean architecture suitable for your RAG/AI system**.
