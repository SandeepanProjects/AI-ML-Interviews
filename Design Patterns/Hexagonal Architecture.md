Yes. **Hexagonal Architecture** is one of the most important architectures to understand if you want to explain your AI projects at a Senior/Staff Engineer level.

The easiest way to understand it is:

> **Business logic stays in the center. Everything external stays outside. The center communicates with the outside world through interfaces called ports, and implementations of those interfaces are adapters.**

It is also called **Ports and Adapters Architecture**.

---

# 1. The problem Hexagonal Architecture solves

Imagine you build a financial compliance application.

A simple implementation might look like this:

```text
FastAPI
   ↓
ReviewService
   ↓
OpenAI
   ↓
PostgreSQL
   ↓
Redis
```

And your service might directly contain:

```python
from openai import AsyncOpenAI
from sqlalchemy.ext.asyncio import AsyncSession

class ReviewService:

    def __init__(self, db: AsyncSession):
        self.db = db
        self.client = AsyncOpenAI()

    async def review(self, material: str):

        response = await self.client.chat.completions.create(
            model="gpt-5",
            messages=[
                {"role": "user", "content": material}
            ]
        )

        # save directly to database
        ...
```

This works.

But now imagine:

* OpenAI → AWS Bedrock
* PostgreSQL → another database
* FastAPI → CLI
* Redis → another cache
* OpenAI → local model
* real LLM → mock LLM for testing

Your business service becomes tightly coupled to infrastructure.

That's the problem Hexagonal Architecture addresses.

---

# 2. The Hexagonal idea

Instead of:

```text
Business Logic
     ↓
OpenAI
PostgreSQL
Redis
FastAPI
```

we create:

```text
                FastAPI
                   │
                   ▼
            ┌─────────────┐
            │   PORT      │
            │ ReviewUseCase│
            └──────┬──────┘
                   │
        ┌──────────▼──────────┐
        │                     │
        │    DOMAIN / CORE    │
        │                     │
        │ Compliance Rules    │
        │ Review Logic        │
        │ Domain Models       │
        │                     │
        └──────────┬──────────┘
                   │
          ┌────────┴────────┐
          ▼                 ▼
     LLM Port          Repository Port
          │                 │
          ▼                 ▼
   OpenAI Adapter     PostgreSQL Adapter
```

The key principle is:

> **The core does not know that OpenAI, PostgreSQL, FastAPI, Redis, or Celery exist.**

---

# 3. What is a Port?

A **port is an interface/contract**.

For example, your application needs an LLM.

Instead of saying:

```python
from openai import AsyncOpenAI
```

inside your business logic, define what the application needs:

```python
from typing import Protocol


class LLMPort(Protocol):

    async def generate(
        self,
        prompt: str
    ) -> str:
        ...
```

That's your port.

It says:

> "I don't care who provides the LLM. I only need something capable of generating text."

---

# 4. What is an Adapter?

An adapter implements that port.

For example:

```python
from openai import AsyncOpenAI


class OpenAIAdapter:

    def __init__(self):
        self.client = AsyncOpenAI()

    async def generate(self, prompt: str) -> str:

        response = await self.client.chat.completions.create(
            model="gpt-5",
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        )

        return response.choices[0].message.content
```

The important part is:

```text
LLMPort
   ↑
   │ implements
   │
OpenAIAdapter
```

Your business logic talks to:

```python
LLMPort
```

not:

```python
OpenAIAdapter
```

---

# 5. Complete example

Let's build a small financial compliance application.

Suppose the requirement is:

> Given financial marketing material, determine whether it contains risky claims.

For example:

```text
"This investment is guaranteed and completely risk-free."
```

We want:

```text
HIGH RISK
```

---

# 6. Project structure

A Hexagonal implementation could look like:

```text
financial_compliance/
│
├── domain/
│   ├── models.py
│   └── rules.py
│
├── application/
│   └── services/
│       └── review_service.py
│
├── ports/
│   ├── llm.py
│   └── repository.py
│
├── adapters/
│   ├── llm/
│   │   └── openai_adapter.py
│   │
│   ├── persistence/
│   │   └── postgres_repository.py
│   │
│   └── api/
│       └── review_router.py
│
└── main.py
```

Now let's understand each layer.

---

# 7. Domain layer

The **domain is the heart of the application**.

It should contain business concepts.

### `domain/models.py`

```python
from dataclasses import dataclass
from enum import Enum


class RiskLevel(str, Enum):

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


@dataclass
class ComplianceReview:

    material: str
    risk: RiskLevel
    explanation: str
```

Notice something important.

There is no:

```python
FastAPI
OpenAI
SQLAlchemy
Redis
LangChain
PostgreSQL
```

inside the domain.

That's intentional.

---

# 8. Domain rules

Suppose we have a business rule:

> "Guaranteed returns" are considered high risk.

### `domain/rules.py`

```python
from .models import RiskLevel


def classify_risk(material: str) -> RiskLevel:

    text = material.lower()

    if "guaranteed" in text:
        return RiskLevel.HIGH

    if "risk-free" in text:
        return RiskLevel.HIGH

    return RiskLevel.LOW
```

This is pure business logic.

It doesn't care where the input came from.

---

# 9. LLM Port

Now the application needs an LLM.

Create:

### `ports/llm.py`

```python
from typing import Protocol


class LLMPort(Protocol):

    async def generate(
        self,
        prompt: str
    ) -> str:
        ...
```

This is an outbound port.

Why?

Because the application is requesting something from the outside world.

```text
Application
     │
     ▼
  LLMPort
     │
     ▼
External LLM
```

---

# 10. Repository Port

Suppose we need to save the review.

Create:

### `ports/repository.py`

```python
from typing import Protocol

from domain.models import ComplianceReview


class ReviewRepository(Protocol):

    async def save(
        self,
        review: ComplianceReview
    ) -> None:
        ...
```

Again, the application doesn't know whether the database is:

```text
PostgreSQL
MongoDB
DynamoDB
Cosmos DB
```

It only knows:

```python
ReviewRepository
```

---

# 11. Application service

Now we can build the actual use case.

### `application/services/review_service.py`

```python
from domain.models import ComplianceReview
from domain.rules import classify_risk

from ports.llm import LLMPort
from ports.repository import ReviewRepository


class ReviewService:

    def __init__(
        self,
        llm: LLMPort,
        repository: ReviewRepository
    ):
        self.llm = llm
        self.repository = repository

    async def review(
        self,
        material: str
    ) -> ComplianceReview:

        risk = classify_risk(material)

        prompt = f"""
        Analyze this financial material.

        Material:
        {material}

        Risk level:
        {risk.value}

        Explain why this material may be problematic.
        """

        explanation = await self.llm.generate(prompt)

        review = ComplianceReview(
            material=material,
            risk=risk,
            explanation=explanation
        )

        await self.repository.save(review)

        return review
```

This is the important part.

The service knows:

```text
LLMPort
ReviewRepository
Domain
```

But it does NOT know:

```text
OpenAI
PostgreSQL
FastAPI
Redis
```

That's Hexagonal Architecture.

---

# 12. OpenAI adapter

Now we implement the port.

### `adapters/llm/openai_adapter.py`

```python
from openai import AsyncOpenAI

from ports.llm import LLMPort


class OpenAIAdapter:

    def __init__(self):

        self.client = AsyncOpenAI()

    async def generate(
        self,
        prompt: str
    ) -> str:

        response = await self.client.chat.completions.create(
            model="gpt-5",
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        )

        return response.choices[0].message.content
```

Conceptually:

```text
             LLMPort
                ▲
                │
         implements
                │
        OpenAIAdapter
```

---

# 13. PostgreSQL adapter

Now implement the repository.

### `adapters/persistence/postgres_repository.py`

Simplified example:

```python
from ports.repository import ReviewRepository
from domain.models import ComplianceReview


class PostgresReviewRepository:

    def __init__(self, session):
        self.session = session

    async def save(
        self,
        review: ComplianceReview
    ) -> None:

        record = ReviewModel(
            material=review.material,
            risk=review.risk.value,
            explanation=review.explanation
        )

        self.session.add(record)

        await self.session.commit()
```

Again:

```text
ReviewRepository
       ▲
       │
       │ implements
       │
PostgresReviewRepository
```

The application doesn't know PostgreSQL exists.

---

# 14. FastAPI adapter

FastAPI is also an adapter.

This is an extremely important point.

Many developers think:

```text
FastAPI = architecture
```

It isn't.

FastAPI is an **inbound adapter**.

Why?

Because it converts HTTP requests into application calls.

### `adapters/api/review_router.py`

```python
from fastapi import APIRouter, Depends

from application.services.review_service import ReviewService


router = APIRouter()


@router.post("/reviews")
async def create_review(
    material: str,
    service: ReviewService = Depends(...)
):

    review = await service.review(material)

    return {
        "risk": review.risk.value,
        "explanation": review.explanation
    }
```

FastAPI does not contain the business logic.

It simply translates:

```text
HTTP request
     ↓
Python application call
```

---

# 15. Main.py is the composition root

Now we wire everything together.

### `main.py`

```python
from fastapi import FastAPI

from adapters.api.review_router import router
from adapters.llm.openai_adapter import OpenAIAdapter
from adapters.persistence.postgres_repository import (
    PostgresReviewRepository
)

from application.services.review_service import ReviewService


app = FastAPI()


llm = OpenAIAdapter()

repository = PostgresReviewRepository(
    session=...
)

service = ReviewService(
    llm=llm,
    repository=repository
)


app.include_router(router)
```

This location is where we decide:

```text
LLMPort
   ↓
OpenAIAdapter

ReviewRepository
   ↓
PostgresReviewRepository
```

This is called **dependency wiring/composition**.

---

# 16. The complete request flow

Suppose the client sends:

```http
POST /reviews
```

with:

```json
{
    "material": "This investment is guaranteed and risk-free."
}
```

The flow is:

```text
                    HTTP Client
                         │
                         ▼
                 ┌───────────────┐
                 │ FastAPI       │
                 │ Adapter       │
                 └───────┬───────┘
                         │
                         ▼
                 ReviewService
                         │
              ┌──────────┴─────────┐
              │                    │
              ▼                    ▼
       classify_risk()          LLMPort
              │                    │
              │                    ▼
              │             OpenAIAdapter
              │                    │
              │                    ▼
              │                  OpenAI
              │
              ▼
         Risk = HIGH
              │
              ▼
       ComplianceReview
              │
              ▼
      ReviewRepository
              │
              ▼
    PostgresRepository
              │
              ▼
        PostgreSQL
```

The critical thing is that the center doesn't depend directly on those technologies.

---

# 17. Inbound vs outbound adapters

This distinction is extremely important for interviews.

There are generally two directions.

## Inbound adapters

They **drive your application**.

Examples:

```text
FastAPI
REST
GraphQL
CLI
Kafka consumer
Celery task
gRPC
```

They call your application.

```text
FastAPI
   │
   ▼
Application
```

---

## Outbound adapters

They are used **by your application**.

Examples:

```text
OpenAI
PostgreSQL
Redis
Qdrant
AWS S3
Kafka producer
Email
```

```text
Application
     │
     ▼
Port
     │
     ▼
Adapter
     │
     ▼
External system
```

---

# 18. Visualizing the Hexagon

The classic mental model is:

```text
                       REST API
                          │
                          ▼
                ┌──────────────────┐
                │                  │
                │  INBOUND PORT    │
                │                  │
                └────────┬─────────┘
                         │
                         ▼
          ┌─────────────────────────────┐
          │                             │
          │       APPLICATION           │
          │                             │
          │       DOMAIN                │
          │                             │
          │   Business Rules            │
          │   Use Cases                 │
          │   Domain Models             │
          │                             │
          └───────┬──────────┬──────────┘
                  │          │
              OUTBOUND    OUTBOUND
                PORT         PORT
                  │          │
                  ▼          ▼
               LLMPort    Repository
                  │          │
                  ▼          ▼
               OpenAI     PostgreSQL
```

The "hexagon" isn't really about having six sides.

The important concept is:

> **The application core is independent of external technologies.**

---

# 19. Why use interfaces?

This is where the architecture becomes powerful.

Imagine you initially use OpenAI.

```python
class OpenAIAdapter:
    ...
```

Six months later you want AWS Bedrock.

You create:

```python
class BedrockAdapter:

    async def generate(
        self,
        prompt: str
    ) -> str:

        ...
```

Both satisfy:

```python
LLMPort
```

Your service doesn't change.

```python
class ReviewService:

    def __init__(self, llm: LLMPort):
        self.llm = llm
```

You can now switch:

```text
              LLMPort
              /     \
             /       \
            ▼         ▼
       OpenAI       Bedrock
```

The core stays untouched.

---

# 20. This becomes extremely useful in AI systems

Your financial AI application could have:

```text
                    LLMPort
                       │
         ┌─────────────┼──────────────┐
         ▼             ▼              ▼
      OpenAI        Bedrock         vLLM
```

Similarly:

```text
                 VectorStorePort
                       │
        ┌──────────────┼──────────────┐
        ▼              ▼              ▼
     pgvector        Qdrant          Pinecone
```

And:

```text
                 EmbeddingPort
                       │
            ┌──────────┼──────────┐
            ▼          ▼          ▼
         OpenAI       HF        Bedrock
```

And:

```text
                 CachePort
                    │
            ┌───────┴───────┐
            ▼               ▼
          Redis          InMemory
```

This is why Hexagonal Architecture is very useful for enterprise AI.

---

# 21. Testing becomes much easier

This is probably the biggest practical advantage.

Suppose your service directly uses OpenAI:

```python
class ReviewService:

    def __init__(self):
        self.client = AsyncOpenAI()
```

Testing becomes difficult because you need to mock OpenAI.

With Hexagonal Architecture:

```python
class FakeLLM:

    async def generate(self, prompt: str) -> str:

        return "This is a potentially misleading claim."
```

Then:

```python
class FakeRepository:

    def __init__(self):
        self.items = []

    async def save(self, review):
        self.items.append(review)
```

Test:

```python
async def test_review():

    llm = FakeLLM()
    repository = FakeRepository()

    service = ReviewService(
        llm=llm,
        repository=repository
    )

    result = await service.review(
        "This investment is guaranteed."
    )

    assert result.risk == RiskLevel.HIGH

    assert len(repository.items) == 1
```

No:

```text
OpenAI
PostgreSQL
Redis
network
```

is required.

That's a major Staff Engineer advantage.

---

# 22. Dependency direction

This is the most important rule to remember.

Bad architecture:

```text
Domain
  ↓
OpenAI
  ↓
PostgreSQL
```

or:

```text
Service
  ↓
FastAPI
  ↓
OpenAI
```

The core becomes coupled to infrastructure.

Hexagonal:

```text
                Infrastructure
               ↗              ↖
              /                \
       Adapters                Adapters
            ↑                    ↑
            │                    │
          Ports                Ports
             \                  /
              \                /
               Application
                    │
                  Domain
```

The dependency should point **toward the core**.

---

# 23. Hexagonal vs traditional layered architecture

Traditional:

```text
Controller
    ↓
Service
    ↓
Repository
    ↓
Database
```

This is fine for many applications.

But the service may directly depend on repository implementations.

Hexagonal:

```text
Controller
    ↓
Application
    ↓
Repository PORT
    ↑
Repository ADAPTER
    ↓
Database
```

The difference is the abstraction boundary.

---

# 24. Hexagonal vs Clean Architecture

These are closely related.

You will often see:

```text
Hexagonal Architecture
Ports & Adapters

Clean Architecture
Entities
Use Cases
Interface Adapters
Frameworks

Onion Architecture
Domain
Application
Infrastructure
```

They all follow a similar philosophy:

> **Keep business logic independent from infrastructure.**

Don't get overly focused on the naming.

At Staff level, explain the dependency rule.

---

# 25. Applying this to your Financial Compliance project

Your project could be organized as:

```text
financial-compliance/
│
├── domain/
│   ├── models/
│   ├── policies/
│   └── rules/
│
├── application/
│   ├── services/
│   └── workflows/
│
├── ports/
│   ├── llm.py
│   ├── repository.py
│   ├── vector_store.py
│   ├── embedding.py
│   ├── guardrails.py
│   └── checkpoint.py
│
├── adapters/
│   ├── api/
│   │   └── fastapi/
│   │
│   ├── llm/
│   │   ├── openai.py
│   │   └── bedrock.py
│   │
│   ├── persistence/
│   │   └── postgres.py
│   │
│   ├── vector/
│   │   └── pgvector.py
│   │
│   ├── cache/
│   │   └── redis.py
│   │
│   └── messaging/
│       └── celery.py
│
└── main.py
```

Then your architecture becomes:

```text
                         FastAPI
                            │
                            ▼
                     Inbound Adapter
                            │
                            ▼
                    Application Service
                            │
                            ▼
                      LangGraph
                            │
             ┌──────────────┼──────────────┐
             ▼              ▼              ▼
          Planner       Supervisor      Evaluator
             │              │              │
             └──────────────┼──────────────┘
                            │
                    Domain / Use Cases
                            │
          ┌─────────────────┼─────────────────┐
          ▼                 ▼                 ▼
      LLM Port         Vector Port       Repository Port
          │                 │                 │
          ▼                 ▼                 ▼
       OpenAI            pgvector         PostgreSQL
```

Then:

```text
Redis
Celery
OpenTelemetry
Guardrails
HITL
```

can also be connected through appropriate ports/adapters depending on whether they're infrastructure concerns or application/domain policies.

---

# 26. One very important distinction for your projects

Don't make **everything** a port.

For example, this is unnecessary:

```python
class StringUtilsPort:
    ...
```

You use ports for things that are:

**external, replaceable, infrastructure-dependent, or operationally significant.**

Good ports:

```text
LLMPort
RepositoryPort
VectorStorePort
EmbeddingPort
CachePort
EventPublisherPort
CheckpointStorePort
```

Usually not necessary:

```text
RiskClassifierPort
StringFormatterPort
DateHelperPort
```

if they're pure domain logic.

---

# 27. The Staff Engineer explanation

If an interviewer asks:

> "Why did you use Hexagonal Architecture?"

A strong answer is:

> "I wanted to keep the financial compliance business logic independent from infrastructure and vendor-specific implementations. I modeled external capabilities such as LLM providers, persistence, vector search, caching and messaging as outbound ports. Concrete implementations such as OpenAI, PostgreSQL, pgvector and Redis live in adapters. FastAPI and Celery act as inbound adapters that invoke application use cases. This allows us to replace infrastructure implementations without changing the domain logic, improves unit-testability, and gives us clear dependency boundaries."

Then give an example:

> "For example, our compliance workflow depends on an `LLMPort` rather than directly on OpenAI. Today the adapter can use OpenAI, but we can switch to Bedrock or vLLM without modifying the compliance use case."

That is a **much stronger answer** than simply saying:

> "We used clean architecture."

---

# 28. The one diagram you should remember

If you remember only one thing, remember this:

```text
                 INBOUND ADAPTERS
          ┌──────────────────────────┐
          │ FastAPI │ Celery │ gRPC │
          └────────────┬─────────────┘
                       │
                       ▼
                APPLICATION CORE
          ┌───────────────────────────┐
          │                           │
          │       Use Cases           │
          │                           │
          │       Domain              │
          │                           │
          │       Business Rules      │
          │                           │
          └────────────┬──────────────┘
                       │
                    PORTS
                       │
          ┌────────────┼────────────┐
          ▼            ▼            ▼
        LLMPort    RepositoryPort  VectorPort
          │            │            │
          ▼            ▼            ▼
       OpenAI       PostgreSQL     pgvector
```

### In one sentence:

**FastAPI/Celery drive the application through inbound ports; the application drives OpenAI/PostgreSQL/pgvector/Redis through outbound ports; adapters implement those ports; the domain remains independent.**

That is Hexagonal Architecture.
