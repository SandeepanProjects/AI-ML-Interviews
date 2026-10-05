Yes. This is the natural next step after understanding **Hexagonal Architecture + Strategy Pattern**.

For your two applications—**Financial Compliance Review** and **Investment Market Intelligence**—the goal is not simply to put code into different folders. The goal is to create **real Python packages with explicit dependencies**, install them into each application, and allow multiple applications to reuse the same capabilities without copying code.

I'll show you the architecture first, then the actual `pyproject.toml`, package code, installation, imports, and runtime flow.

---

# 1. What does "modular" actually mean?

Suppose you have two applications:

```text
Application A
Financial Compliance
```

and

```text
Application B
Investment Intelligence
```

Both need:

* authentication
* PostgreSQL
* Redis
* LLM
* RAG
* embeddings
* observability
* guardrails
* agent runtime

Without modularization, you might do this:

```text
financial-compliance/
├── auth/
├── database/
├── redis/
├── llm/
├── rag/
└── agents/

investment-intelligence/
├── auth/              ← duplicate
├── database/          ← duplicate
├── redis/             ← duplicate
├── llm/               ← duplicate
├── rag/               ← duplicate
└── agents/            ← duplicate
```

This is bad.

You now have two versions of the same infrastructure.

---

# 2. Modular architecture

Instead, create reusable packages:

```text
                    Shared Packages
                         │
        ┌────────────────┼─────────────────┐
        │                │                 │
        ▼                ▼                 ▼
   platform-auth    platform-db       platform-cache
        │                │                 │
        ├────────────────┼─────────────────┤
        │                │                 │
        ▼                ▼                 ▼
      App A            App A             App A
      App B            App B             App B
```

Your applications then contain only business-specific logic.

---

# 3. The architecture I recommend

For your AI platform:

```text
financial-ai-platform/
│
├── packages/
│
│   ├── platform-core/
│   ├── platform-auth/
│   ├── platform-db/
│   ├── platform-cache/
│   ├── platform-messaging/
│   ├── platform-observability/
│   │
│   ├── ai-llm/
│   ├── ai-rag/
│   ├── ai-agents/
│   └── ai-guardrails/
│
│
├── applications/
│
│   ├── financial-compliance/
│   └── investment-intelligence/
│
│
├── deployments/
├── tests/
├── docs/
│
├── pyproject.toml
└── Makefile
```

Think of it as:

```text
packages/
    ↓
Reusable capabilities

applications/
    ↓
Business applications
```

---

# 4. What goes into a package?

This is where people often make mistakes.

A package should represent a **cohesive reusable capability**.

For example:

```text
platform-auth
```

could contain:

```text
JWT
OIDC
RBAC
Principal
Permissions
TenantContext
```

But it should NOT contain:

```text
ComplianceReviewer
InvestmentPortfolio
MarketSignal
PolicyViolation
```

Those belong to applications.

---

# 5. A useful rule

Ask:

> "Could another completely different application reuse this without knowing anything about my business domain?"

If yes → probably a shared package.

For example:

```text
JWT
Redis
PostgreSQL connection
OpenTelemetry
LLM client
Embedding client
```

are reusable.

But:

```text
ComplianceRisk
PolicyViolation
PortfolioRecommendation
MarketSignal
```

are business-specific.

---

# 6. Example package: `platform-core`

Create:

```text
packages/
└── platform-core/
    ├── pyproject.toml
    ├── README.md
    └── src/
        └── platform_core/
            ├── __init__.py
            ├── errors.py
            └── types.py
```

Notice the important part:

```text
src/platform_core/
```

The folder containing the Python package is:

```text
platform_core
```

The project distribution name can be:

```text
platform-core
```

These don't have to be identical.

---

# 7. `pyproject.toml`

Inside:

```text
packages/platform-core/pyproject.toml
```

write:

```toml
[build-system]
requires = ["setuptools>=68"]
build-backend = "setuptools.build_meta"

[project]
name = "platform-core"
version = "0.1.0"
description = "Shared core primitives for financial AI applications"
requires-python = ">=3.11"

[tool.setuptools.packages.find]
where = ["src"]
```

This tells Python:

> This directory is an installable package.

---

# 8. Package code

### `errors.py`

```python
class PlatformError(Exception):
    """Base exception for the platform."""


class ValidationError(PlatformError):
    """Raised when validation fails."""


class AuthorizationError(PlatformError):
    """Raised when authorization fails."""
```

### `types.py`

```python
from dataclasses import dataclass


@dataclass(frozen=True)
class TenantContext:

    tenant_id: str
    user_id: str
```

Now another application can use:

```python
from platform_core.types import TenantContext
```

---

# 9. Creating `platform-auth`

Structure:

```text
packages/platform-auth/
│
├── pyproject.toml
│
└── src/
    └── platform_auth/
        ├── __init__.py
        ├── models.py
        ├── rbac.py
        └── jwt.py
```

---

# 10. Its `pyproject.toml`

```toml
[build-system]
requires = ["setuptools>=68"]
build-backend = "setuptools.build_meta"

[project]
name = "platform-auth"
version = "0.1.0"
requires-python = ">=3.11"

dependencies = [
    "platform-core>=0.1.0",
    "PyJWT>=2.8"
]

[tool.setuptools.packages.find]
where = ["src"]
```

Now you have a dependency:

```text
platform-auth
       │
       ▼
platform-core
```

This is important.

Packages should have **explicit dependencies**.

---

# 11. Authentication package

### `models.py`

```python
from dataclasses import dataclass


@dataclass(frozen=True)
class Principal:

    user_id: str
    tenant_id: str
    roles: tuple[str, ...]
```

### `rbac.py`

```python
from platform_auth.models import Principal


class RBAC:

    def has_permission(
        self,
        principal: Principal,
        permission: str
    ) -> bool:

        if "admin" in principal.roles:
            return True

        if permission == "review:create":
            return "reviewer" in principal.roles

        return False
```

Now any application can use:

```python
from platform_auth.rbac import RBAC
```

---

# 12. Create `platform-db`

```text
packages/platform-db/
│
├── pyproject.toml
│
└── src/
    └── platform_db/
        ├── __init__.py
        ├── session.py
        └── uow.py
```

`pyproject.toml`:

```toml
[build-system]
requires = ["setuptools>=68"]
build-backend = "setuptools.build_meta"

[project]
name = "platform-db"
version = "0.1.0"
requires-python = ">=3.11"

dependencies = [
    "SQLAlchemy>=2.0",
    "asyncpg>=0.29"
]

[tool.setuptools.packages.find]
where = ["src"]
```

---

# 13. Database package

### `session.py`

```python
from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)


class Database:

    def __init__(self, database_url: str):

        self.engine = create_async_engine(
            database_url,
            pool_pre_ping=True,
            pool_size=10,
            max_overflow=20,
        )

        self.session_factory = async_sessionmaker(
            self.engine,
            expire_on_commit=False,
        )

    def session(self) -> AsyncSession:

        return self.session_factory()
```

Now applications don't need to duplicate database setup.

---

# 14. Create `platform-cache`

```text
packages/platform-cache/
│
├── pyproject.toml
│
└── src/
    └── platform_cache/
        ├── __init__.py
        └── redis.py
```

```toml
[build-system]
requires = ["setuptools>=68"]
build-backend = "setuptools.build_meta"

[project]
name = "platform-cache"
version = "0.1.0"
requires-python = ">=3.11"

dependencies = [
    "redis>=5.0"
]

[tool.setuptools.packages.find]
where = ["src"]
```

Implementation:

```python
from redis.asyncio import Redis


class Cache:

    def __init__(self, redis_url: str):

        self.client = Redis.from_url(
            redis_url,
            decode_responses=True
        )

    async def get(self, key: str):

        return await self.client.get(key)

    async def set(
        self,
        key: str,
        value: str,
        ttl: int = 300
    ):

        await self.client.set(
            key,
            value,
            ex=ttl
        )
```

---

# 15. Now the AI packages

The same concept applies.

## `ai-llm`

```text
packages/ai-llm/
└── src/
    └── ai_llm/
        ├── __init__.py
        ├── ports.py
        ├── openai.py
        ├── bedrock.py
        └── router.py
```

The port:

```python
from typing import Protocol


class LLMProvider(Protocol):

    async def generate(
        self,
        prompt: str
    ) -> str:
        ...
```

Now implementations:

```text
LLMProvider
    ▲
    │
    ├── OpenAIProvider
    ├── BedrockProvider
    └── VLLMProvider
```

---

# 16. `ai-rag`

```text
packages/ai-rag/
│
└── src/
    └── ai_rag/
        ├── __init__.py
        ├── models.py
        ├── ports.py
        ├── chunking.py
        ├── embeddings.py
        ├── retrieval.py
        └── reranking.py
```

You can define:

```python
class VectorStore(Protocol):

    async def search(
        self,
        embedding: list[float],
        top_k: int
    ):
        ...
```

Then:

```text
VectorStore
    ▲
    │
    ├── PgVectorAdapter
    ├── QdrantAdapter
    └── PineconeAdapter
```

---

# 17. `ai-agents`

This can contain generic agent infrastructure.

```text
packages/ai-agents/
│
└── src/
    └── ai_agents/
        ├── contracts.py
        ├── runtime.py
        ├── planner.py
        ├── supervisor.py
        ├── evaluator.py
        ├── checkpoint.py
        └── workflow.py
```

But be careful.

Don't put:

```text
ComplianceAgent
PortfolioAgent
RiskAgent
FeeAgent
```

inside the shared package.

Those are application-specific.

---

# 18. `ai-guardrails`

Reusable:

```text
PII detection
Prompt injection
Output schema validation
Citation validation
Content safety
```

But:

```text
SEC marketing rule
Investment suitability rule
Financial promotion rule
```

belong in the compliance application.

This separation is extremely important.

---

# 19. Now create the application

Let's create:

```text
applications/
└── financial-compliance/
    │
    ├── pyproject.toml
    │
    └── src/
        └── financial_compliance/
            ├── __init__.py
            ├── main.py
            ├── api/
            ├── domain/
            ├── application/
            ├── ports/
            └── adapters/
```

This application has its own architecture.

---

# 20. Application `pyproject.toml`

This is where the magic happens.

```toml
[build-system]
requires = ["setuptools>=68"]
build-backend = "setuptools.build_meta"

[project]
name = "financial-compliance"
version = "0.1.0"

requires-python = ">=3.11"

dependencies = [
    "fastapi>=0.115",
    "uvicorn[standard]>=0.30",

    "platform-core>=0.1.0",
    "platform-auth>=0.1.0",
    "platform-db>=0.1.0",
    "platform-cache>=0.1.0",

    "ai-llm>=0.1.0",
    "ai-rag>=0.1.0",
    "ai-agents>=0.1.0",
    "ai-guardrails>=0.1.0"
]

[tool.setuptools.packages.find]
where = ["src"]
```

This application now declares:

```text
financial-compliance
        │
        ├── platform-core
        ├── platform-auth
        ├── platform-db
        ├── platform-cache
        │
        ├── ai-llm
        ├── ai-rag
        ├── ai-agents
        └── ai-guardrails
```

---

# 21. Installing the packages

Suppose your project is:

```text
~/financial-ai-platform
```

Create your virtual environment:

```bash
cd ~/financial-ai-platform

python3 -m venv .venv
```

Activate it:

### macOS/Linux

```bash
source .venv/bin/activate
```

### Windows

```powershell
.venv\Scripts\activate
```

Upgrade pip:

```bash
python -m pip install --upgrade pip
```

---

# 22. Install shared packages

You can install them individually:

```bash
pip install -e packages/platform-core
```

Then:

```bash
pip install -e packages/platform-auth
pip install -e packages/platform-db
pip install -e packages/platform-cache
```

Then AI packages:

```bash
pip install -e packages/ai-llm
pip install -e packages/ai-rag
pip install -e packages/ai-agents
pip install -e packages/ai-guardrails
```

Then application:

```bash
pip install -e applications/financial-compliance
```

The `-e` is important.

---

# 23. What does `-e` mean?

This:

```bash
pip install -e packages/platform-core
```

means:

> Install this package in editable/development mode.

Suppose you change:

```text
packages/platform-core/src/platform_core/errors.py
```

You don't need to reinstall it.

Python uses your local source.

That's ideal during development.

---

# 24. What happens without `-e`?

If you run:

```bash
pip install packages/platform-core
```

pip installs a built package into the virtual environment.

If you then change the source code, the installed version doesn't automatically reflect your changes.

For local development:

```bash
pip install -e ...
```

is usually much more convenient.

---

# 25. But manually installing 10 packages is annoying

Correct.

That's why you create a root `pyproject.toml`.

For example:

```toml
[project]
name = "financial-ai-platform"
version = "0.1.0"

requires-python = ">=3.11"

dependencies = [
    "platform-core",
    "platform-auth",
    "platform-db",
    "platform-cache",
    "platform-messaging",
    "platform-observability",

    "ai-llm",
    "ai-rag",
    "ai-agents",
    "ai-guardrails"
]
```

But local path dependencies need to be handled carefully. For a monorepo, I recommend either:

1. **workspace tooling**, or
2. a Makefile/script that installs each local package editable.

For a Python platform like yours, the second option is simple and transparent.

---

# 26. Create a Makefile

At the root:

```text
Makefile
```

```makefile
install:
	pip install -e packages/platform-core
	pip install -e packages/platform-auth
	pip install -e packages/platform-db
	pip install -e packages/platform-cache
	pip install -e packages/platform-messaging
	pip install -e packages/platform-observability

	pip install -e packages/ai-llm
	pip install -e packages/ai-rag
	pip install -e packages/ai-agents
	pip install -e packages/ai-guardrails

	pip install -e applications/financial-compliance
	pip install -e applications/investment-intelligence
```

Now:

```bash
make install
```

installs everything.

---

# 27. Why installation order matters

Consider:

```text
platform-auth
      ↓
platform-core
```

You should install:

```text
platform-core
↓
platform-auth
```

Similarly:

```text
ai-rag
```

might depend on:

```text
platform-core
ai-llm
```

So:

```text
platform-core
      ↓
ai-llm
      ↓
ai-rag
      ↓
application
```

However, if dependencies are declared properly in `pyproject.toml`, pip can resolve installed package dependencies. Your Makefile is primarily making the local editable workflow explicit.

---

# 28. Importing packages into applications

Now the beautiful part.

Inside:

```text
financial_compliance/
```

you can simply write:

```python
from platform_auth.models import Principal
```

or:

```python
from platform_db.session import Database
```

or:

```python
from platform_cache.redis import Cache
```

or:

```python
from ai_llm.ports import LLMProvider
```

or:

```python
from ai_rag.retrieval import HybridRetriever
```

The application doesn't need:

```python
../../../packages/ai-rag/src/...
```

That's exactly why we created installable packages.

---

# 29. Second application

Now create:

```text
applications/
└── investment-intelligence/
    ├── pyproject.toml
    └── src/
        └── investment_intelligence/
            ├── main.py
            ├── domain/
            ├── application/
            ├── ports/
            └── adapters/
```

Its `pyproject.toml` can use exactly the same shared packages:

```toml
[project]
name = "investment-intelligence"
version = "0.1.0"

dependencies = [
    "fastapi>=0.115",
    "uvicorn[standard]>=0.30",

    "platform-core>=0.1.0",
    "platform-auth>=0.1.0",
    "platform-db>=0.1.0",
    "platform-cache>=0.1.0",

    "ai-llm>=0.1.0",
    "ai-rag>=0.1.0",
    "ai-agents>=0.1.0"
]

[tool.setuptools.packages.find]
where = ["src"]
```

Now:

```text
                   Shared Packages
                         │
          ┌──────────────┴──────────────┐
          │                             │
          ▼                             ▼
Financial Compliance        Investment Intelligence
          │                             │
          ├── auth                      ├── auth
          ├── db                        ├── db
          ├── redis                     ├── redis
          ├── llm                       ├── llm
          ├── rag                       ├── rag
          └── agents                    └── agents
```

No duplication.

---

# 30. But applications can have different versions

This becomes important in enterprise systems.

Suppose:

```text
financial-compliance
```

needs:

```text
ai-rag >= 1.2,<2
```

while:

```text
investment-intelligence
```

needs:

```text
ai-rag >= 1.5,<2
```

You can express those constraints.

For example:

```toml
"ai-rag>=1.2,<2.0"
```

and:

```toml
"ai-rag>=1.5,<2.0"
```

If both applications run in the **same virtual environment**, they must have a compatible installed version.

In production, I recommend:

```text
Application A
    ↓
Container A
    ↓
Virtual environment A

Application B
    ↓
Container B
    ↓
Virtual environment B
```

This gives each application its own dependency environment.

---

# 31. Development vs production

### Development

You can have:

```text
one repository
one virtual environment
multiple editable packages
```

```text
.venv
   │
   ├── platform-core
   ├── platform-auth
   ├── ai-llm
   ├── ai-rag
   ├── compliance
   └── investment
```

### Production

Use separate containers:

```text
                 Docker/Kubernetes
                       │
          ┌────────────┴────────────┐
          ▼                         ▼
 Compliance Container      Investment Container
          │                         │
          ▼                         ▼
 Shared packages            Shared packages
 baked into image           baked into image
```

This is a much cleaner enterprise deployment model.

---

# 32. Version your packages

Suppose you have:

```text
platform-auth 1.0.0
```

You make a backwards-compatible change:

```text
1.1.0
```

Breaking change:

```text
2.0.0
```

The applications can then specify:

```toml
"platform-auth>=1.0,<2.0"
```

This is much better than copying source code between projects.

---

# 33. Eventually publish internal packages

At a larger organization, you might have:

```text
Internal PyPI
```

or:

```text
AWS CodeArtifact
```

or:

```text
GitHub Packages
```

Then instead of:

```bash
pip install -e packages/platform-auth
```

production CI might do:

```bash
pip install platform-auth==1.4.2
```

And:

```bash
pip install ai-rag==2.1.0
```

Your internal package registry becomes:

```text
                  Internal Package Registry
                            │
           ┌────────────────┼────────────────┐
           ▼                ▼                ▼
    platform-auth       ai-rag          ai-agents
           │                │                │
           └────────────────┼────────────────┘
                            ▼
                    Applications
```

---

# 34. Very important: don't create a "god package"

Don't do this:

```text
platform/
├── auth
├── database
├── redis
├── rag
├── agents
├── compliance
├── portfolio
├── market
├── billing
├── notifications
└── everything_else
```

That's just your old monolith packaged differently.

Instead:

```text
platform-core
platform-auth
platform-db
platform-cache
platform-observability

ai-llm
ai-rag
ai-agents
ai-guardrails
```

Each package should have a clear responsibility.

---

# 35. Package dependency graph

This is the Staff-level part.

You want something approximately like:

```text
                     platform-core
                    /      |       \
                   /       |        \
                  ▼        ▼         ▼
          platform-auth  platform-db  platform-cache
                  │         │              │
                  └─────────┼──────────────┘
                            │
                            ▼
                         ai-llm
                            │
                    ┌───────┴───────┐
                    ▼               ▼
                  ai-rag        ai-agents
                    │               │
                    └───────┬───────┘
                            ▼
                     Applications
                    /              \
                   ▼                ▼
          Compliance         Investment
```

But avoid:

```text
platform-core
       ↑
       │
application
       ↑
       │
platform-core
```

or:

```text
ai-rag → compliance → ai-rag
```

Circular dependencies are a major warning sign.

---

# 36. The application should own business logic

For example:

```text
financial-compliance/
│
├── domain/
│   ├── compliance_policy.py
│   ├── risk.py
│   └── violation.py
│
├── application/
│   ├── review_service.py
│   └── approval_service.py
│
└── agents/
    ├── compliance_research_agent.py
    ├── risk_agent.py
    └── compliance_writer_agent.py
```

These shouldn't move into:

```text
ai-agents
```

because they are specific to financial compliance.

Similarly:

```text
investment-intelligence/
│
├── domain/
│   ├── portfolio.py
│   ├── market_signal.py
│   └── recommendation.py
│
└── agents/
    ├── research_agent.py
    ├── market_agent.py
    └── portfolio_agent.py
```

belongs to that application.

---

# 37. How your final architecture looks

For your projects, I would use:

```text
financial-ai-platform/
│
├── packages/
│
│   ├── platform-core
│   │
│   ├── platform-auth
│   │
│   ├── platform-db
│   │
│   ├── platform-cache
│   │
│   ├── platform-messaging
│   │
│   ├── platform-observability
│   │
│   ├── ai-llm
│   │
│   ├── ai-rag
│   │
│   ├── ai-agents
│   │
│   └── ai-guardrails
│
│
├── applications/
│
│   ├── financial-compliance/
│   │
│   │   ├── domain/
│   │   ├── application/
│   │   ├── ports/
│   │   ├── adapters/
│   │   └── main.py
│   │
│   └── investment-intelligence/
│
│       ├── domain/
│       ├── application/
│       ├── ports/
│       ├── adapters/
│       └── main.py
│
│
├── deployments/
│   ├── docker/
│   ├── kubernetes/
│   └── terraform/
│
├── tests/
│
├── docs/
│
├── Makefile
└── README.md
```

---

# 38. The request flow after modularization

Suppose the user calls:

```http
POST /v1/reviews
```

The flow is:

```text
Client
  │
  ▼
FastAPI
  │
  ▼
Compliance Application
  │
  ▼
ReviewService
  │
  ├───────────────► platform-auth
  │
  ├───────────────► ai-guardrails
  │
  ├───────────────► ai-agents
  │                       │
  │                       ▼
  │                  Planner/Supervisor
  │                       │
  │                       ▼
  │                    ai-rag
  │                       │
  │                       ▼
  │                    ai-llm
  │                       │
  │                       ▼
  │                     OpenAI
  │
  ├───────────────► platform-db
  │                       │
  │                       ▼
  │                   PostgreSQL
  │
  └───────────────► platform-cache
                          │
                          ▼
                         Redis
```

This is the real benefit.

The application becomes an **assembly of reusable capabilities** rather than a giant codebase.

---

# 39. The mental model I want you to remember

There are three levels:

### Level 1 — Shared infrastructure

```text
platform-auth
platform-db
platform-cache
platform-observability
```

### Level 2 — Shared AI capabilities

```text
ai-llm
ai-rag
ai-agents
ai-guardrails
```

### Level 3 — Business applications

```text
financial-compliance
investment-intelligence
```

And the dependency direction is:

```text
Business Applications
        │
        ▼
Shared AI Capabilities
        │
        ▼
Platform Capabilities
        │
        ▼
Infrastructure
```

**But the business domain itself should not leak downward into the generic packages.**

---

# 40. One more important distinction

Don't confuse **Python package** with **microservice**.

These are different.

You can have:

```text
ai-rag
```

as a Python package and use it inside:

```text
Compliance Service
```

and:

```text
Investment Service
```

without running a separate RAG microservice.

Later, if RAG needs independent scaling, you can turn it into a service:

```text
Compliance API
      │
      ▼
RAG Service
      │
      ▼
pgvector
```

So:

```text
Package = code reuse boundary
Service = runtime/deployment boundary
```

That's a very important Staff Engineer distinction.

---

## The final picture

Your architecture should ultimately look like this:

```text
                         ┌─────────────────────┐
                         │  Financial          │
                         │  Compliance App     │
                         └──────────┬──────────┘
                                    │
                         ┌──────────▼──────────┐
                         │  Investment         │
                         │  Intelligence App   │
                         └──────────┬──────────┘
                                    │
                 ┌──────────────────┴──────────────────┐
                 │                                     │
                 ▼                                     ▼
          ┌──────────────┐                      ┌──────────────┐
          │ Shared AI    │                      │ Platform     │
          │ Packages     │                      │ Packages     │
          ├──────────────┤                      ├──────────────┤
          │ ai-llm       │                      │ auth         │
          │ ai-rag       │                      │ db           │
          │ ai-agents    │                      │ cache        │
          │ guardrails   │                      │ messaging    │
          └──────┬───────┘                      │ observability│
                 │                              └──────┬───────┘
                 └──────────────────┬──────────────────┘
                                    ▼
                          External Infrastructure
                                    │
                ┌───────────┬───────┼───────┬──────────┐
                ▼           ▼       ▼       ▼          ▼
             OpenAI    PostgreSQL  Redis  pgvector   Celery
```

**The key principle is:**

> **Build shared capabilities once as installable packages, give every package an explicit `pyproject.toml` and dependency contract, install them editable during development, version them for production, and keep business-specific logic inside each application.**

That gives you a genuinely modular architecture rather than simply a large repository split into folders.
