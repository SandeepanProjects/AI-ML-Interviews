Yes. At **Senior/Staff AI Engineer level**, I would not think of “modular packages” as simply putting code into different folders. The goal is to create **independently versioned, privately hosted, reusable Python packages** that can be consumed by multiple AI applications.

For example, instead of this:

```text
financial-ai-project/
├── agents/
├── rag/
├── llm/
├── security/
├── observability/
└── utils/
```

you can evolve toward:

```text
Private GitHub
│
├── ai-platform-core
├── ai-rag
├── ai-agents
├── ai-security
├── ai-observability
├── ai-evaluation
└── financial-advisor-api
```

The important part is that **the repositories remain private**.

---

# 1. What "modular package" means at Staff level

Imagine you have three AI applications:

```text
Financial Advisor
Compliance Review
Market Intelligence
```

All three need:

* LLM client
* RAG
* embeddings
* vector search
* authentication
* RBAC
* tracing
* evaluation
* retry policies
* common schemas

You don't want to copy the same implementation three times.

Instead:

```text
                    ┌─────────────────────┐
                    │ ai-platform-core    │
                    │                     │
                    │ Config              │
                    │ Exceptions          │
                    │ Logging             │
                    │ Common schemas      │
                    │ HTTP utilities      │
                    └──────────┬──────────┘
                               │
          ┌────────────────────┼────────────────────┐
          │                    │                    │
          ▼                    ▼                    ▼
   ┌────────────┐       ┌──────────────┐     ┌──────────────┐
   │ ai-rag     │       │ ai-agents    │     │ ai-security  │
   └─────┬──────┘       └──────┬───────┘     └──────┬───────┘
         │                     │                    │
         └─────────────────────┼────────────────────┘
                               ▼
                    ┌─────────────────────┐
                    │ Financial Advisor   │
                    │ Application         │
                    └─────────────────────┘
```

That is much closer to how I would structure an enterprise AI platform.

---

# 2. Private GitHub is completely fine

You **do not need public GitHub repositories**.

You can create:

```text
GitHub Organization
        │
        ├── Private: ai-platform-core
        ├── Private: ai-rag
        ├── Private: ai-agents
        ├── Private: ai-security
        ├── Private: ai-observability
        └── Private: financial-advisor
```

For example:

```text
github.com/<your-org>/ai-rag
```

with repository visibility:

```text
Private
```

Only people/apps you authorize can access it.

For an enterprise organization, I would normally use:

```text
GitHub Organization
        │
        ├── Teams
        │    ├── AI Platform
        │    ├── Application Teams
        │    └── DevOps
        │
        └── Private repositories
```

---

# 3. First understand Repository vs Package

This distinction is extremely important.

A **GitHub repository** is where your source code lives.

A **Python package** is something another project can install.

For example:

```text
GitHub repository
       │
       │
       ▼
ai-rag
       │
       │ build
       ▼
Python package
       │
       ▼
pip install ai-rag
```

So your architecture is:

```text
Private Git repository
          ↓
Python package
          ↓
Version
          ↓
Package registry
          ↓
Application
```

---

# 4. Don't directly depend on random Git branches

A beginner approach is:

```bash
pip install git+https://github.com/company/ai-rag.git
```

This works, but I would **not make this your primary production dependency strategy**.

Why?

Because your application could effectively depend on whatever happens to be on a branch.

Instead, create releases:

```text
ai-rag
│
├── v1.0.0
├── v1.1.0
├── v1.2.0
└── v2.0.0
```

Then application:

```text
financial-advisor
        │
        └── ai-rag==1.2.0
```

Now you have reproducibility.

---

# 5. Create your first private repository

Let's create:

```text
ai-platform-core
```

Private GitHub repository.

Locally:

```bash
mkdir ai-platform-core
cd ai-platform-core

git init
```

Structure:

```text
ai-platform-core/
│
├── pyproject.toml
├── README.md
├── LICENSE
├── .gitignore
│
├── src/
│   └── ai_platform_core/
│       ├── __init__.py
│       ├── config/
│       │   ├── __init__.py
│       │   └── settings.py
│       │
│       ├── exceptions/
│       │   ├── __init__.py
│       │   └── base.py
│       │
│       ├── logging/
│       │   ├── __init__.py
│       │   └── logger.py
│       │
│       └── schemas/
│           ├── __init__.py
│           └── common.py
│
└── tests/
    └── test_core.py
```

Notice the:

```text
src/
```

layout.

I recommend this for reusable packages because it helps prevent accidental imports from the repository root.

---

# 6. Create `pyproject.toml`

For example:

```toml
[build-system]
requires = ["setuptools>=68"]
build-backend = "setuptools.build_meta"

[project]
name = "ai-platform-core"
version = "1.0.0"

description = "Shared enterprise AI platform primitives"

requires-python = ">=3.11"

dependencies = [
    "pydantic>=2.0"
]

[tool.setuptools.packages.find]
where = ["src"]
```

Now your repository isn't just a collection of Python files.

It is a **real installable Python package**.

---

# 7. Example core package

`settings.py`

```python
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    environment: str = "development"

    openai_api_key: str | None = None

    database_url: str | None = None

    redis_url: str | None = None

    class Config:
        env_file = ".env"


settings = Settings()
```

Then another project can do:

```python
from ai_platform_core.config.settings import settings
```

That's the important part.

---

# 8. Now create a RAG package

Create another **private** repository:

```text
ai-rag
```

Structure:

```text
ai-rag/
│
├── pyproject.toml
├── README.md
│
├── src/
│   └── ai_rag/
│       │
│       ├── __init__.py
│       │
│       ├── ingestion/
│       │   ├── __init__.py
│       │   ├── loader.py
│       │   └── pipeline.py
│       │
│       ├── chunking/
│       │   ├── __init__.py
│       │   └── chunker.py
│       │
│       ├── embeddings/
│       │   ├── __init__.py
│       │   └── embedder.py
│       │
│       ├── retrieval/
│       │   ├── __init__.py
│       │   ├── vector.py
│       │   ├── bm25.py
│       │   └── hybrid.py
│       │
│       ├── reranking/
│       │   └── reranker.py
│       │
│       └── pipeline.py
│
└── tests/
```

Now your RAG capability is independently reusable.

---

# 9. Dependency between private packages

Suppose:

```text
ai-rag
```

needs:

```text
ai-platform-core
```

Your `pyproject.toml` can specify:

```toml
dependencies = [
    "ai-platform-core>=1.0,<2.0"
]
```

That means:

```text
ai-rag
   │
   └── ai-platform-core
```

But your application might have:

```text
financial-advisor
   │
   ├── ai-platform-core
   ├── ai-rag
   ├── ai-agents
   ├── ai-security
   └── ai-observability
```

This is your **internal platform dependency graph**.

---

# 10. Don't create one giant "common" package

This is one of the most important Staff-level lessons.

Don't do:

```text
ai-common
│
├── RAG
├── Agents
├── Security
├── Database
├── Redis
├── ML
├── LLM
├── Observability
├── API
└── Everything else
```

That eventually becomes a **god package**.

Instead:

```text
ai-platform-core
ai-rag
ai-agents
ai-security
ai-observability
ai-evaluation
```

Each package should have a **clear ownership boundary**.

---

# 11. Define package responsibilities

For example:

### `ai-platform-core`

Owns:

```text
Configuration
Common exceptions
Common types
Base interfaces
Correlation IDs
Shared utilities
```

### `ai-rag`

Owns:

```text
Document loading
Chunking
Embedding
Retrieval
Hybrid search
Reranking
Citation generation
```

### `ai-agents`

Owns:

```text
Agent abstractions
Tools
Planner
Supervisor
Executor
Agent state
Handoffs
```

### `ai-security`

Owns:

```text
JWT
RBAC
ABAC
Tool authorization
PII protection
Prompt injection protection
```

### `ai-observability`

Owns:

```text
OpenTelemetry
Tracing
Metrics
LLM token tracking
Latency
Cost
```

### `ai-evaluation`

Owns:

```text
RAGAS
Faithfulness
Answer relevance
Retrieval metrics
LLM-as-judge
Regression evaluation
```

That is much more maintainable.

---

# 12. The biggest Staff-level concept: interfaces

Suppose your RAG package needs a vector database.

Don't hardcode Qdrant everywhere.

Bad:

```python
class Retriever:

    def retrieve(self, query):
        client = QdrantClient(...)
        return client.search(...)
```

Now your entire package is coupled to Qdrant.

Instead:

```python
from typing import Protocol


class VectorStore(Protocol):

    async def search(
        self,
        query: str,
        top_k: int
    ) -> list:
        ...
```

Then:

```text
VectorStore
     │
     ├── QdrantVectorStore
     │
     ├── PgVectorStore
     │
     └── ElasticsearchVectorStore
```

This is **Ports and Adapters / Hexagonal Architecture**.

---

# 13. Then your application chooses the adapter

For example:

```python
vector_store = QdrantVectorStore(
    client=qdrant_client
)
```

or:

```python
vector_store = PgVectorStore(
    session=session
)
```

The RAG orchestration doesn't care.

It sees:

```python
VectorStore
```

This gives you:

```text
Business logic
      │
      ▼
   Port
      │
 ┌────┴────┐
 ▼         ▼
Qdrant    pgvector
```

That's much more enterprise-friendly.

---

# 14. Your packages should expose public APIs

Don't force application developers to know your internal directory structure.

For example, avoid requiring:

```python
from ai_rag.src.ai_rag.retrieval.hybrid.internal.engine.v2 import ...
```

Instead expose:

```python
from ai_rag import RAGPipeline
```

Your:

```text
ai_rag/__init__.py
```

could expose:

```python
from .pipeline import RAGPipeline

__all__ = [
    "RAGPipeline"
]
```

Now consumers have a stable API.

Internals can change.

---

# 15. Versioning becomes extremely important

Use semantic versioning:

```text
MAJOR.MINOR.PATCH
```

For example:

```text
1.0.0
```

### PATCH

Bug fix:

```text
1.0.0 → 1.0.1
```

### MINOR

Backward-compatible functionality:

```text
1.0.1 → 1.1.0
```

### MAJOR

Breaking API:

```text
1.1.0 → 2.0.0
```

Example:

```text
ai-rag==1.4.2
```

Your production application should know exactly which version it uses.

---

# 16. Where should private packages be installed from?

There are several approaches.

### Option A — Private GitHub repository directly

You can install from Git:

```bash
pip install git+ssh://git@github.com/company/ai-rag.git@v1.2.0
```

This can be useful for development or simple internal setups.

But I wouldn't make this the preferred enterprise distribution mechanism.

---

# 17. Better: Private package registry

A stronger architecture is:

```text
Developer
    │
    ▼
Private GitHub repository
    │
    ▼
CI/CD
    │
    ▼
Build Python wheel
    │
    ▼
Private package registry
    │
    ▼
Application
```

For example:

```text
ai-rag
   │
   │ GitHub Actions
   ▼
ai-rag-1.2.0.whl
   │
   ▼
Private package registry
   │
   ▼
financial-advisor
```

Your package remains private.

---

# 18. Why package registry is better

Suppose you have:

```text
financial-advisor
compliance-review
market-intelligence
```

All need:

```text
ai-rag
```

You don't want each application's CI pipeline cloning arbitrary Git repositories.

Instead:

```text
Private Registry

ai-platform-core
   1.0.0
   1.1.0

ai-rag
   1.0.0
   1.1.0
   1.2.0

ai-agents
   1.0.0
   1.1.0
```

Then:

```text
financial-advisor
    ↓
ai-rag==1.2.0
```

Very deterministic.

---

# 19. Your application repository

Now create:

```text
financial-advisor
```

This is **not a package repository**.

It's the actual product/application.

For example:

```text
financial-advisor/
│
├── pyproject.toml
│
├── src/
│   └── financial_advisor/
│       │
│       ├── api/
│       │
│       ├── application/
│       │
│       ├── domain/
│       │
│       ├── infrastructure/
│       │
│       └── workflow/
│
├── tests/
│
├── Dockerfile
│
├── helm/
│
└── terraform/
```

Dependencies:

```toml
dependencies = [
    "ai-platform-core==1.2.0",
    "ai-rag==1.5.0",
    "ai-agents==1.1.0",
    "ai-security==1.0.2",
    "ai-observability==1.3.0"
]
```

Now you have a real modular platform.

---

# 20. The architecture becomes

```text
                    PRIVATE GITHUB
                          │
        ┌─────────────────┼──────────────────┐
        │                 │                  │
        ▼                 ▼                  ▼
 ai-platform-core     ai-rag           ai-agents
        │                 │                  │
        │                 │                  │
        └────────────┬────┴──────────────────┘
                     │
                     ▼
              PRIVATE PACKAGE
                 REGISTRY
                     │
                     ▼
             financial-advisor
                     │
          ┌──────────┼───────────┐
          ▼          ▼           ▼
        FastAPI   LangGraph    Workers
          │          │           │
          └──────────┼───────────┘
                     ▼
                  AWS/K8s
```

That is the mental model I want you to develop.

---

# 21. Now add CI/CD to each package

For `ai-rag`:

```text
Developer
   │
   ▼
git push
   │
   ▼
GitHub Actions
   │
   ├── lint
   ├── type check
   ├── unit tests
   ├── integration tests
   ├── security scan
   ├── build wheel
   └── publish package
```

Example workflow conceptually:

```yaml
name: Build Package

on:
  push:
    tags:
      - "v*"

jobs:

  test:
    runs-on: ubuntu-latest

    steps:
      - uses: actions/checkout@v4

      - name: Setup Python
        uses: actions/setup-python@v5
        with:
          python-version: "3.12"

      - name: Install
        run: pip install -e ".[dev]"

      - name: Test
        run: pytest

      - name: Build
        run: python -m build
```

The important idea isn't the YAML itself.

It's:

```text
Code
 ↓
Test
 ↓
Build
 ↓
Version
 ↓
Publish
```

---

# 22. Private does not mean no automation

A common misunderstanding is:

> "If my GitHub repo is private, I can't use it as a package."

You absolutely can.

You can have:

```text
Private GitHub
       │
       ▼
GitHub Actions
       │
       ▼
Private package registry
       │
       ▼
Private applications
```

Everything can remain private.

---

# 23. Authentication becomes important

Your production environment should not use:

```text
username/password
```

embedded in:

```text
requirements.txt
```

or:

```text
pyproject.toml
```

Instead use:

```text
OIDC
GitHub Actions credentials
Cloud IAM
short-lived tokens
private registry authentication
```

For example:

```text
GitHub Actions
       │
       │ OIDC
       ▼
Cloud IAM
       │
       ▼
Private Registry
```

This is much better than storing long-lived credentials.

---

# 24. Don't put secrets into the repository

Never:

```text
OPENAI_API_KEY=xxxxx
```

inside:

```text
GitHub repository
```

Use:

```text
GitHub Secrets
AWS Secrets Manager
AWS Parameter Store
Kubernetes Secrets
Vault
```

depending on your deployment architecture.

Your package should consume configuration:

```python
settings.openai_api_key
```

but should not own the actual secret.

---

# 25. How your AI project should be divided

For your kind of enterprise AI projects, I would recommend something like:

```text
GitHub Organization
│
├── ai-platform-core
│
├── ai-llm
│
├── ai-rag
│
├── ai-agents
│
├── ai-tools
│
├── ai-security
│
├── ai-guardrails
│
├── ai-observability
│
├── ai-evaluation
│
├── ai-memory
│
├── ai-workflow
│
├── ai-data
│
└── financial-advisor
```

But **do not create all of these repositories immediately**.

That's another Staff-level lesson.

Start with boundaries that have real reuse.

---

# 26. A practical decomposition for your projects

Given your Financial Advisor + Compliance + Market Intelligence projects, I would start with:

```text
Private GitHub Organization
│
├── ai-platform-core
│
├── ai-rag
│
├── ai-agent-runtime
│
├── ai-security
│
├── ai-observability
│
├── ai-evaluation
│
├── financial-advisor
│
├── compliance-review
│
└── market-intelligence
```

Then:

```text
financial-advisor
      │
      ├── ai-platform-core
      ├── ai-rag
      ├── ai-agent-runtime
      ├── ai-security
      ├── ai-observability
      └── ai-evaluation
```

and:

```text
compliance-review
      │
      ├── ai-platform-core
      ├── ai-rag
      ├── ai-agent-runtime
      ├── ai-security
      ├── ai-observability
      └── ai-evaluation
```

This is where the real benefit appears.

---

# 27. What should NOT become a shared package

Suppose Financial Advisor has:

```text
PortfolioAgent
```

and Compliance has:

```text
ComplianceAgent
```

Don't put them into:

```text
ai-agent-runtime
```

just because they're both agents.

Instead:

```text
ai-agent-runtime
    ↓
generic agent infrastructure

financial-advisor
    ↓
PortfolioAgent

compliance-review
    ↓
ComplianceAgent
```

Shared package:

```text
HOW agents execute
```

Application:

```text
WHAT the agent does
```

That distinction is extremely important.

---

# 28. Think in terms of stable vs business-specific code

This is one of the best rules you can use.

### Stable infrastructure

Good candidate for package:

```text
RAG engine
Agent runtime
Security
Observability
Evaluation
LLM abstraction
```

### Business logic

Keep inside application:

```text
PortfolioAgent
ComplianceReviewer
FinancialRiskAgent
InvestmentRecommendation
PolicyRules
```

So:

```text
Reusable capability
        ↓
Private package

Business capability
        ↓
Application repository
```

---

# 29. Your dependency direction matters

You want:

```text
Application
     │
     ▼
Platform packages
     │
     ▼
Infrastructure adapters
```

Not:

```text
ai-rag
   │
   ▼
financial-advisor
```

That would create reverse coupling.

Your reusable packages should **never depend on a specific application**.

---

# 30. Avoid circular dependencies

Bad:

```text
ai-rag
  ↓
ai-agents
  ↓
ai-tools
  ↓
ai-rag
```

This becomes painful.

Instead design:

```text
ai-platform-core
       │
 ┌─────┼────────┐
 ▼     ▼        ▼
RAG   Agents   Security
```

or carefully define interfaces.

---

# 31. Internal API contracts

At Staff level, you should think about package APIs as contracts.

For example:

```python
class Retriever(Protocol):

    async def retrieve(
        self,
        query: str,
        *,
        top_k: int = 10
    ) -> list[Document]:
        ...
```

Applications depend on this contract.

You can internally replace:

```text
Qdrant
```

with:

```text
pgvector
```

without changing the application.

That is the real value of modularity.

---

# 32. Test packages independently

`ai-rag` should have:

```text
tests/
├── unit/
│   ├── test_chunker.py
│   ├── test_retriever.py
│   └── test_reranker.py
│
├── integration/
│   ├── test_qdrant.py
│   └── test_embeddings.py
│
└── contract/
    └── test_retriever_contract.py
```

Then application tests:

```text
financial-advisor/tests/
```

don't need to retest the entire RAG implementation.

They test:

```text
Does Financial Advisor use RAG correctly?
```

rather than:

```text
Does chunking work?
```

That's package ownership.

---

# 33. Package release process

Imagine you fix a bug in hybrid retrieval.

You change:

```text
ai-rag
```

from:

```text
1.5.0
```

to:

```text
1.5.1
```

CI:

```text
commit
 ↓
tests
 ↓
build
 ↓
1.5.1
 ↓
private registry
```

Then Financial Advisor deliberately upgrades:

```text
ai-rag==1.5.0
```

to:

```text
ai-rag==1.5.1
```

and runs:

```text
application tests
```

before deployment.

That gives you controlled dependency upgrades.

---

# 34. What happens when there is a breaking change?

Suppose:

```python
retrieve(query)
```

becomes:

```python
retrieve(request)
```

Don't silently change the existing API.

Create:

```text
ai-rag 2.0.0
```

Then applications can migrate deliberately.

You might temporarily have:

```text
financial-advisor → ai-rag 1.x
compliance-review → ai-rag 2.x
```

This is one of the major advantages of package versioning.

---

# 35. Monorepo vs multi-repository

You asked specifically about GitHub repositories, so this is important.

There are two legitimate architectures.

### Monorepo

```text
ai-platform/
│
├── core/
├── rag/
├── agents/
├── security/
└── applications/
```

One repository.

### Polyrepo

```text
ai-platform-core
ai-rag
ai-agents
ai-security
financial-advisor
compliance-review
```

Multiple repositories.

Neither is automatically better.

---

# 36. When I would choose monorepo

Use monorepo when:

```text
small team
rapid development
highly coupled modules
frequent coordinated changes
```

Example:

```text
10 engineers
3 applications
```

A monorepo may be simpler.

---

# 37. When I would choose polyrepo

Use separate private repositories when:

```text
multiple teams
independent releases
different ownership
strong reuse
separate deployment lifecycles
security boundaries
```

For an enterprise AI platform, polyrepo can make sense when the platform packages have genuinely independent lifecycles.

---

# 38. Don't over-modularize

This is something I especially want you to remember.

Do NOT create:

```text
ai-string-utils
ai-json-utils
ai-date-utils
ai-prompt-utils
ai-token-utils
```

just because you can.

You will end up with:

```text
application
  ↓
15 internal packages
  ↓
42 transitive dependencies
```

and nobody knows what depends on what.

A package should exist because it provides a **meaningful capability or ownership boundary**.

---

# 39. A good dependency graph

I'd aim for something like:

```text
                     ┌─────────────────────┐
                     │ ai-platform-core    │
                     └──────────┬──────────┘
                                │
             ┌──────────────────┼──────────────────┐
             │                  │                  │
             ▼                  ▼                  ▼
         ai-rag          ai-agent-runtime    ai-security
             │                  │                  │
             └────────────┬─────┴──────────────────┘
                          │
                ┌─────────┴──────────┐
                ▼                    ▼
       ai-observability       ai-evaluation
                │                    │
                └─────────┬──────────┘
                          ▼
              ┌───────────────────────┐
              │ Financial Applications │
              └───────────────────────┘
```

Keep dependencies mostly flowing **downward**.

---

# 40. The Staff Engineer checklist

When creating a private AI package, ask:

### Ownership

```text
Who owns this package?
```

### Responsibility

```text
What problem does it solve?
```

### API

```text
What public interfaces does it expose?
```

### Dependencies

```text
What does it depend on?
```

### Stability

```text
How often will the API change?
```

### Versioning

```text
How will breaking changes be handled?
```

### Security

```text
Who can install it?
```

### Testing

```text
Does it have unit/integration/contract tests?
```

### CI/CD

```text
How is it built and released?
```

### Observability

```text
How can consumers observe failures/performance?
```

### Documentation

```text
Can another engineer use it without asking me?
```

If you cannot answer these, the package isn't production-ready yet.

---

# 41. The complete lifecycle

This is the mental model I recommend you memorize:

```text
                  DEVELOPMENT
                       │
                       ▼
             Private GitHub Repo
                       │
                       ▼
                  Pull Request
                       │
          ┌────────────┼────────────┐
          ▼            ▼            ▼
        Lint          Tests       Security
          │            │            │
          └────────────┼────────────┘
                       ▼
                    Merge
                       │
                       ▼
                   Git Tag
                  v1.4.0
                       │
                       ▼
                GitHub Actions
                       │
                       ▼
                  Build Wheel
                       │
                       ▼
             Private Package Registry
                       │
                       ▼
             Application Dependency
                       │
                       ▼
                Integration Tests
                       │
                       ▼
                 Docker Image
                       │
                       ▼
                 Kubernetes/AWS
```

That's the **enterprise package supply chain**.

---

# 42. For your AI projects specifically

I would eventually structure your private GitHub organization roughly like this:

```text
PRIVATE GITHUB ORGANIZATION
│
├── platform
│   ├── ai-platform-core
│   ├── ai-rag
│   ├── ai-agent-runtime
│   ├── ai-security
│   ├── ai-observability
│   └── ai-evaluation
│
├── applications
│   ├── financial-advisor
│   ├── compliance-review
│   └── market-intelligence
│
└── infrastructure
    ├── aws-infrastructure
    ├── kubernetes-platform
    └── github-actions-templates
```

All repositories can remain:

```text
PRIVATE
```

This gives you a much stronger story in a Staff AI Engineer interview:

> "We separated reusable AI platform capabilities from domain-specific applications. Platform components were published as versioned internal Python packages through a private package registry, while applications consumed pinned versions. We enforced API contracts, semantic versioning, CI/CD, automated testing, security scanning, and dependency governance."

That's a **much stronger architectural answer** than saying:

> "I created multiple GitHub repositories and imported the code."

---

## The most important distinction

Think of it as:

```text
GitHub Repository
       ↓
Source ownership
       ↓
CI/CD
       ↓
Versioned Package
       ↓
Private Registry
       ↓
Application
```

And:

```text
Package = reusable capability

Application = business solution
```

For your projects, **RAG, agent runtime, security, observability and evaluation are candidates for reusable packages; Portfolio Agent, Compliance Reviewer, Investment Analyst, financial rules and business workflows belong to the individual applications.**

That boundary is what makes the design genuinely **modular**, rather than merely splitting one codebase into folders.
