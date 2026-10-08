Absolutely. For a **Senior/Staff AI Engineer**, you should understand the Adapter Pattern beyond the textbook definition. In production AI systems, adapters are one of the main mechanisms used to prevent your business/application layer from becoming tightly coupled to vendors such as OpenAI, Anthropic, AWS Bedrock, Qdrant, pgvector, Redis, Kafka, MLflow, etc.

The core idea is:

> **An Adapter translates an external system's interface into the interface your application expects.**

---

# 1. The problem Adapter Pattern solves

Imagine your application wants to generate an answer.

You define what your application needs:

```python
result = await llm.generate(
    prompt="Analyze this portfolio"
)
```

But OpenAI's SDK might expose:

```python
client.responses.create(...)
```

Anthropic might expose:

```python
client.messages.create(...)
```

Bedrock might expose:

```python
bedrock_runtime.invoke_model(...)
```

vLLM might expose an OpenAI-compatible HTTP API.

If your application directly uses all these SDKs:

```text
                    Application
                         │
              ┌──────────┼───────────┐
              ▼          ▼           ▼
           OpenAI     Anthropic    Bedrock
             SDK         SDK          SDK
```

your application becomes vendor-dependent.

That's bad.

Instead:

```text
                  Application
                       │
                       ▼
                  LLM Port
                       │
        ┌──────────────┼──────────────┐
        ▼              ▼              ▼
 OpenAI Adapter   Anthropic Adapter  Bedrock Adapter
        │              │              │
        ▼              ▼              ▼
    OpenAI SDK      Anthropic SDK    AWS SDK
```

Now the application depends on **your interface**, not the vendor SDK.

That is the Adapter Pattern.

---

# 2. Adapter in one sentence

Suppose your application expects:

```python
await llm.generate(prompt)
```

but an external service provides:

```python
client.responses.create(...)
```

The adapter converts:

```text
YOUR APPLICATION INTERFACE
           ↓
        Adapter
           ↓
VENDOR-SPECIFIC INTERFACE
```

So:

```text
Application
    ↓
LLMPort
    ↓
OpenAIAdapter
    ↓
OpenAI SDK
```

---

# 3. Adapter is not the same as Strategy

This is very important given what we've just discussed.

### Adapter

Answers:

> **"How do I make this external implementation fit my application's interface?"**

Example:

```text
LLMPort
   ↑
OpenAIAdapter
```

### Strategy

Answers:

> **"Which algorithm/behavior should I use?"**

Example:

```text
RetrievalStrategy
       │
 ┌─────┼──────┐
 ▼     ▼      ▼
BM25 Vector Hybrid
```

You can use both simultaneously.

For example:

```text
RAG Service
    │
    ▼
Retrieval Strategy
    │
    ▼
Vector Retrieval
    │
    ▼
VectorStore Port
    │
    ▼
Qdrant Adapter
    │
    ▼
Qdrant
```

Here:

* **Strategy** = retrieval behavior
* **Adapter** = Qdrant integration

---

# 4. Start with a production-style LLM Port

For your AI platform, I would not expose vendor SDK objects throughout the application.

Create your own domain/application contract.

For example:

```python
# ai_llm/ports.py

from typing import Protocol

from ai_llm.models import (
    ChatRequest,
    ChatResponse,
)


class LLMProvider(Protocol):

    async def generate(
        self,
        request: ChatRequest,
    ) -> ChatResponse:
        ...
```

Now define your own models.

```python
# ai_llm/models.py

from dataclasses import dataclass


@dataclass(frozen=True)
class ChatRequest:

    messages: list[dict[str, str]]

    model: str | None = None

    temperature: float = 0.0

    max_tokens: int | None = None


@dataclass(frozen=True)
class ChatResponse:

    text: str

    model: str

    input_tokens: int

    output_tokens: int

    finish_reason: str | None = None
```

Notice something important:

Your application doesn't know:

```text
OpenAI response object
Anthropic Message object
Bedrock response JSON
```

It knows only:

```text
ChatRequest
ChatResponse
```

This is a major production design principle.

---

# 5. OpenAI Adapter

Now implement the adapter.

```python
from openai import AsyncOpenAI

from ai_llm.models import ChatRequest, ChatResponse
from ai_llm.ports import LLMProvider


class OpenAIAdapter:

    def __init__(
        self,
        client: AsyncOpenAI,
        default_model: str,
    ):
        self.client = client
        self.default_model = default_model

    async def generate(
        self,
        request: ChatRequest,
    ) -> ChatResponse:

        model = (
            request.model
            or self.default_model
        )

        response = await self.client.responses.create(
            model=model,
            input=request.messages,
        )

        return ChatResponse(
            text=response.output_text,
            model=model,
            input_tokens=(
                response.usage.input_tokens
            ),
            output_tokens=(
                response.usage.output_tokens
            ),
            finish_reason=None,
        )
```

This is an adapter because it translates:

```text
Our ChatRequest
       ↓
OpenAI request format
       ↓
OpenAI SDK
       ↓
OpenAI response
       ↓
Our ChatResponse
```

---

# 6. Anthropic Adapter

Now suppose the company wants Claude.

Don't modify the application.

Create:

```python
from anthropic import AsyncAnthropic

from ai_llm.models import ChatRequest, ChatResponse


class AnthropicAdapter:

    def __init__(
        self,
        client: AsyncAnthropic,
        default_model: str,
    ):
        self.client = client
        self.default_model = default_model

    async def generate(
        self,
        request: ChatRequest,
    ) -> ChatResponse:

        model = (
            request.model
            or self.default_model
        )

        response = await self.client.messages.create(
            model=model,
            messages=request.messages,
            max_tokens=request.max_tokens or 1024,
            temperature=request.temperature,
        )

        return ChatResponse(
            text=response.content[0].text,
            model=model,
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
            finish_reason=response.stop_reason,
        )
```

Now:

```text
                 LLMProvider
                     ▲
                     │
        ┌────────────┼─────────────┐
        │            │             │
        ▼            ▼             ▼
    OpenAI       Anthropic      Bedrock
    Adapter       Adapter       Adapter
```

---

# 7. Why this matters in production

Imagine your application contains:

```python
from openai import AsyncOpenAI
```

in:

```text
ComplianceService
RiskAgent
ResearchAgent
WriterAgent
PortfolioAgent
EvaluationService
```

You've now coupled your entire application to OpenAI.

If you need to migrate to Bedrock:

```text
ComplianceService
RiskAgent
ResearchAgent
WriterAgent
PortfolioAgent
EvaluationService
```

all potentially need changes.

That's expensive.

With adapters:

```text
ComplianceService
RiskAgent
ResearchAgent
WriterAgent
        │
        ▼
   LLMProvider
        │
        ▼
   LLM Router
        │
   ┌────┼─────┐
   ▼    ▼     ▼
OpenAI Claude Bedrock
Adapter Adapter Adapter
```

Vendor changes are isolated.

---

# 8. Adapter doesn't mean "just wrap an SDK"

This is where Staff-level understanding matters.

A weak adapter is:

```python
class OpenAIAdapter:

    def generate(self, prompt):
        return openai.generate(prompt)
```

A production adapter usually handles:

```text
API translation
Authentication
Timeouts
Retries
Rate limits
Error translation
Observability
Token usage
Cost tracking
Tracing
Structured output
Streaming
Provider-specific capabilities
```

For example:

```text
Application
    │
    ▼
LLMPort
    │
    ▼
OpenAIAdapter
    │
    ├── timeout
    ├── retry
    ├── rate-limit handling
    ├── tracing
    ├── token metrics
    ├── cost metrics
    ├── error mapping
    └── request translation
         │
         ▼
      OpenAI
```

That's much closer to production.

---

# 9. Error translation

This is a particularly important responsibility.

Suppose OpenAI throws:

```python
openai.RateLimitError
```

Your application shouldn't have to know about:

```python
openai.RateLimitError
```

Instead, your adapter translates it into your platform error.

For example:

```python
class LLMRateLimitError(Exception):
    pass


class LLMTimeoutError(Exception):
    pass


class LLMProviderError(Exception):
    pass
```

Then:

```python
class OpenAIAdapter:

    async def generate(self, request):

        try:

            response = await self.client.responses.create(
                ...
            )

            return self._map_response(response)

        except openai.RateLimitError as exc:

            raise LLMRateLimitError(
                "LLM provider rate limit exceeded"
            ) from exc

        except openai.APITimeoutError as exc:

            raise LLMTimeoutError(
                "LLM provider timed out"
            ) from exc

        except Exception as exc:

            raise LLMProviderError(
                "LLM provider failure"
            ) from exc
```

Now your application deals with:

```text
LLMRateLimitError
LLMTimeoutError
LLMProviderError
```

rather than vendor-specific exceptions.

That's a strong boundary.

---

# 10. Why error translation is so important

Your supervisor might have logic like:

```python
try:

    result = await llm.generate(request)

except LLMRateLimitError:

    # route to fallback provider

except LLMTimeoutError:

    # retry or fallback

except LLMProviderError:

    # mark workflow failure
```

The supervisor doesn't care whether the failure came from:

```text
OpenAI
Anthropic
Bedrock
vLLM
```

That information remains inside the adapter layer.

---

# 11. Provider fallback

Now we can build a production-style router.

Suppose:

```text
Primary → OpenAI
Fallback → Bedrock
```

You can implement:

```python
class LLMRouter:

    def __init__(
        self,
        primary: LLMProvider,
        fallback: LLMProvider,
    ):
        self.primary = primary
        self.fallback = fallback

    async def generate(
        self,
        request: ChatRequest,
    ) -> ChatResponse:

        try:

            return await self.primary.generate(
                request
            )

        except (
            LLMRateLimitError,
            LLMTimeoutError,
        ):

            return await self.fallback.generate(
                request
            )
```

Architecture:

```text
                 Application
                      │
                      ▼
                  LLMRouter
                      │
              ┌───────┴───────┐
              ▼               ▼
           Primary         Fallback
              │               │
              ▼               ▼
        OpenAI Adapter   Bedrock Adapter
```

Notice:

**Router is not an adapter.**

The router decides **which provider** to use.

The adapters translate each provider's interface.

This is where multiple design patterns work together.

---

# 12. Adapter + Strategy + Factory

A mature AI platform might have:

```text
                    LLMProvider
                         ▲
                         │
          ┌──────────────┼──────────────┐
          │              │              │
          ▼              ▼              ▼
     OpenAIAdapter   ClaudeAdapter   BedrockAdapter
          ▲              ▲              ▲
          └──────────────┼──────────────┘
                         │
                     LLM Router
                         ▲
                         │
                       Factory
```

Different responsibilities:

### Factory

Creates the object.

```text
Factory
   ↓
OpenAIAdapter
```

### Adapter

Translates interfaces.

```text
OpenAIAdapter
   ↓
OpenAI SDK
```

### Strategy

Represents interchangeable behavior.

```text
RoutingStrategy
   ├── CostOptimized
   ├── LatencyOptimized
   └── QualityOptimized
```

### Router

Selects the provider.

```text
Router
   ↓
OpenAI / Claude / Bedrock
```

---

# 13. Adapter in RAG

Adapters aren't only for LLMs.

Your RAG system is an excellent example.

Your application might define:

```python
from typing import Protocol


class VectorStore(Protocol):

    async def similarity_search(
        self,
        query_embedding: list[float],
        top_k: int,
    ) -> list["Document"]:
        ...
```

Now you can implement:

```text
VectorStore
     ▲
     │
 ┌───┼──────────────┐
 ▼   ▼              ▼
pgvector Qdrant   Pinecone
Adapter  Adapter   Adapter
```

---

# 14. pgvector Adapter

```python
class PgVectorAdapter:

    def __init__(self, session):
        self.session = session

    async def similarity_search(
        self,
        query_embedding,
        top_k,
    ):

        rows = await self.session.execute(
            """
            SELECT
                content,
                metadata
            FROM documents
            ORDER BY embedding <=> :embedding
            LIMIT :top_k
            """,
            {
                "embedding": query_embedding,
                "top_k": top_k,
            },
        )

        return [
            Document(
                content=row.content,
                metadata=row.metadata,
            )
            for row in rows
        ]
```

The application doesn't know:

```text
<=> operator
SQL
PostgreSQL
pgvector
```

It just calls:

```python
await vector_store.similarity_search(...)
```

---

# 15. Qdrant Adapter

```python
class QdrantAdapter:

    def __init__(self, client, collection):
        self.client = client
        self.collection = collection

    async def similarity_search(
        self,
        query_embedding,
        top_k,
    ):

        results = await self.client.search(
            collection_name=self.collection,
            query_vector=query_embedding,
            limit=top_k,
        )

        return [
            Document(
                content=result.payload["content"],
                metadata=result.payload["metadata"],
            )
            for result in results
        ]
```

Again:

```text
Application
     │
     ▼
VectorStore
     │
 ┌───┴────┐
 ▼        ▼
pgvector Qdrant
Adapter  Adapter
```

---

# 16. Embedding Adapter

Same concept.

Application:

```python
class EmbeddingProvider(Protocol):

    async def embed(
        self,
        texts: list[str],
    ) -> list[list[float]]:
        ...
```

Adapters:

```text
EmbeddingProvider
        ▲
        │
 ┌──────┼────────┐
 ▼      ▼        ▼
OpenAI  Bedrock  HuggingFace
Adapter Adapter  Adapter
```

Now your ingestion pipeline:

```text
Document
   ↓
Parser
   ↓
Chunker
   ↓
EmbeddingProvider
   ↓
VectorStore
```

doesn't care which embedding vendor you selected.

---

# 17. Adapter in document storage

You may also have:

```python
class ObjectStorage(Protocol):

    async def upload(
        self,
        key: str,
        data: bytes,
    ) -> str:
        ...
```

Adapters:

```text
ObjectStorage
     ▲
     │
 ┌───┼────────┐
 ▼   ▼        ▼
S3  GCS     Azure Blob
```

Your application simply does:

```python
uri = await storage.upload(
    "documents/policy.pdf",
    pdf_bytes
)
```

---

# 18. Adapter in Redis

Define:

```python
class CachePort(Protocol):

    async def get(
        self,
        key: str
    ) -> str | None:
        ...

    async def set(
        self,
        key: str,
        value: str,
        ttl: int
    ) -> None:
        ...
```

Adapter:

```python
class RedisAdapter:

    def __init__(self, client):
        self.client = client

    async def get(self, key):

        return await self.client.get(key)

    async def set(
        self,
        key,
        value,
        ttl,
    ):

        await self.client.set(
            key,
            value,
            ex=ttl,
        )
```

Now your application isn't coupled to Redis.

---

# 19. Adapter in messaging

Your application may need:

```python
class EventPublisher(Protocol):

    async def publish(
        self,
        event: dict,
    ) -> None:
        ...
```

Adapters:

```text
EventPublisher
      ▲
      │
 ┌────┼─────────┐
 ▼    ▼         ▼
Kafka Redis     SNS/SQS
```

Your business logic:

```python
await publisher.publish(
    {
        "event": "review.completed",
        "review_id": review_id,
    }
)
```

doesn't know which messaging technology is used.

---

# 20. Adapter + Hexagonal Architecture

Now connect this to what we discussed earlier.

Hexagonal Architecture says:

```text
                 APPLICATION CORE
                       │
                 ┌─────┴─────┐
                 │   Ports   │
                 └─────┬─────┘
                       │
                  Adapters
                       │
                Infrastructure
```

So adapters are **one of the major implementation mechanisms of Hexagonal Architecture**.

For example:

```text
             Compliance Application
                      │
             ┌────────┴────────┐
             │                 │
             ▼                 ▼
        LLMProvider        RepositoryPort
             │                 │
             ▼                 ▼
       OpenAIAdapter      PostgresAdapter
             │                 │
             ▼                 ▼
          OpenAI           PostgreSQL
```

---

# 21. Inbound adapters

This is another concept you should understand.

Adapters aren't always outbound.

## Outbound adapter

Application → external system.

```text
Application
    ↓
LLMPort
    ↓
OpenAIAdapter
    ↓
OpenAI
```

## Inbound adapter

External client → application.

```text
HTTP Client
    ↓
FastAPI Adapter
    ↓
Application Use Case
```

So FastAPI itself can be considered an **inbound adapter**.

Similarly:

```text
Kafka Consumer
Celery Worker
CLI
gRPC
WebSocket
```

can be inbound adapters.

---

# 22. Production AI architecture

Now let's put all of this together.

Your financial compliance platform might look like:

```text
                        Client
                          │
                          ▼
                   FastAPI Adapter
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
                    Domain Logic
                          │
         ┌────────────────┼─────────────────┐
         │                │                 │
         ▼                ▼                 ▼
      LLM Port       Vector Port      Repository Port
         │                │                 │
         ▼                ▼                 ▼
    LLM Router       Retrieval        PostgreSQL
         │                │
    ┌────┼────┐      ┌────┴─────┐
    ▼    ▼    ▼      ▼          ▼
 OpenAI Claude Bedrock pgvector Qdrant
 Adapter Adapter Adapter Adapter Adapter
```

This is a very realistic Staff-level architecture.

---

# 23. Add Guardrails

You can also define:

```python
class GuardrailPort(Protocol):

    async def validate_input(
        self,
        text: str,
    ) -> None:
        ...

    async def validate_output(
        self,
        output: str,
    ) -> None:
        ...
```

Then:

```text
GuardrailPort
     ▲
     │
 ┌───┴────────────┐
 ▼                ▼
PII Adapter   Injection Adapter
```

But here I'd be careful with terminology.

Not everything needs to be an adapter.

A pure business/policy validator can simply be a domain/application service.

An adapter is most useful when crossing a technical/external boundary.

---

# 24. HITL adapter

Your production HITL system might need a checkpoint store:

```python
class CheckpointStore(Protocol):

    async def save(
        self,
        workflow_id: str,
        state: dict,
    ) -> None:
        ...

    async def load(
        self,
        workflow_id: str,
    ) -> dict | None:
        ...
```

Implementation:

```text
CheckpointStore
      ▲
      │
 ┌────┼───────────┐
 ▼    ▼           ▼
Postgres Redis LangGraph
Adapter Adapter  Adapter
```

For durable enterprise workflows, PostgreSQL would generally be the stronger source of truth; Redis can be useful for ephemeral/cache workloads.

---

# 25. Production adapter responsibilities

For a serious AI platform, I would expect adapters to handle at least these concerns.

### 1. Protocol translation

```text
Your request → Vendor request
Vendor response → Your response
```

### 2. Error normalization

```text
VendorError → PlatformError
```

### 3. Timeout

```python
asyncio.wait_for(...)
```

or provider/client timeout configuration.

### 4. Retry

For transient failures:

```text
429
5xx
network timeout
connection reset
```

But retries should be controlled and idempotent.

### 5. Observability

Capture:

```text
trace_id
request_id
workflow_id
provider
model
latency
input tokens
output tokens
cost
error
```

### 6. Security

Never leak:

```text
API keys
authorization headers
PII
sensitive prompts
```

into logs.

### 7. Rate limiting

Provider-specific limits can be enforced around the adapter or by a shared gateway/router.

### 8. Capability normalization

Providers have different capabilities.

For example:

```text
Provider A → structured output
Provider B → no structured output
Provider C → tool calling
Provider D → different streaming API
```

Your adapter can expose a normalized capability model.

---

# 26. Capability modeling

This is an advanced but useful Staff-level concept.

Don't pretend every provider is identical.

Define:

```python
from dataclasses import dataclass


@dataclass(frozen=True)
class ModelCapabilities:

    supports_tools: bool
    supports_json_schema: bool
    supports_streaming: bool
    supports_vision: bool
    max_context_tokens: int
```

Your provider adapter exposes:

```python
class LLMProvider(Protocol):

    @property
    def capabilities(self) -> ModelCapabilities:
        ...

    async def generate(
        self,
        request: ChatRequest,
    ) -> ChatResponse:
        ...
```

Now the router can make intelligent decisions.

For example:

```python
if request.requires_tools:
    provider = providers.supporting("tools")
```

This is much better than blindly assuming all LLMs behave identically.

---

# 27. Streaming is another adapter boundary

Your abstraction may expose:

```python
class LLMProvider(Protocol):

    async def stream(
        self,
        request: ChatRequest,
    ) -> AsyncIterator[LLMChunk]:
        ...
```

OpenAI might produce one form of streaming event.

Anthropic produces another.

Your adapters normalize both:

```text
OpenAI events
     ↓
OpenAIAdapter
     ↓
LLMChunk
```

and:

```text
Anthropic events
     ↓
AnthropicAdapter
     ↓
LLMChunk
```

Your application sees:

```python
async for chunk in provider.stream(request):
    yield chunk
```

This is exactly what an abstraction boundary should achieve.

---

# 28. Don't over-normalize

There is a subtle Staff-level problem here.

You should **not** force every provider into the lowest common denominator.

Suppose:

```text
OpenAI → reasoning controls
Claude → special thinking configuration
Bedrock → provider-specific parameters
```

If your interface only exposes:

```python
generate(prompt)
```

you may lose important capabilities.

A better design is often:

```python
@dataclass
class ChatRequest:

    messages: list[Message]

    model: str

    temperature: float | None = None

    max_tokens: int | None = None

    tools: list[ToolDefinition] = field(
        default_factory=list
    )

    response_format: ResponseFormat | None = None

    provider_options: dict[str, Any] = field(
        default_factory=dict
    )
```

This gives you:

```text
Common capabilities
        +
Explicit provider-specific extension point
```

without leaking SDK types.

---

# 29. Adapter boundaries should be stable

A good adapter boundary should remain stable even if the vendor changes.

For example:

```python
await llm.generate(
    ChatRequest(...)
)
```

should remain stable if you change:

```text
OpenAI
→ Bedrock
```

or:

```text
Qdrant
→ pgvector
```

or:

```text
Redis
→ another cache
```

The business application should not need to change.

---

# 30. Testing adapters

There are two different things to test.

## Test application logic

Use a fake implementation:

```python
class FakeLLM:

    async def generate(
        self,
        request
    ):

        return ChatResponse(
            text="approved",
            model="fake",
            input_tokens=10,
            output_tokens=5,
        )
```

Then:

```python
service = ComplianceService(
    llm=FakeLLM()
)
```

No network call.

---

## Test the adapter itself

Then separately test:

```text
OpenAIAdapter
```

against:

* mocked SDK responses
* provider sandbox where appropriate
* contract tests

For example:

```python
async def test_openai_adapter_maps_response():

    fake_openai_response = ...

    client = FakeOpenAIClient(
        response=fake_openai_response
    )

    adapter = OpenAIAdapter(
        client=client,
        default_model="..."
    )

    result = await adapter.generate(
        ChatRequest(...)
    )

    assert result.text == "hello"
```

This separates:

```text
Application tests
```

from:

```text
Infrastructure adapter tests
```

---

# 31. Adapter contract tests

This is a very strong enterprise pattern.

Suppose you have:

```text
OpenAIAdapter
AnthropicAdapter
BedrockAdapter
```

All must satisfy:

```text
LLMProvider contract
```

You can create common tests:

```python
async def assert_llm_provider_contract(
    provider: LLMProvider
):

    result = await provider.generate(
        ChatRequest(
            messages=[
                {
                    "role": "user",
                    "content": "Hello"
                }
            ]
        )
    )

    assert result.text
    assert result.model
    assert result.input_tokens >= 0
    assert result.output_tokens >= 0
```

Then run it against each adapter.

This catches inconsistent provider behavior.

---

# 32. How this fits your modular packages

Your modular architecture could be:

```text
packages/
│
├── ai-llm/
│   └── src/ai_llm/
│       ├── models.py
│       ├── ports.py
│       ├── router.py
│       │
│       └── adapters/
│           ├── openai.py
│           ├── anthropic.py
│           ├── bedrock.py
│           └── vllm.py
│
├── ai-rag/
│   └── src/ai_rag/
│       ├── ports.py
│       ├── models.py
│       │
│       └── adapters/
│           ├── pgvector.py
│           ├── qdrant.py
│           └── pinecone.py
│
├── platform-cache/
│   └── src/platform_cache/
│       ├── ports.py
│       └── adapters/
│           └── redis.py
│
└── platform-messaging/
    └── src/platform_messaging/
        ├── ports.py
        └── adapters/
            ├── kafka.py
            └── sqs.py
```

Then:

```text
applications/
│
├── financial-compliance/
│
└── investment-intelligence/
```

Both applications consume the packages.

---

# 33. The dependency direction

This is the most important architectural rule.

You want:

```text
Application
     │
     ▼
     Port
     ▲
     │
   Adapter
     │
     ▼
External System
```

NOT:

```text
Application
     │
     ▼
OpenAI SDK
     │
     ▼
Business Logic
```

And ideally not:

```text
Domain
   ↓
OpenAI SDK
```

The dependency should point inward toward your abstraction.

---

# 34. A complete production request

Let's follow an actual request through your financial compliance platform.

Request:

```http
POST /v1/reviews
```

```json
{
  "material": "This investment is guaranteed and risk-free."
}
```

Flow:

```text
Client
  │
  ▼
FastAPI Adapter
  │
  ▼
ComplianceReviewService
  │
  ▼
LangGraph Workflow
  │
  ├── Guardrails
  │
  ├── Planner
  │
  ├── Supervisor
  │
  ├── Research Agent
  │       │
  │       ▼
  │   Retrieval Strategy
  │       │
  │       ▼
  │   VectorStore Port
  │       │
  │       ▼
  │   PgVector Adapter
  │       │
  │       ▼
  │   PostgreSQL/pgvector
  │
  ├── Risk Agent
  │       │
  │       ▼
  │    LLM Port
  │       │
  │       ▼
  │    LLM Router
  │       │
  │       ├── OpenAI Adapter
  │       ├── Claude Adapter
  │       └── Bedrock Adapter
  │
  ├── Evaluator
  │
  ├── HITL
  │
  └── Repository Port
          │
          ▼
    PostgreSQL Adapter
```

That is where the Adapter Pattern becomes genuinely valuable.

---

# 35. What NOT to do

### Bad

```python
class ComplianceAgent:

    async def run(self):

        from openai import AsyncOpenAI

        client = AsyncOpenAI()

        response = await client.responses.create(
            ...
        )
```

Now your agent knows about OpenAI.

---

### Better

```python
class ComplianceAgent:

    def __init__(
        self,
        llm: LLMProvider,
    ):
        self.llm = llm

    async def run(self, request):

        return await self.llm.generate(
            request
        )
```

Now the agent depends on an abstraction.

---

# 36. The Staff Engineer mental model

When designing a production AI system, ask:

### "What external dependency am I crossing?"

Examples:

```text
LLM
Vector DB
Embedding service
Database
Cache
Object storage
Messaging
Identity provider
Evaluation platform
Observability platform
```

Then ask:

> "Do I want my business/application layer to know the vendor's SDK/API?"

Usually:

**No.**

Create a stable port:

```text
LLMProvider
VectorStore
EmbeddingProvider
Repository
Cache
EventPublisher
ObjectStorage
```

Then create adapters:

```text
OpenAIAdapter
QdrantAdapter
PgVectorAdapter
RedisAdapter
S3Adapter
KafkaAdapter
```

---

# 37. The strongest interview explanation

If you're asked:

> **"How do you use the Adapter Pattern in a production AI system?"**

A Staff-level answer would be:

> "We use adapters at infrastructure boundaries to isolate vendor-specific APIs from the application core. For example, our application depends on an `LLMProvider` contract that exposes normalized request and response models. OpenAI, Anthropic, Bedrock and vLLM each have separate adapters that translate our internal contract into their SDK/API format and normalize responses and exceptions back into platform types. The adapters also provide provider-specific concerns such as timeout handling, retryable error mapping, token telemetry and capability detection. This lets our agents and workflows remain provider-agnostic and allows us to introduce new providers without changing business logic."

Then add:

> "We use the same pattern for vector stores, embeddings, object storage, caching and messaging. Strategy is used above that boundary when we need to select interchangeable behavior, such as vector versus BM25 versus hybrid retrieval. So Strategy determines **which behavior**, while Adapter handles **how an external implementation conforms to our interface**."

That is the distinction I would expect from a **Senior/Staff AI Engineer**.

---

## The architecture to remember

```text
                 BUSINESS APPLICATION
                         │
                         ▼
                 ┌──────────────┐
                 │     PORT     │
                 │              │
                 │ LLMProvider  │
                 │ VectorStore  │
                 │ Repository   │
                 │ Cache        │
                 └──────┬───────┘
                        │
                    ADAPTERS
                        │
       ┌────────────────┼─────────────────┐
       ▼                ▼                 ▼
 OpenAIAdapter     QdrantAdapter    PostgresAdapter
       │                │                 │
       ▼                ▼                 ▼
    OpenAI            Qdrant          PostgreSQL
```

And when you need to choose **between different ways of performing the same operation**:

```text
             RetrievalService
                    │
                    ▼
          RetrievalStrategy
                    │
        ┌───────────┼───────────┐
        ▼           ▼           ▼
      BM25        Vector       Hybrid
     Strategy    Strategy     Strategy
```

So the clean Staff-level combination is:

> **Hexagonal Architecture gives you the boundary → Ports define the contracts → Adapters integrate external systems → Strategy selects interchangeable behavior → Dependency Injection wires everything together.**
