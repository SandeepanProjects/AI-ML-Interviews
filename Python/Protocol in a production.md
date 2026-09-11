Yes. In a **production AI project**, Python `Protocol` is mainly used to create **interfaces/contracts between components without forcing those components to inherit from a common base class**.

For the kind of AI systems you're preparing for—**FastAPI + RAG + LangGraph + LLMs + vector DB + Redis + PostgreSQL**—`Protocol` is particularly useful at the **service/architecture boundaries**.

---

# 1. What is `Protocol`?

Python's `Protocol` comes from:

```python
from typing import Protocol
```

It enables **structural typing** (also called duck typing with static type checking).

Instead of saying:

> "This class must inherit from `LLMProvider`"

you say:

> "Any class that has these methods can be treated as an `LLMProvider`."

Example:

```python
from typing import Protocol


class LLMProvider(Protocol):

    async def generate(
        self,
        prompt: str
    ) -> str:
        ...
```

Now any class implementing `generate()` automatically satisfies this protocol.

```python
class OpenAIProvider:

    async def generate(self, prompt: str) -> str:
        return "Answer from OpenAI"
```

No inheritance is required.

```python
class AnthropicProvider:

    async def generate(self, prompt: str) -> str:
        return "Answer from Claude"
```

Also no inheritance.

Both satisfy:

```python
LLMProvider
```

from the perspective of a type checker such as `mypy` or Pyright.

---

# 2. Why is this useful in production AI?

Imagine your AI application has:

```text
FastAPI
   ↓
Service
   ↓
LLM
   ↓
OpenAI
```

Initially you might write:

```python
from openai import AsyncOpenAI


class AIService:

    def __init__(self):
        self.client = AsyncOpenAI()

    async def answer(self, prompt: str):
        response = await self.client.chat.completions.create(...)
        return response
```

This creates a problem.

Your business logic is now tightly coupled to OpenAI.

Later you might want:

```text
OpenAI
Claude
Azure OpenAI
AWS Bedrock
Llama
vLLM
Mock LLM
```

Instead of changing your business logic every time, define a contract.

```python
class LLMProvider(Protocol):

    async def generate(
        self,
        prompt: str
    ) -> str:
        ...
```

Then your service depends on the **contract**, not the vendor.

---

# 3. Production AI example: LLM provider

A realistic structure:

```text
app/
├── api/
│   └── chat.py
│
├── services/
│   └── chat_service.py
│
├── providers/
│   ├── openai_provider.py
│   ├── anthropic_provider.py
│   └── mock_provider.py
│
├── protocols/
│   └── llm.py
│
└── main.py
```

---

## `protocols/llm.py`

```python
from typing import Protocol


class LLMProvider(Protocol):

    async def generate(
        self,
        prompt: str,
        temperature: float = 0.0
    ) -> str:
        ...
```

This is the **contract**.

It says:

> Any LLM implementation used by my application must provide `generate()`.

---

# 4. OpenAI implementation

```python
from openai import AsyncOpenAI


class OpenAIProvider:

    def __init__(self, client: AsyncOpenAI):
        self.client = client

    async def generate(
        self,
        prompt: str,
        temperature: float = 0.0
    ) -> str:

        response = await self.client.chat.completions.create(
            model="gpt-4.1",
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=temperature
        )

        return response.choices[0].message.content or ""
```

Notice:

```python
class OpenAIProvider:
```

doesn't inherit:

```python
LLMProvider
```

That's intentional.

It nevertheless satisfies the protocol because it has:

```python
generate(...)
```

with the required interface.

---

# 5. Anthropic implementation

```python
class AnthropicProvider:

    async def generate(
        self,
        prompt: str,
        temperature: float = 0.0
    ) -> str:

        # Call Claude API
        return "Claude response"
```

Again:

```python
class AnthropicProvider:
```

doesn't need inheritance.

---

# 6. The service depends on the Protocol

This is where the architecture becomes powerful.

```python
from protocols.llm import LLMProvider


class ChatService:

    def __init__(self, llm: LLMProvider):
        self.llm = llm

    async def answer(self, question: str) -> str:

        prompt = f"""
        Answer the following question:

        {question}
        """

        return await self.llm.generate(prompt)
```

The service doesn't know:

```text
OpenAI
Claude
Azure
Bedrock
Llama
```

It only knows:

```text
LLMProvider
```

That's **dependency inversion**.

---

# 7. Runtime dependency injection

For FastAPI:

```python
from fastapi import Depends


def get_llm() -> LLMProvider:
    return OpenAIProvider(client)


def get_chat_service(
    llm: LLMProvider = Depends(get_llm)
) -> ChatService:

    return ChatService(llm)
```

Then:

```python
@router.post("/chat")
async def chat(
    request: ChatRequest,
    service: ChatService = Depends(get_chat_service)
):

    answer = await service.answer(
        request.question
    )

    return {
        "answer": answer
    }
```

The flow is:

```text
HTTP Request
     ↓
FastAPI Router
     ↓
Dependency Injection
     ↓
ChatService
     ↓
LLMProvider Protocol
     ↓
OpenAIProvider
     ↓
OpenAI API
```

---

# 8. This is extremely useful for testing

This is one of the **most important production uses**.

Suppose your service directly calls OpenAI.

Testing becomes expensive and unreliable.

Instead, create:

```python
class MockLLM:

    async def generate(
        self,
        prompt: str,
        temperature: float = 0.0
    ) -> str:

        return "Mock answer"
```

Because it satisfies the protocol:

```python
LLMProvider
```

you can inject it.

```python
service = ChatService(
    llm=MockLLM()
)
```

Then:

```python
answer = await service.answer(
    "What is RAG?"
)

assert answer == "Mock answer"
```

No API call.

No OpenAI cost.

No network dependency.

---

# 9. Protocol in a RAG system

This becomes even more interesting in your **production RAG architecture**.

Suppose you have:

```text
User
 ↓
FastAPI
 ↓
RAG Service
 ↓
Retriever
 ↓
Qdrant
 ↓
Reranker
 ↓
LLM
```

You can define protocols for each boundary.

---

## Retriever Protocol

```python
from typing import Protocol


class Retriever(Protocol):

    async def retrieve(
        self,
        query: str,
        top_k: int = 5
    ) -> list[dict]:
        ...
```

Now you can have:

```python
class QdrantRetriever:

    async def retrieve(
        self,
        query: str,
        top_k: int = 5
    ) -> list[dict]:

        # Qdrant search
        return []
```

Or:

```python
class ElasticsearchRetriever:

    async def retrieve(
        self,
        query: str,
        top_k: int = 5
    ) -> list[dict]:

        # Elasticsearch search
        return []
```

Or:

```python
class BM25Retriever:

    async def retrieve(
        self,
        query: str,
        top_k: int = 5
    ) -> list[dict]:

        return []
```

The RAG service doesn't care.

---

# 10. RAG service

```python
class RAGService:

    def __init__(
        self,
        retriever: Retriever,
        llm: LLMProvider
    ):
        self.retriever = retriever
        self.llm = llm

    async def answer(
        self,
        question: str
    ) -> str:

        documents = await self.retriever.retrieve(
            question,
            top_k=5
        )

        context = "\n\n".join(
            doc["text"]
            for doc in documents
        )

        prompt = f"""
        Answer using only the following context.

        Context:
        {context}

        Question:
        {question}
        """

        return await self.llm.generate(prompt)
```

Now:

```text
RAGService
   │
   ├── Retriever Protocol
   │      ├── QdrantRetriever
   │      ├── BM25Retriever
   │      └── ElasticsearchRetriever
   │
   └── LLMProvider Protocol
          ├── OpenAI
          ├── Claude
          ├── Azure OpenAI
          └── MockLLM
```

This is a very clean production architecture.

---

# 11. Protocol for embedding models

You can also abstract embeddings.

```python
class EmbeddingProvider(Protocol):

    async def embed(
        self,
        texts: list[str]
    ) -> list[list[float]]:
        ...
```

Implementation:

```python
class OpenAIEmbeddingProvider:

    async def embed(
        self,
        texts: list[str]
    ) -> list[list[float]]:

        # call embedding API
        return []
```

Or:

```python
class HuggingFaceEmbeddingProvider:

    async def embed(
        self,
        texts: list[str]
    ) -> list[list[float]]:

        return []
```

Your ingestion pipeline doesn't care which model is used.

---

# 12. Protocol for vector database

You could define:

```python
class VectorStore(Protocol):

    async def upsert(
        self,
        vectors: list[list[float]],
        metadata: list[dict]
    ) -> None:
        ...

    async def search(
        self,
        vector: list[float],
        top_k: int
    ) -> list[dict]:
        ...
```

Then:

```python
class QdrantStore:
    ...
```

or:

```python
class PineconeStore:
    ...
```

or:

```python
class WeaviateStore:
    ...
```

can implement the same contract.

---

# 13. Protocol for reranking

Production RAG may have:

```text
Query
 ↓
Vector Search
 ↓
Top 50
 ↓
Reranker
 ↓
Top 5
 ↓
LLM
```

Define:

```python
class Reranker(Protocol):

    async def rerank(
        self,
        query: str,
        documents: list[dict],
        top_k: int
    ) -> list[dict]:
        ...
```

Implementation:

```python
class CohereReranker:

    async def rerank(
        self,
        query: str,
        documents: list[dict],
        top_k: int
    ) -> list[dict]:

        # Cohere reranking
        return documents[:top_k]
```

Now your RAG pipeline isn't coupled to Cohere.

---

# 14. Protocol for Redis cache

Another production example:

```python
class Cache(Protocol):

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

Implementation:

```python
class RedisCache:

    async def get(self, key: str) -> str | None:
        ...

    async def set(
        self,
        key: str,
        value: str,
        ttl: int
    ) -> None:
        ...
```

Testing:

```python
class InMemoryCache:

    def __init__(self):
        self.data = {}

    async def get(self, key: str):
        return self.data.get(key)

    async def set(self, key: str, value: str, ttl: int):
        self.data[key] = value
```

Production:

```text
Cache Protocol
     │
     ├── RedisCache       ← Production
     │
     └── InMemoryCache    ← Tests/local development
```

---

# 15. Protocol in LangGraph

You can also use protocols around tools/services used by agents.

For example:

```python
class CustomerRepository(Protocol):

    async def get_customer(
        self,
        customer_id: str
    ) -> dict | None:
        ...
```

Then your LangGraph node can depend on it:

```python
class CustomerAgent:

    def __init__(
        self,
        repository: CustomerRepository,
        llm: LLMProvider
    ):
        self.repository = repository
        self.llm = llm

    async def run(self, customer_id: str):

        customer = await self.repository.get_customer(
            customer_id
        )

        ...
```

The agent doesn't care whether the customer data comes from:

```text
PostgreSQL
DynamoDB
REST API
Mock DB
```

---

# 16. Protocol vs ABC

This is a common interview question.

### ABC

```python
from abc import ABC, abstractmethod


class LLMProvider(ABC):

    @abstractmethod
    async def generate(self, prompt: str) -> str:
        pass
```

Implementation:

```python
class OpenAIProvider(LLMProvider):

    async def generate(self, prompt: str) -> str:
        ...
```

Inheritance is required.

---

### Protocol

```python
from typing import Protocol


class LLMProvider(Protocol):

    async def generate(self, prompt: str) -> str:
        ...
```

Implementation:

```python
class OpenAIProvider:

    async def generate(self, prompt: str) -> str:
        ...
```

No inheritance.

---

# 17. The key difference

Think about it this way:

### ABC

```text
"I belong to this family."
```

### Protocol

```text
"I satisfy this contract."
```

For large AI systems, Protocol is often very convenient because you can define **small contracts at architectural boundaries** without creating inheritance hierarchies.

---

# 18. Protocol + mypy

This is where Protocol becomes especially useful.

Suppose:

```python
class LLMProvider(Protocol):

    async def generate(
        self,
        prompt: str
    ) -> str:
        ...
```

And accidentally:

```python
class OpenAIProvider:

    async def generate(
        self,
        prompt: int
    ) -> str:

        ...
```

You can catch this with static type checking.

For example:

```bash
mypy app/
```

The type checker can identify that the implementation doesn't satisfy the expected contract.

---

# 19. Protocol with attributes

Protocols don't have to contain only methods.

Example:

```python
class ModelProvider(Protocol):

    model_name: str

    async def generate(
        self,
        prompt: str
    ) -> str:
        ...
```

Implementation:

```python
class OpenAIProvider:

    model_name = "gpt-4.1"

    async def generate(
        self,
        prompt: str
    ) -> str:
        ...
```

---

# 20. Protocol for streaming LLMs

For production AI applications, streaming is another excellent use case.

```python
from typing import Protocol, AsyncIterator


class StreamingLLM(Protocol):

    async def stream(
        self,
        prompt: str
    ) -> AsyncIterator[str]:
        ...
```

Implementation:

```python
class OpenAIStreamingProvider:

    async def stream(
        self,
        prompt: str
    ) -> AsyncIterator[str]:

        async for chunk in self.client_stream(prompt):
            yield chunk
```

FastAPI can consume it without knowing the vendor.

---

# 21. Protocol for AI agent tools

Imagine your financial advisor agent has:

```text
Account Tool
Portfolio Tool
Market Tool
Transaction Tool
```

Instead of tightly coupling the agent:

```python
class FinancialAgent:

    def __init__(
        self,
        account_service,
        portfolio_service,
        market_service
    ):
        ...
```

define contracts:

```python
class AccountService(Protocol):

    async def get_balance(
        self,
        customer_id: str
    ) -> float:
        ...
```

```python
class PortfolioService(Protocol):

    async def get_portfolio(
        self,
        customer_id: str
    ) -> dict:
        ...
```

Then the agent depends on interfaces.

This becomes particularly valuable when your enterprise AI system has many external integrations.

---

# 22. Where I would use Protocol in your production AI project

For the **Enterprise Multi-Agent Financial Advisor / RAG architecture** you're working toward, I'd use Protocol around these boundaries:

```text
                     FastAPI
                        │
                        ▼
                 Application Service
                        │
              ┌─────────┴─────────┐
              │                   │
              ▼                   ▼
        Retriever Protocol    LLM Protocol
              │                   │
        ┌─────┴─────┐       ┌─────┴─────┐
        │           │       │           │
     Qdrant       BM25   OpenAI       Claude
        │
        ▼
   Reranker Protocol
        │
   ┌────┴─────┐
   │          │
 Cohere    CrossEncoder

              │
              ▼
        Cache Protocol
              │
        ┌─────┴─────┐
        │           │
      Redis      InMemory

              │
              ▼
      Repository Protocol
              │
        ┌─────┴─────┐
        │           │
    PostgreSQL    Mock
```

This is **dependency inversion + structural typing + testability**.

---

# 23. Don't create Protocol for everything

This is important for a senior-level interview.

You don't want:

```text
UserProtocol
AddressProtocol
StringProtocol
ConfigProtocol
...
```

just because you can.

Use Protocol where you have a **meaningful boundary**.

Good candidates:

```text
LLM provider
Embedding provider
Retriever
Vector store
Reranker
Cache
Repository
External API client
Notification service
Storage service
Tool interface
Model inference service
```

Especially when you expect:

* multiple implementations
* vendor replacement
* testing/mocking
* local vs production implementations
* cloud-provider changes
* clean architecture boundaries

---

# 24. Production folder structure

I'd organize it roughly like this:

```text
app/
│
├── api/
│   └── chat.py
│
├── application/
│   └── rag_service.py
│
├── domain/
│   └── models.py
│
├── protocols/
│   ├── llm.py
│   ├── retriever.py
│   ├── embedding.py
│   ├── reranker.py
│   ├── cache.py
│   ├── vector_store.py
│   └── repository.py
│
├── providers/
│   ├── llm/
│   │   ├── openai.py
│   │   ├── anthropic.py
│   │   └── azure.py
│   │
│   ├── embeddings/
│   │   └── openai.py
│   │
│   ├── retrieval/
│   │   ├── qdrant.py
│   │   └── bm25.py
│   │
│   └── cache/
│       └── redis.py
│
├── infrastructure/
│   ├── postgres.py
│   ├── redis.py
│   └── qdrant.py
│
├── agents/
│   ├── planner.py
│   ├── researcher.py
│   └── writer.py
│
└── main.py
```

The important idea is:

```text
Application layer
       ↓
   Protocols
       ↓
Implementations
       ↓
Infrastructure / external systems
```

rather than:

```text
Application
    ↓
OpenAI SDK
Qdrant SDK
Redis SDK
Postgres
```

everywhere.

---

# 25. Interview answer

If an interviewer asks:

> **"How do you use Protocol in a production AI project?"**

A strong senior-level answer would be:

> "I use Python Protocols to define contracts at the boundaries of my application, particularly where I expect multiple implementations or need easy testing. For example, in a production RAG system I might define `LLMProvider`, `Retriever`, `EmbeddingProvider`, `Reranker`, `VectorStore`, `Cache`, and repository protocols. The application and domain services depend on these protocols rather than concrete vendors such as OpenAI, Qdrant or Redis. Concrete implementations are injected through FastAPI dependency injection. This gives us loose coupling, makes vendor replacement easier, and allows us to inject mock implementations during unit tests. Since Protocol uses structural typing, implementations don't need to inherit from the protocol; static type checkers such as mypy can still verify that they satisfy the contract."

That's a **very good Staff/Senior AI Engineer answer**.

### The mental model to remember

```text
Protocol
   ↓
CONTRACT

Implementation
   ↓
ACTUAL SERVICE

Dependency Injection
   ↓
CHOOSES IMPLEMENTATION

Service
   ↓
DEPENDS ON CONTRACT, NOT VENDOR
```

So in your AI project:

```text
RAGService
    ↓
Retriever Protocol
    ↓
QdrantRetriever

RAGService
    ↓
LLM Protocol
    ↓
OpenAIProvider

RAGService
    ↓
Cache Protocol
    ↓
RedisCache
```

That is where `Protocol` becomes genuinely useful in production—not merely as a Python typing feature, but as a **tool for decoupling your AI application architecture**.
