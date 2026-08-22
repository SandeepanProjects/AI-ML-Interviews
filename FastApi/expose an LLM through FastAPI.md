# How do you expose an LLM through FastAPI?

In a production AI application, I would **not put the LLM call directly inside the FastAPI endpoint**.

Instead, use:

```text
Client
   ↓
FastAPI Router
   ↓
Authentication / Validation
   ↓
Service Layer
   ↓
LLM Client
   ↓
OpenAI / Azure OpenAI / Anthropic / vLLM
   ↓
Response
```

This separation makes the system easier to test, monitor, scale, and replace.

---

# 1. Basic FastAPI LLM endpoint

Suppose the client sends:

```json
{
  "message": "Explain RAG"
}
```

and expects:

```json
{
  "answer": "RAG stands for Retrieval-Augmented Generation..."
}
```

## Request model

```python
from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    message: str = Field(
        min_length=1,
        max_length=10000
    )
```

## Response model

```python
class ChatResponse(BaseModel):
    answer: str
```

---

# 2. Create an LLM client

For example, using an OpenAI-compatible client:

```python
from openai import AsyncOpenAI

client = AsyncOpenAI(
    api_key=settings.OPENAI_API_KEY
)
```

Then create an LLM service:

```python
class LLMService:

    def __init__(self, client):
        self.client = client

    async def generate(
        self,
        message: str,
    ) -> str:

        response = await self.client.chat.completions.create(
            model="gpt-5",
            messages=[
                {
                    "role": "user",
                    "content": message,
                }
            ],
            temperature=0.2,
        )

        return response.choices[0].message.content
```

The important thing is that the **FastAPI endpoint doesn't know how the LLM API works**.

---

# 3. Expose it through FastAPI

```python
from fastapi import APIRouter, Depends

router = APIRouter(
    prefix="/api/v1",
    tags=["chat"],
)


@router.post(
    "/chat",
    response_model=ChatResponse,
)
async def chat(
    request: ChatRequest,
    llm_service: LLMService = Depends(
        get_llm_service
    ),
):

    answer = await llm_service.generate(
        request.message
    )

    return ChatResponse(
        answer=answer
    )
```

Then the client calls:

```http
POST /api/v1/chat
Content-Type: application/json
```

Body:

```json
{
  "message": "What is RAG?"
}
```

Response:

```json
{
  "answer": "RAG stands for Retrieval-Augmented Generation..."
}
```

---

# 4. Why use a service layer?

Don't do this:

```python
@app.post("/chat")
async def chat(request: ChatRequest):

    response = await client.chat.completions.create(
        model="gpt-5",
        messages=[
            {
                "role": "user",
                "content": request.message,
            }
        ],
    )

    return response
```

It works, but becomes difficult to maintain.

Instead:

```text
Router
   ↓
LLMService
   ↓
LLMClient
```

For example:

```text
app/
├── main.py
├── api/
│   └── chat.py
├── services/
│   └── llm_service.py
├── clients/
│   └── llm_client.py
├── schemas/
│   └── chat.py
└── config/
    └── settings.py
```

This allows you to replace:

```text
OpenAI
```

with:

```text
Azure OpenAI
Anthropic
vLLM
Ollama
AWS Bedrock
```

without rewriting your API layer.

---

# 5. Dependency injection

Create the service through FastAPI dependency injection.

```python
def get_llm_service() -> LLMService:

    return LLMService(
        client=client
    )
```

Then:

```python
@router.post("/chat")
async def chat(
    request: ChatRequest,
    llm_service: LLMService = Depends(
        get_llm_service
    ),
):

    return await llm_service.generate(
        request.message
    )
```

This also makes testing much easier.

---

# 6. Add authentication

A production LLM API shouldn't normally be publicly accessible.

For example:

```python
@router.post("/chat")
async def chat(
    request: ChatRequest,
    user=Depends(get_current_user),
    llm_service: LLMService = Depends(
        get_llm_service
    ),
):

    answer = await llm_service.generate(
        request.message
    )

    return ChatResponse(
        answer=answer
    )
```

Now:

```text
Request
   ↓
JWT authentication
   ↓
Current user
   ↓
LLM Service
```

---

# 7. Add authorization / RBAC

You might have:

```text
USER
PREMIUM_USER
ADMIN
```

For example:

```python
@router.post("/chat")
async def chat(
    request: ChatRequest,
    user=Depends(require_role("USER")),
    llm_service: LLMService = Depends(
        get_llm_service
    ),
):
    ...
```

This prevents unauthorized users from accessing the LLM endpoint.

---

# 8. Add request IDs

This becomes very important when debugging LLM applications.

Middleware generates:

```text
request_id = abc-123
```

Then:

```text
POST /chat
request_id=abc-123
```

Your LLM service logs:

```text
request_id=abc-123
model=gpt-5
input_tokens=1200
output_tokens=350
latency_ms=840
```

Now you can trace the complete request.

---

# 9. Measure LLM latency

Don't only measure FastAPI latency.

Measure:

```text
Total API latency
LLM latency
```

For example:

```python
import time


class LLMService:

    async def generate(
        self,
        message: str,
    ):

        start = time.perf_counter()

        response = await self.client.chat.completions.create(
            model="gpt-5",
            messages=[
                {
                    "role": "user",
                    "content": message,
                }
            ],
        )

        latency = (
            time.perf_counter() - start
        )

        logger.info(
            "LLM request completed",
            extra={
                "latency_ms": latency * 1000,
            },
        )

        return response.choices[0].message.content
```

In production you'd send these metrics to your observability system rather than relying only on logs.

---

# 10. Handle LLM failures

LLM APIs can fail because of:

```text
Timeout
Rate limit
Authentication failure
Provider outage
Bad request
Context-length limit
Network error
```

Don't expose raw provider exceptions.

Instead:

```python
try:

    response = await self.client.chat.completions.create(
        model="gpt-5",
        messages=messages,
    )

except Exception as exc:

    logger.exception(
        "LLM request failed"
    )

    raise LLMServiceError(
        "LLM provider unavailable"
    )
```

Then your global exception handler converts that into something like:

```json
{
  "error": {
    "code": "LLM_UNAVAILABLE",
    "message": "AI service temporarily unavailable",
    "request_id": "abc-123"
  }
}
```

---

# 11. Add timeout

Never allow an LLM call to hang indefinitely.

Conceptually:

```python
import asyncio


try:

    response = await asyncio.wait_for(
        self.generate_from_provider(message),
        timeout=30,
    )

except asyncio.TimeoutError:

    raise LLMTimeoutError()
```

In a real provider client, prefer the SDK's native timeout configuration when available.

---

# 12. Add retries

Transient errors can be retried.

For example:

```text
LLM request
   ↓
429 / temporary 5xx
   ↓
wait
   ↓
retry
```

Use exponential backoff:

```text
Attempt 1 → immediate
Attempt 2 → 1 second
Attempt 3 → 2 seconds
Attempt 4 → 4 seconds
```

But **don't blindly retry every error**.

For example:

```text
401 → don't retry
400 → don't retry
invalid request → don't retry

429 → potentially retry
temporary 5xx → potentially retry
timeout → potentially retry
```

---

# 13. Add fallback models

A production AI system can have:

```text
Primary model
     ↓
failure
     ↓
Fallback model
```

For example:

```python
try:
    return await primary_model.generate(
        message
    )

except LLMUnavailableError:

    return await fallback_model.generate(
        message
    )
```

Architecture:

```text
                 Request
                    ↓
              LLM Service
                    ↓
             Primary Model
                    │
             ┌──────┴──────┐
             │             │
           Success       Failure
             │             ↓
             │       Fallback Model
             │             │
             └──────┬──────┘
                    ↓
                Response
```

---

# 14. Streaming LLM responses

For chat applications, you often don't want to wait for the entire response.

Instead:

```text
User
 ↓
FastAPI
 ↓
LLM
 ↓
token 1
token 2
token 3
token 4
...
```

FastAPI can return a streaming response.

For example:

```python
from fastapi.responses import StreamingResponse


async def generate_stream(message: str):

    stream = await client.chat.completions.create(
        model="gpt-5",
        messages=[
            {
                "role": "user",
                "content": message,
            }
        ],
        stream=True,
    )

    async for chunk in stream:

        content = (
            chunk.choices[0]
            .delta
            .content
        )

        if content:
            yield content


@router.post("/chat/stream")
async def chat_stream(
    request: ChatRequest,
):

    return StreamingResponse(
        generate_stream(request.message),
        media_type="text/plain",
    )
```

The client starts receiving output before the entire answer is generated.

For browser applications, **SSE** is often a convenient choice for one-way token streaming.

---

# 15. Expose an LLM with RAG

In your kind of enterprise AI application, you usually wouldn't expose a raw LLM.

Instead:

```text
POST /chat
      ↓
Authentication
      ↓
Chat Service
      ↓
Query understanding
      ↓
Embedding
      ↓
Qdrant retrieval
      ↓
Reranking
      ↓
Prompt construction
      ↓
LLM
      ↓
Response
```

For example:

```python
class ChatService:

    def __init__(
        self,
        retriever,
        llm_service,
    ):
        self.retriever = retriever
        self.llm_service = llm_service

    async def answer(
        self,
        question: str,
    ):

        documents = await self.retriever.search(
            question
        )

        context = "\n\n".join(
            doc.text
            for doc in documents
        )

        prompt = f"""
        Answer the question using only
        the provided context.

        Context:
        {context}

        Question:
        {question}
        """

        return await self.llm_service.generate(
            prompt
        )
```

Then:

```python
@router.post("/chat")
async def chat(
    request: ChatRequest,
    service: ChatService = Depends(
        get_chat_service
    ),
):

    answer = await service.answer(
        request.message
    )

    return ChatResponse(
        answer=answer
    )
```

---

# 16. Production architecture

For a **production FastAPI + LLM/RAG application**, I'd structure it like:

```text
                         Client
                           │
                           ↓
                     API Gateway
                           │
                           ↓
                     FastAPI
                           │
                ┌──────────┴──────────┐
                │                     │
           Middleware             Dependencies
                │                     │
        ┌───────┼───────┐       JWT / RBAC
        │       │       │
     Logging  Metrics  Tracing
                │
                ↓
             Router
                │
                ↓
           Chat Service
                │
       ┌────────┼─────────┐
       ↓        ↓         ↓
    Redis    Retriever  LLM Service
                │          │
                ↓          ↓
              Qdrant   Provider API
                │
                ↓
            Reranker
```

And for enterprise applications:

```text
FastAPI
  │
  ├── PostgreSQL
  │     └── users / tenants / conversations
  │
  ├── Redis
  │     └── cache / rate limit / session
  │
  ├── Qdrant
  │     └── embeddings
  │
  └── LLM
        └── OpenAI / Azure / Anthropic / vLLM
```

---

# 17. What should the API response contain?

Don't necessarily return the provider's raw response.

Instead define your own API contract:

```python
class ChatResponse(BaseModel):

    answer: str
    request_id: str
    model: str
    usage: dict | None = None
```

Response:

```json
{
  "answer": "RAG combines retrieval with generation.",
  "request_id": "abc-123",
  "model": "gpt-5",
  "usage": {
    "input_tokens": 1200,
    "output_tokens": 85
  }
}
```

This gives your application a **stable API contract** even if you later change LLM providers.

---

# 18. Important production considerations

When exposing an LLM through FastAPI, I would consider:

### Security

```text
JWT authentication
RBAC
Tenant isolation
Input validation
Prompt injection defenses
Secrets management
TLS
```

### Reliability

```text
Timeouts
Retries
Exponential backoff
Fallback models
Circuit breaker
Rate limiting
```

### Performance

```text
Async provider calls
Streaming
Redis caching
Connection pooling
Concurrent retrieval
```

### Observability

```text
Request ID
Structured logs
Latency
P50/P95/P99
Token usage
LLM cost
Error rate
Tracing
```

### AI-specific

```text
Prompt versioning
Model versioning
RAG evaluation
Hallucination/faithfulness evaluation
Token limits
Context management
Safety filters
```

---

# 19. Interview answer

If the interviewer asks:

> **"How do you expose an LLM through FastAPI?"**

A strong Senior AI Engineer answer is:

> **"I expose the LLM through a versioned FastAPI endpoint such as `POST /api/v1/chat`. The request is validated with Pydantic and authenticated using JWT. The router delegates to a service layer rather than calling the LLM SDK directly. The service handles prompt construction, model selection, timeouts, retries, fallbacks and observability. The LLM client is injected as a dependency so it can be mocked in tests. For conversational applications I support streaming using SSE or `StreamingResponse`. In a RAG application, the service first retrieves and reranks relevant context from the vector database and then sends the grounded prompt to the LLM. I also track request IDs, latency, token usage, cost and errors."**

The architecture you should remember is:

```text
POST /api/v1/chat
        ↓
Pydantic validation
        ↓
JWT / RBAC
        ↓
Router
        ↓
ChatService
        ↓
RAG / Prompt / Model routing
        ↓
LLMClient
        ↓
OpenAI / Azure / Anthropic / vLLM
        ↓
Response / Streaming
```

**The key interview point:** FastAPI is the **API layer**; it should not become your **LLM business-logic layer**. Keep the provider interaction behind an LLM client/service so the application remains testable, replaceable, and production-ready.
