# How would you stream an LLM response from FastAPI?

Streaming means the client receives the LLM's output **incrementally**, instead of waiting for the complete response.

Without streaming:

```text
User → FastAPI → LLM
                  │
                  │ waits for complete response
                  ↓
             Complete answer
                  ↓
                User
```

With streaming:

```text
User → FastAPI → LLM
                  │
                  ↓
               "RAG"
                  ↓
               " is"
                  ↓
               " a"
                  ↓
               " system..."
                  ↓
                User
```

This improves the **perceived latency** significantly.

---

# 1. Basic FastAPI streaming

FastAPI uses `StreamingResponse`.

```python
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
import asyncio

app = FastAPI()


async def generate_response():
    words = ["Hello", " ", "from", " ", "FastAPI"]

    for word in words:
        yield word
        await asyncio.sleep(0.5)


@app.get("/stream")
async def stream():

    return StreamingResponse(
        generate_response(),
        media_type="text/plain",
    )
```

The key concept is:

```python
async def generator():
    yield "chunk 1"
    yield "chunk 2"
    yield "chunk 3"
```

`StreamingResponse` sends each yielded chunk to the client.

---

# 2. Streaming a real LLM response

A typical architecture is:

```text
Client
   │
   ▼
FastAPI
   │
   ▼
LLM Service
   │
   ▼
LLM Provider
   │
   ├── token/chunk 1
   ├── token/chunk 2
   ├── token/chunk 3
   ▼
Client
```

Example with an asynchronous OpenAI-compatible client:

```python
from openai import AsyncOpenAI

client = AsyncOpenAI(
    api_key="YOUR_API_KEY"
)
```

Create an LLM service:

```python
class LLMService:

    def __init__(self, client: AsyncOpenAI):
        self.client = client

    async def stream(
        self,
        message: str,
    ):
        stream = await self.client.chat.completions.create(
            model="your-model",
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
```

The important part is:

```python
async for chunk in stream:
    yield content
```

The LLM sends chunks, and FastAPI immediately forwards them.

---

# 3. Expose the streaming endpoint

```python
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

app = FastAPI()


class ChatRequest(BaseModel):
    message: str


@app.post("/api/v1/chat/stream")
async def chat_stream(
    request: ChatRequest,
):

    llm_service = LLMService(client)

    return StreamingResponse(
        llm_service.stream(request.message),
        media_type="text/plain",
    )
```

Client sends:

```json
{
  "message": "Explain RAG"
}
```

Instead of waiting for:

```text
RAG is Retrieval-Augmented Generation...
```

the client gradually receives:

```text
RAG
```

then:

```text
RAG is
```

then:

```text
RAG is Retrieval-Augmented
```

and so on.

---

# 4. Better architecture using Dependency Injection

Don't create the LLM client inside every endpoint.

```python
def get_llm_service():

    return LLMService(
        client=client
    )
```

Then:

```python
from fastapi import Depends


@app.post("/api/v1/chat/stream")
async def chat_stream(
    request: ChatRequest,
    llm_service: LLMService = Depends(
        get_llm_service
    ),
):

    return StreamingResponse(
        llm_service.stream(request.message),
        media_type="text/plain",
    )
```

Architecture:

```text
Router
   │
   ▼
LLMService
   │
   ▼
LLM Client
   │
   ▼
LLM Provider
```

This makes testing much easier.

---

# 5. Why SSE is often better for browser applications

For LLM chat applications, I often prefer **Server-Sent Events (SSE)**.

SSE is:

```text
Server
  │
  │ one-way stream
  ▼
Browser
```

The server sends events like:

```text
data: Hello

data: world

data: RAG is...
```

---

# 6. Basic SSE format

SSE requires the format:

```text
data: your message

```

Notice the two newline characters:

```python
yield f"data: {content}\n\n"
```

FastAPI implementation:

```python
from fastapi.responses import StreamingResponse


async def sse_generator(
    llm_service: LLMService,
    message: str,
):
    async for chunk in llm_service.stream(message):
        yield f"data: {chunk}\n\n"


@app.post("/api/v1/chat/stream")
async def chat_stream(
    request: ChatRequest,
    llm_service: LLMService = Depends(
        get_llm_service
    ),
):

    return StreamingResponse(
        sse_generator(
            llm_service,
            request.message,
        ),
        media_type="text/event-stream",
    )
```

The response is now:

```http
Content-Type: text/event-stream
```

---

# 7. Structured SSE events — production approach

Sending raw text is simple but limiting.

A production AI application often sends structured events.

For example:

```text
event: token
data: {"content":"Hello"}

event: token
data: {"content":" world"}

event: metadata
data: {"model":"model-name"}

event: done
data: {}
```

Let's implement that.

```python
import json


async def sse_generator(
    llm_service: LLMService,
    message: str,
):

    try:

        async for chunk in llm_service.stream(message):

            data = {
                "type": "token",
                "content": chunk,
            }

            yield (
                f"event: token\n"
                f"data: {json.dumps(data)}\n\n"
            )

        yield (
            "event: done\n"
            "data: {}\n\n"
        )

    except Exception:

        error = {
            "type": "error",
            "message": "Streaming failed",
        }

        yield (
            f"event: error\n"
            f"data: {json.dumps(error)}\n\n"
        )
```

Then:

```python
@app.post("/api/v1/chat/stream")
async def chat_stream(
    request: ChatRequest,
    llm_service: LLMService = Depends(
        get_llm_service
    ),
):

    return StreamingResponse(
        sse_generator(
            llm_service,
            request.message,
        ),
        media_type="text/event-stream",
    )
```

Now your stream has a clear protocol:

```text
token
  ↓
token
  ↓
token
  ↓
done
```

or:

```text
token
  ↓
error
```

---

# 8. Client disconnect handling

A very important production issue:

```text
User closes browser
        ↓
Should we continue paying for LLM generation?
```

Usually, no.

FastAPI gives you access to the request.

```python
from fastapi import Request


@app.post("/chat/stream")
async def chat_stream(
    request: Request,
    body: ChatRequest,
    llm_service: LLMService = Depends(
        get_llm_service
    ),
):
```

Then:

```python
async def stream_generator():

    async for chunk in llm_service.stream(
        body.message
    ):

        if await request.is_disconnected():
            break

        yield chunk
```

Full version:

```python
@app.post("/chat/stream")
async def chat_stream(
    request: Request,
    body: ChatRequest,
    llm_service: LLMService = Depends(
        get_llm_service
    ),
):

    async def generator():

        async for chunk in llm_service.stream(
            body.message
        ):

            if await request.is_disconnected():

                # Stop upstream generation
                break

            yield chunk

    return StreamingResponse(
        generator(),
        media_type="text/plain",
    )
```

In a production implementation, you should also close/cancel the upstream provider stream if the SDK supports it.

---

# 9. Error handling during streaming

This is tricky.

Before streaming starts, you can return:

```http
500 Internal Server Error
```

But once you've already sent:

```text
HTTP 200 OK
```

you generally cannot change the HTTP status to 500 halfway through the stream.

So:

```text
Stream starts
    ↓
HTTP 200 sent
    ↓
tokens sent
    ↓
LLM fails
```

You cannot reliably send a new HTTP 500 response.

Instead, send an error event:

```text
event: error
data: {"message":"LLM generation failed"}
```

Example:

```python
async def sse_generator():

    try:

        async for chunk in llm_service.stream(
            request.message
        ):

            yield (
                f"event: token\n"
                f"data: {json.dumps({'content': chunk})}\n\n"
            )

    except Exception as exc:

        logger.exception(
            "LLM streaming failed"
        )

        yield (
            "event: error\n"
            "data: {\"message\":\"Generation failed\"}\n\n"
        )

        return

    yield (
        "event: done\n"
        "data: {}\n\n"
    )
```

This is a very good senior-level interview point.

---

# 10. Streaming with timeout

You don't want a provider stream to run forever.

One approach is a total timeout:

```python
import asyncio


async def stream_with_timeout(
    generator,
    timeout: float,
):

    try:

        async with asyncio.timeout(timeout):

            async for item in generator:
                yield item

    except TimeoutError:

        yield (
            "event: error\n"
            "data: {\"message\":\"Generation timeout\"}\n\n"
        )
```

Usage:

```python
async def sse_generator():

    async for item in stream_with_timeout(
        llm_service.stream(message),
        timeout=60,
    ):
        yield item
```

You can also enforce:

* Time to first token
* Total generation timeout
* Maximum tokens
* Maximum stream duration

---

# 11. Streaming with RAG

A common production RAG flow:

```text
User Question
      │
      ▼
Authenticate
      │
      ▼
Embed Query
      │
      ▼
Retrieve Documents
      │
      ▼
Rerank Documents
      │
      ▼
Build Prompt
      │
      ▼
LLM Streaming
      │
      ├── token
      ├── token
      └── done
```

Important:

**Retrieval normally happens before token streaming starts.**

Example:

```python
class ChatService:

    def __init__(
        self,
        retriever,
        llm_service,
    ):
        self.retriever = retriever
        self.llm_service = llm_service

    async def stream_answer(
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
        Answer only using the context.

        Context:
        {context}

        Question:
        {question}
        """

        async for chunk in self.llm_service.stream(
            prompt
        ):
            yield chunk
```

Endpoint:

```python
@app.post("/chat/stream")
async def chat_stream(
    request: ChatRequest,
    service: ChatService = Depends(
        get_chat_service
    ),
):

    return StreamingResponse(
        service.stream_answer(request.message),
        media_type="text/event-stream",
    )
```

---

# 12. Send RAG progress events

For a better UX, you can stream status events before generation.

For example:

```text
event: status
data: {"message":"Searching documents"}

event: status
data: {"message":"Generating answer"}

event: token
data: {"content":"RAG"}

event: token
data: {"content":" is"}

event: done
data: {}
```

Implementation:

```python
async def rag_stream(
    question: str,
):

    yield (
        "event: status\n"
        "data: {\"message\":\"Searching documents\"}\n\n"
    )

    documents = await retriever.search(
        question
    )

    yield (
        "event: status\n"
        "data: {\"message\":\"Generating answer\"}\n\n"
    )

    async for token in llm.stream(
        question,
        documents,
    ):

        yield (
            f"event: token\n"
            f"data: {json.dumps({'content': token})}\n\n"
        )

    yield (
        "event: done\n"
        "data: {}\n\n"
    )
```

This is useful for AI systems because retrieval and reranking can take noticeable time.

---

# 13. Backpressure

A production concern is:

```text
LLM generates faster
        ↓
than
        ↓
Client consumes
```

You need to consider:

```text
Client
  slow
    ↑
FastAPI
    ↑
LLM provider
```

Good practices include:

* Use async streaming.
* Avoid buffering the complete response.
* Yield chunks immediately.
* Apply generation/token limits.
* Detect disconnected clients.
* Cancel upstream generation when possible.
* Avoid unbounded queues.

---

# 14. Streaming architecture

A production architecture might look like:

```text
Browser
   │
   │ SSE
   ▼
Load Balancer
   │
   ▼
FastAPI
   │
   ├── JWT Authentication
   ├── Rate Limiting
   ├── Request ID
   └── Observability
          │
          ▼
      Chat Service
          │
     ┌────┴────┐
     │         │
 Retriever   LLM Client
     │         │
     ▼         ▼
  Qdrant    LLM Provider
```

---

# 15. Complete production-style example

```python
import json
import logging
from collections.abc import AsyncGenerator

from fastapi import (
    APIRouter,
    Depends,
    Request,
)
from fastapi.responses import StreamingResponse
from pydantic import BaseModel


logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/api/v1/chat",
    tags=["chat"],
)


class ChatRequest(BaseModel):
    message: str


class LLMService:

    async def stream(
        self,
        message: str,
    ) -> AsyncGenerator[str, None]:

        stream = await client.chat.completions.create(
            model="your-model",
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


def get_llm_service() -> LLMService:
    return LLMService()


async def generate_sse(
    request: Request,
    message: str,
    llm_service: LLMService,
):

    request_id = getattr(
        request.state,
        "request_id",
        None,
    )

    try:

        async for token in llm_service.stream(
            message
        ):

            if await request.is_disconnected():

                logger.info(
                    "Client disconnected",
                    extra={
                        "request_id": request_id
                    },
                )

                break

            event = {
                "content": token,
                "request_id": request_id,
            }

            yield (
                "event: token\n"
                f"data: {json.dumps(event)}\n\n"
            )

        yield (
            "event: done\n"
            "data: {}\n\n"
        )

    except Exception:

        logger.exception(
            "LLM streaming failed",
            extra={
                "request_id": request_id
            },
        )

        error = {
            "message": "Generation failed",
            "request_id": request_id,
        }

        yield (
            "event: error\n"
            f"data: {json.dumps(error)}\n\n"
        )


@router.post("/stream")
async def stream_chat(
    request: Request,
    body: ChatRequest,
    llm_service: LLMService = Depends(
        get_llm_service
    ),
):

    return StreamingResponse(
        generate_sse(
            request=request,
            message=body.message,
            llm_service=llm_service,
        ),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        },
    )
```

---

# 16. Important interview answer

If an interviewer asks:

> **How would you stream an LLM response from FastAPI?**

A strong answer is:

> **"I use an async generator with FastAPI's `StreamingResponse`. The LLM SDK provides an async stream, and I iterate over it using `async for`, forwarding each token or chunk immediately to the client. For browser applications, I usually use Server-Sent Events with `text/event-stream` and structured events such as `token`, `error`, and `done`. I also handle client disconnects and cancel the upstream stream when possible, enforce timeouts and token limits, and track request IDs, latency, token usage, and errors. For RAG, retrieval and reranking happen before generation, after which the generated answer is streamed token by token."**

## The key code pattern to remember

```python
async def generator():

    async for chunk in llm_stream:

        if await request.is_disconnected():
            break

        yield chunk


return StreamingResponse(
    generator(),
    media_type="text/event-stream",
)
```

This is the core pattern behind production LLM streaming with FastAPI.
