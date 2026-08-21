For a production FastAPI + LLM application, I would implement **token streaming** so the client receives generated tokens incrementally instead of waiting for the complete LLM response.

The architecture would be:

```text
Client
   │
   │ HTTP request
   ▼
FastAPI
   │
   ▼
Service Layer
   │
   ▼
LLM Provider
   │
   │ token1 → token2 → token3 → ...
   ▼
FastAPI StreamingResponse
   │
   ▼
Client receives tokens incrementally
```

## 1. Basic FastAPI implementation

FastAPI's `StreamingResponse` is a natural fit.

```python
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
import asyncio

app = FastAPI()


async def generate_response(prompt: str):
    tokens = [
        "FastAPI ",
        "can ",
        "stream ",
        "LLM ",
        "responses ",
        "incrementally."
    ]

    for token in tokens:
        await asyncio.sleep(0.2)
        yield token


@app.get("/chat")
async def chat(prompt: str):

    return StreamingResponse(
        generate_response(prompt),
        media_type="text/plain"
    )
```

Instead of returning:

```json
{
    "answer": "FastAPI can stream LLM responses incrementally."
}
```

the client receives:

```text
FastAPI
can
stream
LLM
responses
incrementally.
```

as the model generates them.

---

# 2. Using an actual LLM

In a real project, I would keep the LLM code inside a **service layer**, rather than directly inside the FastAPI route.

```text
app/
├── api/
│   └── chat.py
├── services/
│   └── llm_service.py
├── models/
│   └── chat.py
└── main.py
```

### LLM service

For example, conceptually:

```python
class LLMService:

    async def stream(self, prompt: str):

        async for chunk in self.client.stream(
            prompt=prompt
        ):
            yield chunk
```

Then:

```python
@router.post("/chat")
async def chat(request: ChatRequest):

    return StreamingResponse(
        llm_service.stream(request.prompt),
        media_type="text/plain"
    )
```

The important design principle is:

> **The API layer handles HTTP; the service layer handles LLM orchestration.**

---

# 3. I prefer SSE for chat applications

For an AI chat application, I would usually use **Server-Sent Events (SSE)** rather than returning arbitrary plain text.

The flow becomes:

```text
LLM
 │
 ├── token
 ├── token
 ├── token
 ├── token
 │
 ▼
FastAPI
 │
 ▼
SSE
 │
 ▼
Browser
```

Example:

```python
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
import json

app = FastAPI()


async def llm_stream(prompt: str):

    tokens = ["Hello", " ", "world", "!"]

    for token in tokens:
        yield f"data: {json.dumps({'token': token})}\n\n"


@app.get("/chat/stream")
async def stream_chat(prompt: str):

    return StreamingResponse(
        llm_stream(prompt),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        },
    )
```

The browser receives events like:

```text
data: {"token":"Hello"}

data: {"token":" "}

data: {"token":"world"}

data: {"token":"!"}
```

---

# 4. Production version

For a production AI platform, I would stream more than just tokens.

For example:

```json
{
    "type": "token",
    "content": "Hello"
}
```

Then:

```json
{
    "type": "token",
    "content": " world"
}
```

At the end:

```json
{
    "type": "done",
    "usage": {
        "input_tokens": 100,
        "output_tokens": 50
    }
}
```

This gives the frontend structured information.

The event types could be:

```text
token
citation
tool_start
tool_end
error
done
```

This becomes especially useful in **RAG and agentic systems**.

---

# 5. Streaming a RAG response

In a RAG application, I would typically do:

```text
User Question
      │
      ▼
Retriever
      │
      ▼
Documents
      │
      ▼
Prompt Construction
      │
      ▼
LLM
      │
      ├── token
      ├── token
      ├── token
      ▼
SSE
      │
      ▼
Frontend
```

But I would generally **finish retrieval before starting answer generation**.

For example:

```python
async def generate_answer(question: str):

    documents = await retriever.search(question)

    prompt = build_prompt(
        question,
        documents
    )

    async for token in llm.stream(prompt):
        yield {
            "type": "token",
            "content": token
        }
```

You can also send citations separately:

```python
async def stream_answer(question):

    documents = await retriever.search(question)

    yield {
        "type": "sources",
        "sources": [
            document.metadata
            for document in documents
        ]
    }

    prompt = build_prompt(question, documents)

    async for token in llm.stream(prompt):
        yield {
            "type": "token",
            "content": token
        }
```

That allows the UI to display:

```text
Answer:
The company policy allows...

Sources:
📄 HR Policy
📄 Employee Handbook
```

while the answer is still being generated.

---

# 6. Streaming an agent

This becomes more interesting with LangGraph/agents.

You might have:

```text
User
 │
 ▼
Agent
 │
 ├── Thought / decision
 │
 ├── Tool call
 │      │
 │      ▼
 │    Search
 │
 ├── Tool result
 │
 ├── LLM tokens
 │
 └── Final answer
```

I would **not expose hidden chain-of-thought/reasoning** to the client.

Instead, expose safe execution events:

```json
{"type": "status", "message": "Searching documents"}
```

```json
{"type": "tool", "name": "document_search"}
```

```json
{"type": "token", "content": "According"}
```

```json
{"type": "token", "content": " to"}
```

```json
{"type": "done"}
```

This gives the user useful visibility without leaking private reasoning.

---

# 7. Handling client disconnects

This is an important production concern.

Suppose the user closes the browser while the LLM is still generating.

You don't want to continue generating tokens and paying for them.

You should detect cancellation/disconnection where supported:

```python
from asyncio import CancelledError


async def stream(prompt):

    try:
        async for token in llm.stream(prompt):
            yield token

    except CancelledError:
        # Cancel/cleanup the underlying LLM operation
        await llm.cancel()
        raise
```

The exact cancellation mechanism depends on the LLM provider/client.

---

# 8. Error handling

Errors can occur **after the HTTP response has already started streaming**.

That's an important difference from normal FastAPI endpoints.

You can't reliably do:

```python
raise HTTPException(500, "LLM failed")
```

after you've already sent several SSE events.

Instead, send an error event:

```python
async def stream():

    try:
        async for token in llm.stream(prompt):
            yield make_event(
                "token",
                token
            )

    except Exception:
        yield make_event(
            "error",
            "LLM generation failed"
        )

    finally:
        yield make_event(
            "done",
            None
        )
```

The frontend can then stop rendering and show an appropriate error.

---

# 9. Production concerns I would consider

For a senior AI engineer interview, mention these:

### Backpressure

If the client is slower than the LLM producer, you don't want unlimited buffering.

```text
LLM producer
     ↓
bounded queue
     ↓
HTTP stream
     ↓
client
```

A bounded `asyncio.Queue` can be used for more complex pipelines.

### Timeouts

Use:

```text
connection timeout
LLM request timeout
idle timeout
maximum generation time
```

### Authentication

Authenticate the request **before** starting the stream.

```text
JWT
 ↓
RBAC
 ↓
tenant validation
 ↓
LLM request
```

### Rate limiting

Apply per:

```text
user
tenant
API key
IP
```

### Observability

Track:

```text
TTFT              → Time To First Token
total latency
tokens/sec
input tokens
output tokens
LLM cost
errors
disconnects
```

**TTFT is particularly important for streaming applications.**

A user generally perceives:

```text
Request
   │
   ├───────────────┐
   │               │
   │            first token
   │               │
   │<--- TTFT ---->│
   │
   │ token token token token
   │
   └──── total latency ──────>
```

---

# 10. SSE vs WebSocket

This is another common interview question.

| SSE                               | WebSocket                                        |
| --------------------------------- | ------------------------------------------------ |
| Server → client streaming         | Bidirectional                                    |
| Simple HTTP-based model           | Persistent socket                                |
| Excellent for LLM token streaming | Useful for interactive real-time systems         |
| Easy browser integration          | More complex                                     |
| Good default for chat generation  | Better when client must continuously send events |

For a normal ChatGPT-style response:

**I would start with SSE.**

I'd choose WebSockets if I needed continuous bidirectional communication, such as collaborative agents, voice interaction, or real-time session state.

---

## Interview answer

If the interviewer asks **"How would you implement streaming LLM responses?"**, a strong answer is:

> "I would expose an async FastAPI endpoint using `StreamingResponse`, and preferably SSE for a chat application. The service layer would call the LLM's streaming API and yield tokens as they arrive instead of buffering the complete response. I would send structured events such as tokens, citations, tool status, errors, and a final usage event. For production, I'd handle client disconnects and cancellation, timeouts, authentication, rate limiting, backpressure, and observability metrics such as TTFT, tokens per second, total latency, token usage, and cost. For RAG or agentic applications, retrieval or tool execution can happen first, followed by streaming the final answer. I would use WebSockets instead of SSE only when true bidirectional real-time communication is required."

**The key senior-level point:** streaming isn't just `yield token`; you need to think about **cancellation, errors after the stream starts, backpressure, observability, authentication, and cost control**.
