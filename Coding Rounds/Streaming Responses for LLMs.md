# Implement Streaming Responses for LLMs (Senior AI Engineer Interview)

This is a common **Senior AI Engineer**, **LLM Engineer**, and **GenAI Engineer** interview question.

> **"Implement streaming responses for an LLM."**

The interviewer is typically evaluating whether you understand:

* Token streaming
* Async programming
* FastAPI streaming
* Server-Sent Events (SSE)
* WebSockets
* Backpressure
* Cancellation
* Production architecture

---

# Why Stream Responses?

Without streaming:

```text
User
   |
   |------ Request ------>
   |
(wait 10 seconds)
   |
   |<----- Entire response -----
```

The user waits until the entire response is generated.

---

With streaming:

```text
User
   |
   |------ Request ------>
   |
   |<----- Hello
   |<----- Hello, how
   |<----- Hello, how are
   |<----- Hello, how are you?
```

The first token appears almost immediately, improving perceived latency.

---

# Architecture

```text
                 Client
                    |
             HTTP / WebSocket
                    |
             FastAPI Endpoint
                    |
          Async Token Generator
                    |
               LLM Provider
                    |
             Token-by-Token Output
```

---

# Step 1 — Simulate an LLM Stream

```python
import asyncio

class MockLLM:

    async def stream(self, prompt: str):
        text = "Hello! This is a streamed response from the LLM."

        for word in text.split():
            await asyncio.sleep(0.2)  # Simulate token generation
            yield word + " "
```

Usage:

```python
async def main():
    llm = MockLLM()

    async for token in llm.stream("Hi"):
        print(token, end="")
```

Output:

```text
Hello! This is a streamed response from the LLM.
```

Each word arrives independently.

---

# Step 2 — FastAPI Streaming Endpoint (SSE)

```python
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
import asyncio

app = FastAPI()


async def token_generator():

    words = [
        "Hello",
        "this",
        "is",
        "a",
        "streaming",
        "response"
    ]

    for word in words:
        await asyncio.sleep(0.3)
        yield word + " "


@app.get("/stream")
async def stream():

    return StreamingResponse(
        token_generator(),
        media_type="text/plain"
    )
```

When a client connects, it starts receiving chunks as they are generated.

---

# Step 3 — Server-Sent Events (SSE)

SSE is a common choice for one-way streaming from the server to the client.

```python
from fastapi.responses import StreamingResponse

async def event_stream():

    for i in range(5):
        await asyncio.sleep(1)
        yield f"data: Token {i}\n\n"

@app.get("/events")
async def events():
    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream"
    )
```

Browser JavaScript:

```javascript
const source = new EventSource("/events");

source.onmessage = (event) => {
    console.log(event.data);
};
```

---

# Step 4 — Streaming from an LLM Wrapper

Abstract the provider behind a common interface.

```python
class LLM:

    async def stream(self, prompt):

        response = [
            "Artificial",
            "Intelligence",
            "is",
            "transforming",
            "software."
        ]

        for token in response:
            yield token + " "
```

---

# Step 5 — Agent with Streaming

```python
class Agent:

    def __init__(self, llm):
        self.llm = llm

    async def run(self, query):

        async for token in self.llm.stream(query):
            yield token
```

Endpoint:

```python
agent = Agent(LLM())

@app.get("/chat")
async def chat():

    return StreamingResponse(
        agent.run("Hello"),
        media_type="text/plain"
    )
```

---

# Step 6 — Streaming Tool Calls

An agent can stream progress while invoking tools.

```python
async def run():

    yield "Searching...\n"

    await asyncio.sleep(1)

    yield "Found documents...\n"

    await asyncio.sleep(1)

    yield "Generating answer...\n"

    await asyncio.sleep(1)

    yield "Final response."
```

Users receive continuous updates instead of waiting for the final answer.

---

# Step 7 — Cancellation Handling

Clients may disconnect before generation finishes.

```python
import asyncio

async def generator():

    try:
        for i in range(100):
            await asyncio.sleep(1)
            yield f"{i}\n"

    except asyncio.CancelledError:
        print("Client disconnected")
        raise
```

This prevents unnecessary computation after a disconnect.

---

# Step 8 — Streaming with WebSockets

Use WebSockets when bidirectional communication is required.

```python
from fastapi import WebSocket

@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):

    await ws.accept()

    for token in [
        "Hello",
        "from",
        "WebSocket"
    ]:
        await asyncio.sleep(0.5)
        await ws.send_text(token)
```

Use cases include:

* Chat applications
* Voice assistants
* Collaborative editing
* Interactive coding assistants

---

# Streaming Pipeline

```text
User
   |
   v
FastAPI
   |
   v
StreamingResponse
   |
   v
LLM.stream()
   |
   v
yield token
   |
   v
Client receives token immediately
```

---

# Production Architecture

```text
                 Load Balancer
                       |
                FastAPI Workers
                       |
               Async Streaming API
                       |
         +-------------+-------------+
         |                           |
         v                           v
    Redis Cache                LLM Provider
         |                           |
         +-------------+-------------+
                       |
                 Token Generator
                       |
                StreamingResponse
                       |
                     Client
```

---

# Production Best Practices

| Concern          | Recommendation                                                                                                     |
| ---------------- | ------------------------------------------------------------------------------------------------------------------ |
| Async I/O        | Use `async`/`await` end-to-end to avoid blocking worker threads.                                                   |
| Backpressure     | Let the network control send rate; avoid buffering the entire response in memory.                                  |
| Cancellation     | Detect client disconnects and stop generation promptly.                                                            |
| Timeouts         | Apply request and upstream provider timeouts to avoid hanging streams.                                             |
| Heartbeats       | For long-lived SSE connections, send periodic heartbeat events to keep intermediaries from closing the connection. |
| Metrics          | Track first-token latency, total latency, tokens/sec, completion rate, cancellations, and errors.                  |
| Logging          | Log request IDs, model name, token counts, and stream completion status.                                           |
| Security         | Authenticate before opening the stream and enforce rate limits.                                                    |
| Error Handling   | Emit structured error events (for SSE) or close WebSockets with appropriate status codes.                          |
| Resource Cleanup | Ensure upstream LLM connections are closed if the client disconnects.                                              |

---

# SSE vs WebSocket

| Feature                    | Server-Sent Events     | WebSocket            |
| -------------------------- | ---------------------- | -------------------- |
| Direction                  | Server → Client        | Bidirectional        |
| Browser Support            | Native (`EventSource`) | Native (`WebSocket`) |
| Simplicity                 | Very simple            | More complex         |
| Chat Streaming             | Excellent              | Excellent            |
| Live Collaboration         | Limited                | Excellent            |
| Voice/Realtime Interaction | Not suitable           | Excellent            |

---

# Senior AI Engineer Interview Answer

A strong answer is:

> "I implement streaming by exposing the LLM output as an asynchronous generator. Each generated token is yielded immediately to the client using `StreamingResponse` (typically over Server-Sent Events for one-way streaming or WebSockets for bidirectional communication). In production, I make the entire pipeline asynchronous, propagate cancellation when clients disconnect, instrument first-token latency and throughput, and add authentication, rate limiting, retries, and proper cleanup of upstream model connections. This reduces perceived latency while keeping resource usage efficient."

This demonstrates understanding of both the implementation details and the operational considerations expected in production LLM systems.
