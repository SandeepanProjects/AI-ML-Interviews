# FastAPI: Request Logging, Latency Measurement & Correlation IDs

These three are closely related and are **very common production FastAPI interview questions**.

A production request typically flows like this:

```text
Client
  │
  │  X-Request-ID: abc-123
  ↓
Middleware
  │
  ├── Generate/reuse correlation ID
  ├── Start timer
  ├── Log request
  │
  ↓
FastAPI Endpoint
  ↓
Service
  ↓
Database / Redis / Qdrant / LLM
  │
  ↑
Middleware
  ├── Calculate latency
  ├── Log response
  └── Return correlation ID
  ↓
Client
```

---

# 1. How do you implement request logging?

The simplest approach is FastAPI HTTP middleware.

```python
import logging

from fastapi import FastAPI, Request

app = FastAPI()

logger = logging.getLogger("api")


@app.middleware("http")
async def logging_middleware(
    request: Request,
    call_next,
):
    logger.info(
        "Request received: %s %s",
        request.method,
        request.url.path,
    )

    response = await call_next(request)

    logger.info(
        "Request completed: %s %s -> %s",
        request.method,
        request.url.path,
        response.status_code,
    )

    return response
```

If the client sends:

```http
GET /users/123
```

you might get:

```text
Request received: GET /users/123
Request completed: GET /users/123 -> 200
```

---

# 2. What should you log?

In production, I would normally capture things like:

```text
timestamp
request_id
HTTP method
path
status code
latency
user/tenant identifier where appropriate
client information where appropriate
```

For example:

```text
2026-08-22T10:30:21
request_id=8a72...
method=POST
path=/api/chat
status=200
latency_ms=842
```

### Don't log sensitive information

Be careful with:

```text
❌ passwords
❌ JWT/access tokens
❌ API keys
❌ credit-card information
❌ sensitive request bodies
❌ private documents
```

This is especially important in an enterprise AI application because prompts and retrieved documents may themselves contain sensitive information.

---

# 3. How do you measure request latency?

Use a monotonic timer such as:

```python
time.perf_counter()
```

rather than wall-clock time.

```python
import time

@app.middleware("http")
async def timing_middleware(
    request: Request,
    call_next,
):
    start = time.perf_counter()

    response = await call_next(request)

    duration = time.perf_counter() - start

    logger.info(
        "%s %s took %.2f ms",
        request.method,
        request.url.path,
        duration * 1000,
    )

    return response
```

If the request takes 0.842 seconds:

```text
POST /chat took 842.00 ms
```

---

# 4. Why `perf_counter()`?

For measuring elapsed time, prefer:

```python
time.perf_counter()
```

because it is designed for measuring durations.

Don't use:

```python
time.time()
```

as your primary request-duration timer.

The important distinction is:

```text
time.time()
    → wall-clock timestamp

time.perf_counter()
    → elapsed-time measurement
```

---

# 5. Add latency to response headers

You can also expose it:

```python
@app.middleware("http")
async def timing_middleware(
    request: Request,
    call_next,
):
    start = time.perf_counter()

    response = await call_next(request)

    duration = time.perf_counter() - start

    response.headers["X-Process-Time"] = (
        f"{duration:.4f}"
    )

    return response
```

The client might receive:

```http
X-Process-Time: 0.8421
```

However, in production I would generally use **metrics/tracing** for latency rather than relying on a response header.

---

# 6. How do you add correlation IDs?

A **correlation ID** is an identifier that lets you connect all logs generated while processing the same request.

For example:

```text
request_id = 7f9c2b
```

The request might trigger:

```text
FastAPI
   ↓
PostgreSQL
   ↓
Redis
   ↓
Qdrant
   ↓
LLM
```

You want all related logs to contain:

```text
request_id=7f9c2b
```

Then you can search your logging system for:

```text
request_id=7f9c2b
```

and reconstruct what happened.

---

# 7. Implement correlation ID middleware

```python
import uuid

from fastapi import FastAPI, Request

app = FastAPI()


@app.middleware("http")
async def correlation_id_middleware(
    request: Request,
    call_next,
):
    request_id = request.headers.get(
        "X-Request-ID"
    )

    if not request_id:
        request_id = str(uuid.uuid4())

    request.state.request_id = request_id

    response = await call_next(request)

    response.headers["X-Request-ID"] = request_id

    return response
```

Now there are two cases.

### Client provides ID

```http
X-Request-ID: abc-123
```

You reuse:

```text
abc-123
```

### Client doesn't provide one

You generate:

```text
550e8400-e29b-41d4-a716-446655440000
```

---

# 8. Access the correlation ID in your endpoint

Because we stored it in:

```python
request.state.request_id
```

we can access it:

```python
from fastapi import Request


@app.get("/users")
async def get_users(
    request: Request,
):

    request_id = request.state.request_id

    return {
        "request_id": request_id
    }
```

---

# 9. Better approach: combine all three

In production, I wouldn't create three unrelated middleware components unless there is a reason to.

You can combine:

```text
Request ID
+
Logging
+
Latency
```

Example:

```python
import logging
import time
import uuid

from fastapi import FastAPI, Request

app = FastAPI()

logger = logging.getLogger("api")


@app.middleware("http")
async def observability_middleware(
    request: Request,
    call_next,
):
    # 1. Correlation ID
    request_id = request.headers.get(
        "X-Request-ID"
    )

    if not request_id:
        request_id = str(uuid.uuid4())

    request.state.request_id = request_id

    # 2. Start timer
    start = time.perf_counter()

    logger.info(
        "request_started",
        extra={
            "request_id": request_id,
            "method": request.method,
            "path": request.url.path,
        },
    )

    try:

        # 3. Execute endpoint
        response = await call_next(request)

        # 4. Calculate latency
        duration_ms = (
            time.perf_counter() - start
        ) * 1000

        logger.info(
            "request_completed",
            extra={
                "request_id": request_id,
                "method": request.method,
                "path": request.url.path,
                "status_code": response.status_code,
                "latency_ms": round(
                    duration_ms,
                    2,
                ),
            },
        )

        # 5. Return correlation ID
        response.headers[
            "X-Request-ID"
        ] = request_id

        return response

    except Exception:

        duration_ms = (
            time.perf_counter() - start
        ) * 1000

        logger.exception(
            "request_failed",
            extra={
                "request_id": request_id,
                "method": request.method,
                "path": request.url.path,
                "latency_ms": round(
                    duration_ms,
                    2,
                ),
            },
        )

        raise
```

Now every request has:

```text
request_id
method
path
status_code
latency
```

---

# 10. What does the logging output look like?

Conceptually:

```text
request_started
request_id=abc-123
method=POST
path=/api/chat
```

Then:

```text
request_completed
request_id=abc-123
method=POST
path=/api/chat
status_code=200
latency_ms=842.31
```

If something fails:

```text
request_failed
request_id=abc-123
method=POST
path=/api/chat
latency_ms=842.31
exception=...
```

Now you can search:

```text
request_id=abc-123
```

and find the complete request lifecycle.

---

# 11. Why is correlation ID especially important for AI applications?

Imagine:

```text
POST /chat
request_id=abc123
```

Your application performs:

```text
                    /chat
                      │
                      ↓
                 RAG Service
                      │
             ┌────────┼────────┐
             ↓        ↓        ↓
          Redis    Qdrant   PostgreSQL
                       │
                       ↓
                    Reranker
                       │
                       ↓
                      LLM
```

You might have:

```text
request_id=abc123
retrieval_latency_ms=72
```

then:

```text
request_id=abc123
reranking_latency_ms=31
```

then:

```text
request_id=abc123
llm_latency_ms=740
input_tokens=2400
output_tokens=350
```

Finally:

```text
request_id=abc123
total_latency_ms=910
status=200
```

This lets you determine:

> "Why was this particular LLM request slow?"

instead of simply seeing:

```text
POST /chat = 910ms
```

---

# 12. Correlation ID vs Request ID

You will hear both terms.

A **request ID** typically identifies one request.

A **correlation ID** is a broader concept used to correlate related operations across services.

For a simple FastAPI application:

```text
X-Request-ID
```

can serve both purposes.

In a distributed system:

```text
API Gateway
     │
     │ correlation_id=abc
     ↓
FastAPI
     │
     ├── RAG service
     │      │
     │      └── Qdrant
     │
     └── LLM service
```

the same correlation identifier can be propagated across services.

---

# 13. Correlation IDs with `contextvars`

For larger applications, you may want the request ID available to logging code without passing `request` through every function.

Python's `contextvars` is useful for this.

```python
from contextvars import ContextVar

request_id_context: ContextVar[
    str | None
] = ContextVar(
    "request_id",
    default=None,
)
```

Middleware:

```python
@app.middleware("http")
async def correlation_middleware(
    request: Request,
    call_next,
):
    request_id = (
        request.headers.get("X-Request-ID")
        or str(uuid.uuid4())
    )

    request_id_context.set(request_id)

    response = await call_next(request)

    response.headers[
        "X-Request-ID"
    ] = request_id

    return response
```

Now another function can access it:

```python
def get_request_id() -> str | None:
    return request_id_context.get()
```

For example:

```python
async def call_llm():

    request_id = get_request_id()

    logger.info(
        "Calling LLM",
        extra={
            "request_id": request_id
        },
    )
```

You don't have to do:

```python
call_llm(request)
```

through every layer.

---

# 14. Production architecture

For a production FastAPI + AI system, I'd think about observability like this:

```text
                    HTTP Request
                         │
                         ↓
                ┌─────────────────┐
                │    Middleware   │
                │                 │
                │ Request ID      │
                │ Access logging  │
                │ Basic timing    │
                └────────┬────────┘
                         ↓
                    FastAPI Router
                         ↓
                    Dependencies
                         ↓
                     Service
                         ↓
              ┌──────────┼──────────┐
              ↓          ↓          ↓
          PostgreSQL    Redis     Qdrant
                                    ↓
                                Retriever
                                    ↓
                                Reranker
                                    ↓
                                    LLM
```

And observability systems collect:

```text
Logs
  ↓
ELK / Loki / Cloud logging

Metrics
  ↓
Prometheus
  ↓
Grafana

Traces
  ↓
OpenTelemetry
  ↓
Jaeger / Tempo / cloud tracing
```

---

# 15. What should you measure?

Don't only measure average latency.

For production APIs, track:

```text
Request count
Error rate
P50 latency
P95 latency
P99 latency
Status codes
Throughput
```

For AI endpoints additionally:

```text
LLM latency
Time to first token
Total generation time
Input tokens
Output tokens
Token cost
Retrieval latency
Reranking latency
Vector DB latency
Cache hit rate
```

For example:

```text
POST /chat

P50 = 450 ms
P95 = 1.2 s
P99 = 2.8 s
Error rate = 0.4%
```

This is much more useful than only logging:

```text
/chat took 842ms
```

---

# 16. Important interview distinction

### Request logging

Answers:

> **"What happened?"**

```text
POST /chat
status=200
```

### Latency measurement

Answers:

> **"How long did it take?"**

```text
latency=842ms
```

### Correlation ID

Answers:

> **"Which logs/traces belong to the same request?"**

```text
request_id=abc123
```

Together:

```text
request_id=abc123
POST /chat
status=200
latency=842ms
```

---

# 17. Interview-ready answer

If asked:

### **How do you implement request logging?**

> "I use HTTP middleware to log the request method, path, status code, request ID and relevant metadata. I avoid logging secrets, tokens, passwords or sensitive prompt/document contents."

### **How do you measure request latency?**

> "I start a monotonic timer using `time.perf_counter()` before `call_next()`, calculate the elapsed time afterward, and send the latency to structured logs and metrics. In production I'd monitor P50, P95 and P99 rather than just average latency."

### **How do you add correlation IDs?**

> "At the middleware layer, I read an incoming `X-Request-ID` or generate a UUID if one isn't provided. I store it in request state or a `contextvar`, include it in structured logs, and return it in the response header. In a distributed system I propagate the correlation ID across downstream services so I can trace a request end-to-end."

### The mental model to remember

```text
Request
   ↓
Generate/Get Correlation ID
   ↓
Start Timer
   ↓
Log Request
   ↓
call_next()
   ↓
Endpoint + Services + DB + LLM
   ↓
Stop Timer
   ↓
Log Status + Latency
   ↓
Return Correlation ID
   ↓
Response
```

That is the **production-grade answer** interviewers are generally looking for.
