# FastAPI Middleware

Middleware is a layer that runs **around every HTTP request and response**.

Think of it as:

```text
Client
   ↓
Middleware
   ↓
FastAPI Router
   ↓
Service
   ↓
Repository
   ↓
Database
   ↓
Response
   ↑
Middleware
   ↑
Client
```

It is useful for **cross-cutting concerns** such as:

* Request/response logging
* Request ID / correlation ID
* Timing and latency measurement
* CORS
* Authentication-related processing
* Security headers
* Rate limiting
* Metrics
* Tracing
* Exception handling

---

# 1. What is Middleware?

A middleware intercepts a request **before it reaches your endpoint** and can also process the response **after the endpoint finishes**.

Conceptually:

```python
async def middleware(request, call_next):

    # BEFORE endpoint

    response = await call_next(request)

    # AFTER endpoint

    return response
```

The important part is:

```python
await call_next(request)
```

This passes the request to the next middleware/route handler.

---

# 2. Simple FastAPI middleware

FastAPI provides:

```python
@app.middleware("http")
```

Example:

```python
from fastapi import FastAPI, Request

app = FastAPI()


@app.middleware("http")
async def my_middleware(
    request: Request,
    call_next,
):
    print("Request received")

    response = await call_next(request)

    print("Response generated")

    return response
```

Request flow:

```text
GET /users
   ↓
my_middleware
   ↓
/users endpoint
   ↓
my_middleware
   ↓
Response
```

---

# 3. Middleware execution order

Suppose you have:

```python
@app.middleware("http")
async def middleware_a(request, call_next):

    print("A before")

    response = await call_next(request)

    print("A after")

    return response
```

and:

```python
@app.middleware("http")
async def middleware_b(request, call_next):

    print("B before")

    response = await call_next(request)

    print("B after")

    return response
```

Then conceptually:

```text
A before
   ↓
B before
   ↓
Endpoint
   ↓
B after
   ↓
A after
```

Middleware behaves somewhat like nested layers.

---

# 4. Request logging middleware

A common production use case is request logging.

```python
import logging

from fastapi import FastAPI, Request


logger = logging.getLogger(__name__)

app = FastAPI()


@app.middleware("http")
async def logging_middleware(
    request: Request,
    call_next,
):

    logger.info(
        "Request: %s %s",
        request.method,
        request.url.path,
    )

    response = await call_next(request)

    logger.info(
        "Response: %s",
        response.status_code,
    )

    return response
```

Now:

```text
GET /users/123
```

might generate:

```text
Request: GET /users/123
Response: 200
```

---

# 5. Measuring request latency

This is particularly useful for production AI applications.

```python
import time

from fastapi import FastAPI, Request


app = FastAPI()


@app.middleware("http")
async def timing_middleware(
    request: Request,
    call_next,
):

    start = time.perf_counter()

    response = await call_next(request)

    duration = (
        time.perf_counter() - start
    )

    print(
        f"{request.method} "
        f"{request.url.path} "
        f"took {duration:.4f}s"
    )

    return response
```

For example:

```text
POST /chat took 1.2842s
```

---

# 6. Add latency to response headers

You can also expose timing information:

```python
import time

@app.middleware("http")
async def timing_middleware(
    request: Request,
    call_next,
):

    start = time.perf_counter()

    response = await call_next(request)

    duration = (
        time.perf_counter() - start
    )

    response.headers[
        "X-Process-Time"
    ] = str(duration)

    return response
```

The response might contain:

```text
X-Process-Time: 0.245
```

This can be useful during debugging.

For production, though, I'd usually send the measurement to metrics/tracing rather than exposing internal timing unnecessarily.

---

# 7. Request ID / Correlation ID

This is **very important in production systems**.

Suppose one user request triggers:

```text
API
 ↓
PostgreSQL
 ↓
Redis
 ↓
Qdrant
 ↓
LLM
```

You want to identify all logs belonging to the same request.

Create:

```text
request_id = abc-123
```

Middleware:

```python
import uuid

from fastapi import Request


@app.middleware("http")
async def request_id_middleware(
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

    response.headers[
        "X-Request-ID"
    ] = request_id

    return response
```

Now downstream code can access:

```python
request.state.request_id
```

Your logs can contain:

```text
request_id=abc-123
event=llm_call
model=gpt-5
latency=1.24
tokens=1500
```

This is extremely useful for debugging distributed systems.

---

# 8. Custom authentication middleware

You **can** perform authentication-related processing in middleware, but don't automatically put all authentication there.

For example:

```python
@app.middleware("http")
async def auth_middleware(
    request: Request,
    call_next,
):

    token = request.headers.get(
        "Authorization"
    )

    if token:
        request.state.token = token

    return await call_next(request)
```

But for FastAPI JWT authentication, I would generally prefer:

```python
Depends(get_current_user)
```

for endpoint-level authentication.

Why?

Because different endpoints can have different authentication requirements:

```text
GET /health
    → public

GET /profile
    → authenticated

POST /admin/users
    → admin
```

FastAPI dependencies are usually a cleaner fit for that.

---

# 9. Middleware vs Dependency

This is a common interview question.

### Middleware

Runs broadly for requests:

```text
Request
   ↓
Middleware
   ↓
Endpoint
```

Good for:

```text
Logging
Tracing
Request IDs
Metrics
CORS
Global headers
```

### Dependency

Runs when required by a route:

```python
@router.get("/profile")
async def profile(
    user = Depends(get_current_user)
):
    ...
```

Good for:

```text
Authentication
Authorization
Database sessions
Current user
Tenant context
Reusable route-level logic
```

### Easy rule

> **Middleware is for cross-cutting concerns that apply broadly; dependencies are for request-specific application dependencies.**

---

# 10. Exception-handling middleware

You can also create middleware that catches unexpected exceptions.

```python
import logging

from fastapi import Request
from fastapi.responses import JSONResponse


logger = logging.getLogger(__name__)


@app.middleware("http")
async def exception_middleware(
    request: Request,
    call_next,
):

    try:

        return await call_next(request)

    except Exception:

        logger.exception(
            "Unhandled exception"
        )

        return JSONResponse(
            status_code=500,
            content={
                "detail": "Internal server error"
            },
        )
```

However, in a real application I'd normally use FastAPI exception handlers for structured exception handling where appropriate, rather than turning middleware into a giant error-management layer.

---

# 11. Security headers middleware

You can add security-related headers:

```python
@app.middleware("http")
async def security_headers(
    request: Request,
    call_next,
):

    response = await call_next(request)

    response.headers[
        "X-Content-Type-Options"
    ] = "nosniff"

    response.headers[
        "X-Frame-Options"
    ] = "DENY"

    return response
```

For more comprehensive security policy, use an appropriate security configuration rather than manually adding only a couple of headers.

---

# 12. Using `BaseHTTPMiddleware`

FastAPI/Starlette also supports class-based middleware.

```python
from starlette.middleware.base import (
    BaseHTTPMiddleware,
)


class LoggingMiddleware(
    BaseHTTPMiddleware
):

    async def dispatch(
        self,
        request,
        call_next,
    ):

        print(
            f"{request.method} "
            f"{request.url.path}"
        )

        response = await call_next(
            request
        )

        return response
```

Register it:

```python
app.add_middleware(
    LoggingMiddleware
)
```

---

# 13. Function middleware vs class middleware

### Function style

```python
@app.middleware("http")
async def middleware(
    request,
    call_next,
):
    ...
```

Simple and good for small middleware.

### Class style

```python
class LoggingMiddleware(
    BaseHTTPMiddleware
):
    ...
```

Useful when:

* middleware has configuration
* you want reusable middleware
* middleware has more complex behavior

For simple applications, function middleware is often easier to read.

---

# 14. Production example

For a production FastAPI application, I might have:

```text
                    Request
                       │
                       ↓
              ┌─────────────────┐
              │ Request ID      │
              └────────┬────────┘
                       ↓
              ┌─────────────────┐
              │ Logging         │
              └────────┬────────┘
                       ↓
              ┌─────────────────┐
              │ Metrics/Tracing │
              └────────┬────────┘
                       ↓
              ┌─────────────────┐
              │ FastAPI Router  │
              └────────┬────────┘
                       ↓
                  Dependency
                       ↓
                Authentication
                       ↓
                    Service
                       ↓
                  Repository
```

For an AI application:

```text
Request
  ↓
Request ID
  ↓
Tracing
  ↓
Metrics
  ↓
Authentication
  ↓
FastAPI Router
  ↓
AI Service
  ↓
RAG
  ├── Redis
  ├── PostgreSQL
  ├── Qdrant
  └── LLM
```

The request ID allows you to correlate:

```text
API request
    ↓
retrieval
    ↓
reranking
    ↓
LLM call
    ↓
response
```

in your logs and traces.

---

# 15. Important: Middleware should not do everything

A common mistake is creating a giant middleware:

```python
@app.middleware("http")
async def everything(...):
    # authentication
    # authorization
    # database
    # business logic
    # rate limiting
    # RAG
    # LLM call
    # logging
    # ...
```

❌ Don't do this.

Middleware should remain focused on **cross-cutting concerns**.

Instead:

```text
Middleware
    ↓
request ID
logging
metrics
tracing

Dependencies
    ↓
authentication
authorization
database session

Service
    ↓
business logic

Repository
    ↓
database access
```

---

# 16. Middleware and async code

Middleware should be asynchronous when performing async work:

```python
@app.middleware("http")
async def middleware(
    request: Request,
    call_next,
):

    response = await call_next(request)

    return response
```

Don't do blocking operations such as:

```python
time.sleep(5)
```

inside async middleware.

That can block the event loop.

Use:

```python
await asyncio.sleep(5)
```

if you genuinely need an asynchronous delay, although artificial delays obviously shouldn't exist in production middleware.

Similarly, avoid synchronous blocking database or HTTP calls in an async middleware path.

---

# 17. Interview answer

If the interviewer asks:

### **"What is middleware?"**

Say:

> **"Middleware is a component that wraps the HTTP request/response lifecycle. It can execute logic before the request reaches the endpoint and after the endpoint produces a response. I use middleware for cross-cutting concerns such as request IDs, logging, metrics, tracing, CORS, and security headers."**

---

### **"How do you create custom middleware?"**

Show:

```python
@app.middleware("http")
async def logging_middleware(
    request: Request,
    call_next,
):

    start = time.perf_counter()

    response = await call_next(request)

    duration = (
        time.perf_counter() - start
    )

    print(
        request.method,
        request.url.path,
        response.status_code,
        duration,
    )

    return response
```

Then explain:

```text
Request
   ↓
middleware BEFORE
   ↓
call_next()
   ↓
endpoint
   ↓
middleware AFTER
   ↓
Response
```

### The key sentence to remember

> **"`call_next(request)` passes the request to the next layer, and middleware can execute logic both before and after that call."**

For your **Senior AI Engineer interviews**, I'd especially remember **middleware vs `Depends()`**: use middleware for global cross-cutting behavior like **logging, request IDs, metrics and tracing**, and use FastAPI dependencies for things like **JWT authentication, current-user resolution, tenant context, and database sessions**.
