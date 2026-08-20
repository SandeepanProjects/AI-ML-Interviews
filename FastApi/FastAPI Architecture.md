## FastAPI Architecture and Why It Is Faster Than Flask

This is a very common **senior-level FastAPI interview question**.

A strong answer should not simply say *“FastAPI is faster because it is async.”* You should explain the entire request lifecycle and where the performance advantage comes from.

---

# 1. What is FastAPI?

FastAPI is a modern Python web framework designed primarily for building **APIs**.

Its architecture is built around:

* **ASGI** — asynchronous server interface
* **Starlette** — HTTP/web framework underneath FastAPI
* **Pydantic** — request/response validation and serialization
* **Uvicorn** — commonly used ASGI server
* Python `async` / `await`

The high-level architecture is:

```text
                   Client
                     │
                     │ HTTP Request
                     ▼
                  Uvicorn
                ASGI Server
                     │
                     ▼
                 FastAPI
                     │
          ┌──────────┴──────────┐
          │                     │
       Routing              Middleware
          │                     │
          └──────────┬──────────┘
                     ▼
               Dependencies
                     │
                     ▼
              Pydantic Validation
                     │
                     ▼
                 Endpoint
                     │
                     ▼
               Service Layer
                     │
                     ▼
              Repository Layer
                     │
             ┌───────┴────────┐
             ▼                ▼
        PostgreSQL          Redis
```

---

# 2. What happens when a request reaches FastAPI?

Suppose the client sends:

```http
GET /users/123
```

The request goes through several layers.

### Step 1 — Client sends request

```text
Client
   │
   │ GET /users/123
   ▼
```

### Step 2 — Uvicorn receives it

Uvicorn is an **ASGI server**.

```text
Client
   ↓
Uvicorn
```

Uvicorn handles things such as:

* TCP connections
* HTTP
* ASGI protocol
* event loop integration
* connection lifecycle

Then it passes the request to FastAPI.

---

# 3. FastAPI is built on Starlette

This is an important interview point.

FastAPI isn't implementing the entire HTTP framework from scratch.

Conceptually:

```text
FastAPI
   │
   └── Starlette
         │
         └── ASGI
```

Starlette provides many of the underlying web capabilities:

* routing
* middleware
* request/response handling
* WebSockets
* background tasks
* exception handling

FastAPI adds:

* Pydantic validation
* dependency injection
* automatic OpenAPI generation
* automatic Swagger/ReDoc documentation
* API-focused abstractions

---

# 4. Request routing

FastAPI determines which endpoint should handle the request.

For example:

```python
from fastapi import FastAPI

app = FastAPI()


@app.get("/users/{user_id}")
async def get_user(user_id: int):
    return {"user_id": user_id}
```

When:

```http
GET /users/123
```

arrives, FastAPI matches:

```text
/users/{user_id}
```

and extracts:

```text
user_id = 123
```

It also knows that:

```python
user_id: int
```

means the parameter should be an integer.

---

# 5. Pydantic validation

Suppose we have:

```python
from pydantic import BaseModel


class UserCreate(BaseModel):
    name: str
    age: int
    email: str
```

Then:

```python
@app.post("/users")
async def create_user(user: UserCreate):
    return user
```

The client sends:

```json
{
    "name": "Sandeep",
    "age": 37,
    "email": "test@example.com"
}
```

FastAPI/Pydantic validates the request.

Conceptually:

```text
JSON
 ↓
Pydantic
 ↓
Validated Python object
 ↓
Endpoint
```

If:

```json
{
    "name": "Sandeep",
    "age": "hello"
}
```

FastAPI returns a validation error rather than passing invalid data into your business logic.

---

# 6. Dependency Injection

FastAPI also resolves dependencies before executing the endpoint.

Example:

```python
from fastapi import Depends


def get_current_user():
    return {
        "id": 123,
        "role": "admin"
    }


@app.get("/profile")
async def profile(
    user=Depends(get_current_user)
):
    return user
```

The flow becomes:

```text
Request
   ↓
Routing
   ↓
Dependency resolution
   ↓
Authentication
   ↓
Endpoint
```

In production, dependencies are often used for:

* database sessions
* authentication
* authorization
* configuration
* repositories
* services
* tenant context

---

# 7. Endpoint execution

Eventually FastAPI calls:

```python
async def get_user(user_id: int):
    ...
```

If the operation is I/O-bound, we can use asynchronous operations.

For example:

```python
@app.get("/users/{user_id}")
async def get_user(user_id: int):

    user = await repository.get_user(user_id)

    return user
```

The important part is:

```python
await repository.get_user(...)
```

The event loop can work on another request while waiting for the database.

---

# 8. The most important concept: ASGI

This is where the major architectural difference comes in.

Traditional Python web applications commonly use **WSGI**.

FastAPI uses **ASGI**.

```text
WSGI
 │
 └── primarily synchronous request model


ASGI
 │
 ├── synchronous
 ├── asynchronous
 ├── WebSockets
 └── long-lived connections
```

FastAPI's async support is built around ASGI.

---

# 9. What is the event loop?

Suppose we have three requests:

```text
Request A → PostgreSQL
Request B → Redis
Request C → External API
```

All three operations are waiting for I/O.

An async application can conceptually do:

```text
Request A
   ↓
await database
   │
   ├───────────────┐
                   │
Request B          │
   ↓               │
await Redis        │
   │               │
   ├───────────┐   │
               │   │
Request C      │   │
   ↓           │   │
await API      │   │
               │   │
               ▼   ▼
          Event Loop
```

While A is waiting for PostgreSQL, the event loop can work on B or C.

This is **concurrency**, not necessarily parallelism.

---

# 10. Why is FastAPI faster than Flask?

This needs a nuanced answer.

### The simplistic answer

> FastAPI is faster because it is asynchronous.

That's incomplete.

A better answer is:

> FastAPI is built on ASGI and supports asynchronous, non-blocking I/O. When endpoints use async-compatible libraries, a worker can handle many concurrent I/O-bound requests efficiently instead of blocking while waiting for network or database operations.

There are several contributing factors.

---

# 11. FastAPI vs Flask architecture

Conceptually:

```text
              Flask
                │
              WSGI
                │
        Synchronous model
                │
       ┌────────┴────────┐
       │                 │
    Request 1         Request 2
       │                 │
    Blocking          Blocking
```

FastAPI:

```text
             FastAPI
                │
               ASGI
                │
            Event Loop
                │
      ┌─────────┼─────────┐
      │         │         │
    Req A     Req B     Req C
      │         │         │
    await     await     await
      │         │         │
      └─────────┼─────────┘
                │
          Event Loop
```

This becomes especially useful when your application spends significant time waiting for external systems.

---

# 12. Example: Database-heavy API

Imagine an endpoint takes 100 ms because the database takes 100 ms.

### Synchronous/blocking model

Conceptually:

```text
Request 1
   │
   ├── DB query ─────── 100 ms
   │
   ▼
Response
```

During blocking operations, the worker cannot freely use that execution context for other work.

With async:

```text
Request 1
   │
   ├── await DB ─────────────┐
                             │
Request 2                    │
   │                         │
   ├── await DB ─────────┐   │
                         │   │
Request 3                │   │
   │                     │   │
   └── process           │   │
                         ▼   ▼
                      Event Loop
```

The same worker can keep progressing other requests while the I/O is pending.

---

# 13. But async doesn't automatically make everything faster

This is **very important in an interview**.

Suppose you write:

```python
@app.get("/")
async def endpoint():

    time.sleep(10)

    return {"message": "done"}
```

This is bad.

`time.sleep()` is blocking.

It can block the event loop.

Instead:

```python
import asyncio


@app.get("/")
async def endpoint():

    await asyncio.sleep(10)

    return {"message": "done"}
```

The second version allows the event loop to handle other work while waiting.

---

# 14. What about CPU-intensive work?

Async isn't a magic solution for CPU-heavy operations.

For example:

```python
@app.get("/calculate")
async def calculate():

    result = huge_machine_learning_computation()

    return result
```

If the computation takes several seconds and consumes the CPU, `async` doesn't magically make it non-blocking.

For CPU-heavy work, you may use:

```text
FastAPI
   │
   ├── lightweight API work
   │
   └── job queue
          │
          ├── Celery
          ├── workers
          └── GPU workers
```

For AI systems this distinction is particularly important.

---

# 15. FastAPI is excellent for AI APIs

This is where FastAPI becomes very useful in your type of project.

Imagine:

```text
                  Client
                    │
                    ▼
                 FastAPI
                    │
          ┌─────────┼──────────┐
          │         │          │
       PostgreSQL Redis      Qdrant
          │         │          │
          └─────────┼──────────┘
                    │
                 Retriever
                    │
                 Reranker
                    │
                   LLM
```

Your FastAPI application may simultaneously wait for:

* PostgreSQL
* Redis
* Qdrant
* embedding API
* reranker
* LLM API

These are primarily **I/O-bound operations**.

Async programming can therefore provide significant concurrency benefits.

---

# 16. Example of concurrent AI calls

Suppose you want to call:

```text
Qdrant
LLM
Redis
```

independently.

You can potentially perform them concurrently.

```python
import asyncio


async def retrieve():
    ...


async def get_context():
    ...


async def call_llm():
    ...


@app.get("/answer")
async def answer():

    retrieval, context = await asyncio.gather(
        retrieve(),
        get_context()
    )

    answer = await call_llm()

    return {
        "answer": answer
    }
```

Instead of:

```text
retrieve
   ↓
wait
   ↓
context
   ↓
wait
```

independent operations can overlap:

```text
retrieve ──────────┐
                   │
context ───────────┤
                   ▼
                 gather
                   │
                   ▼
                  LLM
```

This can reduce latency when the operations are genuinely independent.

---

# 17. Is Flask slow?

Be careful here.

**Do not say:**

> Flask is slow and FastAPI is fast.

That's an overly simplistic interview answer.

A better statement:

> Flask can absolutely be used for high-performance production systems. The difference is primarily in the underlying concurrency model and the ecosystem around the framework. FastAPI's ASGI architecture makes asynchronous I/O and high-concurrency workloads more natural, while Flask traditionally follows a WSGI/synchronous model.

Modern Flask also supports async views, but the underlying deployment and concurrency model still differs from a native ASGI framework.

---

# 18. FastAPI Architecture — Complete Interview Diagram

You can draw this in an interview:

```text
                         CLIENT
                           │
                           │ HTTP
                           ▼
                    ┌──────────────┐
                    │   Uvicorn    │
                    │  ASGI Server │
                    └──────┬───────┘
                           │
                           ▼
                    ┌──────────────┐
                    │   FastAPI    │
                    └──────┬───────┘
                           │
                    ┌──────▼───────┐
                    │  Middleware  │
                    └──────┬───────┘
                           │
                    ┌──────▼───────┐
                    │    Router    │
                    └──────┬───────┘
                           │
                    ┌──────▼───────┐
                    │ Dependencies │
                    │ Auth / DB DI  │
                    └──────┬───────┘
                           │
                    ┌──────▼───────┐
                    │   Pydantic   │
                    │  Validation  │
                    └──────┬───────┘
                           │
                    ┌──────▼───────┐
                    │   Endpoint   │
                    └──────┬───────┘
                           │
                    ┌──────▼───────┐
                    │   Service    │
                    └──────┬───────┘
                           │
                    ┌──────▼───────┐
                    │  Repository  │
                    └──────┬───────┘
                           │
             ┌─────────────┼─────────────┐
             ▼             ▼             ▼
        PostgreSQL       Redis          Qdrant
```

---

# 19. Strong Interview Answer

If the interviewer asks:

> **"Explain FastAPI architecture and why it is faster than Flask."**

You can answer:

> "FastAPI is an ASGI-based Python framework built on top of Starlette, with Pydantic providing data validation and serialization. It commonly runs behind Uvicorn, which handles the ASGI protocol and integrates with Python's asynchronous event loop.
>
> When a request arrives, Uvicorn passes it to FastAPI, which performs routing, middleware execution, dependency resolution and Pydantic validation before calling the endpoint. In a production application, the endpoint typically delegates to a service layer and repository layer for business logic and data access.
>
> The main performance advantage comes from ASGI and asynchronous, non-blocking I/O. If an endpoint is waiting for PostgreSQL, Redis, Qdrant or an external LLM API, an async endpoint can yield control to the event loop, allowing the worker to process other requests rather than remaining blocked.
>
> However, FastAPI isn't automatically faster in every situation. Async benefits are strongest for I/O-bound workloads and require async-compatible libraries. CPU-heavy or blocking operations can still block execution and may need worker processes or background job systems.
>
> So I would say FastAPI's advantage over traditional Flask deployments is primarily its native ASGI architecture, async concurrency model, and efficient handling of I/O-bound workloads, rather than simply saying that FastAPI is faster because it's async."

### One sentence to remember

**FastAPI → ASGI → async/non-blocking I/O → event loop → high concurrency for I/O-bound workloads.**

For a **Senior AI Engineer interview**, that is the explanation I'd expect you to give rather than just saying *“FastAPI is faster than Flask.”*
