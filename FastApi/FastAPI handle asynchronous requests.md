These four questions are **core FastAPI concurrency questions**. For a senior interview, don't just memorize `async/await`; understand what happens under the hood and when async actually helps.

---

# 1. How does FastAPI handle asynchronous requests?

FastAPI is built on **ASGI**, primarily through **Starlette**, and uses Python's `asyncio` ecosystem.

You can define an async endpoint:

```python
from fastapi import FastAPI

app = FastAPI()


@app.get("/users/{user_id}")
async def get_user(user_id: int):
    user = await fetch_user(user_id)
    return user
```

The important part is:

```python
async def
    ↓
await
    ↓
Non-blocking I/O
```

When the endpoint reaches:

```python
await fetch_user()
```

and `fetch_user()` is waiting for I/O, the event loop can work on other requests instead of sitting idle.

### Example

Imagine 3 requests:

```text
Request A → waiting for DB
Request B → waiting for LLM
Request C → waiting for Redis
```

With asynchronous I/O:

```text
Event Loop
   │
   ├── Request A → DB ─────── waiting
   │
   ├── Request B → LLM ────── waiting
   │
   └── Request C → Redis ──── waiting
                         ↓
              Event loop handles
              other work
```

When the I/O operation completes, the corresponding coroutine can continue.

---

## Async doesn't mean "new thread for every request"

This is a common interview trap.

With async code, you generally aren't doing:

```text
Request
   ↓
New Thread
   ↓
Execute
```

Instead, asynchronous I/O uses cooperative scheduling:

```text
             Event Loop
                 │
       ┌─────────┼─────────┐
       ↓         ↓         ↓
   Coroutine A Coroutine B Coroutine C
       │         │         │
     await     await      await
       │         │         │
       └─────────┴─────────┘
```

The coroutine yields control when it reaches an `await` that is waiting for I/O.

---

# 2. What is an event loop?

The **event loop is the scheduler that manages and runs asynchronous tasks/coroutines.**

A simplified view:

```text
while tasks_exist:

    find_task_that_can_run()

    run_task()

    if task_is_waiting_for_IO:
        suspend_task()

    run_another_task()
```

For example:

```python
async def task_a():
    print("A start")
    await some_io()
    print("A end")


async def task_b():
    print("B start")
    await some_io()
    print("B end")
```

The event loop can interleave their execution.

Conceptually:

```text
A start
B start
A waiting ─────┐
B waiting ───┐ │
              ↓ ↓
           I/O completes
              ↓
A end
B end
```

---

## Why is the event loop important?

Suppose your FastAPI endpoint does:

```python
@app.get("/users")
async def users():
    result = await db.execute(...)
    return result
```

When the database is processing the query:

```text
FastAPI
   ↓
await DB
   ↓
Coroutine pauses
   ↓
Event loop handles other requests
   ↓
DB completes
   ↓
Coroutine resumes
```

This allows one worker to efficiently handle many concurrent I/O operations.

---

# 3. What happens if you put blocking code inside `async def`?

This is **very important for interviews**.

Bad:

```python
import time


@app.get("/users")
async def users():
    time.sleep(5)
    return {"status": "done"}
```

`time.sleep()` is blocking.

You effectively block the event loop:

```text
Event Loop
    │
    ├── Request A
    │      ↓
    │   time.sleep(5)
    │      ↓
    │   BLOCKED
    │
    └── Other requests wait
```

For asynchronous endpoints, use async-compatible libraries:

```python
await asyncio.sleep(5)
```

instead of:

```python
time.sleep(5)
```

---

# 4. What about CPU-heavy work?

Async is primarily useful for **I/O-bound workloads**.

For example:

```text
Good async candidates:

Database
Redis
HTTP APIs
LLM APIs
Qdrant
File/network I/O
```

But something like:

```python
for i in range(10_000_000_000):
    ...
```

is CPU-heavy.

Async doesn't magically make CPU-intensive Python code concurrent.

For CPU-heavy workloads, consider:

```text
Multiple processes
ProcessPoolExecutor
Worker queues
GPU inference
Dedicated compute services
```

---

# 5. How do you make database calls asynchronously?

Use an **async-compatible database driver and ORM/session**.

For PostgreSQL with SQLAlchemy:

```bash
pip install sqlalchemy asyncpg
```

Then:

```python
from sqlalchemy.ext.asyncio import (
    create_async_engine,
    async_sessionmaker,
    AsyncSession,
)

DATABASE_URL = (
    "postgresql+asyncpg://user:password@localhost/mydb"
)

engine = create_async_engine(
    DATABASE_URL,
    pool_pre_ping=True,
)

SessionLocal = async_sessionmaker(
    engine,
    class_=AsyncSession,
    expire_on_commit=False,
)
```

---

# 6. FastAPI database dependency

You can create an async session dependency:

```python
from collections.abc import AsyncGenerator


async def get_db() -> AsyncGenerator[AsyncSession, None]:

    async with SessionLocal() as session:
        yield session
```

Then:

```python
from fastapi import Depends


@app.get("/users")
async def get_users(
    db: AsyncSession = Depends(get_db)
):

    result = await db.execute(
        select(User)
    )

    users = result.scalars().all()

    return users
```

The critical part is:

```python
await db.execute(...)
```

The database operation is asynchronous.

---

# 7. Why not use normal SQLAlchemy session?

You could use synchronous SQLAlchemy:

```python
db.execute(...)
```

but if you're building an async FastAPI application and your workload is heavily I/O-bound, using an async driver/session can prevent database I/O from blocking the event loop.

The stack becomes:

```text
FastAPI
   ↓
async endpoint
   ↓
AsyncSession
   ↓
asyncpg
   ↓
PostgreSQL
```

---

# 8. How do you make multiple external API calls concurrently?

This is one of the most important async patterns.

Suppose you need:

```text
OpenAI
+
Risk API
+
Market API
```

and each takes 1 second.

Sequential code:

```python
result1 = await call_api_1()
result2 = await call_api_2()
result3 = await call_api_3()
```

Approximate latency:

```text
1 sec + 1 sec + 1 sec
= 3 sec
```

If the calls are independent, execute them concurrently.

Use `asyncio.gather()`:

```python
import asyncio


results = await asyncio.gather(
    call_api_1(),
    call_api_2(),
    call_api_3(),
)
```

Approximate latency:

```text
max(1 sec, 1 sec, 1 sec)
≈ 1 sec
```

instead of:

```text
≈ 3 sec
```

---

# 9. FastAPI example

```python
import asyncio
import httpx
from fastapi import FastAPI

app = FastAPI()


async def get_weather(client: httpx.AsyncClient):
    response = await client.get(
        "https://example.com/weather"
    )
    response.raise_for_status()
    return response.json()


async def get_market(client: httpx.AsyncClient):
    response = await client.get(
        "https://example.com/market"
    )
    response.raise_for_status()
    return response.json()


async def get_news(client: httpx.AsyncClient):
    response = await client.get(
        "https://example.com/news"
    )
    response.raise_for_status()
    return response.json()


@app.get("/dashboard")
async def dashboard():

    async with httpx.AsyncClient() as client:

        weather, market, news = await asyncio.gather(
            get_weather(client),
            get_market(client),
            get_news(client),
        )

    return {
        "weather": weather,
        "market": market,
        "news": news,
    }
```

The three network calls can be in flight concurrently.

---

# 10. `asyncio.gather()` vs sequential `await`

### Sequential

```python
a = await service_a()
b = await service_b()
c = await service_c()
```

Timeline:

```text
A ──────
       B ──────
              C ──────

Total ≈ A + B + C
```

### Concurrent

```python
a, b, c = await asyncio.gather(
    service_a(),
    service_b(),
    service_c(),
)
```

Timeline:

```text
A ──────
B ──────
C ──────

Total ≈ max(A, B, C)
```

**provided the operations are independent and genuinely asynchronous.**

---

# 11. Important: don't create a new HTTP client for every request

This is a common production mistake.

You might see:

```python
async with httpx.AsyncClient() as client:
    ...
```

inside every request, which is acceptable for simple examples, but in production you often create a reusable client during application startup/lifespan so that HTTP connections can be pooled and reused.

Conceptually:

```text
FastAPI application
       │
       └── Shared AsyncClient
                │
        ┌───────┼────────┐
        ↓       ↓        ↓
       API A   API B    API C
```

This improves connection reuse and reduces overhead.

---

# 12. Concurrency in an LLM application

This is especially relevant to AI engineering.

Suppose your RAG pipeline needs:

```text
User Query
    │
    ├── Query expansion LLM
    ├── Metadata service
    ├── User profile
    └── Vector search
```

If they're independent:

```python
query_expansion, profile, documents = await asyncio.gather(
    generate_query_variants(query),
    get_user_profile(user_id),
    vector_search(query),
)
```

Then after all complete:

```text
query expansion
       +
user profile
       +
retrieved documents
       ↓
     Reranker
       ↓
       LLM
```

This can significantly reduce end-to-end latency.

---

# 13. Handling failures with concurrent calls

By default:

```python
await asyncio.gather(
    call_a(),
    call_b(),
    call_c(),
)
```

can propagate an exception when one task fails.

Sometimes you want to collect individual results/errors:

```python
results = await asyncio.gather(
    call_a(),
    call_b(),
    call_c(),
    return_exceptions=True,
)
```

Then:

```python
for result in results:
    if isinstance(result, Exception):
        # handle failure
        ...
```

This is useful when some dependencies are optional.

For example:

```text
Primary LLM → required
News API    → optional
Analytics   → optional
```

You might still return the answer if the optional service fails.

---

# 14. Semaphore for controlling concurrency

Suppose you have 1,000 external API calls.

Don't necessarily do:

```python
await asyncio.gather(*all_1000_calls)
```

You could overwhelm:

* your application
* the external API
* your connection pool
* your rate limits

Use a semaphore:

```python
import asyncio

semaphore = asyncio.Semaphore(10)


async def call_api(item):

    async with semaphore:
        return await external_api(item)
```

Now at most 10 calls are active at once.

```text
1000 tasks
    ↓
Semaphore(10)
    ↓
10 concurrent calls
    ↓
next 10
    ↓
next 10
...
```

This is an important production concurrency pattern.

---

# 15. Senior-level interview answer

If asked:

### "How does FastAPI handle asynchronous requests?"

> **"FastAPI is built on ASGI and uses Python's async/await model. Async endpoints run as coroutines managed by an event loop. When an endpoint awaits non-blocking I/O such as a database or HTTP request, the coroutine yields control so the event loop can process other requests. This makes FastAPI efficient for I/O-bound workloads."**

### "What is an event loop?"

> **"The event loop is the scheduler that manages asynchronous coroutines. It runs tasks until they reach an await point where they're waiting for I/O, then schedules other ready tasks and resumes the suspended task when its I/O completes."**

### "How do you make database calls asynchronously?"

> **"I use an async-compatible database driver and ORM/session, such as SQLAlchemy's `AsyncSession` with `asyncpg` for PostgreSQL. Database operations are awaited, for example `await session.execute(...)`, so database I/O doesn't block the event loop."**

### "How do you make multiple external API calls concurrently?"

> **"If the calls are independent, I use an async HTTP client such as `httpx.AsyncClient` and execute the coroutines with `asyncio.gather()`. This reduces latency from approximately the sum of the individual calls to roughly the duration of the slowest call, subject to connection limits and external rate limits. In production, I also use connection pooling, timeouts, retries, and concurrency limits such as semaphores."**

---

## The mental model you should remember

```text
                  FastAPI
                     │
                ASGI Server
                     │
                 Event Loop
                     │
        ┌────────────┼────────────┐
        ↓            ↓            ↓
    Request A    Request B    Request C
        │            │            │
     await DB     await LLM    await Redis
        │            │            │
        └────────────┼────────────┘
                     ↓
              Event loop runs
              other work
```

And for parallel operations inside one request:

```text
                 Request
                    │
              asyncio.gather()
                    │
        ┌───────────┼───────────┐
        ↓           ↓           ↓
      Qdrant      Redis       LLM API
        │           │           │
        └───────────┼───────────┘
                    ↓
              Combine results
```

**The key interview phrase is: *"Async improves concurrency for I/O-bound workloads; it does not automatically make CPU-bound Python code faster."***
