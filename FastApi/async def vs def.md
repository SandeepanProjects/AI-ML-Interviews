# `async def` vs `def` in FastAPI

This is a **very common senior FastAPI interview question**. The important thing is:

> **Don't say "`async def` is always faster than `def`."**
> The correct answer depends on whether your code performs **async I/O or blocking/synchronous work**.

---

# 1. The basic difference

FastAPI allows both:

```python
@app.get("/users")
async def get_users():
    ...
```

and:

```python
@app.get("/users")
def get_users():
    ...
```

The difference is whether the endpoint is a **coroutine function** that can use `await`.

### `async def`

```python
async def get_users():
    result = await database.fetch()
    return result
```

Can use:

```python
await
```

and cooperate with the event loop.

### `def`

```python
def get_users():
    result = database.fetch()
    return result
```

Runs normal synchronous Python code.

---

# 2. What happens with `async def`?

Consider:

```python
@app.get("/users")
async def get_users():

    users = await db.fetch_users()

    return users
```

The flow is:

```text
Request
   ↓
FastAPI
   ↓
async endpoint
   ↓
await database
   │
   │ database is working...
   │
   ▼
Event Loop handles other requests
   │
   ├── Request B
   ├── Request C
   └── Request D
   │
   ▼
Database completes
   │
   ▼
Endpoint resumes
   │
   ▼
Response
```

The key is:

```python
await db.fetch_users()
```

The endpoint can **pause while waiting for I/O**.

---

# 3. What happens with `def`?

Now:

```python
@app.get("/users")
def get_users():

    users = db.fetch_users()

    return users
```

This is synchronous code.

There's no:

```python
await
```

The database call blocks that execution until it returns.

FastAPI can handle normal `def` path operations by running them in an external thread pool rather than directly blocking the event loop.

Conceptually:

```text
Request
   ↓
FastAPI
   ↓
def endpoint
   ↓
Thread Pool
   │
   ├── Thread 1 → DB call
   ├── Thread 2 → DB call
   └── Thread 3 → DB call
```

This is an important FastAPI behavior to know.

---

# 4. Why does FastAPI support both?

Because Python applications use both:

### Async libraries

Examples:

```text
asyncpg
httpx.AsyncClient
redis.asyncio
SQLAlchemy AsyncSession
```

These work naturally with:

```python
async def
```

### Synchronous libraries

Examples:

```text
requests
traditional database drivers
blocking SDKs
legacy libraries
```

These may be used from:

```python
def
```

FastAPI accommodates both programming styles.

---

# 5. Real-world example: Async database

Suppose you're using SQLAlchemy async.

```python
from sqlalchemy import select


@app.get("/users/{user_id}")
async def get_user(
    user_id: int,
    db: AsyncSession = Depends(get_db)
):

    result = await db.execute(
        select(User).where(User.id == user_id)
    )

    return result.scalar_one_or_none()
```

Here:

```python
await db.execute(...)
```

is asynchronous I/O.

So:

```python
async def
```

is appropriate.

---

# 6. Real-world example: synchronous database

Suppose your database library is synchronous:

```python
@app.get("/users/{user_id}")
def get_user(user_id: int):

    user = db.query(User).filter(
        User.id == user_id
    ).first()

    return user
```

Using:

```python
def
```

is appropriate because the database API is blocking/synchronous.

---

# 7. The dangerous combination

One of the most important interview questions is:

> "What happens if you use blocking code inside `async def`?"

Example:

```python
import time


@app.get("/")
async def home():

    time.sleep(5)

    return {"message": "hello"}
```

This is bad.

Why?

Because:

```python
time.sleep(5)
```

blocks the thread running the event loop.

Conceptually:

```text
Event Loop
    │
    ▼
Request A
    │
    ▼
time.sleep(5)
    │
    │ BLOCKED
    │
    ▼
5 seconds later
    │
    ▼
Request A completes
```

During that time, the event loop cannot efficiently process other async tasks on that loop.

---

# 8. Correct async version

Use:

```python
import asyncio


@app.get("/")
async def home():

    await asyncio.sleep(5)

    return {"message": "hello"}
```

Now:

```text
Request A
   │
   ▼
await sleep
   │
   ▼
Event Loop
   ├── Request B
   ├── Request C
   └── Request D
```

When the 5 seconds are over:

```text
Event Loop
    ↓
Resume Request A
```

---

# 9. Another important example: HTTP requests

### Bad for an async endpoint

```python
import requests


@app.get("/weather")
async def weather():

    response = requests.get(
        "https://api.example.com/weather"
    )

    return response.json()
```

`requests.get()` is blocking.

You're using:

```text
async def
    +
blocking requests library
```

which can block the event loop.

---

# 10. Better approach

Use an async HTTP client:

```python
import httpx


@app.get("/weather")
async def weather():

    async with httpx.AsyncClient() as client:

        response = await client.get(
            "https://api.example.com/weather"
        )

    return response.json()
```

Now:

```text
async def
    +
httpx.AsyncClient
    +
await
    ↓
non-blocking I/O
```

This is the pattern you want in a high-concurrency API.

---

# 11. What about CPU-heavy work?

This is another interview trap.

Suppose:

```python
@app.get("/calculate")
async def calculate():

    result = expensive_calculation()

    return result
```

If:

```python
expensive_calculation()
```

takes 10 seconds of CPU time, making the function `async` doesn't make the computation asynchronous.

This:

```python
async def
```

does **not** automatically turn CPU work into background work.

For CPU-heavy tasks, you might use:

```text
FastAPI
   ↓
Task Queue
   ↓
Worker
   ↓
CPU/GPU
```

For example:

```text
FastAPI
   │
   ▼
Celery / task queue
   │
   ▼
Worker
   │
   ▼
ML inference
```

---

# 12. `async def` doesn't mean parallel execution

Suppose you have:

```python
async def task_a():
    ...


async def task_b():
    ...
```

Async allows concurrency:

```text
Task A ──────┐
             │
Task B ──────┤ Event Loop
             │
Task C ──────┘
```

It doesn't mean:

```text
CPU Core 1 → Task A
CPU Core 2 → Task B
CPU Core 3 → Task C
```

That's parallelism.

So remember:

```text
asyncio → concurrency

multiple processes/cores → parallelism
```

---

# 13. When should you use `async def`?

Use `async def` when your endpoint performs **async-compatible I/O**.

Typical examples:

### Database

```python
await db.execute(...)
```

### HTTP API

```python
await client.get(...)
```

### Redis

```python
await redis.get(...)
```

### Qdrant/vector database

```python
await qdrant.search(...)
```

### LLM API

```python
response = await llm.ainvoke(...)
```

For an AI application:

```text
FastAPI
   │
   ├── await PostgreSQL
   ├── await Redis
   ├── await Qdrant
   ├── await embedding API
   └── await LLM API
```

This is a very good use case for `async def`.

---

# 14. When should you use `def`?

Use normal `def` when the operation is naturally synchronous and doesn't need async I/O.

For example:

```python
@app.get("/health")
def health():

    return {"status": "ok"}
```

Or when you're calling a synchronous library and deliberately want FastAPI to handle that sync endpoint appropriately.

For example:

```python
@app.get("/legacy-data")
def get_data():

    return legacy_database_call()
```

---

# 15. FastAPI's important behavior

This is something worth memorizing for interviews.

### `async def`

FastAPI executes the coroutine in the async execution model.

```text
async def
   ↓
event loop
```

### `def`

FastAPI can run the synchronous endpoint using a thread pool so that the blocking function doesn't directly occupy the event-loop execution path.

```text
def
 ↓
thread pool
 ↓
worker thread
```

That's why this isn't simply:

> "`def` blocks FastAPI."

The more accurate answer is:

> A synchronous `def` path operation is handled through FastAPI/Starlette's thread-pool mechanism, whereas an `async def` path operation runs as a coroutine and can directly cooperate with the event loop.

---

# 16. What if a dependency is synchronous?

You might see:

```python
def get_db():
    ...
```

while your endpoint is:

```python
@app.get("/users")
async def get_users(
    db=Depends(get_db)
):
    ...
```

That's allowed.

FastAPI handles synchronous dependencies appropriately.

So you don't need to make **everything** async.

---

# 17. Async vs Def — Interview Table

|                                      | `async def`                              | `def`                             |
| ------------------------------------ | ---------------------------------------- | --------------------------------- |
| Coroutine                            | ✅                                        | ❌                                 |
| Can use `await`                      | ✅                                        | ❌                                 |
| Designed for async I/O               | ✅                                        | ❌                                 |
| Works with sync libraries            | Possible, but blocking code is dangerous | ✅                                 |
| Event loop                           | Directly participates                    | Sync execution handled separately |
| Thread pool                          | Not normally for the endpoint itself     | Yes, FastAPI/Starlette can use it |
| Great for DB/API/Redis async clients | ✅                                        | ❌                                 |
| Automatically faster                 | ❌                                        | ❌                                 |
| Good for simple synchronous code     | ✅ possible                               | ✅                                 |
| CPU-heavy work becomes async         | ❌                                        | ❌                                 |

---

# 18. Production AI Example

Imagine you're building:

```text
POST /chat
```

Your RAG pipeline is:

```text
User
 ↓
FastAPI
 ↓
Authentication
 ↓
PostgreSQL
 ↓
Qdrant
 ↓
Reranker
 ↓
LLM
 ↓
Response
```

Most of these are I/O operations.

You might write:

```python
@app.post("/chat")
async def chat(
    request: ChatRequest,
    db: AsyncSession = Depends(get_db)
):

    user = await user_service.get_user(
        db,
        request.user_id
    )

    documents = await retriever.search(
        request.question
    )

    answer = await llm_service.generate(
        request.question,
        documents
    )

    return {
        "answer": answer
    }
```

This is a good use of async.

The server can spend time waiting for:

```text
PostgreSQL
    ↓
Qdrant
    ↓
LLM
```

without unnecessarily blocking the event loop.

---

# 19. Senior-Level Question: Should every FastAPI endpoint be async?

**No.**

This is a great interview answer.

Say:

> "I don't make every endpoint async by default. I use `async def` when the endpoint performs async-compatible I/O. If the code uses synchronous libraries or is simple synchronous logic, `def` can be appropriate. The important thing is to avoid blocking operations inside an async endpoint because they can reduce the benefits of the event loop."

That's a much stronger answer than:

> "Always use async because it's faster."

---

# 20. A Very Common Interview Scenario

### Interviewer:

> You have an async FastAPI endpoint and it calls `requests.get()`. Is that okay?

Answer:

> "No, not ideal. `requests` is synchronous and blocking. Calling it directly from an async endpoint can block the event loop. I'd prefer an async HTTP client such as `httpx.AsyncClient`, or isolate the synchronous operation appropriately."

---

### Interviewer:

> What if I change `requests.get()` to `await requests.get()`?

Answer:

> "That won't work because `requests.get()` doesn't return an awaitable. `await` only works with asynchronous awaitable operations."

You need:

```python
await client.get(...)
```

where `client` is an async HTTP client.

---

# 21. The Mental Model

Remember this:

```text
                    FastAPI
                       │
             ┌─────────┴─────────┐
             │                   │
         async def              def
             │                   │
             ▼                   ▼
        Event Loop           Thread Pool
             │                   │
             ▼                   ▼
       async I/O            sync/blocking I/O
             │                   │
       await DB/API          DB/API
             │                   │
             ▼                   ▼
      Other requests       Other workers
       can progress
```

And the most important rule:

```text
async def
   +
async I/O
   +
await
   =
efficient concurrency
```

Whereas:

```text
async def
   +
blocking operation
   =
event-loop blocking
```

---

# 22. Best Interview Answer

If they ask:

> **"Explain `async` vs `def` in FastAPI."**

Give this answer:

> "`async def` defines a coroutine and is appropriate when the endpoint performs asynchronous I/O such as async database queries, Redis operations, HTTP calls or LLM API calls. When the coroutine reaches an `await`, it can suspend while waiting for I/O, allowing the event loop to process other work.
>
> A normal `def` endpoint is synchronous. FastAPI/Starlette can execute synchronous path operations through a thread pool, which is useful when using blocking or legacy libraries.
>
> I don't consider `async def` inherently faster. The benefit comes when the code inside it uses non-blocking asynchronous I/O. If I put blocking code such as `time.sleep()` or a synchronous HTTP request inside an async endpoint, I can block the event loop and hurt concurrency.
>
> For CPU-intensive operations, async doesn't provide parallelism, so I'd typically use separate worker processes, a task queue, or dedicated model-serving infrastructure.
>
> Therefore, my rule is: use `async def` with async-compatible I/O, and use `def` for synchronous code rather than making everything async by default."

### One-line version to memorize

**`async def` is for coroutine-based, non-blocking I/O; `def` is for synchronous code, and FastAPI can run sync endpoints in a thread pool.**
