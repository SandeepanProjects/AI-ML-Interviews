# What happens when blocking code is executed inside an async FastAPI endpoint?

This is a **very important senior FastAPI interview question**.

The short answer is:

> **Blocking code inside an `async def` endpoint can block the event loop, preventing it from efficiently processing other asynchronous requests. This increases latency and reduces concurrency.**

Let's understand exactly why.

---

# 1. First understand the event loop

Suppose FastAPI is running an async endpoint:

```python
@app.get("/users")
async def get_users():
    users = await db.fetch_users()
    return users
```

The simplified flow is:

```text
                    FastAPI
                       │
                       ▼
                  Event Loop
                       │
        ┌──────────────┼──────────────┐
        ▼              ▼              ▼
    Request A      Request B      Request C
        │              │              │
      await          await          await
        │              │              │
        ▼              ▼              ▼
    PostgreSQL       Redis          LLM
```

When Request A reaches:

```python
await db.fetch_users()
```

the database operation is I/O-bound.

The coroutine can suspend while waiting.

The event loop can then work on:

```text
Request B
Request C
Request D
...
```

---

# 2. Now introduce blocking code

Suppose you write:

```python
import time

@app.get("/users")
async def get_users():

    time.sleep(5)

    return {"message": "done"}
```

The problem is:

```python
time.sleep(5)
```

is **blocking**.

It doesn't say:

> "I'm waiting for something; event loop, go do other work."

Instead, it says:

> "Stop executing this thread for 5 seconds."

If that thread is running the event loop, the event loop is effectively stuck.

---

# 3. What does that look like?

Imagine three requests arrive:

```text
Request A
Request B
Request C
```

Without blocking:

```text
Event Loop

Request A → await DB ───────────┐
                                │
Request B → await Redis ────┐   │
                             │   │
Request C → await API ──────┤   │
                             │   │
                             ▼   ▼
                         Event Loop
```

The event loop can keep making progress.

But with:

```python
time.sleep(5)
```

you get:

```text
Event Loop

Request A
   │
   ▼
time.sleep(5)
   │
   │
   │ BLOCKED
   │
   ▼
5 seconds later
   │
   ▼
Request A completes
   │
   ▼
Request B
   │
   ▼
Request C
```

So Request B and Request C may have to wait.

---

# 4. Simple demonstration

Consider:

```python
import time

@app.get("/slow")
async def slow():

    time.sleep(5)

    return {"message": "slow"}
```

Now imagine:

```text
10:00:00 → Request A arrives
10:00:01 → Request B arrives
10:00:01 → Request C arrives
```

Request A executes:

```python
time.sleep(5)
```

The event loop is blocked.

Conceptually:

```text
10:00:00 ─────────────── 10:00:05

Request A
████████████████████████
      BLOCKING

Request B
       waiting...

Request C
       waiting...
```

Even though B and C might be simple operations, they can experience unnecessary latency.

---

# 5. This is why `async` does NOT automatically make code non-blocking

This is an important interview point.

Some developers think:

```python
async def
```

means:

> "Everything inside this function is asynchronous."

That's false.

For example:

```python
async def my_function():

    time.sleep(10)
```

The function is declared async, but:

```python
time.sleep()
```

is still synchronous and blocking.

So:

```text
async def
    +
blocking code
    =
potential event-loop blocking
```

---

# 6. Correct version

Instead of:

```python
import time

@app.get("/slow")
async def slow():

    time.sleep(5)

    return {"message": "done"}
```

use an asynchronous operation:

```python
import asyncio

@app.get("/slow")
async def slow():

    await asyncio.sleep(5)

    return {"message": "done"}
```

Now:

```text
Request A
   │
   ▼
await sleep
   │
   └───────────────┐
                   │
                   ▼
               Event Loop
                /   |   \
               /    |    \
              B     C     D
```

When the timer completes:

```text
Event Loop
    │
    ▼
Resume Request A
```

---

# 7. Real-world example: HTTP calls

This is much more important than `time.sleep()` in real applications.

### Bad

```python
import requests

@app.get("/weather")
async def weather():

    response = requests.get(
        "https://api.example.com/weather"
    )

    return response.json()
```

Why is this problematic?

`requests.get()` is synchronous.

While it's waiting for the external server:

```text
FastAPI Event Loop
        │
        ▼
requests.get()
        │
        │ waiting...
        │
        ▼
     BLOCKED
```

---

# 8. Better approach

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

Now the flow is:

```text
FastAPI
   │
   ▼
await client.get()
   │
   ▼
Event Loop
   │
   ├── Request B
   ├── Request C
   └── Request D
   │
   ▼
External API responds
   │
   ▼
Resume Request A
```

---

# 9. Real-world example: database

Suppose you're building a FastAPI application.

### Bad combination

```python
@app.get("/users")
async def users():

    users = synchronous_db.query_users()

    return users
```

If:

```python
synchronous_db.query_users()
```

performs blocking I/O, it can block the event loop.

### Better

Use an async database driver:

```python
@app.get("/users")
async def users():

    result = await async_db.fetch_users()

    return result
```

For example, with SQLAlchemy:

```python
result = await db.execute(
    select(User)
)
```

where `db` is an `AsyncSession`.

---

# 10. What about CPU-heavy code?

Blocking doesn't only mean I/O.

Consider:

```python
@app.get("/calculate")
async def calculate():

    result = huge_cpu_calculation()

    return result
```

If:

```python
huge_cpu_calculation()
```

takes 10 seconds of CPU time, declaring the endpoint `async` doesn't help.

The CPU-intensive function can occupy the execution thread.

Conceptually:

```text
Event Loop
    │
    ▼
CPU-heavy calculation
    │
    │ 10 seconds
    │
    ▼
Other async work delayed
```

---

# 11. How should CPU-heavy work be handled?

For expensive CPU/GPU work, move it outside the request/event-loop path.

For example:

```text
                FastAPI
                   │
                   │ submit job
                   ▼
               Job Queue
                   │
          ┌────────┴────────┐
          ▼                 ▼
       Worker 1          Worker 2
          │                 │
          ▼                 ▼
        CPU                GPU
```

Typical approaches include:

* multiple worker processes
* task queues
* Celery
* dedicated inference servers
* Kubernetes jobs/workers
* GPU workers

For an AI system:

```text
FastAPI
   │
   ├── authentication
   ├── validation
   └── request orchestration
             │
             ▼
       Model Server
             │
             ▼
           GPU
```

---

# 12. What about synchronous functions in FastAPI?

Here's an important nuance.

Suppose you write:

```python
@app.get("/users")
def get_users():

    users = synchronous_db.query_users()

    return users
```

This is different from:

```python
@app.get("/users")
async def get_users():

    users = synchronous_db.query_users()

    return users
```

FastAPI/Starlette can execute a normal `def` path operation in a thread pool.

Conceptually:

```text
                 FastAPI
                    │
              ┌─────┴─────┐
              │           │
          async def      def
              │           │
              ▼           ▼
         Event Loop    Thread Pool
                          │
                          ▼
                    Blocking code
```

That's why blindly converting every endpoint to `async def` can actually be a mistake if the underlying libraries are synchronous.

---

# 13. What about a blocking function inside `async def`?

Suppose you have:

```python
def blocking_operation():
    time.sleep(5)
    return "done"


@app.get("/test")
async def test():

    result = blocking_operation()

    return result
```

This is problematic.

You're calling:

```text
async endpoint
     ↓
blocking_operation()
     ↓
time.sleep()
     ↓
event loop blocked
```

---

# 14. One option: run blocking work in a thread

If you absolutely need to call a blocking function from async code, you can move it to a worker thread.

For example:

```python
import asyncio


def blocking_operation():
    time.sleep(5)
    return "done"


@app.get("/test")
async def test():

    result = await asyncio.to_thread(
        blocking_operation
    )

    return result
```

Conceptually:

```text
Event Loop
    │
    ▼
await asyncio.to_thread(...)
    │
    ▼
Thread Pool
    │
    ▼
blocking_operation()
    │
    ▼
result
    │
    ▼
Event Loop
```

This prevents the blocking function from directly occupying the event-loop thread.

---

# 15. AI/RAG Example

This is especially important for the kind of applications you're preparing for.

Imagine:

```text
POST /chat
```

Your pipeline is:

```text
User
 ↓
FastAPI
 ↓
Auth
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

Suppose you write:

```python
@app.post("/chat")
async def chat(request):

    docs = qdrant.search(request.question)

    response = requests.post(
        LLM_URL,
        json={"prompt": request.question}
    )

    return response.json()
```

This contains potentially blocking operations:

```text
qdrant.search()
requests.post()
```

If they're synchronous APIs, they can block the event loop.

---

# 16. Better AI architecture

Use async clients:

```python
@app.post("/chat")
async def chat(request):

    docs = await qdrant.search(
        request.question
    )

    response = await llm.generate(
        request.question,
        docs
    )

    return response
```

Now the architecture is:

```text
                  FastAPI
                     │
                 Event Loop
                     │
          ┌──────────┴──────────┐
          ▼                     ▼
    await Qdrant           await LLM
          │                     │
          └──────────┬──────────┘
                     ▼
                Response
```

---

# 17. What happens under high traffic?

Suppose you have:

```text
1,000 concurrent requests
```

and each request makes a blocking call:

```python
requests.get(...)
```

You can run into thread/resource limitations and increased latency.

With properly implemented async I/O:

```text
1,000 requests
      │
      ▼
Event Loop
      │
      ├── await DB
      ├── await Redis
      ├── await HTTP
      ├── await Qdrant
      └── await LLM
```

The server can efficiently manage many requests while those operations are waiting.

The exact throughput depends on many other factors, including:

* number of workers
* database pool size
* CPU
* network
* downstream service limits
* application code
* request size
* latency
* rate limits

So don't claim:

> "Async lets FastAPI handle unlimited requests."

It doesn't.

---

# 18. The most important distinction

Remember these three categories:

### 1. Async I/O

```python
await db.execute(...)
```

Good inside:

```python
async def
```

### 2. Blocking I/O

```python
requests.get(...)
```

Don't directly run this inside an async endpoint.

### 3. CPU-heavy work

```python
huge_ml_computation()
```

Don't expect `async` to make it faster.

---

# 19. Interview Table

| Code                           | Inside `async def`? |                                Problem? |
| ------------------------------ | ------------------: | --------------------------------------: |
| `await async_db.execute()`     |                   ✅ |                                      No |
| `await client.get()`           |                   ✅ |                                      No |
| `await redis.get()`            |                   ✅ |                                      No |
| `time.sleep(5)`                |                   ❌ |                       Blocks event loop |
| `requests.get()`               |                   ❌ |                                Blocking |
| synchronous DB query           |                   ❌ |                               Can block |
| CPU-heavy calculation          |                   ❌ |                               Can block |
| `asyncio.sleep(5)`             |                   ✅ |                                      No |
| `await asyncio.to_thread(...)` |                   ✅ | Appropriate for some blocking functions |

---

# 20. Best Senior Interview Answer

If the interviewer asks:

> **"What happens when blocking code is executed inside an async endpoint?"**

Say:

> "When blocking code executes directly inside an async FastAPI endpoint, it can block the event-loop thread. Because the event loop is responsible for scheduling other asynchronous tasks, those tasks cannot make progress while the blocking operation is running. This increases request latency and reduces concurrency.
>
> For example, calling `time.sleep()`, `requests.get()`, or a synchronous database driver directly inside `async def` can block the event loop. I would instead use async-compatible libraries and `await` their I/O operations. If I have to use a blocking library, I can isolate it in a thread using mechanisms such as `asyncio.to_thread()` or use a synchronous `def` endpoint, which FastAPI can execute via its thread-pool mechanism.
>
> For CPU-intensive operations such as model inference or large computations, async isn't the solution because those operations are CPU/GPU-bound. I'd move them to dedicated workers, a task queue, or a model-serving layer."

### The rule to remember

```text
async def
   +
async I/O
   +
await
   ↓
Good concurrency


async def
   +
blocking I/O
   ↓
Event loop blocked


async def
   +
CPU-heavy work
   ↓
Event loop/worker tied up
```

**The key interview phrase:**

> **"Async improves concurrency for I/O-bound work; it does not magically make blocking or CPU-bound code asynchronous."**
