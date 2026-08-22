The key difference is:

> **`time.sleep()` blocks the current thread. `asyncio.sleep()` pauses the current coroutine and allows the event loop to run other tasks.**

---

## 1. `time.sleep()`

`time.sleep()` is **synchronous and blocking**.

```python
import time

print("Start")

time.sleep(3)

print("End")
```

The current thread is blocked for 3 seconds.

In FastAPI:

```python
@app.get("/test")
async def test():
    time.sleep(5)
    return {"message": "done"}
```

This is problematic because you're inside an `async` endpoint but blocking the event loop.

Conceptually:

```text
Event Loop
    │
    ▼
Request A
    │
time.sleep(5)
    │
    ❌ Event loop blocked
    │
    ▼
Other requests have to wait
```

---

# 2. `asyncio.sleep()`

`asyncio.sleep()` is **non-blocking** when awaited.

```python
import asyncio

async def test():
    print("Start")

    await asyncio.sleep(3)

    print("End")
```

When this reaches:

```python
await asyncio.sleep(3)
```

the coroutine pauses and gives control back to the event loop.

```text
Event Loop
    │
    ▼
Request A
    │
await asyncio.sleep()
    │
    ├── Coroutine A paused
    │
    ├── Run Request B
    ├── Run Request C
    └── Run Request D
             │
             ▼
       3 seconds later
             │
             ▼
       Resume Request A
```

---

# 3. Direct comparison

|                                      | `time.sleep()`                   | `asyncio.sleep()` |
| ------------------------------------ | -------------------------------- | ----------------- |
| Type                                 | Synchronous                      | Asynchronous      |
| Blocks thread?                       | ✅ Yes                            | ❌ No              |
| Blocks event loop?                   | ✅ If called on event-loop thread | ❌                 |
| Requires `await`?                    | ❌                                | ✅                 |
| Suitable for async FastAPI endpoint? | ❌ Generally no                   | ✅                 |
| Useful for async I/O simulation      | ❌                                | ✅                 |

---

# 4. See the difference

### Blocking

```python
import time

async def task(name):
    print(f"{name} started")
    time.sleep(2)
    print(f"{name} finished")
```

If you run:

```python
await asyncio.gather(
    task("A"),
    task("B"),
    task("C")
)
```

you don't get useful concurrency because each `time.sleep()` blocks execution.

Conceptually:

```text
A start
A waits 2 sec
A finish

B start
B waits 2 sec
B finish

C start
C waits 2 sec
C finish

≈ 6 seconds
```

---

### Non-blocking

```python
import asyncio

async def task(name):
    print(f"{name} started")
    await asyncio.sleep(2)
    print(f"{name} finished")
```

Now:

```python
await asyncio.gather(
    task("A"),
    task("B"),
    task("C")
)
```

The tasks can wait concurrently:

```text
A start ─────────┐
B start ─────────┤  2 seconds
C start ─────────┘
                 ↓
A finish
B finish
C finish

≈ 2 seconds
```

---

# 5. FastAPI example

### ❌ Bad

```python
@app.get("/slow")
async def slow():
    time.sleep(5)

    return {"message": "done"}
```

You're blocking the event loop.

### ✅ Good

```python
@app.get("/slow")
async def slow():
    await asyncio.sleep(5)

    return {"message": "done"}
```

The event loop can process other requests during those 5 seconds.

---

# 6. But there's an important nuance

`asyncio.sleep()` is **not an alternative to every use of `time.sleep()`**.

If you're writing ordinary synchronous Python:

```python
def process():
    time.sleep(2)
```

that's perfectly reasonable if you intentionally want to block that thread.

The issue is specifically when blocking code runs on the **async event-loop thread**.

---

# 7. What about blocking libraries?

This is a common senior FastAPI interview question.

Suppose you have:

```python
@app.get("/data")
async def get_data():

    result = requests.get(
        "https://example.com"
    )

    return result.json()
```

`requests` is synchronous/blocking.

Even though the endpoint is:

```python
async def
```

the HTTP call can still block the event loop.

Use an async client:

```python
import httpx


@app.get("/data")
async def get_data():

    async with httpx.AsyncClient() as client:
        response = await client.get(
            "https://example.com"
        )

    return response.json()
```

So remember:

```text
async def
   +
blocking library
   =
❌ potentially blocked event loop
```

Whereas:

```text
async def
   +
async library
   +
await
   =
✅ non-blocking I/O
```

---

# 8. If you absolutely must run blocking code

You can move blocking work away from the event loop.

For example:

```python
import asyncio


def blocking_function():
    time.sleep(5)
    return "done"


@app.get("/test")
async def test():

    result = await asyncio.to_thread(
        blocking_function
    )

    return {"result": result}
```

Conceptually:

```text
Event Loop
    │
    ├── async request
    │
    └── to_thread()
           ↓
       Thread
           ↓
    blocking_function()
```

This prevents the blocking operation from directly blocking the event loop.

---

# Interview answer

If the interviewer asks:

**"What's the difference between `asyncio.sleep()` and `time.sleep()`?"**

Say:

> **"`time.sleep()` is a blocking synchronous operation. It blocks the current thread, and if called inside an async FastAPI endpoint, it can block the event loop and prevent other requests from being processed. `asyncio.sleep()` is an awaitable non-blocking operation. When we `await` it, the coroutine is suspended and the event loop can execute other tasks. Therefore, in asynchronous FastAPI code, I use `asyncio.sleep()` for delays and avoid blocking calls on the event-loop thread."**

### One-line memory trick:

```text
time.sleep()
    → BLOCK the thread

await asyncio.sleep()
    → PAUSE the coroutine, free the event loop
```

This distinction is fundamental to understanding **FastAPI async performance, concurrent database/API calls, and LLM streaming applications**.
