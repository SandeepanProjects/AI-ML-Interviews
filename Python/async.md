# Async / Await vs Threading vs Asyncio in Python

This is one of the **most important Senior Python / AI Engineer interview topics**.

You should understand:

* What is asynchronous programming?
* How `async` / `await` works internally
* Difference between threading and asyncio
* When to use which
* How it applies to AI systems (LLM APIs, RAG, data pipelines)

---

# 1. What is Asynchronous Programming?

Normally Python executes sequentially:

```python
def task():
    download()
    process()
    save()
```

Flow:

```text
download (wait 5 sec)
        |
        v
process
        |
        v
save
```

Total time:

```
5 + processing + saving
```

---

With async:

While one task is waiting, Python can execute another task.

Example:

```text
Task A
 |
 |---- waiting for API response
 |
Task B executes


Task A resumes
```

---

# 2. What is `async`?

`async` defines a coroutine.

Example:

```python
async def fetch_data():
    return "data"
```

This is not a normal function.

Calling:

```python
fetch_data()
```

does not execute it.

It returns a coroutine object.

Example:

```python
result = fetch_data()

print(result)
```

Output:

```
<coroutine object fetch_data>
```

---

# 3. What is `await`?

`await` pauses the current coroutine until another async operation completes.

Example:

```python
import asyncio


async def fetch():

    print("Start")

    await asyncio.sleep(2)

    print("Done")


asyncio.run(fetch())
```

Output:

```
Start

(wait 2 seconds)

Done
```

---

# How `await` Works Internally

When Python reaches:

```python
await asyncio.sleep(2)
```

it says:

> I am waiting. Give control back to the event loop.

The event loop runs other tasks.

---

# Event Loop

The heart of asyncio.

Think:

```
                 Event Loop

        -------------------------
        |                       |
        v                       v

     Task A                 Task B

 waiting API             CPU work

        |
        |
     Resume
```

The event loop schedules coroutines.

---

# Example: Without Async

Three API calls:

```python
import time


def api_call():

    time.sleep(2)


start=time.time()


api_call()
api_call()
api_call()


print(time.time()-start)
```

Output:

```
6 seconds
```

---

# With Async

```python
import asyncio
import time


async def api_call():

    await asyncio.sleep(2)


async def main():

    await asyncio.gather(
        api_call(),
        api_call(),
        api_call()
    )


start=time.time()

asyncio.run(main())

print(time.time()-start)
```

Output:

```
~2 seconds
```

Because calls run concurrently.

---

# asyncio.gather()

Runs multiple coroutines together.

Example:

```python
await asyncio.gather(
    task1(),
    task2(),
    task3()
)
```

Internally:

```
Event Loop

Task1 ---- waiting
Task2 ---- waiting
Task3 ---- waiting

Resume when ready
```

---

# Threading vs Asyncio

## Threading

Uses:

```
Multiple OS threads
```

Example:

```
Process

 |
 |-- Thread 1
 |
 |-- Thread 2
 |
 |-- Thread 3
```

---

## Asyncio

Uses:

```
Single thread
+
Event loop
```

Example:

```
Process

 |
 |
Event Loop

 |
 |-- Coroutine 1
 |-- Coroutine 2
 |-- Coroutine 3
```

---

# Major Difference

| Feature           | Threading        | Asyncio       |
| ----------------- | ---------------- | ------------- |
| Execution model   | Multiple threads | Single thread |
| Scheduler         | OS               | Event loop    |
| Context switching | Expensive        | Cheap         |
| Memory            | Higher           | Lower         |
| Parallel CPU      | No (GIL)         | No            |
| Best for          | Blocking I/O     | Massive I/O   |
| Complexity        | Medium           | Higher        |

---

# Example: Threading

```python
import threading
import time


def download():

    print("Downloading")

    time.sleep(2)

    print("Done")


threads=[]


for i in range(5):

    t=threading.Thread(
        target=download
    )

    threads.append(t)

    t.start()


for t in threads:
    t.join()
```

Creates:

```
5 OS threads
```

---

# Example: Asyncio

```python
import asyncio


async def download():

    print("Downloading")

    await asyncio.sleep(2)

    print("Done")


async def main():

    tasks=[]

    for i in range(5):

        tasks.append(
            download()
        )


    await asyncio.gather(*tasks)


asyncio.run(main())
```

Creates:

```
5 coroutines
```

inside:

```
1 thread
```

---

# When to Use Threading?

Use threading for:

## 1. Blocking libraries

Example:

```python
requests.get(url)
```

is blocking.

Thread:

```python
Thread(
 target=requests.get
)
```

---

## 2. File operations

```python
read_file()
write_file()
```

---

## 3. Database drivers without async support

Example:

```python
mysql.connector
```

---

# When to Use Asyncio?

Use asyncio for:

## 1. Many API calls

Example:

AI application:

```
User Request

 |
 |
Parallel calls:

OpenAI API
Vector DB
Database
Search API

 |
 |
Combine response
```

---

## 2. Web servers

FastAPI:

```python
@app.get("/chat")
async def chat():

    response = await llm_call()

    return response
```

---

## 3. WebSockets

Examples:

* Chat applications
* Streaming LLM tokens

---

# Async AI Example

Imagine RAG:

User asks:

```
Explain transformers
```

Need:

1. Retrieve documents
2. Query database
3. Call LLM

Sequential:

```
Vector DB
 2 sec

Database
 1 sec

LLM
 3 sec


Total = 6 sec
```

---

Async:

```python
async def rag_pipeline():

    docs, metadata, answer = await asyncio.gather(

        retrieve_documents(),

        get_user_context(),

        call_llm()

    )
```

Execution:

```
Vector DB  -----
                \
Database --------> Event Loop
                /
LLM API --------

Total ≈ 3 sec
```

---

# Important: Async Does NOT Mean Parallel

Common interview trap.

Async:

```
Concurrency
```

not:

```
Parallel execution
```

Example:

```
Thread:

CPU Core 1
Thread A

CPU Core 2
Thread B
```

Parallel.

---

Async:

```
Single CPU

Task A waiting

Task B runs
```

Concurrent.

---

# Asyncio and CPU Tasks

Bad:

```python
async def calculate():

    huge_matrix_operation()
```

Why?

Because:

```
Event loop blocked
```

Other tasks cannot run.

---

Solution:

Use:

```python
await asyncio.to_thread(cpu_function)
```

or:

```
multiprocessing
```

---

# Threading + Asyncio Together

Production systems often combine them.

Example:

FastAPI:

```
Request
 |
Async Event Loop
 |
 |
Blocking ML Model
 |
Thread Pool
 |
Inference
```

---

Example:

```python
async def endpoint():

    result = await asyncio.to_thread(
        blocking_model_call
    )

    return result
```

---

# GIL Impact

## Threading

Python threads:

```
Thread 1
 |
Python bytecode

Thread 2
 |
waits
```

Because of GIL.

---

## Asyncio

No GIL issue because:

```
one thread
one event loop
```

---

# Comparison with Multiprocessing

|            | Threading | Asyncio      | Multiprocessing |
| ---------- | --------- | ------------ | --------------- |
| CPU work   | ❌         | ❌            | ✅               |
| I/O work   | ✅         | ✅✅           | ❌               |
| Memory     | Medium    | Low          | High            |
| Complexity | Medium    | High         | Medium          |
| GIL        | Affected  | Not relevant | Bypassed        |

---

# Senior AI Engineer Production Example

## LLM Chat System

Architecture:

```
                User

                 |
                 v

             FastAPI

                 |
          Async Event Loop

                 |
        ------------------
        |        |       |
        v        v       v

    VectorDB   Redis   LLM API


                 |
                 v

            Stream Tokens
```

Why async?

Because:

* Thousands of users
* Mostly waiting on network
* Low memory usage
* High throughput

---

# Interview Answers

## Q: Difference between async and threading?

Strong answer:

> Threading uses multiple OS threads managed by the operating system, while asyncio uses a single thread with an event loop that schedules coroutines cooperatively. Threading is useful for blocking I/O, while asyncio is better for handling thousands of concurrent I/O operations.

---

## Q: Does async make CPU code faster?

Answer:

> No. Async improves concurrency for I/O-bound tasks. CPU-bound workloads require multiprocessing or optimized native code.

---

## Q: Why use asyncio in FastAPI?

Answer:

> FastAPI uses asyncio to handle many concurrent requests efficiently. While one request waits for database, vector search, or LLM API responses, the event loop can process other requests.

---

## Senior-level summary

```
Threading
-----------
Multiple OS threads
Good for blocking I/O


Asyncio
-----------
Single thread
Event loop
Good for massive I/O concurrency


Multiprocessing
-----------
Multiple processes
Good for CPU/GPU workloads
```

For modern AI systems:

* **FastAPI + asyncio** → request handling
* **Async HTTP clients** → LLM/API calls
* **Thread pools** → blocking libraries
* **Multiprocessing/GPU workers** → model inference and training

# Implement Async API + Concurrent Requests in Python (FastAPI + asyncio)

This is a **very common Senior AI Engineer interview question**.

Interviewers expect you to know:

* How to build async APIs
* How concurrent requests are handled
* Difference between sequential and concurrent execution
* Async database/API calls
* Handling blocking code
* Production patterns

---

# 1. Basic Async API

Install:

```bash
pip install fastapi uvicorn
```

---

## FastAPI Async Endpoint

```python
from fastapi import FastAPI
import asyncio


app = FastAPI()


@app.get("/hello")
async def hello():

    await asyncio.sleep(2)

    return {
        "message": "Hello"
    }
```

Run:

```bash
uvicorn main:app --reload
```

---

## What Happens Internally?

Request:

```
GET /hello
```

Flow:

```
Client
  |
  v
FastAPI
  |
  v
Async Event Loop
  |
  v
Coroutine created
  |
  v
await asyncio.sleep()
  |
  |
  Event loop handles other requests
  |
  |
Resume coroutine
  |
  v
Response
```

---

# 2. Sequential API Calls (Slow)

Imagine an AI chatbot.

Need:

1. User profile
2. Vector search
3. LLM response

Bad implementation:

```python
import asyncio


async def get_user():

    await asyncio.sleep(2)

    return "User"


async def search_vector_db():

    await asyncio.sleep(3)

    return "Documents"


async def call_llm():

    await asyncio.sleep(5)

    return "Answer"



@app.get("/chat")
async def chat():

    user = await get_user()

    docs = await search_vector_db()

    answer = await call_llm()


    return {
        "user": user,
        "docs": docs,
        "answer": answer
    }
```

---

Execution:

```
get_user
   |
   |----2 sec
   |
search_vector
   |
   |----3 sec
   |
LLM
   |
   |----5 sec


Total = 10 seconds
```

---

# 3. Concurrent Requests Using asyncio.gather()

We can execute independent tasks together.

```python
@app.get("/chat")
async def chat():

    user_task = get_user()

    vector_task = search_vector_db()

    llm_task = call_llm()


    user, docs, answer = await asyncio.gather(
        user_task,
        vector_task,
        llm_task
    )


    return {
        "user": user,
        "docs": docs,
        "answer": answer
    }
```

---

Execution:

```
             Event Loop


User API     -----
                  \
Vector DB    -------> Concurrent
                  /
LLM API      -----

```

Time:

```
max(2,3,5)

= 5 seconds
```

Instead of:

```
2+3+5

=10 seconds
```

---

# 4. Async External API Calls

Example:

Calling multiple services.

Install:

```bash
pip install httpx
```

---

## Async HTTP Client

```python
import httpx


async def call_service(url):

    async with httpx.AsyncClient() as client:

        response = await client.get(url)

        return response.json()
```

---

Use:

```python
@app.get("/aggregate")
async def aggregate():

    result = await asyncio.gather(

        call_service(
            "https://service1.com"
        ),

        call_service(
            "https://service2.com"
        ),

        call_service(
            "https://service3.com"
        )

    )


    return result
```

---

Architecture:

```
             API Gateway

                  |
                  |

        --------------------
        |        |         |
        v        v         v

     Service1 Service2 Service3


```

---

# 5. Concurrent Requests From Multiple Users

Suppose:

100 users hit:

```
/chat
```

at the same time.

## Sync API

```
Request 1
 |
Wait LLM 5 sec
 |
Response


Request 2
 |
Wait LLM 5 sec
 |
Response
```

Limited concurrency.

---

## Async API

```
Event Loop


Request 1
 |
 await LLM


Request 2
 |
 await LLM


Request 3
 |
 await LLM


Resume when responses arrive

```

One worker can handle many connections.

---

# 6. Streaming Async API (LLM Token Streaming)

Very important for AI interviews.

Example:

ChatGPT-like streaming:

```python
from fastapi.responses import StreamingResponse
import asyncio


async def generate_tokens():

    tokens = [
        "Hello",
        "how",
        "are",
        "you"
    ]


    for token in tokens:

        await asyncio.sleep(1)

        yield token



@app.get("/stream")
async def stream():

    return StreamingResponse(
        generate_tokens(),
        media_type="text/plain"
    )
```

---

Client receives:

```
Hello

(wait)

how

(wait)

are

(wait)

you
```

instead of waiting for full response.

---

# 7. Handling Blocking Code

Important interview question.

Bad:

```python
@app.get("/predict")
async def predict():

    result = model.predict(data)

    return result
```

Problem:

If:

```python
model.predict()
```

takes 30 seconds,

it blocks the event loop.

---

Solution:

Move blocking work to thread pool.

```python
@app.get("/predict")
async def predict():

    result = await asyncio.to_thread(
        model.predict,
        data
    )

    return result
```

Flow:

```
Async Event Loop

        |
        |
        v

Thread Pool

        |
        |
        v

Blocking ML Model

```

---

# 8. Async Database Example

Using async DB driver:

```python
@app.get("/users/{id}")
async def get_user(id:int):

    user = await database.fetch_one(
        "SELECT * FROM users WHERE id=:id",
        {
            "id":id
        }
    )

    return user
```

Database waiting does not block other requests.

---

# 9. Limiting Concurrent Requests

Problem:

1000 users call LLM.

GPU overload.

Use Semaphore.

```python
llm_limit = asyncio.Semaphore(10)


async def call_llm():

    async with llm_limit:

        response = await llm.generate()

        return response
```

Now:

```
Only 10 LLM calls running

Others wait
```

---

# 10. Async Retry Pattern

Production APIs fail.

Example:

```python
async def retry_async(
    func,
    retries=3
):

    for attempt in range(retries):

        try:

            return await func()

        except Exception:

            await asyncio.sleep(
                2 ** attempt
            )


    raise Exception(
        "Failed"
    )
```

Backoff:

```
Attempt 1
wait 1 sec

Attempt 2
wait 2 sec

Attempt 3
wait 4 sec
```

---

# 11. Production AI Chat API Architecture

```
                 User

                  |
                  v

              FastAPI

                  |
          Async Event Loop

                  |
     ---------------------------
     |             |            |
     v             v            v

 PostgreSQL     Redis       Vector DB


                  |
                  v

              LLM API


                  |
                  v

           Streaming Response

```

---

# Why Async is Important for AI Systems?

LLM applications are mostly:

```
Request
 |
 |
Waiting...
 |
External API
 |
Vector DB
 |
Database
 |
LLM
```

They spend time waiting, not computing.

Async allows:

```
While Request A waits

Process Request B

Process Request C

Process Request D
```

---

# Interview Questions

## Q1. Why use async API?

Answer:

> Async APIs improve throughput by allowing a server to handle many concurrent I/O-bound requests without blocking threads. While one request waits for database, vector search, or LLM API responses, the event loop can process other requests.

---

## Q2. async vs threading?

Answer:

> Threading uses multiple OS threads, while asyncio uses cooperative multitasking with a single event loop. Asyncio is more efficient for thousands of I/O operations because it avoids thread creation and context switching overhead.

---

## Q3. Does async improve CPU performance?

Answer:

> No. Async improves I/O concurrency. CPU-intensive operations should use multiprocessing, GPU workers, or background job systems.

---

## Q4. How do you handle blocking ML inference in async API?

Answer:

> Move blocking operations to a thread pool using `asyncio.to_thread()`, or use separate inference services behind the API.

---

# Senior AI Engineer Summary

```
FastAPI
   |
async def endpoint
   |
await
   |
asyncio Event Loop
   |
------------------------
|          |            |
DB       Vector DB     LLM
(wait)    (wait)      (wait)
|
Resume when ready
```

Production pattern:

* FastAPI async endpoints
* asyncio.gather() for parallel calls
* Async HTTP clients
* Async database drivers
* Semaphore for rate limiting
* Thread pool for blocking ML code
* StreamingResponse for LLM tokens

This is the architecture used in scalable RAG and LLM applications.
