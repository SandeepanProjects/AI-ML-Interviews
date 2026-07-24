Handling rate limits is a very common **Senior AI Engineer interview question** because every production AI application eventually hits provider limits (OpenAI, Anthropic, Gemini, Azure OpenAI, Pinecone, etc.).

A production system should **never fail immediately** when it receives a **429 Too Many Requests** response. Instead, it should retry, back off, throttle traffic, and fall back gracefully.

---

# What is a Rate Limit?

Suppose OpenAI allows

```text
100 Requests / Minute
```

If your application sends

```text
120 Requests / Minute
```

The last 20 requests may receive

```http
HTTP 429 Too Many Requests
```

Without handling it:

```text
User

↓

LLM

↓

429

↓

Application Crash
```

With proper handling:

```text
User

↓

LLM

↓

429

↓

Retry

↓

Success
```

---

# Production Strategy

A robust production system usually combines multiple techniques.

```text
                User Requests
                      │
                      ▼
               Rate Limiter
                      │
             Queue Requests
                      │
                      ▼
             Retry + Backoff
                      │
         ┌────────────┴────────────┐
         ▼                         ▼
     Success                 Fallback Model
```

---

# 1. Exponential Backoff

Never retry immediately.

Bad:

```text
429

↓

Retry Immediately

↓

429

↓

Retry Immediately

↓

429
```

Good:

```text
429

↓

Wait 1 second

↓

Retry

↓

Wait 2 seconds

↓

Retry

↓

Wait 4 seconds

↓

Retry
```

Python

```python
import time

def call_llm():

    retries = 5

    delay = 1

    for _ in range(retries):

        try:
            return llm.invoke("Hello")

        except RateLimitError:

            time.sleep(delay)

            delay *= 2

    raise Exception("Max retries exceeded")
```

---

# 2. Add Jitter

If 1,000 servers retry after exactly 2 seconds, they overload the service again.

Instead:

```text
Retry

↓

2.3 sec

Retry

↓

1.8 sec

Retry

↓

2.7 sec
```

Python

```python
import random

delay = delay + random.uniform(0, 1)
```

This spreads retry attempts over time.

---

# 3. Respect `Retry-After`

Many APIs return a `Retry-After` header.

Example

```http
429 Too Many Requests

Retry-After: 8
```

Meaning:

```text
Wait exactly 8 seconds.
```

Python

```python
retry_after = response.headers.get(
    "Retry-After"
)

time.sleep(int(retry_after))
```

Prefer the server-provided value over guessing.

---

# 4. Queue Requests

Don't send 1,000 requests simultaneously.

```text
1000 Requests

↓

Queue

↓

10 Workers

↓

LLM
```

Example

```python
import asyncio

queue = asyncio.Queue()
```

Workers

```python
while True:

    job = await queue.get()

    await process(job)
```

---

# 5. Limit Concurrency

Instead of

```text
500 Concurrent Calls
```

Allow only

```text
20 Concurrent Calls
```

Python

```python
import asyncio

semaphore = asyncio.Semaphore(20)

async def ask(prompt):

    async with semaphore:
        return await llm(prompt)
```

This protects both your application and the provider.

---

# 6. Token Bucket Rate Limiter

A common algorithm.

```text
Bucket Capacity = 100

↓

Each Request Uses 1 Token

↓

Bucket Refills Over Time
```

Python

```python
from limiter import Limiter

limiter = Limiter(rate=10)

@limiter
def call_llm():
    ...
```

Useful when you control outgoing traffic.

---

# 7. Redis Distributed Rate Limiting

If you have many application servers:

```text
App 1

App 2

App 3

↓

Redis Counter

↓

LLM
```

All instances share the same request budget.

Example

```python
import redis

r = redis.Redis()

count = r.incr("llm_requests")

r.expire("llm_requests", 60)
```

---

# 8. Cache Responses

Many repeated requests never need to reach the LLM.

```text
User

↓

Redis Cache

↓

Hit

↓

No LLM Call
```

This reduces both cost and the chance of hitting rate limits.

---

# 9. Model Fallback

If one provider is rate limited:

```text
OpenAI

↓

429

↓

Anthropic

↓

Success
```

Python

```python
try:
    return openai_llm(prompt)

except RateLimitError:

    return anthropic_llm(prompt)
```

---

# 10. Circuit Breaker

If failures continue, stop sending requests temporarily.

```text
429

↓

429

↓

429

↓

Circuit Opens

↓

Wait

↓

Retry Later
```

Example

```python
from pybreaker import CircuitBreaker

breaker = CircuitBreaker(
    fail_max=5,
    reset_timeout=60
)
```

This prevents overwhelming a failing dependency.

---

# 11. Batch Requests

Instead of

```text
100 Small Calls
```

Use

```text
1 Batch Call
```

Example

```python
embeddings = embed(
    [
        doc1,
        doc2,
        doc3
    ]
)
```

Batching is especially effective for embedding APIs.

---

# 12. Prioritize Requests

High-priority traffic first.

```text
VIP Users

↓

Paid Users

↓

Free Users
```

Low-priority work can wait in the queue.

---

# 13. Monitor Rate Limits

Track:

```text
429 Errors

Requests/Minute

Retries

Retry Success Rate

Queue Length

Average Wait Time
```

Example

```python
from prometheus_client import Counter

rate_limit_errors = Counter(
    "rate_limit_errors",
    "429 Errors"
)

rate_limit_errors.inc()
```

---

# Production Architecture

```text
                    User
                      │
                      ▼
                API Gateway
                      │
                      ▼
             Redis Rate Limiter
                      │
                      ▼
                Request Queue
                      │
                      ▼
          Semaphore (20 Concurrent)
                      │
                      ▼
             Retry + Backoff
                      │
         ┌────────────┴─────────────┐
         ▼                          ▼
     OpenAI                    Anthropic
         │                          │
         └────────────┬─────────────┘
                      ▼
                Return Response
```

---

# Complete Example

```python
import asyncio
import random

semaphore = asyncio.Semaphore(10)

async def ask_llm(prompt):

    delay = 1

    async with semaphore:

        for _ in range(5):

            try:
                return await llm(prompt)

            except RateLimitError:

                await asyncio.sleep(
                    delay + random.random()
                )

                delay *= 2

    raise Exception("Failed after retries")
```

This combines:

* Concurrency control
* Exponential backoff
* Random jitter
* Maximum retry count

---

# What Senior AI Teams Do

Production AI systems typically combine several strategies:

| Problem                              | Solution                             |
| ------------------------------------ | ------------------------------------ |
| Temporary 429 errors                 | Exponential backoff + jitter         |
| Provider specifies wait time         | Respect `Retry-After` header         |
| Too many concurrent requests         | Semaphores or worker pools           |
| Multiple application instances       | Redis-based distributed rate limiter |
| Repeated prompts                     | Redis/semantic cache                 |
| Provider outage or persistent limits | Multi-provider fallback              |
| Continuous failures                  | Circuit breaker                      |
| Large numbers of small requests      | Batching                             |
| Traffic spikes                       | Request queues and prioritization    |

## Interview Answer

If asked **"How would you handle rate limits in a production AI system?"**, a strong answer is:

> "I would use exponential backoff with jitter for retries, respect the provider's `Retry-After` header when available, limit concurrency with semaphores or worker pools, queue excess requests, cache repeated responses to reduce API calls, monitor 429 rates and retry metrics, and configure fallback models or providers for high availability. For horizontally scaled services, I'd enforce distributed rate limiting with Redis so all instances share the same request budget."

That answer demonstrates knowledge of **resilience, scalability, observability, and cost optimization**, which are all important aspects of production AI systems.
