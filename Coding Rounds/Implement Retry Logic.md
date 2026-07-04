This is one of the **most frequently asked Senior AI Engineer / Backend Engineer / Distributed Systems interview questions**.

Interviewers don't want to know if you can write:

```python
for i in range(3):
    ...
```

They want to know:

* Why retries are needed
* Which errors should be retried
* Which errors should **never** be retried
* Exponential Backoff
* Jitter
* Idempotency
* Circuit Breaker
* Retry Budgets
* Production implementation

In production AI systems, **retry logic is critical** because almost every request depends on external systems:

```text
Client

↓

FastAPI

↓

Redis

↓

PostgreSQL

↓

Qdrant

↓

S3

↓

LLM API (OpenAI, Anthropic, etc.)

↓

Response
```

Any one of these services can fail temporarily.

---

# 1. Why Do We Need Retry Logic?

Suppose your AI application calls an LLM.

```text
POST /chat

↓

OpenAI API
```

Normally

```text
Response

200 OK
```

But sometimes

```text
HTTP 503

Service Unavailable
```

Does it mean

```text
OpenAI is broken?
```

No.

Maybe:

* Temporary overload
* Network timeout
* DNS issue
* Load balancer restart
* Transient cloud issue

Waiting briefly and trying again often succeeds.

---

# 2. Example Without Retry

```python
response = call_openai()
```

Network hiccup

↓

Exception

↓

User gets

```text
500 Internal Server Error
```

Poor user experience.

---

# 3. Retry Flow

```text
Request

↓

Call API

↓

Success?

↓

Yes

↓

Return

No

↓

Wait

↓

Retry

↓

Success?

↓

Return

↓

Else Retry Again

↓

Fail
```

---

# 4. Naive Retry

```python
import time

def retry(func):

    for attempt in range(3):

        try:
            return func()

        except Exception:

            time.sleep(1)

    raise Exception("All retries failed")
```

Works.

But it has problems.

---

# 5. Why Constant Delay Is Bad?

Suppose

```text
10,000 clients
```

All receive

```text
503
```

All wait

```text
1 second
```

Then all retry together.

This creates another traffic spike.

This phenomenon is called the **thundering herd problem**.

---

# 6. Exponential Backoff

Instead of

```text
1

1

1
```

Use

```text
1

2

4

8

16
```

Each retry waits longer.

Formula

[
delay = base \times 2^{attempt}
]

Example

| Attempt | Wait |
| ------- | ---- |
| 1       | 1 s  |
| 2       | 2 s  |
| 3       | 4 s  |
| 4       | 8 s  |

---

# 7. Why Add Jitter?

Even with exponential backoff:

```text
Client1 → 4 sec

Client2 → 4 sec

Client3 → 4 sec
```

They still retry together.

Instead:

```text
Client1

4.2 sec

Client2

3.7 sec

Client3

4.9 sec
```

Randomness spreads retries over time.

Example:

```python
delay = (2 ** attempt) + random.uniform(0, 1)
```

---

# 8. Production Retry Decorator

```python
import random
import time
from functools import wraps

def retry(
    retries=5,
    base_delay=1,
    exceptions=(Exception,)
):

    def decorator(func):

        @wraps(func)
        def wrapper(*args, **kwargs):

            for attempt in range(retries):

                try:
                    return func(*args, **kwargs)

                except exceptions as e:

                    if attempt == retries - 1:
                        raise

                    delay = (
                        base_delay
                        * (2 ** attempt)
                    )

                    delay += random.uniform(0, 1)

                    print(
                        f"Retry {attempt+1}"
                        f" after {delay:.2f}s"
                    )

                    time.sleep(delay)

        return wrapper

    return decorator
```

---

# 9. Using the Decorator

```python
import random

@retry(retries=4)

def unstable():

    if random.random() < 0.7:
        raise Exception("Temporary Error")

    return "Success"

print(unstable())
```

Possible output

```text
Retry 1 after 1.43s

Retry 2 after 2.51s

Retry 3 after 4.77s

Success
```

---

# 10. Async Retry (FastAPI)

Blocking `time.sleep()` should **never** be used inside async code.

Instead:

```python
import asyncio
import random

async def retry_async(func):

    for attempt in range(5):

        try:
            return await func()

        except Exception:

            if attempt == 4:
                raise

            delay = (
                2 ** attempt
            ) + random.random()

            await asyncio.sleep(delay)
```

Notice

```python
await asyncio.sleep()
```

does **not** block the event loop.

---

# 11. Which Errors Should Be Retried?

Retry only **transient failures**.

Examples:

* Network timeout
* Connection reset
* Temporary DNS failure
* HTTP 429 (Too Many Requests)
* HTTP 502 (Bad Gateway)
* HTTP 503 (Service Unavailable)
* HTTP 504 (Gateway Timeout)

These usually succeed after waiting.

---

# 12. Which Errors Should NOT Be Retried?

Never retry permanent errors.

Examples:

```text
400 Bad Request

401 Unauthorized

403 Forbidden

404 Not Found

Validation Error
```

Retrying these wastes time because the request itself is invalid.

---

# 13. Idempotency

Suppose

```text
Transfer

₹1000
```

Request times out.

Did the bank receive it?

You don't know.

Retrying might transfer

```text
₹1000

again.
```

Dangerous.

Therefore:

* **GET** requests are generally safe to retry.
* **POST** requests require care.

Production APIs often use an **Idempotency-Key** so that repeated requests with the same key are executed only once.

---

# 14. Retry Budget

Unlimited retries can overload a struggling service.

Production systems often define:

```text
Maximum retries = 3

Maximum retry time = 10 seconds

Maximum retry rate = 20%
```

This prevents retries from consuming excessive resources.

---

# 15. Circuit Breaker

Suppose

```text
LLM

↓

Always returning 503
```

Without protection:

```text
Retry

Retry

Retry

Retry

Retry
```

The service is overwhelmed further.

A **Circuit Breaker** works like this:

```text
Failures exceed threshold

↓

Open Circuit

↓

Immediately reject requests

↓

Wait

↓

Half-open

↓

Test one request

↓

Healthy?

↓

Close circuit
```

This gives the downstream service time to recover.

---

# 16. AI Production Example

A RAG pipeline:

```text
Client
   │
   ▼
FastAPI
   │
   ├────────► Redis
   ├────────► PostgreSQL
   ├────────► Qdrant
   ├────────► S3
   └────────► LLM API
```

Recommended strategy:

* Redis timeout → retry quickly.
* Vector DB timeout → retry with backoff.
* LLM rate limit (429) → exponential backoff with jitter.
* Authentication failure → fail immediately.
* Invalid prompt → fail immediately.

---

# 17. Production Best Practices

Senior engineers usually combine retries with:

* **Timeouts**: Every network call should have a bounded timeout.
* **Connection pooling**: Reuse HTTP and database connections.
* **Observability**: Record retry counts, latency, and failure reasons.
* **Cancellation**: Respect request cancellation to avoid wasted work.
* **Rate limiting**: Prevent retries from violating upstream quotas.
* **Bulkheads**: Isolate failures so one dependency doesn't consume all resources.
* **Libraries**: Mature retry libraries (e.g., Tenacity in Python) provide configurable retry policies, backoff strategies, and exception filtering.

---

# 18. How to Answer in a Senior AI Engineer Interview

A strong answer would be:

> "Retry logic is used to recover from transient failures such as network timeouts, temporary service unavailability, or rate limiting. A production retry strategy uses exponential backoff with jitter to prevent synchronized retry storms, retries only retryable exceptions, and limits both the number of attempts and the total retry duration. It should be combined with request timeouts, idempotency for operations that modify state, circuit breakers to prevent cascading failures, and comprehensive monitoring. In AI systems, retry logic is commonly applied to calls to LLM providers, vector databases, object storage, and other external services where temporary failures are expected."
