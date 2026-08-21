In a production AI system, I would **not treat timeout, retry, and rate-limit failures as the same error**. I would classify them, apply different retry policies, and combine retries with **exponential backoff + jitter, deadlines, circuit breakers, and model fallback**.

A good architecture is:

```text
FastAPI
   │
   ▼
LLM Service
   │
   ▼
LLM Gateway
   │
   ├── Timeout handling
   ├── Retry policy
   ├── Rate-limit handling
   ├── Exponential backoff + jitter
   ├── Circuit breaker
   ├── Fallback model/provider
   └── Observability
   │
   ▼
LLM Provider
```

## 1. First classify the failure

I would distinguish at least these cases:

| Failure                   | Retry?      | Strategy                 |
| ------------------------- | ----------- | ------------------------ |
| Timeout                   | Usually yes | Exponential backoff      |
| `429 Too Many Requests`   | Yes         | Respect `Retry-After`    |
| `500`                     | Yes         | Exponential backoff      |
| `502/503/504`             | Yes         | Exponential backoff      |
| Invalid request `400`     | No          | Fix request              |
| Authentication `401/403`  | No          | Fix credentials          |
| Context too large         | No          | Reduce/transform prompt  |
| Content policy rejection  | Usually no  | Handle safely            |
| Repeated provider failure | Stop        | Fallback/circuit breaker |

The key point is:

> **Only retry transient failures.**

---

# 2. Exponential backoff

I would never do this:

```python
while True:
    try:
        return await call_llm()
    except Exception:
        continue
```

That's dangerous because it can create:

* infinite retries
* higher costs
* request storms
* worse rate limiting
* increased latency

Instead:

```text
attempt 1 → immediate
attempt 2 → 0.5 sec
attempt 3 → 1 sec
attempt 4 → 2 sec
attempt 5 → 4 sec
```

Usually I add **jitter**:

```text
delay = min(cap, base * 2^attempt) + random_jitter
```

Jitter prevents thousands of workers from retrying simultaneously.

---

# 3. Example implementation

I would centralize this in an LLM gateway rather than duplicating retry logic across every service.

```python
import asyncio
import random


class LLMGateway:

    MAX_RETRIES = 3
    BASE_DELAY = 0.5
    MAX_DELAY = 8.0

    async def generate(self, prompt: str):

        for attempt in range(self.MAX_RETRIES + 1):

            try:
                return await self._call_llm(prompt)

            except RateLimitError as exc:

                if attempt == self.MAX_RETRIES:
                    raise

                delay = self._get_retry_delay(
                    attempt,
                    retry_after=getattr(exc, "retry_after", None)
                )

                await asyncio.sleep(delay)

            except (TimeoutError, TemporaryLLMError):

                if attempt == self.MAX_RETRIES:
                    raise

                delay = self._get_retry_delay(attempt)

                await asyncio.sleep(delay)

            except AuthenticationError:
                raise

            except InvalidRequestError:
                raise

    def _get_retry_delay(
        self,
        attempt: int,
        retry_after: float | None = None,
    ) -> float:

        if retry_after is not None:
            return retry_after

        exponential = min(
            self.MAX_DELAY,
            self.BASE_DELAY * (2 ** attempt)
        )

        jitter = random.uniform(0, 0.2)

        return exponential + jitter
```

The important thing here is that **429 handling is different from normal timeout handling**.

---

# 4. Respect `Retry-After`

For rate limits, the provider may tell us when to retry.

For example:

```text
HTTP 429
Retry-After: 5
```

I would prefer:

```python
await asyncio.sleep(5)
```

rather than blindly calculating my own delay.

Conceptually:

```text
429
 │
 ├── Retry-After exists?
 │       │
 │      YES
 │       │
 │       ▼
 │   wait specified duration
 │
 └── NO
      │
      ▼
 exponential backoff + jitter
```

---

# 5. Timeout handling

I would use **bounded timeouts**.

For example:

```python
import asyncio


async def call_llm(prompt: str):

    try:
        async with asyncio.timeout(30):
            return await client.generate(prompt)

    except TimeoutError:
        raise LLMTimeoutError(
            "LLM request timed out"
        )
```

The exact timeout depends on the workload.

For example:

```text
Simple classification       → short timeout
Normal chat                  → moderate timeout
Large RAG generation        → longer timeout
Long-running agent workflow → workflow-level deadline
```

I prefer a **deadline** rather than allowing each retry to get its own unlimited timeout.

For example:

```text
Total request deadline = 30 seconds

Attempt 1 → 8 sec
Retry wait → 1 sec
Attempt 2 → 8 sec
Retry wait → 2 sec
Attempt 3 → remaining time
```

This prevents:

```text
3 retries × 30 second timeout
= 90+ seconds
```

for a request that should have completed within 30 seconds.

---

# 6. Rate limiting before calling the provider

There's another important distinction.

I don't want every request reaching the LLM provider and then receiving `429`.

I would implement **application-level rate limiting**.

For example:

```text
                    ┌── User A
                    │
Requests ──► Redis Rate Limiter ──► LLM Gateway
                    │
                    ├── User B
                    │
                    └── User C
```

For a multi-tenant AI platform, I might maintain:

```text
tenant_id
    ↓
requests/minute
tokens/minute
concurrent_requests
daily_budget
```

For example:

```python
allowed = await rate_limiter.allow(
    key=f"tenant:{tenant_id}",
    limit=100,
    window=60,
)

if not allowed:
    raise HTTPException(
        status_code=429,
        detail="Rate limit exceeded"
    )
```

Redis is commonly useful here because multiple FastAPI instances need to share the same rate-limit state.

---

# 7. Concurrency limiting

Rate limiting alone isn't enough.

Suppose a tenant is allowed 100 requests/minute, but 100 requests arrive simultaneously.

That could overload the LLM provider.

I would also limit concurrent LLM calls:

```python
semaphore = asyncio.Semaphore(20)


async def call_llm(prompt):

    async with semaphore:
        return await client.generate(prompt)
```

So:

```text
100 requests
     │
     ▼
Rate limiter
     │
     ▼
Concurrency limiter
     │
     ▼
LLM provider
```

---

# 8. Circuit breaker

If the provider is continuously failing, I don't want every request to retry.

For example:

```text
LLM Provider
    │
    X
    X
    X
    X
```

After enough failures:

```text
Circuit OPEN
```

New requests fail fast or go to a fallback provider.

Conceptually:

```text
CLOSED
  │
  │ repeated failures
  ▼
OPEN
  │
  │ wait
  ▼
HALF-OPEN
  │
  ├── success → CLOSED
  │
  └── failure → OPEN
```

This protects both our application and the provider.

---

# 9. Fallback model/provider

For a production AI platform, I would often have:

```text
Primary
OpenAI / Claude / Azure OpenAI
        │
        │ failure
        ▼
Fallback
another model/provider
```

For example:

```python
async def generate(prompt):

    try:
        return await primary_llm.generate(prompt)

    except RetryableLLMError:

        return await fallback_llm.generate(prompt)
```

But I wouldn't blindly fallback for every error.

For example:

```text
Invalid prompt       → don't fallback
Authentication error → don't fallback
Context too large    → modify prompt
Provider outage      → fallback
Timeout              → potentially fallback
429                  → potentially fallback
```

---

# 10. Model fallback based on task

For an enterprise AI system, fallback can be more intelligent.

For example:

```text
                    Request
                       │
                       ▼
                 Model Router
                       │
          ┌────────────┼─────────────┐
          ▼            ▼             ▼
       GPT model    Claude       Llama
          │
          │ failure
          ▼
       fallback
```

I could route based on:

* latency
* cost
* availability
* token limit
* task type
* model capability

For example:

```text
Complex reasoning
    → high-quality model

Simple classification
    → cheaper model

Primary unavailable
    → fallback model
```

---

# 11. Don't retry non-idempotent operations blindly

This is an important production consideration.

If an LLM call triggers a tool:

```text
LLM
 ↓
Tool
 ↓
"charge_customer()"
```

blindly retrying the entire operation could potentially execute the tool twice.

I would use:

* idempotency keys
* tool-level retry policies
* transaction boundaries
* state tracking

For example:

```text
request_id = abc123

charge_customer(request_id=abc123)
```

If the same request is retried, the downstream service can detect the duplicate.

---

# 12. Streaming requires special handling

Streaming is different.

Suppose:

```text
LLM
 ↓
token
 ↓
token
 ↓
token
 ↓
TIMEOUT
```

At this point we've already sent part of the response.

We can't simply return:

```python
HTTP 500
```

because the HTTP response has already started.

Instead:

```json
{"type": "error", "message": "Generation interrupted"}
```

and then terminate the stream.

Also, **retrying after partial token delivery is tricky** because the client may already have rendered part of the answer.

For streaming, I'd generally prefer:

```text
connect
   ↓
stream
   ↓
detect failure
   ↓
send error event
   ↓
close stream
```

rather than automatically restarting the entire generation.

---

# 13. Observability

Every retry should be observable.

I'd log something like:

```json
{
  "request_id": "abc123",
  "tenant_id": "tenant-1",
  "model": "primary-model",
  "attempt": 2,
  "error": "rate_limit",
  "retry_delay_ms": 1500
}
```

Metrics:

```text
llm_requests_total
llm_errors_total
llm_retries_total
llm_rate_limits_total
llm_timeouts_total
llm_fallback_total
llm_latency_seconds
llm_ttft_seconds
llm_tokens_total
llm_cost_total
```

This lets me answer:

> "Are we actually having an LLM reliability problem, or are we just seeing occasional transient failures?"

---

# 14. Production architecture

For the kind of enterprise RAG/agent platform you've been preparing for, I'd structure it approximately like this:

```text
                  FastAPI
                     │
                     ▼
              Authentication
                     │
                     ▼
             Tenant Rate Limit
                     │
                     ▼
              LLM Service
                     │
                     ▼
              LLM Gateway
                     │
       ┌─────────────┼──────────────┐
       │             │              │
       ▼             ▼              ▼
   Timeout        Retry        Circuit Breaker
       │             │              │
       └─────────────┼──────────────┘
                     ▼
               Primary LLM
                     │
              failure/429
                     │
                     ▼
              Fallback Model
                     │
                     ▼
                Response
```

And I would keep these policies **centralized in the LLM Gateway**, rather than implementing retry logic separately in every agent/service.

---

## Interview answer

A strong senior-level answer would be:

> **"I would centralize LLM resilience in an LLM gateway. First, I'd classify errors into transient and permanent failures. I'd retry transient errors such as timeouts, 429s, 500s, 502s, 503s and 504s using bounded exponential backoff with jitter. For 429 responses, I'd respect the provider's `Retry-After` header. I'd use an overall request deadline so retries don't cause unbounded latency. At the application level, I'd also use Redis-based rate limiting and concurrency limits to prevent overwhelming the provider. If the provider continues failing, I'd use a circuit breaker and potentially fail over to another model or provider. Permanent errors such as invalid requests or authentication failures shouldn't be retried. For streaming, I'd handle errors differently because the response may already have started, so I'd send a structured error event and terminate the stream rather than trying to return a new HTTP error. Finally, I'd instrument retries, timeouts, 429s, fallback rate, latency, TTFT, token usage and cost so the entire system is observable."**

### The senior-level keywords to remember

**Retry only transient errors → exponential backoff → jitter → `Retry-After` → deadline → rate limiting → concurrency limiting → circuit breaker → fallback model → idempotency → streaming cancellation → observability.**
