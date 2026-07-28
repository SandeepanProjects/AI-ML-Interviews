The **Retry Pattern** is one of the most important reliability patterns in **Agentic AI** and distributed systems. It enables an AI agent to **automatically recover from transient failures** instead of immediately failing.

Every production AI platform—whether it's an enterprise copilot, RAG system, or autonomous agent—implements retries because failures are inevitable:

* LLM API timeouts
* Rate limits (HTTP 429)
* Temporary network issues
* Tool failures
* Database connection problems
* Vector database unavailability
* External API outages

The key is to retry **intelligently**, not endlessly.

---

# What is the Retry Pattern?

Without retries:

```text
User

↓

LLM Call

↓

Timeout

↓

Request Failed
```

With retries:

```text
User

↓

LLM Call

↓

Timeout

↓

Retry

↓

LLM Call

↓

Success
```

The system automatically attempts the operation again before giving up.

---

# Why Are Retries Needed?

Imagine an AI assistant calls a weather API.

```text
User

↓

Weather Tool

↓

503 Service Unavailable

↓

Retry

↓

Weather Tool

↓

Success

↓

Final Answer
```

Without retries, the user would unnecessarily receive an error even though the service recovered a moment later.

---

# Types of Failures

## Transient Failures (Retry)

These are temporary problems:

* Network timeout
* HTTP 429 (rate limit)
* HTTP 503 (service unavailable)
* Temporary database outage
* Temporary DNS failure

Retries are appropriate here.

---

## Permanent Failures (Don't Retry)

These include:

* Invalid API key (401)
* Permission denied (403)
* Invalid request (400)
* Missing required parameters
* Schema validation errors

Retrying these wastes time and money.

---

# Basic Retry Logic

```python
MAX_RETRIES = 3

for attempt in range(MAX_RETRIES):
    try:
        result = call_llm()
        break
    except TimeoutError:
        continue
```

Execution:

```text
Attempt 1

↓

Timeout

↓

Attempt 2

↓

Timeout

↓

Attempt 3

↓

Success
```

---

# Retry with Exponential Backoff

Never retry immediately.

Instead:

```text
Retry 1 → 1 second

Retry 2 → 2 seconds

Retry 3 → 4 seconds

Retry 4 → 8 seconds
```

Python example:

```python
import time

delay = 1

for _ in range(4):

    try:
        return tool()

    except Exception:

        time.sleep(delay)

        delay *= 2
```

This prevents overwhelming an already struggling service.

---

# Retry with Jitter

If thousands of clients retry simultaneously, they create a **thundering herd** problem.

Instead:

```text
Client A → 2.1 sec

Client B → 2.7 sec

Client C → 3.0 sec
```

Example:

```python
import random
import time

delay = 2

time.sleep(delay + random.uniform(0, 1))
```

---

# Retry in LangGraph

State:

```python
from typing import TypedDict

class AgentState(TypedDict):
    retries: int
    question: str
```

Tool node:

```python
def weather_node(state):

    try:
        return {"weather": weather_api()}

    except Exception:
        return {
            "retries": state["retries"] + 1
        }
```

Router:

```python
MAX_RETRIES = 3

def router(state):

    if state["retries"] >= MAX_RETRIES:
        return "fallback"

    return "retry"
```

Workflow:

```text
Weather Tool

↓

Failure

↓

Retry Count

↓

Retry?

↓

Yes

↓

Try Again

↓

No

↓

Fallback
```

---

# Retry in AgentExecutor

Limit retries inside tools rather than allowing the agent to call the same failing tool indefinitely.

```python
from tenacity import retry, stop_after_attempt

@retry(stop=stop_after_attempt(3))
def weather_tool(city):
    ...
```

This isolates retry logic within the tool.

---

# Retry with Fallback Models

Suppose the primary model fails.

```text
GPT-4.1

↓

Timeout

↓

Retry

↓

Timeout

↓

Fallback

↓

GPT-4.1-mini
```

Example:

```python
try:
    return primary.invoke(prompt)

except Exception:
    return backup.invoke(prompt)
```

---

# Retry in RAG

```text
Question

↓

Retrieve Documents

↓

Timeout

↓

Retry

↓

Retrieve

↓

Success
```

If retrieval repeatedly fails:

```text
Vector Search

↓

Failed

↓

BM25 Search

↓

Failed

↓

Cached Answer
```

---

# Retry for Tool Calls

```text
Agent

↓

SQL Tool

↓

Connection Lost

↓

Retry

↓

Success
```

Retry only if the tool operation is **safe**.

---

# Idempotency

A retry should not accidentally perform the same side effect multiple times.

Safe example:

```text
Read Weather

↓

Retry

↓

Read Weather
```

Unsafe example:

```text
Transfer ₹1,000

↓

Timeout

↓

Retry

↓

Transferred Twice
```

For write operations, use idempotency keys or transaction IDs.

---

# Circuit Breaker + Retry

Retries alone are not enough.

```text
API

↓

Failure

↓

Retry

↓

Failure

↓

Retry

↓

Failure

↓

Circuit Opens

↓

Fallback
```

The circuit breaker prevents endless retries against a failing service.

---

# Retry Budget

Set limits.

```text
Maximum Retries = 3

Maximum Delay = 30 sec

Maximum Request Time = 60 sec
```

This keeps latency predictable.

---

# Observability

Track:

```text
Retry Count

↓

Retry Reason

↓

Latency

↓

Success Rate

↓

Fallback Usage
```

Useful metrics:

* Average retries/request
* Retry success rate
* Retry latency
* Timeouts
* Fallback rate

---

# Production Architecture

```text
                    User
                      │
                      ▼
                LangGraph Agent
                      │
                      ▼
                  Tool Call
                      │
            Success? /      \
                 Yes        No
                  │          │
                  ▼          ▼
            Continue     Retry Policy
                              │
                    Retry Limit Reached?
                        /             \
                      No             Yes
                      │               │
                      ▼               ▼
                  Retry Tool    Fallback Model
                                      │
                                      ▼
                                Return Response
```

---

# Common Retry Mistakes

### 1. Infinite retries

Bad:

```text
Retry

↓

Retry

↓

Retry

↓

Forever
```

Always define a maximum retry count.

---

### 2. Retrying everything

Do **not** retry:

* 400 Bad Request
* 401 Unauthorized
* 403 Forbidden
* Validation errors

---

### 3. No backoff

Bad:

```text
Retry

↓

Retry

↓

Retry

↓

Retry
```

Good:

```text
1 sec

↓

2 sec

↓

4 sec

↓

8 sec
```

---

### 4. Retrying non-idempotent operations

Never blindly retry:

* Payments
* Money transfers
* Order creation
* Sending emails
* Deleting records

Unless idempotency is guaranteed.

---

# Retry vs Reflection

| Retry Pattern                 | Reflection Pattern         |
| ----------------------------- | -------------------------- |
| Handles execution failures    | Improves answer quality    |
| Retries tools or APIs         | Reviews generated output   |
| Focuses on reliability        | Focuses on correctness     |
| Uses backoff and retry limits | Uses critique and revision |

---

# Retry vs Fallback

| Retry                        | Fallback                         |
| ---------------------------- | -------------------------------- |
| Try the same service again   | Switch to another service        |
| Assumes failure is temporary | Assumes failure may persist      |
| Often first strategy         | Used after retries are exhausted |

---

# Interview Follow-Up Questions

## 1. When should you retry?

Retry only for **transient failures**, such as:

* Timeouts
* Rate limits
* Temporary network issues
* Temporary service unavailability

---

## 2. Why use exponential backoff?

It:

* Reduces pressure on overloaded services
* Prevents retry storms
* Improves overall system stability

---

## 3. Why add jitter?

Without jitter, thousands of clients retry at exactly the same moment.

Randomizing retry timing distributes the load.

---

## 4. How many retries should you allow?

Typically:

* 2–5 retries
* Exponential backoff
* Maximum total timeout

The exact numbers depend on the latency requirements of your application.

---

## 5. How do retries fit into a production AI system?

A production-ready retry strategy includes:

* Maximum retry count
* Exponential backoff with jitter
* Timeouts
* Circuit breakers
* Fallback models or tools
* Idempotency for write operations
* Observability and retry metrics
* Alerts when retry rates increase

---

# Complete Enterprise Retry Architecture

```text
                     User Request
                           │
                           ▼
                    LangGraph Agent
                           │
                           ▼
                     Tool / LLM Call
                           │
                 Success? /      \
                      Yes        No
                       │          │
                       ▼          ▼
                Continue     Retry Policy
                                   │
                     Exponential Backoff + Jitter
                                   │
                      Retry Limit Reached?
                         /              \
                       No              Yes
                       │                │
                       ▼                ▼
                  Retry Call     Circuit Breaker
                                          │
                                          ▼
                              Fallback Model / Tool
                                          │
                                          ▼
                                  Return Response
```

## Senior AI Engineer interview tip

A strong interview answer is to say:

> **The Retry Pattern is a reliability mechanism for recovering from transient failures. In production AI systems, retries should be bounded, use exponential backoff with jitter, apply only to retryable errors, respect idempotency, and integrate with circuit breakers, fallback models, and observability. Simply retrying indefinitely is an anti-pattern that increases cost, latency, and the risk of cascading failures.**
