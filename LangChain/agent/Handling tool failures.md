Handling tool failures is one of the **most important production AI engineering topics**. A production AI agent **must assume that tools will fail**.

Interview questions include:

* **What if a tool fails?**
* **How do you recover from tool failures?**
* **How do you retry safely?**
* **How do you build fault-tolerant agents?**

A Senior AI Engineer is expected to explain **resilience patterns**, not just `try/except`.

---

# Why Do Tools Fail?

A tool is usually an external dependency.

```text
               Agent

                  │

                  ▼

             Weather API
             PostgreSQL
             Redis
             Search API
             Vector DB
             Email API
```

Any of these can fail.

Typical reasons:

* Network timeout
* API rate limit (429)
* Authentication failure (401/403)
* Service unavailable (503)
* Database connection failure
* Invalid input
* Tool bug
* Slow response
* Partial results

Your agent must continue gracefully.

---

# What Happens Without Error Handling?

Example:

```text
User

↓

Agent

↓

Weather Tool

↓

Exception

↓

Agent Crashes
```

This is unacceptable in production.

---

# Production Architecture

Instead:

```text
                 Agent

                   │

                   ▼

             Execute Tool

                   │

         ┌─────────┴──────────┐

         ▼                    ▼

      Success              Failure

         │                    │

         ▼                    ▼

 Continue          Retry / Fallback /
                   Human / End
```

---

# Strategy 1: Catch Exceptions

Never allow exceptions to crash the workflow.

Bad:

```python
def weather(city):

    return api.call(city)
```

Good:

```python
from langchain.tools import tool

@tool
def weather(city: str):

    try:

        return api.call(city)

    except Exception as e:

        return {

            "status": "error",

            "message": str(e)
        }
```

Instead of crashing, the tool returns structured information.

---

# Strategy 2: Retry

Many failures are temporary.

Example:

```text
Weather API

↓

Timeout

↓

Retry

↓

Success
```

Typical retry policy:

```text
Attempt 1

↓

Fail

↓

Wait

↓

Attempt 2

↓

Fail

↓

Wait

↓

Attempt 3

↓

Success
```

Example:

```python
import time

MAX_RETRIES = 3

def call_weather(city):

    for attempt in range(MAX_RETRIES):

        try:

            return api.call(city)

        except Exception:

            time.sleep(2 ** attempt)

    raise RuntimeError("Tool failed")
```

Notice the exponential delay.

---

# Why Exponential Backoff?

Bad:

```text
Retry

Retry

Retry

Retry
```

This overloads the service.

Better:

```text
1 second

↓

2 seconds

↓

4 seconds

↓

8 seconds
```

The service gets time to recover.

---

# Strategy 3: Circuit Breaker

Suppose the database is down.

Without protection:

```text
Agent

↓

Database

↓

Fail

↓

Database

↓

Fail

↓

Database

↓

Fail
```

Thousands of requests keep hitting the failing service.

A circuit breaker changes the behavior.

```text
Database

↓

5 failures

↓

Circuit Opens

↓

Skip Database

↓

Fallback
```

After a cooldown period:

```text
Try Again

↓

Recovered?

↓

Yes

↓

Close Circuit
```

---

# Strategy 4: Fallback Tool

Suppose you have:

```text
Primary Search

Backup Search
```

Flow:

```text
Search Tool

↓

Failure

↓

Fallback Search

↓

Answer
```

Example:

```python
def search(state):

    try:

        return primary_search()

    except Exception:

        return backup_search()
```

Enterprise systems often use multiple providers.

```text
OpenAI

↓

Fail

↓

Azure OpenAI

↓

Fail

↓

Anthropic

↓

Success
```

---

# Strategy 5: Retry Limit

Never retry forever.

State:

```python
class AgentState(TypedDict):

    tool_retries: int
```

Router:

```python
MAX_RETRIES = 3

def router(state):

    if state["tool_retries"] >= MAX_RETRIES:

        return "human"

    return "retry"
```

Flow:

```text
Tool

↓

Fail

↓

Retry

↓

Fail

↓

Retry

↓

Fail

↓

Human Review
```

---

# Strategy 6: Timeout

Suppose a tool hangs.

```text
Database

↓

Waiting...

↓

Waiting...

↓

Waiting...
```

The agent never continues.

Instead:

```text
30 seconds

↓

Timeout

↓

Fallback
```

Example:

```python
import requests

requests.get(
    url,
    timeout=10
)
```

Always set timeouts for network operations.

---

# Strategy 7: Validate Tool Output

Not every successful HTTP response is actually usable.

Example:

```python
{
    "temperature": None
}
```

Don't assume success.

```python
if result["temperature"] is None:

    raise ValueError("Invalid data")
```

Validate:

* Required fields
* Types
* Value ranges
* Empty responses

---

# Strategy 8: Human Escalation

Suppose retries fail.

```text
Tool

↓

Retry

↓

Retry

↓

Retry

↓

Still Failed

↓

Human
```

The workflow pauses.

Human decides.

---

# Strategy 9: Alternative Reasoning

Suppose weather fails.

Instead of:

```text
Error
```

Agent responds:

```text
The live weather service is unavailable right now.

Would you like yesterday's weather or a forecast based on historical averages?
```

The conversation continues.

---

# Strategy 10: Observability

Every failure should be logged.

Example:

```text
Workflow:
thread-101

Tool:
weather

Attempts:
3

Latency:
8.4s

Error:
Timeout
```

Metrics to monitor:

* Success rate
* Failure rate
* Retry count
* Timeout rate
* Latency
* Cost
* Circuit breaker status

---

# LangGraph Implementation

State:

```python
from typing import TypedDict

class AgentState(TypedDict):

    retries: int

    tool_failed: bool

    tool_result: dict
```

Tool node:

```python
def weather_node(state):

    try:

        result = weather_api()

        return {

            "tool_result": result,

            "tool_failed": False

        }

    except Exception:

        return {

            "tool_failed": True,

            "retries":
                state["retries"] + 1

        }
```

Router:

```python
MAX_RETRIES = 3

def router(state):

    if not state["tool_failed"]:

        return "generate"

    if state["retries"] >= MAX_RETRIES":

        return "human_review"

    return "retry_tool"
```

Workflow:

```text
Weather Tool

↓

Success?

├── Yes → Generate Answer

└── No

       ↓

Retries < 3?

├── Yes → Retry Tool

└── No → Human Review
```

---

# Production Architecture

```text
                 Agent

                   │

                   ▼

             Tool Router

                   │

                   ▼

            Execute Tool

                   │

      ┌────────────┼─────────────┐

      ▼            ▼             ▼

   Success      Retry        Circuit Breaker

      │            │             │

      ▼            ▼             ▼

 Generate     Fallback      Human Review

      │

      ▼

    Response
```

---

# Best Practices

### 1. Never expose raw exceptions

Bad:

```text
Traceback...
```

Good:

```text
The customer database is temporarily unavailable.
Please try again shortly.
```

---

### 2. Return structured errors

Instead of:

```python
return "failed"
```

Use:

```python
return {

    "status": "error",

    "code": "TIMEOUT",

    "retryable": True
}
```

The agent can make better decisions.

---

### 3. Distinguish retryable vs permanent failures

Retryable:

* Timeout
* 429
* 503
* Temporary network issue

Don't retry:

* Invalid input
* Authentication failure
* Tool not found
* Validation error

---

### 4. Make tools idempotent

If a retry occurs, repeating the request should not create duplicate side effects.

Example:

* Good: reading weather data.
* Risky: transferring ₹10,000 twice.

For side-effecting tools, use idempotency keys or transaction IDs.

---

### 5. Monitor continuously

Track:

* Tool latency
* Failure percentage
* Retry rate
* Availability
* Cost per tool
* Fallback frequency

---

# Interview Questions

### Should every tool be retried?

No. Retry only transient failures such as timeouts, network errors, or HTTP 429/503 responses. Do not retry invalid requests, authentication failures, or malformed inputs until the underlying issue is fixed.

---

### Why use exponential backoff?

It reduces load on struggling services and increases the chance of success without creating a retry storm.

---

### Why use fallback models or tools?

Fallbacks improve availability. If the primary provider or service is unavailable, another provider or alternative tool can continue serving the request.

---

### What if a tool modifies data?

Use idempotency keys, transactions, or compensating actions. Never blindly retry operations like payments or record creation, as retries can duplicate side effects.

---

# Senior AI Engineer Interview Answer

> **In production, I assume every tool can fail. I classify failures into transient and permanent errors. Transient failures such as timeouts or HTTP 429/503 responses are handled with bounded retries and exponential backoff. Persistent failures trigger fallback tools or alternate model providers, and circuit breakers prevent repeatedly calling unhealthy services. Every tool call has a timeout, structured error handling, validation of returned data, and comprehensive logging with metrics. For state-changing operations, I ensure idempotency before retrying. If the agent still cannot complete a high-risk task after the retry budget is exhausted, the workflow escalates to a human reviewer rather than looping indefinitely or failing silently.**
