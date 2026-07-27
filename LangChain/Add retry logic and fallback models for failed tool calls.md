This is one of the **most common production AI interview questions**.

Interviewers are testing whether you know how to build **reliable AI agents**.

In production, tools fail frequently due to:

* API timeout
* Rate limits (429)
* Network failures
* Authentication failures
* Invalid responses
* LLM failures
* Model overload
* Temporary cloud outages

A senior AI engineer should never allow a workflow to fail immediately. Instead, implement:

1. Retry policy
2. Exponential backoff
3. Circuit breaker (optional)
4. Fallback model
5. Human escalation (optional)
6. Logging and monitoring

---

# Production Architecture

```text
                User
                  │
                  ▼
              LangGraph
                  │
                  ▼
            Weather Tool
                  │
          Success? /  \
               Yes    No
                │      │
                ▼      ▼
             Answer   Retry
                        │
                  Success? /  \
                       Yes    No
                        │      │
                        ▼      ▼
                     Answer  Fallback LLM
                                 │
                           Success? / \
                                Yes   No
                                 │     │
                                 ▼     ▼
                              Answer Human Review
```

---

# Step 1: Define State

```python
from typing import TypedDict

class AgentState(TypedDict):
    question: str
    tool_result: str
    answer: str
    retries: int
```

---

# Step 2: Weather Tool

```python
import random

def weather_tool(city: str):

    if random.random() < 0.6:
        raise TimeoutError("Weather API timeout")

    return "28°C Cloudy"
```

This intentionally fails sometimes.

---

# Step 3: Retry Logic

```python
MAX_RETRIES = 3

def call_weather_tool(state):

    try:

        result = weather_tool("Bangalore")

        return {
            "tool_result": result
        }

    except Exception:

        return {
            "retries": state["retries"] + 1
        }
```

---

# Step 4: Conditional Router

```python
def retry_router(state):

    if state.get("tool_result"):
        return "generate"

    if state["retries"] >= MAX_RETRIES:
        return "fallback"

    return "tool"
```

Execution

```text
Tool

↓

Failed

↓

Retries < 3 ?

↓

Yes

↓

Retry
```

After three failures

```text
Tool

↓

Failed

↓

Fallback Model
```

---

# Step 5: Generate Answer

```python
from langchain_openai import ChatOpenAI

primary_llm = ChatOpenAI(
    model="gpt-4.1"
)

def generate_answer(state):

    prompt = f"""
Weather:

{state["tool_result"]}

Answer the user.
"""

    response = primary_llm.invoke(prompt)

    return {
        "answer": response.content
    }
```

---

# Step 6: Fallback Model

Suppose GPT-4.1 is unavailable.

```python
fallback_llm = ChatOpenAI(
    model="gpt-4.1-mini"
)
```

Fallback node

```python
def fallback_node(state):

    response = fallback_llm.invoke(
        state["question"]
    )

    return {
        "answer": response.content
    }
```

---

# LangGraph

```python
from langgraph.graph import StateGraph, END

workflow = StateGraph(AgentState)

workflow.add_node(
    "tool",
    call_weather_tool
)

workflow.add_node(
    "generate",
    generate_answer
)

workflow.add_node(
    "fallback",
    fallback_node
)
```

Conditional edges

```python
workflow.set_entry_point("tool")

workflow.add_conditional_edges(
    "tool",
    retry_router,
    {
        "tool": "tool",
        "generate": "generate",
        "fallback": "fallback"
    }
)

workflow.add_edge(
    "generate",
    END
)

workflow.add_edge(
    "fallback",
    END
)
```

---

# Execution

Initial state

```python
{
    "question":"Weather in Bangalore",
    "retries":0
}
```

Execution

```text
Tool

↓

Timeout

↓

Retry 1

↓

Timeout

↓

Retry 2

↓

Timeout

↓

Retry 3

↓

Fallback Model

↓

Answer
```

---

# Better Retry Strategy (Exponential Backoff)

Instead of retrying immediately:

```text
Retry 1

Wait 1 second

Retry 2

Wait 2 seconds

Retry 3

Wait 4 seconds

Retry 4

Wait 8 seconds
```

Python

```python
import time

def retry(callable_fn):

    delay = 1

    for _ in range(4):

        try:
            return callable_fn()

        except Exception:

            time.sleep(delay)

            delay *= 2

    raise RuntimeError("Failed")
```

Benefits:

* Reduces pressure on overloaded services
* Improves success rate
* Avoids retry storms

---

# Using Tenacity (Recommended)

A production-ready retry library:

```bash
pip install tenacity
```

```python
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=8),
    retry=retry_if_exception_type((TimeoutError, ConnectionError)),
)
def weather_tool(city: str):
    ...
```

This avoids retrying on non-transient errors such as invalid user input.

---

# Multi-Model Fallback

```text
User

↓

GPT-4.1

↓

Failed

↓

GPT-4.1-mini

↓

Failed

↓

Local Llama

↓

Failed

↓

Human
```

Python

```python
models = [
    gpt41,
    gpt41mini,
    llama
]

def ask(question):

    for model in models:

        try:
            return model.invoke(question)

        except Exception:
            pass

    raise RuntimeError("No model available")
```

---

# Tool + LLM Fallback

Suppose the Weather API is down.

```text
Weather API

↓

Failure

↓

LLM General Knowledge

↓

Answer

↓

"I don't have live weather,
but typical weather is..."
```

Implementation

```python
def fallback(state):

    prompt = f"""
The live weather service is unavailable.

Provide a helpful answer to:

{state["question"]}

Clearly state that the weather
may not be current.
"""

    return {
        "answer":
        fallback_llm.invoke(prompt).content
    }
```

Notice the fallback explicitly tells the user that the information is **not live**.

---

# Circuit Breaker

Without a circuit breaker

```text
1000 requests

↓

1000 failures

↓

Server crash
```

With a circuit breaker

```text
Failures > Threshold

↓

Open Circuit

↓

Skip Tool

↓

Fallback
```

Pseudo-code

```python
if circuit_breaker.is_open():
    return fallback_model.invoke(question)

return weather_tool(question)
```

This protects downstream systems during outages.

---

# Logging

```python
logger.info(
    {
        "tool": "weather",
        "retry": retries,
        "latency": latency,
        "fallback": used_fallback,
        "status": "success"
    }
)
```

Monitor:

* Retry count
* Timeout rate
* Fallback rate
* Average latency
* Success rate
* Error categories
* Cost per request

---

# Enterprise Architecture

```text
                 User
                   │
                   ▼
             Planner Agent
                   │
                   ▼
              Tool Router
                   │
            Weather API
           /          \
     Success         Failure
        │               │
        ▼               ▼
     Response        Retry Layer
                         │
                 Success? / \
                      Yes   No
                       │     │
                       ▼     ▼
                  Response  Fallback LLM
                                 │
                           Success? / \
                                Yes   No
                                 │     │
                                 ▼     ▼
                             Response Human Review
```

---

# Interview Follow-Up Questions

## 1. Should every error be retried?

No.

Retry **transient** failures:

* Timeout
* Temporary network issues
* HTTP 429 (respect `Retry-After` when provided)
* HTTP 503/502

Do **not** retry:

* Authentication failures (401/403)
* Invalid input (400)
* Schema validation errors that require code changes
* Permission errors

---

## 2. Why use exponential backoff?

Immediate retries can overwhelm an already unhealthy service.

Backoff:

* Gives the service time to recover
* Reduces cascading failures
* Improves overall reliability

---

## 3. Why use fallback models?

Benefits:

* Higher availability
* Lower user-visible failure rate
* Better resilience during provider outages
* Cost optimization (you can also intentionally route to smaller models)

---

## 4. When should you escalate to a human?

Examples:

* Repeated failures after all retries
* Financial transactions
* Medical recommendations
* Legal advice
* Safety-critical operations

---

## 5. What production improvements would you add?

A production-grade retry and fallback system typically includes:

* Exponential backoff with jitter (to avoid synchronized retries)
* Circuit breakers
* Request timeouts
* Idempotency for retried operations
* Structured error classification
* Multiple fallback providers
* Health checks for models and tools
* Distributed tracing (LangSmith/OpenTelemetry)
* Metrics and alerting
* Dead-letter queues for asynchronous workflows

---

# Complete Production Workflow

```text
                      User Request
                            │
                            ▼
                      LangGraph Agent
                            │
                            ▼
                       Execute Tool
                            │
                    Success? /    \
                         Yes      No
                          │        │
                          ▼        ▼
                    Generate    Retry Layer
                                  │
                           Retry Limit?
                             /        \
                           No         Yes
                           │           │
                           ▼           ▼
                      Retry Tool   Fallback Model
                                        │
                                Success? / \
                                     Yes   No
                                      │     │
                                      ▼     ▼
                              Return Answer Human Review
```

## Senior AI Engineer interview tip

A strong answer goes beyond "retry three times." Explain **how you classify errors**, **which errors are safe to retry**, **why you use exponential backoff with jitter**, **when to switch to a fallback model**, and **how you observe retry and fallback rates in production**. Those design decisions demonstrate production engineering experience rather than just familiarity with the framework.
